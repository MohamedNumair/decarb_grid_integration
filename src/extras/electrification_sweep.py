"""Electrification & Solar PV Penetration Sweep
================================================
For each tariff, interpolates between heat_0/heat_100 and solar_0/solar_100
DECARB outputs at intermediate penetration levels (0-20-40-60-80-100%).

For each percentage step, randomly selects which buildings get switched
from their baseline profile to their electrified/solar profile, aggregates
the system-wide demand, and records peak demand.

Produces:
  - summary_heating.csv   (peak demand per tariff × heating pct)
  - summary_solar.csv     (peak demand per tariff × solar pct)
  - peak_sweep_heating.png (line plot)
  - peak_sweep_solar.png   (line plot)

Usage:
    python electrification_sweep.py --study study_8
"""

import argparse
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_project_root = Path(__file__).parents[1]
sys.path.insert(0, str(_project_root))
try:
    from config.loader import load_config
except ImportError:
    _project_root = Path(__file__).parents[2]
    sys.path.insert(0, str(_project_root))
    from config.loader import load_config
try:
    from src.create_network_inputs import load_building_map, read_decarb_ts
except ImportError:
    sys.path.insert(0, str(_project_root / "src"))
    from create_network_inputs import load_building_map, read_decarb_ts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RANDOM_SEED = 42
PCT_STEPS = [0, 20, 40, 60, 80, 100]
N_HOURS = 8760

TARIFFS = ["flat", "tou", "flat_cap", "tou_cap"]
TARIFF_COLORS = {
    "flat":     "black",
    "tou":      "#1f77b4",
    "flat_cap": "red",
    "tou_cap":  "#D5C655",
}
TARIFF_LABELS = {
    "flat":     "Flat",
    "tou":      "TOU",
    "flat_cap": "Flat w/ CC",
    "tou_cap":  "TOU w/ CC",
}


def opendss_peak_mw(results_dir, case_name):
    """Read system peak demand (MW) from OpenDSS zone_yearly.csv."""
    csv_path = results_dir / case_name / "zone_yearly.csv"
    if not csv_path.exists():
        return np.nan
    df = pd.read_csv(csv_path)
    return df.groupby("hour")["total_kW"].sum().max() / 1000.0


def scale_sweep(df, results_dir, dimension):
    """Scale DECARB-aggregated peaks to match OpenDSS zone_yearly endpoints.

    For each tariff, computes the OpenDSS/DECARB ratio at 0% and 100%,
    linearly interpolates the ratio for intermediate steps, and multiplies.
    """
    df = df.copy()
    for tariff in df["tariff"].unique():
        if dimension == "heating":
            case_0 = f"{tariff}_heat_0_solar_0"
            case_100 = f"{tariff}_heat_100_solar_0"
        else:
            case_0 = f"{tariff}_heat_0_solar_0"
            case_100 = f"{tariff}_heat_0_solar_100"

        opendss_0 = opendss_peak_mw(results_dir, case_0)
        opendss_100 = opendss_peak_mw(results_dir, case_100)

        mask = df["tariff"] == tariff
        decarb_0 = df.loc[mask & (df["pct"] == 0), "peak_mw"].values
        decarb_100 = df.loc[mask & (df["pct"] == 100), "peak_mw"].values

        if (np.isnan(opendss_0) or np.isnan(opendss_100)
                or len(decarb_0) == 0 or len(decarb_100) == 0
                or decarb_0[0] == 0 or decarb_100[0] == 0):
            log.warning(f"  Cannot scale {tariff} {dimension}: "
                        f"missing OpenDSS results")
            continue

        ratio_0 = opendss_0 / decarb_0[0]
        ratio_100 = opendss_100 / decarb_100[0]
        log.info(f"  {tariff} scaling: ratio_0={ratio_0:.4f}, "
                 f"ratio_100={ratio_100:.4f}")

        for idx in df.index[mask]:
            pct = df.at[idx, "pct"]
            t = pct / 100.0
            ratio = ratio_0 + t * (ratio_100 - ratio_0)
            df.at[idx, "peak_mw"] = df.at[idx, "peak_mw"] * ratio

    return df


def read_buy_profile(decarb_dir, bldg_id):
    """Read the 'buy' column from a building's ts.csv, in kW."""
    df = read_decarb_ts(decarb_dir, bldg_id)
    if df is None or len(df) != N_HOURS:
        return None
    return df["buy"].values


def identify_candidates(decarb_dir, bldg_ids, dimension="heating"):
    """Identify which buildings are candidates for switching.

    For heating: candidates are buildings NOT already on electric heating.
    For solar: candidates are ALL buildings (baseline has 0% solar).
    """
    if dimension == "solar":
        return list(bldg_ids)

    candidates = []
    for bldg_id in bldg_ids:
        sp_path = decarb_dir / str(bldg_id) / "in" / "sp.csv"
        if not sp_path.exists():
            continue
        sp = pd.read_csv(sp_path)
        hvac0 = str(sp["pHVAC0"].iloc[0])
        if "Electricity" not in hvac0 and "ASHP" not in hvac0 and "Heat Pump" not in hvac0:
            candidates.append(bldg_id)
    return candidates


def build_batches(candidates, bldg_counts, pct_steps, seed=RANDOM_SEED):
    """Build cumulative sets of building IDs to switch at each percentage.

    Shuffles candidate instances (weighted by count), then assigns buildings
    to percentage buckets.

    Returns dict: {pct: set of bldg_ids to use switched profile}.
    """
    random.seed(seed)

    weighted = []
    for bldg_id in candidates:
        count = bldg_counts.get(bldg_id, 1)
        weighted.extend([bldg_id] * count)

    random.shuffle(weighted)
    total_instances = sum(bldg_counts.get(b, 1) for b in bldg_counts)

    batches = {}
    for pct in pct_steps:
        if pct == 0:
            batches[pct] = set()
        elif pct == 100:
            batches[pct] = set(candidates)
        else:
            target = int(round(total_instances * pct / 100))
            switched = set()
            count_so_far = 0
            for inst_bldg in weighted:
                if inst_bldg in switched:
                    continue
                switched.add(inst_bldg)
                count_so_far += bldg_counts.get(inst_bldg, 1)
                if count_so_far >= target:
                    break
            batches[pct] = switched

    return batches


def preload_profiles(decarb_dir, bldg_ids, max_workers=16):
    """Read all building profiles in parallel. Returns {bldg_id: np.array}."""
    profiles = {}

    def _read(bldg_id):
        return bldg_id, read_buy_profile(decarb_dir, bldg_id)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_read, b) for b in bldg_ids]
        for f in as_completed(futures):
            bldg_id, profile = f.result()
            if profile is not None:
                profiles[bldg_id] = profile

    return profiles


def aggregate_peak_cached(base_profiles, switched_profiles, bldg_ids,
                          bldg_counts, switched_set):
    """Compute system peak demand (MW) using preloaded profiles."""
    total_kw = np.zeros(N_HOURS)
    for bldg_id in bldg_ids:
        if bldg_id in switched_set and bldg_id in switched_profiles:
            profile = switched_profiles[bldg_id]
        elif bldg_id in base_profiles:
            profile = base_profiles[bldg_id]
        else:
            continue

        total_kw += profile * bldg_counts.get(bldg_id, 1)

    return total_kw.max() / 1000.0  # kW → MW


def run_sweep(study_dir, decarb_root, bldg_ids, bldg_counts, tariff,
              dimension, pct_steps):
    """Run the sweep for one tariff and one dimension (heating or solar)."""
    if dimension == "heating":
        base_case = f"{tariff}_heat_0_solar_0"
        switched_case = f"{tariff}_heat_100_solar_0"
    else:
        base_case = f"{tariff}_heat_0_solar_0"
        switched_case = f"{tariff}_heat_0_solar_100"

    base_dir = decarb_root / base_case
    switched_dir = decarb_root / switched_case

    if not base_dir.exists() or not switched_dir.exists():
        log.warning(f"Missing cases for {tariff} {dimension}: "
                    f"{base_case} or {switched_case}")
        return None

    candidates = identify_candidates(base_dir, bldg_ids, dimension)
    log.info(f"  {tariff} {dimension}: {len(candidates)} candidates "
             f"out of {len(bldg_ids)} buildings")

    log.info(f"    Preloading base profiles ({base_case})...")
    base_profiles = preload_profiles(base_dir, bldg_ids)
    log.info(f"    Preloading switched profiles ({switched_case})...")
    switched_profiles = preload_profiles(switched_dir, candidates)
    log.info(f"    Loaded {len(base_profiles)} base, "
             f"{len(switched_profiles)} switched profiles")

    batches = build_batches(candidates, bldg_counts, pct_steps)

    results = []
    for pct in pct_steps:
        switched_set = batches[pct]
        peak_mw = aggregate_peak_cached(base_profiles, switched_profiles,
                                        bldg_ids, bldg_counts, switched_set)
        n_switched = len(switched_set)
        log.info(f"    {pct:3d}%: {n_switched:>5d} bldgs switched, "
                 f"peak = {peak_mw:.2f} MW")
        results.append({
            "tariff": tariff,
            "pct": pct,
            "n_switched": n_switched,
            "peak_mw": peak_mw,
        })

    return results


def plot_sweep(df, dimension, baseline_peak, out_path):
    """Line plot of peak demand vs penetration for all tariffs."""
    fig, ax = plt.subplots(figsize=(11, 7))

    # Collect all labels with positions for overlap avoidance
    all_labels_pos = []  # (x, y, text, color)

    for tariff in TARIFFS:
        tdf = df[df["tariff"] == tariff].sort_values("pct")
        if tdf.empty:
            continue
        color = TARIFF_COLORS[tariff]
        label = TARIFF_LABELS[tariff]
        ax.plot(tdf["pct"], tdf["peak_mw"], "-o", color=color, lw=2.5,
                markersize=8, markeredgecolor="white", label=label)
        for _, row in tdf.iterrows():
            all_labels_pos.append((row["pct"], row["peak_mw"],
                                   f"{row['peak_mw']:.1f}", color))

    # Place labels with vertical offset to avoid overlaps
    min_gap = baseline_peak * 0.025
    for pct in sorted(set(p[0] for p in all_labels_pos)):
        group = [(x, y, t, c) for x, y, t, c in all_labels_pos if x == pct]
        group.sort(key=lambda g: g[1])
        placed_y = []
        for x, y, text, color in group:
            label_y = y + baseline_peak * 0.01
            for py in placed_y:
                if abs(label_y - py) < min_gap:
                    label_y = py + min_gap
            placed_y.append(label_y)
            ax.text(x, label_y, text, ha="center", va="bottom",
                    fontsize=11, color=color, fontweight="bold")

    dim_label = ("ASHP Heating Electrification" if dimension == "heating"
                 else "Rooftop Solar PV Penetration")
    ax.set_title(f"System Peak Electrical Demand by Tariff and {dim_label}",
                 fontsize=20)
    ax.set_xlabel(f"{dim_label}", fontsize=16)
    ax.set_ylabel("Peak Electrical Demand (MW)", fontsize=16)
    ax.set_xticks(PCT_STEPS)
    ax.set_xticklabels([f"{p}%" for p in PCT_STEPS], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    # Tariff legend in upper left
    ax.legend(loc="upper left", fontsize=13, frameon=True)

    # Acronym legend at bottom as separate box
    if dimension == "heating":
        acronym_text = "TOU: Time-of-Use Rate    CC: Capacity Charge    ASHP: Air-Source Heat Pump"
    else:
        acronym_text = "TOU: Time-of-Use Rate    CC: Capacity Charge    PV: Rooftop Solar Panel"
    fig.text(0.5, 0.02, acronym_text, ha="center", va="bottom", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="gray", alpha=0.9))

    fig.subplots_adjust(top=0.92, bottom=0.15, left=0.08, right=0.97)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    log.info(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Electrification & Solar sweep")
    parser.add_argument("--study", required=True)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip sweep, replot from existing summary CSVs")
    args = parser.parse_args()

    cfg = load_config()
    project_root = Path(cfg["paths"]["project_root"]).resolve()
    study_dir = project_root / "out" / "studies" / args.study
    decarb_root = study_dir / "decarb"

    out_dir = study_dir / "opendss_results" / "peak_comparison"
    out_dir.mkdir(exist_ok=True)

    heat_csv = out_dir / "summary_heating.csv"
    solar_csv = out_dir / "summary_solar.csv"

    results_dir = study_dir / "opendss_results"

    if args.plot_only:
        if heat_csv.exists():
            df_heat = pd.read_csv(heat_csv)
            df_heat = scale_sweep(df_heat, results_dir, "heating")
            baseline_peak = df_heat[df_heat["pct"] == 0]["peak_mw"].max()
            plot_sweep(df_heat, "heating", baseline_peak,
                       out_dir / "peak_sweep_heating.png")
        else:
            log.warning(f"No heating summary found: {heat_csv}")

        if solar_csv.exists():
            df_solar = pd.read_csv(solar_csv)
            df_solar = scale_sweep(df_solar, results_dir, "solar")
            baseline_peak = df_solar[df_solar["pct"] == 0]["peak_mw"].max()
            plot_sweep(df_solar, "solar", baseline_peak,
                       out_dir / "peak_sweep_solar.png")
        else:
            log.warning(f"No solar summary found: {solar_csv}")
        return

    if not decarb_root.exists():
        log.error(f"DECARB directory not found: {decarb_root}")
        return

    # Load building map
    bm = load_building_map(study_dir=study_dir)
    bldg_counts = {}
    for _, row in bm.iterrows():
        bid = int(row["bldg_id"])
        bldg_counts[bid] = bldg_counts.get(bid, 0) + int(row["count"])

    bldg_ids = sorted(bldg_counts.keys())
    total_instances = sum(bldg_counts.values())
    log.info(f"Loaded {len(bldg_ids)} buildings, {total_instances} total instances")

    # Run heating sweep
    log.info("\n=== Heating Electrification Sweep ===")
    all_heating = []
    for tariff in TARIFFS:
        results = run_sweep(study_dir, decarb_root, bldg_ids, bldg_counts,
                            tariff, "heating", PCT_STEPS)
        if results:
            all_heating.extend(results)

    if all_heating:
        df_heat = pd.DataFrame(all_heating)
        df_heat.to_csv(heat_csv, index=False)
        log.info(f"Saved: summary_heating.csv")
        df_heat = scale_sweep(df_heat, results_dir, "heating")
        baseline_peak = df_heat[df_heat["pct"] == 0]["peak_mw"].max()
        plot_sweep(df_heat, "heating", baseline_peak,
                   out_dir / "peak_sweep_heating.png")

    # Run solar sweep
    log.info("\n=== Solar PV Penetration Sweep ===")
    all_solar = []
    for tariff in TARIFFS:
        results = run_sweep(study_dir, decarb_root, bldg_ids, bldg_counts,
                            tariff, "solar", PCT_STEPS)
        if results:
            all_solar.extend(results)

    if all_solar:
        df_solar = pd.DataFrame(all_solar)
        df_solar.to_csv(solar_csv, index=False)
        log.info(f"Saved: summary_solar.csv")
        df_solar = scale_sweep(df_solar, results_dir, "solar")
        baseline_peak = df_solar[df_solar["pct"] == 0]["peak_mw"].max()
        plot_sweep(df_solar, "solar", baseline_peak,
                   out_dir / "peak_sweep_solar.png")


if __name__ == "__main__":
    main()
