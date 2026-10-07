"""
Demand interpolation between DECARB bookend cases.
===================================================
Following the approach from ercot_project/src/decarb_electrification_analysis_v4.py:

1. Calculate baseline electrification / PV penetration from metadata
   (in.heating_fuel == "Electricity" for electrification,
    in.has_pv == "Yes" for PV)
2. Round up to nearest 10% interval
3. Randomly assign non-adopted buildings in cumulative batches
   (instance-weighted by bus count from feeder matching)
4. For each intermediate level, aggregate demand using base + delta:
   start with all buildings at heat_0 (or solar_0), then for switched
   buildings add the difference to the heat_100 (or solar_100) profile

Output: interpolation_summary.csv with peak kW, timing, and annual MWh
for each tariff × penetration level combination.
"""

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from config.loader import load_config, load_network_config

from create_network_inputs import (
    build_substitution_map,
    read_decarb_ts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RANDOM_SEED = 42
TOLERANCE_PCT = 0.5   # accept actual% within ±0.5% of target

HEATING_FUEL_COL = "in.heating_fuel"
HEATING_FUEL_ELEC_VAL = "Electricity"
HAS_PV_COL = "in.has_pv"
HAS_PV_VAL = "Yes"

HOUR_ZERO = pd.Timestamp("2018-01-01 00:00")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_parquet(path):
    try:
        return pd.read_parquet(str(path), engine="fastparquet")
    except Exception:
        return pd.read_parquet(str(path), engine="pyarrow")


def load_metadata(cfg):
    """Load the state-level ResStock metadata parquet."""
    state = cfg.get("state", "TX")
    input_dir = Path(cfg["paths"]["input_dir"])
    meta_path = input_dir / f"{state}_upgrade0.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata parquet not found: {meta_path}")
    return pd.read_parquet(str(meta_path))


def get_building_ids(feeder_df):
    """Extract unique matched building IDs from feeder matching file."""
    bldg_col = ("nrel_climate_match" if "nrel_climate_match" in feeder_df.columns
                else "bldg_id")
    return feeder_df[bldg_col].dropna().astype(int).unique().tolist()


def get_instance_counts(feeder_df):
    """Per-building bus instance counts from the feeder matching file.

    Each row in feeder_df is one DSS load (one bus instance).
    """
    bldg_col = ("nrel_climate_match" if "nrel_climate_match" in feeder_df.columns
                else "bldg_id")
    return feeder_df[bldg_col].dropna().astype(int).value_counts().to_dict()


def compute_baseline_pct(metadata_df, matched_bldg_ids, instance_counts,
                         col, val):
    """Instance-weighted baseline percentage for a boolean attribute.

    Returns (baseline_pct, already_adopted_ids, non_adopted_ids).
    """
    adopted_meta_ids = set(
        metadata_df.loc[metadata_df[col] == val, "bldg_id"].unique()
    )
    matched_set = set(matched_bldg_ids)

    already_adopted = matched_set & adopted_meta_ids
    non_adopted = matched_set - adopted_meta_ids

    adopted_instances = sum(instance_counts.get(b, 0) for b in already_adopted)
    total_instances = sum(instance_counts.get(b, 0) for b in matched_set)

    if total_instances == 0:
        return 0.0, already_adopted, non_adopted

    baseline_pct = 100.0 * adopted_instances / total_instances
    return baseline_pct, already_adopted, non_adopted


def build_interpolation_steps(baseline_pct, step=10):
    """Round baseline up to nearest step% and generate steps up to 90%."""
    first = int(math.ceil(baseline_pct / step)) * step
    first = max(first, step)
    return list(range(first, 100, step))


def build_adoption_batches(non_adopted_ids, instance_counts, total_instances,
                           already_adopted_count, pct_steps, seed=RANDOM_SEED):
    """Randomly assign non-adopted buildings to cumulative batches.

    Mirrors build_electrification_batches from ercot_project v4.
    """
    import random
    rng = random.Random(seed)

    bldg_instances = {b: instance_counts.get(b, 1) for b in non_adopted_ids}
    unique_bldgs = list(bldg_instances.keys())
    rng.shuffle(unique_bldgs)

    batches = {}
    cumulative = []
    cumulative_count = 0
    pointer = 0
    tolerance = total_instances * TOLERANCE_PCT / 100.0

    for pct in sorted(pct_steps):
        target_new = int(round(total_instances * pct / 100.0)) - already_adopted_count
        target_new = max(0, target_new)

        while pointer < len(unique_bldgs):
            deficit = target_new - cumulative_count
            if abs(deficit) <= tolerance or deficit <= 0:
                break
            bldg = unique_bldgs[pointer]
            cumulative.append(bldg)
            cumulative_count += bldg_instances[bldg]
            pointer += 1

        actual_pct = (already_adopted_count + cumulative_count) / total_instances * 100
        log.info(f"  {pct:3d}%: {len(cumulative):4d} buildings switched, "
                 f"{cumulative_count:5d} instances -> actual {actual_pct:.1f}%")
        batches[pct] = set(cumulative)

    return batches


# ─────────────────────────────────────────────────────────────────────────────
# Profile loading and aggregation
# ─────────────────────────────────────────────────────────────────────────────

def load_net_profiles(decarb_case_dir, building_ids, sub_map=None):
    """Load net-kW profiles (buy - sell) for all buildings from a DECARB case.

    This matches the OpenDSS loadshape definition:
        net_kw = buy - sell
    Positive = drawing from grid, negative = exporting to grid (solar).

    Returns dict {bldg_id: net_kw_array (8760,)}.
    """
    if sub_map is None:
        sub_map = {}
    profiles = {}
    for bldg_id in building_ids:
        bldg_id = int(bldg_id)
        ts_df = read_decarb_ts(decarb_case_dir, bldg_id)
        if ts_df is None or "buy" not in ts_df.columns:
            if bldg_id in sub_map:
                ts_df = read_decarb_ts(decarb_case_dir, sub_map[bldg_id])
            if ts_df is None or "buy" not in ts_df.columns:
                continue
        buy = ts_df["buy"].values[:8760].astype(np.float64)
        sell = ts_df["sell"].values[:8760].astype(np.float64) if "sell" in ts_df.columns else 0.0
        profiles[bldg_id] = buy - sell
    return profiles


def aggregate_demand(instance_counts, profiles_base, profiles_full=None,
                     switched_ids=None, n_hours=8760):
    """Aggregate demand using base + delta approach.

    1. Sum all buildings at their base (heat_0 / solar_0) profile
    2. For switched buildings, add (full - base) delta

    Returns total_kw array (8760,).
    """
    total_kw = np.zeros(n_hours, dtype=np.float64)

    # Base aggregation
    for bldg_id, count in instance_counts.items():
        bldg_id = int(bldg_id)
        buy = profiles_base.get(bldg_id)
        if buy is None or len(buy) < n_hours:
            continue
        total_kw += buy[:n_hours] * count

    # Add delta for switched buildings
    if switched_ids and profiles_full:
        for bldg_id in switched_ids:
            bldg_id = int(bldg_id)
            count = instance_counts.get(bldg_id, 0)
            if count == 0:
                continue
            base = profiles_base.get(bldg_id)
            full = profiles_full.get(bldg_id)
            if base is None or full is None:
                continue
            if len(base) < n_hours or len(full) < n_hours:
                continue
            total_kw += (full[:n_hours] - base[:n_hours]) * count

    return total_kw


def summarize_demand(total_kw):
    """Compute peak kW, peak datetime, and annual MWh from hourly kW array."""
    peak_idx = int(np.argmax(total_kw))
    peak_kw = float(total_kw[peak_idx])
    peak_dt = HOUR_ZERO + pd.Timedelta(hours=peak_idx)
    annual_mwh = float(np.sum(total_kw)) / 1000.0  # kWh -> MWh
    return peak_kw, peak_dt, annual_mwh


# ─────────────────────────────────────────────────────────────────────────────
# Main interpolation driver
# ─────────────────────────────────────────────────────────────────────────────

def find_bookend_case(decarb_root, tariff_name, elec_level, solar_level):
    """Find the DECARB case directory for a bookend combination."""
    case_name = f"{tariff_name}_heat_{elec_level}_solar_{solar_level}"
    case_dir = decarb_root / case_name
    if case_dir.exists():
        return case_dir, case_name
    return None, None


def interpolate_study_demand(cfg, ncfg, study_dir, elec_levels=None,
                             solar_levels=None, seed=RANDOM_SEED):
    """Run interpolation for a study.

    Produces interpolation_summary.csv with peak kW, timing, and annual MWh
    at each intermediate electrification and PV penetration level.

    elec_levels and solar_levels are accepted for pipeline compatibility
    but ignored — the script auto-detects intermediate steps from metadata.
    """
    # Load metadata
    metadata_df = load_metadata(cfg)
    log.info(f"Loaded metadata: {len(metadata_df)} buildings")

    # Load feeder matching
    feeder_parquet = study_dir / "feeder_summaries_matched.parquet"
    if not feeder_parquet.exists():
        feeder_parquet = Path(cfg["paths"]["input_dir"]) / "feeder_summaries_matched.parquet"
    feeder_df = read_parquet(feeder_parquet)
    log.info(f"Feeder matching: {len(feeder_df)} loads")

    all_bldg_ids = get_building_ids(feeder_df)
    instance_counts = get_instance_counts(feeder_df)
    total_instances = sum(instance_counts.values())
    log.info(f"Unique buildings: {len(all_bldg_ids)}, "
             f"total bus instances: {total_instances}")

    # ── Calculate baseline levels (instance-weighted by bus count) ───────────
    elec_base_pct, elec_adopted, elec_non_adopted = compute_baseline_pct(
        metadata_df, all_bldg_ids, instance_counts,
        HEATING_FUEL_COL, HEATING_FUEL_ELEC_VAL)
    elec_adopted_inst = sum(instance_counts.get(b, 0) for b in elec_adopted)

    pv_base_pct, pv_adopted, pv_non_adopted = compute_baseline_pct(
        metadata_df, all_bldg_ids, instance_counts,
        HAS_PV_COL, HAS_PV_VAL)
    pv_adopted_inst = sum(instance_counts.get(b, 0) for b in pv_adopted)

    log.info(f"Baseline electrification: {elec_base_pct:.1f}% "
             f"({len(elec_adopted)} bldgs, {elec_adopted_inst} instances)")
    log.info(f"Baseline PV penetration:  {pv_base_pct:.1f}% "
             f"({len(pv_adopted)} bldgs, {pv_adopted_inst} instances)")

    elec_steps = build_interpolation_steps(elec_base_pct)
    pv_steps = build_interpolation_steps(pv_base_pct)
    log.info(f"Electrification steps: {elec_steps}")
    log.info(f"PV penetration steps:  {pv_steps}")

    # ── Build adoption batches ──────────────────────────────────────────────
    log.info("Building electrification batches...")
    elec_batches = build_adoption_batches(
        elec_non_adopted, instance_counts, total_instances,
        elec_adopted_inst, elec_steps, seed=seed)

    log.info("Building PV batches...")
    pv_batches = build_adoption_batches(
        pv_non_adopted, instance_counts, total_instances,
        pv_adopted_inst, pv_steps, seed=seed + 1000)

    # ── Detect tariffs ──────────────────────────────────────────────────────
    decarb_root = study_dir / "decarb"
    tariff_names = set()
    for d in decarb_root.iterdir():
        if d.is_dir() and "_heat_" in d.name:
            tariff_names.add(d.name.split("_heat_")[0])
    tariff_names = sorted(tariff_names)
    log.info(f"Tariffs: {tariff_names}")

    # ── Profile cache to avoid reloading ────────────────────────────────────
    profile_cache = {}

    def _load_profiles(case_dir):
        key = str(case_dir)
        if key not in profile_cache:
            sub_map = build_substitution_map(case_dir, all_bldg_ids, cfg)
            profile_cache[key] = load_net_profiles(
                case_dir, all_bldg_ids, sub_map=sub_map)
            log.info(f"  Loaded {len(profile_cache[key])} net profiles from {case_dir.name}")
        return profile_cache[key]

    # ── Collect results ─────────────────────────────────────────────────────
    rows = []

    for tariff in tariff_names:
        log.info(f"\n{'='*60}\nTariff: {tariff}\n{'='*60}")

        # --- Electrification interpolation ---
        for solar_level in [0, 100]:
            base_dir, base_name = find_bookend_case(
                decarb_root, tariff, 0, solar_level)
            full_dir, full_name = find_bookend_case(
                decarb_root, tariff, 100, solar_level)

            if base_dir is None or full_dir is None:
                log.warning(f"Missing bookends for {tariff} elec @ solar={solar_level}")
                continue

            profiles_base = _load_profiles(base_dir)
            profiles_full = _load_profiles(full_dir)

            # Bookend: 0% (= baseline)
            total_kw = aggregate_demand(instance_counts, profiles_base)
            peak_kw, peak_dt, annual_mwh = summarize_demand(total_kw)
            rows.append({
                "tariff": tariff,
                "dimension": "electrification",
                "other_axis": f"solar_{solar_level}",
                "pct": round(elec_base_pct, 1),
                "pct_label": f"heat_0 (baseline {elec_base_pct:.1f}%)",
                "peak_kW": round(peak_kw, 1),
                "peak_date": peak_dt.strftime("%b %d"),
                "peak_time": peak_dt.strftime("%H:%M"),
                "annual_MWh": round(annual_mwh, 1),
            })

            # Intermediate levels
            for pct in elec_steps:
                switched = elec_batches[pct]
                total_kw = aggregate_demand(
                    instance_counts, profiles_base, profiles_full, switched)
                peak_kw, peak_dt, annual_mwh = summarize_demand(total_kw)
                rows.append({
                    "tariff": tariff,
                    "dimension": "electrification",
                    "other_axis": f"solar_{solar_level}",
                    "pct": pct,
                    "pct_label": f"heat_{pct}",
                    "peak_kW": round(peak_kw, 1),
                    "peak_date": peak_dt.strftime("%b %d"),
                    "peak_time": peak_dt.strftime("%H:%M"),
                    "annual_MWh": round(annual_mwh, 1),
                })
                log.info(f"  {tariff} heat_{pct} solar_{solar_level}: "
                         f"peak={peak_kw:.0f} kW @ {peak_dt}")

            # Bookend: 100%
            total_kw = aggregate_demand(instance_counts, profiles_full)
            peak_kw, peak_dt, annual_mwh = summarize_demand(total_kw)
            rows.append({
                "tariff": tariff,
                "dimension": "electrification",
                "other_axis": f"solar_{solar_level}",
                "pct": 100,
                "pct_label": "heat_100",
                "peak_kW": round(peak_kw, 1),
                "peak_date": peak_dt.strftime("%b %d"),
                "peak_time": peak_dt.strftime("%H:%M"),
                "annual_MWh": round(annual_mwh, 1),
            })

        # --- PV penetration interpolation ---
        for heat_level in [0, 100]:
            base_dir, _ = find_bookend_case(
                decarb_root, tariff, heat_level, 0)
            full_dir, _ = find_bookend_case(
                decarb_root, tariff, heat_level, 100)

            if base_dir is None or full_dir is None:
                log.warning(f"Missing bookends for {tariff} PV @ heat={heat_level}")
                continue

            profiles_base = _load_profiles(base_dir)
            profiles_full = _load_profiles(full_dir)

            # Bookend: 0% (= baseline)
            total_kw = aggregate_demand(instance_counts, profiles_base)
            peak_kw, peak_dt, annual_mwh = summarize_demand(total_kw)
            rows.append({
                "tariff": tariff,
                "dimension": "pv_penetration",
                "other_axis": f"heat_{heat_level}",
                "pct": round(pv_base_pct, 1),
                "pct_label": f"solar_0 (baseline {pv_base_pct:.1f}%)",
                "peak_kW": round(peak_kw, 1),
                "peak_date": peak_dt.strftime("%b %d"),
                "peak_time": peak_dt.strftime("%H:%M"),
                "annual_MWh": round(annual_mwh, 1),
            })

            # Intermediate levels
            for pct in pv_steps:
                switched = pv_batches[pct]
                total_kw = aggregate_demand(
                    instance_counts, profiles_base, profiles_full, switched)
                peak_kw, peak_dt, annual_mwh = summarize_demand(total_kw)
                rows.append({
                    "tariff": tariff,
                    "dimension": "pv_penetration",
                    "other_axis": f"heat_{heat_level}",
                    "pct": pct,
                    "pct_label": f"solar_{pct}",
                    "peak_kW": round(peak_kw, 1),
                    "peak_date": peak_dt.strftime("%b %d"),
                    "peak_time": peak_dt.strftime("%H:%M"),
                    "annual_MWh": round(annual_mwh, 1),
                })
                log.info(f"  {tariff} heat_{heat_level} solar_{pct}: "
                         f"peak={peak_kw:.0f} kW @ {peak_dt}")

            # Bookend: 100%
            total_kw = aggregate_demand(instance_counts, profiles_full)
            peak_kw, peak_dt, annual_mwh = summarize_demand(total_kw)
            rows.append({
                "tariff": tariff,
                "dimension": "pv_penetration",
                "other_axis": f"heat_{heat_level}",
                "pct": 100,
                "pct_label": "solar_100",
                "peak_kW": round(peak_kw, 1),
                "peak_date": peak_dt.strftime("%b %d"),
                "peak_time": peak_dt.strftime("%H:%M"),
                "annual_MWh": round(annual_mwh, 1),
            })

    # ── Save summary ────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_path = study_dir / "interpolation_summary.csv"
    df.to_csv(str(out_path), index=False)
    log.info(f"\nSaved: {out_path}  ({len(df)} rows)")

    # Print summary table
    print(f"\n{'='*90}")
    print(df.to_string(index=False))
    print(f"\n{'='*90}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    ncfg = load_network_config()

    study_num = None
    if len(sys.argv) > 1:
        study_num = int(sys.argv[1])

    from create_network_inputs import find_study_dir
    output_dir = Path(cfg["paths"]["output_dir"])
    study_dir = find_study_dir(output_dir, study_num)
    log.info(f"Study: {study_dir.name}")

    interpolate_study_demand(cfg, ncfg, study_dir)
