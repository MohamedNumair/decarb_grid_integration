"""
Compare OpenDSS results across DECARB updates (update_0, update_1, update_2).
───────────────────────────────────────────────────────────────────────────────
For each base case that has _update_1 / _update_2 counterparts, generates:

  1. Yearly overlay  — 4-panel (voltage, zone kW, zone kvar, peak loading)
  2. Peak-day overlay — same 4 panels zoomed to each update's peak day
  3. Annual summary   — CSV with totals and % change vs update_0

Usage:
    python src/compare_updates.py --study 13
    python src/compare_updates.py --study 13 --case flat_cap_heat_0_solar_0
───────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
try:
    from config.loader import load_config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from config.loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

XLABEL = "Day of year"
XDIV = 24.0

UPDATE_LABELS = {
    0: "Baseline (update_0)",
    1: "ASHP only (update_1)",
    2: "ASHP + backup (update_2)",
}

UPDATE_COLORS = {
    0: "#4C72B0",
    1: "#DD8452",
    2: "#55A868",
}


def find_study_dir(output_dir, study_num=None):
    studies_root = Path(output_dir) / "studies"
    if study_num is not None:
        d = studies_root / f"study_{study_num}"
        if not d.exists():
            raise FileNotFoundError(f"Study not found: {d}")
        return d
    dirs = sorted(
        [d for d in studies_root.iterdir() if d.is_dir() and d.name.startswith("study_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    if not dirs:
        raise FileNotFoundError("No studies found")
    return dirs[-1]


def load_case(results_root, case_dir_name):
    csv_path = results_root / case_dir_name / "zone_yearly.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(str(csv_path))


def hourly_zone_agg(df):
    return df.groupby("hour").agg(
        v_min=("v_min_pu", "min"),
        v_max=("v_max_pu", "max"),
        v_mean=("v_mean_pu", "mean"),
        total_kW=("total_kW", "sum"),
        total_kvar=("total_kvar", "sum"),
        peak_loading=("head_loading_pct", "max"),
    )


def find_peak_day(df):
    zone_kw = df.groupby("hour")["total_kW"].sum()
    peak_hour = int(zone_kw.idxmax())
    return peak_hour // 24


def get_day_slice(df, day):
    h_start = day * 24
    h_end = h_start + 24
    return df[(df["hour"] >= h_start) & (df["hour"] < h_end)].copy()


def short_name(feeder_name):
    return feeder_name.split("--")[-1]


def discover_update_groups(results_root, case_filter=None):
    """Find base cases and their _update_N counterparts.

    Returns dict: {base_case_name: {0: dir_name, 1: dir_name, 2: dir_name}}
    """
    completed = set()
    for d in results_root.iterdir():
        if d.is_dir() and (d / "zone_yearly.csv").exists():
            completed.add(d.name)

    # Find base cases (no _update_ suffix)
    base_cases = sorted(c for c in completed if "_update_" not in c)

    groups = {}
    for base in base_cases:
        if case_filter and base != case_filter:
            continue
        updates = {0: base}
        for n in [1, 2]:
            suffixed = f"{base}_update_{n}"
            if suffixed in completed:
                updates[n] = suffixed
        if len(updates) > 1:
            groups[base] = updates

    return groups


# ═══════════════════════════════════════════════════════════════════════════
# 1. Yearly overlay
# ═══════════════════════════════════════════════════════════════════════════

def weekly_peak(series, hours_per_week=168):
    """Downsample an hourly series to weekly peaks."""
    n_weeks = len(series) // hours_per_week
    trimmed = series.values[:n_weeks * hours_per_week]
    peaks = trimmed.reshape(n_weeks, hours_per_week).max(axis=1)
    return np.arange(n_weeks), peaks


def weekly_min(series, hours_per_week=168):
    """Downsample an hourly series to weekly minimums."""
    n_weeks = len(series) // hours_per_week
    trimmed = series.values[:n_weeks * hours_per_week]
    mins = trimmed.reshape(n_weeks, hours_per_week).min(axis=1)
    return np.arange(n_weeks), mins


def plot_yearly_comparison(results_root, base_case, updates, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    for update_num, dir_name in sorted(updates.items()):
        df = load_case(results_root, dir_name)
        if df is None:
            continue
        agg = hourly_zone_agg(df)
        c = UPDATE_COLORS[update_num]
        label = UPDATE_LABELS.get(update_num, f"update_{update_num}")

        # Voltage: weekly max (solid) and weekly min (dotted)
        w, vals = weekly_peak(agg["v_max"], 168)
        axes[0, 0].plot(w, vals, color=c, lw=1.2, label=f"{label} (max)")
        w, vals = weekly_min(agg["v_min"], 168)
        axes[0, 0].plot(w, vals, color=c, lw=0.8, ls=":", label=f"{label} (min)")

        # Load: weekly max (solid) and weekly min (dotted)
        w, vals = weekly_peak(agg["total_kW"] / 1000, 168)
        axes[0, 1].plot(w, vals, color=c, lw=1.2, label=f"{label} (max)")
        w, vals = weekly_min(agg["total_kW"] / 1000, 168)
        axes[0, 1].plot(w, vals, color=c, lw=0.8, ls=":", label=f"{label} (min)")

        # Reactive: weekly max (solid) and weekly min (dotted)
        w, vals = weekly_peak(agg["total_kvar"] / 1000, 168)
        axes[1, 0].plot(w, vals, color=c, lw=1.2, label=f"{label} (max)")
        w, vals = weekly_min(agg["total_kvar"] / 1000, 168)
        axes[1, 0].plot(w, vals, color=c, lw=0.8, ls=":", label=f"{label} (min)")

        # Loading: weekly max (solid) and weekly min (dotted)
        w, vals = weekly_peak(agg["peak_loading"], 168)
        axes[1, 1].plot(w, vals, color=c, lw=1.2, label=f"{label} (max)")
        w, vals = weekly_min(agg["peak_loading"], 168)
        axes[1, 1].plot(w, vals, color=c, lw=0.8, ls=":", label=f"{label} (min)")

    titles = ["Weekly voltage envelope (pu)", "Weekly load envelope (MW)",
              "Weekly reactive envelope (Mvar)", "Weekly feeder loading envelope (%)"]
    ylabels = ["Voltage (pu)", "MW", "Mvar", "Loading (%)"]
    for idx, ax in enumerate(axes.flat):
        ax.set_xlim(0, 52); ax.set_xlabel("Week of year")
        ax.set_ylabel(ylabels[idx]); ax.set_title(titles[idx])
        ax.grid(alpha=0.3)

    # Collect data handles BEFORE adding reference lines
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # ANSI ±5% on voltage subplot
    h1 = axes[0, 0].axhline(1.05, color="red", ls="--", lw=0.8, label="ANSI +5%")
    h2 = axes[0, 0].axhline(0.95, color="red", ls="--", lw=0.8, label="ANSI -5%")
    axes[0, 0].legend(handles=[h1, h2], fontsize=7, loc="best")

    # 100% on loading subplot
    h3 = axes[1, 1].axhline(100.0, color="red", ls="--", lw=0.8, label="100% loading")
    axes[1, 1].legend(handles=[h3], fontsize=7, loc="best")

    # Shared legend for data series
    leg_ax = fig.add_axes([0.0, 0.0, 1.0, 0.08])
    leg_ax.axis("off")
    leg_ax.legend(handles, labels, loc="center",
                  ncol=len(updates), fontsize=9, frameon=True, framealpha=0.9)

    case_titles = {"flat_cap_heat_0_solar_0": "Flat Tariff w/ Cap Charges"}
    title = case_titles.get(base_case, base_case)
    fig.suptitle(f"Update Comparison (Weekly Windows) — {title}", fontsize=13)
    fig.subplots_adjust(bottom=0.12, top=0.93, hspace=0.3, wspace=0.25)
    fig.savefig(str(out_dir / f"update_comparison_{base_case}.png"), dpi=150)
    plt.close(fig)
    log.info(f"  update_comparison_{base_case}.png")


NEW_LABELS = {
    0: "Baseline",
    1: "ASHP only",
    2: "ASHP + backup",
}


def plot_yearly_comparison_new(results_root, base_case, updates, out_dir):
    """Updated version with larger fonts, no 'Weekly' in titles, clean labels."""
    fig, axes = plt.subplots(2, 2, figsize=(22, 11))

    for update_num, dir_name in sorted(updates.items()):
        df = load_case(results_root, dir_name)
        if df is None:
            continue
        agg = hourly_zone_agg(df)
        c = UPDATE_COLORS[update_num]
        label = NEW_LABELS.get(update_num, f"update_{update_num}")

        w, vals = weekly_peak(agg["v_max"], 168)
        axes[0, 0].plot(w, vals, color=c, lw=1.2, label=f"{label} (max)")
        w, vals = weekly_min(agg["v_min"], 168)
        axes[0, 0].plot(w, vals, color=c, lw=0.8, ls=":", label=f"{label} (min)")

        w, vals = weekly_peak(agg["total_kW"] / 1000, 168)
        axes[0, 1].plot(w, vals, color=c, lw=1.2, label=f"{label} (max)")
        w, vals = weekly_min(agg["total_kW"] / 1000, 168)
        axes[0, 1].plot(w, vals, color=c, lw=0.8, ls=":", label=f"{label} (min)")

        w, vals = weekly_peak(agg["total_kvar"] / 1000, 168)
        axes[1, 0].plot(w, vals, color=c, lw=1.2, label=f"{label} (max)")
        w, vals = weekly_min(agg["total_kvar"] / 1000, 168)
        axes[1, 0].plot(w, vals, color=c, lw=0.8, ls=":", label=f"{label} (min)")

        w, vals = weekly_peak(agg["peak_loading"], 168)
        axes[1, 1].plot(w, vals, color=c, lw=1.2, label=f"{label} (max)")
        w, vals = weekly_min(agg["peak_loading"], 168)
        axes[1, 1].plot(w, vals, color=c, lw=0.8, ls=":", label=f"{label} (min)")

    titles = ["Voltage Envelope (pu)", "Load Envelope (MW)",
              "Reactive Envelope (Mvar)", "Feeder Loading Envelope (%)"]
    ylabels = ["Voltage (pu)", "MW", "Mvar", "Loading (%)"]
    for idx, ax in enumerate(axes.flat):
        ax.set_xlim(0, 52)
        ax.set_xlabel("Week of year", fontsize=18)
        ax.set_ylabel(ylabels[idx], fontsize=18)
        ax.set_title(titles[idx], fontsize=20)
        ax.tick_params(labelsize=15)
        ax.grid(alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    h1 = axes[0, 0].axhline(1.05, color="red", ls="--", lw=0.8, label="ANSI +5%")
    h2 = axes[0, 0].axhline(0.95, color="red", ls="--", lw=0.8, label="ANSI -5%")
    axes[0, 0].legend(handles=[h1, h2], fontsize=13, loc="lower right")

    h3 = axes[1, 1].axhline(100.0, color="red", ls="--", lw=0.8, label="100% loading")
    axes[1, 1].legend(handles=[h3], fontsize=13, loc="best")

    leg_ax = fig.add_axes([0.0, 0.0, 1.0, 0.08])
    leg_ax.axis("off")
    leg_ax.legend(handles, labels, loc="center",
                  ncol=len(updates), fontsize=15, frameon=True, framealpha=0.9)

    case_titles = {"flat_cap_heat_0_solar_0": "Flat Tariff w/ Cap Charges"}
    title = case_titles.get(base_case, base_case)
    fig.suptitle(f"Update Comparison (Weekly Windows) — {title}", fontsize=22)
    fig.subplots_adjust(bottom=0.12, top=0.92, left=0.06, right=0.97,
                        hspace=0.3, wspace=0.22)
    fig.savefig(str(out_dir / f"update_comparison_{base_case}_NEW.png"), dpi=150)
    plt.close(fig)
    log.info(f"  update_comparison_{base_case}_NEW.png")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Peak-day overlay
# ═══════════════════════════════════════════════════════════════════════════

def plot_peak_day_comparison(results_root, base_case, updates, out_dir):
    # Line style for accompanying profiles, keyed by whose peak day it is
    ACCOMP_STYLES = {0: "--", 1: "-.", 2: ":"}
    SHORT_LABELS = {0: "Baseline", 1: "ASHP only", 2: "ASHP+backup"}

    # Load all dataframes and find each update's peak day
    loaded = {}
    peak_days = {}
    for update_num, dir_name in sorted(updates.items()):
        df = load_case(results_root, dir_name)
        if df is None:
            continue
        loaded[update_num] = df
        peak_days[update_num] = find_peak_day(df)

    if not loaded:
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # For each update's peak day, plot all updates' profiles on that day
    for owner_num in sorted(peak_days.keys()):
        day = peak_days[owner_num]
        month_day = pd.Timestamp("2018-01-01") + pd.Timedelta(days=day)
        day_str = f"day {day}, {month_day.strftime('%b %d')}"

        for profile_num, df in sorted(loaded.items()):
            day_df = get_day_slice(df, day)
            if day_df.empty:
                continue
            agg = day_df.groupby("hour").agg(
                v_mean=("v_mean_pu", "mean"),
                total_kW=("total_kW", "sum"),
                total_kvar=("total_kvar", "sum"),
                peak_loading=("head_loading_pct", "max"),
            )
            x = [h % 24 for h in agg.index]
            c = UPDATE_COLORS[profile_num]
            profile_label = SHORT_LABELS.get(profile_num, f"upd_{profile_num}")

            if profile_num == owner_num:
                ls, lw = "-", 1.6
                label = f"{profile_label} — own peak ({day_str})"
            else:
                ls = ACCOMP_STYLES[owner_num]
                lw = 1.0
                owner_label = SHORT_LABELS.get(owner_num, f"upd_{owner_num}")
                label = f"{profile_label} @ {owner_label} peak"

            axes[0, 0].plot(x, agg["v_mean"], color=c, ls=ls, lw=lw, label=label)
            axes[0, 1].plot(x, agg["total_kW"] / 1000, color=c, ls=ls, lw=lw, label=label)
            axes[1, 0].plot(x, agg["total_kvar"] / 1000, color=c, ls=ls, lw=lw, label=label)
            axes[1, 1].plot(x, agg["peak_loading"], color=c, ls=ls, lw=lw, label=label)

    titles = ["Mean voltage (pu)", "Zone total load (MW)",
              "Zone total reactive (Mvar)", "Peak feeder loading (%)"]
    ylabels = ["Voltage (pu)", "MW", "Mvar", "Loading (%)"]
    for idx, ax in enumerate(axes.flat):
        ax.set_xlim(0, 23); ax.set_xlabel("Hour")
        ax.set_ylabel(ylabels[idx]); ax.set_title(titles[idx])
        ax.grid(alpha=0.3)

    # Collect data handles BEFORE adding reference lines
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # ANSI ±5% on voltage subplot
    h1 = axes[0, 0].axhline(1.05, color="red", ls="--", lw=0.8, label="ANSI +5%")
    h2 = axes[0, 0].axhline(0.95, color="red", ls="--", lw=0.8, label="ANSI -5%")
    axes[0, 0].legend(handles=[h1, h2], fontsize=7, loc="best")

    # 100% on loading subplot
    h3 = axes[1, 1].axhline(100.0, color="red", ls="--", lw=0.8, label="100% loading")
    axes[1, 1].legend(handles=[h3], fontsize=7, loc="best")

    n_updates = len(peak_days)
    if n_updates > 0:
        leg_ax = fig.add_axes([0.0, 0.0, 1.0, 0.08])
        leg_ax.axis("off")
        leg_ax.legend(handles, labels, loc="center",
                      ncol=n_updates, fontsize=9, frameon=True, framealpha=0.9)

    case_titles = {"flat_cap_heat_0_solar_0": "Flat Tariff w/ Cap Charges"}
    title = case_titles.get(base_case, base_case)
    fig.suptitle(f"Peak Day Comparison — {title}", fontsize=13)
    fig.subplots_adjust(bottom=0.12, top=0.93, hspace=0.3, wspace=0.25)
    fig.savefig(str(out_dir / f"peak_day_updates_{base_case}.png"), dpi=150)
    plt.close(fig)
    log.info(f"  peak_day_updates_{base_case}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Per-feeder comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_feeder_comparison(results_root, base_case, updates, out_dir):
    """Bar chart comparing peak loading and voltage deviation per feeder."""
    feeder_stats = {}

    for update_num, dir_name in sorted(updates.items()):
        df = load_case(results_root, dir_name)
        if df is None:
            continue
        for feeder in df["feeder"].unique():
            fd = df[df["feeder"] == feeder]
            if feeder not in feeder_stats:
                feeder_stats[feeder] = {}
            feeder_stats[feeder][update_num] = {
                "peak_loading": fd["head_loading_pct"].max(),
                "v_min": fd["v_min_pu"].min(),
                "v_max": fd["v_max_pu"].max(),
                "peak_kw": fd.groupby("hour")["total_kW"].sum().max(),
            }

    if not feeder_stats:
        return

    # Sort by baseline peak loading
    feeders_sorted = sorted(
        feeder_stats.keys(),
        key=lambda f: feeder_stats[f].get(0, {}).get("peak_loading", 0),
        reverse=True)
    top_feeders = feeders_sorted[:20]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    bar_width = 0.25
    x_pos = np.arange(len(top_feeders))

    # Peak loading bars
    ax = axes[0]
    for update_num in sorted(updates.keys()):
        vals = [feeder_stats[f].get(update_num, {}).get("peak_loading", 0) for f in top_feeders]
        offset = (update_num - 1) * bar_width
        ax.bar(x_pos + offset, vals, bar_width,
               color=UPDATE_COLORS[update_num],
               label=UPDATE_LABELS.get(update_num, f"update_{update_num}"))
    ax.axhline(100.0, color="red", ls="--", lw=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([short_name(f) for f in top_feeders], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Peak loading (%)")
    ax.set_title("Peak feeder loading by update")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Min voltage bars
    ax = axes[1]
    for update_num in sorted(updates.keys()):
        vals = [feeder_stats[f].get(update_num, {}).get("v_min", 1.0) for f in top_feeders]
        offset = (update_num - 1) * bar_width
        ax.bar(x_pos + offset, vals, bar_width,
               color=UPDATE_COLORS[update_num],
               label=UPDATE_LABELS.get(update_num, f"update_{update_num}"))
    ax.axhline(0.95, color="red", ls="--", lw=0.8, label="ANSI -5%")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([short_name(f) for f in top_feeders], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Min voltage (pu)")
    ax.set_title("Minimum bus voltage by update")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Per-feeder comparison — {base_case} (top 20)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_dir / f"feeder_comparison_{base_case}.png"), dpi=150)
    plt.close(fig)
    log.info(f"  feeder_comparison_{base_case}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Annual summary table
# ═══════════════════════════════════════════════════════════════════════════

def build_annual_summary(results_root, base_case, updates):
    rows = []
    baseline_vals = {}

    for update_num, dir_name in sorted(updates.items()):
        df = load_case(results_root, dir_name)
        if df is None:
            continue
        agg = hourly_zone_agg(df)
        row = {
            "update": f"update_{update_num}",
            "label": UPDATE_LABELS.get(update_num, f"update_{update_num}"),
            "annual_energy_MWh": agg["total_kW"].sum() / 1000,
            "peak_demand_MW": agg["total_kW"].max() / 1000,
            "peak_reactive_Mvar": agg["total_kvar"].max() / 1000,
            "v_min_pu": agg["v_min"].min(),
            "v_max_pu": agg["v_max"].max(),
            "peak_loading_pct": agg["peak_loading"].max(),
            "hours_over_100pct_loading": int((agg["peak_loading"] > 100).sum()),
            "hours_under_0.95pu": int((agg["v_min"] < 0.95).sum()),
            "hours_over_1.05pu": int((agg["v_max"] > 1.05).sum()),
        }

        if update_num == 0:
            baseline_vals = {k: v for k, v in row.items()
                            if isinstance(v, (int, float))}
        elif baseline_vals:
            for key in ["annual_energy_MWh", "peak_demand_MW", "peak_reactive_Mvar",
                        "peak_loading_pct"]:
                bv = baseline_vals.get(key, 0)
                if bv != 0:
                    row[f"{key}_vs_baseline_pct"] = round((row[key] - bv) / bv * 100, 2)

        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compare OpenDSS results across updates")
    parser.add_argument("--study", type=int, default=None, help="Study number (default: latest)")
    parser.add_argument("--case", type=str, default=None,
                        help="Specific base case name (default: all)")
    args = parser.parse_args()

    cfg = load_config()
    output_dir = cfg["paths"]["output_dir"]
    study_dir = find_study_dir(output_dir, args.study)

    results_root = study_dir / "opendss_results"
    if not results_root.exists():
        log.error(f"No results directory: {results_root}")
        return

    groups = discover_update_groups(results_root, case_filter=args.case)

    if not groups:
        log.error("No cases with multiple updates found")
        return

    log.info(f"Study: {study_dir.name} — {len(groups)} case(s) with updates")

    comparison_dir = results_root / "update_comparisons"
    comparison_dir.mkdir(exist_ok=True)

    all_summaries = []

    for base_case, updates in sorted(groups.items()):
        update_list = ", ".join(f"update_{n}" for n in sorted(updates.keys()))
        log.info(f"=== {base_case} ({update_list}) ===")

        plot_yearly_comparison(results_root, base_case, updates, comparison_dir)
        plot_yearly_comparison_new(results_root, base_case, updates, comparison_dir)
        plot_peak_day_comparison(results_root, base_case, updates, comparison_dir)
        plot_feeder_comparison(results_root, base_case, updates, comparison_dir)

        summary = build_annual_summary(results_root, base_case, updates)
        summary.insert(0, "case", base_case)
        all_summaries.append(summary)

        summary_path = comparison_dir / f"annual_summary_{base_case}.csv"
        summary.to_csv(str(summary_path), index=False)
        log.info(f"  {summary_path.name}")

        log.info("\n" + summary.to_string(index=False) + "\n")

    if len(all_summaries) > 1:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_path = comparison_dir / "annual_summary_all.csv"
        combined.to_csv(str(combined_path), index=False)
        log.info(f"Combined summary: {combined_path}")


if __name__ == "__main__":
    main()
