"""
OpenDSS Results Visualization
──────────────────────────────────────────────────────────────────────────────
Generates three types of plots from OpenDSS power flow results:

1. Per-case zone overview (saved to each case's results subfolder)
2. Cross-tariff comparison (same electrification & solar, varying tariff)
3. Cross-scenario comparison (same tariff, varying electrification or solar)

Usage:
    python src/plot_results.py --study 1
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from config.loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ZONE_NAME = "P100U"
XLABEL = "Day of year"
XDIV = 24.0

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860",
          "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#A0522D", "#E377C2"]


def short_name(feeder_name):
    return feeder_name.split("--")[-1]


def parse_case_name(case_name):
    """Parse '{tariff}_heat_{elec}_solar_{solar}[_percent_{pct}]' into components."""
    m = re.match(r"(.+)_heat_(\d+)_solar_(\d+)(?:_percent_(\d+))?$", case_name)
    if not m:
        return None
    result = {
        "tariff": m.group(1),
        "elec": int(m.group(2)),
        "solar": int(m.group(3)),
    }
    if m.group(4) is not None:
        result["percent"] = int(m.group(4))
    return result


def load_case(results_root, case_name):
    csv_path = results_root / case_name / "zone_yearly.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(str(csv_path))


def hourly_zone_agg(df):
    """Aggregate across feeders to zone-level hourly metrics."""
    return df.groupby("hour").agg(
        v_min=("v_min_pu", "min"),
        v_max=("v_max_pu", "max"),
        v_mean=("v_mean_pu", "mean"),
        total_kW=("total_kW", "sum"),
        total_kvar=("total_kvar", "sum"),
        peak_loading=("head_loading_pct", "max"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Per-case zone overview (4-panel)
# ═══════════════════════════════════════════════════════════════════════════

def plot_case_overview(df, case_name, out_dir):
    """4-panel zone overview: voltage, substation load, zone power, feeder loading."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    x_hours = sorted(df["hour"].unique())
    x = [h / XDIV for h in x_hours]

    # Panel 1: voltage envelope
    ax = axes[0, 0]
    by_hour = df.groupby("hour").agg(
        v_min=("v_min_pu", "min"), v_max=("v_max_pu", "max"), v_mean=("v_mean_pu", "mean")
    ).reindex(x_hours)
    ax.plot(x, by_hour["v_mean"], color="#4C72B0", lw=1.2, label="Mean")
    ax.fill_between(x, by_hour["v_min"], by_hour["v_max"], color="#4C72B0", alpha=0.25)
    ax.axhline(1.05, color="red", ls="--", lw=0.8, label="ANSI ±5%")
    ax.axhline(0.95, color="red", ls="--", lw=0.8)
    ax.set_xlim(0, 365); ax.set_xlabel(XLABEL); ax.set_ylabel("Voltage (pu)")
    ax.set_title("Zone-wide voltage envelope")
    ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=7)

    # Panel 2: total kW per substation
    ax = axes[0, 1]
    subs = sorted(df["substation"].unique())
    for i, sub in enumerate(subs):
        sub_kw = df[df["substation"] == sub].groupby("hour")["total_kW"].sum().reindex(x_hours)
        ax.plot(x, sub_kw, color=COLORS[i % len(COLORS)], lw=1.0, label=sub)
    ax.set_xlim(0, 365); ax.set_xlabel(XLABEL); ax.set_ylabel("Active power (kW)")
    ax.set_title("Total load per substation")
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=5, ncol=2)

    # Panel 3: zone total kW and kvar
    ax = axes[1, 0]
    zone_kw = df.groupby("hour")["total_kW"].sum().reindex(x_hours)
    zone_kvar = df.groupby("hour")["total_kvar"].sum().reindex(x_hours)
    ax.plot(x, zone_kw, color="darkgreen", lw=1.2, label="kW")
    ax.plot(x, zone_kvar, color="purple", lw=1.2, label="kvar")
    ax.set_xlim(0, 365); ax.set_xlabel(XLABEL); ax.set_ylabel("Power")
    ax.set_title("Zone total load")
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=7)

    # Panel 4: top feeder loading
    ax = axes[1, 1]
    peak_loading = df.groupby("feeder")["head_loading_pct"].max().sort_values(ascending=False)
    for i, feeder in enumerate(peak_loading.index[:8]):
        fd = df[df["feeder"] == feeder].set_index("hour")["head_loading_pct"].reindex(x_hours)
        ax.plot(x, fd, color=COLORS[i % len(COLORS)], lw=1.0, label=short_name(feeder))
    ax.axhline(100.0, color="red", ls="--", lw=0.8, label="100%")
    ax.set_xlim(0, 365); ax.set_ylim(bottom=0)
    ax.set_xlabel(XLABEL); ax.set_ylabel("Loading (%)")
    ax.set_title("Head-line loading — top feeders")
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=5, ncol=2)

    n_subs = df["substation"].nunique()
    n_feeders = df["feeder"].nunique()
    fig.suptitle(f"Zone {ZONE_NAME} — {case_name} ({n_subs} subs, {n_feeders} feeders)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_dir / "zone_overview.png"), dpi=150)
    plt.close(fig)
    log.info(f"  {case_name}/zone_overview.png")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Cross-tariff comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_tariff_comparison(results_root, cases_by_tariff, elec, solar, out_dir):
    """Compare tariffs at the same electrification/solar level.

    4 panels: voltage mean, zone kW, zone kvar, peak feeder loading.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for i, (tariff, case_name) in enumerate(sorted(cases_by_tariff.items())):
        df = load_case(results_root, case_name)
        if df is None:
            continue
        agg = hourly_zone_agg(df)
        x = agg.index / XDIV
        c = COLORS[i % len(COLORS)]

        axes[0, 0].plot(x, agg["v_mean"], color=c, lw=1.0, label=tariff)
        axes[0, 1].plot(x, agg["total_kW"], color=c, lw=1.0, label=tariff)
        axes[1, 0].plot(x, agg["total_kvar"], color=c, lw=1.0, label=tariff)
        axes[1, 1].plot(x, agg["peak_loading"], color=c, lw=1.0, label=tariff)

    axes[0, 0].axhline(1.05, color="red", ls="--", lw=0.8)
    axes[0, 0].axhline(0.95, color="red", ls="--", lw=0.8)
    axes[1, 1].axhline(100.0, color="red", ls="--", lw=0.8)

    titles = ["Mean voltage (pu)", "Zone total kW", "Zone total kvar", "Peak feeder loading (%)"]
    ylabels = ["Voltage (pu)", "kW", "kvar", "Loading (%)"]
    for idx, ax in enumerate(axes.flat):
        ax.set_xlim(0, 365); ax.set_xlabel(XLABEL)
        ax.set_ylabel(ylabels[idx]); ax.set_title(titles[idx])
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle(f"Tariff comparison — heat {elec}% / solar {solar}%", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"tariff_comparison_heat_{elec}_solar_{solar}.png"
    fig.savefig(str(out_dir / fname), dpi=150)
    plt.close(fig)
    log.info(f"  {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Cross-scenario comparison (electrification or solar under same tariff)
# ═══════════════════════════════════════════════════════════════════════════

def plot_scenario_comparison(results_root, case_pairs, tariff, vary_label,
                             vary_key, fixed_label, fixed_val, out_dir):
    """Compare scenarios varying one dimension (elec or solar) under one tariff.

    case_pairs: list of (label, case_name) tuples
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for i, (label, case_name) in enumerate(case_pairs):
        df = load_case(results_root, case_name)
        if df is None:
            continue
        agg = hourly_zone_agg(df)
        x = agg.index / XDIV
        c = COLORS[i % len(COLORS)]

        axes[0, 0].plot(x, agg["v_mean"], color=c, lw=1.0, label=label)
        axes[0, 1].plot(x, agg["total_kW"], color=c, lw=1.0, label=label)
        axes[1, 0].plot(x, agg["total_kvar"], color=c, lw=1.0, label=label)
        axes[1, 1].plot(x, agg["peak_loading"], color=c, lw=1.0, label=label)

    axes[0, 0].axhline(1.05, color="red", ls="--", lw=0.8)
    axes[0, 0].axhline(0.95, color="red", ls="--", lw=0.8)
    axes[1, 1].axhline(100.0, color="red", ls="--", lw=0.8)

    titles = ["Mean voltage (pu)", "Zone total kW", "Zone total kvar", "Peak feeder loading (%)"]
    ylabels = ["Voltage (pu)", "kW", "kvar", "Loading (%)"]
    for idx, ax in enumerate(axes.flat):
        ax.set_xlim(0, 365); ax.set_xlabel(XLABEL)
        ax.set_ylabel(ylabels[idx]); ax.set_title(titles[idx])
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle(f"{tariff} — {vary_label} comparison ({fixed_label} {fixed_val}%)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"{tariff}_{vary_key}_comparison_{fixed_label}_{fixed_val}.png"
    fig.savefig(str(out_dir / fname), dpi=150)
    plt.close(fig)
    log.info(f"  {fname}")


def plot_scenario_peak_day(results_root, case_pairs, tariff, vary_label,
                           vary_key, fixed_label, fixed_val, out_dir):
    """Peak day comparison varying one dimension (elec or solar) under one tariff."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for i, (label, case_name) in enumerate(case_pairs):
        df = load_case(results_root, case_name)
        if df is None:
            continue
        day = find_peak_day(df)
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
        c = COLORS[i % len(COLORS)]
        month_day = pd.Timestamp("2018-01-01") + pd.Timedelta(days=day)
        full_label = f"{label} (day {day}, {month_day.strftime('%b %d')})"

        axes[0, 0].plot(x, agg["v_mean"], color=c, lw=1.2, label=full_label)
        axes[0, 1].plot(x, agg["total_kW"], color=c, lw=1.2, label=full_label)
        axes[1, 0].plot(x, agg["total_kvar"], color=c, lw=1.2, label=full_label)
        axes[1, 1].plot(x, agg["peak_loading"], color=c, lw=1.2, label=full_label)

    axes[0, 0].axhline(1.05, color="red", ls="--", lw=0.8)
    axes[0, 0].axhline(0.95, color="red", ls="--", lw=0.8)
    axes[1, 1].axhline(100.0, color="red", ls="--", lw=0.8)

    titles = ["Mean voltage (pu)", "Zone total kW", "Zone total kvar", "Peak feeder loading (%)"]
    ylabels = ["Voltage (pu)", "kW", "kvar", "Loading (%)"]
    for idx, ax in enumerate(axes.flat):
        ax.set_xlim(0, 23); ax.set_xlabel("Hour")
        ax.set_ylabel(ylabels[idx]); ax.set_title(titles[idx])
        ax.grid(alpha=0.3); ax.legend(fontsize=6)

    fig.suptitle(f"{tariff} — {vary_label} peak day ({fixed_label} {fixed_val}%)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"peak_day_{tariff}_{vary_key}_{fixed_label}_{fixed_val}.png"
    fig.savefig(str(out_dir / fname), dpi=150)
    plt.close(fig)
    log.info(f"  {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Peak day plots
# ═══════════════════════════════════════════════════════════════════════════

def find_peak_day(df):
    """Find the day (0-364) with the highest total zone kW."""
    zone_kw = df.groupby("hour")["total_kW"].sum()
    peak_hour = int(zone_kw.idxmax())
    return peak_hour // 24


def get_day_slice(df, day):
    """Extract 24 hours for a given day (0-indexed)."""
    h_start = day * 24
    h_end = h_start + 24
    return df[(df["hour"] >= h_start) & (df["hour"] < h_end)].copy()


def plot_peak_day(df, case_name, day, out_dir):
    """4-panel peak-day detail: voltage, per-substation kW, zone kW+kvar, feeder loading."""
    day_df = get_day_slice(df, day)
    if day_df.empty:
        return
    hours = sorted(day_df["hour"].unique())
    x = [h % 24 for h in hours]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Voltage envelope
    ax = axes[0, 0]
    by_hour = day_df.groupby("hour").agg(
        v_min=("v_min_pu", "min"), v_max=("v_max_pu", "max"), v_mean=("v_mean_pu", "mean")
    ).reindex(hours)
    ax.plot(x, by_hour["v_mean"], color="#4C72B0", lw=1.5, label="Mean")
    ax.fill_between(x, by_hour["v_min"], by_hour["v_max"], color="#4C72B0", alpha=0.25)
    ax.axhline(1.05, color="red", ls="--", lw=0.8, label="ANSI ±5%")
    ax.axhline(0.95, color="red", ls="--", lw=0.8)
    ax.set_xlim(0, 23); ax.set_xlabel("Hour"); ax.set_ylabel("Voltage (pu)")
    ax.set_title("Voltage envelope"); ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # Substation kW
    ax = axes[0, 1]
    subs = sorted(day_df["substation"].unique())
    for i, sub in enumerate(subs):
        sub_kw = day_df[day_df["substation"] == sub].groupby("hour")["total_kW"].sum().reindex(hours)
        ax.plot(x, sub_kw, color=COLORS[i % len(COLORS)], lw=1.0, label=sub)
    ax.set_xlim(0, 23); ax.set_xlabel("Hour"); ax.set_ylabel("kW")
    ax.set_title("Substation load"); ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=5, ncol=2)

    # Zone kW + kvar
    ax = axes[1, 0]
    zone_kw = day_df.groupby("hour")["total_kW"].sum().reindex(hours)
    zone_kvar = day_df.groupby("hour")["total_kvar"].sum().reindex(hours)
    ax.plot(x, zone_kw, color="darkgreen", lw=1.5, label="kW")
    ax.plot(x, zone_kvar, color="purple", lw=1.5, label="kvar")
    ax.set_xlim(0, 23); ax.set_xlabel("Hour"); ax.set_ylabel("Power")
    ax.set_title("Zone total"); ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # Top feeder loading
    ax = axes[1, 1]
    peak_loading = day_df.groupby("feeder")["head_loading_pct"].max().sort_values(ascending=False)
    for i, feeder in enumerate(peak_loading.index[:8]):
        fd = day_df[day_df["feeder"] == feeder].set_index("hour")["head_loading_pct"].reindex(hours)
        ax.plot(x, fd, color=COLORS[i % len(COLORS)], lw=1.0, label=short_name(feeder))
    ax.axhline(100.0, color="red", ls="--", lw=0.8)
    ax.set_xlim(0, 23); ax.set_ylim(bottom=0)
    ax.set_xlabel("Hour"); ax.set_ylabel("Loading (%)")
    ax.set_title("Top feeder loading"); ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=5, ncol=2)

    month_day = pd.Timestamp(f"2018-01-01") + pd.Timedelta(days=day)
    date_str = month_day.strftime("%b %d")
    fig.suptitle(f"Peak day — {case_name} (day {day}, {date_str})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_dir / "peak_day.png"), dpi=150)
    plt.close(fig)
    log.info(f"  {case_name}/peak_day.png (day {day}, {date_str})")


def plot_tariff_peak_day(results_root, cases_by_tariff, elec, solar, out_dir):
    """Compare tariffs on each tariff's own peak day — overlaid on one plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for i, (tariff, case_name) in enumerate(sorted(cases_by_tariff.items())):
        df = load_case(results_root, case_name)
        if df is None:
            continue
        day = find_peak_day(df)
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
        c = COLORS[i % len(COLORS)]
        month_day = pd.Timestamp("2018-01-01") + pd.Timedelta(days=day)
        label = f"{tariff} (day {day}, {month_day.strftime('%b %d')})"

        axes[0, 0].plot(x, agg["v_mean"], color=c, lw=1.2, label=label)
        axes[0, 1].plot(x, agg["total_kW"], color=c, lw=1.2, label=label)
        axes[1, 0].plot(x, agg["total_kvar"], color=c, lw=1.2, label=label)
        axes[1, 1].plot(x, agg["peak_loading"], color=c, lw=1.2, label=label)

    axes[0, 0].axhline(1.05, color="red", ls="--", lw=0.8)
    axes[0, 0].axhline(0.95, color="red", ls="--", lw=0.8)
    axes[1, 1].axhline(100.0, color="red", ls="--", lw=0.8)

    titles = ["Mean voltage (pu)", "Zone total kW", "Zone total kvar", "Peak feeder loading (%)"]
    ylabels = ["Voltage (pu)", "kW", "kvar", "Loading (%)"]
    for idx, ax in enumerate(axes.flat):
        ax.set_xlim(0, 23); ax.set_xlabel("Hour")
        ax.set_ylabel(ylabels[idx]); ax.set_title(titles[idx])
        ax.grid(alpha=0.3); ax.legend(fontsize=6)

    fig.suptitle(f"Peak day comparison — heat {elec}% / solar {solar}%", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"peak_day_tariff_heat_{elec}_solar_{solar}.png"
    fig.savefig(str(out_dir / fname), dpi=150)
    plt.close(fig)
    log.info(f"  {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="OpenDSS Results Visualization")
    parser.add_argument("--study", type=int, default=None, help="Study number")
    args = parser.parse_args()

    cfg = load_config()
    output_dir = cfg["paths"]["output_dir"]
    studies_root = output_dir / "studies"

    if args.study is not None:
        study_dir = studies_root / f"study_{args.study}"
    else:
        study_dir = sorted(
            [d for d in studies_root.iterdir() if d.is_dir() and d.name.startswith("study_")],
            key=lambda d: int(d.name.split("_")[1])
        )[-1]

    results_root = study_dir / "opendss_results"
    if not results_root.exists():
        log.error(f"No results directory: {results_root}")
        return

    # Discover completed cases
    all_cases = {}
    for d in sorted(results_root.iterdir()):
        if d.is_dir() and (d / "zone_yearly.csv").exists():
            parsed = parse_case_name(d.name)
            if parsed:
                all_cases[d.name] = parsed

    if not all_cases:
        log.error("No completed cases found")
        return

    log.info(f"Study: {study_dir.name} — {len(all_cases)} completed cases")

    # Comparison output directory
    comparison_dir = results_root / "comparisons"
    comparison_dir.mkdir(exist_ok=True)

    # ── 1. Per-case zone overview + peak day ────────────────────────────
    log.info("=== Per-case zone overviews + peak days ===")
    for case_name in all_cases:
        df = load_case(results_root, case_name)
        if df is not None:
            plot_case_overview(df, case_name, results_root / case_name)
            day = find_peak_day(df)
            plot_peak_day(df, case_name, day, results_root / case_name)

    # ── 2. Cross-tariff comparisons ──────────────────────────────────────
    log.info("=== Cross-tariff comparisons ===")
    # Group cases by (elec, solar) bookend
    bookends = {}
    for case_name, info in all_cases.items():
        key = (info["elec"], info["solar"], info.get("percent"))
        if key not in bookends:
            bookends[key] = {}
        tariff_label = info["tariff"]
        if info.get("percent") is not None:
            tariff_label = f"{tariff_label} ({info['percent']}%)"
        bookends[key][tariff_label] = case_name

    for (elec, solar, pct), tariff_cases in sorted(bookends.items(),
                                                    key=lambda x: x[0]):
        if len(tariff_cases) > 1:
            plot_tariff_comparison(results_root, tariff_cases, elec, solar, comparison_dir)
            plot_tariff_peak_day(results_root, tariff_cases, elec, solar, comparison_dir)

    # ── 3. Cross-scenario comparisons ────────────────────────────────────
    log.info("=== Cross-scenario comparisons ===")
    tariffs = sorted(set(info["tariff"] for info in all_cases.values()))
    elec_levels = sorted(set(info["elec"] for info in all_cases.values()))
    solar_levels = sorted(set(info["solar"] for info in all_cases.values()))
    percent_levels = sorted(set(info["percent"] for info in all_cases.values()
                                if info.get("percent") is not None))

    for tariff in tariffs:
        # Electrification comparison (fixed solar, fixed percent)
        for solar in solar_levels:
            pairs = []
            for case_name, info in sorted(all_cases.items()):
                if (info["tariff"] == tariff and info["solar"] == solar
                        and info.get("percent") is None):
                    pairs.append((f"heat {info['elec']}%", case_name))
            if len(pairs) > 1:
                plot_scenario_comparison(
                    results_root, pairs, tariff,
                    "Electrification", "elec", "solar", solar,
                    comparison_dir)
                plot_scenario_peak_day(
                    results_root, pairs, tariff,
                    "Electrification", "elec", "solar", solar,
                    comparison_dir)

        # Solar comparison (fixed electrification)
        for elec in elec_levels:
            pairs = []
            for case_name, info in sorted(all_cases.items()):
                if (info["tariff"] == tariff and info["elec"] == elec
                        and info.get("percent") is None):
                    pairs.append((f"solar {info['solar']}%", case_name))
            if len(pairs) > 1:
                plot_scenario_comparison(
                    results_root, pairs, tariff,
                    "Solar", "solar", "heat", elec,
                    comparison_dir)
                plot_scenario_peak_day(
                    results_root, pairs, tariff,
                    "Solar", "solar", "heat", elec,
                    comparison_dir)

        # HP sizing comparison (fixed elec and solar, varying percent)
        if percent_levels:
            for elec in elec_levels:
                for solar in solar_levels:
                    pairs = []
                    for case_name, info in sorted(all_cases.items()):
                        if (info["tariff"] == tariff and info["elec"] == elec
                                and info["solar"] == solar
                                and info.get("percent") is not None):
                            pairs.append((f"HP {info['percent']}%", case_name))
                    if len(pairs) > 1:
                        plot_scenario_comparison(
                            results_root, pairs, tariff,
                            "HP Sizing", "percent",
                            f"heat_{elec}_solar", solar,
                            comparison_dir)
                        plot_scenario_peak_day(
                            results_root, pairs, tariff,
                            "HP Sizing", "percent",
                            f"heat_{elec}_solar", solar,
                            comparison_dir)

    log.info(f"All plots saved to {results_root}")


if __name__ == "__main__":
    main()
