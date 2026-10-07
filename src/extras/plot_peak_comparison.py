"""Peak Shaving Comparison bar charts for a study's OpenDSS results.

Replicates the style from ercot_project/src/decarb_case_comparison_v7.py:
  - Bars colored by tariff family
  - Hatched bars for non-baseline electrification/solar combos
  - Percentage delta labels inside bars
  - Acronym legend at bottom

Usage:
    python plot_peak_comparison.py --study study_8
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parents[1]))
try:
    from config.loader import load_config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from config.loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Tariff colors (matching ercot_project) ──────────────────────────────────
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

HEAT_LABELS = {0: "15%", 100: "100%"}
SOLAR_LABELS = {0: "0%", 100: "100%"}


def parse_case_name(case_name):
    """Parse 'flat_cap_heat_100_solar_0' into (tariff, heat_pct, solar_pct)."""
    parts = case_name.split("_")
    solar_idx = parts.index("solar")
    solar_pct = int(parts[solar_idx + 1])
    heat_idx = parts.index("heat")
    heat_pct = int(parts[heat_idx + 1])
    tariff = "_".join(parts[:heat_idx])
    return tariff, heat_pct, solar_pct


def compute_peak_demand_mw(results_dir, case_name):
    """Read zone_yearly.csv and return system peak demand in MW."""
    csv_path = results_dir / case_name / "zone_yearly.csv"
    if not csv_path.exists():
        log.warning(f"Missing: {csv_path}")
        return np.nan
    df = pd.read_csv(csv_path)
    zone_kw = df.groupby("hour")["total_kW"].sum()
    return zone_kw.max() / 1000.0  # kW → MW


def make_bar_label(tariff, heat_pct, solar_pct, include_solar=True):
    """Build x-tick label like 'Flat 15% ASHP\n0% Solar PV'."""
    t = TARIFF_LABELS.get(tariff, tariff)
    h = HEAT_LABELS.get(heat_pct, f"{heat_pct}%")
    if include_solar:
        s = SOLAR_LABELS.get(solar_pct, f"{solar_pct}%")
        return f"{t} {h} ASHP\n{s} Solar PV"
    return f"{t} {h} ASHP"


def plot_peak_bars(cases, results_dir, baseline_case, output_path,
                   title_suffix="", include_solar=True):
    """Draw peak shaving comparison bar chart.

    cases: list of case_name strings in desired bar order.
    """
    # Compute peaks
    peaks = {}
    for c in cases:
        peaks[c] = compute_peak_demand_mw(results_dir, c)
        log.info(f"  {c}: {peaks[c]:.2f} MW")

    baseline_peak = peaks[baseline_case]

    # Build bar data
    n = len(cases)
    x = np.arange(n)
    vals = [peaks[c] for c in cases]

    colors = []
    hatches = []
    labels = []
    for c in cases:
        tariff, heat_pct, solar_pct = parse_case_name(c)
        colors.append(TARIFF_COLORS.get(tariff, "gray"))
        # Solid for heat_0_solar_0, hatched for everything else
        is_baseline_combo = (heat_pct == 0 and solar_pct == 0)
        hatches.append("" if is_baseline_combo else "//")
        labels.append(make_bar_label(tariff, heat_pct, solar_pct, include_solar))

    fig, ax = plt.subplots(figsize=(max(11, n * 1.3), 7))

    bars = ax.bar(x, vals, 0.7,
                  color=colors, hatch=hatches, edgecolor="white", alpha=0.85)

    # Value labels above bars
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=13)

    # Percentage delta labels inside bars (skip baseline)
    for i, c in enumerate(cases):
        if c != baseline_case and not np.isnan(vals[i]) and not np.isnan(baseline_peak):
            pct = 100 * (vals[i] - baseline_peak) / baseline_peak
            ax.text(bars[i].get_x() + bars[i].get_width() / 2,
                    bars[i].get_height() * 0.95,
                    f"({pct:+.1f}%)", ha="center", va="top", fontsize=12,
                    color="white", fontweight="bold")

    ax.set_title("System Peak Electrical Demand", fontsize=20)
    ax.set_ylabel("Peak Electrical Demand (MW)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    # Suptitle
    bl_tariff, bl_heat, bl_solar = parse_case_name(baseline_case)
    bl_display = f"{TARIFF_LABELS[bl_tariff]} Rate - {HEAT_LABELS[bl_heat]} ASHP Electrification"
    if include_solar:
        bl_display += f" & {SOLAR_LABELS[bl_solar]} Solar Rooftop"
    fig.suptitle(f"Peak Shaving Comparison{title_suffix}\nBaseline: {bl_display}",
                 fontsize=20, fontweight="bold", y=0.98)

    # Acronym legend at bottom
    acronym_handles = [
        Line2D([0], [0], color="none", label="TOU: Time-of-Use Rate"),
        Line2D([0], [0], color="none", label="CC: Capacity Charge"),
        Line2D([0], [0], color="none", label="ASHP: Air-Source Heat Pump Penetration"),
    ]
    if include_solar:
        acronym_handles.append(
            Line2D([0], [0], color="none", label="PV: Rooftop Solar Panel Penetration")
        )
    ax.legend(handles=acronym_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.72), ncol=len(acronym_handles),
              fontsize=13, frameon=True, handlelength=0, handletextpad=0,
              borderpad=0.6)

    fig.subplots_adjust(top=0.84, bottom=0.38, left=0.07, right=0.97)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Peak shaving comparison plots")
    parser.add_argument("--study", required=True, help="Study folder name (e.g. study_8)")
    args = parser.parse_args()

    cfg = load_config()
    project_root = Path(cfg["paths"]["project_root"]).resolve()
    results_dir = project_root / "out" / "studies" / args.study / "opendss_results"

    if not results_dir.exists():
        log.error(f"Results directory not found: {results_dir}")
        return

    out_dir = results_dir / "peak_comparison"
    out_dir.mkdir(exist_ok=True)

    # Discover available cases
    all_cases = sorted([
        d.name for d in results_dir.iterdir()
        if d.is_dir() and (d / "zone_yearly.csv").exists()
    ])
    log.info(f"Found {len(all_cases)} cases with results")

    baseline = "flat_heat_0_solar_0"
    if baseline not in all_cases:
        log.error(f"Baseline case '{baseline}' not found")
        return

    # ── Plot 1: No-solar cases only ──
    no_solar = [c for c in all_cases if c.endswith("_solar_0")]
    # Order: flat, tou, flat_cap, tou_cap; within each: heat_0 then heat_100
    tariff_order = ["flat", "tou", "flat_cap", "tou_cap"]
    no_solar_ordered = []
    for t in tariff_order:
        for h in [0, 100]:
            name = f"{t}_heat_{h}_solar_0"
            if name in no_solar:
                no_solar_ordered.append(name)

    log.info(f"\n=== Plot 1: No-solar cases ({len(no_solar_ordered)} bars) ===")
    plot_peak_bars(
        no_solar_ordered, results_dir, baseline,
        out_dir / "peak_shaving_no_solar.png",
        title_suffix="",
        include_solar=False,
    )

    # ── Plot 2: All cases ──
    all_ordered = []
    for t in tariff_order:
        for h in [0, 100]:
            for s in [0, 100]:
                name = f"{t}_heat_{h}_solar_{s}"
                if name in all_cases:
                    all_ordered.append(name)

    log.info(f"\n=== Plot 2: All cases ({len(all_ordered)} bars) ===")
    plot_peak_bars(
        all_ordered, results_dir, baseline,
        out_dir / "peak_shaving_all_cases.png",
    )


if __name__ == "__main__":
    main()
