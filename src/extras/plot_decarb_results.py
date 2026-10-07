"""
DECARB / Interpolation Results Visualization
──────────────────────────────────────────────────────────────────────────────
Generates plots from interpolation_summary.csv and the OpenDSS bookend
summary.csv:

1. Interpolation peak demand & annual energy vs penetration %
2. Season map (summer vs winter peak at each penetration level)
3. Bookend bar chart (OpenDSS 0/100% cases side by side)

Usage:
    python src/plot_decarb_results.py --study 4
──────────────────────────────────────────────────────────────────────────────
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Shared palette ──────────────────────────────────────────────────────────
TARIFF_COLORS = {
    "flat":     "#4C72B0",
    "flat_cap": "#55A868",
    "tou":      "#DD8452",
    "tou_cap":  "#C44E52",
}
TARIFF_LABELS = {
    "flat":     "Flat",
    "flat_cap": "Flat + CC",
    "tou":      "TOU",
    "tou_cap":  "TOU + CC",
}

WINTER_MONTHS = {"Jan", "Feb", "Mar", "Nov", "Dec"}


# ═══════════════════════════════════════════════════════════════════════════
# 1. Interpolation line charts – peak kW & annual MWh vs penetration %
# ═══════════════════════════════════════════════════════════════════════════

def plot_interpolation_peak(interp_df, dimension, other_axis, out_dir):
    """Line chart: peak kW and annual MWh vs penetration % for each tariff."""
    sub = interp_df[
        (interp_df["dimension"] == dimension) &
        (interp_df["other_axis"] == other_axis)
    ].copy()
    if sub.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    tariffs = sorted(sub["tariff"].unique())
    for tariff in tariffs:
        td = sub[sub["tariff"] == tariff].sort_values("pct")
        c = TARIFF_COLORS.get(tariff, "#666")
        label = TARIFF_LABELS.get(tariff, tariff)

        ax1.plot(td["pct"], td["peak_kW"] / 1000, color=c, lw=2,
                 marker="o", markersize=4, label=label)
        ax2.plot(td["pct"], td["annual_MWh"] / 1000, color=c, lw=2,
                 marker="o", markersize=4, label=label)

    # Mark summer→winter crossover on flat tariff
    flat = sub[sub["tariff"] == "flat"].sort_values("pct")
    if len(flat) > 1:
        for i in range(1, len(flat)):
            prev_m = flat.iloc[i - 1]["peak_date"].split()[0]
            cur_m = flat.iloc[i]["peak_date"].split()[0]
            if prev_m not in WINTER_MONTHS and cur_m in WINTER_MONTHS:
                cross_pct = (flat.iloc[i - 1]["pct"] + flat.iloc[i]["pct"]) / 2
                ax1.axvline(cross_pct, color="#888", ls="--", lw=1, alpha=0.7)
                ax1.text(cross_pct + 1, ax1.get_ylim()[1] * 0.98,
                         "summer→winter\npeak shift",
                         fontsize=8, color="#666", va="top")
                break

    ax1.set_xlabel("Penetration (%)")
    ax1.set_ylabel("Peak Demand (MW)")
    ax1.set_title("System Peak Demand")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.set_xlabel("Penetration (%)")
    ax2.set_ylabel("Annual Energy (GWh)")
    ax2.set_title("Annual Energy Consumption")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)

    dim_label = ("Heating Electrification" if dimension == "electrification"
                 else "PV Penetration")
    axis_label = other_axis.replace("_", " ").title()
    fig.suptitle(f"{dim_label} Interpolation — {axis_label}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fname = f"interpolation_{dimension}_{other_axis}.png"
    fig.savefig(str(out_dir / fname), dpi=150)
    plt.close(fig)
    log.info(f"  saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Season map – summer vs winter peak at each penetration level
# ═══════════════════════════════════════════════════════════════════════════

def plot_interpolation_season_map(interp_df, dimension, out_dir):
    """Peak demand with diamond=winter, circle=summer markers."""
    sub = interp_df[interp_df["dimension"] == dimension].copy()
    if sub.empty:
        return

    sub["is_winter"] = sub["peak_date"].apply(
        lambda d: d.split()[0] in WINTER_MONTHS)

    axes_vals = sorted(sub["other_axis"].unique())
    tariffs = sorted(sub["tariff"].unique())

    fig, axes_arr = plt.subplots(1, len(axes_vals),
                                 figsize=(8 * len(axes_vals), 4))
    if len(axes_vals) == 1:
        axes_arr = [axes_arr]

    for ax, axis_val in zip(axes_arr, axes_vals):
        for tariff in tariffs:
            td = sub[(sub["tariff"] == tariff) &
                     (sub["other_axis"] == axis_val)].sort_values("pct")
            c = TARIFF_COLORS.get(tariff, "#666")
            label = TARIFF_LABELS.get(tariff, tariff)

            for _, row in td.iterrows():
                marker = "D" if row["is_winter"] else "o"
                ax.plot(row["pct"], row["peak_kW"] / 1000, color=c,
                        marker=marker, markersize=6)
            ax.plot(td["pct"], td["peak_kW"] / 1000, color=c, lw=1.5,
                    alpha=0.7, label=label)

        ax.set_xlabel("Penetration (%)")
        ax.set_ylabel("Peak Demand (MW)")
        ax.set_title(axis_val.replace("_", " ").title())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    dim_label = ("Heating Electrification" if dimension == "electrification"
                 else "PV Penetration")
    fig.suptitle(f"{dim_label} — Peak Season (◆=winter, ●=summer)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    fname = f"interpolation_{dimension}_season.png"
    fig.savefig(str(out_dir / fname), dpi=150)
    plt.close(fig)
    log.info(f"  saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Bookend bar chart – all 16 OpenDSS 0/100% cases
# ═══════════════════════════════════════════════════════════════════════════

def plot_bookend_summary(summary_df, out_dir):
    """Grouped bar chart of OpenDSS bookend peak demand across all cases."""
    df = summary_df.copy()
    # Parse case name into tariff and scenario parts
    df["scenario"] = df["case"].apply(
        lambda c: "_".join(c.split("_")[c.split("_").index("heat"):]))
    df["tariff"] = df["case"].apply(lambda c: c.split("_heat_")[0])

    scenarios = sorted(df["scenario"].unique())
    tariffs = sorted(df["tariff"].unique())

    fig, ax = plt.subplots(figsize=(14, 6))
    n_tariffs = len(tariffs)
    bar_w = 0.8 / n_tariffs
    x_pos = np.arange(len(scenarios))

    for i, tariff in enumerate(tariffs):
        td = df[df["tariff"] == tariff]
        vals = []
        for sc in scenarios:
            row = td[td["scenario"] == sc]
            vals.append(row["peak_kW"].values[0] / 1000 if len(row) else 0)
        c = TARIFF_COLORS.get(tariff, "#666")
        label = TARIFF_LABELS.get(tariff, tariff)
        bars = ax.bar(x_pos + i * bar_w - 0.4 + bar_w / 2, vals,
                      bar_w * 0.9, color=c, label=label)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{v:.0f}", ha="center", va="bottom",
                        fontsize=7, color="#444")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=9)
    ax.set_ylabel("Peak Demand (MW)")
    ax.set_title("OpenDSS Peak Demand — All Bookend Cases", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    fname = "bookend_peak_summary.png"
    fig.savefig(str(out_dir / fname), dpi=150)
    plt.close(fig)
    log.info(f"  saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run(study_dir):
    """Generate all decarb / interpolation plots for a study directory."""
    out_dir = study_dir / "decarb_results"
    out_dir.mkdir(exist_ok=True)

    # ── Interpolation summary ───────────────────────────────────────────
    interp_path = out_dir / "interpolation_summary.csv"
    if not interp_path.exists():
        # Fall back to old location
        interp_path = study_dir / "interpolation_summary.csv"

    if interp_path.exists():
        interp_df = pd.read_csv(str(interp_path))
        log.info("=== Interpolation plots ===")
        for dim in interp_df["dimension"].unique():
            for axis in interp_df[
                    interp_df["dimension"] == dim]["other_axis"].unique():
                plot_interpolation_peak(interp_df, dim, axis, out_dir)
            plot_interpolation_season_map(interp_df, dim, out_dir)
    else:
        log.warning("No interpolation_summary.csv — skipping interpolation "
                     "plots")

    # ── Bookend summary ─────────────────────────────────────────────────
    summary_path = study_dir / "opendss_results" / "summary.csv"
    if summary_path.exists():
        log.info("=== Bookend summary ===")
        summary_df = pd.read_csv(str(summary_path))
        plot_bookend_summary(summary_df, out_dir)
    else:
        log.warning("No opendss_results/summary.csv — skipping bookend plot")

    log.info(f"Decarb plots saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="DECARB / Interpolation Results Visualization")
    parser.add_argument("--study", type=int, default=None,
                        help="Study number")
    args = parser.parse_args()

    cfg = load_config()
    output_dir = cfg["paths"]["output_dir"]
    studies_root = output_dir / "studies"

    if args.study is not None:
        study_dir = studies_root / f"study_{args.study}"
    else:
        study_dir = sorted(
            [d for d in studies_root.iterdir()
             if d.is_dir() and d.name.startswith("study_")],
            key=lambda d: int(d.name.split("_")[1])
        )[-1]

    if not study_dir.exists():
        log.error(f"Study directory not found: {study_dir}")
        return

    run(study_dir)


if __name__ == "__main__":
    main()
