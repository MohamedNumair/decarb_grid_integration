"""
Figures for the single-building workshop cases (see build_workshop_cases.py),
in the style of plot_results.py.

  peak_day_by_tariff.png        2x2, one panel per tariff: grid import over the peak
                                day for base / heat pump / HP + backup (hourly, full-year
                                cases). The peak day is the one with the highest daily
                                grid import in heat_pump_flat, the same day in every panel.
  solar_bess_summer_week.png    2x2 by tariff: one July week at 15-min for heat pump +
                                PV + battery: electric load, PV, battery, grid (kW),
                                with the battery state of charge below each panel.
  solar_bess_avg_summer_day.png 2x2 by tariff: average day (Jun-Aug, 15-min) of the
                                same series.

TOU peak hours are shaded. ts.csv values are average kW per period; times are the
start of each period in local time (cfg.csv pP0 / tm.csv pP are UTC).

Usage:
    python src/extras/plot_workshop_cases.py C:/.../decarb/examples/single_bldg_updates
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd              # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1]))
from plot_results import COLORS  # noqa: E402

TARIFFS = ["flat", "flat_cap", "tou", "tou_cap"]
TZ = "America/New_York"
TOU_PEAKS = [(6, 10), (19, 23)]           # local hours, decarb.yaml tou / tou_cap


def load_case(case_dir):
    """ts.csv indexed by local period-start time, with derived kW series."""
    cfg = pd.read_csv(case_dir / "in" / "cfg.csv", header=None, index_col=0)[1]
    tm = pd.read_csv(case_dir / "in" / "tm.csv", usecols=["pP"])
    ts = pd.read_csv(case_dir / "out" / "ts.csv")
    bounds = pd.concat([pd.Series([pd.to_datetime(cfg["pP0"])]), pd.to_datetime(tm.pP)],
                       ignore_index=True)
    ts.index = (pd.DatetimeIndex(bounds.iloc[:-1]).tz_localize("UTC").tz_convert(TZ)
                .tz_localize(None))
    ts["step_h"] = bounds.diff().dropna().dt.total_seconds().to_numpy() / 3600
    ts["load"] = ts.dem + ts.hvac_HT + ts.hvac_AC + ts.Qwh + ts.ev_UP
    ts["battery"] = ts.bess_DN - ts.bess_UP          # + discharging, - charging
    ts["grid"] = ts.buy - ts.sell                    # + import, - export
    return ts


def hour_of_day(index):
    return index.hour + index.minute / 60


def day_of_year(index):
    return index.dayofyear - 1 + hour_of_day(index) / 24


def shade_tou(ax, tariff, day_starts):
    """day_starts: x positions of midnight (0 for hour-of-day axes)."""
    if not tariff.startswith("tou"):
        return
    scale = 1 if day_starts == [0] else 1 / 24
    for d0 in day_starts:
        for a, b in TOU_PEAKS:
            ax.axvspan(d0 + a * scale, d0 + b * scale, color="grey", alpha=0.12, lw=0,
                       label="TOU peak" if (d0, a) == (day_starts[0], TOU_PEAKS[0][0]) else None)


def plot_power(ax, x, d):
    for i, (col, label) in enumerate([("load", "Electric load"), ("pv", "PV"),
                                      ("battery", "Battery (+ discharge)"),
                                      ("grid", "Grid (+ import)")]):
        ax.plot(x, d[col], color=COLORS[i], lw=1.2, label=label)
    ax.axhline(0, color="black", lw=0.6)


def fig_peak_day(folder, out):
    hp = load_case(folder / "heat_pump_flat")
    daily = (hp.buy * hp.step_h).groupby(hp.index.normalize()).sum()
    day = daily.idxmax()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    series = [("base", "Gas furnace + AC"), ("heat_pump", "Heat pump"),
              ("hp_backup", "Heat pump + backup heater")]
    for ax, t in zip(axes.flat, TARIFFS):
        shade_tou(ax, t, [0])
        for i, (prefix, label) in enumerate(series):
            ts = load_case(folder / f"{prefix}_{t}")
            d = ts.loc[day: day + pd.Timedelta(hours=23)]
            ax.plot(hour_of_day(d.index), d.buy, color=COLORS[i], lw=1.5, label=label)
        ax.set_xlim(0, 23); ax.set_xlabel("Hour"); ax.set_ylabel("Grid import (kW)")
        ax.set_title(t); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    bldg = (folder / "base_flat" / "source.txt").read_text().split(":")[0]   # "Building N"
    fig.suptitle(f"{bldg} — peak day by tariff ({day:%b %d})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    return day


def fig_summer_week(folder, out):
    ref = load_case(folder / "hp_solar_bess_tou_15min_summer")
    july = ref.loc["2018-07"]
    hot_day = july.load.groupby(july.index.normalize()).mean().idxmax()
    start = hot_day - pd.Timedelta(days=hot_day.weekday())          # Monday of that week
    end = start + pd.Timedelta(days=7)
    day_starts = [start.dayofyear - 1 + k for k in range(7)]
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 3, 1])
    for i, t in enumerate(TARIFFS):
        r, c = 2 * (i // 2), i % 2
        d = load_case(folder / f"hp_solar_bess_{t}_15min_summer").loc[start: end - pd.Timedelta(minutes=15)]
        x = day_of_year(d.index)
        ax = fig.add_subplot(gs[r, c])
        shade_tou(ax, t, day_starts)
        plot_power(ax, x, d)
        ax.set_xlim(x.min(), x.max() + 0.25 / 24); ax.set_ylabel("kW")
        ax.set_title(t); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
        ax.tick_params(labelbottom=False)
        soc = fig.add_subplot(gs[r + 1, c], sharex=ax)
        soc.plot(x, d.bess_SOC, color=COLORS[2], lw=1.2)
        soc.set_ylim(bottom=0); soc.set_ylabel("Battery SOC (kWh)")
        soc.set_xlabel("Day of year"); soc.grid(alpha=0.3)
    fig.suptitle(f"Heat pump + PV + battery — 15-min, week of {start:%b %d}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    return start


def fig_avg_summer_day(folder, out):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    for ax, t in zip(axes.flat, TARIFFS):
        ts = load_case(folder / f"hp_solar_bess_{t}_15min_summer")
        avg = ts[["load", "pv", "battery", "grid"]].groupby(hour_of_day(ts.index)).mean()
        shade_tou(ax, t, [0])
        plot_power(ax, avg.index, avg)
        ax.set_xlim(0, 24); ax.set_xlabel("Hour"); ax.set_ylabel("Average power (kW)")
        ax.set_title(t); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Heat pump + PV + battery — average summer day (Jun–Aug, 15-min)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder with the workshop cases")
    args = ap.parse_args()
    figdir = args.folder / "figures"
    figdir.mkdir(exist_ok=True)
    day = fig_peak_day(args.folder, figdir / "peak_day_by_tariff.png")
    week = fig_summer_week(args.folder, figdir / "solar_bess_summer_week.png")
    fig_avg_summer_day(args.folder, figdir / "solar_bess_avg_summer_day.png")
    print(f"Peak day: {day:%Y-%m-%d}; summer week from {week:%Y-%m-%d}. Figures in {figdir}")


if __name__ == "__main__":
    main()
