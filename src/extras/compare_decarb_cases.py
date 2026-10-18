"""
Compare DECARB results across cases (e.g. the single-building workshop set).

Reads each case's in/ (cfg.csv, tm.csv) and out/ (eq1.csv, ts.csv) and writes one
row per case to comparison.csv: costs from eq1.csv, plus energy totals, peak grid
import and comfort violations computed from ts.csv. ts.csv columns are average kW
per period; period lengths come from the inputs, so hourly and 15-min cases are
handled alike. Totals cover each case's own period
(e.g. a summer-only case is not a full year).

Usage:
    python src/extras/compare_decarb_cases.py C:/.../decarb/examples/single_bldg_updates
    python src/extras/compare_decarb_cases.py C:/.../decarb/examples/single_bldg_updates --cases base_flat heat_pump_flat
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EQ1_COLS = ["total_cost", "energy_bill", "grid_purchases", "capacity_charge", "fuel_purchases",
            "unserved_thermal_cost", "direct_emissions", "indirect_emissions"]
FUEL_COLS = ["Gchp", "Gabp", "Gwh", "Lchp", "Labp", "Lwh"]


def summarize(case_dir):
    in_dir, out_dir = case_dir / "in", case_dir / "out"
    cfg = pd.read_csv(in_dir / "cfg.csv", header=None, index_col=0)[1]
    tm = pd.read_csv(in_dir / "tm.csv")
    ts = pd.read_csv(out_dir / "ts.csv")
    eq1 = pd.read_csv(out_dir / "eq1.csv").set_index("name")["eq"]
    if len(ts) != len(tm):
        raise ValueError(f"{case_dir.name}: ts.csv has {len(ts)} rows, tm.csv {len(tm)}")

    ends = pd.to_datetime(tm.pP)
    h = ends.diff().dt.total_seconds().to_numpy() / 3600
    h[0] = (ends.iloc[0] - pd.to_datetime(cfg["pP0"])).total_seconds() / 3600

    on = tm.pTon.to_numpy() == 1
    excursion = ((tm.pTmn - ts.Temp).clip(lower=0) + (ts.Temp - tm.pTmx).clip(lower=0)).to_numpy()
    row = {
        "case": case_dir.name,
        "periods": len(tm),
        "step_h": float(np.median(h)),
        "start_utc": cfg["pP0"],
        "grid_import_mwh": float((ts.buy * h).sum()) / 1000,
        "grid_export_mwh": float((ts.sell * h).sum()) / 1000,
        "peak_import_kw": float(ts.buy.max()),
        "pv_mwh": float((ts.pv * h).sum()) / 1000,
        "hvac_elec_mwh": float(((ts.hvac_HT + ts.hvac_AC) * h).sum()) / 1000,
        "fuel_mwh": float((ts[[c for c in FUEL_COLS if c in ts]].sum(axis=1) * h).sum()) / 1000,
        "discomfort_degC_h": float((excursion * on * h).sum()),
        "hours_outside_band": float(((excursion > 0.01) & on).sum() * np.median(h)),
    }
    for k in EQ1_COLS:
        row[k] = float(eq1.get(k, np.nan))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder with one sub-folder per case (each with in/ and out/)")
    ap.add_argument("--cases", nargs="*", help="case folder names (default: every case with out/ts.csv)")
    args = ap.parse_args()
    names = args.cases or sorted(d.name for d in args.folder.iterdir() if (d / "out" / "ts.csv").exists())
    df = pd.DataFrame([summarize(args.folder / n) for n in names]).set_index("case")
    out = args.folder / "comparison.csv"
    df.to_csv(out)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.round(2).to_string())
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
