"""
Check that reading feeder loads from SMART-DS .dss files gives the same result
as the hand-made Loads.csv / Buscoords.csv files it replaces.

For every feeder of the configured zone that has both the CSVs and the .dss
files, compares (by load name) the row set and every shared column. Writes any
differences to out/dss_parsing_differences.csv.

Usage:
    python src/extras/check_dss_parsing.py [--zone P100U]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).resolve()
for _root in (_here.parents[1], _here.parents[2]):
    if (_root / "config").is_dir():
        sys.path.insert(0, str(_root))
        sys.path.insert(0, str(_root / "src"))
        break
from config.loader import load_config
from process_network_info import read_feeder_loads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def read_feeder_loads_csv(feeder_path):
    """The previous CSV-based reading, kept here only for comparison."""
    loads_df = pd.read_csv(feeder_path / "Loads.csv", low_memory=False)
    buscoords_df = pd.read_csv(feeder_path / "Buscoords.csv").rename(columns={"Name": "Bus_Name"})
    loads_df["bus1_base"] = loads_df["bus1"].apply(lambda x: str(x).split(".")[0])
    return pd.merge(loads_df, buscoords_df[["Bus_Name", "Longitude", "Latitude"]],
                    left_on="bus1_base", right_on="Bus_Name", how="left")


def compare_feeder(old, new, feeder):
    diffs = []
    old = old.set_index(old["Name"].astype(str).str.lower())
    new = new.set_index(new["Name"].astype(str).str.lower())

    for name in sorted(set(old.index) - set(new.index)):
        diffs.append({"feeder": feeder, "load": name, "column": "(row)", "csv": "present", "dss": "missing"})
    for name in sorted(set(new.index) - set(old.index)):
        diffs.append({"feeder": feeder, "load": name, "column": "(row)", "csv": "missing", "dss": "present"})

    common = sorted(set(old.index) & set(new.index))
    for col in sorted((set(old.columns) & set(new.columns)) - {"Name"}):
        a, b = old.loc[common, col], new.loc[common, col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            same = np.isclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-9, equal_nan=True)
        else:
            same = (a.astype(str).str.lower() == b.astype(str).str.lower()).to_numpy()
        for name in np.array(common)[~same]:
            diffs.append({"feeder": feeder, "load": name, "column": col,
                          "csv": old.at[name, col], "dss": new.at[name, col]})
    return diffs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zone", default=None, help="Zone to check (default: zone.name in config)")
    args = parser.parse_args()

    cfg = load_config()
    zone = args.zone or cfg["zone"]["name"]
    zone_dir = Path(cfg["paths"]["network_dir"]) / zone / "scenarios" / "base_timeseries" / "opendss"
    if not zone_dir.is_dir():
        sys.exit(f"Zone folder not found: {zone_dir}")

    feeders = [f for sub in sorted(zone_dir.iterdir()) if sub.is_dir()
               for f in sorted(sub.iterdir()) if f.is_dir()]
    all_diffs, checked, skipped, n_loads = [], 0, 0, 0
    only_csv_cols, only_dss_cols = set(), set()

    for feeder_dir in feeders:
        has_csv = (feeder_dir / "Loads.csv").exists() and (feeder_dir / "Buscoords.csv").exists()
        has_dss = (feeder_dir / "Loads.dss").exists() and (feeder_dir / "Buscoords.dss").exists()
        if not (has_csv and has_dss):
            skipped += 1
            continue
        old, new = read_feeder_loads_csv(feeder_dir), read_feeder_loads(feeder_dir)
        only_csv_cols |= set(old.columns) - set(new.columns)
        only_dss_cols |= set(new.columns) - set(old.columns)
        all_diffs += compare_feeder(old, new, f"{feeder_dir.parent.name}/{feeder_dir.name}")
        checked += 1
        n_loads += len(old)

    log.info(f"Zone {zone}: compared {checked} feeders ({n_loads} loads), "
             f"skipped {skipped} without both CSV and DSS files")
    if only_csv_cols:
        log.info(f"Columns only in the CSVs (not produced from .dss): {sorted(only_csv_cols)}")
    if only_dss_cols:
        log.info(f"Columns only from .dss: {sorted(only_dss_cols)}")

    if not all_diffs:
        log.info("MATCH: .dss parsing gives the same loads and values as the CSVs")
        return

    diffs = pd.DataFrame(all_diffs)
    out = Path(cfg["paths"]["output_dir"]) / "dss_parsing_differences.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    diffs.to_csv(out, index=False)
    log.warning(f"DIFFERENCES: {len(diffs)} (full list: {out})")
    for col, n in diffs["column"].value_counts().items():
        log.warning(f"  {n:>7}  {col}")
    sys.exit(1)


if __name__ == "__main__":
    main()
