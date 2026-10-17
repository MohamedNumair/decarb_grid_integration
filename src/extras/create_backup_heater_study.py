"""
Create a backup-heater study from an existing DECARB case.

Takes a base study's single case (e.g. study_8/decarb/flat_cap_heat_0_solar_0)
and creates a new study with three cases, each with every building:

  {case}                   As-is building (copied from source),
                           fossil fuel prices set to electricity + margin
  {case}_ashp_no_backup    ASHP replaces legacy HVAC, fossil heater removed (pCHPz0=0),
                           fossil fuel prices set to electricity + margin
  {case}_ashp_backup       ASHP replaces legacy HVAC, fossil heater kept as backup
                           at BACKUP_FRACTION of ASHP heating capacity,
                           fossil fuel prices set to electricity + margin

Usage:
    python create_backup_heater_study.py --source-study 8 --case flat_cap_heat_0_solar_0
    python create_backup_heater_study.py --source-study 8 --case flat_cap_heat_0_solar_0 \
        --fraction 0.1 --fuel-margin 0.01
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
try:
    from create_network_inputs import find_study_dir
except ImportError:
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from create_network_inputs import find_study_dir

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

BACKUP_FRACTION = 0.1
FUEL_MARGIN = 0.01


def read_csv_str(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def next_study_num(output_dir: Path) -> int:
    studies = output_dir / "studies"
    if not studies.exists():
        return 1
    nums = []
    for d in studies.iterdir():
        if d.is_dir() and d.name.startswith("study_"):
            try:
                nums.append(int(d.name.split("_")[1]))
            except ValueError:
                pass
    return max(nums, default=0) + 1


def find_ashp(hvac: pd.DataFrame) -> tuple:
    rows = hvac[hvac["ty"].str.startswith("Mitsubishi ASHP")]
    if len(rows) != 1:
        raise ValueError(
            f"expected exactly one Mitsubishi ASHP row, found {len(rows)}")
    return rows["ty"].iloc[0], float(rows["pHVmx"].iloc[0])


def build_sp(sp: pd.DataFrame, ashp_name: str,
             keep_backup: bool) -> tuple:
    sp = sp.copy()

    # Same logic as create_decarb_inputs.py lines 859-867:
    # if the ASHP name exists in a row, set that row; otherwise first row.
    if ashp_name in sp["pHVAC0"].values:
        sp.loc[sp["pHVAC0"] == ashp_name, "pHVACz0"] = "1"
    else:
        sp.iloc[0, sp.columns.get_loc("pHVAC0")] = ashp_name
        sp.iloc[0, sp.columns.get_loc("pHVACz0")] = "1"

    chp_installed = sp.index[sp["pCHPz0"] == "1"].tolist()
    if len(chp_installed) == 0:
        # No fossil heater installed — pick the first non-zero CHP option
        chp_candidates = sp.index[
            (sp["pCHP0"].astype(str) != "0") &
            (sp["pCHP0"].astype(str) != "")].tolist()
        if not chp_candidates:
            raise ValueError("no CHP options found in sp.csv")
        idx = chp_candidates[0]
        heater_name = sp.loc[idx, "pCHP0"]
        if keep_backup:
            sp.loc[idx, "pCHPz0"] = "1"
    else:
        idx = chp_installed[0]
        heater_name = sp.loc[idx, "pCHP0"]
        if not keep_backup:
            sp.loc[idx, "pCHPz0"] = "0"

    return sp, heater_name


def build_chp(chp: pd.DataFrame, heater_name: str,
              backup_kw: float) -> pd.DataFrame:
    chp = chp.copy()
    mask = chp["ty"] == heater_name
    if mask.sum() != 1:
        raise ValueError(
            f"heater '{heater_name}' not found (or duplicated) in chp.csv")
    chp.loc[mask, "mx"] = repr(backup_kw)
    return chp


def patch_tm_fuel_prices(tm_path: Path, margin: float):
    """Set pGcost and pLcost to pQcostBuy + margin for every timestep."""
    tm = pd.read_csv(tm_path)
    elec_price = tm["pQcostBuy"].astype(float)
    tm["pGcost"] = elec_price + margin
    tm["pLcost"] = elec_price + margin
    tm.to_csv(tm_path, index=False)


def create_ashp_building(src_in: Path, dst_bldg: Path, keep_backup: bool,
                         fraction: float, fuel_margin: float) -> str:
    if dst_bldg.exists():
        shutil.rmtree(dst_bldg)

    shutil.copytree(str(src_in), str(dst_bldg / "in"))
    (dst_bldg / "out").mkdir()

    hvac = read_csv_str(dst_bldg / "in" / "hvac.csv")
    ashp_name, ashp_kw = find_ashp(hvac)

    sp, heater_name = build_sp(
        read_csv_str(dst_bldg / "in" / "sp.csv"), ashp_name, keep_backup)
    sp.to_csv(dst_bldg / "in" / "sp.csv", index=False)

    patch_tm_fuel_prices(dst_bldg / "in" / "tm.csv", fuel_margin)

    msg = f"{ashp_name} ({ashp_kw:.2f} kW)"
    if keep_backup:
        backup_kw = fraction * ashp_kw
        chp = build_chp(
            read_csv_str(dst_bldg / "in" / "chp.csv"), heater_name, backup_kw)
        chp.to_csv(dst_bldg / "in" / "chp.csv", index=False)
        msg += f" + backup {heater_name} @ {backup_kw:.2f} kW"
    else:
        msg += f", removed {heater_name}"
    return msg


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-study", type=int, required=True,
                    help="source study number (e.g. 8)")
    ap.add_argument("--case", type=str, required=True,
                    help="case name within source study (e.g. flat_cap_heat_0_solar_0)")
    ap.add_argument("--fraction", type=float, default=BACKUP_FRACTION,
                    help="backup heater capacity as fraction of ASHP heating kW")
    ap.add_argument("--fuel-margin", type=float, default=FUEL_MARGIN,
                    help="fossil fuel price = electricity price + this margin ($/kWh)")
    args = ap.parse_args()

    output_dir = Path(__file__).parents[1] / "out"
    if not output_dir.exists():
        output_dir = Path(__file__).parents[2] / "out"
    source_study = find_study_dir(output_dir, args.source_study)
    source_case = source_study / "decarb" / args.case

    if not source_case.exists():
        log.error(f"Source case not found: {source_case}")
        return 1

    new_num = next_study_num(output_dir)
    new_study = output_dir / "studies" / f"study_{new_num}"
    asis_case = new_study / "decarb" / args.case
    ashp_cases = {f"{args.case}_ashp_no_backup": False,   # case name: keep_backup
                  f"{args.case}_ashp_backup": True}
    asis_case.mkdir(parents=True)
    log.info(f"Creating study_{new_num} from study_{args.source_study}/{args.case}")

    # Copy feeder matching if it exists
    src_feeder = source_study / "feeder_summaries_matched.parquet"
    if src_feeder.exists():
        shutil.copy2(str(src_feeder),
                      str(new_study / "feeder_summaries_matched.parquet"))

    bldg_dirs = sorted(
        d for d in source_case.iterdir()
        if d.is_dir() and (d / "in").is_dir())

    if not bldg_dirs:
        log.error(f"No buildings with in/ found under {source_case}")
        return 1

    log.info(f"Processing {len(bldg_dirs)} buildings...")

    failures = 0

    for i, src_bldg in enumerate(bldg_dirs, 1):
        bldg_name = src_bldg.name

        # As-is case: copy the inputs only (source results used other fuel
        # prices) and patch fuel prices
        dst_asis = asis_case / bldg_name
        shutil.copytree(str(src_bldg / "in"), str(dst_asis / "in"))
        (dst_asis / "out").mkdir()
        patch_tm_fuel_prices(dst_asis / "in" / "tm.csv", args.fuel_margin)

        src_in = src_bldg / "in"

        for case_name, keep_backup in ashp_cases.items():
            dst_bldg = new_study / "decarb" / case_name / bldg_name
            try:
                msg = create_ashp_building(
                    src_in, dst_bldg, keep_backup,
                    args.fraction, args.fuel_margin)
                if i <= 3 or i % 200 == 0:
                    log.info(f"  {case_name}/{bldg_name}: {msg}")
            except Exception as exc:
                failures += 1
                log.warning(f"  {case_name}/{bldg_name}: ERROR {exc}")

        if i % 200 == 0:
            log.info(f"  {i}/{len(bldg_dirs)} done")

    log.info(f"\nDone: {len(bldg_dirs)} buildings, {failures} failures")
    log.info(f"New study: {new_study}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
