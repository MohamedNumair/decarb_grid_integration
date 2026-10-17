"""
Capacity-charge sweep at prescribed electrification & PV levels.
================================================================
Creates new DECARB case directories by:
  1. Randomly assigning buildings to 50% electrification and 20% PV
     (instance-weighted by bus count, same batching as interpolate_demand.py)
  2. For each building, copying its input folder from the appropriate
     bookend case (heat_0/100 × solar_0/100)
  3. Patching tm.csv to set pQmx=1 and pQmxCost to each CC value

CC values: 250, 375, 500, 625, 750  ($/kW)
Tariff bases: flat, tou

Output case names:  flat_heat_50_solar_20_cap_250, ...
                    tou_heat_50_solar_20_cap_250, ...

After running this script, launch DECARB on the new cases,
then use the network-inputs pipeline or interpolation summary
to aggregate results.

Usage:
    python sweep_capacity_charge.py [study_num]
"""

import logging
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_project_root = Path(__file__).parents[1]
sys.path.insert(0, str(_project_root))
try:
    from config.loader import load_config, load_network_config
except ImportError:
    _project_root = Path(__file__).parents[2]
    sys.path.insert(0, str(_project_root))
    sys.path.insert(0, str(_project_root / "src"))
    from config.loader import load_config, load_network_config
from create_network_inputs import find_study_dir

from interpolate_demand import (
    load_metadata,
    read_parquet,
    get_building_ids,
    get_instance_counts,
    compute_baseline_pct,
    build_adoption_batches,
    build_interpolation_steps,
    HEATING_FUEL_COL, HEATING_FUEL_ELEC_VAL,
    HAS_PV_COL, HAS_PV_VAL,
    RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Sweep parameters ────────────────────────────────────────────────────────
ELEC_PCT = 50
PV_PCT = 20
CC_VALUES = [250, 500, 750]
TARIFF_BASES = ["flat", "tou"]  # base tariff types to sweep CC on


def copy_building_dir(src_case_dir, dst_case_dir, bldg_id):
    """Set up a building's DECARB input directory.

    Symlinks all files EXCEPT tm.csv (which gets copied so we can patch it).
    This avoids copying ~1.2 MB tm.csv × 10 cases redundantly — only the one
    we need to modify gets a real copy.
    """
    src = src_case_dir / str(bldg_id) / "in"
    dst = dst_case_dir / str(bldg_id) / "in"

    if not src.exists():
        return False

    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            dst_file = dst / f.name
            if f.name == "tm.csv":
                # Copy tm.csv so we can patch it
                shutil.copy2(str(f), str(dst_file))
            else:
                # Symlink everything else (saves ~12 GB)
                if dst_file.exists():
                    dst_file.unlink()
                os.symlink(str(f), str(dst_file))

    return True


def patch_tm_capacity_charge(case_dir, bldg_id, cc_value):
    """Set pQmx=1 and pQmxCost=cc_value in a building's tm.csv."""
    tm_path = case_dir / str(bldg_id) / "in" / "tm.csv"
    if not tm_path.exists():
        return False

    df = pd.read_csv(str(tm_path), index_col=0)
    df["pQmx"] = 1
    df["pQmxCost"] = cc_value
    df.to_csv(str(tm_path))
    return True


def next_study_num(output_dir):
    """Find the next available study number."""
    studies_root = Path(output_dir) / "studies"
    if not studies_root.exists():
        return 1
    existing = [int(d.name.split("_")[1]) for d in studies_root.iterdir()
                if d.is_dir() and d.name.startswith("study_")]
    return max(existing, default=0) + 1


def run_sweep(cfg, source_study_dir):
    """Main sweep: create DECARB cases for each tariff × CC value.

    Creates a NEW study directory and copies the feeder matching file
    from the source study. Bookend profiles are read from the source study.
    """
    output_dir = Path(cfg["paths"]["output_dir"])
    new_num = next_study_num(output_dir)
    study_dir = output_dir / "studies" / f"study_{new_num}"
    study_dir.mkdir(parents=True)
    log.info(f"Creating new study: {study_dir.name}")

    # Copy feeder matching from source study
    src_feeder = source_study_dir / "feeder_summaries_matched.parquet"
    if not src_feeder.exists():
        src_feeder = Path(cfg["paths"]["input_dir"]) / "feeder_summaries_matched.parquet"
    dst_feeder = study_dir / "feeder_summaries_matched.parquet"
    shutil.copy2(str(src_feeder), str(dst_feeder))
    log.info(f"Copied feeder matching from {source_study_dir.name}")

    # ── Load metadata and feeder matching ───────────────────────────────────
    metadata_df = load_metadata(cfg)
    log.info(f"Metadata: {len(metadata_df)} buildings")

    feeder_df = read_parquet(dst_feeder)

    all_bldg_ids = get_building_ids(feeder_df)
    instance_counts = get_instance_counts(feeder_df)
    total_instances = sum(instance_counts.values())
    log.info(f"Buildings: {len(all_bldg_ids)}, bus instances: {total_instances}")

    # ── Compute baselines and build batches ─────────────────────────────────
    elec_base, elec_adopted, elec_non = compute_baseline_pct(
        metadata_df, all_bldg_ids, instance_counts,
        HEATING_FUEL_COL, HEATING_FUEL_ELEC_VAL)
    elec_adopted_inst = sum(instance_counts.get(b, 0) for b in elec_adopted)

    pv_base, pv_adopted, pv_non = compute_baseline_pct(
        metadata_df, all_bldg_ids, instance_counts,
        HAS_PV_COL, HAS_PV_VAL)
    pv_adopted_inst = sum(instance_counts.get(b, 0) for b in pv_adopted)

    log.info(f"Baseline electrification: {elec_base:.1f}%")
    log.info(f"Baseline PV penetration:  {pv_base:.1f}%")

    # Build batches at the target percentages
    log.info(f"\nBuilding electrification batch for {ELEC_PCT}%...")
    elec_batches = build_adoption_batches(
        elec_non, instance_counts, total_instances,
        elec_adopted_inst, [ELEC_PCT], seed=RANDOM_SEED)
    elec_switched = elec_batches[ELEC_PCT]

    log.info(f"Building PV batch for {PV_PCT}%...")
    pv_batches = build_adoption_batches(
        pv_non, instance_counts, total_instances,
        pv_adopted_inst, [PV_PCT], seed=RANDOM_SEED + 1000)
    pv_switched = pv_batches[PV_PCT]

    # Combined: for each building, determine its category
    # electrified = already electric OR in elec_switched
    elec_set = elec_adopted | elec_switched
    pv_set = pv_adopted | pv_switched

    log.info(f"\nAssignment summary:")
    log.info(f"  Electrified: {len(elec_set)} buildings "
             f"({sum(instance_counts.get(b,0) for b in elec_set)} instances)")
    log.info(f"  Has PV:      {len(pv_set)} buildings "
             f"({sum(instance_counts.get(b,0) for b in pv_set)} instances)")

    # ── Create cases ────────────────────────────────────────────────────────
    source_decarb = source_study_dir / "decarb"   # bookend sources
    decarb_root = study_dir / "decarb"             # new cases go here
    decarb_root.mkdir(parents=True, exist_ok=True)

    for tariff_base in TARIFF_BASES:
        # Identify the 4 bookend source cases for this tariff
        bookends = {}
        for h in [0, 100]:
            for s in [0, 100]:
                case_name = f"{tariff_base}_heat_{h}_solar_{s}"
                case_dir = source_decarb / case_name
                if not case_dir.exists():
                    # Try with _cap suffix (flat_cap -> flat base)
                    case_name_alt = f"{tariff_base}_cap_heat_{h}_solar_{s}"
                    case_dir_alt = source_decarb / case_name_alt
                    if case_dir_alt.exists():
                        case_dir = case_dir_alt
                    else:
                        log.error(f"Bookend case not found: {case_name}")
                        continue
                bookends[(h, s)] = case_dir
                log.info(f"  Bookend ({h},{s}): {case_dir.name}")

        if len(bookends) < 4:
            log.error(f"Missing bookend cases for tariff {tariff_base}, skipping")
            continue

        for cc_val in CC_VALUES:
            case_name = f"{tariff_base}_cap_{cc_val}_heat_{ELEC_PCT}_solar_{PV_PCT}"
            case_dir = decarb_root / case_name
            log.info(f"\n{'='*60}")
            log.info(f"Creating: {case_name}")
            log.info(f"{'='*60}")

            if case_dir.exists():
                log.info(f"  Already exists, removing...")
                shutil.rmtree(str(case_dir))
            case_dir.mkdir(parents=True)

            copied = 0
            patched = 0
            skipped = 0

            for bldg_id in all_bldg_ids:
                bldg_id = int(bldg_id)

                # Determine which bookend to copy from
                h_level = 100 if bldg_id in elec_set else 0
                s_level = 100 if bldg_id in pv_set else 0
                src_case = bookends.get((h_level, s_level))

                if src_case is None:
                    skipped += 1
                    continue

                if copy_building_dir(src_case, case_dir, bldg_id):
                    copied += 1
                    if patch_tm_capacity_charge(case_dir, bldg_id, cc_val):
                        patched += 1
                else:
                    skipped += 1

            log.info(f"  Copied: {copied}, patched: {patched}, skipped: {skipped}")

    log.info(f"\n{'='*60}")
    log.info(f"Sweep complete. New cases in: {decarb_root}")
    log.info(f"Cases created:")
    for tariff_base in TARIFF_BASES:
        for cc_val in CC_VALUES:
            name = f"{tariff_base}_cap_{cc_val}_heat_{ELEC_PCT}_solar_{PV_PCT}"
            log.info(f"  {name}")
    log.info(f"\nNext: run DECARB on these cases, then aggregate results.")


if __name__ == "__main__":
    cfg = load_config()

    source_study_num = 4  # default: read bookends from study_4
    if len(sys.argv) > 1:
        source_study_num = int(sys.argv[1])

    output_dir = Path(cfg["paths"]["output_dir"])
    source_study_dir = find_study_dir(output_dir, source_study_num)
    log.info(f"Source study: {source_study_dir.name}")

    run_sweep(cfg, source_study_dir)
