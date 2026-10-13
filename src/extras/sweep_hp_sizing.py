"""
HP Sizing Sweep — derivative cases at different sizing percentiles.
===================================================================
For each sizing level (95th, 99th, 100th percentile, and 110% of peak),
re-runs heatpump selection, then creates DECARB input directories by
copying from bookend cases (50% elec, 20% PV) and overwriting hvac.csv
with the new HP sizing.

Output case names:
    flat_heat_50_solar_20_percent_95
    flat_heat_50_solar_20_percent_99
    flat_heat_50_solar_20_percent_100
    flat_heat_50_solar_20_percent_110
    tou_heat_50_solar_20_percent_95
    ...

Usage:
    python sweep_hp_sizing.py --source-study 7

    Optional:
        --percentiles 95,99,100,110
            Override the sizing percentiles (default: 95,99,100,110)
        --elec 50 --pv 20
            Override electrification/PV levels (default: 50/20)
"""

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_project_root = Path(__file__).parents[1]
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(Path(__file__).parent))
try:
    from config.loader import load_config, load_decarb_config
except ImportError:
    _project_root = Path(__file__).parents[2]
    sys.path.insert(0, str(_project_root))
    sys.path.insert(0, str(_project_root / "src"))
    from config.loader import load_config, load_decarb_config
from create_network_inputs import find_study_dir
from heatpump_selection import (
    load_neep_heat_pumps, read_parquet, build_cost_menu,
    find_multi_unit_options, capacity_at_temp, get_sizing_settings,
    _sizing_load_and_temp, _absolute_peak_load_and_temp,
    CONSUMPTION_COLS, METADATA_COLS,
)
from interpolate_demand import (
    load_metadata, get_building_ids, get_instance_counts,
    compute_baseline_pct, build_adoption_batches,
    HEATING_FUEL_COL, HEATING_FUEL_ELEC_VAL,
    HAS_PV_COL, HAS_PV_VAL, RANDOM_SEED,
)
from create_decarb_inputs import is_heat_pump_building, get_resstock_capacities

BTU_HR_TO_KW = 1 / 3412.14

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Defaults
DEFAULT_PERCENTILES = [95, 99, 100, 110]
ELEC_PCT = 50
PV_PCT = 20
TARIFF_BASES = ["flat", "tou", "flat_cap", "tou_cap"]

KW_TO_BTU = 3412.14


# ---------------------------------------------------------------------------
# HP selection at a given percentile
# ---------------------------------------------------------------------------

def select_hp_for_building(building_id, hp_base, metadata_df, consumption_dir,
                           percentile, max_units):
    """Run HP selection for one building at the given sizing percentile.

    percentile <= 100: use that percentile of nonzero heating/cooling loads.
    percentile > 100:  use the absolute peak load scaled by percentile/100.

    Returns (metrics_dict, failed_dict) -- one is None.
    """
    if building_id not in metadata_df.index:
        return None, None

    meta_row = metadata_df.loc[building_id]
    sqft = float(meta_row["in.sqft..ft2"])
    has_ducts = str(meta_row["in.hvac_has_ducts"]).strip().lower() == "yes"

    filepath = consumption_dir / f"{building_id}-0.parquet"
    if not filepath.exists():
        return None, None

    try:
        consumption_df = read_parquet(filepath, columns=CONSUMPTION_COLS)
    except Exception:
        return None, None

    temp_series = consumption_df["out.outdoor_air_drybulb_temp..c"]
    heating_series = consumption_df["out.load.heating.energy_delivered..kbtu"]
    cooling_series = consumption_df["out.load.cooling.energy_delivered..kbtu"]

    # Fallbacks for buildings with no heating or cooling load
    fallback = sqft * 30  # rule-of-thumb BTU/hr
    has_heating = not (heating_series.isna().all() or heating_series.max() == 0
                       or (heating_series > 0).sum() == 0)
    has_cooling = not (cooling_series.isna().all() or cooling_series.max() == 0
                       or (cooling_series > 0).sum() == 0)

    if has_heating:
        abs_h, abs_ht = _absolute_peak_load_and_temp(heating_series, temp_series)
    else:
        abs_h = fallback
        abs_ht = temp_series.min() * 1.8 + 32

    if has_cooling:
        abs_c, abs_ct = _absolute_peak_load_and_temp(cooling_series, temp_series)
    else:
        abs_c = fallback
        abs_ct = temp_series.max() * 1.8 + 32

    # Compute sizing loads
    if percentile > 100:
        scale = percentile / 100.0
        sizing_h, sizing_ht = abs_h * scale, abs_ht
        sizing_c, sizing_ct = abs_c * scale, abs_ct
    elif percentile == 100:
        sizing_h, sizing_ht = abs_h, abs_ht
        sizing_c, sizing_ct = abs_c, abs_ct
    else:
        if has_heating:
            sizing_h, sizing_ht = _sizing_load_and_temp(heating_series, temp_series, percentile)
        else:
            sizing_h, sizing_ht = fallback, temp_series.min() * 1.8 + 32
        if has_cooling:
            sizing_c, sizing_ct = _sizing_load_and_temp(cooling_series, temp_series, percentile)
        else:
            sizing_c, sizing_ct = fallback, temp_series.max() * 1.8 + 32

    # Evaluate HP capacity at sizing outdoor temps
    outdoor_c_h = (sizing_ht - 32) / 1.8
    outdoor_c_c = (sizing_ct - 32) / 1.8

    result = hp_base.copy()
    result["capacity_kw_heating"] = result.apply(
        lambda r: capacity_at_temp(r, outdoor_c_h, "heating"), axis=1)
    result["capacity_kw_cooling"] = result.apply(
        lambda r: capacity_at_temp(r, outdoor_c_c, "cooling"), axis=1)
    result["capacity_btu_heating"] = result["capacity_kw_heating"] * KW_TO_BTU
    result["capacity_btu_cooling"] = result["capacity_kw_cooling"] * KW_TO_BTU
    result["sufficient_capacity"] = (
        (result["capacity_btu_heating"] >= sizing_h) &
        (result["capacity_btu_cooling"] >= sizing_c)
    )

    n_sufficient = result["sufficient_capacity"].sum()

    if n_sufficient == 0:
        hp_multi, n_units = find_multi_unit_options(result, sizing_h, sizing_c, max_units)
        if hp_multi is None:
            return None, {"bldg_id": building_id, "peak_load_btu_hr": abs_h}
        hp_multi = build_cost_menu(hp_multi, sqft, has_ducts)
        cheapest = hp_multi[hp_multi["sufficient_capacity"]].nsmallest(1, "cost_total").iloc[0]
    else:
        result = build_cost_menu(result, sqft, has_ducts)
        cheapest = result[result["sufficient_capacity"]].nsmallest(1, "cost_total").iloc[0]
        n_units = 1

    metrics = {
        "bldg_id": building_id,
        "hp_index": cheapest.name,
        "pHSPF": cheapest["pHSPF"],
        "peak_load_btu_hr_heating": abs_h,
        "peak_load_temp_f_heating": round(abs_ht, 1),
        "peak_load_btu_hr_cooling": abs_c,
        "peak_load_temp_f_cooling": round(abs_ct, 1),
        "heating_capacity_kw": cheapest["capacity_kw_heating"],
        "cooling_capacity_kw": cheapest["capacity_kw_cooling"],
        "rated_heating_capacity_kw": cheapest["pHVmx"],
        "rated_cooling_capacity_kw": cheapest["pACmx"],
        "cop_heating": cheapest["pHVeff"],
        "cop_cooling": cheapest["pACeff"],
        "design_temperature_c": float(cheapest["temp"]),
        "heating_capacity_loss": cheapest["pHVmx_"],
        "cooling_capacity_loss": cheapest["pACmx_"],
        "cop_loss_heating": cheapest["pHVeff_"],
        "cop_loss_cooling": cheapest["pACeff_"],
        "capital_cost": cheapest["cost_total"] * n_units,
        "count": n_units,
    }
    return metrics, None


def run_hp_selection_at_percentile(cfg, hp_base, building_ids, percentile):
    """Run HP selection for all buildings at the given percentile."""
    input_dir = Path(cfg["paths"]["input_dir"])
    state = cfg["state"]

    metadata_path = input_dir / f"{state}_upgrade0.parquet"
    metadata_df = read_parquet(metadata_path, columns=METADATA_COLS).set_index("bldg_id")

    consumption_dir = input_dir / "residential_profiles" / state
    _, max_units = get_sizing_settings(cfg)

    label = f"{percentile}th pct" if percentile <= 100 else f"{percentile}% of peak"
    log.info(f"Running HP selection at {label} for {len(building_ids)} buildings...")

    summary_rows = []
    failed = 0
    for i, bldg_id in enumerate(building_ids, 1):
        metrics, zero = select_hp_for_building(
            bldg_id, hp_base, metadata_df, consumption_dir, percentile, max_units)
        if metrics is not None:
            summary_rows.append(metrics)
        else:
            failed += 1
        if i % 200 == 0:
            log.info(f"  {i}/{len(building_ids)} processed...")

    log.info(f"  Done: {len(summary_rows)} matched, {failed} failed")
    return pd.DataFrame(summary_rows).set_index("bldg_id")


# ---------------------------------------------------------------------------
# Build hvac.csv from hp_summary
# ---------------------------------------------------------------------------

def build_hvac_for_building(template, bldg_id, hp_summary_df, metadata_df=None):
    """Build hvac.csv DataFrame for one building using rated capacity.

    Matches create_decarb_inputs._build_hvac_df: for non-HP buildings,
    generic HVAC row capacities are blended with ResStock capacities
    (average of ResStock + HP), keeping heating and cooling separate.
    The Mitsubishi row always gets the full HP capacity.
    """
    if bldg_id not in hp_summary_df.index:
        return None

    row = hp_summary_df.loc[[bldg_id]]
    count = float(row["count"].iloc[0])
    hp_heating_kw = float(row["rated_heating_capacity_kw"].iloc[0]) * count
    hp_cooling_kw = float(row["rated_cooling_capacity_kw"].iloc[0]) * count

    # Blend generic row capacities with ResStock, matching
    # create_decarb_inputs._build_hvac_df lines 674-680.
    building_row = None
    if metadata_df is not None and bldg_id in metadata_df.index:
        building_row = metadata_df.loc[[bldg_id]]

    if building_row is not None and not is_heat_pump_building(building_row):
        rs_heating_kw, rs_cooling_kw = get_resstock_capacities(building_row)
        heating_kw = (rs_heating_kw + hp_heating_kw) / 2
        cooling_kw = (rs_cooling_kw + hp_cooling_kw) / 2
    else:
        heating_kw = hp_heating_kw
        cooling_kw = hp_cooling_kw

    df = template.copy()
    zeros_mask = df["ty"].astype(str).isin(["0", "0.0"])
    zeros_row = df[zeros_mask].copy()
    df = df[~zeros_mask].copy()
    df.loc[df["pHVmx"] != 0, "pHVmx"] = heating_kw
    df.loc[df["pACmx"] != 0, "pACmx"] = cooling_kw

    hp_index = str(row["hp_index"].iloc[0])
    mitsubishi_row = {
        "ty": f"Mitsubishi ASHP {hp_index}",
        "pHVmx": hp_heating_kw,
        "pACmx": hp_cooling_kw,
        "pHVeff": float(row["cop_heating"].iloc[0]),
        "pACeff": float(row["cop_cooling"].iloc[0]),
        "temp": float(row["design_temperature_c"].iloc[0]),
        "pHVmx_": float(row["heating_capacity_loss"].iloc[0]),
        "pACmx_": float(row["cooling_capacity_loss"].iloc[0]),
        "pHVeff_": float(row["cop_loss_heating"].iloc[0]),
        "pACeff_": float(row["cop_loss_cooling"].iloc[0]),
        "inv": float(row["capital_cost"].iloc[0]),
        "fom": 0,
        "vom": 0,
        "life": 15,
    }
    return pd.concat([df, pd.DataFrame([mitsubishi_row]), zeros_row], ignore_index=True)


def _patch_sp_csv(src_sp_path, dst_sp_path, hp_summary, bldg_id):
    """Copy sp.csv from bookend, updating the Mitsubishi ASHP name and count.

    The bookend sp.csv references whatever HP model the bookend used
    (e.g. "Mitsubishi ASHP 88"). The sweep may select a different model
    at a different percentile (e.g. "Mitsubishi ASHP 232"). We need the
    name in sp.csv to match the row in hvac.csv so DECARB recognizes it
    as pre-existing equipment.
    """
    sp_df = pd.read_csv(str(src_sp_path))

    row = hp_summary.loc[[bldg_id]]
    hp_index = str(row["hp_index"].iloc[0])
    new_name = f"Mitsubishi ASHP {hp_index}"
    count = int(row["count"].iloc[0])

    # Find any existing Mitsubishi row in sp.csv and update it.
    # Always set pHVACz0 = 1: multi-unit count is already baked into the
    # capacity values in hvac.csv (rated_kw * count), matching how
    # create_decarb_inputs.py does it.
    mits_mask = sp_df["pHVAC0"].astype(str).str.startswith("Mitsubishi ASHP")
    if mits_mask.any():
        sp_df.loc[mits_mask, "pHVAC0"] = new_name
        sp_df.loc[mits_mask, "pHVACz0"] = 1
    else:
        # No Mitsubishi row — update the first HVAC row with z0 > 0
        active = sp_df["pHVACz0"].astype(float) > 0
        if active.any():
            idx = sp_df[active].index[0]
            sp_df.loc[idx, "pHVAC0"] = new_name
            sp_df.loc[idx, "pHVACz0"] = 1

    sp_df.to_csv(str(dst_sp_path), index=False)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def next_study_num(output_dir):
    studies_root = Path(output_dir) / "studies"
    if not studies_root.exists():
        return 1
    existing = [int(d.name.split("_")[1]) for d in studies_root.iterdir()
                if d.is_dir() and d.name.startswith("study_")]
    return max(existing, default=0) + 1


def run_sweep(cfg, source_study_dir, percentiles, elec_pct, pv_pct):
    """Create derivative cases at different HP sizing percentiles.

    1. Allocate buildings to 50% elec / 20% PV (same as sweep_capacity_charge)
    2. Run HP selection at each percentile
    3. For each tariff x percentile, create a DECARB case by copying from
       bookend cases and overwriting hvac.csv
    """
    output_dir = Path(cfg["paths"]["output_dir"])
    new_num = next_study_num(output_dir)
    study_dir = output_dir / "studies" / f"study_{new_num}"
    study_dir.mkdir(parents=True)
    log.info(f"Creating new study: {study_dir.name}")

    # Copy feeder matching
    src_feeder = source_study_dir / "feeder_summaries_matched.parquet"
    if not src_feeder.exists():
        src_feeder = Path(cfg["paths"]["input_dir"]) / "feeder_summaries_matched.parquet"
    dst_feeder = study_dir / "feeder_summaries_matched.parquet"
    shutil.copy2(str(src_feeder), str(dst_feeder))

    # ── Building allocation (same logic as sweep_capacity_charge) ──────────
    metadata_df = load_metadata(cfg)
    feeder_df = read_parquet(dst_feeder)

    all_bldg_ids = get_building_ids(feeder_df)
    instance_counts = get_instance_counts(feeder_df)
    total_instances = sum(instance_counts.values())

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

    elec_batches = build_adoption_batches(
        elec_non, instance_counts, total_instances,
        elec_adopted_inst, [elec_pct], seed=RANDOM_SEED)
    elec_switched = elec_batches[elec_pct]

    pv_batches = build_adoption_batches(
        pv_non, instance_counts, total_instances,
        pv_adopted_inst, [pv_pct], seed=RANDOM_SEED + 1000)
    pv_switched = pv_batches[pv_pct]

    elec_set = elec_adopted | elec_switched
    pv_set = pv_adopted | pv_switched

    log.info(f"Electrified: {len(elec_set)} buildings "
             f"({sum(instance_counts.get(b,0) for b in elec_set)} instances)")
    log.info(f"Has PV:      {len(pv_set)} buildings "
             f"({sum(instance_counts.get(b,0) for b in pv_set)} instances)")

    # ── Load HP database and run selection at each percentile ──────────────
    hp_cfg = cfg.get("heatpump", {})
    neep_path = Path(hp_cfg["neep_database"]) if hp_cfg.get("neep_database") else \
        Path(cfg["paths"]["input_dir"]) / "neep_database.csv"
    hp_base = load_neep_heat_pumps(neep_path)

    hp_summaries = {}
    for pct in percentiles:
        hp_summaries[pct] = run_hp_selection_at_percentile(
            cfg, hp_base, list(all_bldg_ids), pct)
        hp_out = study_dir / f"hp_summary_percent_{pct}.csv"
        hp_summaries[pct].to_csv(str(hp_out))
        log.info(f"Saved {hp_out.name}")

    # ── Index metadata by bldg_id for ResStock capacity lookups ─────────────
    meta_indexed = metadata_df.set_index("bldg_id")

    # ── Load hvac template ─────────────────────────────────────────────────
    dcfg = load_decarb_config()
    template_dir = _project_root / dcfg.get("template_dir",
                                              "in/decarb_input_templates")
    hvac_template = pd.read_csv(str(template_dir / "hvac.csv"))

    # ── Locate bookend cases in source study ───────────────────────────────
    source_decarb = source_study_dir / "decarb"
    decarb_root = study_dir / "decarb"
    decarb_root.mkdir(parents=True, exist_ok=True)

    for tariff_base in TARIFF_BASES:
        bookends = {}
        for h in [0, 100]:
            for s in [0, 100]:
                case_name = f"{tariff_base}_heat_{h}_solar_{s}"
                case_dir = source_decarb / case_name
                if not case_dir.exists():
                    log.error(f"Bookend case not found: {case_name}")
                    continue
                bookends[(h, s)] = case_dir
                log.info(f"  Bookend ({h},{s}): {case_dir.name}")

        if len(bookends) < 4:
            log.error(f"Missing bookend cases for tariff {tariff_base}, skipping")
            continue

        for pct in percentiles:
            case_name = f"{tariff_base}_heat_{elec_pct}_solar_{pv_pct}_percent_{pct}"
            case_dir = decarb_root / case_name
            log.info(f"\nCreating: {case_name}")

            if case_dir.exists():
                shutil.rmtree(str(case_dir))
            case_dir.mkdir(parents=True)

            hp_summary = hp_summaries[pct]
            copied = 0
            skipped = 0

            for bldg_id in all_bldg_ids:
                bldg_id = int(bldg_id)

                # Determine bookend source
                h_level = 100 if bldg_id in elec_set else 0
                s_level = 100 if bldg_id in pv_set else 0
                src_case = bookends.get((h_level, s_level))

                if src_case is None:
                    skipped += 1
                    continue

                src_in = src_case / str(bldg_id) / "update_0" / "in"
                if not src_in.exists():
                    skipped += 1
                    continue

                dst_in = case_dir / str(bldg_id) / "update_0" / "in"
                dst_in.mkdir(parents=True, exist_ok=True)

                # Copy files from bookend. For electrified buildings,
                # skip hvac.csv and sp.csv (rebuilt below). For
                # non-electrified buildings, copy everything as-is.
                skip_files = {"hvac.csv", "sp.csv"} if h_level > 0 else set()
                for f in src_in.iterdir():
                    if f.is_file() and f.name not in skip_files:
                        dst_file = dst_in / f.name
                        if dst_file.exists():
                            dst_file.unlink()
                        shutil.copy2(str(f), str(dst_file))

                # Build new hvac.csv from hp_summary at this percentile
                # Only for electrified buildings — non-electrified keep
                # their original fossil equipment from the bookend.
                if h_level > 0 and bldg_id in hp_summary.index:
                    hvac_df = build_hvac_for_building(
                        hvac_template, bldg_id, hp_summary, meta_indexed)
                    if hvac_df is not None:
                        hvac_df.to_csv(str(dst_in / "hvac.csv"), index=False)
                        # Patch sp.csv: update HVAC starting point to match
                        # the new Mitsubishi model name and unit count
                        _patch_sp_csv(src_in / "sp.csv", dst_in / "sp.csv",
                                      hp_summary, bldg_id)
                        copied += 1
                        continue

                # Fallback: copy original hvac.csv and sp.csv
                if (src_in / "hvac.csv").exists():
                    shutil.copy2(str(src_in / "hvac.csv"),
                                 str(dst_in / "hvac.csv"))
                if (src_in / "sp.csv").exists():
                    shutil.copy2(str(src_in / "sp.csv"),
                                 str(dst_in / "sp.csv"))
                skipped += 1

            log.info(f"  {copied} updated, {skipped} kept original")

    log.info(f"\n{'='*60}")
    log.info(f"Sweep complete. New study: {study_dir}")
    log.info(f"Cases created:")
    for tariff_base in TARIFF_BASES:
        for pct in percentiles:
            log.info(f"  {tariff_base}_heat_{elec_pct}_solar_{pv_pct}_percent_{pct}")
    log.info(f"\nHP summaries: hp_summary_percent_<pct>.csv")
    log.info(f"Next: run DECARB on these cases.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HP sizing sweep")
    parser.add_argument("--source-study", type=int, required=True,
                        help="Source study number (must have bookend cases)")
    parser.add_argument("--percentiles", type=str, default="95,99,100,110",
                        help="Comma-separated sizing percentiles (default: 95,99,100,110)")
    parser.add_argument("--elec", type=int, default=50,
                        help="Electrification percentage (default: 50)")
    parser.add_argument("--pv", type=int, default=20,
                        help="PV penetration percentage (default: 20)")
    args = parser.parse_args()

    cfg = load_config()
    output_dir = Path(cfg["paths"]["output_dir"])
    source_study_dir = find_study_dir(output_dir, args.source_study)

    percentiles = [int(x) for x in args.percentiles.split(",")]

    log.info(f"Source study: {source_study_dir.name}")
    log.info(f"Elec: {args.elec}%, PV: {args.pv}%")
    log.info(f"Percentiles: {percentiles}")

    run_sweep(cfg, source_study_dir, percentiles, args.elec, args.pv)
