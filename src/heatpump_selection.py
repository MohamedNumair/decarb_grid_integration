import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from config.loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KW_TO_BTU = 3412.142
BTU_TO_KW = 1 / KW_TO_BTU

NEEP_CAPACITY_COLS = [
    "Minimum Capacity 5°F", "Rated Capacity 5°F", "Maximum Capacity 5°F",
    "Minimum Capacity 17°F", "Rated Capacity 17°F", "Maximum Capacity 17°F",
    "Minimum Capacity 47°F", "Rated Capacity 47°F", "Maximum Capacity 47°F",
    "Minimum Capacity 82°F", "Rated Capacity 82°F", "Maximum Capacity 82°F",
    "Minimum Capacity 95°F", "Rated Capacity 95°F", "Maximum Capacity 95°F",
]

NEEP_COP_COLS = [
    "COP at Min. Capacity 5°F", "COP at Rated Capacity 5°F", "COP at Max. Capacity 5°F",
    "COP at Min. Capacity 17°F", "COP at Rated Capacity 17°F", "COP at Max. Capacity 17°F",
    "COP at Min. Capacity 47°F", "COP at Rated Capacity 47°F", "COP at Max. Capacity 47°F",
    "COP at Min. Capacity 82°F", "COP at Rated Capacity 82°F", "COP at Max. Capacity 82°F",
    "COP at Min. Capacity 95°F", "COP at Rated Capacity 95°F", "COP at Max. Capacity 95°F",
]

NEEP_REQUIRED_COLS = ["Brand Name", "Status", "HSPF (Region IV)"] + NEEP_CAPACITY_COLS + NEEP_COP_COLS

NEEP_TEMP_TIERS = ["5°F", "17°F", "47°F", "82°F", "95°F"]

METADATA_COLS = [
    "bldg_id",
    "in.sqft..ft2",
    "in.hvac_has_ducts",
    "in.hvac_cooling_efficiency",
    "in.hvac_cooling_type",
    "in.hvac_heating_efficiency",
    "in.hvac_heating_type",
    "in.hvac_heating_type_and_fuel",
]

CONSUMPTION_COLS = [
    "out.load.cooling.energy_delivered..kbtu",
    "out.load.heating.energy_delivered..kbtu",
    "out.indoor_operative_temperature.conditioned_space..c",
    "out.outdoor_air_drybulb_temp..c",
]

DESIGN_TEMP_C = (47.0 - 32) / 1.8  # 8.33 °C


def read_parquet(path, columns=None):
    try:
        return pd.read_parquet(str(path), engine="fastparquet", columns=columns)
    except Exception:
        return pd.read_parquet(str(path), engine="pyarrow", columns=columns)


# ---------------------------------------------------------------------------
# NEEP database
# ---------------------------------------------------------------------------

def clean_neep_data(df):
    df = df.copy()
    for tier in NEEP_TEMP_TIERS:
        cap_rated, cap_min, cap_max = f"Rated Capacity {tier}", f"Minimum Capacity {tier}", f"Maximum Capacity {tier}"
        if all(c in df.columns for c in [cap_rated, cap_min, cap_max]):
            mask = df[cap_rated].isna()
            df.loc[mask, cap_rated] = (df.loc[mask, cap_min] + df.loc[mask, cap_max]) / 2

        cop_rated, cop_min, cop_max = f"COP at Rated Capacity {tier}", f"COP at Min. Capacity {tier}", f"COP at Max. Capacity {tier}"
        if all(c in df.columns for c in [cop_rated, cop_min, cop_max]):
            mask = df[cop_rated].isna()
            df.loc[mask, cop_rated] = (df.loc[mask, cop_min] + df.loc[mask, cop_max]) / 2
    return df


def load_neep_heat_pumps(neep_filepath):
    suffix = Path(neep_filepath).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(str(neep_filepath), low_memory=False)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(str(neep_filepath))
    else:
        raise ValueError(f"Unsupported NEEP file type: {suffix}")

    missing = [c for c in NEEP_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing NEEP columns: {missing}")

    df = df[df["Brand Name"].str.contains("mitsubishi", case=False, na=False)]
    df = df[df["Status"].str.strip().str.lower() == "live"]
    df = df.dropna(subset=["HSPF (Region IV)"])
    df = df.dropna(subset=NEEP_CAPACITY_COLS, how="all")
    df = df.reset_index(drop=True)
    df = clean_neep_data(df)

    cop_rated_cols = [c for c in NEEP_COP_COLS if "Rated" in c and c in df.columns]
    cop_mask = (df[cop_rated_cols] > 6).any(axis=1)
    n_excluded = cop_mask.sum()
    df = df[~cop_mask].reset_index(drop=True)

    h_temps_f = np.array([5.0, 17.0, 47.0])
    h_temps_c = (h_temps_f - 32) / 1.8
    h_delta_c = h_temps_c - DESIGN_TEMP_C

    c_temps_f = np.array([47.0, 82.0, 95.0])
    c_temps_c = (c_temps_f - 32) / 1.8
    c_delta_c = c_temps_c - DESIGN_TEMP_C

    records = []
    for _, row in df.iterrows():
        pHVmx = row["Rated Capacity 47°F"] * BTU_TO_KW
        pACmx = row["Rated Capacity 47°F"] * BTU_TO_KW
        pHVeff = row["COP at Rated Capacity 47°F"]
        pACeff = row["COP at Rated Capacity 47°F"]

        h_caps_kw = np.array([row["Rated Capacity 5°F"], row["Rated Capacity 17°F"], row["Rated Capacity 47°F"]]) * BTU_TO_KW
        pHVmx_ = float(np.linalg.lstsq(h_delta_c.reshape(-1, 1), h_caps_kw - pHVmx, rcond=None)[0][0]) / pHVmx if pHVmx > 0 else 0.0

        c_caps_kw = np.array([row["Rated Capacity 47°F"], row["Rated Capacity 82°F"], row["Rated Capacity 95°F"]]) * BTU_TO_KW
        pACmx_ = -float(np.linalg.lstsq(c_delta_c.reshape(-1, 1), c_caps_kw - pACmx, rcond=None)[0][0]) / pACmx if pACmx > 0 else 0.0

        h_cops = np.array([row["COP at Rated Capacity 5°F"], row["COP at Rated Capacity 17°F"], pHVeff])
        pHVeff_ = float(np.linalg.lstsq(h_delta_c.reshape(-1, 1), h_cops - pHVeff, rcond=None)[0][0])

        c_cops = np.array([pACeff, row["COP at Rated Capacity 82°F"], row["COP at Rated Capacity 95°F"]])
        pACeff_ = -float(np.linalg.lstsq(c_delta_c.reshape(-1, 1), c_cops - pACeff, rcond=None)[0][0])

        name_parts = []
        for col in ["Brand Name", "Model Number", "outdoor_model_number"]:
            if col in row.index and pd.notna(row[col]):
                name_parts.append(str(row[col]).strip())
        ty = " | ".join(name_parts) if name_parts else f"unit_{row.name}"

        records.append({
            "ty": ty, "pHVmx": pHVmx, "pACmx": pACmx, "pHVeff": pHVeff, "pACeff": pACeff,
            "temp": DESIGN_TEMP_C, "pHVmx_": pHVmx_, "pACmx_": pACmx_,
            "pHVeff_": pHVeff_, "pACeff_": pACeff_, "pHSPF": row["HSPF (Region IV)"],
        })

    result = pd.DataFrame(records).reset_index(drop=True)
    log.info(f"NEEP: {len(result)} eligible Mitsubishi units ({n_excluded} excluded for COP > 6)")
    return result


# ---------------------------------------------------------------------------
# Capacity / COP interpolation
# ---------------------------------------------------------------------------

def capacity_at_temp(row, outdoor_temp_c, mode):
    delta = outdoor_temp_c - row["temp"]
    if mode == "heating":
        cap = row["pHVmx"] * (1 + row["pHVmx_"] * delta)
    else:
        cap = row["pACmx"] * (1 - row["pACmx_"] * delta)
    return max(cap, 0.0)


def cop_at_temp(row, outdoor_temp_c, mode):
    delta = outdoor_temp_c - row["temp"]
    if mode == "heating":
        cop = row["pHVeff"] + row["pHVeff_"] * delta
    else:
        cop = row["pACeff"] - row["pACeff_"] * delta
    return max(cop, 1.0)


# ---------------------------------------------------------------------------
# Sizing helpers
# ---------------------------------------------------------------------------

def _sizing_load_and_temp(load_series, temp_series, percentile):
    nonzero = load_series[load_series > 0]
    threshold = nonzero.quantile(percentile / 100.0)
    if threshold == 0:
        threshold = load_series.max() * (percentile / 100.0)
    idx = (load_series - threshold).abs().idxmin()
    return load_series.loc[idx] * 4000, temp_series.loc[idx] * 1.8 + 32


def _absolute_peak_load_and_temp(load_series, temp_series):
    nonzero = load_series[load_series > 0]
    idx = nonzero.idxmax()
    return load_series.loc[idx] * 4000, temp_series.loc[idx] * 1.8 + 32


def get_peak_loads(consumption_df, conditioned_area_sqft, percentile):
    temp_series = consumption_df["out.outdoor_air_drybulb_temp..c"]

    heating_series = consumption_df["out.load.heating.energy_delivered..kbtu"]
    if heating_series.isna().all() or heating_series.max() == 0:
        fallback = conditioned_area_sqft * 30
        fallback_temp = temp_series.min() * 1.8 + 32
        sizing_h, sizing_ht = fallback, fallback_temp
        abs_h, abs_ht = fallback, fallback_temp
    else:
        sizing_h, sizing_ht = _sizing_load_and_temp(heating_series, temp_series, percentile)
        abs_h, abs_ht = _absolute_peak_load_and_temp(heating_series, temp_series)

    cooling_series = consumption_df["out.load.cooling.energy_delivered..kbtu"]
    if cooling_series.isna().all() or cooling_series.max() == 0:
        fallback = conditioned_area_sqft * 30
        fallback_temp = temp_series.max() * 1.8 + 32
        sizing_c, sizing_ct = fallback, fallback_temp
        abs_c, abs_ct = fallback, fallback_temp
    else:
        sizing_c, sizing_ct = _sizing_load_and_temp(cooling_series, temp_series, percentile)
        abs_c, abs_ct = _absolute_peak_load_and_temp(cooling_series, temp_series)

    return sizing_h, sizing_ht, sizing_c, sizing_ct, abs_h, abs_ht, abs_c, abs_ct


# ---------------------------------------------------------------------------
# Capacity filtering & multi-unit matching
# ---------------------------------------------------------------------------

def get_capacity_at_design_temp(hp_df, consumption_df, conditioned_area_sqft, percentile):
    (sizing_h, sizing_ht, sizing_c, sizing_ct,
     abs_h, abs_ht, abs_c, abs_ct) = get_peak_loads(consumption_df, conditioned_area_sqft, percentile)

    outdoor_c_h = (sizing_ht - 32) / 1.8
    outdoor_c_c = (sizing_ct - 32) / 1.8

    result = hp_df.copy()
    result["capacity_kw_heating"] = result.apply(lambda r: capacity_at_temp(r, outdoor_c_h, "heating"), axis=1)
    result["capacity_kw_cooling"] = result.apply(lambda r: capacity_at_temp(r, outdoor_c_c, "cooling"), axis=1)
    result["capacity_btu_heating"] = result["capacity_kw_heating"] * KW_TO_BTU
    result["capacity_btu_cooling"] = result["capacity_kw_cooling"] * KW_TO_BTU
    result["sufficient_capacity"] = (
        (result["capacity_btu_heating"] >= sizing_h) &
        (result["capacity_btu_cooling"] >= sizing_c)
    )
    return result, sizing_h, sizing_ht, sizing_c, sizing_ct, abs_h, abs_ht, abs_c, abs_ct


def find_multi_unit_options(hp_df, peak_load_heating, peak_load_cooling, max_units):
    for n in range(2, max_units + 1):
        h_per = peak_load_heating / n
        c_per = peak_load_cooling / n
        eligible = hp_df[
            (hp_df["capacity_btu_heating"] >= h_per) &
            (hp_df["capacity_btu_cooling"] >= c_per)
        ].copy()
        if not eligible.empty:
            eligible["sufficient_capacity"] = True
            return eligible, n
    return None, None


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def estimate_total_cost(capacity_btu, hspf, conditioned_area_sqft, has_ductwork):
    base_appliance = 933.88 + 0.1480 * capacity_btu + 1.1126 * conditioned_area_sqft
    eff_adj = base_appliance * 0.09 * (hspf - 10)
    appliance = base_appliance + eff_adj
    labor = 2286 + 0.1179 * capacity_btu + 0.6556 * conditioned_area_sqft
    ductwork = 0.0 if has_ductwork else 0.883 * conditioned_area_sqft
    scaled_misc = 74.797 + 1.1028 * conditioned_area_sqft
    flat_misc = 180.0
    return appliance + labor + ductwork + scaled_misc + flat_misc


def build_cost_menu(hp_df, conditioned_area_sqft, has_ductwork):
    def _cost(row):
        cap_btu = row["pHVmx"] * KW_TO_BTU
        hspf = float(row.get("pHSPF", 10.0)) if pd.notna(row.get("pHSPF")) else 10.0
        return estimate_total_cost(cap_btu, hspf, conditioned_area_sqft, has_ductwork)
    hp_df = hp_df.copy()
    hp_df["cost_total"] = hp_df.apply(_cost, axis=1)
    return hp_df


# ---------------------------------------------------------------------------
# Per-building processing
# ---------------------------------------------------------------------------

def process_building(building_id, hp_base, metadata_df, consumption_dir, percentile, max_units):
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
    except Exception as e:
        log.warning(f"Building {building_id}: failed to read consumption — {e}")
        return None, None

    try:
        (hp_df, sizing_h, sizing_ht, sizing_c, sizing_ct,
         abs_h, abs_ht, abs_c, abs_ct) = get_capacity_at_design_temp(hp_base, consumption_df, sqft, percentile)
    except Exception as e:
        log.warning(f"Building {building_id}: sizing failed — {e}")
        return None, None

    n_sufficient = hp_df["sufficient_capacity"].sum()

    if n_sufficient == 0:
        hp_multi, n_units = find_multi_unit_options(hp_df, sizing_h, sizing_c, max_units)
        if hp_multi is None:
            return None, {"bldg_id": building_id, "peak_load_btu_hr": abs_h, "outdoor_temp_f": round(abs_ht, 1)}
        hp_multi = build_cost_menu(hp_multi, sqft, has_ducts)
        cheapest = hp_multi[hp_multi["sufficient_capacity"]].nsmallest(1, "cost_total").iloc[0]
    else:
        hp_df = build_cost_menu(hp_df, sqft, has_ducts)
        cheapest = hp_df[hp_df["sufficient_capacity"]].nsmallest(1, "cost_total").iloc[0]
        n_units = 1

    temp_f = consumption_df["out.outdoor_air_drybulb_temp..c"] * 1.8 + 32

    metrics = {
        "bldg_id": building_id,
        "hp_index": cheapest.name,
        "pHSPF": cheapest["pHSPF"],
        "lowest_outdoor_temp_f": round(temp_f.min(), 1),
        "highest_outdoor_temp_f": round(temp_f.max(), 1),
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def get_sizing_settings(cfg):
    hp_cfg = cfg.get("heatpump") or {}
    missing = [k for k in ("sizing_percentile", "max_units") if hp_cfg.get(k) is None]
    if missing:
        raise KeyError(f"config.yaml is missing heatpump.{', heatpump.'.join(missing)}")
    return float(hp_cfg["sizing_percentile"]), int(hp_cfg["max_units"])


def run_heatpump_selection(cfg):
    input_dir = cfg["paths"]["input_dir"]
    output_dir = cfg["paths"]["output_dir"]
    state = cfg["state"]
    release_year = cfg["resstock"]["release_year"]

    hp_cfg = cfg.get("heatpump", {})
    percentile, max_units = get_sizing_settings(cfg)
    log.info(f"Sizing: {percentile:g}th percentile of nonzero load, up to {max_units} units")
    neep_path = hp_cfg.get("neep_database")
    if neep_path:
        neep_path = Path(neep_path)
    else:
        neep_path = input_dir / "neep_database.csv"

    if not neep_path.exists():
        log.error(f"NEEP database not found: {neep_path}")
        return

    hp_base = load_neep_heat_pumps(neep_path)

    metadata_path = input_dir / f"{state}_upgrade0.parquet"
    if not metadata_path.exists():
        log.error(f"Metadata file not found: {metadata_path}")
        return
    metadata_df = read_parquet(metadata_path, columns=METADATA_COLS)
    metadata_df = metadata_df.set_index("bldg_id")

    id_file = input_dir / f"resstock_bldg_ids_{state.lower()}_{release_year}.csv"
    if not id_file.exists():
        log.error(f"Building ID file not found: {id_file}")
        return
    building_ids = pd.read_csv(str(id_file))["bldg_id"].dropna().astype(int).tolist()

    if cfg.get("test_mode") and cfg.get("test_limit"):
        building_ids = building_ids[:cfg["test_limit"]]

    state = cfg["state"]
    consumption_dir = input_dir / "residential_profiles" / state

    heatpump_dir = output_dir / "heatpump_files" / state
    heatpump_dir.mkdir(parents=True, exist_ok=True)

    hp_export = hp_base.copy()
    hp_export.insert(hp_export.columns.get_loc("ty") + 1, "hp_index", hp_export.index)
    hp_export.to_csv(str(heatpump_dir / "available_heatpumps.csv"), index=False)
    log.info(f"Saved {len(hp_base)} available HP units to available_heatpumps.csv")

    log.info(f"Processing {len(building_ids)} buildings for heat pump selection")

    summary_rows = []
    zero_match = []
    processed = 0
    skipped = 0

    for i, bldg_id in enumerate(building_ids, 1):
        metrics, zero = process_building(bldg_id, hp_base, metadata_df, consumption_dir,
                                         percentile, max_units)
        if metrics is not None:
            summary_rows.append(metrics)
            processed += 1
        elif zero is not None:
            zero_match.append(zero)
            skipped += 1
        else:
            skipped += 1

        if i % 100 == 0:
            log.info(f"  [{i}/{len(building_ids)}] processed={processed}, skipped={skipped}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("bldg_id").reset_index(drop=True)
        summary_df.to_csv(str(heatpump_dir / "building_hp_summary.csv"), index=False)
        log.info(f"HP summary saved: {len(summary_df)} buildings")

    if zero_match:
        zero_df = pd.DataFrame(zero_match).sort_values("bldg_id").reset_index(drop=True)
        zero_df.to_csv(str(heatpump_dir / "zero_hp_match_buildings.csv"), index=False)
        log.info(f"{len(zero_match)} buildings with no matching HP unit")

    log.info(f"Heat pump selection done: {processed} matched, {skipped} skipped/failed")
    log.info(f"Results saved to {heatpump_dir}")


if __name__ == "__main__":
    cfg = load_config()
    run_heatpump_selection(cfg)
