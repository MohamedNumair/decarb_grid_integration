import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).parents[1]))
from config.loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def read_parquet(path):
    try:
        return pd.read_parquet(str(path), engine="fastparquet")
    except Exception:
        return pd.read_parquet(str(path), engine="pyarrow")


# ---------------------------------------------------------------------------
# History-based method: classify heating/cooling/shoulder from consumption data
# ---------------------------------------------------------------------------

HISTORY_COLS = [
    "timestamp",
    "out.indoor_radiant_temperature.conditioned_space..c",
    "out.outdoor_air_drybulb_temp..c",
]


def compute_schedule_history(parquet_path, hysteresis):
    df = read_parquet(parquet_path)
    cols_present = [c for c in HISTORY_COLS if c in df.columns]
    df = df[cols_present]

    timestamps = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M")
    indoor = df["out.indoor_radiant_temperature.conditioned_space..c"].values
    outdoor = df["out.outdoor_air_drybulb_temp..c"].values

    y = (indoor > outdoor).astype(float)
    n = len(y)
    steps_per_day = n / 365
    window = int(steps_per_day * hysteresis["smoothing_days"])

    smoothed = uniform_filter1d(y, size=window)

    schedule = np.zeros(n, dtype=int)
    state = 0
    if smoothed[0] >= hysteresis["heat_enter"]:
        state = 1
    elif smoothed[0] <= hysteresis["cool_enter"]:
        state = -1

    for i in range(n):
        if state == 1:
            if smoothed[i] < hysteresis["heat_exit"]:
                state = 0
        elif state == -1:
            if smoothed[i] > hysteresis["cool_exit"]:
                state = 0
        else:
            if smoothed[i] >= hysteresis["heat_enter"]:
                state = 1
            elif smoothed[i] <= hysteresis["cool_enter"]:
                state = -1
        schedule[i] = state

    return timestamps, schedule


def run_history_schedules(cfg):
    input_dir = cfg["paths"]["input_dir"]
    output_dir = cfg["paths"]["output_dir"]
    hysteresis = cfg["schedules"]["hysteresis"]

    state = cfg["state"]
    profiles_dir = input_dir / "residential_profiles" / state
    schedule_dir = input_dir / "schedules" / state
    schedule_dir.mkdir(parents=True, exist_ok=True)
    release_year = cfg["resstock"]["release_year"]
    id_file = input_dir / f"resstock_bldg_ids_{state.lower()}_{release_year}.csv"

    if not id_file.exists():
        log.error(f"Building ID file not found: {id_file}")
        return

    building_ids = pd.read_csv(str(id_file))["bldg_id"].dropna().astype(int).tolist()

    if cfg.get("test_mode") and cfg.get("test_limit"):
        building_ids = building_ids[:cfg["test_limit"]]

    log.info(f"History method: processing {len(building_ids)} buildings")

    processed = 0
    failed = 0
    for i, building_id in enumerate(building_ids, 1):
        building_id = str(building_id)
        ppath = profiles_dir / f"{building_id}-0.parquet"
        if not ppath.exists():
            failed += 1
            continue

        try:
            timestamps, schedule = compute_schedule_history(ppath, hysteresis)
        except Exception as e:
            log.warning(f"Skipped building {building_id}: {e}")
            failed += 1
            continue

        out_df = pd.DataFrame({
            "timestamp": timestamps,
            "building_id": int(building_id),
            "schedule": schedule,
        })
        out_df.to_parquet(str(schedule_dir / f"schedule_{building_id}.parquet"), index=False)
        processed += 1

        if i % 100 == 0:
            log.info(f"  [{i}/{len(building_ids)}] processed")

    log.info(f"History schedules done: {processed} processed, {failed} failed")
    log.info(f"Saved to {schedule_dir}")


# ---------------------------------------------------------------------------
# Metadata-based method: setpoint timeseries from ResStock metadata fields
# ---------------------------------------------------------------------------

SETPOINT_COLS = [
    "bldg_id",
    "in.heating_setpoint",
    "in.heating_setpoint_has_offset",
    "in.heating_setpoint_offset_magnitude",
    "in.heating_setpoint_offset_period",
    "in.cooling_setpoint",
    "in.cooling_setpoint_has_offset",
    "in.cooling_setpoint_offset_magnitude",
    "in.cooling_setpoint_offset_period",
]


def fahrenheit_to_celsius(value):
    return (value - 32) * 5 / 9


def generate_building_schedule_offset(mode, building_row):
    night_start, night_end = 22, 7
    day_start, day_end = 9, 17

    schedule = np.zeros(24)

    has_offset = f"in.{mode}_setpoint_has_offset"
    offset_mag = f"in.{mode}_setpoint_offset_magnitude"
    offset_per = f"in.{mode}_setpoint_offset_period"

    if building_row[has_offset] != "Yes":
        return schedule

    magnitude = int(str(building_row[offset_mag]).replace("F", ""))
    period = str(building_row[offset_per])

    offset_hours = 0
    if period[-1] == "h":
        try:
            offset_hours = int(period[-3:-1])
        except ValueError:
            pass

    if "Night" in period:
        ns = night_start + offset_hours
        ne = night_end + offset_hours
        if ns >= 24:
            ns = ns % 24
            schedule[ns:ne] = 1
        else:
            schedule[ns:] = 1
            schedule[:ne] = 1

    if "Day" in period:
        ds = day_start + offset_hours
        de = day_end + offset_hours
        if ds > de:
            schedule[ds:] = 1
            schedule[:de] = 1
        else:
            schedule[ds:de] = 1

    schedule *= magnitude
    return schedule


def get_base_setpoints(building_row):
    heating = int(str(building_row["in.heating_setpoint"]).replace("F", ""))
    cooling = int(str(building_row["in.cooling_setpoint"]).replace("F", ""))

    if heating > cooling:
        avg = (heating + cooling) / 2
        return avg, avg, avg

    shoulder = (heating + cooling) / 2
    return heating, cooling, shoulder


def generate_setpoint_timeseries(building_row, freq, year):
    heating_sp, cooling_sp, shoulder_sp = get_base_setpoints(building_row)

    ts_index = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq=freq)

    modes = {}
    for mode, base_sp in [("heating", heating_sp), ("cooling", cooling_sp), ("shoulder", shoulder_sp)]:
        setpoints = np.full(len(ts_index), base_sp, dtype=float)

        if mode in ("heating", "cooling"):
            offset_schedule = generate_building_schedule_offset(mode, building_row)
            hours = ts_index.hour
            for h in range(24):
                if offset_schedule[h] != 0:
                    setpoints[hours == h] += offset_schedule[h]

        if mode == "heating":
            min_comfort = setpoints.copy()
            max_comfort = setpoints + 1.8
        elif mode == "cooling":
            min_comfort = setpoints - 1.8
            max_comfort = setpoints.copy()
        else:
            min_comfort = setpoints - 0.9
            max_comfort = setpoints + 0.9

        modes[mode] = pd.DataFrame({
            "setpoint": np.vectorize(fahrenheit_to_celsius)(setpoints),
            "min_comfort_temp": np.vectorize(fahrenheit_to_celsius)(min_comfort),
            "max_comfort_temp": np.vectorize(fahrenheit_to_celsius)(max_comfort),
        }, index=ts_index)

    return modes


def run_metadata_setpoints(cfg):
    input_dir = cfg["paths"]["input_dir"]
    output_dir = cfg["paths"]["output_dir"]
    state = cfg["state"]
    year = cfg["year"]
    freq = cfg["schedules"]["resolution"]

    metadata_path = input_dir / f"{state}_upgrade0.parquet"
    meta = read_parquet(metadata_path)

    if "bldg_id" not in meta.columns:
        meta = meta.reset_index().rename(columns={"index": "bldg_id"})

    available_cols = [c for c in SETPOINT_COLS if c in meta.columns]
    meta = meta[available_cols].copy()
    meta.set_index("bldg_id", inplace=True)

    schedule_dir = input_dir / "schedules" / state
    setpoint_dir = input_dir / "setpoint_timeseries" / state
    setpoint_dir.mkdir(parents=True, exist_ok=True)

    schedule_files = sorted(schedule_dir.glob("schedule_*.parquet"))
    available_ids = [int(f.stem.replace("schedule_", "")) for f in schedule_files]

    if not available_ids:
        log.warning("No schedule files found — run history method first or generate schedules")
        available_ids = meta.index.tolist()

    meta = meta.loc[meta.index.isin(available_ids)]

    if cfg.get("test_mode") and cfg.get("test_limit"):
        meta = meta.head(cfg["test_limit"])

    log.info(f"Metadata method: generating setpoints for {len(meta)} buildings")

    schedule_to_mode = {1: "heating", 0: "shoulder", -1: "cooling"}

    processed = 0
    for i, (bldg_id, row) in enumerate(meta.iterrows(), 1):
        try:
            modes = generate_setpoint_timeseries(row, freq, year)
        except Exception as e:
            log.warning(f"Skipped building {bldg_id}: {e}")
            continue

        schedule_file = schedule_dir / f"schedule_{bldg_id}.parquet"
        if schedule_file.exists():
            season = read_parquet(schedule_file)
            season["timestamp"] = pd.to_datetime(season["timestamp"])
            season.set_index("timestamp", inplace=True)

            # Use schedule index as base (matches original ercot_project)
            mode_ts = {k: v.reindex(season.index) for k, v in modes.items()}
            combined = pd.DataFrame(index=season.index)
            combined["mode"] = season["schedule"]
            for mode_val, mode_name in schedule_to_mode.items():
                mask = season["schedule"] == mode_val
                for col in ["setpoint", "min_comfort_temp", "max_comfort_temp"]:
                    combined.loc[mask, col] = mode_ts[mode_name].loc[mask, col]
        else:
            combined = modes["shoulder"]

        combined.index.name = "timestamp"
        combined.to_parquet(str(setpoint_dir / f"setpoint_{bldg_id}.parquet"))
        processed += 1

        if i % 100 == 0:
            log.info(f"  [{i}/{len(meta)}] processed")

    log.info(f"Metadata setpoints done: {processed} buildings")
    log.info(f"Saved to {setpoint_dir}")


# ---------------------------------------------------------------------------
# Statistical comfort envelope from consumption history
# ---------------------------------------------------------------------------

def compute_statistical_comfort_envelope(combined_ts, consumption_df,
                                         window_days=30, lower_q=0.05,
                                         upper_q=0.95):
    """Override comfort bounds with rolling quantiles of actual indoor temp."""
    indoor_col = "out.indoor_temperature.conditioned_space..c"
    if indoor_col not in consumption_df.columns:
        return combined_ts

    indoor_temp = consumption_df[indoor_col]
    if hasattr(indoor_temp, "dt"):
        pass
    if not isinstance(indoor_temp.index, pd.DatetimeIndex):
        return combined_ts

    indoor_temp = indoor_temp.reindex(combined_ts.index, method="nearest")

    freq = pd.infer_freq(combined_ts.index)
    if freq is None:
        freq = "15min"
    samples_per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))
    window_size = window_days * samples_per_day

    rolling_max = indoor_temp.rolling(window=window_size, min_periods=1).quantile(upper_q)
    rolling_min = indoor_temp.rolling(window=window_size, min_periods=1).quantile(lower_q)

    # Stabilize initial window
    if len(rolling_max) >= window_size:
        first_max = rolling_max.iloc[window_size - 1]
        first_min = rolling_min.iloc[window_size - 1]
        rolling_max.iloc[:window_size] = first_max
        rolling_min.iloc[:window_size] = first_min

    # Enforce minimum comfort band width
    min_band_width = 0.5  # °C
    min_half_width = 0.25
    band_width = rolling_max - rolling_min
    narrow_mask = band_width < min_band_width
    rolling_max[narrow_mask] = np.maximum(
        rolling_max[narrow_mask], indoor_temp[narrow_mask] + min_half_width)
    rolling_min[narrow_mask] = np.minimum(
        rolling_min[narrow_mask], indoor_temp[narrow_mask] - min_half_width)

    # Round to 0.5°C increments
    rolling_min = np.floor(rolling_min * 2) / 2
    rolling_max = np.ceil(rolling_max * 2) / 2

    combined_ts = combined_ts.copy()
    combined_ts["min_comfort_temp"] = rolling_min.values
    combined_ts["max_comfort_temp"] = rolling_max.values
    return combined_ts


def run_setpoints_with_envelope(cfg):
    """Generate setpoints: metadata base + schedule combine + statistical envelope."""
    input_dir = cfg["paths"]["input_dir"]
    state = cfg["state"]
    year = cfg["year"]
    freq = cfg["schedules"]["resolution"]

    metadata_path = input_dir / f"{state}_upgrade0.parquet"
    meta = read_parquet(metadata_path)
    if "bldg_id" not in meta.columns:
        meta = meta.reset_index().rename(columns={"index": "bldg_id"})
    available_cols = [c for c in SETPOINT_COLS if c in meta.columns]
    meta = meta[available_cols].copy()
    meta.set_index("bldg_id", inplace=True)

    schedule_dir = input_dir / "schedules" / state
    setpoint_dir = input_dir / "setpoint_timeseries" / state
    setpoint_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = input_dir / "residential_profiles" / state

    schedule_files = sorted(schedule_dir.glob("schedule_*.parquet"))
    available_ids = [int(f.stem.replace("schedule_", "")) for f in schedule_files]
    meta = meta.loc[meta.index.isin(available_ids)]

    if cfg.get("test_mode") and cfg.get("test_limit"):
        meta = meta.head(cfg["test_limit"])

    log.info(f"Generating setpoints with statistical envelope for {len(meta)} buildings")

    schedule_to_mode = {1: "heating", 0: "shoulder", -1: "cooling"}
    processed = 0
    failed = 0

    for i, (bldg_id, row) in enumerate(meta.iterrows(), 1):
        try:
            modes = generate_setpoint_timeseries(row, freq, year)
        except Exception as e:
            log.warning(f"Skipped building {bldg_id} (setpoint gen): {e}")
            failed += 1
            continue

        # Combine with schedule
        schedule_file = schedule_dir / f"schedule_{bldg_id}.parquet"
        if schedule_file.exists():
            season = read_parquet(schedule_file)
            season["timestamp"] = pd.to_datetime(season["timestamp"])
            season.set_index("timestamp", inplace=True)

            # Use schedule index as base (matches original ercot_project)
            mode_ts = {k: v.reindex(season.index) for k, v in modes.items()}
            combined = pd.DataFrame(index=season.index)
            combined["mode"] = season["schedule"]
            for mode_val, mode_name in schedule_to_mode.items():
                mask = season["schedule"] == mode_val
                for col in ["setpoint", "min_comfort_temp", "max_comfort_temp"]:
                    combined.loc[mask, col] = mode_ts[mode_name].loc[mask, col]
        else:
            combined = modes["shoulder"].copy()
            combined["mode"] = 0

        # Apply statistical comfort envelope from consumption data
        consumption_file = profiles_dir / f"{bldg_id}-0.parquet"
        if consumption_file.exists():
            try:
                consumption_df = read_parquet(consumption_file)
                if "timestamp" in consumption_df.columns:
                    consumption_df["timestamp"] = pd.to_datetime(consumption_df["timestamp"])
                    consumption_df = consumption_df.set_index("timestamp")
                if not isinstance(consumption_df.index, pd.DatetimeIndex):
                    if len(consumption_df) == 35040:
                        consumption_df.index = pd.date_range(
                            start=f"{year}-01-01", periods=35040, freq="15min")
                    elif len(consumption_df) == 8760:
                        consumption_df.index = pd.date_range(
                            start=f"{year}-01-01", periods=8760, freq="h")
                else:
                    consumption_df.index = consumption_df.index - pd.Timedelta(minutes=15)
                combined = compute_statistical_comfort_envelope(combined, consumption_df)
            except Exception as e:
                log.warning(f"Building {bldg_id}: envelope failed, using metadata bounds — {e}")

        combined.index.name = "timestamp"
        combined.to_parquet(str(setpoint_dir / f"setpoint_{bldg_id}.parquet"))
        processed += 1

        if i % 100 == 0:
            log.info(f"  [{i}/{len(meta)}] processed")

    log.info(f"Setpoints done: {processed} processed, {failed} failed")
    log.info(f"Saved to {setpoint_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_schedules(cfg):
    method = cfg["schedules"]["method"]

    if method == "history":
        log.info("Step 1: Generating season schedules from consumption data")
        run_history_schedules(cfg)
        log.info("Step 2: Generating setpoints with statistical comfort envelope")
        run_setpoints_with_envelope(cfg)
    elif method == "metadata":
        log.info("Step 1: Generating season schedules from consumption data")
        run_history_schedules(cfg)
        log.info("Step 2: Generating setpoint timeseries from metadata (no envelope)")
        run_metadata_setpoints(cfg)
    else:
        log.error(f"Unknown schedule method: {method}. Use 'history' or 'metadata'")


if __name__ == "__main__":
    cfg = load_config()
    run_schedules(cfg)
