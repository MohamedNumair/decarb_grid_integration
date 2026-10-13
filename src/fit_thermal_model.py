import time
import logging
import json
import sys
import concurrent.futures
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import lsq_linear
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parents[1]))
from config.loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def read_parquet(path, **kwargs):
    try:
        return pd.read_parquet(path, engine="fastparquet", **kwargs)
    except Exception:
        return pd.read_parquet(path, engine="pyarrow", **kwargs)


# ============================================================
# Column name constants
# ============================================================
TS_CONFIG = {
    "indoor_temp_key":  "out.indoor_temperature.conditioned_space..c",
    "cooling_load_key": "out.load.cooling.energy_delivered..kbtu",
    "heating_load_key": "out.load.heating.energy_delivered..kbtu",
}

SOLAR_COL    = "out.weather.direct_normal_solar_radiation..watt_per_m2"
WIND_COL     = "out.weather.wind_speed..meter_per_second"
HUMIDITY_COL = "out.outdoor_air_relative_humidity..percentage"
OUTDOOR_TEMP_COL = "out.outdoor_air_drybulb_temp..c"

LIGHTING_COL  = "out.electricity.lighting_interior.energy_consumption..kwh"
EQUIPMENT_COL = "out.electricity.plug_loads.energy_consumption..kwh"
DHW_COL       = "out.load.hot_water.energy_delivered..kbtu"
PEOPLE_COL    = "out.people.total_heating_rate..watt"

LIGHTING_FRACTION  = 1.0
EQUIPMENT_FRACTION = 1.0
DHW_FRACTION       = 0

THERMAL_FEATURE_SET = "baseline_internal_gains"

FEATURE_SETS = {
    THERMAL_FEATURE_SET: {
        "temp_diff_lag":               ["outdoor_temp_c", "indoor_temp_c"],
        "radiation":                   "solar_radiation_w",
        "internal_gains_minus_cool_w": [
            "heating_load_w",
            "lighting_gain_w",
            "equipment_gain_w",
            "dhw_gain_w",
            "people_gain_w",
            "cooling_load_w",
        ],
    },
}


def thermal_coeff_csv_path(cfg, feature_name=THERMAL_FEATURE_SET):
    """Path of the regression coefficient CSV written by the thermal model
    stage and read by building matching and DECARB input generation."""
    return (cfg["paths"]["output_dir"] / "thermal_model_outputs" / cfg["state"]
            / feature_name / f"regression_coeff_{feature_name}.csv")


# ============================================================
# Helper: Newey-West HAC standard errors
# ============================================================
def newey_west_cov(X, residuals, n_lags=None):
    T, k = X.shape
    if n_lags is None:
        n_lags = int(4 * (T / 100) ** (2 / 9))

    Xe = X * residuals[:, np.newaxis]
    S  = Xe.T @ Xe / T
    for lag in range(1, n_lags + 1):
        w     = 1 - lag / (n_lags + 1)
        gamma = (Xe[lag:].T @ Xe[:-lag]) / T
        S    += w * (gamma + gamma.T)

    XtX = X.T @ X / T
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    V = XtX_inv @ S @ XtX_inv / T
    return V


# ============================================================
# Create building timeseries from parquet files
# ============================================================
def create_building_timeseries(bldg_ids, cfg, save_outputs=False, skipped=None):
    """Build per-building model inputs. Buildings that can't be used are
    appended to `skipped` (if given) as {"bldg_id", "reason"}."""
    def _skip(bid, reason):
        if skipped is not None:
            skipped.append({"bldg_id": int(bid), "reason": reason})

    input_dir = cfg["paths"]["input_dir"]
    output_dir = cfg["paths"]["output_dir"]
    state = cfg["state"]
    profiles_dir = input_dir / "residential_profiles" / state

    resstock_file = input_dir / f"{state}_upgrade0.parquet"
    if not resstock_file.exists():
        raise FileNotFoundError(f"Missing ResStock metadata: {resstock_file}")

    resstock_df = read_parquet(resstock_file)
    resstock_df = resstock_df[["bldg_id", "in.sqft..ft2", "in.county"]].copy()
    resstock_df = resstock_df.rename(columns={"in.sqft..ft2": "sqft", "in.county": "county_code"})
    resstock_df["bldg_id"] = pd.to_numeric(resstock_df["bldg_id"], errors="coerce").astype("Int64")
    resstock_df["sqm"] = resstock_df["sqft"] * 0.092903

    bldg_df = pd.DataFrame({"bldg_id": pd.array(list(bldg_ids), dtype="Int64")})
    merged_bldg_df = pd.merge(bldg_df, resstock_df, on="bldg_id", how="left", validate="m:1")

    all_ts_dict = {}

    for _, row in merged_bldg_df.iterrows():
        bldg_id = row["bldg_id"]
        sqm     = row["sqm"]

        if pd.isna(bldg_id):
            continue
        if pd.isna(sqm):
            _skip(bldg_id, "not in metadata or no floor area")
            continue

        bldg_path = profiles_dir / f"{bldg_id}-0.parquet"
        if not bldg_path.exists():
            _skip(bldg_id, "profile file missing")
            continue

        try:
            df_bldg = read_parquet(bldg_path)
            if "timestamp" not in df_bldg.columns:
                _skip(bldg_id, "missing column: timestamp")
                continue

            df_bldg["datetime"] = pd.to_datetime(df_bldg["timestamp"])
            df_cols = df_bldg.columns

            if TS_CONFIG["indoor_temp_key"] not in df_cols:
                _skip(bldg_id, f"missing column: {TS_CONFIG['indoor_temp_key']}")
                continue
            if OUTDOOR_TEMP_COL not in df_cols:
                _skip(bldg_id, f"missing column: {OUTDOOR_TEMP_COL}")
                continue

            for load_key in ("cooling_load_key", "heating_load_key"):
                col = TS_CONFIG[load_key]
                if col not in df_cols or df_bldg[col].isna().all():
                    df_bldg[col] = 0

            has_solar    = SOLAR_COL    in df_cols
            has_wind     = WIND_COL     in df_cols
            has_humidity = HUMIDITY_COL in df_cols

            if not has_solar:
                _skip(bldg_id, f"missing column: {SOLAR_COL}")
                continue

            has_lighting  = LIGHTING_COL  in df_cols
            has_equipment = EQUIPMENT_COL in df_cols
            has_dhw       = DHW_COL       in df_cols
            has_people    = PEOPLE_COL    in df_cols

            base_cols = [
                "datetime", OUTDOOR_TEMP_COL,
                TS_CONFIG["indoor_temp_key"],
                TS_CONFIG["cooling_load_key"],
                TS_CONFIG["heating_load_key"],
            ]
            weather_cols  = [c for c, f in [(SOLAR_COL, has_solar), (WIND_COL, has_wind), (HUMIDITY_COL, has_humidity)] if f]
            internal_cols = [c for c, f in [(LIGHTING_COL, has_lighting), (EQUIPMENT_COL, has_equipment), (DHW_COL, has_dhw), (PEOPLE_COL, has_people)] if f]

            df_bldg = df_bldg[base_cols + weather_cols + internal_cols].copy()

            df_bldg.rename(columns={
                OUTDOOR_TEMP_COL:              "outdoor_temp_c",
                TS_CONFIG["indoor_temp_key"]:  "indoor_temp_c",
                TS_CONFIG["cooling_load_key"]: "cooling_load_kbtu",
                TS_CONFIG["heating_load_key"]: "heating_load_kbtu",
            }, inplace=True)

            df_bldg["cooling_load_w"] = df_bldg["cooling_load_kbtu"] * 1172.284
            df_bldg["heating_load_w"] = df_bldg["heating_load_kbtu"] * 1172.284

            if has_solar:
                df_bldg["solar_radiation_wm2"] = df_bldg[SOLAR_COL]
                df_bldg["solar_radiation_w"]   = df_bldg[SOLAR_COL] * sqm
                df_bldg.drop(columns=[SOLAR_COL], inplace=True)
            if has_wind:
                df_bldg.rename(columns={WIND_COL: "wind_speed"}, inplace=True)
            if has_humidity:
                df_bldg.rename(columns={HUMIDITY_COL: "outdoor_humidity"}, inplace=True)

            df_bldg["lighting_gain_w"]  = (df_bldg[LIGHTING_COL]  * 4000 * LIGHTING_FRACTION)  if has_lighting  else 0.0
            df_bldg["equipment_gain_w"] = (df_bldg[EQUIPMENT_COL] * 4000 * EQUIPMENT_FRACTION) if has_equipment else 0.0
            df_bldg["dhw_gain_w"]       = (df_bldg[DHW_COL]       * 1172.284 * DHW_FRACTION)   if has_dhw       else 0.0

            if has_lighting:  df_bldg.drop(columns=[LIGHTING_COL],  inplace=True)
            if has_equipment: df_bldg.drop(columns=[EQUIPMENT_COL], inplace=True)
            if has_dhw:       df_bldg.drop(columns=[DHW_COL],       inplace=True)

            if has_people:
                df_bldg.rename(columns={PEOPLE_COL: "people_gain_w"}, inplace=True)
            else:
                df_bldg["people_gain_w"] = 0.0

        except Exception as e:
            log.warning(f"Error reading {bldg_id}-0.parquet: {e}")
            _skip(bldg_id, f"read error: {type(e).__name__}")
            continue

        all_ts_dict[int(bldg_id)] = df_bldg.copy()

        if save_outputs:
            save_dir = output_dir / "thermal_model_outputs" / state / "timeseries"
            save_dir.mkdir(parents=True, exist_ok=True)
            df_bldg.to_csv(save_dir / f"{bldg_id}_in_ts.csv", index=False)

    return all_ts_dict


# ============================================================
# Compute regression coefficients
# ============================================================
def compute_regression_coefficients(all_ts_dict, required_columns=None, skipped=None):
    def _skip(bid, reason):
        if skipped is not None:
            skipped.append({"bldg_id": int(bid), "reason": reason})

    results = []

    if required_columns is None:
        required_columns = FEATURE_SETS[THERMAL_FEATURE_SET]

    for bldg_id, df in all_ts_dict.items():
        if df.shape[0] < 2:
            _skip(bldg_id, "fewer than 2 timesteps")
            continue

        df["cooling_load_w"] = df["cooling_load_w"].fillna(0)
        df["heating_load_w"] = df["heating_load_w"].fillna(0)
        df["indoor_temp_c"]  = (df["indoor_temp_c"] * 2).round() / 2

        df_lag = df.shift(1)

        X_dict = {}
        skip_building = False

        for feature_name, col_spec in required_columns.items():
            if isinstance(col_spec, list):
                if feature_name == "internal_gains_minus_cool_w":
                    net = df_lag[col_spec[0]].copy()
                    for extra_col in col_spec[1:-1]:
                        net = net + df_lag[extra_col].fillna(0)
                    net = net - df_lag[col_spec[-1]]
                    X_dict[feature_name] = net
                elif feature_name == "heat_minus_cool_w":
                    X_dict[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
                elif feature_name == "temp_diff_lag":
                    if col_spec[0] not in df_lag.columns or col_spec[1] not in df_lag.columns:
                        skip_building = True
                        break
                    X_dict[feature_name] = df_lag[col_spec[0]] - df_lag[col_spec[1]]
            elif "lag" in feature_name:
                if col_spec not in df_lag.columns:
                    skip_building = True
                    break
                X_dict[feature_name] = df_lag[col_spec]
            else:
                if col_spec not in df.columns:
                    skip_building = True
                    break
                X_dict[feature_name] = df[col_spec]

        if skip_building or not X_dict:
            _skip(bldg_id, "missing model feature columns")
            continue

        X = pd.DataFrame(X_dict)
        y = df["indoor_temp_c"] - df_lag["indoor_temp_c"]

        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        if X.shape[0] < 2:
            _skip(bldg_id, "fewer than 2 valid rows after dropping NaNs")
            continue

        X_vals = X.values.astype(np.float64)
        y_vals = y.values.astype(np.float64)

        n_features = X_vals.shape[1]
        result_lsq = lsq_linear(
            X_vals, y_vals,
            bounds=(np.zeros(n_features), np.inf * np.ones(n_features))
        )
        coefficients = result_lsq.x

        y_pred_delta = X_vals @ coefficients
        T_in_actual = df.loc[valid_idx, "indoor_temp_c"].astype(np.float64)
        T_in_lag    = df_lag.loc[valid_idx, "indoor_temp_c"].astype(np.float64)
        T_in_pred   = T_in_lag + y_pred_delta

        ss_res    = np.sum((T_in_actual.values - T_in_pred.values) ** 2)
        ss_tot    = np.sum((T_in_actual.values - T_in_actual.mean()) ** 2)
        r_squared = (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        mse       = mean_squared_error(T_in_actual.values, T_in_pred.values)

        T_obs     = X_vals.shape[0]
        n_lags    = int(4 * (T_obs / 100) ** (2 / 9))
        residuals = y_vals - y_pred_delta
        V_hac     = newey_west_cov(X_vals, residuals, n_lags=n_lags)
        se        = np.sqrt(np.diag(V_hac).clip(0))

        q = n_features
        try:
            V_hac_inv = np.linalg.inv(V_hac)
            f_stat = (coefficients @ V_hac_inv @ coefficients) / q
        except np.linalg.LinAlgError:
            V_hac_inv = np.linalg.pinv(V_hac)
            f_stat = (coefficients @ V_hac_inv @ coefficients) / q

        f_pval = 1 - scipy_stats.chi2.cdf(f_stat * q, df=q)

        row = {
            "bldg_id":   bldg_id,
            "k1":        coefficients[0],
            "k2":        coefficients[1],
            "k3":        coefficients[2],
            "mse":       mse,
            "r_squared": r_squared,
            "nw_lags":   n_lags,
            "se_k1":     se[0],
            "se_k2":     se[1],
            "se_k3":     se[2],
            "f_stat":    f_stat,
            "f_pval":    f_pval,
        }
        results.append(row)

    return results


# ============================================================
# Batch worker
# ============================================================
def _worker(args):
    (batch_idx, batch_ids, cfg, feature_sets, save_outputs, tmp_dir) = args

    skipped = []
    try:
        ts_dict = create_building_timeseries(
            bldg_ids=batch_ids, cfg=cfg, save_outputs=save_outputs, skipped=skipped,
        )

        counts = {}
        for feature_name, feature_cols in feature_sets.items():
            results = compute_regression_coefficients(
                all_ts_dict=ts_dict, required_columns=feature_cols, skipped=skipped,
            )
            if results:
                tmp_file = tmp_dir / f"{feature_name}_batch_{batch_idx:06d}.parquet"
                pd.DataFrame(results).to_parquet(tmp_file, index=False)
            counts[feature_name] = len(results)

        return counts, skipped

    except Exception as e:
        reason = f"batch error: {type(e).__name__}: {e}"
        return f"ERROR in batch {batch_idx}: {e}", [
            {"bldg_id": int(b), "reason": reason} for b in batch_ids]


def _report_skipped(skipped, results_dir, n_input):
    """Save the skipped-building list and log a count per reason."""
    skipped_df = pd.DataFrame(skipped, columns=["bldg_id", "reason"])
    skipped_df = skipped_df.drop_duplicates().sort_values("bldg_id")
    skipped_csv = results_dir / "skipped_buildings.csv"
    skipped_df.to_csv(skipped_csv, index=False)

    if skipped_df.empty:
        log.info(f"No buildings skipped ({n_input} input)")
        return skipped_csv

    log.warning(f"Skipped {skipped_df['bldg_id'].nunique()} of {n_input} buildings "
                f"(list: {skipped_csv}):")
    for reason, n in skipped_df["reason"].value_counts().items():
        log.warning(f"  {n:>6}  {reason}")
    return skipped_csv


def _raise_if_nothing_fitted(n_fitted, feature_name, skipped_csv):
    if n_fitted == 0:
        raise RuntimeError(
            f"Thermal model fitted 0 buildings for {feature_name}. "
            f"See {skipped_csv} for why each building was skipped.")


# ============================================================
# Run pipeline
# ============================================================
def run_pipeline(cfg, bldg_ids, feature_sets=None, save_outputs=False,
                 batch=True, batch_size=100, n_workers=20):
    output_dir = cfg["paths"]["output_dir"]
    state = cfg["state"]
    results_dir = output_dir / "thermal_model_outputs" / state

    if feature_sets is None:
        feature_sets = FEATURE_SETS

    bldg_ids_list = list(bldg_ids)
    total = len(bldg_ids_list)

    for feature_name, feature_cols in feature_sets.items():
        out_dir = results_dir / feature_name
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "feature_set": feature_name,
            "features": list(feature_cols.keys()),
            "constraints": "All coefficients (k1, k2, k3) constrained >= 0",
            "model_form": "T_in(t) = T_in(t-1) + k1*[T_out(t-1)-T_in(t-1)] + k2*Q_rad(t) + k3*[Q_net(t-1)]",
            "standard_errors": "Newey-West HAC, auto lag = int(4*(T/100)^(2/9))",
            "output_formats": ["csv", "parquet"],
            "batched": batch,
            "batch_size": batch_size if batch else None,
            "n_workers": n_workers if batch else None,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        with open(results_dir / feature_name / f"regression_coeff_{feature_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    if batch:
        batches = [bldg_ids_list[i:i + batch_size] for i in range(0, total, batch_size)]
        n_batches = len(batches)
        tmp_dir = results_dir / "_tmp_batch_results"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Leftovers from a run that crashed before merging would otherwise be
        # merged into this run's results
        stale = list(tmp_dir.glob("*_batch_*.parquet"))
        if stale:
            log.warning(f"Removing {len(stale)} leftover batch file(s) from an "
                        f"earlier interrupted run in {tmp_dir}")
            for f in stale:
                f.unlink()

        log.info(f"Running: {total} buildings, {n_batches} batches of {batch_size}, {n_workers} workers")

        worker_args = [
            (batch_idx, batch_ids, cfg, feature_sets, save_outputs, tmp_dir)
            for batch_idx, batch_ids in enumerate(batches)
        ]

        t_start = time.time()
        completed, errors, skipped = 0, [], []

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker, args): args[0] for args in worker_args}
            for future in concurrent.futures.as_completed(futures):
                batch_idx = futures[future]
                result, batch_skipped = future.result()
                skipped.extend(batch_skipped)
                completed += 1

                if isinstance(result, str):
                    errors.append(result)
                    log.error(f"[{completed}/{n_batches}] Batch {batch_idx} FAILED: {result}")
                else:
                    counts_str = " | ".join(f"{fs}: {n}" for fs, n in result.items())
                    log.info(f"[{completed}/{n_batches}] Batch {batch_idx} done "
                             f"({time.time()-t_start:.0f}s) | {counts_str}")

        if errors:
            log.error(f"{len(errors)} batch(es) failed")

        skipped_csv = _report_skipped(skipped, results_dir, total)

        log.info("Merging batch results...")
        for feature_name in feature_sets:
            tmp_files = sorted(tmp_dir.glob(f"{feature_name}_batch_*.parquet"))
            _raise_if_nothing_fitted(len(tmp_files), feature_name, skipped_csv)

            merged_df = pd.concat([pd.read_parquet(f) for f in tmp_files], ignore_index=True)
            out_csv     = thermal_coeff_csv_path(cfg, feature_name)
            out_parquet = out_csv.with_suffix(".parquet")
            merged_df.to_csv(out_csv, index=False)
            merged_df.to_parquet(out_parquet, index=False)
            log.info(f"Saved {feature_name}: {len(merged_df)} buildings -> {out_csv}")

            for f in tmp_files:
                f.unlink()

        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    else:
        log.info(f"Running single-pass: {total} buildings")

        skipped = []
        ts_dict = create_building_timeseries(
            bldg_ids=bldg_ids_list, cfg=cfg, save_outputs=save_outputs, skipped=skipped,
        )

        all_results = {
            feature_name: compute_regression_coefficients(
                all_ts_dict=ts_dict, required_columns=feature_cols, skipped=skipped,
            )
            for feature_name, feature_cols in feature_sets.items()
        }
        skipped_csv = _report_skipped(skipped, results_dir, total)

        for feature_name, results in all_results.items():
            _raise_if_nothing_fitted(len(results), feature_name, skipped_csv)
            results_df = pd.DataFrame(results)
            out_csv     = thermal_coeff_csv_path(cfg, feature_name)
            out_parquet = out_csv.with_suffix(".parquet")
            results_df.to_csv(out_csv, index=False)
            results_df.to_parquet(out_parquet, index=False)
            log.info(f"Saved {feature_name}: {len(results_df)} buildings -> {out_csv}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    cfg = load_config()
    input_dir = cfg["paths"]["input_dir"]
    state = cfg["state"]
    release_year = cfg["resstock"]["release_year"]

    id_file = input_dir / f"resstock_bldg_ids_{state.lower()}_{release_year}.csv"
    if not id_file.exists():
        raise FileNotFoundError(f"Missing building ID list: {id_file}")

    bldg_ids = pd.read_csv(id_file)["bldg_id"].dropna().astype(int).unique()

    if cfg.get("test_mode") and cfg.get("test_limit"):
        bldg_ids = bldg_ids[: cfg["test_limit"]]

    log.info(f"Fitting thermal models for {len(bldg_ids)} buildings")

    use_batching = len(bldg_ids) > 100

    run_pipeline(
        cfg=cfg,
        bldg_ids=bldg_ids,
        save_outputs=False,
        batch=use_batching,
        batch_size=100,
        n_workers=min(20, len(bldg_ids) // 100 + 1),
    )
