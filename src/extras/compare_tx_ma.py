"""
TX vs MA comparison on the same OpenDSS network.

Standalone script that:
  1. Matches TX buildings to the same P100U OpenDSS network used by MA studies
  2. Uses already-solved TX DECARB results (no new DECARB runs needed)
  3. Creates OpenDSS loadshapes from TX DECARB ts.csv outputs
  4. Runs OpenDSS power flow for each TX case
  5. Compares TX and MA results side by side

The TX DECARB cases have NO solar dimension — only heating electrification:
  elec_0_flat, elec_100_flat, elec_0_tou, elec_100_tou,
  elec_0_flat_cap, elec_100_flat_cap, elec_0_tou_cap, elec_100_tou_cap

Usage:
    python src/compare_tx_ma.py --ercot-root <path> [--ma-study N] [--workers 20]

Example (local):
    python src/compare_tx_ma.py --ercot-root Z:/ercot_project --ma-study 8
Example (server):
    python src/compare_tx_ma.py --ercot-root D:/shared/ercot_project --ma-study 8
"""

import argparse
import logging
import math
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[1]
if not (PROJECT_ROOT / "config").is_dir():
    PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# buy/sell converted from kWh per period to kW
from create_network_inputs import read_decarb_ts  # noqa: E402

# ── Column tags (same as building_load_matching.py) ──────────────────────
SUMMER_DEMAND_TAG = "out.qoi.electricity.maximum_daily_peak_summer..kw"
WINTER_DEMAND_TAG = "out.qoi.electricity.maximum_daily_peak_winter..kw"
CLIMATE_TAG = "in.building_america_climate_zone"
COUNTY_TAG = "in.county"

# ── Power factor constants (same as create_network_inputs.py) ────────────
PF_COMPRESSOR = 0.85
PF_RESISTIVE = 1.0
PF_PV = 0.98
PF_BASELINE = 0.95
HOURS_PER_YEAR = 8760

# ── Matching parameters ─────────────────────────────────────────────────
K_NEIGHBORS = 5
RNG_SEED = 42
DEFAULT_USAGE_LIMIT = 1

# ── TX DECARB case names ────────────────────────────────────────────────
TX_CASES = [
    "elec_0_flat_results",
    "elec_100_flat_results",
    "elec_0_tou_results",
    "elec_100_tou_results",
    "elec_0_flat_cap_results",
    "elec_100_flat_cap_results",
    "elec_0_tou_cap_results",
    "elec_100_tou_cap_results",
]


# ═════════════════════════════════════════════════════════════════════════
# Parquet reader
# ═════════════════════════════════════════════════════════════════════════

def read_parquet(path):
    try:
        return pd.read_parquet(str(path), engine="fastparquet")
    except Exception:
        return pd.read_parquet(str(path), engine="pyarrow")


# ═════════════════════════════════════════════════════════════════════════
# Step 1: Match TX buildings to the MA OpenDSS network
# ═════════════════════════════════════════════════════════════════════════

def load_tx_buildings(input_dir, ercot_root):
    """Load TX building metadata, filtered to those with DECARB results."""
    tx_parquet = input_dir / "TX_upgrade0.parquet"
    tx_df = read_parquet(tx_parquet)

    if "bldg_id" not in tx_df.columns:
        tx_df = tx_df.reset_index().rename(columns={"index": "bldg_id"})

    # Filter to buildings that have DECARB results (check first case)
    decarb_dir = Path(ercot_root) / "in" / "decarb_inputs" / TX_CASES[0]
    available_ids = set()
    for d in decarb_dir.iterdir():
        try:
            available_ids.add(int(d.name))
        except ValueError:
            pass

    before = len(tx_df)
    tx_df = tx_df[tx_df["bldg_id"].isin(available_ids)].copy()
    log.info(f"TX buildings: {before} total, {len(tx_df)} with DECARB results")

    return tx_df


def load_ma_feeder_summaries(input_dir, zone_name="P100U"):
    """Load feeder_summaries.parquet and filter to the target zone."""
    feeder_path = input_dir / "feeder_summaries.parquet"
    if not feeder_path.exists():
        raise FileNotFoundError(f"Missing feeder summaries: {feeder_path}")

    feeder_df = read_parquet(feeder_path)

    if zone_name and "zone" in feeder_df.columns:
        feeder_df = feeder_df[feeder_df["zone"] == zone_name].copy()
        log.info(f"Loaded {len(feeder_df)} buses in zone {zone_name}")
    else:
        log.info(f"Loaded {len(feeder_df)} feeder buses (all zones)")

    return feeder_df


def match_tx_to_network(feeder_df, tx_df, usage_limit="auto"):
    """Match TX buildings to MA network buses by summer/winter peak demand.

    Uses the same k-nearest-neighbor approach as building_load_matching.py,
    but without county or climate-zone matching (TX zones don't overlap MA).
    Matches purely on demand magnitude using a global KD-tree.
    """
    rng = np.random.default_rng(RNG_SEED)

    # Compute usage limit
    if usage_limit == "auto":
        n_buses = len(feeder_df)
        n_buildings = len(tx_df)
        usage_limit = math.ceil(15 * n_buses / n_buildings)
        log.info(f"Match diversity: usage_limit=auto → {usage_limit} "
                 f"(ceil(15 × {n_buses}/{n_buildings}))")
    else:
        usage_limit = int(usage_limit)

    # Build KD-tree on TX building demand
    tx_df = tx_df.copy().reset_index(drop=True)
    bldg_ids = tx_df["bldg_id"].to_numpy(dtype=int)
    summer_vals = tx_df[SUMMER_DEMAND_TAG].to_numpy(dtype=float)
    winter_vals = tx_df[WINTER_DEMAND_TAG].to_numpy(dtype=float)
    vectors = np.column_stack([summer_vals, winter_vals])
    tree = cKDTree(vectors)

    # Track usage with a dict (avoids pandas .loc on shrinking DataFrame)
    usage_count = {}
    # Track which rows are still active (not exhausted)
    active_mask = np.ones(len(tx_df), dtype=bool)

    def _rebuild_tree():
        nonlocal tree, vectors
        vectors = np.column_stack([summer_vals[active_mask], winter_vals[active_mask]])
        if len(vectors) > 0:
            tree = cKDTree(vectors)

    # Map from active-subset index back to original index
    def _active_indices():
        return np.where(active_mask)[0]

    match_ids = []
    match_summer = []
    match_winter = []

    for _, row in feeder_df.iterrows():
        bus_kw = row.get("kw")
        if pd.isna(bus_kw):
            match_ids.append(None)
            match_summer.append(None)
            match_winter.append(None)
            continue

        n_active = active_mask.sum()
        if n_active == 0:
            # Pool exhausted — reset
            active_mask[:] = True
            usage_count.clear()
            _rebuild_tree()
            n_active = len(tx_df)
            log.warning("Building pool exhausted, resetting")

        # Use bus kW as target for both summer and winter (best we have)
        target = np.array([[bus_kw, bus_kw]])
        k_actual = min(K_NEIGHBORS, n_active)

        dists, idxs = tree.query(target, k=k_actual)
        if k_actual == 1:
            dists = np.array([dists.item()])
            idxs = np.array([idxs.item()])
        else:
            dists = dists[0]
            idxs = idxs[0]

        # Inverse-distance weighted random pick
        weights = 1.0 / (dists + 1e-6)
        weights /= weights.sum()
        chosen = rng.choice(len(idxs), p=weights)

        # Map from tree index (active subset) back to original index
        active_idx = _active_indices()
        orig_idx = active_idx[idxs[chosen]]

        bldg_id = int(bldg_ids[orig_idx])
        match_ids.append(bldg_id)
        match_summer.append(float(summer_vals[orig_idx]))
        match_winter.append(float(winter_vals[orig_idx]))

        # Track usage and retire if exhausted
        usage_count[bldg_id] = usage_count.get(bldg_id, 0) + 1
        if usage_count[bldg_id] >= usage_limit:
            active_mask[orig_idx] = False
            _rebuild_tree()

    feeder_df = feeder_df.copy()
    feeder_df["nrel_climate_match"] = match_ids
    feeder_df["nrel_climate_summer_kw"] = match_summer
    feeder_df["nrel_climate_winter_kw"] = match_winter

    matched = feeder_df["nrel_climate_match"].notna().sum()
    unique = feeder_df["nrel_climate_match"].nunique()
    log.info(f"TX match: {matched}/{len(feeder_df)} buses matched, "
             f"{unique} unique buildings used")

    return feeder_df


# ═════════════════════════════════════════════════════════════════════════
# Step 2: Create OpenDSS loadshapes from TX DECARB results
# ═════════════════════════════════════════════════════════════════════════

def classify_heating_pf(decarb_case_dir, bldg_id):
    """Determine if heating is compressor-driven (HP) or resistive."""
    sp_path = decarb_case_dir / str(bldg_id) / "in" / "sp.csv"
    hvac_path = decarb_case_dir / str(bldg_id) / "in" / "hvac.csv"
    try:
        sp_df = pd.read_csv(str(sp_path))
        hvac_df = pd.read_csv(str(hvac_path))
        selected = sp_df["pHVAC0"].iloc[0]
        row = hvac_df[hvac_df["ty"] == selected]
        if row.empty:
            return PF_COMPRESSOR
        pHVmx = float(row["pHVmx"].iloc[0])
        pACmx = float(row["pACmx"].iloc[0])
        return PF_RESISTIVE if (pHVmx > 0 and pACmx == 0) else PF_COMPRESSOR
    except (FileNotFoundError, KeyError, IndexError):
        return PF_COMPRESSOR


def compute_kvar_curve(ts_df, pf_ht):
    """Reconstruct reactive power from equipment-level power factors."""
    dem = ts_df["dem"].to_numpy(dtype=float)
    hvac_ht = ts_df["hvac_HT"].to_numpy(dtype=float)
    hvac_ac = ts_df["hvac_AC"].to_numpy(dtype=float)
    pv = ts_df["pv"].to_numpy(dtype=float) if "pv" in ts_df.columns else np.zeros(len(dem))
    baseline = dem - hvac_ht - hvac_ac

    return (
        hvac_ht * np.tan(np.arccos(pf_ht))
        + hvac_ac * np.tan(np.arccos(PF_COMPRESSOR))
        + baseline * np.tan(np.arccos(PF_BASELINE))
        - pv * np.tan(np.arccos(PF_PV))
    )


def build_substitution_map(decarb_case_dir, building_ids, tx_meta):
    """For each failed building, find the closest successful building
    by peak demand and return {failed_id: substitute_id}.

    Same approach as create_network_inputs.build_substitution_map.
    """
    def _check_exists(bldg_id):
        bldg_id = int(bldg_id)
        ts_path = decarb_case_dir / str(bldg_id) / "out" / "ts.csv"
        return bldg_id, ts_path.exists()

    successful = []
    failed = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        for bldg_id, exists in pool.map(_check_exists, building_ids):
            if exists:
                successful.append(bldg_id)
            else:
                failed.append(bldg_id)

    if not failed or not successful:
        if failed:
            log.warning(f"No successful buildings to substitute {len(failed)} failures")
        return {}

    needed_cols = [SUMMER_DEMAND_TAG, WINTER_DEMAND_TAG]
    meta = tx_meta[["bldg_id"] + needed_cols].dropna().copy()
    meta["bldg_id"] = meta["bldg_id"].astype(int)
    meta = meta.set_index("bldg_id")

    success_meta = meta.loc[meta.index.isin(successful)]
    if success_meta.empty:
        return {}

    tree = cKDTree(success_meta[needed_cols].values)

    sub_map = {}
    for bldg_id in failed:
        if bldg_id not in meta.index:
            continue
        target = meta.loc[bldg_id, needed_cols].values.reshape(1, -1)
        dist, idx = tree.query(target, k=1)
        sub_map[bldg_id] = int(success_meta.index[idx[0]])

    log.info(f"Substitution: {len(failed)} failed → {len(sub_map)} mapped to nearest successful")
    return sub_map


def generate_opendss_case(decarb_case_dir, feeder_df, zone_name, output_dir,
                          sub_map=None):
    """Build OpenDSS case directory from TX DECARB outputs.

    Same logic as create_network_inputs.generate_opendss_loadshapes but
    standalone — reads DECARB results from the ercot_project directory.
    """
    if sub_map is None:
        sub_map = {}
    template_dir = PROJECT_ROOT / "in" / "opendss_input_templates"
    zone_template = template_dir / zone_name / "scenarios" / "base_timeseries" / "opendss"
    if not zone_template.exists():
        log.error(f"Zone template not found: {zone_template}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir(exist_ok=True)

    # ── Write per-building profile CSVs (parallelized) ──────────────────
    unique_bldgs = feeder_df["nrel_climate_match"].dropna().astype(int).unique()

    required_cols = {"buy", "sell", "dem", "hvac_HT", "hvac_AC"}

    def _process_building(bldg_id):
        """Read DECARB output, compute profiles, write CSVs. Returns result tuple."""
        bldg_id = int(bldg_id)
        source_id = bldg_id
        ts_df = read_decarb_ts(decarb_case_dir, bldg_id)

        if ts_df is None or not required_cols.issubset(ts_df.columns):
            sub_id = sub_map.get(bldg_id)
            if sub_id is not None:
                ts_df = read_decarb_ts(decarb_case_dir, sub_id)
                if ts_df is not None and required_cols.issubset(ts_df.columns):
                    source_id = sub_id
                else:
                    return ("skip", bldg_id, None, None)
            else:
                return ("skip", bldg_id, None, None)

        pf_ht = classify_heating_pf(decarb_case_dir, source_id)
        net_kw = ts_df["buy"].values - ts_df["sell"].values
        net_kvar = compute_kvar_curve(ts_df, pf_ht)

        np.savetxt(str(profiles_dir / f"decarb_kw_{bldg_id}.csv"),
                   net_kw, fmt="%.6f")
        np.savetxt(str(profiles_dir / f"decarb_kvar_{bldg_id}.csv"),
                   net_kvar, fmt="%.6f")

        pkw = float(np.max(np.abs(net_kw))) or 0.1
        pkvar = float(np.max(np.abs(net_kvar))) or 0.01
        status = "sub" if source_id != bldg_id else "ok"
        return (status, bldg_id, pkw, pkvar)

    peak_kw = {}
    peak_kvar = {}
    processed = 0
    substituted = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_process_building, bid): bid for bid in unique_bldgs}
        for future in as_completed(futures):
            status, bldg_id, pkw, pkvar = future.result()
            if status == "skip":
                skipped += 1
            else:
                peak_kw[bldg_id] = pkw
                peak_kvar[bldg_id] = pkvar
                processed += 1
                if status == "sub":
                    substituted += 1

    log.info(f"Profiles: {processed} buildings written "
             f"({substituted} substituted), {skipped} skipped")

    # ── Copy template feeders and rewrite DSS files ─────────────────────
    def _clear_readonly(path):
        for root, dirs, files in os.walk(str(path)):
            for name in dirs + files:
                try:
                    os.chmod(os.path.join(root, name), stat.S_IWRITE)
                except OSError:
                    pass

    src_feeders = []
    for sub in sorted(zone_template.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("p"):
            continue
        for fd in sorted(sub.iterdir()):
            if fd.is_dir() and (fd / "Master.dss").exists():
                src_feeders.append(fd)

    log.info(f"Copying {len(src_feeders)} feeder templates")

    for src_feeder_dir in src_feeders:
        substation = src_feeder_dir.parent.name
        feeder = src_feeder_dir.name
        dst_feeder_dir = output_dir / substation / feeder

        if dst_feeder_dir.exists():
            _clear_readonly(dst_feeder_dir)
            shutil.rmtree(str(dst_feeder_dir))
        shutil.copytree(str(src_feeder_dir), str(dst_feeder_dir),
                        ignore=shutil.ignore_patterns("*.csv"))

        feeder_loads = feeder_df[
            (feeder_df["substation"] == substation) &
            (feeder_df["feeder"] == feeder)
        ] if "feeder" in feeder_df.columns else pd.DataFrame()

        if feeder_loads.empty:
            continue

        used_bldg_ids = set()
        load_lines = []
        shape_lines = []

        for _, row in feeder_loads.iterrows():
            bldg_id = int(row["nrel_climate_match"])
            load_name = row["Name"]

            if bldg_id not in peak_kw:
                continue

            used_bldg_ids.add(bldg_id)
            shape_name = f"decarb_{bldg_id}"
            kw = peak_kw[bldg_id]
            kvar = peak_kvar[bldg_id]

            load_lines.append(
                f"New Load.{load_name} conn={row['conn']} bus1={row['bus1']} "
                f"kV={row['kv']} Vminpu={row['vminpu']} Vmaxpu={row['vmaxpu']} "
                f"model={row['model']} kW={kw} kvar={kvar} "
                f"Phases={row['phases']} yearly={shape_name}"
            )

        for bldg_id in sorted(used_bldg_ids):
            kw_file = (profiles_dir / f"decarb_kw_{bldg_id}.csv").as_posix()
            kvar_file = (profiles_dir / f"decarb_kvar_{bldg_id}.csv").as_posix()
            shape_lines.append(
                f"New Loadshape.decarb_{bldg_id} npts={HOURS_PER_YEAR} interval=1 "
                f"useactual=true mult=(file={kw_file}) qmult=(file={kvar_file})"
            )

        (dst_feeder_dir / "LoadShapes.dss").write_text(
            "\n\n".join(shape_lines) + "\n")
        (dst_feeder_dir / "Loads.dss").write_text(
            "\n\n".join(load_lines) + "\n")

        master_path = dst_feeder_dir / "Master.dss"
        master_text = master_path.read_text()
        master_text = master_text.replace(
            "Solve mode=yearly stepsize=15m number=35040",
            f"Solve mode=yearly stepsize=1h number={HOURS_PER_YEAR}",
        )
        master_path.write_text(master_text)

    return True


# ═════════════════════════════════════════════════════════════════════════
# Step 3: Run OpenDSS (same as run_opendss.py but standalone)
# ═════════════════════════════════════════════════════════════════════════

NUM_STEPS = 8760
STEPSIZE_H = 1


def discover_feeders(zone_dir):
    """Find all feeders with Master.dss in a case directory."""
    feeders = []
    for sub in sorted(zone_dir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("p"):
            continue
        for feeder_dir in sorted(sub.iterdir()):
            if not feeder_dir.is_dir():
                continue
            master = feeder_dir / "Master.dss"
            if master.exists():
                feeders.append({
                    "substation": sub.name,
                    "feeder": feeder_dir.name,
                    "master": str(master),
                    "dir": str(feeder_dir),
                })
    return feeders


def simulate_feeder(task):
    """Run 8760-hour power flow for a single feeder."""
    import opendssdirect as dss

    case_name = task["case"]
    feeder_name = task["feeder"]
    substation = task["substation"]
    master_path = task["master"]
    feeder_dir = task["dir"]

    t0 = time.time()

    try:
        dss.Text.Command("Clear")
        os.chdir(feeder_dir)

        with open(master_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            s = line.strip()
            if not s:
                continue
            low = s.lower()
            if low.startswith("solve") or low.startswith("export") or low.startswith("plot "):
                continue
            dss.Text.Command(s)

        dss.Text.Command(f"Set mode=yearly stepsize={STEPSIZE_H}h number=1")

        hours = np.empty(NUM_STEPS)
        v_min = np.empty(NUM_STEPS)
        v_max = np.empty(NUM_STEPS)
        v_mean = np.empty(NUM_STEPS)
        tot_kw = np.empty(NUM_STEPS)
        tot_kvar = np.empty(NUM_STEPS)
        loading = np.empty(NUM_STEPS)

        dss.Lines.First()
        head_line = dss.Lines.Name()
        all_lines = [head_line]
        while dss.Lines.Next() != 0:
            all_lines.append(dss.Lines.Name())
        hs_lines = [l for l in all_lines if "p100uhs" in l.lower()]
        if hs_lines:
            head_line = hs_lines[0]

        for step in range(NUM_STEPS):
            dss.Solution.Solve()
            hours[step] = step

            all_vmag = dss.Circuit.AllBusMagPu()
            valid = [v for v in all_vmag if v > 0.01]
            if valid:
                v_min[step] = min(valid)
                v_max[step] = max(valid)
                v_mean[step] = sum(valid) / len(valid)
            else:
                v_min[step] = v_max[step] = v_mean[step] = float("nan")

            p = dss.Circuit.TotalPower()
            tot_kw[step] = -p[0]
            tot_kvar[step] = -p[1]

            dss.Circuit.SetActiveElement(f"Line.{head_line}")
            currents = dss.CktElement.CurrentsMagAng()
            n_cond = dss.CktElement.NumConductors()
            i_max = max(currents[2 * i] for i in range(n_cond))
            norm_amps = dss.CktElement.NormalAmps()
            loading[step] = (i_max / norm_amps * 100) if norm_amps > 0 else 0.0

        elapsed = round(time.time() - t0, 2)

        result_df = pd.DataFrame({
            "feeder": feeder_name,
            "substation": substation,
            "hour": hours.astype(int),
            "v_min_pu": v_min,
            "v_max_pu": v_max,
            "v_mean_pu": v_mean,
            "total_kW": tot_kw,
            "total_kvar": tot_kvar,
            "head_loading_pct": loading,
        })

        return {
            "status": "ok",
            "case": case_name,
            "feeder": feeder_name,
            "substation": substation,
            "elapsed": elapsed,
            "result": result_df,
        }

    except Exception as e:
        return {
            "status": "error",
            "case": case_name,
            "feeder": feeder_name,
            "substation": substation,
            "error": str(e),
            "elapsed": round(time.time() - t0, 2),
        }


def run_opendss_cases(tx_study_dir, cases, workers=20):
    """Run OpenDSS for all TX cases."""
    opendss_root = tx_study_dir / "opendss"
    results_dir = tx_study_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for case_name in cases:
        case_dir = opendss_root / case_name
        if not case_dir.exists():
            log.warning(f"No OpenDSS case directory for {case_name}, skipping")
            continue

        feeders = discover_feeders(case_dir)
        if not feeders:
            log.warning(f"No feeders found in {case_dir}, skipping")
            continue

        tasks = [{**f, "case": case_name} for f in feeders]
        log.info(f"Running {len(tasks)} feeders for case: {case_name}")

        all_results = []
        errors = 0

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(simulate_feeder, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "ok":
                    all_results.append(result["result"])
                    log.info(f"  {result['substation']}/{result['feeder']}: "
                             f"{result['elapsed']}s")
                else:
                    errors += 1
                    log.error(f"  {result['substation']}/{result['feeder']}: "
                              f"FAILED - {result.get('error', 'unknown')}")

        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            out_path = results_dir / f"{case_name}.parquet"
            combined.to_parquet(str(out_path), index=False)
            log.info(f"Saved {case_name}: {len(combined)} rows, "
                     f"{len(all_results)} feeders OK, {errors} errors")
        else:
            log.error(f"No results for {case_name}")


# ═════════════════════════════════════════════════════════════════════════
# Step 4: Compare TX vs MA results
# ═════════════════════════════════════════════════════════════════════════

def load_ma_results(ma_study_dir):
    """Load MA OpenDSS results from a completed study.

    MA results are stored as zone_yearly.csv inside per-case subdirectories
    under opendss_results/.
    """
    results_dir = ma_study_dir / "opendss_results"
    if not results_dir.exists():
        results_dir = ma_study_dir / "results"
    if not results_dir.exists():
        log.warning(f"No results directory found in {ma_study_dir}")
        return {}

    results = {}
    for case_dir in sorted(results_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        csv_path = case_dir / "zone_yearly.csv"
        if csv_path.exists():
            results[case_dir.name] = pd.read_csv(str(csv_path))
        else:
            # Also check for parquet
            parquet_path = case_dir / "zone_yearly.parquet"
            if parquet_path.exists():
                results[case_dir.name] = pd.read_parquet(str(parquet_path))

    log.info(f"Loaded {len(results)} MA result cases from {results_dir}")
    return results


def compute_zone_thermal_profiles(decarb_case_dir, matched_parquet_path,
                                   sub_map=None, workers=20):
    """Aggregate building-level DECARB ts.csv into zone-level hourly profiles.

    Returns dict with keys 'total_kw' and 'thermal_kw', each a numpy array
    of shape (8760,) representing zone-level hourly kW.
    """
    if sub_map is None:
        sub_map = {}

    feeder_df = read_parquet(matched_parquet_path)
    bldg_col = "nrel_climate_match" if "nrel_climate_match" in feeder_df.columns else "bldg_id"

    # Count how many buses each building is mapped to
    bldg_counts = (feeder_df[bldg_col].dropna().astype(int)
                   .value_counts().to_dict())

    required_cols = {"buy", "sell", "hvac_HT", "hvac_AC"}

    def _read_building(bldg_id):
        bldg_id = int(bldg_id)
        source_id = bldg_id
        ts_df = read_decarb_ts(decarb_case_dir, bldg_id)
        if ts_df is None or not required_cols.issubset(ts_df.columns):
            source_id = sub_map.get(bldg_id)
            if source_id is not None:
                ts_df = read_decarb_ts(decarb_case_dir, source_id)
            if ts_df is None or not required_cols.issubset(ts_df.columns):
                return None
        count = bldg_counts.get(bldg_id, 1)
        total = (ts_df["buy"].values - ts_df["sell"].values) * count
        thermal = (ts_df["hvac_HT"].values + ts_df["hvac_AC"].values) * count
        return total, thermal

    zone_total = np.zeros(HOURS_PER_YEAR)
    zone_thermal = np.zeros(HOURS_PER_YEAR)
    n_ok = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_read_building, bid): bid
                   for bid in bldg_counts.keys()}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                zone_total += result[0]
                zone_thermal += result[1]
                n_ok += 1

    log.info(f"Thermal profiles: {n_ok}/{len(bldg_counts)} buildings aggregated")
    return {"total_kw": zone_total, "thermal_kw": zone_thermal}


def compare_results(tx_study_dir, ma_study_dir, ercot_root=None):
    """Generate comparison plots and summary statistics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tx_results_dir = tx_study_dir / "results"
    compare_dir = tx_study_dir / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)

    # Load TX results
    tx_results = {}
    for f in tx_results_dir.iterdir():
        if f.suffix == ".parquet":
            tx_results[f.stem] = pd.read_parquet(str(f))

    # Load MA results
    ma_results = load_ma_results(ma_study_dir)

    if not tx_results:
        log.error("No TX results to compare")
        return
    if not ma_results:
        log.warning("No MA results found — plotting TX results only")

    # ── Summary statistics ──────────────────────────────────────────────
    summary_rows = []
    for case_name, df in tx_results.items():
        by_feeder = df.groupby("feeder")
        summary_rows.append({
            "state": "TX",
            "case": case_name,
            "peak_kW": df["total_kW"].max(),
            "mean_kW": df.groupby("hour")["total_kW"].sum().mean(),
            "v_min": df["v_min_pu"].min(),
            "v_max": df["v_max_pu"].max(),
            "max_loading_pct": df["head_loading_pct"].max(),
            "n_feeders": df["feeder"].nunique(),
        })

    for case_name, df in ma_results.items():
        summary_rows.append({
            "state": "MA",
            "case": case_name,
            "peak_kW": df["total_kW"].max(),
            "mean_kW": df.groupby("hour")["total_kW"].sum().mean(),
            "v_min": df["v_min_pu"].min(),
            "v_max": df["v_max_pu"].max(),
            "max_loading_pct": df["head_loading_pct"].max(),
            "n_feeders": df["feeder"].nunique(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(str(compare_dir / "summary_stats.csv"), index=False)
    log.info(f"Summary statistics saved to {compare_dir / 'summary_stats.csv'}")

    # ── Map TX case names to MA case names ───────────────────────────────
    # TX: elec_100_flat_results     → MA: flat_heat_100_solar_0
    # TX: elec_0_tou_cap_results    → MA: tou_cap_heat_0_solar_0
    tx_to_ma_map = {}
    for tx_case in tx_results:
        parts = tx_case.replace("_results", "").split("_")
        # parts: ['elec', '0'|'100', 'flat'|'tou'] or
        #        ['elec', '0'|'100', 'flat'|'tou', 'cap']
        elec = parts[1]
        tariff = parts[2]
        has_cap = len(parts) > 3 and parts[3] == "cap"

        # Construct MA name: {tariff}[_cap]_heat_{elec}_solar_0
        if has_cap:
            ma_name = f"{tariff}_cap_heat_{elec}_solar_0"
        else:
            ma_name = f"{tariff}_heat_{elec}_solar_0"

        if ma_name in ma_results:
            tx_to_ma_map[tx_case] = ma_name
        else:
            log.warning(f"No MA match for {tx_case} (tried {ma_name})")

    log.info(f"TX→MA case mapping: {tx_to_ma_map}")

    # ── Per-case comparison plots ─────────────────────────────────────
    for tx_case, tx_df in tx_results.items():
        ma_case = tx_to_ma_map.get(tx_case)
        ma_df = ma_results.get(ma_case) if ma_case else None

        # Zone-level total kW by hour
        tx_hourly = tx_df.groupby("hour")["total_kW"].sum().sort_index()
        tx_peak_hour = int(tx_hourly.idxmax())
        tx_peak_day = tx_peak_hour // 24
        tx_day_start = tx_peak_day * 24
        tx_day_end = tx_day_start + 24

        ma_hourly = None
        ma_peak_hour = None
        ma_peak_day = None
        ma_day_start = None
        ma_day_end = None
        if ma_df is not None:
            ma_hourly = ma_df.groupby("hour")["total_kW"].sum().sort_index()
            ma_peak_hour = int(ma_hourly.idxmax())
            ma_peak_day = ma_peak_hour // 24
            ma_day_start = ma_peak_day * 24
            ma_day_end = ma_day_start + 24

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # ── Top: full year with peak markers for both ──────────────
        ax = axes[0]
        days = tx_hourly.index / 24
        ax.plot(days, tx_hourly.values / 1000,
                label="TX", alpha=0.8, linewidth=0.5, color="C0")
        if ma_hourly is not None:
            ma_days = ma_hourly.index / 24
            ax.plot(ma_days, ma_hourly.values / 1000,
                    label="MA", alpha=0.8, linewidth=0.5, color="C1")
        ax.axvline(tx_peak_hour / 24, color="C0", linestyle="--", alpha=0.6,
                   label=f"TX peak (day {tx_peak_day})")
        if ma_peak_hour is not None:
            ax.axvline(ma_peak_hour / 24, color="C1", linestyle="--", alpha=0.6,
                       label=f"MA peak (day {ma_peak_day})")
        ax.set_xlabel("Day of year")
        ax.set_ylabel("Zone Total (MW)")
        title_suffix = f" vs {ma_case}" if ma_case else ""
        ax.set_title(f"TX {tx_case}{title_suffix}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── Bottom: both peak days ─────────────────────────────────
        # Solid = own peak day, dotted = accompanying day on the other's peak
        ax = axes[1]
        hod = range(24)

        # TX on TX peak day (solid)
        tx_on_tx = tx_hourly.loc[tx_day_start:tx_day_end - 1]
        if len(tx_on_tx) == 24:
            ax.plot(hod, tx_on_tx.values / 1000, "o-", color="C0",
                    linewidth=2, label=f"TX — TX peak (day {tx_peak_day})")

        if ma_hourly is not None and ma_day_start is not None:
            # MA on MA peak day (solid)
            ma_on_ma = ma_hourly.loc[ma_day_start:ma_day_end - 1]
            if len(ma_on_ma) == 24:
                ax.plot(hod, ma_on_ma.values / 1000, "s-", color="C1",
                        linewidth=2, label=f"MA — MA peak (day {ma_peak_day})")

            # TX on MA peak day (dotted — accompanying)
            tx_on_ma = tx_hourly.loc[ma_day_start:ma_day_end - 1]
            if len(tx_on_ma) == 24:
                ax.plot(hod, tx_on_ma.values / 1000, "o--", color="C0",
                        linewidth=1.5, alpha=0.5,
                        label=f"TX — MA peak (day {ma_peak_day})")

            # MA on TX peak day (dotted — accompanying)
            ma_on_tx = ma_hourly.loc[tx_day_start:tx_day_end - 1]
            if len(ma_on_tx) == 24:
                ax.plot(hod, ma_on_tx.values / 1000, "s--", color="C1",
                        linewidth=1.5, alpha=0.5,
                        label=f"MA — TX peak (day {tx_peak_day})")

        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Zone Total (MW)")
        ax.set_title("Peak Day Comparison")
        ax.set_xticks(range(0, 24, 2))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(str(compare_dir / f"compare_{tx_case}.png"), dpi=150)
        plt.close(fig)
        log.info(f"Plot saved: compare_{tx_case}.png")

    # ── Thermal vs non-thermal load breakdown on peak days ───────────────
    if ercot_root is not None:
        tx_matched = tx_study_dir / "feeder_summaries_matched.parquet"
        ma_matched = ma_study_dir / "feeder_summaries_matched.parquet"
        decarb_root_tx = Path(ercot_root) / "in" / "decarb_inputs"
        decarb_root_ma = ma_study_dir / "decarb"

        for tx_case in tx_results:
            ma_case = tx_to_ma_map.get(tx_case)
            if ma_case is None:
                continue

            log.info(f"Computing thermal profiles for {tx_case} / {ma_case}...")

            # TX thermal profiles
            tx_decarb_dir = decarb_root_tx / tx_case
            tx_profiles = compute_zone_thermal_profiles(
                tx_decarb_dir, tx_matched)

            # MA thermal profiles
            ma_decarb_dir = decarb_root_ma / ma_case
            ma_profiles = None
            if ma_decarb_dir.exists() and ma_matched.exists():
                ma_profiles = compute_zone_thermal_profiles(
                    ma_decarb_dir, ma_matched)

            tx_total = tx_profiles["total_kw"] / 1000  # MW
            tx_thermal = tx_profiles["thermal_kw"] / 1000

            # Find peak days from OpenDSS results (same as main plots)
            tx_hourly = tx_results[tx_case].groupby("hour")["total_kW"].sum().sort_index()
            tx_peak_day = int(tx_hourly.idxmax()) // 24
            tx_ds = tx_peak_day * 24
            tx_de = tx_ds + 24

            ma_peak_day = None
            ma_total = None
            ma_thermal = None
            if ma_profiles is not None:
                ma_total = ma_profiles["total_kw"] / 1000
                ma_thermal = ma_profiles["thermal_kw"] / 1000
                ma_hourly = ma_results[ma_case].groupby("hour")["total_kW"].sum().sort_index()
                ma_peak_day = int(ma_hourly.idxmax()) // 24
                ma_ds = ma_peak_day * 24
                ma_de = ma_ds + 24

            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            hod = range(24)

            # ── Top panel: TX peak day ─────────────────────────────
            ax = axes[0]
            # TX on TX peak (solid)
            ax.plot(hod, tx_total[tx_ds:tx_de], "-", color="C0",
                    linewidth=2, label=f"TX total (day {tx_peak_day})")
            ax.plot(hod, tx_thermal[tx_ds:tx_de], "--", color="C0",
                    linewidth=1.5, label=f"TX thermal (day {tx_peak_day})")
            # MA on TX peak (accompanying)
            if ma_total is not None:
                ax.plot(hod, ma_total[tx_ds:tx_de], "-", color="C1",
                        linewidth=2, label=f"MA total (day {tx_peak_day})")
                ax.plot(hod, ma_thermal[tx_ds:tx_de], "--", color="C1",
                        linewidth=1.5, label=f"MA thermal (day {tx_peak_day})")
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Zone Total (MW)")
            ax.set_title(f"TX Peak Day (day {tx_peak_day})")
            ax.set_xticks(range(0, 24, 2))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # ── Bottom panel: MA peak day ──────────────────────────
            ax = axes[1]
            if ma_peak_day is not None and ma_total is not None:
                # MA on MA peak (solid)
                ax.plot(hod, ma_total[ma_ds:ma_de], "-", color="C1",
                        linewidth=2, label=f"MA total (day {ma_peak_day})")
                ax.plot(hod, ma_thermal[ma_ds:ma_de], "--", color="C1",
                        linewidth=1.5, label=f"MA thermal (day {ma_peak_day})")
                # TX on MA peak (accompanying)
                ax.plot(hod, tx_total[ma_ds:ma_de], "-", color="C0",
                        linewidth=2, label=f"TX total (day {ma_peak_day})")
                ax.plot(hod, tx_thermal[ma_ds:ma_de], "--", color="C0",
                        linewidth=1.5, label=f"TX thermal (day {ma_peak_day})")
                ax.set_title(f"MA Peak Day (day {ma_peak_day})")
            else:
                ax.text(0.5, 0.5, "No MA thermal data available",
                        transform=ax.transAxes, ha="center")
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Zone Total (MW)")
            ax.set_xticks(range(0, 24, 2))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(str(compare_dir / f"thermal_{tx_case}.png"), dpi=150)
            plt.close(fig)
            log.info(f"Thermal plot saved: thermal_{tx_case}.png")
    else:
        log.warning("No ercot_root provided — skipping thermal breakdown plots")

    # ── Voltage distribution comparison ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for case_name, df in tx_results.items():
        short_name = case_name.replace("_results", "")
        ax.hist(df["v_min_pu"].dropna(), bins=50, alpha=0.5,
                label=f"TX {short_name}", density=True)
    ax.set_xlabel("V_min (pu)")
    ax.set_ylabel("Density")
    ax.set_title("TX Minimum Voltage Distribution")
    ax.legend(fontsize=7)

    ax = axes[1]
    for case_name, df in ma_results.items():
        if "solar_0" not in case_name:
            continue  # only compare solar_0 MA cases (TX has no solar)
        ax.hist(df["v_min_pu"].dropna(), bins=50, alpha=0.5,
                label=f"MA {case_name}", density=True)
    ax.set_xlabel("V_min (pu)")
    ax.set_title("MA Minimum Voltage Distribution (solar_0 only)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(str(compare_dir / "voltage_distributions.png"), dpi=150)
    plt.close(fig)
    log.info("Voltage distribution plot saved")

    log.info(f"\nAll comparison outputs in: {compare_dir}")


# ═════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TX vs MA comparison on the same OpenDSS network")
    parser.add_argument("--ercot-root", required=True,
                        help="Path to ercot_project (e.g. Z:/ercot_project or "
                             "D:/shared/ercot_project)")
    parser.add_argument("--ma-study", type=int, default=None,
                        help="MA study number to compare against (default: latest)")
    parser.add_argument("--zone", default="P100U",
                        help="OpenDSS zone name (default: P100U)")
    parser.add_argument("--workers", type=int, default=20,
                        help="OpenDSS simulation workers (default: 20)")
    parser.add_argument("--skip-matching", action="store_true",
                        help="Skip matching (reuse existing matched parquet)")
    parser.add_argument("--skip-network", action="store_true",
                        help="Skip network input generation (reuse existing)")
    parser.add_argument("--skip-opendss", action="store_true",
                        help="Skip OpenDSS simulation (reuse existing results)")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only run the comparison step")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Specific TX cases to process (default: all 8)")
    args = parser.parse_args()

    ercot_root = Path(args.ercot_root)
    if not ercot_root.exists():
        log.error(f"ercot_project not found: {ercot_root}")
        sys.exit(1)

    input_dir = PROJECT_ROOT / "in"
    output_dir = PROJECT_ROOT / "out"

    # Determine which cases to run
    cases = args.cases if args.cases else TX_CASES
    log.info(f"TX cases: {cases}")

    # Find MA study directory (required — TX output lives inside it)
    studies_root = output_dir / "studies"
    if not studies_root.exists():
        log.error(f"No studies directory: {studies_root}")
        sys.exit(1)

    if args.ma_study is not None:
        ma_study_dir = studies_root / f"study_{args.ma_study}"
    else:
        # Latest study
        study_dirs = sorted(
            [d for d in studies_root.iterdir()
             if d.is_dir() and d.name.startswith("study_")],
            key=lambda d: int(d.name.split("_")[1])
        )
        if not study_dirs:
            log.error("No studies found")
            sys.exit(1)
        ma_study_dir = study_dirs[-1]

    if not ma_study_dir.exists():
        log.error(f"MA study not found: {ma_study_dir}")
        sys.exit(1)

    log.info(f"MA study: {ma_study_dir}")

    # TX comparison output goes inside the MA study directory
    tx_study_dir = ma_study_dir / "tx_comparison"
    tx_study_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Match TX buildings to the network ───────────────────────
    matched_path = tx_study_dir / "feeder_summaries_matched.parquet"

    if args.compare_only:
        log.info("Compare-only mode — skipping matching, network, and OpenDSS")
    elif args.skip_matching and matched_path.exists():
        log.info(f"Skipping matching — using existing {matched_path}")
    else:
        log.info("Step 1: Matching TX buildings to MA network...")
        tx_df = load_tx_buildings(input_dir, ercot_root)
        feeder_df = load_ma_feeder_summaries(input_dir, args.zone)
        matched_df = match_tx_to_network(feeder_df, tx_df)
        matched_df.to_parquet(str(matched_path), index=False)
        log.info(f"Saved TX matching: {matched_path}")

    # ── Step 2: Create OpenDSS loadshapes ───────────────────────────────
    if not args.compare_only and not args.skip_network:
        log.info("Step 2: Creating OpenDSS loadshapes from TX DECARB results...")
        feeder_df = read_parquet(matched_path)
        decarb_root = ercot_root / "in" / "decarb_inputs"

        # Load TX metadata once for substitution maps
        tx_meta = read_parquet(input_dir / "TX_upgrade0.parquet")
        if "bldg_id" not in tx_meta.columns:
            tx_meta = tx_meta.reset_index().rename(columns={"index": "bldg_id"})

        all_bldg_ids = feeder_df["nrel_climate_match"].dropna().astype(int).unique().tolist()

        for case_name in cases:
            decarb_case_dir = decarb_root / case_name
            if not decarb_case_dir.exists():
                log.warning(f"DECARB case not found: {decarb_case_dir}")
                continue

            # Build substitution map for failed buildings
            sub_map = build_substitution_map(decarb_case_dir, all_bldg_ids, tx_meta)

            opendss_case_dir = tx_study_dir / "opendss" / case_name
            log.info(f"  Generating: {case_name}")
            generate_opendss_case(decarb_case_dir, feeder_df,
                                  args.zone, opendss_case_dir,
                                  sub_map=sub_map)

    # ── Step 3: Run OpenDSS ─────────────────────────────────────────────
    if not args.compare_only and not args.skip_opendss:
        log.info("Step 3: Running OpenDSS simulations...")
        run_opendss_cases(tx_study_dir, cases, workers=args.workers)

    # ── Step 4: Compare results ─────────────────────────────────────────
    log.info("Step 4: Comparing TX vs MA results...")
    compare_results(tx_study_dir, ma_study_dir, ercot_root=ercot_root)

    log.info("Done.")


if __name__ == "__main__":
    main()
