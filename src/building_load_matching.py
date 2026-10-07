import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from config.loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

COUNTY_TAG = "in.county"
CLIMATE_TAG = "in.building_america_climate_zone"
SUMMER_DEMAND_TAG = "out.qoi.electricity.maximum_daily_peak_summer..kw"
WINTER_DEMAND_TAG = "out.qoi.electricity.maximum_daily_peak_winter..kw"
PV_TAG = "in.has_pv"

F_STAT_THRESHOLD = 30
COEFF_OUTLIER_FACTOR = 10      # exclude if any k is >10× or <1/10× the median
DEFAULT_USAGE_LIMIT = 1        # each building matched at most once (unique matching)


def read_parquet(path):
    try:
        return pd.read_parquet(str(path), engine="fastparquet")
    except Exception:
        return pd.read_parquet(str(path), engine="pyarrow")


def load_eligible_buildings(cfg):
    input_dir = cfg["paths"]["input_dir"]
    output_dir = cfg["paths"]["output_dir"]
    state = cfg["state"]

    metadata_path = input_dir / f"{state}_upgrade0.parquet"
    nrel_df = read_parquet(metadata_path)

    if "bldg_id" not in nrel_df.columns:
        nrel_df = nrel_df.reset_index().rename(columns={"index": "bldg_id"})

    total_buildings = len(nrel_df)
    log.info(f"Loaded {total_buildings} buildings from metadata")

    thermal_results_dir = output_dir / "thermal_model_outputs" / state / "baseline_internal_gains"
    thermal_csv = thermal_results_dir / "regression_coeff_baseline_internal_gains.csv"

    if not thermal_csv.exists():
        log.warning(f"No thermal model results found at {thermal_csv}, skipping f_stat filter")
        return nrel_df

    thermal_df = pd.read_csv(str(thermal_csv))
    thermal_df["bldg_id"] = thermal_df["bldg_id"].astype(int)

    # --- F-stat filter ---
    fstat_pass = thermal_df[thermal_df["f_stat"] >= F_STAT_THRESHOLD].copy()
    fstat_fail = len(thermal_df) - len(fstat_pass)

    log.info(f"Thermal model results: {len(thermal_df)} buildings fitted")
    log.info(f"  f_stat >= {F_STAT_THRESHOLD}: {len(fstat_pass)} pass")
    log.info(f"  f_stat < {F_STAT_THRESHOLD}: {fstat_fail} excluded")

    # --- Coefficient outlier filter ---
    # Exclude buildings where any coefficient is >10× or <1/10× the median
    # (catches zero coefficients, sign flips, and wildly wrong fits)
    coeff_cols = ["k1", "k2", "k3"]
    outlier_mask = pd.Series(False, index=fstat_pass.index)
    for col in coeff_cols:
        median_val = fstat_pass[col].median()
        if median_val == 0:
            continue
        abs_median = abs(median_val)
        lo = abs_median / COEFF_OUTLIER_FACTOR
        hi = abs_median * COEFF_OUTLIER_FACTOR
        abs_vals = fstat_pass[col].abs()
        outlier_mask |= (abs_vals < lo) | (abs_vals > hi)

    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        outlier_ids = fstat_pass.loc[outlier_mask, "bldg_id"].tolist()
        log.info(f"  Coefficient outlier filter: {n_outliers} excluded "
                 f"(>{COEFF_OUTLIER_FACTOR}× or <1/{COEFF_OUTLIER_FACTOR}× median)")
        log.debug(f"  Outlier building IDs: {outlier_ids}")

    eligible_ids = fstat_pass.loc[~outlier_mask, "bldg_id"]

    eligible_df = nrel_df[nrel_df["bldg_id"].isin(eligible_ids)].copy()
    log.info(f"Eligible buildings after filter: {len(eligible_df)} / {total_buildings}")

    # In test mode, limit the building pool (not the buses)
    if cfg.get("test_mode") and cfg.get("test_limit"):
        limit = cfg["test_limit"]
        if len(eligible_df) > limit:
            eligible_df = eligible_df.head(limit).copy()
            log.info(f"Test mode: building pool limited to {limit}")

    return eligible_df


def load_feeder_summaries(cfg):
    input_dir = cfg["paths"]["input_dir"]
    feeder_path = input_dir / "feeder_summaries.parquet"

    if not feeder_path.exists():
        raise FileNotFoundError(f"Missing feeder summaries: {feeder_path}")

    feeder_df = read_parquet(feeder_path)

    # Filter to configured zone (all buses in that zone)
    zone_name = cfg.get("zone", {}).get("name")
    if zone_name and "zone" in feeder_df.columns:
        feeder_df = feeder_df[feeder_df["zone"] == zone_name].copy()
        log.info(f"Loaded {len(feeder_df)} buses in zone {zone_name}")
    else:
        log.info(f"Loaded {len(feeder_df)} feeder buses (all zones)")

    return feeder_df


def build_kdtrees(valid_df, climate_tag, summer_tag, winter_tag):
    trees = {}
    for zone in valid_df[climate_tag].dropna().unique():
        for summer_dom in [True, False]:
            subset = valid_df[
                (valid_df[climate_tag] == zone)
                & ((valid_df[summer_tag] >= valid_df[winter_tag]) == summer_dom)
            ]
            if not subset.empty:
                vectors = subset[[summer_tag, winter_tag]].to_numpy(dtype=float)
                trees[(zone, summer_dom)] = (subset.copy(), cKDTree(vectors))
    return trees


def match_county(feeder_df, nrel_df, county_tag, summer_tag, winter_tag):
    feeder_df = feeder_df.copy()
    nrel_df = nrel_df.copy()
    nrel_df["max_demand_kw"] = nrel_df[[summer_tag, winter_tag]].max(axis=1)
    county_groups = {c: g for c, g in nrel_df.groupby(county_tag)}

    results = []
    for county_code, kw in zip(feeder_df["county_code"], feeder_df["kw"]):
        if pd.isna(county_code) or pd.isna(kw):
            results.append((None, None, None))
            continue

        county_subset = county_groups.get(county_code)
        if county_subset is None or county_subset.empty:
            results.append((None, None, None))
            continue

        diffs = np.abs(county_subset["max_demand_kw"].values - kw)
        best_idx = diffs.argmin()
        best = county_subset.iloc[best_idx]
        results.append((best["bldg_id"], best[summer_tag], best[winter_tag]))

    results = np.array(results, dtype=object)
    feeder_df["nrel_county_match"] = results[:, 0]
    feeder_df["nrel_county_summer_kw"] = results[:, 1]
    feeder_df["nrel_county_winter_kw"] = results[:, 2]

    matched = feeder_df["nrel_county_match"].notna().sum()
    log.info(f"County match: {matched}/{len(feeder_df)} buses matched")

    return feeder_df


K_NEIGHBORS = 5               # number of nearest neighbors to consider
RNG_SEED = 42                 # reproducibility


def _pick_from_k_nearest(subset, tree, target, k=K_NEIGHBORS):
    """Query k nearest neighbors, return one weighted by inverse distance.

    Closer buildings are more likely to be picked, but not guaranteed —
    this spreads matches across similar buildings for diversity.
    """
    n_available = len(subset)
    k_actual = min(k, n_available)
    dists, idxs = tree.query(target, k=k_actual)

    # tree.query returns scalars when k=1
    if k_actual == 1:
        dists = np.array([dists])
        idxs = np.array([idxs])
    else:
        dists = dists[0]
        idxs = idxs[0]

    # Inverse-distance weights (add small epsilon to avoid div-by-zero)
    weights = 1.0 / (dists + 1e-6)
    weights /= weights.sum()

    chosen_idx = _match_rng.choice(len(idxs), p=weights)
    return subset.iloc[idxs[chosen_idx]]


def _build_global_tree(valid_df, summer_tag, winter_tag):
    """Build a single KD-tree over all buildings (no climate zone split)."""
    if valid_df.empty:
        return None, None
    vectors = valid_df[[summer_tag, winter_tag]].to_numpy(dtype=float)
    return valid_df.copy(), cKDTree(vectors)


def match_climate(feeder_df, nrel_df, valid_df, usage_limit,
                   county_tag, climate_tag, summer_tag, winter_tag):
    """Match feeder buses to NREL buildings with diversity.

    Three-tier fallback:
      1. Climate zone match — same climate zone, k-nearest with random pick
      2. All-zone match — search all climate zones, k-nearest with random pick
      3. Power-only match — global KD-tree on demand, ignore geography

    Buildings are removed from the pool after `usage_limit` matches to
    ensure diversity across the building stock.
    """
    global _match_rng
    _match_rng = np.random.default_rng(RNG_SEED)

    feeder_df = feeder_df.copy()
    valid_df = valid_df.copy()
    valid_df["num_matches"] = 0

    # Tier 1 & 2: climate-zone KD-trees
    zone_trees = build_kdtrees(valid_df, climate_tag, summer_tag, winter_tag)

    # Tier 3: global KD-tree (all buildings, no zone split)
    global_subset, global_tree = _build_global_tree(valid_df, summer_tag, winter_tag)

    # Map bus county codes → climate zones (for same-state matching)
    county_to_climate = (
        nrel_df[[county_tag, climate_tag]]
        .dropna()
        .drop_duplicates()
        .set_index(county_tag)[climate_tag]
        .to_dict()
    )

    # Also map feeder climate zones directly if available
    has_feeder_climate = "climate_zone" in feeder_df.columns

    climate_match_ids = []
    climate_summer = []
    climate_winter = []
    match_tier = []       # track which tier matched each bus

    n_tier1 = n_tier2 = n_tier3 = n_reset = 0

    for i, row in feeder_df.iterrows():
        bus_kw = row["kw"]
        county_code = row.get("county_code")
        county_match = row.get("nrel_county_match")

        if pd.isna(bus_kw):
            climate_match_ids.append(None)
            climate_summer.append(None)
            climate_winter.append(None)
            match_tier.append(None)
            continue

        # Determine target demand vector
        if pd.notna(county_match):
            target_summer = row["nrel_county_summer_kw"]
            target_winter = row["nrel_county_winter_kw"]
        else:
            target_summer = bus_kw
            target_winter = bus_kw

        target = np.array([[target_summer, target_winter]])
        summer_dominant = target_summer >= target_winter

        # --- Tier 1: Same climate zone ---
        climate_zone = county_to_climate.get(county_code)
        # If bus county doesn't map, try the feeder's own climate_zone column
        if climate_zone is None and has_feeder_climate:
            climate_zone = row.get("climate_zone")

        best = None
        tier = None

        # Try same zone + same dominance, then opposite dominance
        for dom in [summer_dominant, not summer_dominant]:
            candidates = zone_trees.get((climate_zone, dom))
            if candidates is not None and not candidates[0].empty:
                best = _pick_from_k_nearest(candidates[0], candidates[1], target)
                tier = 1
                break

        # --- Tier 2: Any climate zone ---
        if best is None:
            best_dist = float("inf")
            best_zone_result = None
            for key, (subset, tree) in zone_trees.items():
                if subset.empty:
                    continue
                k_actual = min(K_NEIGHBORS, len(subset))
                dists, _ = tree.query(target, k=k_actual)
                min_d = np.min(dists) if k_actual > 0 else float("inf")
                if min_d < best_dist:
                    best_dist = min_d
                    best_zone_result = (subset, tree)

            if best_zone_result is not None:
                best = _pick_from_k_nearest(best_zone_result[0], best_zone_result[1], target)
                tier = 2

        # --- Tier 3: Global power-only match ---
        if best is None and global_tree is not None:
            best = _pick_from_k_nearest(global_subset, global_tree, target)
            tier = 3

        # --- Record or reset ---
        if best is None:
            # Exhausted all pools — reset and try again
            n_reset += 1
            valid_df = nrel_df.copy()
            valid_df["num_matches"] = 0
            zone_trees = build_kdtrees(valid_df, climate_tag, summer_tag, winter_tag)
            global_subset, global_tree = _build_global_tree(valid_df, summer_tag, winter_tag)
            log.warning(f"Pool exhausted at bus {i}, reset #{n_reset}")

            # Retry with fresh pool
            if global_tree is not None:
                best = _pick_from_k_nearest(global_subset, global_tree, target)
                tier = 3

        if best is None:
            climate_match_ids.append(None)
            climate_summer.append(None)
            climate_winter.append(None)
            match_tier.append(None)
            continue

        bldg_id = best["bldg_id"]

        # Track usage and enforce diversity limit
        valid_df.loc[valid_df["bldg_id"] == bldg_id, "num_matches"] += 1
        match_count = valid_df.loc[
            valid_df["bldg_id"] == bldg_id, "num_matches"
        ].iloc[0]

        if match_count >= usage_limit:
            valid_df = valid_df[valid_df["bldg_id"] != bldg_id]
            zone_trees = build_kdtrees(valid_df, climate_tag, summer_tag, winter_tag)
            global_subset, global_tree = _build_global_tree(valid_df, summer_tag, winter_tag)

        climate_match_ids.append(bldg_id)
        climate_summer.append(best[summer_tag])
        climate_winter.append(best[winter_tag])
        match_tier.append(tier)

        if tier == 1:
            n_tier1 += 1
        elif tier == 2:
            n_tier2 += 1
        elif tier == 3:
            n_tier3 += 1

    feeder_df["nrel_climate_match"] = climate_match_ids
    feeder_df["nrel_climate_summer_kw"] = climate_summer
    feeder_df["nrel_climate_winter_kw"] = climate_winter
    feeder_df["match_tier"] = match_tier

    matched = feeder_df["nrel_climate_match"].notna().sum()
    unique = feeder_df["nrel_climate_match"].nunique()
    log.info(f"Climate match: {matched}/{len(feeder_df)} buses matched, "
             f"{unique} unique buildings used")
    log.info(f"  Tier 1 (climate zone): {n_tier1} | "
             f"Tier 2 (cross-zone): {n_tier2} | "
             f"Tier 3 (power-only): {n_tier3} | "
             f"Pool resets: {n_reset}")

    return feeder_df, valid_df


def load_pv_load_ids(cfg):
    pv_csv = cfg.get("matching", {}).get("pv_profiles_csv")
    if pv_csv is None:
        pv_csv = cfg["paths"]["input_dir"] / "potential_pv_rooftops.csv"
    else:
        pv_csv = Path(pv_csv)

    if not pv_csv.exists():
        log.warning(f"No rooftop PV profile list at {pv_csv}, treating all buses as non-PV")
        return set()

    pv_df = pd.read_csv(str(pv_csv))
    log.info(f"Loaded {len(pv_df)} rooftop PV candidate loads from {pv_csv}")
    return set(pv_df["load_id"])


def tag_rooftop_pv(feeder_df, pv_load_ids):
    feeder_df = feeder_df.copy()
    load_id = feeder_df["yearly"].str.extract(r"^res_kw_(.+)_pu$")[0]
    feeder_df["has_rooftop_pv"] = load_id.isin(pv_load_ids)
    n_pv = feeder_df["has_rooftop_pv"].sum()
    log.info(f"Tagged {n_pv}/{len(feeder_df)} buses as rooftop PV")
    return feeder_df


def match_pv_statewide(pv_feeder_df, pv_eligible_df, summer_tag, winter_tag):
    pv_eligible_df = pv_eligible_df.copy()
    pv_eligible_df["max_demand_kw"] = pv_eligible_df[[summer_tag, winter_tag]].max(axis=1)

    results = []
    for kw in pv_feeder_df["kw"]:
        if pd.isna(kw):
            results.append((None, None, None))
            continue
        diffs = np.abs(pv_eligible_df["max_demand_kw"].values - kw)
        best = pv_eligible_df.iloc[diffs.argmin()]
        results.append((best["bldg_id"], best[summer_tag], best[winter_tag]))

    results = np.array(results, dtype=object)
    pv_feeder_df = pv_feeder_df.copy()
    pv_feeder_df["nrel_county_match"] = results[:, 0]
    pv_feeder_df["nrel_county_summer_kw"] = results[:, 1]
    pv_feeder_df["nrel_county_winter_kw"] = results[:, 2]
    pv_feeder_df["nrel_climate_match"] = results[:, 0]
    pv_feeder_df["nrel_climate_summer_kw"] = results[:, 1]
    pv_feeder_df["nrel_climate_winter_kw"] = results[:, 2]

    matched = pv_feeder_df["nrel_county_match"].notna().sum()
    log.info(f"PV statewide match: {matched}/{len(pv_feeder_df)} PV buses matched")
    return pv_feeder_df


def run_matching(cfg):
    input_dir = cfg["paths"]["input_dir"]
    pv_rooftop = cfg.get("matching", {}).get("pv_rooftop", False)

    eligible_df = load_eligible_buildings(cfg)
    feeder_df = load_feeder_summaries(cfg)

    # Compute usage_limit: "auto" = ceil(n_buses / n_buildings) + 1 headroom
    usage_limit_cfg = cfg.get("matching", {}).get("usage_limit", DEFAULT_USAGE_LIMIT)
    if usage_limit_cfg == "auto":
        import math
        n_buses = len(feeder_df)
        n_buildings = len(eligible_df)
        usage_limit = math.ceil(15 * n_buses / n_buildings)
        log.info(f"Match diversity: usage_limit=auto → {usage_limit} "
                 f"(ceil(15 × {n_buses}/{n_buildings}))")
    else:
        usage_limit = int(usage_limit_cfg)
        log.info(f"Match diversity: usage_limit={usage_limit}")

    if "kw" not in feeder_df.columns:
        log.error("Feeder summaries missing 'kw' column — cannot match by demand")
        return None

    if pv_rooftop:
        pv_load_ids = load_pv_load_ids(cfg)
        feeder_df = tag_rooftop_pv(feeder_df, pv_load_ids)

        pv_mask = feeder_df["has_rooftop_pv"]
        pv_subset = feeder_df[pv_mask]
        non_pv_subset = feeder_df[~pv_mask]

        pv_eligible = eligible_df[eligible_df[PV_TAG] == "Yes"]
        non_pv_eligible = eligible_df[eligible_df[PV_TAG] != "Yes"]

        log.info(
            f"PV split: {len(pv_subset)} PV buses -> {len(pv_eligible)} PV buildings | "
            f"{len(non_pv_subset)} non-PV buses -> {len(non_pv_eligible)} non-PV buildings"
        )

        matched_pv = match_pv_statewide(
            pv_subset, pv_eligible, SUMMER_DEMAND_TAG, WINTER_DEMAND_TAG,
        )

        non_pv_subset = match_county(
            non_pv_subset, non_pv_eligible,
            COUNTY_TAG, SUMMER_DEMAND_TAG, WINTER_DEMAND_TAG,
        )
        valid_df = non_pv_eligible.copy()
        if "bldg_id" not in valid_df.columns:
            valid_df = valid_df.reset_index().rename(columns={"index": "bldg_id"})

        non_pv_subset, valid_df = match_climate(
            non_pv_subset, non_pv_eligible, valid_df, usage_limit,
            COUNTY_TAG, CLIMATE_TAG, SUMMER_DEMAND_TAG, WINTER_DEMAND_TAG,
        )

        feeder_df = pd.concat([matched_pv, non_pv_subset]).sort_index()
    else:
        feeder_df["has_rooftop_pv"] = False

        feeder_df = match_county(
            feeder_df, eligible_df,
            COUNTY_TAG, SUMMER_DEMAND_TAG, WINTER_DEMAND_TAG,
        )

        valid_df = eligible_df.copy()
        if "bldg_id" not in valid_df.columns:
            valid_df = valid_df.reset_index().rename(columns={"index": "bldg_id"})

        feeder_df, valid_df = match_climate(
            feeder_df, eligible_df, valid_df, usage_limit,
            COUNTY_TAG, CLIMATE_TAG, SUMMER_DEMAND_TAG, WINTER_DEMAND_TAG,
        )

    output_path = input_dir / "feeder_summaries_matched.parquet"
    feeder_df.to_parquet(str(output_path), index=False)
    log.info(f"Saved matched feeder summaries to {output_path}")

    summary_path = input_dir / "matching_summary.csv"
    agg_dict = {
        "total_buses": ("Name", "count"),
        "pv_buses": ("has_rooftop_pv", "sum"),
        "county_matched": ("nrel_county_match", lambda x: x.notna().sum()),
        "climate_matched": ("nrel_climate_match", lambda x: x.notna().sum()),
        "unique_bldgs_matched": ("nrel_climate_match", "nunique"),
    }
    summary = feeder_df.groupby("zone").agg(**agg_dict).reset_index()
    summary.to_csv(str(summary_path), index=False)
    log.info(f"Matching summary saved to {summary_path}")

    return feeder_df


if __name__ == "__main__":
    cfg = load_config()
    run_matching(cfg)
