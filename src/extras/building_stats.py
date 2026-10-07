"""
Building stock statistics — metadata histograms for the full state
housing stock and the model subset used in a given study.

Generates:
    <study_dir>/building_stats/
        all_ma_<col>.png          — histograms for all MA buildings
        all_ma_histogram_bins.csv
        all_ma_histogram_bins.json
        model_<col>.png           — histograms for model subset
        model_histogram_bins.csv
        model_histogram_bins.json
        model_lookup_stats.csv    — count distribution stats
        model_lookup_stats.json
        model_bucketed_histogram.png
        model_cdf.png
        model_top_ids.csv
        model_bottom_ids.csv

Usage:
    python building_stats.py <study_num>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_project_root = Path(__file__).parents[1]
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(Path(__file__).parent))
try:
    from config.loader import load_config
except ImportError:
    _project_root = Path(__file__).parents[2]
    sys.path.insert(0, str(_project_root))
    sys.path.insert(0, str(_project_root / "src"))
    from config.loader import load_config
from create_network_inputs import find_study_dir, read_parquet

# ── Column config ─────────────────────────────────────────────────────────────

COLUMNS = [
    "in.representative_income",
    "in.area_median_income",
    "in.ashrae_iecc_climate_zone_2004",
    "in.geometry_building_type_acs",
    "in.geometry_floor_area",
    "in.geometry_stories",
    "in.heating_fuel",
    "in.hvac_heating_type_and_fuel",
    "in.has_pv",
    "in.electric_vehicle_ownership",
]

NUMERIC_COLUMNS = {
    "in.representative_income",
}

INTEGER_LABEL_COLUMNS = {
    "in.representative_income",
}

CUSTOM_BIN_COLUMNS = {
    "in.geometry_stories": [
        ("1",   1, 2),
        ("2",   2, 3),
        ("3",   3, 4),
        ("4-8", 4, 8),
        ("8+",  8, float("inf")),
    ],
}

NUM_BINS = 5


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✅ Saved: {path}")


def load_parquet_metadata(path, columns):
    """Load parquet with bldg_id as index, selecting only available columns."""
    pf_cols = pd.read_parquet(str(path), columns=["bldg_id"]).columns.tolist()
    # Read just to check available columns
    sample = pd.read_parquet(str(path), columns=["bldg_id"])
    all_cols = pd.read_parquet(str(path)).columns.tolist()

    valid = [c for c in columns if c in all_cols]
    missing = [c for c in columns if c not in all_cols]
    if missing:
        print(f"  [WARN] Columns not found, skipped:\n    " + "\n    ".join(missing))
    print(f"  Loading {len(valid)} matched columns: {valid}")
    df = pd.read_parquet(str(path), columns=["bldg_id"] + valid)
    df = df.set_index("bldg_id")
    return df


# ── Building map from feeder matching ────────────────────────────────────────

def get_building_counts(feeder_path):
    """Parse feeder_summaries_matched.parquet to get {bldg_id: instance_count}.

    Returns (all_bldg_ids: set, aggregated_counts: dict).
    """
    fm = read_parquet(feeder_path)

    bldg_col = None
    for col in ("nrel_climate_match", "nrel_county_match"):
        if col in fm.columns:
            bldg_col = col
            break
    if bldg_col is None:
        raise ValueError("No nrel_climate_match or nrel_county_match column")

    matched = fm[fm[bldg_col].notna()].copy()
    matched["bldg_id"] = matched[bldg_col].astype(int)

    counts = matched.groupby("bldg_id").size().to_dict()
    all_ids = set(counts.keys())
    print(f"  {len(all_ids):,} unique building IDs, "
          f"{sum(counts.values()):,} total bus instances")
    return all_ids, counts


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_statistics(aggregated_counts, top_n=10, percentiles=(50, 75, 90, 95, 99)):
    arr = np.array(list(aggregated_counts.values()))
    sorted_items = sorted(aggregated_counts.items(), key=lambda x: x[1], reverse=True)
    total = int(arr.sum())
    return {
        "total_buildings": total,
        "unique_ids": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "percentiles": {str(p): float(np.percentile(arr, p)) for p in percentiles},
        "top_n": sorted_items[:top_n],
        "bottom_n": sorted_items[-top_n:],
        "top_n_pct": [(id_, count, round(100 * count / total, 4))
                      for id_, count in sorted_items[:top_n]],
        "bottom_n_pct": [(id_, count, round(100 * count / total, 4))
                         for id_, count in sorted_items[-top_n:]],
    }


def print_statistics(stats, top_n=10):
    total = stats["total_buildings"]
    print("  ===== Building Count Report =====")
    print(f"  Total bus instances : {total:,}")
    print(f"  Unique IDs          : {stats['unique_ids']:,}")
    print(f"  Mean count/bldg     : {stats['mean']:.2f}")
    print(f"  Median              : {stats['median']:.2f}")
    print(f"  Std dev             : {stats['std']:.2f}")
    print(f"  Min                 : {stats['min']}")
    print(f"  Max                 : {stats['max']}")
    print("\n  Percentiles:")
    for p, v in stats["percentiles"].items():
        print(f"    p{p:<3}: {v:.1f}")
    print(f"\n  Top {top_n} IDs by count:")
    for id_, count, pct in stats["top_n_pct"]:
        print(f"    {id_:<20} {count:>8,}  ({pct:.2f}%)")
    print(f"\n  Bottom {top_n} IDs by count:")
    for id_, count, pct in stats["bottom_n_pct"]:
        print(f"    {id_:<20} {count:>8,}  ({pct:.2f}%)")
    print("  =================================")


# ── Bucketing ────────────────────────────────────────────────────────────────

def bucketize_counts(aggregated_counts, bucket_size=5, max_value=50):
    bucket_counts = {
        f"{s}-{s + bucket_size}": 0
        for s in range(0, max_value, bucket_size)
    }
    bucket_counts[f"{max_value}+"] = 0

    for total in aggregated_counts.values():
        if total > max_value:
            bucket_counts[f"{max_value}+"] += 1
            continue
        bucket_start = (total // bucket_size) * bucket_size
        if total % bucket_size == 0 and total != 0:
            bucket_start -= bucket_size
        bucket_counts[f"{bucket_start}-{bucket_start + bucket_size}"] += 1

    return bucket_counts


# ── Histogram builders ────────────────────────────────────────────────────────

def _add_percentages(bins, total=None):
    if total is None:
        total = sum(b["count"] for b in bins)
    for b in bins:
        b["pct"] = round(100 * b["count"] / total, 4) if total > 0 else 0.0
    return bins


def histogram_numeric(series, n_bins=NUM_BINS, integer_labels=False):
    clean = series.dropna()
    counts, edges = np.histogram(clean, bins=n_bins)
    bins = [
        {
            "bin_start": float(edges[i]),
            "bin_end": float(edges[i + 1]),
            "label": (f"{int(edges[i])}–{int(edges[i + 1])}" if integer_labels
                      else f"{edges[i]:.4g}–{edges[i + 1]:.4g}"),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]
    return {"type": "numeric", "n_bins": n_bins, "bins": _add_percentages(bins)}


def histogram_custom_bins(series, bin_defs):
    clean = pd.to_numeric(series, errors="coerce").dropna()
    bins = []
    for label, low, high in bin_defs:
        if high == float("inf"):
            count = int((clean >= low).sum())
        else:
            count = int(((clean >= low) & (clean <= high)).sum())
        bins.append({
            "bin_start": float(low),
            "bin_end": None if high == float("inf") else float(high),
            "label": label,
            "count": count,
        })
    return {"type": "custom_numeric", "n_bins": len(bins), "bins": _add_percentages(bins)}


def histogram_categorical(series):
    counts = series.dropna().astype(str).value_counts()
    bins = [{"label": label, "count": int(count)} for label, count in counts.items()]
    return {"type": "categorical", "n_bins": len(bins), "bins": _add_percentages(bins)}


def build_histograms(df, n_bins=NUM_BINS):
    histograms = {}
    for col in df.columns:
        series = df[col]
        if col in CUSTOM_BIN_COLUMNS:
            histograms[col] = histogram_custom_bins(series, CUSTOM_BIN_COLUMNS[col])
        elif col in NUMERIC_COLUMNS:
            histograms[col] = histogram_numeric(series, n_bins,
                                                 integer_labels=col in INTEGER_LABEL_COLUMNS)
        else:
            histograms[col] = histogram_categorical(series)
        print(f"    ✓ {col}  [{histograms[col]['type']}, {histograms[col]['n_bins']} bins]")
    return histograms


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_bucketed_histogram(bucket_counts, out_dir, prefix=""):
    labels, values = zip(*bucket_counts.items())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, values, edgecolor="black")
    ax.set_xlabel("Total Bus Count per Building (Buckets)")
    ax.set_ylabel("Number of Buildings")
    ax.set_title("Distribution of Building Instance Counts")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}bucketed_histogram.png", dpi=150)
    plt.close(fig)


def plot_cdf(aggregated_counts, out_dir, prefix=""):
    values = np.sort(list(aggregated_counts.values()))
    cdf = np.arange(1, len(values) + 1) / len(values)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(values, cdf, linewidth=1.5)
    ax.set_xlabel("Total Bus Count per Building")
    ax.set_ylabel("Cumulative Fraction of Buildings")
    ax.set_title("CDF of Building Instance Counts")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}cdf.png", dpi=150)
    plt.close(fig)


def plot_metadata_histograms(histograms, out_dir, prefix="", max_cat_labels=30):
    for col, info in histograms.items():
        bins = info["bins"]
        counts = [b["count"] for b in bins]
        pcts = [b["pct"] for b in bins]
        labels = [b["label"] for b in bins]
        n = len(bins)

        if info["type"] == "numeric":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
            ax1.bar(range(n), counts, edgecolor="black")
            ax1.set_xticks(range(n))
            ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax1.set_ylabel("Count")
            ax1.set_title(f"{col} — Count")

            ax2.bar(range(n), pcts, edgecolor="black", color="steelblue")
            ax2.set_xticks(range(n))
            ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax2.set_ylabel("Percentage (%)")
            ax2.set_title(f"{col} — %")
        else:
            if n > max_cat_labels:
                bins = bins[:max_cat_labels]
                labels = [b["label"] for b in bins]
                counts = [b["count"] for b in bins]
                pcts = [b["pct"] for b in bins]
                title_suffix = f" (top {max_cat_labels} of {n})"
            else:
                title_suffix = ""

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(labels) * 0.3)))
            ax1.barh(labels[::-1], counts[::-1], edgecolor="black")
            ax1.set_xlabel("Count")
            ax1.set_title(col + title_suffix + " — Count")

            ax2.barh(labels[::-1], pcts[::-1], edgecolor="black", color="steelblue")
            ax2.set_xlabel("Percentage (%)")
            ax2.set_title(col + title_suffix + " — %")

        fig.tight_layout()
        safe_col = col.replace(".", "_").replace("/", "_")
        fig.savefig(out_dir / f"{prefix}{safe_col}.png", dpi=150)
        plt.close(fig)


def plot_adoption_comparison(df_all, df_subset, out_dir, state):
    """Side-by-side bar chart: electric heating, rooftop PV, EV ownership
    for model subset (left) vs full state (right), shown as percentages."""

    def compute_rates(df):
        n = len(df)
        rates = {}
        # Electric heating
        if "in.heating_fuel" in df.columns:
            rates["Electric\nHeating"] = (
                df["in.heating_fuel"].astype(str).str.strip() == "Electricity"
            ).sum() / n * 100
        # Rooftop PV
        if "in.has_pv" in df.columns:
            rates["Rooftop\nPV"] = (
                df["in.has_pv"].astype(str).str.strip() == "Yes"
            ).sum() / n * 100
        # EV ownership
        if "in.electric_vehicle_ownership" in df.columns:
            rates["EV\nOwnership"] = (
                df["in.electric_vehicle_ownership"].astype(str).str.strip() == "Yes"
            ).sum() / n * 100
        return rates

    rates_model = compute_rates(df_subset)
    rates_all = compute_rates(df_all)

    categories = [k for k in rates_model if k in rates_all]
    if not categories:
        print("  [WARN] No adoption columns found, skipping comparison plot")
        return

    model_vals = [rates_model[k] for k in categories]
    all_vals = [rates_all[k] for k in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, model_vals, width, label="Model Subset",
                   color="#4C72B0", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, all_vals, width, label=f"All {state.upper()}",
                   color="#DD8452", edgecolor="black", linewidth=0.5)

    # Add percentage labels on bars
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Percentage of Buildings (%)")
    ax.set_title("Adoption Rates: Model Subset vs Full State")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend()
    ax.set_ylim(0, max(model_vals + all_vals) * 1.25)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "adoption_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: adoption_comparison.png")


# ── Analysis runners ──────────────────────────────────────────────────────────

def run_lookup_analysis(aggregated_counts, out_dir, prefix=""):
    """Bucketed histogram + CDF + top-N from feeder matching counts."""
    print("\n── Lookup analysis ──────────────────────────────────────")
    stats = compute_statistics(aggregated_counts)
    print_statistics(stats)
    save_json(stats, out_dir / f"{prefix}lookup_stats.json")

    csv_rows = {
        "total_bus_instances": stats["total_buildings"],
        "unique_buildings": stats["unique_ids"],
        "mean_count": stats["mean"],
        "median_count": stats["median"],
        "std": stats["std"],
        "min_count": stats["min"],
        "max_count": stats["max"],
        **{f"p{p}": v for p, v in stats["percentiles"].items()},
    }
    pd.DataFrame([csv_rows]).to_csv(out_dir / f"{prefix}lookup_stats.csv", index=False)
    print(f"  ✅ Saved: {out_dir / f'{prefix}lookup_stats.csv'}")

    top_rows = [{"rank": i+1, "bldg_id": id_, "count": count, "pct": pct}
                for i, (id_, count, pct) in enumerate(stats["top_n_pct"])]
    bottom_rows = [{"rank": i+1, "bldg_id": id_, "count": count, "pct": pct}
                   for i, (id_, count, pct) in enumerate(stats["bottom_n_pct"])]
    pd.DataFrame(top_rows).to_csv(out_dir / f"{prefix}top_ids.csv", index=False)
    pd.DataFrame(bottom_rows).to_csv(out_dir / f"{prefix}bottom_ids.csv", index=False)
    print(f"  ✅ Saved: {prefix}top_ids.csv, {prefix}bottom_ids.csv")

    bucket_counts = bucketize_counts(aggregated_counts)
    save_json(bucket_counts, out_dir / f"{prefix}bucket_counts.json")
    plot_bucketed_histogram(bucket_counts, out_dir, prefix=prefix)
    plot_cdf(aggregated_counts, out_dir, prefix=prefix)


def run_metadata_analysis(df, out_dir, prefix=""):
    """Metadata histogram plots + JSON + CSV for a given dataframe slice."""
    print("\n── Metadata histogram analysis ──────────────────────────")
    print(f"  Rows in slice: {len(df):,}")
    histograms = build_histograms(df)

    save_json(histograms, out_dir / f"{prefix}histogram_bins.json")

    csv_rows = []
    for col, info in histograms.items():
        for bin_ in info["bins"]:
            csv_rows.append({
                "column": col,
                "type": info["type"],
                "label": bin_["label"],
                "bin_start": bin_.get("bin_start", ""),
                "bin_end": bin_.get("bin_end", ""),
                "count": bin_["count"],
                "pct": bin_["pct"],
            })
    pd.DataFrame(csv_rows).to_csv(out_dir / f"{prefix}histogram_bins.csv", index=False)
    print(f"  ✅ Saved: {out_dir / f'{prefix}histogram_bins.csv'}")

    plot_metadata_histograms(histograms, out_dir, prefix=prefix)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python building_stats.py <study_num>")
        sys.exit(1)

    study_num = int(sys.argv[1])
    cfg = load_config()
    output_dir = Path(cfg["paths"]["output_dir"])
    input_dir = Path(cfg["paths"]["input_dir"])
    state = cfg["state"]

    study_dir = find_study_dir(output_dir, study_num)
    print(f"Study: {study_dir.name}")

    out_dir = study_dir / "building_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load feeder matching → model subset IDs and instance counts
    feeder_path = study_dir / "feeder_summaries_matched.parquet"
    print(f"\nLoading feeder matching: {feeder_path}")
    all_bldg_ids, aggregated_counts = get_building_counts(feeder_path)

    # 2. Load full state parquet
    parquet_path = input_dir / f"{state}_upgrade0.parquet"
    print(f"\nLoading full state parquet: {parquet_path}")
    df_all = load_parquet_metadata(parquet_path, COLUMNS)
    print(f"  {len(df_all):,} rows loaded.")

    # 3. Filter to model subset
    lookup_ids = pd.Index(list(all_bldg_ids)).astype(df_all.index.dtype)
    df_subset = df_all[df_all.index.isin(lookup_ids)]
    print(f"  {len(df_subset):,} rows match model subset.")

    # ── All state stats ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"ALL {state.upper()} STATS")
    print(f"{'='*55}")
    run_metadata_analysis(df_all, out_dir, prefix=f"all_{state.lower()}_")

    # ── Model subset stats ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("MODEL SUBSET STATS")
    print(f"{'='*55}")
    run_lookup_analysis(aggregated_counts, out_dir, prefix="model_")
    run_metadata_analysis(df_subset, out_dir, prefix="model_")

    # ── Adoption comparison (model vs full state) ───────────────────────────
    print(f"\n{'='*55}")
    print("ADOPTION COMPARISON")
    print(f"{'='*55}")
    plot_adoption_comparison(df_all, df_subset, out_dir, state)

    print(f"\n✅ All stats saved to: {out_dir}")


if __name__ == "__main__":
    main()
