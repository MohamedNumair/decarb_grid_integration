import logging
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
import requests
from botocore import UNSIGNED
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SMARTDS_PROFILES_URL = "https://oedi-data-lake.s3.amazonaws.com/SMART-DS/v0.9/2018/Full_Texas/profiles/"
MAX_WORKERS = 2

def _s3_prefix(cfg):
    rs = cfg["resstock"]
    state = cfg["state"]
    return (
        f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
        f"{rs['release_year']}/{rs['release']}/timeseries_individual_buildings/"
        f"by_state/upgrade=0/state={state}/"
    )


def _s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def building_id_file(cfg):
    """Full list of ResStock building IDs for the state (written by acquisition)."""
    return (cfg["paths"]["input_dir"]
            / f"resstock_bldg_ids_{cfg['state'].lower()}_{cfg['resstock']['release_year']}.csv")


def apply_test_limit(cfg, building_ids):
    """In test mode, keep the test_limit lowest building IDs; otherwise all."""
    building_ids = list(building_ids)
    if cfg.get("test_mode") and cfg.get("test_limit"):
        limited = sorted(building_ids)[: cfg["test_limit"]]
        log.warning(f"TEST MODE: using {len(limited)} of {len(building_ids):,} buildings")
        return limited
    return building_ids


def load_building_ids(cfg):
    """Building IDs every preprocessing stage runs on (test_limit applied)."""
    id_file = building_id_file(cfg)
    if not id_file.exists():
        raise FileNotFoundError(
            f"Missing building ID list: {id_file}. Run the data acquisition stage "
            f"first (stages.data_acquisition: true in config.yaml).")
    building_ids = pd.read_csv(id_file)["bldg_id"].dropna().astype(int).drop_duplicates()
    return apply_test_limit(cfg, building_ids)


def get_valid_building_ids(cfg):
    s3 = _s3_client()
    bucket = cfg["resstock"]["s3_bucket"]
    prefix = _s3_prefix(cfg)
    pattern = re.compile(r"(\d+)-\d+\.parquet")

    log.info(f"Listing building IDs from s3://{bucket}/{prefix}")

    building_ids = []
    continuation_token = None

    while True:
        params = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if continuation_token:
            params["ContinuationToken"] = continuation_token

        response = s3.list_objects_v2(**params)

        for obj in response.get("Contents", []):
            match = pattern.search(obj["Key"])
            if match:
                building_ids.append(int(match.group(1)))

        continuation_token = response.get("NextContinuationToken")
        if not continuation_token:
            break

    log.info(f"Found {len(building_ids)} building IDs")

    # Always save the full list; test_limit is applied when the list is read
    id_file = building_id_file(cfg)
    id_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"bldg_id": building_ids}).to_csv(id_file, index=False)
    log.info(f"Saved building ID list to {id_file}")

    return building_ids


def download_building_files(cfg, building_ids=None):
    if building_ids is None:
        building_ids = load_building_ids(cfg)
    else:
        building_ids = apply_test_limit(cfg, building_ids)

    s3 = _s3_client()
    bucket = cfg["resstock"]["s3_bucket"]
    prefix = _s3_prefix(cfg)
    state = cfg["state"]
    output_dir = cfg["paths"]["input_dir"] / "residential_profiles" / state
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = []

    log.info(f"Downloading {len(building_ids)} buildings to {output_dir}")

    for bldg_id in building_ids:
        local_path = output_dir / f"{bldg_id}-0.parquet"

        if local_path.exists():
            skipped += 1
            continue

        key = f"{prefix}{bldg_id}-0.parquet"
        try:
            s3.download_file(bucket, key, str(local_path))
            downloaded += 1
            if downloaded % 50 == 0:
                log.info(f"  downloaded {downloaded} so far...")
        except Exception as e:
            log.warning(f"Failed: {bldg_id} ({e})")
            failed.append(bldg_id)

    log.info(f"Done. downloaded={downloaded}, skipped={skipped}, failed={len(failed)}")

    if failed:
        failed_file = output_dir / "failed_downloads.csv"
        pd.DataFrame({"bldg_id": failed}).to_csv(failed_file, index=False)
        log.warning(f"Failed IDs saved to {failed_file}")

    return {"downloaded": downloaded, "skipped": skipped, "failed": failed}


def _metadata_s3_key(cfg, fmt):
    rs = cfg["resstock"]
    state = cfg["state"]
    ext = "csv" if fmt == "csv" else "parquet"
    filename = f"{state}_upgrade0.{ext}"
    if fmt == "csv":
        filename += ".gz"
    return (
        f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
        f"{rs['release_year']}/{rs['release']}/metadata_and_annual_results/"
        f"by_state/full/{ext}/state={state}/{filename}"
    )


def download_metadata(cfg):
    s3 = _s3_client()
    bucket = cfg["resstock"]["s3_bucket"]
    state = cfg["state"]
    output_dir = cfg["paths"]["input_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # The parquet is what every pipeline stage reads; the csv.gz is only
    # downloaded for visually checking the metadata by hand.
    files = {
        "csv": f"{state}_upgrade0.csv.gz",
        "parquet": f"{state}_upgrade0.parquet",
    }

    for fmt, filename in files.items():
        local_path = output_dir / filename
        if local_path.exists():
            log.info(f"Metadata {filename} already exists, skipping")
            continue

        key = _metadata_s3_key(cfg, fmt)
        log.info(f"Downloading metadata: s3://{bucket}/{key}")
        try:
            s3.download_file(bucket, key, str(local_path))
            log.info(f"Saved to {local_path}")
        except Exception as e:
            if fmt == "parquet":
                rs = cfg["resstock"]
                raise RuntimeError(
                    f"Failed to download ResStock metadata s3://{bucket}/{key}: {e}\n"
                    f"Check state ({state!r}), resstock.release ({rs['release']!r}) and "
                    f"resstock.release_year ({rs['release_year']!r}) in config.yaml.") from e
            log.warning(f"Failed to download {filename} (only used for manual checks): {e}")


def _collect_profile_filenames(opendss_dir):
    filenames = set()
    for root, _, files in os.walk(str(opendss_dir)):
        for f in files:
            if f == "LoadShapes.dss":
                with open(os.path.join(root, f), "r") as fh:
                    for match in re.finditer(r'file=[\.\./]*profiles/([^\s\)]+)', fh.read()):
                        filenames.add(match.group(1))
    return filenames


def _download_profile(filename, output_dir, max_retries=5):
    url = SMARTDS_PROFILES_URL + filename
    out = os.path.join(str(output_dir), filename)
    if os.path.exists(out):
        return "skipped"
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(out, "wb") as f:
                f.write(resp.content)
            return "downloaded"
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt}/{max_retries} failed for {filename}: {e}")
            time.sleep(5 * attempt)
    log.error(f"Failed to download {filename} after {max_retries} attempts")
    return "failed"


def network_profiles_enabled(cfg):
    matching = cfg.get("matching", {})
    if "download_network_profiles" in matching:
        return bool(matching["download_network_profiles"])
    if "download_kvar" in matching:
        log.warning("config key 'matching.download_kvar' is deprecated; "
                    "rename it to 'matching.download_network_profiles'")
        return bool(matching["download_kvar"])
    return False


def download_network_profiles(cfg, kvar_only=False):
    network_dir = cfg["paths"]["network_dir"]

    zone_dirs = sorted(
        d for d in Path(network_dir).iterdir()
        if d.is_dir()
    )

    from process_network_info import select_test_zones
    zone_dirs = select_test_zones(cfg, zone_dirs)

    for zone_dir in zone_dirs:
        opendss_dir = zone_dir / "scenarios" / "base_timeseries" / "opendss"
        if not opendss_dir.is_dir():
            log.warning(f"No opendss dir for {zone_dir.name}, skipping")
            continue

        all_filenames = _collect_profile_filenames(opendss_dir)
        if kvar_only:
            to_download = sorted(f for f in all_filenames if "_kvar_" in f)
        else:
            to_download = sorted(all_filenames)

        profiles_dir = opendss_dir / "opendss_loadshape_files"
        profiles_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"{zone_dir.name}: {len(to_download)} profile files to check")

        succeeded = 0
        failed = 0
        skipped = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(_download_profile, fn, profiles_dir): fn
                for fn in to_download
            }
            for future in as_completed(futures):
                result = future.result()
                if result == "downloaded":
                    succeeded += 1
                    if succeeded % 50 == 0:
                        log.info(f"  {zone_dir.name}: {succeeded} downloaded so far")
                elif result == "skipped":
                    skipped += 1
                else:
                    failed += 1

        log.info(f"{zone_dir.name}: {succeeded} downloaded, {skipped} already existed, {failed} failed")


if __name__ == "__main__":
    cfg = load_config()
    download_metadata(cfg)
    ids = get_valid_building_ids(cfg)
    download_building_files(cfg, ids)
    if network_profiles_enabled(cfg):
        download_network_profiles(cfg)
