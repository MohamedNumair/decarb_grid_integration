"""
Building-Grid Integration Pipeline Runner
==========================================

Pipeline phases:

1. preprocess       — data acquisition through building matching
2. decarb-inputs    — generate DECARB optimizer input files (bookend cases)
3. (external)       — run DECARB optimizer (Julia, run_decarb.jl)
4. network-inputs   — aggregate DECARB outputs into OpenDSS loadshapes
5. opendss          — run OpenDSS power flow for each case (Julia, run_opendss.jl)

Usage:
    python src/run_pipeline.py preprocess
    python src/run_pipeline.py decarb-inputs
    ... run DECARB optimizer externally ...
    python src/run_pipeline.py network-inputs --study 1
    python src/run_pipeline.py opendss --study 1
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config.loader import load_config, load_decarb_config, load_network_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_stage(name, fn):
    log.info(f"=== {name} ===")
    t = time.time()
    fn()
    log.info(f"{name} done ({time.time() - t:.0f}s)")


def _get_building_ids(cfg):
    import pandas as pd
    input_dir = cfg["paths"]["input_dir"]
    state = cfg["state"]
    release_year = cfg["resstock"]["release_year"]
    id_file = input_dir / f"resstock_bldg_ids_{state.lower()}_{release_year}.csv"
    if not id_file.exists():
        raise FileNotFoundError(f"Missing building ID list: {id_file}")
    bldg_ids = pd.read_csv(str(id_file))["bldg_id"].dropna().astype(int).unique()
    if cfg.get("test_mode") and cfg.get("test_limit"):
        bldg_ids = bldg_ids[: cfg["test_limit"]]
    return bldg_ids


# ---------------------------------------------------------------------------
# Phase 1: Preprocessing
# ---------------------------------------------------------------------------

def run_preprocess(cfg):
    stages = cfg["stages"]

    if stages.get("data_acquisition"):
        from data_acquisition import download_metadata, get_valid_building_ids, download_building_files, download_network_profiles
        def _acq():
            download_metadata(cfg)
            ids = get_valid_building_ids(cfg)
            download_building_files(cfg, ids)
            if cfg.get("matching", {}).get("download_kvar", False):
                download_network_profiles(cfg, kvar_only=True)
        _run_stage("Data Acquisition", _acq)

    if stages.get("thermal_model"):
        from fit_thermal_model import run_pipeline as run_thermal_pipeline
        bldg_ids = _get_building_ids(cfg)
        def _thermal():
            use_batching = len(bldg_ids) > 100
            run_thermal_pipeline(
                cfg=cfg, bldg_ids=bldg_ids,
                save_outputs=False,
                batch=use_batching,
                batch_size=100,
                n_workers=min(20, len(bldg_ids) // 100 + 1),
            )
        _run_stage("Thermal Model", _thermal)

    if stages.get("feeder_processing"):
        from process_network_info import process_all_zones
        _run_stage("Feeder Processing", lambda: process_all_zones(cfg))

    if stages.get("building_matching"):
        from building_load_matching import run_matching
        _run_stage("Building-Load Matching", lambda: run_matching(cfg))

    if stages.get("schedules"):
        from create_schedules import run_schedules
        _run_stage("Schedule Generation", lambda: run_schedules(cfg))

    if stages.get("heatpump_selection"):
        from heatpump_selection import run_heatpump_selection
        _run_stage("Heat Pump Selection", lambda: run_heatpump_selection(cfg))

    log.info("Preprocessing complete.")


# ---------------------------------------------------------------------------
# Phase 2: DECARB Input Generation
# ---------------------------------------------------------------------------

def run_decarb_inputs(cfg):
    dcfg = load_decarb_config()

    log.info("=== DECARB Input Generation (bookend cases) ===")
    t = time.time()
    from create_decarb_inputs import run_decarb_inputs as _run
    study_dir = _run(cfg, dcfg)
    log.info(f"DECARB inputs done ({time.time() - t:.0f}s)")
    log.info(f"Study: {study_dir}")
    log.info("Next: run DECARB optimizer on the generated inputs, then "
             "use 'network-inputs' to aggregate results.")


# ---------------------------------------------------------------------------
# Phase 3: Network Input Generation (after DECARB optimizer has run)
# ---------------------------------------------------------------------------

def run_network_inputs(cfg, args):
    import shutil
    dcfg = load_decarb_config()
    ncfg = load_network_config()

    from create_network_inputs import find_study_dir

    if args.study is not None:
        study_dir = find_study_dir(cfg["paths"]["output_dir"], args.study)
    else:
        study_dir = find_study_dir(cfg["paths"]["output_dir"])

    log.info(f"Using study: {study_dir}")

    # Optionally copy a matching file into the study
    if args.matching:
        matching_src = Path(args.matching)
        if not matching_src.exists():
            log.error(f"Matching file not found: {matching_src}")
            return
        matching_dst = study_dir / "feeder_summaries_matched.parquet"
        shutil.copy2(str(matching_src), str(matching_dst))
        log.info(f"Copied matching file into study: {matching_src.name}")

    # Parse penetration levels
    if args.electrification:
        elec_levels = [int(x) for x in args.electrification.split(",")]
    else:
        elec_levels = dcfg["electrification"]["levels"]

    if args.solar:
        solar_levels = [int(x) for x in args.solar.split(",")]
    else:
        solar_enabled = dcfg.get("rooftop_solar", {}).get("enabled", False)
        solar_levels = [0, 100] if solar_enabled else [0]

    log.info(f"Electrification levels: {elec_levels}")
    log.info(f"Solar levels: {solar_levels}")

    log.info("=== Network Demand Generation ===")
    t = time.time()

    interpolate = ncfg.get("interpolation", {}).get("enabled", False)

    if interpolate:
        from interpolate_demand import interpolate_study_demand
        log.info("Interpolation enabled — computing intermediate penetration levels")
        interpolate_study_demand(
            cfg, ncfg, study_dir,
            elec_levels=elec_levels,
            solar_levels=solar_levels,
            seed=dcfg.get("rooftop_solar", {}).get("seed", 42),
        )
    else:
        from create_network_inputs import run_network_inputs as _run_ni
        log.info("Interpolation disabled — direct aggregation from DECARB outputs")
        _run_ni(cfg, ncfg, study_num=int(study_dir.name.split("_")[1]))

    log.info(f"Network demand done ({time.time() - t:.0f}s)")
    log.info(f"Study output: {study_dir}")


# ---------------------------------------------------------------------------
# Phase 5: OpenDSS Power Flow (Julia subprocess)
# ---------------------------------------------------------------------------

def run_opendss(cfg, args):
    import subprocess

    from create_network_inputs import find_study_dir
    output_dir = cfg["paths"]["output_dir"]

    if args.study is not None:
        study_dir = find_study_dir(output_dir, args.study)
    else:
        study_dir = find_study_dir(output_dir)

    log.info(f"=== OpenDSS Power Flow ===")
    log.info(f"Study: {study_dir}")

    script_path = Path(__file__).parent / "run_opendss.py"
    if not script_path.exists():
        log.error(f"OpenDSS runner not found: {script_path}")
        return

    project_root = Path(__file__).parents[1]
    study_num = study_dir.name.split("_")[1]

    cmd = [
        sys.executable,
        str(script_path),
        "--study", study_num,
    ]

    log.info(f"Running: {' '.join(cmd)}")
    t = time.time()
    result = subprocess.run(cmd, cwd=str(project_root))

    if result.returncode == 0:
        log.info(f"OpenDSS complete ({time.time() - t:.0f}s)")
    else:
        log.error(f"OpenDSS failed with exit code {result.returncode}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Building-Grid Integration Pipeline")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("preprocess",
                   help="Run data preparation stages (data acquisition through HP selection)")

    sub.add_parser("decarb-inputs",
                   help="Generate DECARB optimizer input files (bookend cases)")

    net = sub.add_parser("network-inputs",
                         help="Aggregate DECARB outputs into GenX/OpenDSS inputs")
    net.add_argument("--study", type=int, default=None,
                     help="Study number to use (default: latest)")
    net.add_argument("--matching", type=str, default=None,
                     help="Path to matching parquet to use (copies into study dir). "
                          "If omitted, uses study-level file or global fallback.")
    net.add_argument("--electrification", type=str, default=None,
                     help="Comma-separated electrification levels (e.g. 0,25,50,75,100)")
    net.add_argument("--solar", type=str, default=None,
                     help="Comma-separated solar penetration levels (e.g. 0,30,60,100)")

    dss = sub.add_parser("opendss",
                         help="Run OpenDSS power flow for each case in a study")
    dss.add_argument("--study", type=int, default=None,
                     help="Study number to use (default: latest)")

    res = sub.add_parser("results",
                         help="Generate result plots from OpenDSS outputs")
    res.add_argument("--study", type=int, default=None,
                     help="Study number to use (default: latest)")

    args = parser.parse_args()
    cfg = load_config()

    if args.command == "preprocess":
        run_preprocess(cfg)
    elif args.command == "decarb-inputs":
        run_decarb_inputs(cfg)
    elif args.command == "network-inputs":
        run_network_inputs(cfg, args)
    elif args.command == "opendss":
        run_opendss(cfg, args)
    elif args.command == "results":
        from plot_results import main as plot_main
        sys.argv = ["plot_results.py"]
        if args.study is not None:
            sys.argv += ["--study", str(args.study)]
        plot_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
