import os
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

_config = None


def load_config(config_path=None):
    global _config
    if _config is not None and config_path is None:
        return _config

    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    config_dir = config_path.parent

    env_file = config_dir / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for key in ("project_root", "input_dir", "output_dir", "network_dir", "county_shapefile"):
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = config_dir / p
        cfg["paths"][key] = p

    if cfg["api_keys"].get("nrel") is None:
        cfg["api_keys"]["nrel"] = os.environ.get("NREL_API_KEY")

    _config = cfg
    return cfg


_decarb_config = None


def load_decarb_config(config_path=None):
    global _decarb_config
    if _decarb_config is not None and config_path is None:
        return _decarb_config

    if config_path is None:
        config_path = Path(__file__).parent / "decarb.yaml"
    else:
        config_path = Path(config_path)

    config_dir = config_path.parent

    with open(config_path) as f:
        dcfg = yaml.safe_load(f)

    template_dir = Path(dcfg.get("template_dir", "../in/decarb_input_templates"))
    if not template_dir.is_absolute():
        template_dir = config_dir / template_dir
    dcfg["template_dir"] = template_dir

    _decarb_config = dcfg
    return dcfg


_network_config = None


def load_network_config(config_path=None):
    global _network_config
    if _network_config is not None and config_path is None:
        return _network_config

    if config_path is None:
        config_path = Path(__file__).parent / "network.yaml"
    else:
        config_path = Path(config_path)

    config_dir = config_path.parent

    with open(config_path) as f:
        ncfg = yaml.safe_load(f)

    # Resolve relative paths in genx section
    genx = ncfg.get("genx", {})
    for key in ("base_case", "bus_mapping"):
        val = genx.get(key)
        if val is not None:
            p = Path(val)
            if not p.is_absolute():
                genx[key] = config_dir / p
            else:
                genx[key] = p

    _network_config = ncfg
    return ncfg


def get_path(cfg, *parts):
    return cfg["paths"]["project_root"].joinpath(*parts)


def resolve_hour_indices(cfg):
    """Return a sorted array of 0-based hour indices (into 8760) for the
    configured date_range.  Handles year-wrap when start > end.
    Returns None when the range covers the full year."""
    dr = cfg.get("date_range")
    if dr is None:
        return None
    year = cfg["year"]
    start_str = dr.get("start", "Jan 1")
    end_str = dr.get("end", "Dec 31")

    start_dt = datetime.strptime(f"{start_str} {year}", "%b %d %Y")
    end_dt = datetime.strptime(f"{end_str} {year}", "%b %d %Y")
    end_dt = end_dt.replace(hour=23)

    jan1 = datetime(year, 1, 1)
    start_hour = int((start_dt - jan1).total_seconds() // 3600)
    end_hour = int((end_dt - jan1).total_seconds() // 3600)

    if start_hour <= end_hour:
        indices = np.arange(start_hour, end_hour + 1)
    else:
        indices = np.concatenate([
            np.arange(start_hour, 8760),
            np.arange(0, end_hour + 1),
        ])

    if len(indices) == 8760:
        return None
    return indices
