"""
Per-run log file.

run_pipeline.py calls start_run_log() once per command; it writes everything
logged in the main process to out/logs/{command}_{timestamp}.log, alongside
the console. The path is also put in the PIPELINE_LOG_FILE environment
variable, which worker processes and subprocesses inherit; they call
attach_run_log() at startup so their messages land in the same file.
"""

import logging
import os
import time
from pathlib import Path

LOG_ENV_VAR = "PIPELINE_LOG_FILE"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def _add_file_handler(path):
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == Path(path).resolve():
            return  # already attached in this process
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)


def start_run_log(cfg, command):
    """Create out/logs/{command}_{YYYY-MM-DD_HHMMSS}.log and log to it."""
    log_dir = Path(cfg["paths"]["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{command}_{time.strftime('%Y-%m-%d_%H%M%S')}.log"
    _add_file_handler(path)
    os.environ[LOG_ENV_VAR] = str(path)
    return path


def attach_run_log():
    """In a worker process or subprocess: also log to the run's log file."""
    path = os.environ.get(LOG_ENV_VAR)
    if path:
        _add_file_handler(path)
