"""
Settings record for stages that skip buildings they have already processed.

A stage saves the settings its outputs were made with in a small JSON file
next to those outputs. On the next run, existing outputs are reused only if
the record says the previous run finished and the settings are the same;
otherwise the stage recomputes every building. Each stage includes a version
number in its settings, to be bumped whenever its logic changes in a way that
affects results.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _normalize(settings):
    # Round-trip through JSON so the comparison sees what is stored on disk
    # (tuples become lists, Paths become strings, ...)
    return json.loads(json.dumps(settings, default=str, sort_keys=True))


def _differences(old, new, prefix=""):
    diffs = []
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key, "<missing>"), new.get(key, "<missing>")
        if isinstance(a, dict) and isinstance(b, dict):
            diffs += _differences(a, b, f"{prefix}{key}.")
        elif a != b:
            diffs.append(f"{prefix}{key}: {a} -> {b}")
    return diffs


def can_reuse(record_path, settings, stage):
    """True if existing outputs were made by a finished run with the same
    settings. Logs the reason when they can't be reused."""
    record_path = Path(record_path)
    if not record_path.exists():
        log.info(f"{stage}: no settings record yet; processing all buildings")
        return False
    with open(record_path) as f:
        record = json.load(f)
    if record.get("status") != "done":
        log.warning(f"{stage}: the previous run did not finish; reprocessing all buildings")
        return False
    diffs = _differences(record.get("settings", {}), _normalize(settings))
    if diffs:
        log.warning(f"{stage}: settings changed since the existing outputs were made "
                    f"({'; '.join(diffs)}); reprocessing all buildings")
        return False
    log.info(f"{stage}: settings unchanged; skipping buildings that are already done")
    return True


def write_record(record_path, settings, status):
    """Record the settings with status 'running' (at start) or 'done' (at end)."""
    record_path = Path(record_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, "w") as f:
        json.dump({"status": status, "settings": _normalize(settings)}, f, indent=2)
