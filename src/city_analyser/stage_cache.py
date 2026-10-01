"""Content-addressed cache for the expensive per-panorama model stages.

Model outputs depend only on the panorama bytes, the model and its revision,
and the stage's own settings. Keying them by exactly that, instead of by the
whole analysis configuration, lets OSM association, thresholds and
``ANALYSIS_VERSION`` change without running the models again. Entries live
under ``<data root>/.stage-cache/<stage>/<key[:2]>/<key>.npz`` and are shared by
every dataset that contains the same panorama, or under
``BUILDING_ANALYSIS_STAGE_CACHE_DIR`` when it is set.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Any

from .analysis_cache import atomic_write_bytes, canonical_json_sha256

STAGE_CACHE_DIRECTORY = ".stage-cache"


def stage_cache_enabled() -> bool:
    mode = os.environ.get("BUILDING_ANALYSIS_STAGE_CACHE", "on").strip().lower()
    if mode not in {"on", "off"}:
        raise RuntimeError("BUILDING_ANALYSIS_STAGE_CACHE must be on or off.")
    return mode == "on"


def stage_key(stage: str, identity: dict[str, Any]) -> str:
    return canonical_json_sha256({"stage": stage, **identity})


def stage_path(data_root: Path, stage: str, key: str) -> Path:
    # A remote worker analyzes each request in a temporary data root, so it
    # needs an explicit, persistent location to reuse model outputs.
    configured = os.environ.get("BUILDING_ANALYSIS_STAGE_CACHE_DIR", "").strip()
    root = Path(configured) if configured else data_root / STAGE_CACHE_DIRECTORY
    return root / stage / key[:2] / f"{key}.npz"


def load_stage(data_root: Path, stage: str, key: str) -> dict[str, Any] | None:
    """Stored arrays for ``key``, or ``None`` on a miss or an unreadable entry."""
    if not stage_cache_enabled():
        return None
    np = __import__("numpy")
    path = stage_path(data_root, stage, key)
    try:
        # np.load leaves the file open when the archive is corrupt.
        with open(path, "rb") as source, np.load(source, allow_pickle=False) as stored:
            return {name: stored[name] for name in stored.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        # A missing file is a miss; a truncated one is recomputed and replaced.
        return None


def save_stage(data_root: Path, stage: str, key: str, **arrays: Any) -> None:
    if not stage_cache_enabled():
        return
    np = __import__("numpy")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_write_bytes(stage_path(data_root, stage, key), buffer.getvalue())
