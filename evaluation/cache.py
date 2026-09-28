"""Cached per-panorama model outputs, so association variants can be compared cheaply.

Semantic masks and instance maps are the expensive stages (tens of seconds per
panorama on CPU). They are stored under ``data/.eval-cache/<stage>/<variant>/``
keyed by dataset and image, and reused by every association experiment.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

CACHE_ROOT = Path("data") / ".eval-cache"


@contextmanager
def environment(**values: str) -> Iterator[None]:
    """Temporarily set analysis environment variables for one variant."""
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def panoramas(data_root: Path = Path("data")) -> list[tuple[str, dict[str, Any]]]:
    """Every downloaded panorama once, keyed by its first dataset."""
    seen, result = set(), []
    for manifest in sorted(data_root.glob("*/manifest.json")):
        # Demo datasets re-use panoramas of other datasets for recordings.
        if manifest.parent.name.startswith("demo-"):
            continue
        for image in json.loads(manifest.read_text(encoding="utf-8"))["images"]:
            if image["id"] not in seen and image.get("local_file"):
                seen.add(image["id"])
                result.append((manifest.parent.name, image))
    return result


def cache_path(stage: str, variant: str, dataset_id: str, image_id: str) -> Path:
    return CACHE_ROOT / stage / variant / f"{dataset_id}__{image_id}.npz"


def load(stage: str, variant: str, dataset_id: str, image_id: str) -> dict[str, Any] | None:
    path = cache_path(stage, variant, dataset_id, image_id)
    if not path.is_file():
        return None
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def save(stage: str, variant: str, dataset_id: str, image_id: str, **arrays: Any) -> None:
    path = cache_path(stage, variant, dataset_id, image_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def canonical_digest(payload: Any) -> str:
    from city_analyser.analysis_cache import canonical_json_sha256

    return canonical_json_sha256(payload)
