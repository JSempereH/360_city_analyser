"""Shared, dependency-light validation for panorama analysis cache inputs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any


def canonical_json_sha256(payload: Any) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


@lru_cache(maxsize=4096)
def _file_sha256_for_version(path: str, _version: tuple[int, int, int, int, int]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, memoised by its identity and modification metadata.

    Every viewer click revalidates panoramas and masks; rehashing unchanged
    multi-megabyte files dominated those requests. Atomic replacement changes
    the inode, and in-place writes change size or mtime/ctime.
    """
    status = path.stat()
    return _file_sha256_for_version(
        str(path.resolve()), (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns),
    )


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def sfm_input_sha256(data_root: Path, dataset_id: str, image_id: str) -> str | None:
    path = data_root / dataset_id / "analysis" / f"{image_id}-sfm.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return canonical_json_sha256(payload) if isinstance(payload, dict) and payload.get("version") == 3 else None


def analysis_input_sha256(
    data_root: Path,
    dataset_id: str,
    image: dict[str, Any],
    configuration: dict[str, Any],
) -> str:
    local_file = image.get("local_file")
    if not isinstance(local_file, str):
        raise ValueError("The panorama has no local image file.")
    dataset_dir = (data_root / dataset_id).resolve()
    image_path = (dataset_dir / local_file).resolve()
    if not image_path.is_file() or not image_path.is_relative_to(dataset_dir):
        raise ValueError("The panorama image is unavailable.")
    image_id = str(image.get("id", ""))
    return canonical_json_sha256({
        "image_id": image_id,
        "local_file": local_file,
        "panorama_sha256": file_sha256(image_path),
        "geometry": image.get("selected_geometry") or image.get("computed_geometry") or image.get("geometry"),
        "computed_compass_angle_deg": image.get("computed_compass_angle_deg"),
        "sfm_input_sha256": sfm_input_sha256(data_root, dataset_id, image_id),
        "configuration": configuration,
    })


def analysis_cache_matches(
    payload: object,
    data_root: Path,
    dataset_id: str,
    image_id: str,
    expected_version: int,
    expected_input_sha256: str,
) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("version") == expected_version
        and str(payload.get("image_id")) == image_id
        and payload.get("input_sha256") == expected_input_sha256
        and payload.get("sfm_input_sha256") == sfm_input_sha256(data_root, dataset_id, image_id)
    )
