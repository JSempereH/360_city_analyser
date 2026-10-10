"""Configurable local or remote execution for building analysis.

Only the model stages (``perceive_panorama``) can run remotely. A remote worker
receives a single panorama and returns its semantic confidences and instance
map; pose refinement, OpenStreetMap association and artifact publication always
run here, next to the OSM cache, the footprint corrections and the SfM depth.
Credentials and the worker endpoint never reach the browser.
"""

from __future__ import annotations

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .analysis_cache import file_sha256
from .building_analysis import (
    Perception,
    analyze_panorama,
    associate_panorama,
    nearby_buildings,
    panorama_inputs,
    perceive_panorama,
    perception_configuration,
    perception_from_arrays,
    perception_to_arrays,
)
from .stage_cache import load_stage, save_stage, stage_key


REMOTE_PERCEPTION_PATH = "/api/v2/perception"
PROTOCOL_VERSION = 2
REMOTE_TIMEOUT_SECONDS = 900
MAX_REMOTE_IMAGE_BYTES = 32 * 1024 * 1024
NPZ_SIGNATURE = b"PK"
REMOTE_STAGE = "remote-perception"


@dataclass(frozen=True)
class AnalysisBackend:
    kind: str
    remote_url: str | None = None
    token: str | None = None
    timeout_seconds: int = REMOTE_TIMEOUT_SECONDS

    @property
    def label(self) -> str:
        return "remote GPU" if self.kind == "remote" else "local device"


def _remote_endpoint(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("BUILDING_ANALYSIS_REMOTE_URL must be an HTTP(S) base URL without query parameters")
    return value.rstrip("/") + REMOTE_PERCEPTION_PATH


def analysis_backend_from_environment(environment: Mapping[str, str] | None = None) -> AnalysisBackend:
    environment = os.environ if environment is None else environment
    configured = environment.get("BUILDING_ANALYSIS_BACKEND", "auto").strip().lower()
    remote_url = environment.get("BUILDING_ANALYSIS_REMOTE_URL", "").strip()
    if configured not in {"auto", "local", "remote"}:
        raise ValueError("BUILDING_ANALYSIS_BACKEND must be auto, local, or remote")
    if configured == "remote" or configured == "auto" and remote_url:
        if not remote_url:
            raise ValueError("BUILDING_ANALYSIS_REMOTE_URL is required when BUILDING_ANALYSIS_BACKEND=remote")
        requested_model = environment.get("BUILDING_ANALYSIS_MODEL", "").strip()
        if requested_model in {"", "auto"}:
            raise ValueError("BUILDING_ANALYSIS_MODEL must be explicit when using a remote analysis backend")
        try:
            timeout_seconds = int(environment.get("BUILDING_ANALYSIS_REMOTE_TIMEOUT_S", str(REMOTE_TIMEOUT_SECONDS)))
        except ValueError as error:
            raise ValueError("BUILDING_ANALYSIS_REMOTE_TIMEOUT_S must be an integer") from error
        if not 1 <= timeout_seconds <= 3_600:
            raise ValueError("BUILDING_ANALYSIS_REMOTE_TIMEOUT_S must be between 1 and 3600")
        return AnalysisBackend("remote", _remote_endpoint(remote_url), environment.get("BUILDING_ANALYSIS_REMOTE_TOKEN") or None, timeout_seconds)
    return AnalysisBackend("local")


def analysis_backend_status(backend: AnalysisBackend) -> dict[str, Any]:
    return {"kind": backend.kind, "label": backend.label, "remote_configured": backend.remote_url is not None}


def _remote_error(error: HTTPError) -> RuntimeError:
    try:
        payload = json.loads(error.read().decode("utf-8"))
        message = payload.get("error") if isinstance(payload, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        message = None
    return RuntimeError(f"Remote analysis failed ({error.code}): {message or error.reason}")


def _remote_perception(backend: AnalysisBackend, image_path: Path, progress: Callable[[str], None]) -> Perception:
    if backend.remote_url is None:
        raise RuntimeError("Remote analysis is not configured.")
    image_bytes = image_path.read_bytes()
    if len(image_bytes) > MAX_REMOTE_IMAGE_BYTES:
        raise ValueError(f"The panorama exceeds the {MAX_REMOTE_IMAGE_BYTES // (1024 * 1024)} MB remote analysis limit.")
    progress("Uploading panorama to remote GPU...")
    # The body is the JPEG itself; the configuration travels in a header so
    # the worker can reject a mismatch before decoding anything.
    headers = {
        "Content-Type": "image/jpeg",
        "X-Analysis-Protocol": str(PROTOCOL_VERSION),
        "X-Analysis-Configuration": json.dumps(perception_configuration(), sort_keys=True, separators=(",", ":")),
        "User-Agent": "local-panorama-building-viewer/1.0",
    }
    if backend.token:
        headers["Authorization"] = f"Bearer {backend.token}"
    request = Request(backend.remote_url, data=image_bytes, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=backend.timeout_seconds) as response:  # noqa: S310 - administrator-configured endpoint
            content = response.read()
    except HTTPError as error:
        raise _remote_error(error) from error
    except (OSError, URLError) as error:
        raise RuntimeError(f"Remote analysis could not be reached: {error}") from error
    if not content.startswith(NPZ_SIGNATURE):
        raise RuntimeError("Remote analysis returned an invalid response.")
    np = __import__("numpy")
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in stored.files}
        if not {"building", "vegetation", "model", "revision", "device", "details"} <= arrays.keys():
            raise ValueError("missing arrays")
        return perception_from_arrays(arrays)
    except (ValueError, OSError, EOFError, KeyError) as error:
        raise RuntimeError("Remote analysis returned an invalid response.") from error


def perceive(
    backend: AnalysisBackend, data_root: Path, model_root: Path, image_path: Path, progress: Callable[[str], None],
) -> Perception:
    """Model-stage outputs for one panorama from this machine or from the remote worker."""
    if backend.kind != "remote":
        return perceive_panorama(data_root, model_root, image_path, progress)
    # A local copy of the worker's answer keeps re-association (new footprints,
    # thresholds, corrections) from uploading the panorama again. Pin
    # BUILDING_ANALYSIS_MODEL_REVISION to also invalidate it on a model update.
    key = stage_key(REMOTE_STAGE, {"panorama_sha256": file_sha256(image_path), "configuration": perception_configuration()})
    cached = load_stage(data_root, REMOTE_STAGE, key)
    if cached is not None:
        progress("Reusing cached remote perception...")
        return perception_from_arrays(cached)
    perception = _remote_perception(backend, image_path, progress)
    save_stage(data_root, REMOTE_STAGE, key, **perception_to_arrays(perception))
    return perception


def run_analysis(
    backend: AnalysisBackend,
    data_root: Path,
    model_root: Path,
    dataset_id: str,
    image: dict[str, Any],
    progress: Callable[[str], None],
) -> dict[str, Any]:
    if backend.kind != "remote":
        return analyze_panorama(data_root, model_root, dataset_id, image, progress)
    inputs = panorama_inputs(data_root, dataset_id, image)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="osm-footprints") as pool:
        pending_features = pool.submit(nearby_buildings, inputs.latitude, inputs.longitude)
        perception = perceive(backend, data_root, model_root, inputs.image_path, progress)
        progress("Loading nearby OpenStreetMap building footprints...")
        features = pending_features.result()
    metadata = associate_panorama(data_root, dataset_id, image, inputs, perception, features, progress)
    progress("Remote building analysis complete.")
    return metadata
