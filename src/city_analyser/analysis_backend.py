"""Configurable local or remote execution for building analysis.

The viewer owns the dataset and analysis cache. A remote worker only receives a
single panorama and returns the generated analysis artifacts, so credentials and
the worker endpoint never reach the browser.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .analysis_cache import analysis_input_sha256, sfm_input_sha256
from .building_analysis import ANALYSIS_VERSION, analysis_configuration, analyze_panorama, publish_analysis_artifacts


REMOTE_ANALYSIS_PATH = "/api/v1/building-analysis"
REMOTE_TIMEOUT_SECONDS = 900
MAX_REMOTE_IMAGE_BYTES = 32 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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
    return value.rstrip("/") + REMOTE_ANALYSIS_PATH


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


def _decode_png(value: object, name: str) -> bytes:
    if not isinstance(value, str):
        raise RuntimeError(f"Remote analysis returned no {name} mask.")
    try:
        content = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise RuntimeError(f"Remote analysis returned an invalid {name} mask.") from error
    if not content.startswith(PNG_SIGNATURE):
        raise RuntimeError(f"Remote analysis returned an invalid {name} mask.")
    return content


def _run_remote_analysis(
    backend: AnalysisBackend,
    data_root: Path,
    dataset_id: str,
    image: dict[str, Any],
    progress: Callable[[str], None],
) -> dict[str, Any]:
    if backend.remote_url is None:
        raise RuntimeError("Remote analysis is not configured.")
    local_file = image.get("local_file")
    if not isinstance(local_file, str):
        raise ValueError("The panorama has no local image file.")
    image_path = (data_root / dataset_id / local_file).resolve()
    if not image_path.is_file() or not image_path.is_relative_to((data_root / dataset_id).resolve()):
        raise ValueError("The panorama image is unavailable.")
    image_bytes = image_path.read_bytes()
    if len(image_bytes) > MAX_REMOTE_IMAGE_BYTES:
        raise ValueError(f"The panorama exceeds the {MAX_REMOTE_IMAGE_BYTES // (1024 * 1024)} MB remote analysis limit.")
    image_id = str(image.get("id", ""))
    progress("Uploading panorama to remote GPU...")
    requested_configuration = analysis_configuration()
    payload = {
        "protocol_version": 1,
        "dataset_id": dataset_id,
        # The remote worker needs capture metadata, not the local storage path.
        "image": {key: value for key, value in image.items() if key != "local_file"},
        "image_jpeg_base64": base64.b64encode(image_bytes).decode("ascii"),
        "analysis_configuration": requested_configuration,
    }
    from .sfm_refinement import load_sfm_refinement, sfm_path

    sfm_refinement = load_sfm_refinement(data_root, dataset_id, image_id)
    if sfm_refinement is None:
        sfm_path(data_root, dataset_id, image_id).unlink(missing_ok=True)
    if sfm_refinement is not None:
        payload["sfm_refinement"] = sfm_refinement
    headers = {"Content-Type": "application/json", "User-Agent": "local-panorama-building-viewer/1.0"}
    if backend.token:
        headers["Authorization"] = f"Bearer {backend.token}"
    request = Request(backend.remote_url, data=json.dumps(payload, separators=(",", ":")).encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=backend.timeout_seconds) as response:  # noqa: S310 - administrator-configured endpoint
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise _remote_error(error) from error
    except (OSError, URLError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Remote analysis could not be reached: {error}") from error
    if not isinstance(result, dict) or not isinstance(result.get("analysis"), dict):
        raise RuntimeError("Remote analysis returned an invalid response.")
    metadata = result["analysis"]
    if metadata.get("version") != ANALYSIS_VERSION or metadata.get("image_id") != image_id:
        raise RuntimeError("Remote analysis result is incompatible with this viewer.")
    if metadata.get("analysis_configuration") != requested_configuration:
        raise RuntimeError("Remote analysis used a different model or inference configuration.")
    id_mask = _decode_png(result.get("id_mask_png_base64"), "building ID")
    facade_mask = _decode_png(result.get("facade_mask_png_base64"), "facade")
    instance_mask = _decode_png(result["instance_mask_png_base64"], "instance") if "instance_mask_png_base64" in result else None
    metadata["input_sha256"] = analysis_input_sha256(data_root, dataset_id, image, analysis_configuration())
    metadata["sfm_input_sha256"] = sfm_input_sha256(data_root, dataset_id, image_id)
    metadata = publish_analysis_artifacts(data_root, dataset_id, image_id, metadata, id_mask, facade_mask, instance_mask)
    progress("Remote building analysis complete.")
    return metadata


def run_analysis(
    backend: AnalysisBackend,
    data_root: Path,
    model_root: Path,
    dataset_id: str,
    image: dict[str, Any],
    progress: Callable[[str], None],
) -> dict[str, Any]:
    if backend.kind == "remote":
        return _run_remote_analysis(backend, data_root, dataset_id, image, progress)
    return analyze_panorama(data_root, model_root, dataset_id, image, progress)
