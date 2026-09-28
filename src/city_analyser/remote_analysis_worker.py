"""Authenticated HTTP worker for GPU-backed building analysis."""

from __future__ import annotations

import argparse
import base64
import os
import hmac
import json
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from .analysis_backend import MAX_REMOTE_IMAGE_BYTES, REMOTE_ANALYSIS_PATH
from .building_analysis import analysis_configuration, analysis_paths, analyze_panorama, facade_path, instance_path, preload_models


MAX_REQUEST_BYTES = MAX_REMOTE_IMAGE_BYTES * 2


class AnalysisWorker(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], model_root: Path, token: str | None):
        super().__init__(address, WorkerRequestHandler)
        self.model_root = model_root.resolve()
        self.token = token
        self.inference_lock = threading.Lock()
        self.loaded_models: dict[str, Any] | None = None


class WorkerRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/api/health":
            server = cast(AnalysisWorker, self.server)
            self.send_json({"status": "ok", "busy": server.inference_lock.locked(), "models": server.loaded_models})
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != REMOTE_ANALYSIS_PATH:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        server = cast(AnalysisWorker, self.server)
        if server.token and not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {server.token}"):
            self.send_json({"error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return
        try:
            payload = self.read_json_body()
            with server.inference_lock:
                result = self.analyze(server, payload)
        except ValueError as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        except (OSError, RuntimeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self.send_json(result)

    def analyze(self, server: AnalysisWorker, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("protocol_version") != 1:
            raise ValueError("Unsupported analysis protocol version.")
        image = payload.get("image")
        if not isinstance(image, dict) or not str(image.get("id", "")).isdecimal():
            raise ValueError("The request must include a Mapillary panorama image.")
        encoded_image = payload.get("image_jpeg_base64")
        if not isinstance(encoded_image, str):
            raise ValueError("The request has no panorama image.")
        try:
            image_bytes = base64.b64decode(encoded_image, validate=True)
        except ValueError as error:
            raise ValueError("The panorama image is invalid.") from error
        if not image_bytes or len(image_bytes) > MAX_REMOTE_IMAGE_BYTES:
            raise ValueError(f"The panorama must be smaller than {MAX_REMOTE_IMAGE_BYTES // (1024 * 1024)} MB.")
        image_id = str(image["id"])
        requested_configuration = payload.get("analysis_configuration")
        if not isinstance(requested_configuration, dict) or requested_configuration != analysis_configuration():
            raise ValueError("The worker configuration does not match the requested analysis configuration.")
        with tempfile.TemporaryDirectory(prefix="360-city-analyser-") as directory:
            data_root = Path(directory)
            dataset_id = "request"
            image_directory = data_root / dataset_id / "images"
            image_directory.mkdir(parents=True)
            image_path = image_directory / f"{image_id}.jpg"
            image_path.write_bytes(image_bytes)
            request_image = {**image, "local_file": str(image_path.relative_to(data_root / dataset_id))}
            sfm_refinement = payload.get("sfm_refinement")
            if sfm_refinement is not None:
                if (
                    not isinstance(sfm_refinement, dict) or sfm_refinement.get("version") != 3
                    or str(sfm_refinement.get("image_id")) != image_id
                    or not isinstance(sfm_refinement.get("depth_samples"), list)
                ):
                    raise ValueError("The optional SfM refinement is invalid.")
                analysis_directory = data_root / dataset_id / "analysis"
                analysis_directory.mkdir()
                (analysis_directory / f"{image_id}-sfm.json").write_text(json.dumps(sfm_refinement), encoding="utf-8")
            metadata = analyze_panorama(
                data_root, server.model_root, dataset_id, request_image, lambda _: None,
                sfm_refinement=sfm_refinement if isinstance(sfm_refinement, dict) else None,
            )
            _, id_path = analysis_paths(data_root, dataset_id, image_id)
            result = {
                "analysis": metadata,
                "id_mask_png_base64": base64.b64encode(id_path.read_bytes()).decode("ascii"),
                "facade_mask_png_base64": base64.b64encode(facade_path(data_root, dataset_id, image_id).read_bytes()).decode("ascii"),
            }
            instances = instance_path(data_root, dataset_id, image_id)
            if instances.is_file():
                result["instance_mask_png_base64"] = base64.b64encode(instances.read_bytes()).decode("ascii")
            return result

    def read_json_body(self) -> dict[str, Any]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid request body.") from error
        if not 0 < content_length <= MAX_REQUEST_BYTES:
            raise ValueError("Request body is too large.")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Invalid request body.")
        return payload

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a remote GPU building-analysis worker.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    parser.add_argument("--port", default=8766, type=int, help="HTTP port (default: 8766)")
    parser.add_argument("--model-root", default="models", help="directory for downloaded model weights")
    parser.add_argument(
        "--token", default=os.environ.get("BUILDING_ANALYSIS_REMOTE_TOKEN") or None,
        help="bearer token (default: BUILDING_ANALYSIS_REMOTE_TOKEN, which keeps it out of process listings)",
    )
    parser.add_argument("--preload", action="store_true", help="load all models before accepting requests")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"} and not args.token:
        parser.error("a token is required when binding outside localhost (--token or BUILDING_ANALYSIS_REMOTE_TOKEN)")
    server = AnalysisWorker((args.host, args.port), Path(args.model_root), args.token)
    if args.preload:
        print("Loading models...", flush=True)
        server.loaded_models = preload_models(Path(args.model_root))
        print(f"Models ready on {server.loaded_models['device']}.", flush=True)
    print(f"Building analysis worker: http://{args.host}:{args.port}{REMOTE_ANALYSIS_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
