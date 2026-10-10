"""Authenticated HTTP worker for GPU-backed perception (semantic and instance model stages).

It never touches OpenStreetMap: the client associates the returned model outputs with footprints."""

from __future__ import annotations

import argparse
import hmac
import io
import json
import os
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from .analysis_backend import MAX_REMOTE_IMAGE_BYTES, PROTOCOL_VERSION, REMOTE_PERCEPTION_PATH
from .building_analysis import perceive_panorama, perception_configuration, perception_to_arrays, preload_models


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
        if urlparse(self.path).path != REMOTE_PERCEPTION_PATH:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        server = cast(AnalysisWorker, self.server)
        if server.token and not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {server.token}"):
            self.send_json({"error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return
        try:
            image_bytes = self.read_image_body()
            # Only the model stages hold the lock: OSM association stays on the client.
            with server.inference_lock:
                result = self.perceive(server, image_bytes)
        except ValueError as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        except (OSError, RuntimeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self.send_bytes(result, "application/x-npz")

    def perceive(self, server: AnalysisWorker, image_bytes: bytes) -> bytes:
        if self.headers.get("X-Analysis-Protocol") != str(PROTOCOL_VERSION):
            raise ValueError("Unsupported analysis protocol version.")
        try:
            requested_configuration = json.loads(self.headers.get("X-Analysis-Configuration", ""))
        except json.JSONDecodeError as error:
            raise ValueError("The analysis configuration header is invalid.") from error
        if requested_configuration != perception_configuration():
            raise ValueError("The worker configuration does not match the requested analysis configuration.")
        with tempfile.TemporaryDirectory(prefix="360-city-analyser-") as directory:
            image_path = Path(directory) / "panorama.jpg"
            image_path.write_bytes(image_bytes)
            # The stage cache (BUILDING_ANALYSIS_STAGE_CACHE_DIR) is keyed by the panorama bytes, so it survives the temporary directory.
            perception = perceive_panorama(Path(directory), server.model_root, image_path, lambda _: None)
        buffer = io.BytesIO()
        __import__("numpy").savez_compressed(buffer, **perception_to_arrays(perception))
        return buffer.getvalue()

    def read_image_body(self) -> bytes:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid request body.") from error
        if not 0 < content_length <= MAX_REMOTE_IMAGE_BYTES:
            raise ValueError(f"The panorama must be smaller than {MAX_REMOTE_IMAGE_BYTES // (1024 * 1024)} MB.")
        return self.rfile.read(content_length)

    def send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8", status)

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
    print(f"Building analysis worker: http://{args.host}:{args.port}{REMOTE_PERCEPTION_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
