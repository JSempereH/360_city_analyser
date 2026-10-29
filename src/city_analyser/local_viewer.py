"""Local, dependency-free server for the downloaded panorama viewer."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import uuid
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, cast
from urllib.parse import parse_qs, unquote, urlparse

from .analysis_backend import analysis_backend_from_environment, analysis_backend_status, run_analysis
from .analysis_cache import atomic_write_json
from .attributes.assistant import Assistant, AssistantUnavailable
from .attributes.card import building_card, references
from .attributes.cli import model_settings_from_environment
from .attributes.review import attribute_summary, attributes_signature, save_review
from .building_analysis import ANALYSIS_VERSION, analysis_paths, building_mask, instance_overlay, load_analysis, nearby_buildings, osm_buildings, pick_building
from .footprint_corrections import CorrectionError, all_records, load_record, propose, revert, review
from .footprint_issues import affected_panoramas, analysis_status, building_views, footprint_issues
from .building_inspection import BuildingEvidenceNotFound, build_and_persist_report, load_report, load_report_html
from .mapillary_street_download import MAPILLARY_IMAGE_FIELDS, MAPILLARY_IMAGE_SIZES, available_image_url, download_image, fetch_image, fetch_images, image_record, load_dotenv, point_from_image, write_manifest
from .multi_view_analysis import fuse_nearby_analyses, has_valid_heading, load_nearby_analysis, nearby_analysis_path, nearby_panoramas, panorama_coordinates
from .sfm_refinement import load_sfm_refinement, refine_nearby_poses, sfm_path
from .tile_cache import TILE_PATH, tile

# Static viewer files ship inside the package.
VIEWER_ROOT = Path(__file__).resolve().parent / "viewer"


DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]*$")
FOOTPRINT_CORRECTION_ROUTE = re.compile(r"/api/footprint-corrections/((?:node|way|relation)/[1-9][0-9]*)")
BUILDING_VIEWS_ROUTE = re.compile(r"/api/datasets/([^/]+)/building-views/((?:node|way|relation)/[1-9][0-9]*(?:/part-[1-9][0-9]*)?)")
BUILDING_ATTRIBUTES_ROUTE = re.compile(r"/api/datasets/([^/]+)/building-attributes/((?:node|way|relation)/[1-9][0-9]*(?:/part-[1-9][0-9]*)?)")
ATTRIBUTE_SUMMARY_ROUTE = re.compile(r"/api/datasets/([^/]+)/building-attributes")
ASSISTANT_ROUTE = re.compile(r"/api/datasets/([^/]+)/building-attributes/((?:node|way|relation)/[1-9][0-9]*(?:/part-[1-9][0-9]*)?)/(analyze|chat)")
MAX_CHAT_REQUEST_BYTES = 65_536
# Corrected footprints are GeoJSON polygons and can be larger than other requests.
MAX_FOOTPRINT_REQUEST_BYTES = 262_144
BUILDING_REPORT_ROUTE = re.compile(
    r"/api/datasets/([^/]+)/building-reports/((?:node|way|relation)/[1-9][0-9]*(?:/part-[1-9][0-9]*)?)(/html)?"
)
# City-scale queries stay bounded while allowing a typical urban municipality.
MAX_MAPILLARY_AOI_DEG2 = 0.05
MAX_MAPILLARY_SEARCH_PAGES = 5
MAX_MAPILLARY_RESULTS = 10_000
MAX_MAPILLARY_DOWNLOADS = 10_000
MAPILLARY_DOWNLOAD_SIZES = {"auto", *MAPILLARY_IMAGE_SIZES}


def load_manifest(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid manifest: {path.name}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError(f"manifest has no image list: {path.name}")
    return payload


def dataset_summaries(data_root: Path) -> list[dict]:
    if not data_root.exists():
        return []
    datasets = []
    for manifest_path in data_root.glob("*/manifest.json"):
        try:
            manifest = load_manifest(manifest_path)
        except ValueError:
            continue
        dataset_id = manifest_path.parent.name
        if not DATASET_ID.fullmatch(dataset_id):
            continue
        raw_query = manifest.get("query")
        query: dict = raw_query if isinstance(raw_query, dict) else {}
        datasets.append(
            {
                "id": dataset_id,
                "images": len(manifest["images"]),
                "provider": query.get("provider", "Unknown provider"),
                "acquired_at": query.get("acquired_at"),
            }
        )
    return sorted(datasets, key=lambda item: (item["acquired_at"] or "", item["id"]), reverse=True)


def dataset_manifest(data_root: Path, dataset_id: str) -> dict:
    if not DATASET_ID.fullmatch(dataset_id):
        raise ValueError("invalid dataset id")
    return load_manifest(data_root / dataset_id / "manifest.json")


def delete_dataset(data_root: Path, dataset_id: str) -> None:
    """Delete one discovered dataset, never an arbitrary path under data/."""
    dataset_manifest(data_root, dataset_id)
    dataset_dir = (data_root / dataset_id).resolve()
    if dataset_dir.parent != data_root.resolve():
        raise ValueError("invalid dataset id")
    try:
        shutil.rmtree(dataset_dir)
    except OSError as error:
        raise ValueError(f"could not delete dataset: {dataset_id}") from error


def parse_mapillary_bbox(value: object) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4 or not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        raise ValueError("bbox must contain west, south, east, north coordinates")
    west, south, east, north = (float(item) for item in value)
    if not -180 <= west < east <= 180 or not -90 <= south < north <= 90:
        raise ValueError("bbox is outside WGS84 bounds")
    if (east - west) * (north - south) >= MAX_MAPILLARY_AOI_DEG2:
        raise ValueError("AOI is too large; zoom in and draw a smaller area")
    return west, south, east, north


def search_mapillary_panoramas(
    token: str,
    bbox: tuple[float, float, float, float],
    fetcher: Callable[[str, tuple[float, float, float, float], tuple[str, ...], int | None], list[dict[str, Any]]] = fetch_images,
) -> dict[str, Any]:
    # Coverage is independent of the derivative resolutions exposed by a panorama.
    images = fetcher(token, bbox, (), MAX_MAPILLARY_SEARCH_PAGES)
    panoramas = []
    for image in images:
        if image.get("is_pano") is not True:
            continue
        point = point_from_image(image)
        if point is None:
            continue
        image_id = str(image.get("id", ""))
        if not image_id:
            continue
        panoramas.append(
            {
                "id": image_id,
                "coordinates": point,
                "captured_at": image.get("captured_at"),
                "heading": image.get("computed_compass_angle"),
                "source_page": f"https://www.mapillary.com/app/?pKey={image_id}",
            }
        )
    panoramas.sort(key=lambda item: (item["captured_at"] or 0, item["id"]))
    return {
        "api_images_seen": len(images),
        "panoramas_found": len(panoramas),
        "panoramas": panoramas[:MAX_MAPILLARY_RESULTS],
        "truncated": len(panoramas) > MAX_MAPILLARY_RESULTS or len(images) >= MAX_MAPILLARY_SEARCH_PAGES * 2_000,
    }


def parse_mapillary_download(payload: object) -> tuple[tuple[float, float, float, float], str, list[str] | None, str]:
    if not isinstance(payload, dict):
        raise ValueError("invalid request body")
    bbox = parse_mapillary_bbox(payload.get("bbox"))
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or not DATASET_ID.fullmatch(dataset_id):
        raise ValueError("dataset name contains unsupported characters")
    download_all = payload.get("download_all") is True
    image_ids = None if download_all else payload.get("image_ids")
    if not download_all:
        if not isinstance(image_ids, list) or not 1 <= len(image_ids) <= MAX_MAPILLARY_DOWNLOADS:
            raise ValueError(f"select between 1 and {MAX_MAPILLARY_DOWNLOADS} panoramas")
        if not all(isinstance(image_id, str) and image_id.isdecimal() for image_id in image_ids):
            raise ValueError("invalid Mapillary panorama ids")
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("panorama ids must be unique")
    image_size = payload.get("image_size", "auto")
    if image_size not in MAPILLARY_DOWNLOAD_SIZES:
        raise ValueError("unsupported image size")
    return bbox, dataset_id, image_ids, image_size


def download_mapillary_panoramas(
    data_root: Path,
    token: str,
    bbox: tuple[float, float, float, float],
    dataset_id: str,
    image_ids: list[str] | None,
    image_size: str,
    fetcher: Callable[[str, tuple[float, float, float, float], tuple[str, ...], int | None], list[dict[str, Any]]] = fetch_images,
    progress: Callable[[str], None] | None = None,
    image_fetcher: Callable[[str, str, tuple[str, ...]], dict[str, Any]] = fetch_image,
) -> dict[str, Any]:
    output_dir = data_root / dataset_id
    if output_dir.exists():
        raise ValueError("a dataset with that name already exists")
    if progress:
        progress("Finding Mapillary panoramas in the selected area...")
    available = {
        str(image.get("id")): image
        for image in fetcher(token, bbox, MAPILLARY_IMAGE_FIELDS, None if image_ids is None else MAX_MAPILLARY_SEARCH_PAGES)
        if image.get("is_pano") is True
    }
    if image_ids is not None:
        for number, image_id in enumerate(image_ids, start=1):
            image = available.get(image_id)
            if image is not None and available_image_url(image, image_size) is not None:
                continue
            if progress:
                progress(f"Resolving selected panorama {number}/{len(image_ids)}...")
            resolved = image_fetcher(token, image_id, MAPILLARY_IMAGE_FIELDS)
            if resolved.get("is_pano") is True:
                available[image_id] = resolved
    selected = []
    requested_ids = available if image_ids is None else image_ids
    for image_id in requested_ids:
        image = available.get(image_id)
        if image is None:
            continue
        point = point_from_image(image)
        derivative = available_image_url(image, image_size)
        if point is None or derivative is None:
            continue
        image = dict(image)
        image["_selected_geometry"] = {"type": "Point", "coordinates": list(point)}
        image["_download_image_field"], image["_download_url"] = derivative
        selected.append(image)
    if not selected:
        raise ValueError("none of the selected panoramas are still available from Mapillary")
    if progress:
        progress(f"Found {len(selected)} panoramas. Preparing download...")

    records = []
    try:
        for number, image in enumerate(selected, start=1):
            image_id = str(image["id"])
            if progress:
                progress(f"Downloading panorama {number}/{len(selected)}...")
            destination = output_dir / "images" / f"pano-{image_id}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            _, local_metadata = download_image(image["_download_url"], destination, overwrite=False)
            records.append(image_record(image, f"images/{destination.name}", local_metadata))
        query = {
            "provider": "Mapillary Graph API v4",
            "acquired_at": datetime.now(UTC).isoformat(),
            "bbox_wgs84": bbox,
            "only_panoramas": True,
            "preferred_image_size": image_size,
            "image_fields_requested": MAPILLARY_IMAGE_FIELDS,
            "api_images_seen": len(available),
            "download_mode": "all_aoi_matches" if image_ids is None else "selected",
            "panoramas_requested": len(requested_ids),
            "panoramas_selected": len(records),
            "downloaded_from_viewer": True,
        }
        write_manifest(output_dir, records, query)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return {"dataset_id": dataset_id, "downloaded": len(records), "missing": len(requested_ids) - len(records)}


# Finished jobs are kept for polling, but a long session must not grow forever.
MAX_RETAINED_JOBS = 200


class JobCancelled(Exception):
    """Raised by a job's work when a cancellation was requested; the message becomes the job's."""


def _launch_job(
    server: Any, name: str, message: str, work: Callable[[Callable[..., None]], dict[str, Any]], **initial: Any,
) -> dict[str, Any]:
    """Run ``work(update)`` on a daemon thread as a pollable job.

    ``work`` returns the fields merged into the job on success; any exception
    becomes a failed job, because model, network and COLMAP errors must reach
    the UI instead of killing the thread silently.
    """
    job_id = uuid.uuid4().hex
    cancel = threading.Event()
    with server.analysis_lock:
        finished = [key for key, job in server.analysis_jobs.items() if job.get("status") in {"complete", "failed", "cancelled"}]
        for key in finished[:max(0, len(server.analysis_jobs) - MAX_RETAINED_JOBS + 1)]:
            del server.analysis_jobs[key]
            server.job_cancellations.pop(key, None)
        server.analysis_jobs[job_id] = {"id": job_id, "message": message, "status": "queued", **initial}
        server.job_cancellations[job_id] = cancel

    def update(message: str | None = None, **fields: Any) -> None:
        if message is not None:
            fields["message"] = message
        with server.analysis_lock:
            server.analysis_jobs[job_id].update(status="running", **fields)

    # Long jobs poll this between steps; see cancel_job.
    update.cancelled = cancel.is_set  # type: ignore[attr-defined]

    def run() -> None:
        try:
            result = work(update)
        except JobCancelled as cancelled:
            with server.analysis_lock:
                server.analysis_jobs[job_id].update(message=str(cancelled), status="cancelled")
        except Exception as error:  # noqa: BLE001 - reported to the polling client.
            with server.analysis_lock:
                server.analysis_jobs[job_id].update(message=str(error), status="failed")
        else:
            with server.analysis_lock:
                server.analysis_jobs[job_id].update(status="complete", **result)

    threading.Thread(target=run, daemon=True, name=f"{name}-{job_id[:8]}").start()
    return {"job": server.analysis_jobs[job_id], "status": "queued"}


class ViewerServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], workspace_root: Path, viewer_root: Path = VIEWER_ROOT):
        """Serve ``viewer_root`` and the ``data/``, ``models/`` and ``.env`` of ``workspace_root``."""
        super().__init__(address, ViewerRequestHandler)
        self.workspace_root = workspace_root.resolve()
        self.data_root = (self.workspace_root / "data").resolve()
        self.viewer_root = viewer_root.resolve()
        self.model_root = (self.workspace_root / "models").resolve()
        self.analysis_backend = analysis_backend_from_environment()
        self.analysis_jobs: dict[str, dict[str, Any]] = {}
        self.job_cancellations: dict[str, threading.Event] = {}
        self.analysis_lock = threading.Lock()
        # Small GPUs cannot safely host concurrent transformer inference jobs.
        self.analysis_run_lock = threading.Lock()
        self.report_lock = threading.Lock()
        settings = model_settings_from_environment(self.model_root)
        self.assistant = Assistant(settings) if settings else None
        self.attribute_summaries: dict[str, tuple[str, dict[str, Any]]] = {}
        self.attribute_summary_lock = threading.Lock()
        # Reviews of one building are read, merged and written as a unit.
        self.attribute_review_lock = threading.Lock()

    def has_dataset(self, dataset_id: str) -> bool:
        return bool(DATASET_ID.fullmatch(dataset_id)) and (self.data_root / dataset_id / "manifest.json").is_file()

    def images_by_id(self, dataset_id: str) -> dict[str, dict[str, Any]]:
        return {str(image["id"]): image for image in dataset_manifest(self.data_root, dataset_id)["images"] if isinstance(image, dict) and "id" in image}

    def attribute_summary(self, dataset_id: str) -> dict[str, Any]:
        """The dataset's attribute summary, rebuilt only when an attribute file changed (a survey may be writing)."""
        dataset_dir = self.data_root / dataset_id
        signature = attributes_signature(dataset_dir)
        with self.attribute_summary_lock:
            cached = self.attribute_summaries.get(dataset_id)
            if cached is None or cached[0] != signature:
                cached = (signature, attribute_summary(dataset_dir, dataset_id))
                self.attribute_summaries[dataset_id] = cached
            return cached[1]

    def image_record(self, dataset_id: str, image_id: str) -> dict[str, Any]:
        manifest = dataset_manifest(self.data_root, dataset_id)
        image = next((item for item in manifest["images"] if str(item.get("id")) == image_id), None)
        if not isinstance(image, dict) or not image.get("is_pano") or not image.get("local_file"):
            raise ValueError("panorama not found")
        return image

    def start_analysis(self, dataset_id: str, image_id: str) -> dict[str, Any]:
        image = self.image_record(dataset_id, image_id)
        if not has_valid_heading(image):
            raise ValueError("The panorama has no valid compass heading.")
        existing = load_analysis(self.data_root, dataset_id, image_id)
        if existing:
            return {"analysis": existing, "status": "complete"}

        def work(update: Callable[..., None]) -> dict[str, Any]:
            with self.analysis_run_lock:
                analysis = load_analysis(self.data_root, dataset_id, image_id)
                analysis = analysis or run_analysis(self.analysis_backend, self.data_root, self.model_root, dataset_id, image, update)
            return {"analysis": analysis, "message": "Building analysis complete."}

        return _launch_job(self, "building-analysis", f"Queued {self.analysis_backend.label} building analysis.", work)

    def propose_footprint_correction(self, osm_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        geometry = payload.get("geometry")
        parts = payload.get("parts")
        try:
            shape = __import__("shapely.geometry", fromlist=["shape"]).shape
            if isinstance(parts, list) and parts:
                centroid = __import__("shapely.ops", fromlist=["unary_union"]).unary_union([shape(part) for part in parts]).centroid
            else:
                centroid = shape(geometry).centroid
        except (TypeError, ValueError, KeyError, AttributeError) as error:
            raise CorrectionError("geometry must be a GeoJSON polygon") from error
        # The correction is tied to the element exactly as OSM serves it now.
        base = next((feature for feature in osm_buildings(centroid.y, centroid.x) if feature["properties"]["osm_id"] == osm_id), None)
        if base is None:
            raise CorrectionError(f"{osm_id} is not an OpenStreetMap building near the proposed footprint")
        evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
        return {"revision": propose(base, geometry, "manual", str(payload.get("author", "")), str(payload.get("note", "")), evidence, payload.get("parts"))}

    def review_footprint_correction(self, osm_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            revision_number = int(payload["revision"])
        except (KeyError, TypeError, ValueError) as error:
            raise CorrectionError("revision must be an integer") from error
        revision = review(osm_id, revision_number, str(payload.get("decision", "")), str(payload.get("reviewer", "")), str(payload.get("note", "")))
        result: dict[str, Any] = {"revision": revision}
        if revision["status"] == "accepted":
            # A split has no single geometry; its parts together locate it.
            changed = revision["geometry"] or {"type": "GeometryCollection", "geometries": revision.get("parts", [])}
            result.update(self.start_reassociation(changed))
        return result

    def revert_footprint_correction(self, osm_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        revision = revert(osm_id, str(payload.get("reviewer", "")), str(payload.get("note", "")))
        changed = revision["geometry"] or {"type": "GeometryCollection", "geometries": revision.get("parts", [])}
        return {"revision": revision, **self.start_reassociation(changed)}

    def start_reassociation(self, geometry: dict[str, Any]) -> dict[str, Any]:
        """Re-associate analyzed panoramas that load a changed footprint."""
        targets = [
            (dataset_id, image) for dataset_id, image in affected_panoramas(self.data_root, geometry)
            if has_valid_heading(image) and analysis_paths(self.data_root, dataset_id, str(image["id"]))[0].is_file()
        ]
        def work(update: Callable[..., None]) -> dict[str, Any]:
            for number, (dataset_id, image) in enumerate(targets, start=1):
                update(f"Re-associating panorama {number}/{len(targets)} with the corrected footprint...")
                with self.analysis_run_lock:
                    run_analysis(self.analysis_backend, self.data_root, self.model_root, dataset_id, image, lambda _: None)
            return {"message": f"Re-associated {len(targets)} panoramas.", "reassociated": len(targets)}

        return {"reassociation": _launch_job(self, "footprint-reassociation", f"Queued re-association of {len(targets)} panoramas.", work)}

    def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        """Ask a cancellable job to stop after its current step."""
        with self.analysis_lock:
            job = self.analysis_jobs.get(job_id)
            if job is None:
                return None
            if job.get("cancellable") and job.get("status") in {"queued", "running"}:
                self.job_cancellations[job_id].set()
                job["message"] = "Cancelling after the current panorama..."
            return dict(job)

    def start_batch_analysis(self, dataset_id: str, bbox: list[float] | None, force: bool) -> dict[str, Any]:
        """Analyze every panorama of a dataset, or those inside ``bbox`` (west, south, east, north)."""
        manifest = dataset_manifest(self.data_root, dataset_id)
        if bbox is not None:
            if len(bbox) != 4 or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox) or bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                raise ValueError("bbox must be [west, south, east, north]")
        images = []
        for image in manifest["images"]:
            if not isinstance(image, dict) or not image.get("is_pano") or not image.get("local_file") or not has_valid_heading(image):
                continue
            coordinates = panorama_coordinates(image)
            if bbox is not None and (coordinates is None or not (bbox[0] <= coordinates[0] <= bbox[2] and bbox[1] <= coordinates[1] <= bbox[3])):
                continue
            if not force and load_analysis(self.data_root, dataset_id, str(image["id"]), image) is not None:
                continue
            images.append(image)

        def work(update: Any) -> dict[str, Any]:
            for number, image in enumerate(images, start=1):
                if update.cancelled():
                    raise JobCancelled(f"Cancelled after {number - 1} of {len(images)} panoramas.")
                update(f"Analyzing panorama {number} of {len(images)}...", done=number - 1, current=str(image["id"]))
                with self.analysis_run_lock:
                    run_analysis(self.analysis_backend, self.data_root, self.model_root, dataset_id, image, lambda _: None)
            return {"message": f"Analyzed {len(images)} panorama{'s' if len(images) != 1 else ''}.", "done": len(images)}

        scope = "in the map view" if bbox is not None else "in the dataset"
        return _launch_job(
            self, "batch-analysis", f"Queued analysis of {len(images)} panoramas {scope}.", work,
            total=len(images), done=0, cancellable=True, dataset=dataset_id,
        )

    def start_nearby_analysis(self, dataset_id: str, image_id: str) -> dict[str, Any]:
        current_image = self.image_record(dataset_id, image_id)
        if not has_valid_heading(current_image):
            raise ValueError("The current panorama has no valid compass heading.")
        existing = load_nearby_analysis(self.data_root, dataset_id, image_id)
        if existing:
            return {"analysis": existing, "status": "complete"}
        manifest = dataset_manifest(self.data_root, dataset_id)
        images = [
            image for image in manifest["images"]
            if isinstance(image, dict) and image.get("is_pano") and image.get("local_file") and has_valid_heading(image)
        ]
        selected_images = nearby_panoramas(images, current_image)

        def work(update: Callable[..., None]) -> dict[str, Any]:
            analyzed = []
            for number, (candidate, distance) in enumerate(selected_images, start=1):
                candidate_id = str(candidate["id"])
                prefix = f"Nearby panorama {number}/{len(selected_images)} at {distance:.0f} m"
                update(f"{prefix}: checking analysis cache...")
                with self.analysis_run_lock:
                    analysis = load_analysis(self.data_root, dataset_id, candidate_id)
                    if analysis is None:
                        update(f"{prefix}: running analysis...")
                        analysis = run_analysis(self.analysis_backend, self.data_root, self.model_root, dataset_id, candidate, update)
                analyzed.append((candidate, distance, analysis))
                if candidate_id == image_id:
                    update(current_analysis_complete=True)
            metadata = fuse_nearby_analyses(image_id, analyzed)
            atomic_write_json(nearby_analysis_path(self.data_root, dataset_id, image_id), metadata)
            return {"analysis": metadata, "message": f"Nearby evidence complete from {len(analyzed)} panoramas."}

        return _launch_job(
            self, "nearby-building-analysis",
            f"Queued {len(selected_images)} panoramas for {self.analysis_backend.label} analysis. You can keep browsing.",
            work, current_analysis_complete=False,
        )

    def start_sfm_refinement(self, dataset_id: str, image_id: str) -> dict[str, Any]:
        current_image = self.image_record(dataset_id, image_id)
        if not has_valid_heading(current_image):
            raise ValueError("The current panorama has no valid compass heading.")
        existing = load_sfm_refinement(self.data_root, dataset_id, image_id)
        if existing:
            return {"analysis": existing, "status": "complete"}
        manifest = dataset_manifest(self.data_root, dataset_id)
        images = [
            image for image in manifest["images"]
            if isinstance(image, dict) and image.get("is_pano") and image.get("local_file") and has_valid_heading(image)
        ]

        def work(update: Callable[..., None]) -> dict[str, Any]:
            # COLMAP optionally uses CUDA too, so it shares the small-GPU lock.
            with self.analysis_run_lock:
                refinement = load_sfm_refinement(self.data_root, dataset_id, image_id)
                if refinement is None:
                    refinement = refine_nearby_poses(self.data_root, dataset_id, current_image, images, update)
                    atomic_write_json(sfm_path(self.data_root, dataset_id, image_id), refinement)
            accepted = sum(pose["accepted"] for pose in refinement["poses"].values())
            return {"analysis": refinement, "message": f"COLMAP refined {accepted} nearby panorama poses."}

        return _launch_job(self, "nearby-pose-refinement", "Queued nearby-pose refinement with COLMAP.", work)

    def start_mapillary_download(
        self, bbox: tuple[float, float, float, float], dataset_id: str, image_ids: list[str] | None, image_size: str, token: str,
    ) -> dict[str, Any]:
        if (self.data_root / dataset_id).exists():
            raise ValueError("a dataset with that name already exists")

        def work(update: Callable[..., None]) -> dict[str, Any]:
            result = download_mapillary_panoramas(self.data_root, token, bbox, dataset_id, image_ids, image_size, progress=update)
            message = f"Mapillary download complete: {result['downloaded']} panoramas."
            if result["missing"]:
                message += f" {result['missing']} selected panoramas were unavailable."
            return {"result": result, "message": message}

        return _launch_job(self, "mapillary-download", "Queued Mapillary download.", work)

    def analysis_job(self, job_id: str) -> dict[str, Any] | None:
        with self.analysis_lock:
            job = self.analysis_jobs.get(job_id)
            return dict(job) if job else None


class ViewerRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        server = cast(ViewerServer, self.server)
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/api/buildings":
            try:
                latitude = float(query["lat"][0])
                longitude = float(query["lon"][0])
                if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                    raise ValueError
            except (KeyError, ValueError):
                self.send_json({"error": "lat and lon must be valid WGS84 coordinates"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                self.send_json({"features": nearby_buildings(latitude, longitude)})
            except (OSError, RuntimeError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_GATEWAY)
            return
        if path == "/api/footprint-issues":
            dataset_id = query.get("dataset", [None])[0]
            if dataset_id is not None and not DATASET_ID.fullmatch(dataset_id):
                self.send_json({"error": "invalid dataset id"}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json(footprint_issues(server.data_root, dataset_id))
            return
        if path == "/api/footprint-corrections":
            status = query.get("status", [None])[0]
            records = [
                record for record in all_records()
                if status is None or any(revision.get("status") == status for revision in record["revisions"])
            ]
            self.send_json({"corrections": records})
            return
        status_match = re.fullmatch(r"/api/datasets/([^/]+)/analysis-status", path)
        if status_match:
            dataset_id = unquote(status_match.group(1))
            if not server.has_dataset(dataset_id):
                self.send_json({"error": "dataset not found"}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json({"analysis_version": ANALYSIS_VERSION, "panoramas": analysis_status(server.data_root, dataset_id)})
            return
        views_match = BUILDING_VIEWS_ROUTE.fullmatch(path)
        if views_match:
            dataset_id = unquote(views_match.group(1))
            if not server.has_dataset(dataset_id):
                self.send_json({"error": "dataset not found"}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json(building_views(server.data_root, dataset_id, views_match.group(2)))
            return
        summary_match = ATTRIBUTE_SUMMARY_ROUTE.fullmatch(path)
        if summary_match:
            dataset_id = unquote(summary_match.group(1))
            if not server.has_dataset(dataset_id):
                self.send_json({"error": "dataset not found"}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json(server.attribute_summary(dataset_id))
            return
        attributes_match = BUILDING_ATTRIBUTES_ROUTE.fullmatch(path)
        if attributes_match:
            dataset_id = unquote(attributes_match.group(1))
            if not server.has_dataset(dataset_id):
                self.send_json({"error": "dataset not found"}, HTTPStatus.NOT_FOUND)
                return
            # Reference data (the cadastre) only in evaluation mode (?references=1): the analysis never uses it.
            with_references = query.get("references", [""])[0] == "1"
            card = building_card(server.data_root / dataset_id, dataset_id, attributes_match.group(2), with_references=with_references)
            if card is None:
                # Expected until a survey or city-analyser-attributes reaches this building; reference data may exist already.
                self.send_json({"error": "no attributes for this building yet",
                                **({"references": references(server.data_root / dataset_id, attributes_match.group(2))} if with_references else {})},
                               HTTPStatus.NOT_FOUND)
            else:
                self.send_json(card)
            return
        correction_match = FOOTPRINT_CORRECTION_ROUTE.fullmatch(path)
        if correction_match:
            record = load_record(correction_match.group(1))
            if record is None:
                self.send_json({"error": "no corrections for this building"}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json(record)
            return
        if path.startswith("/api/jobs/"):
            job = server.analysis_job(unquote(path.removeprefix("/api/jobs/")))
            if not job:
                self.send_error(HTTPStatus.NOT_FOUND, "Analysis job not found")
            else:
                self.send_json(job)
            return
        report_match = BUILDING_REPORT_ROUTE.fullmatch(path)
        if report_match:
            dataset_id, building_id, html_suffix = report_match.groups()
            dataset_id = unquote(dataset_id)
            try:
                dataset_manifest(server.data_root, dataset_id)
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND, "Dataset not found")
                return
            if html_suffix:
                report_html = load_report_html(server.data_root / dataset_id, building_id)
                if report_html is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "Building report not found")
                else:
                    self.send_bytes(report_html.encode("utf-8"), "text/html; charset=utf-8")
                return
            report = load_report(server.data_root / dataset_id, building_id)
            if report is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Building report not found")
            else:
                self.send_json(report)
            return
        analysis_match = re.fullmatch(r"/api/datasets/([^/]+)/analysis/([^/]+)(?:/(mask|nearby|sfm))?", path)
        if analysis_match:
            dataset_id, image_id, action = (unquote(value) if value else value for value in analysis_match.groups())
            try:
                server.image_record(dataset_id, image_id)
                if action == "mask":
                    osm_id = query.get("building_id", [""])[0]
                    layer = query.get("layer", [""])[0]
                    mask = (
                        instance_overlay(server.data_root, dataset_id, image_id) if layer == "instances"
                        else building_mask(server.data_root, dataset_id, image_id, osm_id, layer == "inferred")
                    )
                    if mask is None:
                        self.send_error(HTTPStatus.NOT_FOUND, "Building mask not found")
                    else:
                        self.send_bytes(mask, "image/png")
                elif action == "nearby":
                    analysis = load_nearby_analysis(server.data_root, dataset_id, image_id)
                    if analysis is None:
                        self.send_json({"error": "Nearby building analysis not found"}, HTTPStatus.NOT_FOUND)
                    else:
                        self.send_json(analysis)
                elif action == "sfm":
                    analysis = load_sfm_refinement(server.data_root, dataset_id, image_id)
                    if analysis is None:
                        # SfM is optional and usually not built: "none yet" is not an error.
                        self.send_response(HTTPStatus.NO_CONTENT)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                    else:
                        self.send_json(analysis)
                else:
                    analysis = load_analysis(server.data_root, dataset_id, image_id)
                    if analysis is None:
                        # Missing analysis is expected until the user starts it.
                        self.send_json({"error": "Building analysis not found"}, HTTPStatus.NOT_FOUND)
                    else:
                        self.send_json(analysis)
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND, "Panorama not found")
            except RuntimeError as error:
                self.send_json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if path == "/api/datasets":
            self.send_json({"datasets": dataset_summaries(server.data_root)})
            return
        if path.startswith("/api/datasets/"):
            dataset_id = unquote(path.removeprefix("/api/datasets/"))
            try:
                self.send_json(dataset_manifest(server.data_root, dataset_id))
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND, "Dataset not found")
            return
        if path == "/api/health":
            self.send_json({"status": "ok", "time": datetime.now(UTC).isoformat(), "analysis_backend": analysis_backend_status(server.analysis_backend)})
            return
        if path == "/api/assistant":
            self.send_json(server.assistant.status() if server.assistant else {
                "available": False,
                "help": "Start the viewer with ATTRIBUTES_VLM_URL and ATTRIBUTES_VLM_MODEL (an OpenAI-compatible server such as Ollama or vLLM), "
                        "or ATTRIBUTES_VLM_BACKEND=transformers, to analyze and chat in real time.",
            })
            return
        if path == "/api/analysis-backend":
            self.send_json(analysis_backend_status(server.analysis_backend))
            return
        if path == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/viewer/")
            self.end_headers()
            return
        if path == "/viewer":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/viewer/")
            self.end_headers()
            return
        if path == "/viewer/":
            self.serve_file(server.viewer_root, "index.html", cache_control="no-cache")
            return
        tile_match = TILE_PATH.fullmatch(path)
        if tile_match:
            content = tile(server.data_root, *(int(part) for part in tile_match.groups()))
            if content is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Tile not cached and not reachable")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "max-age=86400")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if path.startswith("/viewer/"):
            self.serve_file(server.viewer_root, unquote(path.removeprefix("/viewer/")), cache_control="no-cache")
            return
        if path.startswith("/data/"):
            relative_path = unquote(path.removeprefix("/data/"))
            candidate = (server.data_root / relative_path).resolve()
            if candidate.is_relative_to(server.data_root):
                parts = candidate.relative_to(server.data_root).parts
                if len(parts) >= 3 and parts[1:3] == ("reports", "buildings"):
                    # Screenshots of an OSM element, or of a part of a reviewed split one level deeper.
                    part = len(parts) == 7 and re.fullmatch(r"part-[1-9][0-9]*", parts[5]) is not None
                    allowed_screenshot = bool(
                        (len(parts) == 6 or part)
                        and parts[3] in {"node", "way", "relation"}
                        and parts[4].isdecimal()
                        and parts[-1].startswith("view-")
                        and candidate.suffix.lower() == ".jpg"
                    )
                    if not allowed_screenshot:
                        self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                        return
            self.serve_file(server.data_root, relative_path, cache_control="private, max-age=3600")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_DELETE(self) -> None:  # noqa: N802
        server = cast(ViewerServer, self.server)
        path = urlparse(self.path).path
        if not path.startswith("/api/datasets/"):
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        dataset_id = unquote(path.removeprefix("/api/datasets/"))
        try:
            delete_dataset(server.data_root, dataset_id)
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND, "Dataset not found")
            return
        self.send_json({"deleted": dataset_id})

    def do_POST(self) -> None:  # noqa: N802
        server = cast(ViewerServer, self.server)
        path = urlparse(self.path).path
        cancel_match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/cancel", path)
        if cancel_match:
            job = server.cancel_job(cancel_match.group(1))
            if job is None:
                self.send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json({"job": job}, HTTPStatus.ACCEPTED)
            return
        batch_match = re.fullmatch(r"/api/datasets/([^/]+)/analysis-batch", path)
        if batch_match:
            try:
                payload = self.read_json_body() if int(self.headers.get("Content-Length", "0")) else {}
                result = server.start_batch_analysis(unquote(batch_match.group(1)), payload.get("bbox"), bool(payload.get("force")))
            except ValueError as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            else:
                self.send_json(result, HTTPStatus.ACCEPTED)
            return
        assistant_match = ASSISTANT_ROUTE.fullmatch(path)
        if assistant_match:
            self.assistant_request(server, unquote(assistant_match.group(1)), assistant_match.group(2), assistant_match.group(3))
            return
        attributes_match = BUILDING_ATTRIBUTES_ROUTE.fullmatch(path)
        if attributes_match:
            dataset_id = unquote(attributes_match.group(1))
            if not server.has_dataset(dataset_id):
                self.send_json({"error": "dataset not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                payload = self.read_json_body()
                with server.attribute_review_lock:
                    card = save_review(server.data_root / dataset_id, dataset_id, attributes_match.group(2), payload)
            except ValueError as error:  # ReviewError, or a malformed body
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            else:
                if parse_qs(urlparse(self.path).query).get("references", [""])[0] == "1":
                    card["references"] = references(server.data_root / dataset_id, attributes_match.group(2))
                self.send_json(card)
            return
        correction_match = FOOTPRINT_CORRECTION_ROUTE.fullmatch(path)
        if correction_match:
            try:
                payload = self.read_json_body(MAX_FOOTPRINT_REQUEST_BYTES)
                action = payload.get("action")
                if action == "propose":
                    result = server.propose_footprint_correction(correction_match.group(1), payload)
                elif action == "review":
                    result = server.review_footprint_correction(correction_match.group(1), payload)
                elif action == "revert":
                    result = server.revert_footprint_correction(correction_match.group(1), payload)
                else:
                    raise CorrectionError("action must be propose, review or revert")
            except (CorrectionError, ValueError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except (OSError, RuntimeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_GATEWAY)
            else:
                self.send_json(result, HTTPStatus.CREATED if action == "propose" else HTTPStatus.OK)
            return
        report_match = BUILDING_REPORT_ROUTE.fullmatch(path)
        if report_match and not report_match.group(3):
            dataset_id = unquote(report_match.group(1))
            building_id = report_match.group(2)
            try:
                manifest = dataset_manifest(server.data_root, dataset_id)
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND, "Dataset not found")
                return
            try:
                # Analysis publishes multiple related artifacts; hold its lock so
                # reports never combine metadata and masks from different runs.
                with server.analysis_run_lock, server.report_lock:
                    report = build_and_persist_report(
                        server.data_root,
                        dataset_id,
                        manifest,
                        building_id,
                        expected_analysis_version=ANALYSIS_VERSION,
                    )
            except BuildingEvidenceNotFound as error:
                self.send_json({"error": str(error)}, HTTPStatus.UNPROCESSABLE_ENTITY)
            except (OSError, ValueError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            else:
                self.send_json({"report": report, "links": report["links"]}, HTTPStatus.CREATED)
            return
        analysis_match = re.fullmatch(r"/api/datasets/([^/]+)/analysis/([^/]+)(?:/(pick|nearby|sfm))?", path)
        if analysis_match:
            dataset_id, image_id, action = (unquote(value) if value else value for value in analysis_match.groups())
            try:
                if action == "pick":
                    payload = self.read_json_body()
                    u_value, v_value = payload.get("u"), payload.get("v")
                    if not isinstance(u_value, (int, float)) or not isinstance(v_value, (int, float)):
                        raise ValueError("u and v must be numbers")
                    server.image_record(dataset_id, image_id)
                    u, v = float(u_value), float(v_value)
                    building = pick_building(server.data_root, dataset_id, image_id, u, v)
                    self.send_json({"building": building})
                elif action == "nearby":
                    result = server.start_nearby_analysis(dataset_id, image_id)
                    self.send_json(result, HTTPStatus.OK if result["status"] == "complete" else HTTPStatus.ACCEPTED)
                elif action == "sfm":
                    result = server.start_sfm_refinement(dataset_id, image_id)
                    self.send_json(result, HTTPStatus.OK if result["status"] == "complete" else HTTPStatus.ACCEPTED)
                else:
                    result = server.start_analysis(dataset_id, image_id)
                    self.send_json(result, HTTPStatus.OK if result["status"] == "complete" else HTTPStatus.ACCEPTED)
            except (TypeError, ValueError):
                self.send_json({"error": "Invalid building analysis request."}, HTTPStatus.BAD_REQUEST)
            return
        if path not in {"/api/mapillary/search", "/api/mapillary/download"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            payload = self.read_json_body()
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        load_dotenv(server.workspace_root / ".env")
        token = os.environ.get("MAPILLARY_ACCESS_TOKEN")
        if not token:
            self.send_json({"error": "MAPILLARY_ACCESS_TOKEN is not configured on this server."}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if path == "/api/mapillary/download":
            try:
                bbox, dataset_id, image_ids, image_size = parse_mapillary_download(payload)
                self.send_json(server.start_mapillary_download(bbox, dataset_id, image_ids, image_size, token), HTTPStatus.ACCEPTED)
            except ValueError as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError:
                self.send_json({"error": "Mapillary could not complete the download. Try again later."}, HTTPStatus.BAD_GATEWAY)
            return
        try:
            bbox = parse_mapillary_bbox(payload.get("bbox"))
        except ValueError as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        try:
            self.send_json(search_mapillary_panoramas(token, bbox))
        except RuntimeError as error:
            # request_json already strips the upstream body to a short API error;
            # returning it makes rate limits and invalid credentials actionable.
            self.send_json({"error": f"Mapillary search failed: {error}"}, HTTPStatus.BAD_GATEWAY)

    def assistant_request(self, server: ViewerServer, dataset_id: str, building_id: str, action: str) -> None:
        """Analyze or chat about one building, streaming progress as server-sent events (one JSON object per event)."""
        if not server.has_dataset(dataset_id):
            self.send_json({"error": "dataset not found"}, HTTPStatus.NOT_FOUND)
            return
        if server.assistant is None:
            self.send_json({"error": "no model is configured for the assistant"}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        try:
            payload = self.read_json_body(MAX_CHAT_REQUEST_BYTES) if int(self.headers.get("Content-Length", "0")) else {}
        except ValueError as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True

        def emit(event: dict[str, Any]) -> None:
            self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        images = server.images_by_id(dataset_id)
        try:
            if action == "analyze":
                card = server.assistant.analyze(server.data_root, dataset_id, building_id, images, emit)
                emit({"type": "done", "card": card})
            else:
                text = server.assistant.chat(server.data_root, dataset_id, building_id, payload.get("messages") or [], images,
                                             lambda chunk: emit({"type": "text", "text": chunk}))
                emit({"type": "done", "text": text})
        except (BrokenPipeError, ConnectionResetError):
            return  # the viewer went away; the analysis that finished is saved
        except (AssistantUnavailable, RuntimeError, ValueError, OSError) as error:
            try:
                emit({"type": "error", "error": str(error)})
            except OSError:
                pass

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self, limit: int = 16_384) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if not 0 < content_length <= limit:
            raise ValueError("invalid request body")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid request body")
        return payload

    def serve_file(self, root: Path, relative_path: str, cache_control: str) -> None:
        candidate = (root / relative_path).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        mime_type, _ = mimetypes.guess_type(candidate.name)
        content_type = mime_type or "application/octet-stream"
        try:
            content_size = candidate.stat().st_size
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache_control)
        self.send_header("Content-Length", str(content_size))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        with candidate.open("rb") as source:
            shutil.copyfileobj(source, self.wfile, length=1024 * 1024)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local panorama dataset viewer. Datasets, models and .env are read from the working directory.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    parser.add_argument("--port", default=8765, type=int, help="HTTP port (default: 8765)")
    args = parser.parse_args()
    try:
        server = ViewerServer((args.host, args.port), Path.cwd())
    except ValueError as error:
        parser.error(str(error))
    interrupted = False
    print(f"Panorama viewer: http://{args.host}:{args.port}/viewer/")
    print("Press Ctrl+C to stop the local server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        interrupted = True
        print("\nServer stopped.")
    finally:
        server.server_close()
    if interrupted:
        # CUDA/PyTorch can segfault in CPython's extension-module teardown after
        # inference has run. The OS still releases the closed server and GPU
        # resources, while bypassing the unstable native destructor path.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
