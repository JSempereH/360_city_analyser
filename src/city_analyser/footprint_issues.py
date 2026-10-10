"""Review queue built from the footprint diagnostics of current panorama analyses.

Each analysis records what its footprints failed to explain (see
``building_analysis.associate_panorama``). This module aggregates those
diagnostics per OSM building and per dataset, so the most strongly supported
problems are reviewed first, and finds the panoramas a footprint correction
affects. Diagnostics are suggestions for a person to review, never edits.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator

from .building_analysis import ANALYSIS_VERSION, OSM_QUERY_RADIUS_M
from .footprint_corrections import MAXIMUM_CENTROID_SHIFT_M, load_record
from .multi_view_analysis import distance_meters, panorama_coordinates

# A dataset-wide offset needs several panoramas moved the same way: a
# consistent shift points at misregistered footprints rather than GPS noise.
SYSTEMATIC_OFFSET_MIN_PANORAMAS = 3
SYSTEMATIC_OFFSET_MIN_CONSISTENCY = 0.7
SYSTEMATIC_OFFSET_MIN_M = 2.0
# A footprint matters to every camera that loads it for association.
AFFECTED_RADIUS_M = OSM_QUERY_RADIUS_M + MAXIMUM_CENTROID_SHIFT_M


def _manifest_images(dataset_dir: Path) -> list[dict[str, Any]]:
    try:
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    images = manifest.get("images") if isinstance(manifest, dict) else None
    return [image for image in images if isinstance(image, dict) and "id" in image] if isinstance(images, list) else []


def _datasets(data_root: Path, dataset_id: str | None) -> list[Path]:
    if dataset_id is not None:
        return [data_root / dataset_id]
    return sorted(path.parent for path in data_root.glob("*/manifest.json"))


def current_analyses(data_root: Path, dataset_id: str | None = None) -> Iterator[tuple[str, dict[str, Any], dict[str, Any]]]:
    """``(dataset id, manifest image, analysis metadata)`` for every current analysis."""
    for dataset_dir in _datasets(data_root, dataset_id):
        for image in _manifest_images(dataset_dir):
            path = dataset_dir / "analysis" / f"{image['id']}.json"
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(metadata, dict) and metadata.get("version") == ANALYSIS_VERSION:
                yield dataset_dir.name, image, metadata


def _correction_status(osm_id: str) -> dict[str, Any] | None:
    # Parts of a split ("way/1/part-2") are corrected through their OSM element.
    record = load_record(osm_id.split("/part-")[0])
    if record is None or not record["revisions"]:
        return None
    latest = record["revisions"][-1]
    return {"revision": latest["revision"], "status": latest["status"], "source": latest["source"]}


def _systematic_offset(dataset_id: str, offsets: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(offsets) < SYSTEMATIC_OFFSET_MIN_PANORAMAS:
        return None
    east = sum(offset["east_m"] for offset in offsets) / len(offsets)
    north = sum(offset["north_m"] for offset in offsets) / len(offsets)
    mean_length = sum(math.hypot(offset["east_m"], offset["north_m"]) for offset in offsets) / len(offsets)
    # Length of the mean vector over the mean length: 1 when every shift points the same way.
    consistency = math.hypot(east, north) / mean_length if mean_length else 0.0
    if consistency < SYSTEMATIC_OFFSET_MIN_CONSISTENCY or math.hypot(east, north) < SYSTEMATIC_OFFSET_MIN_M:
        return None
    return {
        "dataset": dataset_id, "panoramas": len(offsets), "mean_east_m": round(east, 2), "mean_north_m": round(north, 2),
        "consistency": round(consistency, 3),
    }


def footprint_issues(data_root: Path, dataset_id: str | None = None) -> dict[str, Any]:
    """Diagnostics grouped per building, unexplained instances, and systematic offsets."""
    buildings: dict[str, dict[str, Any]] = {}
    unexplained = []
    offsets: dict[str, list[dict[str, Any]]] = {}
    for dataset, image, metadata in current_analyses(data_root, dataset_id):
        coordinates = panorama_coordinates(image)
        reference = {"dataset": dataset, "image": str(image["id"])}
        for diagnostic in metadata.get("footprint_diagnostics", []):
            kind = diagnostic.get("type")
            if kind in {"unsupported_footprint", "merged_footprint"}:
                entry = buildings.setdefault(diagnostic["osm_id"], {"osm_id": diagnostic["osm_id"], "issues": {}, "panoramas": []})
                entry["issues"][kind] = entry["issues"].get(kind, 0) + 1
                entry["panoramas"].append({**reference, "type": kind})
            elif kind == "unexplained_instance" and coordinates is not None:
                unexplained.append({
                    **reference, "longitude": coordinates[0], "latitude": coordinates[1],
                    "bearings": diagnostic["bearings"], "evidence_pixels": diagnostic["evidence_pixels"],
                })
            elif kind == "pose_offset" and diagnostic.get("pose_accepted"):
                offsets.setdefault(dataset, []).append(diagnostic)
    ranked = sorted(buildings.values(), key=lambda entry: (-len({(item["dataset"], item["image"]) for item in entry["panoramas"]}), entry["osm_id"]))
    for entry in ranked:
        entry["supporting_panoramas"] = len({(item["dataset"], item["image"]) for item in entry["panoramas"]})
        entry["correction"] = _correction_status(entry["osm_id"])
    systematic = [offset for dataset, values in sorted(offsets.items()) if (offset := _systematic_offset(dataset, values)) is not None]
    return {
        "analysis_version": ANALYSIS_VERSION,
        "buildings": ranked,
        "unexplained_instances": sorted(unexplained, key=lambda item: -item["evidence_pixels"]),
        "systematic_offsets": systematic,
    }


def analysis_status(data_root: Path, dataset_id: str) -> dict[str, str]:
    """``current``, ``stale`` (older analysis version) or ``missing`` for every panorama."""
    statuses = {}
    for image in _manifest_images(data_root / dataset_id):
        path = data_root / dataset_id / "analysis" / f"{image['id']}.json"
        try:
            version = json.loads(path.read_text(encoding="utf-8")).get("version")
        except (OSError, json.JSONDecodeError, AttributeError):
            statuses[str(image["id"])] = "missing"
            continue
        statuses[str(image["id"])] = "current" if version == ANALYSIS_VERSION else "stale"
    return statuses


def building_views(data_root: Path, dataset_id: str, osm_id: str) -> dict[str, Any]:
    """Panoramas of one dataset whose current analysis shows ``osm_id``, best evidence first."""
    views, geometry = [], None
    for _, image, metadata in current_analyses(data_root, dataset_id):
        for building in (metadata.get("building_ids") or {}).values():
            if not isinstance(building, dict) or building.get("osm_id") != osm_id:
                continue
            views.append({
                "image": str(image["id"]), "pixels": int(building.get("pixels", 0)),
                "confidence": float(building.get("confidence", 0.0)), "facades": int(building.get("facades", 0)),
            })
            geometry = geometry or building.get("footprint_geometry")
    views.sort(key=lambda view: (-view["pixels"], view["image"]))
    return {"osm_id": osm_id, "views": views, "geometry": geometry}


def affected_panoramas(data_root: Path, geometry: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Panoramas whose association loads a footprint near ``geometry``."""
    centroid = __import__("shapely.geometry", fromlist=["shape"]).shape(geometry).centroid
    affected = []
    for dataset_dir in _datasets(data_root, None):
        for image in _manifest_images(dataset_dir):
            coordinates = panorama_coordinates(image)
            if coordinates is not None and image.get("local_file") and distance_meters(coordinates, (centroid.x, centroid.y)) <= AFFECTED_RADIUS_M:
                affected.append((dataset_dir.name, image))
    return affected
