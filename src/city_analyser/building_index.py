"""Which panoramas see which building, without running any model.

Analysis is panorama first; attribute work is building first. This index
inverts the relation geometrically (an isovist, as in OpenFACADES): from each
camera position the unobstructed wall intervals of every OSM footprint give
that building's angle of view (AOV), distance and visible wall length. It costs
milliseconds per panorama, so it can decide which panoramas deserve GPU time.
It uses the recorded GPS position, not the refined pose, and ignores trees and
vehicles, so it over-promises visibility; the analysis remains the evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

from .analysis_cache import atomic_write_json, canonical_json_sha256
from .building_analysis import GPS_REPAIR_CLEARANCE_M, GPS_REPAIR_MAXIMUM_SHIFT_M, _is_overhead, nearby_buildings, panorama_inputs
from .facade_geometry import build_scene, free_space_offset, visible_facades

INDEX_VERSION = 2
INDEX_FILENAME = "building-index.json"
# OpenFACADES keeps views that subtend 10 to 120 degrees: smaller ones show too
# little, larger ones are too close and distorted.
DEFAULT_MIN_AOV_DEGREES = 10.0
DEFAULT_MAX_AOV_DEGREES = 120.0
DEFAULT_MAX_VIEWS_PER_BUILDING = 5
DEFAULT_MAX_DISTANCE_M = 50.0


def index_path(data_root: Path, dataset_id: str) -> Path:
    return data_root / dataset_id / INDEX_FILENAME


def _bearing_degrees(point: Any) -> float:
    return math.degrees(math.atan2(float(point[0]), float(point[1]))) % 360.0


def _interval_width(start: float, end: float) -> float:
    return abs((end - start + 180.0) % 360.0 - 180.0)


def panorama_visibility(features: list[dict[str, Any]], latitude: float, longitude: float) -> dict[str, dict[str, float]]:
    """Per OSM building: AOV, nearest visible wall distance, visible wall length and the bearing of its widest wall."""
    scene = build_scene(features, latitude, longitude)
    # Same repair as pose refinement: a GPS fix recorded inside a footprint sees
    # no wall at all until it is moved back to the street.
    blocking = __import__("numpy").array([not _is_overhead(feature.get("properties", {})) for feature in features], dtype=bool)
    repair = free_space_offset(scene, blocking, GPS_REPAIR_CLEARANCE_M, GPS_REPAIR_MAXIMUM_SHIFT_M)
    if repair is not None:
        scene = scene.translated(*repair)
    buildings: dict[str, dict[str, float]] = {}
    for facade in visible_facades(scene):
        osm_id = str(scene.features[facade.feature_index].get("properties", {}).get("osm_id", ""))
        if not osm_id:
            continue
        first, second = _bearing_degrees(facade.start), _bearing_degrees(facade.end)
        width = _interval_width(first, second)
        length = float(math.hypot(*(facade.end - facade.start)))
        entry = buildings.setdefault(osm_id, {"aov_deg": 0.0, "distance_m": math.inf, "visible_wall_m": 0.0, "widest_deg": 0.0, "bearing_deg": 0.0})
        entry["aov_deg"] += width
        entry["visible_wall_m"] += length
        entry["distance_m"] = min(entry["distance_m"], facade.distance)
        if width > entry["widest_deg"]:
            entry["widest_deg"] = width
            entry["bearing_deg"] = (first + (((second - first + 180.0) % 360.0) - 180.0) / 2) % 360.0
    return {
        osm_id: {
            "aov_deg": round(min(item["aov_deg"], 360.0), 2), "distance_m": round(item["distance_m"], 2),
            "visible_wall_m": round(item["visible_wall_m"], 2), "bearing_deg": round(item["bearing_deg"], 2),
        }
        for osm_id, item in buildings.items()
    }


def _images_sha256(images: Iterable[dict[str, Any]]) -> str:
    return canonical_json_sha256(sorted((str(image["id"]), image.get("computed_geometry") or image.get("geometry")) for image in images))


def build_building_index(data_root: Path, dataset_id: str, images: list[dict[str, Any]]) -> dict[str, Any]:
    """Index every panorama with valid coordinates; panoramas without them are listed under ``skipped``."""
    started = time.perf_counter()
    buildings: dict[str, list[dict[str, Any]]] = {}
    # Every footprint within reach of some camera: the denominator of coverage.
    nearby: set[str] = set()
    skipped: list[dict[str, str]] = []
    indexed = 0
    for image in images:
        try:
            inputs = panorama_inputs(data_root, dataset_id, image)
        except ValueError as error:
            skipped.append({"image": str(image.get("id")), "reason": str(error)})
            continue
        indexed += 1
        features = nearby_buildings(inputs.latitude, inputs.longitude)
        nearby.update(str(feature["properties"]["osm_id"]) for feature in features if feature.get("properties", {}).get("osm_id"))
        for osm_id, visibility in panorama_visibility(features, inputs.latitude, inputs.longitude).items():
            buildings.setdefault(osm_id, []).append({"image": inputs.image_id, **visibility})
    for views in buildings.values():
        views.sort(key=lambda view: (-view["aov_deg"], view["distance_m"], view["image"]))
    return {
        "version": INDEX_VERSION,
        "dataset": dataset_id,
        "images_sha256": _images_sha256(images),
        "panoramas": indexed,
        "skipped": skipped,
        "seconds": round(time.perf_counter() - started, 2),
        "nearby_buildings": sorted(nearby),
        "buildings": dict(sorted(buildings.items())),
    }


def load_building_index(data_root: Path, dataset_id: str, images: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The stored index, or ``None`` when it is missing or the panorama set changed."""
    try:
        payload = json.loads(index_path(data_root, dataset_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != INDEX_VERSION or payload.get("images_sha256") != _images_sha256(images):
        return None
    return payload


def select_panoramas(
    index: dict[str, Any],
    building_ids: Iterable[str] | None = None,
    *,
    min_aov_deg: float = DEFAULT_MIN_AOV_DEGREES,
    max_aov_deg: float = DEFAULT_MAX_AOV_DEGREES,
    max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
    max_views_per_building: int = DEFAULT_MAX_VIEWS_PER_BUILDING,
) -> dict[str, Any]:
    """Panoramas worth analyzing for ``building_ids`` (default: all), at most N views per building.

    Returns the image IDs to analyze plus how many buildings are covered and how
    many have no acceptable view. Within the AOV window the widest view wins.
    """
    wanted = sorted(index["buildings"]) if building_ids is None else sorted(set(building_ids))
    chosen: set[str] = set()
    covered, uncovered = 0, []
    for osm_id in wanted:
        views = [
            view for view in index["buildings"].get(osm_id, [])
            if min_aov_deg <= view["aov_deg"] <= max_aov_deg and view["distance_m"] <= max_distance_m
        ]
        if not views:
            uncovered.append(osm_id)
            continue
        covered += 1
        chosen.update(view["image"] for view in views[:max_views_per_building])
    return {"images": sorted(chosen), "buildings_covered": covered, "buildings_uncovered": uncovered}


def main() -> None:
    parser = argparse.ArgumentParser(description="Index which panoramas see which OSM buildings (no models needed).")
    parser.add_argument("dataset", help="dataset directory name under --data-root")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--select", action="store_true", help="also print how many panoramas the default selection keeps")
    args = parser.parse_args()
    from .local_viewer import dataset_manifest

    try:
        images = [image for image in dataset_manifest(args.data_root, args.dataset)["images"] if isinstance(image, dict) and image.get("local_file")]
    except ValueError as error:
        parser.error(str(error))
    index = build_building_index(args.data_root, args.dataset, images)
    atomic_write_json(index_path(args.data_root, args.dataset), index)
    print(f"{len(index['buildings'])} buildings seen by {index['panoramas']} panoramas in {index['seconds']} s -> {index_path(args.data_root, args.dataset)}")
    if args.select:
        selection = select_panoramas(index)
        print(f"default selection: {len(selection['images'])} of {index['panoramas']} panoramas cover {selection['buildings_covered']} buildings; {len(selection['buildings_uncovered'])} have no view within the AOV window")


if __name__ == "__main__":
    main()
