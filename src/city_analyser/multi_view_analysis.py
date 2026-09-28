"""Selection and evidence fusion for analyses from nearby panoramas."""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

from .analysis_cache import canonical_json_sha256
from .building_analysis import ANALYSIS_VERSION, load_analysis

EARTH_RADIUS_M = 6_371_008.8
MULTI_VIEW_ANALYSIS_VERSION = 1
MAX_NEARBY_PANORAMAS = 7
NEARBY_PANORAMA_RADIUS_M = 45.0
MINIMUM_INDEPENDENT_BASELINE_M = 3.0


def panorama_coordinates(image: dict[str, Any]) -> tuple[float, float] | None:
    geometry = image.get("selected_geometry") or image.get("computed_geometry") or image.get("geometry")
    coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in coordinates[:2]):
        return None
    try:
        longitude, latitude = float(coordinates[0]), float(coordinates[1])
    except (TypeError, ValueError):
        return None
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        return None
    return longitude, latitude


def has_valid_heading(image: dict[str, Any]) -> bool:
    value = image.get("computed_compass_angle_deg")
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def distance_meters(first: tuple[float, float], second: tuple[float, float]) -> float:
    first_longitude, first_latitude = (math.radians(value) for value in first)
    second_longitude, second_latitude = (math.radians(value) for value in second)
    latitude_delta = second_latitude - first_latitude
    longitude_delta = second_longitude - first_longitude
    a = math.sin(latitude_delta / 2) ** 2 + math.cos(first_latitude) * math.cos(second_latitude) * math.sin(longitude_delta / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def maximum_independent_indices(coordinates: list[tuple[float, float]]) -> tuple[int, ...]:
    for size in range(len(coordinates), 0, -1):
        for indices in combinations(range(len(coordinates)), size):
            if all(
                distance_meters(coordinates[first], coordinates[second]) >= MINIMUM_INDEPENDENT_BASELINE_M
                for first, second in combinations(indices, 2)
            ):
                return indices
    return ()


def nearby_panoramas(images: list[dict[str, Any]], current_image: dict[str, Any]) -> list[tuple[dict[str, Any], float]]:
    current_coordinates = panorama_coordinates(current_image)
    if current_coordinates is None:
        raise ValueError("The panorama has no coordinates.")
    candidates = []
    current_id = str(current_image.get("id", ""))
    for image in images:
        coordinates = panorama_coordinates(image)
        if coordinates is None:
            continue
        distance = distance_meters(current_coordinates, coordinates)
        if str(image.get("id", "")) == current_id:
            candidates.append((image, 0.0))
        elif distance <= NEARBY_PANORAMA_RADIUS_M:
            candidates.append((image, distance))
    candidates.sort(key=lambda item: (item[1], str(item[0].get("id", ""))))
    return candidates[:MAX_NEARBY_PANORAMAS]


def nearby_analysis_path(data_root: Path, dataset_id: str, image_id: str) -> Path:
    return data_root / dataset_id / "analysis" / f"{image_id}-nearby.json"


def _selection_sha256(selected: list[tuple[dict[str, Any], float]]) -> str:
    return canonical_json_sha256([
        {
            "image_id": str(image.get("id", "")),
            "distance_m": round(distance, 4),
            "coordinates": panorama_coordinates(image),
            "heading_degrees": image.get("computed_compass_angle_deg"),
            "local_file": image.get("local_file"),
        }
        for image, distance in selected
    ])


def load_nearby_analysis(data_root: Path, dataset_id: str, image_id: str) -> dict[str, Any] | None:
    path = nearby_analysis_path(data_root, dataset_id, image_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != MULTI_VIEW_ANALYSIS_VERSION or payload.get("analysis_version") != ANALYSIS_VERSION:
        return None
    try:
        manifest = json.loads((data_root / dataset_id / "manifest.json").read_text(encoding="utf-8"))
        images = [image for image in manifest["images"] if isinstance(image, dict) and has_valid_heading(image)]
        current = next(image for image in images if str(image.get("id")) == image_id)
        expected_selection = _selection_sha256(nearby_panoramas(images, current))
    except (OSError, ValueError, KeyError, StopIteration, TypeError, json.JSONDecodeError):
        return None
    if payload.get("selection_sha256") != expected_selection:
        return None
    inputs = payload.get("analysis_inputs")
    if not isinstance(inputs, list) or not inputs:
        return None
    for item in inputs:
        if not isinstance(item, dict) or not isinstance(item.get("image_id"), str) or not isinstance(item.get("sha256"), str):
            return None
        analysis = load_analysis(data_root, dataset_id, item["image_id"])
        if analysis is None or canonical_json_sha256(analysis) != item["sha256"]:
            return None
    return payload


def fuse_nearby_analyses(image_id: str, analyses: list[tuple[dict[str, Any], float, dict[str, Any]]]) -> dict[str, Any]:
    buildings: dict[str, dict[str, Any]] = {}
    panoramas = []
    analysis_inputs = []
    for image, distance, analysis in analyses:
        current_id = str(image.get("id", ""))
        panoramas.append({"image_id": current_id, "distance_m": round(distance, 1), "matched_buildings": len(analysis.get("building_ids", {}))})
        analysis_inputs.append({"image_id": current_id, "sha256": canonical_json_sha256(analysis)})
        for building in analysis.get("building_ids", {}).values():
            if not isinstance(building, dict) or not isinstance(building.get("osm_id"), str):
                continue
            osm_id = building["osm_id"]
            pixels = int(building.get("pixels", 0))
            fused = buildings.setdefault(
                osm_id,
                {
                    "osm_id": osm_id,
                    "footprint_sources": set(),
                    "panorama_ids": [],
                    "view_evidence": [],
                    "observed_pixels": 0,
                    "inferred_pixels": 0,
                },
            )
            fused["panorama_ids"].append(current_id)
            coordinates = panorama_coordinates(image)
            if coordinates is not None:
                fused["view_evidence"].append((current_id, coordinates, float(building.get("confidence", 0))))
            fused["footprint_sources"].add(str(building.get("footprint_source", "osm")))
            fused["observed_pixels"] += pixels
            fused["inferred_pixels"] += int(building.get("inferred_pixels", 0))
    fused_buildings = []
    for building in buildings.values():
        view_evidence = building.pop("view_evidence")
        independent_indices = maximum_independent_indices([item[1] for item in view_evidence])
        independent = [view_evidence[index] for index in independent_indices]
        building["panorama_ids"] = sorted(set(building["panorama_ids"]))
        building["independent_panorama_ids"] = sorted({item[0] for item in independent})
        building["views"] = len(building["independent_panorama_ids"])
        building["footprint_sources"] = sorted(building["footprint_sources"])
        building["confidence"] = round(sum(item[2] for item in independent) / len(independent), 3) if independent else 0.0
        fused_buildings.append(building)
    fused_buildings.sort(key=lambda item: (-item["views"], -item["observed_pixels"], item["osm_id"]))
    return {
        "version": MULTI_VIEW_ANALYSIS_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "image_id": image_id,
        "selection_sha256": _selection_sha256([(image, distance) for image, distance, _ in analyses]),
        "analysis_inputs": analysis_inputs,
        "panoramas": panoramas,
        "building_ids": fused_buildings,
    }
