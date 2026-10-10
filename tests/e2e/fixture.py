"""A small synthetic workspace for browser tests: no network, no models.

Four OSM footprints along a street, three panoramas, and model-stage outputs
derived from the projected footprints themselves. The real pipeline then
associates them, so the viewer sees genuine, current analyses, including two
footprint diagnostics: one footprint covering two detected buildings and one
detected building with no footprint.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from PIL import Image

from city_analyser import building_analysis

DATASET = "sample"
# A synthetic street running east, away from any real coordinates that matter.
ORIGIN = (2.17, 41.387)
WIDTH, HEIGHT = 1024, 512
METERS_PER_DEGREE = 111_195.0
# (OSM id, version, tags, west, south, east, north in meters from the origin)
FOOTPRINTS = [
    (101, 4, {"building": "commercial", "building:levels": "3"}, 0, 8, 40, 20),
    (102, 2, {"building": "yes", "height": "12"}, 45, 8, 60, 20),
    (103, 7, {"building": "residential", "building:levels": "4"}, 0, -20, 25, -8),
    (104, 1, {"building": "yes"}, 30, -20, 50, -8),
]
# (image id, meters east of the origin, heading in degrees)
CAMERAS = [("1001", 10.0, 90.0), ("1002", 22.0, 80.0), ("1003", 36.0, 100.0)]
MERGED_FOOTPRINT = "way/101"


def lon_lat(east: float, north: float) -> list[float]:
    return [
        ORIGIN[0] + east / (METERS_PER_DEGREE * math.cos(math.radians(ORIGIN[1]))),
        ORIGIN[1] + north / METERS_PER_DEGREE,
    ]


def osm_features() -> list[dict[str, Any]]:
    features = []
    for identifier, version, tags, west, south, east, north in FOOTPRINTS:
        corners = [lon_lat(west, south), lon_lat(east, south), lon_lat(east, north), lon_lat(west, north)]
        element = {
            "type": "way", "id": identifier, "tags": tags, "version": version, "timestamp": "2025-01-01T00:00:00Z",
            "changeset": 1, "geometry": [{"lon": x, "lat": y} for x, y in corners],
        }
        features.append(building_analysis._feature_from_osm(element))
    return features


def _environment(root: Path) -> dict[str, str]:
    return {
        "BUILDING_ANALYSIS_OSM_CACHE_DIR": str(root / "data" / ".osm-cache"),
        "BUILDING_ANALYSIS_OSM_CACHE_DAYS": "36500",
        "BUILDING_ANALYSIS_FOOTPRINT_CORRECTIONS_DIR": str(root / "corrections"),
        "BUILDING_ANALYSIS_MODEL": "b0",
        "BUILDING_ANALYSIS_DEVICE": "cpu",
        "BUILDING_ANALYSIS_INSTANCES": "on",
        "BUILDING_ANALYSIS_STAGE_CACHE": "on",
        "BUILDING_ANALYSIS_STAGE_CACHE_DIR": "",
        "BUILDING_ANALYSIS_BACKEND": "local",
    }


def _write_osm_cache(root: Path, features: list[dict[str, Any]]) -> None:
    directory = root / "data" / ".osm-cache"
    directory.mkdir(parents=True, exist_ok=True)
    cells = set()
    for _, east, _ in CAMERAS:
        longitude, latitude = lon_lat(east, 0)
        cells.add((math.floor(round(latitude, 4) / building_analysis.OSM_CELL_DEGREES), math.floor(round(longitude, 4) / building_analysis.OSM_CELL_DEGREES)))
    for latitude_cell, longitude_cell in cells:
        payload = {"version": building_analysis.OSM_CACHE_VERSION, "features": features}
        (directory / f"{latitude_cell}_{longitude_cell}.json").write_text(json.dumps(payload), encoding="utf-8")


def _model_outputs(features: list[dict[str, Any]], east: float, heading: float) -> tuple[Any, Any, Any]:
    """Building confidence and an instance map that agree with the projected footprints."""
    longitude, latitude = lon_lat(east, 0)
    owners, _, candidates, _ = building_analysis._render_facade_owners(features, latitude, longitude, heading, WIDTH, HEIGHT)
    building = np.where(owners > 0, 230, 0).astype(np.uint8)
    instances = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    label = 0
    for ordinal, feature in enumerate(candidates, start=1):
        mask = owners == ordinal
        if not mask.any():
            continue
        columns = np.flatnonzero(mask.any(axis=0))
        if feature["properties"]["osm_id"] == MERGED_FOOTPRINT and len(columns) > 40:
            # Two buildings in the imagery under one footprint (a missing party wall).
            middle = columns[len(columns) // 2]
            for part in (mask & (np.arange(WIDTH) < middle)[None, :], mask & (np.arange(WIDTH) >= middle)[None, :]):
                label += 1
                instances[part] = label
        else:
            label += 1
            instances[mask] = label
    # A building to the west of the street that OSM does not map.
    west = int(((270.0 - heading + 180.0) % 360.0) / 360.0 * WIDTH)
    rows = slice(int((0.5 - 25 / 180) * HEIGHT), int(0.5 * HEIGHT))
    span = slice(max(west - 25, 0), min(west + 25, WIDTH))
    label += 1
    instances[rows, span] = label
    building[rows, span] = 230
    return building, instances, owners


def _panorama(instances: Any) -> Any:
    """Sky, street and flat-colored facades, enough to look like a panorama in screenshots."""
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[: HEIGHT // 2] = (150, 185, 220)
    image[HEIGHT // 2:] = (95, 95, 92)
    palette = np.array([(196, 160, 120), (170, 120, 90), (210, 190, 150), (150, 150, 140), (185, 140, 110), (120, 135, 150)], dtype=np.uint8)
    filled = instances > 0
    image[filled] = palette[instances[filled] % len(palette)]
    return image


def create_workspace(root: Path) -> dict[str, Any]:
    """Write the dataset and analyze it with the real pipeline. Returns the environment it needs."""
    environment = _environment(root)
    features = osm_features()
    _write_osm_cache(root, features)
    (root / "models").mkdir(parents=True, exist_ok=True)
    dataset = root / "data" / DATASET
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    images, outputs = [], {}
    for number, (image_id, east, heading) in enumerate(CAMERAS):
        building, instances, _ = _model_outputs(features, east, heading)
        Image.fromarray(_panorama(instances)).save(dataset / "images" / f"{image_id}.jpg", quality=88)
        images.append({
            "id": image_id, "is_pano": True, "local_file": f"images/{image_id}.jpg",
            "captured_at_ms": 1_700_000_000_000 + number * 60_000, "sequence_id": "street",
            "computed_geometry": {"type": "Point", "coordinates": lon_lat(east, 0)},
            "computed_compass_angle_deg": heading, "creator": {"username": "fixture"},
        })
        outputs[image_id] = (building, instances)
    (dataset / "manifest.json").write_text(json.dumps({"query": {"provider": "Synthetic fixture"}, "images": images}), encoding="utf-8")

    current = {}

    def semantic(panorama: Any, model_root: Path, progress: Any) -> tuple[Any, Any, str, str, str | None]:
        building, _ = outputs[current["id"]]
        return building, np.zeros_like(building), "cpu", "fixture/semantic", None

    def instances(panorama: Any, model_root: Path, device: Any, settings: Any, progress: Any) -> tuple[Any, list[dict[str, Any]]]:
        _, instance_map = outputs[current["id"]]
        labels = [int(value) for value in np.unique(instance_map) if value]
        return instance_map, [
            {"instance": value, "score": 0.9, "views": [0], "pixels": int((instance_map == value).sum())} for value in labels
        ]

    # The model stages are replaced, so PyTorch is never needed (the browser tests run without it).
    with patch.dict(os.environ, environment), patch.object(building_analysis, "_semantic_confidences", semantic), \
            patch.object(building_analysis, "segment_instances", instances), \
            patch.object(building_analysis, "_require_ml_dependencies", lambda: (np, None, None, None)), \
            patch.object(building_analysis, "inference_device", lambda torch: "cpu"):
        for image in images:
            current["id"] = image["id"]
            building_analysis.analyze_panorama(root / "data", root / "models", DATASET, image, lambda _: None)
    return environment
