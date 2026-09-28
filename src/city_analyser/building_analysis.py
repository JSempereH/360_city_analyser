"""OSM-guided building segmentation for local equirectangular panoramas.

The optional ML dependencies are loaded only when an analysis job starts, so the
viewer and dataset tools remain usable on a machine without PyTorch installed.
"""

from __future__ import annotations

import json
import importlib
import io
import math
import os
import re
import time
import xml.etree.ElementTree as ElementTree
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .building_instances import instance_coverage, instance_settings, segment_instances
from .analysis_cache import analysis_cache_matches, analysis_input_sha256, atomic_write_bytes, file_sha256, sfm_input_sha256
from .panorama_geometry import panorama_to_tile_grid, perspective_tile


# B5 materially improves facade boundaries in shadows and vegetation. It fits
# the project's 4 GB NVIDIA GPU at 512 px tiles; CPU uses B0 to stay usable.
GPU_MODEL_ID = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
CPU_MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
HIGH_ACCURACY_MODEL_ID = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
MODEL_VERSION = "mask2former-mapillary-high-segformer-confidence-fusion-osm-multipolygons"
ANALYSIS_VERSION = 26
EARTH_RADIUS_M = 6_371_008.8
DEFAULT_BUILDING_HEIGHT_M = 10.0
UNKNOWN_FACADE_RENDER_HEIGHT_M = 32.0
CAMERA_HEIGHT_M = 1.7
# Model input resolution when BUILDING_ANALYSIS_TILE_SIZE is unset; these match
# each processor's own training resize so the default compute is unchanged.
SEGFORMER_TILE_SIZE = 512
MASK2FORMER_TILE_SIZE = 384
DEFAULT_SEGMENTATION_BATCH_SIZE = 4
PERSPECTIVE_FOV_DEGREES = 84
PERSPECTIVE_PITCHES_DEG = (0, 45)
_HORIZONTAL_YAWS = tuple(step * math.pi / 3 for step in range(6))
# Close facades commonly extend above a horizon-only 84° perspective tile.
# The upward ring overlaps it and covers the upper stories to the zenith.
PERSPECTIVE_VIEWS = tuple((yaw, math.radians(pitch)) for pitch in PERSPECTIVE_PITCHES_DEG for yaw in _HORIZONTAL_YAWS)
POSE_SEARCH_OFFSETS_M = (-4.0, 0.0, 4.0)
POSE_SEARCH_HEADING_OFFSETS_DEG = (-6.0, 0.0, 6.0)
POSE_SEARCH_MAX_FACADE_DISTANCE_M = 45.0
POSE_MINIMUM_IOU_GAIN = 0.03
POSE_MINIMUM_IOU = 0.08
POSE_MINIMUM_RUNNER_UP_MARGIN = 0.01
# Candidates farther than this from the best pose count as rival hypotheses.
POSE_DISTINCT_POSITION_M = 4.5
POSE_DISTINCT_HEADING_DEG = 6.5
# GPS inside a footprint is moved to open street space this far from walls.
GPS_REPAIR_CLEARANCE_M = 1.0
GPS_REPAIR_MAXIMUM_SHIFT_M = 12.0
# Structures a street camera can legitimately stand under.
OVERHEAD_BUILDING_TYPES = frozenset({"roof", "canopy", "carport", "bridge"})
# An instance names one footprint only when most of it lies on projected
# facades and one footprint clearly dominates those pixels.
INSTANCE_MIN_PROJECTED_SHARE = 0.3
INSTANCE_MIN_OWNER_SHARE = 0.6
# Narrower facade slivers inside a detected building are merged into a neighbor.
INSTANCE_MIN_FACADE_DEGREES = 2.0
OSM_QUERY_RADIUS_M = 140.0
# About 110 m; every camera in a cell reuses one enlarged footprint query.
OSM_CELL_DEGREES = 0.001
OSM_CACHE_VERSION = 1
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)


def _require_ml_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        np = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        Image = importlib.import_module("PIL.Image")
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise RuntimeError(
            "Building analysis requires Pillow, PyTorch, and Transformers. "
            "Run uv sync, then retry."
        ) from error
    return np, torch, Image, transformers


def _parse_osm_number(value: object) -> tuple[float, str] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return (number, "") if math.isfinite(number) else None
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*([+]?(?:\d+(?:[.,]\d+)?|[.,]\d+))\s*([a-zA-Z']*)\s*", value)
    if not match:
        return None
    return float(match.group(1).replace(",", ".")), match.group(2).lower()


def _parse_height(value: object) -> float | None:
    parsed = _parse_osm_number(value)
    if parsed is None:
        return None
    height, unit = parsed
    if unit in {"ft", "foot", "feet", "'"}:
        height *= 0.3048
    elif unit not in {"", "m", "meter", "meters", "metre", "metres"}:
        return None
    return height if 0 <= height <= 500 else None


def _parse_levels(value: object) -> float | None:
    parsed = _parse_osm_number(value)
    if parsed is None or parsed[1]:
        return None
    levels = parsed[0]
    return levels if 0 < levels <= 150 else None


def building_height_source(properties: dict[str, Any]) -> str:
    explicit = _parse_height(properties.get("height"))
    if explicit is not None:
        return "height"
    levels = _parse_levels(properties.get("building:levels"))
    if levels is not None:
        return "levels"
    return "default"


def building_height(properties: dict[str, Any]) -> float:
    explicit = _parse_height(properties.get("height"))
    if explicit is not None:
        return explicit
    levels = _parse_levels(properties.get("building:levels"))
    return max(3.0, levels * 3.2) if levels is not None else DEFAULT_BUILDING_HEIGHT_M


def facade_render_height(properties: dict[str, Any]) -> float:
    """Avoid cutting observed upper stories when OSM omits a building height."""
    return building_height(properties) if building_height_source(properties) != "default" else UNKNOWN_FACADE_RENDER_HEIGHT_M


def facade_bottom_height(properties: dict[str, Any]) -> float:
    explicit = _parse_height(properties.get("min_height"))
    if explicit is not None:
        return explicit
    levels = _parse_levels(properties.get("building:min_level"))
    return levels * 3.2 if levels is not None else 0.0


def _is_building(tags: dict[str, Any]) -> bool:
    building = str(tags.get("building", "")).strip().lower()
    building_part = str(tags.get("building:part", "")).strip().lower()
    return building not in {"", "no"} or building_part not in {"", "no"}


def _coordinate_path(value: object) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    path = []
    for point in value:
        if (
            not isinstance(point, dict)
            or not isinstance(point.get("lon"), (int, float)) or isinstance(point.get("lon"), bool)
            or not isinstance(point.get("lat"), (int, float)) or isinstance(point.get("lat"), bool)
        ):
            return None
        longitude, latitude = float(point["lon"]), float(point["lat"])
        if not math.isfinite(longitude) or not math.isfinite(latitude):
            return None
        path.append([longitude, latitude])
    return path


def _polygonize_member_paths(paths: list[list[list[float]]]) -> list[Any]:
    if not paths:
        return []
    try:
        geometry = importlib.import_module("shapely.geometry")
        operations = importlib.import_module("shapely.ops")
    except ImportError as error:
        raise RuntimeError("OpenStreetMap multipolygons require Shapely. Run uv sync, then retry.") from error
    lines = [geometry.LineString(path) for path in paths if len(path) >= 2]
    return list(operations.polygonize(operations.unary_union(lines))) if lines else []


def _relation_geometry(element: dict[str, Any]) -> dict[str, Any] | None:
    members = element.get("members")
    if not isinstance(members, list):
        return None
    outer_paths, inner_paths = [], []
    for member in members:
        if not isinstance(member, dict) or member.get("type") != "way":
            continue
        path = _coordinate_path(member.get("geometry"))
        if path is None:
            continue
        role = str(member.get("role", "")).lower()
        (inner_paths if role == "inner" else outer_paths).append(path)
    outers = _polygonize_member_paths(outer_paths)
    inners = _polygonize_member_paths(inner_paths)
    if not outers:
        return None
    try:
        geometry = importlib.import_module("shapely.geometry")
        validation = importlib.import_module("shapely.validation")
    except ImportError as error:
        raise RuntimeError("OpenStreetMap multipolygons require Shapely. Run uv sync, then retry.") from error
    polygons = []
    for outer in outers:
        holes = [list(inner.exterior.coords) for inner in inners if outer.covers(inner.representative_point())]
        candidate = geometry.Polygon(outer.exterior.coords, holes)
        if not candidate.is_valid:
            candidate = validation.make_valid(candidate)
        if candidate.geom_type == "Polygon":
            polygons.append(candidate)
        elif candidate.geom_type == "MultiPolygon":
            polygons.extend(candidate.geoms)
    if not polygons:
        return None
    mapped = geometry.mapping(polygons[0] if len(polygons) == 1 else geometry.MultiPolygon(polygons))
    return cast(dict[str, Any], json.loads(json.dumps(mapped)))


def _feature_from_osm(element: dict[str, Any]) -> dict[str, Any] | None:
    tags = cast(dict[str, Any], element.get("tags")) if isinstance(element.get("tags"), dict) else {}
    if not _is_building(tags):
        return None
    osm_type = str(element.get("type", "way"))
    osm_id = element.get("id")
    if not isinstance(osm_id, int):
        return None
    if osm_type == "relation":
        feature_geometry = _relation_geometry(element)
    else:
        ring = _coordinate_path(element.get("geometry"))
        if ring is None or len(ring) < 3:
            return None
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        feature_geometry = {"type": "Polygon", "coordinates": [ring]}
    if feature_geometry is None:
        return None
    properties = {**tags, "osm_id": f"{osm_type}/{osm_id}", "osm_type": osm_type, "footprint_source": "osm"}
    return {"type": "Feature", "properties": properties, "geometry": feature_geometry}


def _features_from_osm_elements(elements: Iterable[object]) -> tuple[dict[str, Any], ...]:
    candidates = [element for element in elements if isinstance(element, dict)]
    features = []
    relation_member_ids = set()
    relations = sorted((element for element in candidates if element.get("type") == "relation"), key=lambda item: int(item.get("id", 0)))
    for relation in relations:
        feature = _feature_from_osm(relation)
        if feature is None:
            continue
        features.append(feature)
        members = relation.get("members")
        if not isinstance(members, list):
            continue
        geometry = None
        try:
            geometry = importlib.import_module("shapely.geometry")
            boundary = geometry.shape(feature["geometry"]).boundary
        except (ImportError, TypeError, ValueError):
            boundary = None
        if boundary is not None and geometry is not None:
            for member in members:
                if (
                    not isinstance(member, dict) or member.get("type") != "way" or not isinstance(member.get("ref"), int)
                    or str(member.get("role", "")).lower() == "inner"
                ):
                    continue
                path = _coordinate_path(member.get("geometry"))
                if path is not None and boundary.buffer(1e-12).covers(geometry.LineString(path)):
                    relation_member_ids.add(int(member["ref"]))
    ways = sorted((element for element in candidates if element.get("type") == "way"), key=lambda item: int(item.get("id", 0)))
    for element in ways:
        if element.get("type") == "way" and element.get("id") in relation_member_ids:
            continue
        feature = _feature_from_osm(element)
        if feature is not None:
            features.append(feature)
    return tuple(features)


def _download_buildings_from_osm_api(latitude: float, longitude: float, radius_m: float) -> tuple[dict[str, Any], ...]:
    """Use the main OSM map API for a small fallback bbox when Overpass is busy."""
    latitude_delta = radius_m / EARTH_RADIUS_M * 180 / math.pi
    longitude_delta = latitude_delta / max(math.cos(math.radians(latitude)), 0.1)
    bbox = f"{longitude - longitude_delta},{latitude - latitude_delta},{longitude + longitude_delta},{latitude + latitude_delta}"
    request = Request(
        f"https://api.openstreetmap.org/api/0.6/map?{urlencode({'bbox': bbox})}",
        headers={"User-Agent": "local-panorama-building-viewer/1.0"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS endpoint
        root = ElementTree.fromstring(response.read())
    nodes = {
        node.attrib["id"]: [float(node.attrib["lon"]), float(node.attrib["lat"])]
        for node in root.findall("node")
        if "id" in node.attrib and "lon" in node.attrib and "lat" in node.attrib
    }
    ways: dict[int, dict[str, Any]] = {}
    for way in root.findall("way"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag") if "k" in tag.attrib and "v" in tag.attrib}
        ring = [nodes[reference.attrib["ref"]] for reference in way.findall("nd") if reference.attrib.get("ref") in nodes]
        if len(ring) < 2 or "id" not in way.attrib:
            continue
        way_id = int(way.attrib["id"])
        ways[way_id] = {"type": "way", "id": way_id, "tags": tags, "geometry": [{"lon": point[0], "lat": point[1]} for point in ring]}
    elements: list[dict[str, Any]] = list(ways.values())
    for relation in root.findall("relation"):
        if "id" not in relation.attrib:
            continue
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in relation.findall("tag") if "k" in tag.attrib and "v" in tag.attrib}
        members = []
        for member in relation.findall("member"):
            try:
                reference = int(member.attrib["ref"])
            except (KeyError, ValueError):
                continue
            source = ways.get(reference)
            if member.attrib.get("type") == "way" and source is not None:
                members.append({"type": "way", "ref": reference, "role": member.attrib.get("role", ""), "geometry": source["geometry"]})
        elements.append({"type": "relation", "id": int(relation.attrib["id"]), "tags": tags, "members": members})
    return _features_from_osm_elements(elements)


def _download_buildings(latitude: float, longitude: float, radius_m: float) -> tuple[dict[str, Any], ...]:
    radius = f"{radius_m:.0f}"
    query = (
        "[out:json][timeout:25];("
        f'way["building"](around:{radius},{latitude},{longitude});'
        f'relation["building"]["type"="multipolygon"](around:{radius},{latitude},{longitude});'
        f'way["building:part"](around:{radius},{latitude},{longitude});'
        f'relation["building:part"]["type"="multipolygon"](around:{radius},{latitude},{longitude});'
        ");"
        "out tags geom;"
    )
    errors = []
    payload = None
    for endpoint in OVERPASS_ENDPOINTS:
        request = Request(
            f"{endpoint}?{urlencode({'data': query})}",
            headers={"User-Agent": "local-panorama-building-viewer/1.0"},
        )
        try:
            with urlopen(request, timeout=20) as response:  # noqa: S310 - fixed HTTPS endpoint
                candidate = json.loads(response.read().decode("utf-8"))
            if isinstance(candidate, dict):
                payload = candidate
                break
            errors.append(f"{endpoint}: invalid response")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{endpoint}: {error}")
    if payload is not None:
        return _features_from_osm_elements(payload.get("elements", []))
    try:
        return _download_buildings_from_osm_api(latitude, longitude, radius_m + 10)
    except (OSError, ElementTree.ParseError) as error:
        errors.append(f"OSM map API: {error}")
    raise RuntimeError("No OpenStreetMap building service responded. " + " | ".join(errors))


def _osm_cache_directory() -> Path | None:
    configured = os.environ.get("BUILDING_ANALYSIS_OSM_CACHE_DIR", "").strip()
    if configured.lower() in {"off", "none", "0"}:
        return None
    return Path(configured) if configured else Path("data") / ".osm-cache"


def _osm_cache_max_age_seconds() -> float:
    configured = os.environ.get("BUILDING_ANALYSIS_OSM_CACHE_DAYS", "30").strip()
    try:
        days = float(configured)
    except ValueError as error:
        raise RuntimeError("BUILDING_ANALYSIS_OSM_CACHE_DAYS must be a number of days.") from error
    return max(0.0, days) * 86_400


@lru_cache(maxsize=64)
def _cell_buildings(latitude_cell: int, longitude_cell: int) -> tuple[dict[str, Any], ...]:
    """Buildings for one OSM_CELL_DEGREES cell, shared by every camera inside it.

    A disk snapshot keeps nearby panoramas and viewer restarts off the public
    Overpass servers, and pins the footprints a run was computed against.
    """
    directory = _osm_cache_directory()
    cache_path = directory / f"{latitude_cell}_{longitude_cell}.json" if directory is not None else None
    if cache_path is not None:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            fresh = time.time() - cache_path.stat().st_mtime <= _osm_cache_max_age_seconds()
            if fresh and isinstance(cached, dict) and cached.get("version") == OSM_CACHE_VERSION and isinstance(cached.get("features"), list):
                return tuple(cached["features"])
        except (OSError, json.JSONDecodeError):
            pass
    center_latitude = (latitude_cell + 0.5) * OSM_CELL_DEGREES
    center_longitude = (longitude_cell + 0.5) * OSM_CELL_DEGREES
    half_diagonal_m = math.hypot(1.0, math.cos(math.radians(center_latitude))) * OSM_CELL_DEGREES / 2 * math.pi / 180 * EARTH_RADIUS_M
    features = _download_buildings(round(center_latitude, 6), round(center_longitude, 6), OSM_QUERY_RADIUS_M + half_diagonal_m)
    if cache_path is not None:
        try:
            atomic_write_bytes(cache_path, json.dumps({
                "version": OSM_CACHE_VERSION,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "center": [center_longitude, center_latitude],
                "features": list(features),
            }).encode("utf-8"))
        except OSError:
            pass
    return features


def nearby_buildings(latitude: float, longitude: float) -> list[dict[str, Any]]:
    """Return cacheable GeoJSON features while preserving OSM identifiers and tags.

    Features are those with a wall within OSM_QUERY_RADIUS_M of the rounded
    camera position, matching an Overpass ``around`` query at that point.
    """
    latitude_key, longitude_key = round(latitude, 4), round(longitude, 4)
    features = _cell_buildings(math.floor(latitude_key / OSM_CELL_DEGREES), math.floor(longitude_key / OSM_CELL_DEGREES))
    geometry = _facade_geometry()
    distances = geometry.nearest_wall_distances(geometry.build_scene(list(features), latitude_key, longitude_key))
    return [feature for feature, distance in zip(features, distances) if distance <= OSM_QUERY_RADIUS_M]


def _facade_geometry() -> Any:
    try:
        return importlib.import_module(".facade_geometry", __package__)
    except ImportError as error:
        raise RuntimeError("Facade projection requires NumPy and pyproj. Run uv sync, then retry.") from error


def _edge_is_shared_with_building(
    feature: dict[str, Any], first: list[float], second: list[float], features: list[dict[str, Any]],
    camera_latitude: float, camera_longitude: float,
) -> bool:
    """Reject a whole wall shared with an adjacent footprint as an internal wall."""
    geometry = _facade_geometry()
    np = __import__("numpy")
    scene = geometry.build_scene(features, camera_latitude, camera_longitude)
    start, end = geometry.project_points(camera_latitude, camera_longitude, np.array([first, second], dtype=np.float64))
    feature_index = next((index for index, candidate in enumerate(features) if candidate is feature), -1)
    return bool(geometry.shared_with_other_feature(scene, feature_index, start, end))


def _render_scene_owners(
    scene: Any, facades: list[Any], heading_degrees: float, width: int, height: int,
) -> tuple[Any, Any, list[dict[str, Any]], dict[str, int]]:
    """Rasterize visible walls nearest-first into an owner/depth buffer."""
    np = __import__("numpy")
    geometry = _facade_geometry()
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    owners = np.zeros((height, width), dtype=np.uint16)
    candidate_features: list[dict[str, Any]] = []
    ordinals: dict[str, int] = {}
    facade_counts: dict[str, int] = {}
    for facade in facades:
        feature = scene.features[facade.feature_index]
        osm_id = str(feature["properties"]["osm_id"])
        if osm_id not in ordinals:
            ordinals[osm_id] = len(candidate_features) + 1
            candidate_features.append(feature)
            facade_counts[osm_id] = 0
        facade_counts[osm_id] += 1
    for facade in sorted(facades, key=lambda item: item.distance):
        feature = scene.features[facade.feature_index]
        properties = feature.get("properties", {})
        geometry.render_facade(
            depth_buffer, owners, ordinals[str(properties["osm_id"])], facade, heading_degrees,
            facade_render_height(properties) - CAMERA_HEIGHT_M, facade_bottom_height(properties) - CAMERA_HEIGHT_M,
        )
    return owners, depth_buffer, candidate_features, facade_counts


def _render_facade_owners(
    features: list[dict[str, Any]], camera_latitude: float, camera_longitude: float, heading_degrees: float, width: int, height: int,
) -> tuple[Any, Any, list[dict[str, Any]], dict[str, int]]:
    """Render all clear OSM walls into a pixel-accurate owner/depth buffer."""
    geometry = _facade_geometry()
    scene = geometry.build_scene(features, camera_latitude, camera_longitude)
    return _render_scene_owners(scene, geometry.visible_facades(scene), heading_degrees, width, height)


def _sfm_depth_samples_from_payload(payload: object, image_id: str) -> list[dict[str, float]]:
    if (
        not isinstance(payload, dict) or payload.get("version") != 3
        or str(payload.get("image_id")) != image_id
    ):
        return []
    samples = payload.get("depth_samples")
    if not isinstance(samples, list):
        return []
    valid = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        try:
            u, v, distance = float(sample["u"]), float(sample["v"]), float(sample["distance_m"])
            track_length = int(sample["track_length"])
            reprojection_error = float(sample["reprojection_error_px"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            0 <= u < 1 and 0 <= v < 1 and math.isfinite(distance) and distance > 0
            and track_length >= 2 and math.isfinite(reprojection_error) and 0 <= reprojection_error <= 8
        ):
            valid.append({"u": u, "v": v, "distance_m": distance, "track_length": track_length, "reprojection_error_px": reprojection_error})
    return valid


def _sfm_depth_samples(data_root: Path, dataset_id: str, image_id: str) -> list[dict[str, float]]:
    """Load sparse metric depths written by the compatible SfM refinement cache."""
    from .sfm_refinement import load_sfm_refinement

    return _sfm_depth_samples_from_payload(load_sfm_refinement(data_root, dataset_id, image_id), image_id)


def _sfm_depth_conflicts(depth_buffer: Any, samples: list[dict[str, float]]) -> tuple[Any, Any, Any]:
    """Mark sparse 3D points that contradict an OSM facade's expected ray depth."""
    np = __import__("numpy")
    known = np.zeros(depth_buffer.shape, dtype=bool)
    foreground = np.zeros(depth_buffer.shape, dtype=bool)
    background = np.zeros(depth_buffer.shape, dtype=bool)
    height, width = depth_buffer.shape
    for sample in samples:
        x = min(width - 1, int(sample["u"] * width))
        y = min(height - 1, int(sample["v"] * height))
        radius = max(1, round(width / 1024))
        for sample_y in range(max(0, y - radius), min(height, y + radius + 1)):
            for sample_x in range(max(0, x - radius), min(width, x + radius + 1)):
                horizontal_depth = float(depth_buffer[sample_y, sample_x])
                if not math.isfinite(horizontal_depth):
                    continue
                elevation = (0.5 - (sample_y + 0.5) / height) * math.pi
                expected_distance = horizontal_depth / max(math.cos(elevation), 0.1)
                tolerance = max(3.0, expected_distance * 0.2)
                known[sample_y, sample_x] = True
                if sample["distance_m"] < expected_distance - tolerance:
                    foreground[sample_y, sample_x] = True
                elif sample["distance_m"] > expected_distance + tolerance:
                    background[sample_y, sample_x] = True
    return known, foreground, background


def inference_device(torch: Any) -> Any:
    """Device for all model stages: BUILDING_ANALYSIS_DEVICE or the best available."""
    configured = os.environ.get("BUILDING_ANALYSIS_DEVICE", "auto").strip().lower()
    if configured == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        return torch.device("mps" if mps is not None and mps.is_available() else "cpu")
    try:
        device = torch.device(configured)
    except RuntimeError as error:
        raise RuntimeError("BUILDING_ANALYSIS_DEVICE must be auto, cpu, cuda, cuda:<index> or mps.") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BUILDING_ANALYSIS_DEVICE requests CUDA, but PyTorch cannot see a CUDA GPU.")
    return device


def _segmentation_model_spec(torch: Any) -> tuple[str, str]:
    """Choose an accurate street-scene model when the available GPU can hold it."""
    requested = os.environ.get("BUILDING_ANALYSIS_MODEL", "auto").strip()
    if requested == "high":
        return HIGH_ACCURACY_MODEL_ID, "mask2former"
    if requested == "balanced":
        return GPU_MODEL_ID, "segformer"
    if re.fullmatch(r"b[0-5]", requested):
        return f"nvidia/segformer-{requested}-finetuned-cityscapes-1024-1024", "segformer"
    if requested not in {"", "auto"}:
        return requested, "mask2former" if "mask2former" in requested.lower() else "segformer"
    device = inference_device(torch)
    if device.type == "cuda" and torch.cuda.get_device_properties(device).total_memory >= 12 * 1024 ** 3:
        return HIGH_ACCURACY_MODEL_ID, "mask2former"
    return (GPU_MODEL_ID, "segformer") if device.type == "cuda" else (CPU_MODEL_ID, "segformer")


def _segmentation_tile_size(architecture: str) -> int:
    """Edge of each perspective tile, which is also the model input resolution."""
    default = MASK2FORMER_TILE_SIZE if architecture == "mask2former" else SEGFORMER_TILE_SIZE
    configured = os.environ.get("BUILDING_ANALYSIS_TILE_SIZE", "").strip()
    if not configured:
        return default
    try:
        tile_size = int(configured)
    except ValueError as error:
        raise RuntimeError("BUILDING_ANALYSIS_TILE_SIZE must be an integer between 256 and 1024.") from error
    if not 256 <= tile_size <= 1024:
        raise RuntimeError("BUILDING_ANALYSIS_TILE_SIZE must be between 256 and 1024.")
    return tile_size


def _segmentation_batch_size() -> int:
    configured = os.environ.get("BUILDING_ANALYSIS_BATCH_SIZE", "").strip()
    if not configured:
        return DEFAULT_SEGMENTATION_BATCH_SIZE
    try:
        batch_size = int(configured)
    except ValueError as error:
        raise RuntimeError(f"BUILDING_ANALYSIS_BATCH_SIZE must be an integer between 1 and {len(PERSPECTIVE_VIEWS)}.") from error
    if not 1 <= batch_size <= len(PERSPECTIVE_VIEWS):
        raise RuntimeError(f"BUILDING_ANALYSIS_BATCH_SIZE must be between 1 and {len(PERSPECTIVE_VIEWS)}.")
    return batch_size


def _segmentation_minimum_confidence() -> float:
    configured = os.environ.get("BUILDING_ANALYSIS_MIN_CONFIDENCE", "0.30").strip()
    try:
        confidence = float(configured)
    except ValueError as error:
        raise RuntimeError("BUILDING_ANALYSIS_MIN_CONFIDENCE must be a number between 0 and 1.") from error
    if not 0 < confidence <= 1:
        raise RuntimeError("BUILDING_ANALYSIS_MIN_CONFIDENCE must be between 0 and 1.")
    return confidence


def _label_id(labels: dict[int, str], category: str) -> int | None:
    return next(
        (key for key, value in labels.items() if value == category or value.endswith(f"--{category}") or value.endswith(f"/{category}")),
        None,
    )


@lru_cache(maxsize=4)
def _load_segmentation_model(model_id: str, architecture: str, model_root: str, device_name: str, revision: str | None) -> tuple[Any, Any]:
    _, torch, _, transformers = _require_ml_dependencies()
    processor = transformers.AutoImageProcessor.from_pretrained(model_id, cache_dir=Path(model_root), revision=revision)
    model_class = transformers.Mask2FormerForUniversalSegmentation if architecture == "mask2former" else transformers.SegformerForSemanticSegmentation
    model = model_class.from_pretrained(model_id, cache_dir=Path(model_root), revision=revision).to(torch.device(device_name)).eval()
    return processor, model


def _tile_scores(
    torch: Any, processor: Any, model: Any, architecture: str, tiles: list[Any], tile_size: int,
    device: Any, building_label: int, vegetation_label: int,
) -> tuple[Any, Any]:
    """Per-pixel building and vegetation confidence for a batch of tiles, on device."""
    # The processor would otherwise resample every tile to its configured size,
    # silently decoupling BUILDING_ANALYSIS_TILE_SIZE from the model resolution.
    inputs = processor(images=tiles, size={"height": tile_size, "width": tile_size}, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        output = model(**inputs)
        if architecture == "mask2former":
            labels = torch.stack(processor.post_process_semantic_segmentation(output, target_sizes=[(tile_size, tile_size)] * len(tiles)))
            return (labels == building_label).float(), (labels == vegetation_label).float()
        logits = torch.nn.functional.interpolate(output.logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
        labels = logits.argmax(dim=1)
        probabilities = torch.nn.functional.softmax(logits.float(), dim=1)
        # Preserve the semantic winner while using its probability to
        # reject uncertain single-view predictions at overlap seams.
        building = probabilities[:, building_label] * (labels == building_label)
        vegetation = probabilities[:, vegetation_label] * (labels == vegetation_label)
        return building, vegetation


def _load_analysis_panorama(image_path: Path) -> Any:
    """Validated equirectangular RGB array at the analysis resolution (<= 2048 wide)."""
    np = __import__("numpy")
    Image = importlib.import_module("PIL.Image")
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        if source.width < 512 or source.height < 256:
            raise ValueError("The panorama resolution is too low for building analysis (minimum 512x256).")
        if not math.isclose(source.width / source.height, 2.0, rel_tol=0.03):
            raise ValueError("The image is not a valid 2:1 equirectangular panorama.")
        analysis_width = min(2048, source.width)
        return np.asarray(source.resize((analysis_width, analysis_width // 2), Image.Resampling.LANCZOS))


def _segment_buildings(image_path: Path, model_root: Path, progress: Callable[[str], None]) -> tuple[Any, Any, int, str, str, str | None]:
    _, torch, _, _ = _require_ml_dependencies()
    device = inference_device(torch)
    model_id, architecture = _segmentation_model_spec(torch)
    requested_revision = os.environ.get("BUILDING_ANALYSIS_MODEL_REVISION", "").strip() or None
    progress(f"Loading {model_id} on {device.type.upper()}...")
    processor, model = _load_segmentation_model(model_id, architecture, str(model_root.resolve()), str(device), requested_revision)
    labels = {int(key): str(value).lower() for key, value in model.config.id2label.items()}
    building_label = _label_id(labels, "building")
    vegetation_label = _label_id(labels, "vegetation")
    if building_label is None or vegetation_label is None:
        raise RuntimeError("The downloaded model does not include building and vegetation labels.")
    panorama = _load_analysis_panorama(image_path)
    analysis_height, analysis_width = panorama.shape[:2]
    tile_size = _segmentation_tile_size(architecture)
    tiles = [perspective_tile(panorama, yaw, pitch, tile_size, PERSPECTIVE_FOV_DEGREES)[0] for yaw, pitch in PERSPECTIVE_VIEWS]
    # Several overlapping perspective views can reach one equirectangular
    # pixel. Keep their strongest semantic evidence instead of a boolean union.
    building_confidence = torch.zeros((analysis_height, analysis_width), dtype=torch.float32, device=device)
    vegetation_confidence = torch.zeros_like(building_confidence)
    batch_size = _segmentation_batch_size()
    start = 0
    while start < len(tiles):
        stop = min(start + batch_size, len(tiles))
        progress(f"Segmenting perspectives {start + 1}-{stop}/{len(tiles)} on {device.type.upper()}...")
        try:
            building, vegetation = _tile_scores(
                torch, processor, model, architecture, tiles[start:stop], tile_size, device, building_label, vegetation_label,
            )
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            # Small GPUs: retry the same tiles with half the batch.
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            continue
        scores = torch.stack((building, vegetation), dim=1).float()
        for offset, (yaw, pitch) in enumerate(PERSPECTIVE_VIEWS[start:stop]):
            grid, visible = panorama_to_tile_grid(torch, analysis_width, analysis_height, yaw, pitch, PERSPECTIVE_FOV_DEGREES, device)
            sampled = torch.nn.functional.grid_sample(scores[offset:offset + 1], grid, mode="bilinear", align_corners=False)[0]
            building_confidence = torch.where(visible, torch.maximum(building_confidence, sampled[0]), building_confidence)
            vegetation_confidence = torch.where(visible, torch.maximum(vegetation_confidence, sampled[1]), vegetation_confidence)
        start = stop
    minimum_confidence = _segmentation_minimum_confidence()
    buildings = (building_confidence >= minimum_confidence).cpu().numpy()
    vegetation = (vegetation_confidence >= minimum_confidence).cpu().numpy()
    resolved_revision = getattr(model.config, "_commit_hash", None) or requested_revision
    return buildings, vegetation, analysis_width, device.type, model_id, resolved_revision


def _absorb_narrow_runs(column_owner: Any, columns: Any, minimum: int) -> None:
    """Give facade slivers narrower than ``minimum`` columns to a wider neighbor.

    A footprint that projects onto a handful of columns inside a detected
    building is usually pose or OSM noise, not a separate visible facade.
    """
    np = __import__("numpy")
    if len(columns) < 2:
        return
    values = column_owner[columns]
    breaks = np.flatnonzero((np.diff(values) != 0) | (np.diff(columns) != 1)) + 1
    runs = [list(run) for run in np.split(np.arange(len(columns)), breaks)]
    changed = True
    while changed and len(runs) > 1:
        changed = False
        widths = [len(run) for run in runs]
        index = int(np.argmin(widths))
        if widths[index] >= minimum:
            break
        neighbors = [candidate for candidate in (index - 1, index + 1) if 0 <= candidate < len(runs)]
        target = max(neighbors, key=lambda candidate: widths[candidate])
        values[runs[index]] = values[runs[target][0]]
        runs[target] = sorted(runs[target] + runs[index])
        del runs[index]
        changed = True
    column_owner[columns] = values


def _drop_fallback_slivers(labels: Any, outside_instances: Any, evidence: Any, bins: int) -> Any:
    """Remove projection-only labels that cover only a sliver of columns.

    Outside detected instances the labels come from projected footprints alone;
    a footprint showing through a few columns there is typically a pose error
    or a sky/semantic false positive rather than a visible facade.
    """
    np = __import__("numpy")
    fallback = outside_instances & evidence & (labels > 0)
    if not fallback.any():
        return labels
    width = labels.shape[1]
    minimum = max(1, round(width * INSTANCE_MIN_FACADE_DEGREES / 360))
    rows, columns = np.nonzero(fallback)
    present = np.zeros((bins, width), dtype=bool)
    present[labels[rows, columns], columns] = True
    narrow = present.sum(axis=1) < minimum
    narrow[0] = False
    if not narrow.any():
        return labels
    cleaned = labels.copy()
    cleaned[outside_instances & narrow[labels]] = 0
    return cleaned


def _instance_labels(instance_map: Any, evidence: Any, owners: Any, bins: int) -> tuple[Any, list[dict[str, Any]]]:
    """Name every visual instance's pixels with OSM ordinals.

    The instance mask decides a building's extent (roof line, sky, trees); the
    projected footprints only name it:

    * ``whole``: one footprint dominates the instance, which takes its name.
    * ``columns``: the detector merged attached facades (a block of row
      houses is one "building" to it). Walls are vertical, so facade seams
      are panorama columns: each column takes the footprint projected most
      in it and the label spans the instance's full height.
    * ``abstain``: the instance mostly lies where no footprint projects.
    """
    np = __import__("numpy")
    width = owners.shape[1]
    labels = np.zeros_like(owners)
    count = int(instance_map.max()) + 1
    votes = np.bincount(
        instance_map[evidence].astype(np.int64) * bins + owners[evidence], minlength=count * bins,
    ).reshape(count, bins)
    details = []
    for label in range(1, count):
        total = int(votes[label].sum())
        projected = int(votes[label, 1:].sum())
        if not total:
            continue
        best = int(np.argmax(votes[label, 1:])) + 1 if projected else 0
        share = float(votes[label, best] / projected) if projected else 0.0
        mask = instance_map == label
        if projected < INSTANCE_MIN_PROJECTED_SHARE * total:
            mode, ordinals = "abstain", []
        elif share >= INSTANCE_MIN_OWNER_SHARE:
            mode, ordinals = "whole", [best]
            labels[mask] = best
        else:
            mode = "columns"
            rows, columns = np.nonzero(mask & evidence & (owners > 0))
            column_votes = np.zeros((width, bins), dtype=np.int64)
            np.add.at(column_votes, (columns, owners[rows, columns]), 1)
            voted = np.flatnonzero(column_votes.sum(axis=1))
            column_owner = column_votes.argmax(axis=1)
            # Columns without projected votes take the nearest voted column.
            instance_columns = np.flatnonzero(mask.any(axis=0))
            position = np.clip(np.searchsorted(voted, instance_columns), 1, len(voted) - 1) if len(voted) > 1 else np.zeros(len(instance_columns), dtype=np.int64)
            if len(voted) > 1:
                left, right = voted[position - 1], voted[position]
                position = np.where(instance_columns - left <= right - instance_columns, position - 1, position)
            column_owner[instance_columns] = column_owner[voted[position]]
            _absorb_narrow_runs(column_owner, instance_columns, max(1, round(width * INSTANCE_MIN_FACADE_DEGREES / 360)))
            mask_rows, mask_columns = np.nonzero(mask)
            labels[mask_rows, mask_columns] = column_owner[mask_columns]
            ordinals = sorted(int(value) for value in np.unique(column_owner[instance_columns]) if value)
        details.append({
            "instance": label, "evidence_pixels": total, "mode": mode,
            "projected_share": round(projected / total, 3), "owner_share": round(share, 3), "ordinals": ordinals,
        })
    return labels, details


def _continue_upward(labels: Any, evidence: Any, owners: Any, open_space: Any) -> Any:
    """Extend each label up its column through unclaimed building pixels.

    OSM heights are often too low, and every view stops at some elevation, so
    a facade's top floors can be left unlabeled although the semantic mask
    sees them. A pixel inherits the label below it only when it is building
    evidence, no projected footprint claims it, and ``open_space`` allows it.
    """
    np = __import__("numpy")
    continued = labels.copy()
    allowed = evidence & (continued == 0) & (owners == 0) & open_space
    for row in range(continued.shape[0] - 2, -1, -1):
        take = allowed[row] & (continued[row + 1] > 0)
        if take.any():
            continued[row, take] = continued[row + 1, take]
    return continued if np.any(continued != labels) else labels


def _assign_buildings(
    buildings: Any, vegetation: Any, features: list[dict[str, Any]], camera_latitude: float, camera_longitude: float, heading_degrees: float,
    sfm_depth_samples: list[dict[str, float]] | None = None, instance_map: Any = None, instance_coverage: Any = None,
) -> tuple[Any, Any, dict[str, dict[str, Any]]]:
    return _assign_buildings_detailed(
        buildings, vegetation, features, camera_latitude, camera_longitude, heading_degrees, sfm_depth_samples,
        instance_map, instance_coverage,
    )[:3]


def _assign_buildings_detailed(
    buildings: Any, vegetation: Any, features: list[dict[str, Any]], camera_latitude: float, camera_longitude: float, heading_degrees: float,
    sfm_depth_samples: list[dict[str, float]] | None = None, instance_map: Any = None, instance_coverage: Any = None,
) -> tuple[Any, Any, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    np = __import__("numpy")
    height, width = buildings.shape
    owners, depth_buffer, candidate_features, facade_counts = _render_facade_owners(
        features, camera_latitude, camera_longitude, heading_degrees, width, height,
    )
    depth_known, foreground_conflicts, background_conflicts = _sfm_depth_conflicts(depth_buffer, sfm_depth_samples or [])
    conflicts = foreground_conflicts | background_conflicts
    bins = len(candidate_features) + 1
    selected_mask = buildings & ~conflicts
    # Image-derived instances decide where one building ends and the next
    # begins; the projected footprints only name them. Pixels outside any
    # linked instance fall back to the projection, but never onto a building
    # an instance already delimits, which would re-introduce the pose error.
    labels = owners
    instance_details: list[dict[str, Any]] = []
    if instance_map is not None and instance_map.any():
        instance_labels, instance_details = _instance_labels(instance_map, selected_mask, owners, bins)
        delimited = np.zeros(bins, dtype=bool)
        delimited[np.unique(instance_labels)] = True
        delimited[0] = False
        labels = np.where(instance_map > 0, instance_labels, np.where(delimited[owners], 0, owners)).astype(owners.dtype)
        labels = _drop_fallback_slivers(labels, instance_map == 0, selected_mask, bins)
    # Inside the instance views SAM already decided where each facade ends;
    # continuation only fills what those views could not see.
    open_space = ~instance_coverage if instance_map is not None and instance_coverage is not None else np.ones_like(buildings)
    labels = _continue_upward(labels, selected_mask, owners, open_space)
    flat_owners, flat_labels = owners.ravel(), labels.ravel()

    def counts(values: Any, mask: Any) -> Any:
        return np.bincount(values[mask.ravel()], minlength=bins)

    available_counts = np.bincount(flat_owners, minlength=bins)
    # Complete only tree-occluded portions of an otherwise visible facade.
    # Unknown and building-class pixels remain excluded to avoid selecting a
    # facade hidden behind another building or a non-modelled obstruction.
    completion_mask = vegetation & ~background_conflicts
    claimed_counts = counts(flat_labels, selected_mask)
    # Coverage of the projected facade by direct evidence carrying its label.
    agreeing_counts = counts(flat_labels, selected_mask & (labels == owners))
    inferred_counts = counts(flat_owners, completion_mask)
    known_counts = counts(flat_owners, depth_known)
    conflict_counts = counts(flat_owners, conflicts)
    accepted = np.zeros(bins, dtype=bool)
    metadata = {}
    for ordinal, feature in enumerate(candidate_features, start=1):
        available = int(available_counts[ordinal])
        claimed = int(claimed_counts[ordinal])
        if not available or claimed < max(16, int(available * 0.025)):
            continue
        accepted[ordinal] = True
        osm_id = str(feature["properties"]["osm_id"])
        metadata[str(ordinal)] = {
            "osm_id": osm_id,
            "footprint_source": str(feature["properties"].get("footprint_source", "osm")),
            "alternative_footprint_id": feature["properties"].get("alternative_footprint_id"),
            # The viewer map and raycast must use the exact OSM geometry that produced this mask.
            "footprint_geometry": feature["geometry"],
            "facades": facade_counts[osm_id],
            "pixels": claimed,
            "inferred_pixels": int(inferred_counts[ordinal]),
            "confidence": round(int(agreeing_counts[ordinal]) / max(available, 1), 3),
            "height_m": building_height(feature.get("properties", {})),
            "height_source": building_height_source(feature.get("properties", {})),
            "render_height_m": facade_render_height(feature.get("properties", {})),
            "min_height_m": facade_bottom_height(feature.get("properties", {})),
            "sfm_depth_samples": int(known_counts[ordinal]),
            "sfm_depth_conflicts": int(conflict_counts[ordinal]),
        }
        if instance_details:
            metadata[str(ordinal)]["visual_instances"] = [detail["instance"] for detail in instance_details if ordinal in detail["ordinals"]]
    zero = np.zeros((), dtype=np.uint16)
    assignments = np.where(accepted[labels] & selected_mask, labels, zero)
    vegetation_assignments = np.where(accepted[owners] & completion_mask, owners, zero)
    for detail in instance_details:
        detail["osm_ids"] = [metadata[str(ordinal)]["osm_id"] for ordinal in detail.pop("ordinals") if accepted[ordinal]]
    return assignments, vegetation_assignments, metadata, instance_details


def _shift_camera_position(latitude: float, longitude: float, east: float, north: float) -> tuple[float, float]:
    try:
        pyproj = importlib.import_module("pyproj")
    except ImportError as error:
        raise RuntimeError("Geographic projection requires pyproj. Run uv sync, then retry.") from error
    distance = math.hypot(east, north)
    if distance == 0:
        return latitude, longitude
    bearing = math.degrees(math.atan2(east, north))
    shifted_longitude, shifted_latitude, _ = pyproj.Geod(ellps="WGS84").fwd(longitude, latitude, bearing, distance)
    return float(shifted_latitude), float(shifted_longitude)


def _is_overhead(properties: dict[str, Any]) -> bool:
    building = str(properties.get("building", properties.get("building:part", ""))).strip().lower()
    return building in OVERHEAD_BUILDING_TYPES or facade_bottom_height(properties) > CAMERA_HEIGHT_M


def _pose_iou(observed_buildings: Any, scene: Any, facades: list[Any], heading_degrees: float) -> float:
    height, width = observed_buildings.shape
    projected, _, _, _ = _render_scene_owners(scene, facades, heading_degrees, width, height)
    projected = projected != 0
    union = int((observed_buildings | projected).sum())
    return int((observed_buildings & projected).sum()) / union if union else 0.0


def _refine_camera_pose(
    buildings: Any, features: list[dict[str, Any]], camera_latitude: float, camera_longitude: float, heading_degrees: float,
) -> dict[str, Any]:
    """Refine noisy Mapillary GPS/heading only when facade evidence clearly agrees."""
    geometry = _facade_geometry()
    np = __import__("numpy")
    full_scene = geometry.build_scene(features, camera_latitude, camera_longitude)
    blocking = np.array([not _is_overhead(feature.get("properties", {})) for feature in features], dtype=bool)
    repair = geometry.free_space_offset(full_scene, blocking, GPS_REPAIR_CLEARANCE_M, GPS_REPAIR_MAXIMUM_SHIFT_M)
    base_east, base_north = repair if repair is not None else (0.0, 0.0)
    gps_repair_m = round(math.hypot(base_east, base_north), 2)
    wall_distances = geometry.nearest_wall_distances(full_scene.translated(base_east, base_north))
    probe_features = [feature for feature, distance in zip(features, wall_distances) if distance <= POSE_SEARCH_MAX_FACADE_DISTANCE_M]
    base_latitude, base_longitude = _shift_camera_position(camera_latitude, camera_longitude, base_east, base_north)
    if not probe_features:
        return {
            "latitude": base_latitude, "longitude": base_longitude, "heading_degrees": heading_degrees,
            "facade_iou": 0.0, "facade_iou_gain": 0.0, "runner_up_iou": 0.0, "gps_repair_m": gps_repair_m,
            "accepted": False, "reason": "no_nearby_facades",
        }
    # A low-resolution, binary facade IoU is sufficient for a small local search
    # and avoids turning semantic segmentation into an expensive pose optimizer.
    # Wall visibility depends only on position, so it is solved once per offset
    # in the fixed local frame and reused for every heading. A camera recorded
    # inside a footprint searches around its repaired street position instead.
    observed = buildings[::8, ::8]
    probe_scene = geometry.build_scene(probe_features, camera_latitude, camera_longitude)
    base_scene = probe_scene.translated(base_east, base_north)
    original_score = _pose_iou(observed, base_scene, geometry.visible_facades(base_scene), heading_degrees)
    candidates = []
    for east in POSE_SEARCH_OFFSETS_M:
        for north in POSE_SEARCH_OFFSETS_M:
            candidate_latitude, candidate_longitude = _shift_camera_position(camera_latitude, camera_longitude, base_east + east, base_north + north)
            shifted_scene = probe_scene.translated(base_east + east, base_north + north)
            shifted_facades = geometry.visible_facades(shifted_scene)
            for heading_offset in POSE_SEARCH_HEADING_OFFSETS_DEG:
                candidate_heading = heading_degrees + heading_offset
                score = _pose_iou(observed, shifted_scene, shifted_facades, candidate_heading)
                candidates.append((
                    score, -abs(east) - abs(north), -abs(heading_offset), candidate_latitude, candidate_longitude, candidate_heading,
                    east, north, heading_offset,
                ))
    candidates.sort(reverse=True)
    best_score, _, _, best_latitude, best_longitude, best_heading, best_east, best_north, best_offset = candidates[0]
    # Ambiguity means a clearly *different* pose fits almost as well; the
    # best pose's grid neighbors always score similarly on a smooth optimum.
    runner_up = max(
        (
            candidate[0] for candidate in candidates[1:]
            if math.hypot(candidate[6] - best_east, candidate[7] - best_north) > POSE_DISTINCT_POSITION_M
            or abs(candidate[8] - best_offset) > POSE_DISTINCT_HEADING_DEG
        ),
        default=original_score,
    )
    gain = best_score - original_score
    accepted = bool(
        best_score >= POSE_MINIMUM_IOU
        and gain >= POSE_MINIMUM_IOU_GAIN
        and best_score - runner_up >= POSE_MINIMUM_RUNNER_UP_MARGIN
    )
    if not accepted:
        best_latitude, best_longitude, best_heading = base_latitude, base_longitude, heading_degrees
    if best_score < POSE_MINIMUM_IOU:
        reason = "low_absolute_fit"
    elif gain < POSE_MINIMUM_IOU_GAIN:
        reason = "insufficient_gain"
    elif best_score - runner_up < POSE_MINIMUM_RUNNER_UP_MARGIN:
        reason = "ambiguous_best_pose"
    else:
        reason = "accepted"
    return {
        "latitude": best_latitude,
        "longitude": best_longitude,
        "heading_degrees": best_heading % 360.0,
        "facade_iou": round(original_score, 4),
        "facade_iou_gain": round(gain if accepted else 0.0, 4),
        "best_facade_iou": round(best_score, 4),
        "runner_up_iou": round(runner_up, 4),
        "gps_repair_m": gps_repair_m,
        "accepted": accepted,
        "reason": reason,
    }


def preload_models(model_root: Path) -> dict[str, Any]:
    """Load every model the current configuration uses, e.g. at worker start-up."""
    _, torch, _, _ = _require_ml_dependencies()
    device = inference_device(torch)
    model_id, architecture = _segmentation_model_spec(torch)
    revision = os.environ.get("BUILDING_ANALYSIS_MODEL_REVISION", "").strip() or None
    _load_segmentation_model(model_id, architecture, str(model_root.resolve()), str(device), revision)
    settings = instance_settings()
    if settings.enabled:
        importlib.import_module("building_instances")._load_models(settings.detector, settings.segmenter, str(model_root.resolve()), str(device))
    return {"device": str(device), "segmentation_model": model_id, "instances": settings.as_configuration()}


def analysis_paths(data_root: Path, dataset_id: str, image_id: str) -> tuple[Path, Path]:
    directory = data_root / dataset_id / "analysis"
    return directory / f"{image_id}.json", directory / f"{image_id}-ids.png"


def facade_path(data_root: Path, dataset_id: str, image_id: str) -> Path:
    return data_root / dataset_id / "analysis" / f"{image_id}-facades.png"


def instance_path(data_root: Path, dataset_id: str, image_id: str) -> Path:
    return data_root / dataset_id / "analysis" / f"{image_id}-instances.png"


def analysis_configuration() -> dict[str, Any]:
    return {
        "analysis_version": ANALYSIS_VERSION,
        "model_version": MODEL_VERSION,
        "requested_model": os.environ.get("BUILDING_ANALYSIS_MODEL", "auto").strip(),
        "requested_model_revision": os.environ.get("BUILDING_ANALYSIS_MODEL_REVISION", "").strip() or None,
        "tile_size": os.environ.get("BUILDING_ANALYSIS_TILE_SIZE", "").strip(),
        "minimum_confidence": os.environ.get("BUILDING_ANALYSIS_MIN_CONFIDENCE", "0.30").strip(),
        "camera_height_m": CAMERA_HEIGHT_M,
        "perspective_yaws": len(_HORIZONTAL_YAWS),
        "perspective_pitch_degrees": list(PERSPECTIVE_PITCHES_DEG),
        "perspective_fov_degrees": PERSPECTIVE_FOV_DEGREES,
        "instances": instance_settings().as_configuration(),
    }


def _dataset_image(data_root: Path, dataset_id: str, image_id: str) -> dict[str, Any] | None:
    try:
        manifest = json.loads((data_root / dataset_id / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    images = manifest.get("images") if isinstance(manifest, dict) else None
    if not isinstance(images, list):
        return None
    return next((image for image in images if isinstance(image, dict) and str(image.get("id")) == image_id), None)


def publish_analysis_artifacts(
    data_root: Path, dataset_id: str, image_id: str, metadata: dict[str, Any], id_mask: bytes, facade_mask: bytes,
    instance_mask: bytes | None = None,
) -> dict[str, Any]:
    import hashlib

    published = dict(metadata)
    published["artifact_sha256"] = {
        "id_mask": hashlib.sha256(id_mask).hexdigest(),
        "facade_mask": hashlib.sha256(facade_mask).hexdigest(),
    }
    metadata_path, id_path = analysis_paths(data_root, dataset_id, image_id)
    atomic_write_bytes(id_path, id_mask)
    atomic_write_bytes(facade_path(data_root, dataset_id, image_id), facade_mask)
    if instance_mask is not None:
        published["artifact_sha256"]["instance_mask"] = hashlib.sha256(instance_mask).hexdigest()
        atomic_write_bytes(instance_path(data_root, dataset_id, image_id), instance_mask)
    atomic_write_bytes(metadata_path, (json.dumps(published, indent=2) + "\n").encode("utf-8"))
    return published


def _analysis_artifacts_match(
    payload: dict[str, Any], id_path: Path, facade_mask_path: Path, instance_mask_path: Path | None = None,
) -> bool:
    width, height = payload.get("width"), payload.get("height")
    hashes = payload.get("artifact_sha256")
    if not isinstance(width, int) or not isinstance(height, int) or not isinstance(hashes, dict):
        return False
    masks = {"id_mask": id_path, "facade_mask": facade_mask_path}
    if "instance_mask" in hashes:
        if instance_mask_path is None:
            return False
        masks["instance_mask"] = instance_mask_path
    try:
        if any(not path.is_file() or hashes.get(name) != file_sha256(path) for name, path in masks.items()):
            return False
        Image = importlib.import_module("PIL.Image")
        for path in masks.values():
            with Image.open(path) as source:
                if source.size != (width, height):
                    return False
                source.verify()
    except (OSError, SyntaxError):
        return False
    return True


def load_analysis(data_root: Path, dataset_id: str, image_id: str) -> dict[str, Any] | None:
    metadata_path, id_path = analysis_paths(data_root, dataset_id, image_id)
    facade_mask_path = facade_path(data_root, dataset_id, image_id)
    if not metadata_path.is_file() or not id_path.is_file() or not facade_mask_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    image = _dataset_image(data_root, dataset_id, image_id)
    if not isinstance(payload, dict) or image is None:
        return None
    try:
        input_sha256 = analysis_input_sha256(data_root, dataset_id, image, analysis_configuration())
    except (OSError, ValueError):
        return None
    return payload if (
        analysis_cache_matches(payload, data_root, dataset_id, image_id, ANALYSIS_VERSION, input_sha256)
        and _analysis_artifacts_match(payload, id_path, facade_mask_path, instance_path(data_root, dataset_id, image_id))
    ) else None


def analyze_panorama(
    data_root: Path,
    model_root: Path,
    dataset_id: str,
    image: dict[str, Any],
    progress: Callable[[str], None],
    sfm_refinement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_id = str(image["id"])
    geometry = image.get("selected_geometry") or image.get("computed_geometry") or image.get("geometry") or {}
    coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        raise ValueError("The panorama has no coordinates.")
    local_file = image.get("local_file")
    if not isinstance(local_file, str):
        raise ValueError("The panorama has no local image file.")
    image_path = (data_root / dataset_id / local_file).resolve()
    if not image_path.is_file() or not image_path.is_relative_to((data_root / dataset_id).resolve()):
        raise ValueError("The panorama image is unavailable.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in coordinates[:2]):
        raise ValueError("The panorama coordinates are invalid.")
    try:
        longitude, latitude = float(coordinates[0]), float(coordinates[1])
        raw_heading = image["computed_compass_angle_deg"]
        if isinstance(raw_heading, bool):
            raise ValueError
        heading = float(raw_heading)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("The panorama has no valid compass heading.") from error
    if not math.isfinite(longitude) or not math.isfinite(latitude) or not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
        raise ValueError("The panorama coordinates are invalid.")
    if not math.isfinite(heading):
        raise ValueError("The panorama has no valid compass heading.")
    heading %= 360.0
    if sfm_refinement is None:
        from .sfm_refinement import load_sfm_refinement, sfm_path

        if load_sfm_refinement(data_root, dataset_id, image_id) is None:
            sfm_path(data_root, dataset_id, image_id).unlink(missing_ok=True)
    input_sha256 = analysis_input_sha256(data_root, dataset_id, image, analysis_configuration())
    # Footprints come from the network on a cache miss; fetch them while the
    # model segments instead of after it.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="osm-footprints")
    try:
        pending_features = pool.submit(nearby_buildings, latitude, longitude)
        buildings, vegetation, width, device, model_id, model_revision = _segment_buildings(image_path, model_root, progress)
        progress("Loading nearby OpenStreetMap building footprints...")
        features = pending_features.result()
    finally:
        # A failed segmentation must not wait for a slow footprint server.
        pool.shutdown(wait=False, cancel_futures=True)
    settings = instance_settings()
    instance_map, detected_instances, coverage = None, [], None
    if settings.enabled:
        _, torch, _, _ = _require_ml_dependencies()
        instance_map, detected_instances = segment_instances(
            _load_analysis_panorama(image_path), model_root, inference_device(torch), settings, progress,
        )
        coverage = instance_coverage(settings, instance_map.shape[1], instance_map.shape[0])
    progress("Refining camera pose against projected building facades...")
    # Detected instances exclude sky and trees the semantic mask can confuse
    # with facades, which makes a cleaner target for the pose search.
    pose_target = buildings & (instance_map > 0) if instance_map is not None and instance_map.any() else buildings
    refined_pose = _refine_camera_pose(pose_target, features, latitude, longitude, heading)
    progress("Associating building masks with OpenStreetMap footprints...")
    assignments, facade_assignments, building_ids, instance_links = _assign_buildings_detailed(
        buildings, vegetation, features,
        float(refined_pose["latitude"]), float(refined_pose["longitude"]), float(refined_pose["heading_degrees"]),
        _sfm_depth_samples_from_payload(sfm_refinement, image_id) if sfm_refinement is not None
        else _sfm_depth_samples(data_root, dataset_id, image_id),
        instance_map, coverage,
    )
    links_by_instance = {link["instance"]: link for link in instance_links}
    visual_instances = [{**detected, **links_by_instance.get(detected["instance"], {"osm_ids": []})} for detected in detected_instances]
    Image = importlib.import_module("PIL.Image")
    id_output, facade_output = io.BytesIO(), io.BytesIO()
    Image.fromarray(assignments, mode="I;16").save(id_output, format="PNG")
    Image.fromarray(facade_assignments, mode="I;16").save(facade_output, format="PNG")
    instance_output = None
    if instance_map is not None:
        instance_output = io.BytesIO()
        Image.fromarray(instance_map.astype("uint16"), mode="I;16").save(instance_output, format="PNG")
    metadata = {
        "version": ANALYSIS_VERSION,
        "image_id": image_id,
        "width": width,
        "height": int(assignments.shape[0]),
        "model": model_id,
        "model_revision": model_revision,
        "model_version": MODEL_VERSION,
        "analysis_configuration": analysis_configuration(),
        "segmentation_minimum_confidence": _segmentation_minimum_confidence(),
        "device": device,
        "input_sha256": input_sha256,
        "sfm_input_sha256": sfm_input_sha256(data_root, dataset_id, image_id),
        "heading_degrees": heading,
        "refined_pose": refined_pose,
        "footprint_sources": {
            "osm_candidates": len(features),
        },
        "building_ids": building_ids,
        "visual_instances": visual_instances,
    }
    metadata = publish_analysis_artifacts(
        data_root, dataset_id, image_id, metadata, id_output.getvalue(), facade_output.getvalue(),
        instance_output.getvalue() if instance_output is not None else None,
    )
    progress(f"Analysis complete: {len(building_ids)} buildings matched.")
    return metadata


@lru_cache(maxsize=8)
def _decoded_mask(path: str, sha256: str) -> Any:
    """Decoded uint16 mask; ``sha256`` is the digest load_analysis just validated."""
    Image = importlib.import_module("PIL.Image")
    with Image.open(path) as source:
        values = __import__("numpy").asarray(source)
    values.setflags(write=False)
    return values


def _validated_mask(data_root: Path, dataset_id: str, image_id: str, metadata: dict[str, Any], inferred: bool) -> Any:
    _, id_path = analysis_paths(data_root, dataset_id, image_id)
    path = facade_path(data_root, dataset_id, image_id) if inferred else id_path
    return _decoded_mask(str(path), str(metadata["artifact_sha256"]["facade_mask" if inferred else "id_mask"]))


@lru_cache(maxsize=64)
def _encoded_building_mask(path: str, sha256: str, ordinal: int | None) -> bytes:
    np = __import__("numpy")
    Image = importlib.import_module("PIL.Image")
    values = _decoded_mask(path, sha256)
    mask = np.where(values > 0 if ordinal is None else values == ordinal, 255, 0).astype("uint8")
    output = io.BytesIO()
    # Served to the browser only; fast compression keeps selection responsive.
    Image.fromarray(mask, mode="L").save(output, format="PNG", compress_level=1)
    return output.getvalue()


def building_mask(data_root: Path, dataset_id: str, image_id: str, osm_id: str, inferred: bool = False) -> bytes | None:
    metadata = load_analysis(data_root, dataset_id, image_id)
    if not metadata:
        return None
    ordinal = next((int(key) for key, item in metadata.get("building_ids", {}).items() if item.get("osm_id") == osm_id), None)
    if osm_id and ordinal is None:
        return None
    _, id_path = analysis_paths(data_root, dataset_id, image_id)
    path = facade_path(data_root, dataset_id, image_id) if inferred else id_path
    return _encoded_building_mask(str(path), str(metadata["artifact_sha256"]["facade_mask" if inferred else "id_mask"]), ordinal)


def pick_building(data_root: Path, dataset_id: str, image_id: str, u: float, v: float) -> dict[str, Any] | None:
    metadata = load_analysis(data_root, dataset_id, image_id)
    if not metadata or not 0 <= u < 1 or not 0 <= v < 1:
        return None
    values = _validated_mask(data_root, dataset_id, image_id, metadata, inferred=False)
    height, width = values.shape
    row, column = min(height - 1, int(v * height)), min(width - 1, int(u * width))
    building = metadata.get("building_ids", {}).get(str(int(values[row, column])))
    instance_sha256 = metadata["artifact_sha256"].get("instance_mask")
    if building is None or instance_sha256 is None:
        return building
    instance = int(_decoded_mask(str(instance_path(data_root, dataset_id, image_id)), str(instance_sha256))[row, column])
    return {**building, "visual_instance": instance or None}
