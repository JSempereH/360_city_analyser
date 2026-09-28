"""Vectorized OSM facade geometry in a camera-centered metric frame.

Every footprint is projected once into an azimuthal-equidistant frame around the
recorded camera. Visibility, shared-wall rejection and z-buffer rasterization
then operate on NumPy edge arrays, and pose candidates are simple translations
of that frame instead of fresh geographic projections.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np


# Tolerances inherited from the scalar implementation; they are part of the
# analysis contract because they decide which walls are considered visible.
MINIMUM_FACING_DISTANCE_M = 0.5
FACING_SAMPLE_DISTANCE_M = 0.2
SHARED_WALL_TOLERANCE_M = 0.25
VISIBILITY_SAMPLE_SPACING_M = 1.5
MAXIMUM_VISIBILITY_SAMPLES = 64
RAY_FRACTION_EPSILON = 1e-5
PARALLEL_EPSILON = 1e-9
MINIMUM_RENDER_DISTANCE_M = 0.25
MINIMUM_RENDER_EDGE_M = 0.1
# Upper bound on (samples x edges) booleans evaluated at once.
OCCLUSION_CHUNK_ELEMENTS = 4_000_000
# Open space must reach this far to count as street rather than a courtyard.
STREET_CONNECTIVITY_RADIUS_M = 40.0


def geometry_polygons(feature: dict[str, Any]) -> list[list[list[list[float]]]]:
    """Return validated, closed lon/lat rings grouped by polygon."""
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        return []
    coordinates = geometry.get("coordinates")
    if geometry.get("type") == "Polygon":
        candidates = [coordinates]
    elif geometry.get("type") == "MultiPolygon":
        candidates = coordinates
    else:
        return []
    if not isinstance(candidates, list):
        return []
    polygons = []
    for polygon in candidates:
        if not isinstance(polygon, list) or not polygon:
            continue
        rings = []
        for ring in polygon:
            if not isinstance(ring, list) or len(ring) < 4:
                continue
            normalized = []
            for point in ring:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    normalized = []
                    break
                try:
                    longitude, latitude = float(point[0]), float(point[1])
                except (TypeError, ValueError):
                    normalized = []
                    break
                if not math.isfinite(longitude) or not math.isfinite(latitude):
                    normalized = []
                    break
                normalized.append([longitude, latitude])
            if normalized:
                if normalized[0] != normalized[-1]:
                    normalized.append(normalized[0])
                rings.append(normalized)
        if rings:
            polygons.append(rings)
    return polygons


@lru_cache(maxsize=128)
def local_transformer(camera_latitude: float, camera_longitude: float) -> Any:
    try:
        pyproj = importlib.import_module("pyproj")
    except ImportError as error:
        raise RuntimeError("Geographic projection requires pyproj. Run uv sync, then retry.") from error
    local = pyproj.CRS.from_proj4(
        f"+proj=aeqd +lat_0={camera_latitude:.10f} +lon_0={camera_longitude:.10f} +datum=WGS84 +units=m +no_defs"
    )
    return pyproj.Transformer.from_crs("EPSG:4326", local, always_xy=True)


def project_points(camera_latitude: float, camera_longitude: float, points: np.ndarray) -> np.ndarray:
    """Project an (N, 2) lon/lat array to local east/north meters in one call."""
    if not len(points):
        return np.zeros((0, 2), dtype=np.float64)
    east, north = local_transformer(camera_latitude, camera_longitude).transform(points[:, 0], points[:, 1])
    return np.column_stack((np.asarray(east, dtype=np.float64), np.asarray(north, dtype=np.float64)))


@dataclass(frozen=True)
class FootprintScene:
    """OSM rings and wall edges expressed relative to the camera at (0, 0)."""

    features: tuple[dict[str, Any], ...]
    # Closed rings per polygon; ``polygon_feature[p]`` owns ``polygons[p]``.
    polygons: tuple[tuple[np.ndarray, ...], ...]
    polygon_feature: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    edge_feature: np.ndarray
    edge_polygon: np.ndarray

    def translated(self, east: float, north: float) -> FootprintScene:
        """Re-center the frame on a camera displaced by (east, north) meters."""
        offset = np.array([east, north], dtype=np.float64)
        return FootprintScene(
            self.features,
            tuple(tuple(ring - offset for ring in polygon) for polygon in self.polygons),
            self.polygon_feature,
            self.starts - offset,
            self.ends - offset,
            self.edge_feature,
            self.edge_polygon,
        )


def build_scene(features: list[dict[str, Any]], camera_latitude: float, camera_longitude: float) -> FootprintScene:
    raw_polygons: list[tuple[int, list[list[list[float]]]]] = [
        (feature_index, polygon)
        for feature_index, feature in enumerate(features)
        for polygon in geometry_polygons(feature)
    ]
    ring_lengths = [len(ring) for _, polygon in raw_polygons for ring in polygon]
    flat = np.array([point for _, polygon in raw_polygons for ring in polygon for point in ring], dtype=np.float64).reshape(-1, 2)
    projected = project_points(camera_latitude, camera_longitude, flat)
    rings = np.split(projected, np.cumsum(ring_lengths)[:-1]) if ring_lengths else []
    polygons, polygon_feature = [], []
    starts, ends, edge_feature, edge_polygon = [], [], [], []
    ring_cursor = 0
    for polygon_index, (feature_index, polygon) in enumerate(raw_polygons):
        local_rings = tuple(rings[ring_cursor:ring_cursor + len(polygon)])
        ring_cursor += len(polygon)
        polygons.append(local_rings)
        polygon_feature.append(feature_index)
        for ring in local_rings:
            starts.append(ring[:-1])
            ends.append(ring[1:])
            edge_feature.append(np.full(len(ring) - 1, feature_index, dtype=np.int32))
            edge_polygon.append(np.full(len(ring) - 1, polygon_index, dtype=np.int32))
    empty_points = np.zeros((0, 2), dtype=np.float64)
    empty_index = np.zeros(0, dtype=np.int32)
    return FootprintScene(
        tuple(features),
        tuple(polygons),
        np.asarray(polygon_feature, dtype=np.int32),
        np.concatenate(starts) if starts else empty_points,
        np.concatenate(ends) if ends else empty_points,
        np.concatenate(edge_feature) if edge_feature else empty_index,
        np.concatenate(edge_polygon) if edge_polygon else empty_index,
    )


def _points_in_ring(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd crossing test of many points against one closed ring."""
    first, second = ring[:-1], ring[1:]
    point_east, point_north = points[:, :1], points[:, 1:]
    straddles = (first[:, 1] > point_north) != (second[:, 1] > point_north)
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing_east = (second[:, 0] - first[:, 0]) * (point_north - first[:, 1]) / (second[:, 1] - first[:, 1]) + first[:, 0]
    return np.count_nonzero(straddles & (point_east < crossing_east), axis=1) % 2 == 1


def _points_in_polygon(points: np.ndarray, polygon: tuple[np.ndarray, ...]) -> np.ndarray:
    inside = _points_in_ring(points, polygon[0])
    for hole in polygon[1:]:
        inside &= ~_points_in_ring(points, hole)
    return inside


def facing_edges(scene: FootprintScene) -> np.ndarray:
    """Keep exterior walls whose outside faces the camera, including concave outlines."""
    midpoints = (scene.starts + scene.ends) / 2
    distances = np.hypot(midpoints[:, 0], midpoints[:, 1])
    lengths = np.hypot(*(scene.ends - scene.starts).T)
    facing = np.zeros(len(scene.starts), dtype=bool)
    candidates = distances >= MINIMUM_FACING_DISTANCE_M
    if not candidates.any():
        return facing
    safe_distances = np.where(candidates, distances, 1.0)
    offsets = np.minimum(np.minimum(FACING_SAMPLE_DISTANCE_M, distances / 4), lengths / 4)
    toward_camera = -midpoints / safe_distances[:, None]
    exterior = midpoints + toward_camera * offsets[:, None]
    interior = midpoints - toward_camera * offsets[:, None]
    for polygon_index in np.unique(scene.edge_polygon[candidates]):
        selected = np.flatnonzero(candidates & (scene.edge_polygon == polygon_index))
        polygon = scene.polygons[polygon_index]
        facing[selected] = ~_points_in_polygon(exterior[selected], polygon) & _points_in_polygon(interior[selected], polygon)
    return facing


def shared_with_other_feature(scene: FootprintScene, feature_index: int, start: np.ndarray, end: np.ndarray) -> bool:
    """True when another footprint's wall covers nearly all of this wall line."""
    direction = end - start
    length = float(np.hypot(*direction))
    if length < MINIMUM_FACING_DISTANCE_M:
        return False
    others = scene.edge_feature != feature_index
    first = scene.starts[others] - start
    second = scene.ends[others] - start
    first_offset = np.abs(first[:, 0] * direction[1] - first[:, 1] * direction[0]) / length
    second_offset = np.abs(second[:, 0] * direction[1] - second[:, 1] * direction[0]) / length
    collinear = np.maximum(first_offset, second_offset) <= SHARED_WALL_TOLERANCE_M
    first_projection = first @ direction / length
    second_projection = second @ direction / length
    overlap = np.maximum(
        0.0,
        np.minimum(length, np.maximum(first_projection, second_projection))
        - np.maximum(0.0, np.minimum(first_projection, second_projection)),
    )
    return bool(np.any(collinear & (overlap >= length - SHARED_WALL_TOLERANCE_M)))


def _clear_lines_of_sight(scene: FootprintScene, targets: np.ndarray) -> np.ndarray:
    """Whether the segment camera->target crosses no footprint edge before the target."""
    clear = np.ones(len(targets), dtype=bool)
    if not len(targets) or not len(scene.starts):
        return clear
    edges = scene.ends - scene.starts
    anchor_cross_edge = scene.starts[:, 0] * edges[:, 1] - scene.starts[:, 1] * edges[:, 0]
    chunk = max(1, OCCLUSION_CHUNK_ELEMENTS // len(edges))
    for offset in range(0, len(targets), chunk):
        target = targets[offset:offset + chunk]
        target_east, target_north = target[:, :1], target[:, 1:]
        denominator = target_east * edges[:, 1] - target_north * edges[:, 0]
        valid = np.abs(denominator) >= PARALLEL_EPSILON
        with np.errstate(divide="ignore", invalid="ignore"):
            ray_fraction = anchor_cross_edge / denominator
            edge_fraction = (scene.starts[:, 0] * target_north - scene.starts[:, 1] * target_east) / denominator
        blocked = (
            valid
            & (ray_fraction > RAY_FRACTION_EPSILON) & (ray_fraction < 1 - RAY_FRACTION_EPSILON)
            & (edge_fraction >= 0) & (edge_fraction <= 1)
        )
        clear[offset:offset + chunk] = ~blocked.any(axis=1)
    return clear


@dataclass(frozen=True)
class VisibleFacade:
    feature_index: int
    start: np.ndarray
    end: np.ndarray
    distance: float


def visible_facades(scene: FootprintScene) -> list[VisibleFacade]:
    """Every exterior, unshared wall interval with an unobstructed sight line."""
    facing = facing_edges(scene)
    candidates = [
        index for index in np.flatnonzero(facing)
        if not shared_with_other_feature(scene, int(scene.edge_feature[index]), scene.starts[index], scene.ends[index])
    ]
    if not candidates:
        return []
    starts, ends = scene.starts[candidates], scene.ends[candidates]
    lengths = np.hypot(*(ends - starts).T)
    counts = np.clip(np.ceil(lengths / VISIBILITY_SAMPLE_SPACING_M).astype(np.int64), 1, MAXIMUM_VISIBILITY_SAMPLES)
    owners = np.repeat(np.arange(len(candidates)), counts)
    local_index = np.arange(len(owners)) - np.repeat(np.cumsum(counts) - counts, counts)
    fractions = (local_index + 0.5) / counts[owners]
    samples = starts[owners] + (ends - starts)[owners] * fractions[:, None]
    clear = np.split(_clear_lines_of_sight(scene, samples), np.cumsum(counts)[:-1])
    facades = []
    for position, edge_index in enumerate(candidates):
        flags, total = clear[position], int(counts[position])
        start, direction = starts[position], ends[position] - starts[position]
        # Collapse runs of clear samples into visible wall intervals.
        padded = np.concatenate(([False], flags, [False]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        for run_start, run_end in zip(changes[::2], changes[1::2]):
            first = start + direction * (run_start / total)
            second = start + direction * (run_end / total)
            midpoint = (first + second) / 2
            facades.append(VisibleFacade(int(scene.edge_feature[edge_index]), first, second, float(np.hypot(*midpoint))))
    return facades


def nearest_wall_distances(scene: FootprintScene) -> np.ndarray:
    """Distance from the camera to each feature's closest wall (inf when none)."""
    distances = np.full(len(scene.features), np.inf)
    if not len(scene.starts):
        return distances
    edges = scene.ends - scene.starts
    squared = np.einsum("ij,ij->i", edges, edges)
    nonzero = squared > 0
    fractions = np.zeros(len(edges))
    fractions[nonzero] = np.clip(-np.einsum("ij,ij->i", scene.starts[nonzero], edges[nonzero]) / squared[nonzero], 0, 1)
    nearest = scene.starts + edges * fractions[:, None]
    edge_distances = np.where(nonzero, np.hypot(nearest[:, 0], nearest[:, 1]), np.inf)
    np.minimum.at(distances, scene.edge_feature, edge_distances)
    return distances


def _wrap_radians(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def render_facade(
    depth_buffer: np.ndarray, owner_buffer: np.ndarray, owner: int, facade: VisibleFacade,
    heading_degrees: float, wall_top_m: float, wall_bottom_m: float,
) -> None:
    """Rasterize one vertical wall with an exact per-column ray/segment depth.

    The horizontal distance changes across an oblique facade. Storing the
    intersection distance per panorama column avoids the ordering errors of
    one mean distance per projected trapezoid. ``wall_*_m`` are relative to
    the camera height.
    """
    height, width = depth_buffer.shape
    (first_east, first_north), (second_east, second_north) = facade.start, facade.end
    edge_east, edge_north = second_east - first_east, second_north - first_north
    if math.hypot(edge_east, edge_north) < MINIMUM_RENDER_EDGE_M:
        return
    heading = math.radians(heading_degrees)
    first_angle = _wrap_radians(math.atan2(first_east, first_north) - heading)
    second_angle = first_angle + _wrap_radians(math.atan2(second_east, second_north) - heading - first_angle)
    if abs(second_angle - first_angle) >= math.pi:
        return
    first_column = (first_angle / (2 * math.pi) + 0.5) * width
    second_column = first_column + (second_angle - first_angle) / (2 * math.pi) * width
    start_column = math.floor(min(first_column, second_column))
    end_column = math.ceil(max(first_column, second_column))
    if end_column - start_column >= width:
        return
    columns = np.arange(start_column, end_column + 1, dtype=np.int32)
    relative_angles = ((columns + 0.5) / width - 0.5) * 2 * math.pi
    ray_east = np.sin(heading + relative_angles)
    ray_north = np.cos(heading + relative_angles)
    denominator = ray_east * edge_north - ray_north * edge_east
    valid = np.abs(denominator) > PARALLEL_EPSILON
    distances = np.full(columns.shape, np.inf, dtype=np.float32)
    fractions = np.full(columns.shape, -1.0, dtype=np.float32)
    distances[valid] = (first_east * edge_north - first_north * edge_east) / denominator[valid]
    fractions[valid] = (first_east * ray_north[valid] - first_north * ray_east[valid]) / denominator[valid]
    valid &= (distances > MINIMUM_RENDER_DISTANCE_M) & (fractions >= -1e-5) & (fractions <= 1.00001)
    if not valid.any():
        return
    columns, distances = columns[valid] % width, distances[valid]
    column_distances = distances.astype(np.float64)
    tops = np.maximum(0, np.ceil((0.5 - np.arctan2(wall_top_m, column_distances) / math.pi) * height)).astype(np.int64)
    bottoms = np.minimum(height - 1, np.floor((0.5 - np.arctan2(wall_bottom_m, column_distances) / math.pi) * height)).astype(np.int64)
    drawn = bottoms >= tops
    if not drawn.any():
        return
    columns, distances, tops, bottoms = columns[drawn], distances[drawn], tops[drawn], bottoms[drawn]
    first_row, last_row = int(tops.min()), int(bottoms.max())
    rows = np.arange(first_row, last_row + 1)[:, None]
    # Columns are unique modulo the width, so this gather/scatter is safe.
    window = np.ix_(np.arange(first_row, last_row + 1), columns)
    current = depth_buffer[window]
    closer = (rows >= tops) & (rows <= bottoms) & (distances < current)
    if not closer.any():
        return
    depth_buffer[window] = np.where(closer, distances, current)
    owner_buffer[window] = np.where(closer, owner, owner_buffer[window])


def free_space_offset(
    scene: FootprintScene, blocking_features: np.ndarray, clearance_m: float, maximum_shift_m: float,
) -> tuple[float, float] | None:
    """Smallest (east, north) move that puts the camera in open space.

    Street-level GPS in urban canyons often lands a few meters inside a
    footprint, from where no wall faces the camera. Returns ``(0, 0)`` when
    the camera is already clear, the nearest point at least ``clearance_m``
    from every blocking footprint otherwise, and ``None`` when no such point
    exists within ``maximum_shift_m``.
    """
    try:
        geometry = importlib.import_module("shapely.geometry")
        operations = importlib.import_module("shapely.ops")
    except ImportError as error:
        raise RuntimeError("Camera position repair requires Shapely. Run uv sync, then retry.") from error
    polygons = [
        geometry.Polygon(rings[0], rings[1:]).buffer(0)
        for feature_index, rings in zip(scene.polygon_feature, scene.polygons)
        if blocking_features[feature_index]
    ]
    if not polygons:
        return 0.0, 0.0
    obstacles = operations.unary_union(polygons).buffer(clearance_m)
    camera = geometry.Point(0.0, 0.0)
    if not obstacles.contains(camera):
        return 0.0, 0.0
    # Courtyards and light wells are open space too, but enclosed by the block.
    # Keep only open space that reaches the edge of a wider disk, i.e. streets.
    neighborhood = camera.buffer(STREET_CONNECTIVITY_RADIUS_M)
    open_space = neighborhood.difference(obstacles)
    components = getattr(open_space, "geoms", [open_space])
    streets = [component for component in components if not component.is_empty and component.intersects(neighborhood.exterior)]
    free = operations.unary_union(streets).intersection(camera.buffer(maximum_shift_m)) if streets else None
    if free is None or free.is_empty:
        return None
    nearest = operations.nearest_points(free, camera)[0]
    return float(nearest.x), float(nearest.y)
