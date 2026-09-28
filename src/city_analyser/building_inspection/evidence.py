"""Read and validate per-panorama evidence for one building."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ..analysis_cache import file_sha256
from ..building_analysis import load_analysis

from .models import EvidenceCollection, InputIssue, ViewEvidence


class BuildingEvidenceNotFound(ValueError):
    """Raised when no current analysis directly associates the requested building."""


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _nonnegative_integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _coordinates(image: dict[str, Any]) -> tuple[float, float] | None:
    geometry = image.get("selected_geometry") or image.get("computed_geometry") or image.get("geometry")
    coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        return None
    longitude, latitude = _number(coordinates[0]), _number(coordinates[1])
    if longitude is None or latitude is None or not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
        return None
    return longitude, latitude


def _mask_frame(path: Path, ordinal: int, expected_width: int, expected_height: int) -> tuple[float, float, float, int]:
    np = __import__("numpy")
    Image = __import__("PIL.Image", fromlist=["Image"])
    with Image.open(path) as source:
        values = np.asarray(source)
    if values.shape != (expected_height, expected_width):
        raise ValueError("building ID mask dimensions do not match analysis metadata")
    rows, columns = np.where(values == ordinal)
    if not len(columns):
        raise ValueError("building ID mask contains no pixels for its metadata ordinal")
    angles = ((columns.astype(np.float64) + 0.5) / expected_width - 0.5) * 2 * math.pi
    mean_angle = math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
    offsets = (angles - mean_angle + math.pi) % (2 * math.pi) - math.pi
    minimum_offset, maximum_offset = float(offsets.min()), float(offsets.max())
    yaw = mean_angle + (minimum_offset + maximum_offset) / 2
    yaw_deg = (math.degrees(yaw) + 180) % 360 - 180
    pitches = (0.5 - (rows.astype(np.float64) + 0.5) / expected_height) * 180
    minimum_pitch, maximum_pitch = float(pitches.min()), float(pitches.max())
    pitch_deg = (minimum_pitch + maximum_pitch) / 2
    horizontal_span = math.degrees(maximum_offset - minimum_offset)
    vertical_span = maximum_pitch - minimum_pitch
    fov_deg = max(45.0, min(100.0, max(horizontal_span, vertical_span) * 1.25))
    return round(yaw_deg, 2), round(max(-75.0, min(75.0, pitch_deg)), 2), round(fov_deg, 2), len(columns)


def _valid_geometry(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("type") not in {"Polygon", "MultiPolygon"}:
        return None
    coordinates = value.get("coordinates")
    polygons = [coordinates] if value.get("type") == "Polygon" else coordinates
    if not isinstance(polygons, list) or not polygons:
        return None
    if any(
        not isinstance(polygon, list) or not polygon or not isinstance(polygon[0], list) or len(polygon[0]) < 4
        for polygon in polygons
    ):
        return None
    return value


def collect_building_evidence(
    dataset_dir: Path,
    manifest: dict[str, Any],
    building_id: str,
    *,
    expected_analysis_version: int,
) -> EvidenceCollection:
    images = [
        image for image in manifest.get("images", [])
        if isinstance(image, dict) and image.get("is_pano") is True and isinstance(image.get("local_file"), str) and image.get("id") is not None
    ]
    images.sort(key=lambda image: (image.get("captured_at_ms") or 0, str(image["id"])))
    views: list[ViewEvidence] = []
    issues: list[InputIssue] = []
    compatible = stale = incomplete = invalid = 0
    analysis_hashes: list[str] = []
    analysis_dir = dataset_dir / "analysis"

    for image in images:
        image_id = str(image["id"])
        metadata_path = analysis_dir / f"{image_id}.json"
        if not metadata_path.is_file():
            continue
        id_mask_path = analysis_dir / f"{image_id}-ids.png"
        inferred_mask_path = analysis_dir / f"{image_id}-facades.png"
        if not id_mask_path.is_file() or not inferred_mask_path.is_file():
            incomplete += 1
            issues.append(InputIssue("incomplete_analysis", image_id, "Analysis metadata exists without both mask artifacts."))
            continue
        try:
            raw = metadata_path.read_bytes()
            analysis = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            invalid += 1
            issues.append(InputIssue("invalid_analysis", image_id, "Analysis metadata is unreadable or invalid JSON."))
            continue
        if not isinstance(analysis, dict):
            invalid += 1
            issues.append(InputIssue("invalid_analysis", image_id, "Analysis metadata is not an object."))
            continue
        if analysis.get("version") != expected_analysis_version:
            stale += 1
            continue
        if str(analysis.get("image_id")) != image_id:
            invalid += 1
            issues.append(InputIssue("image_id_mismatch", image_id, "Analysis image_id does not match the manifest."))
            continue
        if load_analysis(dataset_dir.parent, dataset_dir.name, image_id) is None:
            stale += 1
            issues.append(InputIssue("stale_analysis_inputs", image_id, "Analysis inputs changed after this artifact was generated."))
            continue
        width, height = analysis.get("width"), analysis.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            invalid += 1
            issues.append(InputIssue("invalid_dimensions", image_id, "Analysis dimensions are invalid."))
            continue
        building_ids = analysis.get("building_ids")
        if not isinstance(building_ids, dict):
            invalid += 1
            issues.append(InputIssue("invalid_buildings", image_id, "Analysis building_ids is not an object."))
            continue
        compatible += 1
        matches = [(ordinal, item) for ordinal, item in building_ids.items() if isinstance(item, dict) and item.get("osm_id") == building_id]
        if not matches:
            continue
        if len(matches) != 1:
            invalid += 1
            issues.append(InputIssue("duplicate_building", image_id, "Analysis contains the requested building more than once."))
            continue
        ordinal_text, building = matches[0]
        try:
            ordinal = int(ordinal_text)
        except (TypeError, ValueError):
            invalid += 1
            issues.append(InputIssue("invalid_ordinal", image_id, "Building mask ordinal is invalid."))
            continue
        geometry = _valid_geometry(building.get("footprint_geometry"))
        coordinates = _coordinates(image)
        recorded_heading = _number(image.get("computed_compass_angle_deg"))
        observed_pixels = _nonnegative_integer(building.get("pixels"))
        inferred_pixels = _nonnegative_integer(building.get("inferred_pixels", 0))
        visible_facades = _nonnegative_integer(building.get("facades", 0))
        projection_coverage = _number(building.get("confidence"))
        sfm_samples = _nonnegative_integer(building.get("sfm_depth_samples", 0))
        sfm_conflicts = _nonnegative_integer(building.get("sfm_depth_conflicts", 0))
        if (
            geometry is None or coordinates is None or recorded_heading is None or observed_pixels is None or inferred_pixels is None
            or visible_facades is None or projection_coverage is None or not 0 <= projection_coverage <= 1
            or sfm_samples is None or sfm_conflicts is None or sfm_conflicts > sfm_samples
        ):
            invalid += 1
            issues.append(InputIssue("invalid_building_evidence", image_id, "Building evidence is missing valid geometry, pose, or metrics."))
            continue
        try:
            yaw_deg, pitch_deg, fov_deg, mask_pixels = _mask_frame(id_mask_path, ordinal, width, height)
        except (OSError, ValueError) as error:
            invalid += 1
            issues.append(InputIssue("invalid_mask", image_id, str(error)))
            continue
        if mask_pixels != observed_pixels:
            issues.append(InputIssue("pixel_count_mismatch", image_id, f"Mask has {mask_pixels} pixels; metadata reports {observed_pixels}."))
            observed_pixels = mask_pixels
        panorama_path = (dataset_dir / str(image["local_file"])).resolve()
        if not panorama_path.is_file() or not panorama_path.is_relative_to(dataset_dir.resolve()):
            invalid += 1
            issues.append(InputIssue("invalid_panorama", image_id, "Source panorama is unavailable."))
            continue
        raw_refined_pose = analysis.get("refined_pose")
        refined_pose: dict[str, Any] = raw_refined_pose if isinstance(raw_refined_pose, dict) else {}
        digest = hashlib.sha256(raw).hexdigest()
        panorama_digest = file_sha256(panorama_path)
        id_mask_digest = file_sha256(id_mask_path)
        inferred_mask_digest = file_sha256(inferred_mask_path)
        analysis_hashes.append(f"{image_id}:{digest}:{panorama_digest}:{id_mask_digest}:{inferred_mask_digest}")
        longitude, latitude = coordinates
        views.append(ViewEvidence(
            image_id=image_id,
            local_file=str(image["local_file"]),
            source_page=str(image["source_page"]) if image.get("source_page") else None,
            captured_at_ms=int(image["captured_at_ms"]) if isinstance(image.get("captured_at_ms"), int) else None,
            recorded_longitude=longitude,
            recorded_latitude=latitude,
            recorded_heading_deg=recorded_heading,
            analysis_width=width,
            analysis_height=height,
            mask_ordinal=ordinal,
            observed_pixels=observed_pixels,
            inferred_pixels=inferred_pixels,
            projection_coverage=projection_coverage,
            visible_facades=visible_facades,
            footprint_source=str(building.get("footprint_source", "osm")),
            alternative_footprint_id=str(building["alternative_footprint_id"]) if building.get("alternative_footprint_id") else None,
            footprint_geometry=geometry,
            pose_facade_iou=_number(refined_pose.get("facade_iou")),
            pose_facade_iou_gain=_number(refined_pose.get("facade_iou_gain")),
            sfm_depth_samples=sfm_samples,
            sfm_depth_conflicts=sfm_conflicts,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            fov_deg=fov_deg,
            analysis_relative_path=f"analysis/{image_id}.json",
            analysis_sha256=digest,
            panorama_relative_path=str(image["local_file"]),
            panorama_sha256=panorama_digest,
            id_mask_relative_path=f"analysis/{image_id}-ids.png",
            id_mask_sha256=id_mask_digest,
            inferred_mask_relative_path=f"analysis/{image_id}-facades.png",
            inferred_mask_sha256=inferred_mask_digest,
        ))

    manifest_digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    fingerprint = hashlib.sha256((manifest_digest + "\n" + "\n".join(sorted(analysis_hashes))).encode("ascii")).hexdigest()
    if not views:
        raise BuildingEvidenceNotFound(f"No current analysis directly associates {building_id} with this dataset.")
    return EvidenceCollection(
        views=tuple(views),
        manifest_panorama_count=len(images),
        compatible_analysis_count=compatible,
        stale_analysis_count=stale,
        incomplete_analysis_count=incomplete,
        invalid_analysis_count=invalid,
        issues=tuple(issues),
        input_fingerprint_sha256=fingerprint,
    )
