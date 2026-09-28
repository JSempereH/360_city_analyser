"""COLMAP-based, GPS-anchored pose refinement for nearby 360 panoramas."""

from __future__ import annotations

import json
import importlib
import math
import os
import tempfile
from pathlib import Path
from statistics import median
from typing import Any, Callable

from .analysis_cache import canonical_json_sha256, file_sha256
from .multi_view_analysis import distance_meters, has_valid_heading, nearby_panoramas, panorama_coordinates
from .panorama_geometry import perspective_tile


SFM_VERSION = 3
SFM_CUBE_SIZE = 768
SFM_FACES = (("front", 0.0), ("right", math.pi / 2), ("back", math.pi), ("left", 3 * math.pi / 2))
SFM_FIELD_OF_VIEW_DEGREES = 110.0
SFM_MAX_POSITION_SHIFT_M = 15.0
SFM_MAX_RIG_CENTER_SPREAD_M = 2.0
SFM_MAX_MODEL_REPROJECTION_ERROR_PX = 4.0
SFM_MIN_POINT_TRACK_LENGTH = 2
SFM_MAX_POINT_REPROJECTION_ERROR_PX = 8.0
WGS84_A = 6_378_137.0
WGS84_E2 = 6.69437999014e-3


def sfm_path(data_root: Path, dataset_id: str, image_id: str) -> Path:
    return data_root / dataset_id / "analysis" / f"{image_id}-sfm.json"


def _sfm_input_sha256(data_root: Path, dataset_id: str, current_image: dict[str, Any], images: list[dict[str, Any]]) -> str:
    dataset_dir = (data_root / dataset_id).resolve()
    _heading_degrees(current_image)
    selected = nearby_panoramas([image for image in images if has_valid_heading(image)], current_image)
    inputs = []
    for image, _ in selected:
        local_file = image.get("local_file")
        if not isinstance(local_file, str):
            raise ValueError("Nearby panorama metadata is invalid.")
        image_path = (dataset_dir / local_file).resolve()
        if not image_path.is_file() or not image_path.is_relative_to(dataset_dir):
            raise ValueError("A nearby panorama image is unavailable.")
        inputs.append({
            "id": str(image.get("id", "")),
            "panorama_sha256": file_sha256(image_path),
            "coordinates": panorama_coordinates(image),
            "heading_degrees": _heading_degrees(image),
        })
    return canonical_json_sha256({
        "version": SFM_VERSION,
        "cube_size": SFM_CUBE_SIZE,
        "field_of_view_degrees": SFM_FIELD_OF_VIEW_DEGREES,
        "inputs": inputs,
    })


def load_sfm_refinement(data_root: Path, dataset_id: str, image_id: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(sfm_path(data_root, dataset_id, image_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != SFM_VERSION or str(payload.get("image_id")) != image_id:
        return None
    try:
        manifest = json.loads((data_root / dataset_id / "manifest.json").read_text(encoding="utf-8"))
        images = manifest["images"]
        current = next(image for image in images if isinstance(image, dict) and str(image.get("id")) == image_id)
        expected = _sfm_input_sha256(data_root, dataset_id, current, images)
    except (OSError, ValueError, KeyError, StopIteration, TypeError, json.JSONDecodeError):
        return None
    return payload if payload.get("input_sha256") == expected else None


def _require_pycolmap() -> Any:
    try:
        return importlib.import_module("pycolmap")
    except ImportError as error:
        raise RuntimeError("Pose refinement requires PyCOLMAP. Run uv sync, then retry.") from error


def _require_sfm_dependencies() -> tuple[Any, Any]:
    try:
        np = __import__("numpy")
        Image = __import__("PIL.Image", fromlist=["Image"])
    except ImportError as error:
        raise RuntimeError("Pose refinement requires Pillow and NumPy. Run uv sync, then retry.") from error
    return np, Image


def _write_cube_faces(data_root: Path, dataset_id: str, images: list[dict[str, Any]], destination: Path) -> dict[str, str]:
    np, Image = _require_sfm_dependencies()
    face_images = {}
    for image in images:
        local_file = image.get("local_file")
        image_id = str(image.get("id", ""))
        if not isinstance(local_file, str) or not image_id.isdecimal():
            raise ValueError("Nearby panorama metadata is invalid.")
        image_path = (data_root / dataset_id / local_file).resolve()
        if not image_path.is_file() or not image_path.is_relative_to((data_root / dataset_id).resolve()):
            raise ValueError("A nearby panorama image is unavailable.")
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            source.thumbnail((SFM_CUBE_SIZE * 4, SFM_CUBE_SIZE * 2), Image.Resampling.LANCZOS)
            panorama = np.asarray(source)
        for face_name, yaw in SFM_FACES:
            tile, _, _ = perspective_tile(panorama, yaw, 0.0, SFM_CUBE_SIZE, SFM_FIELD_OF_VIEW_DEGREES)
            name = f"pano-{image_id}-{face_name}.jpg"
            Image.fromarray(tile).save(destination / name, quality=94)
            face_images[name] = image_id
    return face_images


def _heading_degrees(image: dict[str, Any]) -> float:
    try:
        raw_heading = image["computed_compass_angle_deg"]
        if isinstance(raw_heading, bool):
            raise ValueError
        heading = float(raw_heading)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Panorama {image.get('id', '')} has no valid computed compass heading.") from error
    if not math.isfinite(heading):
        raise ValueError(f"Panorama {image.get('id', '')} has no valid computed compass heading.")
    return heading % 360.0


def _write_match_pairs(path: Path, images: list[dict[str, Any]]) -> None:
    """Match only physical overlaps instead of unrelated panorama cube faces."""
    names = {
        str(image["id"]): {face_name: f"pano-{image['id']}-{face_name}.jpg" for face_name, _ in SFM_FACES}
        for image in images
    }
    headings = {str(image["id"]): _heading_degrees(image) for image in images}
    pairs = set()
    for image in images:
        faces = names[str(image["id"])]
        face_names = tuple(faces)
        for first, second in zip(face_names, face_names[1:] + face_names[:1]):
            pairs.add(tuple(sorted((faces[first], faces[second]))))
    ordered_images = sorted(images, key=lambda image: str(image["id"]))
    for index, first_image in enumerate(ordered_images):
        first_id = str(first_image["id"])
        for second_image in ordered_images[index + 1:]:
            second_id = str(second_image["id"])
            for first_face, first_yaw in SFM_FACES:
                first_direction = headings[first_id] + math.degrees(first_yaw)
                for second_face, second_yaw in SFM_FACES:
                    second_direction = headings[second_id] + math.degrees(second_yaw)
                    separation = abs((first_direction - second_direction + 180.0) % 360.0 - 180.0)
                    if separation < SFM_FIELD_OF_VIEW_DEGREES:
                        pairs.add(tuple(sorted((names[first_id][first_face], names[second_id][second_face]))))
    path.write_text("\n".join(" ".join(pair) for pair in sorted(pairs)) + "\n", encoding="utf-8")


def _ecef_to_wgs84(position: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = position
    longitude = math.atan2(y, x)
    horizontal = math.hypot(x, y)
    latitude = math.atan2(z, horizontal * (1 - WGS84_E2))
    for _ in range(8):
        radius = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(latitude) ** 2)
        latitude = math.atan2(z + WGS84_E2 * radius * math.sin(latitude), horizontal)
    return math.degrees(longitude), math.degrees(latitude)


def _cube_face_uv(face_name: str, coordinates: tuple[float, float]) -> tuple[float, float] | None:
    """Convert a COLMAP cube-face observation back to panorama UV coordinates."""
    face_yaw = dict(SFM_FACES).get(face_name)
    if face_yaw is None:
        return None
    x, y = coordinates
    focal_length = SFM_CUBE_SIZE / (2 * math.tan(math.radians(SFM_FIELD_OF_VIEW_DEGREES) / 2))
    local_x = (float(x) - SFM_CUBE_SIZE / 2) / focal_length
    local_y = (SFM_CUBE_SIZE / 2 - float(y)) / focal_length
    magnitude = math.sqrt(local_x ** 2 + local_y ** 2 + 1)
    local_x, local_y, local_z = local_x / magnitude, local_y / magnitude, 1 / magnitude
    world_x = math.cos(face_yaw) * local_x + math.sin(face_yaw) * local_z
    world_z = -math.sin(face_yaw) * local_x + math.cos(face_yaw) * local_z
    return (
        (0.5 + math.atan2(world_x, world_z) / (2 * math.pi)) % 1,
        0.5 - math.asin(max(-1, min(1, local_y))) / math.pi,
    )


def _sparse_depth_samples(
    model: Any, face_images: dict[str, str], image_id: str, accepted_panorama_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return deduplicated metric depths for sparse points observed in one panorama."""
    np, _ = _require_sfm_dependencies()
    samples: dict[tuple[float, float], tuple[float, dict[str, Any]]] = {}
    for point_id in model.point3D_ids():
        point = model.point3D(point_id)
        observations = list(point.track.elements)
        observed_panorama_ids = {
            face_images[model.image(observation.image_id).name]
            for observation in observations
            if model.image(observation.image_id).name in face_images
        }
        if accepted_panorama_ids is not None:
            observed_panorama_ids &= accepted_panorama_ids
        if image_id not in observed_panorama_ids or len(observed_panorama_ids) < 2:
            continue
        track_length_value = getattr(point.track, "length", None)
        try:
            raw_track_length = track_length_value() if callable(track_length_value) else track_length_value
            track_length = int(raw_track_length) if isinstance(raw_track_length, (int, float)) else len(observations)
        except (TypeError, ValueError, OverflowError):
            track_length = len(observations)
        if track_length < SFM_MIN_POINT_TRACK_LENGTH:
            continue
        reprojection_error = None
        has_error = getattr(point, "has_error", None)
        try:
            error_is_available = has_error() if callable(has_error) else has_error
        except (TypeError, ValueError):
            error_is_available = False
        if error_is_available is not False:
            try:
                candidate_error = float(point.error)
            except (AttributeError, TypeError, ValueError):
                candidate_error = math.nan
            if math.isfinite(candidate_error) and candidate_error >= 0:
                reprojection_error = candidate_error
        if reprojection_error is None or reprojection_error > SFM_MAX_POINT_REPROJECTION_ERROR_PX:
            continue
        point_xyz = np.asarray(point.xyz, dtype=np.float64)
        for observation in observations:
            image = model.image(observation.image_id)
            if face_images.get(image.name) != image_id:
                continue
            face_name = Path(image.name).stem.rsplit("-", 1)[-1]
            point2d = image.point2D(observation.point2D_idx)
            uv = _cube_face_uv(face_name, tuple(point2d.xy))
            if uv is None:
                continue
            distance = float(np.linalg.norm(point_xyz - np.asarray(image.projection_center(), dtype=np.float64)))
            if not math.isfinite(distance) or distance <= 0:
                continue
            key = (round(uv[0], 4), round(uv[1], 4))
            sample = {"u": key[0], "v": key[1], "distance_m": round(distance, 2), "track_length": track_length}
            if reprojection_error is not None:
                sample["reprojection_error_px"] = round(reprojection_error, 3)
            if key not in samples or distance < samples[key][0]:
                samples[key] = (distance, sample)
    return [sample for _, (_, sample) in sorted(samples.items())]


def _can_report_metric_depth(poses: dict[str, dict[str, Any]], current_id: str) -> bool:
    current_pose = poses.get(current_id)
    return bool(
        current_pose is not None
        and current_pose.get("accepted") is True
        and sum(pose.get("accepted") is True for pose in poses.values()) >= 3
    )


def refine_nearby_poses(
    data_root: Path,
    dataset_id: str,
    current_image: dict[str, Any],
    images: list[dict[str, Any]],
    progress: Callable[[str], None],
) -> dict[str, Any]:
    selected = nearby_panoramas(images, current_image)
    if len(selected) < 2:
        raise ValueError("At least two nearby downloaded panoramas are required for pose refinement.")
    selected_images = [image for image, _ in selected]
    input_sha256 = _sfm_input_sha256(data_root, dataset_id, current_image, images)
    _heading_degrees(current_image)
    for image in selected_images:
        _heading_degrees(image)
    pycolmap = _require_pycolmap()
    np, _ = _require_sfm_dependencies()
    current_id = str(current_image["id"])
    with tempfile.TemporaryDirectory(prefix="panorama-sfm-") as directory:
        workspace = Path(directory)
        faces = workspace / "faces"
        faces.mkdir()
        progress(f"Preparing {len(selected_images)} nearby panoramas as calibrated cube faces...")
        face_images = _write_cube_faces(data_root, dataset_id, selected_images, faces)
        database = workspace / "database.db"
        sparse = workspace / "sparse"
        sparse.mkdir()
        focal_length = SFM_CUBE_SIZE / (2 * math.tan(math.radians(SFM_FIELD_OF_VIEW_DEGREES) / 2))
        use_gpu = os.environ.get("SFM_COLMAP_USE_GPU", "0") == "1"
        device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu
        reader_options = pycolmap.ImageReaderOptions(
            camera_model="PINHOLE",
            camera_params=f"{focal_length},{focal_length},{SFM_CUBE_SIZE / 2},{SFM_CUBE_SIZE / 2}",
        )
        extraction_options = pycolmap.FeatureExtractionOptions(max_image_size=SFM_CUBE_SIZE, use_gpu=use_gpu)
        matching_options = pycolmap.FeatureMatchingOptions(use_gpu=use_gpu)
        progress("PyCOLMAP: extracting SIFT features...")
        pycolmap.extract_features(database, faces, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader_options, extraction_options=extraction_options, device=device)
        pairs_path = workspace / "match-pairs.txt"
        _write_match_pairs(pairs_path, selected_images)
        progress("PyCOLMAP: matching overlapping nearby cube faces...")
        pycolmap.match_image_pairs(
            database,
            matching_options=matching_options,
            pairing_options=pycolmap.ImportedPairingOptions(match_list_path=pairs_path),
            device=device,
        )
        progress("PyCOLMAP: reconstructing sparse nearby-panoramas model...")
        models = pycolmap.incremental_mapping(database, faces, sparse)
        if not models:
            raise RuntimeError("PyCOLMAP could not reconstruct a connected nearby-panoramas model.")
        model = max(models.values(), key=lambda candidate: candidate.num_reg_images())
        names, locations = [], []
        by_id = {str(image["id"]): image for image in selected_images}
        for name, image_id in face_images.items():
            coordinates = panorama_coordinates(by_id[image_id])
            if coordinates is None:
                continue
            longitude, latitude = coordinates
            names.append(name)
            locations.append([latitude, longitude, 0.0])
        progress("PyCOLMAP: aligning the sparse model to Mapillary GPS...")
        ecef_locations = pycolmap.GPSTransform().ellipsoid_to_ecef(np.asarray(locations, dtype=np.float64))
        alignment = pycolmap.align_reconstruction_to_locations(
            model, names, ecef_locations, min_common_images=8, ransac_options=pycolmap.RANSACOptions(max_error=10.0),
        )
        if alignment is None:
            raise RuntimeError("PyCOLMAP could not align the sparse model to Mapillary GPS.")
        model.transform(alignment)
        try:
            model_reprojection_error = float(model.compute_mean_reprojection_error())
        except (AttributeError, TypeError, ValueError):
            model_reprojection_error = math.inf
        centers: dict[str, list[tuple[float, float, float]]] = {}
        for registered_image_id in model.reg_image_ids():
            registered = model.image(registered_image_id)
            if registered.name not in face_images:
                continue
            center = registered.projection_center()
            centers.setdefault(face_images[registered.name], []).append((float(center[0]), float(center[1]), float(center[2])))
    poses = {}
    by_id = {str(image["id"]): image for image in selected_images}
    for image_id, face_centers in centers.items():
        if len(face_centers) < 2:
            continue
        center = tuple(median(coordinate) for coordinate in zip(*face_centers))
        center_spread = max(math.dist(center, face_center) for face_center in face_centers)
        longitude, latitude = _ecef_to_wgs84(center)
        original = panorama_coordinates(by_id[image_id])
        if original is None:
            continue
        shift = distance_meters(original, (longitude, latitude))
        poses[image_id] = {
            "latitude": latitude,
            "longitude": longitude,
            "registered_faces": len(face_centers),
            "position_shift_m": round(shift, 2),
            "rig_center_spread_m": round(center_spread, 2),
            "accepted": bool(
                shift <= SFM_MAX_POSITION_SHIFT_M
                and center_spread <= SFM_MAX_RIG_CENTER_SPREAD_M
                and model_reprojection_error <= SFM_MAX_MODEL_REPROJECTION_ERROR_PX
            ),
        }
    accepted = sum(pose["accepted"] for pose in poses.values())
    if accepted < 2:
        raise RuntimeError("COLMAP produced too few GPS-consistent panorama poses.")
    depth_samples = []
    if _can_report_metric_depth(poses, current_id):
        progress("PyCOLMAP: extracting sparse multiview depth for the current panorama...")
        depth_samples = _sparse_depth_samples(
            model, face_images, current_id, {image_id for image_id, pose in poses.items() if pose.get("accepted") is True},
        )
    return {
        "version": SFM_VERSION,
        "image_id": current_id,
        "input_sha256": input_sha256,
        "panoramas_requested": len(selected_images),
        "cube_faces_registered": sum(pose["registered_faces"] for pose in poses.values()),
        "accepted_panorama_poses": accepted,
        "mean_reprojection_error_px": round(model_reprojection_error, 3) if math.isfinite(model_reprojection_error) else None,
        "poses": poses,
        "depth_samples": depth_samples,
    }
