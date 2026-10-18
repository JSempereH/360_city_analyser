"""Download a small, georeferenced sample of Mapillary 360-degree panoramas.

The script intentionally defaults to a small sample. It records per-image
provenance and attribution; it does not create an unbounded Mapillary mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError


API_URL = "https://graph.mapillary.com/images"
MAPILLARY_APP_URL = "https://www.mapillary.com/app/?pKey={image_id}"
EARTH_RADIUS_M = 6_371_008.8
MAPILLARY_IMAGE_SIZES = ("256", "1024", "2048", "original")
MAPILLARY_IMAGE_FIELDS = tuple(f"thumb_{size}_url" for size in MAPILLARY_IMAGE_SIZES)
MAX_IMAGE_BYTES = 50 * 1024 * 1024
EQUIRECTANGULAR_RATIO_TOLERANCE = 0.05


def status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs without overriding explicit environment variables."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def parse_pair(value: str, name: str) -> tuple[float, float]:
    try:
        first, second = (float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be two comma-separated numbers") from error
    return first, second


def parse_center(value: str) -> tuple[float, float]:
    lat, lon = parse_pair(value, "center")
    validate_lat_lon(lat, lon)
    return lat, lon


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    try:
        min_lon, min_lat, max_lon, max_lat = (float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("bbox must be min_lon,min_lat,max_lon,max_lat") from error
    validate_lat_lon(min_lat, min_lon)
    validate_lat_lon(max_lat, max_lon)
    if min_lon >= max_lon or min_lat >= max_lat:
        raise argparse.ArgumentTypeError("bbox minimums must be smaller than maximums")
    if (max_lon - min_lon) * (max_lat - min_lat) >= 0.01:
        raise argparse.ArgumentTypeError("bbox is too large for the Mapillary images endpoint (max 0.01 deg2)")
    return min_lon, min_lat, max_lon, max_lat


def validate_lat_lon(lat: float, lon: float) -> None:
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise argparse.ArgumentTypeError("coordinates are outside WGS84 bounds")


def bbox_from_center(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    if radius_m <= 0:
        raise ValueError("radius must be positive")
    lat_delta = radius_m / 111_320
    lon_scale = 111_320 * math.cos(math.radians(lat))
    if abs(lon_scale) < 1:
        raise ValueError("a center/radius query is unsupported close to the poles")
    lon_delta = radius_m / lon_scale
    bbox = (lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta)
    if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= 0.01:
        raise ValueError("center/radius area is too large for the Mapillary images endpoint")
    return bbox


def haversine_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    lat_a, lon_a, lat_b, lon_b = map(math.radians, (lat_a, lon_a, lat_b, lon_b))
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    half_chord = math.sin(delta_lat / 2) ** 2 + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(half_chord))


def point_from_image(image: dict[str, Any]) -> tuple[float, float] | None:
    """Prefer Mapillary's corrected position but retain both geometries in output."""
    for field in ("computed_geometry", "geometry"):
        geometry = image.get(field)
        if not isinstance(geometry, dict) or geometry.get("type") != "Point":
            continue
        coordinates = geometry.get("coordinates")
        if isinstance(coordinates, list) and len(coordinates) >= 2:
            lon, lat = coordinates[:2]
            if (
                isinstance(lon, (int, float)) and not isinstance(lon, bool)
                and isinstance(lat, (int, float)) and not isinstance(lat, bool)
            ):
                longitude, latitude = float(lon), float(lat)
                if math.isfinite(longitude) and math.isfinite(latitude) and -180 <= longitude <= 180 and -90 <= latitude <= 90:
                    return longitude, latitude
    return None


def request_json(url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "open-georeferenced-street-dataset/0.1"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504}:
                detail = error.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"Mapillary API returned HTTP {error.code}: {detail}") from error
            last_error = error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
        status(f"  Request failed for now ({last_error}); retrying {attempt + 1}/3...")
        time.sleep(2**attempt)
    raise RuntimeError(f"Mapillary API request failed after retries: {last_error}")


def _image_fields(image_fields: tuple[str, ...]) -> str:
    return ",".join(
        [
            "id",
            "geometry",
            "computed_geometry",
            "captured_at",
            "compass_angle",
            "computed_compass_angle",
            "computed_rotation",
            "is_pano",
            "sequence",
            *image_fields,
            "width",
            "height",
            "creator",
        ]
    )


def fetch_image(token: str, image_id: str, image_fields: tuple[str, ...]) -> dict[str, Any]:
    """Fetch one selected Mapillary image without relying on AOI pagination order."""
    payload = request_json(f"https://graph.mapillary.com/{image_id}", {"access_token": token, "fields": _image_fields(image_fields)})
    if not isinstance(payload.get("id"), (str, int)):
        raise RuntimeError("Mapillary API response did not contain the selected image")
    return payload


def fetch_images(token: str, bbox: tuple[float, float, float, float], image_fields: tuple[str, ...], max_pages: int | None) -> list[dict[str, Any]]:
    fields = _image_fields(image_fields)
    params = {
        "access_token": token,
        "bbox": ",".join(f"{coordinate:.7f}" for coordinate in bbox),
        "fields": fields,
        "is_pano": "true",
        "limit": "2000",
    }
    images: list[dict[str, Any]] = []
    next_url: str | None = API_URL
    next_params: dict[str, str] | None = params
    page = 1
    while next_url is not None and (max_pages is None or page <= max_pages):
        status(f"Querying Mapillary, page {page}{f'/{max_pages}' if max_pages is not None else ''}...")
        payload = request_json(next_url, next_params)
        data = payload.get("data", [])
        if not isinstance(data, list):
            raise RuntimeError("Mapillary API response did not contain an image list")
        images.extend(image for image in data if isinstance(image, dict))
        status(f"  Received {len(data)} images; {len(images)} so far.")
        paging = payload.get("paging")
        next_url = paging.get("next") if isinstance(paging, dict) else None
        next_params = None
        page += 1
    return images


def available_image_url(image: dict[str, Any], preferred_size: str) -> tuple[str, str] | None:
    """Return the requested derivative, or the best smaller available fallback."""
    if preferred_size == "auto":
        sizes = tuple(reversed(MAPILLARY_IMAGE_SIZES))
    else:
        try:
            sizes = MAPILLARY_IMAGE_SIZES[: MAPILLARY_IMAGE_SIZES.index(preferred_size) + 1]
        except ValueError:
            raise ValueError("unsupported image size") from None
        sizes = tuple(reversed(sizes))
    for size in sizes:
        field = f"thumb_{size}_url"
        url = image.get(field)
        if isinstance(url, str) and url:
            return field, url
    return None


class PanoramaSelector:
    """Keeps panoramas at least ``min_spacing_m`` apart, earliest capture first, up to ``max_images``.

    The positions kept so far live in a grid of cells one spacing wide, so each
    candidate is compared with its neighbors only. The previous all-pairs check
    took hours at city scale. It can be fed one tile of results at a time.
    """

    def __init__(self, center: tuple[float, float] | None, radius_m: float | None, min_spacing_m: float, max_images: int):
        self.center, self.radius_m, self.min_spacing_m, self.max_images = center, radius_m, min_spacing_m, max_images
        self.selected: list[dict[str, Any]] = []
        self._seen: set[str] = set()
        self._cells: dict[tuple[int, int], list[tuple[float, float]]] = {}
        self._cell_deg = max(min_spacing_m, 0.001) / 111_320

    def _cell(self, lat: float, lon: float) -> tuple[int, int]:
        # Longitude cells are widened at the latitude of the point, so neighbors always cover the spacing.
        lon_deg = self._cell_deg / max(math.cos(math.radians(min(abs(lat), 89.0))), 0.01)
        return math.floor(lat / self._cell_deg), math.floor(lon / lon_deg)

    def _too_close(self, lat: float, lon: float) -> bool:
        if self.min_spacing_m <= 0:
            return False
        lat_cell, _ = self._cell(lat, lon)
        # A wide latitude band may use a different longitude cell width; scan the nearby columns explicitly.
        reach = 2 + int(1 / max(math.cos(math.radians(min(abs(lat), 89.0))), 0.01))
        lon_deg = self._cell_deg / max(math.cos(math.radians(min(abs(lat), 89.0))), 0.01)
        lon_cell = math.floor(lon / lon_deg)
        for row in range(lat_cell - 1, lat_cell + 2):
            for column in range(lon_cell - reach, lon_cell + reach + 1):
                for other_lat, other_lon in self._cells.get((row, column), ()):
                    if haversine_m(lat, lon, other_lat, other_lon) < self.min_spacing_m:
                        return True
        return False

    def add(self, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Newly selected panoramas from ``images`` (already seen IDs and full selections add nothing)."""
        candidates: list[dict[str, Any]] = []
        for image in images:
            if image.get("is_pano") is not True or str(image.get("id")) in self._seen:
                continue
            point = point_from_image(image)
            if point is None:
                continue
            lon, lat = point
            if self.center and self.radius_m and haversine_m(self.center[0], self.center[1], lat, lon) > self.radius_m:
                continue
            image = dict(image)
            image["_selected_geometry"] = {"type": "Point", "coordinates": [lon, lat]}
            candidates.append(image)
        candidates.sort(key=lambda image: (image.get("captured_at", 0), image.get("id", "")))
        added: list[dict[str, Any]] = []
        for image in candidates:
            if len(self.selected) >= self.max_images:
                break
            lon, lat = image["_selected_geometry"]["coordinates"]
            self._seen.add(str(image.get("id")))
            if self._too_close(lat, lon):
                continue
            self.selected.append(image)
            self._cells.setdefault(self._cell(lat, lon), []).append((lat, lon))
            added.append(image)
        return added


def select_panoramas(
    images: list[dict[str, Any]],
    center: tuple[float, float] | None,
    radius_m: float | None,
    min_spacing_m: float,
    max_images: int,
) -> list[dict[str, Any]]:
    return PanoramaSelector(center, radius_m, min_spacing_m, max_images).add(images)


def normalize_compass_heading(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("computed compass angle must be a finite number")
    heading = float(value)
    if not math.isfinite(heading):
        raise ValueError("computed compass angle must be a finite number")
    return heading % 360.0


def camera_up(rotation: object) -> list[float] | None:
    """World up in the panorama's own frame (x right, y up, z toward the image center), or None.

    ``rotation`` is Mapillary's ``computed_rotation``: the angle-axis rotation from the topocentric
    frame (z up) to the camera frame (x right, y down, z forward) of its 3D reconstruction. A level
    camera gives [0, 1, 0]; the viewer turns the panorama so that this vector points up.
    """
    if not isinstance(rotation, (list, tuple)) or len(rotation) != 3 or not all(isinstance(value, (int, float)) for value in rotation):
        return None
    angle = math.sqrt(sum(value * value for value in rotation))
    if not math.isfinite(angle):
        return None
    if angle < 1e-12:
        up = (0.0, 0.0, 1.0)
    else:
        kx, ky, kz = (value / angle for value in rotation)
        cosine, sine = math.cos(angle), math.sin(angle)
        # Rodrigues: R @ (0, 0, 1) = cos(a) z + sin(a) (k x z) + (1 - cos(a)) (k . z) k
        up = (sine * ky + (1 - cosine) * kz * kx, -sine * kx + (1 - cosine) * kz * ky, cosine + (1 - cosine) * kz * kz)
    return [round(up[0], 6), round(-up[1], 6), round(up[2], 6)]  # camera y points down


def image_record(image: dict[str, Any], local_file: str | None, local_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    creator_value = image.get("creator")
    creator_id = creator_value.get("id") if isinstance(creator_value, dict) else None
    creator_username = creator_value.get("username") if isinstance(creator_value, dict) else None
    image_id = str(image["id"])
    sequence = image.get("sequence")
    width = local_metadata["width"] if local_metadata is not None else image.get("width")
    height = local_metadata["height"] if local_metadata is not None else image.get("height")
    return {
        "id": image_id,
        "source_page": MAPILLARY_APP_URL.format(image_id=image_id),
        "local_file": local_file,
        "captured_at_ms": image.get("captured_at"),
        "computed_compass_angle_deg": normalize_compass_heading(image.get("computed_compass_angle")),
        "raw_compass_angle_deg": normalize_compass_heading(image.get("compass_angle")),
        "computed_rotation": image.get("computed_rotation"),
        "camera_up": camera_up(image.get("computed_rotation")),
        "sequence_id": (
            str(sequence.get("id")) if isinstance(sequence, dict) and sequence.get("id") is not None
            else str(sequence) if isinstance(sequence, (str, int)) and not isinstance(sequence, bool)
            else None
        ),
        "is_pano": True,
        "selected_geometry": image["_selected_geometry"],
        "geometry": image.get("geometry"),
        "computed_geometry": image.get("computed_geometry"),
        "width": width,
        "height": height,
        "sha256": local_metadata["sha256"] if local_metadata is not None else None,
        "download_image_field": image.get("_download_image_field"),
        "creator": {"id": creator_id, "username": creator_username},
        "license": "CC BY-SA 4.0 (Mapillary image license; verify current terms before redistribution)",
    }


def inspect_local_image(path: Path, sha256: str | None = None) -> dict[str, Any]:
    if sha256 is None:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        sha256 = digest.hexdigest()
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise RuntimeError(f"downloaded image is corrupt or unsupported: {path.name}") from error
    if height <= 0 or not math.isclose(width / height, 2.0, rel_tol=EQUIRECTANGULAR_RATIO_TOLERANCE):
        raise RuntimeError(f"downloaded image is not equirectangular (expected an aspect ratio near 2:1): {path.name}")
    return {"width": width, "height": height, "sha256": sha256}


def download_image(url: str, destination: Path, overwrite: bool) -> tuple[bool, dict[str, Any]]:
    if destination.exists() and not overwrite:
        return False, inspect_local_image(destination)
    request = Request(url, headers={"User-Agent": "open-georeferenced-street-dataset/0.1"})
    temporary: Path | None = None
    try:
        with urlopen(request, timeout=90) as response:
            content_type = response.headers.get_content_type()
            if not content_type.startswith("image/"):
                raise RuntimeError(f"expected an image for {destination.name}, received {content_type}")
            content_length = response.headers.get("Content-Length")
            try:
                declared_size = int(content_length) if content_length else None
            except ValueError:
                declared_size = None
            if declared_size is not None and declared_size > MAX_IMAGE_BYTES:
                raise RuntimeError(f"refusing to download a file larger than 50 MB: {destination.name}")

            digest = hashlib.sha256()
            bytes_downloaded = 0
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".part",
                dir=destination.parent,
                delete=False,
            ) as output:
                temporary = Path(output.name)
                while chunk := response.read(1024 * 1024):
                    bytes_downloaded += len(chunk)
                    if bytes_downloaded > MAX_IMAGE_BYTES:
                        raise RuntimeError(f"refusing to download a file larger than 50 MB: {destination.name}")
                    output.write(chunk)
                    digest.update(chunk)

        metadata = inspect_local_image(temporary, digest.hexdigest())
        temporary.replace(destination)
        temporary = None
        return True, metadata
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}
MAX_TILE_AREA_DEG2 = 0.008
DOWNLOAD_ATTEMPTS = 4
# Seconds before the first retry; doubles each time. Tests set it to 0.
DOWNLOAD_BACKOFF_S = 2.0
CHECKPOINT_INTERVAL_S = 120.0


def download_with_retries(url: str, destination: Path, overwrite: bool, attempts: int = DOWNLOAD_ATTEMPTS) -> tuple[bool, dict[str, Any]]:
    """``download_image`` retried on network trouble; bad content (not an image, too large) is not retried."""
    for attempt in range(1, attempts + 1):
        try:
            return download_image(url, destination, overwrite)
        except HTTPError as error:
            if error.code not in TRANSIENT_HTTP_CODES or attempt == attempts:
                raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt == attempts:
                raise
        time.sleep(DOWNLOAD_BACKOFF_S * 2 ** (attempt - 1))
    raise AssertionError("unreachable")


def tile_bbox(bbox: tuple[float, float, float, float], max_area_deg2: float = MAX_TILE_AREA_DEG2) -> list[tuple[float, float, float, float]]:
    """Split a bounding box into an equal grid of tiles, each below the Mapillary images endpoint's area limit."""
    west, south, east, north = bbox
    area = (east - west) * (north - south)
    if area < max_area_deg2:
        return [bbox]
    across = math.ceil(math.sqrt(area / max_area_deg2 * (east - west) / (north - south)))
    down = math.ceil(area / max_area_deg2 / across)
    width, height = (east - west) / across, (north - south) / down
    return [
        (west + column * width, south + row * height, west + (column + 1) * width, south + (row + 1) * height)
        for row in range(down) for column in range(across)
    ]


def parse_tiled_bbox(value: str) -> tuple[float, float, float, float]:
    """Like ``parse_bbox`` but without the single-query area limit; the area is tiled."""
    try:
        min_lon, min_lat, max_lon, max_lat = (float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("bbox must be min_lon,min_lat,max_lon,max_lat") from error
    validate_lat_lon(min_lat, min_lon)
    validate_lat_lon(max_lat, max_lon)
    if min_lon >= max_lon or min_lat >= max_lat:
        raise argparse.ArgumentTypeError("bbox minimums must be smaller than maximums")
    return min_lon, min_lat, max_lon, max_lat


def _replace_text(path: Path, text: str) -> None:
    """Write a new file and swap it in: a dataset copy that hard-links this one keeps its own version."""
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_manifest(output_dir: Path, records: list[dict[str, Any]], query: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _replace_text(output_dir / "manifest.json", json.dumps({"query": query, "images": records}, indent=2, ensure_ascii=False) + "\n")
    features = [
        {"type": "Feature", "geometry": record["selected_geometry"], "properties": {key: value for key, value in record.items() if key != "selected_geometry"}}
        for record in records
    ]
    _replace_text(output_dir / "manifest.geojson", json.dumps({"type": "FeatureCollection", "features": features}, indent=2, ensure_ascii=False) + "\n")
    attribution_lines = [
        "# Attribution", "", "Mapillary images are licensed CC BY-SA 4.0. Keep this file with the images and review Mapillary's current terms before redistributing a dataset.", ""
    ]
    for record in records:
        username = record["creator"].get("username") or record["creator"].get("id") or "unknown contributor"
        attribution_lines.append(f"- Image {record['id']} by {username}: {record['source_page']} (CC BY-SA 4.0)")
    _replace_text(output_dir / "ATTRIBUTION.md", "\n".join(attribution_lines) + "\n")


def add_camera_rotations(output_dir: Path, token: str, workers: int = 8) -> dict[str, int]:
    """Fill ``computed_rotation`` and ``camera_up`` for a downloaded dataset's panoramas that lack them."""
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    missing = [record for record in manifest["images"] if record.get("camera_up") is None]

    def fetch(record: dict[str, Any]) -> tuple[dict[str, Any], object]:
        payload = request_json(f"https://graph.mapillary.com/{record['id']}", {"access_token": token, "fields": "id,computed_rotation"})
        return record, payload.get("computed_rotation")

    found = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record, rotation in pool.map(fetch, missing):
            record["computed_rotation"] = rotation
            record["camera_up"] = camera_up(rotation)
            found += record["camera_up"] is not None
    write_manifest(output_dir, manifest["images"], manifest.get("query") or {})
    return {"panoramas": len(manifest["images"]), "asked": len(missing), "found": found}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a small georeferenced Mapillary panorama sample.")
    area = parser.add_mutually_exclusive_group(required=True)
    area.add_argument("--center", type=parse_center, help="WGS84 latitude,longitude at the street center")
    area.add_argument("--bbox", type=parse_bbox, help="min_lon,min_lat,max_lon,max_lat around one street")
    area.add_argument("--tiled-bbox", type=parse_tiled_bbox, help="min_lon,min_lat,max_lon,max_lat of any size, e.g. a city: queried tile by tile")
    area.add_argument("--add-rotations", action="store_true", help="download nothing: add the camera rotations the viewer levels panoramas with to the dataset in --output")
    parser.add_argument("--radius-m", type=float, default=150, help="search radius for --center queries (default: 150)")
    parser.add_argument("--output", type=Path, required=True, help="directory for JPEGs and manifests")
    parser.add_argument("--max-images", type=int, default=25, help="maximum panoramas to download (default: 25)")
    parser.add_argument("--min-spacing-m", type=float, default=10, help="minimum distance between selected camera positions (default: 10)")
    parser.add_argument("--max-api-pages", type=int, default=3, help="maximum API pages to inspect (default: 3)")
    parser.add_argument("--image-size", choices=("auto", *MAPILLARY_IMAGE_SIZES), default="auto", help="preferred Mapillary derivative; falls back to smaller available versions")
    parser.add_argument("--access-token", help="Mapillary token; defaults to MAPILLARY_ACCESS_TOKEN")
    parser.add_argument("--dry-run", action="store_true", help="write manifests but do not download JPEGs")
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads (default: 4)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing JPEG files")
    return parser


def _checkpoint(output_dir: Path, records: list[dict[str, Any]], query: dict[str, Any]) -> None:
    write_manifest(output_dir, records, {**query, "panoramas_selected": len(records)})


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv()
    token = args.access_token or os.environ.get("MAPILLARY_ACCESS_TOKEN")
    if not token:
        print("Set MAPILLARY_ACCESS_TOKEN or pass --access-token.", file=sys.stderr)
        return 2
    if args.add_rotations:
        try:
            result = add_camera_rotations(args.output, token, args.workers)
        except (OSError, ValueError, KeyError, HTTPError, URLError, RuntimeError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        print(f"{args.output}: camera rotation for {result['found']} of the {result['asked']} panoramas that lacked it ({result['panoramas']} in all).")
        return 0
    if args.max_images < 1 or args.min_spacing_m < 0 or args.max_api_pages < 1 or args.workers < 1:
        print("max-images, max-api-pages and workers must be positive; min-spacing-m cannot be negative.", file=sys.stderr)
        return 2

    center = args.center
    try:
        if center:
            center_lat, center_lon = center
            bbox = bbox_from_center(center_lat, center_lon, args.radius_m)
            tiles = [bbox]
        elif args.tiled_bbox:
            bbox = args.tiled_bbox
            tiles = tile_bbox(bbox)
        else:
            bbox = args.bbox
            tiles = [bbox]
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    image_dir = args.output / "images"
    records: list[dict[str, Any]] = []
    selector = PanoramaSelector(center, args.radius_m if center else None, args.min_spacing_m, args.max_images)
    api_images_seen = 0
    truncated_tiles = 0
    last_checkpoint = time.monotonic()
    query = {
        "provider": "Mapillary Graph API v4",
        "acquired_at": datetime.now(UTC).isoformat(),
        "bbox_wgs84": bbox,
        "center_wgs84": center,
        "radius_m": args.radius_m if center else None,
        "only_panoramas": True,
        "min_camera_spacing_m": args.min_spacing_m,
        "preferred_image_size": args.image_size,
        "image_fields_requested": MAPILLARY_IMAGE_FIELDS,
        "tiles": len(tiles),
    }

    def fetch_one(image: dict[str, Any]) -> dict[str, Any] | None:
        image_id = str(image["id"])
        try:
            image["computed_compass_angle"] = normalize_compass_heading(image.get("computed_compass_angle"))
        except ValueError as error:
            print(f"Skipping {image_id}: {error}", file=sys.stderr)
            return None
        filename = f"pano-{image_id}.jpg"
        derivative = available_image_url(image, args.image_size)
        if derivative is None:
            print(f"Skipping {image_id}: Mapillary did not return a downloadable derivative.", file=sys.stderr)
            return None
        image_field, url = derivative
        image["_download_image_field"] = image_field
        local_metadata = None
        if not args.dry_run:
            image_dir.mkdir(parents=True, exist_ok=True)
            try:
                _, local_metadata = download_with_retries(url, image_dir / filename, args.overwrite)
            except (HTTPError, URLError, RuntimeError, TimeoutError, ConnectionError) as error:
                print(f"Skipping {image_id}: {error}", file=sys.stderr)
                return None
        return image_record(image, f"images/{filename}" if not args.dry_run else None, local_metadata)

    for number, tile in enumerate(tiles, start=1):
        if len(selector.selected) >= args.max_images:
            break
        try:
            if len(tiles) > 1:
                status(f"Tile {number}/{len(tiles)}: {len(records)} panoramas so far.")
            images = fetch_images(token, tile, MAPILLARY_IMAGE_FIELDS, args.max_api_pages)
        except (RuntimeError, ValueError) as error:
            if len(tiles) == 1:
                print(f"Error: {error}", file=sys.stderr)
                return 1
            print(f"Tile {number}/{len(tiles)} skipped: {error}", file=sys.stderr)
            continue
        api_images_seen += len(images)
        if len(images) >= args.max_api_pages * 2_000:
            truncated_tiles += 1
        new = selector.add(images)
        if not new and number == 1 and len(tiles) == 1:
            status("No 360 panoramas found for this area. Try another point or a larger radius.")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for done, record in enumerate(pool.map(fetch_one, new), start=1):
                if record is not None:
                    records.append(record)
                if done % 50 == 0 or done == len(new):
                    status(f"  Downloaded {done}/{len(new)} of this tile ({len(records)} total).")
        # A crash later keeps what was fetched so far (rerunning skips the files already on disk). Rewriting
        # the manifest costs more as it grows, so it is saved at most every CHECKPOINT_INTERVAL_S.
        if time.monotonic() - last_checkpoint >= CHECKPOINT_INTERVAL_S:
            _checkpoint(args.output, records, {**query, "api_images_seen": api_images_seen})
            last_checkpoint = time.monotonic()
    if truncated_tiles:
        status(f"Warning: {truncated_tiles} tile(s) reached --max-api-pages; raise it or use smaller tiles to see every panorama there.")
    status("Writing GeoJSON, JSON, and attribution manifests...")
    _checkpoint(args.output, records, {**query, "api_images_seen": api_images_seen, "tiles_truncated": truncated_tiles})
    print(f"Selected {len(records)} georeferenced panoramas from {api_images_seen} API images.")
    print(f"Dataset manifest: {args.output / 'manifest.geojson'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
