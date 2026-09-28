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
        status(f"  Consulta temporalmente fallida ({last_error}); reintentando {attempt + 1}/3...")
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
        status(f"Consultando Mapillary, pagina {page}{f'/{max_pages}' if max_pages is not None else ''}...")
        payload = request_json(next_url, next_params)
        data = payload.get("data", [])
        if not isinstance(data, list):
            raise RuntimeError("Mapillary API response did not contain an image list")
        images.extend(image for image in data if isinstance(image, dict))
        status(f"  Recibidas {len(data)} imagenes; acumuladas: {len(images)}.")
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


def select_panoramas(
    images: list[dict[str, Any]],
    center: tuple[float, float] | None,
    radius_m: float | None,
    min_spacing_m: float,
    max_images: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for image in images:
        if image.get("is_pano") is not True:
            continue
        point = point_from_image(image)
        if point is None:
            continue
        lon, lat = point
        if center and radius_m and haversine_m(center[0], center[1], lat, lon) > radius_m:
            continue
        image = dict(image)
        image["_selected_geometry"] = {"type": "Point", "coordinates": [lon, lat]}
        candidates.append(image)

    candidates.sort(key=lambda image: (image.get("captured_at", 0), image.get("id", "")))
    selected: list[dict[str, Any]] = []
    for image in candidates:
        lon, lat = image["_selected_geometry"]["coordinates"]
        if any(
            haversine_m(lat, lon, previous["_selected_geometry"]["coordinates"][1], previous["_selected_geometry"]["coordinates"][0])
            < min_spacing_m
            for previous in selected
        ):
            continue
        selected.append(image)
        if len(selected) >= max_images:
            break
    return selected


def normalize_compass_heading(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("computed compass angle must be a finite number")
    heading = float(value)
    if not math.isfinite(heading):
        raise ValueError("computed compass angle must be a finite number")
    return heading % 360.0


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


def write_manifest(output_dir: Path, records: list[dict[str, Any]], query: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps({"query": query, "images": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    features = [
        {"type": "Feature", "geometry": record["selected_geometry"], "properties": {key: value for key, value in record.items() if key != "selected_geometry"}}
        for record in records
    ]
    (output_dir / "manifest.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    attribution_lines = [
        "# Attribution", "", "Mapillary images are licensed CC BY-SA 4.0. Keep this file with the images and review Mapillary's current terms before redistributing a dataset.", ""
    ]
    for record in records:
        username = record["creator"].get("username") or record["creator"].get("id") or "unknown contributor"
        attribution_lines.append(f"- Image {record['id']} by {username}: {record['source_page']} (CC BY-SA 4.0)")
    (output_dir / "ATTRIBUTION.md").write_text("\n".join(attribution_lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a small georeferenced Mapillary panorama sample.")
    area = parser.add_mutually_exclusive_group(required=True)
    area.add_argument("--center", type=parse_center, help="WGS84 latitude,longitude at the street center")
    area.add_argument("--bbox", type=parse_bbox, help="min_lon,min_lat,max_lon,max_lat around one street")
    parser.add_argument("--radius-m", type=float, default=150, help="search radius for --center queries (default: 150)")
    parser.add_argument("--output", type=Path, required=True, help="directory for JPEGs and manifests")
    parser.add_argument("--max-images", type=int, default=25, help="maximum panoramas to download (default: 25)")
    parser.add_argument("--min-spacing-m", type=float, default=10, help="minimum distance between selected camera positions (default: 10)")
    parser.add_argument("--max-api-pages", type=int, default=3, help="maximum API pages to inspect (default: 3)")
    parser.add_argument("--image-size", choices=("auto", *MAPILLARY_IMAGE_SIZES), default="auto", help="preferred Mapillary derivative; falls back to smaller available versions")
    parser.add_argument("--access-token", help="Mapillary token; defaults to MAPILLARY_ACCESS_TOKEN")
    parser.add_argument("--dry-run", action="store_true", help="write manifests but do not download JPEGs")
    parser.add_argument("--overwrite", action="store_true", help="replace existing JPEG files")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv()
    token = args.access_token or os.environ.get("MAPILLARY_ACCESS_TOKEN")
    if not token:
        print("Set MAPILLARY_ACCESS_TOKEN or pass --access-token.", file=sys.stderr)
        return 2
    if args.max_images < 1 or args.min_spacing_m < 0 or args.max_api_pages < 1:
        print("max-images and max-api-pages must be positive; min-spacing-m cannot be negative.", file=sys.stderr)
        return 2

    center = args.center
    try:
        if center:
            center_lat, center_lon = center
            bbox = bbox_from_center(center_lat, center_lon, args.radius_m)
        else:
            bbox = args.bbox
        images = fetch_images(token, bbox, MAPILLARY_IMAGE_FIELDS, args.max_api_pages)
        status("Filtering 360 panoramas and spacing nearby positions...")
        selected = select_panoramas(images, center, args.radius_m if center else None, args.min_spacing_m, args.max_images)
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    image_dir = args.output / "images"
    records: list[dict[str, Any]] = []
    if not selected:
        status("No 360 panoramas found for this area. Try another point or a larger radius.")
    for index, image in enumerate(selected, start=1):
        image_id = str(image["id"])
        try:
            image["computed_compass_angle"] = normalize_compass_heading(image.get("computed_compass_angle"))
        except ValueError as error:
            print(f"Skipping {image_id}: {error}", file=sys.stderr)
            continue
        filename = f"pano-{image_id}.jpg"
        destination = image_dir / filename
        derivative = available_image_url(image, args.image_size)
        if derivative is None:
            print(f"Skipping {image_id}: Mapillary did not return a downloadable derivative.", file=sys.stderr)
            continue
        image_field, url = derivative
        image["_download_image_field"] = image_field
        local_metadata = None
        if not args.dry_run:
            image_dir.mkdir(parents=True, exist_ok=True)
            try:
                status(f"Downloading panorama {index}/{len(selected)} ({image_id})...")
                downloaded, local_metadata = download_image(url, destination, args.overwrite)
                status("  Saved." if downloaded else "  Already exists; keeping the local file.")
            except (HTTPError, URLError, RuntimeError, TimeoutError) as error:
                print(f"Skipping {image_id}: {error}", file=sys.stderr)
                continue
        else:
            status(f"Selected panorama {index}/{len(selected)} ({image_id}); dry-run, not downloading.")
        records.append(image_record(image, f"images/{filename}" if not args.dry_run else None, local_metadata))

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
        "api_images_seen": len(images),
        "panoramas_selected": len(records),
    }
    status("Writing GeoJSON, JSON, and attribution manifests...")
    write_manifest(args.output, records, query)
    print(f"Selected {len(records)} georeferenced panoramas from {len(images)} API images.")
    print(f"Dataset manifest: {args.output / 'manifest.geojson'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
