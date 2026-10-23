#!/usr/bin/env python3
"""Contact sheets to check building masks by eye: one image per building, one panel per view.

Each panel is the panorama cropped around the building's mask (equirectangular,
so nothing is resampled), the mask filled in orange with a yellow outline, other
buildings outlined in blue, the building's OSM footprint projected from the
analysis pose outlined in green (where the building should be), and the view's
facts in the header. Run it with the BUILDING_ANALYSIS_* settings of the analysis.

    uv run python tools/mask_review.py valencia-centro-demo --output /tmp/masks [--building way/1] [--min-score 0.01]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PAD = 0.08  # of the panorama width, around the mask
PANEL_HEIGHT = 420


def _views(dataset_dir: Path) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {}
    for path in sorted((dataset_dir / "analysis").glob("*.json")):
        if path.name.endswith("-sfm.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for ordinal, record in (payload.get("building_ids") or {}).items():
            found.setdefault(record["osm_id"], []).append({"image": str(payload["image_id"]), "ordinal": int(ordinal), "pixels": record.get("pixels", 0)})
    return found


def _projected(analysis: dict, image: dict, building_id: str, width: int, height: int) -> np.ndarray | None:
    """Pixels where the building's OSM walls project, from the pose the analysis used."""
    from city_analyser.building_analysis import _render_facade_owners, camera_up_vector, nearby_buildings
    from city_analyser.multi_view_analysis import panorama_coordinates

    pose = analysis.get("refined_pose") or {}
    longitude, latitude = panorama_coordinates(image)
    heading = pose["heading_degrees"] if pose.get("accepted") else float(image["computed_compass_angle_deg"])
    camera = (pose.get("latitude", latitude), pose.get("longitude", longitude))
    features = nearby_buildings(latitude, longitude)
    owners, _, candidates, _ = _render_facade_owners(features, camera[0], camera[1], heading, width, height, camera_up_vector(analysis.get("camera_up")))
    for ordinal, feature in enumerate(candidates, start=1):
        if feature["properties"].get("osm_id") == building_id:
            return owners == ordinal
    return None


def _panel(panorama: np.ndarray, ids: np.ndarray, ordinal: int, title: str, projected: np.ndarray | None = None) -> Image.Image:
    height, width = ids.shape
    scale_x, scale_y = panorama.shape[1] / width, panorama.shape[0] / height
    mask = ids == ordinal
    rows, columns = np.nonzero(mask)
    # Center on the mask by its circular mean, so a building across the panorama seam stays whole.
    angles = columns / width * 2 * np.pi
    center = int((np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()) % (2 * np.pi)) / (2 * np.pi) * width)
    shift = width // 2 - center
    ids_rolled = np.roll(ids, shift, axis=1)
    projected_rolled = np.roll(projected, shift, axis=1) if projected is not None else None
    pano_rolled = np.roll(panorama, int(shift * scale_x), axis=1)
    rows, columns = np.nonzero(ids_rolled == ordinal)
    pad = int(PAD * width)
    left, right = max(0, columns.min() - pad), min(width, columns.max() + pad)
    top, bottom = max(0, rows.min() - pad // 2), min(height, rows.max() + pad // 2)
    crop_ids = ids_rolled[top:bottom, left:right]
    crop_projected = projected_rolled[top:bottom, left:right] if projected_rolled is not None else None
    crop = pano_rolled[int(top * scale_y):int(bottom * scale_y), int(left * scale_x):int(right * scale_x)]
    image = Image.fromarray(crop).resize((crop_ids.shape[1], crop_ids.shape[0]))
    overlay = np.array(image).astype(np.float32)
    selected = crop_ids == ordinal
    overlay[selected] = overlay[selected] * 0.45 + np.array([255, 140, 30]) * 0.55
    others = (crop_ids > 0) & ~selected
    edges = others & (np.roll(crop_ids, 1, 0) != crop_ids) | others & (np.roll(crop_ids, 1, 1) != crop_ids)
    overlay[edges] = [40, 120, 255]
    outline = selected & ((np.roll(selected, 1, 0) != selected) | (np.roll(selected, 1, 1) != selected))
    overlay[outline] = [255, 230, 0]
    if crop_projected is not None:
        border = crop_projected & ((np.roll(crop_projected, 1, 0) != crop_projected) | (np.roll(crop_projected, 1, 1) != crop_projected))
        border = border | np.roll(border, 1, 0) | np.roll(border, 1, 1)
        overlay[border] = [0, 230, 80]
    result = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    result = result.resize((max(1, int(result.width * PANEL_HEIGHT / result.height)), PANEL_HEIGHT))
    canvas = Image.new("RGB", (max(result.width, 360), PANEL_HEIGHT + 22), (20, 20, 20))
    canvas.paste(result, (0, 22))
    ImageDraw.Draw(canvas).text((6, 5), title, fill=(255, 255, 255))
    return canvas


def _map_inset(features: list[dict], building_id: str, camera: tuple[float, float], size: int = 340, radius_m: float = 60.0) -> Image.Image:
    """Footprints around the camera (north up), the target in green, bearing rays every 30 degrees."""
    import math

    latitude, longitude = camera
    scale = size / (2 * radius_m)

    def point(lon: float, lat: float) -> tuple[float, float]:
        east = (lon - longitude) * 111_320 * math.cos(math.radians(latitude))
        north = (lat - latitude) * 110_574
        return size / 2 + east * scale, size / 2 - north * scale

    image = Image.new("RGB", (size, size), (245, 245, 240))
    draw = ImageDraw.Draw(image)
    for feature in features:
        geometry = feature.get("geometry") or {}
        polygons = [geometry["coordinates"]] if geometry.get("type") == "Polygon" else geometry.get("coordinates", []) if geometry.get("type") == "MultiPolygon" else []
        target = feature.get("properties", {}).get("osm_id") == building_id
        for polygon in polygons:
            ring = [point(x, y) for x, y in polygon[0]]
            if len(ring) > 2:
                draw.polygon(ring, fill=(60, 200, 90) if target else (200, 200, 195), outline=(0, 120, 40) if target else (120, 120, 115))
    for bearing in range(0, 360, 30):
        angle = math.radians(bearing)
        end = (size / 2 + math.sin(angle) * size * 0.48, size / 2 - math.cos(angle) * size * 0.48)
        draw.line([(size / 2, size / 2), end], fill=(90, 90, 255), width=1)
        draw.text((end[0] - 10, end[1] - 6), str(bearing), fill=(20, 20, 160))
    draw.ellipse([size / 2 - 5, size / 2 - 5, size / 2 + 5, size / 2 + 5], fill=(230, 40, 40))
    draw.text((4, 4), f"{radius_m:.0f} m radius, north up", fill=(0, 0, 0))
    return image


def _full_panel(panorama: np.ndarray, ids: np.ndarray, ordinal: int, title: str, projected: np.ndarray | None, heading: float, inset: Image.Image | None = None) -> Image.Image:
    """The whole panorama with compass bearings, to tell which side of the street a mask is on."""
    overlay = panorama.astype(np.float32)
    selected = ids == ordinal
    overlay[selected] = overlay[selected] * 0.45 + np.array([255, 140, 30]) * 0.55
    others = ids > 0
    edges = others & ((np.roll(ids, 1, 0) != ids) | (np.roll(ids, 1, 1) != ids)) & ~selected
    overlay[edges] = [40, 120, 255]
    if projected is not None:
        border = projected & ((np.roll(projected, 1, 0) != projected) | (np.roll(projected, 1, 1) != projected))
        for shift in range(3):
            overlay[np.roll(border, shift, 0)] = [0, 230, 80]
    image = Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).resize((1600, 800))
    canvas = Image.new("RGB", (1600, 800 + 40), (20, 20, 20))
    canvas.paste(image, (0, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill=(255, 255, 255))
    for bearing in range(0, 360, 30):
        x = int(((bearing - heading + 180) % 360) / 360 * 1600)
        draw.line([(x, 22), (x, 822)], fill=(255, 255, 255), width=1)
        draw.text((x + 3, 824), f"{bearing}°", fill=(255, 255, 255))
    if inset is not None:
        canvas.paste(inset, (canvas.width - inset.width - 4, 26))
    return canvas


def presentation_frames(dataset_dir: Path, building_ids: list[str] | None, output: Path) -> None:
    """Each tour view as the presentation shows it (16:9, same direction, tilt and zoom), mask included."""
    import math

    from city_analyser.panorama_geometry import perspective_tile

    manifest = {str(image["id"]): image for image in json.loads((dataset_dir / "manifest.json").read_text())["images"]}
    for sidecar_path in sorted((dataset_dir / "crops").rglob("crop-*.json")):
        sidecar = json.loads(sidecar_path.read_text())
        building_id, image_id = sidecar["building"]["id"], str(sidecar["source"]["image_id"])
        if building_ids and building_id not in building_ids:
            continue
        analysis = json.loads((dataset_dir / "analysis" / f"{image_id}.json").read_text())
        ordinal = next(int(key) for key, record in analysis["building_ids"].items() if record["osm_id"] == building_id)
        view = sidecar["view"]
        # The same framing rules as presentation.mjs.
        pitch, fov = max(-5.0, min(22.0, view["pitch_deg"])), max(60.0, min(95.0, view["fov_deg"] * 1.1))
        with Image.open(dataset_dir / "analysis" / f"{image_id}-ids.png") as source:
            ids = np.asarray(source).astype(np.int32)
        with Image.open(dataset_dir / manifest[image_id]["local_file"]) as source:
            panorama = np.asarray(source.convert("RGB").resize((ids.shape[1] * 2, ids.shape[0] * 2)))
        args = (math.radians(view["yaw_deg"]), math.radians(pitch), 960, fov, 540)
        tile, _, _ = perspective_tile(panorama, *args)
        labels, _, _ = perspective_tile(ids, *args)
        overlay = tile.astype(np.float32)
        selected = labels == ordinal
        overlay[selected] = overlay[selected] * 0.58 + np.array([255, 145, 30]) * 0.42
        outline = selected & ((np.roll(selected, 1, 0) != selected) | (np.roll(selected, 1, 1) != selected))
        overlay[outline] = [255, 220, 140]
        image = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
        ImageDraw.Draw(image).text((8, 6), f"{building_id} | {image_id} | yaw {view['yaw_deg']:.0f} pitch {pitch:.0f} fov {fov:.0f}", fill=(255, 255, 0))
        image.save(output / f"{building_id.replace('/', '_')}__{image_id}.jpg", quality=85)
        print(building_id, image_id)


PALETTE = [(255, 140, 30), (40, 160, 255), (60, 200, 90), (230, 60, 160), (250, 220, 40), (140, 90, 230), (30, 210, 210), (240, 80, 60)]


def overview(dataset_dir: Path, output: Path) -> None:
    """One image per panorama: every building mask in its own color, labelled, with compass bearings."""
    from PIL import ImageFont  # noqa: F401  (default bitmap font)

    manifest = {str(image["id"]): image for image in json.loads((dataset_dir / "manifest.json").read_text())["images"]}
    for path in sorted((dataset_dir / "analysis").glob("*.json")):
        if path.name.endswith("-sfm.json"):
            continue
        analysis = json.loads(path.read_text())
        image_id = str(analysis["image_id"])
        with Image.open(dataset_dir / "analysis" / f"{image_id}-ids.png") as source:
            ids = np.asarray(source).astype(np.int32)
        with Image.open(dataset_dir / manifest[image_id]["local_file"]) as source:
            panorama = np.asarray(source.convert("RGB").resize((ids.shape[1], ids.shape[0])))
        overlay = panorama.astype(np.float32)
        labels = []
        for index, (ordinal, record) in enumerate(sorted(analysis["building_ids"].items(), key=lambda item: int(item[0]))):
            mask = ids == int(ordinal)
            if not mask.any():
                continue
            color = np.array(PALETTE[index % len(PALETTE)])
            overlay[mask] = overlay[mask] * 0.5 + color * 0.5
            edge = mask & ((np.roll(mask, 1, 0) != mask) | (np.roll(mask, 1, 1) != mask))
            overlay[edge] = color
            rows, columns = np.nonzero(mask)
            labels.append((int(np.median(columns)), int(rows.min()), f"{record['osm_id']} {int(mask.sum())}", tuple(int(c) for c in color)))
        width = 2400
        scale = width / ids.shape[1]
        image = Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).resize((width, int(ids.shape[0] * scale)))
        canvas = Image.new("RGB", (width, image.height + 24), (20, 20, 20))
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        heading = float(analysis["heading_degrees"])
        for bearing in range(0, 360, 10):
            x = int(((bearing - heading + 180) % 360) / 360 * width)
            draw.line([(x, 0), (x, image.height)], fill=(255, 255, 255) if bearing % 30 == 0 else (120, 120, 120))
            if bearing % 30 == 0:
                draw.text((x + 3, image.height + 6), f"{bearing}", fill=(255, 255, 255))
        for x, y, text, color in labels:
            draw.rectangle([x * scale - 2, y * scale - 2, x * scale + 6 * len(text) + 2, y * scale + 12], fill=(0, 0, 0))
            draw.text((x * scale, y * scale), text, fill=color)
        canvas.save(output / f"overview__{image_id}.jpg", quality=86)
        print(image_id, len(labels), "buildings")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--building", action="append")
    parser.add_argument("--min-pixels", type=int, default=0, help="skip views whose mask is smaller")
    parser.add_argument("--full", action="store_true", help="one image per view: the whole panorama with compass bearings")
    parser.add_argument("--overview", action="store_true", help="one image per panorama with every building mask labelled")
    parser.add_argument("--presentation", action="store_true", help="one image per exported crop, framed as the viewer's presentation mode shows it")
    args = parser.parse_args()
    dataset_dir = args.data_root / args.dataset
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {str(image["id"]): image for image in json.loads((dataset_dir / "manifest.json").read_text())["images"]}
    if args.overview:
        overview(dataset_dir, args.output)
        return
    if args.presentation:
        presentation_frames(dataset_dir, args.building, args.output)
        return
    views = _views(dataset_dir)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    analyses = {path.stem: json.loads(path.read_text()) for path in (dataset_dir / "analysis").glob("*.json") if not path.name.endswith("-sfm.json")}
    for building_id in sorted(args.building or views):
        panels = []
        for view in sorted(views.get(building_id, []), key=lambda item: -item["pixels"]):
            if view["pixels"] < args.min_pixels:
                continue
            if view["image"] not in cache:
                with Image.open(dataset_dir / "analysis" / f"{view['image']}-ids.png") as source:
                    ids = np.asarray(source).astype(np.int32)
                with Image.open(dataset_dir / manifest[view["image"]]["local_file"]) as source:
                    panorama = np.asarray(source.convert("RGB").resize((ids.shape[1], ids.shape[0])))
                cache = {view["image"]: (panorama, ids)}  # one panorama in memory at a time
            panorama, ids = cache[view["image"]]
            projected = _projected(analyses[view["image"]], manifest[view["image"]], building_id, ids.shape[1], ids.shape[0])
            if args.full:
                analysis = analyses[view["image"]]
                pose = analysis.get("refined_pose") or {}
                heading = pose["heading_degrees"] if pose.get("accepted") else float(manifest[view["image"]]["computed_compass_angle_deg"])
                from city_analyser.building_analysis import nearby_buildings
                from city_analyser.multi_view_analysis import panorama_coordinates

                longitude, latitude = panorama_coordinates(manifest[view["image"]])
                camera = (pose.get("latitude", latitude), pose.get("longitude", longitude))
                inset = _map_inset(nearby_buildings(latitude, longitude), building_id, camera)
                _full_panel(panorama, ids, view["ordinal"], f"{building_id} | image {view['image']} | {view['pixels']} px | heading {heading:.0f}", projected, heading, inset).save(
                    args.output / f"{building_id.replace('/', '_')}__{view['image']}.jpg", quality=85)
                continue
            panels.append(_panel(panorama, ids, view["ordinal"], f"{building_id} | image {view['image']} | ordinal {view['ordinal']} | {view['pixels']} px", projected))
        if not panels:
            continue
        sheet = Image.new("RGB", (sum(panel.width for panel in panels) + 6 * (len(panels) - 1), panels[0].height), (0, 0, 0))
        x = 0
        for panel in panels:
            sheet.paste(panel, (x, 0))
            x += panel.width + 6
        if sheet.width > 2400:
            sheet = sheet.resize((2400, int(sheet.height * 2400 / sheet.width)))
        sheet.save(args.output / f"{building_id.replace('/', '_')}.jpg", quality=85)
        print(building_id, len(panels), "views")


if __name__ == "__main__":
    main()
