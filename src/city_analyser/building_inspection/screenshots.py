"""Generate reproducible perspective screenshots from ranked panorama evidence."""

from __future__ import annotations

import math
from pathlib import Path

from ..panorama_geometry import perspective_tile

from .models import RankedView


SCREENSHOT_SIZE = 960
MAX_REPORT_SCREENSHOTS = 4


def render_screenshots(dataset_dir: Path, view: RankedView, clean_destination: Path, annotated_destination: Path) -> None:
    np = __import__("numpy")
    Image = __import__("PIL.Image", fromlist=["Image"])
    image_path = (dataset_dir / view.evidence.local_file).resolve()
    if not image_path.is_file() or not image_path.is_relative_to(dataset_dir.resolve()):
        raise ValueError("Panorama image is unavailable.")
    analysis_dir = dataset_dir / "analysis"
    with Image.open(image_path) as source:
        panorama = np.asarray(source.convert("RGB"))
    with Image.open(analysis_dir / f"{view.evidence.image_id}-ids.png") as source:
        observed_mask = np.asarray(source)
    with Image.open(analysis_dir / f"{view.evidence.image_id}-facades.png") as source:
        inferred_mask = np.asarray(source)
    yaw = math.radians(view.evidence.yaw_deg)
    pitch = math.radians(view.evidence.pitch_deg)
    tile, _, _ = perspective_tile(panorama, yaw, pitch, SCREENSHOT_SIZE, view.evidence.fov_deg)
    observed, _, _ = perspective_tile(observed_mask, yaw, pitch, SCREENSHOT_SIZE, view.evidence.fov_deg)
    inferred, _, _ = perspective_tile(inferred_mask, yaw, pitch, SCREENSHOT_SIZE, view.evidence.fov_deg)
    annotated = tile.astype(np.float32)
    observed = observed == view.evidence.mask_ordinal
    inferred = inferred == view.evidence.mask_ordinal
    annotated[inferred] = annotated[inferred] * 0.72 + np.array([51, 166, 255], dtype=np.float32) * 0.28
    annotated[observed] = annotated[observed] * 0.62 + np.array([255, 145, 31], dtype=np.float32) * 0.38
    clean_destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tile).save(clean_destination, format="JPEG", quality=90, optimize=True)
    Image.fromarray(np.clip(annotated, 0, 255).astype(np.uint8)).save(annotated_destination, format="JPEG", quality=90, optimize=True)
