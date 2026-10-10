"""Cheap, deterministic quality facts for one equirectangular panorama.

Blur, exposure and file size decide whether facade evidence can be trusted
(OpenFACADES filters the same properties before building attributes are read
from an image). The facts are recorded in the analysis metadata, flags are
derived from configurable thresholds, and ranking penalizes flagged views; no
panorama is hidden from the analysis.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

QUALITY_VERSION = 1
# Fixed width, so the same scene scores the same at any download resolution.
QUALITY_WIDTH = 2048
# Facades live between -20 and +40 degrees of elevation; sky and nadir (which
# stitching often blacks out) would dominate a whole-image statistic.
FACADE_BAND_DEGREES = (-20.0, 40.0)
DEFAULT_MIN_SHARPNESS = 100.0
DEFAULT_MIN_BRIGHTNESS = 20.0
DEFAULT_MAX_BRIGHTNESS = 200.0
DEFAULT_MIN_FILE_BYTES = 20 * 1024
# Ranking weight of a flagged view; one multiplier whatever the number of flags.
FLAGGED_RANK_WEIGHT = 0.5


def _threshold(name: str, default: float) -> float:
    configured = os.environ.get(name, "").strip()
    try:
        return float(configured) if configured else default
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number.") from error


def quality_thresholds() -> dict[str, float]:
    return {
        "min_sharpness": _threshold("BUILDING_ANALYSIS_MIN_SHARPNESS", DEFAULT_MIN_SHARPNESS),
        "min_brightness": _threshold("BUILDING_ANALYSIS_MIN_BRIGHTNESS", DEFAULT_MIN_BRIGHTNESS),
        "max_brightness": _threshold("BUILDING_ANALYSIS_MAX_BRIGHTNESS", DEFAULT_MAX_BRIGHTNESS),
        "min_file_bytes": _threshold("BUILDING_ANALYSIS_MIN_FILE_BYTES", DEFAULT_MIN_FILE_BYTES),
    }


def laplacian_variance(gray: Any) -> float:
    """Variance of the 4-neighbor Laplacian: low values mean few sharp edges (blur)."""
    np = __import__("numpy")
    values = gray.astype(np.float32)
    laplacian = values[:-2, 1:-1] + values[2:, 1:-1] + values[1:-1, :-2] + values[1:-1, 2:] - 4 * values[1:-1, 1:-1]
    return float(laplacian.var())


def quality_flags(facts: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    flags = []
    if facts["sharpness"] < thresholds["min_sharpness"]:
        flags.append("blurry")
    if facts["brightness"] < thresholds["min_brightness"]:
        flags.append("dark")
    if facts["brightness"] > thresholds["max_brightness"]:
        flags.append("overexposed")
    if facts["file_bytes"] < thresholds["min_file_bytes"]:
        flags.append("small_file")
    return flags


def panorama_quality(image_path: Path) -> dict[str, Any]:
    np = __import__("numpy")
    Image = importlib.import_module("PIL.Image")
    with Image.open(image_path) as source:
        gray = source.convert("L")
        gray = gray.resize((QUALITY_WIDTH, QUALITY_WIDTH // 2), Image.Resampling.LANCZOS)
    height = gray.height
    top = round(height * (90.0 - FACADE_BAND_DEGREES[1]) / 180.0)
    bottom = round(height * (90.0 - FACADE_BAND_DEGREES[0]) / 180.0)
    band = np.asarray(gray)[top:bottom]
    facts: dict[str, Any] = {
        "version": QUALITY_VERSION,
        "sharpness": round(laplacian_variance(band), 2),
        "brightness": round(float(band.mean()), 2),
        "file_bytes": image_path.stat().st_size,
    }
    facts["flags"] = quality_flags(facts, quality_thresholds())
    return facts


def rank_weight(quality: dict[str, Any] | None) -> float:
    """Ranking multiplier: 1 for unknown or clean views, ``FLAGGED_RANK_WEIGHT`` for flagged ones."""
    if not isinstance(quality, dict) or not quality.get("flags"):
        return 1.0
    return FLAGGED_RANK_WEIGHT
