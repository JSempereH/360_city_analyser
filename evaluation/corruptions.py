"""Synthetic image degradations for the robustness benchmark.

Crowdsourced street imagery is blurred, noisy, badly exposed or partly hidden.
Following OpenFACADES (after Hendrycks and Dietterich), each corruption has
three severities, is deterministic for a given seed, and acts on an RGB
``uint8`` equirectangular array. ``relative_degradation`` and
``relative_mce`` turn per-severity errors into the scores the paper reports.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.ndimage import uniform_filter1d

SEVERITIES = (1, 2, 3)
# Share of the panorama a black square hides.
OCCLUSION_AREA = {1: 0.10, 2: 0.25, 3: 0.50}
# Horizontal box-blur length in pixels at 2048 px width (scaled with the image).
MOTION_BLUR_PIXELS = {1: 9, 2: 25, 3: 51}
NOISE_SIGMA = {1: 12.0, 2: 30.0, 3: 60.0}
BRIGHT_GAIN = {1: 1.3, 2: 1.7, 3: 2.5}
DARK_GAIN = {1: 0.7, 2: 0.4, 3: 0.15}
# Facades sit around the horizon: occlusions are centered on this elevation band.
OCCLUSION_ROW_CENTER = 0.42


def _check(severity: int) -> None:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}")


def occlusion(image: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    _check(severity)
    height, width = image.shape[:2]
    side = int(round((OCCLUSION_AREA[severity] * height * width) ** 0.5))
    side_y, side_x = min(side, height), min(int(round(OCCLUSION_AREA[severity] * height * width / max(min(side, height), 1))), width)
    rng = np.random.default_rng(seed)
    left = int(rng.integers(0, max(width - side_x, 0) + 1))
    top = int(np.clip(OCCLUSION_ROW_CENTER * height - side_y / 2, 0, height - side_y))
    result = image.copy()
    result[top:top + side_y, left:left + side_x] = 0
    return result


def motion_blur(image: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    _check(severity)
    length = max(1, round(MOTION_BLUR_PIXELS[severity] * image.shape[1] / 2048))
    blurred = uniform_filter1d(image.astype(np.float32), size=length, axis=1, mode="wrap")
    return np.clip(blurred, 0, 255).astype(np.uint8)


def gaussian_noise(image: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    _check(severity)
    noise = np.random.default_rng(seed).normal(0.0, NOISE_SIGMA[severity], image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def brightness_up(image: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    _check(severity)
    return np.clip(image.astype(np.float32) * BRIGHT_GAIN[severity], 0, 255).astype(np.uint8)


def brightness_down(image: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    _check(severity)
    return np.clip(image.astype(np.float32) * DARK_GAIN[severity], 0, 255).astype(np.uint8)


CORRUPTIONS: dict[str, Callable[[np.ndarray, int, int], np.ndarray]] = {
    "occlusion": occlusion,
    "motion_blur": motion_blur,
    "gaussian_noise": gaussian_noise,
    "brightness_up": brightness_up,
    "brightness_down": brightness_down,
}


def corrupt(image: np.ndarray, name: str, severity: int, seed: int = 0) -> np.ndarray:
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {name!r}; choose from {sorted(CORRUPTIONS)}")
    return CORRUPTIONS[name](image, severity, seed)


def parse_corruption(value: str) -> tuple[str, int]:
    """``motion_blur:2`` to ``("motion_blur", 2)``."""
    name, _, severity = value.partition(":")
    if name not in CORRUPTIONS or not severity.isdecimal() or int(severity) not in SEVERITIES:
        raise ValueError(f"expected NAME:SEVERITY with NAME in {sorted(CORRUPTIONS)} and SEVERITY in {SEVERITIES}, got {value!r}")
    return name, int(severity)


def relative_degradation(clean_error: float, errors_by_severity: dict[int, float]) -> float:
    """Error added by one corruption, summed over its severities."""
    return float(sum(error - clean_error for error in errors_by_severity.values()))


def relative_mce(degradations: dict[str, float], reference: dict[str, float]) -> float | None:
    """Mean over corruptions of this configuration's degradation divided by the reference's.

    1.0 means as fragile as the reference, below 1.0 more robust. Corruptions
    the reference does not degrade are skipped (the ratio is undefined).
    """
    ratios = [degradations[name] / reference[name] for name in degradations if name in reference and reference[name] > 0]
    return float(np.mean(ratios)) if ratios else None
