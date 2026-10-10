"""Combine per-view answers into one prediction per building, abstaining when weak.

OpenFACADES keeps the most frequent label across views. Here views are
weighted by their ranking score (image quality included), the result keeps the
whole distribution, and a building with too few views or too little agreement
abstains instead of receiving a label.
"""

from __future__ import annotations

import math
from typing import Sequence

from .models import CategoricalPrediction, NumericPrediction, abstain_categorical, abstain_numeric

DEFAULT_MIN_VIEWS = 1
DEFAULT_MIN_SHARE = 0.5


def fuse_categorical(
    attribute: str,
    classes: Sequence[str],
    per_view: Sequence[tuple[dict[str, float], float]],
    source: str,
    *,
    kind: str = "vote_share",
    min_views: int = DEFAULT_MIN_VIEWS,
    min_share: float = DEFAULT_MIN_SHARE,
) -> CategoricalPrediction:
    """Weighted mean of per-view score dictionaries over ``classes``; abstain below ``min_share`` or ``min_views``."""
    usable = [(scores, weight) for scores, weight in per_view if scores and weight > 0]
    if len(usable) < min_views or not usable:
        return abstain_categorical(attribute, source, f"{len(usable)} usable views, {min_views} required")
    total = sum(weight for _, weight in usable)
    pooled = {name: sum(scores.get(name, 0.0) * weight for scores, weight in usable) / total for name in classes}
    mass = sum(pooled.values())
    if mass <= 0:
        return abstain_categorical(attribute, source, "views named no known class")
    pooled = {name: round(value / mass, 6) for name, value in pooled.items()}
    best = max(pooled.values())
    if best < min_share:
        return CategoricalPrediction(attribute, pooled, kind, source, True, f"top share {best:.2f} is below {min_share:.2f}")
    return CategoricalPrediction(attribute, pooled, kind, source)


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    order = sorted(range(len(values)), key=lambda index: values[index])
    half = sum(weights) / 2
    running = 0.0
    for index in order:
        running += weights[index]
        if running >= half:
            return values[index]
    return values[order[-1]]


def fuse_numeric(
    attribute: str, per_view: Sequence[tuple[float | None, float]], source: str, *, min_views: int = DEFAULT_MIN_VIEWS,
) -> NumericPrediction:
    """Weighted median, with the weighted median absolute deviation as the spread."""
    usable = [(value, weight) for value, weight in per_view if value is not None and math.isfinite(value) and weight > 0]
    if len(usable) < min_views or not usable:
        return abstain_numeric(attribute, source, f"{len(usable)} usable views, {min_views} required")
    values, weights = [value for value, _ in usable], [weight for _, weight in usable]
    median = _weighted_median(values, weights)
    spread = _weighted_median([abs(value - median) for value in values], weights)
    return NumericPrediction(attribute, round(median, 3), round(spread, 3), source)
