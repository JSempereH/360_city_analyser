"""Scores for attribute predictions, with the baselines they must beat.

Abstentions lower coverage, not accuracy, and coverage is always reported: a
provider that answers only easy buildings looks accurate and covers little.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np

# Numeric answers within this distance of the reference count as right.
TOLERANCE = {"floors": 1.0, "construction_year": 10.0}


def categorical_scores(truth: Sequence[str], predicted: Sequence[str | None], classes: Sequence[str]) -> dict[str, Any]:
    """``predicted`` holds ``None`` for abstentions; scores cover the answered subset."""
    answered = [(t, p) for t, p in zip(truth, predicted) if p is not None]
    result: dict[str, Any] = {"n": len(truth), "answered": len(answered), "coverage": round(len(answered) / len(truth), 4) if truth else 0.0}
    if not answered:
        return result
    true, guess = [t for t, _ in answered], [p for _, p in answered]
    result["accuracy"] = round(float(np.mean([t == p for t, p in answered])), 4)
    recalls, f1s = [], []
    for name in classes:
        tp = sum(t == name and p == name for t, p in answered)
        support, predicted_count = true.count(name), guess.count(name)
        if support:
            recalls.append(tp / support)
        if support or predicted_count:
            precision = tp / predicted_count if predicted_count else 0.0
            recall = tp / support if support else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    result["balanced_accuracy"] = round(float(np.mean(recalls)), 4) if recalls else None
    result["macro_f1"] = round(float(np.mean(f1s)), 4) if f1s else None
    result["majority_baseline"] = round(Counter(true).most_common(1)[0][1] / len(true), 4)
    result["confusion"] = {f"{t} -> {p}": count for (t, p), count in sorted(Counter(answered).items())}
    return result


def numeric_scores(truth: Sequence[float], predicted: Sequence[float | None], tolerance: float | None = None) -> dict[str, Any]:
    answered = [(t, p) for t, p in zip(truth, predicted) if p is not None]
    result: dict[str, Any] = {"n": len(truth), "answered": len(answered), "coverage": round(len(answered) / len(truth), 4) if truth else 0.0}
    if not answered:
        return result
    true, guess = np.array([t for t, _ in answered]), np.array([p for _, p in answered])
    result["mae"] = round(float(np.abs(true - guess).mean()), 3)
    result["bias"] = round(float((guess - true).mean()), 3)  # negative: the provider undercounts
    spread = float(((true - true.mean()) ** 2).sum())
    result["r2"] = round(1.0 - float(((true - guess) ** 2).sum()) / spread, 4) if spread > 0 else None
    result["median_baseline_mae"] = round(float(np.abs(true - np.median(true)).mean()), 3)
    if tolerance is not None:
        result["within_tolerance"] = round(float(np.mean(np.abs(true - guess) <= tolerance)), 4)
        result["tolerance"] = tolerance
    return result


def risk_coverage(truth: Sequence[str], predicted: Sequence[dict[str, float]]) -> list[dict[str, float]]:
    """Accuracy of the most confident answers at several coverages (confidence = top score)."""
    rows = sorted(((max(scores.values()), scores, t) for t, scores in zip(truth, predicted) if scores), key=lambda row: -row[0])
    curve = []
    for share in (0.25, 0.5, 0.75, 1.0):
        kept = rows[: max(1, round(len(rows) * share))] if rows else []
        if kept:
            hits = [max(scores, key=scores.get) == t for _, scores, t in kept]  # type: ignore[arg-type]
            curve.append({"coverage": round(len(kept) / len(truth), 4), "accuracy": round(float(np.mean(hits)), 4)})
    return curve


def evaluate(predictions: dict[str, dict[str, Any]], truth: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """``predictions``: building -> attribute -> prediction dict (``BuildingAttributes.to_dict`` form); ``truth``: building -> attribute -> value.

    Every building with a reference value counts; a building the provider did not describe counts as an abstention.
    """
    from ..attributes.taxonomy import CATEGORICAL_ATTRIBUTES, NUMERIC_ATTRIBUTES

    report: dict[str, Any] = {}
    for attribute, classes in CATEGORICAL_ATTRIBUTES.items():
        buildings = sorted(b for b in truth if truth[b].get(attribute) is not None)
        if not buildings:
            continue
        guesses, distributions = [], []
        for b in buildings:
            prediction = (predictions.get(b) or {}).get(attribute) or {"abstained": True, "scores": {}}
            scores = prediction.get("scores") or {}
            guesses.append(None if prediction.get("abstained") or not scores else max(scores, key=scores.get))
            distributions.append({} if prediction.get("abstained") else scores)
        report[attribute] = categorical_scores([truth[b][attribute] for b in buildings], guesses, classes)
        report[attribute]["risk_coverage"] = risk_coverage([truth[b][attribute] for b in buildings], distributions)
    for attribute in NUMERIC_ATTRIBUTES:
        buildings = sorted(b for b in truth if truth[b].get(attribute) is not None)
        if not buildings:
            continue
        values = []
        for b in buildings:
            prediction = (predictions.get(b) or {}).get(attribute) or {}
            values.append(None if prediction.get("abstained") else prediction.get("value"))
        report[attribute] = numeric_scores([float(truth[b][attribute]) for b in buildings], values, TOLERANCE.get(attribute))
    return report
