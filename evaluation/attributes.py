"""Scores for attribute predictions, with the baselines they must beat.

    uv run python -m evaluation.attributes --dataset valencia-centro --provider osm-tag-prior

Labels here come from OSM tags, which are noisy and cover few buildings: the
scores say how well a provider agrees with OSM where OSM has an answer, not how
accurate it is. A visual provider is only worth adopting if it beats the
majority-class baseline and, where tags exist, the tag prior (see the roadmap).
Abstentions lower coverage, not accuracy, and are always reported.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np


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
    majority = Counter(true).most_common(1)[0][1] / len(true)
    result["majority_baseline"] = round(majority, 4)
    return result


def numeric_scores(truth: Sequence[float], predicted: Sequence[float | None]) -> dict[str, Any]:
    answered = [(t, p) for t, p in zip(truth, predicted) if p is not None]
    result: dict[str, Any] = {"n": len(truth), "answered": len(answered), "coverage": round(len(answered) / len(truth), 4) if truth else 0.0}
    if not answered:
        return result
    true, guess = np.array([t for t, _ in answered]), np.array([p for _, p in answered])
    result["mae"] = round(float(np.abs(true - guess).mean()), 3)
    spread = float(((true - true.mean()) ** 2).sum())
    result["r2"] = round(1.0 - float(((true - guess) ** 2).sum()) / spread, 4) if spread > 0 else None
    result["median_baseline_mae"] = round(float(np.abs(true - np.median(true)).mean()), 3)
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
    """``predictions``: building -> attribute -> prediction dict (``BuildingAttributes.to_dict`` form); ``truth``: building -> attribute -> value."""
    from city_analyser.attributes.taxonomy import CATEGORICAL_ATTRIBUTES, NUMERIC_ATTRIBUTES

    report: dict[str, Any] = {}
    for attribute, classes in CATEGORICAL_ATTRIBUTES.items():
        buildings = sorted(b for b in predictions if truth.get(b, {}).get(attribute) is not None)
        guesses = []
        for b in buildings:
            prediction = predictions[b][attribute]
            guesses.append(None if prediction["abstained"] or not prediction["scores"] else max(prediction["scores"], key=prediction["scores"].get))
        if buildings:
            report[attribute] = categorical_scores([truth[b][attribute] for b in buildings], guesses, classes)
            report[attribute]["risk_coverage"] = risk_coverage(
                [truth[b][attribute] for b in buildings], [{} if predictions[b][attribute]["abstained"] else predictions[b][attribute]["scores"] for b in buildings],
            )
    for attribute in NUMERIC_ATTRIBUTES:
        buildings = sorted(b for b in predictions if truth.get(b, {}).get(attribute) is not None)
        if buildings:
            report[attribute] = numeric_scores([float(truth[b][attribute]) for b in buildings], [predictions[b][attribute]["value"] for b in buildings])
    return report


def load_predictions(dataset_dir: Path, provider: str) -> dict[str, dict[str, Any]]:
    predictions = {}
    for path in sorted((dataset_dir / "attributes").rglob(f"attributes-{provider}.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        predictions[payload["building"]["id"]] = payload["predictions"]
    return predictions


def truth_from_osm_tags(dataset_dir: Path, building_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    from city_analyser.attributes.cli import osm_tags
    from city_analyser.attributes.prior import osm_prior
    from city_analyser.local_viewer import dataset_manifest

    manifest = dataset_manifest(dataset_dir.parent, dataset_dir.name)
    truth = {}
    for building_id in building_ids:
        tags = osm_tags(dataset_dir, manifest, building_id)
        if tags is None:
            continue
        prior = osm_prior(building_id, tags).to_dict()["predictions"]
        truth[building_id] = {
            "building_type": max(prior["building_type"]["scores"], key=prior["building_type"]["scores"].get) if not prior["building_type"]["abstained"] else None,
            "surface_material": max(prior["surface_material"]["scores"], key=prior["surface_material"]["scores"].get) if not prior["surface_material"]["abstained"] else None,
            "floors": prior["floors"]["value"], "construction_year": prior["construction_year"]["value"],
        }
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--provider", required=True, help="file suffix of attributes-<provider>.json (e.g. osm-tag-prior, vlm_<model>)")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    dataset_dir = (args.data_root / args.dataset).resolve()
    predictions = load_predictions(dataset_dir, args.provider)
    if not predictions:
        raise SystemExit(f"no attributes-{args.provider}.json files under {dataset_dir / 'attributes'}; run city-analyser-attributes first")
    report = evaluate(predictions, truth_from_osm_tags(dataset_dir, sorted(predictions)))
    (dataset_dir / "attributes" / f"evaluation-{args.provider}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
