"""Human review of a building's structural attributes, and the queue of answers worth reviewing.

A reviewer confirms or corrects any attribute: the GEM material, the lateral
load-resisting system, storeys, use, facade material and construction year. The review is one more provider,
``human-review``, which the card prefers over OSM tags and models. Every
decision is appended to ``reviews.jsonl`` next to it, so a withdrawn or
replaced review stays on record. Reviewed buildings are the labeled data a
model must be scored against before its answers are trusted.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..building_inspection.reporting import BUILDING_ID, _building_parts
from .card import building_card
from .cli import attributes_path, read_attributes, write_attributes
from .models import BuildingAttributes, CategoricalPrediction, NumericPrediction, abstain_categorical, abstain_numeric
from .taxonomy import BUILDING_TYPES, GEM_LLRS, GEM_MATERIALS, SURFACE_MATERIALS
from .vlm import MAX_FLOORS

REVIEW_PROVIDER = "human-review"
SOURCE = "human_review"
REVIEWABLE: dict[str, tuple[str, ...] | None] = {
    "gem_material": tuple(GEM_MATERIALS), "gem_llrs": tuple(GEM_LLRS), "floors": None,
    "building_type": BUILDING_TYPES, "surface_material": SURFACE_MATERIALS, "construction_year": None,
}
# Accepted range of each numeric attribute, whole numbers only.
NUMERIC_RANGES = {"floors": (1, MAX_FLOORS), "construction_year": (1000, 2100)}
MAX_NOTE = 500
MAX_REVIEWER = 80
# Model answers below this vote share are put in front of a reviewer. A view also gives its second
# choice half a vote, so a single confident view already scores 2/3: only lower shares mean disagreement.
WEAK_SHARE = 0.6


class ReviewError(ValueError):
    """The request cannot be applied (bad value, unknown building); the message is for the reviewer."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _prediction(attribute: str, value: Any) -> CategoricalPrediction | NumericPrediction:
    classes = REVIEWABLE[attribute]
    if classes is None:
        if value is None:
            return abstain_numeric(attribute, SOURCE, "the reviewer could not tell from the imagery")
        low, high = NUMERIC_RANGES[attribute]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high or value != int(value):
            raise ReviewError(f"{attribute} must be a whole number from {low} to {high}")
        return NumericPrediction(attribute, float(value), 0.0, SOURCE)
    if value is None:
        return abstain_categorical(attribute, SOURCE, "the reviewer could not tell from the imagery")
    if value not in classes:
        raise ReviewError(f"{attribute} must be one of {', '.join(classes)}, or null")
    return CategoricalPrediction(attribute, {name: 1.0 if name == value else 0.0 for name in classes}, "review", SOURCE)


def _log(dataset_dir: Path, building_id: str, entry: dict[str, Any]) -> None:
    path = dataset_dir / "attributes" / Path(*_building_parts(building_id)) / "reviews.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def review_history(dataset_dir: Path, building_id: str) -> list[dict[str, Any]]:
    path = dataset_dir / "attributes" / Path(*_building_parts(building_id)) / "reviews.jsonl"
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def save_review(dataset_dir: Path, dataset_id: str, building_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Apply ``{"action": "review", "reviewer", "note", "values": {attribute: value or null}}`` or ``{"action": "withdraw", "reviewer"}``.

    Only the attributes named in ``values`` are reviewed; ``null`` records that
    the reviewer could not tell. Returns the updated card.
    """
    if not BUILDING_ID.fullmatch(building_id):
        raise ReviewError("invalid building id")
    reviewer = str(payload.get("reviewer") or "").strip()[:MAX_REVIEWER]
    if not reviewer:
        raise ReviewError("a reviewer name is required")
    note = str(payload.get("note") or "").strip()[:MAX_NOTE]
    action = payload.get("action", "review")
    if action == "withdraw":
        path = attributes_path(dataset_dir, building_id, REVIEW_PROVIDER)
        if not path.is_file():
            raise ReviewError("this building has no review to withdraw")
        path.unlink()
        _log(dataset_dir, building_id, {"action": "withdraw", "reviewer": reviewer, "note": note, "at": _now()})
    elif action == "review":
        values = payload.get("values")
        if not isinstance(values, dict) or not values or not set(values) <= set(REVIEWABLE):
            raise ReviewError(f"values must name at least one of {', '.join(REVIEWABLE)}")
        card = building_card(dataset_dir, dataset_id, building_id)
        if card is None:
            raise ReviewError("this building has not been surveyed yet")
        # What the reviewer looked at, so a later model answer can be compared with the decision.
        models = [name for name in card["providers"] if name.startswith("vlm:")]
        model = read_attributes(dataset_dir, building_id, models[0]) if models else None
        based_on = {"provider": models[0], "inputs_sha256": (model or {}).get("run", {}).get("inputs_sha256")} if models else None
        predictions = {attribute: _prediction(attribute, value) for attribute, value in values.items()}
        previous = read_attributes(dataset_dir, building_id, REVIEW_PROVIDER)
        # A partial review keeps the attributes reviewed earlier.
        for attribute, prediction in (previous or {}).get("predictions", {}).items():
            if attribute not in predictions and attribute in REVIEWABLE:
                predictions[attribute] = _restore(prediction)
        # An evaluation label is a review chosen as ground truth, kept apart from ordinary corrections.
        run = {"reviewer": reviewer, "note": note, "reviewed_at": _now(), "based_on": based_on, "evaluation": bool(payload.get("evaluation"))}
        write_attributes(dataset_dir, BuildingAttributes(building_id, REVIEW_PROVIDER, predictions, run=run))
        _log(dataset_dir, building_id, {"action": "review", "values": values, **run, "at": run["reviewed_at"]})
    else:
        raise ReviewError("action must be review or withdraw")
    card = building_card(dataset_dir, dataset_id, building_id)
    if card is None:
        raise ReviewError("this building has no attributes left")
    return card


def _restore(prediction: dict[str, Any]) -> CategoricalPrediction | NumericPrediction:
    if prediction.get("type") == "numeric":
        return NumericPrediction(prediction["attribute"], prediction.get("value"), prediction.get("spread"), SOURCE, bool(prediction.get("abstained")), prediction.get("reason"))
    return CategoricalPrediction(prediction["attribute"], prediction.get("scores") or {}, "review", SOURCE, bool(prediction.get("abstained")), prediction.get("reason"))


def _review_reason(card: dict[str, Any]) -> tuple[str | None, float]:
    """Why a model answer deserves a reviewer, and how urgently (lower first)."""
    attributes = card["attributes"]
    reasons, urgency = [], 1.0
    for attribute, label in (("gem_material", "material"), ("gem_llrs", "lateral system")):
        entry = attributes.get(attribute) or {}
        if not str(entry.get("provider", "")).startswith("vlm:"):
            continue
        share = entry.get("share")
        if entry.get("abstained") and entry.get("leaning"):
            reasons.append(f"split vote on the {label}")
            urgency = min(urgency, float(share or 0.0))
        elif not entry.get("abstained") and isinstance(share, (int, float)) and share < WEAK_SHARE:
            reasons.append(f"weak agreement on the {label} ({round(share * 100)}%)")
            urgency = min(urgency, float(share))
    floors = attributes.get("floors") or {}
    model_floors = next((other for other in floors.get("others", []) if str(other.get("provider", "")).startswith("vlm:") and not other.get("abstained")), None)
    if floors.get("source") == "osm_tag" and model_floors and abs(float(model_floors["value"]) - float(floors["value"])) >= 1:
        reasons.append(f"model says {model_floors['value']:g} floors, OSM {floors['value']:g}")
        urgency = min(urgency, 0.5)
    if card.get("view_errors"):
        reasons.append(f"the model failed on {card['view_errors']} view{'s' if card['view_errors'] != 1 else ''}")
        urgency = min(urgency, 0.6)
    return ("; ".join(reasons) or None), urgency


def attributes_signature(dataset_dir: Path) -> str:
    """Changes whenever an attribute file is written or removed; cheap enough to check on every request."""
    entries = []
    for directory, _, files in os.walk(dataset_dir / "attributes"):
        for name in files:
            if name.endswith(".json"):
                stat = os.stat(os.path.join(directory, name))
                entries.append(f"{directory}/{name}:{stat.st_mtime_ns}:{stat.st_size}")
    return str(hash("\n".join(sorted(entries))))


def attribute_summary(dataset_dir: Path, dataset_id: str) -> dict[str, Any]:
    """One compact entry per building with attributes: what the map colors and the review queue need."""
    buildings: dict[str, Any] = {}
    root = dataset_dir / "attributes"
    for directory in sorted({path.parent for path in root.rglob("attributes-*.json")} if root.is_dir() else ()):
        building_id = directory.relative_to(root).as_posix()
        if not BUILDING_ID.fullmatch(building_id):
            continue
        card = building_card(dataset_dir, dataset_id, building_id)
        if card is None:
            continue
        attributes = card["attributes"]

        def value(name: str) -> Any:
            entry = attributes.get(name) or {}
            return None if entry.get("abstained") else entry.get("value")

        reviewed = card.get("review") is not None
        has_model = any(name.startswith("vlm:") for name in card["providers"])
        reason, urgency = _review_reason(card) if has_model and not reviewed else (None, 1.0)
        buildings[building_id] = {
            "gem_material": value("gem_material"), "gem_llrs": value("gem_llrs"), "floors": value("floors"),
            "gem": card["gem"]["string"], "reviewed": reviewed, "model": has_model,
            "needs_review": reason, "urgency": round(urgency, 4), "image": card.get("best_image"),
            # Ranking score of the best view: how much of the building that view shows.
            "score": max((view["score"] for view in card.get("views", []) if isinstance(view.get("score"), (int, float))), default=None),
            # Mask quality: how many views were set aside (and why) next to the ones the model read.
            "set_aside": len(card.get("set_aside", [])),
            "set_aside_reasons": sorted({reason for view in card.get("set_aside", []) for reason in view["reasons"]}),
            "set_aside_image": (card.get("set_aside") or [{}])[0].get("image_id"),
            "clean_views": len(card.get("views", [])) if not card.get("no_clean_view") else 0,
            "no_clean_view": bool(card.get("no_clean_view")),
        }
    counts = {
        "buildings": len(buildings), "reviewed": sum(entry["reviewed"] for entry in buildings.values()),
        "needs_review": sum(entry["needs_review"] is not None for entry in buildings.values()),
        "set_aside": sum(entry["set_aside"] > 0 for entry in buildings.values()),
        "no_clean_view": sum(entry["no_clean_view"] for entry in buildings.values()),
    }
    return {"dataset": dataset_id, "buildings": buildings, "counts": counts}
