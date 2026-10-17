"""Attribute predictions with explicit abstention and evidence references."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .taxonomy import TAXONOMY_VERSION

# What the numbers of a categorical prediction mean. Only "probability" may be
# read as one, and only after calibration; "vote_share" is a normalized count
# of per-view answers; "tag" is a human-entered OSM value and "review" a
# reviewer's decision in this project (all mass on one class).
SCORE_KINDS = ("probability", "vote_share", "tag", "review")


@dataclass(frozen=True, slots=True)
class CategoricalPrediction:
    attribute: str
    scores: dict[str, float]
    kind: str
    source: str
    abstained: bool = False
    reason: str | None = None

    def top(self, count: int = 1) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda item: (-item[1], item[0]))[:count]

    def to_dict(self) -> dict[str, Any]:
        return {"attribute": self.attribute, "type": "categorical", "scores": self.scores, "score_kind": self.kind,
                "source": self.source, "abstained": self.abstained, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class NumericPrediction:
    attribute: str
    value: float | None
    spread: float | None
    source: str
    abstained: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"attribute": self.attribute, "type": "numeric", "value": self.value, "spread": self.spread,
                "source": self.source, "abstained": self.abstained, "reason": self.reason}


Prediction = CategoricalPrediction | NumericPrediction


def abstain_categorical(attribute: str, source: str, reason: str) -> CategoricalPrediction:
    return CategoricalPrediction(attribute, {}, "vote_share", source, True, reason)


def abstain_numeric(attribute: str, source: str, reason: str) -> NumericPrediction:
    return NumericPrediction(attribute, None, None, source, True, reason)


@dataclass(frozen=True, slots=True)
class BuildingAttributes:
    building_id: str
    provider: str
    predictions: dict[str, Prediction]
    # Crops (image IDs and hashes) the predictions rest on; empty for tag priors.
    evidence: list[dict[str, Any]] = field(default_factory=list)
    taxonomy_version: str = TAXONOMY_VERSION
    # How the result was made (prompt version, fingerprint of its inputs), so a rerun can skip it.
    run: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "taxonomy_version": self.taxonomy_version,
            "building": {"id": self.building_id, "identity_source": "openstreetmap"},
            "provider": self.provider,
            "predictions": {name: prediction.to_dict() for name, prediction in sorted(self.predictions.items())},
            "evidence": self.evidence,
            "run": self.run,
        }
