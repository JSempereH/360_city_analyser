"""Experiment files: a dataset, the reference sources and the runs to compare (see docs/experiments.md)."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..groundtruth import SOURCES

RUN_KINDS = ("prior", "vlm", "saved")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class RunConfig:
    id: str
    kind: str
    # vlm runs
    backend: str = "openai"
    model: str | None = None
    url: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    regional_guidance: bool = True
    min_views: int = 1
    min_share: float = 0.5
    # saved runs: a provider whose answers are already in the dataset (a survey, a Colab import, the curated demo)
    provider: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value not in (None, {}, [])}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: str
    truth: tuple[str, ...]
    runs: tuple[RunConfig, ...]
    buildings: str = "with_truth"
    limit: int | None = None
    per_building: int = 3
    path: Path | None = None
    # BUILDING_ANALYSIS_* settings the dataset was analyzed with: analyses made with others do not count.
    analysis: dict[str, str] = field(default_factory=dict)


def load_config(path: Path) -> ExperimentConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        name, dataset = str(raw["name"]), str(raw["dataset"])
    except KeyError as error:
        raise ValueError(f"{path}: an experiment needs {error.args[0]!r}") from error
    if not _ID.fullmatch(name):
        raise ValueError(f"{path}: the name {name!r} must be letters, digits, dots, dashes or underscores")
    truth = tuple(raw.get("truth", SOURCES))
    unknown = [source for source in truth if source not in SOURCES]
    if unknown:
        raise ValueError(f"{path}: unknown reference sources {unknown} (known: {', '.join(SOURCES)})")
    buildings = raw.get("buildings", "with_truth")
    if buildings not in {"with_truth", "all"}:
        raise ValueError(f"{path}: buildings must be with_truth or all")
    runs = []
    for item in raw.get("runs", []):
        run = RunConfig(**{key: item[key] for key in item})
        if not _ID.fullmatch(run.id):
            raise ValueError(f"{path}: run id {run.id!r} must be letters, digits, dots, dashes or underscores")
        if run.kind not in RUN_KINDS:
            raise ValueError(f"{path}: run {run.id} has kind {run.kind!r} (one of {', '.join(RUN_KINDS)})")
        if run.kind == "vlm" and not run.model:
            raise ValueError(f"{path}: vlm run {run.id} needs a model")
        if run.kind == "saved" and not run.provider:
            raise ValueError(f"{path}: saved run {run.id} needs the provider whose answers it scores")
        runs.append(run)
    ids = [run.id for run in runs]
    if not runs or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: an experiment needs runs with distinct ids")
    analysis = {str(key): str(value) for key, value in (raw.get("analysis") or {}).items()}
    wrong = [key for key in analysis if not key.startswith("BUILDING_ANALYSIS_")]
    if wrong:
        raise ValueError(f"{path}: [analysis] takes BUILDING_ANALYSIS_* settings, not {wrong}")
    limit = raw.get("limit")
    return ExperimentConfig(name, dataset, truth, tuple(runs), buildings, int(limit) if limit else None, int(raw.get("per_building", 3)), path, analysis)
