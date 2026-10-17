"""Run an experiment's runs into their own folders, resumably, without touching the dataset's attributes.

Each run writes ``data/<dataset>/experiments/<name>/<run>/attributes/...``
(same layout as a dataset, so the card and metrics code read it unchanged) and
``run.json`` with its configuration, versions, code revision, timings and errors.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..analysis_cache import atomic_write_json
from ..attributes.cli import ModelSettings, attributes_path, load_crops, osm_tags, read_attributes, write_attributes
from ..attributes.prior import osm_prior
from ..attributes.regional import GUIDANCE_VERSION
from ..attributes.taxonomy import TAXONOMY_VERSION
from ..attributes.vlm import PROMPT_VERSION, VLMProvider, inputs_fingerprint
from ..building_analysis import ANALYSIS_VERSION
from ..building_inspection.crops import analyzed_building_ids, crops_directory, export_building_crops
from ..groundtruth import load_truth
from .config import ExperimentConfig, RunConfig


def experiment_dir(data_root: Path, config: ExperimentConfig) -> Path:
    return data_root / config.dataset / "experiments" / config.name


def code_revision() -> str | None:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{commit}{'+changes' if dirty else ''}"


def select_buildings(data_root: Path, config: ExperimentConfig) -> list[str]:
    dataset_dir = data_root / config.dataset
    analyzed = analyzed_building_ids(dataset_dir.resolve())
    if config.buildings == "with_truth":
        with_truth = set().union(*(load_truth(dataset_dir, source) for source in config.truth))
        analyzed = [building for building in analyzed if building in with_truth]
    return analyzed[: config.limit] if config.limit else analyzed


def _run_prior(dataset_dir: Path, run_dir: Path, buildings: list[str], record: Callable[..., None]) -> None:
    from ..local_viewer import dataset_manifest

    manifest = dataset_manifest(dataset_dir.parent, dataset_dir.name)
    for building in buildings:
        started = time.perf_counter()
        tags = osm_tags(dataset_dir.resolve(), manifest, building)
        if tags is None:
            record(building, "no_tags", time.perf_counter() - started)
            continue
        write_attributes(run_dir, osm_prior(building, tags))
        record(building, "written", time.perf_counter() - started)


def _run_saved(dataset_dir: Path, run_dir: Path, run: RunConfig, buildings: list[str], record: Callable[..., None]) -> None:
    for building in buildings:
        source = attributes_path(dataset_dir, building, run.provider or "")
        if not source.is_file():
            record(building, "missing", 0.0)
            continue
        # A snapshot, so the report does not change when the dataset's answers do.
        target = attributes_path(run_dir, building, run.provider or "")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        record(building, "copied", 0.0)


def _run_vlm(dataset_dir: Path, run_dir: Path, run: RunConfig, buildings: list[str], record: Callable[..., None], model_root: Path, log: Callable[[str], None]) -> None:
    from ..local_viewer import dataset_manifest

    manifest = dataset_manifest(dataset_dir.parent, dataset_dir.name)
    images = {str(image["id"]): image for image in manifest["images"] if isinstance(image, dict) and "id" in image}
    settings = ModelSettings(run.backend, run.model or "", run.url, options=dict(run.options), cache_dir=model_root)
    provider: VLMProvider | None = None
    for number, building in enumerate(buildings, start=1):
        crops = load_crops(dataset_dir.resolve(), building, images)
        if not crops:
            record(building, "no_crops", 0.0)
            continue
        if not run.regional_guidance:
            crops = [replace(crop, region=None) for crop in crops]
        name = settings.provider_name
        previous = read_attributes(run_dir, building, name)
        if previous and (previous.get("run") or {}).get("inputs_sha256") == inputs_fingerprint(name, crops):
            record(building, "unchanged", 0.0)
            continue
        if provider is None:
            log(f"  loading {settings.model} ({settings.backend})")
            provider = VLMProvider(settings.client(), name, min_views=run.min_views, min_share=run.min_share)
        started = time.perf_counter()
        result = provider.predict(building, crops)
        write_attributes(run_dir, result)
        errors = sum("error" in view for view in result.evidence)
        record(building, "written" if not errors else "written_with_errors", time.perf_counter() - started, views=len(crops), errors=errors)
        log(f"  [{number}/{len(buildings)}] {building}: {len(crops)} views in {time.perf_counter() - started:.1f} s")


@contextmanager
def analysis_settings(settings: dict[str, str]) -> Iterator[None]:
    """The experiment's BUILDING_ANALYSIS_* settings for the length of the run, then the previous ones."""
    previous = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


NO_EVIDENCE = ("no analysis of this dataset matches the current BUILDING_ANALYSIS_* settings, so no building has crops or tags. "
               "Set the ones the dataset was analyzed with under [analysis] in the experiment file "
               "(for example BUILDING_ANALYSIS_MODEL = \"balanced\"), or analyze it again with these.")


def run_experiment(
    data_root: Path, config: ExperimentConfig, *, model_root: Path = Path("models"), only: set[str] | None = None, log: Callable[[str], None] = print,
) -> dict[str, Any]:
    with analysis_settings(config.analysis):
        return _run_experiment(data_root, config, model_root=model_root, only=only, log=log)


def _run_experiment(
    data_root: Path, config: ExperimentConfig, *, model_root: Path, only: set[str] | None, log: Callable[[str], None],
) -> dict[str, Any]:
    dataset_dir = data_root / config.dataset
    if not (dataset_dir / "manifest.json").is_file():
        raise ValueError(f"no dataset at {dataset_dir}")
    buildings = select_buildings(data_root, config)
    if not buildings:
        raise ValueError("no building to run: build reference data first (city-analyser-groundtruth) or set buildings = \"all\"")
    missing_crops = [building for building in buildings if not any(crops_directory(dataset_dir.resolve(), building).glob("crop-*.json"))]
    if missing_crops and any(run.kind == "vlm" for run in config.runs):
        log(f"Exporting crops for {len(missing_crops)} buildings...")
        exported = export_building_crops(data_root, config.dataset, missing_crops, per_building=config.per_building, mask_others=True)
        if len(missing_crops) == len(buildings) and not exported["buildings"]:
            raise ValueError(NO_EVIDENCE)
    summaries = {}
    for run in config.runs:
        if only and run.id not in only:
            continue
        run_dir = experiment_dir(data_root, config) / run.id
        log(f"Run {run.id} ({run.kind}) on {len(buildings)} buildings")
        rows: list[dict[str, Any]] = []

        def record(building: str, status: str, seconds: float, **extra: Any) -> None:
            rows.append({"building": building, "status": status, "seconds": round(seconds, 3), **extra})

        started = datetime.now(UTC).isoformat(timespec="seconds")
        if run.kind == "prior":
            _run_prior(dataset_dir, run_dir, buildings, record)
        elif run.kind == "saved":
            _run_saved(dataset_dir, run_dir, run, buildings, record)
        else:
            _run_vlm(dataset_dir, run_dir, run, buildings, record, model_root, log)
        timed = [row["seconds"] for row in rows if row["seconds"] > 0]
        summary = {
            "experiment": config.name, "dataset": config.dataset, "run": run.to_dict(), "analysis": config.analysis, "started_at": started,
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"), "code_revision": code_revision(),
            "versions": {"prompt": PROMPT_VERSION, "guidance": GUIDANCE_VERSION, "taxonomy": TAXONOMY_VERSION, "analysis": ANALYSIS_VERSION},
            "buildings": len(buildings), "statuses": {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})},
            "seconds_per_building": {"median": round(statistics.median(timed), 2), "mean": round(statistics.mean(timed), 2)} if timed else None,
            "rows": rows,
        }
        atomic_write_json(run_dir / "run.json", summary)
        summaries[run.id] = summary
        log(f"  {summary['statuses']}")
        if rows and all(row["status"] in {"no_tags", "no_crops"} for row in rows) and run.kind != "saved":
            log(f"  Warning: {NO_EVIDENCE}")
    return summaries


def run_predictions(run_dir: Path) -> dict[str, dict[str, Any]]:
    """``{building: predictions}`` written by one run."""
    predictions = {}
    for path in sorted((run_dir / "attributes").rglob("attributes-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        predictions[payload["building"]["id"]] = payload["predictions"]
    return predictions
