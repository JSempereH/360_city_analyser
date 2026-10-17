"""Score every run of an experiment against every reference source, side by side."""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..analysis_cache import atomic_write_bytes
from ..groundtruth import load_truth
from .config import ExperimentConfig
from .metrics import evaluate
from .runner import experiment_dir, run_predictions, select_buildings

SOURCE_NOTES = {
    "catastro": "Spanish cadastre: storeys, year (may be a refurbishment) and use; nothing about the structure.",
    "evaluation_set": "Engineer labels from the same imagery: the only reference for the GEM structure.",
    "osm_tags": "OSM tags: sparse and noisy; the prior is built from them, so the prior's score here is not a measurement.",
}

# Reference sources whose records carry a use, split into groups scored apart: a cathedral's
# cadastral year and storeys do not measure the same thing as an apartment block's.
GROUPS = {"catastro": {"residential": lambda record: str(record.get("use") or "").startswith("1_"),
                       "non-residential": lambda record: bool(record.get("use")) and not str(record["use"]).startswith("1_")}}


def answered(predictions: dict[str, dict[str, Any]], attribute: str) -> set[str]:
    """Buildings the run gave an answer for, abstentions excluded."""
    out = set()
    for building, attributes in predictions.items():
        prediction = attributes.get(attribute) or {}
        if not prediction.get("abstained") and (prediction.get("value") is not None or prediction.get("scores")):
            out.add(building)
    return out


def common_buildings(truth: dict[str, dict[str, Any]], answers: list[set[str]], attribute: str) -> set[str]:
    """Buildings with a reference value that every run answering at least half of them answered.

    Runs that rarely answer (the prior, from sparse tags) are left out, or the subset would be
    empty; they are still scored on it, with their coverage.
    """
    buildings = {b for b, record in truth.items() if record.get(attribute) is not None}
    deciding = [run_answers for run_answers in answers if 2 * len(run_answers & buildings) >= len(buildings)]
    for run_answers in deciding:
        buildings &= run_answers
    return buildings


def build_report(data_root: Path, config: ExperimentConfig) -> dict[str, Any]:
    dataset_dir = data_root / config.dataset
    selected = set(select_buildings(data_root, config))
    truths = {source: {b: v for b, v in load_truth(dataset_dir, source).items() if b in selected} for source in config.truth}
    runs, all_predictions = {}, {}
    for run in config.runs:
        run_dir = experiment_dir(data_root, config) / run.id
        summary_path = run_dir / "run.json"
        if not summary_path.is_file():
            runs[run.id] = {"missing": True}
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        predictions = run_predictions(run_dir)
        all_predictions[run.id] = predictions
        runs[run.id] = {
            "run": summary["run"], "versions": summary["versions"], "code_revision": summary["code_revision"],
            "statuses": summary["statuses"], "seconds_per_building": summary["seconds_per_building"], "described": len(predictions),
            "scores": {source: evaluate(predictions, truth) for source, truth in truths.items()},
        }
    common: dict[str, dict[str, int]] = {}
    for source, truth in truths.items():
        common[source] = {}
        attributes = {attribute for run in runs.values() if not run.get("missing") for attribute in run["scores"][source]}
        for attribute in sorted(attributes):
            subset = common_buildings(truth, [answered(predictions, attribute) for predictions in all_predictions.values()], attribute)
            common[source][attribute] = len(subset)
            for run_id, predictions in all_predictions.items():
                scored = evaluate(predictions, {b: {attribute: truth[b][attribute]} for b in subset}).get(attribute)
                runs[run_id].setdefault("common", {}).setdefault(source, {})[attribute] = scored
    groups: dict[str, dict[str, int]] = {}
    for source, truth in truths.items():
        for group, member in GROUPS.get(source, {}).items():
            subset = {b: record for b, record in truth.items() if member(record)}
            groups.setdefault(source, {})[group] = len(subset)
            for run_id, predictions in all_predictions.items():
                runs[run_id].setdefault("groups", {}).setdefault(source, {})[group] = evaluate(predictions, subset)
    return {
        "experiment": config.name, "dataset": config.dataset, "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "buildings": len(selected), "truth": {source: {"buildings": len(truth), "note": SOURCE_NOTES[source],
                                       "common": common[source], "groups": groups.get(source, {})} for source, truth in truths.items()},
        "runs": runs,
    }


def _cells(scores: dict[str, Any] | None) -> list[str]:
    """n, coverage, main score, second score, baseline, bias, for one run on one attribute."""
    if not scores:
        return ["-"] * 6

    def percent(value: float | None) -> str:
        return "-" if value is None else f"{value * 100:.0f}%"

    if "mae" in scores:
        return [str(scores["n"]), percent(scores["coverage"]), f"MAE {scores['mae']:g}", f"±{scores['tolerance']:g}: {percent(scores.get('within_tolerance'))}",
                f"median MAE {scores['median_baseline_mae']:g}", f"{scores['bias']:+g}"]
    if "accuracy" in scores:
        return [str(scores["n"]), percent(scores["coverage"]), f"acc {percent(scores['accuracy'])}", f"macro F1 {scores['macro_f1']:.2f}" if scores.get("macro_f1") is not None else "-",
                f"majority {percent(scores['majority_baseline'])}", "-"]
    return [str(scores["n"]), percent(scores["coverage"]), "no answer", "-", "-", "-"]


HEADERS = ["Run", "n", "Coverage", "Score", "Second", "Baseline", "Bias"]


def tables(report: dict[str, Any]) -> list[tuple[str, str, list[tuple[str, list[list[str]]]]]]:
    """(source, attribute, [(caption, rows)]) for every attribute any run was scored on.

    The first table scores every building with a reference value; then the same buildings for
    every run, so abstentions do not change what is compared; then each group of the source.
    """
    out = []
    present = {run_id: run for run_id, run in report["runs"].items() if not run.get("missing")}
    for source, truth in report["truth"].items():
        attributes = sorted({attribute for run in present.values() for attribute in run["scores"][source]})
        for attribute in attributes:
            rows = [[run_id, *_cells(None if run.get("missing") else run["scores"][source].get(attribute))] for run_id, run in report["runs"].items()]
            captioned = [("All buildings with a reference value", rows)]
            same = truth.get("common", {}).get(attribute)
            if same:
                answering = [run_id for run_id, run in present.items() if (run["scores"][source].get(attribute) or {}).get("answered")]
                if len(answering) > 1:
                    captioned.append((f"Same {same} buildings, answered by every run that answers at least half", [[run_id, *_cells(present[run_id]["common"][source][attribute])] for run_id in answering]))
            for group, size in truth.get("groups", {}).items():
                if size and attribute != "building_type":  # the type is derived from the same use
                    captioned.append((f"{group.capitalize()} ({size})", [[run_id, *_cells(run["groups"][source][group].get(attribute))] for run_id, run in present.items()]))
            out.append((source, attribute, captioned))
    return out


def to_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Experiment {report['experiment']}", "", f"Dataset `{report['dataset']}`, {report['buildings']} buildings, {report['created_at']}.", ""]
    lines += ["| Run | Kind | Model | Described | Statuses | s / building | Code |", "|---|---|---|---|---|---|---|"]
    for run_id, run in report["runs"].items():
        if run.get("missing"):
            lines.append(f"| {run_id} | not run | | | | | |")
            continue
        config = run["run"]
        timing = run["seconds_per_building"]["median"] if run["seconds_per_building"] else "-"
        statuses = ", ".join(f"{key} {value}" for key, value in run["statuses"].items())
        guidance = "" if config.get("regional_guidance", True) else " (no regional guidance)"
        lines.append(f"| {run_id} | {config['kind']} | {(config.get('model') or config.get('provider') or '-')}{guidance} | {run['described']} | {statuses} | {timing} | {run['code_revision'] or '-'} |")
    current = None
    for source, attribute, captioned in tables(report):
        if source != current:
            current = source
            lines += ["", f"## Against {source} ({report['truth'][source]['buildings']} buildings)", "", report["truth"][source]["note"]]
        lines += ["", f"### {attribute}"]
        for caption, rows in captioned:
            lines += ["", f"{caption}:", "", "| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
            lines += ["| " + " | ".join(row) + " |" for row in rows]
    lines += ["", "Coverage is the share of buildings with a reference value that the run answered; abstentions lower coverage, not the score. "
              "Bias is the mean signed error (negative: the run undercounts). The same-buildings table compares runs on the buildings all of them answered, so abstentions do not change what is compared. A run is worth adopting only if it beats the baseline and the prior."]
    return "\n".join(lines) + "\n"


def to_html(report: dict[str, Any]) -> str:
    def table(headers: list[str], rows: list[list[str]]) -> str:
        head = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    sections, current = [], None
    for source, attribute, captioned in tables(report):
        if source != current:
            current = source
            sections.append(f"<h2>Against {html.escape(source)} ({report['truth'][source]['buildings']} buildings)</h2><p class=note>{html.escape(report['truth'][source]['note'])}</p>")
        sections.append(f"<h3>{html.escape(attribute)}</h3>" + "".join(f"<p class=caption>{html.escape(caption)}</p>{table(HEADERS, rows)}" for caption, rows in captioned))
    style = (":root{color-scheme:light dark;--bg:#fbfbf9;--fg:#1c2420;--muted:#56645a;--line:#dde2d8}"
             "@media (prefers-color-scheme:dark){:root{--bg:#101413;--fg:#ecf0e8;--muted:#a3b2a8;--line:#2f3934}}"
             "body{margin:0 auto;max-width:960px;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}"
             "table{border-collapse:collapse;width:100%;margin:8px 0 18px;font-variant-numeric:tabular-nums}"
             "th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left}th{color:var(--muted);font-weight:600}"
             ".note{color:var(--muted)}.caption{margin:4px 0 0;color:var(--muted);font-size:13px}")
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Experiment {html.escape(report['experiment'])}</title><style>{style}</style></head><body>"
            f"<h1>Experiment {html.escape(report['experiment'])}</h1><p class=note>Dataset {html.escape(report['dataset'])}, {report['buildings']} buildings, {report['created_at']}</p>"
            + "".join(sections) + "</body></html>\n")


def write_report(data_root: Path, config: ExperimentConfig) -> Path:
    report = build_report(data_root, config)
    folder = experiment_dir(data_root, config)
    atomic_write_bytes(folder / "report.json", (json.dumps(report, indent=2) + "\n").encode("utf-8"))
    atomic_write_bytes(folder / "report.md", to_markdown(report).encode("utf-8"))
    atomic_write_bytes(folder / "report.html", to_html(report).encode("utf-8"))
    return folder
