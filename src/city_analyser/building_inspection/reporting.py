"""Application service and persistent HTML/JSON reports."""

from __future__ import annotations

import html
import json
import os
import re
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlparse

from .. import building_analysis
from ..analysis_cache import atomic_write_bytes, file_sha256

from .assessment import assess_footprint
from .evidence import collect_building_evidence
from .models import RankedView
from .ranking import RANKING_FORMULA_VERSION, rank_views
from .screenshots import MAX_REPORT_SCREENSHOTS, render_screenshots


REPORT_SCHEMA_VERSION = 1
BUILDING_ID = re.compile(r"^(node|way|relation)/([1-9][0-9]*)$")


def _building_parts(building_id: str) -> tuple[str, str]:
    match = BUILDING_ID.fullmatch(building_id)
    if not match:
        raise ValueError("building id must be an OpenStreetMap footprint identifier")
    return match.group(1), match.group(2)


def report_paths(dataset_dir: Path, building_id: str) -> tuple[Path, Path]:
    osm_type, osm_number = _building_parts(building_id)
    directory = dataset_dir / "reports" / "buildings" / osm_type / osm_number
    return directory / f"report-v{REPORT_SCHEMA_VERSION}.json", directory / f"report-v{REPORT_SCHEMA_VERSION}.html"


def _safe_source_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else None


def _report_links(dataset_id: str, building_id: str) -> dict[str, str]:
    osm_type, osm_number = _building_parts(building_id)
    base = f"/api/datasets/{quote(dataset_id, safe='')}/building-reports/{osm_type}/{osm_number}"
    return {"json": base, "html": f"{base}/html"}


def _screenshot_url(dataset_id: str, building_id: str, filename: str) -> str:
    osm_type, osm_number = _building_parts(building_id)
    return f"/data/{quote(dataset_id, safe='')}/reports/buildings/{osm_type}/{osm_number}/{quote(filename, safe='')}"


def _report_directory(dataset_dir: Path, building_id: str) -> Path:
    return report_paths(dataset_dir, building_id)[0].parent


def _with_screenshots(dataset_dir: Path, dataset_id: str, building_id: str, views: tuple[RankedView, ...]) -> tuple[RankedView, ...]:
    report_dir = _report_directory(dataset_dir, building_id)
    report_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for view in views:
        if view.rank <= MAX_REPORT_SCREENSHOTS:
            clean_descriptor, clean_name = tempfile.mkstemp(prefix=f".view-{view.rank:02d}-clean.", suffix=".jpg", dir=report_dir)
            annotated_descriptor, annotated_name = tempfile.mkstemp(prefix=f".view-{view.rank:02d}-overlay.", suffix=".jpg", dir=report_dir)
            os.close(clean_descriptor)
            os.close(annotated_descriptor)
            clean_temporary = Path(clean_name)
            annotated_temporary = Path(annotated_name)
            try:
                render_screenshots(dataset_dir, view, clean_temporary, annotated_temporary)
                filenames = []
                for label, temporary in (("clean", clean_temporary), ("overlay", annotated_temporary)):
                    digest = file_sha256(temporary)
                    filename = f"view-{view.rank:02d}-{label}-{digest[:12]}.jpg"
                    destination = report_dir / filename
                    if destination.exists():
                        temporary.unlink()
                    else:
                        os.replace(temporary, destination)
                    filenames.append(filename)
            except Exception:
                clean_temporary.unlink(missing_ok=True)
                annotated_temporary.unlink(missing_ok=True)
                raise
            rendered.append(replace(
                view,
                clean_screenshot_url=_screenshot_url(dataset_id, building_id, filenames[0]),
                screenshot_url=_screenshot_url(dataset_id, building_id, filenames[1]),
            ))
        else:
            rendered.append(view)
    return tuple(rendered)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def render_report_html(report: dict[str, Any]) -> str:
    assessment = report["assessment"]
    checks = "".join(
        f'<li class="{_escape(check["status"])}">{_escape(check["summary"])}</li>'
        for check in assessment["checks"]
    )
    cards = []
    for view in report["ranked_views"]:
        if not view.get("screenshot_url"):
            continue
        clean_url = view.get("clean_screenshot_url") or view["screenshot_url"]
        screenshot = (
            f'<div class="evidence-image"><img class="clean" src="{_escape(clean_url)}" alt="Building view from panorama {_escape(view["image_id"])}">'
            f'<img class="annotated" src="{_escape(view["screenshot_url"])}" alt="Building evidence overlay from panorama {_escape(view["image_id"])}"></div>'
        )
        source_url = _safe_source_url(view.get("source_page"))
        source = f'<a href="{_escape(source_url)}" rel="noreferrer">Mapillary source</a>' if source_url else ""
        coverage = float(view["evidence"]["projection_coverage"])
        cards.append(
            f'<article class="view-card"><header><span class="rank">#{view["rank"]}</span><div><h3>Inspection view</h3><small>Panorama {_escape(view["image_id"])}</small></div></header>'
            f'{screenshot}<p class="metric"><strong>{coverage:.0%}</strong> of the projected facade has direct visual evidence.</p>'
            f'<div class="actions"><a class="button" href="{_escape(view["viewer"]["url"])}">Review in viewer</a>{source}</div></article>'
        )
    limitations = "".join(f"<li>{_escape(item)}</li>" for item in assessment["limitations"])
    issues = "".join(f'<li>{_escape(item["image_id"])}: {_escape(item["detail"])}</li>' for item in report["inputs"]["issues"])
    issues_section = f"<section><h2>Input issues</h2><ul>{issues}</ul></section>" if issues else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
  <title>Building evidence report: {_escape(report['building']['id'])}</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; background:#101512; color:#edf0e9; }}
    body {{ max-width:1180px; margin:0 auto; padding:40px 24px 72px; }}
    h1 {{ margin:.2rem 0; font-size:clamp(2rem,5vw,4.6rem); letter-spacing:-.05em; }}
    h2 {{ margin-top:2.5rem; }} .eyebrow {{ color:#e9a65d; letter-spacing:.14em; text-transform:uppercase; }}
    .summary {{ border-left:4px solid #e9783d; padding:16px 20px; background:#18201b; }}
    .warning {{ padding:18px; border:1px solid #996035; background:#2a2118; color:#ffd7a8; }}
    .checks {{ display:grid; gap:10px; padding:0; list-style:none; }} .checks li {{ padding:14px; background:#18201b; border-left:3px solid #6f8576; }}
    .checks .caution {{ border-color:#e9783d; }}
    .overlay-toggle {{ position:absolute; width:1px; height:1px; opacity:0; pointer-events:none; }}
    .evidence-heading {{ display:flex; align-items:center; justify-content:space-between; gap:20px; margin-top:2.5rem; }}
    .evidence-heading h2 {{ margin:0; }} .evidence-heading label {{ padding:9px 12px; border:1px solid #6f8576; cursor:pointer; white-space:nowrap; }}
    .evidence-heading label::before {{ content:""; display:inline-block; width:.75rem; height:.75rem; margin-right:8px; border:1px solid #9eaca2; vertical-align:-.05rem; }}
    #evidence-overlay:checked ~ main .evidence-heading label::before {{ background:#e9783d; box-shadow:inset 0 0 0 2px #18201b; }}
    #evidence-overlay:not(:checked) ~ main .annotated {{ display:none; }}
    .views {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr)); gap:18px; }}
    .view-card {{ background:#18201b; border:1px solid #344139; padding:16px; }} .view-card header {{ display:flex; gap:12px; align-items:center; }}
    .view-card h3 {{ margin:0; }} .view-card small {{ color:#9eaca2; }} .rank {{ color:#e9a65d; font:700 1.4rem ui-monospace,monospace; }}
    .evidence-image {{ position:relative; aspect-ratio:1; margin-top:14px; background:#0b0f0d; }} .evidence-image img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }}
    .metric {{ min-height:2.8em; }}
    .actions {{ display:flex; gap:16px; align-items:center; flex-wrap:wrap; }} a {{ color:#f2bd7f; }} .button {{ padding:10px 14px; background:#e9783d; color:#111; text-decoration:none; font-weight:700; }}
    .technical {{ margin-top:2.5rem; padding:16px 18px; border:1px solid #344139; }} .technical summary {{ cursor:pointer; font-weight:700; }}
    .provenance {{ overflow-wrap:anywhere; color:#9eaca2; }}
    @media (max-width:640px) {{ .evidence-heading {{ align-items:flex-start; flex-direction:column; }} body {{ padding-inline:14px; }} }}
  </style>
</head>
<body>
  <input id="evidence-overlay" class="overlay-toggle" type="checkbox" checked>
  <main>
    <p class="eyebrow">Building visual inspection</p>
    <h1>{_escape(report['building']['id'])}</h1>
    <p>Dataset <strong>{_escape(report['dataset']['id'])}</strong> · {_escape(report['dataset']['provider'])} · generated {_escape(report['generated_at'])}</p>
    <section class="summary"><h2>{_escape(assessment['conclusion']['label'])}</h2><p>Scope: visible facade and footprint association.</p></section>
    <p class="warning"><strong>GEM classification: not run.</strong> Structural attributes require a separate calibrated model and cannot be established from these masks alone.</p>
    <section class="evidence-heading"><h2>Best available views</h2><label for="evidence-overlay">Show orange/blue evidence overlay</label></section>
    <section class="views">{''.join(cards)}</section>
    <details class="technical"><summary>Technical details and limitations</summary>
      <h2>Assessment checks</h2><ul class="checks">{checks}</ul>
      {issues_section}
      <h2>Limitations</h2><ul>{limitations}</ul>
      <p class="provenance">Input fingerprint: {_escape(report['inputs']['fingerprint_sha256'])}<br>Ranking formula: {_escape(report['ranking_formula_version'])} · <a href="{_escape(report['links']['json'])}">Full JSON record</a></p>
    </details>
  </main>
</body>
</html>
"""


def build_and_persist_report(
    data_root: Path,
    dataset_id: str,
    manifest: dict[str, Any],
    building_id: str,
    *,
    expected_analysis_version: int,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    _building_parts(building_id)
    dataset_dir = (data_root / dataset_id).resolve()
    if dataset_dir.parent != data_root.resolve() or not dataset_dir.is_dir():
        raise ValueError("invalid dataset directory")
    collection = collect_building_evidence(dataset_dir, manifest, building_id, expected_analysis_version=expected_analysis_version)
    ranked = rank_views(collection.views, dataset_id, building_id)
    ranked = _with_screenshots(dataset_dir, dataset_id, building_id, ranked)
    raw_query = manifest.get("query")
    query: dict[str, Any] = raw_query if isinstance(raw_query, dict) else {}
    timestamp = (now or (lambda: datetime.now(UTC)))().astimezone(UTC).isoformat()
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": timestamp,
        "dataset": {
            "id": dataset_id,
            "provider": query.get("provider", "Unknown provider"),
            "acquired_at": query.get("acquired_at"),
            "manifest_panorama_count": collection.manifest_panorama_count,
        },
        "building": {"id": building_id, "identity_source": "openstreetmap"},
        "inputs": {
            "required_analysis_version": expected_analysis_version,
            "compatible_analysis_count": collection.compatible_analysis_count,
            "matching_analysis_count": len(collection.views),
            "stale_analysis_count": collection.stale_analysis_count,
            "incomplete_analysis_count": collection.incomplete_analysis_count,
            "invalid_analysis_count": collection.invalid_analysis_count,
            "fingerprint_sha256": collection.input_fingerprint_sha256,
            "analysis_files": [
                {
                    "image_id": view.image_id,
                    "artifacts": [
                        {"path": view.analysis_relative_path, "sha256": view.analysis_sha256, "role": "analysis_metadata"},
                        {"path": view.panorama_relative_path, "sha256": view.panorama_sha256, "role": "source_panorama"},
                        {"path": view.id_mask_relative_path, "sha256": view.id_mask_sha256, "role": "building_id_mask"},
                        {"path": view.inferred_mask_relative_path, "sha256": view.inferred_mask_sha256, "role": "vegetation_inference_mask"},
                    ],
                }
                for view in collection.views
            ],
            "issues": [issue.to_dict() for issue in collection.issues],
        },
        "ranking_formula_version": RANKING_FORMULA_VERSION,
        "assessment": assess_footprint(building_id, ranked, collection),
        "ranked_views": [view.to_dict() for view in ranked],
        "links": _report_links(dataset_id, building_id),
    }
    json_path, html_path = report_paths(dataset_dir, building_id)
    atomic_write_bytes(html_path, render_report_html(report).encode("utf-8"))
    atomic_write_bytes(json_path, (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return report


def load_report(dataset_dir: Path, building_id: str) -> dict[str, Any] | None:
    json_path, _ = report_paths(dataset_dir, building_id)
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != REPORT_SCHEMA_VERSION:
        return None
    building = payload.get("building")
    inputs = payload.get("inputs")
    ranked_views = payload.get("ranked_views")
    if (
        not isinstance(building, dict)
        or building.get("identity_source") != "openstreetmap"
        or building.get("id") != building_id
        or not isinstance(building.get("id"), str)
        or BUILDING_ID.fullmatch(building["id"]) is None
        or not isinstance(inputs, dict)
        or inputs.get("required_analysis_version") != building_analysis.ANALYSIS_VERSION
        or not isinstance(ranked_views, list)
    ):
        return None
    for view in ranked_views:
        evidence = view.get("evidence") if isinstance(view, dict) else None
        if not isinstance(evidence, dict) or evidence.get("footprint_source") != "osm":
            return None
        alternative_id = evidence.get("alternative_footprint_id")
        if alternative_id is not None and (not isinstance(alternative_id, str) or BUILDING_ID.fullmatch(alternative_id) is None):
            return None
    return payload


def load_report_html(dataset_dir: Path, building_id: str) -> str | None:
    report = load_report(dataset_dir, building_id)
    return render_report_html(report) if report is not None else None
