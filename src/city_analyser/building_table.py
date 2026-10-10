"""One row per OSM building: what the analyses saw, how well, and any attributes.

The analysis stores results per panorama; a consumer wants them per building.
``city-analyser-export`` reads every current analysis once and writes a
building table (GeoJSON, CSV and, when ``pyarrow`` is installed, GeoParquet),
a view table that says which panoramas back each row, a coverage report and an
export manifest with hashes and versions. Coverage is reported against the
footprints within reach of the captured streets, so buildings nobody
photographed are visible as a gap instead of silently missing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .analysis_cache import atomic_write_bytes, atomic_write_json
from .building_analysis import ANALYSIS_VERSION, analysis_configuration, analysis_paths, load_analysis
from .building_index import build_building_index, index_path, load_building_index
from .building_inspection.models import RANKING_FORMULA_VERSION
from .building_inspection.ranking import view_score
from .run_ledger import RunLedger, file_signature, ledger_path, settings_fingerprint

EXPORT_SCHEMA_VERSION = 1
# A view counts as usable evidence when it is not flagged for quality and the
# building fills at least this share of the panorama. Unvalidated: reported with the export.
DEFAULT_MIN_OBSERVED_FRACTION = 0.005
ATTRIBUTION = (
    "Footprints and identifiers: OpenStreetMap contributors, ODbL 1.0. Imagery-derived values: Mapillary contributors, "
    "CC BY-SA 4.0. Derived databases may carry share-alike obligations; check both licenses before redistributing."
)
ROW_COLUMNS = (
    "building_id", "osm_type", "osm_id", "parent_osm_id", "centroid_lon", "centroid_lat", "osm_version", "footprint_correction_revision",
    "association_status", "views", "clean_views", "flagged_views", "best_image_id", "best_score", "best_observed_fraction",
    "best_projection_coverage", "first_captured_ms", "last_captured_ms",
)
VIEW_COLUMNS = ("building_id", "image_id", "captured_at_ms", "observed_fraction", "projection_coverage", "score", "quality_flags", "usable")
ATTRIBUTE_COLUMNS = (
    "attributes_provider", "building_type", "building_type_score", "building_type_kind", "building_type_abstained",
    "surface_material", "surface_material_score", "surface_material_kind", "surface_material_abstained",
    "floors", "floors_spread", "floors_abstained", "construction_year", "construction_year_spread", "construction_year_abstained",
)


def _building_parts(building_id: str) -> tuple[str, str, str | None]:
    parts = building_id.split("/")
    return parts[0], parts[1], f"{parts[0]}/{parts[1]}" if len(parts) > 2 else None


def _current_analyses(
    data_root: Path, dataset_id: str, images: list[dict[str, Any]], verify: bool,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], str]], dict[str, int]]:
    """``(image, analysis, sha256 of its JSON)`` for every current analysis, plus counts.

    "Current" is answered from the run ledger when it knows the panorama (no hashing of
    images); anything else, or everything with ``verify``, is validated from the artifacts.
    """
    ledger = RunLedger(ledger_path(data_root, dataset_id))
    fingerprint = settings_fingerprint("analysis", analysis_configuration())
    found, counts = [], {"analyzed_current": 0, "not_analyzed": 0, "stale_or_invalid": 0}
    for image in images:
        image_id = str(image["id"])
        metadata_path, _ = analysis_paths(data_root, dataset_id, image_id)
        try:
            raw = metadata_path.read_bytes()
        except OSError:
            counts["not_analyzed"] += 1
            continue
        try:
            analysis = json.loads(raw)
            signature = file_signature(data_root / dataset_id / str(image["local_file"]))
        except (json.JSONDecodeError, OSError):
            counts["stale_or_invalid"] += 1
            continue
        trusted = not verify and isinstance(analysis, dict) and analysis.get("version") == ANALYSIS_VERSION and ledger.is_done("analysis", image_id, fingerprint, signature)
        if not trusted and (not isinstance(analysis, dict) or load_analysis(data_root, dataset_id, image_id, image) is None):
            counts["stale_or_invalid"] += 1
            continue
        counts["analyzed_current"] += 1
        found.append((image, analysis, hashlib.sha256(raw).hexdigest()))
    return found, counts


def collect_buildings(
    analyses: list[tuple[dict[str, Any], dict[str, Any], str]], min_observed_fraction: float,
) -> dict[str, dict[str, Any]]:
    """Group the per-panorama building records by OSM ID."""
    buildings: dict[str, dict[str, Any]] = {}
    for image, analysis, _ in analyses:
        width, height = analysis.get("width"), analysis.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            continue
        quality = analysis.get("image_quality") if isinstance(analysis.get("image_quality"), dict) else None
        flags = list(quality.get("flags", [])) if quality else []
        for item in (analysis.get("building_ids") or {}).values():
            if not isinstance(item, dict) or not isinstance(item.get("osm_id"), str):
                continue
            pixels, coverage = item.get("pixels"), item.get("confidence")
            if not isinstance(pixels, int) or isinstance(pixels, bool) or not isinstance(coverage, (int, float)):
                continue
            fraction = pixels / (width * height)
            score = view_score(fraction, float(coverage), quality)
            entry = buildings.setdefault(item["osm_id"], {"views": []})
            entry["views"].append({
                "image_id": str(image["id"]), "captured_at_ms": image.get("captured_at_ms") if isinstance(image.get("captured_at_ms"), int) else None,
                "observed_fraction": round(fraction, 6), "projection_coverage": round(float(coverage), 4), "score": round(score, 8),
                "quality_flags": flags, "usable": not flags and fraction >= min_observed_fraction,
                "geometry": item.get("footprint_geometry"), "osm_version": item.get("osm_version"), "correction": item.get("footprint_correction"),
            })
    return buildings


def _centroid(geometry: Any) -> tuple[float | None, float | None]:
    try:
        from shapely.geometry import shape

        point = shape(geometry).centroid
        return round(point.x, 7), round(point.y, 7)
    except Exception:  # noqa: BLE001 - a malformed footprint must not stop the export
        return None, None


def association_status(views: list[dict[str, Any]]) -> str:
    usable = {view["image_id"] for view in views if view["usable"]}
    if len(usable) >= 2:
        return "multi_view"
    if len(usable) == 1:
        return "single_view"
    return "weak_views_only"


def build_rows(buildings: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for building_id in sorted(buildings):
        views = sorted(buildings[building_id]["views"], key=lambda view: (-view["score"], view["image_id"]))
        best = views[0]
        osm_type, osm_id, parent = _building_parts(building_id)
        longitude, latitude = _centroid(best["geometry"])
        captured = [view["captured_at_ms"] for view in views if view["captured_at_ms"] is not None]
        correction = best["correction"] if isinstance(best["correction"], dict) else None
        rows.append({
            "building_id": building_id, "osm_type": osm_type, "osm_id": osm_id, "parent_osm_id": parent,
            "centroid_lon": longitude, "centroid_lat": latitude, "osm_version": best["osm_version"],
            "footprint_correction_revision": correction.get("revision") if correction else None,
            "association_status": association_status(views), "views": len(views),
            "clean_views": sum(view["usable"] for view in views), "flagged_views": sum(bool(view["quality_flags"]) for view in views),
            "best_image_id": best["image_id"], "best_score": best["score"], "best_observed_fraction": best["observed_fraction"],
            "best_projection_coverage": best["projection_coverage"],
            "first_captured_ms": min(captured) if captured else None, "last_captured_ms": max(captured) if captured else None,
            "_geometry": best["geometry"], "_views": views,
        })
    return rows


def attribute_columns(payload: dict[str, Any] | None, provider: str | None) -> dict[str, Any]:
    """Flat columns for one building's attribute file; every column is present (``None`` when absent)."""
    columns: dict[str, Any] = dict.fromkeys(ATTRIBUTE_COLUMNS)
    if payload is None:
        return columns
    columns["attributes_provider"] = payload.get("provider", provider)
    for name, prediction in (payload.get("predictions") or {}).items():
        if prediction.get("type") == "categorical" and name in ("building_type", "surface_material"):
            scores = prediction.get("scores") or {}
            top = max(scores, key=lambda key: scores[key]) if scores else None
            columns[name] = top if not prediction.get("abstained") else None
            columns[f"{name}_score"] = round(scores[top], 4) if top is not None else None
            columns[f"{name}_kind"] = prediction.get("score_kind")
            columns[f"{name}_abstained"] = bool(prediction.get("abstained"))
        elif prediction.get("type") == "numeric" and name in ("floors", "construction_year"):
            columns[name] = prediction.get("value")
            columns[f"{name}_spread"] = prediction.get("spread")
            columns[f"{name}_abstained"] = bool(prediction.get("abstained"))
    return columns


def coverage_report(
    dataset_id: str, rows: list[dict[str, Any]], panorama_counts: dict[str, int], total_panoramas: int,
    index: dict[str, Any] | None, min_observed_fraction: float,
) -> dict[str, Any]:
    matched = {row["building_id"] for row in rows}
    footprints: dict[str, Any] = {"matched": len(matched)}
    fractions: dict[str, float | None] = {}
    if index is not None:
        nearby, visible = set(index.get("nearby_buildings", [])), set(index["buildings"])
        footprints.update({"near_captured_streets": len(nearby), "in_line_of_sight": len(visible)})
        denominator = len(nearby) or None
        fractions["matched_of_nearby"] = round(len(matched & nearby) / denominator, 4) if denominator else None
        fractions["line_of_sight_of_nearby"] = round(len(visible & nearby) / denominator, 4) if denominator else None
        missing = sorted(visible - matched)
        footprints["seen_but_not_matched"] = len(missing)
        footprints["seen_but_not_matched_examples"] = missing[:20]
    status = Counter(row["association_status"] for row in rows)
    footprints.update({"multi_view": status["multi_view"], "single_view": status["single_view"], "weak_views_only": status["weak_views_only"]})
    if index is not None and footprints.get("near_captured_streets"):
        usable = sum(row["clean_views"] > 0 for row in rows)
        fractions["usable_view_of_nearby"] = round(usable / footprints["near_captured_streets"], 4)
    histogram = Counter("1" if row["views"] == 1 else "2" if row["views"] == 2 else "3-5" if row["views"] <= 5 else "6+" for row in rows)
    return {
        "dataset": dataset_id, "analysis_version": ANALYSIS_VERSION,
        "panoramas": {"in_manifest": total_panoramas, **panorama_counts},
        "footprints": footprints, "fractions": fractions,
        "views_per_building": {key: histogram.get(key, 0) for key in ("1", "2", "3-5", "6+")},
        "parameters": {"min_observed_fraction": min_observed_fraction},
        "note": "Coverage is measured against footprints near captured streets; buildings with no line of sight, "
                "or hidden by trees and vehicles, are not evidence of anything. The usable-view threshold is a heuristic.",
    }


def _csv_bytes(columns: tuple[str, ...], records: list[dict[str, Any]]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow({key: (";".join(value) if isinstance(value, list) else value) for key, value in record.items() if key in columns})
    return output.getvalue().encode("utf-8")


def _geojson(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": row["building_id"], "geometry": row["_geometry"], "properties": {key: row.get(key) for key in columns}}
            for row in rows
        ],
    }


def _write_parquet(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> bool:
    """GeoParquet 1.0 (WKB geometry, CRS84) when pyarrow is installed; returns whether it was written."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False
    from shapely.geometry import shape

    table = pa.table({**{key: [row.get(key) for row in rows] for key in columns}, "geometry": [shape(row["_geometry"]).wkb for row in rows]})
    geo = {"version": "1.0.0", "primary_column": "geometry", "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Polygon", "MultiPolygon"]}}}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"geo": json.dumps(geo).encode("utf-8")})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return True


def export_buildings(
    data_root: Path, dataset_id: str, *, provider: str | None = None, min_observed_fraction: float = DEFAULT_MIN_OBSERVED_FRACTION,
    verify: bool = False, use_index: bool = True, output_dir: Path | None = None,
) -> dict[str, Any]:
    from .attributes.cli import attributes_path
    from .local_viewer import dataset_manifest

    dataset_dir = (data_root / dataset_id).resolve()
    images = [image for image in dataset_manifest(data_root, dataset_id)["images"] if isinstance(image, dict) and image.get("local_file")]
    analyses, panorama_counts = _current_analyses(data_root, dataset_id, images, verify)
    rows = build_rows(collect_buildings(analyses, min_observed_fraction))
    for row in rows:
        payload = None
        if provider:
            try:
                payload = json.loads(attributes_path(dataset_dir, row["building_id"], provider).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
        row.update(attribute_columns(payload, provider))
    index = None
    if use_index:
        index = load_building_index(data_root, dataset_id, images)
        if index is None:
            index = build_building_index(data_root, dataset_id, images)
            atomic_write_json(index_path(data_root, dataset_id), index)
    coverage = coverage_report(dataset_id, rows, panorama_counts, len(images), index, min_observed_fraction)
    output = output_dir or dataset_dir / "exports"
    columns = ROW_COLUMNS + ATTRIBUTE_COLUMNS
    files: dict[str, bytes] = {
        "buildings.geojson": (json.dumps(_geojson(rows, columns), separators=(",", ":")) + "\n").encode("utf-8"),
        "buildings.csv": _csv_bytes(columns, rows),
        "building_views.csv": _csv_bytes(VIEW_COLUMNS, [{"building_id": row["building_id"], **view} for row in rows for view in row["_views"]]),
        "coverage.json": (json.dumps(coverage, indent=2) + "\n").encode("utf-8"),
    }
    for name, content in files.items():
        atomic_write_bytes(output / name, content)
    written = list(files)
    if _write_parquet(output / "buildings.parquet", rows, columns):
        written.append("buildings.parquet")
    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION, "dataset": dataset_id, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "buildings": len(rows), "attributes_provider": provider,
        "versions": {"analysis": ANALYSIS_VERSION, "ranking_formula": RANKING_FORMULA_VERSION},
        "inputs_sha256": hashlib.sha256("\n".join(sorted(f"{image['id']}:{digest}" for image, _, digest in analyses)).encode("ascii")).hexdigest(),
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in written},
        "usable_view_threshold": {"min_observed_fraction": min_observed_fraction, "excludes_flagged_quality": True},
        "attribution": ATTRIBUTION,
    }
    atomic_write_json(output / "export-manifest.json", manifest)
    return {"output": str(output), "files": written + ["export-manifest.json"], "buildings": len(rows), "coverage": coverage}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one row per OSM building (GeoJSON, CSV, GeoParquet if pyarrow is installed) with a coverage report.")
    parser.add_argument("dataset", help="dataset directory name under --data-root")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=None, help="directory (default: <dataset>/exports)")
    parser.add_argument("--provider", default=None, help="attributes provider whose files to join, e.g. osm-tag-prior or vlm_<model>")
    parser.add_argument("--min-observed-fraction", type=float, default=DEFAULT_MIN_OBSERVED_FRACTION, help="share of the panorama a building must fill for a view to count as usable (default: %(default)s)")
    parser.add_argument("--verify", action="store_true", help="validate every analysis from its artifacts instead of trusting the run ledger (hashes files; slow on big datasets)")
    parser.add_argument("--no-index", action="store_true", help="skip the footprint index; coverage is then reported only for matched buildings")
    args = parser.parse_args()
    try:
        result = export_buildings(
            args.data_root, args.dataset, provider=args.provider, min_observed_fraction=args.min_observed_fraction,
            verify=args.verify, use_index=not args.no_index, output_dir=args.output,
        )
    except ValueError as error:
        parser.error(str(error))
    coverage = result["coverage"]
    footprints = coverage["footprints"]
    print(f"{result['buildings']} buildings -> {result['output']} ({', '.join(result['files'])})")
    print(f"panoramas: {coverage['panoramas']}")
    print(f"footprints: {footprints}")
    if coverage["fractions"]:
        print(f"coverage: {coverage['fractions']}")
    if "buildings.parquet" not in result["files"]:
        print("GeoParquet skipped: install pyarrow (uv sync --extra parquet) to write it.")


if __name__ == "__main__":
    main()
