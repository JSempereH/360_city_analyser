"""Mask quality per view: does a building's mask in one panorama agree with where its footprint projects?

Three independent sources describe a building in a panorama: the mask the
analysis assigned it (image), its OSM walls rendered from the refined pose
(geometry) and, for the roof, its known or default height. Where they disagree
the view is suspect. Each signal below answers one failure seen when masks were
reviewed by eye (``docs/experiments.md``, point 4) and comes with a readable
reason; a view is flagged, never deleted, so a wrong pose or footprint that
makes a good mask look bad can still be told apart by a person.

    uv run city-analyser-mask-quality score valencia-ciutat-vella
    uv run city-analyser-mask-quality evaluate benchmarks/mask-labels/*.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .building_analysis import panorama_analysis_files

QUALITY_VERSION = 1
QUALITY_FILE = "mask-quality.json"

# Below this share of the panorama a mask is a sliver: too few pixels to read storeys or structure.
MIN_PIXEL_SHARE = 0.002
# Share of the mask lying in columns where the building's walls do not project: it reaches into a neighbor.
MAX_OUTSIDE_SHARE = 0.3
# A gap wider than this between mask columns, where the walls do not project either, splits two buildings.
SPLIT_GAP_DEGREES = 3.0
# Share of the mask above the expected roof (height plus a margin for attics and parapets): it leaks into the sky.
MAX_ABOVE_ROOF_SHARE = 0.25
ROOF_MARGIN = 1.3
ROOF_MARGIN_M = 3.0
# Many separate pieces, or a mask that fills little of its own column span: speckle.
MAX_COMPONENTS = 4
MIN_COMPONENT_SHARE = 0.02
MIN_FILL = 0.45
# Share of the mask more than this far above the horizon: an arch or overhang seen from below.
OVERHEAD_ELEVATION_DEGREES = 50.0
MAX_OVERHEAD_SHARE = 0.4
# Share of the projected, unhidden facade the mask covers; below it a large visible facade is mostly missing.
MIN_COVERAGE = 0.15
MISSING_MIN_PROJECTED_SHARE = 0.01
CAMERA_HEIGHT_M = 1.7

REASONS = {
    "interior": "the panorama was taken inside a building footprint",
    "sliver": "the mask is too small to show the facade",
    "neighbor": "much of the mask lies where the building's walls do not project",
    "street": "the mask is split in two with a gap where the walls do not project",
    "sky": "much of the mask is above the expected roof",
    "speckle": "the mask is broken into scattered pieces",
    "overhead": "the mask is mostly high above the horizon (an arch or overhang)",
    "missing": "most of the projected facade has no mask",
}


def view_signals(
    ids: np.ndarray, ordinal: int, owners: np.ndarray, depth: np.ndarray, *, height_m: float, hidden: np.ndarray | None = None,
    camera_inside: bool = False,
) -> dict[str, Any]:
    """The signals of one building's mask (``ids == ordinal``) against its projection (``owners == ordinal``, ``depth`` in meters)."""
    from scipy import ndimage

    height, width = ids.shape
    mask = ids == ordinal
    projected = owners == ordinal
    pixels = int(mask.sum())
    signals: dict[str, Any] = {"pixels": pixels, "pixel_share": round(pixels / (height * width), 5), "camera_inside": camera_inside}
    if not pixels:
        return signals
    rows, columns = np.nonzero(mask)
    mask_columns = np.unique(columns)
    projected_columns = projected.any(axis=0)
    signals["width_degrees"] = round(len(mask_columns) * 360 / width, 2)
    signals["outside_share"] = round(float((~projected_columns[columns]).mean()), 3)
    # Gaps narrower than SPLIT_GAP_DEGREES never split; wider ones split unless the walls project across them.
    signals["runs"] = _runs_with_minimum_gap(mask_columns, width, projected_columns, max(1, round(width * SPLIT_GAP_DEGREES / 360)))
    # Expected roof per column: the nearest projected wall of this building at its height plus a margin.
    near = np.where(projected, depth, np.inf).min(axis=0)
    roof_m = height_m * ROOF_MARGIN + ROOF_MARGIN_M - CAMERA_HEIGHT_M
    roof_row = np.full(width, -1, dtype=np.int64)
    known = np.isfinite(near)
    roof_row[known] = np.floor((0.5 - np.arctan2(roof_m, near[known]) / math.pi) * height).astype(np.int64)
    in_known = known[columns]
    signals["above_roof_share"] = round(float((rows[in_known] < roof_row[columns[in_known]]).mean()), 3) if in_known.any() else None
    labeled = ndimage.label(mask)[0]
    sizes = np.bincount(labeled.ravel())[1:]
    signals["components"] = int((sizes >= MIN_COMPONENT_SHARE * pixels).sum())
    tops = np.full(width, height, dtype=np.int64)
    bottoms = np.full(width, -1, dtype=np.int64)
    np.minimum.at(tops, columns, rows)
    np.maximum.at(bottoms, columns, rows)
    span = int((bottoms[mask_columns] - tops[mask_columns] + 1).sum())
    signals["fill"] = round(pixels / max(span, 1), 3)
    elevation = 90.0 - (rows + 0.5) / height * 180.0
    signals["overhead_share"] = round(float((elevation > OVERHEAD_ELEVATION_DEGREES).mean()), 3)
    expected = projected & ~hidden if hidden is not None else projected
    expected_pixels = int(expected.sum())
    signals["projected_share"] = round(expected_pixels / (height * width), 5)
    signals["coverage"] = round(int((mask & expected).sum()) / expected_pixels, 3) if expected_pixels else None
    return signals


def _runs_with_minimum_gap(columns: np.ndarray, width: int, joinable: np.ndarray, minimum_gap: int) -> int:
    """Runs of mask columns, where only gaps of at least ``minimum_gap`` columns with no projected wall in them separate two."""
    if len(columns) == 0:
        return 0
    splits = 0
    for index, column in enumerate(columns):
        previous = columns[index - 1] if index else columns[-1] - width
        gap = np.arange(previous + 1, column) % width
        if len(gap) >= minimum_gap and not joinable[gap].any():
            splits += 1
    return max(splits, 1)


def assess(signals: dict[str, Any]) -> dict[str, Any]:
    """``{"ok", "reasons"}``: every rule the view breaks, in order of how certain it is."""
    reasons = []
    if signals.get("camera_inside"):
        reasons.append("interior")
    if signals["pixel_share"] < MIN_PIXEL_SHARE:
        reasons.append("sliver")
    if signals["pixels"]:
        if signals["overhead_share"] > MAX_OVERHEAD_SHARE:
            reasons.append("overhead")
        if signals["outside_share"] > MAX_OUTSIDE_SHARE:
            reasons.append("neighbor")
        if signals["runs"] > 1:
            reasons.append("street")
        if (signals["above_roof_share"] or 0) > MAX_ABOVE_ROOF_SHARE:
            reasons.append("sky")
        if signals["components"] > MAX_COMPONENTS or signals["fill"] < MIN_FILL:
            reasons.append("speckle")
        if signals["coverage"] is not None and signals["projected_share"] >= MISSING_MIN_PROJECTED_SHARE and signals["coverage"] < MIN_COVERAGE:
            reasons.append("missing")
    return {"ok": not reasons, "reasons": reasons}


def building_height_m(record: dict[str, Any]) -> float:
    """The height a building's walls were drawn with: its known height, or the drawing default when OSM has none."""
    from .building_analysis import DEFAULT_BUILDING_HEIGHT_M

    height_m = float(record.get("height_m") or DEFAULT_BUILDING_HEIGHT_M)
    return float(record.get("render_height_m") or height_m) if record.get("height_source") == "default" else height_m


def score_views(
    ids: np.ndarray, hidden_ids: np.ndarray | None, owners: np.ndarray, depth: np.ndarray, building_ids: dict[str, dict[str, Any]],
    camera_inside: bool,
) -> dict[str, dict[str, Any]]:
    """``{ordinal: {"version", "ok", "reasons", "signals"}}`` for every building the analysis named, ``owners`` drawn
    with the same ordinals as ``ids``."""
    results = {}
    for ordinal, record in building_ids.items():
        number = int(ordinal)
        hidden = hidden_ids == number if hidden_ids is not None else None
        signals = view_signals(ids, number, owners, depth, height_m=building_height_m(record), hidden=hidden, camera_inside=camera_inside)
        results[ordinal] = {"version": QUALITY_VERSION, **assess(signals), "signals": signals}
    return results


def stored_quality(analysis: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    """``{osm_id: quality}`` as the analysis recorded it while associating (version 33 on), or None."""
    records = analysis.get("building_ids") or {}
    stored = {record["osm_id"]: record.get("mask_quality") for record in records.values()}
    if not stored or any(not isinstance(value, dict) or value.get("version") != QUALITY_VERSION for value in stored.values()):
        return None
    return stored


def camera_inside_footprint(features: list[dict[str, Any]], latitude: float, longitude: float) -> bool:
    from .facade_geometry import _points_in_polygon, build_scene

    scene = build_scene(features, latitude, longitude)
    origin = np.zeros((1, 2))
    return any(bool(_points_in_polygon(origin, polygon)[0]) for polygon in scene.polygons)


def panorama_quality(dataset_dir: Path, image_id: str) -> dict[str, dict[str, Any]]:
    """``{osm_id: {"signals", "ok", "reasons"}}`` for every building the analysis of one panorama named.

    The interior test uses the refined pose: the recorded GPS of a narrow street often falls inside a
    footprint, while the analysis repairs it to open space unless the camera really is enclosed.
    """
    from PIL import Image

    from .building_analysis import _render_facade_owners, camera_up_vector, nearby_buildings

    analysis = json.loads((dataset_dir / "analysis" / f"{image_id}.json").read_text(encoding="utf-8"))
    stored = stored_quality(analysis)
    if stored is not None:
        return stored
    with Image.open(dataset_dir / "analysis" / f"{image_id}-ids.png") as source:
        ids = np.asarray(source).astype(np.int32)
    hidden_path = dataset_dir / "analysis" / f"{image_id}-facades.png"
    hidden_ids = None
    if hidden_path.is_file():
        with Image.open(hidden_path) as source:
            hidden_ids = np.asarray(source).astype(np.int32)
    pose = analysis["refined_pose"]
    latitude, longitude, heading = float(pose["latitude"]), float(pose["longitude"]), float(pose["heading_degrees"])
    # The same footprints and tilt the analysis projected (analyses before version 30 recorded no tilt).
    features = nearby_buildings(latitude, longitude)
    owners, depth, candidates, _ = _render_facade_owners(
        features, latitude, longitude, heading, ids.shape[1], ids.shape[0], camera_up_vector(analysis.get("camera_up")),
    )
    ordinals = {str(feature["properties"]["osm_id"]): index for index, feature in enumerate(candidates, start=1)}
    # The analysis rendered the same footprints from the same pose; renumber the walls with its ordinals.
    lookup = np.zeros(len(candidates) + 1, dtype=np.int32)
    named = {}
    for ordinal, record in analysis["building_ids"].items():
        rendered = ordinals.get(record["osm_id"])
        if rendered is not None:
            lookup[rendered] = int(ordinal)
            named[ordinal] = record
    scored = score_views(ids, hidden_ids, lookup[owners], depth, named, camera_inside_footprint(features, latitude, longitude))
    return {named[ordinal]["osm_id"]: result for ordinal, result in scored.items()}


def analysis_versions(dataset_dir: Path) -> list[int]:
    """The ``ANALYSIS_VERSION`` values of a dataset's analyses: labels describe the masks of one version."""
    return sorted({
        json.loads(path.read_text(encoding="utf-8")).get("version")
        for path in panorama_analysis_files(dataset_dir)
    })


def dataset_quality(data_root: Path, dataset_id: str, log: Any = print) -> dict[str, Any]:
    dataset_dir = data_root / dataset_id
    views = []
    for path in panorama_analysis_files(dataset_dir):
        image_id = path.stem
        for building, result in panorama_quality(dataset_dir, image_id).items():
            views.append({"image": image_id, "building": building, **result})
        log(f"  {image_id}: {sum(view['image'] == image_id for view in views)} views")
    payload = {"version": QUALITY_VERSION, "dataset": dataset_id, "analysis_versions": analysis_versions(dataset_dir), "views": views,
               "flagged": dict(Counter(reason for view in views for reason in view["reasons"]))}
    (dataset_dir / QUALITY_FILE).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return payload


def labels_from_curation(source_dir: Path, demo_dir: Path) -> list[dict[str, Any]]:
    """Benchmark labels from a demo made of ``source_dir``: every view its curation edited is bad, with the failure the edit fixed;
    of the views the demo then showed (its crops), one a reviewer answered is good and one left as a sliver is bad.
    Views the demo never showed were not reviewed."""
    from .demo import SLIVER_SCORE

    curation = json.loads((demo_dir / "curation.json").read_text(encoding="utf-8"))
    edits: dict[tuple[str, str], set[str]] = {}
    for edit in curation.get("masks", []):
        reasons = edits.setdefault((str(edit["image"]), edit["building"]), set())
        if edit.get("remove"):
            reasons.add("removed")
        if "keep_bearings" in edit:
            reasons.add("neighbor")
        if "roof_line" in edit and "keep_bearings" in edit:
            reasons.add("sky")
        if {"claim_bearings", "fill_bearings", "add"} & set(edit):
            reasons.add("missing")
        if "rename_to" in edit:
            reasons.add("wrong_id")
    answered = {(str(image), building) for building, views in curation.get("answers", {}).items() for image in views}
    shown = {}
    for sidecar_path in (demo_dir / "crops").rglob("crop-*.json"):
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        view = (str(sidecar["source"]["image_id"]), sidecar["building"]["id"])
        shown[view] = view in answered or sidecar["evidence"]["score"] >= SLIVER_SCORE
    labels = []
    for path in panorama_analysis_files(source_dir):
        analysis = json.loads(path.read_text(encoding="utf-8"))
        for record in analysis["building_ids"].values():
            view = (str(analysis["image_id"]), record["osm_id"])
            if view in edits:
                labels.append({"image": view[0], "building": view[1], "label": "bad", "reasons": sorted(edits[view])})
            elif view in shown:
                labels.append({"image": view[0], "building": view[1], "label": "good"} if shown[view]
                              else {"image": view[0], "building": view[1], "label": "bad", "reasons": ["sliver"]})
    return labels


def evaluate(labels: Iterable[dict[str, Any]], scored: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """Bad views caught (recall), good views dropped (false alarms) and recall per labeled reason."""
    caught = missed = dropped = kept = 0
    per_reason: dict[str, list[int]] = {}
    unscored = 0
    for label in labels:
        result = scored.get((label["image"], label["building"]))
        if result is None:
            unscored += 1
            continue
        flagged = not result["ok"]
        if label["label"] == "bad":
            caught, missed = caught + flagged, missed + (not flagged)
            for reason in label.get("reasons", []):
                per_reason.setdefault(reason, [0, 0])
                per_reason[reason][0] += flagged
                per_reason[reason][1] += 1
        else:
            dropped, kept = dropped + flagged, kept + (not flagged)
    return {
        "bad": caught + missed, "good": dropped + kept, "unscored": unscored,
        "recall": round(caught / max(caught + missed, 1), 3), "false_alarms": round(dropped / max(dropped + kept, 1), 3),
        "recall_by_reason": {reason: f"{hit}/{total}" for reason, (hit, total) in sorted(per_reason.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help=f"score every view of a dataset into data/<dataset>/{QUALITY_FILE}")
    score.add_argument("dataset")
    curation = commands.add_parser("labels-from-curation", help="benchmark labels from a curated demo")
    curation.add_argument("dataset", help="the dataset the demo was copied from (masks before the curation)")
    curation.add_argument("demo", help="the demo dataset, with its curation.json and crops")
    curation.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("evaluate", help="compare the scores with labeled views")
    check.add_argument("labels", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "score":
        payload = dataset_quality(args.data_root, args.dataset)
        print(f"{len(payload['views'])} views, flagged: {payload['flagged']}")
    elif args.command == "labels-from-curation":
        labels = labels_from_curation(args.data_root / args.dataset, args.data_root / args.demo)
        document = {"dataset": args.dataset, "analysis_versions": analysis_versions(args.data_root / args.dataset),
                    "note": f"Derived from the curation and crops of {args.demo} with city-analyser-mask-quality labels-from-curation: "
                    "edited views are bad, views the demo showed unedited are good.", "labels": labels}
        args.output.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        print(f"{len(labels)} views, {sum(label['label'] == 'bad' for label in labels)} bad")
    else:
        for path in args.labels:
            document = json.loads(path.read_text(encoding="utf-8"))
            quality_path = args.data_root / document["dataset"] / QUALITY_FILE
            if not quality_path.is_file():
                print(f"{path}: no {quality_path}; run `city-analyser-mask-quality score {document['dataset']}` first")
                continue
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            # A re-analysis changes the masks, so labels only describe the analysis version they were made on.
            if document.get("analysis_versions") != quality.get("analysis_versions"):
                print(f"{path.name}: labeled on analysis version {document.get('analysis_versions')}, "
                      f"scored on {quality.get('analysis_versions')}: label the current masks again")
                continue
            scored = {(view["image"], view["building"]): view for view in quality["views"]}
            print(path.name, json.dumps(evaluate(document["labels"], scored)))


if __name__ == "__main__":
    main()
