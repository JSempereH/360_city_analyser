"""Reference values to measure attribute providers against: the cadastre, an engineer-labeled evaluation set and OSM tags.

Each source measures different attributes and is never mixed with another in
one score (see docs/experiments.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..attributes.taxonomy import ALL_ATTRIBUTES

SOURCES = ("catastro", "evaluation_set", "osm_tags")
# What each source can tell; anything else from it is ignored.
SOURCE_ATTRIBUTES = {
    "catastro": ("floors", "construction_year", "building_type"),
    "evaluation_set": ALL_ATTRIBUTES,
    "osm_tags": ("floors", "construction_year", "building_type", "surface_material"),
}


def _value(prediction: dict[str, Any]) -> Any:
    if prediction.get("abstained"):
        return None
    if prediction.get("type") == "numeric":
        return prediction.get("value")
    scores = prediction.get("scores") or {}
    return max(scores, key=lambda name: scores[name]) if scores else None


def _attribute_files(dataset_dir: Path, file_name: str) -> list[dict[str, Any]]:
    payloads = []
    for path in sorted((dataset_dir / "attributes").rglob(file_name)):
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return payloads


def load_truth(dataset_dir: Path, source: str) -> dict[str, dict[str, Any]]:
    """``{building: {attribute: value}}`` from one source; values a source cannot tell are left out."""
    if source not in SOURCE_ATTRIBUTES:
        raise ValueError(f"unknown reference source {source!r} (one of {', '.join(SOURCES)})")
    attributes = SOURCE_ATTRIBUTES[source]
    truth: dict[str, dict[str, Any]] = {}
    if source == "catastro":
        path = dataset_dir / "groundtruth" / "catastro.json"
        if path.is_file():
            for building, record in json.loads(path.read_text(encoding="utf-8"))["buildings"].items():
                values = {name: record.get(name) for name in attributes if record.get(name) is not None}
                if values and record.get("use"):
                    values["use"] = record["use"]  # context for grouping, not scored
                truth[building] = values
    elif source == "evaluation_set":
        for payload in _attribute_files(dataset_dir, "attributes-human-review.json"):
            if (payload.get("run") or {}).get("evaluation"):
                truth[payload["building"]["id"]] = {
                    name: value for name, prediction in payload["predictions"].items()
                    if name in attributes and (value := _value(prediction)) is not None
                }
    elif source == "osm_tags":
        for payload in _attribute_files(dataset_dir, "attributes-osm-tag-prior.json"):
            truth[payload["building"]["id"]] = {
                name: value for name, prediction in payload["predictions"].items()
                if name in attributes and (value := _value(prediction)) is not None
            }
    return {building: values for building, values in truth.items() if values}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reference values for a dataset's buildings, to score attribute providers against.")
    commands = parser.add_subparsers(dest="command", required=True)
    catastro = commands.add_parser("catastro", help="match the dataset's OSM footprints to the Spanish cadastre (storeys, year, use)")
    catastro.add_argument("dataset")
    catastro.add_argument("--data-root", type=Path, default=Path("data"))
    catastro.add_argument("--cache-dir", type=Path, default=None, help="where downloaded cadastral files are kept (default: <data-root>/.groundtruth-cache)")
    catastro.add_argument("--min-iou", type=float, default=0.5, help="footprint overlap a match needs (default: %(default)s)")
    summary = commands.add_parser("summary", help="how many buildings each source covers, per attribute")
    summary.add_argument("dataset")
    summary.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.command == "catastro":
        from .catastro import build_reference

        try:
            result = build_reference(args.data_root, args.dataset, args.cache_dir or args.data_root / ".groundtruth-cache", min_iou=args.min_iou)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        found = result["match"]
        print(f"{found['matched']} of {found['osm_footprints']} footprints matched to the cadastre ({found['cadastral_buildings_in_area']} cadastral buildings in the area) "
              f"-> {args.data_root / args.dataset / 'groundtruth' / 'catastro.json'}")
        return
    dataset_dir = args.data_root / args.dataset
    for source in SOURCES:
        truth = load_truth(dataset_dir, source)
        counts = {name: sum(name in values for values in truth.values()) for name in ALL_ATTRIBUTES}
        print(f"{source:15s} {len(truth):4d} buildings  " + "  ".join(f"{name} {count}" for name, count in counts.items() if count))


if __name__ == "__main__":
    main()
