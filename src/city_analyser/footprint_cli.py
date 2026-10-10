"""List and export reviewed footprint corrections.

``export --format osm`` writes a JOSM file of accepted corrections for a person
to compare with OpenStreetMap and map by hand. It is marked ``upload="never"``:
corrections are never uploaded automatically, as the OpenStreetMap Automated
Edits code of conduct requires.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .footprint_corrections import active_revision, all_records

SOURCE_TAG = "survey;Mapillary street-level imagery"


def accepted_corrections() -> list[tuple[str, dict[str, Any]]]:
    pairs = ((record["osm_id"], active_revision(record)) for record in all_records())
    return [(osm_id, revision) for osm_id, revision in pairs if revision is not None]


def _outlines(revision: dict[str, Any]) -> list[tuple[dict[str, Any], int | None]]:
    """``(geometry, part number)`` for a geometry correction or for each part of a split."""
    if revision.get("kind") == "split":
        return [(part, number) for number, part in enumerate(revision["parts"], start=1)]
    return [(revision["geometry"], None)]


def as_geojson(corrections: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "osm_id": osm_id, "part": part, "revision": revision["revision"], "kind": revision.get("kind", "geometry"),
                    "base_version": revision.get("base_version"), "source": revision["source"], "author": revision.get("author"),
                    "reviewer": revision.get("reviewer"), "note": revision.get("note"), "reviewed_at": revision.get("reviewed_at"),
                },
            }
            for osm_id, revision in corrections
            for geometry, part in _outlines(revision)
        ],
    }


def _outer_rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    return [polygon[0] for polygon in polygons]


def as_osm_xml(corrections: list[tuple[str, dict[str, Any]]]) -> bytes:
    """A JOSM reference layer: one new way per corrected outer ring, never uploadable."""
    root = ElementTree.Element("osm", version="0.6", generator="360-city-analyser", upload="never")
    next_id = -1
    for osm_id, revision in corrections:
        outlines = _outlines(revision)
        for geometry, part in outlines:
            for ring in _outer_rings(geometry):
                node_ids = []
                # The closing vertex repeats the first; the way reuses its node instead.
                for longitude, latitude in ring[:-1]:
                    ElementTree.SubElement(root, "node", id=str(next_id), lon=f"{longitude:.8f}", lat=f"{latitude:.8f}", version="0")
                    node_ids.append(next_id)
                    next_id -= 1
                way = ElementTree.SubElement(root, "way", id=str(next_id), version="0")
                next_id -= 1
                for node_id in [*node_ids, node_ids[0]]:
                    ElementTree.SubElement(way, "nd", ref=str(node_id))
                tags = {
                    "fixme": (
                        f"Reviewed {'split of' if part else 'correction of'} {osm_id} (version {revision.get('base_version')})"
                        + (f", part {part} of {len(outlines)}" if part else "") + f": {revision.get('note', '')}"
                    ),
                    "note": f"360-city-analyser footprint correction revision {revision['revision']}; compare with {osm_id} and edit that element by hand",
                    "source": SOURCE_TAG,
                }
                for key, value in tags.items():
                    ElementTree.SubElement(way, "tag", k=key, v=value)
    ElementTree.indent(root)
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="List and export reviewed footprint corrections.")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="show every corrected building and its latest revision")
    listing.add_argument("--status", choices=["proposed", "accepted", "rejected", "superseded", "reverted"], help="only revisions with this status")
    export = commands.add_parser("export", help="export accepted corrections")
    export.add_argument("--format", choices=["geojson", "osm"], default="geojson")
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "list":
        for record in all_records():
            revisions = [revision for revision in record["revisions"] if args.status is None or revision["status"] == args.status]
            for revision in revisions:
                print(f"{record['osm_id']}\trevision {revision['revision']}\t{revision['status']}\t{revision['source']}\t{revision.get('note', '')}")
        return
    corrections = accepted_corrections()
    if args.format == "geojson":
        args.output.write_text(json.dumps(as_geojson(corrections), indent=2) + "\n", encoding="utf-8")
    else:
        args.output.write_bytes(as_osm_xml(corrections))
    print(f"Exported {len(corrections)} accepted corrections to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
