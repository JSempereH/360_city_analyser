"""Reviewed, versioned local corrections of OpenStreetMap footprint geometry.

OpenStreetMap stays the footprint source and the building identity. A
correction replaces the geometry of one OSM element, but only once a person has
accepted it and only while OSM still serves the element version it was drawn
on. When OSM changes the element, the correction is reported as stale and the
OSM geometry is used until someone reviews it again. Nothing is ever deleted:
each element keeps its full revision history in
``data/footprint-corrections/<type>-<id>.json`` (or under
``BUILDING_ANALYSIS_FOOTPRINT_CORRECTIONS_DIR``).
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .analysis_cache import atomic_write_json, canonical_json_sha256

OSM_ELEMENT_ID = re.compile(r"^(node|way|relation)/[1-9][0-9]*$")
# Buildings in analyses and reports: OSM elements, and the parts a reviewed
# split creates from one ("way/123/part-2").
BUILDING_ID = re.compile(r"^(node|way|relation)/[1-9][0-9]*(?:/part-[1-9][0-9]*)?$")
MAXIMUM_PARTS = 20
# Parts may touch along a party wall but must not cover each other.
MAXIMUM_PART_OVERLAP = 0.01
SOURCES = {"manual", "pipeline", "cadastre"}
DECISIONS = {"accepted", "rejected"}
# A correction may move a footprint, not attach it to a different building.
MAXIMUM_CENTROID_SHIFT_M = 50.0
EARTH_RADIUS_M = 6_371_008.8

_write_lock = threading.Lock()
_record_cache: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


class CorrectionError(ValueError):
    """A correction request that cannot be applied as asked."""


def corrections_directory() -> Path:
    configured = os.environ.get("BUILDING_ANALYSIS_FOOTPRINT_CORRECTIONS_DIR", "").strip()
    return Path(configured) if configured else Path("data") / "footprint-corrections"


def _record_path(osm_id: str) -> Path:
    if not OSM_ELEMENT_ID.fullmatch(osm_id):
        raise CorrectionError(f"invalid OSM element id: {osm_id}")
    return corrections_directory() / f"{osm_id.replace('/', '-')}.json"


def _read_record(path: Path) -> dict[str, Any] | None:
    try:
        status = path.stat()
    except OSError:
        return None
    signature = (status.st_mtime_ns, status.st_size)
    cached = _record_cache.get(str(path))
    if cached is not None and cached[0] == signature:
        return cached[1]
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or not isinstance(record.get("revisions"), list):
        return None
    _record_cache[str(path)] = (signature, record)
    return record


def load_record(osm_id: str) -> dict[str, Any] | None:
    """Full revision history of one element, or ``None`` when it has none."""
    return _read_record(_record_path(osm_id))


def all_records() -> list[dict[str, Any]]:
    directory = corrections_directory()
    if not directory.is_dir():
        return []
    records = (_read_record(path) for path in sorted(directory.glob("*.json")))
    return [record for record in records if record is not None]


def active_revision(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """The accepted revision in force, if any (a later acceptance supersedes earlier ones)."""
    if record is None:
        return None
    accepted = [revision for revision in record["revisions"] if revision.get("status") == "accepted"]
    return accepted[-1] if accepted else None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _validated_geometry(geometry: object) -> dict[str, Any]:
    if not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        raise CorrectionError("a footprint correction must be a GeoJSON Polygon or MultiPolygon")
    try:
        shape = __import__("shapely.geometry", fromlist=["shape"]).shape(geometry)
    except (TypeError, ValueError, KeyError, IndexError, AttributeError) as error:
        raise CorrectionError(f"invalid footprint geometry: {error}") from error
    if shape.is_empty or not shape.is_valid or shape.area <= 0:
        raise CorrectionError("the footprint geometry must be a valid, non-empty polygon")
    west, south, east, north = shape.bounds
    if not all(math.isfinite(value) for value in shape.bounds) or west < -180 or east > 180 or south < -90 or north > 90:
        raise CorrectionError("footprint coordinates must be longitude/latitude degrees")
    return json.loads(json.dumps(geometry))


def _centroid_distance_m(first: dict[str, Any], second: dict[str, Any]) -> float:
    shape = __import__("shapely.geometry", fromlist=["shape"]).shape
    a, b = shape(first).centroid, shape(second).centroid
    mean_latitude = math.radians((a.y + b.y) / 2)
    east = math.radians(b.x - a.x) * math.cos(mean_latitude) * EARTH_RADIUS_M
    north = math.radians(b.y - a.y) * EARTH_RADIUS_M
    return math.hypot(east, north)


def _validated_parts(base_geometry: dict[str, Any], parts: object) -> list[dict[str, Any]]:
    if not isinstance(parts, list) or not 2 <= len(parts) <= MAXIMUM_PARTS:
        raise CorrectionError(f"a split needs between 2 and {MAXIMUM_PARTS} parts")
    validated = [_validated_geometry(part) for part in parts]
    shape = __import__("shapely.geometry", fromlist=["shape"]).shape
    shapes = [shape(part) for part in validated]
    for index, first in enumerate(shapes):
        if _centroid_distance_m(base_geometry, validated[index]) > MAXIMUM_CENTROID_SHIFT_M:
            raise CorrectionError(f"part {index + 1} lies more than {MAXIMUM_CENTROID_SHIFT_M:.0f} m from the OSM element")
        for other_index, second in enumerate(shapes[index + 1:], start=index + 1):
            if first.intersection(second).area > MAXIMUM_PART_OVERLAP * min(first.area, second.area):
                raise CorrectionError(f"parts {index + 1} and {other_index + 1} overlap")
    return validated


def propose(
    base_feature: dict[str, Any],
    geometry: object,
    source: str,
    author: str,
    note: str,
    evidence: list[dict[str, Any]] | None = None,
    parts: object = None,
) -> dict[str, Any]:
    """Record a proposed geometry, or a split into ``parts``, for the OSM element ``base_feature`` describes.

    ``base_feature`` must be the element as OSM currently serves it, so the
    correction is tied to that exact version and geometry. A split replaces
    the element with buildings ``<element>/part-1`` ... ``<element>/part-N``.
    """
    properties = base_feature.get("properties", {})
    osm_id = str(properties.get("osm_id", ""))
    if properties.get("footprint_source") != "osm":
        raise CorrectionError("corrections apply to OpenStreetMap footprints only")
    if source not in SOURCES:
        raise CorrectionError(f"source must be one of {sorted(SOURCES)}")
    if not author.strip() or not note.strip():
        raise CorrectionError("a correction needs an author and a note explaining it")
    if parts is not None:
        corrected, validated_parts, kind = None, _validated_parts(base_feature["geometry"], parts), "split"
    else:
        corrected, validated_parts, kind = _validated_geometry(geometry), None, "geometry"
        if _centroid_distance_m(base_feature["geometry"], corrected) > MAXIMUM_CENTROID_SHIFT_M:
            raise CorrectionError(f"the corrected footprint moved more than {MAXIMUM_CENTROID_SHIFT_M:.0f} m from the OSM element")
    with _write_lock:
        path = _record_path(osm_id)
        record = _read_record(path) or {"osm_id": osm_id, "revisions": []}
        revision = {
            "revision": len(record["revisions"]) + 1,
            "kind": kind,
            "geometry": corrected,
            **({"parts": validated_parts} if validated_parts else {}),
            "base_version": properties.get("osm_version"),
            "base_timestamp": properties.get("osm_timestamp"),
            "base_geometry_sha256": canonical_json_sha256(base_feature["geometry"]),
            "source": source,
            "status": "proposed",
            "author": author.strip(),
            "note": note.strip(),
            "created_at": _now(),
            "evidence": evidence or [],
        }
        record = {**record, "revisions": [*record["revisions"], revision]}
        atomic_write_json(path, record)
        _record_cache.pop(str(path), None)
    return revision


def review(osm_id: str, revision_number: int, decision: str, reviewer: str, note: str = "") -> dict[str, Any]:
    """Accept or reject a proposed revision; accepting supersedes the previous accepted one."""
    if decision not in DECISIONS:
        raise CorrectionError(f"decision must be one of {sorted(DECISIONS)}")
    if not reviewer.strip():
        raise CorrectionError("a review needs a reviewer")
    with _write_lock:
        path = _record_path(osm_id)
        record = _read_record(path)
        if record is None:
            raise CorrectionError(f"{osm_id} has no footprint corrections")
        revisions = [dict(revision) for revision in record["revisions"]]
        matches = [revision for revision in revisions if revision.get("revision") == revision_number]
        if not matches:
            raise CorrectionError(f"{osm_id} has no revision {revision_number}")
        target = matches[0]
        if target.get("status") != "proposed":
            raise CorrectionError(f"revision {revision_number} is already {target.get('status')}")
        if decision == "accepted":
            for revision in revisions:
                if revision.get("status") == "accepted":
                    revision["status"] = "superseded"
        target.update({"status": decision, "reviewer": reviewer.strip(), "review_note": note.strip(), "reviewed_at": _now()})
        record = {**record, "revisions": revisions}
        atomic_write_json(path, record)
        _record_cache.pop(str(path), None)
    return target


def revert(osm_id: str, reviewer: str, note: str = "") -> dict[str, Any]:
    """Withdraw the accepted correction in force, so OpenStreetMap's geometry applies again.

    The revision stays in the history as ``reverted``, with who did it and why.
    """
    if not reviewer.strip():
        raise CorrectionError("reverting a correction needs a reviewer")
    with _write_lock:
        path = _record_path(osm_id)
        record = _read_record(path)
        active = active_revision(record)
        if record is None or active is None:
            raise CorrectionError(f"{osm_id} has no accepted correction to revert")
        revisions = [dict(revision) for revision in record["revisions"]]
        target = next(revision for revision in revisions if revision.get("revision") == active["revision"])
        target.update({"status": "reverted", "reverted_by": reviewer.strip(), "revert_note": note.strip(), "reverted_at": _now()})
        atomic_write_json(path, {**record, "revisions": revisions})
        _record_cache.pop(str(path), None)
    return target


def correction_state(feature: dict[str, Any], record: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None]:
    """``("applied" | "stale" | "none", revision)`` for an OSM feature as served now."""
    revision = active_revision(record)
    if revision is None:
        return "none", None
    current_version = feature.get("properties", {}).get("osm_version")
    # Without a version on either side the element cannot be shown unchanged.
    if current_version is None or revision.get("base_version") != current_version:
        return "stale", revision
    return "applied", revision


def apply_corrections(features: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """Copies of ``features`` with accepted, current corrections applied.

    Input features (shared by the per-cell cache) are never modified.
    """
    corrected = []
    for feature in features:
        osm_id = str(feature.get("properties", {}).get("osm_id", ""))
        record = load_record(osm_id) if OSM_ELEMENT_ID.fullmatch(osm_id) and corrections_directory().is_dir() else None
        state, revision = correction_state(feature, record)
        if state == "none" or revision is None:
            corrected.append(feature)
            continue
        summary = {
            "revision": revision["revision"], "source": revision["source"], "base_version": revision.get("base_version"),
            "reviewed_at": revision.get("reviewed_at"), "kind": revision.get("kind", "geometry"),
        }
        if state == "applied" and revision.get("kind") == "split":
            parts = revision["parts"]
            for number, part in enumerate(parts, start=1):
                corrected.append({
                    **feature,
                    "properties": {
                        **feature["properties"], "osm_id": f"{osm_id}/part-{number}", "parent_osm_id": osm_id,
                        "footprint_correction": {**summary, "part": number, "parts": len(parts)},
                    },
                    "geometry": part,
                })
        elif state == "applied":
            corrected.append({
                **feature,
                "properties": {**feature["properties"], "footprint_correction": summary},
                "geometry": revision["geometry"],
            })
        else:
            corrected.append({**feature, "properties": {**feature["properties"], "footprint_correction_stale": summary}})
    return corrected


def footprints_sha256(features: list[dict[str, Any]]) -> str:
    """Identity of the exact footprints an analysis used (OSM versions and corrections)."""
    entries = [
        [
            str(feature["properties"].get("osm_id")),
            feature["properties"].get("osm_version"),
            (feature["properties"].get("footprint_correction") or {}).get("revision"),
            canonical_json_sha256(feature["geometry"]),
        ]
        for feature in features
    ]
    return canonical_json_sha256(sorted(entries, key=lambda entry: entry[0]))
