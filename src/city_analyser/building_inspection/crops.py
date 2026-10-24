"""Export ranked perspective crops of buildings, with provenance, for attribute work.

A crop is the panorama reprojected to a pinhole view centered on the building's
instance mask, with the field of view fitted to the mask bounds (OpenFACADES
fits it to a detector box instead). Each crop is written beside a JSON sidecar
naming the panorama, OSM building, view parameters, evidence, quality facts
and hashes, so that a label or prediction can always be traced back. Output is
deterministic for the same inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ..analysis_cache import atomic_write_bytes
from ..building_analysis import ANALYSIS_VERSION, panorama_analysis_files
from ..panorama_geometry import perspective_tile
from .evidence import BuildingEvidenceNotFound, collect_building_evidence
from .models import RANKING_FORMULA_VERSION, RankedView
from .ranking import rank_views
from .reporting import BUILDING_ID, _building_parts

CROP_SCHEMA_VERSION = 1
# InternVL-style vision encoders tile at 448 px.
DEFAULT_CROP_SIZE = 448
DEFAULT_CROPS_PER_BUILDING = 3
OTHER_BUILDING_FILL = 127


def crops_directory(dataset_dir: Path, building_id: str) -> Path:
    return dataset_dir / "crops" / Path(*_building_parts(building_id))


# Next to a building's crops: the views left out because their mask was flagged, with the reasons.
SET_ASIDE_FILE = "set-aside.json"


def set_aside_views(dataset_dir: Path, building_id: str) -> list[dict[str, Any]]:
    """``[{"image_id", "rank", "reasons"}]`` of the views a crop export left out, or ``[]``."""
    path = crops_directory(dataset_dir, building_id) / SET_ASIDE_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [view for view in payload.get("views", []) if isinstance(view, dict)]


def analyzed_building_ids(dataset_dir: Path) -> list[str]:
    """OSM buildings named by a current analysis of this dataset."""
    found: set[str] = set()
    for path in panorama_analysis_files(dataset_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("version") != ANALYSIS_VERSION or not isinstance(payload.get("building_ids"), dict):
            continue
        found.update(
            item["osm_id"] for item in payload["building_ids"].values()
            if isinstance(item, dict) and isinstance(item.get("osm_id"), str) and BUILDING_ID.fullmatch(item["osm_id"])
        )
    return sorted(found)


def _render(panorama: Any, id_mask: Any, view: RankedView, size: int, mask_others: bool) -> bytes:
    np = __import__("numpy")
    Image = __import__("PIL.Image", fromlist=["Image"])
    import io
    import math

    yaw, pitch, fov = math.radians(view.evidence.yaw_deg), math.radians(view.evidence.pitch_deg), view.evidence.fov_deg
    tile, _, _ = perspective_tile(panorama, yaw, pitch, size, fov)
    if mask_others:
        labels, _, _ = perspective_tile(id_mask, yaw, pitch, size, fov)
        other = (labels > 0) & (labels != view.evidence.mask_ordinal)
        tile = tile.copy()
        tile[other] = OTHER_BUILDING_FILL
    output = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(tile)).save(output, format="JPEG", quality=92, optimize=True)
    return output.getvalue()


def _sidecar(dataset_id: str, building_id: str, view: RankedView, file_name: str, digest: str, size: int, mask_others: bool) -> dict[str, Any]:
    evidence = view.evidence
    return {
        "schema_version": CROP_SCHEMA_VERSION,
        "crop": {"file": file_name, "sha256": digest, "size": size, "mask_other_buildings": mask_others},
        "building": {"id": building_id, "identity_source": "openstreetmap", "osm_version": evidence.osm_version,
                     "footprint_correction": evidence.footprint_correction},
        "source": {
            "dataset": dataset_id, "image_id": evidence.image_id, "source_page": evidence.source_page,
            "captured_at_ms": evidence.captured_at_ms, "panorama_sha256": evidence.panorama_sha256,
            "id_mask_sha256": evidence.id_mask_sha256, "analysis_sha256": evidence.analysis_sha256,
            "attribution": "Mapillary contributors, CC BY-SA 4.0; see the dataset's ATTRIBUTION.md",
        },
        "view": {"yaw_deg": evidence.yaw_deg, "pitch_deg": evidence.pitch_deg, "fov_deg": evidence.fov_deg, "framing_basis": "observed_mask_bounds"},
        "evidence": {
            "rank": view.rank, "score": view.score, "ranking_formula_version": RANKING_FORMULA_VERSION,
            "observed_fraction": view.observed_fraction, "inferred_fraction": view.inferred_fraction,
            "projection_coverage": evidence.projection_coverage, "visible_facades": evidence.visible_facades,
        },
        "image_quality": evidence.image_quality,
    }


def export_building_crops(
    data_root: Path,
    dataset_id: str,
    building_ids: Iterable[str] | None = None,
    *,
    per_building: int = DEFAULT_CROPS_PER_BUILDING,
    size: int = DEFAULT_CROP_SIZE,
    mask_others: bool = False,
    skip_flagged: bool = False,
    keep_flagged_masks: bool = False,
    clean_views: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Write the top ``per_building`` crops of each building; returns counts and the buildings without evidence.

    Views whose mask the analysis flagged (``mask_quality``) are left out and listed in ``set-aside.json``
    with their reasons, unless ``keep_flagged_masks``; ``clean_views`` (``{building: {image ids}}``, e.g. a
    reviewer's curation) are kept whatever the flag says. A building with only flagged views gets no crop.
    """
    from ..local_viewer import dataset_manifest

    np = __import__("numpy")
    Image = __import__("PIL.Image", fromlist=["Image"])
    dataset_dir = (data_root / dataset_id).resolve()
    manifest = dataset_manifest(data_root, dataset_id)
    wanted = analyzed_building_ids(dataset_dir) if building_ids is None else sorted(set(building_ids))
    tasks: list[tuple[str, RankedView]] = []
    missing: list[str] = []
    no_clean_view: list[str] = []
    set_aside: dict[str, list[dict[str, Any]]] = {}
    for building_id in wanted:
        try:
            collection = collect_building_evidence(dataset_dir, manifest, building_id, expected_analysis_version=ANALYSIS_VERSION)
        except BuildingEvidenceNotFound:
            missing.append(building_id)
            continue
        views = [view for view in rank_views(collection.views, dataset_id, building_id) if not (skip_flagged and (view.evidence.image_quality or {}).get("flags"))]
        if not views:
            missing.append(building_id)
            continue
        if not keep_flagged_masks:
            vouched = (clean_views or {}).get(building_id, set())
            flagged = [view for view in views if view.evidence.mask_flagged and view.evidence.image_id not in vouched]
            set_aside[building_id] = [
                {"image_id": view.evidence.image_id, "rank": view.rank, "reasons": list(view.evidence.mask_quality.get("reasons", []))}
                for view in flagged
            ]
            views = [view for view in views if view not in flagged]
            if not views:
                no_clean_view.append(building_id)
                continue
        tasks.extend((building_id, view) for view in views[:per_building])
    # One panorama decode serves every crop taken from it.
    tasks.sort(key=lambda item: (item[1].evidence.image_id, item[0]))
    written = 0
    kept: dict[str, set[str]] = {building_id: set() for building_id in wanted}
    loaded_id, panorama, id_mask = None, None, None
    for building_id, view in tasks:
        if view.evidence.image_id != loaded_id:
            with Image.open(dataset_dir / view.evidence.local_file) as source:
                panorama = np.asarray(source.convert("RGB"))
            with Image.open(dataset_dir / view.evidence.id_mask_relative_path) as source:
                id_mask = np.asarray(source)
            loaded_id = view.evidence.image_id
        content = _render(panorama, id_mask, view, size, mask_others)
        directory = crops_directory(dataset_dir, building_id)
        name = f"crop-{view.rank:02d}-{view.evidence.image_id}.jpg"
        atomic_write_bytes(directory / name, content)
        sidecar = _sidecar(dataset_id, building_id, view, name, hashlib.sha256(content).hexdigest(), size, mask_others)
        atomic_write_bytes((directory / name).with_suffix(".json"), (json.dumps(sidecar, indent=2) + "\n").encode("utf-8"))
        kept[building_id].update((name, Path(name).with_suffix(".json").name))
        written += 1
    # Crops of an earlier ranking (another rank, a panorama that lost the
    # building) would otherwise be read as current evidence.
    for building_id, names in kept.items():
        directory = crops_directory(dataset_dir, building_id)
        for stale in (*directory.glob("crop-*.jpg"), *directory.glob("crop-*.json")):
            if stale.name not in names:
                stale.unlink(missing_ok=True)
        if set_aside.get(building_id):
            document = {"note": "Views left out of the crops because their mask disagrees with the projected walls (mask_quality).",
                        "views": set_aside[building_id]}
            atomic_write_bytes(directory / SET_ASIDE_FILE, (json.dumps(document, indent=2) + "\n").encode("utf-8"))
        else:
            (directory / SET_ASIDE_FILE).unlink(missing_ok=True)
    return {"buildings": len(wanted) - len(missing) - len(no_clean_view), "crops": written, "buildings_without_evidence": missing,
            "buildings_without_clean_view": no_clean_view}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ranked perspective crops of analyzed buildings with provenance sidecars.")
    parser.add_argument("dataset", help="dataset directory name under --data-root")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--building", action="append", help="OSM ID (way/123); repeat for several. Default: every analyzed building")
    parser.add_argument("--per-building", type=int, default=DEFAULT_CROPS_PER_BUILDING)
    parser.add_argument("--size", type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument("--mask-others", action="store_true", help="paint pixels of other buildings gray, to isolate the target")
    parser.add_argument("--skip-flagged", action="store_true", help="do not export views flagged blurry, dark or overexposed")
    parser.add_argument("--keep-flagged-masks", action="store_true", help="export views whose mask was flagged too (by default they are set aside)")
    args = parser.parse_args()
    try:
        result = export_building_crops(
            args.data_root, args.dataset, args.building, per_building=args.per_building, size=args.size,
            mask_others=args.mask_others, skip_flagged=args.skip_flagged, keep_flagged_masks=args.keep_flagged_masks,
        )
    except ValueError as error:
        parser.error(str(error))
    print(f"{result['crops']} crops for {result['buildings']} buildings under {args.data_root / args.dataset / 'crops'}")
    if result["buildings_without_evidence"]:
        print(f"{len(result['buildings_without_evidence'])} buildings had no usable view")
    if result["buildings_without_clean_view"]:
        print(f"{len(result['buildings_without_clean_view'])} buildings had only flagged masks (see set-aside.json in their crops folder)")


if __name__ == "__main__":
    main()
