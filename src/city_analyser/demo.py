"""city-analyser-demo: a presentation copy of an analyzed dataset, ready for the viewer's presentation mode.

The copy (``<dataset>-demo``) shares the panoramas (hard links, not duplicates) and
copies the analyses, so the real dataset is never touched. It then exports the
ranked crops, writes the OSM-tag prior and, for every building without a real
vision-language answer, writes a *simulated* one (provider
``vlm:simulated-demo``): storeys from the OSM tag or the analyzed height,
structure from age and use, with per-view noise and a few deliberately split
votes so the presentation also shows abstentions. Simulated answers go through
the same provider, fusion and card code as real ones, and the presentation
labels them as simulated. A dataset surveyed with a real model is replayed as is.

    uv run city-analyser-demo valencia-centro
    uv run city-analyser-viewer      # then open /viewer/?dataset=valencia-centro-demo&present=1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

from .attributes.cli import load_crops, osm_tags, predict_prior, read_attributes, write_attributes
from .attributes.vlm import VLMProvider
from .building_analysis import ANALYSIS_VERSION, panorama_analysis_files
from .building_inspection.crops import analyzed_building_ids, export_building_crops

SIMULATED_PROVIDER = "vlm:simulated-demo"

# Observation templates by structure; {floors} is filled in. Several per class, so views differ.
_OBSERVATIONS = {
    ("MUR", "LWAL"): [
        "Thick load-bearing walls with small, vertically aligned openings and deep window reveals; {floors} rows of windows and no visible columns or slab edges.",
        "Openings stack exactly above each other and are narrow relative to the wall piers, typical of bearing masonry; balconies on stone corbels; {floors} storeys.",
        "Solid ground floor with arched doorways and a heavy cornice; wall thickness at the reveals suggests unreinforced masonry; {floors} window rows.",
    ],
    ("CR", "LFINF"): [
        "Regular grid of openings with horizontal slab edges at every storey; brick infill between columns; {floors} storeys above an open ground floor with shop fronts.",
        "Continuous balconies project from the slabs and the ground floor is glazed between columns, a reinforced concrete frame with masonry infill; {floors} storeys.",
        "Uniform bays and thin piers between wide windows, slab lines visible in the render; the infill does not carry load; {floors} window rows.",
    ],
    ("CR", "LFM"): [
        "Wide openings and a fully glazed ground floor; columns and beams carry the facade, which is a light curtain; {floors} storeys.",
    ],
    ("CR", "LDUAL"): [
        "Concrete frame with solid wall panels at the corners and around the stair core visible at the side; {floors} storeys of regular openings.",
    ],
    ("S", "LFM"): [
        "Long spans, metal cladding and a regular rhythm of slender columns; no masonry visible; {floors} levels.",
    ],
}
_MONUMENT_OBSERVATIONS = [
    "A tower or monumental building: tall, nearly blind stone walls with few, irregular openings; the storeys cannot be counted from the facade.",
    "Massive ashlar masonry with buttresses and small openings at irregular heights; there are no window rows to count storeys from.",
]
# Tall buildings whose height is not storeys: towers, churches, monuments.
_ORDINARY_USES = {"apartments", "residential", "house", "office", "commercial", "retail", "hotel"}
MONUMENT_HEIGHT_M = 25.0
_FACADE_BY_STRUCTURE = {("MUR", "LWAL"): ("plaster", "stone"), ("CR", "LFINF"): ("plaster", "brick"), ("CR", "LFM"): ("glass", "concrete"),
                        ("CR", "LDUAL"): ("concrete", "plaster"), ("S", "LFM"): ("metal", "glass")}
_ALTERNATIVE = {("MUR", "LWAL"): ("CR", "LFINF"), ("CR", "LFINF"): ("MUR", "LWAL"), ("CR", "LFM"): ("CR", "LFINF"),
                ("CR", "LDUAL"): ("CR", "LFINF"), ("S", "LFM"): ("CR", "LFM")}


def _seed(building_id: str) -> random.Random:
    return random.Random(int(hashlib.sha256(building_id.encode("utf-8")).hexdigest()[:12], 16))


def _structure(rng: random.Random, year: int | None, use: str | None) -> tuple[str, str]:
    if use in {"industrial", "garage"}:
        return ("S", "LFM")
    if use == "religious" or (year is not None and year < 1940):
        return ("MUR", "LWAL")
    if year is not None and year >= 1990:
        return rng.choice([("CR", "LFINF"), ("CR", "LDUAL"), ("CR", "LFM")])
    if year is not None:
        return ("CR", "LFINF")
    return rng.choices([("MUR", "LWAL"), ("CR", "LFINF"), ("CR", "LDUAL")], weights=[45, 45, 10])[0]


def simulated_answers(building_id: str, views: int, tags: dict[str, Any], height_m: float | None) -> list[dict[str, Any]]:
    """One plausible answer per view, deterministic for a building: what a model might say, not a measurement."""
    rng = _seed(building_id)
    building = str(tags.get("building", "")).lower()
    try:
        floors: int | None = int(round(float(str(tags.get("building:levels", "")).replace(",", "."))))
    except ValueError:
        floors = None
    if floors is None and height_m and height_m >= MONUMENT_HEIGHT_M and building not in _ORDINARY_USES:
        # A tower's height says nothing about storeys: the model abstains, as a careful one would.
        return [{
            "observations": _MONUMENT_OBSERVATIONS[(index + rng.randint(0, 9)) % len(_MONUMENT_OBSERVATIONS)], "floors": None,
            "gem_material": "MUR", "alternate_gem_material": None, "gem_llrs": "LWAL", "alternate_gem_llrs": None,
            "building_type": "religious" if building in {"church", "cathedral", "yes", ""} else "public", "surface_material": "stone",
            "construction_year": None,
        } for index in range(views)]
    if floors is None:
        floors = int(round(height_m / 3.1)) if height_m else rng.randint(3, 7)
    floors = max(1, min(floors, 12))
    year = None
    start = str(tags.get("start_date") or "")[:4]
    if start.isdigit():
        year = int(start)
    use = {"apartments": "apartments", "residential": "apartments", "house": "house", "retail": "retail", "commercial": "office",
           "office": "office", "church": "religious", "cathedral": "religious", "industrial": "industrial", "school": "education",
           "hotel": "hotel", "public": "public"}.get(str(tags.get("building", "")).lower())
    material, llrs = _structure(rng, year, use)
    if year is None:
        year = {("MUR", "LWAL"): rng.randint(1870, 1935)}.get((material, llrs), rng.randint(1955, 1985))
    # About one building in six shows views that disagree, so the presentation also shows a flag for review.
    split = rng.random() < 0.17 and views >= 2
    answers = []
    for index in range(views):
        structure = (material, llrs)
        if (split and index % 2 == 1) or (not split and rng.random() < 0.12):
            structure = _ALTERNATIVE[structure]
        view_floors = floors + (rng.choice([-1, 1]) if rng.random() < 0.18 else 0)
        templates = _OBSERVATIONS.get(structure, _OBSERVATIONS[("CR", "LFINF")])
        facade = _FACADE_BY_STRUCTURE.get(structure, ("plaster", "brick"))
        alternative = _ALTERNATIVE.get(structure, structure)
        answers.append({
            "observations": templates[(index + rng.randint(0, 9)) % len(templates)].format(floors=max(1, view_floors)),
            "floors": max(1, view_floors),
            "gem_material": structure[0], "alternate_gem_material": alternative[0] if alternative[0] != structure[0] else None,
            "gem_llrs": structure[1], "alternate_gem_llrs": alternative[1] if alternative[1] != structure[1] else None,
            "building_type": use or "apartments", "surface_material": facade[0], "alternate_surface_material": facade[1],
            "construction_year": year + rng.randint(-6, 6),
        })
    return answers


def _height(dataset_dir: Path, building_id: str) -> float | None:
    for path in panorama_analysis_files(dataset_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for record in (payload.get("building_ids") or {}).values():
            if isinstance(record, dict) and record.get("osm_id") == building_id and isinstance(record.get("height_m"), (int, float)):
                return float(record["height_m"])
    return None


def copy_dataset(data_root: Path, source: str, target: str) -> Path:
    """The demo copy: panoramas hard-linked (copied across disks), analyses and manifest copied, nothing of the source modified."""
    source_dir, target_dir = (data_root / source).resolve(), data_root / target
    if not (source_dir / "manifest.json").is_file():
        raise ValueError(f"no dataset at {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "manifest.geojson", "ATTRIBUTION.md", "building-index.json", "batch-ledger.jsonl"):
        if (source_dir / name).is_file():
            shutil.copy2(source_dir / name, target_dir / name)
    # Hard links, not a symlinked folder: analyses refuse images that resolve outside their dataset.
    images = target_dir / "images"
    images.mkdir(exist_ok=True)
    for source_image in sorted((source_dir / "images").glob("*")):
        linked = images / source_image.name
        if source_image.is_file() and not linked.exists():
            try:
                linked.hardlink_to(source_image)
            except OSError:
                shutil.copy2(source_image, linked)
    if (source_dir / "analysis").is_dir():
        shutil.copytree(source_dir / "analysis", target_dir / "analysis", dirs_exist_ok=True)
    return target_dir


CURATION_FILE = "curation.json"
# Ranking score below which a view shows too little of the facade to judge storeys or structure.
SLIVER_SCORE = 0.03
SLIVER_OBSERVATION = ("Only a narrow sliver of this facade is visible from the captured positions, partly hidden by the buildings "
                      "in front; storeys and structure cannot be judged from it.")
CURATED_PROVIDER = "vlm:curated-demo"


def _column_bearings(width: int, heading: float) -> Any:
    import numpy as np

    return (heading + ((np.arange(width) + 0.5) / width - 0.5) * 360.0) % 360.0


def _in_range(bearings: Any, start: float, end: float) -> Any:
    """Columns whose bearing lies in [start, end] going clockwise (so 352 to 40 crosses north)."""
    return ((bearings - start) % 360.0) <= ((end - start) % 360.0)


def small_views(dataset_dir: Path, min_pixels: int) -> list[dict[str, Any]]:
    """Removal edits for every building view smaller than ``min_pixels``: slivers a presentation should not show."""
    edits = []
    for path in panorama_analysis_files(dataset_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        edits.extend(
            {"image": str(payload["image_id"]), "building": record["osm_id"], "remove": True}
            for record in (payload.get("building_ids") or {}).values() if record.get("pixels", 0) < min_pixels
        )
    return edits


def _record_from_osm(dataset_dir: Path, image_id: str, building_id: str) -> dict[str, Any]:
    """An analysis building record for an OSM footprint near the panorama, as the association writes them."""
    from .building_analysis import nearby_buildings
    from .multi_view_analysis import panorama_coordinates

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    image = next(item for item in manifest["images"] if str(item.get("id")) == image_id)
    longitude, latitude = panorama_coordinates(image)
    feature = next((item for item in nearby_buildings(latitude, longitude) if item["properties"].get("osm_id") == building_id), None)
    if feature is None:
        raise ValueError(f"{building_id} is not an OpenStreetMap footprint near {image_id}")
    tags = feature["properties"]
    try:
        height = float(str(tags.get("height", "")).replace(",", ".").replace("m", "").strip())
        source = "height"
    except ValueError:
        height, source = 10.0, "default"
    return {
        "osm_id": building_id, "footprint_source": "osm", "alternative_footprint_id": None,
        "osm_version": tags.get("osm_version"), "osm_timestamp": tags.get("osm_timestamp"), "footprint_correction": None,
        "facades": 1, "pixels": 0, "inferred_pixels": 0, "confidence": 1.0, "height_m": height, "height_source": source,
        "render_height_m": height, "min_height_m": 0.0, "sfm_depth_samples": 0, "sfm_depth_conflicts": 0, "visual_instances": [],
        "footprint_geometry": feature["geometry"], "curated": True,
    }


def apply_mask_curation(dataset_dir: Path, edits: list[dict[str, Any]]) -> list[str]:
    """Reviewed corrections of building masks in a demo copy; returns what was done, one line per edit.

    Each edit names an ``image`` and a ``building`` and one action, in compass
    bearings (degrees) as tools/mask_review.py --full shows them:
    ``keep_bearings: [a, b]`` drops the building's pixels outside that range
    (with ``max_elevation`` also above that angle, with ``roof_line`` also above
    a traced outline of [bearing, elevation] points, and with ``within_instances``
    also where the instance segmentation saw no building),
    ``claim_bearings: [a, b]`` gives it every pixel the instance segmentation
    saw as a building in that range (with ``add: true`` for a building the analysis
    missed, created from its OSM footprint), ``rename_to`` gives the mask to another
    OSM building, ``fill_bearings: [a, b]`` with a
    ``ground_line`` (and optional ``roof_line``) of [bearing, elevation] points
    makes every pixel between them the building, ``remove: true`` drops it from the
    panorama. The facade (tree-occluded) mask gets the same edit, pixel counts
    are recounted and the artifact hashes rewritten, so the analysis stays valid.
    """
    import hashlib
    import io

    import numpy as np
    from PIL import Image

    from .analysis_cache import atomic_write_bytes

    done = []
    for image_id in sorted({str(edit["image"]) for edit in edits}):
        metadata_path = dataset_dir / "analysis" / f"{image_id}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        masks = {}
        for key, suffix in (("id_mask", "ids"), ("facade_mask", "facades")):
            with Image.open(dataset_dir / "analysis" / f"{image_id}-{suffix}.png") as source:
                masks[key] = np.asarray(source).astype(np.int64)
        instances_path = dataset_dir / "analysis" / f"{image_id}-instances.png"
        instances = None
        if instances_path.is_file():
            with Image.open(instances_path) as source:
                instances = np.asarray(source).astype(np.int64)
        width = masks["id_mask"].shape[1]
        bearings = _column_bearings(width, float(metadata["heading_degrees"]))
        ordinals = {record["osm_id"]: int(ordinal) for ordinal, record in metadata["building_ids"].items()}
        for edit in (edit for edit in edits if str(edit["image"]) == image_id):
            building = edit["building"]
            if edit.get("add") and building not in ordinals:
                # A building the analysis missed (e.g. a landmark the detector did not box): a record from OSM.
                ordinal = max([int(key) for key in metadata["building_ids"]] + [0]) + 1
                metadata["building_ids"][str(ordinal)] = _record_from_osm(dataset_dir, image_id, building)
                ordinals[building] = ordinal
                done.append(f"{image_id} {building}: added from OpenStreetMap")
            if building not in ordinals:
                raise ValueError(f"{building} is not in the analysis of {image_id}")
            ordinal = ordinals[building]
            if "rename_to" in edit:
                # The mask is right but the analysis named the wrong footprint.
                metadata["building_ids"][str(ordinal)] = {**_record_from_osm(dataset_dir, image_id, edit["rename_to"]),
                                                          "pixels": 0, "inferred_pixels": 0}
                ordinals[edit["rename_to"]] = ordinals.pop(building)
                done.append(f"{image_id} {building}: renamed to {edit['rename_to']}")
                continue
            if edit.get("remove"):
                for mask in masks.values():
                    mask[mask == ordinal] = 0
                done.append(f"{image_id} {building}: removed")
            elif "keep_bearings" in edit:
                outside = np.broadcast_to(~_in_range(bearings, *edit["keep_bearings"])[None, :], masks["id_mask"].shape)
                if "max_elevation" in edit:
                    # Above a roof line the "continue upward" fill can reach into the sky.
                    rows = masks["id_mask"].shape[0]
                    elevation = 90.0 - (np.arange(rows) + 0.5) / rows * 180.0
                    outside = outside | (elevation > float(edit["max_elevation"]))[:, None]
                if "roof_line" in edit:
                    # A traced outline, [[bearing, elevation], ...] clockwise: nothing of the building above it.
                    points = sorted(((float(b) - edit["keep_bearings"][0]) % 360.0, float(e)) for b, e in edit["roof_line"])
                    offsets = (bearings - edit["keep_bearings"][0]) % 360.0
                    limit = np.interp(offsets, [b for b, _ in points], [e for _, e in points])
                    rows = masks["id_mask"].shape[0]
                    elevation = 90.0 - (np.arange(rows) + 0.5) / rows * 180.0
                    outside = outside | (elevation[:, None] > limit[None, :])
                if edit.get("within_instances") and instances is not None:
                    # The instance segmentation follows the real outline, where the fill does not.
                    outside = outside | (instances == 0)
                for mask in masks.values():
                    mask[(mask == ordinal) & outside] = 0
                done.append(f"{image_id} {building}: kept bearings {edit['keep_bearings']}" + (f" below {edit['max_elevation']} degrees" if "max_elevation" in edit else "")
                            + (" below the traced roof line" if "roof_line" in edit else "")
                            + (" within detected instances" if edit.get("within_instances") else ""))
            elif edit.get("add") and not any(key in edit for key in ("fill_bearings", "claim_bearings")):
                pass
            elif "fill_bearings" in edit:
                # The facade fills the view (camera close to the wall): every pixel between its edges,
                # above the traced foot of the wall and below an optional roof line, is the building.
                start = edit["fill_bearings"][0]
                offsets = (bearings - start) % 360.0
                rows = masks["id_mask"].shape[0]
                elevation = (90.0 - (np.arange(rows) + 0.5) / rows * 180.0)[:, None]

                def line(points: list[list[float]]) -> Any:
                    ordered = sorted(((float(b) - start) % 360.0, float(e)) for b, e in points)
                    return np.interp(offsets, [b for b, _ in ordered], [e for _, e in ordered])[None, :]

                inside = _in_range(bearings, *edit["fill_bearings"])[None, :] & (elevation >= line(edit["ground_line"]))
                if "roof_line" in edit:
                    inside = inside & (elevation <= line(edit["roof_line"]))
                for mask in masks.values():
                    mask[(mask == ordinal) & ~inside] = 0
                masks["id_mask"][inside] = ordinal
                done.append(f"{image_id} {building}: filled bearings {edit['fill_bearings']} above the traced ground line")
            elif "claim_bearings" in edit:
                inside = _in_range(bearings, *edit["claim_bearings"])[None, :]
                seen = (masks["id_mask"] > 0) | (instances > 0 if instances is not None else False)
                masks["id_mask"][(masks["id_mask"] == ordinal) & ~inside] = 0
                masks["id_mask"][seen & inside] = ordinal
                done.append(f"{image_id} {building}: claimed bearings {edit['claim_bearings']}")
            else:
                raise ValueError(f"unknown mask edit {edit}")
        for key, suffix in (("id_mask", "ids"), ("facade_mask", "facades")):
            buffer = io.BytesIO()
            Image.fromarray(masks[key].astype("uint16")).save(buffer, format="PNG")
            atomic_write_bytes(dataset_dir / "analysis" / f"{image_id}-{suffix}.png", buffer.getvalue())
            metadata["artifact_sha256"][key] = hashlib.sha256(buffer.getvalue()).hexdigest()
        for ordinal, record in list(metadata["building_ids"].items()):
            record["pixels"] = int((masks["id_mask"] == int(ordinal)).sum())
            record["inferred_pixels"] = int((masks["facade_mask"] == int(ordinal)).sum())
            if not record["pixels"] and not record["inferred_pixels"]:
                del metadata["building_ids"][ordinal]
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return done


def prepare_demo(data_root: Path, dataset_id: str, *, per_building: int = 3, force: bool = False, log: Any = print) -> dict[str, Any]:
    from .local_viewer import dataset_manifest

    dataset_dir = (data_root / dataset_id).resolve()
    manifest = dataset_manifest(data_root, dataset_id)
    curation_path = dataset_dir / CURATION_FILE
    curation = json.loads(curation_path.read_text(encoding="utf-8")) if curation_path.is_file() else {}
    for line in apply_mask_curation(dataset_dir, curation.get("masks", [])):
        log(f"  mask: {line}")
    if curation.get("min_view_pixels"):
        removed = apply_mask_curation(dataset_dir, small_views(dataset_dir, int(curation["min_view_pixels"])))
        log(f"  mask: {len(removed)} views under {curation['min_view_pixels']} pixels removed")
    excluded = set(curation.get("exclude", []))
    curated = curation.get("answers", {})
    buildings = analyzed_building_ids(dataset_dir)
    if not buildings:
        raise ValueError(f"{dataset_id} has no current analysis (version {ANALYSIS_VERSION}); run city-analyser-analyze first")
    log(f"{len(buildings)} analyzed buildings; exporting crops...")
    # A view the reviewer edited or answered was checked by eye: it is clean whatever the mask rules say.
    vouched: dict[str, set[str]] = {}
    for edit in curation.get("masks", []):
        if not edit.get("remove"):
            vouched.setdefault(edit.get("rename_to") or edit["building"], set()).add(str(edit["image"]))
    for building_id, views in curated.items():
        vouched.setdefault(building_id, set()).update(str(image_id) for image_id in views)
    export_building_crops(data_root, dataset_id, buildings, per_building=per_building, mask_others=True, clean_views=vouched)
    images_by_id = {str(image["id"]): image for image in manifest["images"] if isinstance(image, dict) and "id" in image}
    counts = {"real": 0, "simulated": 0, "curated": 0, "excluded": 0, "no_crops": 0, "prior": 0}
    for building_id in buildings:
        if predict_prior(dataset_dir, manifest, building_id) == "written":
            counts["prior"] += 1
        for stale in (dataset_dir / "attributes" / building_id).glob("attributes-vlm_*demo.json"):
            stale.unlink()
        if building_id in excluded:
            counts["excluded"] += 1  # a reviewer found its masks unreliable: no model answer, so not presented
            continue
        crops = load_crops(dataset_dir, building_id, images_by_id)
        if not crops:
            counts["no_crops"] += 1
            continue
        if building_id in curated:
            views = curated[building_id]
            missing = [crop.reference["image_id"] for crop in crops if crop.reference["image_id"] not in views and crop.weight >= SLIVER_SCORE]
            if missing:
                raise ValueError(f"{building_id}: no curated answer for the views {missing}")
            # Slivers seen from other positions add no evidence: they abstain.
            answers = iter(json.dumps(views.get(crop.reference["image_id"], {"observations": SLIVER_OBSERVATION})) for crop in crops)
            write_attributes(dataset_dir, VLMProvider(lambda _images, _prompt: next(answers), CURATED_PROVIDER).predict(building_id, crops))
            counts["curated"] += 1
            continue
        real = [path for path in (dataset_dir / "attributes" / building_id).glob("attributes-vlm_*.json") if "simulated" not in path.name]
        if real and not force:
            counts["real"] += 1
            continue
        tags = osm_tags(dataset_dir, manifest, building_id) or {}
        if max(crop.weight for crop in crops) < SLIVER_SCORE:
            # Only a sliver of facade was captured: a careful model says so instead of guessing.
            answers = iter(json.dumps({"observations": SLIVER_OBSERVATION}) for _ in crops)
            write_attributes(dataset_dir, VLMProvider(lambda _images, _prompt: next(answers), SIMULATED_PROVIDER).predict(building_id, crops))
            counts["sliver"] = counts.get("sliver", 0) + 1
            continue
        answers = iter(json.dumps(answer) for answer in simulated_answers(building_id, len(crops), tags, _height(dataset_dir, building_id)))
        result = VLMProvider(lambda _images, _prompt: next(answers), SIMULATED_PROVIDER).predict(building_id, crops)
        write_attributes(dataset_dir, result)
        counts["simulated"] += 1
    stored = read_attributes(dataset_dir, buildings[0], SIMULATED_PROVIDER)
    return {"dataset": dataset_id, "buildings": len(buildings), **counts, "example": stored and stored["building"]["id"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a presentation copy of an analyzed dataset for the viewer's presentation mode (?present=1).")
    parser.add_argument("dataset", help="analyzed dataset under --data-root")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--name", help="name of the copy (default: <dataset>-demo)")
    parser.add_argument("--per-building", type=int, default=3, help="views per building (default: %(default)s)")
    parser.add_argument("--in-place", action="store_true", help="prepare the dataset itself instead of a copy (only for datasets made for demos)")
    parser.add_argument("--force", action="store_true", help="simulate answers even for buildings a real model already described")
    args = parser.parse_args()
    try:
        target = args.dataset if args.in_place else (args.name or f"{args.dataset}-demo")
        if not args.in_place:
            copy_dataset(args.data_root, args.dataset, target)
        result = prepare_demo(args.data_root, target, per_building=args.per_building, force=args.force)
    except ValueError as error:
        parser.error(str(error))
    print(f"{target}: {result['buildings']} buildings, {result['curated']} curated and {result['simulated']} simulated answers, "
          f"{result['real']} real answers kept, {result['excluded']} excluded, {result['no_crops']} without crops. Open http://127.0.0.1:8765/viewer/?dataset={target}&present=1")


if __name__ == "__main__":
    main()
