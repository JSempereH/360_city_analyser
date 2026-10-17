"""city-analyser-survey: walk a downloaded dataset across the city and describe every building it sees.

This is the unattended version of what a person does in the viewer. The
panoramas are split into square tiles that are visited in a serpentine order,
and in each tile the survey:

1. analyzes the tile's panoramas (``batch_analysis``: resumable, retried, ledgered);
2. picks the buildings whose every geometric view (``building_index``) has now
   been visited, so that a building is described once, with all its evidence;
3. exports their best crops, writes the OSM-tag prior and asks a
   vision-language model about the GEM v3 structural system (material and
   lateral load-resisting system), floors, use, facade material and age.

Every step is idempotent. Analyses are in the run ledger, crops are
deterministic, and a model answer is kept while its inputs fingerprint does not
change. ``survey-state.json`` records the tiles and buildings already handled,
so an interrupted survey (Ctrl+C, SIGTERM, a lost Colab session) continues
where it stopped. The viewer reads the results when a building is selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import threading
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .analysis_cache import atomic_write_json
from .attributes.cli import add_vlm_arguments, model_settings_from_args, predict_prior, predict_vlm
from .attributes.taxonomy import gem_taxonomy_string
from .attributes.vlm import VLMProvider
from .batch_analysis import _install_stop_handlers, analyze_dataset
from .building_index import DEFAULT_MAX_DISTANCE_M, build_building_index, index_path, load_building_index, select_panoramas
from .building_inspection.crops import analyzed_building_ids, export_building_crops
from .multi_view_analysis import panorama_coordinates
from .run_ledger import DONE, RunLedger, ledger_path

SURVEY_VERSION = 1
STATE_FILENAME = "survey-state.json"
DEFAULT_TILE_M = 250.0
# Buildings whose crops are exported together: each panorama is decoded once per group.
DESCRIBE_GROUP = 16
METERS_PER_DEGREE_LATITUDE = 110_574.0
METERS_PER_DEGREE_LONGITUDE = 111_320.0


def survey_state_path(data_root: Path, dataset_id: str) -> Path:
    return data_root / dataset_id / STATE_FILENAME


def plan_tiles(images: Iterable[dict[str, Any]], tile_m: float) -> list[dict[str, Any]]:
    """Square tiles of ``tile_m`` meters holding the panoramas, in a serpentine order (row by row, alternating direction).

    Panoramas without coordinates belong to no tile: they cannot be associated with footprints.
    """
    if not tile_m > 0:
        raise ValueError("the tile size must be positive")
    located = [(str(image["id"]), coordinates) for image in images if (coordinates := panorama_coordinates(image)) is not None]
    if not located:
        return []
    west = min(coordinates[0] for _, coordinates in located)
    south = min(coordinates[1] for _, coordinates in located)
    scale = METERS_PER_DEGREE_LONGITUDE * math.cos(math.radians(south))
    cells: dict[tuple[int, int], list[str]] = {}
    for image_id, (longitude, latitude) in located:
        cell = (int((latitude - south) * METERS_PER_DEGREE_LATITUDE // tile_m), int((longitude - west) * scale // tile_m))
        cells.setdefault(cell, []).append(image_id)
    rows = sorted({row for row, _ in cells})
    tiles = []
    for order, row in enumerate(rows):
        columns = sorted((column for cell_row, column in cells if cell_row == row), reverse=order % 2 == 1)
        tiles.extend({"id": f"r{row}c{column}", "images": sorted(cells[(row, column)])} for column in columns)
    return tiles


def _key(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()[:16]


def _load_state(path: Path, settings: dict[str, Any], fresh: bool = False) -> dict[str, Any]:
    state = None
    if not fresh:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    # Other settings describe buildings differently: start over (model answers are still reused by fingerprint).
    if not isinstance(state, dict) or state.get("version") != SURVEY_VERSION or state.get("settings") != settings:
        state = {"version": SURVEY_VERSION, "settings": settings, "tiles": {}, "buildings": {}}
    return state


def _save_state(path: Path, state: dict[str, Any], tiles_total: int) -> None:
    statuses: dict[str, int] = {}
    for record in state["buildings"].values():
        for step in ("prior", "vlm"):
            if step in record:
                statuses[f"{step}_{record[step]}"] = statuses.get(f"{step}_{record[step]}", 0) + 1
    state["summary"] = {
        "tiles_done": len(state["tiles"]), "tiles_total": tiles_total, "buildings_described": len(state["buildings"]),
        "steps": dict(sorted(statuses.items())), "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    atomic_write_json(path, state)


def _describe_line(building_id: str, record: dict[str, Any], result: Any) -> str:
    parts = [building_id]
    if result is not None:
        def top(attribute: str) -> str | None:
            prediction = result.predictions[attribute]
            return None if prediction.abstained or not prediction.scores else prediction.top()[0][0]

        floors = result.predictions["floors"]
        parts.append(f"GEM {gem_taxonomy_string(top('gem_material'), top('gem_llrs'), None if floors.abstained else floors.value)}")
    parts.extend(f"{step} {record[step]}" for step in ("evidence", "prior", "vlm") if step in record and not (step == "vlm" and result is not None))
    return "  " + ", ".join(parts)


def run_survey(
    data_root: Path, model_root: Path, dataset_id: str, *, provider_factory: Callable[[], VLMProvider] | None = None,
    tile_m: float = DEFAULT_TILE_M, per_building: int = 3, max_views_per_building: int | None = None,
    max_tiles: int | None = None, max_buildings: int | None = None, skip_low_quality: bool = False,
    force: bool = False, stop: threading.Event | None = None, log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Survey a dataset; returns counts and the final state (``finished``, ``partial``, ``interrupted`` or ``aborted``).

    ``provider_factory`` builds the vision-language provider the first time a
    building is ready (the model loads after the first tile is analyzed); with
    ``None`` only the OSM-tag prior is written. ``max_tiles`` and
    ``max_buildings`` bound one run for a small validation; a rerun continues.
    """
    from .local_viewer import dataset_manifest

    stop = stop or threading.Event()
    started = time.perf_counter()
    dataset_dir = (data_root / dataset_id).resolve()
    manifest = dataset_manifest(data_root, dataset_id)
    images = [image for image in manifest["images"] if isinstance(image, dict) and image.get("local_file")]
    images_by_id = {str(image["id"]): image for image in images}
    index = load_building_index(data_root, dataset_id, images)
    if index is None:
        log("Building the geometric index (which panoramas see which footprint)...")
        index = build_building_index(data_root, dataset_id, images)
        atomic_write_json(index_path(data_root, dataset_id), index)
    surveyed = set(images_by_id)
    if max_views_per_building is not None:
        surveyed = set(select_panoramas(index, min_aov_deg=0.0, max_views_per_building=max_views_per_building)["images"])
    # The panoramas that should show each building, within the distance the index trusts.
    expected: dict[str, set[str]] = {
        building_id: {view["image"] for view in views if view["distance_m"] <= DEFAULT_MAX_DISTANCE_M} & surveyed
        for building_id, views in index["buildings"].items()
    }
    by_image: dict[str, set[str]] = {}
    for building_id, views in expected.items():
        for image_id in views:
            by_image.setdefault(image_id, set()).add(building_id)
    tiles = plan_tiles((images_by_id[image_id] for image_id in sorted(surveyed)), tile_m)
    settings = {
        "tile_m": tile_m, "per_building": per_building, "max_views_per_building": max_views_per_building,
        "skip_low_quality": skip_low_quality, "provider": "prior+vlm" if provider_factory else "prior",
    }
    state_path = survey_state_path(data_root, dataset_id)
    state = _load_state(state_path, settings, fresh=force)
    provider: list[VLMProvider | None] = [None]
    counts = {"tiles_analyzed": 0, "buildings_described": 0, "model_answers": 0}
    log(f"Survey of {dataset_id}: {len(surveyed)} panoramas in {len(tiles)} tiles of {tile_m:g} m; "
        f"{len(state['tiles'])} tiles and {len(state['buildings'])} buildings already done.")

    def done_views(building_id: str, ledger: RunLedger) -> list[str]:
        return [image_id for image_id in expected.get(building_id, ()) if (ledger.get("analysis", image_id) or {}).get("status") == DONE]

    def describe(building_ids: list[str], ledger: RunLedger) -> bool:
        """Crops, prior and model answer for each building; returns ``False`` when the run must stop."""
        for start in range(0, len(building_ids), DESCRIBE_GROUP):
            group = building_ids[start:start + DESCRIBE_GROUP]
            if max_buildings is not None:
                group = group[: max(0, max_buildings - counts["buildings_described"])]
            if stop.is_set() or not group:
                return False
            if provider_factory is not None and provider[0] is None:
                log("Loading the vision-language model...")
                provider[0] = provider_factory()
            try:
                exported = export_building_crops(data_root, dataset_id, group, per_building=per_building, mask_others=True)
                without_evidence = set(exported["buildings_without_evidence"])
            except (OSError, ValueError) as error:
                log(f"  crops failed for {len(group)} buildings ({type(error).__name__}: {error}); they are retried next run")
                continue
            for building_id in group:
                if stop.is_set():
                    return False
                record: dict[str, Any] = {"views": _key(done_views(building_id, ledger)), "at": datetime.now(UTC).isoformat(timespec="seconds")}
                result = None
                if building_id in without_evidence:
                    record["evidence"] = "none"  # the index expected it, but no analysis matched it
                else:
                    try:
                        record["prior"] = predict_prior(dataset_dir, manifest, building_id)
                    except (OSError, RuntimeError, ValueError) as error:
                        record["prior"] = "error"
                        record["prior_error"] = f"{type(error).__name__}: {error}"[:300]
                    if provider[0] is not None:
                        try:
                            record["vlm"], result = predict_vlm(provider[0], dataset_dir, building_id, images_by_id)
                        except (OSError, ValueError, KeyError) as error:
                            record["vlm"] = "error"
                            record["vlm_error"] = f"{type(error).__name__}: {error}"[:300]
                        counts["model_answers"] += result is not None
                state["buildings"][building_id] = record
                counts["buildings_described"] += 1
                log(_describe_line(building_id, record, result))
            _save_state(state_path, state, len(tiles))
        return max_buildings is None or counts["buildings_described"] < max_buildings

    final_state = "finished"
    for number, tile in enumerate(tiles, start=1):
        if stop.is_set():
            final_state = "interrupted"
            break
        if max_tiles is not None and counts["tiles_analyzed"] >= max_tiles:
            final_state = "partial"
            break
        tile_key = _key(tile["images"])
        if state["tiles"].get(tile["id"]) != tile_key:
            log(f"Tile {number}/{len(tiles)} {tile['id']}: analyzing {len(tile['images'])} panoramas")
            report = analyze_dataset(data_root, model_root, dataset_id, image_ids=tile["images"], skip_low_quality=skip_low_quality, stop=stop)
            counts["tiles_analyzed"] += 1
            if report["state"] != "finished":
                final_state = report["state"]
                if report.get("abort_reason"):
                    log(f"Stopped: {report['abort_reason']}")
                break
            if not report["failed"]:
                # A tile with failures is analyzed again by the next run; done panoramas are skipped in a second.
                state["tiles"][tile["id"]] = tile_key
        ledger = RunLedger(ledger_path(data_root, dataset_id))
        visited = {image_id for other in tiles[:number] for image_id in other["images"]}
        candidates = sorted({building_id for image_id in tile["images"] for building_id in by_image.get(image_id, ())})
        # Ready: every expected view visited, at least one analyzed (a tile whose panoramas all failed
        # describes nothing), and the analyzed views changed since the building was last described.
        ready = [
            building_id for building_id in candidates
            if expected[building_id] <= visited and (views := done_views(building_id, ledger))
            and (state["buildings"].get(building_id) or {}).get("views") != _key(views)
        ]
        if ready:
            log(f"Tile {number}/{len(tiles)} {tile['id']}: describing {len(ready)} buildings")
        if not describe(ready, ledger):
            final_state = "interrupted" if stop.is_set() else "partial"
            break
        _save_state(state_path, state, len(tiles))
    else:
        # Buildings an analysis matched although the geometric index did not expect them there.
        ledger = RunLedger(ledger_path(data_root, dataset_id))
        rest = [building_id for building_id in analyzed_building_ids(dataset_dir) if not expected.get(building_id) and building_id not in state["buildings"]]
        if rest:
            log(f"Describing {len(rest)} buildings outside the geometric index")
            if not describe(rest, ledger):
                final_state = "interrupted" if stop.is_set() else "partial"
    _save_state(state_path, state, len(tiles))
    return {
        "dataset": dataset_id, "state": final_state, "tiles": len(tiles), "tiles_done": len(state["tiles"]),
        "buildings_described_total": len(state["buildings"]), **counts, "wall_s": round(time.perf_counter() - started, 1),
        "state_file": str(state_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk a downloaded dataset tile by tile: analyze the panoramas, then describe each building "
        "(GEM v3 structural material and system, floors, use, facade, age) with the OSM-tag prior and a vision-language model. "
        "Reruns continue where the last run stopped; the viewer shows the results when a building is selected.",
    )
    parser.add_argument("dataset", help="dataset directory name under --data-root")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="datasets directory (default: data)")
    parser.add_argument("--model-root", type=Path, default=Path("models"), help="model cache directory, also for the vision-language model (default: models)")
    parser.add_argument("--tile-size", type=float, default=DEFAULT_TILE_M, help="tile side in meters (default: %(default)s)")
    parser.add_argument("--per-building", type=int, default=3, help="best crops per building shown to the model (default: %(default)s)")
    parser.add_argument("--max-views-per-building", type=int, default=None, help="analyze only the N best panoramas per building (geometric index) to save GPU time")
    parser.add_argument("--max-tiles", type=int, default=None, help="analyze at most this many tiles in this run (small validation runs)")
    parser.add_argument("--max-buildings", type=int, default=None, help="describe at most this many buildings in this run")
    parser.add_argument("--skip-low-quality", action="store_true", help="do not analyze panoramas flagged blurry, dark or overexposed")
    parser.add_argument("--no-vlm", action="store_true", help="write only the OSM-tag prior (no vision-language model)")
    parser.add_argument("--force", action="store_true", help="forget the survey state and visit every tile again (analyses and model answers are still reused when current)")
    add_vlm_arguments(parser)
    args = parser.parse_args()
    try:
        settings = model_settings_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    if not args.no_vlm and settings.backend == "openai" and not settings.url:
        parser.error("the openai backend needs --vlm-url (or ATTRIBUTES_VLM_URL)")

    def vlm_provider() -> VLMProvider:
        return VLMProvider(settings.client(), settings.provider_name, min_views=args.min_views, min_share=args.min_share)

    stop = threading.Event()
    _install_stop_handlers(stop)
    try:
        result = run_survey(
            args.data_root, args.model_root, args.dataset, provider_factory=None if args.no_vlm else vlm_provider, tile_m=args.tile_size,
            per_building=args.per_building, max_views_per_building=args.max_views_per_building, max_tiles=args.max_tiles,
            max_buildings=args.max_buildings, skip_low_quality=args.skip_low_quality, force=args.force, stop=stop,
        )
    except ValueError as error:
        parser.error(str(error))
    print(
        f"{result['state']}: {result['tiles_done']}/{result['tiles']} tiles done, {result['buildings_described']} buildings described in this run "
        f"({result['model_answers']} new model answers), {result['buildings_described_total']} in total, {result['wall_s']:.0f} s. State: {result['state_file']}"
    )
    if result["state"] == "aborted":
        sys.exit(1)
    if result["state"] == "interrupted":
        sys.exit(130)


if __name__ == "__main__":
    main()
