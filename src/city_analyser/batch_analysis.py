"""Analyze every panorama of a dataset, overlapping model inference with OSM association.

The model stages (``perceive_panorama``) run on the main thread while the
previous panorama is associated with OSM footprints on a worker thread, and
footprints are fetched ahead. A JSON report records per-panorama stage times,
stage-cache hits and peak memory, so runs on different machines are comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .building_analysis import (
    _require_ml_dependencies,
    associate_panorama,
    load_analysis,
    nearby_buildings,
    panorama_inputs,
    perceive_panorama,
)
from .local_viewer import dataset_manifest

# Footprint requests kept in flight ahead of the model stages.
OSM_PREFETCH = 4


def _peak_memory() -> dict[str, float]:
    # ru_maxrss is in KiB on Linux.
    peak = {"peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        peak["peak_cuda_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    return peak


def analyze_dataset(data_root: Path, model_root: Path, dataset_id: str, force: bool = False, limit: int | None = None) -> dict[str, Any]:
    images = [image for image in dataset_manifest(data_root, dataset_id)["images"] if isinstance(image, dict) and image.get("local_file")]
    if not force:
        images = [image for image in images if load_analysis(data_root, dataset_id, str(image["id"])) is None]
    images = images[:limit] if limit is not None else images
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="osm-footprints") as osm_pool, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="association") as association_pool:
        footprints: dict[int, Future] = {}
        pending: Future | None = None

        def prefetch(index: int) -> None:
            for ahead in range(index, min(index + OSM_PREFETCH, len(images))):
                if ahead not in footprints:
                    try:
                        inputs = panorama_inputs(data_root, dataset_id, images[ahead])
                    except ValueError:
                        continue
                    footprints[ahead] = osm_pool.submit(nearby_buildings, inputs.latitude, inputs.longitude)

        def associate(index: int, inputs: Any, perception: Any, row: dict[str, Any]) -> dict[str, Any]:
            association_started = time.perf_counter()
            features = footprints.pop(index).result()
            row["osm_wait_s"] = round(time.perf_counter() - association_started, 2)
            metadata = associate_panorama(data_root, dataset_id, images[index], inputs, perception, features, lambda _: None)
            row["associate_s"] = round(time.perf_counter() - association_started, 2)
            row["buildings"] = len(metadata.get("building_ids", {}))
            return row

        for index, image in enumerate(images):
            row: dict[str, Any] = {"image": str(image["id"])}
            try:
                inputs = panorama_inputs(data_root, dataset_id, image)
            except ValueError as error:
                rows.append({**row, "error": str(error)})
                continue
            prefetch(index)
            cache_hits: list[str] = []
            perceive_started = time.perf_counter()
            perception = perceive_panorama(
                data_root, model_root, inputs.image_path,
                lambda message: cache_hits.append(message) if message.startswith("Reusing cached") else None,
            )
            row["perceive_s"] = round(time.perf_counter() - perceive_started, 2)
            row["cached_stages"] = len(cache_hits)
            if pending is not None:
                rows.append(pending.result())
                _print_row(rows[-1], len(rows), len(images))
            pending = association_pool.submit(associate, index, inputs, perception, row)
        if pending is not None:
            rows.append(pending.result())
            _print_row(rows[-1], len(rows), len(images))
    return {
        "dataset": dataset_id,
        "panoramas": len(images),
        "wall_s": round(time.perf_counter() - started, 2),
        "configuration": {name: value for name, value in sorted(os.environ.items()) if name.startswith("BUILDING_ANALYSIS_")},
        **_peak_memory(),
        "rows": rows,
    }


def _print_row(row: dict[str, Any], done: int, total: int) -> None:
    if "error" in row:
        print(f"[{done}/{total}] {row['image']}: skipped, {row['error']}", flush=True)
        return
    print(
        f"[{done}/{total}] {row['image']}: models {row['perceive_s']:.1f} s ({row['cached_stages']} cached stages), "
        f"association {row['associate_s']:.1f} s, {row['buildings']} buildings",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze every panorama of a downloaded dataset.")
    parser.add_argument("dataset", help="dataset directory name under --data-root")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="datasets directory (default: data)")
    parser.add_argument("--model-root", type=Path, default=Path("models"), help="model cache directory (default: models)")
    parser.add_argument("--force", action="store_true", help="re-associate panoramas whose analysis is already current")
    parser.add_argument("--limit", type=int, default=None, help="analyze at most this many panoramas")
    parser.add_argument("--report", type=Path, default=None, help="write per-panorama timings and peak memory as JSON")
    args = parser.parse_args()
    _require_ml_dependencies()
    try:
        report = analyze_dataset(args.data_root, args.model_root, args.dataset, args.force, args.limit)
    except ValueError as error:
        parser.error(str(error))
    analyzed = [row for row in report["rows"] if "error" not in row]
    print(f"{len(analyzed)} panoramas in {report['wall_s']:.1f} s; peak RSS {report['peak_rss_mib']:.0f} MiB")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
