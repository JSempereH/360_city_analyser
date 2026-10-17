"""Analyze every panorama of a dataset, overlapping model inference with OSM association.

The model stages run locally or on a remote worker (``BUILDING_ANALYSIS_BACKEND``) while the
previous panorama is associated with OSM footprints on a worker thread, and
footprints are fetched ahead. A JSON report records per-panorama stage times,
stage-cache hits and peak memory, so runs on different machines are comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Collection

from .analysis_backend import AnalysisBackend, analysis_backend_from_environment, perceive
from .building_analysis import (
    analysis_configuration,
    associate_panorama,
    load_analysis,
    nearby_buildings,
    panorama_inputs,
    perception_configuration,
)
from .analysis_cache import atomic_write_json
from .building_index import build_building_index, index_path, load_building_index, select_panoramas
from .image_quality import panorama_quality, quality_thresholds
from .local_viewer import dataset_manifest
from .run_ledger import DONE, FAILED, MAX_TOTAL_ATTEMPTS, SKIPPED, RunLedger, file_signature, ledger_path, settings_fingerprint
from .run_progress import ProgressTracker, format_duration, progress_path

# Footprint requests kept in flight ahead of the model stages.
OSM_PREFETCH = 4


def select_images(
    data_root: Path, dataset_id: str, images: list[dict[str, Any]], building_ids: list[str] | None,
    max_views_per_building: int | None, min_aov_deg: float,
) -> list[dict[str, Any]]:
    """Keep only the panoramas that best show ``building_ids`` (all buildings when ``None``), from the geometric index.

    The index is rebuilt when the panorama set changed. No AOV floor is applied
    by default: on the local benchmark every building the analysis matched was
    in the index, but only a third had a 10 degree view.
    """
    all_images = [image for image in dataset_manifest(data_root, dataset_id)["images"] if isinstance(image, dict) and image.get("local_file")]
    index = load_building_index(data_root, dataset_id, all_images)
    if index is None:
        index = build_building_index(data_root, dataset_id, all_images)
        atomic_write_json(index_path(data_root, dataset_id), index)
    selection = select_panoramas(
        index, building_ids, min_aov_deg=min_aov_deg,
        max_views_per_building=max_views_per_building if max_views_per_building is not None else 10**9,
    )
    print(f"Index selection: {len(selection['images'])} panoramas cover {selection['buildings_covered']} buildings; {len(selection['buildings_uncovered'])} have no acceptable view.", flush=True)
    chosen = set(selection["images"])
    return [image for image in images if str(image["id"]) in chosen]


def _peak_memory() -> dict[str, float]:
    # ru_maxrss is in KiB on Linux.
    peak = {"peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        peak["peak_cuda_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    return peak


class BatchAborted(RuntimeError):
    """Raised by callers that treat an aborted run as an error (the command exits 1)."""


# Pause before a retry; tests set it to 0.
RETRY_BACKOFF_S = 2.0
MAX_BACKOFF_S = 30.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_CONSECUTIVE_FAILURES = 10


def _is_permanent(error: BaseException) -> bool:
    """Bad input will fail the same way again; everything else (network, OSM, GPU memory) may not."""
    return isinstance(error, ValueError)


def _release_gpu_memory() -> None:
    import gc

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _retry(step: Any, attempts: int, stop: threading.Event | None, label: str) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return step()
        except Exception as error:  # noqa: BLE001 - classified below; KeyboardInterrupt is not an Exception
            if _is_permanent(error) or attempt == attempts or (stop is not None and stop.is_set()):
                raise
            print(f"  {label}: attempt {attempt}/{attempts} failed ({type(error).__name__}: {error}); retrying", flush=True)
            _release_gpu_memory()
            time.sleep(min(RETRY_BACKOFF_S * 2 ** (attempt - 1), MAX_BACKOFF_S))


def analyze_dataset(
    data_root: Path, model_root: Path, dataset_id: str, force: bool = False, limit: int | None = None,
    backend: AnalysisBackend | None = None, concurrency: int | None = None, perception_only: bool = False,
    skip_low_quality: bool = False, only_buildings: list[str] | None = None, max_views_per_building: int | None = None,
    min_aov_deg: float = 0.0, verify: bool = False, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES, retry_failed: bool = False,
    stop: threading.Event | None = None, image_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    """Analyze a dataset's panoramas, resumably and without letting one bad panorama stop the run.

    Finished work is remembered in the run ledger (see ``run_ledger``), so a
    rerun, after a crash or a Colab disconnect, starts at the first unfinished
    panorama; ``force`` redoes everything and ``verify`` re-validates finished
    analyses from the artifacts. Transient failures are retried, then recorded
    and left for the next run; permanent ones (bad input) are recorded and not
    retried. ``perception_only`` just fills the stage cache (no OSM, no artifacts).
    ``image_ids`` restricts the run to those panoramas (e.g. one spatial tile).
    """
    backend = backend or analysis_backend_from_environment()
    # A remote worker serializes the GPU, but upload, download and the previous
    # panorama's association overlap with it. Local models run one at a time.
    concurrency = concurrency or (2 if backend.kind == "remote" else 1)
    if concurrency < 1 or max_attempts < 1 or max_consecutive_failures < 1:
        raise ValueError("concurrency, max_attempts and max_consecutive_failures must be at least 1")
    stop = stop or threading.Event()
    stage = "perception" if perception_only else "analysis"
    configuration: dict[str, Any] = perception_configuration() if perception_only else analysis_configuration()
    if skip_low_quality:
        configuration = {**configuration, "skip_low_quality": quality_thresholds()}
    fingerprint = settings_fingerprint(stage, configuration)
    ledger = RunLedger(ledger_path(data_root, dataset_id))
    candidates = [image for image in dataset_manifest(data_root, dataset_id)["images"] if isinstance(image, dict) and image.get("local_file")]
    if image_ids is not None:
        wanted = {str(image_id) for image_id in image_ids}
        candidates = [image for image in candidates if str(image.get("id")) in wanted]
    if only_buildings is not None or max_views_per_building is not None:
        candidates = select_images(data_root, dataset_id, candidates, only_buildings, max_views_per_building, min_aov_deg)
    rows: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    already_done = already_resolved = 0
    for image in candidates:
        image_id = str(image["id"])
        try:
            signature = file_signature(data_root / dataset_id / str(image["local_file"]))
        except OSError:
            images.append(image)  # reported with its real error when its turn comes
            continue
        entry = ledger.get(stage, image_id)
        if not force and entry is not None and entry.get("settings") == fingerprint and entry.get("size") == signature["size"] and entry.get("mtime_ns") == signature["mtime_ns"]:
            if entry["status"] == SKIPPED or (entry["status"] == FAILED and not retry_failed and ledger.total_failures(stage, image_id) >= MAX_TOTAL_ATTEMPTS):
                already_resolved += 1
                continue
            if entry["status"] == DONE and not (verify and stage == "analysis" and load_analysis(data_root, dataset_id, image_id, image) is None):
                already_done += 1
                continue
        elif not force and entry is None and stage == "analysis" and load_analysis(data_root, dataset_id, image_id, image) is not None:
            # Analyzed before the ledger existed (viewer, older runs): validate once, then remember.
            ledger.record(stage, image_id, DONE, fingerprint, signature)
            already_done += 1
            continue
        images.append(image)
    images = images[:limit] if limit is not None else images
    tracker = ProgressTracker(
        progress_path(data_root, dataset_id), len(candidates), already_done + already_resolved,
        {"stage": stage, "backend": backend.kind, "dataset": dataset_id},
    )
    tracker.write(force=True)
    state_lock = threading.Lock()
    consecutive_failures = [0]
    aborted = [False]
    started = time.perf_counter()

    def settle(image: dict[str, Any], row: dict[str, Any], error: BaseException | None) -> None:
        """Record the outcome of one panorama in the ledger, the progress file and the row."""
        image_id = str(image["id"])
        try:
            signature = file_signature(data_root / dataset_id / str(image["local_file"]))
        except OSError:
            signature = None
        with state_lock:
            if error is None:
                ledger.record(stage, image_id, DONE, fingerprint, signature, seconds=row.get("perceive_s", 0) + row.get("associate_s", 0), buildings=row.get("buildings"))
                tracker.finished_one(image_id, True)
                consecutive_failures[0] = 0
                return
            permanent = _is_permanent(error)
            message = f"{type(error).__name__}: {error}"
            row["error"] = message
            if permanent:
                ledger.record(stage, image_id, SKIPPED, fingerprint, signature, error=message)
            else:
                ledger.record(stage, image_id, FAILED, fingerprint, signature, error=message, attempts=ledger.total_failures(stage, image_id) + 1)
                consecutive_failures[0] += 1
                if consecutive_failures[0] >= max_consecutive_failures:
                    aborted[0] = True
                    stop.set()
            tracker.finished_one(image_id, False, message, permanent)

    if skip_low_quality:
        # Cheap CPU check before any model or upload: flagged panoramas never reach the GPU.
        kept = []
        for image in images:
            try:
                flags = panorama_quality(panorama_inputs(data_root, dataset_id, image).image_path)["flags"]
            except ValueError:
                kept.append(image)  # reported below with its real error
                continue
            if flags:
                rows.append({"image": str(image["id"]), "error": f"low image quality ({', '.join(flags)})"})
                _print_row(rows[-1], len(rows), len(images))
                settle(image, {}, ValueError(f"low image quality ({', '.join(flags)})"))
            else:
                kept.append(image)
        images = kept
    final_state = "finished"
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="osm-footprints") as osm_pool, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="association") as association_pool, \
            ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="perception") as perception_pool:
        footprints: dict[int, Future] = {}
        perceptions: dict[int, Future] = {}
        pending: Future | None = None

        def prefetch(index: int) -> None:
            for ahead in range(index, min(index + OSM_PREFETCH, len(images))):
                if ahead not in footprints:
                    try:
                        inputs = panorama_inputs(data_root, dataset_id, images[ahead])
                    except ValueError:
                        continue
                    footprints[ahead] = osm_pool.submit(nearby_buildings, inputs.latitude, inputs.longitude)

        def perceive_one(inputs: Any) -> tuple[Any, float, int]:
            cache_hits: list[str] = []
            perceive_started = time.perf_counter()
            perception = _retry(
                lambda: perceive(
                    backend, data_root, model_root, inputs.image_path,
                    lambda message: cache_hits.append(message) if message.startswith("Reusing cached") else None,
                ),
                max_attempts, stop, f"models {inputs.image_id}",
            )
            return perception, time.perf_counter() - perceive_started, len(cache_hits)

        def schedule(index: int) -> None:
            for ahead in range(index, min(index + concurrency, len(images))):
                if ahead not in perceptions:
                    try:
                        inputs = panorama_inputs(data_root, dataset_id, images[ahead])
                    except ValueError:
                        continue
                    perceptions[ahead] = perception_pool.submit(perceive_one, inputs)

        def associate(index: int, inputs: Any, perception: Any, row: dict[str, Any]) -> dict[str, Any]:
            association_started = time.perf_counter()
            first_attempt = [True]

            def step() -> dict[str, Any]:
                if first_attempt[0] and index in footprints:
                    features = footprints.pop(index).result()
                else:
                    features = nearby_buildings(inputs.latitude, inputs.longitude)
                first_attempt[0] = False
                row["osm_wait_s"] = round(time.perf_counter() - association_started, 2)
                return associate_panorama(data_root, dataset_id, images[index], inputs, perception, features, lambda _: None)

            try:
                metadata = _retry(step, max_attempts, stop, f"association {inputs.image_id}")
            except Exception as error:  # noqa: BLE001
                settle(images[index], row, error)
                return row
            row["associate_s"] = round(time.perf_counter() - association_started, 2)
            row["buildings"] = len(metadata.get("building_ids", {}))
            settle(images[index], row, None)
            return row

        def collect(future: Future) -> None:
            rows.append(future.result())
            _print_row(rows[-1], len(rows), len(images), tracker.eta_seconds())

        try:
            for index, image in enumerate(images):
                if stop.is_set():
                    break
                row: dict[str, Any] = {"image": str(image["id"])}
                try:
                    inputs = panorama_inputs(data_root, dataset_id, image)
                except ValueError as error:
                    rows.append({**row, "error": str(error)})
                    settle(image, rows[-1], error)
                    continue
                tracker.current = [inputs.image_id]
                if not perception_only:
                    prefetch(index)
                schedule(index)
                try:
                    perception, perceive_s, cached_stages = perceptions.pop(index).result()
                except Exception as error:  # noqa: BLE001
                    footprints.pop(index, None)
                    # Settle the previous panorama first, so outcomes are counted in order and a
                    # success still being associated resets the consecutive-failure count.
                    if pending is not None:
                        collect(pending)
                        pending = None
                    rows.append({**row, "error": f"{type(error).__name__}: {error}"})
                    settle(image, rows[-1], error)
                    _print_row(rows[-1], len(rows), len(images), tracker.eta_seconds())
                    continue
                row["perceive_s"], row["cached_stages"] = round(perceive_s, 2), cached_stages
                if perception_only:
                    rows.append(row)
                    settle(image, row, None)
                    _print_row(row, len(rows), len(images), tracker.eta_seconds())
                    continue
                if pending is not None:
                    collect(pending)
                pending = association_pool.submit(associate, index, inputs, perception, row)
            if pending is not None:
                collect(pending)
            if stop.is_set():
                final_state = "aborted" if aborted[0] else "interrupted"
        except BaseException:
            final_state = "interrupted"
            raise
        finally:
            tracker.close(final_state)
    return {
        "dataset": dataset_id,
        "backend": backend.kind,
        "perception_only": perception_only,
        "state": final_state,
        "panoramas": len(images),
        "already_done": already_done,
        "already_resolved": already_resolved,
        "failed": tracker.failed,
        "skipped": tracker.skipped,
        "wall_s": round(time.perf_counter() - started, 2),
        "configuration": {name: value for name, value in sorted(os.environ.items()) if name.startswith("BUILDING_ANALYSIS_") and "TOKEN" not in name},
        **_peak_memory(),
        "rows": rows,
    }


def _print_row(row: dict[str, Any], done: int, total: int, eta_seconds: float | None = None) -> None:
    suffix = f" | ETA {format_duration(eta_seconds)}" if eta_seconds else ""
    if "error" in row and "perceive_s" not in row:
        print(f"[{done}/{total}] {row['image']}: skipped, {row['error']}{suffix}", flush=True)
        return
    if "error" in row:
        print(f"[{done}/{total}] {row['image']}: FAILED, {row['error']}{suffix}", flush=True)
        return
    line = f"[{done}/{total}] {row['image']}: models {row['perceive_s']:.1f} s ({row['cached_stages']} cached stages)"
    if "associate_s" in row:
        line += f", association {row['associate_s']:.1f} s, {row['buildings']} buildings"
    print(line + suffix, flush=True)


def _install_stop_handlers(stop: threading.Event) -> None:
    """First interrupt or SIGTERM (what Colab and cloud schedulers send) finishes the current panorama; the second aborts."""
    def handler(signum: int, frame: Any) -> None:
        if stop.is_set():
            raise KeyboardInterrupt
        print("\nStopping after the current panorama; progress is saved. Interrupt again to abort now.", flush=True)
        stop.set()

    for name in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(name, handler)
        except ValueError:
            pass  # not the main thread (e.g. inside a notebook kernel thread)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze every panorama of a downloaded dataset. Reruns resume where the last run stopped; "
        "progress is in <dataset>/batch-progress.json (render it with city-analyser-progress).",
    )
    parser.add_argument("dataset", help="dataset directory name under --data-root")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="datasets directory (default: data)")
    parser.add_argument("--model-root", type=Path, default=Path("models"), help="model cache directory (default: models)")
    parser.add_argument("--force", action="store_true", help="redo every panorama, ignoring the run ledger and current analyses")
    parser.add_argument("--verify", action="store_true", help="re-validate panoramas the ledger says are done from their artifacts (hashes the files; slow on big datasets)")
    parser.add_argument("--retry-failed", action="store_true", help=f"retry panoramas that already failed {MAX_TOTAL_ATTEMPTS} times in earlier runs")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS, help="tries per step for transient errors such as network, OSM or GPU memory (default: %(default)s)")
    parser.add_argument("--max-consecutive-failures", type=int, default=DEFAULT_MAX_CONSECUTIVE_FAILURES, help="stop the run after this many panoramas in a row fail, which usually means a broken environment (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=None, help="analyze at most this many panoramas")
    parser.add_argument("--perception-only", action="store_true", help="run only the model stages and fill the stage cache (no OSM, no artifacts); e.g. on a Colab GPU")
    parser.add_argument("--skip-low-quality", action="store_true", help="do not analyze panoramas flagged blurry, dark, overexposed or tiny (thresholds: BUILDING_ANALYSIS_MIN_SHARPNESS and friends)")
    parser.add_argument("--only-buildings", help="comma-separated OSM IDs (way/123,...): analyze only the panoramas that best show them (geometric index, no models)")
    parser.add_argument("--max-views-per-building", type=int, default=None, help="keep at most N panoramas per building (OpenFACADES uses 5); implies the geometric index")
    parser.add_argument("--min-aov", type=float, default=0.0, help="minimum angle of view of a building, in degrees, for the index selection (default 0: OpenFACADES uses 10)")
    parser.add_argument("--concurrency", type=int, default=None, help="panoramas perceived at once (default: 2 with a remote worker, otherwise 1)")
    parser.add_argument("--report", type=Path, default=None, help="write per-panorama timings and peak memory as JSON")
    args = parser.parse_args()
    stop = threading.Event()
    _install_stop_handlers(stop)
    try:
        report = analyze_dataset(
            args.data_root, args.model_root, args.dataset, args.force, args.limit, concurrency=args.concurrency,
            perception_only=args.perception_only, skip_low_quality=args.skip_low_quality,
            only_buildings=args.only_buildings.split(",") if args.only_buildings else None,
            max_views_per_building=args.max_views_per_building, min_aov_deg=args.min_aov, verify=args.verify,
            max_attempts=args.max_attempts, max_consecutive_failures=args.max_consecutive_failures,
            retry_failed=args.retry_failed, stop=stop,
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    finally:
        try:
            from .run_progress import progress_html

            (args.data_root / args.dataset / "progress.html").write_text(progress_html(args.data_root, args.dataset), encoding="utf-8")
        except (OSError, ValueError, ImportError):
            pass
    completed = [row for row in report["rows"] if "error" not in row]
    print(
        f"{report['state']}: {len(completed)} panoramas done in {report['wall_s']:.1f} s "
        f"({report['already_done']} already done, {report['failed']} failed, {report['skipped']} skipped); peak RSS {report['peak_rss_mib']:.0f} MiB"
    )
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["failed"] and report["state"] == "finished":
        print("Failed panoramas are retried by running the same command again.")
    if report["state"] == "aborted":
        print(f"Aborted after {args.max_consecutive_failures} consecutive failures; see the errors above or in progress.html.", file=sys.stderr)
        sys.exit(1)
    if report["state"] == "interrupted":
        sys.exit(130)


if __name__ == "__main__":
    main()
