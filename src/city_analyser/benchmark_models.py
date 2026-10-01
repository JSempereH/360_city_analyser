"""Measure local building-segmentation models on one panorama without caching output."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .building_analysis import _load_analysis_panorama, _load_segmentation_model, _require_ml_dependencies, _segment_buildings
from .download_models import MODEL_ALIASES


DEFAULT_MODELS = ("b0", "b2", "b5")


def benchmark_model(image_path: Path, model_root: Path, model_id: str) -> dict[str, Any]:
    _, torch, _, _ = _require_ml_dependencies()
    milestones: dict[str, float] = {}
    started = time.perf_counter()

    def progress(message: str) -> None:
        if message.startswith("Segmenting perspectives 1-"):
            milestones["loaded"] = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    previous_model = os.environ.get("BUILDING_ANALYSIS_MODEL")
    os.environ["BUILDING_ANALYSIS_MODEL"] = model_id
    try:
        _, _, _, device, resolved_model, resolved_revision = _segment_buildings(_load_analysis_panorama(image_path), model_root, progress)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        finished = time.perf_counter()
        result: dict[str, Any] = {
            "model": resolved_model,
            "model_revision": resolved_revision,
            "device": device,
            "load_seconds": round(milestones.get("loaded", finished) - started, 2),
            "segmentation_seconds": round(finished - milestones.get("loaded", started), 2),
            "total_seconds": round(finished - started, 2),
            "status": "ok",
        }
        if torch.cuda.is_available():
            result["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
            result["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 1024 ** 2, 1)
        return result
    except (OSError, RuntimeError, ValueError) as error:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"model": model_id, "status": "failed", "error": str(error)}
    finally:
        _load_segmentation_model.cache_clear()
        if previous_model is None:
            os.environ.pop("BUILDING_ANALYSIS_MODEL", None)
        else:
            os.environ["BUILDING_ANALYSIS_MODEL"] = previous_model


def _benchmark() -> bool:
    parser = argparse.ArgumentParser(description="Benchmark local building segmentation models on one panorama.")
    parser.add_argument("image", type=Path, help="local equirectangular JPEG panorama")
    parser.add_argument("--model", action="append", help=f"alias ({', '.join(MODEL_ALIASES)}) or Hugging Face model ID; repeatable")
    parser.add_argument("--model-root", type=Path, default=Path("models"), help="model cache directory")
    parser.add_argument("--tile-size", type=int, default=None, help="perspective tile edge in pixels (256-1024)")
    parser.add_argument("--batch-size", type=int, default=None, help="perspective tiles per forward pass (1-12)")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON results path")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    if args.tile_size is not None:
        os.environ["BUILDING_ANALYSIS_TILE_SIZE"] = str(args.tile_size)
    if args.batch_size is not None:
        os.environ["BUILDING_ANALYSIS_BATCH_SIZE"] = str(args.batch_size)
    models = [MODEL_ALIASES.get(name, name) for name in args.model or DEFAULT_MODELS]
    results = []
    for model_id in models:
        print(f"Benchmarking {model_id}...")
        result = benchmark_model(args.image, args.model_root, model_id)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
    output = {"image": str(args.image), "tile_size": args.tile_size, "batch_size": args.batch_size, "results": results}
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return any(result.get("device") == "cuda" for result in results)


def main() -> None:
    if _benchmark():
        # Some PyTorch/CUDA builds hang during CPython extension teardown after
        # inference. Results have already been flushed, so bypass that teardown.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
