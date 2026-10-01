#!/usr/bin/env python3
"""Building-separation experiments over every downloaded panorama.

    uv run python -m evaluation.run semantic --variant b5-384 --model b5 --tile-size 384
    uv run python -m evaluation.run instances --variant gdino-tiny
    uv run python -m evaluation.run associate --semantic b5-384 --instances gdino-tiny --tag v2
    uv run python -m evaluation.run score --tag v2

Expensive model stages are cached (see ``evaluation.cache``); association and
scoring rerun in seconds, which is what threshold and policy tuning needs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from city_analyser import building_analysis as analysis
from city_analyser.download_models import MODEL_ALIASES
from evaluation import cache

REPORT_ROOT = cache.CACHE_ROOT / "reports"


def _coordinates(image: dict[str, Any]) -> tuple[float, float, float]:
    longitude, latitude = image["selected_geometry"]["coordinates"][:2]
    return float(latitude), float(longitude), float(image["computed_compass_angle_deg"])


def command_semantic(args: argparse.Namespace) -> None:
    model = MODEL_ALIASES.get(args.model, args.model)
    settings = {"BUILDING_ANALYSIS_MODEL": model}
    if args.tile_size:
        settings["BUILDING_ANALYSIS_TILE_SIZE"] = str(args.tile_size)
    with cache.environment(**settings):
        for dataset_id, image in cache.panoramas():
            if cache.load("semantic", args.variant, dataset_id, image["id"]) is not None:
                continue
            started = time.perf_counter()
            panorama = analysis._load_analysis_panorama(Path("data") / dataset_id / image["local_file"])
            buildings, vegetation, *_ = analysis._segment_buildings(panorama, Path("models"), lambda _: None)
            cache.save("semantic", args.variant, dataset_id, image["id"], buildings=buildings, vegetation=vegetation)
            print(f"{dataset_id} {image['id']} {time.perf_counter() - started:.1f}s", flush=True)


def command_instances(args: argparse.Namespace) -> None:
    from city_analyser import building_instances as instances

    settings = {}
    if args.detector:
        settings["BUILDING_ANALYSIS_INSTANCE_DETECTOR"] = args.detector
    if args.segmenter:
        settings["BUILDING_ANALYSIS_INSTANCE_SEGMENTER"] = args.segmenter
    if args.box_threshold is not None:
        settings["BUILDING_ANALYSIS_INSTANCE_BOX_THRESHOLD"] = str(args.box_threshold)
    torch = __import__("torch")
    settings_path = cache.CACHE_ROOT / "instances" / args.variant / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    annotated = None
    if args.only_annotated:
        from evaluation import annotations

        annotated = {item["image"] for item in annotations.load_all() if item["buildings"]}
    with cache.environment(**settings):
        configuration = instances.instance_settings()
        for dataset_id, image in cache.panoramas():
            if annotated is not None and image["id"] not in annotated:
                continue
            if cache.load("instances", args.variant, dataset_id, image["id"]) is not None:
                continue
            started = time.perf_counter()
            panorama = analysis._load_analysis_panorama(Path("data") / dataset_id / image["local_file"])
            instance_map, details = instances.segment_instances(panorama, Path("models"), torch.device("cpu"), configuration, lambda _: None)
            cache.save("instances", args.variant, dataset_id, image["id"], instance_map=instance_map, details=np.array(json.dumps(details)))
            print(f"{dataset_id} {image['id']} {time.perf_counter() - started:.1f}s instances={len(details)}", flush=True)


def _instance_environment(variant: str) -> dict[str, str]:
    """Settings an instance cache variant was produced with (stored beside it)."""
    path = cache.CACHE_ROOT / "instances" / variant / "settings.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def command_associate(args: argparse.Namespace) -> None:
    output = REPORT_ROOT / args.tag
    output.mkdir(parents=True, exist_ok=True)
    if args.no_repair:
        analysis.GPS_REPAIR_MAXIMUM_SHIFT_M = 0.0
    import ast

    for assignment in args.set or []:
        name, value = assignment.split("=", 1)
        if not hasattr(analysis, name):
            raise SystemExit(f"unknown building_analysis constant: {name}")
        setattr(analysis, name, ast.literal_eval(value))
    rows = []
    for dataset_id, image in cache.panoramas():
        semantic = cache.load("semantic", args.semantic, dataset_id, image["id"])
        if semantic is None:
            continue
        instance_map = None
        if args.instances:
            stored = cache.load("instances", args.instances, dataset_id, image["id"])
            if stored is None:
                continue
            instance_map = stored["instance_map"]
        latitude, longitude, heading = _coordinates(image)
        features = analysis.nearby_buildings(latitude, longitude)
        started = time.perf_counter()
        observed = semantic["buildings"]
        coverage = None
        if instance_map is not None:
            from city_analyser import building_instances

            with cache.environment(**_instance_environment(args.instances)):
                coverage = building_instances.instance_coverage(building_instances.instance_settings(), instance_map.shape[1], instance_map.shape[0])
            if args.pose_mask == "instances":
                # Same target as analyze_panorama.
                observed = observed & (instance_map > 0)
        pose = analysis._refine_camera_pose(observed, features, latitude, longitude, heading)
        ids, _, metadata, _ = analysis._assign_buildings_detailed(
            semantic["buildings"], semantic["vegetation"], features,
            pose["latitude"], pose["longitude"], pose["heading_degrees"], None, instance_map, coverage,
        )
        elapsed = time.perf_counter() - started
        visual = ids.astype(np.int64) * 65536 + (instance_map.astype(np.int64) * (ids > 0) if instance_map is not None else 0)
        np.savez_compressed(output / f"{dataset_id}__{image['id']}.npz", ids=ids, visual=visual)
        coverage = float((ids > 0).sum() / max(int(semantic["buildings"].sum()), 1))
        rows.append({
            "dataset": dataset_id, "image": image["id"], "pose": pose["reason"], "gps_repair_m": pose.get("gps_repair_m", 0.0),
            "matched": len(metadata), "coverage": round(coverage, 3), "seconds": round(elapsed, 2),
        })
        if args.overlays:
            _overlay(dataset_id, image, ids, instance_map, output / f"{dataset_id}__{image['id']}.jpg")
    (output / "association.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['dataset']:16s} {row['image']:>17s} {row['pose']:20s} repair {row['gps_repair_m']:4.1f} matched {row['matched']:3d} coverage {row['coverage']:.3f}")
    if rows:
        print(f"mean coverage {np.mean([row['coverage'] for row in rows]):.3f}; panoramas without matches {sum(row['matched'] == 0 for row in rows)}")


def _overlay(dataset_id: str, image: dict[str, Any], ids: Any, instance_map: Any, destination: Path) -> None:
    Image = __import__("PIL.Image", fromlist=["Image"])
    height = ids.shape[0]
    panorama = analysis._load_analysis_panorama(Path("data") / dataset_id / image["local_file"]).astype(np.float32)
    palette = np.random.default_rng(3).integers(40, 255, (int(ids.max()) + 1, 3))
    labeled = ids > 0
    panorama[labeled] = panorama[labeled] * 0.45 + palette[ids][labeled] * 0.55
    key = ids.astype(np.int64) * 65536 + (instance_map.astype(np.int64) * labeled if instance_map is not None else 0)
    edges = np.zeros_like(labeled)
    edges[:, 1:] |= key[:, 1:] != key[:, :-1]
    edges[1:] |= key[1:] != key[:-1]
    panorama[edges] = 255
    Image.fromarray(panorama[height // 20: height * 5 // 8].astype(np.uint8)).resize((1600, 450)).save(destination, quality=88)


def command_score(args: argparse.Namespace) -> None:
    from evaluation import annotations, metrics

    results = []
    for annotation in annotations.load_all():
        if not annotation["buildings"]:
            continue
        stored = REPORT_ROOT / args.tag / f"{annotation['dataset']}__{annotation['image']}.npz"
        if not stored.is_file():
            continue
        with np.load(stored) as data:
            predicted = data["visual"] if args.visual and "visual" in data.files else data["ids"]
        truth, region = annotations.ground_truth(annotation)
        results.append({"image": annotation["image"], **metrics.separation_scores(truth, predicted, region)})
    summary = metrics.summarize(results)
    (REPORT_ROOT / args.tag / ("scores-visual.json" if args.visual else "scores.json")).write_text(json.dumps({"summary": summary, "panoramas": results}, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(f"{result['image']:>17s} " + " ".join(f"{key} {value:.3f}" for key, value in result.items() if isinstance(value, float)))
    print("SUMMARY", json.dumps(summary))


# Four horizon views plus four upward views, so tall facades are annotated to their top.
ANNOTATION_VIEWS = [
    {"yaw": yaw, "pitch": pitch, "fov": 100, "size": 768} for pitch in (10, 50) for yaw in (0, 90, 180, 270)
]


def command_annotate(args: argparse.Namespace) -> None:
    from evaluation import annotations

    path = annotations.ANNOTATION_ROOT / f"{args.dataset}__{args.image}.json"
    if path.is_file():
        annotation = json.loads(path.read_text(encoding="utf-8"))
        missing = [view for view in ANNOTATION_VIEWS if view not in annotation["views"]]
        if missing:
            annotation["views"].extend(missing)
            path.write_text(json.dumps(annotation, indent=2) + "\n", encoding="utf-8")
    else:
        annotation = {"dataset": args.dataset, "image": args.image, "views": ANNOTATION_VIEWS, "buildings": [], "ignore": []}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(annotation, indent=2) + "\n", encoding="utf-8")
    for index in range(len(annotation["views"])) if args.view is None else [args.view]:
        destination = cache.CACHE_ROOT / "annotate" / f"{args.dataset}__{args.image}__view{index}.jpg"
        annotations.render_view_for_annotation(annotation, index, destination)
        print(destination)


def command_ground_truth(args: argparse.Namespace) -> None:
    from evaluation import annotations

    for annotation in annotations.load_all():
        if args.image and annotation["image"] != args.image:
            continue
        destination = cache.CACHE_ROOT / "annotate" / f"{annotation['dataset']}__{annotation['image']}__truth.jpg"
        annotations.render_ground_truth(annotation, destination)
        truth, _ = annotations.ground_truth(annotation)
        print(destination, "buildings", int(truth.max()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    semantic = commands.add_parser("semantic")
    semantic.add_argument("--variant", required=True)
    semantic.add_argument("--model", required=True)
    semantic.add_argument("--tile-size", type=int)
    semantic.set_defaults(handler=command_semantic)
    instances = commands.add_parser("instances")
    instances.add_argument("--variant", required=True)
    instances.add_argument("--detector")
    instances.add_argument("--segmenter")
    instances.add_argument("--only-annotated", action="store_true")
    instances.add_argument("--box-threshold", type=float)
    instances.set_defaults(handler=command_instances)
    associate = commands.add_parser("associate")
    associate.add_argument("--semantic", required=True)
    associate.add_argument("--instances")
    associate.add_argument("--tag", required=True)
    associate.add_argument("--no-repair", action="store_true")
    associate.add_argument("--overlays", action="store_true")
    associate.add_argument("--set", action="append", help="override a building_analysis constant, e.g. POSE_SEARCH_OFFSETS_M=(-4,0,4)")
    associate.add_argument("--pose-mask", choices=("semantic", "instances"), default="instances")
    associate.set_defaults(handler=command_associate)
    score = commands.add_parser("score")
    score.add_argument("--tag", required=True)
    score.add_argument("--visual", action="store_true", help="score OSM label x visual instance instead of OSM labels")
    score.set_defaults(handler=command_score)
    annotate = commands.add_parser("annotate")
    annotate.add_argument("--dataset", required=True)
    annotate.add_argument("--image", required=True)
    annotate.add_argument("--view", type=int)
    annotate.set_defaults(handler=command_annotate)
    truth = commands.add_parser("ground-truth")
    truth.add_argument("--image")
    truth.set_defaults(handler=command_ground_truth)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
