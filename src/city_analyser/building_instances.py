"""Visual building instances from text-prompted detection and box-prompted masks.

Semantic segmentation only says *building*; it cannot tell where one facade
ends and the next begins. Grounding DINO proposes one box per visible building
("building."), SAM 2 turns each box into a mask that follows the real facade
edges, and the tile masks are merged into one panorama-wide instance map.
Instance boundaries therefore come from the image, not from the projected OSM
footprints, so they survive GPS/heading error and coarse OSM block polygons.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .panorama_geometry import panorama_to_tile_grid, perspective_tile


DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"
SEGMENTER_ID = "facebook/sam2.1-hiera-small"
DETECTION_PROMPT = "building."
ULTRALYTICS_PROMPT = "building"
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
# Portrait views: 90 degrees wide, about 112 tall, centered 27 degrees up. A
# 30 m facade 5 m away spans roughly -19 to +80 degrees of elevation, so one
# ring of square views would cut either its base or its top.
INSTANCE_TILE_SIZE = 512
INSTANCE_TILE_HEIGHT = 768
INSTANCE_PITCH_DEGREES = 27.0
# Minimum share of a tile-boundary overlap two masks must agree on to merge.
MERGE_OVERLAP_IOU = 0.5
MERGE_MINIMUM_OVERLAP_PIXELS = 200
# A box is a "group" detection when other boxes cover most of it.
GROUP_BOX_COVERAGE = 0.6
MINIMUM_INSTANCE_PIXELS = 400


@dataclass(frozen=True)
class InstanceSettings:
    enabled: bool
    views: int
    field_of_view_degrees: float
    detector_size: int
    detector: str = DETECTOR_ID
    segmenter: str = SEGMENTER_ID
    box_threshold: float = BOX_THRESHOLD

    def as_configuration(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "detector": self.detector,
            "segmenter": self.segmenter,
            "prompt": DETECTION_PROMPT,
            "box_threshold": self.box_threshold,
            "text_threshold": TEXT_THRESHOLD,
            "views": self.views,
            "field_of_view_degrees": self.field_of_view_degrees,
            "pitch_degrees": INSTANCE_PITCH_DEGREES,
            "tile_size": [INSTANCE_TILE_SIZE, INSTANCE_TILE_HEIGHT],
            "detector_size": self.detector_size,
        }


def _bounded_int(name: str, default: int, low: int, high: int) -> int:
    configured = os.environ.get(name, "").strip()
    try:
        value = int(configured) if configured else default
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer between {low} and {high}.") from error
    if not low <= value <= high:
        raise RuntimeError(f"{name} must be between {low} and {high}.")
    return value


def instance_settings() -> InstanceSettings:
    mode = os.environ.get("BUILDING_ANALYSIS_INSTANCES", "on").strip().lower()
    if mode not in {"on", "off"}:
        raise RuntimeError("BUILDING_ANALYSIS_INSTANCES must be on or off.")
    views = _bounded_int("BUILDING_ANALYSIS_INSTANCE_VIEWS", 6, 4, 8)
    # Neighboring views overlap by 30 degrees so a building cut by one tile
    # edge is whole in the next tile and the two masks can be merged.
    return InstanceSettings(
        enabled=mode == "on",
        views=views,
        field_of_view_degrees=360.0 / views + 30.0,
        detector_size=_bounded_int("BUILDING_ANALYSIS_INSTANCE_DETECTOR_SIZE", 800, 384, 1024),
        # Any Grounding-DINO-family checkpoint (MM-Grounding-DINO, LLMDet) and
        # any SAM 2 checkpoint load through the same classes.
        detector=os.environ.get("BUILDING_ANALYSIS_INSTANCE_DETECTOR", "").strip() or DETECTOR_ID,
        segmenter=os.environ.get("BUILDING_ANALYSIS_INSTANCE_SEGMENTER", "").strip() or SEGMENTER_ID,
        # Detectors are calibrated differently; the default suits Grounding DINO.
        box_threshold=float(os.environ.get("BUILDING_ANALYSIS_INSTANCE_BOX_THRESHOLD", "").strip() or BOX_THRESHOLD),
    )


# Optional Ultralytics detectors ("ultralytics:yoloe-11l-seg.pt"). The package
# is AGPL-3.0 and deliberately not a project dependency.
ULTRALYTICS_PREFIX = "ultralytics:"
# Segmenter value that reuses a segmentation detector's own masks (no SAM).
DETECTOR_MASKS = "detector"


def _transformers() -> Any:
    try:
        return importlib.import_module("transformers")
    except ImportError as error:
        raise RuntimeError("Building instances require PyTorch and Transformers. Run uv sync, then retry.") from error


@lru_cache(maxsize=2)
def _load_detector(detector_id: str, model_root: str, device: str) -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    if detector_id.startswith(ULTRALYTICS_PREFIX):
        try:
            ultralytics = importlib.import_module("ultralytics")
        except ImportError as error:
            raise RuntimeError(
                f"{detector_id} needs the optional 'ultralytics' package (AGPL-3.0); "
                "install it separately, e.g. uv run --with ultralytics ...",
            ) from error
        name = detector_id.removeprefix(ULTRALYTICS_PREFIX)
        directory = Path(model_root) / "ultralytics"
        directory.mkdir(parents=True, exist_ok=True)
        # Ultralytics resolves weights and its text encoder relative to the cwd.
        with contextlib.chdir(directory):
            if "yoloe" in name.lower():
                model = ultralytics.YOLOE(name)
                model.set_classes([ULTRALYTICS_PROMPT], model.get_text_pe([ULTRALYTICS_PROMPT]))
            else:
                model = ultralytics.YOLOWorld(name)
                model.set_classes([ULTRALYTICS_PROMPT])
        return None, model
    transformers = _transformers()
    processor = transformers.AutoProcessor.from_pretrained(detector_id, cache_dir=Path(model_root))
    model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(detector_id, cache_dir=Path(model_root))
    return processor, model.to(torch.device(device)).eval()


@lru_cache(maxsize=2)
def _load_segmenter(segmenter_id: str, model_root: str, device: str) -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    transformers = _transformers()
    processor = transformers.Sam2Processor.from_pretrained(segmenter_id, cache_dir=Path(model_root))
    model = transformers.Sam2Model.from_pretrained(segmenter_id, cache_dir=Path(model_root)).to(torch.device(device)).eval()
    return processor, model


def _load_models(detector_id: str, segmenter_id: str, model_root: str, device: str) -> tuple[Any, Any, Any, Any]:
    segmenter = (None, None) if segmenter_id == DETECTOR_MASKS else _load_segmenter(segmenter_id, model_root, device)
    return (*_load_detector(detector_id, model_root, device), *segmenter)


def _ultralytics_device(device: Any) -> str:
    if device.type == "cuda":
        return str(device.index or 0)
    return device.type


def _detect_ultralytics(model: Any, tiles: list[Any], settings: "InstanceSettings", device: Any) -> list[dict[str, Any]]:
    detections = []
    for tile in tiles:
        result = model.predict(
            tile, conf=settings.box_threshold, imgsz=settings.detector_size,
            device=_ultralytics_device(device), half=device.type == "cuda", verbose=False,
        )[0]
        detection = {"boxes": result.boxes.xyxy.float().cpu(), "scores": result.boxes.conf.float().cpu()}
        if result.masks is not None and len(result.boxes):
            # Segmentation checkpoints (YOLOE-seg) also return instance masks
            # at the original tile size, which can stand in for SAM.
            detection["masks"] = _resize_masks(result.masks.data.float().cpu(), tile.size[1], tile.size[0])
        detections.append(detection)
    return detections


def _resize_masks(masks: Any, height: int, width: int) -> Any:
    torch = importlib.import_module("torch")
    if masks.shape[-2:] != (height, width):
        masks = torch.nn.functional.interpolate(masks[:, None], size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    return masks > 0.5


def _without_group_boxes(boxes: list[list[float]], scores: list[float]) -> tuple[list[list[float]], list[float]]:
    """Drop boxes that mostly consist of other detected buildings."""
    def area(box: list[float]) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    kept_boxes, kept_scores = [], []
    for index, box in enumerate(boxes):
        inner = [
            other for other_index, other in enumerate(boxes)
            if other_index != index and area(other) < area(box)
            and other[0] >= box[0] - 2 and other[1] >= box[1] - 2 and other[2] <= box[2] + 2 and other[3] <= box[3] + 2
        ]
        if len(inner) >= 2 and sum(area(other) for other in inner) >= GROUP_BOX_COVERAGE * area(box):
            continue
        kept_boxes.append(box)
        kept_scores.append(scores[index])
    return kept_boxes, kept_scores


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        self.parent[self.find(first)] = self.find(second)


def _detector_batch_size(device: Any, views: int) -> int:
    configured = os.environ.get("BUILDING_ANALYSIS_INSTANCE_BATCH_SIZE", "").strip()
    if configured:
        return _bounded_int("BUILDING_ANALYSIS_INSTANCE_BATCH_SIZE", 1, 1, views)
    return 1 if device.type == "cpu" else views


def _view_yaws(settings: InstanceSettings) -> list[float]:
    return [index * 2 * math.pi / settings.views for index in range(settings.views)]


def instance_coverage(settings: InstanceSettings, width: int, height: int) -> Any:
    """Panorama pixels seen by at least one instance view (numpy bool array)."""
    torch = importlib.import_module("torch")
    covered = torch.zeros((height, width), dtype=torch.bool)
    for yaw in _view_yaws(settings):
        _, visible = panorama_to_tile_grid(
            torch, width, height, yaw, math.radians(INSTANCE_PITCH_DEGREES), settings.field_of_view_degrees, "cpu",
            INSTANCE_TILE_HEIGHT / INSTANCE_TILE_SIZE,
        )
        covered |= visible
    return covered.numpy()


def _detect_grounding(
    detector_processor: Any, detector: Any, tiles: list[Any], settings: "InstanceSettings", device: Any, aspect: float, autocast: Any,
) -> list[dict[str, Any]]:
    torch = importlib.import_module("torch")
    detections: list[dict[str, Any]] = []
    # On CPUs batching barely helps and multiplies peak memory (several GB for
    # six 800x1200 views), so detect one view at a time; GPUs batch all views
    # and halve the batch on out-of-memory.
    batch_size = _detector_batch_size(device, len(tiles))
    start = 0
    while start < len(tiles):
        batch = tiles[start:start + batch_size]
        inputs = detector_processor(
            images=batch, text=[DETECTION_PROMPT] * len(batch), return_tensors="pt",
            size={"shortest_edge": settings.detector_size, "longest_edge": math.ceil(settings.detector_size * aspect)},
        ).to(device)
        try:
            with torch.inference_mode(), autocast:
                outputs = detector(**inputs)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            continue
        detections.extend(detector_processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=settings.box_threshold, text_threshold=TEXT_THRESHOLD,
            target_sizes=[(INSTANCE_TILE_HEIGHT, INSTANCE_TILE_SIZE)] * len(batch),
        ))
        del inputs, outputs
        start += len(batch)

    return detections


def segment_instances(
    panorama: Any, model_root: Path, device: Any, settings: InstanceSettings, progress: Callable[[str], None],
) -> tuple[Any, list[dict[str, Any]]]:
    """Return an (H, W) uint16 instance map (0 = none) and per-instance details."""
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    height, width = panorama.shape[:2]
    detector_processor, detector, segmenter_processor, segmenter = _load_models(
        settings.detector, settings.segmenter, str(model_root.resolve()), str(device),
    )
    yaws = _view_yaws(settings)
    pitch = math.radians(INSTANCE_PITCH_DEGREES)
    aspect = INSTANCE_TILE_HEIGHT / INSTANCE_TILE_SIZE
    Image = importlib.import_module("PIL.Image")
    tiles = [
        Image.fromarray(perspective_tile(panorama, yaw, pitch, INSTANCE_TILE_SIZE, settings.field_of_view_degrees, INSTANCE_TILE_HEIGHT)[0])
        for yaw in yaws
    ]
    # Half precision on GPUs; CPUs keep float32 (no fast half kernels).
    autocast = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")

    progress(f"Detecting buildings in {len(tiles)} views...")
    if settings.detector.startswith(ULTRALYTICS_PREFIX):
        detections = _detect_ultralytics(detector, tiles, settings, device)
    else:
        detections = _detect_grounding(detector_processor, detector, tiles, settings, device, aspect, autocast)

    masks, scores, views = [], [], []
    visibility = []
    for view, (tile, yaw, detection) in enumerate(zip(tiles, yaws, detections)):
        grid, visible = panorama_to_tile_grid(torch, width, height, yaw, pitch, settings.field_of_view_degrees, device, aspect)
        visibility.append(visible)
        boxes, box_scores = _without_group_boxes(detection["boxes"].tolist(), detection["scores"].tolist())
        if not boxes:
            continue
        progress(f"Outlining {len(boxes)} buildings in view {view + 1}/{len(tiles)}...")
        if settings.segmenter == DETECTOR_MASKS:
            if "masks" not in detection:
                raise RuntimeError(f"{settings.detector} returns no masks; choose a SAM 2 segmenter.")
            kept = [index for index, box in enumerate(detection["boxes"].tolist()) if box in boxes]
            tile_masks = detection["masks"][kept]
        else:
            prompt = segmenter_processor(images=tile, input_boxes=[boxes], return_tensors="pt").to(device)
            with torch.inference_mode(), autocast:
                predicted = segmenter(**prompt, multimask_output=False)
            tile_masks = segmenter_processor.post_process_masks(predicted.pred_masks.float().cpu(), prompt["original_sizes"].cpu())[0][:, 0]
        sampled = torch.nn.functional.grid_sample(
            tile_masks[:, None].float().to(device), grid.expand(len(boxes), -1, -1, -1), mode="nearest", align_corners=False,
        )[:, 0] > 0.5
        for mask, score in zip(sampled & visible, box_scores):
            masks.append(mask)
            scores.append(float(score))
            views.append(view)

    # Merge the same building seen by neighboring views: compare masks only
    # where both views see the panorama, since each is clipped by its tile.
    union_find = _UnionFind(len(masks))
    for first in range(len(masks)):
        for second in range(first + 1, len(masks)):
            if views[first] == views[second]:
                continue
            shared = visibility[views[first]] & visibility[views[second]]
            first_part, second_part = masks[first] & shared, masks[second] & shared
            first_size, second_size = int(first_part.sum()), int(second_part.sum())
            if min(first_size, second_size) < MERGE_MINIMUM_OVERLAP_PIXELS:
                continue
            intersection = int((first_part & second_part).sum())
            if intersection / (first_size + second_size - intersection) >= MERGE_OVERLAP_IOU:
                union_find.union(first, second)
    groups: dict[int, list[int]] = {}
    for index in range(len(masks)):
        groups.setdefault(union_find.find(index), []).append(index)
    merged = []
    for members in groups.values():
        mask = torch.zeros((height, width), dtype=torch.bool, device=device)
        for member in members:
            mask |= masks[member]
        merged.append((mask, max(scores[member] for member in members), sorted({views[member] for member in members})))

    # Where instances still overlap, the more specific (smaller) one wins.
    instance_map = torch.zeros((height, width), dtype=torch.int32, device=device)
    merged.sort(key=lambda item: -int(item[0].sum()))
    for label, (mask, _, _) in enumerate(merged, start=1):
        instance_map[mask] = label
    instance_map = instance_map.cpu().numpy()
    pixels = np.bincount(instance_map.ravel(), minlength=len(merged) + 1)
    instances = []
    relabel = np.zeros(len(merged) + 1, dtype=np.uint16)
    for label, (_, score, member_views) in enumerate(merged, start=1):
        if pixels[label] < MINIMUM_INSTANCE_PIXELS:
            continue
        relabel[label] = len(instances) + 1
        instances.append({"instance": len(instances) + 1, "score": round(score, 3), "views": member_views, "pixels": int(pixels[label])})
    return relabel[instance_map], instances
