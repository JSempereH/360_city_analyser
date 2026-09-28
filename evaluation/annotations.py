"""SAM-assisted building-instance annotations stored as reproducible prompts.

Each ``evaluation/annotations/<dataset>__<image>.json`` lists perspective
views and, per real building, a box (plus optional points) in one or more of
those views. A large SAM 2 checkpoint, different from the pipeline's, turns
the prompts into masks, which are reprojected into the panorama. The same
``building`` id in several views is one building. ``ignore`` boxes mark
regions that are not scored (distant clutter, ambiguous structures).

Only the prompts are versioned; ground-truth rasters are cached under
``data/.eval-cache/ground-truth``.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from evaluation import cache
from city_analyser.panorama_geometry import panorama_to_tile_grid, perspective_tile

ANNOTATION_ROOT = Path(__file__).resolve().parent / "annotations"
ANNOTATION_SEGMENTER = "facebook/sam2.1-hiera-large"
ANALYSIS_WIDTH = 2048
# Scored elevation band. Upward views (pitch 50, fov 100) reach 100 degrees,
# so the top of tall facades is scored too.
SCORED_ELEVATION_DEGREES = (-30.0, 80.0)


def load_all() -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(ANNOTATION_ROOT.glob("*.json"))]


def _panorama(annotation: dict[str, Any]) -> np.ndarray:
    from city_analyser import building_analysis

    manifest = json.loads((Path("data") / annotation["dataset"] / "manifest.json").read_text(encoding="utf-8"))
    image = next(item for item in manifest["images"] if item["id"] == annotation["image"])
    return building_analysis._load_analysis_panorama(Path("data") / annotation["dataset"] / image["local_file"])


def view_image(annotation: dict[str, Any], view: dict[str, Any]) -> np.ndarray:
    tile, _, _ = perspective_tile(_panorama(annotation), math.radians(view["yaw"]), math.radians(view["pitch"]), view["size"], view["fov"])
    return tile


@lru_cache(maxsize=1)
def _segmenter() -> tuple[Any, Any]:
    transformers = __import__("transformers")
    processor = transformers.Sam2Processor.from_pretrained(ANNOTATION_SEGMENTER, cache_dir="models")
    model = transformers.Sam2Model.from_pretrained(ANNOTATION_SEGMENTER, cache_dir="models").eval()
    return processor, model


def _prompt_masks(tile: np.ndarray, prompts: list[dict[str, Any]]) -> list[np.ndarray]:
    torch = __import__("torch")
    Image = __import__("PIL.Image", fromlist=["Image"])
    processor, model = _segmenter()
    masks = []
    for prompt in prompts:
        kwargs: dict[str, Any] = {"input_boxes": [[prompt["box"]]]}
        if prompt.get("points"):
            kwargs["input_points"] = [[[point[:2] for point in prompt["points"]]]]
            kwargs["input_labels"] = [[[int(point[2]) for point in prompt["points"]]]]
        inputs = processor(images=Image.fromarray(tile), return_tensors="pt", **kwargs)
        with torch.inference_mode():
            output = model(**inputs, multimask_output=False)
        masks.append(processor.post_process_masks(output.pred_masks, inputs["original_sizes"])[0][0, 0].numpy())
    return masks


def _to_panorama(mask: np.ndarray, view: dict[str, Any]) -> np.ndarray:
    torch = __import__("torch")
    grid, visible = panorama_to_tile_grid(
        torch, ANALYSIS_WIDTH, ANALYSIS_WIDTH // 2, math.radians(view["yaw"]), math.radians(view["pitch"]), view["fov"], "cpu",
    )
    sampled = torch.nn.functional.grid_sample(torch.from_numpy(mask[None, None].astype(np.float32)), grid, mode="nearest", align_corners=False)
    return ((sampled[0, 0] > 0.5) & visible).numpy()


def view_visibility(view: dict[str, Any]) -> np.ndarray:
    torch = __import__("torch")
    _, visible = panorama_to_tile_grid(
        torch, ANALYSIS_WIDTH, ANALYSIS_WIDTH // 2, math.radians(view["yaw"]), math.radians(view["pitch"]), view["fov"], "cpu",
    )
    return visible.numpy()


def ground_truth(annotation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Instance ids (0 none, -1 ignored) and the exhaustively annotated region."""
    digest = cache.canonical_digest(annotation)
    stored = cache.load("ground-truth", digest[:16], annotation["dataset"], annotation["image"])
    if stored is not None:
        return stored["truth"], stored["region"]
    height = ANALYSIS_WIDTH // 2
    truth = np.zeros((height, ANALYSIS_WIDTH), dtype=np.int32)
    region = np.zeros((height, ANALYSIS_WIDTH), dtype=bool)
    ids = {name: index for index, name in enumerate(sorted({item["building"] for item in annotation["buildings"]}), start=1)}
    painted: list[tuple[int, np.ndarray]] = []
    ignored = np.zeros_like(region)
    for view_index, view in enumerate(annotation["views"]):
        region |= view_visibility(view)
        tile = view_image(annotation, view)
        prompts = [item for item in annotation["buildings"] if item["view"] == view_index]
        for item, mask in zip(prompts, _prompt_masks(tile, prompts)):
            painted.append((ids[item["building"]], _to_panorama(mask, view)))
        for box in annotation.get("ignore", []):
            if box["view"] == view_index:
                x0, y0, x1, y1 = (int(value) for value in box["box"])
                ignore_mask = np.zeros(tile.shape[:2], dtype=bool)
                ignore_mask[y0:y1, x0:x1] = True
                ignored |= _to_panorama(ignore_mask, view)
    # Merge each building's views, then let smaller buildings win overlaps.
    merged: dict[int, np.ndarray] = {}
    for building, mask in painted:
        merged[building] = merged.get(building, np.zeros_like(region)) | mask
    for building, mask in sorted(merged.items(), key=lambda item: -int(item[1].sum())):
        truth[mask] = building
    truth[ignored & (truth == 0)] = -1
    low, high = SCORED_ELEVATION_DEGREES
    region[: int((0.5 - high / 180) * height)] = False
    region[int((0.5 - low / 180) * height):] = False
    cache.save("ground-truth", digest[:16], annotation["dataset"], annotation["image"], truth=truth, region=region)
    return truth, region


def render_view_for_annotation(annotation: dict[str, Any], view_index: int, destination: Path) -> None:
    """Tile with a labeled 64 px grid, to read box coordinates from."""
    Image = __import__("PIL.Image", fromlist=["Image"])
    ImageDraw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"])
    view = annotation["views"][view_index]
    image = Image.fromarray(view_image(annotation, view))
    draw = ImageDraw.Draw(image)
    for position in range(0, view["size"], 64):
        draw.line([(position, 0), (position, view["size"])], fill=(255, 255, 0), width=1)
        draw.line([(0, position), (view["size"], position)], fill=(255, 255, 0), width=1)
        draw.text((position + 2, 2), str(position), fill=(255, 0, 0))
        draw.text((2, position + 2), str(position), fill=(255, 0, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=90)


def render_ground_truth(annotation: dict[str, Any], destination: Path) -> None:
    Image = __import__("PIL.Image", fromlist=["Image"])
    truth, _ = ground_truth(annotation)
    panorama = _panorama(annotation).astype(np.float32)
    palette = np.random.default_rng(7).integers(40, 255, (int(truth.max()) + 1, 3))
    labeled = truth > 0
    panorama[labeled] = panorama[labeled] * 0.4 + palette[truth[labeled]] * 0.6
    panorama[truth < 0] = panorama[truth < 0] * 0.3
    edges = np.zeros_like(labeled)
    edges[:, 1:] |= truth[:, 1:] != truth[:, :-1]
    edges[1:] |= truth[1:] != truth[:-1]
    panorama[edges] = 255
    height = truth.shape[0]
    Image.fromarray(panorama[height // 20: height * 5 // 8].astype(np.uint8)).resize((1600, 450)).save(destination, quality=90)
