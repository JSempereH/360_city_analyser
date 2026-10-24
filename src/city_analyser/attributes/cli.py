"""city-analyser-attributes: attributes for every analyzed building of a dataset."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shapely.geometry import shape

from ..analysis_cache import atomic_write_bytes
from ..building_analysis import ANALYSIS_VERSION, nearby_buildings
from ..building_inspection.crops import analyzed_building_ids, crops_directory, export_building_crops, set_aside_views
from ..building_inspection.evidence import BuildingEvidenceNotFound, collect_building_evidence
from ..building_inspection.reporting import _building_parts
from ..multi_view_analysis import panorama_coordinates
from .models import BuildingAttributes
from .prior import osm_prior
from .regional import region_at
from .vlm import Crop, OpenAICompatibleClient, VLMProvider, inputs_fingerprint, no_clean_view_result

PRIOR_PROVIDER = "osm-tag-prior"


def provider_slug(provider: str) -> str:
    """File-name form of a provider name: ``vlm:Qwen/Qwen3-VL-4B-Instruct`` -> ``vlm_Qwen_Qwen3-VL-4B-Instruct``."""
    return provider.replace(":", "_").replace("/", "_")


def attributes_path(dataset_dir: Path, building_id: str, provider: str) -> Path:
    return dataset_dir / "attributes" / Path(*_building_parts(building_id)) / f"attributes-{provider_slug(provider)}.json"


def read_attributes(dataset_dir: Path, building_id: str, provider: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(attributes_path(dataset_dir, building_id, provider).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_attributes(dataset_dir: Path, result: BuildingAttributes) -> Path:
    path = attributes_path(dataset_dir, result.building_id, result.provider)
    atomic_write_bytes(path, (json.dumps(result.to_dict(), indent=2) + "\n").encode("utf-8"))
    return path


def osm_tags(dataset_dir: Path, manifest: dict[str, Any], building_id: str) -> dict[str, Any] | None:
    """Tags of the footprint as OSM serves them, found through the building's analyzed geometry.

    Looked up first where the analysis looked them up, at the camera (the same
    cached OSM cell, so no network request), then at the footprint centroid.
    """
    try:
        view = collect_building_evidence(dataset_dir, manifest, building_id, expected_analysis_version=ANALYSIS_VERSION).views[0]
    except BuildingEvidenceNotFound:
        return None
    centroid = shape(view.footprint_geometry).centroid
    for latitude, longitude in ((view.recorded_latitude, view.recorded_longitude), (centroid.y, centroid.x)):
        for feature in nearby_buildings(latitude, longitude):
            if feature["properties"].get("osm_id") == building_id:
                return dict(feature["properties"])
    return None


def capture_context(image: dict[str, Any] | None) -> str | None:
    """Where and when a panorama was taken, in words a model can use to place a building in a region and an era.

    The country is named: from coordinates alone a model guesses the city, and guesses wrong.
    """
    if not image:
        return None
    parts = []
    coordinates = panorama_coordinates(image)
    if coordinates is not None:
        region = region_at(*coordinates)
        parts.append(f"photo taken{f' in {region.name}' if region else ''} at latitude {coordinates[1]:.4f}, longitude {coordinates[0]:.4f}")
    captured = image.get("captured_at_ms")
    if isinstance(captured, (int, float)) and not isinstance(captured, bool) and captured > 0:
        parts.append(f"in {datetime.fromtimestamp(captured / 1000, UTC):%Y-%m}")
    return " ".join(parts) or None


def load_crops(dataset_dir: Path, building_id: str, images_by_id: dict[str, dict[str, Any]] | None = None) -> list[Crop]:
    """Exported crops of a building, best first; ``images_by_id`` (manifest records) adds the capture context."""
    crops = []
    for sidecar_path in sorted(crops_directory(dataset_dir, building_id).glob("crop-*.json")):
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        image_path = sidecar_path.with_suffix(".jpg")
        image_id = str(sidecar["source"]["image_id"])
        reference = {
            "image_id": image_id, "crop_sha256": sidecar["crop"]["sha256"], "rank": sidecar["evidence"]["rank"],
            "crop_file": image_path.relative_to(dataset_dir).as_posix(),
            # Where the crop looks in the panorama, so a viewer can turn there while the model reads it.
            "view": {key: sidecar["view"][key] for key in ("yaw_deg", "pitch_deg", "fov_deg") if key in sidecar.get("view", {})},
        }
        image = (images_by_id or {}).get(image_id)
        context = capture_context(image) if images_by_id is not None else None
        coordinates = panorama_coordinates(image) if image else None
        region = region_at(*coordinates) if coordinates else None
        crops.append(Crop(image_path.read_bytes(), float(sidecar["evidence"]["score"]), reference, context, region.id if region else None))
    return crops


def predict_prior(dataset_dir: Path, manifest: dict[str, Any], building_id: str) -> str:
    """Write the OSM-tag prior of one building; returns ``written`` or ``no_tags``."""
    tags = osm_tags(dataset_dir, manifest, building_id)
    if tags is None:
        return "no_tags"
    write_attributes(dataset_dir, osm_prior(building_id, tags))
    return "written"


def predict_vlm(
    provider: VLMProvider, dataset_dir: Path, building_id: str, images_by_id: dict[str, dict[str, Any]], force: bool = False,
) -> tuple[str, BuildingAttributes | None]:
    """Ask the model about one building's exported crops, unless its inputs did not change since the last answer.

    Returns ``(status, result)``: ``written``, ``unchanged``, ``no_clean_view`` (every view's mask was flagged:
    the model is not asked and its result abstains, saying why) or ``no_crops``.
    """
    crops = load_crops(dataset_dir, building_id, images_by_id)
    previous = read_attributes(dataset_dir, building_id, provider.name)
    if not crops:
        set_aside = set_aside_views(dataset_dir, building_id)
        if not set_aside:
            return "no_crops", None
        result = no_clean_view_result(building_id, provider.name, set_aside)
        if not force and previous and (previous.get("run") or {}).get("inputs_sha256") == result.run["inputs_sha256"]:
            return "unchanged", None
        write_attributes(dataset_dir, result)
        return "no_clean_view", result
    if not force and previous and (previous.get("run") or {}).get("inputs_sha256") == inputs_fingerprint(provider.name, crops):
        return "unchanged", None
    result = provider.predict(building_id, crops)
    write_attributes(dataset_dir, result)
    return "written", result


@dataclass(frozen=True)
class ModelSettings:
    """Which vision-language model to use and how to reach it; the same for the CLI, the survey and the viewer."""

    backend: str
    model: str
    url: str | None = None
    token: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    cache_dir: Path | None = None
    dtype: str = "auto"
    batch_size: int = 4

    @property
    def provider_name(self) -> str:
        return f"vlm:{self.model}"

    def client(self) -> Any:
        """``openai``: a model server at ``url`` (vLLM, Ollama, LM Studio, a cloud API); ``transformers``: the checkpoint in this process."""
        if self.backend == "openai":
            if not self.url:
                raise ValueError("the openai backend needs a server URL (--vlm-url or ATTRIBUTES_VLM_URL)")
            return OpenAICompatibleClient(self.url, self.model, self.token, options=self.options)
        if self.backend == "transformers":
            from .local_vlm import TransformersVLM

            return TransformersVLM(self.model, cache_dir=self.cache_dir, dtype=self.dtype, batch_size=self.batch_size)
        raise ValueError(f"unknown model backend {self.backend!r}")


def _options(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        options = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"model options must be a JSON object: {error}") from error
    if not isinstance(options, dict):
        raise ValueError("model options must be a JSON object")
    return options


def model_settings_from_environment(model_root: Path | None = None) -> ModelSettings | None:
    """The viewer's assistant model, from ATTRIBUTES_VLM_*; ``None`` when none is configured."""
    from .local_vlm import DEFAULT_MODEL

    url, backend = os.environ.get("ATTRIBUTES_VLM_URL"), os.environ.get("ATTRIBUTES_VLM_BACKEND")
    if not url and not backend:
        return None
    return ModelSettings(
        backend=backend or "openai", model=os.environ.get("ATTRIBUTES_VLM_MODEL", DEFAULT_MODEL), url=url,
        token=os.environ.get("ATTRIBUTES_VLM_TOKEN"), options=_options(os.environ.get("ATTRIBUTES_VLM_OPTIONS")),
        cache_dir=Path(os.environ["ATTRIBUTES_VLM_CACHE_DIR"]) if os.environ.get("ATTRIBUTES_VLM_CACHE_DIR") else model_root,
    )


def model_settings_from_args(args: argparse.Namespace) -> ModelSettings:
    return ModelSettings(
        backend=args.vlm_backend, model=args.vlm_model, url=args.vlm_url, token=os.environ.get("ATTRIBUTES_VLM_TOKEN"),
        options=_options(args.vlm_options), cache_dir=args.vlm_cache_dir or args.model_root, dtype=args.vlm_dtype, batch_size=args.vlm_batch_size,
    )


def add_vlm_arguments(parser: argparse.ArgumentParser) -> None:
    from .local_vlm import DEFAULT_MODEL, DTYPES

    group = parser.add_argument_group("vision-language model")
    group.add_argument(
        "--vlm-backend", choices=("transformers", "openai"), default=os.environ.get("ATTRIBUTES_VLM_BACKEND", "openai" if os.environ.get("ATTRIBUTES_VLM_URL") else "transformers"),
        help="transformers: load the model in this process (GPU recommended); openai: call an OpenAI-compatible server such as vLLM or Ollama (env ATTRIBUTES_VLM_BACKEND)",
    )
    group.add_argument("--vlm-model", default=os.environ.get("ATTRIBUTES_VLM_MODEL", DEFAULT_MODEL), help="Hugging Face ID or served model name (env ATTRIBUTES_VLM_MODEL; default: %(default)s)")
    group.add_argument("--vlm-url", default=os.environ.get("ATTRIBUTES_VLM_URL"), help="server base URL for the openai backend, e.g. http://127.0.0.1:11434 for Ollama (env ATTRIBUTES_VLM_URL)")
    group.add_argument("--vlm-options", default=os.environ.get("ATTRIBUTES_VLM_OPTIONS"), help='extra JSON request fields for the server, e.g. \'{"reasoning_effort": "none"}\' for Ollama reasoning models (env ATTRIBUTES_VLM_OPTIONS)')
    group.add_argument("--vlm-dtype", choices=DTYPES, default="auto", help="weights precision for the transformers backend (auto: bfloat16 where supported, else float16 on GPU)")
    group.add_argument("--vlm-batch-size", type=int, default=4, help="views per forward pass for the transformers backend")
    group.add_argument("--vlm-cache-dir", type=Path, default=None, help="where the transformers backend keeps the model weights (default: --model-root)")
    group.add_argument("--min-views", type=int, default=1, help="views a building needs before the model's answer is kept")
    group.add_argument("--min-share", type=float, default=0.5, help="top vote share below which an attribute abstains")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict building attributes for an analyzed dataset (prior from OSM tags, or a vision-language model).")
    parser.add_argument("dataset", help="dataset directory name under --data-root")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-root", type=Path, default=Path("models"), help="where the transformers backend keeps model weights")
    parser.add_argument("--provider", choices=("prior", "vlm"), default="prior")
    parser.add_argument("--building", action="append", help="OSM ID (way/123); default: every analyzed building")
    parser.add_argument("--per-building", type=int, default=3, help="crops per building for the vlm provider")
    parser.add_argument("--force", action="store_true", help="ask the model again even when a building's crops did not change")
    add_vlm_arguments(parser)
    args = parser.parse_args()
    from ..local_viewer import dataset_manifest

    try:
        manifest = dataset_manifest(args.data_root, args.dataset)
    except ValueError as error:
        parser.error(str(error))
    dataset_dir = (args.data_root / args.dataset).resolve()
    buildings = sorted(set(args.building)) if args.building else analyzed_building_ids(dataset_dir)
    provider: VLMProvider | None = None
    if args.provider == "vlm":
        try:
            settings = model_settings_from_args(args)
            client = settings.client()
        except ValueError as error:
            parser.error(str(error))
        provider = VLMProvider(client, settings.provider_name, min_views=args.min_views, min_share=args.min_share)
        export_building_crops(args.data_root, args.dataset, buildings, per_building=args.per_building, mask_others=True)
    images_by_id = {str(image["id"]): image for image in manifest["images"] if isinstance(image, dict) and "id" in image}
    counts: dict[str, int] = {}
    abstained = 0
    for building_id in buildings:
        if provider is None:
            status = predict_prior(dataset_dir, manifest, building_id)
            if status == "no_tags":
                print(f"{building_id}: no current analysis or footprint tags (same BUILDING_ANALYSIS_* settings as the analysis?), skipped")
        else:
            status, result = predict_vlm(provider, dataset_dir, building_id, images_by_id, args.force)
            if status == "no_crops":
                print(f"{building_id}: no crops, skipped")
            if result is not None:
                abstained += sum(prediction.abstained for prediction in result.predictions.values())
        counts[status] = counts.get(status, 0) + 1
    summary = ", ".join(f"{count} {status.replace('_', ' ')}" for status, count in sorted(counts.items()))
    print(f"{len(buildings)} buildings under {dataset_dir / 'attributes'}: {summary or 'none'}; {abstained} attribute abstentions in new answers")


if __name__ == "__main__":
    main()
