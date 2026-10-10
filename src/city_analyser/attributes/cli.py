"""city-analyser-attributes: attributes for every analyzed building of a dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from shapely.geometry import shape

from ..analysis_cache import atomic_write_bytes
from ..building_analysis import ANALYSIS_VERSION, nearby_buildings
from ..building_inspection.crops import analyzed_building_ids, crops_directory, export_building_crops
from ..building_inspection.evidence import BuildingEvidenceNotFound, collect_building_evidence
from ..building_inspection.reporting import _building_parts
from .models import BuildingAttributes
from .prior import osm_prior
from .vlm import Crop, OpenAICompatibleClient, VLMProvider


def attributes_path(dataset_dir: Path, building_id: str, provider: str) -> Path:
    return dataset_dir / "attributes" / Path(*_building_parts(building_id)) / f"attributes-{provider}.json"


def osm_tags(dataset_dir: Path, manifest: dict[str, Any], building_id: str) -> dict[str, Any] | None:
    """Tags of the footprint as OSM serves them, found through the building's analyzed geometry."""
    try:
        view = collect_building_evidence(dataset_dir, manifest, building_id, expected_analysis_version=ANALYSIS_VERSION).views[0]
    except BuildingEvidenceNotFound:
        return None
    centroid = shape(view.footprint_geometry).centroid
    for feature in nearby_buildings(centroid.y, centroid.x):
        if feature["properties"].get("osm_id") == building_id:
            return dict(feature["properties"])
    return None


def load_crops(dataset_dir: Path, building_id: str) -> list[Crop]:
    crops = []
    for sidecar_path in sorted(crops_directory(dataset_dir, building_id).glob("crop-*.json")):
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        image_path = sidecar_path.with_suffix(".jpg")
        reference = {"image_id": sidecar["source"]["image_id"], "crop_sha256": sidecar["crop"]["sha256"], "rank": sidecar["evidence"]["rank"]}
        crops.append(Crop(image_path.read_bytes(), float(sidecar["evidence"]["score"]), reference))
    return crops


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict building attributes for an analyzed dataset (prior from OSM tags, or a vision-language model).")
    parser.add_argument("dataset", help="dataset directory name under --data-root")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--provider", choices=("prior", "vlm"), default="prior")
    parser.add_argument("--building", action="append", help="OSM ID (way/123); default: every analyzed building")
    parser.add_argument("--vlm-url", default=os.environ.get("ATTRIBUTES_VLM_URL"), help="OpenAI-compatible server base URL (env ATTRIBUTES_VLM_URL)")
    parser.add_argument("--vlm-model", default=os.environ.get("ATTRIBUTES_VLM_MODEL"), help="served model name (env ATTRIBUTES_VLM_MODEL)")
    parser.add_argument("--per-building", type=int, default=3, help="crops per building for the vlm provider")
    parser.add_argument("--min-views", type=int, default=1)
    parser.add_argument("--min-share", type=float, default=0.5, help="top vote share below which a building abstains")
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
        if not args.vlm_url or not args.vlm_model:
            parser.error("--provider vlm needs --vlm-url and --vlm-model (or ATTRIBUTES_VLM_URL and ATTRIBUTES_VLM_MODEL)")
        client = OpenAICompatibleClient(args.vlm_url, args.vlm_model, os.environ.get("ATTRIBUTES_VLM_TOKEN"))
        provider = VLMProvider(client, f"vlm:{args.vlm_model}", min_views=args.min_views, min_share=args.min_share)
        export_building_crops(args.data_root, args.dataset, buildings, per_building=args.per_building)
    written = abstained = 0
    for building_id in buildings:
        if provider is None:
            tags = osm_tags(dataset_dir, manifest, building_id)
            if tags is None:
                print(f"{building_id}: no current analysis or footprint tags (same BUILDING_ANALYSIS_* settings as the analysis?), skipped")
                continue
            result: BuildingAttributes = osm_prior(building_id, tags)
        else:
            crops = load_crops(dataset_dir, building_id)
            if not crops:
                print(f"{building_id}: no crops, skipped")
                continue
            result = provider.predict(building_id, crops)
        path = attributes_path(dataset_dir, building_id, result.provider.replace(":", "_").replace("/", "_"))
        atomic_write_bytes(path, (json.dumps(result.to_dict(), indent=2) + "\n").encode("utf-8"))
        written += 1
        abstained += sum(prediction.abstained for prediction in result.predictions.values())
    print(f"{written} buildings written under {dataset_dir / 'attributes'}; {abstained} attribute abstentions")


if __name__ == "__main__":
    main()
