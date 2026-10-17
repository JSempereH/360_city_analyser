"""Scores for attribute predictions, with the baselines they must beat.

    uv run python -m evaluation.attributes --dataset valencia-centro --provider osm-tag-prior

Labels here come from OSM tags, which are noisy and cover few buildings: the
scores say how well a provider agrees with OSM where OSM has an answer, not how
accurate it is. A visual provider is only worth adopting if it beats the
majority-class baseline and, where tags exist, the tag prior (see the roadmap).
Abstentions lower coverage, not accuracy, and are always reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence



# The scores live in the package so experiments can use them; re-exported for this script and its tests.
from city_analyser.experiments.metrics import categorical_scores, evaluate, numeric_scores, risk_coverage  # noqa: E402,F401


def load_predictions(dataset_dir: Path, provider: str) -> dict[str, dict[str, Any]]:
    predictions = {}
    for path in sorted((dataset_dir / "attributes").rglob(f"attributes-{provider}.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        predictions[payload["building"]["id"]] = payload["predictions"]
    return predictions


def truth_from_osm_tags(dataset_dir: Path, building_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    from city_analyser.attributes.cli import osm_tags
    from city_analyser.attributes.prior import osm_prior
    from city_analyser.local_viewer import dataset_manifest

    manifest = dataset_manifest(dataset_dir.parent, dataset_dir.name)
    truth = {}
    for building_id in building_ids:
        tags = osm_tags(dataset_dir, manifest, building_id)
        if tags is None:
            continue
        prior = osm_prior(building_id, tags).to_dict()["predictions"]
        truth[building_id] = {
            "building_type": max(prior["building_type"]["scores"], key=prior["building_type"]["scores"].get) if not prior["building_type"]["abstained"] else None,
            "surface_material": max(prior["surface_material"]["scores"], key=prior["surface_material"]["scores"].get) if not prior["surface_material"]["abstained"] else None,
            "floors": prior["floors"]["value"], "construction_year": prior["construction_year"]["value"],
        }
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--provider", required=True, help="file suffix of attributes-<provider>.json (e.g. osm-tag-prior, vlm_<model>)")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    dataset_dir = (args.data_root / args.dataset).resolve()
    predictions = load_predictions(dataset_dir, args.provider)
    if not predictions:
        raise SystemExit(f"no attributes-{args.provider}.json files under {dataset_dir / 'attributes'}; run city-analyser-attributes first")
    report = evaluate(predictions, truth_from_osm_tags(dataset_dir, sorted(predictions)))
    (dataset_dir / "attributes" / f"evaluation-{args.provider}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
