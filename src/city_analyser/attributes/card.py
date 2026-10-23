"""What the viewer shows for one building: every provider's attributes, merged by one fixed rule.

A reviewer's decision wins, then a human-entered OSM tag, then the newest
vision-language answer. The other providers' values stay in the card, so a reader
sees when a model disagrees with OSM, and an abstention is shown with its
reason instead of a value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..building_inspection.reporting import _building_parts
from .taxonomy import ALL_ATTRIBUTES, BUILDING_TYPES, GEM_LLRS, GEM_MATERIALS, GEM_TAXONOMY_VERSION, SURFACE_MATERIALS, TAXONOMY_VERSION, gem_taxonomy_string

CARD_VERSION = 2
MAX_ALTERNATIVES = 2
MAX_OBSERVATIONS = 3
# The Sede Electrónica del Catastro page of a parcel (the first 14 characters of its reference).
CADASTRE_URL = "https://www1.sedecatastro.gob.es/CYCBienInmueble/OVCListaBienes.aspx?rc1={rc1}&rc2={rc2}"
_reference_cache: dict[Path, tuple[float, dict[str, Any]]] = {}


def references(dataset_dir: Path, building_id: str) -> dict[str, Any]:
    """Reference data shown next to the answers: the cadastre's storeys, year and use, if matched.

    Shown to the reviewer, never to the model: the experiments score the model against it.
    """
    path = dataset_dir / "groundtruth" / "catastro.json"
    try:
        modified = path.stat().st_mtime
    except OSError:
        return {}
    cached = _reference_cache.get(path)
    if cached is None or cached[0] != modified:
        try:
            buildings = json.loads(path.read_text(encoding="utf-8")).get("buildings") or {}
        except (OSError, json.JSONDecodeError, AttributeError):
            return {}
        cached = _reference_cache[path] = (modified, buildings)
    record = cached[1].get(building_id)
    if not isinstance(record, dict):
        return {}
    reference = str(record.get("reference") or "")
    return {"catastro": {
        **{key: record.get(key) for key in ("floors", "construction_year", "building_type", "use", "dwellings", "reference", "iou")},
        "url": CADASTRE_URL.format(rc1=quote(reference[:7]), rc2=quote(reference[7:14])) if len(reference) >= 14 else None,
        "note": "Spanish cadastre: storeys above ground, year of construction (may be a refurbishment) and main use.",
    }}


def _summary(prediction: dict[str, Any], provider: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"provider": provider, "source": prediction.get("source"), "abstained": bool(prediction.get("abstained")), "reason": prediction.get("reason")}
    if prediction.get("type") == "numeric":
        summary.update(value=prediction.get("value"), spread=prediction.get("spread"))
        return summary
    ranked = sorted(((name, float(score)) for name, score in (prediction.get("scores") or {}).items() if score), key=lambda item: (-item[1], item[0]))
    summary.update(
        value=ranked[0][0] if ranked and not summary["abstained"] else None, share=round(ranked[0][1], 4) if ranked else None,
        score_kind=prediction.get("score_kind"),
        alternatives=[{"value": name, "share": round(score, 4)} for name, score in ranked[1:1 + MAX_ALTERNATIVES]],
    )
    if summary["abstained"] and ranked:
        summary["leaning"] = ranked[0][0]  # what the split vote leaned to, shown only as such
    return summary


def _load(directory: Path, building_id: str) -> list[tuple[float, dict[str, Any]]]:
    payloads = []
    for path in directory.glob("attributes-*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            modified = path.stat().st_mtime
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and (payload.get("building") or {}).get("id") == building_id and isinstance(payload.get("predictions"), dict):
            payloads.append((modified, payload))
    return payloads


def building_card(dataset_dir: Path, dataset_id: str, building_id: str, *, with_references: bool = False) -> dict[str, Any] | None:
    """The merged attributes of one building, or ``None`` when no provider has written any.

    ``with_references`` adds the reference data (the cadastre) for evaluation. It is off by
    default: the analysis works without reference data, so the viewer shows it only when asked.
    """
    loaded = _load(dataset_dir / "attributes" / Path(*_building_parts(building_id)), building_id)
    if not loaded:
        return None
    # Reviews, then OSM tags, then models, newest first: the order in which a value is preferred.
    rank = {"human-review": 0, "osm-tag-prior": 1}
    loaded.sort(key=lambda item: (rank.get(str(item[1].get("provider")), 2), -item[0], str(item[1].get("provider"))))
    attributes: dict[str, Any] = {}
    for attribute in ALL_ATTRIBUTES:
        summaries = [_summary(payload["predictions"][attribute], str(payload.get("provider"))) for _, payload in loaded if isinstance(payload["predictions"].get(attribute), dict)]
        if not summaries:
            continue
        # A reviewer who could not tell overrides a model that claimed to.
        reviewed = summaries[0] if summaries[0]["provider"] == "human-review" else None
        chosen = reviewed or next((summary for summary in summaries if not summary["abstained"] and summary.get("value") is not None), summaries[-1])
        entry = {**chosen, "others": [summary for summary in summaries if summary is not chosen]}
        meanings = {"gem_material": GEM_MATERIALS, "gem_llrs": GEM_LLRS}.get(attribute, {})
        if chosen.get("value") in meanings:
            entry["meaning"] = meanings[chosen["value"]]
        attributes[attribute] = entry
    observations = []
    for _, payload in loaded:
        for view in payload.get("evidence") or []:
            text = (view.get("answer") or {}).get("observations")
            if not isinstance(text, str) or len(observations) >= MAX_OBSERVATIONS:
                continue
            crop = view.get("crop_file")
            observations.append({
                "provider": payload.get("provider"), "image_id": view.get("image_id"), "rank": view.get("rank"), "text": text,
                "crop_url": f"/data/{quote(dataset_id, safe='')}/{quote(crop)}" if isinstance(crop, str) and crop.startswith("crops/") else None,
            })
    observations.sort(key=lambda item: (item["rank"] is None, item["rank"] or 0))
    review = next((payload.get("run") for _, payload in loaded if payload.get("provider") == "human-review"), None)
    model = next((payload for _, payload in loaded if str(payload.get("provider", "")).startswith("vlm:")), None)
    views = sorted((model or {}).get("evidence") or [], key=lambda view: (view.get("rank") is None, view.get("rank") or 0))
    known = {name: entry.get("value") for name, entry in attributes.items() if not entry.get("abstained")}
    view_details = []
    for view in views:
        crop = view.get("crop_file")
        angles, score = None, None
        if isinstance(crop, str) and crop.startswith("crops/"):
            # The crop's framing is the panorama direction that shows the building; it moves the viewer's camera there.
            try:
                sidecar = json.loads((dataset_dir / crop).with_suffix(".json").read_text(encoding="utf-8"))
                angles = {key: sidecar["view"][key] for key in ("yaw_deg", "pitch_deg", "fov_deg")}
                score = sidecar["evidence"].get("score")
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                angles = None
        answer = view.get("answer") or {}
        view_details.append({
            "image_id": view.get("image_id"), "rank": view.get("rank"), "view": angles, "score": score, "error": view.get("error"),
            "crop_url": f"/data/{quote(dataset_id, safe='')}/{quote(crop)}" if isinstance(crop, str) and crop.startswith("crops/") else None,
            "answer": {key: answer.get(key) for key in ("observations", "floors", "gem_material", "gem_llrs", "building_type", "construction_year") if answer.get(key) is not None},
            "reasoning": view.get("reasoning"),
        })
    gem = {
        "string": gem_taxonomy_string(known.get("gem_material"), known.get("gem_llrs"), known.get("floors")),
        "complete": all(known.get(name) is not None for name in ("gem_material", "gem_llrs", "floors")),
        "taxonomy": GEM_TAXONOMY_VERSION,
        "codes": {**GEM_MATERIALS, **GEM_LLRS},
    }
    return {
        "card_version": CARD_VERSION, "taxonomy_version": TAXONOMY_VERSION, "building": building_id, "gem": gem,
        "providers": [str(payload.get("provider")) for _, payload in loaded],
        "rule": "A review wins, then an OpenStreetMap tag, then the newest model answer.",
        "attributes": attributes, "observations": observations,
        "review": {key: review.get(key) for key in ("reviewer", "note", "reviewed_at", "evaluation")} if review else None,
        "best_image": views[0].get("image_id") if views else None,
        "view_errors": sum("error" in view for view in views),
        "model": (model or {}).get("provider"),
        # The classes a reviewer can choose from (GEM codes are in gem.codes).
        "classes": {"building_type": list(BUILDING_TYPES), "surface_material": list(SURFACE_MATERIALS)},
        "views": view_details,
        **({"references": references(dataset_dir, building_id)} if with_references else {}),
    }
