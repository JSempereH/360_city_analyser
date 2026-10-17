"""Baseline without images: attributes read from the footprint's own OSM tags.

This is the baseline any visual provider has to beat. A missing or ambiguous
tag is an abstention, never a guess.
"""

from __future__ import annotations

from typing import Any, Mapping

from .models import BuildingAttributes, CategoricalPrediction, NumericPrediction, abstain_categorical, abstain_numeric
from .taxonomy import BUILDING_TYPES, SURFACE_MATERIALS, floors_from_osm, material_from_osm, type_from_osm, year_from_osm

SOURCE = "osm_tag"


def _tag_prediction(attribute: str, classes: tuple[str, ...], label: str | None, tag: str) -> CategoricalPrediction:
    if label is None:
        return abstain_categorical(attribute, SOURCE, f"no unambiguous {tag} tag")
    return CategoricalPrediction(attribute, {name: 1.0 if name == label else 0.0 for name in classes}, "tag", SOURCE)


def osm_prior(building_id: str, tags: Mapping[str, Any]) -> BuildingAttributes:
    floors, year = floors_from_osm(tags.get("building:levels")), year_from_osm(tags.get("start_date"))
    predictions = {
        "building_type": _tag_prediction("building_type", BUILDING_TYPES, type_from_osm(tags.get("building")), "building"),
        "surface_material": _tag_prediction("surface_material", SURFACE_MATERIALS, material_from_osm(tags.get("building:material")), "building:material"),
        # OSM has no established tag for the load-bearing structure, and wall
        # material does not imply it (brick infill hides concrete frames).
        "gem_material": abstain_categorical("gem_material", SOURCE, "OpenStreetMap has no structural material tag"),
        "gem_llrs": abstain_categorical("gem_llrs", SOURCE, "OpenStreetMap has no structural system tag"),
        "floors": NumericPrediction("floors", floors, 0.0, SOURCE) if floors is not None else abstain_numeric("floors", SOURCE, "no building:levels tag"),
        "construction_year": NumericPrediction("construction_year", float(year), 0.0, SOURCE) if year is not None else abstain_numeric("construction_year", SOURCE, "no start_date tag"),
    }
    return BuildingAttributes(building_id, "osm-tag-prior", predictions)
