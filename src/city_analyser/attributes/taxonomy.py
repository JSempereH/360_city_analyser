"""Attribute taxonomy (OpenFACADES classes) and OSM tag parsing."""

from __future__ import annotations

import re
from typing import Any

TAXONOMY_VERSION = "openfacades-v1"
BUILDING_TYPES = ("apartments", "house", "retail", "office", "hotel", "industrial", "religious", "education", "public", "garage")
SURFACE_MATERIALS = ("brick", "wood", "concrete", "metal", "stone", "glass", "plaster")
CATEGORICAL_ATTRIBUTES: dict[str, tuple[str, ...]] = {"building_type": BUILDING_TYPES, "surface_material": SURFACE_MATERIALS}
NUMERIC_ATTRIBUTES = ("floors", "construction_year")
ALL_ATTRIBUTES = (*CATEGORICAL_ATTRIBUTES, *NUMERIC_ATTRIBUTES)

# OSM ``building=*`` values that name a class unambiguously. ``yes`` and
# ``residential`` are deliberately absent: they do not say apartments or house.
_TYPE_BY_OSM_VALUE = {
    "apartments": "apartments", "dormitory": "apartments",
    "house": "house", "detached": "house", "semidetached_house": "house", "terrace": "house", "bungalow": "house",
    "retail": "retail", "supermarket": "retail", "kiosk": "retail",
    "office": "office", "commercial": "office",
    "hotel": "hotel",
    "industrial": "industrial", "warehouse": "industrial", "factory": "industrial",
    "church": "religious", "chapel": "religious", "cathedral": "religious", "mosque": "religious", "temple": "religious",
    "synagogue": "religious", "religious": "religious", "monastery": "religious",
    "school": "education", "university": "education", "college": "education", "kindergarten": "education",
    "public": "public", "civic": "public", "government": "public", "hospital": "public", "train_station": "public", "fire_station": "public",
    "garage": "garage", "garages": "garage", "parking": "garage", "carport": "garage",
}
_MATERIAL_BY_OSM_VALUE = {
    "brick": "brick", "wood": "wood", "timber_framing": "wood", "concrete": "concrete", "reinforced_concrete": "concrete",
    "metal": "metal", "steel": "metal", "stone": "stone", "glass": "glass", "plaster": "plaster", "stucco": "plaster",
}
_YEAR = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)")


def type_from_osm(value: Any) -> str | None:
    return _TYPE_BY_OSM_VALUE.get(str(value).strip().lower()) if value is not None else None


def material_from_osm(value: Any) -> str | None:
    return _MATERIAL_BY_OSM_VALUE.get(str(value).strip().lower()) if value is not None else None


def floors_from_osm(value: Any) -> float | None:
    try:
        floors = float(str(value).replace(",", ".").strip())
    except ValueError:
        return None
    return floors if floors >= 0 and floors == floors else None


def year_from_osm(value: Any) -> int | None:
    """First plausible four-digit year of ``start_date`` ("1905", "1905-03-01", "~1900", "1900s")."""
    match = _YEAR.search(str(value)) if value is not None else None
    return int(match.group(1)) if match else None
