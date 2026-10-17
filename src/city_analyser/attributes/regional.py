"""What an engineer knows about how a region builds, given to the model with each view.

A street-level photo rarely shows the structure: render, cladding and shop
fronts hide it. Engineers fill the gap with the construction history of the
place (which systems were built in which period) and read the facade for the
cues that confirm or contradict it. The model gets the same knowledge, chosen
by the camera position. It never gets OSM tags, which the evaluation scores
against; construction history is general knowledge, not a label.

Bump ``GUIDANCE_VERSION`` when a text changes: it is part of the prompt fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass

GUIDANCE_VERSION = 1

GENERAL = (
    "How to judge the structure: the facade material is what covers the walls, not the structure. "
    "Read the proportion of openings to solid piers, the wall thickness visible at window reveals, slab edges or "
    "bands at every floor, the regularity of bays and any columns. Openings stacked exactly above each other in "
    "narrow bays between thick piers point to load-bearing walls (LWAL); wide openings, thin piers, ribbon windows "
    "or slab edges expressed at every floor point to a frame (LFINF when masonry fills it). A ground floor opened "
    "for shop fronts or parking under solid upper floors is a soft storey: always say so, it matters for "
    "earthquakes. Count storeys by window rows, including the ground floor, a mezzanine with its own windows and "
    "an attic with windows. Monuments, towers and churches have no countable storeys: say so and give null."
)


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    # (west, south, east, north) boxes; a camera inside any of them belongs to the region.
    boxes: tuple[tuple[float, float, float, float], ...]
    guidance: str
    # Boxes inside ``boxes`` that belong to another country.
    excluded: tuple[tuple[float, float, float, float], ...] = ()


SPAIN = Region(
    "es", "Spain",
    boxes=((-9.4, 35.9, 3.4, 43.9), (1.1, 38.6, 4.4, 40.1), (-18.2, 27.6, -13.4, 29.5)),
    excluded=((-9.6, 36.9, -6.2, 42.2),),  # mainland Portugal
    guidance=(
        "Construction history of Spain, use it to weigh what the facade shows: historic centres built before about "
        "1940 are almost always unreinforced load-bearing masonry (MUR, LWAL): solid brick or rubble stone walls with "
        "lime mortar and timber or iron-joist floors, often with cast-iron columns behind ground-floor shops. "
        "Between about 1940 and 1960 steel was scarce: load-bearing brick and early reinforced concrete frames both "
        "occur, so be less certain and give the other as the alternate. From the 1960s most buildings are reinforced "
        "concrete frames with brick infill (CR, LFINF); from the 1970s waffle or flat slabs on columns without beams "
        "are very common (CR, LFLSINF). Steel frames (S) are rare outside recent offices and industrial sheds."
    ),
)

REGIONS = (SPAIN,)


def _inside(box: tuple[float, float, float, float], longitude: float, latitude: float) -> bool:
    west, south, east, north = box
    return west <= longitude <= east and south <= latitude <= north


def region_at(longitude: float | None, latitude: float | None) -> Region | None:
    if longitude is None or latitude is None:
        return None
    for region in REGIONS:
        if any(_inside(box, longitude, latitude) for box in region.boxes) and not any(_inside(box, longitude, latitude) for box in region.excluded):
            return region
    return None


def guidance_for(region_id: str | None) -> str:
    """General rules plus the region's construction history, when the region is known."""
    region = next((item for item in REGIONS if item.id == region_id), None)
    return f"{GENERAL} {region.guidance}" if region else GENERAL
