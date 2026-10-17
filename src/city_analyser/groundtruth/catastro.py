"""Reference values from the Spanish cadastre (INSPIRE buildings) for a dataset's buildings.

The cadastre publishes, per municipality, every building's footprint with its
construction year, use and dwellings, and storeys per building part. This
module finds the municipality from the panorama positions, downloads its
buildings once, reads only those inside the dataset's area (streaming, so
memory stays low for large cities) and matches them one to one with the
dataset's OSM footprints: a pair must be each other's best match and overlap
with an intersection over union of at least ``MIN_IOU``. Unmatched footprints
are left out, never guessed.

Data: Dirección General del Catastro, INSPIRE download service (free of
charge, with attribution to the D. G. del Catastro as author and owner).
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from xml.etree import ElementTree

FEED_URL = "https://www.catastro.hacienda.gob.es/INSPIRE/buildings/ES.SDGC.BU.atom.xml"
ATTRIBUTION = "Dirección General del Catastro, INSPIRE buildings (author and owner of the information)"
MIN_IOU = 0.5
# Area around the panoramas whose cadastral buildings are read, in meters.
MARGIN_M = 150.0
USE_TO_TYPE = {"3_industrial": "industrial", "4_1_office": "office", "4_2_retail": "retail", "4_3_publicServices": "public"}
_YEAR = re.compile(r"^(\d{4})")


def _get(url: str, timeout: int = 300) -> bytes:
    # The feeds list http:// links that answer with an empty body; the same paths work over https.
    request = Request(url.replace("http://", "https://", 1), headers={"User-Agent": "360-city-analyser/1.0"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed public endpoint
        return response.read()


def _entries(feed: bytes) -> Iterator[dict[str, Any]]:
    text = feed.decode("latin-1")
    for entry in re.findall(r"<entry>(.*?)</entry>", text, re.DOTALL):
        title = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
        link = re.search(r'<link[^>]*rel="enclosure"[^>]*href="([^"]+)"', entry) or re.search(r'href="([^"]+)"', entry)
        polygon = re.search(r"<georss:polygon>(.*?)</georss:polygon>", entry, re.DOTALL)
        epsg = re.search(r"EPSG/0/(\d+)", entry)
        if not (title and link and polygon):
            continue
        values = [float(value) for value in polygon.group(1).split()]
        latitudes, longitudes = values[0::2], values[1::2]
        yield {"title": title.group(1).strip(), "url": link.group(1), "bbox": (min(longitudes), min(latitudes), max(longitudes), max(latitudes)),
               "epsg": int(epsg.group(1)) if epsg else None}


def _contains(bbox: tuple[float, float, float, float], longitude: float, latitude: float) -> bool:
    return bbox[0] <= longitude <= bbox[2] and bbox[1] <= latitude <= bbox[3]


def find_municipality(longitude: float, latitude: float, cache_dir: Path) -> dict[str, Any]:
    """The municipality whose bounding box is the smallest containing the point: code, name, zip URL, CRS."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cached(name: str, url: str) -> bytes:
        path = cache_dir / name
        if not path.is_file():
            path.write_bytes(_get(url))
        return path.read_bytes()

    candidates = []
    for office in _entries(cached("feed.xml", FEED_URL)):
        if not _contains(office["bbox"], longitude, latitude):
            continue
        code = re.search(r"/(\d{2})/[^/]*$", office["url"])
        for municipality in _entries(cached(f"feed-{code.group(1) if code else 'x'}.xml", office["url"])):
            if municipality["url"].endswith(".zip") and _contains(municipality["bbox"], longitude, latitude):
                west, south, east, north = municipality["bbox"]
                candidates.append(((east - west) * (north - south), municipality))
    if not candidates:
        raise ValueError(f"no cadastral municipality covers {latitude:.4f}, {longitude:.4f} (the cadastre does not cover Navarre or the Basque Country)")
    municipality = min(candidates, key=lambda item: item[0])[1]
    match = re.match(r"(\d+)-(.*?) buildings", municipality["title"])
    return {"code": match.group(1) if match else municipality["title"], "name": match.group(2).title() if match else municipality["title"],
            "url": municipality["url"], "epsg": municipality["epsg"] or 25830}


def download(url: str, cache_dir: Path) -> Path:
    path = cache_dir / url.rsplit("/", 1)[-1]
    if not path.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        temporary.write_bytes(_get(url, timeout=1800))
        temporary.replace(path)
    return path


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _features(stream: Any, feature: str) -> Iterator[ElementTree.Element]:
    """Each ``feature`` element of a GML stream, cleared once handled so memory stays flat."""
    context = ElementTree.iterparse(stream, events=("start", "end"))
    _, root = next(context)
    for event, element in context:
        if event == "end" and _local(element.tag) == feature:
            yield element
            root.clear()


def _child_text(element: ElementTree.Element, name: str) -> str | None:
    for child in element.iter():
        if _local(child.tag) == name and child.text and child.text.strip():
            return child.text.strip()
    return None


def _polygon(element: ElementTree.Element) -> Any:
    from shapely.geometry import MultiPolygon, Polygon

    polygons = []
    for patch in (node for node in element.iter() if _local(node.tag) == "PolygonPatch"):
        rings = []
        for ring in (node for node in patch.iter() if _local(node.tag) == "posList" and node.text):
            values = [float(value) for value in (ring.text or "").split()]
            rings.append(list(zip(values[0::2], values[1::2])))
        if rings:
            polygons.append(Polygon(rings[0], rings[1:]))
    valid = [polygon.buffer(0) if not polygon.is_valid else polygon for polygon in polygons if not polygon.is_empty]
    return valid[0] if len(valid) == 1 else MultiPolygon([part for polygon in valid for part in getattr(polygon, "geoms", [polygon])]) if valid else None


def read_buildings(zip_path: Path, bbox: tuple[float, float, float, float]) -> dict[str, dict[str, Any]]:
    """Cadastral buildings intersecting ``bbox`` (in the file's projected CRS), with storeys from their parts."""
    west, south, east, north = bbox
    buildings: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        building_file = next(name for name in names if name.endswith(".building.gml"))
        with archive.open(building_file) as stream:
            for element in _features(stream, "Building"):
                lower, upper = _child_text(element, "lowerCorner"), _child_text(element, "upperCorner")
                if lower and upper:
                    (x0, y0), (x1, y1) = (map(float, lower.split()), map(float, upper.split()))
                    if x1 < west or x0 > east or y1 < south or y0 > north:
                        continue
                reference = _child_text(element, "localId")
                geometry = _polygon(element)
                if not reference or geometry is None:
                    continue
                year = _YEAR.match(_child_text(element, "beginning") or "")
                dwellings = _child_text(element, "numberOfDwellings")
                buildings[reference] = {
                    "geometry": geometry, "use": _child_text(element, "currentUse"),
                    "construction_year": int(year.group(1)) if year and int(year.group(1)) >= 1000 else None,
                    "dwellings": int(dwellings) if dwellings and dwellings.isdigit() else None, "floors": None,
                }
        part_file = next((name for name in names if name.endswith(".buildingpart.gml")), None)
        if part_file:
            with archive.open(part_file) as stream:
                for element in _features(stream, "BuildingPart"):
                    part_id = _child_text(element, "localId") or ""
                    building = buildings.get(part_id.split("_part")[0])
                    floors = _child_text(element, "numberOfFloorsAboveGround")
                    if building is not None and floors and floors.isdigit():
                        building["floors"] = max(building["floors"] or 0, int(floors)) or None
    return buildings


def building_type(use: str | None, dwellings: int | None) -> str | None:
    if use == "1_residential":
        return None if dwellings is None else "apartments" if dwellings > 1 else "house"
    return USE_TO_TYPE.get(use or "")


def match(osm: dict[str, Any], cadastre: dict[str, Any], min_iou: float = MIN_IOU) -> list[tuple[str, str, float]]:
    """One-to-one pairs (osm id, cadastral reference, IoU) that are each other's best match above ``min_iou``."""
    from shapely.strtree import STRtree

    references = list(cadastre)
    tree = STRtree([cadastre[reference]["geometry"] for reference in references])
    scores: list[tuple[str, str, float]] = []
    for osm_id, geometry in osm.items():
        for index in tree.query(geometry):
            other = cadastre[references[int(index)]]["geometry"]
            union = geometry.union(other).area
            if union > 0:
                scores.append((osm_id, references[int(index)], geometry.intersection(other).area / union))
    best_osm: dict[str, tuple[str, float]] = {}
    best_cadastre: dict[str, tuple[str, float]] = {}
    for osm_id, reference, iou in scores:
        if iou > best_osm.get(osm_id, ("", 0.0))[1]:
            best_osm[osm_id] = (reference, iou)
        if iou > best_cadastre.get(reference, ("", 0.0))[1]:
            best_cadastre[reference] = (osm_id, iou)
    return sorted((osm_id, reference, round(iou, 4)) for osm_id, (reference, iou) in best_osm.items()
                  if iou >= min_iou and best_cadastre[reference][0] == osm_id)


def dataset_footprints(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    """OSM footprints (GeoJSON geometry) of the buildings the dataset's analyses name."""
    footprints: dict[str, dict[str, Any]] = {}
    for path in sorted((dataset_dir / "analysis").glob("*.json")):
        if path.name.endswith("-sfm.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in (payload.get("building_ids") or {}).values():
            if isinstance(record, dict) and record.get("footprint_geometry") and record.get("osm_id"):
                footprints.setdefault(record["osm_id"], record["footprint_geometry"])
    return footprints


def build_reference(data_root: Path, dataset_id: str, cache_dir: Path, *, min_iou: float = MIN_IOU, log: Any = print) -> dict[str, Any]:
    from pyproj import Transformer
    from shapely.geometry import box, shape
    from shapely.ops import transform

    from ..local_viewer import dataset_manifest
    from ..multi_view_analysis import panorama_coordinates

    dataset_dir = data_root / dataset_id
    points = [coordinates for image in dataset_manifest(data_root, dataset_id)["images"] if (coordinates := panorama_coordinates(image))]
    if not points:
        raise ValueError(f"{dataset_id} has no georeferenced panorama")
    longitude = sorted(point[0] for point in points)[len(points) // 2]
    latitude = sorted(point[1] for point in points)[len(points) // 2]
    municipality = find_municipality(longitude, latitude, cache_dir)
    log(f"Municipality {municipality['code']} {municipality['name']}; downloading its buildings (cached after the first time)...")
    archive = download(municipality["url"], cache_dir)
    forward = Transformer.from_crs("EPSG:4326", f"EPSG:{municipality['epsg']}", always_xy=True).transform
    projected = [forward(*point) for point in points]
    area = box(min(x for x, _ in projected) - MARGIN_M, min(y for _, y in projected) - MARGIN_M,
               max(x for x, _ in projected) + MARGIN_M, max(y for _, y in projected) + MARGIN_M)
    cadastre = read_buildings(archive, area.bounds)
    osm = {osm_id: transform(forward, shape(geometry)) for osm_id, geometry in dataset_footprints(dataset_dir).items()}
    pairs = match(osm, cadastre, min_iou)
    buildings = {}
    for osm_id, reference, iou in pairs:
        record = cadastre[reference]
        buildings[osm_id] = {
            "reference": reference, "iou": iou, "floors": record["floors"], "construction_year": record["construction_year"],
            "use": record["use"], "dwellings": record["dwellings"], "building_type": building_type(record["use"], record["dwellings"]),
        }
    result = {
        "source": "catastro", "dataset": dataset_id, "municipality": {key: municipality[key] for key in ("code", "name", "url")},
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"), "attribution": ATTRIBUTION,
        "match": {"min_iou": min_iou, "osm_footprints": len(osm), "cadastral_buildings_in_area": len(cadastre), "matched": len(buildings)},
        "buildings": buildings,
    }
    output = dataset_dir / "groundtruth" / "catastro.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
