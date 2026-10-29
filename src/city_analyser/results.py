"""city-analyser-results: carry a survey's results from a GPU machine (Colab, a cloud VM) to the viewer.

``export`` packs what a survey wrote for a dataset into one zip: the manifest,
analyses, crops, attributes, index, ledger and survey state (panoramas only with
``--with-images``), plus ``results-manifest.json`` with the BUILDING_ANALYSIS_*
settings of the run. ``import`` unpacks it into ``data/<dataset>`` on the
viewer's machine. It never touches a reviewer's decisions
(``attributes-human-review.json``, ``reviews.jsonl``): a model run imported
later never undoes a correction, and the card keeps preferring the review.

An analysis is valid only under the settings it was made with, so ``import``
prints the settings to start the viewer with.

``--demo`` packs a presentation copy for another machine: also its route and
curation (``tour.json``, ``curation.json``), the cadastre match (``groundtruth/``,
shown only in evaluation mode), the panoramas, and the OSM footprints and map
tiles of its area, so it plays offline exactly as it was reviewed.

``share`` packs one or more analyzed datasets for a colleague who has only a
clone of the repository: the archive is unpacked into the checkout as it is
(``data/<dataset>/`` and the area's caches, nothing to import), and carries a
generated ``HOW-TO-VIEW.md`` with the command, settings and addresses to open
each dataset, offline included.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

RESULTS_MANIFEST = "results-manifest.json"
FILES = ("manifest.json", "manifest.geojson", "ATTRIBUTION.md", "building-index.json", "batch-ledger.jsonl", "survey-state.json")
FOLDERS = ("analysis", "crops", "attributes")
# A presentation copy (city_analyser.demo) adds its route, its curation and the reference data.
DEMO_FILES = ("tour.json", "curation.json")
DEMO_FOLDERS = ("groundtruth", "images")
# The data root's shared caches, carried under this prefix and unpacked next to the datasets.
CACHES = "_caches"
CACHE_FOLDERS = {"osm": ".osm-cache", "tiles": ".tile-cache"}
# A reviewer's work stays where it was made.
PROTECTED = ("attributes-human-review.json", "reviews.jsonl")
# Paths and secrets are machine-specific; the rest decides whether an analysis is current.
SKIPPED_SETTINGS = ("TOKEN", "_DIR", "BACKEND", "REMOTE")


def analysis_settings() -> dict[str, str]:
    return {name: value for name, value in sorted(os.environ.items())
            if name.startswith("BUILDING_ANALYSIS_") and not any(part in name for part in SKIPPED_SETTINGS)}


def _area_cells(dataset_dir: Path) -> set[str]:
    """OSM cache cells (``<lat cell>_<lon cell>``) around the dataset's panoramas, with one cell of margin."""
    from .building_analysis import OSM_CELL_DEGREES
    from .multi_view_analysis import panorama_coordinates

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    cells = set()
    for image in manifest.get("images", []):
        coordinates = panorama_coordinates(image) if isinstance(image, dict) else None
        if coordinates:
            longitude, latitude = coordinates
            row, column = int(latitude // OSM_CELL_DEGREES), int(longitude // OSM_CELL_DEGREES)
            cells.update(f"{row + dr}_{column + dc}" for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    return cells


def export_results(data_root: Path, dataset_id: str, output: Path, *, with_images: bool = False, demo: bool = False) -> dict[str, Any]:
    dataset_dir = data_root / dataset_id
    if not (dataset_dir / "manifest.json").is_file():
        raise ValueError(f"no dataset at {dataset_dir}")
    names = (*FILES, *(DEMO_FILES if demo else ()))
    files = [dataset_dir / name for name in names if (dataset_dir / name).is_file()]
    folders = (*FOLDERS, *(DEMO_FOLDERS if demo else ("images",) if with_images else ()))
    for folder in folders:
        files.extend(path for path in sorted((dataset_dir / folder).rglob("*")) if path.is_file() and path.name not in PROTECTED)
    caches: list[tuple[Path, str]] = []
    if demo:
        cells = _area_cells(dataset_dir)
        osm = data_root / CACHE_FOLDERS["osm"]
        caches.extend((path, f"{CACHES}/osm/{path.name}") for path in sorted(osm.glob("*.json")) if path.stem in cells)
        tiles = data_root / CACHE_FOLDERS["tiles"]
        caches.extend((path, f"{CACHES}/tiles/{path.relative_to(tiles).as_posix()}") for path in sorted(tiles.rglob("*.png")))
    summary = {
        "dataset": dataset_id, "created_at": datetime.now(UTC).isoformat(timespec="seconds"), "settings": analysis_settings(),
        "with_images": with_images or demo, "demo": demo, "files": len(files), "cached_files": len(caches),
        "attributes": sum(1 for path in files if path.name.startswith("attributes-") and path.suffix == ".json"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(RESULTS_MANIFEST, json.dumps(summary, indent=2) + "\n")
        for path in files:
            # JPEG and PNG are already compressed.
            compress = zipfile.ZIP_STORED if path.suffix.lower() in {".jpg", ".jpeg", ".png"} else zipfile.ZIP_DEFLATED
            archive.write(path, PurePosixPath(dataset_id, path.relative_to(dataset_dir).as_posix()).as_posix(), compress_type=compress)
        for path, name in caches:
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix == ".png" else zipfile.ZIP_DEFLATED)
    return summary


def import_results(archive_path: Path, data_root: Path, name: str | None = None) -> dict[str, Any]:
    """Unpack into ``data_root/<name or the exported dataset>``; returns counts and the run's settings."""
    with zipfile.ZipFile(archive_path) as archive:
        try:
            summary = json.loads(archive.read(RESULTS_MANIFEST))
        except KeyError as error:
            raise ValueError(f"{archive_path} is not a results archive (no {RESULTS_MANIFEST})") from error
        source = summary["dataset"]
        target = data_root / (name or source)
        written = kept = 0
        for member in archive.infolist():
            if member.filename == RESULTS_MANIFEST or member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            # Every entry must live under the exported dataset or a known cache, with no way out of it.
            cache = len(parts) >= 3 and parts[0] == CACHES and parts[1] in CACHE_FOLDERS
            if len(parts) < 2 or (parts[0] != source and not cache) or any(part in {"..", ""} for part in parts) or member.filename.startswith("/"):
                raise ValueError(f"unsafe path in the archive: {member.filename}")
            destination = data_root.joinpath(CACHE_FOLDERS[parts[1]], *parts[2:]) if cache else target.joinpath(*parts[1:])
            if destination.name in PROTECTED:
                kept += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".importing")
            temporary.write_bytes(archive.read(member))
            temporary.replace(destination)
            written += 1
    images = target / "images"
    return {**summary, "target": str(target), "written": written, "protected_skipped": kept,
            "images_present": images.is_dir() and any(images.iterdir())}


# What ``share`` carries of a dataset: everything a viewer reads, nothing a local workflow keeps for itself.
SHARE_FILES = (*FILES, *DEMO_FILES, "mask-quality.json")
SHARE_FOLDERS = (*FOLDERS, *DEMO_FOLDERS)
SHARE_README = "HOW-TO-VIEW.md"
# Analysis configuration keys and the environment variables that set them (building_analysis.analysis_configuration).
CONFIGURATION_VARIABLES = {
    "requested_model": "BUILDING_ANALYSIS_MODEL", "requested_model_revision": "BUILDING_ANALYSIS_MODEL_REVISION",
    "tile_size": "BUILDING_ANALYSIS_TILE_SIZE", "minimum_confidence": "BUILDING_ANALYSIS_MIN_CONFIDENCE",
}


def _analysis_files(dataset_dir: Path) -> list[Path]:
    from .building_analysis import panorama_analysis_files

    return panorama_analysis_files(dataset_dir)


def recorded_settings(dataset_dir: Path) -> dict[str, str]:
    """The BUILDING_ANALYSIS_* settings a dataset was analyzed with, read from its analyses (they must agree)."""
    configurations = {json.dumps(json.loads(path.read_text(encoding="utf-8")).get("analysis_configuration"), sort_keys=True)
                      for path in _analysis_files(dataset_dir)}
    if not configurations:
        raise ValueError(f"{dataset_dir.name} has no analysis to share")
    if len(configurations) > 1:
        raise ValueError(f"{dataset_dir.name} mixes analyses made with different settings; analyze it again first")
    configuration = json.loads(configurations.pop()) or {}
    return {variable: str(configuration[key]) for key, variable in CONFIGURATION_VARIABLES.items()
            if configuration.get(key) not in (None, "")}


def _settings_reproduce(dataset_dir: Path, settings: dict[str, str]) -> bool:
    """Whether a viewer started with only ``settings`` sees the dataset's analyses as current (other variables at default)."""
    from unittest.mock import patch

    from .building_analysis import analysis_configuration

    stored = json.loads(_analysis_files(dataset_dir)[0].read_text(encoding="utf-8")).get("analysis_configuration")
    clean = {name: value for name, value in os.environ.items() if not name.startswith("BUILDING_ANALYSIS_")}
    with patch.dict(os.environ, {**clean, **settings}, clear=True):
        return analysis_configuration() == stored


def _tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    import math

    def latitude(row: int) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * row / 2**z))))

    return x / 2**z * 360 - 180, latitude(y + 1), (x + 1) / 2**z * 360 - 180, latitude(y)


def _area(dataset_dir: Path, margin_deg: float = 0.003) -> tuple[float, float, float, float]:
    from .multi_view_analysis import panorama_coordinates

    points = [panorama_coordinates(image) for image in json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8")).get("images", [])
              if isinstance(image, dict)]
    points = [point for point in points if point]
    if not points:
        raise ValueError(f"{dataset_dir.name} has no located panorama")
    longitudes, latitudes = [point[0] for point in points], [point[1] for point in points]
    return min(longitudes) - margin_deg, min(latitudes) - margin_deg, max(longitudes) + margin_deg, max(latitudes) + margin_deg


def _view_guide(entries: list[dict[str, Any]], archive_name: str) -> str:
    """HOW-TO-VIEW.md for whoever receives the archive: install, unpack, start, open."""
    settings = entries[0]["settings"]
    same = all(entry["settings"] == settings for entry in entries)
    def command(values: dict[str, str]) -> str:
        return " ".join(f"{name}={value}" for name, value in values.items())
    def powershell(values: dict[str, str]) -> str:
        return "; ".join(f'$env:{name}="{value}"' for name, value in values.items())
    lines = [
        "# How to view these results", "",
        f"This archive holds the analysis results of {', '.join(f'`{entry['dataset']}`' for entry in entries)} for 360 City Analyser: "
        "panoramas, building masks, crops and the AI's answers for each building. Viewing them needs no GPU, no AI model, "
        "no Mapillary account and, once unpacked, no network.", "",
        "## 1. Install (once)", "",
        "You need [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/) (Linux, macOS or Windows).", "",
        "```bash", "git clone https://github.com/JSempereH/360_city_analyser.git", "cd 360_city_analyser", "uv sync", "```", "",
        "`uv sync` installs the base dependencies only; the GPU extras are not needed.", "",
        "## 2. Unpack this archive (once)", "",
        "Extract it **into the `360_city_analyser` folder**, so that a `data` folder appears next to `README.md`. On Linux or macOS:", "",
        "```bash", f"unzip /path/to/{archive_name} -d /path/to/360_city_analyser", "```", "",
        "On Windows: right-click the archive, \"Extract All...\", and choose the `360_city_analyser` folder.", "",
        "## 3. Start the viewer", "",
        "From the `360_city_analyser` folder (these settings are the ones the panoramas were analyzed with; "
        "with other ones the viewer treats the analyses as out of date):", "",
    ]
    groups = [entries] if same else [[entry] for entry in entries]
    for group in groups:
        values = {**group[0]["settings"], "CITY_ANALYSER_OFFLINE": "1"}
        if not same:
            lines.append(f"For `{group[0]['dataset']}`:")
            lines.append("")
        lines += ["```bash", f"{command(values)} uv run city-analyser-viewer", "```", "",
                  f"Windows PowerShell: `{powershell(values)}; uv run city-analyser-viewer`", ""]
    lines += ["`CITY_ANALYSER_OFFLINE=1` keeps the viewer off the network; leave it out to let the map fetch tiles the archive does not carry. "
              "Leave the terminal open while you use the viewer.", "", "## 4. Open", "", "In Chrome or Firefox:", ""]
    for entry in entries:
        base = f"http://127.0.0.1:8765/viewer/?dataset={entry['dataset']}"
        lines.append(f"- `{entry['dataset']}` ({entry['buildings']} buildings described): {base}")
        if entry["buildings"]:
            lines.append(f"  - as a presentation that walks the buildings by itself: {base}&present=1&live=0&speed=2")
    lines += ["", "In the viewer, click a building in the panorama or on the map to see its card: the GEM v3 record "
              "(material / lateral load-resisting system / height), each attribute with its source, and the AI's notes per view. "
              "Press `?` for the keyboard shortcuts; in a presentation, Space pauses, the right arrow skips ahead, "
              "H hides the notes, F hides the side columns and Esc leaves it.", ""]
    return "\n".join(lines)


def share_results(data_root: Path, dataset_ids: list[str], output: Path) -> dict[str, Any]:
    """An archive to unpack into another checkout as it is: ``data/<dataset>/`` for each dataset, the OSM footprints and
    map tiles of their areas, and HOW-TO-VIEW.md."""
    entries, files, cells, areas = [], [], set(), []
    for dataset_id in dataset_ids:
        dataset_dir = data_root / dataset_id
        if not (dataset_dir / "manifest.json").is_file():
            raise ValueError(f"no dataset at {dataset_dir}")
        settings = recorded_settings(dataset_dir)
        if not _settings_reproduce(dataset_dir, settings):
            raise ValueError(f"{dataset_id} was analyzed with settings beyond {', '.join(sorted(settings))} "
                             "(for example the instance detector's); they cannot be written in the instructions")
        names = [dataset_dir / name for name in SHARE_FILES if (dataset_dir / name).is_file()]
        for folder in SHARE_FOLDERS:
            names.extend(path for path in sorted((dataset_dir / folder).rglob("*")) if path.is_file() and not path.name.endswith(".importing"))
        files.extend(names)
        cells |= _area_cells(dataset_dir)
        areas.append(_area(dataset_dir))
        described = {path.parent for path in (dataset_dir / "attributes").rglob("attributes-vlm_*.json")}
        entries.append({"dataset": dataset_id, "settings": settings, "buildings": len(described)})
    osm = data_root / CACHE_FOLDERS["osm"]
    caches = [path for path in sorted(osm.glob("*.json")) if path.stem in cells]
    tiles = data_root / CACHE_FOLDERS["tiles"]
    for path in sorted(tiles.rglob("*.png")):
        try:
            z, x, y = int(path.parent.parent.name), int(path.parent.name), int(path.stem)
        except ValueError:
            continue
        west, south, east, north = _tile_bounds(z, x, y)
        if any(west <= area[2] and east >= area[0] and south <= area[3] and north >= area[1] for area in areas):
            caches.append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(SHARE_README, _view_guide(entries, output.name))
        for path in (*files, *caches):
            compress = zipfile.ZIP_STORED if path.suffix.lower() in {".jpg", ".jpeg", ".png"} else zipfile.ZIP_DEFLATED
            archive.write(path, PurePosixPath("data", path.relative_to(data_root).as_posix()).as_posix(), compress_type=compress)
    return {"datasets": entries, "files": len(files), "cached_files": len(caches), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Move a survey's results between machines, e.g. from Colab to the viewer on a laptop.")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="pack a dataset's results into a zip")
    export.add_argument("dataset")
    export.add_argument("output", type=Path)
    export.add_argument("--data-root", type=Path, default=Path("data"))
    export.add_argument("--with-images", action="store_true", help="include the panoramas (needed when the target machine does not have them)")
    export.add_argument("--demo", action="store_true", help="a presentation copy for another machine: also its route, curation, reference data, "
                        "panoramas and the OSM footprints and map tiles of its area")
    shared = commands.add_parser("share", help="pack analyzed datasets for a colleague: unzip into a checkout and open, with a generated HOW-TO-VIEW.md")
    shared.add_argument("datasets", nargs="+", help="dataset names under --data-root")
    shared.add_argument("--output", type=Path, required=True, help="the zip to write")
    shared.add_argument("--data-root", type=Path, default=Path("data"))
    imported = commands.add_parser("import", help="unpack a results zip into data/, keeping local reviews")
    imported.add_argument("archive", type=Path)
    imported.add_argument("--data-root", type=Path, default=Path("data"))
    imported.add_argument("--name", help="dataset name on this machine (default: the exported one)")
    args = parser.parse_args()
    try:
        if args.command == "share":
            summary = share_results(args.data_root, args.datasets, args.output)
            for entry in summary["datasets"]:
                print(f"{entry['dataset']}: {entry['buildings']} buildings described, analyzed with "
                      f"{' '.join(f'{name}={value}' for name, value in entry['settings'].items())}")
            size = args.output.stat().st_size / 2**20
            print(f"{args.output}: {summary['files']} files and {summary['cached_files']} cached footprints and map tiles, {size:.0f} MiB. "
                  f"Share it with the instructions inside ({SHARE_README}).")
            return
        if args.command == "export":
            summary = export_results(args.data_root, args.dataset, args.output, with_images=args.with_images, demo=args.demo)
            print(f"{args.output}: {summary['files']} files, {summary['attributes']} attribute files of {args.dataset}")
            if not summary["attributes"]:
                print("Warning: no building was described yet, so the archive holds no results; see the survey's output.", file=sys.stderr)
            return
        summary = import_results(args.archive, args.data_root, args.name)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(f"Imported {summary['written']} files into {summary['target']} (local reviews kept).")
    if not summary["images_present"]:
        print("The panoramas are not there: copy the dataset's images/ folder, or export again with --with-images.")
    settings = " ".join(f"{name}={value}" for name, value in summary["settings"].items())
    print(f"Start the viewer with the settings of that run, or its analyses will look out of date:\n  {settings} uv run city-analyser-viewer")


if __name__ == "__main__":
    main()
