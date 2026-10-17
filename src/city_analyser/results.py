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
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

RESULTS_MANIFEST = "results-manifest.json"
FILES = ("manifest.json", "manifest.geojson", "ATTRIBUTION.md", "building-index.json", "batch-ledger.jsonl", "survey-state.json")
FOLDERS = ("analysis", "crops", "attributes")
# A reviewer's work stays where it was made.
PROTECTED = ("attributes-human-review.json", "reviews.jsonl")
# Paths and secrets are machine-specific; the rest decides whether an analysis is current.
SKIPPED_SETTINGS = ("TOKEN", "_DIR", "BACKEND", "REMOTE")


def analysis_settings() -> dict[str, str]:
    return {name: value for name, value in sorted(os.environ.items())
            if name.startswith("BUILDING_ANALYSIS_") and not any(part in name for part in SKIPPED_SETTINGS)}


def export_results(data_root: Path, dataset_id: str, output: Path, *, with_images: bool = False) -> dict[str, Any]:
    dataset_dir = data_root / dataset_id
    if not (dataset_dir / "manifest.json").is_file():
        raise ValueError(f"no dataset at {dataset_dir}")
    files = [dataset_dir / name for name in FILES if (dataset_dir / name).is_file()]
    for folder in (*FOLDERS, *(("images",) if with_images else ())):
        files.extend(path for path in sorted((dataset_dir / folder).rglob("*")) if path.is_file() and path.name not in PROTECTED)
    summary = {
        "dataset": dataset_id, "created_at": datetime.now(UTC).isoformat(timespec="seconds"), "settings": analysis_settings(),
        "with_images": with_images, "files": len(files),
        "attributes": sum(1 for path in files if path.name.startswith("attributes-") and path.suffix == ".json"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(RESULTS_MANIFEST, json.dumps(summary, indent=2) + "\n")
        for path in files:
            # JPEG and PNG are already compressed.
            compress = zipfile.ZIP_STORED if path.suffix.lower() in {".jpg", ".jpeg", ".png"} else zipfile.ZIP_DEFLATED
            archive.write(path, PurePosixPath(dataset_id, path.relative_to(dataset_dir).as_posix()).as_posix(), compress_type=compress)
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
            # Every entry must live under the exported dataset, with no way out of it.
            if len(parts) < 2 or parts[0] != source or any(part in {"..", ""} for part in parts) or member.filename.startswith("/"):
                raise ValueError(f"unsafe path in the archive: {member.filename}")
            destination = target.joinpath(*parts[1:])
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Move a survey's results between machines, e.g. from Colab to the viewer on a laptop.")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="pack a dataset's results into a zip")
    export.add_argument("dataset")
    export.add_argument("output", type=Path)
    export.add_argument("--data-root", type=Path, default=Path("data"))
    export.add_argument("--with-images", action="store_true", help="include the panoramas (needed when the target machine does not have them)")
    imported = commands.add_parser("import", help="unpack a results zip into data/, keeping local reviews")
    imported.add_argument("archive", type=Path)
    imported.add_argument("--data-root", type=Path, default=Path("data"))
    imported.add_argument("--name", help="dataset name on this machine (default: the exported one)")
    args = parser.parse_args()
    try:
        if args.command == "export":
            summary = export_results(args.data_root, args.dataset, args.output, with_images=args.with_images)
            print(f"{args.output}: {summary['files']} files, {summary['attributes']} attribute files of {args.dataset}")
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
