# Architecture

## Repository layout

```text
src/city_analyser/          installable package (import name `city_analyser`)
  building_analysis.py      segmentation, OSM footprints, pose refinement, facade association
  building_instances.py     per-building instance masks (Grounding DINO + SAM 2 or YOLOE)
  facade_geometry.py        vectorized OSM wall projection and occlusion
  panorama_geometry.py      equirectangular <-> perspective projection
  multi_view_analysis.py    evidence fusion across nearby captures
  sfm_refinement.py         optional PyCOLMAP sparse depth
  analysis_backend.py       local or remote inference dispatch
  analysis_cache.py         fingerprints and atomic artifact writes
  building_inspection/      building-level reports (below)
  viewer/                   static web viewer served by local_viewer.py
  local_viewer.py           city-analyser-viewer
  mapillary_street_download.py  city-analyser-download
  download_models.py        city-analyser-models
  remote_analysis_worker.py city-analyser-worker
  benchmark_models.py       city-analyser-benchmark
evaluation/                 separation benchmark and annotations (development only)
tools/                      demo recording (development only)
tests/                      Python and JavaScript tests
```

Every command reads `data/`, `models/` and `.env` from the working directory, so run them from the repository root.

## Building inspection

`building_inspection` turns existing panorama analysis artifacts into an auditable building-level report. It deliberately does not run segmentation or infer a GEM class.

The dependency flow is one-way:

```text
manifest + analysis JSON/PNG
        |
        v
building_inspection.evidence
        |
        +--> building_inspection.ranking
        +--> building_inspection.assessment
        +--> building_inspection.screenshots
        |
        v
building_inspection.reporting --> report snapshot (JSON + HTML + JPEG)
```

The package modules have narrow responsibilities:

- `models.py`: immutable domain values shared by the package.
- `evidence.py`: validates current analysis artifacts and extracts evidence for one OSM building ID.
- `ranking.py`: normalizes direct facade pixels by analysis resolution and ranks views deterministically.
- `assessment.py`: reports factual checks about image-to-footprint association without claiming surveyed accuracy.
- `screenshots.py`: renders perspective crops with the existing orange/blue evidence convention.
- `reporting.py`: orchestrates the use case and atomically persists JSON and HTML snapshots.

`panorama_geometry.py` is shared by segmentation, SfM, and report screenshots. This prevents projection implementations from drifting.

`building_instances.py` produces the visual instance map: Grounding DINO boxes and SAM 2 masks in portrait perspective views, reprojected into the panorama with `panorama_geometry.panorama_to_tile_grid` and merged across overlapping views. `building_analysis._instance_labels` then names each instance with projected footprints (whole instance, per-column split for merged attached facades, or abstain), and `_continue_upward` fills roof lines that views or OSM heights cut.

`facade_geometry.py` holds the OSM footprint geometry used by analysis. Footprints are projected once into a camera-centered azimuthal-equidistant frame; wall facing, shared-wall rejection, line-of-sight occlusion and z-buffer rasterization are vectorized over edge arrays. Pose-search candidates are translations of that frame, and wall visibility is solved once per candidate position and reused across headings.

## Public API

The supported Python API is exported by `city_analyser.building_inspection`:

```python
from city_analyser.building_inspection import (
    BuildingEvidenceNotFound,
    build_and_persist_report,
    load_report,
    report_paths,
)
```

Internal modules can evolve without becoming server contracts. Persisted reports have their own `schema_version`, but the loader also requires the current `ANALYSIS_VERSION` and OSM-only evidence. There is intentionally no migration path for pre-release artifacts.

## HTTP integration

The stdlib server exposes:

```text
POST /api/datasets/{dataset}/building-reports/{osm-type}/{osm-id}
GET  /api/datasets/{dataset}/building-reports/{osm-type}/{osm-id}
GET  /api/datasets/{dataset}/building-reports/{osm-type}/{osm-id}/html
```

Reports are stored under `data/{dataset}/reports/buildings/{osm-type}/{osm-id}/`.

The viewer state contract is implemented in `src/city_analyser/viewer/deep-link.mjs` and uses stable image/building IDs rather than list indexes:

```text
/viewer/?dataset=...&image=...&building=way%2F...&yaw=...&pitch=...&fov=...
```

## Scope and future GEM integration

The current assessment concerns only association between direct image evidence and a footprint snapshot. Projection coverage is not a calibrated probability that the footprint is correct.

A future GEM provider should be added behind a separate interface and consume the ranked evidence. It must return per-attribute distributions, evidence references, taxonomy/specification versions, and explicit abstentions. It must not overwrite the footprint assessment or reuse its scores as structural probabilities.

## Current artifact contract

Panorama analysis metadata also lists `visual_instances` (detector score, views, naming mode and OSM IDs per instance), and each building lists the instances that carry it. The instance PNG is hashed like the other masks and is optional in the contract, so analyses without the instance stage stay valid.

Panorama analysis metadata records one input fingerprint over the panorama bytes, recorded GPS and heading, SfM payload, model request and projection settings. Both PNG masks have recorded hashes and dimensions. The masks are atomically replaced first and metadata is published last, so an interrupted run cannot validate a mixed generation. `load_analysis` rejects the current artifacts whenever any of these checks fails; older versions are recalculated rather than migrated.

OSM geometry remains embedded in each accepted building record because the exact `Polygon` or `MultiPolygon` used for projection must also drive map display and reports. OpenStreetMap is the only footprint source.
