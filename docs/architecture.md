# Architecture

## Repository layout

```text
src/city_analyser/          installable package (import name `city_analyser`)
  building_analysis.py      segmentation, OSM footprints, pose refinement, facade association
  building_instances.py     per-building instance masks (Grounding DINO + SAM 2 or YOLOE)
  facade_geometry.py        vectorized OSM wall projection and occlusion
  panorama_geometry.py      equirectangular <-> perspective projection
  image_quality.py          sharpness, brightness and file-size facts and flags per panorama
  building_index.py         city-analyser-index: which panoramas see which building (geometry only)
  attributes/               attribute providers (OSM-tag prior, vision-language), fusion and abstention
  multi_view_analysis.py    evidence fusion across nearby captures
  sfm_refinement.py         optional PyCOLMAP sparse depth
  analysis_backend.py       local or remote perception (model stages) dispatch; association is always local
  analysis_cache.py         fingerprints and atomic artifact writes
  stage_cache.py            content-addressed cache of model-stage outputs
  footprint_corrections.py  reviewed, versioned OSM footprint corrections
  footprint_issues.py       review queue from per-panorama footprint diagnostics
  footprint_cli.py          city-analyser-footprints (list, export for JOSM)
  batch_analysis.py         city-analyser-analyze (whole dataset, pipelined, resumable)
  building_table.py         city-analyser-export: per-building table, views, coverage report, export manifest
  run_ledger.py             append-only record of finished, failed and skipped panoramas
  run_progress.py           progress file, ETA, and the progress page with a status map
  building_inspection/      building-level reports and crop export (below)
  viewer/                   static web viewer served by local_viewer.py (app.js plus pure, tested
                            modules: deep links, map geometry, navigation, review queue, footprint
                            editor, wall projection)
  local_viewer.py           city-analyser-viewer
  mapillary_street_download.py  city-analyser-download
  download_models.py        city-analyser-models
  remote_analysis_worker.py city-analyser-worker (GPU perception only, POST /api/v2/perception)
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
- `crops.py`: `city-analyser-crops`, ranked perspective crops with provenance sidecars for attribute providers.

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

## Attribute providers

`city_analyser.attributes` implements the provider interface this page describes. A provider takes ranked crops (or OSM tags) and returns, per attribute, a prediction that is either categorical (`scores` plus `score_kind`: `probability`, `vote_share` or `tag`) or numeric (`value` and `spread`), with `abstained` and `reason` set when evidence is insufficient, and a list of the crops it used. `fusion` pools per-view answers weighted by the view's ranking score and keeps the whole distribution. The OSM-tag `prior` is the baseline; the `vlm` provider reaches a multimodal model through an OpenAI-compatible endpoint and never loads model code in this process. Provider output lives under `attributes/` and never feeds the footprint assessment. Image quality facts recorded by the analysis (`image_quality`) halve the ranking score of flagged views (ranking formula `direct-facade-evidence-v2`, report schema 3).

## Scope and future GEM integration

The current assessment concerns only association between direct image evidence and a footprint snapshot. Projection coverage is not a calibrated probability that the footprint is correct.

A future GEM provider should be added behind a separate interface and consume the ranked evidence. It must return per-attribute distributions, evidence references, taxonomy/specification versions, and explicit abstentions. It must not overwrite the footprint assessment or reuse its scores as structural probabilities.

## Analysis stages and caching

`analyze_panorama` runs in two phases:

- `perceive_panorama`: the model stages. Semantic confidences (uint8, 255 = 1.0) and the instance map are cached by `stage_cache` under a key built from the panorama SHA-256, the model ID, the revision in the local Hugging Face cache, and the stage's settings and `*_STAGE_VERSION`. The minimum confidence is applied when the confidences are read, so it is not part of the key. On a full cache hit the panorama is not decoded and no model is loaded.
- `associate_panorama`: pose refinement, OSM projection and naming, and artifact publication. It is pure CPU work and is what `ANALYSIS_VERSION` versions.

Model input views are sampled bilinearly on the inference device (`panorama_geometry.perspective_tiles_torch`); projection grids depend only on geometry and are memoized. `batch_analysis` overlaps `perceive_panorama` for one panorama with `associate_panorama` for the previous one and prefetches OSM footprints. With a remote worker only `perceive_panorama` moves: the request is the JPEG plus the perception configuration, the answer is an `.npz` of confidences and the instance map, and `--perception-only` runs just that stage to fill the stage cache.

## Batch runs

`batch_analysis` is the city-scale entry point. It keeps three things apart. The run ledger (`batch-ledger.jsonl`) is the only fast source for "is this panorama done": one JSON line per event, flushed and synced, keyed by stage (`analysis` or `perception`), carrying a fingerprint of the settings that stage depends on and the image's size and modification time. The stage cache and the artifacts remain the evidence; the ledger is an index over them, and `--verify` re-checks it against them. Progress (`batch-progress.json`) is advisory and rewritten at most every 5 seconds. Failures are classified once: `ValueError` is permanent input trouble (recorded `skipped`, never retried), everything else is transient (retried with backoff, then recorded `failed` and retried by the next run up to a cap). `load_analysis` takes the manifest record as an argument so that loops over a dataset do not re-read the manifest for every panorama.

## Current artifact contract

Panorama analysis metadata also lists `visual_instances` (detector score, views, naming mode and OSM IDs per instance), and each building lists the instances that carry it. The instance PNG is hashed like the other masks and is optional in the contract, so analyses without the instance stage stay valid.

Panorama analysis metadata records one input fingerprint over the panorama bytes, recorded GPS and heading, SfM payload, model request and projection settings. Both PNG masks have recorded hashes and dimensions. The masks are atomically replaced first and metadata is published last, so an interrupted run cannot validate a mixed generation. `load_analysis` rejects the current artifacts whenever any of these checks fails; older versions are recalculated rather than migrated.

OSM geometry remains embedded in each accepted building record because the exact `Polygon` or `MultiPolygon` used for projection must also drive map display and reports. OpenStreetMap is the footprint source and the building identity.

## Footprint corrections

`nearby_buildings` returns `osm_buildings` (the elements as OSM serves them, with `osm_version`, `osm_timestamp` and `osm_changeset`; user names are not kept) after `footprint_corrections.apply_corrections`. A correction replaces an element's geometry only when it is accepted and its `base_version` equals the version OSM serves; otherwise the feature carries `footprint_correction_stale` and keeps the OSM geometry. Applied corrections add `footprint_correction` (revision, source, base version, review time) and keep `footprint_source: "osm"`.

Analysis metadata records `footprints_sha256` (OSM IDs, versions, correction revisions and geometry hashes of every candidate footprint), the corrected IDs under `footprint_sources.corrected`, and `footprint_diagnostics` (`merged_footprint`, `unexplained_instance`, `unsupported_footprint`, `pose_offset`). Building records copy `osm_version`, `osm_timestamp` and `footprint_correction`; reports (schema 2) state an applied correction as a caution and a limitation. `footprint_issues` aggregates diagnostics across current analyses and selects the panoramas a changed footprint affects, which the viewer re-associates after a review. The HTTP routes are `GET /api/footprint-issues`, `GET /api/footprint-corrections[?status=]` and `GET`/`POST /api/footprint-corrections/{osm-type}/{osm-id}` (`action`: `propose` with `geometry` or `parts`, `review`, or `revert`, which withdraws the accepted correction).

A split correction (`kind: "split"`) replaces the element with buildings `{osm-type}/{osm-id}/part-{n}` that carry `parent_osm_id`. Building identifiers in analyses, reports (`reports/buildings/{osm-type}/{osm-id}/part-{n}/`), report routes and deep links accept this suffix; corrections themselves always target the OSM element.

## Viewer API

Besides analysis, reports and corrections, the viewer uses:

```text
GET  /api/datasets/{dataset}/analysis-status                      current | stale | missing per panorama
GET  /api/datasets/{dataset}/building-views/{building-id}         analyzed panoramas showing a building, best first
GET  /api/datasets/{dataset}/analysis/{image}/mask?layer=instances  RGBA overlay, one color per visual instance
POST /api/datasets/{dataset}/analysis-batch                       {bbox?, force?}: background analysis job
POST /api/jobs/{job-id}/cancel                                    stop a cancellable job after its current step
```

Jobs report `total` and `done` when they process several panoramas; a cancelled job ends with status `cancelled`.
