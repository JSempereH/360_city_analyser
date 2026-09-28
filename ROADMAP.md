# Reliability roadmap

## Goal

Make the panorama-to-OSM building pipeline reliable across varied cities, cameras, weather, and urban forms. Reliability means the system either returns a well-supported result or abstains with a specific reason. It does not mean forcing an assignment for every visible building.

The roadmap keeps these invariants:

- OpenStreetMap is the only building-footprint source.
- Original Mapillary GPS controls visible camera positions and navigation.
- Refined poses are internal to image-to-footprint association.
- SfM supplies optional depth evidence and never moves the displayed capture track.
- Reports are deterministic and do not run an LLM or GEM classifier.
- Pre-production analysis artifacts are recalculated rather than migrated.

## Current baseline

The implemented baseline includes:

- Validated 2:1 Mapillary panoramas with finite GPS and compass heading.
- Perspective-view SegFormer or Mask2Former segmentation.
- OSM ways, multipolygon relations, holes, multiple outer rings, and building parts.
- Shapely relation assembly and pyproj local geographic projection.
- Per-ray facade depth ordering and partial-edge footprint occlusion.
- Conservative semantic pose refinement with explicit abstention.
- Input fingerprints, mask hashes, atomic artifact publication, and OSM-only reports.
- Spatially independent multi-view evidence.
- Heading-aware PyCOLMAP matching with GPS, reprojection, track, and rig-spread gates.
- Local and explicitly configured remote inference.

This is an engineering baseline, not evidence of cross-city accuracy. Synthetic tests and live OSM smoke tests cover behavior, while model quality still requires labeled evaluation.

## Phase 1: measurable baseline

### 1.1 Build a labeled pilot

Start with 100-150 panoramas from at least four cities. Expand to six cities and 600-900 panoramas only after the annotation protocol is stable.

The city set should cover:

- Dense historic attached buildings.
- Regular mid-rise blocks.
- Detached or setback buildings and sloped streets.
- Tropical vegetation and high-rise morphology.
- Irregular or peri-urban morphology.
- Adverse illumination, rain, snow, glare, or stitching defects.

Keep complete Mapillary sequences and OSM building IDs in one split. Hold out at least one city and one camera family from threshold selection.

### 1.2 Label the outputs the product uses

For each eligible panorama, record:

- Visible building-instance masks.
- Exact OSM assignment or `OSM_NULL`.
- Visible OSM facade edges or edge intervals.
- Occlusion fraction and occluder category.
- Ambiguous cases that should not contribute to primary accuracy metrics.

Use two reviewers for OSM assignment and facade-edge labels. Freeze source image hashes and the OSM extract used for evaluation.

### 1.3 Implement quality metrics

Required metrics:

| Output | Primary metrics |
|---|---|
| Building mask | Spherical IoU, spherical boundary IoU, detection F1 |
| OSM assignment | Exact-ID precision/recall, `OSM_NULL` precision/recall, adjacent-footprint confusion |
| End to end | OSM-aware panoptic quality |
| Facade evidence | Length-weighted edge precision/recall/F1 |
| Occlusion | Fraction MAE, class-weighted kappa |
| Confidence | Brier score, adaptive ECE, risk-coverage |
| SfM depth | Coverage, median-scaled AbsRel, conflict precision |

Weight equirectangular rows by `cos(elevation)` for image-space metrics. Report each city, camera, weather, resolution, and occlusion stratum separately; averages must not hide a failing domain.

### 1.4 Initial release gates

These are starting targets and should be locked after the pilot annotation review:

| Gate | Requirement |
|---|---|
| CI | 100% schema validity, deterministic artifacts, no crash, quality regression below 2 points |
| Offline alpha | Assignment F1 >= 0.85 and facade F1 >= 0.60 |
| Shadow beta | Accepted assignment precision >= 0.97 at coverage >= 0.70; ECE <= 0.05 |
| Limited canary | Accepted assignment precision >= 0.98 at coverage >= 0.80; facade F1 >= 0.75 |
| Domain expansion | New city/camera slice passes the canary floors before joining the supported set |

Promotion must use confidence intervals over spatial clusters or sequences. A hard-gate failure cannot be averaged away by a better city.

## Phase 2: geometry and camera correctness

### 2.1 Resolve building-part identity

Define and test one policy for `building` and `building:part`:

- Associate visual parts with their containing parent building when the parent is unambiguous.
- Preserve part geometry and height for projection.
- Abstain when overlapping parts map to multiple possible parents.
- Keep the report identity stable at the chosen OSM target.

### 2.2 Add orientation calibration

- Ingest pitch, roll, camera model, and stitch metadata when available.
- Add horizon estimation with a confidence score when metadata is unavailable.
- Keep an explicit `orientation_unknown` state.
- Validate forward/inverse panorama projection to within one analysis pixel.
- Use one documented camera-height convention across backend and viewer.

### 2.3 Complete vertical geometry

- Support roof height and common roof shapes where they materially affect facade projection.
- Define terrain/elevation behavior for sloped streets.
- Evaluate raised parts, arcades, and buildings above passageways.

## Phase 3: calibrated perception and fusion

### 3.1 Establish the segmentation baseline

- Benchmark B0/B5 tile sizes 320, 384, and 512 on the actual 4 GB GPU.
- Measure quality and latency rather than selecting by VRAM alone.
- Calibrate confidence independently for SegFormer and Mask2Former.
- Fine-tune on legally usable panorama street data only if the cross-city benchmark shows a domain gap.

### 3.2 Improve assignment confidence

Create separate confidence values for:

- Mask validity.
- OSM assignment.
- Facade projection.
- Occlusion estimate.
- Complete-record validity.

An abstained record may retain masks and diagnostics, but it must not emit an authoritative OSM assignment.

### 3.3 Strengthen multi-view evidence

- Require both baseline and azimuth diversity for independent confirmation.
- Detect duplicate or near-duplicate captures using sequence metadata and image hashes.
- Normalize evidence by view or visible facade area, never raw raster size.
- Preserve disagreement instead of averaging incompatible geometry variants.

## Phase 4: SfM and depth

### 4.1 Validate the current sparse pipeline

- Evaluate reconstruction success by city, camera, baseline, and sequence continuity.
- Measure registered faces, accepted panorama poses, reprojection error, track length, center spread, and GPS residual.
- Build synthetic metric scenes for scale and depth regression tests.
- Require three non-collinear accepted GPS controls for metric depth.

### 4.2 Introduce an explicit panorama rig

- Model cube faces as fixed rotations with a shared optical center.
- Compare the rig reconstruction against the current post-validation approach.
- Reject models that cannot preserve the rig constraint.

### 4.3 Evaluate modern matching only when justified

Evaluate LightGlue with a controlled keypoint budget if SIFT matching is a measured failure source. Compare reconstruction success, runtime, memory, and license constraints before replacing the baseline.

### 4.4 Evaluate monocular geometry separately

Candidate experiments:

| Model | Intended role | Deployment constraint |
|---|---|---|
| Depth Anything V2 Small | Local ordinal/relative depth | Benchmark at batch 1 on 4 GB; use compatible weights |
| MoGe-2 Small | Point maps, normals, or metric geometry experiments | Verify checkpoint license and memory |
| VGGT | Remote shadow reconstruction | Experimental; do not gate the main pipeline on it |

Monocular depth must not be called metric unless validated against independent metric ground truth. It should first be tested as an occlusion or consistency signal.

## Phase 5: operational readiness

Add structured telemetry for every job:

- Job and correlation IDs.
- Queue, start, finish, and stage timings.
- Model ID and resolved revision.
- Input fingerprint and device profile.
- Peak GPU memory.
- OSM candidate, projected, matched, and abstained counts.
- Pose and SfM quality statistics.
- Typed error code and retryability.

Initial operational targets:

| Indicator | Target |
|---|---|
| Valid jobs producing complete artifacts | >= 99.5% |
| Unhandled crashes or non-finite output | < 0.1% |
| Deterministic reruns for identical inputs | 100% artifact-hash equality |
| Traceable model/config/input identity | 100% |
| Concurrent inference on a single small GPU | Serialized |

## Model adoption rule

Do not add a model because it is newer or performs well on its own benchmark. Add it only when:

1. A versioned evaluation slice demonstrates a current failure.
2. The candidate improves that slice without violating mandatory floors elsewhere.
3. Its weights and datasets have acceptable licenses.
4. It fits either the declared local profile or the remote profile.
5. It exposes enough confidence or diagnostics to support abstention.

## Status of the separation work (September 2026)

A first separation benchmark exists (`evaluation/`, 8 panoramas, 52 buildings) and the pipeline now adds GPS-in-footprint repair, a Grounding DINO + SAM 2 instance stage, column splitting of merged facades and roof-line continuation. See [docs/state-of-the-art.md](docs/state-of-the-art.md) for measured results, model licenses and candidate upgrades (learned OSM localization, SAM 3, panoramic depth).

## Immediate checklist

1. Select four pilot cities and 100-150 panoramas.
2. Define the annotation schema for masks, OSM IDs, facade edges, and abstentions.
3. Implement spherical mask and exact-ID metrics.
4. Run the current baseline unchanged and review its 50 highest-confidence errors.
5. Decide whether the first measured bottleneck is camera orientation, segmentation, OSM-part identity, or matching.
6. Implement only the highest-impact correction and rerun the same frozen benchmark.

## Explicitly out of scope

- GEM structural classification without a separately calibrated labeled pipeline.
- LLM-generated structural attributes.
- Silent fusion of non-OSM building footprints.
- Using SfM coordinates for visible navigation.
- Claims of universal city support based only on synthetic tests or a single-city demo.
