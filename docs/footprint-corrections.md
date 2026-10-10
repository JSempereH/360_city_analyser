# Footprint corrections

OpenStreetMap stays the footprint source and the building identity. When the imagery shows that an OSM footprint is wrong, a person can record a **local correction**: a reviewed, versioned replacement of that element's geometry. Corrections are explicit, never silent, and never uploaded to OSM automatically.

## Why footprints go wrong

Errors seen by this pipeline:

- **Displacement** of a few meters (published cases reach about 9 m): footprints traced from misregistered aerial imagery, or traced along the roof instead of the base.
- **Blocks mapped as one polygon** without party walls. This is the most common case in the separation benchmark and the cause of most column splits.
- **Missing, demolished or outdated** buildings.

The camera pose is a second source of disagreement. Pose refinement already moves a GPS fix out of footprints and searches small shifts, and a consistent shift across many panoramas points at the data rather than the GPS (see *Systematic offsets* below).

## How corrections work

```text
OSM element (version v) ──> nearby_buildings ──> accepted correction drawn on version v? ──> analysis
                                                  yes: use the corrected geometry
                                                  no (OSM now serves v' != v): use OSM, report the correction as stale
```

- Each OSM element has one JSON file in `data/footprint-corrections/` (`way-123.json`), or under `BUILDING_ANALYSIS_FOOTPRINT_CORRECTIONS_DIR`. It holds the element's full revision history. Nothing is deleted.
- A revision records the geometry, the OSM version and geometry hash it was drawn on, its source (`manual`, `pipeline` or `cadastre`), author, note, evidence, status and review.
- Status flow: `proposed` -> `accepted` or `rejected`. Accepting a revision marks the previously accepted one `superseded`. An accepted correction can be withdrawn: it becomes `reverted` (with who, when and why) and the OSM geometry applies again.
- A correction applies only while OSM serves the version it was drawn on. When the element changes in OSM, the correction is reported as **stale** and the OSM geometry is used until someone reviews it again.
- A **geometry** correction moves or redraws a footprint by at most 50 m (centroid to centroid).
- A **split** replaces the element with 2 to 20 parts (for example a block mapped as one polygon, cut along its party walls). Parts may touch but not overlap by more than 1 % of the smaller one. Each part becomes a building with its own identity, `<element>/part-<n>` (`way/123/part-2`), keeps the element's tags and records `parent_osm_id`. Parts are changed by proposing a new revision of the parent element; single parts cannot be edited yet.
- Adding a building that OSM lacks entirely is not supported yet.
- Analyses record each building's `osm_version` and `footprint_correction`, the list of corrected footprints, and `footprints_sha256`, the identity of the exact footprints they used. Building reports state an applied correction as a caution and a limitation.

## Workflow in the viewer

1. Select the footprint (in the panorama, on the map, or by searching `way/123`) and click **Edit footprint** in the Building tab, or use **Edit a footprint on the map** in the Review tab.
2. Drag vertices or a part's square handle. Click an edge dot to add a vertex; double-click a vertex to remove it. **Split**, then click two vertices of one part, cuts it between them. `Ctrl+Z` and `Ctrl+Shift+Z` undo and redo.
3. While editing, the camera-facing walls of your outline (blue) and of the OSM original (dashed) are drawn on the panorama from the analysis pose, so the outline can be aligned with the real facade.
4. Enter your name (in the Review tab) and why the footprint is wrong, then **Propose correction**.
5. In the **Review** tab, `N` opens the next item in the panorama that shows it. `A` accepts and `X` rejects a proposal. A reviewer should be someone other than the author whenever possible.
6. **Revert to OpenStreetMap** in the Building tab withdraws an accepted correction (for a part, the whole split) after a confirmation and an optional reason.
7. Accepting or reverting a correction re-associates every analyzed panorama within the association radius of the footprint. With the stage cache this takes about a second per panorama, runs no model and does not load PyTorch.

Corrections also apply with a remote GPU worker: the worker only runs the models, and footprint association always happens on your machine (see [Remote inference](remote-inference.md)).

## Diagnostics: where to look

Every analysis records `footprint_diagnostics`, and **Footprint review** aggregates them per building:

| Diagnostic | Meaning | Likely cause |
|---|---|---|
| `merged_footprint` | One footprint names two or more separate buildings in the image | Block mapped as one polygon |
| `unexplained_instance` | A building in the image with no footprint projecting onto it (with its compass bearings) | Missing or displaced footprint |
| `unsupported_footprint` | A large, unoccluded projected facade with almost no building in the image | Displaced, demolished or wrongly tagged footprint |
| `pose_offset` | How far pose refinement moved the camera | GPS error, or misregistered data when consistent |

**Systematic offsets:** when at least 3 accepted poses in a dataset shift the same way (consistency of at least 0.7) by at least 2 m on average, the review lists a possible dataset-wide misregistration.

All thresholds are heuristics. On the separation benchmark (8 annotated panoramas, 52 buildings, October 2026):

| Diagnostic | Flagged | Precision | Recall |
|---|---|---|---|
| `merged_footprint` | 6 | 0.83 (5 of 6) | 0.31 (5 of 16 footprints covering several annotated buildings) |
| `unexplained_instance` | 3 | 0.67 (2 of 3) | not measured |
| `unsupported_footprint` | 0 | not measured | not measured |

Diagnostics are therefore mostly right when they fire but miss most merged blocks. Treat them as a review queue, not as evidence. Reproduce with:

```bash
uv run python -m evaluation.run associate --semantic b5-384-v2 --instances gdino-tiny-v2 --tag diag
uv run python -m evaluation.run diagnostics --tag diag
```

## Exporting for OpenStreetMap

```bash
uv run city-analyser-footprints list --status proposed
uv run city-analyser-footprints export --format osm --output corrections.osm
uv run city-analyser-footprints export --format geojson --output corrections.geojson
```

The `.osm` file is a JOSM reference layer: one new way per corrected outline, with `fixme`, `note` and `source` tags naming the original element and version. It is marked `upload="never"`, so JOSM refuses to upload it. Use it to edit the original element by hand.

Rules to follow before changing OSM:

- Mapillary allows deriving data from its images to contribute to OpenStreetMap.
- Systematic edits must follow the [Automated Edits code of conduct](https://wiki.openstreetmap.org/wiki/Automated_Edits_code_of_conduct) and the [Import Guidelines](https://wiki.osm.org/Import/Guidelines): review every change, and consult the local community before any organized effort.
- Geometry from other sources (cadastre, EUBUCCO) may only enter OSM where its license is compatible and the local community agrees.

## Next steps

1. Add buildings missing from OSM, and edit single parts of a split.
2. Suggest corrections from cadastral footprints or EUBUCCO, with license checks.
3. Triangulate facade lines and party walls from several panoramas with the existing SfM to propose translations and splits automatically.
