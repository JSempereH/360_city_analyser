# Experiments: what the tool needs to analyze cities (October 2026)

This page plans the work between a tool that can describe buildings and one that can run experiments and report how well it does. It covers seven points. Points 1 and 2 are implemented (sections 1 and 2 also say how to use them); points 3 to 7 are analyzed here for later.

The question every experiment must answer first: **does a vision-language model add anything over the baseline it replaces**, the OSM-tag prior and, in Spain, the cadastre? Without reference data that question has no answer, which is why points 1 and 2 come first.

| # | Point | Status |
|---|---|---|
| 1 | Reference data to measure against | Implemented: cadastre (Spain), engineer-labeled evaluation set, OSM tags |
| 2 | Experiment management | Implemented: experiment files, isolated runs, comparison report |
| 3 | Enough imagery | In progress: Mapillary works again; Ciutat Vella (Valencia) has 89 panoramas and 133 buildings matched to the cadastre |
| 4 | Automatic mask quality control | Planned below |
| 5 | Scale (frozen OSM, GPU throughput, job queue) | Planned below |
| 6 | GEM detail that matters for seismic risk | Planned below |
| 7 | Outputs for risk analysis | Planned below |

The last section compares the tool with related systems and lists what it lacks against them.

## 1. Reference data

Three sources, each measuring different attributes, never mixed in one number:

| Source | Attributes it measures | Coverage | Caveats |
|---|---|---|---|
| **Cadastre** (Spain, INSPIRE buildings) | storeys above ground, construction year, use | Every building outside Navarre and the Basque Country | Year can be that of a major refurbishment; storeys can omit an attic under the roof; use is per building, not per facade. It says nothing about the structure. |
| **Evaluation set** (engineer reviews marked as ground truth) | every attribute, including the GEM material and lateral system | As many buildings as someone labels (aim: 200 to 500 per pilot) | Only source for the structure. Labels come from the same photos the model sees, so they measure agreement with an engineer reading the imagery, not with the building as built. |
| **OSM tags** | use, storeys, year, facade material where tagged | Sparse and uneven | Noisy, and the prior provider is built from them: never use OSM to score the prior. |

**Cadastre.** `city-analyser-groundtruth catastro <dataset>` finds the municipality from the panorama positions, downloads its INSPIRE buildings once (`data/.groundtruth-cache/`), reads only the buildings inside the dataset's area (streaming, so memory stays low even for Madrid), and matches them to the dataset's OSM footprints one to one: each pair must be the other's best match and overlap with an intersection over union of at least 0.5. Unmatched footprints are left out rather than guessed. The result is `data/<dataset>/groundtruth/catastro.json`, with the cadastral reference, the overlap and the values for each matched building. Storeys are the highest count of any part of the building (the cadastre stores them per part). Use maps `1_residential` to apartments (several dwellings) or house (one), `3_industrial` to industrial, `4_1_office` to office, `4_2_retail` to retail and `4_3_publicServices` to public; agriculture has no class here and is left out.

**Evaluation set.** In the review form, "Use as ground truth" marks a review as a label. Labels are kept apart from ordinary corrections because an evaluation set must be chosen before looking at a model's answers and left alone afterwards: pick the buildings (for example every building of a few blocks, or a random sample), label them from the imagery, and do not correct them after seeing a run.

## 2. Experiment management

An experiment is a TOML file in `experiments/` that names a dataset, the reference sources and the runs to compare:

```toml
name = "valencia-pilot"
dataset = "valencia-ciutat-vella"
truth = ["catastro", "evaluation_set", "osm_tags"]
buildings = "with_truth"          # or "all"; limit = 200 for a quick check

[analysis]                        # the BUILDING_ANALYSIS_* settings the dataset was analyzed with
BUILDING_ANALYSIS_MODEL = "balanced"
BUILDING_ANALYSIS_TILE_SIZE = "384"

[[runs]]
id = "prior"
kind = "prior"                    # the OSM-tag baseline

[[runs]]
id = "qwen3vl-4b"
kind = "vlm"
backend = "openai"                # or "transformers"
url = "http://127.0.0.1:11434"
model = "qwen3.5:4b"
options = { reasoning_effort = "none" }

[[runs]]
id = "qwen3vl-4b-no-region"
kind = "vlm"
backend = "openai"
url = "http://127.0.0.1:11434"
model = "qwen3.5:4b"
options = { reasoning_effort = "none" }
regional_guidance = false         # ablation: the same model without the construction history

[[runs]]
id = "survey"
kind = "saved"                    # answers already in the dataset, e.g. a Colab survey
provider = "vlm:Qwen/Qwen3-VL-8B-Instruct"
```

Analyses only count under the settings they were made with, so `[analysis]` names them; an experiment whose settings match no analysis stops with that explanation instead of scoring empty runs. `city-analyser-experiment run experiments/valencia-pilot.toml` runs each run into `data/<dataset>/experiments/<name>/<run>/`, never into the dataset's own attributes, so experiments do not change what the viewer shows. A run is resumable: a building whose inputs did not change is not asked again. Each run records its configuration, the prompt, guidance and taxonomy versions, the code revision, timings and errors in `run.json`.

`city-analyser-experiment report experiments/valencia-pilot.toml` scores every run against every reference source and writes `report.json`, `report.md` and `report.html` next to the runs: per attribute, the number of buildings with a reference value, coverage (answered share; abstentions lower coverage, not accuracy), accuracy, balanced accuracy and macro F1 for classes, mean absolute error and the share within one storey for storeys, mean absolute error and the share within ten years for years, the majority-class or median baseline, and the risk-coverage curve. Two more tables per attribute keep the comparison fair: the same buildings for every run (those answered by every run that answers at least half, so one run's abstentions do not change what the others are scored on), and, for the cadastre, residential and non-residential buildings apart (a cathedral's cadastral year and storeys do not measure what an apartment block's do). A run is only worth adopting if it beats both the baseline and the prior on the attributes it claims.

**First experiment to run.** One neighborhood of Valencia: the cadastre for storeys, year and use on every matched building, 200 buildings labeled for the structure, and four runs: the prior, Qwen3-VL-4B, Qwen3-VL-8B and the 4B without regional guidance. This says whether a model adds anything, which model size is worth the GPU, and whether the regional guidance helps.

## Order and dependencies for points 3 to 7

| Order | Point | Depends on | Estimate | Unlocks |
|---|---|---|---|---|
| 1 | 3. Imagery (Mapillary token, Panoramax) | nothing | 2-3 days | a real pilot neighborhood, experiments with several views per building |
| 2 | 4. Mask quality control | 3 (enough views to afford discarding bad ones) | 4-5 days for phases 0-2, 9-12 in all | trustworthy crops without hand curation |
| 3 | 5a. Frozen OSM snapshot | nothing (can go in parallel with 3) | 2 days | reproducible experiments |
| 4 | 6. GEM detail | 1-2 (reference data and experiments to measure it) | 3-4 days | vulnerability-relevant attributes |
| 5 | 5b-c. GPU throughput, job queue | 3 (a city to run) | 3-5 days | city-scale runs |
| 6 | 7. Area outputs | 6 (the attributes to aggregate) | 3-4 days | comparison with published exposure models |

Estimates are working days for the code, tests and docs, not for running the data. Each point ends with an experiment from section 2 that shows its effect; a step that cannot be measured is not done.

## 3. Enough imagery

**Update (October 17).** A new token works: `valencia-ciutat-vella` was downloaded (89 panoramas at 10 m spacing), analyzed on CPU in 2 h 20 min without failures, and matched to the cadastre (133 of 151 footprints). The pilot experiment (`experiments/valencia-pilot.toml`) runs on it. Panoramax is still to do.

**Diagnosis (checked October 2026).** The Mapillary token in `.env` is not active: the Graph API answers "An active access token must be used" even to `/me`, so every search returns empty. It is not a permissions problem. Panoramax, the open alternative, has over 480 equirectangular (360°) pictures in central Madrid (the query limit was reached, so more), mostly from 2023 to 2026; in central Valencia it lists 441 pictures whose metadata does not say whether they are 360°.

**Tasks.**

1. **Mapillary access.** Create a new client token at mapillary.com/dashboard/developers (an application with read access), put it in `.env`, and add a `city-analyser-download --check-token` that calls `/me` and says plainly whether the token works. Today a dead token looks like an area without imagery.
2. **Panoramax source.** A second downloader behind the same manifest format: search with `GET https://api.panoramax.xyz/api/search?bbox=...` (STAC), keep pictures whose `pers:interior_orientation.field_of_view` is 360 or whose image is 2:1, read position and heading from the STAC item, download the full-resolution asset, and record `provider: panoramax`, the picture's license (CC-BY-SA or the instance's) and the author in `ATTRIBUTION.md`. Panoramax is federated: the API above aggregates instances, and each item names its own.
3. **Selection for a pilot.** Reuse the spacing filter (`--min-spacing-m 10`) and the geometric index to choose panoramas that see the most buildings from two or more sides; report the expected views per building before downloading.
4. **Pilot neighborhood.** 2,000 to 5,000 buildings in Madrid (where Panoramax 360° coverage is proven) or Valencia (where the cadastre is already matched, if Mapillary works).

**Risks.** Panoramax headings can be missing or rough for some contributors (the pose refinement exists for this); mixing two sources in one dataset mixes camera heights and exposures (record the source per panorama and allow filtering in experiments).

**Done when** a pilot neighborhood has at least two usable views (coverage report, `usable_view_of_nearby`) for most buildings in line of sight, and the experiment on it runs end to end.

## 4. Automatic mask quality control

**Update (October 22): phases 0, 1 and 2 done; phase 3 next.**

Phases 0 and 2 (October 21, analysis version 29):

- **Benchmark.** The labels are in `benchmarks/mask-labels/`. Each file records the analysis version whose masks it describes, and `evaluate` refuses to score labels against masks from another version.
  - The 94 crop views of the Ciutat Vella pilot were labeled by eye from `tools/mask_review.py` panels. 58 of them are bad: the pilot's model read a bad crop most of the time.
  - The curated demos give 88 more through `city-analyser-mask-quality labels-from-curation`. A view the curation edited is bad. A view the demo showed with a reviewer's answer is good. A view it left as a sliver is bad. Views the demo never showed were not reviewed and are not used.
- **New failure.** The labels add a cause the demo did not have: panoramas taken inside the Mercat Central (`interior`). It is detected by the analysis's repaired camera still lying inside a footprint. The recorded GPS cannot be used for this: in narrow streets it falls inside a footprint for 199 of 531 views.
- **Score.** `city-analyser-mask-quality score` (`mask_quality.py`) applies thresholded rules, one per failure. On the 182 version-29 views they catch 70% of the bad views (92/132) and flag 14% of the good ones (7/50).
  - Ciutat Vella: 86% caught, 17% false alarms.
  - Madrid demo: 65% caught, 8% false alarms.
  - Valencia demo: 42% caught; it has only one good view.
- **Misses.** What the rules miss is mostly the demo's hand fixes of masks that reach a neighbor where the projected walls reach it too. Geometry cannot see those; it needs the image (phase 4) or better footprints.
- **Thresholds.** Sweeping each threshold trades recall for false alarms one for one, so they were left at their first values. A logistic regression waits for more labels.

Phase 1 (October 22, analysis version 31). Both changes act after the model stages, so Ciutat Vella re-associated from the stage cache in 5.5 minutes on CPU, with a peak of 460 MiB.

- **Leveling.** Walls are rendered for a level camera and resampled into the panorama's tilt (`camera_up`), with the viewer's rotation.
  - On the 36 panoramas tilted more than 5 degrees, the overlap between projected walls and detected buildings rose from 0.200 to 0.224, and was better in 22 of them.
  - The opposite rotation lowers it to 0.165, which confirms the sign.
- **Fallback cleanup.** Pieces and holes under 0.02% of the panorama are removed where only the projection names the pixels.
- **Cadastre heights: tried and removed.**
  - Version 30 gave footprints without an OSM height their cadastre storeys.
  - The analysis must work where no cadastre exists, and floors are scored against that same cadastre, so it would have been both a dependency and a leak.
  - The analysis reads no reference data; footprints without a height are still drawn 32 m high.
- **Effect across the dataset** (version 29 to 31):
  - Speckle flags fell from 64 to 37, street splits from 19 to 11, overhead views from 45 to 39 and neighbor leaks from 76 to 69.
  - Clean views rose from 178 of 564 to 192 of 538; buildings with a clean view from 71 to 76.
- **Effect on what the model reads.** The pilot's crops were exported again and labeled by eye.
  - 59 of 95 crop views are bad (62%), as in version 29 (58 of 94). 21 buildings have a good view, as before.
  - The dominant failures, distant slivers and the market interior, are not causes phase 1 can remove: phase 3 has to leave them out.
- **Score on the new masks.** It catches 85% of the bad views (50/59) and flags 14% of the good ones (5/36).
- **Demos.** `make presentation-data` rebuilt both demos with the same curated answers (10 and 23), and the curation's mask edits still apply.
  - Their labels from the curation describe version-29 masks and are kept as `*-v29.json`.
  - The version 29 and 30 analyses of Ciutat Vella are kept under `data/valencia-ciutat-vella/history/` for comparison.

**Next: phase 3, use the flag (planned October 22, about 2 days).** Nothing in it reads reference data. The cadastre stays an evaluation tool, shown in the viewer only in evaluation mode (`references=1`).

1. **Score while associating.** `associate_panorama` already has the projected walls, their depth and the masks, so it computes `mask_quality.view_signals` and `assess` for every building it names. It stores `{"reasons", "signals"}` in `building_ids[ordinal]["mask_quality"]`, and `ANALYSIS_VERSION` goes up. `city-analyser-mask-quality score` then only collects them, and `evaluate` is unchanged.
2. **Crops and ranking.** The ranking (`building_inspection.evidence`) keeps every view but marks the flagged ones. `export_building_crops` takes the best `per_building` clean views. Sidecars record the views set aside with their reasons, so nothing is deleted. `--keep-flagged` gives the old behavior, for comparison.
3. **Attributes and survey.** A building without a clean view is not sent to the model. Its model result abstains with "no clean view" and the reasons, while the OSM-tag prior still applies. A building with some clean views is asked only about those.
4. **Viewer.**
   - The card lists the views set aside and why: "2 views set aside: sliver, panorama inside a building".
   - The Review tab gets a "Mask to check" queue with the reason, which opens the panorama on that facade.
   - The map gets a "Mask quality" color layer: the share of a building's views that are clean.
   - In the presentation, the tour skips buildings without a clean view (instead of `min-score` alone), and the Look phase says why a view was set aside.
5. **Curated demos.** A curation is a human review, so a view it edited or answered counts as clean whatever the rules say. Its mask edits are applied before the views are counted.
6. **Tests and checks.** Unit tests for each piece, and the e2e presentation test with a flagged view. Screenshots at three sizes. Re-association of Ciutat Vella and of the demos (minutes, from the stage cache).

**Then phase 5: measure.** The pilot is run again with the filter and with `--keep-flagged`: the OSM-tag prior, qwen3.5:4b with and without regional guidance, and a new Claude run on the clean crops. The pilot's model answers are stale anyway, because version 31 changed the crops. It is scored against the cadastre and the evaluation set, which are references only. The filter stays if the scores hold or improve while coverage drops little. Expect coverage to drop: today 21 of the 40 buildings have a clean crop view.

**Diagnosis.** In the demo about thirty masks were wrong in five recurring ways, each of which a footprint projection can detect. The Ciutat Vella pilot (October 17) added three causes upstream of them:

- **Speckle.** Pixels no instance (detector plus SAM) covers fall back to the footprint projection, and the semantic mask (SegFormer at 384 px) is noisy there: some masks come out mottled with loose blobs.
- **Tilted cameras.** The analysis assumes a level camera, but 36 of the 89 Ciutat Vella panoramas are tilted more than 5 degrees (up to 25). Projected walls then shift vertically, which spoils exactly the "mask above the roof" signal. The viewer already levels panoramas with Mapillary's `computed_rotation` (`camera_up` in the manifest); the analysis does not.
- **Default height.** Most footprints are projected 10 m high because OSM has no height; the cadastre now gives storeys for most of them.

| Failure seen in the demo | Signal |
|---|---|
| The mask reaches into the neighbor | Mask extends beyond the bearing range of the building's projected walls |
| The mask crosses a street (two buildings, one id) | Mask splits into two bearing runs with a gap where the projection has none |
| The side facade of a corner is missing | A projected wall in line of sight has no mask pixels |
| The mask leaks into the sky | Mask top far above the projected roof height |
| A distant sliver gets a building id | Mask area tiny relative to the projected facade |
| Speckle (pilot) | Many small components, low solidity |

The approach is to remove the causes first, then score what remains by the agreement of three independent sources: the instance mask (image), the footprint projection (geometry) and the semantic mask. Where they disagree the view is suspect, and the score says why.

**Phase 0. Labeled benchmark (half a day).** Positives: the demo curation's mask corrections (`demo/*.curation.json`), labeled by failure type. More: a quick labeling mode in the viewer (one key for good, bad, and the reason) for about 150 views of Ciutat Vella, some 30 to 40 minutes of review. Metrics: bad views caught (recall), good views dropped (false alarms), and the effect on the pilot experiment.

**Phase 1. Remove the causes in the analysis (2 days).**

1. Clean the projection fallback: drop small connected components and close holes only in pixels outside instances; SAM masks are left alone.
2. Level before projecting: use `camera_up` in the wall projection and in the analysis' perspective views.
3. Cadastre height: storeys times about 3.1 m where OSM has no height, for the projection and as the expected roof for phase 2.

These change analysis results: bump `ANALYSIS_VERSION` and re-analyze (Ciutat Vella: about 2.5 h on CPU, minutes on a Colab GPU).

**Phase 2. Quality score per view (2 days).** `mask_quality.py`, one signal per row of the table above, each with a readable reason: bearing overlap between mask and projected walls, number of separate bearing runs, mask top against the expected roof, fragmentation (components and solidity), and area against the projected facade minus what trees hide (the vegetation mask). Thresholded rules calibrated on phase 0 first; a logistic regression on the same signals once there are a few hundred labels.

**Phase 3. Use it (1 to 2 days).** Views below the threshold leave the crops and the ranking, recorded with their reason, not deleted, and the model uses the next best view; a building without a clean view abstains with "no clean view" instead of answering from a bad mask. In the viewer: a "Mask to check" review queue with the reason, and a map layer colored by mask quality.

**Phase 4. Correct in the viewer (3 to 4 days).** First the three operations the demo curation used most (remove the view, keep a bearing range, trace a roof line), then claiming pixels for another building and filling between edges. Stored as reviewed, versioned corrections in `mask-corrections/`, like the footprint corrections, and re-applied at association time so a re-analysis keeps them.

**Phase 5. Measure (half a day plus the run).** The pilot experiment with and without the filter, against the run that finished on October 17 as the "before": the filter stays if the scores against the cadastre improve or hold while coverage drops little.

**Risks.** A wrong projection (pose or OSM footprint) makes a good mask look bad: the score marks a view as inconsistent with a reason, not as wrong, and the footprint diagnostics help tell the cases apart. Trees legitimately hide facades: the vegetation mask is discounted from the expected coverage.

**Start with** phases 0 to 2 (4 to 5 days, no new interface beyond quick labeling), which show with data how much the masks improve before the correction tools are built.

**Done when** the score separates the demo's corrected masks from the rest (most bad ones caught, few good ones dropped) and the experiment shows no loss.

## 5. Scale

**a. Frozen OSM snapshot.** Today footprints come from Overpass per 110 m cell and are cached for a while; a rerun months later may see other footprints, so results are not reproducible and city runs hit rate limits. Geofabrik publishes daily regional extracts (for example `europe/spain/valencia-latest.osm.pbf` and `madrid-latest.osm.pbf`).

1. `city-analyser-osm freeze <dataset>`: for a small area, one Overpass query for the dataset's bounding box; for a city, read the regional PBF (optional extra `osm` with pyosmium, BSD) and keep buildings only. Write `data/<dataset>/osm-snapshot.parquet` (GeoParquet, with the extract date and source).
2. With a snapshot present, `_cell_buildings` reads it instead of Overpass; the snapshot date goes into `footprints_sha256` and into every experiment's `run.json`.
3. Footprint corrections keep working on top (they are versioned against the OSM element version).

**b. GPU throughput.** Replace the estimates of the scaling page with measurements: the survey on 100 panoramas with `--report` on T4, L4 and A100 (Colab), with the transformers backend and with vLLM, recording seconds per panorama and per view and peak memory.

**c. Several GPUs.** A lease file per tile in shared storage (or a Postgres `SKIP LOCKED` table for larger setups): each worker takes the next free tile, the survey state merges, and the model runs behind one vLLM server per GPU.

**Done when** a rerun of an experiment on a frozen snapshot gives identical inputs, and a 10,000-panorama run has measured times.

## 6. GEM detail for seismic risk

**What the facade can show,** beyond material, lateral system and height, and what the regional guidance already asks the model to notice. The codes below follow the GEM Building Taxonomy v3 attributes (structural irregularity, floor, material technology); confirm each code against the GEM v3 glossary before implementing, as was done for the material and lateral system.

| Attribute | GEM v3 group | Classes to start with | Why it matters |
|---|---|---|---|
| Vertical irregularity | Structural irregularity, vertical type | soft storey (`SOS`), setback, none | Ground floors opened for shops are the main weakness of masonry centres |
| Floor system | Floor material | timber, steel or iron with masonry infill, concrete, unknown | Flexible timber floors make masonry walls act alone |
| Masonry unit | Material technology (masonry) | fired clay solid brick, hollow brick, stone (rubble or dressed) | Strength and typology |
| Roof shape | Roof | flat, pitched | Weight and shape |

**Tasks.**

1. Taxonomy, parser and prompt: each new attribute with its classes and abstention, the GEM code in the composed string (`MUR+CLBRS/LWAL/HEX:5/IRVP:SOS`), `PROMPT_VERSION` bump.
2. Card, review form, export columns and the experiment metrics: no new code beyond the attribute lists, if the lists stay the single source.
3. Reference: soft storeys and roof shape can be labeled from the imagery in the evaluation set; floor system and masonry unit mostly cannot (an inspection or the building's file is needed), so the model must be allowed, and expected, to abstain on them.

**Done when** the evaluation set covers the new attributes on a pilot and the experiment shows where the model can and cannot tell them.

## 7. Outputs for risk analysis

**Problem.** Results are per building; exposure and risk models work with assets grouped by area and taxonomy string.

**Tasks.**

1. **Areas.** Census sections (INE "secciones censales" boundaries for Spain) or a regular grid; each building is assigned by its footprint centroid.
2. **Aggregation** (`city-analyser-export --by section`): per area, buildings and dwellings by GEM taxonomy string, the storeys distribution, and the share of buildings described, abstained and reviewed, so a reader sees how much of the area the numbers rest on.
3. **Formats.** An OpenQuake exposure model (asset CSV with location, taxonomy string, number of buildings and dwellings; costs and occupants left to the user, who has the economic data), and a comparison table against the European exposure model ESRM20 for the same municipality.
4. **Uncertainty.** Carry the vote shares into the aggregation as a distribution over taxonomy strings per area, not only the top answer.

**Done when** a pilot area produces an exposure file OpenQuake accepts and a side-by-side comparison with ESRM20.

## Related tools and what this one lacks (October 2026)

Four systems are close to this one; none covers the whole loop from imagery to a reviewed record.

| System | What it does | Closest to |
|---|---|---|
| [OpenFACADES](https://arxiv.org/abs/2504.02866) (NUS, 2025) | Mapillary and OSM footprints joined by isovists to pick vantage points, facades detected in panoramas and reprojected, then **vision-language models fine-tuned on about 31,000 labeled images** predict type, material, age and floors; about 1.2 million images, half a million buildings. Paper under CC BY-NC-ND. | The analysis chain (see [OpenFACADES reuse plan](openfacades-reuse-plan.md)) |
| [BRAILS++](https://simcenter.designsafe-ci.org/products/backend-components/brails/) (NHERI SimCenter) | Building inventories for regional hazard simulation: stories, roof shape, height and soft storeys from Street View, with imputation of missing values; feeds the SimCenter damage workflows. | The purpose: an inventory for risk |
| [OpenBuildingMap](https://gee-community-catalog.org/projects/obm/) (GFZ, 2025) | 2.7 billion footprints with occupancy, height, stories and floorspace as GEM tags, derived from OSM and footprint datasets without imagery (ODbL); combined with ESRM20 in the European high-resolution exposure model. | The OSM-tag prior, at global scale |
| GFZ remote rapid visual screening, and [GEM's street-view classifier](https://www.globalquakemodel.org/GEMNews/deep-learning-speeds-up-building-classification) | Experts label buildings in GEM from omnidirectional Street View with OSM footprints; GEM's pipeline rectifies the facade and classifies structural system (88% accuracy), storeys and construction period. | The review tab; the attribute models |

**What they do and this tool does not.**

1. **Fine-tuned models.** OpenFACADES shows fine-tuned open models beating single-attribute classifiers and zero-shot GPT-4o. Here Qwen runs zero-shot and, in the first experiments, does not beat the baselines.
2. **Labels for the structure.** GEM's 88% rests on thousands of engineer-labeled buildings. The cadastre says nothing about structure, so `gem_material` and `gem_llrs` cannot be measured yet.
3. **A published prior to beat.** OpenBuildingMap and ESRM20 already give a GEM estimate for every building in Europe; that is the honest baseline.
4. **Output a risk model reads.** BRAILS feeds damage simulations, OBM and ESRM20 feed OpenQuake; this tool exports a table.
5. **Rectified facades.** A fronto-parallel facade makes storeys easier to count than a perspective crop.

**What this tool has that they do not.** The whole loop in one place: analysis, a 360° viewer, a card per building, a human review that wins over the model, a chat about the building with the model that described it, and a presentation mode; instance masks per building associated to OSM with occlusion and pose repair, measured on a benchmark; GEM v3 answers with visible reasoning, regional guidance, abstention and multi-view votes; reproducible experiments against the cadastre; Colab runs imported without losing local reviews.

**What to add, in order.**

1. **An engineer-labeled evaluation set**: a review mode that walks a reviewer through 100 to 200 buildings chosen to cover the typologies of an area, with "Use as ground truth" on.
2. **Reference values in the card**: the cadastre's storeys, year and use next to the answer, disagreements marked, and later the OBM or ESRM20 class of the area. Implemented for the cadastre (October 17).
3. **OpenBuildingMap or ESRM20 as an experiment run**, like the OSM prior: a model that does not beat it adds nothing.
4. **Fine-tuning** (LoRA on Qwen3-VL) once there are a few hundred to a few thousand reviewed buildings, plus OpenFACADES labels if their license allows.
5. **OpenQuake exposure export** and aggregation by area (point 7).
6. **Vulnerability attributes** (point 6): open ground floor (soft storey), irregularity, position in the block, roof.
7. **Viewer**: a height ruler to count storeys on a facade, the views of one building side by side, map filters by attribute or by disagreement with the cadastre, and the panorama leveling of the viewer applied to the analysis as well.

