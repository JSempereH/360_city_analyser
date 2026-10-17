# Experiments: what the tool needs to analyze cities (October 2026)

This page plans the work between a tool that can describe buildings and one that can run experiments and report how well it does. It covers seven points. Points 1 and 2 are implemented (sections 1 and 2 also say how to use them); points 3 to 7 are analyzed here for later.

The question every experiment must answer first: **does a vision-language model add anything over the baseline it replaces**, the OSM-tag prior and, in Spain, the cadastre? Without reference data that question has no answer, which is why points 1 and 2 come first.

| # | Point | Status |
|---|---|---|
| 1 | Reference data to measure against | Implemented: cadastre (Spain), engineer-labeled evaluation set, OSM tags |
| 2 | Experiment management | Implemented: experiment files, isolated runs, comparison report |
| 3 | Enough imagery | Planned below |
| 4 | Automatic mask quality control | Planned below |
| 5 | Scale (frozen OSM, GPU throughput, job queue) | Planned below |
| 6 | GEM detail that matters for seismic risk | Planned below |
| 7 | Outputs for risk analysis | Planned below |

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

`city-analyser-experiment run experiments/valencia-pilot.toml` runs each run into `data/<dataset>/experiments/<name>/<run>/`, never into the dataset's own attributes, so experiments do not change what the viewer shows. A run is resumable: a building whose inputs did not change is not asked again. Each run records its configuration, the prompt, guidance and taxonomy versions, the code revision, timings and errors in `run.json`.

`city-analyser-experiment report experiments/valencia-pilot.toml` scores every run against every reference source and writes `report.json`, `report.md` and `report.html` next to the runs: per attribute, the number of buildings with a reference value, coverage (answered share; abstentions lower coverage, not accuracy), accuracy, balanced accuracy and macro F1 for classes, mean absolute error and the share within one storey for storeys, mean absolute error and the share within ten years for years, the majority-class or median baseline, and the risk-coverage curve. Two more tables per attribute keep the comparison fair: the same buildings for every run (those answered by every run that answers at least half, so one run's abstentions do not change what the others are scored on), and, for the cadastre, residential and non-residential buildings apart (a cathedral's cadastral year and storeys do not measure what an apartment block's do). A run is only worth adopting if it beats both the baseline and the prior on the attributes it claims.

**First experiment to run.** One neighborhood of Valencia: the cadastre for storeys, year and use on every matched building, 200 buildings labeled for the structure, and four runs: the prior, Qwen3-VL-4B, Qwen3-VL-8B and the 4B without regional guidance. This says whether a model adds anything, which model size is worth the GPU, and whether the regional guidance helps.

## Order and dependencies for points 3 to 7

| Order | Point | Depends on | Estimate | Unlocks |
|---|---|---|---|---|
| 1 | 3. Imagery (Mapillary token, Panoramax) | nothing | 2-3 days | a real pilot neighborhood, experiments with several views per building |
| 2 | 4. Mask quality control | 3 (enough views to afford discarding bad ones) | 4-6 days | trustworthy crops without hand curation |
| 3 | 5a. Frozen OSM snapshot | nothing (can go in parallel with 3) | 2 days | reproducible experiments |
| 4 | 6. GEM detail | 1-2 (reference data and experiments to measure it) | 3-4 days | vulnerability-relevant attributes |
| 5 | 5b-c. GPU throughput, job queue | 3 (a city to run) | 3-5 days | city-scale runs |
| 6 | 7. Area outputs | 6 (the attributes to aggregate) | 3-4 days | comparison with published exposure models |

Estimates are working days for the code, tests and docs, not for running the data. Each point ends with an experiment from section 2 that shows its effect; a step that cannot be measured is not done.

## 3. Enough imagery

**Diagnosis (checked October 2026).** The Mapillary token in `.env` is not active: the Graph API answers "An active access token must be used" even to `/me`, so every search returns empty. It is not a permissions problem. Panoramax, the open alternative, has over 480 equirectangular (360°) pictures in central Madrid (the query limit was reached, so more), mostly from 2023 to 2026; in central Valencia it lists 441 pictures whose metadata does not say whether they are 360°.

**Tasks.**

1. **Mapillary access.** Create a new client token at mapillary.com/dashboard/developers (an application with read access), put it in `.env`, and add a `city-analyser-download --check-token` that calls `/me` and says plainly whether the token works. Today a dead token looks like an area without imagery.
2. **Panoramax source.** A second downloader behind the same manifest format: search with `GET https://api.panoramax.xyz/api/search?bbox=...` (STAC), keep pictures whose `pers:interior_orientation.field_of_view` is 360 or whose image is 2:1, read position and heading from the STAC item, download the full-resolution asset, and record `provider: panoramax`, the picture's license (CC-BY-SA or the instance's) and the author in `ATTRIBUTION.md`. Panoramax is federated: the API above aggregates instances, and each item names its own.
3. **Selection for a pilot.** Reuse the spacing filter (`--min-spacing-m 10`) and the geometric index to choose panoramas that see the most buildings from two or more sides; report the expected views per building before downloading.
4. **Pilot neighborhood.** 2,000 to 5,000 buildings in Madrid (where Panoramax 360° coverage is proven) or Valencia (where the cadastre is already matched, if Mapillary works).

**Risks.** Panoramax headings can be missing or rough for some contributors (the pose refinement exists for this); mixing two sources in one dataset mixes camera heights and exposures (record the source per panorama and allow filtering in experiments).

**Done when** a pilot neighborhood has at least two usable views (coverage report, `usable_view_of_nearby`) for most buildings in line of sight, and the experiment on it runs end to end.

## 4. Automatic mask quality control

**Diagnosis.** In the demo about thirty masks were wrong in five recurring ways, each of which a footprint projection can detect:

| Failure seen in the demo | Signal |
|---|---|
| The mask reaches into the neighbor | Mask extends beyond the bearing range of the building's projected walls |
| The mask crosses a street (two buildings, one id) | Mask splits into two bearing runs with a gap where the projection has none |
| The side facade of a corner is missing | A projected wall in line of sight has no mask pixels |
| The mask leaks into the sky | Mask top far above the projected roof height |
| A distant sliver gets a building id | Mask area tiny relative to the projected facade |

**Tasks.**

1. **Consistency score per view** (`mask_quality.py`): bearing intersection-over-union between mask and projected walls, share of projected walls covered, overshoot above the roof line, number of disjoint runs. The analysis already has both inputs; `tools/mask_review.py` computes the projection.
2. **Use it.** Views below a threshold are left out of crops and the ranking (recorded, not deleted), and listed in the viewer's review queue as "mask to check" with the reason.
3. **Fix it in the viewer.** The operations the demo curation used (keep a bearing range, trace a roof line, claim detected pixels, fill between edges, remove the view, give the mask to another footprint), drawn on the panorama and stored as reviewed, versioned corrections like the footprint corrections (`mask-corrections/`), applied at association time so a re-analysis keeps them.
4. **Calibrate the threshold** on the demo, whose 30 hand corrections are labeled examples of bad masks, and on a sample of the pilot.
5. **Measure.** An experiment with and without the filter: the filter is kept if model scores against the cadastre improve or hold while coverage drops little.

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
