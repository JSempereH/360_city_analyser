# User guide

## Setup

Choose the PyTorch build that matches the machine:

```bash
uv sync --extra cu126   # NVIDIA GPU (CUDA 12.6 wheels)
uv sync --extra cpu     # CPU only, avoids ~3 GB of CUDA libraries
```

This installs the `city_analyser` package from `src/` and the `city-analyser-*` commands used below. Run them from the repository root: datasets (`data/`), model weights (`models/`) and `.env` are read from the working directory. [architecture.md](architecture.md) describes the layout.

Pre-download the segmentation weights into `models/` so analysis also works offline:

```bash
uv run city-analyser-models            # b0, b1, b2, b5 and the Mask2Former "high" profile
uv run city-analyser-models all        # SegFormer b0-b5 plus "high" (~1.8 GB)
uv run city-analyser-models b0 b5      # aliases or Hugging Face model IDs
```

Add a Mapillary developer token to `.env` or the shell:

```bash
MAPILLARY_ACCESS_TOKEN=MLY|...
```

Do not commit tokens or downloaded imagery.

## Download

```bash
uv run city-analyser-download \
  --center 9.9281,-84.0907 \
  --radius-m 150 \
  --max-images 25 \
  --min-spacing-m 10 \
  --output data/san-jose-pilot
```

Use `--bbox=min_lon,min_lat,max_lon,max_lat` instead of `--center` when needed; keep the `=`, or a western longitude such as `-0.38` is read as an option. `--image-size auto` selects the best available derivative. Where panoramas are closer than `--min-spacing-m`, the earliest capture is kept; `--captured-after 2025-01-01` leaves out older ones, so a recent, sharper camera is not displaced by an old one at the same spot (`notebooks/colab_madrid_salesas.ipynb` uses it). The manifest also keeps each camera's rotation from Mapillary's 3D reconstruction, which the viewer uses to level panoramas taken with a tilted camera; for a dataset downloaded before this, `city-analyser-download --add-rotations --output data/<dataset>` adds it without downloading images.

The downloader requests panoramas at API level, limits each response to 50 MB, decodes the JPEG, requires an approximately 2:1 image, and records dimensions and SHA-256. A dataset contains `images/`, `manifest.json`, `manifest.geojson`, and `ATTRIBUTION.md`; manifest coordinates are camera locations.

## Viewer

```bash
uv run city-analyser-viewer
```

Open [http://127.0.0.1:8765/viewer/](http://127.0.0.1:8765/viewer/).

The viewer provides panorama and map navigation, Mapillary downloads, OSM footprints, deep links, background analysis, optional SfM, and building reports. Nearby navigation uses original GPS, valid headings, known sequence metadata when available, and loaded OSM obstructions.

- **Details panel** (right; a drawer on tablets and a bottom sheet on phones): **Building** shows the selected footprint (OSM version, correction, observed share, analyzed views) with report, edit and "Go to best view" actions; **Review** is the footprint review queue; **Jobs** lists background jobs with progress and cancellation, and every status message.
- **Selection and search:** any footprint can be selected, even one without a facade in the current panorama. The sidebar search takes an OSM element (`way/123`, `relation/45`, `way/123/part-2`) and opens its best analyzed view, or coordinates (`9.934, -84.078`) and opens the nearest panorama.
- **Layers:** building masks, tree-occluded facades and detected instances on the panorama; footprints, captures, view direction, footprint diagnostics and analysis coverage on the map.
- **Batch analysis:** **Advanced > Analyze whole dataset** or **Analyze panoramas in map view** run in the background and skip current analyses; the panorama list marks each capture as current, stale or not analyzed.
- **Settings and shortcuts** (`?`): theme (system, dark, light) and every keyboard shortcut. Panel, tab, map and layer choices are remembered per browser.

## Analysis

Click **Analyze buildings** to process the current panorama and up to six nearby eligible captures. The pipeline:

1. Validates finite GPS/heading and an equirectangular image of at least 512x256.
2. Segments perspective views with SegFormer or Mask2Former (building / vegetation evidence). The model is chosen by `BUILDING_ANALYSIS_MODEL` (below).
3. Detects individual buildings: Grounding DINO proposes one box per visible building in six portrait views (90 degrees wide, -29 to +83 degrees of elevation) and SAM 2 outlines each box. Masks are merged across views into one instance map.
4. Loads OSM ways, multipolygon relations, holes, and building parts (disk-cached per ~110 m cell).
5. Moves a GPS fix that falls inside a footprint to the nearest street-connected open space (never a courtyard), then refines the pose against the detected buildings.
6. Projects facade walls with depth ordering and partial-edge occlusion, and names each detected building with the footprint that dominates it. When the detector merged attached facades (row houses), the building is split by panorama columns, because walls are vertical; this also happens when one footprint dominates but another wins a continuous run of at least 3 degrees holding 15% of the instance, so a neighbor seen as one with the building gets its own part. Walls are projected for a level camera and then turned into the panorama's own tilt (`camera_up`, from Mapillary's `computed_rotation`, the same rotation the viewer levels with), so a tilted capture does not shift them up or down. A footprint OSM gives no height is drawn 32 m high. The analysis never reads reference data (the cadastre or labels): those only measure it.
7. Continues labels up through building pixels no footprint claims, so low OSM heights and view limits do not cut roof lines. Where no detected instance covers the pixels, the semantic mask decides the extent: pieces of it smaller than 0.02% of the panorama are dropped there and holes as small are closed, so those masks are not mottled.
8. Publishes direct facade evidence, vegetation-completed evidence, and the visual instance map.

Artifacts are written under `data/<dataset>/analysis/` (`<id>.json`, `<id>-ids.png`, `<id>-facades.png`, `<id>-instances.png`). Cache identity includes source bytes, GPS, heading, every model setting, SfM input, and mask hashes. Models remain loaded per process.

The model stages are cached separately, under `data/.stage-cache/`, keyed by the panorama bytes, the model and its downloaded revision, and the stage's own settings. Changing association code, `ANALYSIS_VERSION` or the minimum confidence therefore re-associates panoramas in seconds instead of running the models again, and a panorama shared by several datasets is segmented once.

To analyze a whole dataset from the command line, overlapping model inference with OSM association:

```bash
uv run city-analyser-analyze san-jose-pilot --report analysis-report.json
```

The report lists per-panorama model and association times, cached stages and peak memory. Panoramas whose analysis is already current are skipped unless `--force` is given.

| Variable | Purpose |
|---|---|
| `BUILDING_ANALYSIS_DEVICE=auto` | `auto`, `cpu`, `cuda`, `cuda:<n>` or `mps` for every model stage |
| `BUILDING_ANALYSIS_MODEL=auto` | Default: Mask2Former `high` on a CUDA GPU with at least 12 GB, SegFormer-B5 on a smaller GPU, SegFormer-B0 otherwise |
| `BUILDING_ANALYSIS_MODEL=balanced` | SegFormer-B5 profile; `b0`..`b5` select a SegFormer size |
| `BUILDING_ANALYSIS_MODEL=high` | Large Mapillary Mask2Former profile |
| `BUILDING_ANALYSIS_MODEL=<model-id>` | Compatible Hugging Face model |
| `BUILDING_ANALYSIS_MODEL_REVISION=<commit>` | Immutable model revision |
| `BUILDING_ANALYSIS_TILE_SIZE=384` | Semantic tile edge = model input resolution, 256-1024 (default 512 SegFormer, 384 Mask2Former) |
| `BUILDING_ANALYSIS_BATCH_SIZE=4` | Semantic tiles per forward pass, 1-12; CUDA out-of-memory halves it |
| `BUILDING_ANALYSIS_MIN_CONFIDENCE=0.30` | Semantic evidence threshold |
| `BUILDING_ANALYSIS_INSTANCES=on` | Building instance stage (`off` falls back to projection-only separation) |
| `BUILDING_ANALYSIS_INSTANCE_DETECTOR=<model-id>` | Grounding-DINO-family detector (default `IDEA-Research/grounding-dino-tiny`) |
| `BUILDING_ANALYSIS_INSTANCE_SEGMENTER=<model-id>` | SAM 2 checkpoint (default `facebook/sam2.1-hiera-small`) |
| `BUILDING_ANALYSIS_INSTANCE_VIEWS=6` | Instance views around the horizon, 4-8 |
| `BUILDING_ANALYSIS_INSTANCE_DETECTOR_SIZE=800` | Detector input short edge, 384-1024 |
| `BUILDING_ANALYSIS_INSTANCE_BATCH_SIZE` | Views per detector pass (default 1 on CPU, all on GPU) |
| `BUILDING_ANALYSIS_INSTANCE_BOX_THRESHOLD=0.30` | Detector confidence threshold (detectors are calibrated differently) |
| `BUILDING_ANALYSIS_OSM_CACHE_DIR=off` | OSM footprint snapshot directory (default `data/.osm-cache`), or `off` |
| `BUILDING_ANALYSIS_OSM_CACHE_DAYS=30` | Age after which a cached OSM cell is fetched again |
| `BUILDING_ANALYSIS_STAGE_CACHE=on` | Model-stage cache; `off` always runs the models |
| `BUILDING_ANALYSIS_STAGE_CACHE_DIR` | Model-stage cache directory (default `data/.stage-cache`); set it on a remote worker (see [Remote inference](remote-inference.md)) to keep the cache between requests |
| `BUILDING_ANALYSIS_FOOTPRINT_CORRECTIONS_DIR` | Reviewed footprint corrections (default `data/footprint-corrections`) |

The semantic tile size is the resolution the model actually runs at, so it directly trades latency for detail:

```bash
uv run city-analyser-benchmark IMAGE.jpg --model b0 --model b5 --tile-size 384 --output benchmark.json
```

This measures runtime and memory, not accuracy; use `evaluation/` for accuracy.

### Hardware

Measured on a 4-core laptop CPU (no GPU), one 2048x1024 panorama:

| Stage | Time | Peak RAM |
|---|---|---|
| SegFormer-B0 / B5 semantic (512 px / 384 px tiles) | ~5 s / ~18 s | < 2 GB |
| Instances (Grounding DINO tiny + SAM 2.1 small, 6 views) | ~95 s | ~2.5 GB |
| OSM, pose and association | ~1-2 s | < 1 GB |

**Fast CPU mode (optional, AGPL-3.0).** YOLOE with its own masks instead of Grounding DINO + SAM 2 runs the instance stage in about 6 s per panorama on the same CPU, at a small measured cost (separation F 0.799 vs 0.813, detected 0.94 vs 0.97, more leak). Ultralytics is AGPL-3.0, so it is an opt-in extra, not a dependency:

```bash
uv sync --extra cpu --extra yolo
export BUILDING_ANALYSIS_INSTANCE_DETECTOR=ultralytics:yoloe-11l-seg.pt
export BUILDING_ANALYSIS_INSTANCE_SEGMENTER=detector
export BUILDING_ANALYSIS_INSTANCE_BOX_THRESHOLD=0.10
```

The first run downloads the YOLOE weights and its MobileCLIP text encoder into `models/ultralytics/`. Measured end to end with SegFormer-B0: about 10 s per panorama warm and 1.8 GB peak RAM on the 4-core laptop CPU.

On CPU-only machines either accept the instance cost, reduce `BUILDING_ANALYSIS_INSTANCE_VIEWS`, or set `BUILDING_ANALYSIS_INSTANCES=off`, or send inference to a GPU worker ([remote inference](remote-inference.md)). Run one analysis process at a time on machines with 16 GB RAM or less. GPU memory has not been measured in this repository. Weights for SegFormer-B5 plus the instance models are about 0.6 GB in half precision, so a 4 GB GPU should hold them; batches halve automatically on CUDA out-of-memory. Budget more for Mask2Former `high` (about 216 M parameters).

## SfM and multi-view

**Advanced > Build sparse depth (SfM)** runs optional heading-aware PyCOLMAP reconstruction. Depth is accepted only after GPS, pose-count, reprojection, track-length, center-spread, and multi-panorama observation checks. Invalid or stale SfM is ignored; run building analysis again after building depth.

Nearby evidence is fused by OSM ID. Captures less than 3 m apart do not count as independent confirmation, and confidence is averaged per independent view rather than per pixel.

## Footprint corrections

When the imagery shows that an OSM footprint is wrong, select it and click **Edit footprint**: move vertices or the whole outline, or **Split** it along a party wall, while its walls are drawn live on the panorama next to the OSM original; then propose the correction. The **Review** tab is a queue of proposals and of suspected problems found by the analyses (several buildings in one footprint, buildings without a footprint, footprints without a building, consistent pose shifts): `N`/`P` move through it, opening the panorama that shows each item, and `A`/`X` accept or reject proposals. An accepted correction applies only while OSM serves the element version it was drawn on, and affected panoramas are re-associated in seconds. `uv run city-analyser-footprints export --format osm` writes a JOSM layer for mapping corrections in OSM by hand; nothing is uploaded automatically. See [docs/footprint-corrections.md](footprint-corrections.md).

## Reports

Select a matched OSM building and click **Generate report**. Reports contain ranked evidence, input checks, limitations, and artifact hashes. Only current OSM-based reports load; JSON and HTML are not exposed by the generic `/data/` route.

Reports assess image-to-footprint association. They do not certify surveyed position, structural systems, foundations, reinforcement, code compliance, or GEM attributes; GEM attributes from a vision-language model are a separate output (see [Survey agent](#survey-agent)).

## Image quality, building index, crops and attributes

These tools come from the [OpenFACADES reuse plan](openfacades-reuse-plan.md).

**Image quality.** Every analysis records `image_quality` in its metadata: sharpness (variance of the Laplacian over the facade band of the panorama, resized to 2048 px wide), brightness, file size and the flags `blurry`, `dark`, `overexposed` and `small_file`. Flagged views stay listed but their report score is halved and the reason is shown. Thresholds are `BUILDING_ANALYSIS_MIN_SHARPNESS` (default 100), `BUILDING_ANALYSIS_MIN_BRIGHTNESS` (20), `BUILDING_ANALYSIS_MAX_BRIGHTNESS` (200) and `BUILDING_ANALYSIS_MIN_FILE_BYTES` (20480); changing one re-associates the affected panoramas. The sharpness default was set on 24 local panoramas (clean ones scored 405 to 1087; a mild blur scored 178 and a stronger one 33), so recalibrate it on your own imagery. `city-analyser-analyze --skip-low-quality` leaves flagged panoramas out before any model runs.

**Building index.** `city-analyser-index <dataset>` writes `building-index.json`: for every OSM building, which panoramas see it, with angle of view, distance and visible wall length, from the footprints alone (milliseconds per panorama, no models). It uses the recorded GPS position repaired out of footprints, ignores trees and vehicles, and therefore lists more buildings than the analysis later matches. `city-analyser-analyze --only-buildings way/1,way/2 --max-views-per-building 5` analyzes only the panoramas that best show those buildings (`--min-aov` sets a minimum angle of view; the default is 0 because requiring 10 degrees, as OpenFACADES does for its crops, dropped most buildings of a street from the local benchmark).

**Crops.** `city-analyser-crops <dataset>` writes, for every analyzed building, its best `--per-building` views as 448 px perspective crops under `crops/<osm-type>/<id>/`, each with a JSON sidecar naming the panorama, OSM version, view angles, evidence, image quality and SHA-256 hashes. `--mask-others` paints other buildings gray (neighboring facades in a row otherwise give near-identical crops) and `--skip-flagged` omits flagged views.

**Mask quality.** `city-analyser-mask-quality score <dataset>` checks every view of every building against its OSM walls rendered from the analysis pose and writes `mask-quality.json` with the signals and a list of reasons per view: `interior` (the panorama was taken inside a footprint), `sliver` (under 0.2% of the panorama), `neighbor` (over 30% of the mask where the walls do not project), `street` (two runs of mask with a gap of 3 degrees or more the walls do not cross), `sky` (over 25% above the expected roof), `speckle`, `overhead` (mostly above 50 degrees of elevation) and `missing` (under 15% of a visible facade). Thresholds are constants in `mask_quality.py`, set on the labeled views in `benchmarks/mask-labels/`; `city-analyser-mask-quality evaluate benchmarks/mask-labels/*.json` measures them (bad views caught, good views flagged, per reason). Each label file records the analysis version whose masks it describes; after a re-analysis `evaluate` asks for new labels instead of scoring old ones. The analysis scores every view while it associates (`mask_quality` in each building's record), and `city-analyser-mask-quality score` only collects those scores. Flagged views stay in the analysis and the ranking but are left out of the crops: `city-analyser-crops` writes them, with their reasons, to `set-aside.json` next to the building's crops (`--keep-flagged-masks` exports them anyway). A building whose every view is flagged is not shown to the model: its answer abstains with "no clean view" and the reasons. In the viewer the card says which views were set aside and why, the Review tab has a "Mask to check" queue (buildings without a clean view first) and Layers > Building colors has a "Mask quality" layer. A curated demo's edited or answered views count as clean.

**Attributes.** `city-analyser-attributes <dataset>` writes `attributes/<osm-type>/<id>/attributes-<provider>.json`. The default `prior` provider reads OSM tags (`building`, `building:material`, `building:levels`, `start_date`) and abstains where a tag is missing or ambiguous; it is the baseline a visual provider must beat. `--provider vlm --vlm-url <server> --vlm-model <name>` asks a multimodal model behind an OpenAI-compatible chat endpoint (for example vLLM on a cloud GPU) one structured question per crop and pools the answers per building, weighting each by its ranking score. `--vlm-backend transformers --vlm-model Qwen/Qwen3-VL-4B-Instruct` loads the model in the same process instead (GPU recommended; `--vlm-cache-dir` sets where the weights go). Predictions use the classes of the OpenFACADES taxonomy (10 building types, 7 surface materials, floors, construction year) plus the structure in the [GEM Building Taxonomy v3](https://github.com/gem/gem_taxonomy): `gem_material`, the material of the lateral load-resisting system (`CR`, `MUR`, `MCF`, `S`, `W` and the other v3 codes), and `gem_llrs`, the type of that system (`LFM`, `LFINF`, `LWAL`, `LDUAL`, `LFLS` and others). The unknown codes `MAT99` and `L99` are never classes: they are abstentions. The model is asked to write what it observes (openings, columns, slab edges, wall thickness, storey lines) before answering, and those observations are stored with each view. Every file records the prompt version and a fingerprint of its inputs, so an unchanged building is not asked again. Predictions keep the full distribution, and abstain when views are too few or disagree (`--min-views`, `--min-share`). Model answers are votes, not calibrated probabilities, and the file says so (`score_kind`). Score a provider against OSM tags with `uv run python -m evaluation.attributes --dataset <dataset> --provider <provider>`; it reports coverage, accuracy, macro-F1, the majority-class baseline and a risk-coverage curve. OSM tags are noisy and sparse, so this measures agreement with OSM, not accuracy.

**Robustness.** `evaluation/corruptions.py` degrades panoramas (occlusion, motion blur, Gaussian noise, brighter, darker; three severities each). Run the `semantic` and `instances` evaluation commands with `--corruption motion_blur:2` and a distinct `--variant`, associate and score with a tag named `<prefix>-motion_blur-2`, then `uv run python -m evaluation.run robustness --clean <tag> --prefix <prefix>` prints the separation error added by each corruption, and the relative mean corruption error against a reference configuration when one is given.

## Survey agent

`city-analyser-survey <dataset>` is the unattended version of the viewer workflow: it walks a downloaded dataset across the city and leaves a description of every building it sees. The panoramas are split into square tiles (`--tile-size`, default 250 m) visited in a serpentine order. In each tile it analyzes the panoramas with `city-analyser-analyze`, then picks the buildings whose every view in the geometric index has been visited, exports their best crops (other buildings painted gray), writes the OSM-tag prior and asks the vision-language model. A building at a tile border therefore waits for the next tile instead of being described twice.

```bash
uv run city-analyser-survey valencia-centro --vlm-model Qwen/Qwen3-VL-4B-Instruct                  # model in process (GPU)
uv run city-analyser-survey valencia-centro --vlm-backend openai --vlm-url http://gpu:8000 --vlm-model Qwen/Qwen3-VL-8B-Instruct
uv run city-analyser-survey valencia-centro --no-vlm                                                 # OSM-tag prior only
```

Every step is idempotent and the run can be stopped at any time (Ctrl+C or SIGTERM finish the current step). `survey-state.json` records the tiles and buildings already handled; a rerun continues there, and a building is described again only when one of its views changed. `--max-tiles` and `--max-buildings` bound a run for a small validation; `--max-views-per-building` analyzes only the best panoramas of each building to save GPU time; `--force` revisits everything but still reuses current analyses and model answers.

In the viewer, clicking a footprint on the map or a building in the panorama opens a card with the GEM string (for example `CR/LFINF/HEX:5`; `MAT99`, `L99` and `H99` mark what could not be determined), each attribute with its source, and the model's observations next to the crops it saw. A human-entered OSM tag wins over the model, and the card says when the two disagree. `city-analyser-export <dataset> --provider vlm_<model>` writes the same attributes, GEM string included, to GeoJSON, CSV and GeoParquet.

**Map colors.** Layers > Building colors paints the footprints by GEM material, GEM lateral system, floors or review status, with a legend of the classes on screen and their counts. Materials and systems are grouped into three families each (concrete, masonry, other; frame, wall, dual or other) so neighboring footprints stay distinguishable, including for color-blind readers; hovering a footprint shows its exact GEM string. Buildings the survey has not reached stay faint. The colors refresh every minute, so a survey running in the background fills the map as it goes.

**Review.** The card's **Review** button opens a form that starts from what the card shows: pick the GEM material and lateral system (or "Cannot tell from the imagery"), the storey count, and an optional note. The review is stored as one more provider, `attributes-human-review.json`, and wins over OSM tags and the model; every decision, including a withdrawal, is appended to `reviews.jsonl` with the reviewer's name. Review > Show > **Structure to check** lists the model answers worth a look, weakest first: a split vote, a vote share below 60 % (a single confident view scores 67 %, because each view gives its second choice half a vote), a storey count that contradicts OSM, or views the model failed on. <kbd>A</kbd> (Confirm) records the values shown as your review, <kbd>E</kbd> opens the form, <kbd>N</kbd> moves on. Reviewed buildings are the labeled set to score the model against.

`city-analyser-export <dataset> --provider merged` exports what the card shows; each `*_kind` column says where the value came from (`review`, `tag` or `vote_share`). `--provider human-review` exports only the reviews.

The model's answers are uncalibrated votes from a general-purpose model, not surveyed structural data: check a sample against the crops, review what the queue flags, and score the model against reviewed buildings before using its answers in a risk model.

## Assistant: analyze and talk in real time

Start the viewer with a vision-language model and the **Assistant** tab comes alive for the selected building:

```bash
# A local model with Ollama (ollama pull qwen3.5:4b), or any OpenAI-compatible server (vLLM, LM Studio, a cloud API)
ATTRIBUTES_VLM_URL=http://127.0.0.1:11434 ATTRIBUTES_VLM_MODEL=qwen3.5:4b \
ATTRIBUTES_VLM_OPTIONS='{"reasoning_effort": "none"}' uv run city-analyser-viewer
# Or the model in the viewer process (GPU): ATTRIBUTES_VLM_BACKEND=transformers ATTRIBUTES_VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct
```

- **Analyze now** asks the model about every view of the building while the camera turns to the view it is reading; its field notes stream in, then the answer is saved like a survey's and the card shows it.
- **Chat**: ask why it chose a structure, how it counted storeys, or whether a correction is supported. The model sees the building's best photos, its current record (with the source of each value and any review) and the region's construction history. It never changes the record.
- **Edit values** (or Review on the card) corrects any attribute by hand: GEM material and lateral system, storeys, use, facade material and construction year. Your decision wins over OSM tags and any model, and stays when new model results are imported.

`ATTRIBUTES_VLM_OPTIONS` passes extra request fields to the server. Reasoning models served by Ollama (Qwen3.5, for example) otherwise spend the answer budget thinking out of sight and return nothing; `{"reasoning_effort": "none"}` makes them write their reasoning as the visible field notes instead. On a CPU an answer takes one to two minutes per view; on a GPU a few seconds.

**Regional knowledge.** Every question to the model carries the construction history of the region the camera is in (`attributes/regional.py`), as an engineer would bring it: for Spain, load-bearing masonry before about 1940, mixed systems in the 1940s and 1950s, concrete frames with brick infill from the 1960s and waffle or flat slabs (`LFLSINF`) from the 1970s, plus the rules to read a facade and to flag soft storeys. It is general knowledge, never the building's OSM tags.

## Presentation mode

The viewer can run the survey as a show, to explain what the tool does. Press **▶ Run survey** in the toolbar (or open the viewer with `present=1`): it says what the audience is about to see, then walks the panorama sequence by itself, building after building. The HUD at the bottom names the phase of each visit:

| Phase | What happens on screen |
|---|---|
| Walk | The camera turns to the next capture and steps into it, as in Street View: horizontal turns, then a gentle look up, never a roll. Panoramas taken with a tilted camera are shown leveled. |
| Look | It clicks the building: its mask (the orange area) appears, and the crop the model was given is shown next to its field notes for that view. A view the model could not use is set aside, in its own words. |
| Reason | It compares the storey counts and the structure across the views, and says when it will not guess. |
| Record | It fills in floors, GEM material and lateral load-resisting system, composes the GEM v3 string, "saves" it and colors the footprint on the map. |
| Check | It says which answers it leaves to an engineer. In evaluation mode (`references=1`) and when the building is matched to the cadastre, it also compares the record with it and explains a difference (a mezzanine or set-back attic the street does not show, a cadastral year that is a refurbishment). |

To show the panorama large, <kbd>F</kbd> (or the focus button in the toolbar) hides the library column, the details panel and the capture strip, and brings them back as they were; it works in the viewer and during the presentation, and is remembered. During the presentation, <kbd>H</kbd> (or **Hide notes** in the bar at the bottom) folds the assistant's notes away while they keep filling, and the map button hides the mini-map.

Each walk ends with a summary: what was found, the map enlarged and colored by GEM material, lateral system and floors in turn, and the review queue an engineer would work through. Then it starts again, or continues with the next dataset of `then`. <kbd>Space</kbd> pauses, <kbd>→</kbd> skips ahead, <kbd>Esc</kbd> returns to the viewer. The presentation writes nothing.

```bash
make presentation-data      # rebuild both demo copies from demo/*.curation.json and match them to the cadastre
make presentation           # serve them; the command prints the URL
# http://127.0.0.1:8765/viewer/?dataset=valencia-centro-demo&present=1&then=madrid-centro-demo
```

`city-analyser-demo <dataset>` prepares a copy of an analyzed dataset (panoramas hard-linked, analyses copied, the original untouched): crops, OSM-tag priors and, for buildings without a real model answer, simulated answers built from the OSM tags, the analyzed height and the building's age and use. They pass through the same fusion and card code as real answers. A dataset surveyed with a real model (`city-analyser-survey`) is presented as it is, with its model named in the HUD; demo answers are marked "Demo dataset".

For a public presentation, `demo/<dataset>.curation.json` holds a reviewed version: mask corrections checked by eye with `tools/mask_review.py --full` (keep a bearing range, trace a roof line, fill or claim views, remove a view), buildings to `exclude`, and per-view `answers` written from the crops (provider `vlm:curated-demo`). `make presentation-data` rebuilds both demo copies from it, matches them to the cadastre and `make presentation` serves them.

With a model configured, the presentation runs **live**: the model analyzes each building as the camera reaches it, and the field notes on screen are its own, generated then. Without one, or with `live=0`, it replays the saved answers.

URL options, kept while the presentation runs: `speed=2` (pace; the default 1.5 keeps a building to about 40 seconds, and 1 reads at about 330 words a minute), `min-score=0.03` (leave out buildings whose best view shows only a sliver of facade; a higher value gives a shorter tour), `then=other-demo,third-demo` (continue with other datasets after each walk), `live=0` (replay even with a model configured), `references=1` (evaluation mode, below).

**Evaluation mode.** The analysis and the model never read reference data, and most cities have no cadastre, so by default the viewer does not show it either. With `references=1` in the viewer URL (with or without `present=1`), the building card adds the cadastre's value next to each answer (marked when they differ) and the parcel with a link to its page, and the presentation's Check phase compares the record with it. Use it to evaluate, not to present the tool.

## From Colab (or any GPU machine) to the viewer

Run the survey where there is a GPU, then bring the results to the viewer on your laptop:

```bash
uv run city-analyser-results export valencia-centro results.zip --with-images    # on Colab or the GPU machine
uv run city-analyser-results import results.zip                                  # on the laptop
```

The archive holds the analyses, crops, model answers, index and survey state (and the panoramas with `--with-images`). Importing never overwrites a reviewer's decision (`attributes-human-review.json`, `reviews.jsonl`), so you can correct values in the viewer and import a later run without losing them. An analysis is only valid under the `BUILDING_ANALYSIS_*` settings it was made with: the import prints them so you can start the viewer with the same ones.

## Experiments

To know whether a model is worth using, score it against reference data: the Spanish cadastre (storeys, year, use), an evaluation set of buildings an engineer labeled in the viewer ("Use as ground truth" in the review form; the only reference for the structure) and OSM tags. An experiment file in `experiments/` names a dataset, the reference sources and the runs to compare (the OSM-tag prior, models with and without the regional guidance, answers already saved by a survey):

```bash
uv run city-analyser-groundtruth catastro valencia-ciutat-vella  # match footprints to the cadastre
uv run city-analyser-experiment run experiments/valencia-pilot.toml
```

Runs go to `data/<dataset>/experiments/<name>/<run>/` and never change what the viewer shows; the report (`report.html`, `.md`, `.json`) compares coverage, accuracy, macro F1, error and bias per attribute and source. See [Experiments](experiments.md) for the plan, the caveats of each source and the next steps.

## Processing a whole city

Everything below is built to be stopped and restarted: Colab disconnects, the laptop sleeps, a server answers 503. Nothing finished is redone.

**Download.** `city-analyser-download --tiled-bbox=min_lon,min_lat,max_lon,max_lat --output data/<city> --max-images 200000 --image-size 2048 --workers 8` queries the area tile by tile (the Mapillary images endpoint accepts at most 0.01 deg² per query), removes panoramas seen twice on a tile border, keeps cameras at least `--min-spacing-m` apart, downloads in parallel with retries on network errors, and saves the manifest as it goes. Rerunning the same command skips the files already on disk. `--image-size 2048` avoids downloading originals the analysis shrinks anyway. A tile that reaches `--max-api-pages` is reported, because panoramas beyond the cap in that tile were not seen.

**Analysis.** `city-analyser-analyze <city>` writes two small files next to the manifest:

- `batch-ledger.jsonl` records every panorama as done, failed or skipped, with the settings and the image's size and modification time. A rerun starts at the first unfinished panorama and answers "already done?" from the ledger in about half a second per 20,000 panoramas, without hashing images (earlier versions re-read the whole manifest for every panorama, which took over an hour at that size). Changing a model or threshold, or replacing an image, redoes exactly the affected panoramas; `--force` redoes everything and `--verify` re-validates finished panoramas from their files.
- `batch-progress.json` holds counts, the recent rate, an estimated time left (from the last 40 completions, so a burst of cache hits does not promise an impossible finish) and the latest errors.

**Errors.** One bad panorama never stops the run. A transient error (network, OpenStreetMap, GPU memory) is retried `--max-attempts` times with backoff and, if it keeps failing, recorded as failed and retried by the next run, up to 6 failures in total (`--retry-failed` overrides that). Bad input (no heading, wrong aspect ratio) is recorded as skipped and not retried. If `--max-consecutive-failures` panoramas fail in a row (default 10) the run stops with state `aborted`, because that usually means a broken environment and continuing would only burn GPU hours. An interrupt or SIGTERM, which is what Colab sends, finishes the current panorama and saves; a second interrupt aborts at once.

**Watching it.** `city-analyser-progress <city>` writes `progress.html`: a progress bar, percent resolved, estimated time left, panoramas per minute, counts by status and a map of every panorama colored by status (done, failed, skipped, pending), plus the latest errors. `--watch 15` regenerates it every 15 seconds until the run ends. The page is self-contained, so it opens from a browser, from Google Drive's file preview, or inline in a notebook (the Colab notebook does this). `city-analyser-analyze` also writes it when it finishes.

**Results.** `city-analyser-export` (next section) writes the per-building table and the coverage report.

**Limits that remain.** The unmeasured one is throughput: no GPU run has been timed, so the estimated time left only becomes meaningful once the first panoramas have finished. Thousands of small files in one Drive folder are slow to list and write; for a very large city work on the notebook's local disk and copy results to Drive periodically. Crop export and the building index read every analysis, which is fine for hundreds of buildings and slow for a whole city.

## Building table and coverage report

`city-analyser-export <dataset> [--provider osm-tag-prior]` turns the per-panorama analyses into per-building results under `<dataset>/exports/`:

| File | Content |
|---|---|
| `buildings.geojson`, `buildings.csv` | One row per OSM building (parts of split footprints included): identity, centroid, OSM version, applied correction revision, number of views and of usable and flagged ones, `association_status` (`multi_view`, `single_view` or `weak_views_only`), the best view and its score, capture dates, and the joined attribute columns (value, score and score kind, abstention flag; empty when no provider file exists) |
| `buildings.parquet` | The same as GeoParquet 1.0 (WKB footprints, CRS84); written only when `pyarrow` is installed (`uv sync --extra parquet`) |
| `building_views.csv` | Every panorama behind every row, with observed fraction, projection coverage, score, quality flags and whether it counts as usable |
| `coverage.json` | Panoramas analyzed, current, stale; footprints near the captured streets, in line of sight, matched, and usable; share per status; views-per-building histogram; examples of buildings seen but not matched |
| `export-manifest.json` | Versions, row count, a hash of the inputs, a hash of every file, the usable-view threshold and the OSM and Mapillary attribution |

It reads each analysis once and, like the batch run, trusts the run ledger instead of hashing every image (`--verify` validates from the files). Coverage is measured against the footprints within reach of the captured streets (the building index is built on first use; `--no-index` skips it), so buildings nobody photographed show up as a gap. A view counts as usable when it is not flagged for quality and the building fills at least `--min-observed-fraction` of the panorama (default 0.005). That threshold is a heuristic and not validated: on the 6-panorama `sj-centro` sample it classed 23 of 44 matched buildings as `weak_views_only`, 16 % of nearby footprints had a usable view, and 34 % were matched at all. Treat those as the order of magnitude to expect from sparse street captures, not as a quality score.

## Evaluation

`evaluation/` measures building separation against annotated panoramas. Annotations are SAM-assisted prompts (boxes and points per building in eight perspective views, including four upward views for roof lines) stored in `evaluation/annotations/`; a larger SAM 2 than the pipeline's turns them into masks. Expensive model outputs are cached under `data/.eval-cache/`.

```bash
uv run python -m evaluation.run semantic --variant b5-384 --model b5 --tile-size 384
uv run python -m evaluation.run instances --variant gdino-tiny
uv run python -m evaluation.run associate --semantic b5-384 --instances gdino-tiny --tag current --overlays
uv run python -m evaluation.run score --tag current            # OSM labels
uv run python -m evaluation.run score --tag current --visual   # OSM label x visual instance
```

Scores: completeness (a building under one label), purity (a label on one building), their F-score, detection, `top_detected` (upper quarter of each facade labeled), leak (labels outside buildings), and merge/split counts.

## Verification

```bash
make test
```

This runs the Python and JavaScript test suites and a syntax check of every Python module and viewer script. Run `uv run python -m unittest discover -s tests -v` for verbose test output.

Browser tests run the real viewer against a small synthetic workspace (four footprints, three panoramas, model outputs derived from the footprints; no network data, no models, no PyTorch): layout at every screen size and theme, the keyboard review queue, reverting and splitting corrections, search and deep links, batch analysis and layers:

```bash
make e2e    # installs Playwright's Chromium on first use
```

GitHub Actions runs `make lint`, `make test` and the browser tests on every push to `main` and every pull request (`.github/workflows/ci.yml`); layout screenshots are kept as an artifact when a browser test fails.

`tools/viewer_check.py` (used by the layout test) checks the viewer in a headless browser at desktop, laptop, tablet and phone sizes, in dark and light themes: overlapping or clipped controls, elements outside the screen, console and HTTP errors, and the memory of the viewer and the browser (it stops if available memory drops below 2 GB). It also runs against any running viewer whose analyses are cached:

```bash
uv run --no-project --with playwright==1.62.0 python tools/viewer_check.py --dataset demo-san-jose --output /tmp/viewer-check
```

## Limits

- Cross-city accuracy has not passed a labeled benchmark.
- Building parts still need a final parent-building identity policy.
- Pitch, roll, terrain, roofs, and confidence are not fully calibrated.
- SfM does not yet enforce a rigid panorama rig during optimization.
- LightGlue, Depth Anything, MoGe-2, and VGGT are candidates, not dependencies.
- The separation benchmark has 8 annotated panoramas (52 buildings, 3 cities): enough to compare variants, not to certify accuracy.
- Open-vocabulary detectors treat a block of attached facades as one building; separation inside it relies on the projected footprints and therefore on the pose.

The prioritized work, metrics, and model-adoption rules are in [roadmap.md](roadmap.md). Scaling, hardware, and structural-attribute plans are in [docs/scaling-and-attributes.md](scaling-and-attributes.md).

## License and privacy

The code in this repository is licensed under the [European Union Public Licence v. 1.2](https://github.com/JSempereH/360_city_analyser/blob/main/LICENSE) (EUPL-1.2). The license covers the code only: downloaded imagery, OpenStreetMap data and third-party model weights keep their own terms, described below.

Model weights have their own licenses: SegFormer checkpoints (NVIDIA Source Code License) are for non-commercial research and evaluation only, and the Mapillary Vistas Mask2Former checkpoint is trained on a research-licensed dataset. Grounding DINO, MM-Grounding-DINO, LLMDet, SAM 2 and Depth Anything V2 Small are Apache-2.0; EoMT is MIT. The optional Ultralytics YOLOE/YOLO-World detectors are AGPL-3.0 (commercial use needs an Ultralytics license). Check these before any commercial use.

Mapillary documents ordinary imagery under CC BY-SA 4.0, subject to current platform terms and image-specific restrictions. Preserve `ATTRIBUTION.md`, do not reverse face or plate blurring, and verify current [API documentation](https://www.mapillary.com/developer/api-documentation/), [image license](https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data), and [terms](https://www.mapillary.com/terms) before redistribution.
