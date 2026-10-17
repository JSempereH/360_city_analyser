# Reuse plan: OpenFACADES

[OpenFACADES](https://arxiv.org/abs/2504.02866) (Liang, Xie, Zhao, Stouffs, Biljecki; ISPRS J. Photogramm. Remote Sens. 230, 2025; code and data at <https://github.com/seshing/OpenFACADES>) turns Mapillary panoramas and OSM footprints into a building-attribute dataset and fine-tunes a vision-language model (VLM) on it. This page records what the paper measured, what this project reused, and what is still open. Numbers in the tables below are the paper's own validation results and have not been reproduced here. See [Status](#status) for what exists now.

## Status

| Item | State | Evidence |
|---|---|---|
| 0. Repository review | Done (read on GitHub and Hugging Face; not cloned or run) | below |
| 1. Quality gate | Implemented: `image_quality`, ranking v2, `--skip-low-quality` | Clean local panoramas scored 405 to 1087 sharpness; a Gaussian blur of radius 1 scored 178 and radius 2 scored 33, so the default threshold is 100. 24 panoramas from 4 cities is a small basis |
| 2. Robustness benchmark | Implemented: `evaluation/corruptions.py`, `--corruption`, `robustness` command | Unit-tested. **Not run on the annotated benchmark**: it needs the instance stage on 8 panoramas per corruption and severity (15 settings), which is hours on this CPU. Run it on a GPU worker |
| 3. Building index | Implemented: `city-analyser-index`, `--only-buildings`, `--max-views-per-building` | On the analyzed local panoramas (57 matched buildings) the index contained all 57 once GPS repair was applied, and contained 0 of 16 without it for a camera recorded inside a footprint. Requiring a 10 degree angle of view kept only 19 of 57, so the selection default has no floor. The index also lists many buildings the analysis does not match, as intended |
| 4. Crop export | Implemented: `city-analyser-crops` | Run on a real analysis of `sj-bench` (18 buildings, 33 crops). Neighboring attached buildings give near-identical crops, which is why `--mask-others` exists. Occluders (vegetation, vehicles) are not masked |
| 5. Attribute provider | Interface, OSM-tag prior, vision-language client (server or in-process Transformers), GEM v3 structure (`gem_material`, `gem_llrs`), fusion with abstention, scoring, CLI and the `city-analyser-survey` agent implemented; tested with fake answers and run end to end with a tiny random-weight Qwen3-VL checkpoint | **No real model has been run and nothing has been fine-tuned or evaluated on a held-out city.** There was no GPU here; the Colab notebook is the first place to run one. The prior scored against OSM tags agrees with itself by construction (on `sj-bench`, 14 of 18 buildings had a usable `building` tag and 10 had `building:levels`), so it only shows how sparse the tags are |

### Repository review (step 0)

- **Code:** `seshing/OpenFACADES`, MIT license, `src/openfacades/` with `detection`, `footprint`, `streetview`, `visualization` and `vlm` packages, `run.py` (Mapillary) and `run_gsv.py` (Google Street View), and a `train/` guide for InternVL fine-tuning.
- **Dataset:** `seshing/openfacades-dataset` on Hugging Face, CC BY 4.0, 5.2 GB, JSONL training files and individual building images. The dataset viewer fails and the card does not list columns, so the schema is unknown, and it does not say whether Mapillary image IDs are included.
- **Weights:** a Hugging Face collection of fine-tuned InternVL models is linked, but their license was not checked.
- **Licensing caveat:** the images derive from Mapillary (CC BY-SA 4.0) and the labels from OpenStreetMap (ODbL), so the dataset's CC BY 4.0 does not by itself settle what may be redistributed or used commercially. Treat it as research material until that is clarified.
- **Not checked:** the exact blur-score definition in their code (their table says at most 30, which reads inverted for a Laplacian variance), and any overlap of their Mapillary images with our cities.

## What the paper does

1. **Candidate selection.** Mapillary metadata filters (`is_pano`, duplicate coordinates, `quality_score`) and OSM filters (area above 20 m², not underground). An isovist (line of sight from the camera to sampled perimeter points, blocked by other footprints) gives each building an angle of view (AOV) and azimuth range.
2. **Building image.** The azimuth range crops the panorama, Grounding DINO-B boxes the building, and the box is reprojected to a pinhole view whose focal length follows the AOV.
3. **Image filters** (their Table 2): AOV between 10 and 120 degrees, at most 5 images per building, at least one building within 30 m, file size at least 20 KB, building share at least 0.2, wall share at most 0.3, vegetation share at most 0.75, outdoor scene, Laplacian blur score at most 30 (as their table states it), brightness between 20 and 200.
4. **Dataset.** 31,180 labeled images from 7 cities. Labels are OSM `building`, `building:material`, `start_date` and `building:levels`; multi-attribute answers and captions come from GPT-4o. Applied at scale it annotates 1.2 million images (about 503,000 buildings). Across all footprints 31.1 % of buildings end up with an image (14 % in Helsinki to 62 % in Amsterdam); in the 2.5 km x 2.5 km city centers it is 49.4 %.
5. **Model.** Full fine-tuning of InternVL3-2B (3 epochs, learning rate 8e-6), one structured prompt for all attributes.

Reported results (validation set, fine-tuned InternVL3-2B versus zero-shot GPT-4o):

| Attribute | InternVL3-2B | GPT-4o zero-shot | Best single-attribute CNN/ViT |
|---|---|---|---|
| Type, accuracy | 61.3 % | 57.7 % | 56.1 % (Swin-B) |
| Material, accuracy | 68.1 % | 65.4 % | 69.2 % (Swin-B) |
| Floors, R² | 0.789 | 0.721 | 0.777 (ResNet101) |
| Age, R² | 0.710 | 0.645 | 0.738 (ResNet101) |

Other findings worth keeping:

- On an unseen city (Brussels) the VLM reaches 63.8 % type accuracy against at most 32.5 % for the CNN/ViT baselines. Per-class results show hotel and public confused with office, concrete with plaster, pre-1900 ages and buildings above 15 floors predicted poorly.
- Full fine-tuning beat LoRA (overall R² 0.707 versus 0.670 at the same learning rate and 3 epochs). 5 to 10 % of the data already gives most of the gain, and 1B to 4B models end up close.
- Adding GPT-generated labels slightly lowered attribute accuracy; OSM labels helped. OSM labels are noisy and the authors say so.
- Synthetic corruptions: motion blur hurts most (floors error 0.21 to 0.48, age error 0.29 to 1.45 at severe), occlusion and lighting much less.
- Stated limitations: unverified GPT captions, noisy and misaligned crowdsourced data, no manual image selection, so selection bias is unaddressed.

## What this project already has

| Need | Here today |
|---|---|
| Panorama to perspective reprojection | `panorama_geometry.perspective_tile` (equivalent to their equations 2 to 9) |
| Image-to-footprint association | Pose refinement, per-pixel instance masks (Grounding DINO tiny + SAM 2), occlusion, abstention. The paper uses GPS and heading only |
| Ranked evidence per building | `building_inspection.ranking` and `screenshots` (up to 4 crops per report) |
| Wall visibility from a camera position | `facade_geometry` (vectorized, occlusion-aware) |
| Attribute stage | At the time of the review, none; now the attribute providers and the survey agent (step 5) |
| Image quality gate | None. Only panorama validity checks at download |
| Robustness benchmark | None. `evaluation/` measures separation on clean images |

## Reuse candidates, in priority order

Each item lists the change, where it lands, and how to know it worked.

### 1. Image quality gate (small, high value)

Add `image_quality.py` with blur (Laplacian variance), brightness, night detection and file-size checks, computed once per panorama and stored in the analysis metadata as facts, not as a hidden filter. `ranking` then penalizes degraded views and the viewer shows why. Start from the paper's thresholds, but re-derive the blur one on our own panoramas: the paper's value is for 1200x720 perspective crops, not equirectangular panoramas, so apply it to the projected crops.

- Done when: the gate runs in `city-analyser-analyze`, its numbers are in the artifact contract, and the 8 benchmark panoramas show whether low-quality views correlate with lower separation F.
- Risk: dense greenery gets filtered out (the authors warn about this). Keep thresholds configurable and report excluded counts.

### 2. Robustness benchmark (small, no product risk)

Extend `evaluation/` with their corruption suite: occlusion at 10, 25 and 50 %, motion blur, Gaussian noise and brightness at three severities, scored as relative corruption error against the clean run. It shows which degradations break pose refinement and instance separation, and justifies the thresholds of item 1.

- Done when: `python -m evaluation.run score` can report F under each corruption and the report states the worst case.
- Constraint: the instance stage costs about 90 s per panorama on CPU. Use the stage cache and run on a GPU worker ([remote inference](remote-inference.md)).

### 3. Building-centric candidate index (medium, cuts GPU cost)

The paper works building first: for each footprint it finds which cameras see it (AOV 10 to 120 degrees, at most 5 images), then processes only those. We work panorama first. Add a cheap index that, without running models, lists for each building the panoramas whose projected visible wall exceeds a threshold. Use it to choose which panoramas to send to the GPU when only some buildings matter, and to cap evidence per building.

- Where: `facade_geometry.visible_facades` plus the cached OSM cells; output a Parquet or JSON index next to the manifest.
- Done when: on a dataset, the indexed subset gives the same buildings as a full run for at least 95 % of buildings while perceiving fewer panoramas (measure and record the ratio).

### 4. Building crop export (medium, prerequisite for attributes)

Add `building_inspection/crops.py`: for a building, render a perspective crop centered on its instance mask, with the field of view fitted to the mask bounding box (their AOV-driven focal length, but driven by our masks instead of a DINO box), plus a sidecar with image id, OSM id, yaw, pitch, FOV, observed fraction, quality metrics and artifact hashes. Optionally mask occluders so a vegetation-covered base is not mistaken for facade.

- Done when: crops are deterministic (same inputs, same bytes) and each carries the provenance fields the GEM provider contract requires.
- Reuses `screenshots.render_screenshots` rather than a second projection implementation.

### 5. Attribute provider baseline (large, follows items 1 to 4)

Implement the GEM provider interface from [architecture](architecture.md) and evaluate in this order, as the [scaling plan](scaling-and-attributes.md) already requires:

1. cadastral or OSM prior without images;
2. the paper's released model or dataset as an external baseline on our crops, if its license allows it (step 0);
3. our own fine-tune on OSM-derived labels.

Adjust the earlier plan with the paper's evidence: prefer full fine-tuning of a small (1B to 2B) open VLM with a low learning rate over LoRA when labels are scarce, train on OSM labels first, and add the top-2 prediction trick (`alternate_building_type`) to expose ambiguity. Fuse views by pooling distributions and abstaining when evidence is weak, not by taking the most frequent label as the paper does. Evaluate on a held-out city, as the paper's Brussels test did, and report per-class results and calibration, not only accuracy.

- Done when: the provider beats the prior on a held-out city and the report separates image-to-footprint association from structural attributes, as the architecture requires.

## Not adopted

- GPS-only geometric association. It is the weak point the paper itself lists (misaligned image coordinates); our pose refinement and instance masks exist for this.
- GPT-4o captions as training labels. Closed, unverified, and the ablation shows they reduce attribute accuracy.
- Most-frequent-label aggregation without abstention.
- A single dataset-wide Grounding DINO-B crop step. Our instance stage already separates attached facades.

## What remains

1. Run the robustness benchmark on a GPU worker and use it to confirm or move the sharpness threshold.
2. Download the released dataset, read its schema and decide whether it may train or evaluate here; run its released model on our crops as an external baseline.
3. Run a real vision-language model behind `city-analyser-attributes --provider vlm` on a GPU and score it against the tag prior and the majority class on a held-out city.
4. Decide the labeled data for our own cities (see the scaling plan) so the evaluation does not rest only on noisy OSM tags.
5. Make crop export scale: it reads every analysis once per building, which is fine for hundreds of buildings and slow for city scale.

## Open questions

- Which license covers the fine-tuned weights? The paper itself is CC BY-NC-ND; OSM labels are ODbL and Mapillary imagery is CC BY-SA, which affect redistribution of derived datasets.
- Does the authors' blur threshold apply to our resolution, or must it be recalibrated on panorama crops?
- Is a 2B VLM affordable on the target GPU, and which of Qwen3-VL or InternVL3 should we start from?
- Which labels do we have for our own cities (Valencia and Madrid via the Catastro, see the scaling plan) to avoid training only on noisy OSM tags?
