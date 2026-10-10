# Scaling, hardware and building attributes (October 2026)

This review covers four questions:

- how to make model execution and caching cheaper;
- what architecture and hardware a city-scale run needs;
- which models to use for each stage, including a vision-language model (VLM) for structural attributes;
- how to validate attribute predictions against a European census.

Measured numbers come from this repository on a 4-core laptop CPU (no GPU). GPU and throughput figures are estimates and are labeled as such. Model and dataset claims come from the cited sources and have not been reproduced here.

## 1. Efficiency review

The critical path is `analyze_panorama` in `building_analysis.py`. The review found these issues:

| # | Issue | Status |
|---|---|---|
| 1 | One cache key covered both model outputs and OSM association (`ANALYSIS_VERSION`), so any association or threshold change re-ran every model | **Fixed**: content-addressed stage cache (`stage_cache.py`) |
| 2 | Model outputs were cached only by `evaluation/cache.py`, keyed by variant name | Kept on purpose: named variants suit experiments. Production uses the stage cache |
| 3 | The panorama was decoded and LANCZOS-resized twice (semantic and instance stages) | **Fixed**: decoded at most once, not at all on a full cache hit |
| 4 | Perspective views were built on CPU with nearest-neighbor sampling, then resized and normalized on CPU by the Hugging Face processor | **Fixed**: batched bilinear sampling on the inference device; normalization on device for the semantic models |
| 5 | Projection grids were recomputed for every panorama (18 views plus instance coverage) | **Fixed**: memoized per geometry |
| 6 | Weights load in fp32; only autocast runs in fp16 | Not changed: autocast already computes in fp16 on CUDA, the saving is about 0.6 GB of weights, and bf16 would cost SAM box precision. Revisit when GPU memory is measured |
| 7 | The SAM 2 image encoder ran once per view | **Fixed**: views are encoded in batches; only the prompt decoder runs per view |
| 8 | Mask merging compared full-resolution mask pairs, O(N^2 x W x H) | **Fixed**: comparison restricted to the bounding box of each view pair's shared region (identical result, tested) |
| 9 | One panorama at a time, no overlap between CPU and model work | **Partly fixed**: `city-analyser-analyze` overlaps model stages with association and prefetches OSM, locally and with a remote worker (which now runs the model stages only, so its GPU lock never covers CPU work). Cross-panorama GPU batching is not implemented |
| 10 | OSM comes from Overpass per ~110 m cell: fine locally, rate-limited and not reproducible at scale | Open: use a frozen Geofabrik extract (section 2) |
| 11 | The downloader fetches the largest derivative, but analysis caps width at 2048 px | Open: download 2048 px derivatives for batch runs |

The project's own benchmark (`docs/state-of-the-art.md`) also shows that instances now decide building extent: Mask2Former brings no gain over SegFormer-B5, and the semantic stage mainly supplies vegetation and a fallback. That stage is a candidate for simplification (section 4).

### Measured effect

On the 3 `demo-san-jose` panoramas, CPU only, `BUILDING_ANALYSIS_MODEL=balanced`, `BUILDING_ANALYSIS_TILE_SIZE=384`:

| Case | Before | After |
|---|---|---|
| Full analysis, models loaded | 117-131 s per panorama | 89-98 s per panorama |
| Re-analysis after an association or `ANALYSIS_VERSION` change | same as a full analysis | **0.5-0.6 s per panorama** |

Run-to-run variation on this laptop is about ±15 % (a warm baseline run was slower than a cold one), and the detector code did not change. **CPU model speed is therefore unchanged within noise.** The measured gain is the stage cache. The device-side view sampling and batched SAM encoding target GPUs and have not been measured.

Quality on the separation benchmark (8 annotated panoramas, 52 buildings) is unchanged within noise:

| Metric | Before | After |
|---|---|---|
| F (OSM) | 0.8132 | 0.8125 |
| F (visual) | 0.8383 | 0.8395 |
| Detected | 0.9696 | 0.9702 |
| Top quarter detected | 0.9714 | 0.9756 |
| Leak | 0.0814 | 0.0825 |
| Merges / splits | 9 / 3 | 9 / 3 |

## 2. Architecture at scale

Principle: **idempotent stages whose artifacts are keyed by the hash of their inputs**. Each stage stores its output under `hash(inputs + stage settings + model revision)`, so only what changed is recomputed. The stage cache is the first piece of this.

```text
ingest ──> perception (GPU) ──> association (CPU) ──> per-building fusion ──> attributes (GPU/VLM) ──> publish
Mapillary/   semantic+instances    pose + OSM projection     multi-view per OSM ID     classifier/VLM, GEM         PostGIS + viewer
Panoramax    [+depth]              per panorama              per spatial tile          with abstention             + reports
```

| Component | One city (about 10^5 panoramas) | One country (10^6 or more) |
|---|---|---|
| Orchestration | Postgres job table (`SELECT ... FOR UPDATE SKIP LOCKED`) and Python workers | Ray Data for batched GPU inference, or an autoscaled queue |
| Partitioning | Spatial tiles (H3 resolution 7-8 or OSM cells), so OSM loads once per tile and fusion stays local | Same, plus coverage-based priority |
| OSM | **Frozen Geofabrik extract (PBF) loaded into GeoParquet or PostGIS**, no Overpass | Same, versioned by date |
| Images and artifacts | Local NVMe disk or MinIO | S3 or GCS (object storage) |
| Results | PostGIS (buildings, assignments, attributes) plus Parquet for analysis | Same |
| GPU | One worker, batches of K panoramas, CPU -> GPU -> CPU pipeline | N stateless workers; vLLM for the VLM |
| Observability | Stage timings, peak memory, model revision, IDs (roadmap phase 5) | Same, plus queue metrics |

Indicative volume: with one capture every 10 m, a city with 1,000 km of streets has about **100,000 panoramas**. At about 1 MB per panorama (2048 px JPEG) plus about 0.2 MB of masks, that is **about 120 GB**.

## 3. Hardware requirements

The CPU figures are measured on the 4-core laptop (about 1.5-2 minutes per panorama for the full pipeline, most of it in the instance stage). **The GPU figures are estimates**; `city-analyser-analyze --report` records per-panorama timings and peak memory so they can be measured.

| Profile | CPU / RAM | GPU | Expected throughput |
|---|---|---|---|
| Local development (viewer and tests) | 8 or more cores, **32 GB** | 8-16 GB (RTX 4060 Ti 16 GB or 4070) | A few seconds per panorama |
| Perception worker | 16 vCPU, **64 GB** (decoding and association run on CPU), NVMe | **24 GB** (L4, A10G or RTX 4090) | Estimated 1-3 s per panorama. 100k panoramas take about 1-4 days on one GPU (about 140 days on the laptop CPU) |
| 8B VLM (Qwen3-VL-8B) | 32 GB | 24 GB in bf16 (about 17 GB of weights); 8-12 GB quantized to 4 bits | Hundreds of crops per minute with vLLM |
| 32B VLM | 64 GB | 48 GB (L40S) quantized, or 80 GB (H100) in bf16 | Quality reference and teacher |

The current weights (SegFormer-B5, Grounding DINO tiny and SAM 2.1 small) are about 0.6 GB in fp16, so an 8 GB GPU is enough for inference. With 24 GB, several panoramas can share a batch and a small VLM can stay loaded alongside.

## 4. Recommended models per stage

License column: **P** means permissive (Apache or MIT); **NC** means non-commercial. The intended use (research or commercial) is not decided yet, so permissive models are the default and non-commercial ones are options.

| Stage | Now | Recommendation | License |
|---|---|---|---|
| Semantic (vegetation and fallback) | SegFormer-B5 (NC) | Mask2Former is already measured to bring no gain. Try **removing the stage** and detecting vegetation with the same Grounded-SAM (prompt "tree."), or use EoMT (MIT) | SegFormer is NC: replace it for commercial use |
| Instances | Grounding DINO tiny + SAM 2.1 small (P) | Keep it: it is the best measured option (F 0.813). Test **SAM 3** on GPU as a single model that segments "building" and "tree" directly | SAM 3 has its own license: check it |
| Pose against OSM | Silhouette search | OrienterNet or OSMLoc on GPU (next roadmap step) | OrienterNet is NC |
| Depth | Optional SfM | Depth Anything V2 Small to mark building edges and occlusions | P |
| **Attributes (GEM)** | **Not implemented** | See section 5: a supervised classifier (SigLIP 2 or DINOv3 with multi-view pooling) as the baseline, and a VLM (Qwen3-VL) to explain and act as a teacher | SigLIP 2 and Qwen3-VL: P; DINOv3 has its own license |

## 5. Structural attributes with a VLM

**Not integrated, on purpose.** The user guide states that reports "do not run an LLM or GEM classifier", and the roadmap lists LLM-generated structural attributes as out of scope. [the architecture page](architecture.md) already defines how one would be added: a **separate GEM provider** that consumes the ranked evidence and returns per-attribute distributions, evidence references, the taxonomy version and explicit abstentions.

What a facade shows, from most to least reliable:

1. Height and number of storeys.
2. Occupancy.
3. Construction period.
4. Material (masonry or concrete; render often hides it).
5. Lateral load-resisting system (frame or walls).
6. Irregularities such as an open ground floor.

Published results for supervised models trained in a single city:

- About 83-88 % accuracy for the structural system, about 78 % for the number of storeys and about 69 % for the construction period.
- The Alvalade dataset (Lisbon), with 10 attributes, reaches **57 % macro accuracy**.
- General-purpose VLM APIs used zero-shot score below a fine-tuned classifier. They are useful to explain a decision and as weak labelers.

Recommended design, in phases:

1. **Prior without images:** cadastral year and storeys plus location give a GEM distribution. **This is the baseline to beat**; in Europe the construction year already predicts much of the typology.
2. **Visual classifier:** the best crops per building (already ranked by `building_inspection.ranking`), a backbone (SigLIP 2 or DINOv3) with one head per attribute and multi-view pooling, temperature-calibrated, with abstention.
3. **VLM:** Qwen3-VL-8B (Apache-2.0) fine-tuned with LoRA on the same labels. It receives the crops and the prior and returns JSON with a distribution, cited evidence and "not visible". A closed VLM serves only as a research reference.
4. Bayesian combination of the prior and the visual evidence, emitted in the provider format defined in the architecture page.

## 6. Benchmarking against a European census

| City / source | Level | What it validates | Notes |
|---|---|---|---|
| **Lisbon**: INE Censos 2021 (BGRI) and the Alvalade dataset | BGRI: statistical subsection (aggregated). Alvalade: **per building** (4,007 buildings, 10 attributes) | **Structure type** (concrete or masonry; check that the variables are still published for 2021), period, storeys | **Recommended pilot**: real structural labels and active seismic-risk research. Request the Alvalade data from its authors |
| **Spain** (Valencia and Madrid are already downloaded): Catastro INSPIRE | Per building, open | Construction year, storeys above ground, use, dwellings. **Not** the structural system | Ideal for height, period, use and the prior. Lorca (2011) for damage |
| **Italy**: Da.D.O. (AeDES forms) | Per building, about 320,000 | **Structural typology** and material | Free for research. Caveat: post-earthquake surveys, and buildings may have been rebuilt since |
| **France**: BDNB and Panoramax | Per building, Licence Ouverte 2.0 | Wall material (check the field), year, storeys | Best open option at scale; Panoramax provides open 360 imagery |
| ESRM20 (European exposure model) | Municipality | Distribution of GEM classes | Aggregated comparison |

Protocol:

1. **Matching:** OSM footprint to census or cadastre building by overlap (IoU >= 0.5, one to one). Ambiguous cases are set aside. Filter by the gap between image date and census date.
2. **Spatial splits:** hold out whole neighborhoods or a whole city, so neighboring facades do not leak between training and test.
3. **Per-building metrics:** macro F1, balanced accuracy, confusion matrix, ECE and Brier score, and risk-coverage with abstention.
4. **Aggregated metrics** (BGRI, ESRM20): distance between the predicted and census distributions per subsection (Jensen-Shannon or total variation) and per-class count error.
5. **Mandatory baselines:** majority class, cadastral prior and visual classifier. A VLM is adopted only if it beats the prior and the classifier (roadmap model adoption rule).
6. **Pilot:** one district with 2,000-5,000 buildings (Alvalade, or a Valencia neighborhood for cadastral attributes).

## 7. Next steps, in order

1. Attribute benchmark with Lisbon (BGRI and Alvalade) and Catastro (Valencia), with the prior and the metrics above.
2. Multi-view visual classifier.
3. VLM prototype behind the GEM provider interface.
4. Frozen OSM snapshot in GeoParquet.
5. Postgres job queue and H3 partitions.
6. Measure GPU throughput and memory with `city-analyser-analyze --report` and replace the estimates in section 3.

## Sources

- [BDNB on data.gouv.fr](https://www.data.gouv.fr/datasets/base-de-donnees-nationale-des-batiments) · [CSTB BDNB](https://www.cstb.fr/bases-donnees/base-donnees-nationale-batiments)
- [INE Censos 2021](https://censos.ine.pt/xportal/xmain?xpgid=censos21_produtos&xpid=CENSOS21&xlang=pt) · [BGRI 2021 geopackage](https://www.cascais.pt/sites/default/files/anexos/gerais/new/13bgri_geopackage_censos_2021.pdf)
- [Deep Learning for Building Attribute Classification from Street-View Images for Seismic Exposure Modeling (Alvalade)](https://www.mdpi.com/2076-3417/16/2/875) · [GEM Foundation: AI and Street View for exposure models](https://www.globalquakemodel.org/GEMNews/silva-earthquake-spectra-webinar-exposure-modelling-ai)
- [Automating building typology identification (Earthquake Spectra, 2025)](https://onlinelibrary.wiley.com/doi/10.1177/87552930251327435) · [Street-view building structural type recognition (YOLO11)](https://www.tandfonline.com/doi/full/10.1080/17538947.2025.2586899)
- [Observed damage database of past Italian earthquakes (Da.D.O.)](https://bgo.ogs.it/sites/default/files/pdf/bgta0254_Dolce.pdf)
- [Facade material mapping with VLMs (Scientific Reports, 2026)](https://www.nature.com/articles/s41598-026-51028-6)
- [Best VLMs 2026](https://mixpeek.com/curated-lists/best-vision-language-models) · [InternVL3.5](https://arxiv.org/pdf/2508.18265)
