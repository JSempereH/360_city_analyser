# Techniques and state of the art (September 2026)

This review maps each pipeline stage to current alternatives and records what was measured in this repository. "Measured" numbers come from `evaluation/` (8 annotated panoramas, 52 buildings, San José / Madrid / Valencia) or from CPU benchmarks on a 4-core laptop; everything else is from the cited papers and model cards and has not been reproduced here.

## 1. Semantic segmentation (building / vegetation evidence)

| Option | Status here | Notes |
|---|---|---|
| SegFormer B0-B5, Cityscapes ([NVlabs](https://github.com/NVlabs/SegFormer)) | Default (`b0` CPU, B5 `balanced`) | Fast (B0 ~5 s, B5 ~18 s per panorama on CPU). Cityscapes has 19 classes and bright overcast sky is often labeled *building* (Valencia). **Non-commercial license.** |
| Mask2Former Swin-L, Mapillary Vistas | `high` profile | Better street-level coverage (66 classes, trained on worldwide street imagery); less sky confusion measured (upper-quarter "building" share 0.93 to 0.68-0.74 on Valencia). ~33 s per panorama on CPU. Research-licensed training data. |
| EoMT ([CVPR 2025](https://huggingface.co/docs/transformers/main/en/model_doc/eomt)), EoMT-DINOv3 | Candidate | Plain ViT segmentation, faster than Mask2Former at similar quality; MIT. HF checkpoints are Cityscapes/ADE20K/COCO, not Vistas. |
| OneFormer, InternImage | Candidate | Strong Vistas/ADE results; heavier; no gain expected for a two-class need. |

Recommendation: keep SegFormer for speed, but budget a commercially licensed replacement (EoMT or a Mask2Former retrained on permissive data) before any commercial use. In this pipeline the semantic mask matters less now: instance masks decide building extent where they exist.

## 2. Building instances (separating buildings)

No off-the-shelf panoptic model separates buildings: in Cityscapes, Mapillary Vistas, ADE20K and COCO panoptic, *building* is a "stuff" class without instances. Current options are open-vocabulary detection plus promptable segmentation:

| Option | Status here | Notes |
|---|---|---|
| Grounding DINO tiny + SAM 2.1 small | Default | Apache-2.0. ~95 s per panorama on CPU (6 portrait views), ~2.5 GB peak RAM. Masks follow facade edges and exclude sky/trees; attached facades come out as one block. |
| MM-Grounding-DINO tiny ([OpenMMLab](https://huggingface.co/openmmlab-community)) | Measured, see below | Re-trained Grounding DINO with more grounding data; same classes in `transformers`. |
| LLMDet tiny ([CVPR 2025](https://huggingface.co/iSEE-Laboratory/llmdet_tiny)) | Measured, see below | Grounding DINO trained with LLM-generated captions; strong LVIS zero-shot. |
| SAM 3 / 3.1 ([Meta, Nov 2025 / Mar 2026](https://arxiv.org/abs/2511.16719)) | Not used | Text-prompted *concept* segmentation returns every instance of "building" directly, the closest model to this task. Gated (manual approval), custom license, ~3.4 GB; realistic only on a GPU or cloud worker. Worth a controlled test once access is granted. |
| YOLOE-11L / YOLO-World-X (Ultralytics) | Optional fast mode | Real-time open-vocabulary detectors, ~1-2 s per view on CPU; YOLOE-seg also returns masks. AGPL-3.0. |
| WeDetect (CVPR 2026) | Not used | Fast retrieval-style open-vocabulary detector. |

Measured on the separation benchmark with the same SAM 2.1 small masks and association policy (F = separation F-score with OSM labels):

| Detector | Box threshold | F | Detected | Top quarter | Leak | Splits |
|---|---|---|---|---|---|---|
| Grounding DINO tiny | 0.30 | **0.813** | **0.970** | **0.971** | 0.082 | **3** |
| MM-Grounding-DINO tiny (o365, goldg, grit, v3det) | 0.30 | 0.631 | 0.632 | 0.659 | 0.079 | 7 |
| LLMDet tiny | 0.30 | 0.661 | 0.694 | 0.712 | 0.056 | 9 |
| LLMDet tiny | 0.20 | 0.725 | 0.765 | 0.801 | 0.054 | 8 |
| YOLO-World-X | 0.05 | 0.689 | 0.744 | 0.830 | 0.104 | 9 |
| YOLOE-11L | 0.25 | 0.732 | 0.857 | 0.897 | 0.081 | 7 |
| YOLOE-11L | 0.10 | 0.790 | 0.920 | 0.943 | 0.085 | 4 |
| YOLOE-11L, own masks (no SAM) | 0.10 | 0.799 | 0.940 | 0.949 | 0.102 | 4 |

CPU time per panorama for the instance stage: Grounding DINO + SAM 2.1 ~95 s, YOLOE + SAM 2.1 ~22 s, YOLOE with its own masks ~6 s. YOLOE without SAM is therefore the practical fast mode on CPU (documented in the README as an opt-in AGPL extra); Grounding DINO + SAM remains the default for quality and licensing.

The newer detectors score better on COCO/LVIS, but for "building." in street panoramas they detect fewer whole buildings and split facades into parts (LLMDet reaches 0.94 visual purity but 15 visual splits). Grounding DINO tiny stays the default; the detector is configurable (`BUILDING_ANALYSIS_INSTANCE_DETECTOR`, `BUILDING_ANALYSIS_INSTANCE_BOX_THRESHOLD`).

[SeamSeg](https://github.com/mapillary/seamseg) (Mapillary, CVPR 2019) was also reviewed: panoptic segmentation trained on Vistas, where *building* has no instances, pinned to PyTorch 1.1 / CUDA 10.1, non-commercial weights, and superseded by the Vistas Mask2Former already used here.

## 3. Associating image evidence with OSM

The pipeline projects OSM walls from the camera pose (exact ray/segment depth, occlusion, shared-wall rejection) and names image instances with the projected footprints. The measured failure modes are pose and footprint quality, not projection maths:

- **GPS inside footprints**: 11 of 20 downloaded panoramas had their GPS fix 0.8-6.1 m inside an OSM building; from there no wall faces the camera. Moving the camera to the nearest street-connected open space (courtyards excluded) raised mean coverage from 0.41 to 0.77 and removed all zero-match panoramas.
- **Pose ambiguity test**: the original runner-up test compared the best pose with its own grid neighbor and rejected half the refinements as ambiguous. Comparing against clearly different poses (> 4.5 m or > 6.5 degrees) doubled accepted poses (5 to 11 of 20).
- **Negative result**: choosing the repaired GPS position by image fit over all street positions within 12 m made separation worse (F 0.813 to 0.764) and was 10-40x slower; silhouette IoU favors large, wrong shifts. It was removed.

State of the art for this sub-problem is learned image-to-OSM localization: [OrienterNet](https://arxiv.org/abs/2304.02009) (CVPR 2023, sub-meter on its benchmark), [OSMLoc](https://arxiv.org/abs/2411.08665) (depth-guided BEV, 2025/26), and 2026 work such as coarse-to-fine OSM re-localization and AutoCompass. They match a bird's-eye feature map to rasterized OSM and would replace the silhouette search. Constraints: OrienterNet weights are CC BY-NC; they are trained on perspective images (a panorama must be split); a GPU is advisable. This is the most promising upgrade for attached-facade separation, because column splits inside a detected block are only as good as the pose.

## 4. Depth

Sparse PyCOLMAP depth is optional and gated. Monocular depth for panoramas has matured: Depth Anything V2 (Apache-2.0 for Small), [PanDA](https://openaccess.thecvf.com/content/CVPR2025/papers/Cao_PanDA_Towards_Panoramic_Depth_Anything_with_Unlabeled_Panoramas_and_Mobius_CVPR_2025_paper.pdf) (CVPR 2025), [Depth Any Camera](https://github.com/yuliangguo/depth_any_camera) (CVPR 2025, metric, any camera), DA² and [Depth Any Panoramas](https://arxiv.org/abs/2512.16913) (2025). The useful signal here is *relative* depth discontinuities: a depth step inside a detected block is strong evidence of an occluding building edge, and depth ordering could check the OSM z-buffer. Not integrated yet; must be validated as an occlusion signal before it may influence assignments.

## 5. Building attributes

[OpenFACADES](https://arxiv.org/abs/2504.02866) (2025) combines Mapillary, OSM isovists and fine-tuned vision-language models to predict building type, material, age and floors, with open code and 1.2 M images. It is the closest published system to where this project's roadmap goes after association, and a good reference for the GEM-attribute stage the roadmap keeps out of scope for now.

## 6. What was measured

Separation on the annotated benchmark (SegFormer-B5 semantic at 384 px):

| Variant | F (OSM) | F (visual) | Top quarter labeled | Detected | Leak | Merges | Splits |
|---|---|---|---|---|---|---|---|
| Projection, original GPS | 0.653 | - | - | 0.776 | 0.111 | 11 | 9 |
| Projection + GPS repair | 0.759 | - | 0.949 | 0.924 | 0.127 | 12 | 11 |
| + instances, whole-instance naming only | 0.767 | 0.793 | 0.817 | 0.856 | 0.092 | 9 | 5 |
| + column split, upward continuation, sliver merge, pose fixes (current) | **0.813** | **0.838** | **0.971** | **0.970** | **0.082** | 9 | 3 |

With Mask2Former-Vistas (`high`) instead of SegFormer-B5 for the semantic stage the current pipeline scores F 0.792 (visual 0.816, detected 0.923, top quarter 0.933), and 0.737 projection-only: no gain over B5 once instances decide building extent, at about twice the CPU time.

"Visual" scores OSM label x visual instance, i.e. how well the image separates buildings even where OSM maps a block as one footprint.

## 7. Recommended next steps

1. Replace silhouette pose search with learned OSM localization (OrienterNet/OSMLoc class) on the GPU worker, evaluated on this benchmark.
2. Test SAM 3 on the GPU worker as a drop-in instance stage.
3. Add panoramic relative depth (Depth Anything V2 Small / DAC) as a seam and occlusion cue inside merged blocks.
4. Grow the benchmark (more cities, two annotators) before tuning thresholds further; with 8 panoramas, differences below ~0.02 F are noise.
5. Move to commercially licensed semantic weights if the project is to be used commercially.
