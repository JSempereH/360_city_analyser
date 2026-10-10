---
title: Home
hide:
  - navigation
---

# 360 City Analyser

<div class="hero">
<img src="assets/logo.svg" alt="360 City Analyser logo" width="96" height="96">
<div>Local-first tools to download Mapillary 360 panoramas, view them, segment visible buildings, and associate facade evidence with OpenStreetMap footprints. Run it on your laptop, or send only the model stages to a GPU.</div>
</div>

![Viewer: building masks, selection in the panorama and on the mini-map, and moving between captures](viewer-demo.gif){ .demo }

## Where to start

- **[User guide](guide.md)**: install, download a street, open the viewer, run the analysis, generate reports.
- **[Remote inference](remote-inference.md)**: run the models on a GPU box, a cloud VM or Google Colab.
- **[Architecture](architecture.md)**: modules, caching, artifact contract and HTTP API.
- **[Footprint corrections](footprint-corrections.md)**: reviewed, versioned fixes to OSM footprints.
- **[Scaling and attributes](scaling-and-attributes.md)**: efficiency review, city-scale design, hardware, and structural attributes.
- **[State of the art](state-of-the-art.md)**, **[Roadmap](roadmap.md)** and **[OpenFACADES reuse plan](openfacades-reuse-plan.md)**: measured results, prioritized work and what to reuse from related research.

## Principles

- OSM is the building-footprint source and identity. Local footprint corrections are explicit, reviewed and versioned against the OSM element version; never silent.
- Original Mapillary GPS controls visible camera positions and navigation.
- Refined poses only support image-to-footprint association.
- SfM is optional depth evidence and never replaces the displayed GPS track.
- Reports are deterministic and do not run an LLM or GEM classifier.
- The system abstains when its evidence is insufficient or ambiguous.

## License

Code under the [EUPL-1.2](https://github.com/JSempereH/360_city_analyser/blob/main/LICENSE). Imagery, OpenStreetMap data and model weights keep their own terms; see the end of the [user guide](guide.md#license-and-privacy).
