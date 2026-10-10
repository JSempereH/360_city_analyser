[![360 City Analyser](docs/assets/banner.svg)](https://jsempereh.github.io/360_city_analyser/)

[![CI](https://github.com/JSempereH/360_city_analyser/actions/workflows/ci.yml/badge.svg)](https://github.com/JSempereH/360_city_analyser/actions/workflows/ci.yml)
[![License: EUPL-1.2](https://img.shields.io/badge/license-EUPL--1.2-ffd500?labelColor=0b0b0b)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-ffd500?labelColor=0b0b0b)](pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-web-ffd500?labelColor=0b0b0b)](https://jsempereh.github.io/360_city_analyser/)

Local-first tools to download Mapillary 360 panoramas, view them, segment visible buildings, and associate facade evidence with OpenStreetMap footprints.

![Viewer: building masks, selection in the panorama and on the mini-map, and moving between captures](docs/viewer-demo.gif)

**Documentation:** https://jsempereh.github.io/360_city_analyser/ (sources in [docs/](docs/)).

## Quick start

```bash
uv sync --extra cpu     # or --extra cu126 on an NVIDIA GPU
uv run city-analyser-models
uv run city-analyser-viewer
```

Run the commands from the repository root: datasets (`data/`), model weights (`models/`) and `.env` are read from the working directory.

| Topic | Page |
|---|---|
| Install, download, viewer, analysis, reports, evaluation, limits, licenses | [docs/guide.md](docs/guide.md) |
| Run the models on a GPU box, cloud VM or Google Colab | [docs/remote-inference.md](docs/remote-inference.md) |
| Modules, caching, artifact contract, HTTP API | [docs/architecture.md](docs/architecture.md) |
| Footprint corrections | [docs/footprint-corrections.md](docs/footprint-corrections.md) |
| Scaling, hardware, structural attributes | [docs/scaling-and-attributes.md](docs/scaling-and-attributes.md) |
| Measured results and candidate models | [docs/state-of-the-art.md](docs/state-of-the-art.md) |
| Prioritized work | [docs/roadmap.md](docs/roadmap.md) |

Preview the site locally with `make docs` (http://127.0.0.1:8000/, live reload; `make docs-build` runs the strict build used in CI).

## Principles

- OSM is the building-footprint source and identity. Local footprint corrections are explicit, reviewed and versioned against the OSM element version; never silent.
- Original Mapillary GPS controls visible camera positions and navigation.
- Refined poses only support image-to-footprint association.
- SfM is optional depth evidence and never replaces the displayed GPS track.
- Reports are deterministic and do not run an LLM or GEM classifier.
- The system abstains when its evidence is insufficient or ambiguous.

## License

Code under the [European Union Public Licence v. 1.2](LICENSE) (EUPL-1.2). Imagery, OpenStreetMap data and model weights keep their own terms; read [License and privacy](docs/guide.md#license-and-privacy) before redistributing or using anything commercially.
