.PHONY: viewer models test e2e lint worker-image evaluate demo-gif docs docs-build presentation-data presentation presentation-offline

# Override on CPU-only machines: make viewer TORCH=cpu
TORCH ?= cu126

viewer:
	uv sync --locked --extra $(TORCH)
	uv run city-analyser-viewer

models:
	uv run city-analyser-models

test:
	uv run python -m unittest discover -s tests
	node --test tests/*.mjs
	uv run python -m compileall -q src evaluation tools
	node --input-type=module --check < src/city_analyser/viewer/app.js
	node --input-type=module --check < src/city_analyser/viewer/map-window.js

# Browser tests on a synthetic workspace; Playwright and Chromium are test-only tools.
PLAYWRIGHT ?= playwright==1.62.0

e2e:
	uv run --with $(PLAYWRIGHT) playwright install chromium
	uv run --with $(PLAYWRIGHT) python -m unittest discover -s tests/e2e -v

lint:
	uvx ruff check --select F,E9,RUF059 .

worker-image:
	docker build -f Dockerfile.worker -t 360-city-analyser-worker .

evaluate:
	uv run python -m evaluation.run associate --semantic b5-384 --instances gdino-tiny --tag current --overlays
	uv run python -m evaluation.run score --tag current

# Needs the viewer running with the analysis settings used for data/demo-san-jose.
demo-gif:
	uv run --no-project --with playwright==1.62.0 python tools/record_demo.py --output docs/viewer-demo.gif

# Same pinned Material version as the Docs workflow.
MKDOCS ?= uvx --with mkdocs-material==9.6.14 mkdocs
DOCS_ADDR ?= 127.0.0.1:8000

# Live-reloading preview at http://$(DOCS_ADDR)/
docs:
	$(MKDOCS) serve --dev-addr $(DOCS_ADDR)

# Static build into site/; broken links fail it, as in CI.
docs-build:
	$(MKDOCS) build --strict

# Presentation: analyzed copies of valencia-centro and madrid-centro with the reviewed masks
# and answers in demo/*.curation.json, matched to the cadastre (the presentation cross-checks
# each record with it), then the viewer at the presentation URL.
PRESENTATION_ENV = BUILDING_ANALYSIS_MODEL=balanced BUILDING_ANALYSIS_TILE_SIZE=384

presentation-data:
	for city in valencia-centro madrid-centro madrid-salesas; do \
	  $(PRESENTATION_ENV) uv run city-analyser-analyze $$city && \
	  rm -rf data/$$city-demo && mkdir -p data/$$city-demo && \
	  cp demo/$$city-demo.curation.json data/$$city-demo/curation.json && \
	  $(PRESENTATION_ENV) uv run city-analyser-demo $$city && \
	  uv run city-analyser-groundtruth catastro $$city-demo || exit 1; \
	done

presentation:
	@echo "Open http://127.0.0.1:8765/viewer/?dataset=madrid-salesas-demo&present=1&live=0&speed=2 (press F for full screen)"
	@echo "Longer tour: http://127.0.0.1:8765/viewer/?dataset=valencia-centro-demo&present=1&then=madrid-centro-demo"
	$(PRESENTATION_ENV) uv run city-analyser-viewer

# The same, with no network at all: map tiles, footprints and libraries come from the local caches.
presentation-offline:
	@echo "Open http://127.0.0.1:8765/viewer/?dataset=madrid-salesas-demo&present=1&live=0&speed=2 (press F for full screen)"
	CITY_ANALYSER_OFFLINE=1 $(PRESENTATION_ENV) uv run city-analyser-viewer
