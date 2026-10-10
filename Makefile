.PHONY: viewer models test e2e lint worker-image evaluate demo-gif docs docs-build

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
