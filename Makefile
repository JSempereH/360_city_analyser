.PHONY: viewer models test worker-image evaluate demo-gif

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

worker-image:
	docker build -f Dockerfile.worker -t 360-city-analyser-worker .

evaluate:
	uv run python -m evaluation.run associate --semantic b5-384 --instances gdino-tiny --tag current --overlays
	uv run python -m evaluation.run score --tag current

# Needs the viewer running with the analysis settings used for data/demo-san-jose.
demo-gif:
	uv run --no-project --with playwright==1.62.0 python tools/record_demo.py --output docs/viewer-demo.gif
