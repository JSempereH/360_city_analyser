# Remote inference

Only the model stages (semantic segmentation and building instances) run on the worker. It receives the panorama JPEG and returns the semantic confidences and the instance map. Pose refinement, OpenStreetMap association, footprint corrections, SfM depth and artifact publication stay on your machine, so the worker needs no internet access to OSM and local footprint corrections apply to remote analyses.

Start a worker with an explicit model. `--preload` loads every model before the first request, and the token is read from the environment so it stays out of process listings:

```bash
export BUILDING_ANALYSIS_REMOTE_TOKEN='use-a-long-random-secret'
BUILDING_ANALYSIS_MODEL=high uv run --extra cu126 city-analyser-worker --host 0.0.0.0 --port 8766 --preload
```

Or as a container on any NVIDIA host (local workstation or cloud GPU VM with the NVIDIA container toolkit):

```bash
docker build -f Dockerfile.worker -t 360-city-analyser-worker .
docker run --gpus all -p 8766:8766 -v "$PWD/models:/models" -v "$PWD/cache:/cache" \
  -e BUILDING_ANALYSIS_REMOTE_TOKEN -e BUILDING_ANALYSIS_MODEL=high 360-city-analyser-worker
```

`GET /api/health` reports the device and loaded models. The worker keeps its model outputs in `/cache/stages`, so a repeated panorama is not recomputed.

Point the viewer or the batch command at it with matching model, revision, tile-size and instance settings (the worker rejects any mismatch):

```bash
export BUILDING_ANALYSIS_BACKEND=remote
export BUILDING_ANALYSIS_REMOTE_URL=https://gpu.example.com
export BUILDING_ANALYSIS_REMOTE_TOKEN='use-a-long-random-secret'
export BUILDING_ANALYSIS_REMOTE_TIMEOUT_S=900
export BUILDING_ANALYSIS_MODEL=high
uv run city-analyser-viewer                       # or: uv run city-analyser-analyze <dataset>
```

`city-analyser-analyze` works with a remote worker too: it keeps up to `--concurrency` uploads in flight (default 2) while it associates the previous panorama, and the client needs no PyTorch. The client keeps a copy of every answer under `data/.stage-cache/remote-perception`, so re-associating after a footprint or threshold change does not upload again; pin `BUILDING_ANALYSIS_MODEL_REVISION` if you want a model update to invalidate that copy. The worker serializes GPU work, so run one worker per GPU and use HTTPS outside a trusted private network.

## Google Colab (or any notebook GPU)

A ready notebook is in [`notebooks/colab_quickstart.ipynb`](https://github.com/JSempereH/360_city_analyser/blob/main/notebooks/colab_quickstart.ipynb): Drive for persistence, install, models, analysis, index, crops, attributes and an optional vLLM server. It has not been run on Colab yet.

Colab is a good fit for a one-off batch. Instead of exposing a server, run only the model stages there and bring the cache back:

1. Put `data/<dataset>` (manifest and images) on Google Drive and open a GPU runtime.
2. Install the project (`pip install uv`, `uv sync --extra cu126`) and run
   `BUILDING_ANALYSIS_MODEL=high BUILDING_ANALYSIS_STAGE_CACHE_DIR=/content/drive/MyDrive/stage-cache uv run city-analyser-analyze <dataset> --data-root /content/drive/MyDrive/data --perception-only`.
3. Copy the stage-cache folder to `data/.stage-cache` on your machine (or point `BUILDING_ANALYSIS_STAGE_CACHE_DIR` at it) and run `city-analyser-analyze <dataset>` with the same model settings. Every model stage is a cache hit, so each panorama takes about a second.

`--perception-only` needs neither OSM nor your corrections. The stage-cache key includes the model revision read from the `models/` directory, so a mismatch is never wrong, only a cache miss. To make the keys agree, copy the Colab `models/` directory (or at least its `refs/` folders) back as well, or download the weights locally at the same time. Free Colab sessions are time limited and their terms restrict using them as a public server, which is another reason to prefer this batch flow over a tunnel to a Colab-hosted worker.
