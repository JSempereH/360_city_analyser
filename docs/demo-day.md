# Demo day

A checklist for showing the presentation to an audience. It replays saved answers (no model, no GPU) and needs no network.

## Presenting on another machine

The repository holds the code and the curations, not the data (`data/` is not in git): a fresh clone cannot rebuild the exact demo without downloading the panoramas again and analyzing them on a GPU, and an analysis made elsewhere can differ slightly from the one that was reviewed. So the demo travels as one zip.

**Who prepared the demo** runs `make demo-bundle` and shares `madrid-salesas-demo-bundle.zip` (about 430 MB) by a file link. It holds the panoramas, analyses, crops, the answers of every building, the route, the curation, and the OSM footprints and map tiles of the area.

**Who presents it** needs Git and [uv](https://docs.astral.sh/uv/) (Linux, macOS or Windows; no GPU, no PyTorch, no Mapillary token):

```bash
git clone https://github.com/JSempereH/360_city_analyser.git
cd 360_city_analyser
uv sync                     # the base dependencies only, about 500 MB
uv run city-analyser-results import /path/to/madrid-salesas-demo-bundle.zip
make presentation-offline   # without make: CITY_ANALYSER_OFFLINE=1 BUILDING_ANALYSIS_MODEL=balanced BUILDING_ANALYSIS_TILE_SIZE=384 uv run city-analyser-viewer
```

Do not run `make presentation-data` on that machine: it rebuilds the demos from the source datasets, which the bundle does not carry.

## The day before

1. On the machine that will present: import the latest bundle as above (it replaces the previous copy), or, on the machine where the demo was made, `git pull` and `make presentation-data`.
2. Rehearse once **without** a network: disconnect, `make presentation-offline`, and check that the map, the masks and the summary all appear.

## On the day

1. Close other heavy programs; the viewer uses about 200 MB and the browser about 1 GB.
2. `make presentation-offline` (or `make presentation` if the room has a reliable network).
3. Open `http://127.0.0.1:8765/viewer/?dataset=madrid-salesas-demo&present=1&live=0&speed=2` in a full-screen browser window (F11), then press <kbd>F</kbd> so the panorama fills it.
4. The walk takes about 2 minutes 15 seconds: four buildings from the 1900s to today, then the map colored by structure and the review queue. It starts again by itself.

## Keys

| Key | Effect |
|---|---|
| <kbd>Space</kbd> | Pause and resume |
| <kbd>→</kbd> | Skip to the next building |
| <kbd>H</kbd> | Fold the assistant's notes away, to show more of the panorama |
| <kbd>F</kbd> | Focus: hide the side columns |
| <kbd>Esc</kbd> | Back to the normal viewer |

## If something goes wrong

- **A gray map:** the tiles of that area are not in the cache (only the ones viewed while the bundle was made are). The walk still works; with a network, `make presentation` fetches and keeps them.
- **No masks or a stalled walk:** reload the page; the presentation resumes from the address.
- **Out of memory:** close the browser tabs and other programs and reload.
- **Detected-building colors or other diagnostic layers:** the presentation hides them and restores them on exit.

What the audience sees is computed only from the street images and OpenStreetMap footprints. The cadastre is used to evaluate the answers, and the viewer shows it only in evaluation mode (`references=1`).
