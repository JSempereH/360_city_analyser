# Demo day

A checklist for showing the presentation to an audience. It replays saved answers (no model, no GPU) and needs no network once the rehearsal below has been run on the same laptop.

## The day before

1. `git pull` and `uv sync --extra cpu`.
2. `make presentation-data` rebuilds the three demo copies (Valencia centro, Madrid centro, Madrid Salesas). It takes a few minutes and uses under 1 GB of RAM.
3. Rehearse once **with** a network: `make presentation`, open the address it prints and let the walk run to the end. This fills the map tile cache (`data/.tile-cache/`) for the area shown.
4. Rehearse once **without** one: disconnect, `make presentation-offline`, and check that the map, the masks and the summary all appear.

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

- **A gray map:** the tiles of that area were never seen with a network. The walk still works; rehearse once online to fill the cache.
- **No masks or a stalled walk:** reload the page; the presentation resumes from the address.
- **Out of memory:** close the browser tabs and other programs and reload.
- **Detected-building colors or other diagnostic layers:** the presentation hides them and restores them on exit.

What the audience sees is computed only from the street images and OpenStreetMap footprints. The cadastre is used to evaluate the answers, and the viewer shows it only in evaluation mode (`references=1`).
