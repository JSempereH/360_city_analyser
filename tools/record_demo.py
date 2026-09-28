#!/usr/bin/env python3
"""Record a short GIF of the viewer: masks, building selection and the mini-map.

Prerequisites:
  * a dataset whose panoramas are already analyzed (analysis is not triggered
    here, so the recording shows instant results), and
  * the viewer running with the *same* analysis settings, e.g.

        BUILDING_ANALYSIS_MODEL=balanced BUILDING_ANALYSIS_TILE_SIZE=384 \\
            uv run --extra cpu city-analyser-viewer

Run (Playwright is a recording-only tool, not a project dependency):

    uv run --no-project --with playwright==1.62.0 python tools/record_demo.py --rehearse
    uv run --no-project --with playwright==1.62.0 python tools/record_demo.py

``--rehearse`` saves one screenshot per step instead of recording, to check
the choreography. The GIF is converted with ffmpeg (palette per clip).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

VIEWPORT = {"width": 1280, "height": 720}
# Headless Chromium on the host GPU renders the WebGL panorama at 60 fps.
CHROMIUM_ARGS = ["--enable-gpu", "--use-angle=gl", "--ignore-gpu-blocklist"]

# Visible cursor and click ripple: headless recordings contain no pointer.
CURSOR_SCRIPT = """
window.addEventListener('DOMContentLoaded', () => {
  const style = document.createElement('style');
  style.textContent = `
    #demo-cursor { position: fixed; z-index: 99999; width: 22px; height: 22px; pointer-events: none;
      left: -40px; top: -40px; transform: translate(-3px, -2px);
      background: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='22' height='22'><path d='M2 2 L2 18 L7 13 L11 21 L14 19.5 L10 12 L17 12 Z' fill='white' stroke='black' stroke-width='1.6' stroke-linejoin='round'/></svg>") no-repeat; }
    .demo-ripple { position: fixed; z-index: 99998; width: 14px; height: 14px; margin: -7px 0 0 -7px; border-radius: 50%;
      border: 3px solid #ff8a2a; pointer-events: none; animation: demo-ripple 0.6s ease-out forwards; }
    @keyframes demo-ripple { to { transform: scale(3.2); opacity: 0; } }`;
  document.head.append(style);
  const cursor = document.createElement('div');
  cursor.id = 'demo-cursor';
  document.body.append(cursor);
  const move = (event) => { cursor.style.left = event.clientX + 'px'; cursor.style.top = event.clientY + 'px'; };
  window.addEventListener('pointermove', move, true);
  window.addEventListener('pointerdown', (event) => {
    move(event);
    const ripple = document.createElement('div');
    ripple.className = 'demo-ripple';
    ripple.style.left = event.clientX + 'px';
    ripple.style.top = event.clientY + 'px';
    document.body.append(ripple);
    setTimeout(() => ripple.remove(), 700);
  }, true);
});
"""


class Director:
    """Runs the choreography; in rehearsal mode it screenshots each step."""

    def __init__(self, page: Page, rehearse_dir: Path | None) -> None:
        self.page = page
        self.rehearse_dir = rehearse_dir
        self.step = 0
        self.position = (800, 380)

    def hold(self, seconds: float, label: str) -> None:
        self.page.wait_for_timeout(int(seconds * 1000))
        if self.rehearse_dir is not None:
            self.step += 1
            self.page.screenshot(path=str(self.rehearse_dir / f"{self.step:02d}-{label}.png"))
            status = self.page.locator("#building-status").text_content() or ""
            print(f"{self.step:02d} {label}: {status.strip()[:110]}")

    def glide(self, x: float, y: float, seconds: float = 0.6) -> None:
        steps = max(2, int(seconds * 30))
        self.page.mouse.move(x, y, steps=steps)
        self.position = (x, y)

    def click(self, x: float, y: float) -> None:
        self.glide(x, y)
        self.page.wait_for_timeout(150)
        self.page.mouse.click(x, y)

    def drag(self, start: tuple[float, float], end: tuple[float, float], seconds: float) -> None:
        self.glide(*start, seconds=0.4)
        self.page.mouse.down()
        self.page.mouse.move(*end, steps=max(2, int(seconds * 30)))
        self.page.mouse.up()
        self.position = end

    def click_footprint(self, osm_id: str) -> None:
        """Click a mini-map footprint by OSM id (found by selecting candidates offscreen-free)."""
        box = self.page.evaluate(FOOTPRINT_BOX, osm_id)
        if box is None:
            raise RuntimeError(f"footprint {osm_id} is not visible on the mini-map")
        self.click(*box)

    def click_capture(self, panorama_number: int) -> None:
        box = self.page.evaluate(CAPTURE_BOX, panorama_number)
        if box is None:
            raise RuntimeError(f"panorama {panorama_number} is not on the mini-map")
        self.click(*box)


# Leaflet keeps the GeoJSON feature on each layer; walk the SVG paths and ask
# Leaflet which one carries the requested OSM id. The page exposes nothing
# global, so read it from the map container's layer registry.
FOOTPRINT_BOX = """(osmId) => {
  const map = window.__demoMap;
  let target = null;
  map.eachLayer((layer) => { if (layer.feature?.properties?.osm_id === osmId && layer._path) target = layer._path; });
  if (!target) return null;
  const box = target.getBoundingClientRect();
  const view = document.querySelector('#location-map-canvas').getBoundingClientRect();
  const x = Math.min(Math.max(box.left + box.width / 2, view.left + 8), view.right - 8);
  const y = Math.min(Math.max(box.top + box.height / 2, view.top + 8), view.bottom - 8);
  return [x, y];
}"""
CAPTURE_BOX = """(number) => {
  const map = window.__demoMap;
  let target = null;
  map.eachLayer((layer) => {
    if (layer.getTooltip?.()?.getContent?.() === `Panorama ${number}` && layer._path) target = layer._path;
  });
  if (!target) return null;
  const box = target.getBoundingClientRect();
  return [box.left + box.width / 2, box.top + box.height / 2];
}"""
# Capture the Leaflet map instance when the viewer creates it.
MAP_HOOK_SCRIPT = """
(() => {
  let original;
  Object.defineProperty(window, 'L', {
    configurable: true,
    get() { return original; },
    set(value) {
      original = value;
      const create = value.map;
      value.map = function (element, options) {
        const map = create.call(this, element, options);
        if (element && element.id === 'location-map-canvas') window.__demoMap = map;
        return map;
      };
    },
  });
})();
"""


def choreography(director: Director) -> None:
    page = director.page
    # 1. Overview: every matched building tinted, mini-map with view cone.
    director.hold(1.8, "overview")
    # 2. Look across the street, slowly, to the Instituto Jimenez block.
    director.drag((980, 470), (520, 470), seconds=2.6)
    director.hold(0.5, "rotated")
    # 3. Pick a facade in the panorama: outlined mask + highlighted footprint.
    director.click(640, 330)
    director.hold(2.2, "picked-in-panorama")
    # 4. Pick a footprint on the mini-map: the view turns to that building.
    director.click_footprint("way/992065315")
    director.hold(2.4, "picked-on-map")
    # 5. Jump to the neighboring capture from the mini-map.
    director.click_capture(3)
    page.wait_for_function("() => document.querySelector('#status').textContent.includes('Panorama 3')")
    director.hold(1.2, "next-panorama")
    # 6. Select a building of the new scene and let the result sit.
    director.click(470, 260)
    director.hold(2.8, "picked-second-scene")


def record(url: str, output: Path, rehearse: bool, width: int, fps: int) -> None:
    rehearse_dir = output.parent / "rehearsal" if rehearse else None
    if rehearse_dir is not None:
        shutil.rmtree(rehearse_dir, ignore_errors=True)
        rehearse_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="viewer-demo-") as video_dir, sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=CHROMIUM_ARGS)
        options = {"viewport": VIEWPORT, "device_scale_factor": 1}
        if not rehearse:
            options.update(record_video_dir=video_dir, record_video_size=VIEWPORT)
        context = browser.new_context(**options)
        context.add_init_script(MAP_HOOK_SCRIPT)
        context.add_init_script(CURSOR_SCRIPT)
        page = context.new_page()
        started = time.perf_counter()
        page.goto(url)
        # Ready when the panorama, its analysis overlay and the footprints are in.
        page.wait_for_function("() => document.querySelector('#building-status').textContent.includes('matched buildings')")
        page.wait_for_function("() => document.querySelectorAll('#location-map-canvas path.leaflet-interactive').length > 10")
        page.wait_for_timeout(1500)  # map tiles
        page.mouse.move(800, 380)
        ready = time.perf_counter() - started
        director = Director(page, rehearse_dir)
        choreography(director)
        video = page.video
        context.close()
        browser.close()
        if rehearse:
            print(f"rehearsal screenshots in {rehearse_dir}")
            return
        source = Path(video.path())
        clip = output.with_suffix(".webm")
        shutil.copy(source, clip)
    palette_filter = (
        f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ready:.2f}", "-i", str(clip), "-vf", palette_filter, "-loop", "0", str(output)],
        check=True,
    )
    print(f"{output} ({output.stat().st_size / 1e6:.1f} MB), source clip {clip}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--dataset", default="demo-san-jose")
    parser.add_argument("--image", default="3594589034100632")
    parser.add_argument("--start", default="yaw=-60&pitch=6&fov=95", help="initial view query")
    parser.add_argument("--output", type=Path, default=Path("docs/viewer-demo.gif"))
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--rehearse", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    url = f"{args.base_url}/viewer/?dataset={args.dataset}&image={args.image}&{args.start}"
    record(url, args.output, args.rehearse, args.width, args.fps)


if __name__ == "__main__":
    main()
