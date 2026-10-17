#!/usr/bin/env python3
"""Check the viewer layout in a real browser: overlaps, overflow, errors, memory.

For each viewport and UI state it records:

* console and page errors;
* visible buttons, inputs and text that another element covers (sampled with
  ``elementFromPoint``), i.e. overlapping panels or controls;
* text clipped without an ellipsis, and elements outside the viewport that no
  scrollable container holds;
* a screenshot, and the resident memory of the viewer and the browser.

It aborts when available system memory drops below ``--min-available-mib``, so
it never competes with other processes for RAM. Run it against a running viewer
whose analyses are already cached (no model runs):

    uv run --no-project --with playwright==1.62.0 python tools/viewer_check.py \\
        --base-url http://127.0.0.1:8765 --dataset demo-san-jose --output /tmp/viewer-check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

VIEWPORTS = {"desktop": (1440, 900), "laptop": (1280, 720), "tablet": (1024, 768), "phone": (390, 844)}

# Runs in the page: every visible control or text leaf that something else covers,
# clipped text, and elements outside the viewport with no scrollable ancestor.
LAYOUT_PROBE = """
() => {
  const selector = "button, select, input, textarea, label, a, h1, h2, h3, p, li, small, strong, kbd, summary, [role=tab]";
  const visible = (element) => {
    const style = getComputedStyle(element);
    const box = element.getBoundingClientRect();
    return box.width > 1 && box.height > 1 && style.visibility !== "hidden" && style.display !== "none" && Number(style.opacity) > 0.05
      && !element.closest("[hidden], dialog:not([open]), .leaflet-container")
      && !(element.closest("details:not([open])") && !element.matches("summary"));
  };
  const describe = (element) => {
    const id = element.id ? `#${element.id}` : "";
    const classes = element.classList.length ? `.${[...element.classList].slice(0, 2).join(".")}` : "";
    const text = (element.textContent || element.value || "").trim().replace(/\\s+/g, " ").slice(0, 40);
    return `${element.tagName.toLowerCase()}${id}${classes}${text ? ` "${text}"` : ""}`;
  };
  const scrollable = (element) => {
    for (let node = element.parentElement; node; node = node.parentElement) {
      const style = getComputedStyle(node);
      if (/(auto|scroll)/.test(style.overflowY + style.overflowX)) return node;
    }
    return null;
  };
  const covered = [], clipped = [], outside = [];
  for (const element of document.querySelectorAll(selector)) {
    if (!visible(element)) continue;
    const box = element.getBoundingClientRect();
    const container = scrollable(element);
    const clip = container ? container.getBoundingClientRect() : { left: 0, top: 0, right: innerWidth, bottom: innerHeight };
    // Points scrolled out of view inside a container are not overlaps.
    const points = [[0.5, 0.5], [0.15, 0.5], [0.85, 0.5]].map(([fx, fy]) => [box.left + box.width * fx, box.top + box.height * fy])
      .filter(([x, y]) => x >= Math.max(0, clip.left) && x <= Math.min(innerWidth, clip.right) - 1 && y >= Math.max(0, clip.top) && y <= Math.min(innerHeight, clip.bottom) - 1);
    for (const [x, y] of points) {
      const top = document.elementFromPoint(x, y);
      // Drawers and sheets marked data-overlay cover content on purpose.
      const overlay = top?.closest("[data-overlay]");
      if (top && top !== element && !element.contains(top) && !top.contains(element) && !(overlay && !overlay.contains(element))) {
        covered.push({ element: describe(element), by: describe(top) });
        break;
      }
    }
    const style = getComputedStyle(element);
    if (!["INPUT", "SELECT", "TEXTAREA"].includes(element.tagName) && style.textOverflow !== "ellipsis" && !/(auto|scroll)/.test(style.overflowX)) {
      if (/(hidden|clip)/.test(style.overflowX)) {
        // Hidden overflow: wider content is cut off.
        if (element.scrollWidth > element.clientWidth + 1 && element.clientWidth > 0) clipped.push(describe(element));
      } else {
        // Visible overflow: only the element's own text spilling out of its box counts;
        // positioned decorations (::after labels) are deliberate and not in the range.
        const range = document.createRange();
        range.selectNodeContents(element);
        const text = range.getBoundingClientRect();
        if (text.width > 0 && (text.left < box.left - 1 || text.right > box.right + 1)) clipped.push(describe(element));
      }
    }
    if (!container && (box.right > innerWidth + 1 || box.bottom > innerHeight + 1 || box.left < -1 || box.top < -1)) {
      outside.push(describe(element));
    }
  }
  return { covered, clipped, outside };
}
"""


def _meminfo_available_mib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return float("inf")


def _rss_mib(pattern: str) -> float:
    """Total resident memory of processes whose command line matches ``pattern``."""
    output = subprocess.run(["ps", "-eo", "rss,args"], capture_output=True, text=True, check=True).stdout
    return round(sum(int(line.split(None, 1)[0]) for line in output.splitlines()[1:] if pattern in line) / 1024, 1)


class Checker:
    def __init__(self, page: Any, output: Path, viewport: str, minimum_available_mib: float) -> None:
        self.page, self.output, self.viewport, self.minimum = page, output, viewport, minimum_available_mib
        self.results: list[dict[str, Any]] = []
        self.errors: list[str] = []
        page.on("pageerror", lambda error: self.errors.append(f"pageerror: {error}"))
        # Failed requests are reported with their URL below; the console only repeats them without it.
        page.on("console", lambda message: self.errors.append(f"console: {message.text}") if message.type == "error" and "Failed to load resource" not in message.text else None)
        page.on("response", self._response)

    def _response(self, response: Any) -> None:
        url = response.url
        # The API answers 404 for "no analysis / no SfM / no report yet"; map tiles are third-party.
        expected = response.status == 404 and "/api/datasets/" in url
        if response.status >= 400 and not expected and "tile.openstreetmap.org" not in url:
            self.errors.append(f"http {response.status}: {url}")

    def check(self, state: str) -> dict[str, Any]:
        available = _meminfo_available_mib()
        if available < self.minimum:
            raise SystemExit(f"stopping: only {available:.0f} MiB of memory available (minimum {self.minimum:.0f})")
        self.page.wait_for_timeout(400)
        layout = self.page.evaluate(LAYOUT_PROBE)
        # The viewer reports startup and loading failures in #error rather than throwing.
        shown_error = self.page.evaluate("(() => { const e = document.querySelector('#error'); return e && !e.hidden ? e.textContent.trim() : null; })()")
        if shown_error:
            self.errors.append(f"viewer error: {shown_error}")
        screenshot = self.output / f"{self.viewport}-{state}.png"
        self.page.screenshot(path=str(screenshot))
        result = {
            "viewport": self.viewport, "state": state, **layout, "errors": self.errors[:],
            "memory_mib": {"viewer": _rss_mib("city-analyser-viewer"), "browser": _rss_mib("headless_shell") or _rss_mib("chrom"), "available": round(available)},
            "screenshot": str(screenshot),
        }
        self.errors.clear()
        self.results.append(result)
        problems = len(result["covered"]) + len(result["clipped"]) + len(result["outside"]) + len(result["errors"])
        print(f"{self.viewport:8s} {state:28s} problems {problems:3d} · viewer {result['memory_mib']['viewer']} MiB · browser {result['memory_mib']['browser']} MiB", flush=True)
        return result


def _default_states(base_url: str, dataset: str, image: str | None) -> list[tuple[str, Callable[[Any], None]]]:
    """UI states every viewport is checked in; later states build on earlier ones."""
    def open_viewer(page: Any) -> None:
        query = f"?dataset={dataset}" + (f"&image={image}" if image else "")
        page.goto(f"{base_url}/viewer/{query}")
        page.wait_for_function("document.querySelector('#loading')?.hidden === true", timeout=60_000)
        page.wait_for_timeout(1500)

    def open_panel(page: Any) -> None:
        toggle = page.locator("#panel-toggle-button")
        if toggle.get_attribute("aria-expanded") != "true":
            toggle.click()
            page.wait_for_timeout(300)

    def select_building(page: Any) -> None:
        target = page.locator('#location-map-canvas path.leaflet-interactive[fill="#ecba74"]').first
        if target.count() and target.is_visible():
            # A mouse click at the footprint's center: whatever lies on top must still select it.
            box = target.bounding_box()
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        open_panel(page)

    def building_review(page: Any) -> None:
        # The card offers a review once the building has surveyed attributes. A details
        # drawer (narrow screens) lies over the card, so it is closed first, as a person would.
        close = page.locator("#panel-close-button")
        if close.is_visible():
            close.click()
        button = page.locator("#building-card-review")
        if button.is_visible():
            button.click()
            page.wait_for_timeout(300)

    def review_tab(page: Any) -> None:
        cancel = page.locator("#building-card-review-cancel")
        if cancel.is_visible():
            cancel.click()
        open_panel(page)
        page.click("#tab-review")
        page.wait_for_timeout(800)

    def review_current(page: Any) -> None:
        page.keyboard.press("n")
        page.wait_for_timeout(2500)

    def footprint_edit(page: Any) -> None:
        button = page.locator("#footprint-edit-button")
        if button.is_visible():
            button.click()
            target = page.locator('#location-map-canvas path.leaflet-interactive[fill="#ecba74"]').first
            if target.count() and target.is_visible():
                target.click(force=True)

    def layers_menu(page: Any) -> None:
        cancel = page.locator("#footprint-cancel-button")
        if cancel.is_visible():
            cancel.click()
        close = page.locator("#panel-close-button")
        if close.is_visible():
            close.click()
        page.click("#layers-menu summary")
        page.wait_for_timeout(300)

    def building_colors(page: Any) -> None:
        page.select_option("#attribute-color-select", "gem_material")
        page.click("#layers-menu summary")
        page.wait_for_timeout(800)

    def help_dialog(page: Any) -> None:
        if page.locator("#layers-menu").get_attribute("open") is not None:
            page.click("#layers-menu summary")
        page.keyboard.press("?")
        page.wait_for_timeout(300)

    def assistant_tab(page: Any) -> None:
        if page.locator("#help-dialog").get_attribute("open") is not None:
            page.click("#close-help-dialog-button")
        open_panel(page)
        cancel = page.locator("#footprint-cancel-button")
        if cancel.is_visible():
            cancel.click()
        page.click("#tab-assistant")
        page.wait_for_timeout(300)

    def jobs_tab(page: Any) -> None:
        if page.locator("#help-dialog").get_attribute("open") is not None:
            page.click("#close-help-dialog-button")
        open_panel(page)
        cancel = page.locator("#footprint-cancel-button")
        if cancel.is_visible():
            cancel.click()
        page.click("#tab-jobs")

    return [
        ("loaded", open_viewer), ("building-selected", select_building), ("building-review", building_review), ("review-tab", review_tab),
        ("review-current", review_current), ("footprint-edit", footprint_edit), ("layers-menu", layers_menu), ("building-colors", building_colors),
        ("help-dialog", help_dialog), ("assistant-tab", assistant_tab), ("jobs-tab", jobs_tab),
    ]


def check_viewports(
    browser: Any, base_url: str, dataset: str, image: str | None, output: Path, viewports: list[str],
    color_scheme: str = "dark", minimum_available_mib: float = 2048,
) -> list[dict[str, Any]]:
    """Walk every UI state at each viewport size; one result per state (also used by tests/e2e)."""
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in viewports:
        width, height = VIEWPORTS[name]
        context = browser.new_context(viewport={"width": width, "height": height}, color_scheme=color_scheme, device_scale_factor=1)
        page = context.new_page()
        checker = Checker(page, output, f"{name}-{color_scheme}" if color_scheme != "dark" else name, minimum_available_mib)
        for state, action in _default_states(base_url, dataset, image):
            action(page)
            checker.check(state)
        results.extend(checker.results)
        context.close()
    return results


def problems(result: dict[str, Any]) -> list[str]:
    return [f"covered: {item}" for item in result["covered"]] + [f"clipped: {item}" for item in result["clipped"]] \
        + [f"outside: {item}" for item in result["outside"]] + result["errors"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the viewer layout in a headless browser.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viewports", nargs="+", choices=sorted(VIEWPORTS), default=list(VIEWPORTS))
    parser.add_argument("--color-scheme", choices=["dark", "light"], default="dark")
    parser.add_argument("--min-available-mib", type=float, default=2048)
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    args.output.mkdir(parents=True, exist_ok=True)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"{args.base_url}/api/health", timeout=2)
            break
        except OSError:
            time.sleep(0.5)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        results = check_viewports(browser, args.base_url, args.dataset, args.image, args.output, args.viewports, args.color_scheme, args.min_available_mib)
        browser.close()
    (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    failing = [result for result in results if problems(result)]
    print(f"{len(failing)} of {len(results)} states have layout problems or errors; report: {args.output / 'report.json'}")
    sys.exit(1 if failing else 0)


if __name__ == "__main__":
    main()
