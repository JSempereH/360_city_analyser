"""Browser tests of the viewer against a synthetic workspace (no network data, no models).

Run with ``make e2e``. Playwright is a test-only tool, so these tests are
skipped when it is not installed; ``make test`` does not collect them.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import patch

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixture  # noqa: E402

_spec = importlib.util.spec_from_file_location("viewer_check", HERE.parents[1] / "tools" / "viewer_check.py")
viewer_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(viewer_check)

DATASET = fixture.DATASET
FIRST, SECOND, THIRD = (camera[0] for camera in fixture.CAMERAS)
STATUS_DOTS = "[...document.querySelectorAll('#image-list .status-dot')].map((dot) => dot.dataset.status)"
URL_PARAMETER = "(name) => new URLSearchParams(location.search).get(name)"
OVERLAY_PIXELS = """() => {
  const canvas = document.querySelector('#projection-overlay');
  const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
  let blue = 0;
  for (let index = 0; index < data.length; index += 4) if (data[index + 3] > 100 && data[index + 2] > 200 && data[index] < 90) blue++;
  return blue;
}"""


@unittest.skipIf(sync_playwright is None, "Playwright is not installed; run make e2e")
class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        environment = patch.dict(os.environ, fixture.create_workspace(self.root))
        environment.start()
        self.addCleanup(environment.stop)
        from city_analyser.local_viewer import ViewerServer

        server = ViewerServer(("127.0.0.1", 0), self.root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=5)))
        self.base = f"http://127.0.0.1:{server.server_port}"

    # Helpers

    def api(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def new_page(self, width: int = 1440, height: int = 900) -> Any:
        context = self.browser.new_context(viewport={"width": width, "height": height})
        # The reviewer name is remembered per browser; tests start with one.
        context.add_init_script("localStorage.setItem('footprint-author', 'E2E reviewer')")
        self.addCleanup(context.close)
        page = context.new_page()
        self.page_errors: list[str] = []
        page.on("pageerror", lambda error: self.page_errors.append(str(error)))
        return page

    def open_viewer(self, page: Any, image: str = FIRST, query: str = "") -> None:
        page.goto(f"{self.base}/viewer/?dataset={DATASET}&image={image}{query}")
        page.wait_for_function("document.querySelector('#loading').hidden", timeout=60_000)
        page.wait_for_function("document.querySelector('#map-message').textContent.includes('nearby building footprints')", timeout=60_000)
        self.assertTrue(page.locator("#error").is_hidden(), page.text_content("#error"))

    def search(self, page: Any, text: str) -> None:
        page.fill("#search-input", text)
        page.press("#search-input", "Enter")
        page.wait_for_function("!document.querySelector('#search-message').hidden", timeout=30_000)
        page.wait_for_timeout(800)

    def propose_shift(self, osm_id: str, east_degrees: float = 0.00001) -> dict[str, Any]:
        feature = next(item for item in fixture.osm_features() if item["properties"]["osm_id"] == osm_id)
        ring = [[x + east_degrees, y] for x, y in feature["geometry"]["coordinates"][0]]
        return self.api(f"/api/footprint-corrections/{osm_id}", {
            "action": "propose", "geometry": {"type": "Polygon", "coordinates": [ring]}, "author": "fixture", "note": "facade further east",
            "evidence": [{"dataset": DATASET, "image": FIRST}],
        })

    def wait_for_job(self, job_id: str) -> dict[str, Any]:
        import time

        for _ in range(300):
            job = self.api(f"/api/jobs/{job_id}")
            if job["status"] in {"complete", "failed", "cancelled"}:
                return job
            time.sleep(0.1)
        self.fail(f"job {job_id} did not finish")

    def tearDown(self) -> None:
        self.assertEqual(getattr(self, "page_errors", []), [])

    # Tests

    def test_layout_has_no_overlaps_or_errors_at_any_size_and_theme(self) -> None:
        # CI keeps the screenshots as an artifact when this test fails.
        output = Path(os.environ.get("VIEWER_E2E_SCREENSHOTS") or self.root / "layout")
        results = viewer_check.check_viewports(self.browser, self.base, DATASET, FIRST, output, list(viewer_check.VIEWPORTS), "dark")
        results += viewer_check.check_viewports(self.browser, self.base, DATASET, FIRST, output, ["desktop", "phone"], "light")
        failures = {f"{result['viewport']} {result['state']}": viewer_check.problems(result) for result in results if viewer_check.problems(result)}
        self.assertEqual(failures, {})
        self.assertGreater(len(results), 30)

    def test_review_queue_opens_and_accepts_a_proposal_from_the_keyboard(self) -> None:
        self.propose_shift("way/102")
        page = self.new_page()
        self.open_viewer(page, image=THIRD)
        page.click("#tab-review")
        page.wait_for_function("document.querySelector('#review-position').textContent.includes('to review')", timeout=20_000)
        page.keyboard.press("n")
        page.wait_for_function("document.querySelector('#review-current-kind').textContent === 'Proposed correction'", timeout=20_000)
        self.assertEqual(page.evaluate(URL_PARAMETER, "image"), FIRST)
        page.keyboard.press("a")
        page.wait_for_function("document.querySelector('#footprint-review-message').textContent.includes('Re-associated')", timeout=60_000)

        record = self.api("/api/footprint-corrections/way/102")
        self.assertEqual([revision["status"] for revision in record["revisions"]], ["accepted"])
        self.assertEqual(record["revisions"][0]["reviewer"], "E2E reviewer")
        analysis = self.api(f"/api/datasets/{DATASET}/analysis/{FIRST}")
        self.assertIn("way/102", analysis["footprint_sources"]["corrected"])

    def test_revert_returns_to_the_openstreetmap_geometry(self) -> None:
        revision = self.propose_shift("way/102")["revision"]
        job = self.api("/api/footprint-corrections/way/102", {"action": "review", "revision": revision["revision"], "decision": "accepted", "reviewer": "fixture"})
        self.wait_for_job(job["reassociation"]["job"]["id"])
        page = self.new_page()
        page.on("dialog", lambda dialog: dialog.accept("GPS error, not the footprint") if dialog.type == "prompt" else dialog.accept())
        self.open_viewer(page)
        self.search(page, "way/102")
        self.assertTrue(page.locator("#building-revert-button").is_visible())
        page.click("#building-revert-button")
        page.wait_for_function("document.querySelector('#building-status').textContent.includes('reverted') || document.querySelector('#building-status').textContent.includes('matched buildings')", timeout=60_000)
        page.wait_for_timeout(1500)

        record = self.api("/api/footprint-corrections/way/102")
        self.assertEqual(record["revisions"][0]["status"], "reverted")
        self.assertEqual(record["revisions"][0]["revert_note"], "GPS error, not the footprint")
        analysis = self.api(f"/api/datasets/{DATASET}/analysis/{FIRST}")
        self.assertNotIn("way/102", analysis["footprint_sources"]["corrected"])
        self.assertTrue(page.locator("#building-revert-button").is_hidden())

    def test_split_turns_one_footprint_into_part_buildings(self) -> None:
        page = self.new_page()
        self.open_viewer(page, image=SECOND)
        self.search(page, fixture.MERGED_FOOTPRINT)
        page.click("#building-edit-button")
        page.wait_for_selector(".footprint-midpoint")
        edges = page.locator(".footprint-midpoint").count()
        page.locator(".footprint-midpoint").nth(0).click(force=True)
        page.locator(".footprint-midpoint").nth(edges // 2 + 1).click(force=True)
        page.click("#footprint-split-button")
        page.locator(".footprint-vertex").nth(1).click(force=True)
        page.locator(".footprint-vertex").nth(edges // 2 + 2).click(force=True)
        page.click("#footprint-split-button")
        self.assertEqual(page.locator(".footprint-mover").count(), 2)
        page.fill("#footprint-note", "Two buildings share this footprint.")
        page.click("#footprint-edit-form button[type=submit]")
        page.wait_for_function("document.querySelector('#footprint-review-message').textContent.includes('Proposed revision')", timeout=20_000)
        revision = self.api(f"/api/footprint-corrections/{fixture.MERGED_FOOTPRINT}")["revisions"][-1]
        self.assertEqual((revision["kind"], len(revision["parts"])), ("split", 2))
        job = self.api(f"/api/footprint-corrections/{fixture.MERGED_FOOTPRINT}", {"action": "review", "revision": revision["revision"], "decision": "accepted", "reviewer": "fixture"})
        self.wait_for_job(job["reassociation"]["job"]["id"])

        buildings = {building["osm_id"] for building in self.api(f"/api/datasets/{DATASET}/analysis/{SECOND}")["building_ids"].values()}
        self.assertTrue({f"{fixture.MERGED_FOOTPRINT}/part-1", f"{fixture.MERGED_FOOTPRINT}/part-2"} & buildings)
        self.assertNotIn(fixture.MERGED_FOOTPRINT, buildings)

    def test_search_selection_and_review_links(self) -> None:
        page = self.new_page()
        self.open_viewer(page)
        self.search(page, "way/104")
        self.assertEqual(page.text_content("#building-title"), "way/104")
        longitude, latitude = fixture.lon_lat(fixture.CAMERAS[2][1], 0)
        self.search(page, f"{latitude}, {longitude}")
        self.assertEqual(page.evaluate(URL_PARAMETER, "image"), THIRD)
        self.assertIn("0 m", page.text_content("#search-message"))
        page.goto(f"{self.base}/viewer/?dataset={DATASET}&image={FIRST}&review=merged_footprint:{fixture.MERGED_FOOTPRINT}")
        page.wait_for_function(f"document.querySelector('#review-current-label')?.textContent === '{fixture.MERGED_FOOTPRINT}'", timeout=30_000)
        self.assertEqual(page.get_attribute("#tab-review", "aria-selected"), "true")

    def test_batch_analysis_fills_missing_analyses_and_updates_the_list(self) -> None:
        (self.root / "data" / DATASET / "analysis" / f"{THIRD}.json").unlink()
        page = self.new_page()
        self.open_viewer(page)
        page.wait_for_function("document.querySelectorAll('#image-list .status-dot').length === 3", timeout=20_000)
        self.assertEqual(page.evaluate(STATUS_DOTS), ["current", "current", "missing"])
        page.click(".viewer-actions details.advanced-actions:not(.layers-menu) summary")
        page.click("#analyze-dataset-button")
        page.wait_for_function("[...document.querySelectorAll('#job-list li')].some((item) => item.dataset.status === 'complete')", timeout=60_000)
        page.wait_for_function(f"({STATUS_DOTS}).every((status) => status === 'current')", timeout=20_000)
        self.assertTrue(page.locator("#job-list progress").is_visible())

    def test_layers_projection_and_undo(self) -> None:
        page = self.new_page()
        self.open_viewer(page)
        page.click("#layers-menu summary")
        page.check('[data-layer="instances"]')
        page.wait_for_timeout(800)
        self.assertTrue(page.evaluate("performance.getEntriesByType('resource').some((entry) => entry.name.includes('layer=instances'))"))
        page.click("#layers-menu summary")
        self.search(page, "way/103")
        page.click("#building-edit-button")
        page.wait_for_selector(".footprint-mover")
        page.wait_for_timeout(500)
        self.assertGreater(page.evaluate(OVERLAY_PIXELS), 0, "the edited walls are drawn on the panorama")
        box = page.locator(".footprint-mover").first.bounding_box()
        page.mouse.move(box["x"] + 6, box["y"] + 6)
        page.mouse.down()
        page.mouse.move(box["x"] + 40, box["y"] + 6, steps=8)
        page.mouse.up()
        page.wait_for_timeout(300)
        self.assertTrue(page.is_enabled("#footprint-undo-button"))
        page.keyboard.press("Control+z")
        page.wait_for_timeout(300)
        self.assertTrue(page.is_enabled("#footprint-redo-button"))
        page.click("#footprint-cancel-button")
        page.wait_for_timeout(300)
        self.assertEqual(page.evaluate(OVERLAY_PIXELS), 0)


if __name__ == "__main__":
    unittest.main()
