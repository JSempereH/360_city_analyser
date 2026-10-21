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
        record = self.api("/api/footprint-corrections/way/102")
        self.assertEqual(record["revisions"][0]["status"], "reverted")
        self.assertEqual(record["revisions"][0]["revert_note"], "GPS error, not the footprint")
        # Re-association runs as a background job: wait for its result, not for a fixed time.
        import time

        for _ in range(300):
            if "way/102" not in self.api(f"/api/datasets/{DATASET}/analysis/{FIRST}")["footprint_sources"]["corrected"]:
                break
            time.sleep(0.1)
        else:
            self.fail("the re-association did not drop the reverted correction")
        page.wait_for_function("document.querySelector('#building-revert-button').hidden", timeout=30_000)

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

    def test_clicking_a_footprint_shows_the_surveyed_attributes(self) -> None:
        from city_analyser.attributes.vlm import VLMProvider
        from city_analyser.survey import run_survey

        answer = json.dumps({"observations": "Slab edges and columns at every storey.", "floors": 3, "gem_material": "CR", "gem_llrs": "LFINF"})
        run_survey(self.root / "data", self.root / "models", DATASET, provider_factory=lambda: VLMProvider(lambda images, prompt: answer, "vlm:e2e"), tile_m=20, log=lambda _: None)
        page = self.new_page()
        self.open_viewer(page)
        footprint = page.locator('#location-map-canvas path.leaflet-interactive[fill="#ecba74"]').first
        box = footprint.bounding_box()
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.wait_for_function("document.querySelector('#building-card-attributes').children.length > 0", timeout=20_000)
        card = page.text_content("#building-card")
        self.assertIn("CR/LFINF/HEX:3", card)
        self.assertIn("Reinforced concrete (CR)", card)
        page.click("#building-card-why summary")
        self.assertIn("Slab edges", page.text_content("#building-card-observations"))
        page.keyboard.press("Escape")
        self.assertTrue(page.locator("#building-card").is_hidden())
        page.click("#panel-toggle-button") if page.locator("#building-attributes-button").is_hidden() else None
        page.click("#building-attributes-button")
        self.assertTrue(page.locator("#building-card").is_visible())

    def test_footprints_colored_by_structure_and_a_review_from_the_card(self) -> None:
        from city_analyser.attributes.vlm import VLMProvider
        from city_analyser.survey import run_survey

        answer = json.dumps({"floors": 3, "gem_material": "CR", "gem_llrs": "LFINF"})
        run_survey(self.root / "data", self.root / "models", DATASET, provider_factory=lambda: VLMProvider(lambda images, prompt: answer, "vlm:e2e"), tile_m=20, log=lambda _: None)
        page = self.new_page()
        self.open_viewer(page, query="&building=way%2F104")
        page.click("#layers-menu summary")
        page.select_option("#attribute-color-select", "gem_material")
        page.click("#layers-menu summary")
        page.wait_for_function("document.querySelectorAll('#map-legend-items li').length > 0", timeout=20_000)
        self.assertIn("Concrete", page.text_content("#map-legend"))
        concrete = 'path.leaflet-interactive[fill="#2a78d6"]'
        colored = page.locator(f"#location-map-canvas {concrete}").count()
        self.assertGreater(colored, 0)
        page.wait_for_function("!document.querySelector('#building-card-review').hidden", timeout=20_000)
        page.click("#building-card-review")
        page.select_option("#review-gem-material", "MUR")
        page.fill("#review-note", "Load-bearing brick walls")
        page.click("#building-card-review-form button[type=submit]")
        page.wait_for_function("document.querySelector('#building-card-review-status').textContent.includes('E2E reviewer')", timeout=20_000)
        self.assertIn("MUR/LFINF/HEX:3", page.text_content("#building-card"))
        page.wait_for_function(f"document.querySelectorAll('#location-map-canvas {concrete}').length === {colored - 1}", timeout=20_000)
        self.assertIn("1 reviewed", page.text_content("#map-legend"))
        review = json.loads((self.root / "data" / DATASET / "attributes" / "way" / "104" / "attributes-human-review.json").read_text())
        self.assertEqual((review["run"]["reviewer"], review["predictions"]["gem_material"]["score_kind"]), ("E2E reviewer", "review"))

    def test_presentation_mode_surveys_by_itself_and_exits_cleanly(self) -> None:
        from city_analyser.demo import prepare_demo

        prepare_demo(self.root / "data", DATASET, log=lambda _: None)
        page = self.new_page()
        page.goto(f"{self.base}/viewer/?dataset={DATASET}&image={FIRST}&present=1&speed=20")
        # It opens by saying what the audience will see, phase by phase, as the HUD names them.
        page.wait_for_selector(".presentation-intro", timeout=60_000)
        self.assertEqual(page.locator(".presentation-phases li").all_text_contents(), ["Walk", "Look", "Reason", "Record", "Check"])
        page.wait_for_selector(".presentation-thinking-body", timeout=60_000)
        # Loading the dataset rewrites the URL; the presentation keeps its options.
        self.assertIn("speed=20", page.url)
        # It runs inside the normal viewer: the sidebar stays, the play button turns into stop.
        self.assertTrue(page.locator(".sidebar").is_visible())
        self.assertEqual(page.get_attribute("#presentation-button", "aria-pressed"), "true")
        self.assertIn("Demo dataset", page.text_content(".presentation-hud"))
        page.wait_for_selector(".presentation-save.saved, .presentation-save.flagged", timeout=90_000)
        self.assertRegex(page.text_content(".presentation-gem"), r"^(MUR|CR|S|MAT99)/")
        self.assertIn("Thought for", page.text_content(".presentation-thinking-header"))
        # Each saved building is colored on the map; at this speed the count may already be past one.
        page.wait_for_function("/[1-9][0-9]* saved/.test(document.querySelector('#map-legend-title').textContent)", timeout=30_000)
        self.assertIn("present=1", page.url)
        # The walk ends with what was found, the map colored by each attribute and the review queue, then the
        # panel is as it was.
        page.wait_for_function("document.querySelector('.presentation-panel').textContent.includes('Starting the walk again')", timeout=180_000)
        self.assertIn("described and saved", page.text_content(".presentation-panel"))
        self.assertTrue(page.locator("#panel-building").is_visible())
        self.assertFalse(page.locator(".app-shell.presenting-summary").count())
        page.keyboard.press(" ")
        self.assertIn("paused", page.get_attribute(".presentation", "class"))
        page.keyboard.press("Escape")
        page.wait_for_function("!document.querySelector('.presentation') && !document.querySelector('.app-shell').classList.contains('presenting')", timeout=10_000)
        self.assertEqual(page.get_attribute("#presentation-button", "aria-pressed"), "false")
        self.assertNotIn("present=1", page.url)
        # The play button starts it again.
        page.click("#presentation-button")
        page.wait_for_selector(".presentation-thinking-body", timeout=30_000)
        page.click("#presentation-button")
        page.wait_for_function("!document.querySelector('.presentation')", timeout=10_000)
        # Nothing was written: a presentation only shows what the attribute files hold.
        self.assertFalse(list((self.root / "data" / DATASET / "attributes").rglob("attributes-human-review.json")))

    def test_assistant_chats_analyzes_live_and_drives_a_live_presentation(self) -> None:
        sys.path.insert(0, str(HERE.parents[0]))
        from test_assistant import NOTES, FakeModelServer

        from city_analyser.demo import prepare_demo
        from city_analyser.local_viewer import ViewerServer

        prepare_demo(self.root / "data", DATASET, log=lambda _: None)
        model = FakeModelServer()
        self.addCleanup(model.close)
        with patch.dict(os.environ, {"ATTRIBUTES_VLM_URL": model.url, "ATTRIBUTES_VLM_MODEL": "fake-vl"}):
            server = ViewerServer(("127.0.0.1", 0), self.root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=5)))
        self.base = f"http://127.0.0.1:{server.server_port}"
        page = self.new_page()
        self.open_viewer(page, query="&building=way%2F103")
        page.click("#tab-assistant")
        page.wait_for_function("document.querySelector('#assistant-status').textContent.includes('fake-vl')", timeout=20_000)
        page.fill("#assistant-input", "Why masonry?")
        page.keyboard.press("Enter")
        page.wait_for_function(f"document.querySelector('#assistant-log').textContent.includes({json.dumps(NOTES[:30])})", timeout=30_000)
        self.assertIn("Why masonry?", page.text_content("#assistant-log"))
        page.click("#assistant-analyze-button")
        page.wait_for_function("document.querySelector('#assistant-log').textContent.includes('Saved:')", timeout=60_000)
        # The log survives the end of the request (it used to be redrawn from the chat alone).
        page.wait_for_function("!document.querySelector('#assistant-send-button').disabled", timeout=30_000)
        analysis = page.text_content("#assistant-log").split("Analysis", 1)[1]
        self.assertIn("Saved: MUR/LWAL", analysis)
        self.assertIn("View 1", analysis)
        self.assertIn("→ 4 storeys · MUR · LWAL", analysis)
        self.assertNotIn('"gem_material"', analysis)  # the model's JSON stays out of sight
        page.click("#assistant-edit-button")
        page.wait_for_selector("#building-card-review-form:not([hidden])", timeout=10_000)
        page.select_option("#review-building-type", "office")
        page.fill("#review-construction-year", "1905")
        page.click("#building-card-review-form button[type=submit]")
        page.wait_for_function("document.querySelector('#building-card-review-status').textContent.includes('E2E reviewer')", timeout=20_000)
        self.assertIn("Office", page.text_content("#building-card-attributes"))
        page.goto(f"{self.base}/viewer/?dataset={DATASET}&image={FIRST}&present=1&speed=20")
        page.wait_for_function("document.querySelector('.presentation-badge')?.textContent.startsWith('Live')", timeout=60_000)
        page.wait_for_function(f"document.querySelector('.presentation-thinking-body')?.textContent.includes({json.dumps(NOTES[:30])})", timeout=90_000)
        page.wait_for_selector(".presentation-save.saved, .presentation-save.flagged", timeout=120_000)
        self.assertTrue(any(request["body"].get("stream") for request in model.requests))

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
