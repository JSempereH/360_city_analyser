import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from city_analyser.attributes.vlm import VLMProvider
from city_analyser.survey import plan_tiles, run_survey, survey_state_path

sys.path.insert(0, str(Path(__file__).resolve().parent / "e2e"))
import fixture  # noqa: E402

ANSWER = json.dumps({
    "observations": "Regular grid of openings with slab edges at every storey.", "floors": 4,
    "gem_material": "CR", "alternate_gem_material": "MCF", "gem_llrs": "LFINF", "alternate_gem_llrs": "LFM", "building_type": "apartments",
})


def image(image_id, longitude, latitude):
    return {"id": image_id, "computed_geometry": {"type": "Point", "coordinates": [longitude, latitude]}}


class TilePlanTests(unittest.TestCase):
    def test_tiles_are_visited_row_by_row_in_alternating_directions(self):
        step = 0.0019  # about 160 m of longitude at this latitude, 210 m of latitude
        images = [image(f"{row}{column}", column * step, 40 + row * step) for row in range(2) for column in range(3)]
        images.append({"id": "no-gps"})
        tiles = plan_tiles(images, 150)
        self.assertEqual([tile["images"] for tile in tiles], [["00"], ["01"], ["02"], ["12"], ["11"], ["10"]])
        self.assertEqual(plan_tiles(images, 10_000)[0]["images"], sorted(item["id"] for item in images[:-1]))
        with self.assertRaises(ValueError):
            plan_tiles(images, 0)


class SurveyWorkspace(unittest.TestCase):
    """The synthetic workspace of the browser tests, with a fake vision-language model."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        environment = patch.dict(os.environ, fixture.create_workspace(self.root))
        environment.start()
        self.addCleanup(environment.stop)
        # Footprints and tags come from the fixture's OSM cache: any network request is a bug.
        network = patch("city_analyser.building_analysis.urlopen", side_effect=AssertionError("network request"))
        network.start()
        self.addCleanup(network.stop)
        self.calls = []

    def provider(self):
        def generate(_images, prompt):
            self.calls.append(prompt)
            return ANSWER

        return VLMProvider(generate, "vlm:test")

    def survey(self, **options):
        return run_survey(self.root / "data", self.root / "models", fixture.DATASET, provider_factory=self.provider, tile_m=20, log=lambda _: None, **options)



class SurveyTests(SurveyWorkspace):
    def test_walks_every_tile_describes_every_building_and_a_rerun_repeats_nothing(self):
        result = self.survey()
        dataset_dir = self.root / "data" / fixture.DATASET
        state = json.loads(survey_state_path(self.root / "data", fixture.DATASET).read_text(encoding="utf-8"))
        self.assertEqual((result["state"], result["tiles"], result["tiles_done"]), ("finished", 2, 2))
        self.assertEqual(sorted(state["buildings"]), ["way/101", "way/102", "way/103", "way/104"])
        self.assertEqual(state["summary"]["steps"], {"prior_written": 4, "vlm_written": 4})
        stored = json.loads((dataset_dir / "attributes" / "way" / "101" / "attributes-vlm_test.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["predictions"]["gem_material"]["scores"]["CR"], max(stored["predictions"]["gem_material"]["scores"].values()))
        asked = len(self.calls)
        self.assertGreater(asked, 0)
        again = self.survey()
        self.assertEqual((again["state"], again["buildings_described"], len(self.calls)), ("finished", 0, asked))
        # A new survey with other settings revisits everything but reuses every model answer.
        forced = self.survey(force=True)
        self.assertEqual((forced["buildings_described"], forced["model_answers"], len(self.calls)), (4, 0, asked))

    def test_small_runs_stop_early_and_the_next_run_continues(self):
        first = self.survey(max_buildings=1)
        self.assertEqual((first["state"], first["buildings_described"]), ("partial", 1))
        rest = self.survey()
        self.assertEqual((rest["state"], rest["buildings_described"], rest["buildings_described_total"]), ("finished", 3, 4))

    def test_buildings_without_an_analyzed_view_wait_for_the_next_run(self):
        # As on Colab with corrupted weights: no panorama is analyzed, so there is nothing to describe yet.
        failed_report = {"state": "finished", "failed": 3}
        with patch("city_analyser.survey.analyze_dataset", return_value=failed_report):
            failed = self.survey(max_buildings=2)
        state = json.loads(survey_state_path(self.root / "data", fixture.DATASET).read_text(encoding="utf-8"))
        self.assertEqual((failed["buildings_described"], state["buildings"], self.calls), (0, {}, []))
        # The next run analyzes them (here: validates the fixture's analyses) and describes every building.
        self.assertEqual(self.survey()["buildings_described"], 4)

    def test_a_stop_request_ends_the_run_as_interrupted(self):
        stop = threading.Event()
        stop.set()
        self.assertEqual(self.survey(stop=stop)["state"], "interrupted")

    def test_prior_only_survey_needs_no_model(self):
        result = run_survey(self.root / "data", self.root / "models", fixture.DATASET, provider_factory=None, tile_m=20, log=lambda _: None)
        self.assertEqual((result["state"], result["model_answers"], self.calls), ("finished", 0, []))

    def test_the_viewer_serves_the_merged_card(self):
        from city_analyser.local_viewer import ViewerServer

        self.survey()
        server = ViewerServer(("127.0.0.1", 0), self.root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=5)))
        base = f"http://127.0.0.1:{server.server_port}/api/datasets/{fixture.DATASET}/building-attributes"
        with urllib.request.urlopen(f"{base}/way/101", timeout=10) as response:
            card = json.load(response)
        with self.assertRaises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"{base}/way/999", timeout=10)
        self.assertEqual(card["building"], "way/101")
        self.assertEqual(card["attributes"]["floors"]["source"], "osm_tag")  # building:levels=3 wins over the model's 4
        self.assertTrue(card["gem"]["string"].startswith("CR/LFINF/HEX:3"))
        self.assertEqual(missing.exception.code, 404)
        missing.exception.close()


class ReviewTests(SurveyWorkspace):
    """Human review on top of a finished survey."""

    def setUp(self):
        super().setUp()
        self.survey()
        self.dataset_dir = self.root / "data" / fixture.DATASET

    def test_a_review_wins_keeps_earlier_attributes_and_can_be_withdrawn(self):
        from city_analyser.attributes.review import ReviewError, review_history, save_review

        card = save_review(self.dataset_dir, fixture.DATASET, "way/101", {"reviewer": "Ana", "note": "thick walls", "values": {"gem_material": "MUR", "gem_llrs": "LWAL"}})
        self.assertEqual(card["gem"]["string"], "MUR/LWAL/HEX:3")
        self.assertEqual((card["attributes"]["gem_material"]["source"], card["review"]["reviewer"]), ("human_review", "Ana"))
        model = next(other for other in card["attributes"]["gem_material"]["others"] if other["provider"] == "vlm:test")
        self.assertEqual(model["value"], "CR")  # the model's answer stays visible
        card = save_review(self.dataset_dir, fixture.DATASET, "way/101", {"reviewer": "Ana", "values": {"floors": None}})
        self.assertEqual(card["gem"]["string"], "MUR/LWAL/H99")  # "cannot tell" overrides the OSM tag and the model
        self.assertTrue(card["attributes"]["floors"]["abstained"])
        for bad in ({"values": {"gem_material": "CR"}}, {"reviewer": "Ana", "values": {"gem_material": "MAT99"}},
                    {"reviewer": "Ana", "values": {"floors": 2.5}}, {"reviewer": "Ana", "values": {"roof": "flat"}}, {"reviewer": "Ana", "action": "delete"}):
            with self.assertRaises(ReviewError):
                save_review(self.dataset_dir, fixture.DATASET, "way/101", bad)
        card = save_review(self.dataset_dir, fixture.DATASET, "way/101", {"reviewer": "Ben", "action": "withdraw"})
        self.assertEqual((card["review"], card["gem"]["string"]), (None, "CR/LFINF/HEX:3"))
        self.assertEqual([entry["action"] for entry in review_history(self.dataset_dir, "way/101")], ["review", "review", "withdraw"])
        with self.assertRaises(ReviewError):
            save_review(self.dataset_dir, fixture.DATASET, "way/999", {"reviewer": "Ana", "values": {"gem_material": "CR"}})

    def test_the_summary_flags_answers_worth_a_reviewer_and_drops_reviewed_ones(self):
        from city_analyser.attributes.review import attribute_summary, attributes_signature, save_review

        before = attributes_signature(self.dataset_dir)
        summary = attribute_summary(self.dataset_dir, fixture.DATASET)
        entry = summary["buildings"]["way/101"]
        self.assertEqual((entry["gem_material"], entry["gem_llrs"], entry["floors"], entry["reviewed"]), ("CR", "LFINF", 3.0, False))
        self.assertIn("model says 4 floors, OSM 3", entry["needs_review"])  # building:levels=3 in the fixture
        self.assertTrue(entry["image"])
        save_review(self.dataset_dir, fixture.DATASET, "way/101", {"reviewer": "Ana", "values": {"floors": 3}})
        self.assertNotEqual(attributes_signature(self.dataset_dir), before)
        after = attribute_summary(self.dataset_dir, fixture.DATASET)
        self.assertTrue(after["buildings"]["way/101"]["reviewed"])
        self.assertIsNone(after["buildings"]["way/101"]["needs_review"])
        self.assertEqual(after["counts"]["reviewed"], 1)

    def test_the_merged_export_takes_what_the_card_shows(self):
        import csv

        from city_analyser.attributes.review import save_review
        from city_analyser.building_table import export_buildings

        save_review(self.dataset_dir, fixture.DATASET, "way/102", {"reviewer": "Ana", "values": {"gem_material": "MCF"}})
        export_buildings(self.root / "data", fixture.DATASET, provider="merged", use_index=False, output_dir=self.root / "export")
        rows = {row["building_id"]: row for row in csv.DictReader((self.root / "export" / "buildings.csv").open(encoding="utf-8"))}
        self.assertEqual((rows["way/102"]["gem_material"], rows["way/102"]["gem_material_kind"]), ("MCF", "review"))
        self.assertEqual((rows["way/101"]["gem_material"], rows["way/101"]["gem_material_kind"]), ("CR", "vote_share"))
        self.assertEqual(rows["way/101"]["floors"], "3.0")  # the OSM tag wins over the model's 4
        self.assertEqual(rows["way/102"]["gem_taxonomy"], "MCF/LFINF/HEX:4")

    def test_the_viewer_serves_the_summary_and_saves_reviews(self):
        from city_analyser.local_viewer import ViewerServer

        server = ViewerServer(("127.0.0.1", 0), self.root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=5)))
        base = f"http://127.0.0.1:{server.server_port}/api/datasets/{fixture.DATASET}/building-attributes"

        def post(path, body):
            request = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.load(response)

        with urllib.request.urlopen(base, timeout=10) as response:
            self.assertEqual(json.load(response)["counts"]["buildings"], 4)
        card = post("/way/102", {"reviewer": "Ana", "values": {"gem_llrs": "LDUAL"}})
        self.assertEqual(card["attributes"]["gem_llrs"]["value"], "LDUAL")
        with urllib.request.urlopen(base, timeout=10) as response:
            self.assertEqual(json.load(response)["counts"]["reviewed"], 1)  # the cached summary saw the new file
        with self.assertRaises(urllib.error.HTTPError) as rejected:
            post("/way/102", {"reviewer": "Ana", "values": {"gem_llrs": "frame"}})
        self.assertEqual(rejected.exception.code, 400)
        rejected.exception.close()


if __name__ == "__main__":
    unittest.main()
