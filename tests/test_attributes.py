import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from city_analyser.attributes.card import building_card
from city_analyser.attributes.cli import attributes_path, capture_context, load_crops, osm_tags, predict_vlm, write_attributes
from city_analyser.attributes.fusion import fuse_categorical, fuse_numeric
from city_analyser.attributes.prior import osm_prior
from city_analyser.attributes.taxonomy import BUILDING_TYPES, GEM_LLRS, GEM_MATERIALS, floors_from_osm, gem_taxonomy_string, material_from_osm, type_from_osm, year_from_osm
from city_analyser.attributes.vlm import PROMPT, Crop, OpenAICompatibleClient, VLMProvider, build_prompt, inputs_fingerprint, parse_answer, view_votes
from city_analyser.building_inspection.crops import crops_directory
from city_analyser.building_inspection.crops import export_building_crops
from evaluation import attributes as scoring
from tests.test_building_inspection import BuildingFixture


class TaxonomyAndPriorTests(unittest.TestCase):
    def test_tag_parsing(self):
        self.assertEqual(type_from_osm("Church"), "religious")
        self.assertIsNone(type_from_osm("yes"))
        self.assertIsNone(type_from_osm("residential"))
        self.assertEqual(material_from_osm("reinforced_concrete"), "concrete")
        self.assertEqual(floors_from_osm("4,5"), 4.5)
        self.assertIsNone(floors_from_osm("many"))
        for value, year in (("1905", 1905), ("1905-03-01", 1905), ("~1900", 1900), ("1900s", 1900), ("C19", None), (None, None)):
            self.assertEqual(year_from_osm(value), year)

    def test_prior_reads_tags_and_abstains_without_them(self):
        prior = osm_prior("way/1", {"building": "apartments", "building:levels": "5", "start_date": "1932"}).to_dict()["predictions"]
        self.assertEqual(prior["building_type"]["scores"]["apartments"], 1.0)
        self.assertEqual(prior["building_type"]["score_kind"], "tag")
        self.assertEqual(prior["floors"]["value"], 5.0)
        self.assertEqual(prior["construction_year"]["value"], 1932.0)
        self.assertTrue(prior["surface_material"]["abstained"])
        self.assertTrue(osm_prior("way/2", {"building": "yes"}).to_dict()["predictions"]["building_type"]["abstained"])


class GemTaxonomyTests(unittest.TestCase):
    def test_answers_use_gem_v3_codes_and_unknown_codes_abstain(self):
        answer = parse_answer('{"observations": "  Columns and slab edges\nat each storey. ", "gem_material": "cr", "alternate_gem_material": "MCF", '
                              '"gem_llrs": "lfinf", "alternate_gem_llrs": "LFM", "floors": "5"}')
        self.assertEqual((answer["gem_material"], answer["alternate_gem_material"], answer["gem_llrs"], answer["alternate_gem_llrs"]), ("CR", "MCF", "LFINF", "LFM"))
        self.assertEqual((answer["floors"], answer["observations"]), (5.0, "Columns and slab edges at each storey."))
        unknown = parse_answer('{"gem_material": "MAT99", "gem_llrs": "L99", "floors": 0}')
        self.assertEqual((unknown["gem_material"], unknown["gem_llrs"], unknown["floors"]), (None, None, None))
        for code in (*GEM_MATERIALS, *GEM_LLRS):
            self.assertIn(f"{code} = ", PROMPT)

    def test_the_answer_after_any_reasoning_is_the_one_kept(self):
        text = 'Thinking: maybe {"gem_material": "W"}... no, the walls are thick. {"gem_material": "MUR", "gem_llrs": "LWAL"}'
        self.assertEqual(parse_answer(text)["gem_material"], "MUR")

    def test_gem_string(self):
        self.assertEqual(gem_taxonomy_string("CR", "LFINF", 5.0), "CR/LFINF/HEX:5")
        self.assertEqual(gem_taxonomy_string(None, "LWAL", None), "MAT99/LWAL/H99")

    def test_the_prior_never_claims_a_structure(self):
        prior = osm_prior("way/1", {"building": "apartments", "building:material": "brick"}).to_dict()["predictions"]
        self.assertTrue(prior["gem_material"]["abstained"])
        self.assertTrue(prior["gem_llrs"]["abstained"])

    def test_context_is_appended_to_the_prompt(self):
        self.assertEqual(build_prompt(None), PROMPT)
        self.assertTrue(build_prompt("photo taken in 2024-05").endswith("Context: photo taken in 2024-05."))
        image = {"computed_geometry": {"type": "Point", "coordinates": [-0.37, 39.47]}, "captured_at_ms": 1_717_200_000_000}
        self.assertEqual(capture_context(image), "photo taken in Spain at latitude 39.4700, longitude -0.3700 in 2024-06")
        self.assertIsNone(capture_context({}))


class FusionTests(unittest.TestCase):
    def test_weighted_pooling_keeps_the_distribution_and_abstains_when_split(self):
        views = [({"house": 1.0}, 3.0), ({"office": 1.0}, 1.0)]
        pooled = fuse_categorical("building_type", BUILDING_TYPES, views, "t")
        self.assertFalse(pooled.abstained)
        self.assertAlmostEqual(pooled.scores["house"], 0.75)
        split = fuse_categorical("building_type", BUILDING_TYPES, [({"house": 1.0}, 1.0), ({"office": 1.0}, 1.0)], "t", min_share=0.6)
        self.assertTrue(split.abstained)
        self.assertEqual(len(split.scores), len(BUILDING_TYPES))  # the evidence stays visible

    def test_abstains_without_enough_usable_views(self):
        self.assertTrue(fuse_categorical("building_type", BUILDING_TYPES, [({}, 1.0), ({"house": 1.0}, 0.0)], "t").abstained)
        self.assertTrue(fuse_categorical("building_type", BUILDING_TYPES, [({"house": 1.0}, 1.0)], "t", min_views=2).abstained)
        self.assertTrue(fuse_numeric("floors", [(None, 1.0)], "t").abstained)

    def test_numeric_weighted_median_and_spread(self):
        prediction = fuse_numeric("floors", [(4.0, 1.0), (5.0, 1.0), (30.0, 0.2)], "t")
        self.assertEqual(prediction.value, 5.0)
        self.assertLess(prediction.spread, 2.0)


class VLMTests(unittest.TestCase):
    def test_answer_validation(self):
        good = parse_answer('Sure! {"building_type": "Apartments", "alternate_building_type": "office", "surface_material": "brick", '
                            '"alternate_surface_material": "wood", "floors": 6, "construction_year": 1925}')
        self.assertEqual((good["building_type"], good["floors"], good["construction_year"]), ("apartments", 6.0, 1925.0))
        bad = parse_answer('{"building_type": "castle", "floors": -3, "construction_year": 12, "surface_material": null}')
        self.assertEqual((bad["building_type"], bad["floors"], bad["construction_year"], bad["surface_material"]), (None, None, None, None))
        self.assertEqual(parse_answer("no json at all")["building_type"], None)
        self.assertIn("building_type", PROMPT)

    def test_votes_weight_the_alternate_less(self):
        votes = view_votes({"building_type": "house", "alternate_building_type": "office"}, "building_type")
        self.assertAlmostEqual(votes["house"], 2 / 3)
        self.assertAlmostEqual(votes["office"], 1 / 3)

    def test_provider_pools_views_by_weight(self):
        answers = iter([
            '{"building_type": "apartments", "alternate_building_type": "house", "surface_material": "brick", "floors": 5, "construction_year": 1930}',
            '{"building_type": "apartments", "alternate_building_type": "office", "surface_material": "plaster", "floors": 6, "construction_year": 1935}',
            '{"building_type": "office", "surface_material": "glass", "floors": 20, "construction_year": 2010}',
        ])
        provider = VLMProvider(lambda images, prompt: next(answers), "vlm:test")
        crops = [Crop(b"a", 3.0, {"image_id": "1"}), Crop(b"b", 2.0, {"image_id": "2"}), Crop(b"c", 0.5, {"image_id": "3"})]
        result = provider.predict("way/9", crops).to_dict()
        self.assertEqual(result["predictions"]["building_type"]["scores"]["apartments"], max(result["predictions"]["building_type"]["scores"].values()))
        self.assertFalse(result["predictions"]["building_type"]["abstained"])
        self.assertEqual(result["predictions"]["floors"]["value"], 5.0)
        self.assertEqual([item["image_id"] for item in result["evidence"]], ["1", "2", "3"])
        self.assertEqual(result["predictions"]["building_type"]["score_kind"], "vote_share")

    def test_a_failing_view_is_recorded_and_the_others_still_count(self):
        def generate(images, prompt):
            if images[0] == b"bad":
                raise RuntimeError("Model server failed (503): busy")
            return '{"observations": "Thick walls.", "gem_material": "MUR", "gem_llrs": "LWAL", "floors": 3}'

        provider = VLMProvider(generate, "vlm:test", retry_pause_s=0)
        crops = [Crop(b"bad", 2.0, {"image_id": "1"}, "photo taken in 2020-01"), Crop(b"good", 1.0, {"image_id": "2"})]
        result = provider.predict("way/9", crops).to_dict()
        self.assertEqual(result["predictions"]["gem_material"]["scores"]["MUR"], 1.0)
        self.assertIn("503", result["evidence"][0]["error"])
        self.assertEqual(result["evidence"][1]["answer"]["observations"], "Thick walls.")
        self.assertEqual(result["run"]["inputs_sha256"], inputs_fingerprint("vlm:test", crops))
        self.assertNotEqual(inputs_fingerprint("vlm:test", crops), inputs_fingerprint("vlm:test", crops[:1]))

    def test_views_go_in_one_batch_when_the_model_supports_it(self):
        class Batched:
            def __init__(self):
                self.batches = []

            def __call__(self, images, prompt):
                raise AssertionError("asked one by one")

            def batch(self, requests):
                self.batches.append(len(requests))
                return ['{"gem_llrs": "LFINF"}'] * len(requests)

        model = Batched()
        result = VLMProvider(model, "vlm:test").predict("way/9", [Crop(b"a", 1.0, {}), Crop(b"b", 1.0, {})])
        self.assertEqual(model.batches, [2])
        self.assertEqual(result.predictions["gem_llrs"].top()[0][0], "LFINF")

    def test_client_posts_images_and_prompt_and_reads_the_answer(self):
        received = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                received["path"] = self.path
                received["auth"] = self.headers.get("Authorization")
                received["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                body = json.dumps({"choices": [{"message": {"content": "{\"floors\": 3}"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = OpenAICompatibleClient(f"http://127.0.0.1:{server.server_address[1]}", "m", "key")
            text = client([b"jpegbytes"], "question")
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(text, '{"floors": 3}')
        self.assertEqual(received["path"], "/v1/chat/completions")
        self.assertEqual(received["auth"], "Bearer key")
        content = received["body"]["messages"][0]["content"]
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(content[1]["text"], "question")
        with self.assertRaises(ValueError):
            OpenAICompatibleClient("ftp://x", "m")
        with self.assertRaisesRegex(RuntimeError, "no usable answer"):
            OpenAICompatibleClient("http://127.0.0.1:1", "m", timeout_seconds=1)([b"x"], "q")


class ScoringTests(unittest.TestCase):
    def test_categorical_scores_report_coverage_and_baseline(self):
        scores = scoring.categorical_scores(["a", "a", "b", "b"], ["a", "b", None, "b"], ["a", "b"])
        self.assertEqual((scores["n"], scores["answered"], scores["coverage"]), (4, 3, 0.75))
        self.assertAlmostEqual(scores["accuracy"], 2 / 3, places=3)
        self.assertEqual(scores["majority_baseline"], 0.6667)
        self.assertEqual(scoring.categorical_scores(["a"], [None], ["a"])["answered"], 0)

    def test_numeric_scores(self):
        scores = scoring.numeric_scores([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, None])
        self.assertEqual((scores["answered"], scores["mae"], scores["r2"]), (3, 0.0, 1.0))

    def test_risk_coverage_improves_on_the_confident_answers(self):
        truth = ["a", "a", "b", "b"]
        predicted = [{"a": 0.9, "b": 0.1}, {"a": 0.8, "b": 0.2}, {"a": 0.55, "b": 0.45}, {"a": 0.6, "b": 0.4}]
        curve = scoring.risk_coverage(truth, predicted)
        self.assertEqual(curve[0]["accuracy"], 1.0)
        self.assertEqual(curve[-1]["accuracy"], 0.5)

    def test_evaluate_ignores_buildings_without_truth(self):
        prior = {"a": osm_prior("way/1", {"building": "house"}).to_dict()["predictions"], "b": osm_prior("way/2", {}).to_dict()["predictions"]}
        report = scoring.evaluate(prior, {"a": {"building_type": "house"}})
        self.assertEqual(report["building_type"]["accuracy"], 1.0)
        self.assertNotIn("floors", report)


class DatasetIntegrationTests(BuildingFixture, unittest.TestCase):
    def test_tags_and_crops_feed_the_provider_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, manifest = self.create_dataset(root)
            feature = {"type": "Feature", "properties": {"osm_id": "way/42", "building": "retail", "building:levels": "3"}, "geometry": {}}
            with patch("city_analyser.attributes.cli.nearby_buildings", return_value=[feature]):
                tags = osm_tags(dataset_dir, manifest, "way/42")
            export_building_crops(root, "sample", ["way/42"], size=32, per_building=2)
            crops = load_crops(dataset_dir.resolve(), "way/42")
            result = VLMProvider(lambda images, prompt: '{"building_type": "retail", "floors": 3}', "vlm:t").predict("way/42", crops)
            path = attributes_path(dataset_dir, "way/42", "osm-tag-prior")
        self.assertEqual(tags["building"], "retail")
        self.assertEqual([crop.reference["rank"] for crop in crops], [1, 2])
        self.assertGreater(crops[0].weight, 0)
        self.assertEqual(result.predictions["floors"].value, 3.0)
        self.assertEqual(path.name, "attributes-osm-tag-prior.json")

    def test_tags_are_looked_up_at_the_camera_before_the_centroid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, manifest = self.create_dataset(root)
            feature = {"type": "Feature", "properties": {"osm_id": "way/42", "building": "retail"}, "geometry": {}}
            asked = []
            with patch("city_analyser.attributes.cli.nearby_buildings", side_effect=lambda lat, lon: asked.append((lat, lon)) or [feature]):
                osm_tags(dataset_dir, manifest, "way/42")
        self.assertEqual(len(asked), 1)
        self.assertAlmostEqual(asked[0][0], 52.3)

    def test_crops_of_an_earlier_export_do_not_survive_a_new_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, _ = self.create_dataset(root)
            export_building_crops(root, "sample", ["way/42"], size=32, per_building=2)
            export_building_crops(root, "sample", ["way/42"], size=32, per_building=1)
            names = sorted(path.name for path in crops_directory(dataset_dir.resolve(), "way/42").iterdir())
        self.assertEqual(len(names), 2)
        self.assertTrue(all(name.startswith("crop-01-") for name in names))

    def test_an_unchanged_building_is_not_asked_again(self):
        calls = []

        def generate(images, prompt):
            calls.append(prompt)
            return '{"gem_material": "CR", "gem_llrs": "LFINF", "floors": 4}'

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, manifest = self.create_dataset(root)
            dataset_dir = dataset_dir.resolve()
            images = {image["id"]: image for image in manifest["images"]}
            export_building_crops(root, "sample", ["way/42"], size=32, per_building=2)
            provider = VLMProvider(generate, "vlm:test")
            first, _ = predict_vlm(provider, dataset_dir, "way/42", images)
            second, _ = predict_vlm(provider, dataset_dir, "way/42", images)
            export_building_crops(root, "sample", ["way/42"], size=32, per_building=1)
            third, _ = predict_vlm(provider, dataset_dir, "way/42", images)
            stored = json.loads(attributes_path(dataset_dir, "way/42", "vlm:test").read_text(encoding="utf-8"))
        self.assertEqual((first, second, third), ("written", "unchanged", "written"))
        self.assertEqual(len(calls), 3)  # two views, then one
        self.assertIn("Context: photo taken at latitude 52.3000", calls[0])
        self.assertTrue(stored["evidence"][0]["crop_file"].startswith("crops/way/42/crop-01-"))


class CardTests(unittest.TestCase):
    def test_a_tag_wins_the_model_is_kept_and_abstentions_say_why(self):
        from city_analyser.attributes.models import BuildingAttributes

        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = Path(directory)
            self.assertIsNone(building_card(dataset_dir, "city", "way/7"))
            write_attributes(dataset_dir, osm_prior("way/7", {"building": "apartments", "building:levels": "5"}))
            answers = iter([
                '{"observations": "Infilled concrete frame.", "gem_material": "CR", "gem_llrs": "LFINF", "floors": 4, "building_type": "office"}',
                '{"observations": "Infilled concrete frame.", "gem_material": "CR", "gem_llrs": "LDUAL", "floors": 4}',
            ])
            crops = [Crop(b"a", 1.0, {"image_id": "1", "rank": 1, "crop_file": "crops/way/7/crop-01-1.jpg"}), Crop(b"b", 1.0, {"image_id": "2", "rank": 2})]
            vlm: BuildingAttributes = VLMProvider(lambda images, prompt: next(answers), "vlm:test", min_share=0.6).predict("way/7", crops)
            write_attributes(dataset_dir, vlm)
            card = building_card(dataset_dir, "my city", "way/7")
        attributes = card["attributes"]
        self.assertEqual((attributes["floors"]["value"], attributes["floors"]["source"]), (5.0, "osm_tag"))
        self.assertEqual(attributes["floors"]["others"][0]["value"], 4.0)
        self.assertEqual(attributes["building_type"]["value"], "apartments")
        self.assertEqual((attributes["gem_material"]["value"], attributes["gem_material"]["meaning"]), ("CR", "reinforced concrete"))
        self.assertTrue(attributes["gem_llrs"]["abstained"])  # split between LFINF and LDUAL
        self.assertIn(attributes["gem_llrs"]["leaning"], {"LFINF", "LDUAL"})
        self.assertEqual((card["gem"]["string"], card["gem"]["complete"]), ("CR/L99/HEX:5", False))
        self.assertEqual(card["observations"][0]["crop_url"], "/data/my%20city/crops/way/7/crop-01-1.jpg")
        self.assertEqual(len(card["observations"]), 2)


if __name__ == "__main__":
    unittest.main()
