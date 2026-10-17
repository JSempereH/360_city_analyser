import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from city_analyser.attributes.assistant import chat_messages, record_summary
from city_analyser.attributes.cli import model_settings_from_environment
from city_analyser.attributes.regional import guidance_for, region_at
from city_analyser.attributes.review import ReviewError, save_review
from city_analyser.attributes.vlm import PROMPT, Crop, OpenAICompatibleClient, VLMProvider, build_prompt, visible_reasoning
from city_analyser.results import export_results, import_results

sys.path.insert(0, str(Path(__file__).resolve().parent / "e2e"))
import fixture  # noqa: E402

NOTES = "I count four rows of windows. Openings stack above each other between thick piers, so the walls carry the load."
ANSWER = NOTES + '\n{"observations": "stacked openings between thick piers", "floors": 4, "gem_material": "MUR", "gem_llrs": "LWAL", "building_type": "apartments"}'


class FakeModelServer:
    """An OpenAI-compatible chat endpoint that streams like Ollama or vLLM; records the requests it gets."""

    def __init__(self, reply: str = ANSWER):
        self.requests: list[dict] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append({"path": self.path, "body": body})
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for start in range(0, len(reply), 7):
                        chunk = {"choices": [{"delta": {"content": reply[start:start + 7]}}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                payload = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class RegionalPromptTests(unittest.TestCase):
    def test_regions_by_camera_position(self):
        self.assertEqual(region_at(-0.3758, 39.4758).id, "es")  # Valencia
        self.assertEqual(region_at(-3.7038, 40.4168).id, "es")  # Madrid
        self.assertEqual(region_at(2.65, 39.57).id, "es")  # Palma
        self.assertIsNone(region_at(-9.14, 38.72))  # Lisbon is Portugal
        self.assertIsNone(region_at(2.35, 48.86))
        self.assertIsNone(region_at(None, None))

    def test_the_prompt_carries_the_region_and_asks_for_notes_before_the_json(self):
        spain = build_prompt("photo taken in 2017-06", "es")
        self.assertIn("before about 1940", spain)
        self.assertIn("LFLSINF", spain)
        self.assertIn("soft storey", spain)
        self.assertNotIn("before about 1940", PROMPT)
        self.assertLess(spain.index("plain prose"), spain.index("JSON object"))
        self.assertTrue(spain.endswith("Context: photo taken in 2017-06."))
        self.assertIn("Spain", guidance_for("es"))

    def test_notes_are_kept_apart_from_the_answer(self):
        self.assertEqual(visible_reasoning(ANSWER), NOTES)
        self.assertEqual(visible_reasoning(NOTES + "\n```json\n{"), NOTES)


class ClientAndProviderTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModelServer()
        self.addCleanup(self.model.close)

    def test_streaming_reports_each_view_and_stores_the_notes(self):
        client = OpenAICompatibleClient(self.model.url + "/v1", "fake", options={"reasoning_effort": "none"})
        events = []
        crops = [Crop(b"a", 1.0, {"image_id": "1"}, "photo", "es"), Crop(b"b", 0.5, {"image_id": "2"}, None, None)]
        result = VLMProvider(client, "vlm:fake").predict("way/1", crops, on_event=events.append).to_dict()
        self.assertEqual([event["type"] for event in events if event["type"] != "text"], ["view", "answer", "view", "answer"])
        self.assertEqual("".join(event["text"] for event in events if event["type"] == "text" and event["index"] == 0), ANSWER)
        self.assertEqual(result["evidence"][0]["reasoning"], NOTES)
        self.assertEqual(result["predictions"]["gem_material"]["scores"]["MUR"], 1.0)
        request = self.model.requests[0]
        self.assertEqual(request["path"], "/v1/chat/completions")  # a base URL ending in /v1 is not doubled
        self.assertEqual(request["body"]["reasoning_effort"], "none")
        self.assertTrue(request["body"]["stream"])
        self.assertIn("before about 1940", request["body"]["messages"][0]["content"][1]["text"])
        self.assertNotIn("before about 1940", self.model.requests[1]["body"]["messages"][0]["content"][1]["text"])

    def test_an_empty_answer_says_why(self):
        empty = FakeModelServer(reply="")
        self.addCleanup(empty.close)
        with self.assertRaisesRegex(RuntimeError, "reasoning_effort"):
            OpenAICompatibleClient(empty.url, "fake").chat([{"role": "user", "content": "hi"}])

    def test_settings_come_from_the_environment(self):
        self.assertIsNone(model_settings_from_environment())
        with patch.dict(os.environ, {"ATTRIBUTES_VLM_URL": self.model.url, "ATTRIBUTES_VLM_MODEL": "m", "ATTRIBUTES_VLM_OPTIONS": '{"reasoning_effort": "none"}'}):
            settings = model_settings_from_environment()
        self.assertEqual((settings.backend, settings.provider_name, settings.options), ("openai", "vlm:m", {"reasoning_effort": "none"}))
        with patch.dict(os.environ, {"ATTRIBUTES_VLM_URL": self.model.url, "ATTRIBUTES_VLM_OPTIONS": "[1]"}):
            with self.assertRaises(ValueError):
                model_settings_from_environment()


class ChatContextTests(unittest.TestCase):
    def test_the_model_gets_the_record_the_photos_and_the_conversation(self):
        card = {"building": "way/1", "gem": {"string": "MUR/LWAL/HEX:4"}, "review": {"reviewer": "Ana"},
                "attributes": {"floors": {"value": 4, "source": "osm_tag"}, "gem_llrs": {"abstained": True, "reason": "split", "source": "vlm:m"}},
                "views": [{"reasoning": "Thick walls."}]}
        history = [{"role": "user", "content": "Why masonry?"}, {"role": "assistant", "content": "Thick walls."}, {"role": "user", "content": "Floors?"}]
        messages = chat_messages(card, [b"a", b"b", b"c", b"d"], history, "es")
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertIn('"why_undetermined": "split"', messages[0]["content"])
        self.assertIn("Ana", messages[0]["content"])
        self.assertIn("before about 1940", messages[0]["content"])
        self.assertEqual(sum(part["type"] == "image" for part in messages[1]["content"]), 3)
        with self.assertRaises(ValueError):
            chat_messages(card, [b"a"], [{"role": "assistant", "content": "hi"}], None)
        self.assertEqual(record_summary(None), {"record": "this building has not been described yet"})


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        environment = patch.dict(os.environ, fixture.create_workspace(self.root))
        environment.start()
        self.addCleanup(environment.stop)
        network = patch("city_analyser.building_analysis.urlopen", side_effect=AssertionError("network request"))
        network.start()
        self.addCleanup(network.stop)
        self.dataset_dir = self.root / "data" / fixture.DATASET

    def test_every_attribute_can_be_corrected_by_hand(self):
        from city_analyser.demo import prepare_demo

        prepare_demo(self.root / "data", fixture.DATASET, log=lambda _: None)
        card = save_review(self.dataset_dir, fixture.DATASET, "way/101", {"reviewer": "Ana", "values": {
            "building_type": "office", "surface_material": "stone", "construction_year": 1912, "floors": 3}})
        self.assertEqual(card["attributes"]["building_type"]["value"], "office")
        self.assertEqual(card["attributes"]["construction_year"]["value"], 1912.0)
        self.assertEqual(card["attributes"]["surface_material"]["source"], "human_review")
        self.assertEqual(card["classes"]["building_type"][0], "apartments")
        for bad in ({"construction_year": 1912.5}, {"construction_year": 3000}, {"building_type": "castle"}):
            with self.assertRaises(ReviewError):
                save_review(self.dataset_dir, fixture.DATASET, "way/101", {"reviewer": "Ana", "values": bad})

    def test_results_travel_without_overwriting_local_reviews(self):
        from city_analyser.demo import prepare_demo

        prepare_demo(self.root / "data", fixture.DATASET, log=lambda _: None)
        archive = self.root / "results.zip"
        summary = export_results(self.root / "data", fixture.DATASET, archive)
        self.assertGreater(summary["attributes"], 0)
        self.assertEqual(summary["settings"]["BUILDING_ANALYSIS_MODEL"], "b0")
        self.assertNotIn("BUILDING_ANALYSIS_OSM_CACHE_DIR", summary["settings"])
        # The laptop has reviewed a building since: importing the Colab run keeps that decision.
        save_review(self.dataset_dir, fixture.DATASET, "way/101", {"reviewer": "Ana", "values": {"floors": 5}})
        result = import_results(archive, self.root / "data")
        self.assertTrue(result["images_present"])
        review = json.loads((self.dataset_dir / "attributes" / "way" / "101" / "attributes-human-review.json").read_text())
        self.assertEqual(review["predictions"]["floors"]["value"], 5.0)
        # A crafted archive cannot write outside the dataset.
        bad = self.root / "bad.zip"
        with zipfile.ZipFile(bad, "w") as handle:
            handle.writestr("results-manifest.json", json.dumps({"dataset": "sample", "settings": {}}))
            handle.writestr("sample/../../escape.txt", "x")
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            import_results(bad, self.root / "data")
        self.assertFalse((self.root / "escape.txt").exists())

    def test_the_viewer_analyzes_and_chats_in_real_time(self):
        from city_analyser.demo import prepare_demo
        from city_analyser.local_viewer import ViewerServer

        prepare_demo(self.root / "data", fixture.DATASET, log=lambda _: None)
        model = FakeModelServer()
        self.addCleanup(model.close)
        with patch.dict(os.environ, {"ATTRIBUTES_VLM_URL": model.url, "ATTRIBUTES_VLM_MODEL": "fake-vl"}):
            server = ViewerServer(("127.0.0.1", 0), self.root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=5)))
        base = f"http://127.0.0.1:{server.server_port}"

        def events(path, body):
            request = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=30) as response:
                text = response.read().decode()
            return [json.loads(block[5:]) for block in text.split("\n\n") if block.startswith("data:")]

        with urllib.request.urlopen(base + "/api/assistant", timeout=10) as response:
            self.assertEqual(json.load(response)["model"], "fake-vl")
        analysis = events(f"/api/datasets/{fixture.DATASET}/building-attributes/way/103/analyze", {})
        kinds = [event["type"] for event in analysis]
        self.assertEqual((kinds[0], kinds[-1]), ("start", "done"))
        self.assertIn("text", kinds)
        self.assertTrue(analysis[1]["reference"]["view"]["yaw_deg"] is not None)
        card = analysis[-1]["card"]
        self.assertEqual(card["model"], "vlm:fake-vl")  # the newest answer is the one shown
        self.assertEqual(card["views"][0]["reasoning"], NOTES)
        chat = events(f"/api/datasets/{fixture.DATASET}/building-attributes/way/103/chat", {"messages": [{"role": "user", "content": "Why masonry?"}]})
        self.assertEqual(chat[-1]["type"], "done")
        self.assertEqual("".join(event["text"] for event in chat if event["type"] == "text"), ANSWER)
        system = model.requests[-1]["body"]["messages"][0]["content"]
        self.assertIn("way/103", system)
        bad = events(f"/api/datasets/{fixture.DATASET}/building-attributes/way/103/chat", {"messages": []})
        self.assertEqual(bad[-1]["type"], "error")

    def test_without_a_model_the_viewer_says_how_to_configure_one(self):
        from city_analyser.local_viewer import ViewerServer

        server = ViewerServer(("127.0.0.1", 0), self.root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=5)))
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/assistant", timeout=10) as response:
            status = json.load(response)
        self.assertFalse(status["available"])
        self.assertIn("ATTRIBUTES_VLM_URL", status["help"])
        request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/datasets/{fixture.DATASET}/building-attributes/way/103/chat",
                                         data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(urllib.error.HTTPError) as unavailable:
            urllib.request.urlopen(request, timeout=10)
        self.assertEqual(unavailable.exception.code, 503)
        unavailable.exception.close()


if __name__ == "__main__":
    unittest.main()
