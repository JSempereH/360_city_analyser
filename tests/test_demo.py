import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from city_analyser.attributes.card import building_card
from city_analyser.demo import SIMULATED_PROVIDER, copy_dataset, prepare_demo, simulated_answers

sys.path.insert(0, str(Path(__file__).resolve().parent / "e2e"))
import fixture  # noqa: E402


class SimulatedAnswerTests(unittest.TestCase):
    def test_answers_are_deterministic_plausible_and_follow_the_tags(self):
        first = simulated_answers("way/1", 3, {"building:levels": "6", "start_date": "1965"}, None)
        self.assertEqual(first, simulated_answers("way/1", 3, {"building:levels": "6", "start_date": "1965"}, None))
        self.assertEqual(len(first), 3)
        self.assertTrue(all(abs(answer["floors"] - 6) <= 1 for answer in first))
        self.assertEqual(simulated_answers("way/2", 1, {"start_date": "1890"}, 12.0)[0]["gem_material"], "MUR")
        self.assertEqual(simulated_answers("way/3", 1, {"building": "industrial"}, 9.0)[0]["gem_llrs"], "LFM")
        self.assertEqual(simulated_answers("way/4", 1, {}, 15.5)[0]["floors"], 5)
        # A 51 m bell tower has no countable storeys: the simulated model abstains instead of answering 16.
        tower = simulated_answers("way/5", 2, {"name": "el Micalet", "height": "51"}, 51.0)
        self.assertEqual([answer["floors"] for answer in tower], [None, None])
        self.assertEqual(tower[0]["gem_material"], "MUR")
        # Across many buildings some views disagree, so the presentation shows flags too.
        splits = sum(len({answer["gem_llrs"] for answer in simulated_answers(f"way/{n}", 3, {"start_date": "1970"}, None)}) > 1 for n in range(200))
        self.assertTrue(10 < splits < 120)


class PrepareDemoTests(unittest.TestCase):
    def test_a_copy_gets_crops_priors_and_simulated_answers_and_the_source_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, fixture.create_workspace(root)), \
                    patch("city_analyser.building_analysis.urlopen", side_effect=AssertionError("network request")):
                copy_dataset(root / "data", fixture.DATASET, "sample-demo")
                result = prepare_demo(root / "data", "sample-demo", log=lambda _: None)
                card = building_card(root / "data" / "sample-demo", "sample-demo", "way/101")
            self.assertFalse((root / "data" / fixture.DATASET / "attributes").exists())
            image = next((root / "data" / "sample-demo" / "images").iterdir())
            self.assertEqual(image.stat().st_ino, (root / "data" / fixture.DATASET / "images" / image.name).stat().st_ino)
        # A building seen only as a sliver of facade abstains instead of receiving an invented answer.
        self.assertEqual((result["buildings"], result["simulated"] + result.get("sliver", 0)), (4, 4))
        self.assertEqual(card["model"], SIMULATED_PROVIDER)
        self.assertTrue(card["views"][0]["view"]["fov_deg"] > 0)
        self.assertTrue(card["views"][0]["answer"]["observations"])
        json.dumps(card)


if __name__ == "__main__":
    unittest.main()
