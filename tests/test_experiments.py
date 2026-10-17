import json
import os
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from city_analyser.attributes.review import save_review
from city_analyser.experiments.config import load_config
from city_analyser.experiments.metrics import numeric_scores
from city_analyser.experiments.report import build_report, to_markdown, write_report
from city_analyser.experiments.runner import run_experiment
from city_analyser.groundtruth import load_truth
from city_analyser.groundtruth.catastro import building_type, match, read_buildings

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "e2e"))
import fixture  # noqa: E402
from test_assistant import FakeModelServer  # noqa: E402

GML_HEAD = ('<?xml version="1.0" encoding="UTF-8"?><gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml/3.2" '
            'xmlns:bu-core2d="http://inspire.ec.europa.eu/schemas/bu-core2d/4.0" xmlns:bu-ext2d="http://inspire.jrc.ec.europa.eu/schemas/bu-ext2d/2.0" '
            'xmlns:base="http://inspire.ec.europa.eu/schemas/base/3.3">')


def building(reference, x, y, size, year, use, dwellings):
    ring = f"{x} {y} {x + size} {y} {x + size} {y + size} {x} {y + size} {x} {y}"
    return (f'<gml:featureMember><bu-ext2d:Building gml:id="ES.SDGC.BU.{reference}"><gml:boundedBy><gml:Envelope>'
            f"<gml:lowerCorner>{x} {y}</gml:lowerCorner><gml:upperCorner>{x + size} {y + size}</gml:upperCorner></gml:Envelope></gml:boundedBy>"
            f"<bu-core2d:dateOfConstruction><bu-core2d:DateOfEvent><bu-core2d:beginning>{year}-01-01T00:00:00</bu-core2d:beginning></bu-core2d:DateOfEvent></bu-core2d:dateOfConstruction>"
            f"<bu-core2d:inspireId><base:Identifier><base:localId>{reference}</base:localId></base:Identifier></bu-core2d:inspireId>"
            f"<gml:PolygonPatch><gml:exterior><gml:LinearRing><gml:posList>{ring}</gml:posList></gml:LinearRing></gml:exterior></gml:PolygonPatch>"
            f"<bu-ext2d:currentUse>{use}</bu-ext2d:currentUse><bu-ext2d:numberOfDwellings>{dwellings}</bu-ext2d:numberOfDwellings>"
            f'<bu-ext2d:numberOfFloorsAboveGround xsi:nil="true"></bu-ext2d:numberOfFloorsAboveGround></bu-ext2d:Building></gml:featureMember>')


def part(reference, number, floors):
    return (f'<gml:featureMember><bu-ext2d:BuildingPart gml:id="p"><bu-core2d:inspireId><base:Identifier><base:localId>{reference}_part{number}</base:localId>'
            f"</base:Identifier></bu-core2d:inspireId><bu-ext2d:numberOfFloorsAboveGround>{floors}</bu-ext2d:numberOfFloorsAboveGround></bu-ext2d:BuildingPart></gml:featureMember>")


class CatastroTests(unittest.TestCase):
    def test_reads_buildings_in_the_area_with_storeys_from_their_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "A.ES.SDGC.BU.99999.zip"
            buildings = building("A1", 0, 0, 10, 1905, "1_residential", 6) + building("B2", 20, 0, 10, 1975, "4_2_retail", 0) + building("FAR", 9000, 9000, 10, 2000, "3_industrial", 0)
            parts = part("A1", 1, 5) + part("A1", 2, 2) + part("B2", 1, 3) + part("FAR", 1, 1)
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("A.ES.SDGC.BU.99999.building.gml", GML_HEAD.replace("<gml:FeatureCollection", '<gml:FeatureCollection xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"') + buildings + "</gml:FeatureCollection>")
                handle.writestr("A.ES.SDGC.BU.99999.buildingpart.gml", GML_HEAD + parts + "</gml:FeatureCollection>")
            found = read_buildings(archive, (-5, -5, 100, 100))
        self.assertEqual(sorted(found), ["A1", "B2"])  # FAR is outside the area
        self.assertEqual((found["A1"]["floors"], found["A1"]["construction_year"], found["A1"]["dwellings"]), (5, 1905, 6))
        self.assertEqual(found["B2"]["use"], "4_2_retail")

    def test_matching_is_one_to_one_above_the_overlap_threshold(self):
        from shapely.geometry import box

        cadastre = {"A": {"geometry": box(0, 0, 10, 10)}, "B": {"geometry": box(10, 0, 20, 10)}, "C": {"geometry": box(40, 0, 50, 10)}}
        osm = {"way/1": box(0.5, 0, 10.5, 10), "way/2": box(10, 0, 30, 10), "way/3": box(1, 1, 9, 9)}
        # way/3 also overlaps A but less than way/1 does; way/2 overlaps B with IoU 0.5 exactly.
        self.assertEqual(match(osm, cadastre), [("way/1", "A", 0.9048), ("way/2", "B", 0.5)])

    def test_uses_map_to_classes_and_residential_needs_dwellings(self):
        self.assertEqual(building_type("1_residential", 6), "apartments")
        self.assertEqual(building_type("1_residential", 1), "house")
        self.assertIsNone(building_type("1_residential", None))
        self.assertEqual(building_type("4_1_office", 0), "office")
        self.assertIsNone(building_type("2_agriculture", 0))


class ExperimentTests(unittest.TestCase):
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
        from city_analyser.demo import prepare_demo

        prepare_demo(self.root / "data", fixture.DATASET, log=lambda _: None)
        self.dataset_dir = self.root / "data" / fixture.DATASET
        reference = {"buildings": {
            "way/101": {"floors": 3, "construction_year": 1960, "building_type": "office"},
            "way/103": {"floors": 5, "construction_year": 1900, "building_type": "apartments"},
        }}
        (self.dataset_dir / "groundtruth").mkdir()
        (self.dataset_dir / "groundtruth" / "catastro.json").write_text(json.dumps(reference))
        self.model = FakeModelServer()
        self.addCleanup(self.model.close)

    def config(self, extra_runs=""):
        path = self.root / "experiment.toml"
        path.write_text(f'''name = "check"
dataset = "{fixture.DATASET}"
truth = ["catastro", "evaluation_set", "osm_tags"]

[[runs]]
id = "prior"
kind = "prior"

[[runs]]
id = "model"
kind = "vlm"
url = "{self.model.url}"
model = "fake-vl"

[[runs]]
id = "model-no-region"
kind = "vlm"
url = "{self.model.url}"
model = "fake-vl"
regional_guidance = false
{extra_runs}''')
        return load_config(path)

    def test_truth_sources_and_the_evaluation_set(self):
        save_review(self.dataset_dir, fixture.DATASET, "way/102", {"reviewer": "Ana", "evaluation": True, "values": {"gem_material": "CR", "gem_llrs": "LFINF", "floors": 4}})
        save_review(self.dataset_dir, fixture.DATASET, "way/104", {"reviewer": "Ana", "values": {"floors": 2}})  # a correction, not a label
        self.assertEqual(load_truth(self.dataset_dir, "evaluation_set"), {"way/102": {"gem_material": "CR", "gem_llrs": "LFINF", "floors": 4.0}})
        self.assertEqual(load_truth(self.dataset_dir, "catastro")["way/103"]["floors"], 5)
        self.assertEqual(load_truth(self.dataset_dir, "osm_tags")["way/101"]["floors"], 3.0)
        with self.assertRaises(ValueError):
            load_truth(self.dataset_dir, "census")

    def test_runs_are_isolated_resumable_and_compared(self):
        save_review(self.dataset_dir, fixture.DATASET, "way/102", {"reviewer": "Ana", "evaluation": True, "values": {"gem_material": "MUR", "gem_llrs": "LWAL"}})
        before = sorted(path.name for path in (self.dataset_dir / "attributes").rglob("*.json"))
        config = self.config('\n[[runs]]\nid = "curated"\nkind = "saved"\nprovider = "vlm:simulated-demo"\n')
        summaries = run_experiment(self.root / "data", config, log=lambda _: None)
        self.assertEqual(sorted(path.name for path in (self.dataset_dir / "attributes").rglob("*.json")), before)  # the dataset is untouched
        # Buildings with a reference value: way/101 and way/103 (cadastre and OSM tags), way/102 (evaluation set).
        self.assertEqual(summaries["model"]["statuses"], {"written": 3})
        self.assertEqual(summaries["curated"]["statuses"], {"copied": 3})
        self.assertIn("before about 1940", json.dumps(self.model.requests[0]["body"]))  # the fixture is in Barcelona
        asked = len(self.model.requests)
        again = run_experiment(self.root / "data", config, only={"model"}, log=lambda _: None)
        self.assertEqual((again["model"]["statuses"], len(self.model.requests)), ({"unchanged": 3}, asked))
        report = build_report(self.root / "data", config)
        floors = report["runs"]["model"]["scores"]["catastro"]["floors"]
        self.assertEqual((floors["n"], floors["coverage"], floors["mae"], floors["bias"]), (2, 1.0, 1.0, 0.0))
        structure = report["runs"]["model"]["scores"]["evaluation_set"]["gem_material"]
        self.assertEqual((structure["n"], structure["accuracy"]), (1, 1.0))
        self.assertEqual(report["runs"]["prior"]["scores"]["evaluation_set"]["gem_material"]["coverage"], 0.0)
        self.assertEqual(report["truth"]["catastro"]["common"]["floors"], 2)
        markdown = to_markdown(report)
        self.assertIn("## Against catastro (2 buildings)", markdown)
        self.assertIn("Same 2 buildings", markdown)
        self.assertIn("| model-no-region |", markdown)
        folder = write_report(self.root / "data", config)
        self.assertTrue((folder / "report.html").read_text().startswith("<!doctype html>"))

    def test_without_regional_guidance_the_prompt_has_only_the_general_rules(self):
        from city_analyser.attributes.vlm import build_prompt

        self.assertNotIn("before about 1940", build_prompt("photo", None))
        config = self.config()
        run_experiment(self.root / "data", config, only={"model-no-region"}, log=lambda _: None)
        # The capture context still names the country; the construction history is what the ablation removes.
        self.assertTrue(self.model.requests)
        self.assertTrue(all("before about 1940" not in json.dumps(request["body"]) for request in self.model.requests))

    def test_analysis_settings_apply_during_the_run_and_a_mismatch_is_explained(self):
        import shutil

        shutil.rmtree(self.dataset_dir / "crops", ignore_errors=True)
        config = self.config()
        mismatched = replace(config, analysis={"BUILDING_ANALYSIS_MODEL": "b5"})  # the fixture was analyzed with b0
        with self.assertRaisesRegex(ValueError, r"\[analysis\]"):
            run_experiment(self.root / "data", mismatched, only={"model"}, log=lambda _: None)
        self.assertEqual(os.environ["BUILDING_ANALYSIS_MODEL"], "b0")  # restored afterwards
        self.assertEqual(run_experiment(self.root / "data", config, only={"model"}, log=lambda _: None)["model"]["statuses"], {"written": 2})

    def test_bad_experiment_files_are_refused(self):
        path = self.root / "bad.toml"
        for text in ('dataset = "x"', 'name = "a b"\ndataset = "x"\n[[runs]]\nid = "p"\nkind = "prior"',
                     'name = "a"\ndataset = "x"\n[[runs]]\nid = "m"\nkind = "vlm"', 'name = "a"\ndataset = "x"\ntruth = ["census"]\n[[runs]]\nid = "p"\nkind = "prior"',
                     'name = "a"\ndataset = "x"\n[[runs]]\nid = "p"\nkind = "prior"\n[[runs]]\nid = "p"\nkind = "prior"',
                     'name = "a"\ndataset = "x"\n[analysis]\nHOME = "/tmp"\n[[runs]]\nid = "p"\nkind = "prior"'):
            path.write_text(text)
            with self.assertRaises(ValueError):
                load_config(path)


class ReportTests(unittest.TestCase):
    def test_runs_are_also_compared_on_the_buildings_all_of_them_answered(self):
        from city_analyser.experiments.report import answered, common_buildings

        def floors(value):
            return {"floors": {"value": value, "abstained": value is None}}

        truth = {f"way/{n}": {"floors": 5} for n in range(4)}
        bold = {f"way/{n}": floors(5) for n in range(4)}
        careful = {"way/0": floors(5), "way/1": floors(5), "way/2": floors(None)}
        rare = {"way/3": floors(4)}  # answers a quarter: does not shrink the subset
        sets = [answered(run, "floors") for run in (bold, careful, rare)]
        self.assertEqual(sets[1], {"way/0", "way/1"})
        self.assertEqual(common_buildings(truth, sets, "floors"), {"way/0", "way/1"})


class MetricTests(unittest.TestCase):
    def test_numeric_scores_report_bias_and_tolerance(self):
        scores = numeric_scores([5, 5, 4], [4, 4, None], tolerance=1)
        self.assertEqual((scores["answered"], scores["mae"], scores["bias"], scores["within_tolerance"]), (2, 1.0, -1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
