import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from city_analyser.analysis_cache import analysis_input_sha256, canonical_json_sha256
from city_analyser.building_analysis import ANALYSIS_VERSION, analysis_configuration, publish_analysis_artifacts
from city_analyser.multi_view_analysis import MULTI_VIEW_ANALYSIS_VERSION, _selection_sha256, fuse_nearby_analyses, load_nearby_analysis, nearby_analysis_path, nearby_panoramas

Image = __import__("PIL.Image", fromlist=["Image"])


def image(identifier, longitude, latitude):
    return {
        "id": identifier,
        "is_pano": True,
        "local_file": f"images/{identifier}.jpg",
        "computed_compass_angle_deg": 10,
        "computed_geometry": {"type": "Point", "coordinates": [longitude, latitude]},
    }


class MultiViewAnalysisTests(unittest.TestCase):
    def test_nearby_cache_tracks_its_analysis_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "sample"
            (dataset_dir / "images").mkdir(parents=True)
            current = image("1", 4.9, 52.3)
            current["computed_compass_angle_deg"] = 10
            Image.fromarray(np.zeros((256, 512, 3), dtype=np.uint8)).save(dataset_dir / current["local_file"])
            (dataset_dir / "manifest.json").write_text(json.dumps({"images": [current]}), encoding="utf-8")
            analysis = {
                "version": ANALYSIS_VERSION, "image_id": "1", "building_ids": {}, "width": 2, "height": 2,
                "sfm_input_sha256": None,
                "input_sha256": analysis_input_sha256(root, "sample", current, analysis_configuration()),
            }
            import io
            output = io.BytesIO()
            Image.fromarray(np.zeros((2, 2), dtype=np.uint16)).save(output, format="PNG")
            analysis = publish_analysis_artifacts(root, "sample", "1", analysis, output.getvalue(), output.getvalue())
            analysis_dir = dataset_dir / "analysis"
            nearby = {
                "version": MULTI_VIEW_ANALYSIS_VERSION,
                "analysis_version": ANALYSIS_VERSION,
                "image_id": "1",
                "selection_sha256": _selection_sha256([(current, 0.0)]),
                "analysis_inputs": [{"image_id": "1", "sha256": canonical_json_sha256(analysis)}],
            }
            nearby_analysis_path(root, "sample", "1").write_text(json.dumps(nearby), encoding="utf-8")

            self.assertEqual(load_nearby_analysis(root, "sample", "1"), nearby)
            headingless = image("2", 4.9001, 52.3)
            headingless.pop("computed_compass_angle_deg")
            (dataset_dir / "manifest.json").write_text(json.dumps({"images": [current, headingless]}), encoding="utf-8")
            self.assertEqual(load_nearby_analysis(root, "sample", "1"), nearby)
            analysis["building_ids"] = {"1": {"osm_id": "way/9"}}
            analysis["building_ids"] = {"1": {"osm_id": "way/9"}}
            (analysis_dir / "1.json").write_text(json.dumps(analysis), encoding="utf-8")
            self.assertIsNone(load_nearby_analysis(root, "sample", "1"))

    def test_nearby_cache_is_invalidated_when_the_selected_view_set_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "sample"
            dataset_dir.mkdir()
            current = image("1", 4.9, 52.3)
            manifest = {"images": [current]}
            (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            nearby = {
                "version": MULTI_VIEW_ANALYSIS_VERSION,
                "analysis_version": ANALYSIS_VERSION,
                "image_id": "1",
                "selection_sha256": _selection_sha256([(current, 0.0)]),
                "analysis_inputs": [{"image_id": "missing", "sha256": "unused"}],
            }
            path = nearby_analysis_path(root, "sample", "1")
            path.parent.mkdir()
            path.write_text(json.dumps(nearby), encoding="utf-8")
            manifest["images"].append(image("2", 4.9001, 52.3))
            (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            self.assertIsNone(load_nearby_analysis(root, "sample", "1"))

    def test_nearby_selection_keeps_current_and_sorts_by_distance(self):
        current = image("1", 4.9, 52.3)
        close = image("2", 4.9001, 52.3)
        farther = image("3", 4.9003, 52.3)
        distant = image("4", 4.91, 52.3)

        selected = nearby_panoramas([distant, farther, current, close], current)

        self.assertEqual([item[0]["id"] for item in selected], ["1", "2", "3"])
        self.assertEqual(selected[0][1], 0.0)

    def test_fusion_combines_building_evidence_from_multiple_views(self):
        current = image("1", 4.9, 52.3)
        neighbor = image("2", 4.9001, 52.3)
        result = fuse_nearby_analyses("1", [
            (current, 0.0, {"building_ids": {"1": {"osm_id": "way/9", "pixels": 100, "inferred_pixels": 20, "confidence": 0.8, "footprint_source": "osm"}}}),
            (neighbor, 8.0, {"building_ids": {"1": {"osm_id": "way/9", "pixels": 300, "inferred_pixels": 0, "confidence": 0.6, "footprint_source": "osm"}}}),
        ])

        building = result["building_ids"][0]
        self.assertEqual(building["osm_id"], "way/9")
        self.assertEqual(building["views"], 2)
        self.assertEqual(building["observed_pixels"], 400)
        self.assertEqual(building["inferred_pixels"], 20)
        self.assertEqual(building["footprint_sources"], ["osm"])
        self.assertEqual(building["confidence"], 0.7)

    def test_duplicate_capture_positions_do_not_count_as_independent_views(self):
        first = image("1", 4.9, 52.3)
        duplicate = image("2", 4.900001, 52.3)

        result = fuse_nearby_analyses("1", [
            (first, 0.0, {"building_ids": {"1": {"osm_id": "way/9", "pixels": 100, "confidence": 0.8}}}),
            (duplicate, 0.1, {"building_ids": {"1": {"osm_id": "way/9", "pixels": 100, "confidence": 0.8}}}),
        ])

        self.assertEqual(result["building_ids"][0]["views"], 1)
        self.assertEqual(result["building_ids"][0]["panorama_ids"], ["1", "2"])

    def test_independent_view_selection_finds_the_largest_subset(self):
        middle = image("1", 0, 0)
        west = image("2", -0.00002, 0)
        east = image("3", 0.00002, 0)
        analysis = {"building_ids": {"1": {"osm_id": "way/9", "pixels": 100, "confidence": 0.8}}}

        result = fuse_nearby_analyses("1", [(middle, 0, analysis), (west, 2.2, analysis), (east, 2.2, analysis)])

        self.assertEqual(result["building_ids"][0]["views"], 2)
        self.assertEqual(result["building_ids"][0]["independent_panorama_ids"], ["2", "3"])
