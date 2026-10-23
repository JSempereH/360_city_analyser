import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from city_analyser.mask_quality import assess, camera_inside_footprint, evaluate, labels_from_curation, view_signals

HEIGHT, WIDTH = 180, 360  # one pixel per degree


def scene(columns: slice, rows: slice = slice(60, 90), distance: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """A building whose walls project onto ``columns`` (rows ``rows``) at ``distance`` meters."""
    owners = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    owners[rows, columns] = 1
    depth = np.where(owners == 1, distance, np.inf).astype(np.float32)
    return owners, depth


class ViewSignalTests(unittest.TestCase):
    def reasons(self, ids: np.ndarray, owners: np.ndarray, depth: np.ndarray, **options) -> list[str]:
        return assess(view_signals(ids, 1, owners, depth, height_m=options.pop("height_m", 20.0), **options))["reasons"]

    def test_a_mask_that_matches_its_projection_is_clean(self):
        owners, depth = scene(slice(100, 140))
        self.assertEqual(self.reasons(owners.copy(), owners, depth), [])

    def test_each_failure_is_named(self):
        owners, depth = scene(slice(100, 140))
        sliver = np.zeros_like(owners)
        sliver[80:85, 110:112] = 1
        self.assertIn("sliver", self.reasons(sliver, owners, depth))
        neighbor = owners.copy()
        neighbor[60:90, 140:180] = 1
        self.assertIn("neighbor", self.reasons(neighbor, owners, depth))
        # Two buildings across a street: a wide gap the walls do not project across.
        street_owners, street_depth = scene(slice(100, 120))
        street_owners[60:90, 140:160] = 1
        street_depth[street_owners == 1] = 20.0
        street = street_owners.copy()
        self.assertIn("street", self.reasons(street, street_owners, street_depth))
        # 20 m away a 10 m building's roof (with margin) is about 37 degrees up; the mask reaches 70.
        sky = owners.copy()
        sky[20:90, 100:140] = 1
        self.assertIn("sky", self.reasons(sky, owners, depth, height_m=10.0))
        speckle = np.zeros_like(owners)
        speckle[60:90:4, 100:140:4] = 1
        self.assertIn("speckle", self.reasons(speckle, owners, depth))
        overhead = np.zeros_like(owners)
        overhead[5:30, 100:140] = 1
        self.assertIn("overhead", self.reasons(overhead, *scene(slice(100, 140), slice(5, 30))))
        missing_owners, missing_depth = scene(slice(60, 200))
        missing = np.zeros_like(owners)
        missing[60:90, 60:70] = 1
        self.assertIn("missing", self.reasons(missing, missing_owners, missing_depth))
        self.assertEqual(self.reasons(owners.copy(), owners, depth, camera_inside=True), ["interior"])

    def test_facade_hidden_by_trees_is_not_missing(self):
        owners, depth = scene(slice(60, 200))
        mask = np.zeros_like(owners)
        mask[60:90, 60:80] = 1
        hidden = np.zeros_like(owners, dtype=bool)
        hidden[:, 80:200] = True
        self.assertNotIn("missing", self.reasons(mask, owners, depth, hidden=hidden))

    def test_a_mask_across_the_panorama_seam_is_one_run(self):
        owners, depth = scene(slice(0, 360))
        mask = np.zeros_like(owners)
        mask[60:90, 340:360] = 1
        mask[60:90, 0:20] = 1
        self.assertEqual(view_signals(mask, 1, owners, depth, height_m=20.0)["runs"], 1)


class CameraInsideTests(unittest.TestCase):
    def test_a_camera_inside_a_footprint_is_detected(self):
        square = {"type": "Feature", "properties": {"osm_id": "way/1"},
                  "geometry": {"type": "Polygon", "coordinates": [[[0.0, 0.0], [0.001, 0.0], [0.001, 0.001], [0.0, 0.001], [0.0, 0.0]]]}}
        self.assertTrue(camera_inside_footprint([square], 0.0005, 0.0005))
        self.assertFalse(camera_inside_footprint([square], 0.002, 0.0005))


class BenchmarkTests(unittest.TestCase):
    def test_curated_demo_views_become_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            source, demo = Path(directory) / "source", Path(directory) / "demo"
            (source / "analysis").mkdir(parents=True)
            analysis = {"image_id": "7", "building_ids": {str(n): {"osm_id": f"way/{n}"} for n in range(1, 5)}}
            (source / "analysis" / "7.json").write_text(json.dumps(analysis))
            curation = {"masks": [{"image": "7", "building": "way/1", "keep_bearings": [10, 20]}], "answers": {"way/2": {"7": {}}}}
            demo.mkdir()
            (demo / "curation.json").write_text(json.dumps(curation))
            for building, score in (("way/1", 0.5), ("way/2", 0.01), ("way/3", 0.01)):
                crop = demo / "crops" / building / "crop-01-7.json"
                crop.parent.mkdir(parents=True)
                crop.write_text(json.dumps({"source": {"image_id": "7"}, "building": {"id": building}, "evidence": {"score": score}}))
            labels = {label["building"]: (label["label"], label.get("reasons")) for label in labels_from_curation(source, demo)}
        # way/1 was corrected, way/2 answered by the reviewer, way/3 left as a sliver; way/4 was never shown.
        self.assertEqual(labels, {"way/1": ("bad", ["neighbor"]), "way/2": ("good", None), "way/3": ("bad", ["sliver"])})

    def test_evaluation_counts_caught_and_dropped_views(self):
        labels = [{"image": "1", "building": "a", "label": "bad", "reasons": ["sliver"]},
                  {"image": "1", "building": "b", "label": "bad", "reasons": ["neighbor"]},
                  {"image": "1", "building": "c", "label": "good"}, {"image": "2", "building": "a", "label": "good"}]
        scored = {("1", "a"): {"ok": False}, ("1", "b"): {"ok": True}, ("1", "c"): {"ok": False}}
        result = evaluate(labels, scored)
        self.assertEqual((result["recall"], result["false_alarms"], result["unscored"]), (0.5, 1.0, 1))
        self.assertEqual(result["recall_by_reason"], {"neighbor": "0/1", "sliver": "1/1"})


if __name__ == "__main__":
    unittest.main()
