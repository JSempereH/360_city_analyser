import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from city_analyser import batch_analysis
from city_analyser.analysis_cache import atomic_write_json
from city_analyser.building_index import build_building_index, index_path, load_building_index, panorama_visibility, select_panoramas

LATITUDE, LONGITUDE = 52.3, 4.9


def _square(osm_id: str, south: float, north: float, west: float, east: float) -> dict:
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return {"type": "Feature", "properties": {"osm_id": osm_id, "osm_type": "way"}, "geometry": {"type": "Polygon", "coordinates": [ring]}}


# About 11 m north of the camera and 7 m wide.
AHEAD = _square("way/1", 52.30010, 52.30020, 4.89995, 4.90005)
# A block the camera stands inside.
AROUND = _square("way/2", 52.2999, 52.3001, 4.8999, 4.9001)


class VisibilityTests(unittest.TestCase):
    def test_angle_of_view_and_distance_of_a_facing_wall(self):
        seen = panorama_visibility([AHEAD], LATITUDE, LONGITUDE)["way/1"]
        self.assertTrue(25 < seen["aov_deg"] < 45, seen)
        self.assertAlmostEqual(seen["distance_m"], 11.1, delta=1.0)
        self.assertTrue(seen["bearing_deg"] < 5 or seen["bearing_deg"] > 355)

    def test_a_wall_behind_another_is_not_visible(self):
        behind = _square("way/3", 52.30030, 52.30040, 4.89995, 4.90005)
        self.assertNotIn("way/3", panorama_visibility([AHEAD, behind], LATITUDE, LONGITUDE))

    def test_a_camera_recorded_inside_a_footprint_is_repaired_to_the_street(self):
        seen = panorama_visibility([AROUND], LATITUDE, LONGITUDE)
        self.assertIn("way/2", seen)


def _index() -> dict:
    view = lambda image, aov, distance: {"image": image, "aov_deg": aov, "distance_m": distance, "visible_wall_m": 10.0, "bearing_deg": 0.0}  # noqa: E731
    return {"buildings": {
        "way/1": [view("a", 90.0, 8.0), view("b", 40.0, 15.0), view("c", 30.0, 20.0), view("d", 5.0, 30.0), view("e", 150.0, 3.0)],
        "way/2": [view("d", 20.0, 40.0)],
        "way/3": [view("a", 4.0, 60.0)],
    }}


class SelectionTests(unittest.TestCase):
    def test_aov_window_distance_and_view_cap(self):
        selection = select_panoramas(_index(), None, max_views_per_building=2)
        self.assertEqual(selection["images"], ["a", "b", "d"])  # way/1: a, b; way/2: d; way/3 has no acceptable view
        self.assertEqual(selection["buildings_covered"], 2)
        self.assertEqual(selection["buildings_uncovered"], ["way/3"])

    def test_building_subset_and_no_floor(self):
        subset = select_panoramas(_index(), ["way/3"], min_aov_deg=0, max_distance_m=100)
        self.assertEqual(subset["images"], ["a"])
        self.assertEqual(select_panoramas(_index(), ["way/9"])["buildings_uncovered"], ["way/9"])


class IndexFileTests(unittest.TestCase):
    def _dataset(self, root: Path) -> list[dict]:
        (root / "set" / "images").mkdir(parents=True)
        images = []
        for number in range(2):
            (root / "set" / "images" / f"{number}.jpg").write_bytes(b"x")
            images.append({"id": str(number), "local_file": f"images/{number}.jpg", "computed_compass_angle_deg": 0,
                           "computed_geometry": {"type": "Point", "coordinates": [LONGITUDE, LATITUDE]}})
        (root / "set" / "manifest.json").write_text(json.dumps({"images": images}), encoding="utf-8")
        return images

    def test_build_store_reload_and_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = self._dataset(root)
            images.append({"id": "bad", "local_file": "images/missing.jpg"})
            with patch("city_analyser.building_index.nearby_buildings", return_value=[AHEAD]):
                index = build_building_index(root, "set", images)
            self.assertEqual(index["panoramas"], 2)
            self.assertEqual(index["skipped"][0]["image"], "bad")
            self.assertEqual(len(index["buildings"]["way/1"]), 2)
            atomic_write_json(index_path(root, "set"), index)
            self.assertEqual(load_building_index(root, "set", images), index)
            self.assertIsNone(load_building_index(root, "set", images[:1]))

    def test_batch_selection_filters_images_and_reuses_the_stored_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = self._dataset(root)
            with patch("city_analyser.building_index.nearby_buildings", return_value=[AHEAD]) as nearby, patch("builtins.print"):
                kept = batch_analysis.select_images(root, "set", images, ["way/1"], 1, 0.0)
                batch_analysis.select_images(root, "set", images, None, None, 0.0)
            self.assertEqual(len(kept), 1)
            self.assertEqual(nearby.call_count, 2)  # indexed once, then loaded from disk


if __name__ == "__main__":
    unittest.main()
