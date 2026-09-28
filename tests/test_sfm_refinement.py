import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from city_analyser.sfm_refinement import (
    _can_report_metric_depth,
    _cube_face_uv,
    _ecef_to_wgs84,
    _require_pycolmap,
    _sparse_depth_samples,
    _write_match_pairs,
    load_sfm_refinement,
)


class SfmRefinementTests(unittest.TestCase):
    def test_cache_requires_matching_image_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = root / "sample" / "analysis"
            analysis.mkdir(parents=True)
            (analysis / "123-sfm.json").write_text(json.dumps({"version": 3, "image_id": "other"}), encoding="utf-8")

            self.assertIsNone(load_sfm_refinement(root, "sample", "123"))

    def test_ecef_equator_point_converts_to_wgs84(self):
        longitude, latitude = _ecef_to_wgs84((6_378_137, 0, 0))

        self.assertAlmostEqual(longitude, 0.0, places=6)
        self.assertAlmostEqual(latitude, 0.0, places=6)

    def test_pycolmap_error_explains_uv_sync_when_unavailable(self):
        with patch("city_analyser.sfm_refinement.importlib.import_module", side_effect=ImportError):
            with self.assertRaisesRegex(RuntimeError, "uv sync"):
                _require_pycolmap()

    def test_match_pairs_only_connect_physical_cube_overlaps(self):
        images = [
            {"id": "1", "computed_compass_angle_deg": 0},
            {"id": "2", "computed_compass_angle_deg": 0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.txt"
            _write_match_pairs(path, images)
            pairs = path.read_text(encoding="utf-8").splitlines()

        self.assertIn("pano-1-front.jpg pano-1-right.jpg", pairs)
        self.assertIn("pano-1-front.jpg pano-2-front.jpg", pairs)
        self.assertNotIn("pano-1-front.jpg pano-2-back.jpg", pairs)

    def test_match_pairs_account_for_heading_and_connect_every_panorama_pair(self):
        images = [
            {"id": "1", "computed_compass_angle_deg": 0},
            {"id": "2", "computed_compass_angle_deg": 90},
            {"id": "3", "computed_compass_angle_deg": 0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.txt"
            second_path = Path(directory) / "second.txt"
            _write_match_pairs(first_path, images)
            _write_match_pairs(second_path, list(reversed(images)))
            pairs = first_path.read_text(encoding="utf-8").splitlines()

            self.assertEqual(first_path.read_text(encoding="utf-8"), second_path.read_text(encoding="utf-8"))

        self.assertIn("pano-1-front.jpg pano-2-left.jpg", pairs)
        self.assertNotIn("pano-1-front.jpg pano-2-right.jpg", pairs)
        self.assertIn("pano-1-front.jpg pano-3-front.jpg", pairs)

    def test_match_pairs_reject_nonfinite_or_missing_headings(self):
        for invalid_image in (
            {"id": "2"},
            {"id": "2", "computed_compass_angle_deg": float("nan")},
            {"id": "2", "computed_compass_angle_deg": float("inf")},
        ):
            with self.subTest(image=invalid_image):
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaisesRegex(ValueError, "heading"):
                        _write_match_pairs(
                            Path(directory) / "pairs.txt",
                            [{"id": "1", "computed_compass_angle_deg": 0}, invalid_image],
                        )

    def test_cube_face_observation_maps_to_panorama_coordinates(self):
        front = _cube_face_uv("front", (384, 384))
        right = _cube_face_uv("right", (384, 384))

        self.assertEqual(front, (0.5, 0.5))
        self.assertEqual(right, (0.75, 0.5))

    def test_sparse_depth_samples_include_quality_and_filter_very_weak_points(self):
        import numpy as np

        observations = [SimpleNamespace(image_id=10, point2D_idx=0), SimpleNamespace(image_id=11, point2D_idx=0)]

        def point(track_length, error):
            track = SimpleNamespace(elements=observations, length=lambda: track_length)
            return SimpleNamespace(xyz=(0, 0, 5), track=track, error=error, has_error=lambda: True)

        points = {
            1: point(3, 1.23456),
            2: point(1, 0.2),
            3: point(4, 20.0),
        }
        image = SimpleNamespace(
            name="pano-1-front.jpg",
            projection_center=lambda: (0, 0, 0),
            point2D=lambda _: SimpleNamespace(xy=(384, 384)),
        )
        second_image = SimpleNamespace(name="pano-2-front.jpg")
        model = SimpleNamespace(
            point3D_ids=lambda: points.keys(),
            point3D=lambda point_id: points[point_id],
            image=lambda image_id: image if image_id == 10 else second_image,
        )

        with patch("city_analyser.sfm_refinement._require_sfm_dependencies", return_value=(np, None)):
            samples = _sparse_depth_samples(model, {image.name: "1", second_image.name: "2"}, "1", {"1", "2"})

        self.assertEqual(
            samples,
            [{"u": 0.5, "v": 0.5, "distance_m": 5.0, "track_length": 3, "reprojection_error_px": 1.235}],
        )
        with patch("city_analyser.sfm_refinement._require_sfm_dependencies", return_value=(np, None)):
            same_panorama = _sparse_depth_samples(model, {image.name: "1", second_image.name: "1"}, "1", {"1"})
        self.assertEqual(same_panorama, [])

    def test_metric_depth_requires_current_and_three_accepted_poses(self):
        two_accepted = {"1": {"accepted": True}, "2": {"accepted": True}, "3": {"accepted": False}}
        three_accepted = {**two_accepted, "3": {"accepted": True}}

        self.assertFalse(_can_report_metric_depth(two_accepted, "1"))
        self.assertTrue(_can_report_metric_depth(three_accepted, "1"))
        self.assertFalse(_can_report_metric_depth(three_accepted, "missing"))
        self.assertFalse(_can_report_metric_depth({**three_accepted, "1": {"accepted": False}}, "1"))


if __name__ == "__main__":
    unittest.main()
