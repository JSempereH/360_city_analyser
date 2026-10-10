import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageFilter

from city_analyser.building_inspection.models import ViewEvidence
from city_analyser.building_inspection.ranking import rank_views, view_evidence_score
from city_analyser.image_quality import FLAGGED_RANK_WEIGHT, laplacian_variance, panorama_quality, quality_flags, quality_thresholds, rank_weight

np = __import__("numpy")


def _textured_panorama(path: Path, blur: float = 0.0, brightness: float = 1.0) -> None:
    rng = np.random.default_rng(5)
    base = rng.integers(0, 255, (256, 512), dtype=np.uint8)
    image = Image.fromarray(base).convert("RGB").resize((1024, 512), Image.Resampling.NEAREST)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    if brightness != 1.0:
        image = Image.fromarray(np.clip(np.asarray(image) * brightness, 0, 255).astype(np.uint8))
    image.save(path, quality=95)


class ImageQualityTests(unittest.TestCase):
    def test_sharp_image_is_clean_and_blur_lowers_sharpness_and_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            sharp, blurred = Path(directory) / "sharp.jpg", Path(directory) / "blurred.jpg"
            _textured_panorama(sharp)
            _textured_panorama(blurred, blur=6)
            clean, soft = panorama_quality(sharp), panorama_quality(blurred)
        self.assertGreater(clean["sharpness"], soft["sharpness"] * 5)
        self.assertNotIn("blurry", clean["flags"])
        self.assertIn("blurry", soft["flags"])

    def test_exposure_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            dark, bright = Path(directory) / "dark.jpg", Path(directory) / "bright.jpg"
            _textured_panorama(dark, brightness=0.05)
            _textured_panorama(bright, brightness=4.0)
            self.assertIn("dark", panorama_quality(dark)["flags"])
            self.assertIn("overexposed", panorama_quality(bright)["flags"])

    def test_thresholds_come_from_the_environment_and_flags_are_pure(self):
        with patch.dict("os.environ", {"BUILDING_ANALYSIS_MIN_SHARPNESS": "5"}):
            thresholds = quality_thresholds()
        self.assertEqual(thresholds["min_sharpness"], 5.0)
        facts = {"sharpness": 10.0, "brightness": 100.0, "file_bytes": 10_000_000}
        self.assertEqual(quality_flags(facts, thresholds), [])
        self.assertEqual(quality_flags({**facts, "file_bytes": 10}, thresholds), ["small_file"])
        with patch.dict("os.environ", {"BUILDING_ANALYSIS_MIN_SHARPNESS": "x"}), self.assertRaisesRegex(RuntimeError, "MIN_SHARPNESS"):
            quality_thresholds()

    def test_laplacian_of_a_constant_image_is_zero(self):
        self.assertEqual(laplacian_variance(np.full((10, 10), 7, dtype=np.uint8)), 0.0)

    def test_rank_weight(self):
        self.assertEqual(rank_weight(None), 1.0)
        self.assertEqual(rank_weight({"flags": []}), 1.0)
        self.assertEqual(rank_weight({"flags": ["blurry", "dark"]}), FLAGGED_RANK_WEIGHT)


def _evidence(image_id: str, quality: dict | None) -> ViewEvidence:
    return ViewEvidence(
        image_id=image_id, local_file="images/a.jpg", source_page=None, captured_at_ms=None, recorded_longitude=0.0, recorded_latitude=0.0,
        recorded_heading_deg=0.0, analysis_width=100, analysis_height=50, mask_ordinal=1, observed_pixels=500, inferred_pixels=0,
        projection_coverage=0.8, visible_facades=1, footprint_source="osm", alternative_footprint_id=None, footprint_geometry={},
        osm_version=1, footprint_correction=None, pose_facade_iou=None, pose_facade_iou_gain=None, sfm_depth_samples=0, sfm_depth_conflicts=0,
        yaw_deg=0.0, pitch_deg=0.0, fov_deg=60.0, analysis_relative_path="a", analysis_sha256="a", panorama_relative_path="p", panorama_sha256="p",
        id_mask_relative_path="i", id_mask_sha256="i", inferred_mask_relative_path="f", inferred_mask_sha256="f", image_quality=quality,
    )


class RankingQualityTests(unittest.TestCase):
    def test_flagged_view_ranks_below_an_equal_clean_view_and_says_why(self):
        flagged = _evidence("1", {"sharpness": 3.0, "brightness": 90.0, "file_bytes": 10**6, "flags": ["blurry"]})
        clean = _evidence("2", {"sharpness": 500.0, "brightness": 90.0, "file_bytes": 10**6, "flags": []})
        self.assertAlmostEqual(view_evidence_score(flagged), view_evidence_score(clean) / 2)
        ranked = rank_views((flagged, clean), "sample", "way/1")
        self.assertEqual([view.evidence.image_id for view in ranked], ["2", "1"])
        self.assertTrue(any("blurry" in reason for reason in ranked[1].reasons))
        self.assertEqual(ranked[1].to_dict()["image_quality"]["flags"], ["blurry"])


if __name__ == "__main__":
    unittest.main()
