import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluation import corruptions, run


def _image() -> np.ndarray:
    return np.random.default_rng(1).integers(40, 200, (64, 1024, 3), dtype=np.uint8)


class CorruptionTests(unittest.TestCase):
    def test_occlusion_hides_the_requested_share_and_is_deterministic(self):
        image = _image()
        for severity, share in corruptions.OCCLUSION_AREA.items():
            result = corruptions.corrupt(image, "occlusion", severity, seed=3)
            hidden = float((result == 0).all(axis=2).mean())
            self.assertAlmostEqual(hidden, share, delta=0.03)
            np.testing.assert_array_equal(result, corruptions.corrupt(image, "occlusion", severity, seed=3))

    def test_blur_reduces_detail_and_noise_adds_it_with_severity(self):
        image = _image()
        detail = lambda a: float(np.abs(np.diff(a.astype(np.float32), axis=1)).mean())  # noqa: E731
        blurs = [detail(corruptions.corrupt(image, "motion_blur", s)) for s in corruptions.SEVERITIES]
        self.assertGreater(detail(image), blurs[0])
        self.assertGreater(blurs[0], blurs[2])
        deviations = [float(np.abs(corruptions.corrupt(image, "gaussian_noise", s).astype(int) - image).mean()) for s in corruptions.SEVERITIES]
        self.assertLess(deviations[0], deviations[2])

    def test_brightness_moves_the_mean_in_both_directions(self):
        image = _image()
        up = [corruptions.corrupt(image, "brightness_up", s).mean() for s in corruptions.SEVERITIES]
        down = [corruptions.corrupt(image, "brightness_down", s).mean() for s in corruptions.SEVERITIES]
        self.assertTrue(image.mean() < up[0] < up[2])
        self.assertTrue(image.mean() > down[0] > down[2])

    def test_parsing_and_validation(self):
        self.assertEqual(corruptions.parse_corruption("motion_blur:2"), ("motion_blur", 2))
        for bad in ("motion_blur", "motion_blur:4", "nope:1", "motion_blur:x"):
            with self.assertRaises(ValueError):
                corruptions.parse_corruption(bad)
        with self.assertRaises(ValueError):
            corruptions.corrupt(_image(), "occlusion", 9)

    def test_relative_scores(self):
        self.assertAlmostEqual(corruptions.relative_degradation(0.2, {1: 0.25, 2: 0.3, 3: 0.4}), 0.35)
        self.assertAlmostEqual(corruptions.relative_mce({"a": 0.2, "b": 0.1}, {"a": 0.4, "b": 0.1, "c": 1.0}), 0.75)
        self.assertIsNone(corruptions.relative_mce({"a": 0.2}, {"a": 0.0}))


class RobustnessCommandTests(unittest.TestCase):
    def test_table_from_scored_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def score(tag: str, separation_f: float) -> None:
                (root / tag).mkdir(parents=True)
                (root / tag / "scores.json").write_text(json.dumps({"summary": {"separation_f": separation_f}}), encoding="utf-8")

            score("clean", 0.8)
            score("x-motion_blur-1", 0.75)
            score("x-motion_blur-2", 0.6)
            score("ref-clean", 0.8)
            score("ref-motion_blur-1", 0.7)
            with patch.object(run, "REPORT_ROOT", root):
                run.command_robustness(Namespace(clean="clean", prefix="x", reference_clean="ref-clean", reference_prefix="ref"))
            summary = json.loads((root / "clean" / "robustness.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["corruptions"][0]["corruption"], "motion_blur")
        self.assertAlmostEqual(summary["corruptions"][0]["added_error"], (0.25 - 0.2) + (0.4 - 0.2), places=3)
        self.assertIsNotNone(summary["relative_mce"])


if __name__ == "__main__":
    unittest.main()
