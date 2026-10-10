import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from city_analyser.analysis_backend import AnalysisBackend
from city_analyser.batch_analysis import analyze_dataset


def _dataset(root: Path, count: int) -> None:
    (root / "set" / "images").mkdir(parents=True)
    images = []
    for number in range(count):
        (root / "set" / "images" / f"{number}.jpg").write_bytes(b"jpeg")
        images.append({
            "id": str(number), "local_file": f"images/{number}.jpg", "computed_compass_angle_deg": 0,
            "computed_geometry": {"type": "Point", "coordinates": [4.9, 52.3]},
        })
    (root / "set" / "manifest.json").write_text(json.dumps({"images": images}), encoding="utf-8")


class BatchAnalysisTests(unittest.TestCase):
    def test_remote_backend_perceives_remotely_and_associates_locally(self):
        backend = AnalysisBackend("remote", "https://gpu.example.test/api/v2/perception")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root, 3)
            with patch("city_analyser.batch_analysis.perceive", return_value="perception") as perceive, \
                    patch("city_analyser.batch_analysis.nearby_buildings", return_value=[]), \
                    patch("city_analyser.batch_analysis.associate_panorama", return_value={"building_ids": {"a": 1}}) as associate:
                report = analyze_dataset(root, root, "set", backend=backend)
        self.assertEqual((perceive.call_count, associate.call_count), (3, 3))
        self.assertEqual([row["image"] for row in report["rows"]], ["0", "1", "2"])
        self.assertEqual(report["backend"], "remote")

    def test_perception_only_skips_osm_and_association(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _dataset(root, 2)
            with patch("city_analyser.batch_analysis.perceive", return_value="perception") as perceive, \
                    patch("city_analyser.batch_analysis.nearby_buildings") as nearby, \
                    patch("city_analyser.batch_analysis.associate_panorama") as associate:
                report = analyze_dataset(root, root, "set", backend=AnalysisBackend("local"), perception_only=True)
        self.assertEqual(perceive.call_count, 2)
        nearby.assert_not_called()
        associate.assert_not_called()
        self.assertEqual(len(report["rows"]), 2)


if __name__ == "__main__":
    unittest.main()
