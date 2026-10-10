import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from city_analyser.analysis_backend import AnalysisBackend, analysis_backend_from_environment, perceive, run_analysis
from city_analyser.building_analysis import Perception, perception_configuration, perception_to_arrays


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class AnalysisBackendTests(unittest.TestCase):
    def test_auto_uses_local_when_no_remote_url_is_configured(self):
        backend = analysis_backend_from_environment({})

        self.assertEqual(backend.kind, "local")
        self.assertEqual(backend.label, "local device")

    def test_auto_uses_remote_when_remote_url_is_configured(self):
        backend = analysis_backend_from_environment({
            "BUILDING_ANALYSIS_REMOTE_URL": "https://gpu.example.test/worker/",
            "BUILDING_ANALYSIS_REMOTE_TOKEN": "secret",
            "BUILDING_ANALYSIS_REMOTE_TIMEOUT_S": "120",
            "BUILDING_ANALYSIS_MODEL": "balanced",
        })

        self.assertEqual(backend.kind, "remote")
        self.assertEqual(backend.remote_url, "https://gpu.example.test/worker/api/v2/perception")
        self.assertEqual(backend.timeout_seconds, 120)

    def test_remote_mode_requires_a_remote_url(self):
        with self.assertRaisesRegex(ValueError, "REMOTE_URL"):
            analysis_backend_from_environment({"BUILDING_ANALYSIS_BACKEND": "remote"})

    def test_remote_mode_requires_an_explicit_model(self):
        with self.assertRaisesRegex(ValueError, "MODEL"):
            analysis_backend_from_environment({"BUILDING_ANALYSIS_BACKEND": "remote", "BUILDING_ANALYSIS_REMOTE_URL": "https://gpu.example.test"})

    def test_remote_perception_is_cached_and_associated_locally(self):
        np = __import__("numpy")
        perception = Perception(
            np.full((4, 8), 200, dtype=np.uint8), np.zeros((4, 8), dtype=np.uint8), "model", "rev", "cuda",
            np.zeros((4, 8), dtype=np.uint16), [{"instance": 1}],
        )
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **perception_to_arrays(perception))
        backend = AnalysisBackend("remote", "https://gpu.example.test/api/v2/perception", "secret")
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            image_path = data_root / "sample" / "images" / "pano.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"jpeg")
            with patch("city_analyser.analysis_backend.urlopen", return_value=_Response(buffer.getvalue())) as urlopen:
                first = perceive(backend, data_root, data_root / "models", image_path, lambda _: None)
                second = perceive(backend, data_root, data_root / "models", image_path, lambda _: None)

            self.assertEqual(urlopen.call_count, 1)
            request = urlopen.call_args.args[0]
            self.assertEqual(request.data, b"jpeg")
            self.assertEqual(request.get_header("Authorization"), "Bearer secret")
            self.assertEqual(json.loads(request.get_header("X-analysis-configuration")), perception_configuration())
            for result in (first, second):
                self.assertEqual((result.model_id, result.model_revision, result.device), ("model", "rev", "cuda"))
                self.assertEqual(result.detected_instances, [{"instance": 1}])
                self.assertTrue((result.building_confidence == 200).all())

    def test_remote_run_associates_with_local_footprints(self):
        np = __import__("numpy")
        perception = Perception(np.zeros((4, 8), dtype=np.uint8), np.zeros((4, 8), dtype=np.uint8), "m", None, "cpu", None, [])
        backend = AnalysisBackend("remote", "https://gpu.example.test/api/v2/perception")
        image = {"id": "123", "local_file": "images/pano.jpg", "computed_compass_angle_deg": 10, "computed_geometry": {"type": "Point", "coordinates": [4.9, 52.3]}}
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            (data_root / "sample" / "images").mkdir(parents=True)
            (data_root / "sample" / "images" / "pano.jpg").write_bytes(b"jpeg")
            footprints = [{"id": "local-footprint"}]
            with patch("city_analyser.analysis_backend.perceive", return_value=perception), \
                    patch("city_analyser.analysis_backend.nearby_buildings", return_value=footprints) as nearby, \
                    patch("city_analyser.analysis_backend.associate_panorama", return_value={"ok": True}) as associate:
                result = run_analysis(backend, data_root, data_root / "models", "sample", image, lambda _: None)

            self.assertEqual(result, {"ok": True})
            nearby.assert_called_once_with(52.3, 4.9)
            self.assertIs(associate.call_args.args[4], perception)
            self.assertIs(associate.call_args.args[5], footprints)


if __name__ == "__main__":
    unittest.main()
