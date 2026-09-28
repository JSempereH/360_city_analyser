import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from city_analyser.analysis_backend import ANALYSIS_VERSION, AnalysisBackend, analysis_backend_from_environment, run_analysis
from city_analyser.building_analysis import analysis_configuration, analysis_paths, facade_path


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

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
        self.assertEqual(backend.remote_url, "https://gpu.example.test/worker/api/v1/building-analysis")
        self.assertEqual(backend.timeout_seconds, 120)

    def test_remote_mode_requires_a_remote_url(self):
        with self.assertRaisesRegex(ValueError, "REMOTE_URL"):
            analysis_backend_from_environment({"BUILDING_ANALYSIS_BACKEND": "remote"})

    def test_remote_mode_requires_an_explicit_model(self):
        with self.assertRaisesRegex(ValueError, "MODEL"):
            analysis_backend_from_environment({"BUILDING_ANALYSIS_BACKEND": "remote", "BUILDING_ANALYSIS_REMOTE_URL": "https://gpu.example.test"})

    def test_remote_result_is_cached_without_sending_the_local_path(self):
        png = b"\x89PNG\r\n\x1a\nmock"
        metadata = {"version": ANALYSIS_VERSION, "image_id": "123", "building_ids": {}, "analysis_configuration": analysis_configuration()}
        response = _Response({
            "analysis": metadata,
            "id_mask_png_base64": base64.b64encode(png).decode("ascii"),
            "facade_mask_png_base64": base64.b64encode(png).decode("ascii"),
        })
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            image_path = data_root / "sample" / "images" / "pano.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"jpeg")
            analysis_dir = data_root / "sample" / "analysis"
            analysis_dir.mkdir()
            sfm = {"version": 3, "depth_samples": [{"u": 0.5, "v": 0.5, "distance_m": 12.0}]}
            (analysis_dir / "123-sfm.json").write_text(json.dumps(sfm), encoding="utf-8")
            image = {
                "id": "123",
                "local_file": "images/pano.jpg",
                "computed_compass_angle_deg": 10,
                "computed_geometry": {"type": "Point", "coordinates": [4.9, 52.3]},
            }
            with patch("city_analyser.analysis_backend.urlopen", return_value=response) as urlopen:
                result = run_analysis(AnalysisBackend("remote", "https://gpu.example.test/api/v1/building-analysis", "secret"), data_root, data_root / "models", "sample", image, lambda _: None)

            request = urlopen.call_args.args[0]
            request_payload = json.loads(request.data.decode("utf-8"))
            self.assertNotIn("local_file", request_payload["image"])
            self.assertNotIn("sfm_refinement", request_payload)
            self.assertEqual(request.get_header("Authorization"), "Bearer secret")
            self.assertEqual(result["version"], metadata["version"])
            self.assertIn("input_sha256", result)
            self.assertIn("artifact_sha256", result)
            metadata_path, id_path = analysis_paths(data_root, "sample", "123")
            self.assertEqual(json.loads(metadata_path.read_text(encoding="utf-8")), result)
            self.assertEqual(id_path.read_bytes(), png)
            self.assertEqual(facade_path(data_root, "sample", "123").read_bytes(), png)


if __name__ == "__main__":
    unittest.main()
