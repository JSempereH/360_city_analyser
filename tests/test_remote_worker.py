import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from city_analyser.analysis_backend import AnalysisBackend, perceive
from city_analyser.building_analysis import Perception
from city_analyser.remote_analysis_worker import AnalysisWorker


class RemoteWorkerTests(unittest.TestCase):
    def setUp(self):
        self.server = AnalysisWorker(("127.0.0.1", 0), Path("models"), "secret")
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/api/v2/perception"
        np = __import__("numpy")
        self.perception = Perception(
            np.full((4, 8), 7, dtype=np.uint8), np.zeros((4, 8), dtype=np.uint8), "model", None, "cpu",
            np.ones((4, 8), dtype=np.uint16), [{"instance": 1}],
        )

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_client_and_worker_round_trip_without_touching_osm(self):
        backend = AnalysisBackend("remote", self.url, "secret")
        with tempfile.TemporaryDirectory() as directory, \
                patch("city_analyser.remote_analysis_worker.perceive_panorama", return_value=self.perception) as perceive_panorama:
            image_path = Path(directory) / "pano.jpg"
            image_path.write_bytes(b"jpeg")
            result = perceive(backend, Path(directory), Path(directory), image_path, lambda _: None)
        self.assertEqual(perceive_panorama.call_count, 1)
        self.assertEqual(result.detected_instances, [{"instance": 1}])
        self.assertTrue((result.building_confidence == 7).all())
        self.assertTrue((result.instance_map == 1).all())

    def test_rejects_missing_token_and_mismatched_configuration(self):
        request = Request(self.url, data=b"jpeg", headers={"X-Analysis-Protocol": "2", "X-Analysis-Configuration": "{}"}, method="POST")
        with self.assertRaises(HTTPError) as unauthorized:
            urlopen(request)
        self.assertEqual(unauthorized.exception.code, 401)
        request.add_header("Authorization", "Bearer secret")
        with self.assertRaises(HTTPError) as mismatch:
            urlopen(request)
        self.assertEqual(mismatch.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
