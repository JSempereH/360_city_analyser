import json
import tempfile
import threading
import unittest
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

Image = __import__("PIL.Image", fromlist=["Image"])

from city_analyser.analysis_cache import analysis_input_sha256, file_sha256
from city_analyser.building_analysis import ANALYSIS_VERSION, analysis_configuration
from city_analyser.building_inspection import BuildingEvidenceNotFound, build_and_persist_report, load_report, report_paths
from city_analyser.building_inspection.assessment import geometry_fingerprint
from city_analyser.building_inspection.evidence import collect_building_evidence
from city_analyser.building_inspection.ranking import rank_views
from city_analyser.local_viewer import ViewerServer


class BuildingFixture:
    def create_dataset(self, root: Path, *, second_view: bool = True, provider: str = "Mapillary") -> tuple[Path, dict]:
        dataset_dir = root / "sample"
        (dataset_dir / "images").mkdir(parents=True)
        (dataset_dir / "analysis").mkdir()
        image_ids = ["100", "200"] if second_view else ["100"]
        images = []
        for index, image_id in enumerate(image_ids):
            local_file = f"images/pano-{image_id}.jpg"
            panorama = np.zeros((200, 400, 3), dtype=np.uint8)
            panorama[:, :, :] = [40 + index * 20, 90, 120]
            Image.fromarray(panorama).save(dataset_dir / local_file)
            images.append({
                "id": image_id,
                "is_pano": True,
                "local_file": local_file,
                "captured_at_ms": 1000 + index,
                "computed_compass_angle_deg": 10,
                "selected_geometry": {"type": "Point", "coordinates": [4.9 + index * 0.0001, 52.3]},
                "source_page": f"https://example.test/{image_id}",
            })
            ids = np.zeros((100, 200), dtype=np.uint16)
            ids[30:70, 45 + index * 10:85 + index * 10] = 1
            inferred = np.zeros_like(ids)
            inferred[25:30, 45 + index * 10:85 + index * 10] = 1
            Image.fromarray(ids).save(dataset_dir / "analysis" / f"{image_id}-ids.png")
            Image.fromarray(inferred).save(dataset_dir / "analysis" / f"{image_id}-facades.png")
            geometry = {
                "type": "Polygon",
                "coordinates": [[[4.9, 52.3], [4.9001, 52.3], [4.9001, 52.3001], [4.9, 52.3]]],
            }
            analysis = {
                "version": ANALYSIS_VERSION,
                "image_id": image_id,
                "width": 200,
                "height": 100,
                "refined_pose": {"facade_iou": 0.25 + index * 0.05, "facade_iou_gain": 0.04},
                "building_ids": {
                    "1": {
                        "osm_id": "way/42",
                        "footprint_source": "osm",
                        "alternative_footprint_id": None,
                        "footprint_geometry": geometry,
                        "facades": 2,
                        "pixels": 1600,
                        "inferred_pixels": 200,
                        "confidence": 0.6 + index * 0.1,
                        "sfm_depth_samples": 3,
                        "sfm_depth_conflicts": index,
                    }
                },
            }
            analysis["input_sha256"] = analysis_input_sha256(root, "sample", images[-1], analysis_configuration())
            analysis["sfm_input_sha256"] = None
            analysis["artifact_sha256"] = {
                "id_mask": file_sha256(dataset_dir / "analysis" / f"{image_id}-ids.png"),
                "facade_mask": file_sha256(dataset_dir / "analysis" / f"{image_id}-facades.png"),
            }
            (dataset_dir / "analysis" / f"{image_id}.json").write_text(json.dumps(analysis), encoding="utf-8")
        manifest = {"query": {"provider": provider, "acquired_at": "2026-01-01T00:00:00Z"}, "images": images}
        (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return dataset_dir, manifest

class BuildingInspectionTests(BuildingFixture, unittest.TestCase):
    def write_minimal_report(self, dataset_dir: Path, **changes) -> dict:
        payload = {
            "schema_version": 1,
            "building": {"id": "way/42", "identity_source": "openstreetmap"},
            "inputs": {"required_analysis_version": ANALYSIS_VERSION},
            "ranked_views": [{"evidence": {"footprint_source": "osm", "alternative_footprint_id": None}}],
        }
        payload.update(changes)
        json_path, _ = report_paths(dataset_dir, "way/42")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def test_collects_and_ranks_direct_evidence_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir, manifest = self.create_dataset(Path(directory))
            collection = collect_building_evidence(dataset_dir, manifest, "way/42", expected_analysis_version=ANALYSIS_VERSION)
            ranked = rank_views(collection.views, "sample", "way/42")

        self.assertEqual([view.evidence.image_id for view in ranked], ["200", "100"])
        self.assertEqual(collection.compatible_analysis_count, 2)
        self.assertIn("building=way%2F42", ranked[0].viewer_url)
        self.assertAlmostEqual(ranked[0].evidence.yaw_deg, -45.0, places=1)

    def test_report_persists_clean_and_annotated_screenshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, manifest = self.create_dataset(root, provider="<script>alert(1)</script>")
            report = build_and_persist_report(
                root,
                "sample",
                manifest,
                "way/42",
                expected_analysis_version=ANALYSIS_VERSION,
                now=lambda: datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
            )
            json_path, html_path = report_paths(dataset_dir, "way/42")
            html = html_path.read_text(encoding="utf-8")

            self.assertTrue(json_path.is_file())
            self.assertEqual(len(list(json_path.parent.glob("view-01-clean-*.jpg"))), 1)
            self.assertEqual(len(list(json_path.parent.glob("view-01-overlay-*.jpg"))), 1)
            self.assertIsNotNone(report["ranked_views"][0]["clean_screenshot_url"])
            self.assertEqual(load_report(dataset_dir, "way/42"), report)
            self.assertEqual(report["assessment"]["conclusion"]["code"], "consistent_multi_view_association")
            self.assertEqual(report["assessment"]["non_assessments"][0]["status"], "not_performed")
            self.assertIn("&lt;script&gt;", html)
            self.assertNotIn("<script>alert(1)</script>", html)
            self.assertNotIn("<script", html)
            self.assertIn("GEM classification: not run", html)
            self.assertIn('id="evidence-overlay"', html)
            self.assertIn("Technical details and limitations", html)

    def test_missing_building_evidence_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir, manifest = self.create_dataset(Path(directory), second_view=False)
            with self.assertRaises(BuildingEvidenceNotFound):
                collect_building_evidence(dataset_dir, manifest, "way/99", expected_analysis_version=ANALYSIS_VERSION)

    def test_report_paths_reject_non_osm_identifiers(self):
        with self.assertRaises(ValueError):
            report_paths(Path("dataset"), "microsoft/120202110-84625")

    def test_load_report_rejects_stale_analysis_version(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = Path(directory)
            self.write_minimal_report(
                dataset_dir,
                inputs={"required_analysis_version": ANALYSIS_VERSION - 1},
            )

            self.assertIsNone(load_report(dataset_dir, "way/42"))

    def test_load_report_rejects_non_osm_payloads(self):
        cases = {
            "Microsoft building identity": {
                "building": {"id": "way/42", "identity_source": "microsoft"},
            },
            "non-OSM building ID": {
                "building": {"id": "microsoft/120202110-84625", "identity_source": "openstreetmap"},
            },
            "non-OSM footprint source": {
                "ranked_views": [{"evidence": {"footprint_source": "microsoft", "alternative_footprint_id": "microsoft/120202110-84625"}}],
            },
            "non-OSM footprint ID": {
                "ranked_views": [{"evidence": {"footprint_source": "osm", "alternative_footprint_id": "microsoft/120202110-84625"}}],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = Path(directory)
            for label, changes in cases.items():
                with self.subTest(label):
                    self.write_minimal_report(dataset_dir, **changes)
                    self.assertIsNone(load_report(dataset_dir, "way/42"))

    def test_report_evidence_rejects_analysis_when_sfm_input_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir, manifest = self.create_dataset(Path(directory), second_view=False)
            (dataset_dir / "analysis" / "100-sfm.json").write_text(
                json.dumps({"version": 3, "image_id": "100", "depth_samples": []}), encoding="utf-8",
            )

            with self.assertRaises(BuildingEvidenceNotFound):
                collect_building_evidence(dataset_dir, manifest, "way/42", expected_analysis_version=ANALYSIS_VERSION)

    def test_report_evidence_rejects_a_replaced_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir, manifest = self.create_dataset(Path(directory), second_view=False)
            Image.fromarray(np.zeros((100, 200), dtype=np.uint16)).save(dataset_dir / "analysis" / "100-ids.png")

            with self.assertRaises(BuildingEvidenceNotFound):
                collect_building_evidence(dataset_dir, manifest, "way/42", expected_analysis_version=ANALYSIS_VERSION)

    def test_geometry_fingerprint_is_independent_of_key_order(self):
        first = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 0]]]}
        second = {"coordinates": [[[0, 0], [1, 0], [0, 0]]], "type": "Polygon"}
        self.assertEqual(geometry_fingerprint(first), geometry_fingerprint(second))

class BuildingReportApiTests(BuildingFixture, unittest.TestCase):
    def test_report_http_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            (project_root / "models").mkdir()
            self.create_dataset(project_root / "data", second_view=False)
            server = ViewerServer(("127.0.0.1", 0), project_root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}/api/datasets/sample/building-reports/way/42"
            try:
                with urlopen(Request(base, method="POST"), timeout=10) as response:
                    created = json.loads(response.read())
                    self.assertEqual(response.status, 201)
                with urlopen(base, timeout=10) as response:
                    loaded = json.loads(response.read())
                with urlopen(f"{base}/html", timeout=10) as response:
                    report_html = response.read().decode("utf-8")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertEqual(created["report"]["building"]["id"], "way/42")
        self.assertEqual(loaded["schema_version"], 1)
        self.assertIn("text/html", response.headers.get_content_type())
        self.assertIn("Review in viewer", report_html)

    def test_static_data_route_denies_report_documents_but_serves_screenshots(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            (project_root / "models").mkdir()
            self.create_dataset(project_root / "data", second_view=False)
            server = ViewerServer(("127.0.0.1", 0), project_root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            api_url = f"http://127.0.0.1:{server.server_port}/api/datasets/sample/building-reports/way/42"
            data_url = f"http://127.0.0.1:{server.server_port}/data/sample/reports/buildings/way/42"
            try:
                with urlopen(Request(api_url, method="POST"), timeout=10) as response:
                    created = json.loads(response.read())
                legacy_dir = project_root / "data" / "sample" / "reports" / "buildings" / "microsoft" / "1"
                legacy_dir.mkdir(parents=True)
                (legacy_dir / "view-01-overlay-deadbeef.jpg").write_bytes(b"legacy")
                for filename in ("report-v1.json", "report-v1.html"):
                    with self.subTest(filename), self.assertRaises(HTTPError) as raised:
                        urlopen(f"{data_url}/{filename}", timeout=10)
                    self.assertEqual(raised.exception.code, HTTPStatus.NOT_FOUND)
                with self.assertRaises(HTTPError) as raised:
                    urlopen(f"http://127.0.0.1:{server.server_port}/data/sample/reports/buildings/microsoft/1/view-01-overlay-deadbeef.jpg", timeout=10)
                self.assertEqual(raised.exception.code, HTTPStatus.NOT_FOUND)
                screenshot_url = created["report"]["ranked_views"][0]["screenshot_url"]
                with urlopen(f"http://127.0.0.1:{server.server_port}{screenshot_url}", timeout=10) as response:
                    self.assertEqual(response.status, HTTPStatus.OK)
                    self.assertEqual(response.headers.get_content_type(), "image/jpeg")
                    self.assertTrue(response.read())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
