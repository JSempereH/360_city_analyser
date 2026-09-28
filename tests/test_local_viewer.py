import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from city_analyser.local_viewer import BUILDING_REPORT_ROUTE, MAX_MAPILLARY_SEARCH_PAGES, ViewerServer, delete_dataset, dataset_manifest, dataset_summaries, download_mapillary_panoramas, parse_mapillary_bbox, parse_mapillary_download, search_mapillary_panoramas
from city_analyser.mapillary_street_download import MAPILLARY_IMAGE_FIELDS


class LocalViewerTests(unittest.TestCase):
    def write_manifest(self, root: Path, dataset_id: str, images: list) -> None:
        dataset_dir = root / dataset_id
        dataset_dir.mkdir()
        (dataset_dir / "manifest.json").write_text(
            json.dumps({"query": {"provider": "Mapillary", "acquired_at": "2026-01-01T00:00:00Z"}, "images": images}),
            encoding="utf-8",
        )

    def test_discovers_valid_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_manifest(root, "sample-360", [{"id": "one"}])
            self.assertEqual(dataset_summaries(root), [{"id": "sample-360", "images": 1, "provider": "Mapillary", "acquired_at": "2026-01-01T00:00:00Z"}])

    def test_rejects_path_traversal_dataset_id(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                dataset_manifest(Path(directory), "../.env")

    def test_report_route_rejects_a_non_osm_footprint(self):
        self.assertIsNone(BUILDING_REPORT_ROUTE.fullmatch("/api/datasets/sample/building-reports/microsoft/120202110-84625"))

    def test_background_analysis_supports_a_single_panorama(self):
        panorama = {
            "id": "one",
            "is_pano": True,
            "local_file": "images/one.jpg",
            "computed_compass_angle_deg": 10,
            "computed_geometry": {"type": "Point", "coordinates": [4.9, 52.3]},
        }

        class ImmediateThread:
            def __init__(self, target, **_kwargs):
                self.target = target

            def start(self):
                self.target()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = SimpleNamespace(
                analysis_backend=SimpleNamespace(label="test backend"),
                analysis_jobs={},
                analysis_lock=threading.Lock(),
                analysis_run_lock=threading.Lock(),
                data_root=root,
                image_record=lambda _dataset_id, _image_id: panorama,
                model_root=root / "models",
            )
            with (
                patch("city_analyser.local_viewer.dataset_manifest", return_value={"images": [panorama]}),
                patch("city_analyser.local_viewer.load_analysis", return_value=None),
                patch("city_analyser.local_viewer.load_nearby_analysis", return_value=None),
                patch("city_analyser.local_viewer.run_analysis", return_value={"image_id": "one", "building_ids": {}}),
                patch("city_analyser.local_viewer.threading.Thread", ImmediateThread),
            ):
                result = ViewerServer.start_nearby_analysis(cast(Any, server), "sample", "one")

        self.assertEqual(result["job"]["status"], "complete")
        self.assertTrue(result["job"]["current_analysis_complete"])
        self.assertEqual(result["job"]["analysis"]["panoramas"][0]["image_id"], "one")

    def test_deletes_a_dataset_and_its_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_manifest(root, "sample-360", [{"id": "one"}])
            images = root / "sample-360" / "images"
            images.mkdir()
            (images / "panorama.jpg").write_bytes(b"image")

            delete_dataset(root, "sample-360")

            self.assertFalse((root / "sample-360").exists())

    def test_mapillary_search_returns_only_located_panoramas(self):
        def fetcher(token, bbox, image_field, max_pages):
            self.assertEqual(token, "test-token")
            self.assertEqual(bbox, (4.8, 52.3, 4.9, 52.4))
            self.assertEqual((image_field, max_pages), ((), MAX_MAPILLARY_SEARCH_PAGES))
            return [
                {"id": "pano", "is_pano": True, "captured_at": 2, "computed_geometry": {"type": "Point", "coordinates": [4.85, 52.35]}},
                {"id": "flat", "is_pano": False, "geometry": {"type": "Point", "coordinates": [4.86, 52.36]}},
            ]

        result = search_mapillary_panoramas("test-token", (4.8, 52.3, 4.9, 52.4), fetcher)

        self.assertEqual(result["panoramas_found"], 1)
        self.assertEqual(result["panoramas"][0]["coordinates"], (4.85, 52.35))

    def test_rejects_an_oversized_mapillary_aoi(self):
        with self.assertRaises(ValueError):
            parse_mapillary_bbox([0, 0, 1, 1])

    def test_parses_bounded_mapillary_download(self):
        bbox, dataset_id, image_ids, image_size = parse_mapillary_download(
            {
                "bbox": [-84.091, 9.927, -84.09, 9.928],
                "dataset_id": "san-jose-sample",
                "image_ids": ["123", "456"],
                "image_size": "auto",
            }
        )
        self.assertEqual(bbox, (-84.091, 9.927, -84.09, 9.928))
        self.assertEqual(dataset_id, "san-jose-sample")
        self.assertEqual(image_ids, ["123", "456"])
        self.assertEqual(image_size, "auto")

    def test_parses_unbounded_mapillary_download(self):
        _, _, image_ids, _ = parse_mapillary_download(
            {"bbox": [-84.091, 9.927, -84.09, 9.928], "dataset_id": "san-jose-city", "download_all": True}
        )

        self.assertIsNone(image_ids)

    def test_download_uses_an_available_fallback_resolution(self):
        def fetcher(token, bbox, image_fields, max_pages):
            self.assertEqual(image_fields, MAPILLARY_IMAGE_FIELDS)
            return [
                {
                    "id": "123",
                    "is_pano": True,
                    "computed_geometry": {"type": "Point", "coordinates": [4.85, 52.35]},
                    "thumb_1024_url": "https://example.test/pano-1024.jpg",
                }
            ]

        def downloader(url, destination, overwrite):
            self.assertEqual(url, "https://example.test/pano-1024.jpg")
            destination.write_bytes(b"image")
            return True, {"width": 1024, "height": 512, "sha256": "a" * 64}

        with tempfile.TemporaryDirectory() as directory:
            with patch("city_analyser.local_viewer.download_image", downloader):
                result = download_mapillary_panoramas(
                    Path(directory),
                    "test-token",
                    (4.8, 52.3, 4.9, 52.4),
                    "sample-360",
                    ["123"],
                    "original",
                    fetcher,
                )

            manifest = dataset_manifest(Path(directory), "sample-360")
            self.assertEqual(result, {"dataset_id": "sample-360", "downloaded": 1, "missing": 0})
            self.assertEqual(manifest["images"][0]["download_image_field"], "thumb_1024_url")

    def test_download_all_uses_unbounded_pagination(self):
        def fetcher(token, bbox, image_fields, max_pages):
            self.assertIsNone(max_pages)
            return [{"id": "123", "is_pano": True, "computed_geometry": {"type": "Point", "coordinates": [4.85, 52.35]}, "thumb_256_url": "https://example.test/pano.jpg"}]

        with tempfile.TemporaryDirectory() as directory:
            with patch("city_analyser.local_viewer.download_image", return_value=(True, {"width": 256, "height": 128, "sha256": "a" * 64})):
                result = download_mapillary_panoramas(Path(directory), "test-token", (4.8, 52.3, 4.9, 52.4), "city-360", None, "auto", fetcher)

        self.assertEqual(result["downloaded"], 1)

    def test_download_reports_progress(self):
        def fetcher(token, bbox, image_fields, max_pages):
            return [{"id": "123", "is_pano": True, "computed_geometry": {"type": "Point", "coordinates": [4.85, 52.35]}, "thumb_256_url": "https://example.test/pano.jpg"}]

        progress = []
        with tempfile.TemporaryDirectory() as directory:
            with patch("city_analyser.local_viewer.download_image", return_value=(True, {"width": 256, "height": 128, "sha256": "a" * 64})):
                download_mapillary_panoramas(Path(directory), "test-token", (4.8, 52.3, 4.9, 52.4), "progress-360", ["123"], "auto", fetcher, progress.append)

        self.assertIn("Finding Mapillary panoramas in the selected area...", progress)
        self.assertIn("Downloading panorama 1/1...", progress)

    def test_download_resolves_selected_ids_missing_from_aoi_page(self):
        def fetcher(token, bbox, image_fields, max_pages):
            return []

        def image_fetcher(token, image_id, image_fields):
            return {"id": image_id, "is_pano": True, "computed_geometry": {"type": "Point", "coordinates": [4.85, 52.35]}, "thumb_256_url": "https://example.test/pano.jpg"}

        with tempfile.TemporaryDirectory() as directory:
            with patch("city_analyser.local_viewer.download_image", return_value=(True, {"width": 256, "height": 128, "sha256": "a" * 64})):
                result = download_mapillary_panoramas(
                    Path(directory), "test-token", (4.8, 52.3, 4.9, 52.4), "resolved-360", ["123"], "auto", fetcher, image_fetcher=image_fetcher,
                )

        self.assertEqual(result["downloaded"], 1)


if __name__ == "__main__":
    unittest.main()


class JobRetentionTests(unittest.TestCase):
    def test_launching_a_job_prunes_the_oldest_finished_jobs(self):
        from city_analyser import local_viewer

        class ImmediateThread:
            def __init__(self, target, **_kwargs):
                self.target = target

            def start(self):
                self.target()

        jobs = {f"old-{index}": {"id": f"old-{index}", "status": "complete"} for index in range(local_viewer.MAX_RETAINED_JOBS)}
        jobs["running"] = {"id": "running", "status": "running"}
        server = SimpleNamespace(analysis_jobs=jobs, analysis_lock=threading.Lock())
        with patch("city_analyser.local_viewer.threading.Thread", ImmediateThread):
            result = local_viewer._launch_job(server, "test", "Queued.", lambda update: {"message": "Done."})

        self.assertEqual(result["job"]["status"], "complete")
        self.assertLessEqual(len(jobs), local_viewer.MAX_RETAINED_JOBS)
        self.assertIn("running", jobs)
        self.assertNotIn("old-0", jobs)
