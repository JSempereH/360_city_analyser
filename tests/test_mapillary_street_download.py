import json
import hashlib
import io
import math
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from PIL import Image

from city_analyser.mapillary_street_download import (
    available_image_url,
    bbox_from_center,
    camera_up,
    download_image,
    fetch_images,
    haversine_m,
    image_record,
    normalize_compass_heading,
    point_from_image,
    select_panoramas,
)


def panorama(image_id, lon, lat, captured_at=1):
    return {
        "id": image_id,
        "is_pano": True,
        "captured_at": captured_at,
        "computed_geometry": {"type": "Point", "coordinates": [lon, lat]},
    }


def image_bytes(size=(200, 100)):
    output = io.BytesIO()
    Image.new("RGB", size, "blue").save(output, format="JPEG")
    return output.getvalue()


class FakeResponse:
    def __init__(self, body, content_length=None):
        self.body = io.BytesIO(body)
        self.headers = Message()
        self.headers["Content-Type"] = "image/jpeg"
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, size=-1):
        return self.body.read(size)


class MapillaryStreetDownloadTests(unittest.TestCase):
    def test_center_bbox_contains_center(self):
        min_lon, min_lat, max_lon, max_lat = bbox_from_center(9.9281, -84.0907, 150)
        self.assertLess(min_lat, 9.9281)
        self.assertGreater(max_lat, 9.9281)
        self.assertLess(min_lon, -84.0907)
        self.assertGreater(max_lon, -84.0907)

    def test_haversine_at_equator(self):
        self.assertAlmostEqual(haversine_m(0, 0, 0, 0.001), 111.2, delta=1)

    def test_selection_filters_non_panos_and_near_duplicates(self):
        images = [
            panorama("one", -84.09, 9.92, 1),
            panorama("two", -84.09001, 9.92001, 2),
            panorama("three", -84.089, 9.92, 3),
            {"id": "flat", "is_pano": False, "geometry": {"type": "Point", "coordinates": [-84.08, 9.92]}},
        ]
        selected = select_panoramas(images, None, None, min_spacing_m=50, max_images=10)
        self.assertEqual([item["id"] for item in selected], ["one", "three"])

    def test_download_falls_back_to_an_available_smaller_derivative(self):
        image = {"thumb_1024_url": "https://example.test/pano-1024.jpg"}

        self.assertEqual(available_image_url(image, "original"), ("thumb_1024_url", "https://example.test/pano-1024.jpg"))
        self.assertEqual(available_image_url(image, "auto"), ("thumb_1024_url", "https://example.test/pano-1024.jpg"))

    def test_unbounded_fetch_follows_all_mapillary_pages(self):
        with patch("city_analyser.mapillary_street_download.request_json", side_effect=[
            {"data": [{"id": "one"}], "paging": {"next": "https://example.test/page-2"}},
            {"data": [{"id": "two"}], "paging": {}},
        ]) as request_json:
            images = fetch_images("token", (4.8, 52.3, 4.9, 52.4), (), None)

        self.assertEqual([image["id"] for image in images], ["one", "two"])
        self.assertEqual(request_json.call_count, 2)
        self.assertEqual(request_json.call_args_list[0].args[1]["is_pano"], "true")

    def test_download_verifies_image_and_returns_local_metadata(self):
        body = image_bytes()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "pano.jpg"
            stale_part = destination.with_suffix(".part")
            stale_part.write_bytes(b"unrelated")

            with patch("city_analyser.mapillary_street_download.urlopen", return_value=FakeResponse(body)):
                downloaded, metadata = download_image("https://example.test/pano.jpg", destination, overwrite=False)

            self.assertTrue(downloaded)
            self.assertEqual(metadata, {"width": 200, "height": 100, "sha256": hashlib.sha256(body).hexdigest()})
            self.assertEqual(destination.read_bytes(), body)
            self.assertEqual(stale_part.read_bytes(), b"unrelated")
            self.assertEqual(list(Path(directory).glob(".pano.jpg.*.part")), [])

    def test_download_enforces_streaming_size_limit_despite_content_length(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "pano.jpg"
            response = FakeResponse(b"x" * 11, content_length=1)

            with patch("city_analyser.mapillary_street_download.MAX_IMAGE_BYTES", 10), patch("city_analyser.mapillary_street_download.urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "larger than 50 MB"):
                    download_image("https://example.test/pano.jpg", destination, overwrite=False)

            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_download_rejects_corrupt_and_non_equirectangular_images(self):
        for body, message in ((b"not an image", "corrupt or unsupported"), (image_bytes((100, 100)), "not equirectangular")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "pano.jpg"
                with patch("city_analyser.mapillary_street_download.urlopen", return_value=FakeResponse(body)):
                    with self.assertRaisesRegex(RuntimeError, message):
                        download_image("https://example.test/pano.jpg", destination, overwrite=False)
                self.assertFalse(destination.exists())
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_image_record_uses_local_metadata_and_normalized_heading(self):
        image = panorama("one", -84.09, 9.92)
        image.update({"width": 1, "height": 1, "computed_compass_angle": -10, "sequence": "sequence-1", "_selected_geometry": image["computed_geometry"]})
        metadata = {"width": 200, "height": 100, "sha256": "abc123"}

        record = image_record(image, "images/pano-one.jpg", metadata)

        self.assertEqual(record["width"], 200)
        self.assertEqual(record["height"], 100)
        self.assertEqual(record["sha256"], "abc123")
        self.assertEqual(record["computed_compass_angle_deg"], 350.0)
        self.assertEqual(record["sequence_id"], "sequence-1")

    def test_point_rejects_boolean_nonfinite_and_out_of_range_coordinates(self):
        for coordinates in ([True, False], [math.nan, 0], [181, 0], [0, 91]):
            with self.subTest(coordinates=coordinates):
                self.assertIsNone(point_from_image({"computed_geometry": {"type": "Point", "coordinates": coordinates}}))

    def test_compass_heading_must_be_finite(self):
        for value in (math.nan, math.inf, -math.inf, "10", True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite number"):
                normalize_compass_heading(value)


if __name__ == "__main__":
    unittest.main()


class CameraTiltTests(unittest.TestCase):
    def test_world_up_in_the_panorama_frame_from_the_computed_rotation(self):
        # Mapillary 1986345691512470, Valencia: its 3D reconstruction has the camera tilted 18.5 degrees.
        up = camera_up([0.40423832881247, 1.6357406040857, -2.2891004430312])
        self.assertEqual([round(value, 3) for value in up], [-0.054, 0.948, 0.313])
        # A camera rotated a quarter turn about its right axis from looking straight up looks at the horizon, level.
        self.assertEqual([round(value, 6) + 0.0 for value in camera_up([1.5707963267948966, 0, 0])], [0.0, 1.0, 0.0])
        self.assertIsNone(camera_up(None))
        self.assertIsNone(camera_up([1, "x", 0]))


class CityScaleDownloadTests(unittest.TestCase):
    def test_tiles_cover_the_box_with_small_areas(self):
        from city_analyser.mapillary_street_download import MAX_TILE_AREA_DEG2, tile_bbox

        box = (-0.45, 39.40, -0.30, 39.55)
        tiles = tile_bbox(box)
        self.assertGreater(len(tiles), 1)
        for west, south, east, north in tiles:
            self.assertLess((east - west) * (north - south), MAX_TILE_AREA_DEG2 + 1e-12)
        self.assertAlmostEqual(sum((e - w) * (n - s) for w, s, e, n in tiles), (box[2] - box[0]) * (box[3] - box[1]), places=9)
        self.assertEqual((min(t[0] for t in tiles), min(t[1] for t in tiles), max(t[2] for t in tiles), max(t[3] for t in tiles)), box)
        self.assertEqual(tile_bbox((0.0, 0.0, 0.05, 0.05)), [(0.0, 0.0, 0.05, 0.05)])

    def test_grid_selection_matches_the_all_pairs_rule_and_deduplicates_across_tiles(self):
        import random

        from city_analyser.mapillary_street_download import PanoramaSelector, select_panoramas

        rng = random.Random(7)
        images = [
            {"id": str(i), "is_pano": True, "captured_at": rng.randint(0, 20),
             "computed_geometry": {"type": "Point", "coordinates": [4.9 + rng.random() * 0.004, 52.3 + rng.random() * 0.004]}}
            for i in range(600)
        ]
        everything = [item["id"] for item in select_panoramas(images, None, None, 12, 10_000)]
        expected, kept = [], []
        for image in sorted(images, key=lambda item: (item["captured_at"], item["id"])):
            lon, lat = image["computed_geometry"]["coordinates"]
            if all(haversine_m(lat, lon, o[1], o[0]) >= 12 for o in kept):
                kept.append((lon, lat))
                expected.append(image["id"])
        self.assertEqual(everything, expected)
        selector = PanoramaSelector(None, None, 12, 10_000)
        first = selector.add(images[:300])
        again = selector.add(images[:300])  # tile borders return the same panoramas twice
        self.assertEqual(again, [])
        self.assertTrue(first)
        self.assertEqual(len(PanoramaSelector(None, None, 0, 5).add(images)), 5)  # the cap applies

    def test_downloads_retry_transient_errors_only(self):
        from city_analyser import mapillary_street_download as module

        calls = []

        def flaky(url, destination, overwrite):
            calls.append(url)
            if len(calls) < 3:
                raise HTTPError(url, 503, "busy", None, None)
            return True, {"ok": True}

        with patch.object(module, "download_image", flaky), patch.object(module, "DOWNLOAD_BACKOFF_S", 0.0):
            self.assertEqual(module.download_with_retries("u", Path("x"), False), (True, {"ok": True}))
        self.assertEqual(len(calls), 3)

        def forbidden(url, destination, overwrite):
            calls.append("403")
            raise HTTPError(url, 403, "expired", None, None)

        calls.clear()
        with patch.object(module, "download_image", forbidden), patch.object(module, "DOWNLOAD_BACKOFF_S", 0.0), self.assertRaises(HTTPError):
            module.download_with_retries("u", Path("x"), False)
        self.assertEqual(calls, ["403"])

    def test_tiled_run_downloads_in_parallel_and_writes_the_manifest(self):
        import sys
        import tempfile

        from city_analyser import mapillary_street_download as module

        def api_image(number: int, lon: float) -> dict:
            return {
                "id": str(number), "is_pano": True, "captured_at": number, "computed_compass_angle": 10.0, "creator": {"username": "u"},
                "computed_geometry": {"type": "Point", "coordinates": [lon, 52.3]}, "thumb_2048_url": f"https://img.test/{number}.jpg",
                "geometry": {"type": "Point", "coordinates": [lon, 52.3]}, "width": 2000, "height": 1000,
            }

        def fetch(token, tile, fields, pages):
            # every tile also returns one panorama on the shared border, as the real endpoint does
            return [api_image(1, 4.9), api_image(int(tile[0] * 1000) + 100, tile[0] + 0.01)]

        def fake_download(url, destination, overwrite):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"jpeg")
            return True, {"local_file_size_bytes": 4}

        with tempfile.TemporaryDirectory() as directory:
            argv = ["prog", "--tiled-bbox", "4.0,52.0,4.4,52.4", "--output", directory, "--min-spacing-m", "1", "--max-images", "100", "--workers", "3", "--access-token", "t"]
            with patch.object(sys, "argv", argv), patch.object(module, "fetch_images", fetch), patch.object(module, "download_with_retries", fake_download), \
                    patch.object(module, "image_record", lambda image, local, meta: {"id": image["id"], "local_file": local, "selected_geometry": image["_selected_geometry"], "creator": image["creator"], "source_page": "s"}), \
                    patch.object(module, "status"):
                self.assertEqual(module.main(), 0)
            manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
        ids = [item["id"] for item in manifest["images"]]
        self.assertEqual(ids.count("1"), 1)  # deduplicated across tiles
        self.assertGreater(len(ids), 2)
        self.assertGreater(manifest["query"]["tiles"], 1)
