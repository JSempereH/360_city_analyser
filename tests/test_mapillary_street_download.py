import hashlib
import io
import math
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from city_analyser.mapillary_street_download import (
    available_image_url,
    bbox_from_center,
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
