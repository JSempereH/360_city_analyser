import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from city_analyser import building_analysis, tile_cache

PNG = b"\x89PNG\r\n\x1a\nrest-of-a-tile"


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def read(self, _limit):
        return self.content

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class TileCacheTests(unittest.TestCase):
    def test_a_tile_is_fetched_once_kept_and_served_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(tile_cache, "urlopen", return_value=FakeResponse(PNG)) as fetch:
                self.assertEqual(tile_cache.tile(root, 16, 32000, 24000), PNG)
                self.assertIn("360-city-analyser", fetch.call_args.args[0].get_header("User-agent"))
            with patch.object(tile_cache, "urlopen", side_effect=AssertionError("served from disk")), \
                    patch.dict(os.environ, {"CITY_ANALYSER_OFFLINE": "1"}):
                self.assertEqual(tile_cache.tile(root, 16, 32000, 24000), PNG)
                self.assertIsNone(tile_cache.tile(root, 16, 32001, 24000))
            self.assertIsNone(tile_cache.tile(root, 3, 8, 0))  # outside the zoom level's grid
            with patch.object(tile_cache, "urlopen", return_value=FakeResponse(b"<html>blocked</html>")):
                self.assertIsNone(tile_cache.tile(root, 16, 1, 1))


class StaleFootprintTests(unittest.TestCase):
    def test_an_expired_snapshot_is_used_when_no_service_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            path = cache / "40430_-3694.json"
            path.write_text(json.dumps({"version": building_analysis.OSM_CACHE_VERSION, "features": [{"id": "way/1"}]}))
            old = time.time() - 400 * 86_400
            os.utime(path, (old, old))
            building_analysis._cell_buildings.cache_clear()
            try:
                with patch.object(building_analysis, "_osm_cache_directory", return_value=cache), \
                        patch.object(building_analysis, "_download_buildings", side_effect=RuntimeError("offline")):
                    self.assertEqual(building_analysis._cell_buildings(40430, -3694), ({"id": "way/1"},))
                    with self.assertRaises(RuntimeError):
                        building_analysis._cell_buildings(40431, -3694)
            finally:
                building_analysis._cell_buildings.cache_clear()


if __name__ == "__main__":
    unittest.main()
