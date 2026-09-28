import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from city_analyser.analysis_cache import analysis_input_sha256
from city_analyser.building_analysis import ANALYSIS_VERSION, _assign_buildings, _edge_is_shared_with_building, _feature_from_osm, _features_from_osm_elements, _segmentation_minimum_confidence, _segmentation_tile_size, _sfm_depth_conflicts, _sfm_depth_samples, _sfm_depth_samples_from_payload, analysis_configuration, building_height, building_height_source, facade_bottom_height, facade_render_height, load_analysis, publish_analysis_artifacts
from city_analyser.panorama_geometry import perspective_tile
from unittest.mock import patch

from city_analyser import building_analysis
from city_analyser import facade_geometry
from city_analyser.analysis_cache import file_sha256

Image = __import__("PIL.Image", fromlist=["Image"])


class BuildingAnalysisTests(unittest.TestCase):
    def test_analysis_cache_is_invalidated_when_sfm_input_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "sample"
            (dataset_dir / "images").mkdir(parents=True)
            image_path = dataset_dir / "images" / "123.jpg"
            Image.fromarray(np.zeros((256, 512, 3), dtype=np.uint8)).save(image_path)
            image = {"id": "123", "local_file": "images/123.jpg", "computed_compass_angle_deg": 10, "computed_geometry": {"type": "Point", "coordinates": [4.9, 52.3]}}
            (dataset_dir / "manifest.json").write_text(json.dumps({"images": [image]}), encoding="utf-8")
            metadata = {
                "version": ANALYSIS_VERSION, "image_id": "123", "sfm_input_sha256": None,
                "input_sha256": analysis_input_sha256(root, "sample", image, analysis_configuration()), "width": 2, "height": 2,
            }
            mask = Image.fromarray(np.zeros((2, 2), dtype=np.uint16))
            import io
            output = io.BytesIO()
            mask.save(output, format="PNG")
            metadata = publish_analysis_artifacts(root, "sample", "123", metadata, output.getvalue(), output.getvalue())

            self.assertEqual(load_analysis(root, "sample", "123"), metadata)
            (dataset_dir / "analysis" / "123-ids.png").write_bytes(b"corrupt")
            self.assertIsNone(load_analysis(root, "sample", "123"))
            metadata = publish_analysis_artifacts(root, "sample", "123", metadata, output.getvalue(), output.getvalue())
            analysis_dir = dataset_dir / "analysis"
            (analysis_dir / "123-sfm.json").write_text(json.dumps({"version": 3, "depth_samples": []}), encoding="utf-8")
            self.assertIsNone(load_analysis(root, "sample", "123"))

    def test_analysis_cache_is_invalidated_when_panorama_bytes_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "sample"
            (dataset_dir / "images").mkdir(parents=True)
            image_path = dataset_dir / "images" / "123.jpg"
            Image.fromarray(np.zeros((256, 512, 3), dtype=np.uint8)).save(image_path)
            image = {"id": "123", "local_file": "images/123.jpg", "computed_compass_angle_deg": 10, "computed_geometry": {"type": "Point", "coordinates": [4.9, 52.3]}}
            (dataset_dir / "manifest.json").write_text(json.dumps({"images": [image]}), encoding="utf-8")
            metadata = {
                "version": ANALYSIS_VERSION, "image_id": "123", "sfm_input_sha256": None,
                "input_sha256": analysis_input_sha256(root, "sample", image, analysis_configuration()), "width": 2, "height": 2,
            }
            import io
            output = io.BytesIO()
            Image.fromarray(np.zeros((2, 2), dtype=np.uint16)).save(output, format="PNG")
            publish_analysis_artifacts(root, "sample", "123", metadata, output.getvalue(), output.getvalue())

            Image.fromarray(np.ones((256, 512, 3), dtype=np.uint8)).save(image_path)

            self.assertIsNone(load_analysis(root, "sample", "123"))

    def test_osm_feature_preserves_identifier_and_tags(self):
        feature = _feature_from_osm(
            {
                "type": "way",
                "id": 42,
                "tags": {"building": "apartments", "height": "18 m"},
                "geometry": [
                    {"lon": 4.8950, "lat": 52.3710},
                    {"lon": 4.8951, "lat": 52.3710},
                    {"lon": 4.8951, "lat": 52.3711},
                ],
            }
        )
        if feature is None:
            self.fail("Expected a valid OSM feature")

        self.assertEqual(feature["properties"]["osm_id"], "way/42")
        self.assertEqual(feature["properties"]["building"], "apartments")
        self.assertEqual(feature["geometry"]["coordinates"][0][0], feature["geometry"]["coordinates"][0][-1])

    def test_height_uses_explicit_value_then_levels_then_default(self):
        self.assertEqual(building_height({"height": "18 m", "building:levels": "2"}), 18.0)
        self.assertEqual(building_height({"building:levels": "4"}), 12.8)
        self.assertEqual(building_height({"building:levels": "1"}), 3.2)
        self.assertAlmostEqual(building_height({"height": "30 ft"}), 9.144)
        self.assertEqual(facade_bottom_height({"building:min_level": "2"}), 6.4)
        self.assertEqual(building_height({}), 10.0)

    def test_osm_multipolygon_preserves_holes_and_suppresses_member_ways(self):
        def path(points):
            return [{"lon": longitude, "lat": latitude} for longitude, latitude in points]

        outer = [(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001), (0, 0)]
        inner = [(0.0002, 0.0002), (0.0008, 0.0002), (0.0008, 0.0008), (0.0002, 0.0008), (0.0002, 0.0002)]
        features = _features_from_osm_elements([
            {"type": "way", "id": 10, "tags": {"building": "yes"}, "geometry": path(outer)},
            {"type": "way", "id": 11, "tags": {}, "geometry": path(inner)},
            {"type": "way", "id": 12, "tags": {"building": "no"}, "geometry": path(outer)},
            {"type": "relation", "id": 20, "tags": {"type": "multipolygon", "building": "apartments"}, "members": [
                {"type": "way", "ref": 10, "role": "outer", "geometry": path(outer)},
                {"type": "way", "ref": 11, "role": "inner", "geometry": path(inner)},
            ]},
        ])

        self.assertEqual([feature["properties"]["osm_id"] for feature in features], ["relation/20"])
        self.assertEqual(features[0]["geometry"]["type"], "Polygon")
        self.assertEqual(len(features[0]["geometry"]["coordinates"]), 2)

    def test_incomplete_relation_does_not_suppress_a_valid_member_building(self):
        geometry = [
            {"lon": 0, "lat": 0}, {"lon": 0.001, "lat": 0},
            {"lon": 0.001, "lat": 0.001}, {"lon": 0, "lat": 0},
        ]
        features = _features_from_osm_elements([
            {"type": "way", "id": 10, "tags": {"building": "yes"}, "geometry": geometry},
            {"type": "relation", "id": 20, "tags": {"type": "multipolygon", "building": "yes"}, "members": [
                {"type": "way", "ref": 10, "role": "outer", "geometry": geometry[:2]},
            ]},
        ])

        self.assertEqual([feature["properties"]["osm_id"] for feature in features], ["way/10"])

    def test_unused_relation_member_does_not_suppress_its_standalone_way(self):
        first = [
            {"lon": 0, "lat": 0}, {"lon": 0.001, "lat": 0},
            {"lon": 0.001, "lat": 0.001}, {"lon": 0, "lat": 0},
        ]
        second = [
            {"lon": 0.002, "lat": 0}, {"lon": 0.003, "lat": 0},
            {"lon": 0.003, "lat": 0.001}, {"lon": 0.002, "lat": 0},
        ]
        features = _features_from_osm_elements([
            {"type": "way", "id": 10, "tags": {"building": "yes"}, "geometry": first},
            {"type": "way", "id": 11, "tags": {"building": "yes"}, "geometry": second},
            {"type": "relation", "id": 20, "tags": {"type": "multipolygon", "building": "yes"}, "members": [
                {"type": "way", "ref": 10, "role": "outer", "geometry": first},
                {"type": "way", "ref": 11, "role": "outer", "geometry": second[:2]},
            ]},
        ])

        self.assertEqual([feature["properties"]["osm_id"] for feature in features], ["relation/20", "way/11"])

    def test_unknown_height_uses_a_tall_render_envelope_without_changing_reported_height(self):
        self.assertEqual(building_height_source({"height": "18"}), "height")
        self.assertEqual(building_height_source({"building:levels": "4"}), "levels")
        self.assertEqual(building_height_source({}), "default")
        self.assertEqual(facade_render_height({"building:levels": "4"}), 12.8)
        self.assertGreater(facade_render_height({}), building_height({}))

    def test_nearer_facade_blocks_a_farther_footprint_on_the_same_ray(self):
        def feature(identifier, west, south, east, north):
            return _feature_from_osm(
                {
                    "type": "way",
                    "id": identifier,
                    "tags": {"building": "yes", "height": "16"},
                    "geometry": [
                        {"lon": west, "lat": south}, {"lon": east, "lat": south},
                        {"lon": east, "lat": north}, {"lon": west, "lat": north},
                    ],
                }
            )

        near = feature(1, -0.00005, 0.0002, 0.00005, 0.0003)
        far = feature(2, -0.00002, 0.0005, 0.00002, 0.0007)
        if near is None or far is None:
            self.fail("Expected valid OSM features")
        buildings = np.ones((1024, 2048), dtype=bool)
        vegetation = np.zeros_like(buildings)

        _, _, metadata = _assign_buildings(buildings, vegetation, [near, far], 0, 0, 0)

        self.assertIn("way/1", {item["osm_id"] for item in metadata.values()})
        self.assertNotIn("way/2", {item["osm_id"] for item in metadata.values()})

    def test_sparse_sfm_depth_marks_only_incompatible_facade_samples(self):
        depth_buffer = np.full((8, 8), 20.0, dtype=np.float32)
        known, foreground, background = _sfm_depth_conflicts(depth_buffer, [
            {"u": 0.5, "v": 0.5, "distance_m": 20.0},
            {"u": 0.25, "v": 0.5, "distance_m": 10.0},
            {"u": 0.75, "v": 0.5, "distance_m": 35.0},
        ])

        self.assertTrue(known[4, 4])
        self.assertFalse(foreground[4, 4])
        self.assertFalse(background[4, 4])
        self.assertTrue(foreground[4, 2])
        self.assertTrue(background[4, 6])

    def test_stale_sfm_depth_samples_are_not_consumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "sample"
            (dataset_dir / "images").mkdir(parents=True)
            (dataset_dir / "analysis").mkdir()
            image = {"id": "123", "local_file": "images/123.jpg", "computed_compass_angle_deg": 10, "computed_geometry": {"type": "Point", "coordinates": [4.9, 52.3]}}
            Image.fromarray(np.zeros((256, 512, 3), dtype=np.uint8)).save(dataset_dir / image["local_file"])
            (dataset_dir / "manifest.json").write_text(json.dumps({"images": [image]}), encoding="utf-8")
            (dataset_dir / "analysis" / "123-sfm.json").write_text(json.dumps({
                "version": 3,
                "image_id": "123",
                "input_sha256": "stale",
                "depth_samples": [{"u": 0.5, "v": 0.5, "distance_m": 10, "track_length": 3, "reprojection_error_px": 1}],
            }), encoding="utf-8")

            self.assertEqual(_sfm_depth_samples(root, "sample", "123"), [])

    def test_validated_remote_sfm_payload_can_supply_depth(self):
        sample = {"u": 0.5, "v": 0.5, "distance_m": 10, "track_length": 3, "reprojection_error_px": 1}
        payload = {"version": 3, "image_id": "123", "depth_samples": [sample]}

        self.assertEqual(_sfm_depth_samples_from_payload(payload, "123"), [sample])
        self.assertEqual(_sfm_depth_samples_from_payload(payload, "other"), [])

    def test_shared_footprint_wall_is_not_a_visible_facade(self):
        def feature(identifier, west, south, east, north):
            return _feature_from_osm(
                {
                    "type": "way",
                    "id": identifier,
                    "tags": {"building": "yes"},
                    "geometry": [
                        {"lon": west, "lat": south}, {"lon": east, "lat": south},
                        {"lon": east, "lat": north}, {"lon": west, "lat": north},
                    ],
                }
            )

        west = feature(20, 0.0, 0.0, 0.0001, 0.0001)
        east = feature(21, 0.0001, 0.0, 0.0002, 0.0001)
        if west is None or east is None:
            self.fail("Expected valid OSM features")

        self.assertTrue(_edge_is_shared_with_building(west, [0.0001, 0.0], [0.0001, 0.0001], [west, east], 0, 0))

    def test_upward_perspective_tile_samples_the_upper_panorama(self):
        panorama = np.zeros((100, 200, 3), dtype=np.uint8)
        _, _, horizon_rows = perspective_tile(panorama, 0, 0, 64, 84)
        _, _, upward_rows = perspective_tile(panorama, 0, np.deg2rad(45), 64, 84)

        self.assertLess(int(np.median(upward_rows)), int(np.median(horizon_rows)))

    def test_assigned_building_keeps_the_geometry_used_for_its_mask(self):
        feature = _feature_from_osm(
            {
                "type": "way",
                "id": 99,
                "tags": {"building": "yes", "height": "12"},
                "geometry": [
                    {"lon": -0.0001, "lat": 0.0001}, {"lon": 0.0001, "lat": 0.0001},
                    {"lon": 0.0001, "lat": 0.0003}, {"lon": -0.0001, "lat": 0.0003},
                ],
            }
        )
        if feature is None:
            self.fail("Expected a valid footprint")
        buildings = np.ones((512, 1024), dtype=bool)
        _, _, metadata = _assign_buildings(buildings, np.zeros_like(buildings), [feature], 0, 0, 0)

        self.assertEqual(metadata["1"]["footprint_geometry"], feature["geometry"])

    def test_custom_segmentation_tile_size_is_bounded(self):
        previous = os.environ.get("BUILDING_ANALYSIS_TILE_SIZE")
        os.environ["BUILDING_ANALYSIS_TILE_SIZE"] = "320"
        try:
            self.assertEqual(_segmentation_tile_size("segformer"), 320)
            os.environ["BUILDING_ANALYSIS_TILE_SIZE"] = "128"
            with self.assertRaises(RuntimeError):
                _segmentation_tile_size("segformer")
        finally:
            if previous is None:
                os.environ.pop("BUILDING_ANALYSIS_TILE_SIZE", None)
            else:
                os.environ["BUILDING_ANALYSIS_TILE_SIZE"] = previous

    def test_segmentation_confidence_is_bounded(self):
        previous = os.environ.get("BUILDING_ANALYSIS_MIN_CONFIDENCE")
        os.environ["BUILDING_ANALYSIS_MIN_CONFIDENCE"] = "0.45"
        try:
            self.assertEqual(_segmentation_minimum_confidence(), 0.45)
            os.environ["BUILDING_ANALYSIS_MIN_CONFIDENCE"] = "0"
            with self.assertRaises(RuntimeError):
                _segmentation_minimum_confidence()
        finally:
            if previous is None:
                os.environ.pop("BUILDING_ANALYSIS_MIN_CONFIDENCE", None)
            else:
                os.environ["BUILDING_ANALYSIS_MIN_CONFIDENCE"] = previous


class FacadeGeometryTests(unittest.TestCase):
    @staticmethod
    def square(identifier, west, south, east, north, height="12"):
        feature = _feature_from_osm({
            "type": "way", "id": identifier, "tags": {"building": "yes", "height": height},
            "geometry": [
                {"lon": west, "lat": south}, {"lon": east, "lat": south},
                {"lon": east, "lat": north}, {"lon": west, "lat": north},
            ],
        })
        assert feature is not None
        return feature

    def test_translated_scene_matches_a_fresh_projection_at_the_shifted_camera(self):
        features = [self.square(1, -0.0001, 0.0002, 0.0001, 0.0003), self.square(2, 0.0003, -0.0001, 0.0004, 0.0001)]
        latitude, longitude = building_analysis._shift_camera_position(0.0, 0.0, 4.0, -4.0)
        translated = facade_geometry.build_scene(features, 0.0, 0.0).translated(4.0, -4.0)
        fresh = facade_geometry.build_scene(features, latitude, longitude)

        np.testing.assert_allclose(translated.starts, fresh.starts, atol=1e-3)
        moved = building_analysis._render_scene_owners(translated, facade_geometry.visible_facades(translated), 10.0, 512, 256)[0]
        projected = building_analysis._render_facade_owners(features, latitude, longitude, 10.0, 512, 256)[0]
        self.assertLessEqual(int((moved != projected).sum()), 2)

    def test_nearest_wall_distance_uses_the_closest_edge(self):
        scene = facade_geometry.build_scene([self.square(1, -0.0001, 0.0002, 0.0001, 0.0003)], 0.0, 0.0)

        self.assertAlmostEqual(float(facade_geometry.nearest_wall_distances(scene)[0]), 22.1, delta=0.2)

    def test_nearby_buildings_keeps_only_walls_within_the_query_radius(self):
        near = self.square(1, 0.0005, 0.0005, 0.0006, 0.0006)
        far = self.square(2, 0.0020, 0.0020, 0.0021, 0.0021)
        with patch("city_analyser.building_analysis._cell_buildings", return_value=(near, far)):
            result = building_analysis.nearby_buildings(0.00001, 0.00001)

        self.assertEqual([feature["properties"]["osm_id"] for feature in result], ["way/1"])

    def test_default_tile_size_follows_each_model_input_resolution(self):
        with patch.dict(os.environ, {"BUILDING_ANALYSIS_TILE_SIZE": ""}):
            self.assertEqual(_segmentation_tile_size("segformer"), 512)
            self.assertEqual(_segmentation_tile_size("mask2former"), 384)

    def test_memoized_file_hash_follows_rewrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            path.write_bytes(b"first")
            first = file_sha256(path)
            path.write_bytes(b"second-content")

            self.assertNotEqual(file_sha256(path), first)
