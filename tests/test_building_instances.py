import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from city_analyser import building_analysis
from city_analyser import building_instances
from city_analyser import facade_geometry
from city_analyser.building_analysis import _feature_from_osm
from evaluation.metrics import separation_scores
from city_analyser.panorama_geometry import panorama_to_tile_grid, perspective_tile


def square(identifier, west, south, east, north):
    feature = _feature_from_osm({
        "type": "way", "id": identifier, "tags": {"building": "yes", "height": "12"},
        "geometry": [
            {"lon": west, "lat": south}, {"lon": east, "lat": south},
            {"lon": east, "lat": north}, {"lon": west, "lat": north},
        ],
    })
    assert feature is not None
    return feature


class InstanceLabelTests(unittest.TestCase):
    def test_instance_dominated_by_one_footprint_takes_its_name_over_its_whole_extent(self):
        owners = np.zeros((6, 8), dtype=np.uint16)
        owners[3:, 1:7] = 1  # projection stops below the real roof line
        instances = np.zeros((6, 8), dtype=np.int32)
        instances[1:, 1:7] = 1
        evidence = instances > 0

        labels, details = building_analysis._instance_labels(instances, evidence, owners, bins=2)

        self.assertEqual(details[0]["mode"], "whole")
        self.assertTrue((labels[1:, 1:7] == 1).all())

    def test_merged_attached_facades_are_split_by_columns_over_the_full_height(self):
        owners = np.zeros((6, 12), dtype=np.uint16)
        owners[3:, :6] = 1
        owners[3:, 6:] = 2
        instances = np.ones((6, 12), dtype=np.int32)

        labels, details = building_analysis._instance_labels(instances, instances > 0, owners, bins=3)

        self.assertEqual(details[0]["mode"], "columns")
        self.assertEqual(details[0]["ordinals"], [1, 2])
        self.assertTrue((labels[:, :6] == 1).all() and (labels[:, 6:] == 2).all())

    def test_instance_mostly_outside_projected_footprints_abstains(self):
        owners = np.zeros((4, 10), dtype=np.uint16)
        owners[:, 0] = 1
        instances = np.ones((4, 10), dtype=np.int32)

        labels, details = building_analysis._instance_labels(instances, instances > 0, owners, bins=2)

        self.assertEqual(details[0]["mode"], "abstain")
        self.assertFalse(labels.any())

    def test_narrow_facade_slivers_join_a_wider_neighbour(self):
        owners = np.array([1] * 10 + [2] * 2 + [3] * 10)
        columns = np.arange(len(owners))

        building_analysis._absorb_narrow_runs(owners, columns, minimum=4)

        self.assertNotIn(2, set(owners.tolist()))
        self.assertEqual(owners[:10].tolist(), [1] * 10)

    def test_upward_continuation_fills_unclaimed_building_pixels_only(self):
        labels = np.zeros((5, 3), dtype=np.uint16)
        labels[3:, :] = 7
        evidence = np.ones((5, 3), dtype=bool)
        evidence[0, 2] = False
        owners = np.zeros((5, 3), dtype=np.uint16)
        owners[1, 1] = 9  # another projected footprint claims this pixel

        continued = building_analysis._continue_upward(labels, evidence, owners, np.ones((5, 3), dtype=bool))

        self.assertTrue((continued[:3, 0] == 7).all())
        self.assertEqual(continued[1, 1], 0)
        self.assertEqual(continued[0, 2], 0)


class GeometryTests(unittest.TestCase):
    def test_portrait_tiles_round_trip_and_reach_high_elevations(self):
        width, height = 512, 256
        panorama = np.random.default_rng(0).random((height, width)).astype(np.float32)
        yaw, pitch = 1.0, math.radians(building_instances.INSTANCE_PITCH_DEGREES)
        _, source_x, source_y = perspective_tile(panorama, yaw, pitch, 64, 90, 96)
        grid, visible = panorama_to_tile_grid(torch, width, height, yaw, pitch, 90, "cpu", 96 / 64)
        sampled = grid[0].numpy()[source_y, source_x]
        expected_x = (np.arange(64) + 0.5) / 64 * 2 - 1
        expected_y = (np.arange(96) + 0.5) / 96 * 2 - 1

        self.assertLess(np.abs(sampled[..., 0] - expected_x[None, :]).max() * 32, 1.0)
        self.assertLess(np.abs(sampled[..., 1] - expected_y[:, None]).max() * 48, 1.0)
        top_row = np.flatnonzero(visible.numpy().any(axis=1))[0]
        self.assertGreater((0.5 - (top_row + 0.5) / height) * 180, 80)

    def test_instance_views_cover_the_whole_horizon(self):
        settings = building_instances.InstanceSettings(enabled=True, views=6, field_of_view_degrees=90.0, detector_size=800)
        coverage = building_instances.instance_coverage(settings, 256, 128)

        self.assertTrue(coverage[64].all())

    def test_courtyards_are_not_used_as_repaired_camera_positions(self):
        # A closed ring of buildings around a small courtyard, the camera in its wall.
        ring = [square(1, -0.0003, -0.0003, 0.0003, -0.0001), square(2, -0.0003, 0.0001, 0.0003, 0.0003),
                square(3, -0.0003, -0.0001, -0.0001, 0.0001), square(4, 0.00005, -0.0001, 0.0003, 0.0001)]
        scene = facade_geometry.build_scene(ring, 0.0, -0.00012)
        offset = facade_geometry.free_space_offset(scene, np.ones(len(ring), dtype=bool), 1.0, 12.0)

        self.assertIsNone(offset)


class DeviceAndPoseTests(unittest.TestCase):
    def test_explicit_cuda_device_fails_clearly_without_a_gpu(self):
        fake_torch = SimpleNamespace(device=torch.device, cuda=SimpleNamespace(is_available=lambda: False), backends=SimpleNamespace())
        with patch.dict(os.environ, {"BUILDING_ANALYSIS_DEVICE": "cuda"}):
            with self.assertRaises(RuntimeError):
                building_analysis.inference_device(fake_torch)
        with patch.dict(os.environ, {"BUILDING_ANALYSIS_DEVICE": "auto"}):
            self.assertEqual(building_analysis.inference_device(fake_torch).type, "cpu")

    def test_smooth_optimum_is_not_reported_as_ambiguous(self):
        feature = square(5, -0.0001, 0.0002, 0.0001, 0.0003)
        width, height = 512, 256
        owners, _, _, _ = building_analysis._render_facade_owners([feature], 0.0, 0.0, 0.0, width, height)
        pose = building_analysis._refine_camera_pose(owners > 0, [feature], 0.0, 0.0, 6.0)

        self.assertNotEqual(pose["reason"], "ambiguous_best_pose")


class SeparationMetricTests(unittest.TestCase):
    def test_merged_buildings_lower_purity_and_count_as_a_merge(self):
        truth = np.zeros((10, 40), dtype=np.int32)
        truth[:, :20], truth[:, 20:] = 1, 2
        merged = np.ones_like(truth)
        separate = truth.copy()

        self.assertEqual(separation_scores(truth, separate)["separation_f"], 1.0)
        scores = separation_scores(truth, merged)
        self.assertAlmostEqual(scores["purity"], 0.5)
        self.assertEqual(scores["merges"], 0)  # below the sliver threshold on this tiny raster

    def test_upper_band_coverage_detects_cut_roof_lines(self):
        truth = np.zeros((40, 10), dtype=np.int32)
        truth[:, :] = 1
        predicted = truth.copy()
        predicted[:10] = 0

        self.assertEqual(separation_scores(truth, predicted)["top_detected"], 0.0)


if __name__ == "__main__":
    unittest.main()
