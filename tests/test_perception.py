import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from city_analyser import building_analysis
from city_analyser import building_instances
from city_analyser.panorama_geometry import perspective_tile, perspective_tiles_torch
from city_analyser.stage_cache import STAGE_CACHE_DIRECTORY, load_stage, save_stage

Image = __import__("PIL.Image", fromlist=["Image"])


class PerceptionCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.image_path = self.root / "sample" / "images" / "1.jpg"
        self.image_path.parent.mkdir(parents=True)
        Image.fromarray(np.full((256, 512, 3), 90, dtype=np.uint8)).save(self.image_path)
        self.model_root = self.root / "models"
        self.calls = {"semantic": 0, "instances": 0}
        environment = patch.dict(os.environ, {
            "BUILDING_ANALYSIS_DEVICE": "cpu", "BUILDING_ANALYSIS_MODEL": "balanced",
            "BUILDING_ANALYSIS_INSTANCES": "on", "BUILDING_ANALYSIS_STAGE_CACHE": "on", "BUILDING_ANALYSIS_STAGE_CACHE_DIR": "",
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.addCleanup(self.directory.cleanup)

    def fake_semantic(self, panorama, model_root, progress):
        self.calls["semantic"] += 1
        building = np.zeros(panorama.shape[:2], dtype=np.uint8)
        building[:, :100] = 200
        return building, np.zeros_like(building), "cpu", "fake/model", "abc"

    def fake_instances(self, panorama, model_root, device, settings, progress):
        self.calls["instances"] += 1
        instance_map = np.zeros(panorama.shape[:2], dtype=np.uint16)
        instance_map[:, :100] = 1
        return instance_map, [{"instance": 1, "score": 0.9, "views": [0], "pixels": int((instance_map > 0).sum())}]

    def perceive(self):
        with patch.object(building_analysis, "_semantic_confidences", self.fake_semantic), \
                patch.object(building_analysis, "segment_instances", self.fake_instances):
            return building_analysis.perceive_panorama(self.root, self.model_root, self.image_path, lambda _: None)

    def test_second_run_reuses_both_model_stages(self):
        first = self.perceive()
        second = self.perceive()

        self.assertEqual(self.calls, {"semantic": 1, "instances": 1})
        np.testing.assert_array_equal(first.building_confidence, second.building_confidence)
        np.testing.assert_array_equal(first.instance_map, second.instance_map)
        self.assertEqual(second.detected_instances, first.detected_instances)
        self.assertEqual((second.model_id, second.model_revision, second.device), ("fake/model", "abc", "cpu"))

    def test_association_settings_do_not_invalidate_model_stages(self):
        self.perceive()
        with patch.dict(os.environ, {"BUILDING_ANALYSIS_MIN_CONFIDENCE": "0.9"}), \
                patch.object(building_analysis, "ANALYSIS_VERSION", building_analysis.ANALYSIS_VERSION + 1):
            self.perceive()

        self.assertEqual(self.calls, {"semantic": 1, "instances": 1})

    def test_model_and_view_settings_invalidate_their_stage(self):
        self.perceive()
        with patch.dict(os.environ, {"BUILDING_ANALYSIS_MODEL": "b0"}):
            self.perceive()
        self.assertEqual(self.calls, {"semantic": 2, "instances": 1})
        with patch.dict(os.environ, {"BUILDING_ANALYSIS_TILE_SIZE": "384"}):
            self.perceive()
        self.assertEqual(self.calls, {"semantic": 3, "instances": 1})
        with patch.dict(os.environ, {"BUILDING_ANALYSIS_INSTANCE_VIEWS": "8"}):
            self.perceive()
        self.assertEqual(self.calls, {"semantic": 3, "instances": 2})

    def test_downloaded_model_revision_is_part_of_the_key(self):
        self.perceive()
        reference = self.model_root / "models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024" / "refs" / "main"
        reference.parent.mkdir(parents=True)
        reference.write_text("f" * 40, encoding="utf-8")
        self.perceive()

        self.assertEqual(self.calls["semantic"], 2)

    def test_changed_panorama_bytes_miss_the_cache(self):
        self.perceive()
        Image.fromarray(np.full((256, 512, 3), 91, dtype=np.uint8)).save(self.image_path)
        self.perceive()

        self.assertEqual(self.calls, {"semantic": 2, "instances": 2})

    def test_disabled_cache_always_runs_the_models(self):
        with patch.dict(os.environ, {"BUILDING_ANALYSIS_STAGE_CACHE": "off"}):
            self.perceive()
            self.perceive()

        self.assertEqual(self.calls, {"semantic": 2, "instances": 2})
        self.assertFalse((self.root / STAGE_CACHE_DIRECTORY).exists())

    def test_configured_directory_holds_the_cache(self):
        shared = self.root / "shared-cache"
        with patch.dict(os.environ, {"BUILDING_ANALYSIS_STAGE_CACHE_DIR": str(shared)}):
            self.perceive()
            self.perceive()

        self.assertEqual(self.calls, {"semantic": 1, "instances": 1})
        self.assertTrue(any(shared.glob("semantic/*/*.npz")))
        self.assertFalse((self.root / STAGE_CACHE_DIRECTORY).exists())

    def test_truncated_entry_is_a_miss(self):
        save_stage(self.root, "semantic", "ab" + "0" * 62, building=np.ones((2, 2), dtype=np.uint8))
        path = self.root / STAGE_CACHE_DIRECTORY / "semantic" / "ab" / ("ab" + "0" * 62 + ".npz")
        path.write_bytes(path.read_bytes()[:20])

        self.assertIsNone(load_stage(self.root, "semantic", "ab" + "0" * 62))

    def test_minimum_confidence_applies_to_quantized_confidences(self):
        confidence = np.array([[76, 77, 255]], dtype=np.uint8)
        with patch.dict(os.environ, {"BUILDING_ANALYSIS_MIN_CONFIDENCE": "0.30"}):
            np.testing.assert_array_equal(building_analysis._confident(confidence), [[False, True, True]])


class PreloadTests(unittest.TestCase):
    def test_preload_loads_semantic_and_instance_models(self):
        with patch.dict(os.environ, {"BUILDING_ANALYSIS_DEVICE": "cpu", "BUILDING_ANALYSIS_MODEL": "b0", "BUILDING_ANALYSIS_INSTANCES": "on"}), \
                patch.object(building_analysis, "_load_segmentation_model") as semantic, \
                patch.object(building_analysis, "_load_instance_models") as instances:
            loaded = building_analysis.preload_models(Path("models"))

        semantic.assert_called_once()
        instances.assert_called_once()
        self.assertEqual(loaded["device"], "cpu")


class TileSamplingTests(unittest.TestCase):
    def test_batched_tiles_match_the_reference_projection_across_the_seam(self):
        height, width = 256, 512
        rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        panorama = np.stack([
            128 + 100 * np.sin(2 * np.pi * columns / width),
            128 + 100 * np.cos(np.pi * rows / height),
            128 + 60 * np.sin(4 * np.pi * columns / width) * np.cos(np.pi * rows / height),
        ], axis=-1).astype(np.uint8)
        source = torch.from_numpy(panorama).permute(2, 0, 1)[None].float()
        views = [(0.0, 0.0), (math.radians(179.9), 0.0), (2.0, math.radians(45))]

        tiles = perspective_tiles_torch(torch, source, views, 96, 84, 144)

        for index, (yaw, pitch) in enumerate(views):
            reference = perspective_tile(panorama, yaw, pitch, 96, 84, 144)[0].astype(float)
            sampled = tiles[index].permute(1, 2, 0).numpy()
            self.assertLess(np.abs(reference - sampled).mean(), 1.0)


class MaskMergeTests(unittest.TestCase):
    def reference_groups(self, masks, views, visibility):
        union_find = building_instances._UnionFind(len(masks))
        for first in range(len(masks)):
            for second in range(first + 1, len(masks)):
                if views[first] == views[second]:
                    continue
                shared = visibility[views[first]] & visibility[views[second]]
                first_part, second_part = masks[first] & shared, masks[second] & shared
                first_size, second_size = int(first_part.sum()), int(second_part.sum())
                if min(first_size, second_size) < building_instances.MERGE_MINIMUM_OVERLAP_PIXELS:
                    continue
                intersection = int((first_part & second_part).sum())
                if intersection / (first_size + second_size - intersection) >= building_instances.MERGE_OVERLAP_IOU:
                    union_find.union(first, second)
        groups = {}
        for index in range(len(masks)):
            groups.setdefault(union_find.find(index), []).append(index)
        return sorted(sorted(group) for group in groups.values())

    def test_cropped_comparison_matches_the_full_resolution_reference(self):
        generator = np.random.default_rng(7)
        height, width = 120, 240
        visibility = []
        for view in range(4):
            visible = torch.zeros((height, width), dtype=torch.bool)
            start = view * 60 - 20
            for column in range(start, start + 100):
                visible[:, column % width] = True  # view 0 wraps across the seam
            visibility.append(visible)
        masks, views = [], []
        for view in range(4):
            for _ in range(5):
                top, left = int(generator.integers(0, 80)), int(generator.integers(0, width))
                mask = torch.zeros((height, width), dtype=torch.bool)
                for column in range(left, left + int(generator.integers(30, 90))):
                    mask[top:top + 40, column % width] = True
                masks.append(mask & visibility[view])
                views.append(view)

        groups = building_instances._merge_groups(masks, views, visibility)

        self.assertEqual(sorted(sorted(group) for group in groups), self.reference_groups(masks, views, visibility))
        self.assertLess(len(groups), len(masks))


class BatchAnalysisTests(unittest.TestCase):
    def test_every_valid_panorama_is_perceived_and_associated_in_order(self):
        from city_analyser import batch_analysis

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample" / "images").mkdir(parents=True)
            images = []
            for identifier in ("1", "2", "3"):
                Image.fromarray(np.zeros((256, 512, 3), dtype=np.uint8)).save(root / "sample" / "images" / f"{identifier}.jpg")
                images.append({
                    "id": identifier, "local_file": f"images/{identifier}.jpg", "computed_compass_angle_deg": 10,
                    "computed_geometry": {"type": "Point", "coordinates": [4.9, 52.3]},
                })
            images[1]["computed_compass_angle_deg"] = None  # invalid heading: reported, not analyzed
            (root / "sample" / "manifest.json").write_text(json.dumps({"images": images}), encoding="utf-8")
            associated = []

            def associate(data_root, dataset_id, image, inputs, perception, features, progress, sfm_refinement=None):
                associated.append((image["id"], perception, features))
                return {"building_ids": {"way/1": {}}}

            with patch.object(batch_analysis, "perceive_panorama", side_effect=lambda *args: f"perception-{args[2].stem}"), \
                    patch.object(batch_analysis, "associate_panorama", side_effect=associate), \
                    patch.object(batch_analysis, "nearby_buildings", return_value=["footprint"]), \
                    patch("builtins.print"):
                report = batch_analysis.analyze_dataset(root, root / "models", "sample")

        self.assertEqual([item[:2] for item in associated], [("1", "perception-1"), ("3", "perception-3")])
        self.assertEqual([row["image"] for row in report["rows"]], ["2", "1", "3"])
        self.assertIn("heading", report["rows"][0]["error"])
        self.assertEqual(report["rows"][1]["buildings"], 1)


if __name__ == "__main__":
    unittest.main()
