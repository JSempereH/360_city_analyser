import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

Image = __import__("PIL.Image", fromlist=["Image"])

from city_analyser.building_inspection.crops import analyzed_building_ids, crops_directory, export_building_crops
from tests.test_building_inspection import BuildingFixture


class CropExportTests(BuildingFixture, unittest.TestCase):
    def test_exports_ranked_crops_with_provenance_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, _ = self.create_dataset(root)
            self.assertEqual(analyzed_building_ids(dataset_dir), ["way/42"])
            first = export_building_crops(root, "sample", size=64, per_building=2)
            target = crops_directory(dataset_dir, "way/42")
            crops = sorted(target.glob("crop-*.jpg"))
            sidecars = sorted(target.glob("crop-*.json"))
            first_bytes = [path.read_bytes() for path in crops]
            export_building_crops(root, "sample", size=64, per_building=2)
            second_bytes = [path.read_bytes() for path in sorted(target.glob("crop-*.jpg"))]
            sidecar = json.loads(sidecars[0].read_text(encoding="utf-8"))
            with Image.open(crops[0]) as crop:
                size = crop.size

        self.assertEqual(first, {"buildings": 1, "crops": 2, "buildings_without_evidence": [], "buildings_without_clean_view": []})
        self.assertEqual([path.name for path in crops], ["crop-01-200.jpg", "crop-02-100.jpg"])
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(size, (64, 64))
        self.assertEqual(sidecar["building"]["id"], "way/42")
        self.assertEqual(sidecar["source"]["image_id"], "200")
        self.assertEqual(sidecar["evidence"]["rank"], 1)
        self.assertEqual(len(sidecar["crop"]["sha256"]), 64)
        self.assertEqual(sidecar["view"]["framing_basis"], "observed_mask_bounds")

    def test_unknown_building_is_reported_not_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_dataset(root)
            result = export_building_crops(root, "sample", ["way/999"], size=64)
        self.assertEqual(result, {"buildings": 0, "crops": 0, "buildings_without_evidence": ["way/999"], "buildings_without_clean_view": []})

    def test_views_with_a_flagged_mask_are_set_aside_with_their_reasons(self):
        from city_analyser.building_inspection.crops import set_aside_views

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, _ = self.create_dataset(root)
            meta_path = dataset_dir / "analysis" / "200.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for record in meta["building_ids"].values():
                record["mask_quality"] = {"version": 1, "ok": False, "reasons": ["neighbor"], "signals": {}}
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            target = crops_directory(dataset_dir, "way/42")
            filtered = export_building_crops(root, "sample", size=64, per_building=2)
            names = sorted(path.name for path in target.glob("crop-*.jpg"))
            aside = set_aside_views(dataset_dir, "way/42")
            # A reviewer's curation vouches for the view, so it comes back; so does --keep-flagged-masks.
            export_building_crops(root, "sample", size=64, per_building=2, clean_views={"way/42": {"200"}})
            vouched = sorted(path.name for path in target.glob("crop-*.jpg"))
            vouched_aside = set_aside_views(dataset_dir, "way/42")
        self.assertEqual((filtered["crops"], names), (1, ["crop-02-100.jpg"]))
        self.assertEqual(aside, [{"image_id": "200", "rank": 1, "reasons": ["neighbor"]}])
        self.assertEqual((vouched, vouched_aside), (["crop-01-200.jpg", "crop-02-100.jpg"], []))

    def test_only_the_approved_views_of_a_building_are_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, _ = self.create_dataset(root)
            result = export_building_crops(root, "sample", size=64, per_building=3, only_views={"way/42": {"100"}})
            names = sorted(path.name for path in crops_directory(dataset_dir, "way/42").glob("crop-*.jpg"))
        self.assertEqual((result["crops"], names), (1, ["crop-02-100.jpg"]))

    def test_mask_others_paints_other_buildings_gray(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, _ = self.create_dataset(root, second_view=False)
            ids_path = dataset_dir / "analysis" / "100-ids.png"
            ids = np.asarray(Image.open(ids_path)).copy()
            ids[30:70, 86:100] = 2  # a second building next to the target
            Image.fromarray(ids).save(ids_path)
            # The id mask changed, so refresh the analysis hash that guards it.
            from city_analyser.analysis_cache import file_sha256
            meta_path = dataset_dir / "analysis" / "100.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["artifact_sha256"]["id_mask"] = file_sha256(ids_path)
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            export_building_crops(root, "sample", ["way/42"], size=96, per_building=1, mask_others=True)
            masked = np.asarray(Image.open(next(crops_directory(dataset_dir, "way/42").glob("crop-*.jpg"))))
        self.assertTrue(((masked > 120) & (masked < 135)).all(axis=2).any())


if __name__ == "__main__":
    unittest.main()
