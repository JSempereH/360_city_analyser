import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from city_analyser import building_table
from city_analyser.attributes.cli import attributes_path
from city_analyser.attributes.prior import osm_prior
from city_analyser.building_table import association_status, attribute_columns, export_buildings
from tests.test_building_inspection import BuildingFixture

INDEX = {"nearby_buildings": ["way/42", "way/43", "way/44", "way/45"], "buildings": {"way/42": [], "way/43": []}}


class ExportTests(BuildingFixture, unittest.TestCase):
    def export(self, root: Path, **options):
        with patch.object(building_table, "load_building_index", return_value=INDEX):
            return export_buildings(root, "sample", **options)

    def test_row_views_coverage_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_dataset(root)
            result = self.export(root, min_observed_fraction=0.001)
            output = Path(result["output"])
            geojson = json.loads((output / "buildings.geojson").read_text(encoding="utf-8"))
            rows = list(csv.DictReader((output / "buildings.csv").open(encoding="utf-8")))
            views = list(csv.DictReader((output / "building_views.csv").open(encoding="utf-8")))
            manifest = json.loads((output / "export-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(result["buildings"], 1)
        row = rows[0]
        self.assertEqual((row["building_id"], row["osm_type"], row["osm_id"], row["views"], row["association_status"]), ("way/42", "way", "42", "2", "multi_view"))
        self.assertEqual(row["best_image_id"], "200")  # the ranking in the fixture puts view 200 first
        self.assertEqual(sorted(view["image_id"] for view in views), ["100", "200"])
        self.assertEqual(geojson["features"][0]["id"], "way/42")
        self.assertEqual(geojson["features"][0]["geometry"]["type"], "Polygon")
        self.assertEqual(geojson["features"][0]["properties"]["views"], 2)
        footprints = result["coverage"]["footprints"]
        self.assertEqual((footprints["matched"], footprints["near_captured_streets"], footprints["in_line_of_sight"]), (1, 4, 2))
        self.assertEqual(footprints["seen_but_not_matched_examples"], ["way/43"])
        self.assertEqual(result["coverage"]["fractions"]["matched_of_nearby"], 0.25)
        self.assertEqual(result["coverage"]["panoramas"]["analyzed_current"], 2)
        self.assertEqual(manifest["buildings"], 1)
        self.assertIn("ODbL", manifest["attribution"])
        self.assertEqual(len(manifest["inputs_sha256"]), 64)
        self.assertEqual(set(manifest["files"]), {"buildings.geojson", "buildings.csv", "building_views.csv", "coverage.json"} | ({"buildings.parquet"} if "buildings.parquet" in manifest["files"] else set()))

    def test_unusable_views_make_the_building_weak(self):
        views = [{"image_id": "1", "usable": False}, {"image_id": "2", "usable": False}]
        self.assertEqual(association_status(views), "weak_views_only")
        self.assertEqual(association_status([{"image_id": "1", "usable": True}, {"image_id": "1", "usable": True}]), "single_view")

    def test_stale_analyses_are_counted_not_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, _ = self.create_dataset(root)
            meta_path = dataset_dir / "analysis" / "100.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["version"] = 1
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            result = self.export(root)
        counts = result["coverage"]["panoramas"]
        self.assertEqual((counts["analyzed_current"], counts["stale_or_invalid"]), (1, 1))
        self.assertEqual(result["coverage"]["footprints"]["matched"], 1)

    def test_attributes_are_joined_flat_and_missing_ones_are_null(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, _ = self.create_dataset(root)
            payload = osm_prior("way/42", {"building": "retail", "building:levels": "4"}).to_dict()
            path = attributes_path(dataset_dir.resolve(), "way/42", "osm-tag-prior")
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            joined = self.export(root, provider="osm-tag-prior")
            rows = list(csv.DictReader((Path(joined["output"]) / "buildings.csv").open(encoding="utf-8")))
            without = self.export(root, provider="vlm_nothing")  # same output folder: read the first result before
        self.assertEqual((rows[0]["building_type"], rows[0]["building_type_kind"], rows[0]["floors"], rows[0]["surface_material_abstained"]), ("retail", "tag", "4.0", "True"))
        self.assertEqual(rows[0]["attributes_provider"], "osm-tag-prior")
        self.assertEqual(without["buildings"], 1)

    def test_attribute_columns_are_stable_without_a_file(self):
        columns = attribute_columns(None, None)
        self.assertEqual(set(columns), set(building_table.ATTRIBUTE_COLUMNS))
        self.assertTrue(all(value is None for value in columns.values()))

    def test_export_is_deterministic_apart_from_the_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_dataset(root)
            first = self.export(root)
            hashes = json.loads((Path(first["output"]) / "export-manifest.json").read_text())["files"]
            second = self.export(root)
            again = json.loads((Path(second["output"]) / "export-manifest.json").read_text())["files"]
        self.assertEqual(hashes, again)

    def test_geoparquet_when_pyarrow_is_available(self):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_dataset(root)
            result = self.export(root)
            table = pq.read_table(Path(result["output"]) / "buildings.parquet")
        self.assertEqual(table.num_rows, 1)
        self.assertIn(b"geo", table.schema.metadata)
        self.assertEqual(json.loads(table.schema.metadata[b"geo"])["primary_column"], "geometry")


if __name__ == "__main__":
    unittest.main()
