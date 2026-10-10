import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from city_analyser import building_analysis
from city_analyser import footprint_corrections as corrections
from city_analyser.footprint_issues import affected_panoramas, footprint_issues


def osm_way(identifier=99, west=-0.0001, south=0.0001, east=0.0001, north=0.0003, version=3):
    return {
        "type": "way", "id": identifier, "tags": {"building": "yes", "height": "12"},
        "version": version, "timestamp": "2024-05-01T10:00:00Z", "changeset": 1234, "user": "someone", "uid": 42,
        "geometry": [{"lon": west, "lat": south}, {"lon": east, "lat": south}, {"lon": east, "lat": north}, {"lon": west, "lat": north}],
    }


def polygon(west, south, east, north):
    return {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]]}


class OsmMetadataTests(unittest.TestCase):
    def test_features_keep_version_timestamp_and_changeset_but_not_the_user(self):
        feature = building_analysis._feature_from_osm(osm_way())

        properties = feature["properties"]
        self.assertEqual((properties["osm_version"], properties["osm_timestamp"], properties["osm_changeset"]), (3, "2024-05-01T10:00:00Z", 1234))
        self.assertNotIn("user", properties)
        self.assertNotIn("uid", properties)

    def test_map_api_fallback_reads_element_versions(self):
        xml = b"""<osm>
          <node id="1" lon="0" lat="0"/><node id="2" lon="0.0001" lat="0"/><node id="3" lon="0.0001" lat="0.0001"/>
          <way id="7" version="5" timestamp="2023-01-02T00:00:00Z" changeset="88" user="x">
            <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="1"/><tag k="building" v="yes"/>
          </way>
        </osm>"""

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return xml

        with patch.object(building_analysis, "urlopen", return_value=Response()):
            features = building_analysis._download_buildings_from_osm_api(0, 0, 50)

        self.assertEqual(features[0]["properties"]["osm_version"], 5)
        self.assertEqual(features[0]["properties"]["osm_changeset"], 88)


class CorrectionStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        environment = patch.dict(os.environ, {"BUILDING_ANALYSIS_FOOTPRINT_CORRECTIONS_DIR": self.directory.name})
        environment.start()
        self.addCleanup(environment.stop)
        self.addCleanup(self.directory.cleanup)
        self.feature = building_analysis._feature_from_osm(osm_way())
        self.moved = polygon(-0.00008, 0.0001, 0.00012, 0.0003)

    def propose(self, geometry=None, **overrides):
        arguments = {"source": "manual", "author": "tester", "note": "facade is 2 m east in every panorama"}
        return corrections.propose(self.feature, geometry or self.moved, **{**arguments, **overrides})

    def test_proposals_are_not_applied_until_accepted(self):
        revision = self.propose()

        self.assertEqual((revision["revision"], revision["status"], revision["base_version"]), (1, "proposed", 3))
        self.assertIs(corrections.apply_corrections([self.feature])[0], self.feature)

    def test_accepted_correction_replaces_geometry_without_touching_the_cached_feature(self):
        self.propose()
        corrections.review("way/99", 1, "accepted", "reviewer")

        applied = corrections.apply_corrections([self.feature])[0]

        self.assertEqual(applied["geometry"], self.moved)
        self.assertEqual(applied["properties"]["footprint_correction"]["revision"], 1)
        self.assertEqual(applied["properties"]["footprint_source"], "osm")
        self.assertNotIn("footprint_correction", self.feature["properties"])
        self.assertNotEqual(self.feature["geometry"], self.moved)

    def test_new_osm_version_makes_the_correction_stale(self):
        self.propose()
        corrections.review("way/99", 1, "accepted", "reviewer")
        newer = building_analysis._feature_from_osm(osm_way(version=4))

        applied = corrections.apply_corrections([newer])[0]

        self.assertEqual(applied["geometry"], newer["geometry"])
        self.assertEqual(applied["properties"]["footprint_correction_stale"]["base_version"], 3)
        self.assertNotIn("footprint_correction", applied["properties"])

    def test_accepting_a_later_revision_supersedes_the_earlier_one(self):
        self.propose()
        corrections.review("way/99", 1, "accepted", "reviewer")
        later = polygon(-0.00009, 0.0001, 0.00011, 0.0003)
        self.propose(later)
        corrections.review("way/99", 2, "accepted", "reviewer")

        record = corrections.load_record("way/99")

        self.assertEqual([revision["status"] for revision in record["revisions"]], ["superseded", "accepted"])
        self.assertEqual(corrections.apply_corrections([self.feature])[0]["geometry"], later)

    def test_reverting_restores_the_osm_geometry_and_keeps_the_history(self):
        self.propose()
        corrections.review("way/99", 1, "accepted", "reviewer")

        reverted = corrections.revert("way/99", "second reviewer", "the shift was a GPS error")

        self.assertEqual((reverted["status"], reverted["reverted_by"]), ("reverted", "second reviewer"))
        self.assertIs(corrections.apply_corrections([self.feature])[0], self.feature)
        self.assertEqual([revision["status"] for revision in corrections.load_record("way/99")["revisions"]], ["reverted"])
        with self.assertRaisesRegex(corrections.CorrectionError, "no accepted correction"):
            corrections.revert("way/99", "second reviewer")
        with self.assertRaisesRegex(corrections.CorrectionError, "reviewer"):
            corrections.revert("way/99", " ")

    def test_rejected_and_reviewed_revisions_cannot_be_reviewed_again(self):
        self.propose()
        corrections.review("way/99", 1, "rejected", "reviewer", "pose error, not the footprint")

        with self.assertRaisesRegex(corrections.CorrectionError, "already rejected"):
            corrections.review("way/99", 1, "accepted", "reviewer")
        self.assertIs(corrections.apply_corrections([self.feature])[0], self.feature)

    def test_invalid_requests_are_refused(self):
        bowtie = {"type": "Polygon", "coordinates": [[[0, 0], [0.0001, 0.0001], [0.0001, 0], [0, 0.0001], [0, 0]]]}
        with self.assertRaisesRegex(corrections.CorrectionError, "valid"):
            self.propose(bowtie)
        with self.assertRaisesRegex(corrections.CorrectionError, "moved more than"):
            self.propose(polygon(0.01, 0.01, 0.0102, 0.0102))
        with self.assertRaisesRegex(corrections.CorrectionError, "note"):
            self.propose(note=" ")
        with self.assertRaisesRegex(corrections.CorrectionError, "source"):
            self.propose(source="guess")
        with self.assertRaisesRegex(corrections.CorrectionError, "invalid OSM element id"):
            corrections.review("way/../x", 1, "accepted", "reviewer")

    def test_osm_export_is_a_never_uploadable_josm_layer_of_accepted_corrections(self):
        from xml.etree import ElementTree

        from city_analyser.footprint_cli import accepted_corrections, as_geojson, as_osm_xml

        self.propose()
        self.assertEqual(accepted_corrections(), [])
        corrections.review("way/99", 1, "accepted", "reviewer")

        root = ElementTree.fromstring(as_osm_xml(accepted_corrections()))
        geojson = as_geojson(accepted_corrections())

        self.assertEqual(root.get("upload"), "never")
        ways = root.findall("way")
        self.assertEqual(len(ways), 1)
        references = [int(node.get("ref")) for node in ways[0].findall("nd")]
        self.assertEqual(len(references), 5)
        self.assertEqual(references[0], references[-1])
        self.assertTrue(all(int(node.get("id")) < 0 for node in root.findall("node")))
        self.assertIn("way/99", {tag.get("k"): tag.get("v") for tag in ways[0].findall("tag")}["fixme"])
        self.assertEqual(geojson["features"][0]["properties"]["osm_id"], "way/99")

    def test_accepted_split_replaces_the_element_with_its_parts(self):
        halves = [polygon(-0.0001, 0.0001, 0.0, 0.0003), polygon(0.0, 0.0001, 0.0001, 0.0003)]
        revision = corrections.propose(self.feature, None, "manual", "tester", "party wall at the middle", parts=halves)
        corrections.review("way/99", revision["revision"], "accepted", "reviewer")

        parts = corrections.apply_corrections([self.feature])

        self.assertEqual([part["properties"]["osm_id"] for part in parts], ["way/99/part-1", "way/99/part-2"])
        self.assertTrue(all(part["properties"]["parent_osm_id"] == "way/99" for part in parts))
        self.assertEqual(parts[1]["properties"]["footprint_correction"]["kind"], "split")
        self.assertEqual((parts[1]["properties"]["footprint_correction"]["part"], parts[1]["properties"]["footprint_correction"]["parts"]), (2, 2))
        self.assertEqual(parts[0]["properties"]["height"], "12")
        self.assertEqual(parts[0]["geometry"], halves[0])
        self.assertTrue(all(corrections.BUILDING_ID.fullmatch(part["properties"]["osm_id"]) for part in parts))

    def test_invalid_splits_are_refused(self):
        with self.assertRaisesRegex(corrections.CorrectionError, "between 2"):
            corrections.propose(self.feature, None, "manual", "tester", "one part", parts=[self.moved])
        overlapping = [polygon(-0.0001, 0.0001, 0.00005, 0.0003), polygon(-0.00005, 0.0001, 0.0001, 0.0003)]
        with self.assertRaisesRegex(corrections.CorrectionError, "overlap"):
            corrections.propose(self.feature, None, "manual", "tester", "overlapping", parts=overlapping)

    def test_split_exports_one_way_per_part(self):
        from xml.etree import ElementTree

        from city_analyser.footprint_cli import accepted_corrections, as_geojson, as_osm_xml

        halves = [polygon(-0.0001, 0.0001, 0.0, 0.0003), polygon(0.0, 0.0001, 0.0001, 0.0003)]
        corrections.propose(self.feature, None, "manual", "tester", "party wall", parts=halves)
        corrections.review("way/99", 1, "accepted", "reviewer")

        ways = ElementTree.fromstring(as_osm_xml(accepted_corrections())).findall("way")
        fixmes = [{tag.get("k"): tag.get("v") for tag in way.findall("tag")}["fixme"] for way in ways]

        self.assertEqual(len(ways), 2)
        self.assertIn("part 2 of 2", fixmes[1])
        self.assertEqual([feature["properties"]["part"] for feature in as_geojson(accepted_corrections())["features"]], [1, 2])

    def test_footprint_identity_changes_with_an_applied_correction(self):
        before = corrections.footprints_sha256([self.feature])
        self.propose()
        corrections.review("way/99", 1, "accepted", "reviewer")

        self.assertNotEqual(corrections.footprints_sha256(corrections.apply_corrections([self.feature])), before)


class DiagnosticTests(unittest.TestCase):
    def test_bearing_range_wraps_across_north(self):
        width = 360
        columns = np.r_[350:360, 0:10]

        self.assertEqual(building_analysis._bearing_range(columns, width, 180.0), [350.5, 9.5])

    def test_unexplained_and_merged_instances_are_reported(self):
        instance_map = np.zeros((100, 400), dtype=np.uint16)
        instance_map[20:80, 10:60] = 1
        instance_map[20:80, 100:150] = 2
        instance_map[20:80, 200:260] = 3
        links = [
            {"instance": 1, "evidence_pixels": 3000, "mode": "abstain", "projected_share": 0.05, "owner_share": 0.0, "osm_ids": []},
            {"instance": 2, "evidence_pixels": 3000, "mode": "whole", "projected_share": 0.9, "owner_share": 1.0, "osm_ids": ["way/5"]},
            {"instance": 3, "evidence_pixels": 3600, "mode": "whole", "projected_share": 0.9, "owner_share": 1.0, "osm_ids": ["way/5"]},
        ]

        diagnostics = building_analysis._instance_diagnostics(instance_map, links, 0.0)

        self.assertEqual([item["type"] for item in diagnostics], ["unexplained_instance", "merged_footprint"])
        np.testing.assert_allclose(diagnostics[0]["bearings"], [189.45, 233.55], atol=0.1)
        self.assertEqual([item["instance"] for item in diagnostics[1]["instances"]], [2, 3])

    def test_projected_footprint_without_building_evidence_is_unsupported(self):
        feature = building_analysis._feature_from_osm(osm_way())
        empty = np.zeros((512, 1024), dtype=bool)
        diagnostics = []

        building_analysis._assign_buildings_detailed(empty, empty, [feature], 0, 0, 0, diagnostics=diagnostics)

        self.assertEqual(diagnostics[0]["type"], "unsupported_footprint")
        self.assertEqual(diagnostics[0]["osm_id"], "way/99")
        self.assertEqual(diagnostics[0]["building_share"], 0.0)

    def test_tree_occluded_footprint_is_not_reported(self):
        feature = building_analysis._feature_from_osm(osm_way())
        empty = np.zeros((512, 1024), dtype=bool)
        diagnostics = []

        building_analysis._assign_buildings_detailed(empty, np.ones_like(empty), [feature], 0, 0, 0, diagnostics=diagnostics)

        self.assertEqual(diagnostics, [])


class IssueQueueTests(unittest.TestCase):
    def write_analysis(self, root, dataset, image_id, longitude, diagnostics):
        dataset_dir = root / dataset
        (dataset_dir / "analysis").mkdir(parents=True, exist_ok=True)
        manifest_path = dataset_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"images": []}
        manifest["images"].append({"id": image_id, "local_file": f"images/{image_id}.jpg", "computed_geometry": {"type": "Point", "coordinates": [longitude, 0.0]}})
        manifest_path.write_text(json.dumps(manifest))
        metadata = {"version": building_analysis.ANALYSIS_VERSION, "image_id": image_id, "footprint_diagnostics": diagnostics}
        (dataset_dir / "analysis" / f"{image_id}.json").write_text(json.dumps(metadata))

    def test_issues_are_ranked_by_supporting_panoramas_and_offsets_need_consistency(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"BUILDING_ANALYSIS_FOOTPRINT_CORRECTIONS_DIR": str(Path(directory) / "corrections")}):
            root = Path(directory)
            shifted = {"type": "pose_offset", "east_m": 3.0, "north_m": 0.2, "heading_delta_deg": 0, "gps_repair_m": 0, "pose_accepted": True}
            for index in range(3):
                self.write_analysis(root, "city", str(index), 0.0001 * index, [
                    {"type": "unsupported_footprint", "osm_id": "way/1", "projected_pixels": 9000, "building_share": 0.0, "vegetation_share": 0.0},
                    shifted,
                ])
            self.write_analysis(root, "city", "9", 0.0, [
                {"type": "merged_footprint", "osm_id": "way/2", "instances": []},
                {"type": "unexplained_instance", "instance": 4, "evidence_pixels": 5000, "projected_share": 0.0, "bearings": [10.0, 30.0]},
            ])
            self.write_analysis(root, "other", "5", 0.0, [
                {"type": "pose_offset", "east_m": 3.0, "north_m": 0.0, "pose_accepted": True},
                {"type": "pose_offset", "east_m": -3.0, "north_m": 0.0, "pose_accepted": True},
            ])

            issues = footprint_issues(root)

        self.assertEqual([entry["osm_id"] for entry in issues["buildings"]], ["way/1", "way/2"])
        self.assertEqual(issues["buildings"][0]["supporting_panoramas"], 3)
        self.assertEqual(issues["unexplained_instances"][0]["bearings"], [10.0, 30.0])
        self.assertEqual([offset["dataset"] for offset in issues["systematic_offsets"]], ["city"])

    def test_building_views_rank_panoramas_by_evidence(self):
        from city_analyser.footprint_issues import building_views

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_analysis(root, "city", "a", 0.0, [])
            self.write_analysis(root, "city", "b", 0.0001, [])
            for image_id, pixels in (("a", 100), ("b", 900)):
                path = root / "city" / "analysis" / f"{image_id}.json"
                metadata = json.loads(path.read_text())
                metadata["building_ids"] = {"1": {"osm_id": "way/9", "pixels": pixels, "confidence": 0.5, "facades": 1, "footprint_geometry": polygon(0, 0, 0.0001, 0.0001)}}
                path.write_text(json.dumps(metadata))

            result = building_views(root, "city", "way/9")
            missing = building_views(root, "city", "way/10")

        self.assertEqual([view["image"] for view in result["views"]], ["b", "a"])
        self.assertEqual(result["geometry"]["type"], "Polygon")
        self.assertEqual((missing["views"], missing["geometry"]), ([], None))

    def test_analysis_status_separates_current_stale_and_missing(self):
        from city_analyser.footprint_issues import analysis_status

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_analysis(root, "city", "now", 0.0, [])
            self.write_analysis(root, "city", "old", 0.0, [])
            old = root / "city" / "analysis" / "old.json"
            old.write_text(json.dumps({**json.loads(old.read_text()), "version": 1}))
            manifest = root / "city" / "manifest.json"
            manifest.write_text(json.dumps({"images": [*json.loads(manifest.read_text())["images"], {"id": "never", "local_file": "images/never.jpg"}]}))

            self.assertEqual(analysis_status(root, "city"), {"now": "current", "old": "stale", "never": "missing"})

    def test_affected_panoramas_are_those_within_the_association_radius(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_analysis(root, "city", "near", 0.0005, [])
            self.write_analysis(root, "city", "far", 0.01, [])

            affected = affected_panoramas(root, polygon(-0.0001, -0.0001, 0.0001, 0.0001))

        self.assertEqual([image["id"] for _, image in affected], ["near"])

    def test_affected_panoramas_accept_the_parts_of_a_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_analysis(root, "city", "near", 0.0005, [])
            parts = {"type": "GeometryCollection", "geometries": [polygon(-0.0001, -0.0001, 0.0, 0.0001), polygon(0.0, -0.0001, 0.0001, 0.0001)]}

            self.assertEqual([image["id"] for _, image in affected_panoramas(root, parts)], ["near"])


class CorrectionApiTests(unittest.TestCase):
    def request(self, base, path, payload=None):
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(f"{base}{path}", data=data, method="POST" if payload is not None else "GET", headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def test_propose_review_and_list_round_trip(self):
        import threading

        from city_analyser import local_viewer

        feature = building_analysis._feature_from_osm(osm_way())
        moved = polygon(-0.00008, 0.0001, 0.00012, 0.0003)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"BUILDING_ANALYSIS_FOOTPRINT_CORRECTIONS_DIR": str(Path(directory) / "corrections"), "BUILDING_ANALYSIS_BACKEND": "local"}), \
                patch.object(local_viewer, "osm_buildings", return_value=[feature]):
            (Path(directory) / "data").mkdir()
            server = local_viewer.ViewerServer(("127.0.0.1", 0), Path(directory))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                proposed = self.request(base, "/api/footprint-corrections/way/99", {"action": "propose", "geometry": moved, "author": "a", "note": "shifted"})
                wrong_id = self.request(base, "/api/footprint-corrections/way/100", {"action": "propose", "geometry": moved, "author": "a", "note": "shifted"})
                pending = self.request(base, "/api/footprint-corrections?status=proposed")
                reviewed = self.request(base, "/api/footprint-corrections/way/99", {"action": "review", "revision": 1, "decision": "accepted", "reviewer": "b"})
                record = self.request(base, "/api/footprint-corrections/way/99")
                reverted = self.request(base, "/api/footprint-corrections/way/99", {"action": "revert", "reviewer": "c", "note": "wrong"})
                reverted_again = self.request(base, "/api/footprint-corrections/way/99", {"action": "revert", "reviewer": "c"})
                missing = self.request(base, "/api/footprint-corrections/way/7")
                issues = self.request(base, "/api/footprint-issues")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertEqual((proposed[0], proposed[1]["revision"]["status"]), (201, "proposed"))
        self.assertEqual(wrong_id[0], 400)
        self.assertEqual([item["osm_id"] for item in pending[1]["corrections"]], ["way/99"])
        self.assertEqual(reviewed[1]["revision"]["status"], "accepted")
        self.assertIn("reassociation", reviewed[1])
        self.assertEqual(record[1]["revisions"][0]["reviewer"], "b")
        self.assertEqual((reverted[0], reverted[1]["revision"]["status"]), (200, "reverted"))
        self.assertIn("reassociation", reverted[1])
        self.assertEqual(reverted_again[0], 400)
        self.assertEqual(missing[0], 404)
        self.assertEqual(issues[1]["buildings"], [])


if __name__ == "__main__":
    unittest.main()


class PartIdentityTests(unittest.TestCase):
    def test_parts_have_report_paths_and_routes(self):
        from city_analyser.building_inspection.reporting import report_paths
        from city_analyser.local_viewer import BUILDING_REPORT_ROUTE, BUILDING_VIEWS_ROUTE

        json_path, _ = report_paths(Path("dataset"), "way/12/part-3")

        self.assertEqual(json_path.parent, Path("dataset/reports/buildings/way/12/part-3"))
        self.assertEqual(BUILDING_REPORT_ROUTE.fullmatch("/api/datasets/d/building-reports/way/12/part-3").group(2), "way/12/part-3")
        self.assertIsNotNone(BUILDING_VIEWS_ROUTE.fullmatch("/api/datasets/d/building-views/way/12/part-3"))
        with self.assertRaises(ValueError):
            report_paths(Path("dataset"), "way/12/part-0")

    def test_part_report_links_keep_the_part_segment(self):
        from city_analyser.building_inspection.reporting import _report_links, _screenshot_url

        self.assertEqual(_report_links("d", "way/12/part-3")["html"], "/api/datasets/d/building-reports/way/12/part-3/html")
        self.assertEqual(_screenshot_url("d", "way/12/part-3", "a.jpg"), "/data/d/reports/buildings/way/12/part-3/a.jpg")
