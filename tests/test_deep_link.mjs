import assert from "node:assert/strict";
import test from "node:test";

import { buildViewerSearch, findImageIndex, parseSearchQuery, parseViewerTarget } from "../src/city_analyser/viewer/deep-link.mjs";

test("round trips a complete viewer target", () => {
  const search = buildViewerSearch({
    datasetId: "sample data",
    imageId: "123",
    buildingId: "way/42",
    yawDeg: -31.25,
    pitchDeg: 12.5,
    fovDeg: 84,
  });
  assert.deepEqual(parseViewerTarget(search), {
    datasetId: "sample data",
    imageId: "123",
    buildingId: "way/42",
    yawDeg: -31.25,
    pitchDeg: 12.5,
    fovDeg: 84,
    reviewKey: null,
  });
});

test("carries a review item and rejects malformed keys", () => {
  assert.equal(parseViewerTarget(buildViewerSearch({ datasetId: "d", reviewKey: "proposal:way/1:2" })).reviewKey, "proposal:way/1:2");
  assert.equal(parseViewerTarget("?review=%3Cscript%3E").reviewKey, null);
  assert.equal(parseViewerTarget("?review=other:way/1").reviewKey, null);
});

test("reads OSM elements and coordinates from the search box", () => {
  assert.deepEqual(parseSearchQuery("way/123"), { type: "osm", osmId: "way/123" });
  assert.deepEqual(parseSearchQuery(" Relation 45 "), { type: "osm", osmId: "relation/45" });
  assert.deepEqual(parseSearchQuery("987"), { type: "osm", osmId: "way/987" });
  assert.deepEqual(parseSearchQuery("9.9341, -84.0783"), { type: "coordinates", latitude: 9.9341, longitude: -84.0783 });
  assert.equal(parseSearchQuery("95, 10"), null);
  assert.equal(parseSearchQuery("main street"), null);
});

test("normalizes and clamps view parameters", () => {
  const target = parseViewerTarget("?yaw=540&pitch=1000&fov=2&building=invalid");
  assert.equal(target.yawDeg, -180);
  assert.ok(target.pitchDeg < 86);
  assert.equal(target.fovDeg, 45);
  assert.equal(target.buildingId, null);
});

test("finds a panorama by stable ID rather than index", () => {
  const images = [{ id: 200 }, { id: "100" }];
  assert.equal(findImageIndex(images, "100"), 1);
  assert.equal(findImageIndex(images, "missing"), -1);
});

test("rejects non-OSM building identifiers", () => {
  assert.equal(parseViewerTarget("?building=microsoft%2F120202110-84625").buildingId, null);
});
