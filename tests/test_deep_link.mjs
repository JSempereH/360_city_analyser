import assert from "node:assert/strict";
import test from "node:test";

import { buildViewerSearch, findImageIndex, parseViewerTarget } from "../src/city_analyser/viewer/deep-link.mjs";

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
  });
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
