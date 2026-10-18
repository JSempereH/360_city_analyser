import assert from "node:assert/strict";
import test from "node:test";

import { applyMatrix, LEVEL_IDENTITY, levelMatrix, nearbyPanoramas, nearestPanoramasByBearing, projectGroundPanoramaToView } from "../src/city_analyser/viewer/navigation.mjs";

test("uses the Mapillary track spacing for navigation distances", () => {
  const targets = nearbyPanoramas([
    { coordinates: [4.9030607775901, 52.36795816785] },
    { coordinates: [4.9029969389023, 52.367937040217] },
  ], 0);

  assert.equal(targets.length, 1);
  assert.ok(targets[0].distance > 4.8 && targets[0].distance < 5.1);
});

test("does not connect nearby panoramas from different known sequences", () => {
  const targets = nearbyPanoramas([
    { coordinates: [4.9030607, 52.3679581], sequenceId: "street-a" },
    { coordinates: [4.9029969, 52.3679370], sequenceId: "street-b" },
    { coordinates: [4.9030000, 52.3679400], sequenceId: "street-a" },
  ], 0);

  assert.deepEqual(targets.map((target) => target.index), [2]);
});

test("projects closer ground targets lower in the panorama", () => {
  const close = projectGroundPanoramaToView(0, 5, 0, 0, 90, 2, 2);
  const distant = projectGroundPanoramaToView(0, 20, 0, 0, 90, 2, 2);

  assert.equal(close.left, 50);
  assert.equal(distant.left, 50);
  assert.ok(close.top > distant.top);
  assert.ok(distant.top > 50);
});

test("keeps only the nearest panorama in each direction", () => {
  const targets = [
    { bearing: 2, distance: 15, index: 1 },
    { bearing: 358, distance: 5, index: 2 },
    { bearing: 90, distance: 8, index: 3 },
    { bearing: 95, distance: 4, index: 4 },
    { bearing: 180, distance: 12, index: 5 },
  ];

  assert.deepEqual(nearestPanoramasByBearing(targets).map((target) => target.index), [4, 2, 5]);
});

test("does not move projected navigation points", () => {
  const projection = projectGroundPanoramaToView(12, 15, 0, 0, 90, 2, 2);

  assert.ok(projection.left > 50);
  assert.ok(projection.top > 50);
});

test("a tilted panorama is leveled by the smallest rotation", () => {
  const up = [-0.05434, 0.948265, 0.312795];  // Mapillary 1986345691512470: 18.5 degrees of tilt
  const level = levelMatrix(up);
  applyMatrix(level, [0, 1, 0]).forEach((value, index) => assert.ok(Math.abs(value - up[index]) < 1e-6));
  const forward = applyMatrix(level, [0, 0, 1]);
  assert.ok(Math.abs(Math.atan2(forward[0], forward[2]) * 180 / Math.PI) < 1);  // the center keeps its heading
  const [x, y, z] = applyMatrix(level, [0.6, 0, 0.8]);
  assert.ok(Math.abs(Math.hypot(x, y, z) - 1) < 1e-9);  // a rotation keeps lengths
  assert.deepEqual(levelMatrix(undefined), LEVEL_IDENTITY);
  assert.deepEqual(levelMatrix([0, -1, 0]), LEVEL_IDENTITY);  // an upside-down estimate is not trusted
});

