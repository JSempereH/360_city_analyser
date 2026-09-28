import assert from "node:assert/strict";
import test from "node:test";

import { nearbyPanoramas, nearestPanoramasByBearing, projectGroundPanoramaToView } from "../src/city_analyser/viewer/navigation.mjs";

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
