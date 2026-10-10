import assert from "node:assert/strict";
import test from "node:test";

import { createHistory, editableRings, splitRing, edgeMidpoints, polygonFromRings, translatedRings, withInsertedVertex, withoutVertex } from "../src/city_analyser/viewer/footprint-editor.mjs";

const square = {
  type: "Polygon",
  coordinates: [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.4]]],
};

test("polygons round-trip through open latitude/longitude rings", () => {
  const rings = editableRings(square);
  assert.deepEqual(rings.outer, [[0, 0], [0, 1], [1, 1], [1, 0]]);
  assert.equal(rings.holes.length, 1);
  assert.deepEqual(polygonFromRings(rings), square);
});

test("multi-part footprints are not editable", () => {
  assert.equal(editableRings({ type: "MultiPolygon", coordinates: [square.coordinates, square.coordinates] }), null);
  assert.ok(editableRings({ type: "MultiPolygon", coordinates: [square.coordinates] }));
});

test("translation moves holes with the outer ring", () => {
  const moved = translatedRings(editableRings(square), 0.5, -1);
  assert.deepEqual(moved.outer[0], [0.5, -1]);
  assert.deepEqual(moved.holes[0][0], [0.9, -0.6]);
});

test("vertices are inserted at edge midpoints and never drop below a triangle", () => {
  const { outer } = editableRings(square);
  const midpoints = edgeMidpoints(outer);
  assert.deepEqual(midpoints[0], [0, 0.5]);
  const inserted = withInsertedVertex(outer, 0, midpoints[0]);
  assert.deepEqual(inserted, [[0, 0], [0, 0.5], [0, 1], [1, 1], [1, 0]]);
  const triangle = withoutVertex(outer, 0);
  assert.equal(triangle.length, 3);
  assert.equal(withoutVertex(triangle, 0).length, 3);
});

test("history undoes and redoes, and a new change drops the redo branch", () => {
  const history = createHistory("a");
  assert.equal(history.canUndo(), false);
  history.commit("b");
  history.commit("c");
  assert.equal(history.undo(), "b");
  assert.equal(history.undo(), "a");
  assert.equal(history.undo(), "a");
  assert.equal(history.redo(), "b");
  history.commit("d");
  assert.equal(history.canRedo(), false);
  assert.equal(history.current(), "d");
});

test("a ring is cut along the chord between two vertices", () => {
  const hexagon = [[0, 0], [0, 1], [0, 2], [1, 2], [1, 1], [1, 0]];
  const [first, second] = splitRing(hexagon, 1, 4);
  assert.deepEqual(first, [[0, 1], [0, 2], [1, 2], [1, 1]]);
  assert.deepEqual(second, [[1, 1], [1, 0], [0, 0], [0, 1]]);
  assert.deepEqual(splitRing(hexagon, 4, 1), [first, second]);
  assert.equal(splitRing(hexagon, 2, 2), null);
  assert.equal(splitRing(hexagon, 2, 3), null);
  assert.equal(splitRing(hexagon, 0, 5), null);
});
