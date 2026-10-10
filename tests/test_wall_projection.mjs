import assert from "node:assert/strict";
import test from "node:test";

import { CAMERA_HEIGHT_M, facingWalls, projectPoint, wallHeight, wallPolylines } from "../src/city_analyser/viewer/wall-projection.mjs";

const camera = [0, 0];
const view = { headingDeg: 0, yaw: 0, pitch: 0, fovDeg: 90, width: 800, height: 400 };
const north = (meters) => [0, meters / 111195];
const east = (meters) => [meters / 111195, 0];

test("a point straight ahead at eye height lands on the screen center", () => {
  const [x, y] = projectPoint(camera, north(20), CAMERA_HEIGHT_M, view);
  assert.ok(Math.abs(x - 400) < 1e-6);
  assert.ok(Math.abs(y - 200) < 1e-6);
});

test("points behind the viewer are not drawn", () => {
  assert.equal(projectPoint(camera, north(-20), 0, view), null);
});

test("heading and yaw turn the view the way the shader does", () => {
  const right = projectPoint(camera, east(20), CAMERA_HEIGHT_M, { ...view, headingDeg: 45 });
  assert.ok(right[0] > 400, "a point east of a north-east heading is right of center");
  // Looking right (positive yaw) brings that point back toward the center.
  const centered = projectPoint(camera, east(20), CAMERA_HEIGHT_M, { ...view, headingDeg: 45, yaw: Math.PI / 4 });
  assert.ok(Math.abs(centered[0] - 400) < 1e-6);
});

test("higher points appear higher, and pitch moves them down", () => {
  const ground = projectPoint(camera, north(10), 0, view);
  const top = projectPoint(camera, north(10), 10, view);
  assert.ok(top[1] < ground[1]);
  // Negative pitch looks up in app.js, so the top of the wall moves down the screen.
  const pitched = projectPoint(camera, north(10), 10, { ...view, pitch: -0.3 });
  assert.ok(pitched[1] > top[1]);
});

test("walls passing behind the camera are split, not drawn across the screen", () => {
  // A block east of the camera whose west wall runs from 20 m behind to 20 m ahead.
  const ring = [[0.00005, -0.00018], [0.00014, -0.00018], [0.00014, 0.00018], [0.00005, 0.00018], [0.00005, -0.00018]];
  const lines = wallPolylines(camera, ring, 10, view);
  assert.ok(lines.length > 0);
  lines.flat().forEach(([x, y]) => assert.ok(Number.isFinite(x) && Number.isFinite(y)));
  // Nothing is drawn for a camera standing inside the footprint.
  assert.deepEqual(wallPolylines(camera, [[-0.0001, -0.0001], [0.0001, -0.0001], [0.0001, 0.0001], [-0.0001, 0.0001]], 10, view), []);
});

test("wall heights follow the analysis rules", () => {
  assert.equal(wallHeight({ height: "12,5" }), 12.5);
  assert.equal(wallHeight({ "building:levels": "4" }), 12.8);
  assert.equal(wallHeight({ "building:levels": "0.5" }), 3);
  assert.equal(wallHeight({}), 10);
});

test("only walls facing the camera are drawn, whatever the ring orientation", () => {
  // A square 10-20 m north of the camera: only its south wall faces it.
  const square = [[-0.00005, 0.00009], [0.00005, 0.00009], [0.00005, 0.00018], [-0.00005, 0.00018]];
  assert.deepEqual(facingWalls(camera, square), [true, false, false, false]);
  // Reversed, the south wall is the third edge.
  assert.deepEqual(facingWalls(camera, [...square].reverse()), [false, false, true, false]);
});
