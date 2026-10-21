import assert from "node:assert/strict";
import test from "node:test";

import { answerLine, boldSpans, framingOf, parseEvents, visibleReasoning } from "../src/city_analyser/viewer/assistant.mjs";
import { levelMatrix } from "../src/city_analyser/viewer/navigation.mjs";

test("events are parsed per block and an unfinished block is kept", () => {
  const { events, rest } = parseEvents('data: {"type":"text","text":"Fo"}\n\ndata: {"type":"text","text":"ur"}\n\ndata: {"ty');
  assert.deepEqual(events.map((event) => event.text), ["Fo", "ur"]);
  assert.equal(rest, 'data: {"ty');
  assert.deepEqual(parseEvents("data: not json\n\n").events, []);
});

test("only the field notes are shown, also while the JSON is arriving", () => {
  assert.equal(visibleReasoning("Four storeys.\nThick walls.\n{\"floors\": 4"), "Four storeys.\nThick walls.");
  assert.equal(visibleReasoning("Notes so far"), "Notes so far");
  assert.equal(visibleReasoning("<think>hidden</think>Visible.\n```json\n{}"), "Visible.");
  assert.equal(visibleReasoning('{"floors": 3}'), "");
});

test("answer line", () => {
  assert.equal(answerLine({ floors: 4, gem_material: "MUR", gem_llrs: "LWAL" }), "4 storeys · MUR · LWAL");
  assert.equal(answerLine(null), "");
});

test("bold markers become spans and nothing else is interpreted", () => {
  assert.deepEqual(boldSpans("It has **4 storeys** and <b>x</b>"), [
    { text: "It has ", bold: false }, { text: "4 storeys", bold: true }, { text: " and <b>x</b>", bold: false },
  ]);
  assert.deepEqual(boldSpans("unclosed **bold"), [{ text: "unclosed **bold", bold: false }]);
});

test("a crop's framing tilts at most 22 degrees up and is wide enough for the storeys", () => {
  // A crop looks up to 30 degrees with 100 of field; shown like that facades lean back, so the camera looks up gently.
  assert.deepEqual(framingOf({ yaw_deg: 30, pitch_deg: 40, fov_deg: 100 }), { yawDeg: 30, pitchDeg: -15, fov: 85 });
  assert.deepEqual(framingOf({ yaw_deg: -10, pitch_deg: -20, fov_deg: 40 }), { yawDeg: -10, pitchDeg: 3, fov: 65 });
  assert.equal(framingOf({}), null);
  // On a tilted panorama the crop's direction is turned into the leveled view. Here world up leans toward the
  // image center, so the camera looked 10 degrees up: 5 degrees below the raw center is 5 degrees above the horizon.
  const tilt = 10 * Math.PI / 180;
  const level = levelMatrix([0, Math.cos(tilt), Math.sin(tilt)]);
  const leveled = framingOf({ yaw_deg: 0, pitch_deg: -5, fov_deg: 100 }, level);
  assert.equal(leveled.yawDeg, 0);
  assert.ok(Math.abs(leveled.pitchDeg - -5) < 0.01, `pitch ${leveled.pitchDeg}`);
});
