import assert from "node:assert/strict";
import test from "node:test";

import { answerLine, boldSpans, framingOf, parseEvents, visibleReasoning } from "../src/city_analyser/viewer/assistant.mjs";

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
  assert.deepEqual(framingOf({ yaw_deg: 30, pitch_deg: 40, fov_deg: 100 }), { yawDeg: 30, pitchDeg: -22, fov: 95 });
  assert.deepEqual(framingOf({ yaw_deg: -10, pitch_deg: -20, fov_deg: 40 }), { yawDeg: -10, pitchDeg: 5, fov: 60 });
  assert.equal(framingOf({}), null);
});
