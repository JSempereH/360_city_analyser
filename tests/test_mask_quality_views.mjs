import assert from "node:assert/strict";
import test from "node:test";

import { attributeClass, attributeLabel, legendItems } from "../src/city_analyser/viewer/attribute-map.mjs";
import { setAsideNote } from "../src/city_analyser/viewer/building-card.mjs";
import { buildReviewQueue } from "../src/city_analyser/viewer/review-queue.mjs";

const summary = {
  dataset: "city",
  buildings: {
    "way/1": { gem: "MUR/LWAL", set_aside: 0, clean_views: 3, no_clean_view: false },
    "way/2": { gem: "CR/LFINF", set_aside: 1, clean_views: 2, no_clean_view: false, set_aside_reasons: ["sliver"], set_aside_image: "7" },
    "way/3": { gem: "MAT99/L99", set_aside: 2, clean_views: 0, no_clean_view: true, set_aside_reasons: ["interior", "speckle"], set_aside_image: 9 },
    "way/4": { gem: "CR/LFINF", set_aside: 1, clean_views: 1, no_clean_view: false, reviewed: true, set_aside_reasons: ["sky"] },
  },
};

test("the card says which views were set aside and why, and when the model was not asked", () => {
  assert.equal(setAsideNote({ set_aside: [] }), null);
  assert.equal(setAsideNote({ set_aside: [{ why: "the mask is too small to show the facade" }] }),
    "1 view set aside, not shown to the model: the mask is too small to show the facade.");
  assert.equal(setAsideNote({ no_clean_view: true, set_aside: [{ why: "a" }, { why: "a" }] }),
    "No clean view of this building: 2 views set aside (a). The model was not asked.");
});

test("the mask queue lists unreviewed buildings with set-aside views, those without a clean view first", () => {
  const items = buildReviewQueue([], {}, { kind: "mask" }, summary);
  assert.deepEqual(items.map((item) => item.osmId), ["way/3", "way/2"]);
  assert.equal(items[0].image, "9");
  assert.equal(items[0].detail, "No clean view · interior, speckle");
  assert.equal(items[1].detail, "1 view set aside · sliver");
});

test("the mask quality layer has three labeled states", () => {
  assert.deepEqual(["way/1", "way/2", "way/3"].map((id) => attributeClass("mask_quality", summary.buildings[id]).key),
    ["clean", "some_set_aside", "no_clean_view"]);
  assert.deepEqual(legendItems("mask_quality", summary).map((item) => [item.key, item.count]), [["clean", 1], ["some_set_aside", 2], ["no_clean_view", 1]]);
  assert.equal(attributeLabel("mask_quality", "way/3", summary.buildings["way/3"]), "way/3 · MAT99/L99 · no clean view");
});

test("the tour skips buildings without a clean view and the visit says what was set aside", async () => {
  const { planTour, reasoningScript } = await import("../src/city_analyser/viewer/presentation-script.mjs");
  const tour = planTour({ buildings: { "way/1": { model: true, image: "1" }, "way/2": { model: true, image: "2", no_clean_view: true } } }, ["1", "2"]);
  assert.deepEqual(tour.map((stop) => stop.osmId), ["way/1"]);
  const steps = reasoningScript({ building: "way/1", attributes: {}, views: [{ image_id: "1", answer: { observations: "A facade" } }],
    set_aside: [{ why: "the mask is too small to show the facade" }] });
  assert.match(steps[0].text, /One more view was set aside before reading: the mask is too small to show the facade\.$/);
});
