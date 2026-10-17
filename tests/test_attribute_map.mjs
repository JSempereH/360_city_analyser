import assert from "node:assert/strict";
import test from "node:test";

import { attributeClass, attributeLabel, COLOR_MODES, legendItems } from "../src/city_analyser/viewer/attribute-map.mjs";
import { buildReviewQueue } from "../src/city_analyser/viewer/review-queue.mjs";
import { reviewFormValues, reviewOptions, reviewStatus, sourceLabel } from "../src/city_analyser/viewer/building-card.mjs";

const summary = {
  dataset: "city",
  buildings: {
    "way/1": { gem_material: "CR", gem_llrs: "LFINF", floors: 5, gem: "CR/LFINF/HEX:5", reviewed: false, model: true, needs_review: null, urgency: 1, image: "10" },
    "way/2": { gem_material: "MUR", gem_llrs: null, floors: 2, gem: "MUR/L99/HEX:2", reviewed: false, model: true, needs_review: "split vote on the lateral system", urgency: 0.5, image: "11" },
    "way/3": { gem_material: null, gem_llrs: "LWAL", floors: 9, gem: "MAT99/LWAL/HEX:9", reviewed: true, model: true, needs_review: null, urgency: 1, image: null },
    "way/4": { gem_material: "W", gem_llrs: null, floors: null, gem: "W/L99/H99", reviewed: false, model: true, needs_review: "weak agreement on the material (60%)", urgency: 0.6, image: null },
    "way/5": { gem_material: null, gem_llrs: null, floors: 3, gem: "MAT99/L99/HEX:3", reviewed: false, model: false, needs_review: null, urgency: 1, image: null },
  },
};

test("GEM codes fall into three families and unknown values are not determined", () => {
  assert.equal(attributeClass("gem_material", summary.buildings["way/1"]).key, "concrete");
  assert.equal(attributeClass("gem_material", summary.buildings["way/2"]).key, "masonry");
  assert.equal(attributeClass("gem_material", summary.buildings["way/4"]).key, "other");
  assert.equal(attributeClass("gem_material", summary.buildings["way/3"]).key, "undetermined");
  assert.equal(attributeClass("gem_llrs", summary.buildings["way/3"]).key, "wall");
  assert.equal(attributeClass("gem_material", undefined), null);
  assert.equal(attributeClass("none", summary.buildings["way/1"]), null);
});

test("floors use ordered bins and review status needs a model answer", () => {
  assert.deepEqual(["way/1", "way/2", "way/3", "way/4"].map((id) => attributeClass("floors", summary.buildings[id]).key), ["5-7", "1-2", "8+", "undetermined"]);
  assert.deepEqual(["way/1", "way/2", "way/3", "way/5"].map((id) => attributeClass("review", summary.buildings[id])?.key ?? null), ["model_only", "needs_review", "reviewed", null]);
});

test("the legend lists only classes on screen, in fixed order, with counts", () => {
  const items = legendItems("gem_material", summary);
  assert.deepEqual(items.map((item) => [item.key, item.count]), [["concrete", 1], ["masonry", 1], ["other", 1], ["undetermined", 2]]);
  assert.deepEqual(legendItems("none", summary), []);
  assert.equal(Object.keys(COLOR_MODES)[0], "none");
});

test("hover labels carry the GEM string so color is never alone", () => {
  assert.equal(attributeLabel("gem_material", "way/1", summary.buildings["way/1"]), "way/1 · CR/LFINF/HEX:5");
  assert.equal(attributeLabel("review", "way/2", summary.buildings["way/2"]), "way/2 · MUR/L99/HEX:2 · split vote on the lateral system");
  assert.equal(attributeLabel("floors", "way/9", undefined), "way/9 · not surveyed yet");
});

test("structure items join the review queue, weakest first, reviewed ones excluded", () => {
  const queue = buildReviewQueue([], null, { kind: "attributes" }, summary);
  assert.deepEqual(queue.map((item) => item.osmId), ["way/2", "way/4"]);
  assert.equal(queue[0].key, "attributes:city:way/2");
  assert.deepEqual(queue[0].values, { gem_material: "MUR", gem_llrs: null, floors: 2 });
  assert.equal(buildReviewQueue([], null, { kind: "proposal" }, summary).length, 0);
});

test("review form starts from the card and lists GEM codes by attribute", () => {
  const card = {
    gem: { codes: { CR: "reinforced concrete", MUR: "unreinforced masonry", LFINF: "infilled frame", LWAL: "wall" } },
    attributes: { gem_material: { value: "CR" }, gem_llrs: { value: null, abstained: true }, floors: { value: 4.5 } },
    review: { reviewer: "Ana", reviewed_at: "2026-10-16T10:00:00+00:00", note: "thick walls" },
  };
  assert.deepEqual(reviewFormValues(card), { gem_material: "CR", gem_llrs: "", floors: 5, building_type: "", surface_material: "", construction_year: "" });
  const withUse = { ...card, classes: { building_type: ["apartments", "office"], surface_material: ["brick"] }, attributes: { ...card.attributes, building_type: { value: "office" }, construction_year: { value: 1924.4 } } };
  assert.equal(reviewFormValues(withUse).building_type, "office");
  assert.equal(reviewFormValues(withUse).construction_year, 1924);
  assert.deepEqual(reviewOptions(withUse).building_type, [["apartments", "Apartments"], ["office", "Office"]]);
  assert.deepEqual(reviewOptions(card).gem_llrs.map(([code]) => code), ["LFINF", "LWAL"]);
  assert.equal(reviewOptions(card).gem_material[1][1], "Unreinforced masonry (MUR)");
  assert.equal(reviewStatus(card), "Reviewed by Ana on 2026-10-16: thick walls");
  assert.equal(reviewStatus({}), "");
  assert.equal(sourceLabel({ source: "human_review" }), "Reviewed");
});
