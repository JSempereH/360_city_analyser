import assert from "node:assert/strict";
import test from "node:test";

import { planTour, reasoningScript } from "../src/city_analyser/viewer/presentation-script.mjs";

const codes = { CR: "reinforced concrete", MUR: "unreinforced masonry", LFINF: "infilled frame", LWAL: "wall" };

function card(overrides = {}) {
  return {
    building: "way/7",
    gem: { string: "CR/LFINF/HEX:5", codes },
    model: "vlm:simulated-demo",
    attributes: {
      floors: { source: "osm_tag", value: 5, others: [{ provider: "vlm:simulated-demo", value: 5 }] },
      gem_material: { source: "vlm:simulated-demo", value: "CR", share: 0.83, score_kind: "vote_share" },
      gem_llrs: { source: "vlm:simulated-demo", value: "LFINF", share: 1, score_kind: "vote_share" },
    },
    views: [
      { image_id: "10", rank: 1, view: { yaw_deg: 30, pitch_deg: 12, fov_deg: 60 }, answer: { observations: "Slab edges at every storey.", floors: 5, gem_material: "CR", gem_llrs: "LFINF" } },
      { image_id: "11", rank: 2, view: null, error: "Model server failed", answer: {} },
    ],
    ...overrides,
  };
}

test("the tour follows the panorama sequence and skips buildings without a model answer", () => {
  const summary = { buildings: {
    "way/1": { model: true, image: "12" }, "way/2": { model: true, image: "10" }, "way/3": { model: false, image: "10" }, "way/4": { model: true, image: null },
  } };
  assert.deepEqual(planTour(summary, ["10", "11", "12"]).map((stop) => stop.osmId), ["way/2", "way/1", "way/4"]);
  assert.deepEqual(planTour(null, []), []);
});

test("a confident building is reasoned view by view, then decided and saved", () => {
  const steps = reasoningScript(card());
  assert.deepEqual(steps.map((step) => step.type), ["intro", "view", "view", "floors", "structure", "decision", "save"]);
  assert.equal(steps[1].imageId, "10");
  assert.match(steps[1].text, /^View 1\. Slab edges at every storey\.\n→ 5 storeys · reinforced concrete \(CR\) · infilled frame \(LFINF\)/);
  assert.match(steps[2].text, /set it aside/);
  assert.match(steps[3].text, /OpenStreetMap records 5 levels, consistent/);
  assert.match(steps[4].text, /reinforced concrete \(CR\), 83% of the votes/);
  const decision = steps[5];
  assert.equal(decision.needsReview, false);
  assert.deepEqual(decision.fields.map((field) => field.value), ["5", "CR · Reinforced concrete", "LFINF · Infilled frame"]);
  assert.equal(steps[6].text, "Saved to the building record.");
});

test("a split vote or a contradiction with OSM is flagged, never guessed", () => {
  const split = card({ attributes: {
    floors: { source: "osm_tag", value: 5, others: [{ provider: "vlm:simulated-demo", value: 3 }] },
    gem_material: { source: "vlm:simulated-demo", value: "CR", share: 1, score_kind: "vote_share" },
    gem_llrs: { source: "vlm:simulated-demo", value: null, abstained: true, leaning: "LWAL", share: 0.5 },
  } });
  const steps = reasoningScript(split);
  assert.match(steps.find((step) => step.type === "structure").text, /disagree on the lateral system; wall \(LWAL\) leads with 50%\. I will not guess/);
  assert.match(steps.find((step) => step.type === "floors").text, /images suggest 3/);
  const decision = steps.find((step) => step.type === "decision");
  assert.equal(decision.needsReview, true);
  assert.equal(decision.fields[2].value, "L99 · not determined");
  assert.match(steps.at(-1).text, /flagged for an engineer/);
});

test("slivers of facade are left out of the tour unless nothing else is left", () => {
  const summary = { buildings: { "way/1": { model: true, image: "10", score: 0.3 }, "way/2": { model: true, image: "10", score: 0.001 }, "way/3": { model: true, image: "11" } } };
  assert.deepEqual(planTour(summary, ["10", "11"], { minScore: 0.03 }).map((stop) => stop.osmId), ["way/1", "way/3"]);
  assert.deepEqual(planTour({ buildings: { "way/2": summary.buildings["way/2"] } }, ["10"], { minScore: 0.03 }).map((stop) => stop.osmId), ["way/2"]);
});

test("a view shows the model's field notes when it wrote them", () => {
  const notes = card({ views: [{ image_id: "10", rank: 1, view: null, reasoning: "Four rows of windows between thick piers.", answer: { observations: "short", floors: 4 } }] });
  assert.match(reasoningScript(notes).find((step) => step.type === "view").text, /^View 1\. Four rows of windows between thick piers\.\n→ 4 storeys/);
});
