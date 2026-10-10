import assert from "node:assert/strict";
import test from "node:test";

import { bearingCenter, buildReviewQueue, preservedIndex, stepIndex } from "../src/city_analyser/viewer/review-queue.mjs";

const corrections = [{
  osm_id: "way/1",
  revisions: [
    { revision: 1, status: "rejected", note: "old", author: "a" },
    { revision: 2, status: "proposed", note: "moved east", author: "b", geometry: { type: "Polygon", coordinates: [] }, evidence: [{ dataset: "city", image: 7 }] },
  ],
}];
const issues = {
  buildings: [
    { osm_id: "way/5", issues: { merged_footprint: 1 }, supporting_panoramas: 1, panoramas: [{ dataset: "city", image: "2", type: "merged_footprint" }], correction: null },
    { osm_id: "way/6", issues: { unsupported_footprint: 2 }, supporting_panoramas: 3, panoramas: [{ dataset: "other", image: "9", type: "unsupported_footprint" }], correction: null },
    { osm_id: "way/7", issues: { merged_footprint: 1 }, supporting_panoramas: 2, panoramas: [{ dataset: "city", image: "3", type: "merged_footprint" }], correction: { status: "accepted" } },
  ],
  unexplained_instances: [{ dataset: "city", image: "4", bearings: [350, 20], evidence_pixels: 9000, longitude: 1, latitude: 2 }],
};

test("proposals come first, then problems by support, then unexplained buildings", () => {
  const queue = buildReviewQueue(corrections, issues);
  assert.deepEqual(queue.map((item) => item.key), [
    "proposal:way/1:2", "unsupported_footprint:way/6", "merged_footprint:way/5", "unexplained_instance:city:4:350-20",
  ]);
  assert.equal(queue[0].image, "7");
  assert.equal(queue[0].dataset, "city");
});

test("filters by kind, dataset and resolution", () => {
  assert.deepEqual(buildReviewQueue(corrections, issues, { kind: "merged_footprint" }).map((item) => item.osmId), ["way/5"]);
  assert.deepEqual(buildReviewQueue(corrections, issues, { kind: "merged_footprint", unresolvedOnly: false }).map((item) => item.osmId), ["way/7", "way/5"]);
  assert.ok(buildReviewQueue(corrections, issues, { dataset: "city" }).every((item) => item.dataset === "city"));
});

test("navigation wraps and keeps the reviewer's place after a rebuild", () => {
  assert.equal(stepIndex(3, 2, 1), 0);
  assert.equal(stepIndex(3, 0, -1), 2);
  assert.equal(stepIndex(3, -1, 1), 0);
  assert.equal(stepIndex(0, -1, 1), -1);
  const items = [{ key: "a" }, { key: "c" }];
  assert.equal(preservedIndex(items, "c", 0), 1);
  assert.equal(preservedIndex(items, "b", 1), 1);
  assert.equal(preservedIndex(items, "b", 5), 1);
  assert.equal(preservedIndex([], "b", 0), -1);
  assert.equal(preservedIndex(items, undefined, -1), -1);
});

test("bearing centers wrap across north", () => {
  assert.equal(bearingCenter([350, 20]), 5);
  assert.equal(bearingCenter([10, 30]), 20);
});
