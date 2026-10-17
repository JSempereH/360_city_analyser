import assert from "node:assert/strict";
import test from "node:test";

import { attributeRows, gemRow, observationItems, sourceLabel, valueLabel } from "../src/city_analyser/viewer/building-card.mjs";

const card = {
  gem: { string: "CR/L99/HEX:5", complete: false, codes: { CR: "reinforced concrete", LFINF: "infilled frame", LDUAL: "dual frame-wall system" } },
  attributes: {
    gem_material: { provider: "vlm:m", source: "vlm:m", value: "CR", share: 0.83, score_kind: "vote_share", alternatives: [{ value: "MCF", share: 0.17 }], others: [] },
    gem_llrs: { provider: "vlm:m", source: "vlm:m", value: null, abstained: true, reason: "top share 0.50 is below 0.60", leaning: "LFINF", others: [] },
    floors: { provider: "osm-tag-prior", source: "osm_tag", value: 5, spread: 0, others: [{ provider: "vlm:m", source: "vlm:m", value: 4, spread: 0 }] },
    building_type: { provider: "osm-tag-prior", source: "osm_tag", value: "apartments", share: 1, score_kind: "tag", others: [] },
    construction_year: { provider: "vlm:m", source: "vlm:m", value: 1972, spread: 8, others: [] },
  },
  observations: [{ text: "Slab edges at every storey.", rank: 1 }, { text: "Slab edges at every storey.", rank: 2 }, { text: " ", rank: 3 }],
};

test("rows follow the card order and say where each value comes from", () => {
  const rows = attributeRows(card);
  assert.deepEqual(rows.map((row) => row.label), ["Material", "Lateral system", "Floors", "Use", "Built"]);
  assert.equal(rows[0].value, "Reinforced concrete (CR)");
  assert.equal(rows[0].source, "AI, 83% of votes");
  assert.match(rows[0].title, /Also considered: MCF 17%/);
  assert.equal(rows[2].source, "OSM tag");
  assert.equal(rows[2].note, "AI says 4");
  assert.equal(rows[4].value, "about 1972 (±8)");
});

test("an abstention shows why and what the vote leaned to", () => {
  const row = attributeRows(card)[1];
  assert.equal(row.value, "Not determined");
  assert.equal(row.abstained, true);
  assert.equal(row.title, "top share 0.50 is below 0.60");
  assert.equal(row.note, "leans infilled frame (LFINF)");
});

test("the GEM row marks an incomplete string", () => {
  const row = gemRow(card);
  assert.equal(row.value, "CR/L99/HEX:5");
  assert.match(row.title, /MAT99, L99 and H99/);
  assert.equal(gemRow({ gem: card.gem, attributes: { floors: card.attributes.floors } }), null);
});

test("labels, sources and observations", () => {
  assert.equal(valueLabel("surface_material"), "Surface material");
  assert.equal(valueLabel("LFINF", card), "Infilled frame (LFINF)");
  assert.equal(sourceLabel({ provider: "vlm:m", score_kind: "probability", share: 0.4 }), "AI");
  assert.deepEqual(observationItems(card).map((item) => item.rank), [1]);
  assert.deepEqual(attributeRows(null), []);
});
