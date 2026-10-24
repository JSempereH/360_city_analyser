// Footprint colors from GET /api/datasets/{dataset}/building-attributes (the attribute summary).
// Pure, so the classes, the legend and the wording are tested without a browser.
//
// Neighboring footprints are compared all against all, so a categorical mode
// uses at most three hues (the reference palette's all-pairs limit, checked
// for color vision deficiency); the exact GEM code is in the hover label and
// the card. Floors use one blue ramp, light to dark, whose lightest step still
// stands out from the map tiles.

export const COLOR_MODES = {
  none: "Analysis (default)",
  gem_material: "GEM material",
  gem_llrs: "GEM lateral system",
  floors: "Floors",
  review: "Review status",
  mask_quality: "Mask quality",
};

const UNDETERMINED = { key: "undetermined", label: "Not determined", color: "#898781" };

const GROUPS = {
  gem_material: [
    { key: "concrete", label: "Concrete (CR, CU, C99, SRC)", color: "#2a78d6", codes: ["CR", "CU", "C99", "SRC"] },
    { key: "masonry", label: "Masonry (MUR, MCF, MR, M99)", color: "#eb6834", codes: ["MUR", "MCF", "MR", "M99"] },
    { key: "other", label: "Steel, metal, wood or earth", color: "#1baf7a", codes: ["S", "ME", "W", "EU", "ER", "E99"] },
  ],
  gem_llrs: [
    { key: "frame", label: "Frame (LFM, LFINF, LFBR, LPB, LFLS, LFLSINF)", color: "#2a78d6", codes: ["LFM", "LFINF", "LFBR", "LPB", "LFLS", "LFLSINF"] },
    { key: "wall", label: "Wall (LWAL)", color: "#eb6834", codes: ["LWAL"] },
    { key: "other", label: "Dual, hybrid or none (LDUAL, LH, LN)", color: "#1baf7a", codes: ["LDUAL", "LH", "LN"] },
  ],
  floors: [
    { key: "1-2", label: "1 to 2 floors", color: "#6da7ec", max: 2 },
    { key: "3-4", label: "3 to 4 floors", color: "#2a78d6", max: 4 },
    { key: "5-7", label: "5 to 7 floors", color: "#1c5cab", max: 7 },
    { key: "8+", label: "8 floors or more", color: "#0d366b", max: Infinity },
  ],
  // Status colors with a label in the legend, never color alone.
  review: [
    { key: "reviewed", label: "Reviewed", color: "#0ca30c" },
    { key: "needs_review", label: "Needs review", color: "#fab219" },
    { key: "model_only", label: "Model answer, not reviewed", color: "#898781" },
  ],
  // Whether the model read clean views of the building (mask_quality), with a label in the legend.
  mask_quality: [
    { key: "clean", label: "Every view clean", color: "#0ca30c" },
    { key: "some_set_aside", label: "Some views set aside", color: "#fab219" },
    { key: "no_clean_view", label: "No clean view", color: "#d4351c" },
  ],
};

/** The class of one building in a mode, or null when the summary has nothing for it (not surveyed). */
export function attributeClass(mode, entry) {
  if (!entry || !GROUPS[mode]) return null;
  if (mode === "mask_quality") {
    if (entry.no_clean_view) return GROUPS.mask_quality[2];
    return entry.set_aside > 0 ? GROUPS.mask_quality[1] : GROUPS.mask_quality[0];
  }
  if (mode === "review") {
    if (entry.reviewed) return GROUPS.review[0];
    if (!entry.model) return null;
    return entry.needs_review ? GROUPS.review[1] : GROUPS.review[2];
  }
  const value = entry[mode];
  if (value === null || value === undefined) return UNDETERMINED;
  if (mode === "floors") return GROUPS.floors.find((group) => Number(value) <= group.max) || UNDETERMINED;
  return GROUPS[mode].find((group) => group.codes.includes(value)) || UNDETERMINED;
}

/** Legend rows for the classes present in the summary, in the mode's fixed order, with counts. */
export function legendItems(mode, summary) {
  if (!GROUPS[mode]) return [];
  const counts = new Map();
  for (const entry of Object.values(summary?.buildings || {})) {
    const group = attributeClass(mode, entry);
    if (group) counts.set(group.key, (counts.get(group.key) || 0) + 1);
  }
  return [...GROUPS[mode], UNDETERMINED]
    .filter((group, index, all) => counts.has(group.key) && all.findIndex((other) => other.key === group.key) === index)
    .map((group) => ({ ...group, count: counts.get(group.key) }));
}

/** Hover label of a footprint: the building, its GEM string, and the value of the mode. */
export function attributeLabel(mode, osmId, entry) {
  if (!entry) return `${osmId} · not surveyed yet`;
  const parts = [osmId, entry.gem];
  if (mode === "floors") parts.push(entry.floors === null || entry.floors === undefined ? "floors not determined" : `${entry.floors} floors`);
  if (mode === "review") parts.push(entry.reviewed ? "reviewed" : entry.needs_review || "model answer");
  if (mode === "mask_quality") parts.push(entry.no_clean_view ? "no clean view" : entry.set_aside ? `${entry.set_aside} views set aside` : "every view clean");
  return parts.filter(Boolean).join(" · ");
}
