// Rows of the building card from GET /api/datasets/{dataset}/building-attributes/{id}.
// Pure, so the wording and the source rule are tested without a browser.

export const CARD_ATTRIBUTES = [
  ["gem_material", "Material"],
  ["gem_llrs", "Lateral system"],
  ["floors", "Floors"],
  ["building_type", "Use"],
  ["surface_material", "Facade"],
  ["construction_year", "Built"],
];

const GEM_ATTRIBUTES = new Set(["gem_material", "gem_llrs"]);

function capitalized(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** Readable name of a class; GEM codes keep their code: "Reinforced concrete (CR)". */
export function valueLabel(value, card = null) {
  if (value === null || value === undefined) return "";
  const meaning = card?.gem?.codes?.[value];
  if (meaning) return `${capitalized(meaning)} (${value})`;
  return capitalized(String(value).replaceAll("_", " "));
}

function isTag(entry) {
  return entry?.source === "osm_tag" || entry?.source === "human_review";
}

function sourceName(entry) {
  return entry?.source === "human_review" ? "Review" : entry?.source === "osm_tag" ? "OSM" : "AI";
}

function formatNumeric(attribute, entry) {
  const value = Number(entry.value);
  if (!Number.isFinite(value)) return "";
  const spread = Number(entry.spread);
  if (attribute === "construction_year") {
    // A model's year is an estimate; a tagged one is a record.
    return isTag(entry) ? String(Math.round(value)) : `about ${Math.round(value)}${spread >= 5 ? ` (±${Math.round(spread)})` : ""}`;
  }
  return `${Number.isInteger(value) ? value : value.toFixed(1)}${!isTag(entry) && spread >= 0.5 ? ` (±${spread % 1 ? spread.toFixed(1) : spread})` : ""}`;
}

function shownValue(attribute, entry, card) {
  if (!entry || entry.abstained || entry.value === null || entry.value === undefined) return "";
  return attribute === "floors" || attribute === "construction_year" ? formatNumeric(attribute, entry) : valueLabel(entry.value, card);
}

/** Where a value comes from, in two or three words. */
export function sourceLabel(entry) {
  if (!entry) return "";
  if (entry.source === "human_review") return "Reviewed";
  if (entry.source === "osm_tag") return "OSM tag";
  if (String(entry.provider || "").startsWith("vlm:")) {
    return Number.isFinite(entry.share) && entry.score_kind === "vote_share" ? `AI, ${Math.round(entry.share * 100)}% of votes` : "AI";
  }
  return entry.provider || entry.source || "";
}

/** The GEM v3 string with what is known (MAT99, L99 and H99 mark the rest), or null without any GEM attribute. */
export function gemRow(card) {
  const attributes = card?.attributes || {};
  if (!card?.gem?.string || ![...GEM_ATTRIBUTES].some((attribute) => attributes[attribute])) return null;
  return {
    label: "GEM",
    value: card.gem.string,
    title: `GEM Building Taxonomy v3: material / lateral load-resisting system / height.${card.gem.complete ? "" : " MAT99, L99 and H99 mark what could not be determined."}`,
  };
}

// Reference attributes the cadastre records, and how far an answer may be from it and still agree.
const CADASTRE_ATTRIBUTES = { floors: 0.5, construction_year: 10, building_type: null };

/**
 * The cadastre's value for one attribute: `{ text, disagrees, title }`, or null.
 * The cadastre counts storeys a street view may not show (a mezzanine, a set-back attic), and its
 * year can be a refurbishment, so a disagreement is something to look at, not an error.
 */
export function referenceFor(card, attribute, entry = card?.attributes?.[attribute]) {
  const cadastre = card?.references?.catastro;
  const reference = cadastre?.[attribute];
  if (!cadastre || reference === null || reference === undefined || !(attribute in CADASTRE_ATTRIBUTES)) return null;
  const text = `Cadastre ${attribute === "building_type" ? valueLabel(reference).toLowerCase() : Math.round(Number(reference))}`;
  const answered = entry && !entry.abstained && entry.value !== null && entry.value !== undefined;
  const tolerance = CADASTRE_ATTRIBUTES[attribute];
  const disagrees = Boolean(answered) && (tolerance === null ? entry.value !== reference : Math.abs(Number(entry.value) - Number(reference)) > tolerance);
  const title = attribute === "construction_year" ? "The cadastral year may be a refurbishment." : attribute === "floors" ? "Storeys above ground in the cadastre, including any the street does not show." : "";
  return { text, disagrees, title };
}

/** The cadastre row of the card, with the parcel reference and its page, or null without a match. */
export function referenceRow(card) {
  const cadastre = card?.references?.catastro;
  if (!cadastre?.reference) return null;
  const parcel = String(cadastre.reference).slice(0, 14);
  return { label: "Cadastre", value: parcel, url: cadastre.url || null, title: `${cadastre.note || ""} Footprint overlap ${Math.round(Number(cadastre.iou) * 100)}%.`.trim() };
}

/**
 * One row per attribute: `{ attribute, label, value, source, note, abstained, title, reference }`.
 * `note` names a disagreeing provider ("AI says 4") or what a split vote leaned to; `reference` is the
 * cadastre's value (see referenceFor). An attribute only the cadastre records gets a row too.
 */
export function attributeRows(card) {
  const attributes = card?.attributes || {};
  return CARD_ATTRIBUTES.filter(([attribute]) => attributes[attribute] || referenceFor(card, attribute)).map(([attribute, label]) => {
    const entry = attributes[attribute];
    const reference = referenceFor(card, attribute);
    if (!entry) return { attribute, label, value: "Not surveyed", source: "", note: "", abstained: true, title: "", reference };
    const value = shownValue(attribute, entry, card);
    const row = { attribute, label, value, source: value ? sourceLabel(entry) : "", note: "", abstained: !value, title: "", reference };
    if (!value) {
      row.value = "Not determined";
      row.title = entry.reason || "";
      if (entry.leaning) row.note = `leans ${valueLabel(entry.leaning, card).replace(/^./, (first) => first.toLowerCase())}`;
      return row;
    }
    const disagreeing = (entry.others || []).find((other) => {
      const otherValue = shownValue(attribute, other, card);
      return otherValue && otherValue !== value && !(attribute === "floors" && Math.abs(Number(other.value) - Number(entry.value)) < 1);
    });
    if (disagreeing) row.note = `${sourceName(disagreeing)} says ${shownValue(attribute, disagreeing, card).replace(/^about /, "~")}`;
    if (entry.alternatives?.length && !isTag(entry)) {
      row.title = `Also considered: ${entry.alternatives.map((item) => `${valueLabel(item.value, card)} ${Math.round(item.share * 100)}%`).join(", ")}`;
    }
    return row;
  });
}

/** The model's observations, best view first, without repeating identical sentences. */
export function observationItems(card) {
  const seen = new Set();
  return (card?.observations || []).filter((item) => {
    const text = String(item?.text || "").trim();
    if (!text || seen.has(text)) return false;
    seen.add(text);
    return true;
  });
}

/**
 * What the card says about views left out of the model's crops because their mask was flagged, or null.
 * Without any clean view the model was not asked, and the card says that instead of showing empty answers.
 */
export function setAsideNote(card) {
  const views = card?.set_aside || [];
  if (!views.length) return null;
  const reasons = [...new Set(views.map((view) => view.why).filter(Boolean))];
  const count = views.length === 1 ? "1 view" : `${views.length} views`;
  if (card.no_clean_view) return `No clean view of this building: ${count} set aside (${reasons.join("; ")}). The model was not asked.`;
  return `${count} set aside, not shown to the model: ${reasons.join("; ")}.`;
}

/** Choices for the review form: `{ gem_material: [[code, label]], gem_llrs, building_type, surface_material }`. */
export function reviewOptions(card) {
  const codes = Object.entries(card?.gem?.codes || {});
  // Every lateral load-resisting system code starts with L; no material code does.
  const classes = (name) => (card?.classes?.[name] || []).map((value) => [value, valueLabel(value)]);
  return {
    gem_material: codes.filter(([code]) => !code.startsWith("L")).map(([code]) => [code, valueLabel(code, card)]),
    gem_llrs: codes.filter(([code]) => code.startsWith("L")).map(([code]) => [code, valueLabel(code, card)]),
    building_type: classes("building_type"),
    surface_material: classes("surface_material"),
  };
}

/** What the review form starts from: the values the card shows, "" where none is determined. */
export function reviewFormValues(card) {
  const attributes = card?.attributes || {};
  const shown = (name) => {
    const entry = attributes[name];
    return entry && !entry.abstained && entry.value !== null && entry.value !== undefined ? entry.value : "";
  };
  const whole = (value) => (value === "" ? "" : Math.round(Number(value)));
  return {
    gem_material: shown("gem_material"), gem_llrs: shown("gem_llrs"), floors: whole(shown("floors")),
    building_type: shown("building_type"), surface_material: shown("surface_material"), construction_year: whole(shown("construction_year")),
  };
}

/** "Reviewed by Ana on 2026-10-16: thick walls", or "" without a review. */
export function reviewStatus(card) {
  const review = card?.review;
  if (!review) return "";
  return `${review.evaluation ? "Ground truth label" : "Reviewed"} by ${review.reviewer} on ${String(review.reviewed_at || "").slice(0, 10)}${review.note ? `: ${review.note}` : ""}`;
}
