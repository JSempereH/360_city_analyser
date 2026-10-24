// What the presentation mode does and says, built from the attribute summary and the building cards.
// Pure, so the tour order and the wording are tested without a browser; presentation.mjs plays it.

import { referenceFor, valueLabel } from "./building-card.mjs";

/**
 * Buildings to visit, in the order a person walking the sequence would meet them:
 * by the position of their best panorama in the dataset, model answers only.
 * `imageOrder` lists the dataset's image IDs in viewer order. Buildings whose best
 * view scores below `minScore` (a sliver of facade) are left out; when none
 * reaches it, the best-seen few are shown rather than every sliver.
 */
const FALLBACK_STOPS = 3;

export function planTour(summary, imageOrder, { minScore = 0 } = {}) {
  const position = new Map(imageOrder.map((id, index) => [String(id), index]));
  // A building whose every view was set aside has nothing to show: the tour skips it.
  const answered = Object.entries(summary?.buildings || {}).filter(([, entry]) => entry.model && !entry.no_clean_view);
  const visible = answered.filter(([, entry]) => !Number.isFinite(entry.score) || entry.score >= minScore);
  const best = [...answered].sort((first, second) => (second[1].score ?? 0) - (first[1].score ?? 0)).slice(0, FALLBACK_STOPS);
  return (visible.length ? visible : best)
    .map(([osmId, entry]) => ({ osmId, image: entry.image ? String(entry.image) : null, order: position.has(String(entry.image)) ? position.get(String(entry.image)) : Infinity }))
    .sort((first, second) => first.order - second.order || first.osmId.localeCompare(second.osmId))
    .map(({ osmId, image }) => ({ osmId, image }));
}

function lower(label) {
  return label.charAt(0).toLowerCase() + label.slice(1);
}

function percent(share) {
  return Number.isFinite(share) ? `${Math.round(share * 100)}%` : "";
}

function floorsList(values) {
  return values.map((value, index) => `view ${index + 1}: ${value}`).join(", ");
}

/**
 * Steps of one building's visit:
 * `{ type: "intro" | "view" | "floors" | "structure" | "decision" | "save", text, ... }`.
 * A "view" step carries the panorama and camera framing to move to before its text.
 */
// Views left out before the model read anything, because their mask disagreed with the footprint.
function setAsideSentence(card) {
  const views = card?.set_aside || [];
  if (!views.length) return "";
  const reasons = [...new Set(views.map((view) => view.why).filter(Boolean))];
  return ` ${views.length === 1 ? "One more view was" : `${views.length} more views were`} set aside before reading: ${reasons.join("; ")}.`;
}

export function reasoningScript(card, { maxViews = 3 } = {}) {
  const attributes = card?.attributes || {};
  const views = (card?.views || []).filter((view) => view.image_id).slice(0, maxViews);
  const steps = [];
  const label = (code) => valueLabel(code, card);
  steps.push({
    type: "intro",
    text: `Building ${card.building}. ${views.length === 1 ? "One view of it was captured" : `${views.length} views of it were captured`}; I will look at ${views.length === 1 ? "it" : "each one"} before deciding, starting with the clearest.${setAsideSentence(card)}`,
  });
  views.forEach((view, index) => {
    const answer = view.answer || {};
    const read = [
      answer.floors ? `${answer.floors} storeys` : null,
      answer.gem_material ? `${lower(label(answer.gem_material))}` : null,
      answer.gem_llrs ? `${lower(label(answer.gem_llrs))}` : null,
    ].filter(Boolean).join(" · ");
    // A view without a usable answer is set aside, in the model's own words when it gave them; its crop is
    // not shown, since what remains visible in it is mostly the neighbors.
    const setAside = Boolean(view.error) || !(view.reasoning || answer.observations);
    const reason = String(answer.observations || "").trim().replace(/[.;\s]+$/, "");
    steps.push({
      type: "view", index, imageId: String(view.image_id), view: view.view, cropUrl: setAside ? null : view.crop_url, setAside,
      // The model's own field notes when it wrote them, otherwise its one-line observations.
      text: setAside
        ? `View ${index + 1}. ${reason || "The image does not show enough of the facade"}; I set it aside.`
        : `View ${index + 1}. ${view.reasoning || answer.observations}${read ? `\n→ ${read}` : ""}`,
    });
  });
  const floors = attributes.floors;
  const counted = views.map((view) => view.answer?.floors).filter((value) => Number.isFinite(value));
  if (floors || counted.length) {
    const agree = counted.length > 1 && counted.every((value) => value === counted[0]);
    let text = counted.length
      ? `Counting window rows per view (${floorsList(counted)}). ${counted.length === 1 ? "" : agree ? "The views agree." : "The views differ by a storey, so I take the median weighted by how well each view shows the building."}`
      : "No view has regular window rows to count storeys from, so I leave the storey count open.";
    if (floors?.source === "osm_tag") {
      const model = (floors.others || []).find((other) => String(other.provider || "").startsWith("vlm:") && Number.isFinite(other.value));
      text += model && Math.abs(model.value - floors.value) >= 1
        ? ` OpenStreetMap records ${floors.value} levels while the images suggest ${model.value}. I keep the mapped value and flag the difference.`
        : ` OpenStreetMap records ${floors.value} levels, consistent with the images.`;
    } else if (floors?.source === "human_review") {
      text += ` An engineer already reviewed it: ${floors.value} storeys.`;
    }
    steps.push({ type: "floors", text: text.trim(), counts: counted, value: floors && !floors.abstained ? floors.value : null });
  }
  const material = attributes.gem_material || {};
  const system = attributes.gem_llrs || {};
  const decided = (entry) => !entry.abstained && entry.value;
  const parts = [];
  if (decided(material)) parts.push(`Material of the structure: ${lower(label(material.value))}${material.share && material.score_kind === "vote_share" ? `, ${percent(material.share)} of the votes` : ""}.`);
  else if (material.leaning) parts.push(`The views disagree on the material; ${lower(label(material.leaning))} leads with ${percent(material.share)}, not enough to decide.`);
  if (decided(system)) parts.push(`Lateral load-resisting system: ${lower(label(system.value))}${system.share && system.score_kind === "vote_share" ? `, ${percent(system.share)} of the votes` : ""}.`);
  else if (system.leaning) parts.push(`The views disagree on the lateral system; ${lower(label(system.leaning))} leads with ${percent(system.share)}. I will not guess.`);
  const needsReview = !decided(material) || !decided(system) || (floors?.others || []).some((other) => String(other.provider || "").startsWith("vlm:") && floors.source === "osm_tag" && Math.abs((other.value ?? floors.value) - floors.value) >= 1);
  if (parts.length) steps.push({ type: "structure", text: parts.join(" ") });
  const fields = [
    { key: "floors", label: "Floors", value: floors && !floors.abstained ? String(floors.value) : "Not determined" },
    { key: "gem_material", label: "GEM material", value: decided(material) ? `${material.value} · ${label(material.value).replace(/ \([A-Z0-9]+\)$/, "")}` : "MAT99 · not determined" },
    { key: "gem_llrs", label: "GEM lateral system", value: decided(system) ? `${system.value} · ${label(system.value).replace(/ \([A-Z0-9]+\)$/, "")}` : "L99 · not determined" },
  ];
  steps.push({ type: "decision", fields, gem: card.gem?.string || "", needsReview });
  steps.push({ type: "save", text: needsReview ? "Saved, and flagged for an engineer to review." : "Saved to the building record." });
  return steps;
}

/**
 * After the record is saved, the assistant compares it with the Spanish cadastre, when the building
 * is matched to it: `{ text, differs }`, or null. The model never saw the cadastre; this is the
 * check an engineer would make, and a difference is explained rather than hidden.
 */
export function cadastreCheck(card) {
  const cadastre = card?.references?.catastro;
  if (!cadastre) return null;
  const attributes = card.attributes || {};
  const answered = (name) => {
    const entry = attributes[name];
    return entry && !entry.abstained && entry.value !== null && entry.value !== undefined ? entry : null;
  };
  const sentences = [];
  let differs = false;
  const floors = answered("floors");
  if (Number.isFinite(cadastre.floors)) {
    const reference = referenceFor(card, "floors");
    if (!floors) sentences.push(`The cadastre records ${cadastre.floors} storeys above ground; I could not count them from the street.`);
    else if (reference?.disagrees) {
      differs = true;
      sentences.push(`The cadastre records ${cadastre.floors} storeys above ground and I counted ${Math.round(floors.value)}: the usual cause is a mezzanine or a set-back attic that the street does not show.`);
    } else sentences.push(`The cadastre records ${cadastre.floors} storeys above ground, as counted.`);
  }
  const year = answered("construction_year");
  if (Number.isFinite(cadastre.construction_year) && year) {
    if (referenceFor(card, "construction_year")?.disagrees) {
      differs = true;
      sentences.push(`It dates the building ${cadastre.construction_year}, where the facade reads as about ${Math.round(year.value)}; a cadastral year is often that of a refurbishment.`);
    } else sentences.push(`Its year, ${cadastre.construction_year}, agrees with the facade.`);
  }
  const use = answered("building_type");
  if (cadastre.building_type && use) {
    if (referenceFor(card, "building_type")?.disagrees) {
      differs = true;
      sentences.push(`Its use is ${valueLabel(cadastre.building_type).toLowerCase()}, not ${valueLabel(use.value).toLowerCase()}.`);
    } else sentences.push(`Same use: ${valueLabel(cadastre.building_type).toLowerCase()}.`);
  }
  if (!sentences.length) return null;
  return { text: `Cross-check with the Spanish cadastre (parcel ${String(cadastre.reference || "").slice(0, 14)}). ${sentences.join(" ")}`, differs };
}

const MATERIAL_WORDS = { CR: "reinforced concrete", CU: "unreinforced concrete", C99: "concrete", SRC: "composite steel and concrete", S: "steel", ME: "metal", MUR: "unreinforced masonry", MCF: "confined masonry", MR: "reinforced masonry", M99: "masonry", EU: "unreinforced earth", ER: "reinforced earth", E99: "earth", W: "wood" };

/** What the assistant says at the end of a walk: the structures it found and what an engineer should check. */
export function surveySummary(summary, saved) {
  const entries = Object.entries(summary?.buildings || {}).filter(([osmId, entry]) => entry.model && (!saved || saved.has(osmId)));
  if (!entries.length) return "";
  const counts = new Map();
  for (const [, entry] of entries) {
    const word = MATERIAL_WORDS[entry.gem_material] || "structure not determined";
    counts.set(word, (counts.get(word) || 0) + 1);
  }
  const parts = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([word, count]) => `${count} ${word}`);
  const flagged = entries.filter(([, entry]) => entry.needs_review).length;
  const list = parts.length > 1 ? `${parts.slice(0, -1).join(", ")} and ${parts.at(-1)}` : parts[0];
  return `${entries.length} building${entries.length === 1 ? "" : "s"} described and saved: ${list}. The map is colored by what was found. `
    + (flagged ? `${flagged} ${flagged === 1 ? "is" : "are"} flagged for an engineer; they wait in the Review tab, under "Structure to check", weakest answer first.` : "None needs an engineer's check.");
}

