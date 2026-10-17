// What the presentation mode does and says, built from the attribute summary and the building cards.
// Pure, so the tour order and the wording are tested without a browser; presentation.mjs plays it.

import { valueLabel } from "./building-card.mjs";

/**
 * Buildings to visit, in the order a person walking the sequence would meet them:
 * by the position of their best panorama in the dataset, model answers only.
 * `imageOrder` lists the dataset's image IDs in viewer order. Buildings whose best
 * view scores below `minScore` (a sliver of facade) are left out, unless that
 * would leave nothing to show.
 */
export function planTour(summary, imageOrder, { minScore = 0 } = {}) {
  const position = new Map(imageOrder.map((id, index) => [String(id), index]));
  const answered = Object.entries(summary?.buildings || {}).filter(([, entry]) => entry.model);
  const visible = answered.filter(([, entry]) => !Number.isFinite(entry.score) || entry.score >= minScore);
  return (visible.length ? visible : answered)
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
export function reasoningScript(card, { maxViews = 3 } = {}) {
  const attributes = card?.attributes || {};
  const views = (card?.views || []).filter((view) => view.image_id).slice(0, maxViews);
  const steps = [];
  const label = (code) => valueLabel(code, card);
  steps.push({
    type: "intro",
    text: `Building ${card.building}. ${views.length === 1 ? "One view of it was captured" : `${views.length} views of it were captured`}; I will look at ${views.length === 1 ? "it" : "each one"} before deciding, starting with the clearest.`,
  });
  views.forEach((view, index) => {
    const answer = view.answer || {};
    const read = [
      answer.floors ? `${answer.floors} storeys` : null,
      answer.gem_material ? `${lower(label(answer.gem_material))}` : null,
      answer.gem_llrs ? `${lower(label(answer.gem_llrs))}` : null,
    ].filter(Boolean).join(" · ");
    steps.push({
      type: "view", index, imageId: String(view.image_id), view: view.view, cropUrl: view.crop_url,
      // The model's own field notes when it wrote them, otherwise its one-line observations.
      text: view.error || !(view.reasoning || answer.observations)
        ? `View ${index + 1}: the image does not show enough of the facade; I set it aside.`
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
