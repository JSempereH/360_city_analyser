// One ordered queue for review: footprint proposals waiting for a decision
// first, then suspected footprint problems by how many panoramas support them,
// then buildings the imagery shows without a footprint, then model answers
// about a building's structure that deserve a human look (weakest first).

export const REVIEW_KINDS = {
  proposal: "Proposed correction",
  merged_footprint: "Several buildings in one footprint",
  unsupported_footprint: "No building in the imagery",
  unexplained_instance: "Building without a footprint",
  attributes: "Structure to check",
  mask: "Mask to check",
};

function proposalItems(records) {
  return records.flatMap((record) => record.revisions
    .filter((revision) => revision.status === "proposed")
    .map((revision) => {
      const evidence = (revision.evidence || []).find((item) => item?.dataset && item?.image) || {};
      return {
        key: `proposal:${record.osm_id}:${revision.revision}`,
        kind: "proposal",
        osmId: record.osm_id,
        revision: revision.revision,
        // A split is previewed as all of its parts.
        geometry: revision.geometry || { type: "GeometryCollection", geometries: revision.parts || [] },
        dataset: evidence.dataset || null,
        image: evidence.image ? String(evidence.image) : null,
        label: `${record.osm_id} · revision ${revision.revision}`,
        detail: `${revision.kind === "split" ? `Split into ${(revision.parts || []).length} parts · ` : ""}${revision.note} · by ${revision.author}`,
      };
    }));
}

function buildingItems(buildings) {
  return buildings.flatMap((entry) => Object.entries(entry.issues || {}).map(([kind, count]) => {
    const panorama = (entry.panoramas || []).find((item) => item.type === kind) || entry.panoramas?.[0] || {};
    return {
      key: `${kind}:${entry.osm_id}`,
      kind,
      osmId: entry.osm_id,
      dataset: panorama.dataset || null,
      image: panorama.image ? String(panorama.image) : null,
      support: entry.supporting_panoramas || 0,
      correction: entry.correction?.status || null,
      label: entry.osm_id,
      detail: `${count} flag${count === 1 ? "" : "s"} · ${entry.supporting_panoramas} panorama${entry.supporting_panoramas === 1 ? "" : "s"}${entry.correction ? ` · correction ${entry.correction.status}` : ""}`,
    };
  })).sort((first, second) => second.support - first.support || first.label.localeCompare(second.label));
}

function unexplainedItems(instances) {
  return instances.map((entry) => ({
    key: `unexplained_instance:${entry.dataset}:${entry.image}:${entry.bearings.join("-")}`,
    kind: "unexplained_instance",
    osmId: null,
    dataset: entry.dataset,
    image: String(entry.image),
    bearings: entry.bearings,
    coordinates: [entry.longitude, entry.latitude],
    label: `${entry.dataset}/${entry.image}`,
    detail: `Bearings ${entry.bearings[0]}° to ${entry.bearings[1]}°`,
  }));
}

// Buildings of the attribute summary whose model answer is split, weak or contradicts OSM.
function attributeItems(summary) {
  if (!summary?.buildings) return [];
  return Object.entries(summary.buildings)
    .filter(([, entry]) => entry.needs_review && !entry.reviewed)
    .map(([osmId, entry]) => ({
      key: `attributes:${summary.dataset}:${osmId}`,
      kind: "attributes",
      osmId,
      dataset: summary.dataset,
      image: entry.image ? String(entry.image) : null,
      urgency: Number.isFinite(entry.urgency) ? entry.urgency : 1,
      values: { gem_material: entry.gem_material, gem_llrs: entry.gem_llrs, floors: entry.floors },
      label: osmId,
      detail: `${entry.gem} · ${entry.needs_review}`,
    }))
    .sort((first, second) => first.urgency - second.urgency || first.label.localeCompare(second.label));
}

// Buildings with views whose mask was set aside, those without any clean view first.
function maskItems(summary) {
  if (!summary?.buildings) return [];
  return Object.entries(summary.buildings)
    .filter(([, entry]) => entry.set_aside > 0 && !entry.reviewed)
    .map(([osmId, entry]) => ({
      key: `mask:${summary.dataset}:${osmId}`,
      kind: "mask",
      osmId,
      dataset: summary.dataset,
      image: entry.set_aside_image ? String(entry.set_aside_image) : entry.image ? String(entry.image) : null,
      urgency: entry.no_clean_view ? 0 : entry.set_aside / (entry.set_aside + (entry.clean_views || 0)),
      label: osmId,
      detail: `${entry.no_clean_view ? "No clean view" : `${entry.set_aside} view${entry.set_aside === 1 ? "" : "s"} set aside`} · ${(entry.set_aside_reasons || []).join(", ")}`,
    }))
    .sort((first, second) => first.urgency - second.urgency || first.label.localeCompare(second.label));
}

// filters: { kind: "all" | one of REVIEW_KINDS, dataset: id or null for all,
// unresolvedOnly: hide problems whose building already has an accepted correction }
// attributeSummary: the current dataset's attribute summary, or null.
export function buildReviewQueue(corrections, issues, filters = {}, attributeSummary = null) {
  const { kind = "all", dataset = null, unresolvedOnly = true } = filters;
  const items = [
    ...proposalItems(corrections || []),
    ...buildingItems(issues?.buildings || []),
    ...unexplainedItems(issues?.unexplained_instances || []),
    ...attributeItems(attributeSummary),
    ...maskItems(attributeSummary),
  ];
  return items.filter((item) => (kind === "all" || item.kind === kind)
    && (!dataset || !item.dataset || item.dataset === dataset)
    && !(unresolvedOnly && item.correction === "accepted"));
}

export function stepIndex(length, index, step) {
  if (!length) return -1;
  if (index < 0) return step > 0 ? 0 : length - 1;
  return (index + step + length) % length;
}

// Keep the reviewer on the same item after the queue is rebuilt, or on the
// item that took its place when it was resolved.
export function preservedIndex(items, previousKey, previousIndex) {
  const same = items.findIndex((item) => item.key === previousKey);
  if (same >= 0) return same;
  // Nothing was selected yet: stay before the first item, so "next" opens it.
  if (!items.length || previousIndex < 0) return -1;
  return Math.min(Math.max(previousIndex, 0), items.length - 1);
}

export function bearingCenter([from, to]) {
  const span = ((to - from) % 360 + 360) % 360;
  return (from + span / 2) % 360;
}
