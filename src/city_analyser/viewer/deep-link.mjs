const DEFAULT_FOV = 90;
const MIN_FOV = 45;
const MAX_FOV = 110;
const MAX_PITCH_DEG = 90 - 0.08 * 180 / Math.PI;
const BUILDING_ID = /^(?:node|way|relation)\/[1-9][0-9]*(?:\/part-[1-9][0-9]*)?$/;
// Review queue item keys, e.g. "merged_footprint:way/5" or "proposal:way/1:2".
const REVIEW_KEY = /^(?:proposal|merged_footprint|unsupported_footprint|unexplained_instance):[\w/:.\- ]{1,200}$/;

function optionalNumber(parameters, name, minimum, maximum, normalize = false) {
  const raw = parameters.get(name);
  if (raw === null || raw.trim() === "") return null;
  const value = Number(raw);
  if (!Number.isFinite(value)) return null;
  const normalized = normalize ? ((value + 180) % 360 + 360) % 360 - 180 : value;
  return Math.max(minimum, Math.min(maximum, normalized));
}

export function parseViewerTarget(search) {
  const parameters = search instanceof URLSearchParams ? search : new URLSearchParams(search);
  const buildingId = parameters.get("building");
  return {
    datasetId: parameters.get("dataset") || null,
    imageId: parameters.get("image") || null,
    buildingId: buildingId && BUILDING_ID.test(buildingId) ? buildingId : null,
    yawDeg: optionalNumber(parameters, "yaw", -180, 180, true),
    pitchDeg: optionalNumber(parameters, "pitch", -MAX_PITCH_DEG, MAX_PITCH_DEG),
    fovDeg: optionalNumber(parameters, "fov", MIN_FOV, MAX_FOV),
    reviewKey: REVIEW_KEY.test(parameters.get("review") || "") ? parameters.get("review") : null,
  };
}

export function buildViewerSearch(target) {
  const parameters = new URLSearchParams();
  if (target.datasetId) parameters.set("dataset", target.datasetId);
  if (target.imageId) parameters.set("image", String(target.imageId));
  if (target.buildingId && BUILDING_ID.test(target.buildingId)) parameters.set("building", target.buildingId);
  if (Number.isFinite(target.yawDeg)) parameters.set("yaw", Number(target.yawDeg).toFixed(2));
  if (Number.isFinite(target.pitchDeg)) parameters.set("pitch", Number(target.pitchDeg).toFixed(2));
  parameters.set("fov", Number.isFinite(target.fovDeg) ? Number(target.fovDeg).toFixed(2) : DEFAULT_FOV.toFixed(2));
  if (target.reviewKey && REVIEW_KEY.test(target.reviewKey)) parameters.set("review", target.reviewKey);
  return `?${parameters.toString()}`;
}

// A search box entry: an OSM element ("way/123", "relation/45", or a bare
// number, read as a way) or coordinates ("9.93, -84.08").
export function parseSearchQuery(text) {
  const query = String(text || "").trim().toLowerCase().replace(/\s+/g, " ");
  const element = query.match(/^(?:(node|way|relation)\s*\/?\s*)?([1-9][0-9]*)(?:\/(part-[1-9][0-9]*))?$/);
  if (element) return { type: "osm", osmId: `${element[1] || "way"}/${element[2]}${element[3] ? `/${element[3]}` : ""}` };
  const coordinates = query.match(/^(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)$/);
  if (coordinates) {
    const latitude = Number(coordinates[1]);
    const longitude = Number(coordinates[2]);
    if (Math.abs(latitude) <= 90 && Math.abs(longitude) <= 180) return { type: "coordinates", latitude, longitude };
  }
  return null;
}

export function findImageIndex(images, imageId) {
  if (imageId === null || imageId === undefined) return 0;
  return images.findIndex((image) => String(image.id) === String(imageId));
}

export { DEFAULT_FOV, MAX_PITCH_DEG };
