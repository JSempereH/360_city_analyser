import { coordinatesOf, destinationPoint, mergeBuildingFootprints, normalizeDegrees, outerRingsOf, segmentIntersectionFraction, sortedLocalPanoramas } from "./map-geometry.mjs";

const MAP_BEAM_FOV = 22;
const MAP_BEAM_DISTANCE_M = 20;
const elements = {
  direction: document.querySelector("#direction"),
  map: document.querySelector("#map"),
  message: document.querySelector("#message"),
};

let datasetId = new URLSearchParams(window.location.search).get("dataset");
let images = [];
let map;
let captures;
let buildings;
let view;
let camera;
let captureMarkers = [];
let loadedDataset = null;
let loadedBuildingLocation = null;
let baseBuildingFeatures = [];
let selectedBuildingId = null;
let visibleBuildingIds = null;
let stateRequest = 0;
const channel = "BroadcastChannel" in window ? new BroadcastChannel("panorama-viewer-map") : null;

function viewObstructionDistance(latitude, longitude, heading) {
  if (!buildings) return null;
  const endpoint = destinationPoint(latitude, longitude, heading, MAP_BEAM_DISTANCE_M);
  let nearest = null;
  buildings.eachLayer((layer) => {
    outerRingsOf(layer.feature?.geometry).forEach((ring) => {
      if (ring.length < 2) return;
      const points = ring.map(([pointLongitude, pointLatitude]) => [pointLatitude, pointLongitude]);
      points.forEach((point, index) => {
        const next = points[(index + 1) % points.length];
        const fraction = segmentIntersectionFraction([latitude, longitude], endpoint, point, next);
        if (fraction !== null && (nearest === null || fraction < nearest)) nearest = fraction;
      });
    });
  });
  return nearest === null ? null : MAP_BEAM_DISTANCE_M * nearest;
}

function initializeMap() {
  if (map) return;
  map = window.L.map(elements.map, { scrollWheelZoom: true, zoomControl: true });
  window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 19,
  }).addTo(map);
  captures = window.L.layerGroup().addTo(map);
  buildings = window.L.geoJSON(null, {
    onEachFeature: (feature, layer) => layer.on("click", () => selectBuilding(feature.properties.osm_id)),
    style: buildingStyle,
  }).addTo(map);
  view = window.L.layerGroup().addTo(map);
}

function buildingStyle(feature) {
  const selected = feature.properties?.osm_id === selectedBuildingId;
  const directlyVisible = !visibleBuildingIds || visibleBuildingIds.has(feature.properties?.osm_id);
  const style = selected
    ? { color: "#fff", fillColor: "#ff8a2a", fillOpacity: 0.58, weight: 3 }
    : !directlyVisible
      ? { color: "#59685f", fillColor: "#59685f", fillOpacity: 0.08, weight: 1 }
      : { color: "#d47a36", fillColor: "#ecba74", fillOpacity: 0.28, weight: 1 };
  return style;
}

function refreshBuildingStyles() {
  buildings?.eachLayer((layer) => layer.setStyle(buildingStyle(layer.feature)));
}

function selectBuilding(osmId) {
  if (visibleBuildingIds && !visibleBuildingIds.has(osmId)) {
    elements.message.textContent = "This footprint has no directly visible facade in the current panorama.";
    return;
  }
  selectedBuildingId = osmId || null;
  refreshBuildingStyles();
  channel?.postMessage({ datasetId, focus: true, osmId: selectedBuildingId, type: "building-select" });
}

async function loadBuildingFootprints(coordinates, requestId) {
  const [longitude, latitude] = coordinates;
  const key = `${latitude.toFixed(4)},${longitude.toFixed(4)}`;
  if (key === loadedBuildingLocation) return;
  const response = await fetch(`/api/buildings?lat=${encodeURIComponent(latitude)}&lon=${encodeURIComponent(longitude)}`);
  if (!response.ok) throw new Error(`Building service did not respond (${response.status}).`);
  const payload = await response.json();
  if (requestId !== stateRequest) return;
  const features = Array.isArray(payload.features) ? payload.features : [];
  loadedBuildingLocation = key;
  baseBuildingFeatures = features;
}

async function loadDataset(nextDatasetId, requestId) {
  if (!nextDatasetId || nextDatasetId === loadedDataset) return;
  elements.message.textContent = "Loading dataset captures...";
  const response = await fetch(`/api/datasets/${encodeURIComponent(nextDatasetId)}`);
  if (!response.ok) throw new Error(`Dataset could not be read (${response.status}).`);
  const manifest = await response.json();
  if (requestId !== stateRequest) return;
  datasetId = nextDatasetId;
  loadedDataset = nextDatasetId;
  images = sortedLocalPanoramas(manifest);
  initializeMap();
  captures.clearLayers();
  captureMarkers = [];
  const points = [];
  images.forEach((image, index) => {
    const coordinates = coordinatesOf(image);
    if (!coordinates) return;
    const [longitude, latitude] = coordinates;
    points.push([latitude, longitude]);
    const marker = window.L.circleMarker([latitude, longitude], {
      color: "#7d231c", fillColor: "#d95045", fillOpacity: 0.45, radius: 5, weight: 2,
    }).addTo(captures).bindTooltip(`Panorama ${index + 1}`, { direction: "top" }).on("click", () => {
      channel?.postMessage({ datasetId, index, type: "select" });
    });
    captureMarkers.push({ imageIndex: index, marker });
  });
  if (points.length) map.fitBounds(points, { maxZoom: 18, padding: [30, 30] });
  elements.message.textContent = `${images.length} local panoramas. Select a point to open it in the viewer.`;
}

async function applyState(state) {
  if (!state?.datasetId) return;
  const requestId = ++stateRequest;
  if (state.datasetId !== loadedDataset) await loadDataset(state.datasetId, requestId);
  if (requestId !== stateRequest) return;
  const image = images[state.index];
  const coordinates = image && coordinatesOf(image);
  if (!coordinates) return;
  const [longitude, latitude] = coordinates;
  captureMarkers.forEach(({ imageIndex, marker }) => {
    const current = imageIndex === state.index;
    const markerCoordinates = coordinatesOf(images[imageIndex]);
    if (markerCoordinates) marker.setLatLng([markerCoordinates[1], markerCoordinates[0]]);
    marker.setStyle({ color: "#7d231c", fillOpacity: current ? 0 : 0.45, opacity: current ? 0 : 1, weight: 2 });
    marker.setRadius(5);
  });
  visibleBuildingIds = Array.isArray(state.visibleBuildingIds) ? new Set(state.visibleBuildingIds) : null;
  selectedBuildingId = state.selectedBuildingId || null;
  refreshBuildingStyles();
  await loadBuildingFootprints(coordinates, requestId);
  if (requestId !== stateRequest) return;
  const buildingFeatures = mergeBuildingFootprints(baseBuildingFeatures, state.visibleBuildingFootprints);
  buildings.clearLayers();
  buildings.addData({ type: "FeatureCollection", features: buildingFeatures });
  refreshBuildingStyles();
  if (camera) camera.remove();
  camera = window.L.circleMarker([latitude, longitude], {
    color: "#fff", fillColor: "#d95f3d", fillOpacity: 0.85, radius: 8, weight: 2,
  }).addTo(map).bindTooltip("Mapillary panorama position", { direction: "top" });
  const recordedHeading = Number.isFinite(image.computed_compass_angle_deg) ? image.computed_compass_angle_deg : null;
  if (recordedHeading === null) {
    view.clearLayers();
    elements.direction.textContent = "Current view orientation unavailable";
    map.setView([latitude, longitude], Math.max(map.getZoom(), 18));
    return;
  }
  const heading = normalizeDegrees(recordedHeading + state.yaw * 180 / Math.PI);
  const obstructionDistance = viewObstructionDistance(latitude, longitude, heading);
  const beamDistance = obstructionDistance ?? MAP_BEAM_DISTANCE_M;
  const arc = [];
  for (let step = 0; step <= 12; step += 1) {
    arc.push(destinationPoint(latitude, longitude, heading - MAP_BEAM_FOV / 2 + MAP_BEAM_FOV * step / 12, beamDistance));
  }
  view.clearLayers();
  window.L.polygon([[latitude, longitude], ...arc], {
    color: obstructionDistance === null ? "#2878c9" : "#c95b2b", fillColor: obstructionDistance === null ? "#3c9bdf" : "#e57b45", fillOpacity: 0.2, interactive: false, weight: 2,
  }).addTo(view);
  window.L.polyline([[latitude, longitude], destinationPoint(latitude, longitude, heading, beamDistance)], {
    color: obstructionDistance === null ? "#195c9e" : "#a84222", interactive: false, weight: 3,
  }).addTo(view);
  map.setView([latitude, longitude], Math.max(map.getZoom(), 18));
  elements.direction.textContent = obstructionDistance === null
    ? `Current view: ${heading.toFixed(0)}°`
    : `Current view: ${heading.toFixed(0)}° · blocked by a footprint at ${obstructionDistance.toFixed(0)} m`;
}

channel?.addEventListener("message", (event) => {
  if (event.data?.type === "building-select" && event.data.datasetId === datasetId) {
    selectedBuildingId = event.data.osmId || null;
    refreshBuildingStyles();
    return;
  }
  applyState(event.data).catch((error) => { elements.message.textContent = error.message; });
});

if (!channel) elements.message.textContent = "This browser does not support synchronization between windows.";
else channel.postMessage({ type: "request-state" });
