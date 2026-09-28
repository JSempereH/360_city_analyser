import { nearbyPanoramas, nearestPanoramasByBearing, projectGroundPanoramaToView, signedDegrees } from "./navigation.mjs";
import { buildViewerSearch, DEFAULT_FOV, findImageIndex, parseViewerTarget } from "./deep-link.mjs";
import { coordinatesOf, destinationPoint, mergeBuildingFootprints, normalizeDegrees, outerRingsOf, segmentIntersectionFraction, sortedLocalPanoramas } from "./map-geometry.mjs";

const elements = {
  analyzeBuildingsButton: document.querySelector("#analyze-buildings-button"),
  refineNearbyPosesButton: document.querySelector("#refine-nearby-poses-button"),
  buildingStatus: document.querySelector("#building-status"),
  canvas: document.querySelector("#panorama-canvas"),
  capturedAt: document.querySelector("#captured-at"),
  coordinates: document.querySelector("#coordinates"),
  creator: document.querySelector("#creator"),
  deleteDatasetButton: document.querySelector("#delete-dataset-button"),
  datasetSelect: document.querySelector("#dataset-select"),
  datasetSummary: document.querySelector("#dataset-summary"),
  error: document.querySelector("#error"),
  heading: document.querySelector("#heading"),
  generateReportButton: document.querySelector("#generate-report-button"),
  imageList: document.querySelector("#image-list"),
  loading: document.querySelector("#loading"),
  mapillaryAoiMap: document.querySelector("#mapillary-aoi-map"),
  mapillaryClearAoiButton: document.querySelector("#mapillary-clear-aoi-button"),
  mapillaryDatasetName: document.querySelector("#mapillary-dataset-name"),
  mapillaryDrawAoiButton: document.querySelector("#mapillary-draw-aoi-button"),
  mapillaryDownloadButton: document.querySelector("#mapillary-download-button"),
  mapillaryDownloadAllButton: document.querySelector("#mapillary-download-all-button"),
  mapillaryImageSize: document.querySelector("#mapillary-image-size"),
  mapillaryDialog: document.querySelector("#mapillary-dialog"),
  mapillaryResultList: document.querySelector("#mapillary-result-list"),
  mapillaryResults: document.querySelector("#mapillary-results"),
  mapillarySearchButton: document.querySelector("#mapillary-search-button"),
  mapillarySearchStatus: document.querySelector("#mapillary-search-status"),
  mapillarySelectAllButton: document.querySelector("#mapillary-select-all-button"),
  mapPanel: document.querySelector("#map-panel"),
  mapDragHandle: document.querySelector("#map-drag-handle"),
  mapDetachButton: document.querySelector("#map-detach-button"),
  mapPopoutButton: document.querySelector("#map-popout-button"),
  mapToggleButton: document.querySelector("#map-toggle-button"),
  mapDirection: document.querySelector("#map-direction"),
  mapMessage: document.querySelector("#map-message"),
  mapCanvas: document.querySelector("#location-map-canvas"),
  maskToggleButton: document.querySelector("#mask-toggle-button"),
  metadata: document.querySelector("#metadata"),
  nextButton: document.querySelector("#next-button"),
  previousButton: document.querySelector("#previous-button"),
  resetButton: document.querySelector("#reset-button"),
  searchMapillaryButton: document.querySelector("#search-mapillary-button"),
  sceneNavigation: document.querySelector("#scene-navigation"),
  sequencePosition: document.querySelector("#sequence-position"),
  sourceLink: document.querySelector("#source-link"),
  status: document.querySelector("#status"),
  closeMapillaryDialogButton: document.querySelector("#close-mapillary-dialog-button"),
};

const vertexShaderSource = `
  attribute vec2 position;
  varying vec2 screenPosition;
  void main() {
    screenPosition = position;
    gl_Position = vec4(position, 0.0, 1.0);
  }
`;

const fragmentShaderSource = `
  precision highp float;
  varying vec2 screenPosition;
  uniform sampler2D panorama;
  uniform sampler2D selectionMask;
  uniform sampler2D vegetationMask;
  uniform float selectionVisible;
  uniform float selectionEmphasis;
  uniform vec2 maskSize;
  uniform float vegetationVisible;
  uniform float aspect;
  uniform float yaw;
  uniform float pitch;
  uniform float fov;
  const float PI = 3.141592653589793;
  vec3 rotateYaw(vec3 vector, float angle) {
    float cosine = cos(angle); float sine = sin(angle);
    return vec3(cosine * vector.x + sine * vector.z, vector.y, -sine * vector.x + cosine * vector.z);
  }
  vec3 rotatePitch(vec3 vector, float angle) {
    float cosine = cos(angle); float sine = sin(angle);
    return vec3(vector.x, cosine * vector.y - sine * vector.z, sine * vector.y + cosine * vector.z);
  }
  void main() {
    // fov is horizontal, so the perspective stays stable when the panel size changes.
    float field = tan(radians(fov) * 0.5) / aspect;
    // The center of an equirectangular panorama is the forward direction.
    vec3 direction = normalize(vec3(screenPosition.x * aspect * field, screenPosition.y * field, 1.0));
    // Pitch around the viewer's local right axis first, then yaw around world up.
    // Applying pitch after yaw uses a global axis and makes the horizon roll.
    direction = rotateYaw(rotatePitch(direction, pitch), yaw);
    float longitude = atan(direction.x, direction.z);
    float latitude = asin(clamp(direction.y, -1.0, 1.0));
    vec2 panoramaUv = vec2(0.5 + longitude / (2.0 * PI), 0.5 - latitude / PI);
    vec4 panoramaColor = texture2D(panorama, panoramaUv);
    float selected = texture2D(selectionMask, panoramaUv).r * selectionVisible;
    float vegetation = texture2D(vegetationMask, panoramaUv).r * vegetationVisible;
    vec4 completedColor = mix(panoramaColor, vec4(0.20, 0.65, 1.0, 1.0), vegetation * 0.18);
    // Overview: a light tint on every matched building. Selection: a stronger
    // fill plus an outline, so one building stands out on dark facades too.
    float fillStrength = mix(0.26, 0.42, selectionEmphasis);
    vec4 color = mix(completedColor, vec4(1.0, 0.57, 0.12, 1.0), selected * fillStrength);
    if (selectionEmphasis > 0.5 && selectionVisible > 0.5) {
      vec2 texel = 1.5 / maskSize;
      float neighbors = texture2D(selectionMask, panoramaUv + vec2(texel.x, 0.0)).r
        + texture2D(selectionMask, panoramaUv - vec2(texel.x, 0.0)).r
        + texture2D(selectionMask, panoramaUv + vec2(0.0, texel.y)).r
        + texture2D(selectionMask, panoramaUv - vec2(0.0, texel.y)).r;
      float outline = selected * step(0.5, 4.0 - neighbors);
      color = mix(color, vec4(1.0, 0.86, 0.55, 1.0), outline);
    }
    gl_FragColor = color;
  }
`;

let datasets = [];
let datasetId = null;
let images = [];
let currentIndex = 0;
let activeLoad = 0;
let yaw = 0;
let pitch = 0;
let fov = DEFAULT_FOV;
let pointer = null;
// Keep a small margin from the poles to avoid the equirectangular singularity,
// while allowing inspection of tall building facades.
const MAX_PITCH = Math.PI / 2 - 0.08;
const SAFE_TEXTURE_SIZE = 4096;
const MAP_BEAM_FOV = 22;
const MAP_BEAM_DISTANCE_M = 20;
const ASSUMED_CAMERA_HEIGHT_M = 2.1;
let activeTexture = null;
let selectionTexture = null;
let vegetationTexture = null;
let selectionIsSingleBuilding = false;
let selectionMaskSize = [2048, 1024];
let currentAnalysis = null;
let currentSfmRefinement = null;
let selectedBuildingId = null;
let buildingOverlayVisible = true;
let analysisRequest = 0;
let sfmRequest = 0;
let maskLoadRequest = 0;
let analysisJobRunning = false;
let analysisBackend = { kind: "local", label: "local device" };
let locationMap = null;
let cameraMarker = null;
let buildingLayer = null;
let viewLayer = null;
let captureLayer = null;
let currentMapImage = null;
let viewUpdateFrame = null;
let mapLoad = 0;
const buildingCache = new Map();
let currentBuildingFeatures = [];
let sceneNavigationTargets = new Map();
const sceneNavigationControls = new Map();
let mapillaryAoiMap = null;
let aoiRectangle = null;
let aoiStart = null;
let mapillaryAoiDrawing = false;
let mapillaryResultsLayer = null;
let mapillarySearchBbox = null;
let mapillarySearchResults = [];
let mapillaryDownloadCandidates = [];
const mapillaryResultControls = new Map();
const selectedMapillaryIds = new Set();
const mapWindowChannel = "BroadcastChannel" in window ? new BroadcastChannel("panorama-viewer-map") : null;
let datasetLoadRequest = 0;

// WebGL2 samples non-power-of-two panoramas with REPEAT wrapping; WebGL1 does
// not and renders them black, so it is only a fallback with resized textures.
const gl = elements.canvas.getContext("webgl2", { antialias: true }) || elements.canvas.getContext("webgl", { antialias: true });
const webgl2 = typeof WebGL2RenderingContext !== "undefined" && gl instanceof WebGL2RenderingContext;
let program;
let uniforms;

function formatDate(value) {
  if (!value) return "No date";
  return new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

function syncViewerUrl() {
  const image = images[currentIndex];
  if (!datasetId || !image) return;
  const search = buildViewerSearch({
    datasetId,
    imageId: image.id,
    buildingId: selectedBuildingId,
    yawDeg: yaw * 180 / Math.PI,
    pitchDeg: -pitch * 180 / Math.PI,
    fovDeg: fov,
  });
  window.history.replaceState(null, "", `${window.location.pathname}${search}`);
}

function coordinateLabel(image) {
  const coordinates = coordinatesOf(image);
  return coordinates ? `${coordinates[1].toFixed(6)}, ${coordinates[0].toFixed(6)}` : "No coordinates";
}

function showError(message) {
  elements.error.textContent = message;
  elements.error.hidden = false;
  elements.loading.hidden = true;
  setActivity(elements.loading, false);
  elements.status.textContent = "The panorama could not be displayed.";
  setActivity(elements.status, false);
}

function clearError() {
  elements.error.hidden = true;
}

function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) || "WebGL could not compile.");
  return shader;
}

function initializeRenderer() {
  if (!gl) throw new Error("This browser does not support WebGL.");
  program = gl.createProgram();
  gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexShaderSource));
  gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentShaderSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) || "WebGL could not start.");
  gl.useProgram(program);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  uniforms = {
    aspect: gl.getUniformLocation(program, "aspect"), fov: gl.getUniformLocation(program, "fov"), panorama: gl.getUniformLocation(program, "panorama"),
    pitch: gl.getUniformLocation(program, "pitch"), selectionMask: gl.getUniformLocation(program, "selectionMask"),
    selectionVisible: gl.getUniformLocation(program, "selectionVisible"), vegetationMask: gl.getUniformLocation(program, "vegetationMask"),
    selectionEmphasis: gl.getUniformLocation(program, "selectionEmphasis"), maskSize: gl.getUniformLocation(program, "maskSize"),
    vegetationVisible: gl.getUniformLocation(program, "vegetationVisible"), yaw: gl.getUniformLocation(program, "yaw"),
  };
  gl.uniform1i(uniforms.panorama, 0);
  gl.uniform1i(uniforms.selectionMask, 1);
  gl.uniform1i(uniforms.vegetationMask, 2);
  clearSelectionMask();
  gl.clearColor(0.03, 0.04, 0.035, 1);
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();
}

function resizeCanvas() {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(elements.canvas.clientWidth * ratio));
  const height = Math.max(1, Math.floor(elements.canvas.clientHeight * ratio));
  if (elements.canvas.width !== width || elements.canvas.height !== height) {
    elements.canvas.width = width;
    elements.canvas.height = height;
  }
  gl.viewport(0, 0, width, height);
  render();
}

function render() {
  if (!program) return;
  gl.useProgram(program);
  gl.uniform1f(uniforms.aspect, elements.canvas.width / elements.canvas.height);
  gl.uniform1f(uniforms.yaw, yaw);
  gl.uniform1f(uniforms.pitch, pitch);
  gl.uniform1f(uniforms.fov, fov);
  gl.uniform1f(uniforms.selectionVisible, selectionTexture && buildingOverlayVisible ? 1 : 0);
  gl.uniform1f(uniforms.vegetationVisible, vegetationTexture && buildingOverlayVisible ? 1 : 0);
  gl.uniform1f(uniforms.selectionEmphasis, selectionIsSingleBuilding ? 1 : 0);
  gl.uniform2f(uniforms.maskSize, selectionMaskSize[0], selectionMaskSize[1]);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

function uploadTexture(image) {
  const reportedMaxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  // 8K equirectangular textures use roughly 128 MB uncompressed and can make
  // otherwise capable integrated GPUs lose the WebGL context. A 4K cap keeps
  // the viewer responsive while preserving twice the detail of thumb_2048.
  const maxTextureSize = Math.min(reportedMaxTextureSize, SAFE_TEXTURE_SIZE);
  let textureSource = image;
  let reduced = false;
  const powerOfTwo = (value) => value > 0 && (value & (value - 1)) === 0;
  const needsPowerOfTwo = !webgl2 && (!powerOfTwo(image.naturalWidth) || !powerOfTwo(image.naturalHeight));
  if (image.naturalWidth > maxTextureSize || image.naturalHeight > maxTextureSize || needsPowerOfTwo) {
    const scale = Math.min(maxTextureSize / image.naturalWidth, maxTextureSize / image.naturalHeight, 1);
    const resized = document.createElement("canvas");
    resized.width = Math.floor(image.naturalWidth * scale);
    resized.height = Math.floor(image.naturalHeight * scale);
    if (needsPowerOfTwo) {
      // Largest power of two that does not upscale the panorama.
      resized.width = 2 ** Math.floor(Math.log2(resized.width));
      resized.height = 2 ** Math.floor(Math.log2(resized.height));
    }
    resized.getContext("2d").drawImage(image, 0, 0, resized.width, resized.height);
    textureSource = resized;
    reduced = true;
  }
  const texture = gl.createTexture();
  while (gl.getError() !== gl.NO_ERROR) {
    // Clear errors from a failed previous upload before checking this one.
  }
  gl.bindTexture(gl.TEXTURE_2D, texture);
  // Mapillary JPEG rows already use the orientation expected by this shader.
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, textureSource);
  const textureError = gl.getError();
  if (textureError !== gl.NO_ERROR) {
    gl.deleteTexture(texture);
    throw new Error(`WebGL no pudo cargar la textura (error ${textureError}).`);
  }
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  if (activeTexture) gl.deleteTexture(activeTexture);
  activeTexture = texture;
  render();
  return {
    reduced,
    height: textureSource.height || textureSource.naturalHeight,
    maxTextureSize,
    originalHeight: image.naturalHeight,
    originalWidth: image.naturalWidth,
    width: textureSource.width || textureSource.naturalWidth,
  };
}

function uploadSelectionMask(source, singleBuilding = false) {
  selectionIsSingleBuilding = singleBuilding;
  selectionMaskSize = [source.naturalWidth || source.width, source.naturalHeight || source.height];
  const texture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  if (selectionTexture) gl.deleteTexture(selectionTexture);
  selectionTexture = texture;
  gl.activeTexture(gl.TEXTURE0);
  render();
}

function uploadVegetationMask(source) {
  const texture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  if (vegetationTexture) gl.deleteTexture(vegetationTexture);
  vegetationTexture = texture;
  gl.activeTexture(gl.TEXTURE0);
  render();
}

function clearVegetationMask() {
  if (vegetationTexture) gl.deleteTexture(vegetationTexture);
  vegetationTexture = null;
  render();
}

function clearSelectionMask() {
  if (selectionTexture) gl.deleteTexture(selectionTexture);
  selectionTexture = null;
  if (vegetationTexture) gl.deleteTexture(vegetationTexture);
  vegetationTexture = null;
  render();
}

function showBuildingStatus(message) {
  elements.buildingStatus.textContent = message || "";
  elements.buildingStatus.hidden = !message;
}

function setActivity(element, running) {
  element.classList.toggle("is-running", running);
  element.setAttribute("aria-busy", String(running));
}

function analysisButtonLabel() {
  return "Analyze buildings";
}

function refineNearbyPosesButtonLabel() {
  return "Build sparse depth (SfM)";
}

function setAnalysisJobRunning(running) {
  analysisJobRunning = running;
  setBuildingAnalysisRunning(running);
  const hasCurrentImage = Boolean(images[currentIndex]);
  elements.analyzeBuildingsButton.disabled = running || !hasCurrentImage;
  elements.refineNearbyPosesButton.disabled = running || images.length < 2;
  updateReportButton();
  if (!running) {
    elements.analyzeBuildingsButton.textContent = analysisButtonLabel();
    elements.refineNearbyPosesButton.textContent = refineNearbyPosesButtonLabel();
  }
}

async function loadAnalysisBackend() {
  const response = await fetch("/api/analysis-backend");
  if (!response.ok) throw new Error("Building analysis configuration is unavailable.");
  const backend = await response.json();
  if (backend?.kind !== "local" && backend?.kind !== "remote") throw new Error("Building analysis configuration is invalid.");
  if (typeof backend.label === "string" && backend.label) analysisBackend = backend;
  elements.analyzeBuildingsButton.textContent = analysisButtonLabel();
  elements.analyzeBuildingsButton.title = `Analyze this panorama first, then combine up to six nearby views in the background on the ${analysisBackend.label}.`;
  elements.refineNearbyPosesButton.title = "Optionally triangulate sparse facade depth with multi-view SfM. Visible positions remain on Mapillary GPS.";
}

function setBuildingAnalysisRunning(running) {
  setActivity(elements.buildingStatus, running);
}

function updateMetadata(image) {
  const heading = image.computed_compass_angle_deg;
  elements.capturedAt.textContent = formatDate(image.captured_at_ms);
  elements.coordinates.textContent = coordinateLabel(image);
  elements.heading.textContent = Number.isFinite(heading) ? `${heading.toFixed(1)}°` : "No heading";
  elements.creator.textContent = image.creator?.username || image.creator?.id || "Unknown";
  elements.sourceLink.href = image.source_page || "#";
  elements.sourceLink.hidden = !image.source_page;
  elements.metadata.hidden = false;
}

function mapCacheKey(coordinates) {
  return `${coordinates[1].toFixed(3)},${coordinates[0].toFixed(3)}`;
}

function initializeMap() {
  if (locationMap) return;
  locationMap = window.L.map(elements.mapCanvas, { scrollWheelZoom: true, zoomControl: true });
  window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 19,
  }).addTo(locationMap);
  buildingLayer = window.L.geoJSON(null, {
    onEachFeature: (feature, layer) => layer.on("click", () => selectBuilding(feature.properties.osm_id, true, true)),
    style: buildingStyle,
  }).addTo(locationMap);
  viewLayer = window.L.layerGroup().addTo(locationMap);
  captureLayer = window.L.layerGroup().addTo(locationMap);
  new ResizeObserver(() => locationMap.invalidateSize()).observe(elements.mapCanvas);
}

function initializeMapillaryAoiMap() {
  if (mapillaryAoiMap) return;
  mapillaryAoiMap = window.L.map(elements.mapillaryAoiMap, { scrollWheelZoom: true, zoomControl: true }).setView([20, 0], 2);
  window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 19,
  }).addTo(mapillaryAoiMap);
  mapillaryResultsLayer = window.L.layerGroup().addTo(mapillaryAoiMap);
  const setDrawing = (drawing) => {
    mapillaryAoiDrawing = drawing;
    elements.mapillaryDrawAoiButton.setAttribute("aria-pressed", String(drawing));
    elements.mapillaryDrawAoiButton.textContent = drawing ? "Cancel drawing" : "Draw AOI";
    elements.mapillaryAoiMap.classList.toggle("drawing-aoi", drawing);
    if (!drawing) {
      aoiStart = null;
      mapillaryAoiMap.dragging.enable();
    } else {
      elements.mapillarySearchStatus.textContent = "Drag on the map to draw a small search area.";
    }
  };
  const clearAoi = () => {
    aoiStart = null;
    if (aoiRectangle) aoiRectangle.remove();
    aoiRectangle = null;
    mapillaryResultsLayer.clearLayers();
    mapillarySearchBbox = null;
    mapillarySearchResults = [];
    mapillaryDownloadCandidates = [];
    selectedMapillaryIds.clear();
    mapillaryResultControls.clear();
    elements.mapillaryResultList.replaceChildren();
    elements.mapillaryResults.hidden = true;
    elements.mapillaryDownloadButton.disabled = true;
    elements.mapillaryDownloadAllButton.disabled = true;
    elements.mapillarySelectAllButton.disabled = true;
    elements.mapillarySearchButton.disabled = true;
    elements.mapillaryClearAoiButton.disabled = true;
    elements.mapillarySearchStatus.textContent = "Pan the map, then draw a small area to search.";
  };
  const startAoi = (event) => {
    if (!mapillaryAoiDrawing) return;
    aoiStart = event.latlng;
    if (aoiRectangle) aoiRectangle.remove();
    mapillaryResultsLayer.clearLayers();
    mapillarySearchBbox = null;
    mapillarySearchResults = [];
    mapillaryDownloadCandidates = [];
    selectedMapillaryIds.clear();
    mapillaryResultControls.clear();
    elements.mapillaryResultList.replaceChildren();
    elements.mapillaryResults.hidden = true;
    elements.mapillaryDownloadButton.disabled = true;
    elements.mapillaryDownloadAllButton.disabled = true;
    elements.mapillarySelectAllButton.disabled = true;
    aoiRectangle = window.L.rectangle([aoiStart, aoiStart], { color: "#ecba74", fillColor: "#ecba74", fillOpacity: 0.15, weight: 2 }).addTo(mapillaryAoiMap);
    mapillaryAoiMap.dragging.disable();
    elements.mapillarySearchButton.disabled = true;
    elements.mapillarySearchStatus.textContent = "Draw the AOI and release to search.";
  };
  const updateAoi = (event) => {
    if (mapillaryAoiDrawing && aoiStart && aoiRectangle) aoiRectangle.setBounds(window.L.latLngBounds(aoiStart, event.latlng));
  };
  const finishAoi = (event) => {
    if (!mapillaryAoiDrawing || !aoiStart || !aoiRectangle) return;
    aoiRectangle.setBounds(window.L.latLngBounds(aoiStart, event.latlng));
    aoiStart = null;
    mapillaryAoiMap.dragging.enable();
    const bounds = aoiRectangle.getBounds();
    const hasArea = bounds.getNorth() !== bounds.getSouth() && bounds.getEast() !== bounds.getWest();
    elements.mapillarySearchButton.disabled = !hasArea;
    elements.mapillaryClearAoiButton.disabled = !hasArea;
    elements.mapillarySearchStatus.textContent = hasArea ? "AOI ready to query Mapillary." : "The AOI must have an area.";
    setDrawing(false);
  };
  mapillaryAoiMap.on("mousedown touchstart", startAoi);
  mapillaryAoiMap.on("mousemove touchmove", updateAoi);
  mapillaryAoiMap.on("mouseup touchend", finishAoi);
  elements.mapillaryDrawAoiButton.addEventListener("click", () => setDrawing(!mapillaryAoiDrawing));
  elements.mapillaryClearAoiButton.addEventListener("click", () => {
    setDrawing(false);
    clearAoi();
  });
}

function openMapillaryDialog() {
  if (!window.L) {
    showError("The map library could not load. Check your internet connection.");
    return;
  }
  elements.mapillaryDialog.showModal();
  const hasMap = Boolean(mapillaryAoiMap);
  initializeMapillaryAoiMap();
  elements.mapillaryDatasetName.value = suggestedMapillaryDatasetName();
  const coordinates = images[currentIndex] && coordinatesOf(images[currentIndex]);
  if (!hasMap && coordinates) mapillaryAoiMap.setView([coordinates[1], coordinates[0]], 16);
  window.setTimeout(() => mapillaryAoiMap.invalidateSize(), 0);
}

function renderMapillarySearchResults(panoramas) {
  mapillaryResultsLayer.clearLayers();
  mapillaryResultControls.clear();
  selectedMapillaryIds.clear();
  mapillaryDownloadCandidates = panoramas;
  mapillarySearchResults = panoramas.slice(0, 50);
  elements.mapillaryResultList.replaceChildren();
  elements.mapillaryResults.hidden = mapillarySearchResults.length === 0;
  elements.mapillarySelectAllButton.disabled = mapillarySearchResults.length === 0;
  elements.mapillaryDownloadAllButton.disabled = mapillaryDownloadCandidates.length === 0;
  mapillarySearchResults.forEach((panorama) => {
    const [longitude, latitude] = panorama.coordinates;
    const marker = window.L.circleMarker([latitude, longitude], {
      color: "#5d2131", fillColor: "#db6070", fillOpacity: 0.45, radius: 4, weight: 1.5,
    }).addTo(mapillaryResultsLayer);
    const item = document.createElement("li");
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.addEventListener("change", () => setMapillarySelection(panorama.id, checkbox.checked));
    const text = document.createElement("span");
    text.textContent = `${formatDate(panorama.captured_at)} · ${latitude.toFixed(5)}, ${longitude.toFixed(5)}`;
    label.append(checkbox, text);
    item.append(label);
    elements.mapillaryResultList.append(item);
    mapillaryResultControls.set(panorama.id, { checkbox, marker });
    marker.bindTooltip(`Panorama ${panorama.id}`, { direction: "top" }).on("click", () => setMapillarySelection(panorama.id, !selectedMapillaryIds.has(panorama.id)));
  });
  updateMapillaryDownloadControls();
}

function setMapillarySelection(imageId, selected) {
  const control = mapillaryResultControls.get(imageId);
  if (!control) return;
  if (selected) selectedMapillaryIds.add(imageId);
  else selectedMapillaryIds.delete(imageId);
  control.checkbox.checked = selected;
  control.marker.setStyle(selected
    ? { color: "#fff", fillColor: "#ecba74", fillOpacity: 1, radius: 6 }
    : { color: "#5d2131", fillColor: "#db6070", fillOpacity: 0.45, radius: 4 });
  updateMapillaryDownloadControls();
}

function updateMapillaryDownloadControls() {
  const selected = selectedMapillaryIds.size;
  elements.mapillaryDownloadButton.disabled = selected === 0;
  elements.mapillaryDownloadButton.textContent = selected ? `Download selected (${selected})` : "Download selected";
  elements.mapillarySelectAllButton.textContent = selected === mapillarySearchResults.length && selected > 0
    ? "Deselect all"
    : "Select displayed";
}

function suggestedMapillaryDatasetName() {
  return `mapillary-${new Date().toISOString().replace(/[-:]/g, "").slice(0, 13).toLowerCase()}`;
}

async function searchMapillaryAoi() {
  if (!aoiRectangle) return;
  const bounds = aoiRectangle.getBounds();
  const bbox = [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()];
  elements.mapillarySearchButton.disabled = true;
  elements.mapillarySearchStatus.textContent = "Consultando Mapillary...";
  setActivity(elements.mapillarySearchStatus, true);
  try {
    const response = await fetch("/api/mapillary/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bbox }),
    });
    const body = await response.text();
    let result;
    try {
      result = JSON.parse(body);
    } catch {
      if (response.status === 501) {
        throw new Error("The running server does not include Mapillary search. Stop it and run uv run city-analyser-viewer again.");
      }
    throw new Error(`The server returned an invalid response (${response.status}).`);
    }
    if (!response.ok) throw new Error(result.error || `Mapillary did not respond (${response.status}).`);
    mapillarySearchBbox = bbox;
    renderMapillarySearchResults(result.panoramas);
    const displayed = mapillarySearchResults.length;
    elements.mapillarySearchStatus.textContent = result.panoramas_found
      ? `${result.panoramas_found.toLocaleString()} 360 panoramas found. Showing ${displayed} for selection; download all to fetch every result${result.truncated ? "; Mapillary returned the city-scale limit, so further coverage may exist." : ""}.`
      : "No 360 panoramas were found in this AOI.";
  } catch (error) {
    elements.mapillarySearchStatus.textContent = `Search could not complete: ${error.message}`;
  } finally {
    elements.mapillarySearchButton.disabled = false;
    setActivity(elements.mapillarySearchStatus, false);
  }
}

async function downloadMapillaryPanoramas(imageIds, allResults = false) {
  if (!mapillarySearchBbox || (!allResults && !imageIds.length)) return;
  const datasetId = elements.mapillaryDatasetName.value.trim();
  if (!datasetId) {
    elements.mapillarySearchStatus.textContent = "Enter a name for the new dataset.";
    return;
  }
  if (allResults && !window.confirm("Download every Mapillary panorama in this AOI, not only the displayed or selected items? This can require substantial disk space and take a long time.")) return;
  elements.mapillaryDownloadButton.disabled = true;
  elements.mapillaryDownloadAllButton.disabled = true;
  elements.mapillarySearchStatus.textContent = allResults ? "Downloading every Mapillary panorama in this AOI..." : `Downloading ${imageIds.length.toLocaleString()} panoramas from Mapillary...`;
  setActivity(elements.mapillarySearchStatus, true);
  try {
    const response = await fetch("/api/mapillary/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        bbox: mapillarySearchBbox,
        dataset_id: datasetId,
        image_ids: imageIds,
        download_all: allResults,
        image_size: elements.mapillaryImageSize.value,
      }),
    });
    const body = await response.text();
    let result;
    try {
      result = JSON.parse(body);
    } catch {
      if (response.status === 404 || response.status === 501) {
        throw new Error("The running server does not include Mapillary downloads. Stop it and run uv run city-analyser-viewer again.");
      }
        throw new Error(`The server returned an invalid response (${response.status}).`);
    }
    if (!response.ok) throw new Error(result.error || `Download failed (${response.status}).`);
    if (result.status !== "queued" || !result.job?.id) throw new Error("The server did not create a download job.");
    await pollMapillaryDownload(result.job.id);
  } catch (error) {
    elements.mapillarySearchStatus.textContent = `Download failed: ${error.message}`;
    updateMapillaryDownloadControls();
  } finally {
    elements.mapillaryDownloadAllButton.disabled = mapillaryDownloadCandidates.length === 0;
    setActivity(elements.mapillarySearchStatus, false);
  }
}

async function pollMapillaryDownload(jobId) {
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
  if (!response.ok) throw new Error("Mapillary download job was lost.");
  const job = await response.json();
  elements.mapillarySearchStatus.textContent = job.message;
  if (job.status === "complete") {
    const catalogResponse = await fetch("/api/datasets");
    if (!catalogResponse.ok) throw new Error("The dataset downloaded, but the local library could not be updated.");
    const catalog = await catalogResponse.json();
    datasets = catalog.datasets || [];
    elements.datasetSelect.replaceChildren(...datasets.map((dataset) => {
      const option = document.createElement("option");
      option.value = dataset.id;
      option.textContent = `${dataset.id} (${dataset.images})`;
      return option;
    }));
    elements.datasetSelect.disabled = false;
    elements.mapillarySearchStatus.textContent = `${job.message} Select the new dataset to open it.`;
    return;
  }
  if (job.status === "failed") throw new Error(job.message);
  await new Promise((resolve) => window.setTimeout(resolve, 1000));
  return pollMapillaryDownload(jobId);
}

function downloadSelectedMapillaryPanoramas() {
  return downloadMapillaryPanoramas([...selectedMapillaryIds]);
}

function downloadAllMapillaryPanoramas() {
  return downloadMapillaryPanoramas([], true);
}

function recordedHeading(image) {
  return Number.isFinite(image?.computed_compass_angle_deg) ? image.computed_compass_angle_deg : null;
}

function updateSceneNavigation() {
  const currentImage = images[currentIndex];
  const currentHeading = recordedHeading(currentImage);
  if (currentHeading === null) {
    sceneNavigationControls.forEach((button) => button.remove());
    sceneNavigationControls.clear();
    sceneNavigationTargets.clear();
    elements.sceneNavigation.hidden = true;
    return;
  }
  const visualHeading = normalizeDegrees(currentHeading + yaw * 180 / Math.PI);
  const panoramas = images.map((image) => ({ coordinates: coordinatesOf(image), sequenceId: image.sequence_id || null }));
  const aspect = elements.canvas.width / elements.canvas.height;
  const projectedTargets = nearbyPanoramas(panoramas, currentIndex).filter((target) => (
    recordedHeading(images[target.index]) !== null
    && navigationLineIsClear(coordinatesOf(currentImage), coordinatesOf(images[target.index]))
  )).map((target) => ({
    ...target,
    projection: projectGroundPanoramaToView(target.bearing, target.distance, visualHeading, pitch, fov, aspect, ASSUMED_CAMERA_HEIGHT_M),
  })).filter((target) => target.projection);
  const visibleTargets = nearestPanoramasByBearing(projectedTargets);
  sceneNavigationTargets = new Map(visibleTargets.map((target) => [target.index, target]));
  sceneNavigationControls.forEach((button, index) => {
    if (sceneNavigationTargets.has(index)) return;
    button.remove();
    sceneNavigationControls.delete(index);
  });
  visibleTargets.forEach((target) => {
    let button = sceneNavigationControls.get(target.index);
    if (!button) {
      button = document.createElement("button");
      button.type = "button";
      button.className = "scene-navigation-point";
      button.dataset.index = String(target.index);
      const number = document.createElement("span");
      number.className = "scene-navigation-number";
      number.textContent = String(target.index + 1);
      button.append(number);
      elements.sceneNavigation.append(button);
      sceneNavigationControls.set(target.index, button);
    }
    button.style.setProperty("--scene-point-left", `${target.projection.left}%`);
    button.style.setProperty("--scene-point-top", `${target.projection.top}%`);
    button.style.setProperty("--scene-point-size", `${Math.max(20, Math.min(38, 18 + 80 / (target.distance + 2)))}px`);
    button.dataset.distance = `${Math.round(target.distance)} m`;
    button.setAttribute("aria-label", `Move to nearby panorama ${target.index + 1}, ${Math.round(target.distance)} meters away`);
    button.title = `Panorama ${target.index + 1} · ${Math.round(target.distance)} m`;
  });
  elements.sceneNavigation.hidden = visibleTargets.length === 0;
}

function moveToNearbyPanorama(target) {
  if (!target) return;
  const nextImage = images[target.index];
  const nextHeading = recordedHeading(nextImage);
  if (nextHeading === null) return;
  yaw = signedDegrees(target.bearing - nextHeading) * Math.PI / 180;
  pitch = 0;
  selectImage(target.index);
}

function mapWindowState() {
  if (!datasetId || !images.length) return null;
  const visibleBuildingIds = currentAnalysis
    ? Object.values(currentAnalysis.building_ids || {}).map((building) => building.osm_id)
    : null;
  return {
    datasetId,
    index: currentIndex,
    selectedBuildingId,
    type: "state",
    visibleBuildingIds,
    visibleBuildingFootprints: currentAnalysis ? Object.values(currentAnalysis.building_ids || {}) : null,
    yaw,
  };
}

function broadcastMapWindowState() {
  const state = mapWindowState();
  if (state) mapWindowChannel?.postMessage(state);
}

function setMapPanelVisible(visible) {
  elements.mapPanel.hidden = !visible;
  elements.mapToggleButton.setAttribute("aria-expanded", String(visible));
  elements.mapToggleButton.setAttribute("aria-label", visible ? "Hide map" : "Show map");
  elements.mapToggleButton.title = visible ? "Hide map" : "Show map";
  if (visible) window.setTimeout(() => locationMap?.invalidateSize(), 0);
}

function openMapWindow() {
  if (!datasetId) return;
  const url = new URL("map-window.html", window.location.href);
  url.searchParams.set("dataset", datasetId);
  const popup = window.open(url.toString(), "panorama-map", "popup,width=900,height=700,resizable=yes,scrollbars=no");
  if (!popup) {
    elements.status.textContent = "Firefox blocked the map window. Allow pop-ups for this site.";
    return;
  }
  setMapPanelVisible(false);
  popup.focus();
  window.setTimeout(broadcastMapWindowState, 100);
}

function viewObstructionDistance(latitude, longitude, heading) {
  if (!buildingLayer) return null;
  const endpoint = destinationPoint(latitude, longitude, heading, MAP_BEAM_DISTANCE_M);
  let nearest = null;
  buildingLayer.eachLayer((layer) => {
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

function navigationLineIsClear(start, end) {
  if (!buildingLayer || !start || !end) return true;
  const rayStart = [start[1], start[0]];
  const rayEnd = [end[1], end[0]];
  let blocked = false;
  buildingLayer.eachLayer((layer) => {
    outerRingsOf(layer.feature?.geometry).forEach((ring) => {
      const points = ring.map(([longitude, latitude]) => [latitude, longitude]);
      points.forEach((point, index) => {
        const fraction = segmentIntersectionFraction(rayStart, rayEnd, point, points[(index + 1) % points.length]);
        if (fraction !== null && fraction < 0.98) blocked = true;
      });
    });
  });
  return !blocked;
}

function updateViewIndicator() {
  if (!locationMap || !viewLayer || !currentMapImage) return;
  const coordinates = coordinatesOf(currentMapImage);
  if (!coordinates) return;
  const [longitude, latitude] = coordinates;
  const capturedHeading = Number.isFinite(currentMapImage.computed_compass_angle_deg)
    ? currentMapImage.computed_compass_angle_deg
    : null;
  if (capturedHeading === null) {
    viewLayer.clearLayers();
    elements.mapDirection.textContent = "Current view orientation unavailable";
    elements.mapDirection.hidden = false;
    return;
  }
  const heading = normalizeDegrees(capturedHeading + yaw * 180 / Math.PI);
  const obstructionDistance = viewObstructionDistance(latitude, longitude, heading);
  const beamDistance = obstructionDistance ?? MAP_BEAM_DISTANCE_M;
  const halfFov = MAP_BEAM_FOV / 2;
  const arc = [];
  for (let step = 0; step <= 12; step += 1) {
    arc.push(destinationPoint(latitude, longitude, heading - halfFov + MAP_BEAM_FOV * step / 12, beamDistance));
  }
  viewLayer.clearLayers();
  window.L.polygon([[latitude, longitude], ...arc], {
    color: obstructionDistance === null ? "#2878c9" : "#c95b2b", fillColor: obstructionDistance === null ? "#3c9bdf" : "#e57b45", fillOpacity: 0.2, interactive: false, weight: 2,
  }).addTo(viewLayer);
  window.L.polyline([[latitude, longitude], destinationPoint(latitude, longitude, heading, beamDistance)], {
    color: obstructionDistance === null ? "#195c9e" : "#a84222", interactive: false, weight: 3,
  }).addTo(viewLayer);
  elements.mapDirection.hidden = false;
  elements.mapDirection.textContent = obstructionDistance === null
    ? `Current view: ${heading.toFixed(0)}° · guide beam: ${MAP_BEAM_FOV}°`
    : `Current view: ${heading.toFixed(0)}° · blocked by a footprint at ${obstructionDistance.toFixed(0)} m`;
}

function renderCaptureLocations() {
  if (!captureLayer) return;
  captureLayer.clearLayers();
  images.forEach((image, index) => {
    if (index === currentIndex) return;
    const coordinates = coordinatesOf(image);
    if (!coordinates) return;
    const [longitude, latitude] = coordinates;
    window.L.circleMarker([latitude, longitude], {
      color: "#7d231c", fillColor: "#d95045", fillOpacity: 0.35, radius: 5, weight: 2,
    }).addTo(captureLayer).bindTooltip(`Panorama ${index + 1}`, { direction: "top" }).on("click", () => selectImage(index));
  });
}

function scheduleViewIndicator() {
  if (viewUpdateFrame !== null) return;
  viewUpdateFrame = window.requestAnimationFrame(() => {
    viewUpdateFrame = null;
    updateViewIndicator();
    updateSceneNavigation();
    broadcastMapWindowState();
    syncViewerUrl();
  });
}

function toggleMapPanel() {
  const floating = elements.mapPanel.classList.toggle("floating");
  if (!floating) {
    elements.mapPanel.style.removeProperty("left");
    elements.mapPanel.style.removeProperty("top");
    elements.mapPanel.style.removeProperty("width");
    elements.mapPanel.style.removeProperty("height");
  }
  elements.mapDetachButton.textContent = floating ? "Dock" : "Detach";
  window.setTimeout(() => locationMap?.invalidateSize(), 0);
}

function bindMapPanelDrag() {
  let drag = null;
  elements.mapDragHandle.addEventListener("pointerdown", (event) => {
    if (!elements.mapPanel.classList.contains("floating") || event.target.closest("button")) return;
    const rectangle = elements.mapPanel.getBoundingClientRect();
    drag = { id: event.pointerId, offsetX: event.clientX - rectangle.left, offsetY: event.clientY - rectangle.top };
    elements.mapDragHandle.setPointerCapture(event.pointerId);
  });
  elements.mapDragHandle.addEventListener("pointermove", (event) => {
    if (!drag || event.pointerId !== drag.id) return;
    const left = Math.max(0, Math.min(window.innerWidth - elements.mapPanel.offsetWidth, event.clientX - drag.offsetX));
    const top = Math.max(0, Math.min(window.innerHeight - elements.mapPanel.offsetHeight, event.clientY - drag.offsetY));
    elements.mapPanel.style.left = `${left}px`;
    elements.mapPanel.style.top = `${top}px`;
  });
  const stopDrag = (event) => {
    if (!drag || event.pointerId !== drag.id) return;
    drag = null;
    locationMap?.invalidateSize();
  };
  elements.mapDragHandle.addEventListener("pointerup", stopDrag);
  elements.mapDragHandle.addEventListener("pointercancel", stopDrag);
}

async function loadBuildingFootprints(coordinates) {
  const key = mapCacheKey(coordinates);
  if (buildingCache.has(key)) return buildingCache.get(key);
  const [longitude, latitude] = coordinates;
  const response = await fetch(`/api/buildings?lat=${encodeURIComponent(latitude)}&lon=${encodeURIComponent(longitude)}`);
  if (!response.ok) throw new Error(`Building service did not respond (${response.status}).`);
  const payload = await response.json();
  const features = Array.isArray(payload.features) ? payload.features : [];
  buildingCache.set(key, features);
  return features;
}

function buildingStyle(feature) {
  const selected = feature.properties?.osm_id === selectedBuildingId;
  const directlyVisible = !currentAnalysis || analysisHasBuilding(feature.properties?.osm_id);
  const style = selected
    ? { color: "#fff", fillColor: "#ff8a2a", fillOpacity: 0.58, weight: 3 }
    : !directlyVisible
      ? { color: "#59685f", fillColor: "#59685f", fillOpacity: 0.08, weight: 1 }
      : { color: "#d47a36", fillColor: "#ecba74", fillOpacity: 0.28, weight: 1 };
  return style;
}

function refreshBuildingStyles() {
  buildingLayer?.eachLayer((layer) => layer.setStyle(buildingStyle(layer.feature)));
}

function applyAnalysisFootprints() {
  if (!buildingLayer) return;
  const features = mergeBuildingFootprints(currentBuildingFeatures, currentAnalysis?.building_ids);
  buildingLayer.clearLayers();
  buildingLayer.addData({ type: "FeatureCollection", features });
  refreshBuildingStyles();
}

function analysisHasBuilding(osmId) {
  return Object.values(currentAnalysis?.building_ids || {}).some((building) => building.osm_id === osmId);
}

function updateReportButton() {
  elements.generateReportButton.disabled = analysisJobRunning || !selectedBuildingId || !analysisHasBuilding(selectedBuildingId);
}

function updateMaskToggle() {
  elements.maskToggleButton.disabled = !currentAnalysis;
  elements.maskToggleButton.textContent = buildingOverlayVisible ? "Hide overlay" : "Show overlay";
  elements.maskToggleButton.setAttribute("aria-pressed", String(buildingOverlayVisible));
}

function toggleBuildingOverlay() {
  buildingOverlayVisible = !buildingOverlayVisible;
  updateMaskToggle();
  render();
}

function loadSelectionMask(osmId) {
  const requestId = ++maskLoadRequest;
  if (!currentAnalysis || !analysisHasBuilding(osmId)) {
    clearSelectionMask();
    return;
  }
  const image = new Image();
  image.onload = () => { if (requestId === maskLoadRequest) uploadSelectionMask(image, true); };
  image.onerror = () => { if (requestId === maskLoadRequest) clearSelectionMask(); };
  image.src = `/api/datasets/${encodeURIComponent(datasetId)}/analysis/${encodeURIComponent(images[currentIndex].id)}/mask?building_id=${encodeURIComponent(osmId)}&t=${Date.now()}`;
  const vegetation = new Image();
  vegetation.onload = () => { if (requestId === maskLoadRequest) uploadVegetationMask(vegetation); };
  vegetation.onerror = () => {};
  vegetation.src = `/api/datasets/${encodeURIComponent(datasetId)}/analysis/${encodeURIComponent(images[currentIndex].id)}/mask?building_id=${encodeURIComponent(osmId)}&layer=inferred&t=${Date.now()}`;
}

function loadBuildingOverviewMask() {
  const requestId = ++maskLoadRequest;
  if (!currentAnalysis) {
    clearSelectionMask();
    return;
  }
  clearVegetationMask();
  const image = new Image();
  image.onload = () => { if (requestId === maskLoadRequest) uploadSelectionMask(image); };
  image.onerror = () => { if (requestId === maskLoadRequest) clearSelectionMask(); };
  image.src = `/api/datasets/${encodeURIComponent(datasetId)}/analysis/${encodeURIComponent(images[currentIndex].id)}/mask?t=${Date.now()}`;
}

function focusBuilding(osmId) {
  const image = images[currentIndex];
  if (!image || !buildingLayer) return;
  let feature = null;
  buildingLayer.eachLayer((layer) => {
    if (layer.feature?.properties?.osm_id === osmId) feature = layer.feature;
  });
  const vertices = outerRingsOf(feature?.geometry).flatMap((ring) => ring.slice(0, -1));
  const analysisPose = currentAnalysis?.image_id === image.id ? currentAnalysis.refined_pose : null;
  const camera = Number.isFinite(analysisPose?.longitude) && Number.isFinite(analysisPose?.latitude)
    ? [analysisPose.longitude, analysisPose.latitude]
    : coordinatesOf(image);
  if (!vertices.length || !camera) return;
  const [longitude, latitude] = vertices.reduce((total, point) => [total[0] + point[0], total[1] + point[1]], [0, 0]);
  const targetLongitude = longitude / vertices.length;
  const targetLatitude = latitude / vertices.length;
  const [cameraLongitude, cameraLatitude] = camera;
  const longitudeDelta = (targetLongitude - cameraLongitude) * Math.PI / 180;
  const latitudeRadians = cameraLatitude * Math.PI / 180;
  const targetLatitudeRadians = targetLatitude * Math.PI / 180;
  const bearing = Math.atan2(
    Math.sin(longitudeDelta) * Math.cos(targetLatitudeRadians),
    Math.cos(latitudeRadians) * Math.sin(targetLatitudeRadians) - Math.sin(latitudeRadians) * Math.cos(targetLatitudeRadians) * Math.cos(longitudeDelta),
  ) * 180 / Math.PI;
  const capturedHeading = Number.isFinite(analysisPose?.heading_degrees)
    ? analysisPose.heading_degrees
    : recordedHeading(image);
  if (capturedHeading === null) return;
  yaw = ((normalizeDegrees(bearing - capturedHeading + 180) - 180) * Math.PI / 180);
  render();
  scheduleViewIndicator();
}

function selectBuilding(osmId, broadcast = true, focus = false) {
  if (osmId && currentAnalysis && !analysisHasBuilding(osmId)) {
    selectedBuildingId = null;
    loadBuildingOverviewMask();
    refreshBuildingStyles();
    showBuildingStatus("This footprint has no directly visible facade in the current panorama.");
    updateReportButton();
    updateMaskToggle();
    if (broadcast) broadcastMapWindowState();
    return;
  }
  selectedBuildingId = osmId || null;
  if (focus && selectedBuildingId) focusBuilding(selectedBuildingId);
  refreshBuildingStyles();
  if (selectedBuildingId) {
    loadSelectionMask(selectedBuildingId);
    const details = Object.values(currentAnalysis?.building_ids || {}).find((building) => building.osm_id === selectedBuildingId);
    const depth = details?.sfm_depth_samples
      ? ` · SfM depth: ${details.sfm_depth_samples} point${details.sfm_depth_samples === 1 ? "" : "s"}${details.sfm_depth_conflicts ? `, ${details.sfm_depth_conflicts} rejected` : ""}`
      : "";
    showBuildingStatus(details
      ? `Selected ${selectedBuildingId} · ${details.facades || 1} visible facade${details.facades === 1 ? "" : "s"} · observed ${Math.round(details.confidence * 100)}%${depth} · orange observed, blue tree-occluded`
      : `Selected ${selectedBuildingId} · no image mask for this panorama`);
  } else {
    loadBuildingOverviewMask();
    showBuildingStatus(currentAnalysis ? `${Object.keys(currentAnalysis.building_ids || {}).length} matched buildings. Click a mask or footprint to select one.` : "");
  }
  updateReportButton();
  updateMaskToggle();
  syncViewerUrl();
  if (broadcast) broadcastMapWindowState();
}

async function updateMap(image) {
  currentMapImage = image;
  currentBuildingFeatures = [];
  buildingLayer?.clearLayers();
  const coordinates = coordinatesOf(image);
  if (!coordinates) {
    mapLoad += 1;
    elements.mapCanvas.hidden = true;
    elements.mapDirection.hidden = true;
    elements.mapMessage.textContent = "This panorama has no coordinates to display on the map.";
    setActivity(elements.mapMessage, false);
    return;
  }
  if (!window.L) {
    mapLoad += 1;
    elements.mapCanvas.hidden = true;
    elements.mapMessage.textContent = "The map library could not load. Check your internet connection.";
    setActivity(elements.mapMessage, false);
    return;
  }
  initializeMap();
  elements.mapCanvas.hidden = false;
  renderCaptureLocations();
  centerMapOnCurrentCamera();
  renderCameraMarker();
  updateViewIndicator();
  window.setTimeout(() => {
    if (currentMapImage === image) centerMapOnCurrentCamera();
  }, 0);
  const loadId = ++mapLoad;
  elements.mapMessage.textContent = "Loading nearby building footprints...";
  setActivity(elements.mapMessage, true);
  try {
    const features = await loadBuildingFootprints(coordinates);
    if (loadId !== mapLoad) return;
    currentBuildingFeatures = features;
    applyAnalysisFootprints();
    updateViewIndicator();
    const footprintCount = buildingLayer.getLayers().length;
    elements.mapMessage.textContent = footprintCount
      ? `${footprintCount} nearby building footprints.`
      : "No nearby building footprints in OpenStreetMap.";
  } catch (error) {
    if (loadId !== mapLoad) return;
    currentBuildingFeatures = [];
    buildingLayer.clearLayers();
    elements.mapMessage.textContent = `Building footprints could not load: ${error.message}`;
  } finally {
    if (loadId === mapLoad) setActivity(elements.mapMessage, false);
  }
}

function renderCameraMarker() {
  if (!locationMap || !currentMapImage) return;
  const coordinates = coordinatesOf(currentMapImage);
  if (!coordinates) return;
  const [longitude, latitude] = coordinates;
  if (cameraMarker) cameraMarker.remove();
  cameraMarker = window.L.circleMarker([latitude, longitude], {
    color: "#fff", fillColor: "#d95f3d", fillOpacity: 0.7, radius: 7, weight: 2,
  }).addTo(locationMap).bindTooltip("Mapillary panorama position", { direction: "top" });
}

function applyRefinedMapPose() {
  applyAnalysisFootprints();
  renderCaptureLocations();
  renderCameraMarker();
  updateViewIndicator();
  updateSceneNavigation();
  centerMapOnCurrentCamera();
  broadcastMapWindowState();
}

function centerMapOnCurrentCamera() {
  if (!locationMap || !currentMapImage) return;
  const coordinates = coordinatesOf(currentMapImage);
  if (!coordinates) return;
  const [longitude, latitude] = coordinates;
  locationMap.invalidateSize({ pan: false });
  locationMap.setView([latitude, longitude], 18, { animate: false });
}

async function loadBuildingAnalysis(image) {
  const requestId = ++analysisRequest;
  const requestedDataset = datasetId;
  const requestedImageId = image.id;
  maskLoadRequest += 1;
  currentAnalysis = null;
  setBuildingAnalysisRunning(analysisJobRunning);
  clearSelectionMask();
  updateMaskToggle();
  elements.analyzeBuildingsButton.disabled = analysisJobRunning;
  elements.analyzeBuildingsButton.textContent = analysisButtonLabel();
  try {
    const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/analysis/${encodeURIComponent(image.id)}`);
    if (requestId !== analysisRequest || requestedDataset !== datasetId || images[currentIndex]?.id !== requestedImageId) return;
    if (response.status === 404) {
      if (selectedBuildingId) selectBuilding(selectedBuildingId, false);
      updateReportButton();
      return;
    }
    if (!response.ok) throw new Error(`Analysis unavailable (${response.status}).`);
    const analysis = await response.json();
    if (requestId !== analysisRequest || requestedDataset !== datasetId || images[currentIndex]?.id !== requestedImageId) return;
    currentAnalysis = analysis;
    updateMetadata(image);
    elements.analyzeBuildingsButton.textContent = analysisButtonLabel();
    if (selectedBuildingId) selectBuilding(selectedBuildingId, false);
    else selectBuilding(null, false);
    applyRefinedMapPose();
  } catch (error) {
    if (requestId === analysisRequest) showBuildingStatus(`Building analysis unavailable: ${error.message}`);
  }
}

function showNearbyAnalysisSummary(analysis) {
  const buildings = Array.isArray(analysis?.building_ids) ? analysis.building_ids : [];
  const views = analysis?.panoramas?.length || 0;
  const confirmed = buildings.filter((building) => building.views >= 2).length;
  showBuildingStatus(`Nearby evidence from ${views} panoramas: ${buildings.length} matched buildings${confirmed ? ` · ${confirmed} confirmed in multiple views` : ""}.`);
}

function showSfmRefinementSummary(refinement) {
  const accepted = Object.values(refinement?.poses || {}).filter((pose) => pose.accepted).length;
  const samples = Array.isArray(refinement?.depth_samples) ? refinement.depth_samples.length : 0;
  showBuildingStatus(`COLMAP aligned ${accepted} nearby cameras and triangulated ${samples} sparse depth points. Run Analyze buildings to apply its optional depth checks.`);
}

async function loadSfmRefinement(image) {
  const requestId = ++sfmRequest;
  const requestedDataset = datasetId;
  const requestedImageId = image.id;
  currentSfmRefinement = null;
  updateSceneNavigation();
  elements.refineNearbyPosesButton.disabled = analysisJobRunning || images.length < 2;
  elements.refineNearbyPosesButton.textContent = refineNearbyPosesButtonLabel();
  try {
    const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/analysis/${encodeURIComponent(image.id)}/sfm`);
    if (requestId !== sfmRequest || requestedDataset !== datasetId || images[currentIndex]?.id !== requestedImageId || response.status === 404) return;
    if (!response.ok) throw new Error(`Pose refinement unavailable (${response.status}).`);
    const refinement = await response.json();
    if (requestId !== sfmRequest || requestedDataset !== datasetId || images[currentIndex]?.id !== requestedImageId) return;
    currentSfmRefinement = refinement;
    updateMetadata(image);
    applyRefinedMapPose();
  } catch (error) {
    if (requestId === sfmRequest) console.warn("Nearby pose refinement could not load.", error);
  }
}

async function pollNearbyBuildingAnalysis(jobId, requestedDataset, imageId, currentLoaded = false) {
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
  if (!response.ok) throw new Error("Nearby analysis job was lost.");
  const job = await response.json();
  showBuildingStatus(job.message);
  const showingRequestedImage = datasetId === requestedDataset && images[currentIndex]?.id === imageId;
  if (!currentLoaded && job.current_analysis_complete) {
    currentLoaded = true;
    if (showingRequestedImage) await loadBuildingAnalysis(images[currentIndex]);
  }
  if (job.status === "complete") {
    setBuildingAnalysisRunning(false);
    if (showingRequestedImage) {
      if (!currentLoaded) await loadBuildingAnalysis(images[currentIndex]);
      showNearbyAnalysisSummary(job.analysis);
    } else showBuildingStatus("Building analysis completed in the background.");
    setAnalysisJobRunning(false);
    return;
  }
  if (job.status === "failed") {
    setBuildingAnalysisRunning(false);
    setAnalysisJobRunning(false);
    throw new Error(job.message);
  }
  window.setTimeout(() => {
    pollNearbyBuildingAnalysis(jobId, requestedDataset, imageId, currentLoaded).catch((error) => {
      setBuildingAnalysisRunning(false);
      setAnalysisJobRunning(false);
      showBuildingStatus(`Building analysis failed: ${error.message}`);
    });
  }, 1000);
}

async function analyzeBuildings() {
  const image = images[currentIndex];
  if (!image) return;
  const requestedDataset = datasetId;
  setAnalysisJobRunning(true);
  elements.analyzeBuildingsButton.textContent = "Analyzing in background...";
  setBuildingAnalysisRunning(true);
  showBuildingStatus(`Starting ${analysisBackend.label} analysis with this view, then nearby views...`);
  try {
    const response = await fetch(`/api/datasets/${encodeURIComponent(requestedDataset)}/analysis/${encodeURIComponent(image.id)}/nearby`, { method: "POST" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Building analysis could not start (${response.status}).`);
    if (result.status === "complete") {
      setBuildingAnalysisRunning(false);
      if (datasetId === requestedDataset && images[currentIndex]?.id === image.id) {
        await loadBuildingAnalysis(images[currentIndex]);
        showNearbyAnalysisSummary(result.analysis);
      } else showBuildingStatus("Building analysis completed in the background.");
      setAnalysisJobRunning(false);
      return;
    }
    await pollNearbyBuildingAnalysis(result.job.id, requestedDataset, image.id);
  } catch (error) {
    setBuildingAnalysisRunning(false);
    setAnalysisJobRunning(false);
    showBuildingStatus(`Building analysis failed: ${error.message}`);
  }
}

async function pollSfmRefinement(jobId, imageId) {
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
  if (!response.ok) throw new Error("Sparse depth job was lost.");
  const job = await response.json();
  showBuildingStatus(job.message);
  if (job.status === "complete") {
    setBuildingAnalysisRunning(false);
    if (images[currentIndex]?.id === imageId) {
      currentSfmRefinement = job.analysis;
      updateMetadata(images[currentIndex]);
      elements.refineNearbyPosesButton.disabled = false;
      elements.refineNearbyPosesButton.textContent = refineNearbyPosesButtonLabel();
      showSfmRefinementSummary(job.analysis);
      applyRefinedMapPose();
    }
    setAnalysisJobRunning(false);
    return;
  }
  if (job.status === "failed") {
    setBuildingAnalysisRunning(false);
    setAnalysisJobRunning(false);
    throw new Error(job.message);
  }
  window.setTimeout(() => {
    pollSfmRefinement(jobId, imageId).catch((error) => {
      setBuildingAnalysisRunning(false);
      setAnalysisJobRunning(false);
      showBuildingStatus(`Sparse depth reconstruction failed: ${error.message}`);
    });
  }, 1000);
}

async function refineNearbyPoses() {
  const image = images[currentIndex];
  if (!image) return;
  setAnalysisJobRunning(true);
  elements.refineNearbyPosesButton.textContent = "Building sparse depth...";
  setBuildingAnalysisRunning(true);
  showBuildingStatus("Queueing optional sparse depth reconstruction with COLMAP...");
  try {
    const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/analysis/${encodeURIComponent(image.id)}/sfm`, { method: "POST" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Sparse depth reconstruction could not start (${response.status}).`);
    if (result.status === "complete") {
      setBuildingAnalysisRunning(false);
      currentSfmRefinement = result.analysis;
      updateMetadata(image);
      elements.refineNearbyPosesButton.disabled = false;
      elements.refineNearbyPosesButton.textContent = refineNearbyPosesButtonLabel();
      applyRefinedMapPose();
      showSfmRefinementSummary(result.analysis);
      setAnalysisJobRunning(false);
      return;
    }
    await pollSfmRefinement(result.job.id, image.id);
  } catch (error) {
    setBuildingAnalysisRunning(false);
    setAnalysisJobRunning(false);
    showBuildingStatus(`Sparse depth reconstruction failed: ${error.message}`);
  }
}

function rotateYaw(vector, angle) {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return [cosine * vector[0] + sine * vector[2], vector[1], -sine * vector[0] + cosine * vector[2]];
}

function rotatePitch(vector, angle) {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return [vector[0], cosine * vector[1] - sine * vector[2], sine * vector[1] + cosine * vector[2]];
}

function panoramaUvAt(event) {
  const rectangle = elements.canvas.getBoundingClientRect();
  const x = (event.clientX - rectangle.left) / rectangle.width * 2 - 1;
  const y = 1 - (event.clientY - rectangle.top) / rectangle.height * 2;
  const aspect = rectangle.width / rectangle.height;
  const field = Math.tan(fov * Math.PI / 360) / aspect;
  let direction = [x * aspect * field, y * field, 1];
  const magnitude = Math.hypot(...direction);
  direction = direction.map((value) => value / magnitude);
  direction = rotateYaw(rotatePitch(direction, pitch), yaw);
  return {
    u: (0.5 + Math.atan2(direction[0], direction[2]) / (2 * Math.PI) + 1) % 1,
    v: 0.5 - Math.asin(Math.max(-1, Math.min(1, direction[1]))) / Math.PI,
  };
}

async function pickBuildingAt(event) {
  if (!currentAnalysis || !images[currentIndex]) return;
  const requestedDataset = datasetId;
  const requestedImageId = images[currentIndex].id;
  const { u, v } = panoramaUvAt(event);
  const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/analysis/${encodeURIComponent(images[currentIndex].id)}/pick`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ u, v }),
  });
  if (!response.ok) return;
  const result = await response.json();
  if (datasetId !== requestedDataset || images[currentIndex]?.id !== requestedImageId) return;
  if (result.building?.osm_id) selectBuilding(result.building.osm_id);
}

function renderList() {
  elements.imageList.replaceChildren(...images.map((image, index) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-current", String(index === currentIndex));
    const indexElement = document.createElement("span");
    indexElement.className = "list-index";
    indexElement.textContent = String(index + 1).padStart(2, "0");
    const content = document.createElement("span");
    const date = document.createElement("span");
    date.className = "list-date";
    date.textContent = formatDate(image.captured_at_ms);
    const location = document.createElement("span");
    location.className = "list-location";
    location.textContent = coordinateLabel(image);
    content.append(date, location);
    button.append(indexElement, content);
    button.addEventListener("click", () => selectImage(index));
    item.append(button);
    return item;
  }));
}

function updateControls() {
  const hasImages = images.length > 0;
  elements.sequencePosition.textContent = hasImages ? `${currentIndex + 1} / ${images.length}` : "0 / 0";
  elements.previousButton.disabled = !hasImages;
  elements.nextButton.disabled = !hasImages;
  updateSceneNavigation();
}

function resetView() {
  yaw = 0;
  pitch = 0;
  fov = DEFAULT_FOV;
  render();
  scheduleViewIndicator();
}

function assetUrl(image) {
  const encodedFile = image.local_file.split("/").map(encodeURIComponent).join("/");
  return `/data/${encodeURIComponent(datasetId)}/${encodedFile}`;
}

function selectImage(index, reset = false) {
  if (!images.length) return;
  currentIndex = (index + images.length) % images.length;
  const image = images[currentIndex];
  if (reset) resetView();
  renderList();
  updateControls();
  updateMetadata(image);
  updateMap(image);
  loadBuildingAnalysis(image);
  loadSfmRefinement(image);
  scheduleViewIndicator();
  clearError();
  const loadId = ++activeLoad;
  elements.loading.hidden = false;
  elements.loading.textContent = `Loading panorama ${currentIndex + 1} of ${images.length}...`;
  setActivity(elements.loading, true);
  elements.status.textContent = `Dataset ${datasetId}: panorama ${currentIndex + 1} of ${images.length}. Drag to look around.`;
  const textureImage = new Image();
  textureImage.onload = () => {
    if (loadId !== activeLoad) return;
    try {
      const texture = uploadTexture(textureImage);
      elements.loading.hidden = true;
      setActivity(elements.loading, false);
      if (texture.reduced) {
        elements.status.textContent = `Panorama ${currentIndex + 1} of ${images.length}. ${texture.originalWidth}×${texture.originalHeight} resized to ${texture.width}×${texture.height} for the GPU.`;
      } else {
        elements.status.textContent = `Panorama ${currentIndex + 1} of ${images.length}. Local resolution: ${texture.width}×${texture.height}.`;
      }
    } catch (error) {
      showError(error.message);
    }
  };
  textureImage.onerror = () => {
    if (loadId === activeLoad) showError(`${image.local_file} was not found in this dataset.`);
  };
  textureImage.src = assetUrl(image);
}

async function loadDataset(nextDatasetId, target = null) {
  const requestId = ++datasetLoadRequest;
  activeLoad += 1;
  elements.loading.hidden = false;
  elements.loading.textContent = "Reading dataset manifest...";
  setActivity(elements.loading, true);
  elements.status.textContent = `Loading ${nextDatasetId}...`;
  setActivity(elements.status, true);
  clearError();
  try {
    const response = await fetch(`/api/datasets/${encodeURIComponent(nextDatasetId)}`);
    if (!response.ok) throw new Error(`Dataset could not be read (${response.status}).`);
    const manifest = await response.json();
    if (requestId !== datasetLoadRequest) return;
    datasetId = nextDatasetId;
    images = sortedLocalPanoramas(manifest);
    currentIndex = 0;
    const summary = datasets.find((dataset) => dataset.id === datasetId);
    elements.datasetSummary.textContent = `${images.length} local panoramas${summary?.provider ? ` · ${summary.provider}` : ""}`;
    elements.deleteDatasetButton.disabled = false;
    updateControls();
    if (!images.length) {
      elements.imageList.replaceChildren();
      showError("This manifest does not contain downloaded local panoramas.");
      return;
    }
    const requestedIndex = findImageIndex(images, target?.imageId);
    if (requestedIndex < 0) throw new Error(`Panorama ${target.imageId} is not part of dataset ${datasetId}.`);
    yaw = Number.isFinite(target?.yawDeg) ? target.yawDeg * Math.PI / 180 : 0;
    // Public links use the conventional positive-up pitch; WebGL stores positive-down.
    pitch = Number.isFinite(target?.pitchDeg) ? -target.pitchDeg * Math.PI / 180 : 0;
    fov = Number.isFinite(target?.fovDeg) ? target.fovDeg : DEFAULT_FOV;
    selectedBuildingId = target?.buildingId || null;
    updateReportButton();
    selectImage(requestedIndex);
  } catch (error) {
    if (requestId === datasetLoadRequest) showError(error.message);
  } finally {
    if (requestId === datasetLoadRequest) setActivity(elements.status, false);
  }
}

async function deleteCurrentDataset() {
  if (!datasetId || !window.confirm(`Dataset "${datasetId}" and all its local images will be permanently deleted.`)) return;
  const deletedId = datasetId;
  elements.deleteDatasetButton.disabled = true;
  elements.status.textContent = `Deleting ${deletedId}...`;
  setActivity(elements.status, true);
  try {
    const response = await fetch(`/api/datasets/${encodeURIComponent(deletedId)}`, { method: "DELETE" });
    if (!response.ok) throw new Error(`Dataset could not be deleted (${response.status}).`);
    datasets = datasets.filter((dataset) => dataset.id !== deletedId);
    elements.datasetSelect.replaceChildren(...datasets.map((dataset) => {
      const option = document.createElement("option");
      option.value = dataset.id;
      option.textContent = `${dataset.id} (${dataset.images})`;
      return option;
    }));
    if (!datasets.length) {
      datasetId = null;
      images = [];
      activeLoad += 1;
      elements.datasetSelect.disabled = true;
      elements.imageList.replaceChildren();
      elements.metadata.hidden = true;
      elements.datasetSummary.textContent = "No local datasets remain.";
      elements.status.textContent = "The dataset was deleted.";
      elements.loading.hidden = true;
      updateControls();
      return;
    }
    elements.datasetSelect.value = datasets[0].id;
    await loadDataset(datasets[0].id);
  } catch (error) {
    elements.deleteDatasetButton.disabled = false;
    showError(error.message);
  } finally {
    setActivity(elements.status, false);
  }
}

async function generateBuildingReport() {
  if (!datasetId || !selectedBuildingId || !analysisHasBuilding(selectedBuildingId)) return;
  const [osmType, osmNumber] = selectedBuildingId.split("/");
  const popup = window.open("", "_blank");
  elements.generateReportButton.disabled = true;
  elements.generateReportButton.textContent = "Generating report...";
  showBuildingStatus("Generating ranked views, annotated screenshots, and footprint assessment...");
  try {
    const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/building-reports/${encodeURIComponent(osmType)}/${encodeURIComponent(osmNumber)}`, { method: "POST" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Report could not be generated (${response.status}).`);
    showBuildingStatus(`Report generated from ${result.report.ranked_views.length} panorama view${result.report.ranked_views.length === 1 ? "" : "s"}.`);
    if (popup) popup.location.href = result.links.html;
    else window.open(result.links.html, "_blank", "noopener");
  } catch (error) {
    popup?.close();
    showBuildingStatus(`Report generation failed: ${error.message}`);
  } finally {
    elements.generateReportButton.textContent = "Generate report";
    updateReportButton();
  }
}

function bindInteraction() {
  elements.canvas.addEventListener("pointerdown", (event) => {
    pointer = { id: event.pointerId, moved: false, startX: event.clientX, startY: event.clientY, x: event.clientX, y: event.clientY };
    elements.canvas.setPointerCapture(event.pointerId);
    elements.canvas.classList.add("dragging");
  });
  elements.canvas.addEventListener("pointermove", (event) => {
    if (!pointer || event.pointerId !== pointer.id) return;
    // Drag the panorama itself: down looks up, right looks left.
    yaw -= (event.clientX - pointer.x) * 0.006;
    pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, pitch - (event.clientY - pointer.y) * 0.006));
    pointer = { ...pointer, moved: pointer.moved || Math.hypot(event.clientX - pointer.startX, event.clientY - pointer.startY) > 3, x: event.clientX, y: event.clientY };
    render();
    scheduleViewIndicator();
  });
  const endPointer = (event, cancelled = false) => {
    if (!pointer || event.pointerId !== pointer.id) return;
    const clicked = !pointer.moved;
    pointer = null;
    elements.canvas.classList.remove("dragging");
    if (clicked && !cancelled) pickBuildingAt(event);
  };
  elements.canvas.addEventListener("pointerup", endPointer);
  elements.canvas.addEventListener("pointercancel", (event) => endPointer(event, true));
  elements.canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    fov = Math.max(45, Math.min(110, fov + event.deltaY * 0.04));
    render();
    scheduleViewIndicator();
  }, { passive: false });
  elements.previousButton.addEventListener("click", () => selectImage(currentIndex - 1));
  elements.nextButton.addEventListener("click", () => selectImage(currentIndex + 1));
  elements.resetButton.addEventListener("click", resetView);
  elements.searchMapillaryButton.addEventListener("click", openMapillaryDialog);
  elements.analyzeBuildingsButton.addEventListener("click", analyzeBuildings);
  elements.refineNearbyPosesButton.addEventListener("click", refineNearbyPoses);
  elements.generateReportButton.addEventListener("click", generateBuildingReport);
  elements.maskToggleButton.addEventListener("click", toggleBuildingOverlay);
  elements.closeMapillaryDialogButton.addEventListener("click", () => elements.mapillaryDialog.close());
  elements.mapillarySearchButton.addEventListener("click", searchMapillaryAoi);
  elements.mapillarySelectAllButton.addEventListener("click", () => {
    const shouldSelect = selectedMapillaryIds.size !== mapillarySearchResults.length;
    mapillarySearchResults.forEach((panorama) => setMapillarySelection(panorama.id, shouldSelect));
  });
  elements.mapillaryDownloadButton.addEventListener("click", downloadSelectedMapillaryPanoramas);
  elements.mapillaryDownloadAllButton.addEventListener("click", downloadAllMapillaryPanoramas);
  elements.datasetSelect.addEventListener("change", () => loadDataset(elements.datasetSelect.value));
  elements.deleteDatasetButton.addEventListener("click", deleteCurrentDataset);
  elements.mapDetachButton.addEventListener("click", toggleMapPanel);
  elements.mapPopoutButton.addEventListener("click", openMapWindow);
  elements.mapToggleButton.addEventListener("click", () => setMapPanelVisible(elements.mapPanel.hidden));
  elements.sceneNavigation.addEventListener("click", (event) => {
    const button = event.target.closest(".scene-navigation-point");
    if (button) moveToNearbyPanorama(sceneNavigationTargets.get(Number(button.dataset.index)));
  });
  bindMapPanelDrag();
  mapWindowChannel?.addEventListener("message", (event) => {
    const message = event.data;
    if (message?.type === "request-state") {
      broadcastMapWindowState();
      return;
    }
    if (message?.type === "select" && message.datasetId === datasetId && Number.isInteger(message.index)) {
      selectImage(message.index);
    }
    if (message?.type === "building-select" && message.datasetId === datasetId) {
      selectBuilding(message.osmId, true, Boolean(message.focus));
    }
  });
  window.addEventListener("keydown", (event) => {
    if (event.target.closest?.("input, select, textarea, [contenteditable=true]")) return;
    if (event.key === "ArrowLeft") { event.preventDefault(); selectImage(currentIndex - 1); }
    if (event.key === "ArrowRight") { event.preventDefault(); selectImage(currentIndex + 1); }
    if (event.key.toLowerCase() === "r") resetView();
  });
}

async function start() {
  try {
    initializeRenderer();
    bindInteraction();
    setActivity(elements.loading, true);
    await loadAnalysisBackend();
    const response = await fetch("/api/datasets");
    if (!response.ok) throw new Error("The local library could not be read.");
    const catalog = await response.json();
    datasets = catalog.datasets || [];
    if (!datasets.length) throw new Error("No manifests found in data/. Download a dataset first.");
    elements.datasetSelect.replaceChildren(...datasets.map((dataset) => {
      const option = document.createElement("option");
      option.value = dataset.id;
      option.textContent = `${dataset.id} (${dataset.images})`;
      return option;
    }));
    elements.datasetSelect.disabled = false;
    const target = parseViewerTarget(window.location.search);
    const first = datasets.some((dataset) => dataset.id === target.datasetId) ? target.datasetId : datasets[0].id;
    elements.datasetSelect.value = first;
    await loadDataset(first, first === target.datasetId ? target : null);
  } catch (error) {
    showError(`${error.message} Run uv run city-analyser-viewer and open http://127.0.0.1:8765/viewer/.`);
  }
}

start();
