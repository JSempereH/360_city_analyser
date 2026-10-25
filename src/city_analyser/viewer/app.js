import { applyMatrix, bearingDegrees, distanceMeters, LEVEL_IDENTITY, levelMatrix, nearbyPanoramas, nearestPanoramasByBearing, projectGroundPanoramaToView, signedDegrees } from "./navigation.mjs";
import { buildViewerSearch, DEFAULT_FOV, findImageIndex, parseSearchQuery, parseViewerTarget } from "./deep-link.mjs";
import { coordinatesOf, destinationPoint, geometryCentroid, mergeBuildingFootprints, nearestImageIndex, normalizeDegrees, outerRingsOf, segmentIntersectionFraction, sortedLocalPanoramas } from "./map-geometry.mjs";
import { editableRings, startFootprintEdit } from "./footprint-editor.mjs";
import { bearingCenter, buildReviewQueue, preservedIndex, REVIEW_KINDS, stepIndex } from "./review-queue.mjs";
import { wallHeight, wallPolylines } from "./wall-projection.mjs";
import { attributeRows, gemRow, observationItems, referenceRow, reviewFormValues, reviewOptions, reviewStatus, setAsideNote } from "./building-card.mjs";
import { attributeClass, attributeLabel, COLOR_MODES, legendItems } from "./attribute-map.mjs";
import { DEFAULT_SPEED, startPresentation } from "./presentation.mjs";
import { answerLine, boldSpans, framingOf, streamEvents, visibleReasoning } from "./assistant.mjs";

const elements = {
  analyzeBuildingsButton: document.querySelector("#analyze-buildings-button"),
  refineNearbyPosesButton: document.querySelector("#refine-nearby-poses-button"),
  analyzeDatasetButton: document.querySelector("#analyze-dataset-button"),
  analyzeMapViewButton: document.querySelector("#analyze-map-view-button"),
  buildingStatus: document.querySelector("#building-status"),
  buildingCard: document.querySelector("#building-card"),
  buildingCardAttributes: document.querySelector("#building-card-attributes"),
  buildingCardClose: document.querySelector("#building-card-close"),
  buildingCardMessage: document.querySelector("#building-card-message"),
  buildingCardMore: document.querySelector("#building-card-more"),
  buildingCardObservations: document.querySelector("#building-card-observations"),
  buildingCardTitle: document.querySelector("#building-card-title"),
  buildingCardWhy: document.querySelector("#building-card-why"),
  buildingCardReview: document.querySelector("#building-card-review"),
  buildingCardReviewStatus: document.querySelector("#building-card-review-status"),
  buildingCardReviewForm: document.querySelector("#building-card-review-form"),
  buildingCardReviewMessage: document.querySelector("#building-card-review-message"),
  buildingCardReviewCancel: document.querySelector("#building-card-review-cancel"),
  buildingCardReviewWithdraw: document.querySelector("#building-card-review-withdraw"),
  reviewGemMaterial: document.querySelector("#review-gem-material"),
  reviewGemLlrs: document.querySelector("#review-gem-llrs"),
  reviewFloors: document.querySelector("#review-floors"),
  reviewNote: document.querySelector("#review-note"),
  reviewEvaluation: document.querySelector("#review-evaluation"),
  reviewBuildingType: document.querySelector("#review-building-type"),
  reviewSurfaceMaterial: document.querySelector("#review-surface-material"),
  reviewConstructionYear: document.querySelector("#review-construction-year"),
  assistantStatus: document.querySelector("#assistant-status"),
  assistantBuilding: document.querySelector("#assistant-building"),
  assistantAnalyzeButton: document.querySelector("#assistant-analyze-button"),
  assistantEditButton: document.querySelector("#assistant-edit-button"),
  assistantLog: document.querySelector("#assistant-log"),
  assistantForm: document.querySelector("#assistant-form"),
  assistantInput: document.querySelector("#assistant-input"),
  assistantSendButton: document.querySelector("#assistant-send-button"),
  attributeColorSelect: document.querySelector("#attribute-color-select"),
  mapLegend: document.querySelector("#map-legend"),
  mapLegendTitle: document.querySelector("#map-legend-title"),
  mapLegendItems: document.querySelector("#map-legend-items"),
  buildingAttributesButton: document.querySelector("#building-attributes-button"),
  canvasWrap: document.querySelector("#canvas-wrap"),
  transitionCanvas: document.querySelector("#transition-canvas"),
  presentationButton: document.querySelector("#presentation-button"),
  canvas: document.querySelector("#panorama-canvas"),
  projectionOverlay: document.querySelector("#projection-overlay"),
  footprintUndoButton: document.querySelector("#footprint-undo-button"),
  footprintSplitButton: document.querySelector("#footprint-split-button"),
  footprintRedoButton: document.querySelector("#footprint-redo-button"),
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
  layerInputs: [...document.querySelectorAll("[data-layer]")],
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
  closeHelpDialogButton: document.querySelector("#close-help-dialog-button"),
  helpButton: document.querySelector("#help-button"),
  helpDialog: document.querySelector("#help-dialog"),
  themeSelect: document.querySelector("#theme-select"),
  footprintCancelButton: document.querySelector("#footprint-cancel-button"),
  footprintEditButton: document.querySelector("#footprint-edit-button"),
  footprintEditForm: document.querySelector("#footprint-edit-form"),
  footprintEditTarget: document.querySelector("#footprint-edit-target"),
  footprintNote: document.querySelector("#footprint-note"),
  footprintRefreshButton: document.querySelector("#footprint-refresh-button"),
  footprintReviewMessage: document.querySelector("#footprint-review-message"),
  appShell: document.querySelector(".app-shell"),
  buildingBestViewButton: document.querySelector("#building-best-view-button"),
  buildingRevertButton: document.querySelector("#building-revert-button"),
  searchForm: document.querySelector("#search-form"),
  searchInput: document.querySelector("#search-input"),
  searchMessage: document.querySelector("#search-message"),
  buildingDetails: document.querySelector("#building-details"),
  buildingEditButton: document.querySelector("#building-edit-button"),
  buildingEmpty: document.querySelector("#building-empty"),
  buildingFacts: document.querySelector("#building-facts"),
  buildingOsmLink: document.querySelector("#building-osm-link"),
  buildingTitle: document.querySelector("#building-title"),
  jobList: document.querySelector("#job-list"),
  jobsCount: document.querySelector("#jobs-count"),
  messageLog: document.querySelector("#message-log"),
  panelCloseButton: document.querySelector("#panel-close-button"),
  panelToggleButton: document.querySelector("#panel-toggle-button"),
  focusButton: document.querySelector("#focus-button"),
  buildingCardSetAside: document.querySelector("#building-card-set-aside"),
  reviewCount: document.querySelector("#review-count"),
  reviewAcceptButton: document.querySelector("#review-accept-button"),
  reviewAllDatasets: document.querySelector("#review-all-datasets"),
  reviewCurrent: document.querySelector("#review-current"),
  reviewCurrentDetail: document.querySelector("#review-current-detail"),
  reviewCurrentKind: document.querySelector("#review-current-kind"),
  reviewCurrentLabel: document.querySelector("#review-current-label"),
  reviewEditButton: document.querySelector("#review-edit-button"),
  reviewerName: document.querySelector("#reviewer-name"),
  reviewHideResolved: document.querySelector("#review-hide-resolved"),
  reviewKind: document.querySelector("#review-kind"),
  reviewList: document.querySelector("#review-list"),
  reviewNextButton: document.querySelector("#review-next-button"),
  reviewPosition: document.querySelector("#review-position"),
  reviewPreviousButton: document.querySelector("#review-previous-button"),
  reviewRejectButton: document.querySelector("#review-reject-button"),
  reviewShowButton: document.querySelector("#review-show-button"),
  sidePanel: document.querySelector("#side-panel"),
  tabs: [...document.querySelectorAll('.panel-tabs [role="tab"]')],
};

const PANEL_TABS = ["building", "review", "assistant", "jobs"];
const PREFERENCES_KEY = "viewer-preferences";

// Per-viewer conveniences: theme, panel state, open tab and map visibility.
function preferences() {
  try {
    const stored = JSON.parse(window.localStorage.getItem(PREFERENCES_KEY) || "{}");
    return stored && typeof stored === "object" ? stored : {};
  } catch {
    return {};
  }
}

function savePreference(name, value) {
  try {
    window.localStorage.setItem(PREFERENCES_KEY, JSON.stringify({ ...preferences(), [name]: value }));
  } catch {
    // Private windows may refuse storage; the setting applies to this page only.
  }
}

function applyTheme(theme) {
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  elements.themeSelect.value = ["light", "dark"].includes(theme) ? theme : "system";
}
const MAX_LOGGED_MESSAGES = 100;
// Below this width the details panel overlays the viewer instead of taking a column.
const OVERLAY_PANEL_QUERY = window.matchMedia("(max-width: 1180px)");
const PHONE_QUERY = window.matchMedia("(max-width: 760px)");

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
  uniform sampler2D instanceMask;
  uniform float instanceVisible;
  uniform float selectionVisible;
  uniform float selectionEmphasis;
  uniform float fillStrength;
  uniform vec2 maskSize;
  uniform float vegetationVisible;
  uniform float aspect;
  uniform float yaw;
  uniform float pitch;
  uniform float fov;
  uniform mat3 level;
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
    // Undo the camera's tilt, so the horizon stays level whatever the capture.
    direction = level * direction;
    float longitude = atan(direction.x, direction.z);
    float latitude = asin(clamp(direction.y, -1.0, 1.0));
    vec2 panoramaUv = vec2(0.5 + longitude / (2.0 * PI), 0.5 - latitude / PI);
    vec4 panoramaColor = texture2D(panorama, panoramaUv);
    float selected = texture2D(selectionMask, panoramaUv).r * selectionVisible;
    float vegetation = texture2D(vegetationMask, panoramaUv).r * vegetationVisible;
    vec4 instanceColor = texture2D(instanceMask, panoramaUv);
    vec4 instancedColor = mix(panoramaColor, vec4(instanceColor.rgb, 1.0), instanceColor.a * 0.38 * instanceVisible);
    vec4 completedColor = mix(instancedColor, vec4(0.20, 0.65, 1.0, 1.0), vegetation * 0.18);
    // Overview: a light tint on every matched building. Selection: a stronger
    // fill plus an outline, so one building stands out on dark facades too.
    vec4 color = mix(completedColor, vec4(1.0, 0.57, 0.12, 1.0), selected * fillStrength);
    if (selectionEmphasis > 0.5 && selectionVisible > 0.01) {
      vec2 texel = 1.5 / maskSize;
      float neighbors = texture2D(selectionMask, panoramaUv + vec2(texel.x, 0.0)).r
        + texture2D(selectionMask, panoramaUv - vec2(texel.x, 0.0)).r
        + texture2D(selectionMask, panoramaUv + vec2(0.0, texel.y)).r
        + texture2D(selectionMask, panoramaUv - vec2(0.0, texel.y)).r;
      float outline = selected * step(0.5, 4.0 - neighbors);
      color = mix(color, vec4(1.0, 0.86, 0.55, 1.0), outline * selectionVisible);
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
// The rotation that levels the panorama on screen (see levelMatrix), set with its texture.
let panoramaLevel = LEVEL_IDENTITY;
let selectionTexture = null;
// A new selection mask fades in instead of appearing at once.
let maskOpacity = 1;
let maskFadeFrame = null;

function fadeInMask(durationMs = 500) {
  window.cancelAnimationFrame(maskFadeFrame);
  const started = performance.now();
  const step = (now) => {
    maskOpacity = Math.min(1, (now - started) / durationMs);
    render();
    if (maskOpacity < 1) maskFadeFrame = window.requestAnimationFrame(step);
  };
  maskOpacity = 0;
  maskFadeFrame = window.requestAnimationFrame(step);
}
let vegetationTexture = null;
let selectionIsSingleBuilding = false;
let selectionMaskSize = [2048, 1024];
let currentAnalysis = null;
let currentSfmRefinement = null;
let selectedBuildingId = null;
let buildingCardRequest = 0;
// The card closed by the user stays closed for that building until it is reopened or another one is selected.
let dismissedBuildingCard = null;
let renderedBuildingCard = null;
// The card on screen (for the review form) and the dataset's attribute summary (map colors, review queue).
let currentBuildingCard = null;
// A review form asked for before its building's card had loaded.
let pendingReviewForm = null;
let attributeSummary = null;
let attributeSummaryRequest = 0;
let attributeSummaryTimer = null;
// A survey may be writing while the viewer is open: colors refresh this often.
const ATTRIBUTE_SUMMARY_REFRESH_MS = 60_000;
// The assistant (model status, one chat per building, the request in flight).
let assistantStatus = null;
const chatHistories = new Map();
let assistantRequest = null;
// Presentation mode: the running controller, and the buildings it has "saved" (only those are colored).
let presentation = null;
let presentationLive = null;
// Resolved when the next panorama texture is on screen (smooth transitions wait for it).
const textureWaiters = [];
let presentationRevealed = null;
// What the presentation colors the map by; the viewer's saved choice is left alone.
let presentationColorMode = "gem_material";
let presentationRestore = null;
// The layers a presentation shows, whatever the viewer had on.
const PRESENTATION_LAYERS = { masks: true, vegetation: true, instances: false, coverage: false, footprints: true };
const LAYER_DEFAULTS = { masks: true, vegetation: true, instances: false, footprints: true, captures: true, view: true, diagnostics: true, coverage: false };
const LAYERS_KEY = "viewer-layers";
const layers = { ...LAYER_DEFAULTS, ...storedLayers() };
let instanceTexture = null;
let diagnosticsLayer = null;
let coverageStatus = {};

function storedLayers() {
  try {
    const stored = JSON.parse(window.localStorage.getItem(LAYERS_KEY) || "{}");
    return Object.fromEntries(Object.entries(stored).filter(([name, value]) => name in LAYER_DEFAULTS && typeof value === "boolean"));
  } catch {
    return {};
  }
}
let analysisRequest = 0;
let sfmRequest = 0;
let maskLoadRequest = 0;
let analysisJobRunning = false;
let analysisBackend = { kind: "local", label: "local device" };
let locationMap = null;
let cameraMarker = null;
let buildingLayer = null;
let footprintPreviewLayer = null;
let footprintEditMode = false;
let footprintEditSession = null;
let footprintEditFeature = null;
const FOOTPRINT_AUTHOR_KEY = "footprint-author";
let projection = null;
let reviewData = { corrections: [], issues: null };
let reviewQueue = [];
let reviewIndex = -1;
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

// The presentation's options from the page's own URL (speed, min-score, then, live), read once: loading
// a dataset rewrites the URL before the presentation starts.
const PRESENTATION_OPTIONS = ["speed", "min-score", "then", "live"];
// Evaluation mode (?references=1): the card shows reference data (the cadastre) next to the answers. The analysis
// never uses it, and a city without a cadastre must look the same, so it is off unless asked for.
const SHOW_REFERENCES = new URLSearchParams(window.location.search).get("references") === "1";
const REFERENCES_QUERY = SHOW_REFERENCES ? "?references=1" : "";
const presentationParameters = new URLSearchParams(
  [...new URLSearchParams(window.location.search)].filter(([name]) => PRESENTATION_OPTIONS.includes(name)),
);

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
    reviewKey: !elements.sidePanel.hidden && currentPanelTab() === "review" ? reviewQueue[reviewIndex]?.key : null,
  });
  // A reload during a presentation resumes it, with the same options.
  const options = presentationParameters.toString();
  const presenting = presentation ? `${search ? "&" : "?"}present=1${options ? `&${options}` : ""}` : "";
  const evaluation = SHOW_REFERENCES ? `${search || presenting ? "&" : "?"}references=1` : "";
  window.history.replaceState(null, "", `${window.location.pathname}${search}${presenting}${evaluation}`);
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
    fillStrength: gl.getUniformLocation(program, "fillStrength"),
    vegetationVisible: gl.getUniformLocation(program, "vegetationVisible"), yaw: gl.getUniformLocation(program, "yaw"),
    instanceMask: gl.getUniformLocation(program, "instanceMask"), instanceVisible: gl.getUniformLocation(program, "instanceVisible"),
    level: gl.getUniformLocation(program, "level"),
  };
  gl.uniform1i(uniforms.panorama, 0);
  gl.uniform1i(uniforms.selectionMask, 1);
  gl.uniform1i(uniforms.vegetationMask, 2);
  gl.uniform1i(uniforms.instanceMask, 3);
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
  // WebGL 1 takes no transpose flag, so the row-major matrix goes in transposed (column-major).
  const [a, b, c, d, e, f, g, h, i] = panoramaLevel;
  gl.uniformMatrix3fv(uniforms.level, false, [a, d, g, b, e, h, c, f, i]);
  gl.uniform1f(uniforms.selectionVisible, selectionTexture && layers.masks ? maskOpacity : 0);
  gl.uniform1f(uniforms.vegetationVisible, vegetationTexture && layers.vegetation ? 1 : 0);
  gl.uniform1f(uniforms.instanceVisible, instanceTexture && layers.instances ? 1 : 0);
  gl.uniform1f(uniforms.selectionEmphasis, selectionIsSingleBuilding ? 1 : 0);
  // A lighter fill while presenting: the facade stays readable behind the mask at close range.
  gl.uniform1f(uniforms.fillStrength, selectionIsSingleBuilding ? (presentationRevealed ? 0.28 : 0.42) : 0.26);
  gl.uniform2f(uniforms.maskSize, selectionMaskSize[0], selectionMaskSize[1]);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  drawProjection();
}

// The camera the analysis projected from: its refined pose when available.
function projectionCamera() {
  const image = images[currentIndex];
  if (!image) return null;
  const pose = currentAnalysis && currentAnalysis.image_id === image.id ? currentAnalysis.refined_pose : null;
  if (Number.isFinite(pose?.latitude) && Number.isFinite(pose?.heading_degrees)) return { position: [pose.longitude, pose.latitude], headingDeg: pose.heading_degrees };
  const coordinates = coordinatesOf(image);
  const heading = recordedHeading(image);
  return coordinates && heading !== null ? { position: coordinates, headingDeg: heading } : null;
}

// Walls of the footprint being edited, drawn over the panorama next to the OSM original.
function drawProjection() {
  const overlay = elements.projectionOverlay;
  const context = overlay.getContext("2d");
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = elements.canvas.clientWidth;
  const height = elements.canvas.clientHeight;
  if (overlay.width !== Math.floor(width * ratio) || overlay.height !== Math.floor(height * ratio)) {
    overlay.width = Math.floor(width * ratio);
    overlay.height = Math.floor(height * ratio);
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const camera = projectionCamera();
  if (!projection || !camera) return;
  const view = { headingDeg: camera.headingDeg, yaw, pitch, fovDeg: fov, width, height };
  const stroke = (geometry, color, dash, lineWidth) => {
    context.strokeStyle = color;
    context.lineWidth = lineWidth;
    context.setLineDash(dash);
    outerRingsOf(geometry).forEach((ring) => {
      wallPolylines(camera.position, ring, projection.height, view).forEach((line) => {
        context.beginPath();
        line.forEach(([x, y], index) => (index ? context.lineTo(x, y) : context.moveTo(x, y)));
        context.stroke();
      });
    });
  };
  if (projection.original) stroke(projection.original, "rgba(220, 226, 222, 0.85)", [5, 5], 1.5);
  (projection.edited || []).forEach((geometry) => stroke(geometry, "#2f80ed", [], 2.5));
}

function updateUndoButtons() {
  elements.footprintUndoButton.disabled = !footprintEditSession?.canUndo();
  elements.footprintRedoButton.disabled = !footprintEditSession?.canRedo();
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
    throw new Error(`WebGL could not load the texture (error ${textureError}).`);
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
  fadeInMask();
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

function uploadInstanceOverlay(source) {
  const texture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE3);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  // Nearest sampling keeps instance colors from blending at their borders.
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  if (instanceTexture) gl.deleteTexture(instanceTexture);
  instanceTexture = texture;
  gl.activeTexture(gl.TEXTURE0);
  render();
}

function clearInstanceOverlay() {
  if (instanceTexture) gl.deleteTexture(instanceTexture);
  instanceTexture = null;
  render();
}

let instanceOverlayRequest = 0;

function loadInstanceOverlay() {
  const requestId = ++instanceOverlayRequest;
  if (!layers.instances || !currentAnalysis) {
    clearInstanceOverlay();
    return;
  }
  const image = new Image();
  image.onload = () => { if (requestId === instanceOverlayRequest) uploadInstanceOverlay(image); };
  image.onerror = () => { if (requestId === instanceOverlayRequest) clearInstanceOverlay(); };
  image.src = `/api/datasets/${encodeURIComponent(datasetId)}/analysis/${encodeURIComponent(images[currentIndex].id)}/mask?layer=instances&t=${Date.now()}`;
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
  if (message) logMessage(message);
}

// Every status message is kept, newest first, so a message that is
// replaced quickly can still be read in the Jobs tab.
function logMessage(message) {
  if (elements.messageLog.firstElementChild?.dataset.message === message) return;
  const item = document.createElement("li");
  item.dataset.message = message;
  const time = document.createElement("time");
  const now = new Date();
  time.dateTime = now.toISOString();
  time.textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  item.append(time, document.createTextNode(message));
  elements.messageLog.prepend(item);
  while (elements.messageLog.children.length > MAX_LOGGED_MESSAGES) elements.messageLog.lastElementChild.remove();
}

const trackedJobs = new Map();

// Jobs started from this page, polled until they finish, listed in the Jobs tab.
const FINISHED_JOB_STATUSES = new Set(["complete", "failed", "cancelled"]);

function trackJob(job, label, onFinished = null) {
  if (!job?.id || trackedJobs.has(job.id)) return;
  trackedJobs.set(job.id, { ...job, label, startedAt: new Date() });
  renderJobs();
  const poll = async () => {
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(job.id)}`);
      if (!response.ok) throw new Error("The job is no longer known to the server.");
      const latest = await response.json();
      trackedJobs.set(job.id, { ...trackedJobs.get(job.id), ...latest });
      renderJobs();
      if (!FINISHED_JOB_STATUSES.has(latest.status)) window.setTimeout(poll, 1500);
      else {
        logMessage(`${label}: ${latest.message}`);
        onFinished?.(latest);
      }
    } catch (error) {
      trackedJobs.set(job.id, { ...trackedJobs.get(job.id), status: "failed", message: error.message });
      renderJobs();
    }
  };
  window.setTimeout(poll, 500);
}

function renderJobs() {
  const jobs = [...trackedJobs.values()].reverse();
  const running = jobs.filter((job) => !FINISHED_JOB_STATUSES.has(job.status)).length;
  elements.jobsCount.textContent = String(running);
  elements.jobsCount.hidden = running === 0;
  if (!jobs.length) {
    const empty = document.createElement("li");
    empty.className = "panel-empty";
    empty.textContent = "No jobs in this session.";
    elements.jobList.replaceChildren(empty);
    return;
  }
  elements.jobList.replaceChildren(...jobs.map((job) => {
    const item = document.createElement("li");
    item.dataset.status = job.status;
    const title = document.createElement("strong");
    title.textContent = `${job.label} · ${job.status}`;
    const message = document.createElement("span");
    message.textContent = job.message || "";
    item.append(title, message);
    if (Number.isFinite(job.total) && job.total > 0) {
      const progress = document.createElement("progress");
      progress.max = job.total;
      progress.value = job.status === "complete" ? job.total : job.done || 0;
      progress.setAttribute("aria-label", `${job.label}: ${progress.value} of ${job.total}`);
      item.append(progress);
    }
    if (job.cancellable && !FINISHED_JOB_STATUSES.has(job.status)) {
      const cancel = document.createElement("button");
      cancel.type = "button";
      cancel.textContent = "Cancel";
      cancel.addEventListener("click", async () => {
        cancel.disabled = true;
        const response = await fetch(`/api/jobs/${encodeURIComponent(job.id)}/cancel`, { method: "POST" });
        if (response.ok) trackedJobs.set(job.id, { ...trackedJobs.get(job.id), ...(await response.json()).job });
        renderJobs();
      });
      item.append(cancel);
    }
    return item;
  }));
}

function setPanelTab(name, focusTab = false) {
  savePreference("tab", name);
  PANEL_TABS.forEach((tab) => {
    const selected = tab === name;
    const button = document.querySelector(`#tab-${tab}`);
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    document.querySelector(`#panel-${tab}`).hidden = !selected;
    if (selected && focusTab) button.focus();
  });
  if (name === "review") refreshFootprintReview();
}

function currentPanelTab() {
  return PANEL_TABS.find((tab) => document.querySelector(`#tab-${tab}`).getAttribute("aria-selected") === "true");
}

function setSidePanelVisible(visible, remember = true) {
  // Opening the details leaves focus, which hides them.
  if (visible && isFocused()) setFocus(false, remember);
  // Only the docked desktop panel is remembered; drawers open on demand.
  if (remember && !OVERLAY_PANEL_QUERY.matches) savePreference("panel", visible);
  elements.sidePanel.hidden = !visible;
  elements.appShell.classList.toggle("panel-closed", !visible);
  elements.panelToggleButton.setAttribute("aria-expanded", String(visible));
  // The panorama canvas and the map change size with the panel.
  window.setTimeout(() => {
    resizeCanvas();
    locationMap?.invalidateSize();
  }, 0);
}

// Focus: the panorama fills the window, with the library column and the details panel hidden (not closed),
// so a walk or a presentation can be shown large. F or the focus button toggles it.
function isFocused() {
  return elements.appShell.classList.contains("focus");
}

function setFocus(on, remember = true) {
  if (remember) savePreference("focus", on);
  elements.appShell.classList.toggle("focus", on);
  elements.focusButton.setAttribute("aria-pressed", String(on));
  elements.focusButton.title = on ? "Show the side columns (F)" : "Focus on the panorama: hide the side columns (F)";
  window.setTimeout(() => {
    resizeCanvas();
    locationMap?.invalidateSize();
  }, 0);
}

function openPanelTab(name) {
  setSidePanelVisible(true);
  setPanelTab(name);
}

function selectedBuildingFeature() {
  return featureById(selectedBuildingId);
}

function addFact(term, value) {
  const dt = document.createElement("dt");
  dt.textContent = term;
  const dd = document.createElement("dd");
  dd.textContent = value;
  elements.buildingFacts.append(dt, dd);
}

function renderBuildingPanel() {
  const osmId = selectedBuildingId;
  elements.buildingEmpty.hidden = Boolean(osmId);
  elements.buildingDetails.hidden = !osmId;
  if (!osmId) return;
  const details = Object.values(currentAnalysis?.building_ids || {}).find((building) => building.osm_id === osmId);
  const properties = selectedBuildingFeature()?.properties || {};
  elements.buildingTitle.textContent = osmId;
  elements.buildingFacts.replaceChildren();
  const version = details?.osm_version ?? properties.osm_version;
  if (version) addFact("OSM version", `${version}${details?.osm_timestamp || properties.osm_timestamp ? ` · ${String(details?.osm_timestamp || properties.osm_timestamp).slice(0, 10)}` : ""}`);
  const correction = details?.footprint_correction || properties.footprint_correction;
  addFact("Footprint", correction
    ? `${correction.kind === "split" ? "Split" : "Corrected"} locally (revision ${correction.revision}, ${correction.source})`
    : properties.footprint_correction_stale ? "Correction is stale: OSM changed since it was drawn" : "As served by OpenStreetMap");
  if (properties.building && properties.building !== "yes") addFact("OSM building type", properties.building);
  if (details) {
    addFact("Visible facades", String(details.facades || 1));
    addFact("Observed", `${Math.round(details.confidence * 100)}% of the projected facade`);
    if (Number.isFinite(details.height_m)) addFact("Height used", `${details.height_m.toFixed(1)} m (${details.height_source || "default"})`);
    if (details.sfm_depth_samples) addFact("SfM depth", `${details.sfm_depth_samples} points${details.sfm_depth_conflicts ? `, ${details.sfm_depth_conflicts} rejected` : ""}`);
  } else {
    addFact("This panorama", currentAnalysis ? "No directly visible facade" : "Not analyzed yet");
  }
  elements.buildingRevertButton.hidden = !correction;
  const parent = properties.parent_osm_id || details?.footprint_correction?.kind === "split" && osmId.split("/part-")[0];
  if (parent && parent !== osmId) addFact("Split part", `Part ${correction?.part ?? "?"} of ${correction?.parts ?? "?"} of ${parent}`);
  elements.buildingOsmLink.href = `https://www.openstreetmap.org/${osmId.split("/part-")[0]}`;
  loadBuildingViews(osmId);
}

const buildingViewsRequests = new Map();

async function fetchBuildingViews(osmId) {
  const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/building-views/${osmId}`);
  if (!response.ok) throw new Error(`Building views unavailable (${response.status}).`);
  return response.json();
}

async function loadBuildingViews(osmId) {
  const requestId = (buildingViewsRequests.get("current") || 0) + 1;
  buildingViewsRequests.set("current", requestId);
  elements.buildingBestViewButton.hidden = true;
  try {
    const result = await fetchBuildingViews(osmId);
    if (buildingViewsRequests.get("current") !== requestId || selectedBuildingId !== osmId) return;
    const views = result.views || [];
    addFact("Analyzed views", views.length ? `${views.length} panorama${views.length === 1 ? "" : "s"} in this dataset` : "None in this dataset yet");
    const best = views[0];
    elements.buildingBestViewButton.hidden = !best || String(best.image) === String(images[currentIndex]?.id);
  } catch (error) {
    console.warn(error);
  }
}

// Open the panorama that shows a building best, or the nearest one when no analysis shows it.
async function goToBestView(osmId, knownGeometry = null) {
  const result = await fetchBuildingViews(osmId);
  const best = result.views?.[0];
  if (best) {
    await showPanorama(datasetId, best.image);
    selectBuilding(osmId, true, true);
    return `${osmId}: best of ${result.views.length} analyzed panorama${result.views.length === 1 ? "" : "s"}.`;
  }
  const centroid = geometryCentroid(knownGeometry || result.geometry || featureById(osmId)?.geometry);
  if (!centroid) return `${osmId} is not in any analysis of this dataset, and its footprint is not on the map.`;
  const index = nearestImageIndex(images, centroid, distanceMeters);
  if (index < 0) return "This dataset has no georeferenced panoramas.";
  if (index !== currentIndex) await selectImage(index);
  selectBuilding(osmId, true, false);
  lookAtBearing(bearingDegrees(coordinatesOf(images[index]), centroid));
  return `${osmId} is not analyzed yet; showing the nearest panorama, ${Math.round(distanceMeters(coordinatesOf(images[index]), centroid))} m away.`;
}

async function runSearch(event) {
  event.preventDefault();
  const query = parseSearchQuery(elements.searchInput.value);
  const say = (message) => {
    elements.searchMessage.textContent = message;
    elements.searchMessage.hidden = !message;
    if (message) logMessage(message);
  };
  if (!query) {
    say("Enter an OSM element such as way/123, or coordinates such as 9.934, -84.078.");
    return;
  }
  if (!images.length) return;
  try {
    if (query.type === "osm") {
      say(await goToBestView(query.osmId));
      return;
    }
    const point = [query.longitude, query.latitude];
    const index = nearestImageIndex(images, point, distanceMeters);
    if (index < 0) {
      say("This dataset has no georeferenced panoramas.");
      return;
    }
    if (index !== currentIndex) await selectImage(index);
    lookAtBearing(bearingDegrees(coordinatesOf(images[index]), point));
    say(`Nearest panorama is ${Math.round(distanceMeters(coordinatesOf(images[index]), point))} m from that point.`);
  } catch (error) {
    say(`Search failed: ${error.message}`);
  }
}

// Withdraw an accepted correction; a part of a split reverts the whole split.
async function revertSelectedCorrection() {
  if (!selectedBuildingId) return;
  const element = selectedBuildingId.split("/part-")[0];
  const reviewer = requireReviewerName();
  if (!reviewer) return;
  const split = element !== selectedBuildingId;
  if (!window.confirm(`Revert ${element}${split ? " and all parts of its split" : ""} to the OpenStreetMap geometry? The correction stays in the history as reverted.`)) return;
  const note = window.prompt("Why is the correction being reverted? (optional)") ?? "";
  try {
    const result = await footprintRequest(`/api/footprint-corrections/${element}`, { action: "revert", reviewer, note });
    showBuildingStatus(`Revision ${result.revision.revision} of ${element} reverted.`);
    if (result.reassociation?.job) {
      trackJob(result.reassociation.job, "Re-associate panoramas");
      await followFootprintJob(result.reassociation.job.id);
    } else if (result.reassociation?.message) {
      showBuildingStatus(result.reassociation.message);
    }
    if (split) selectBuilding(element, true, false);
    refreshFootprintReview(false);
  } catch (error) {
    showBuildingStatus(`Revert failed: ${error.message}`);
  }
}

function editSelectedFootprint() {
  const feature = selectedBuildingFeature();
  if (!feature) return;
  openPanelTab("review");
  setFootprintEditMode(true);
  beginFootprintEdit(feature);
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
  elements.analyzeDatasetButton.disabled = !images.length;
  elements.analyzeMapViewButton.disabled = !images.length;
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
  window.L.tileLayer("/tiles/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 19,
  }).addTo(locationMap);
  buildingLayer = window.L.geoJSON(null, {
    onEachFeature: (feature, layer) => layer.on("click", () => {
      if (footprintEditMode) beginFootprintEdit(feature);
      else selectBuilding(feature.properties.osm_id, true, true);
    }),
    style: buildingStyle,
  }).addTo(locationMap);
  footprintPreviewLayer = window.L.layerGroup().addTo(locationMap);
  diagnosticsLayer = window.L.layerGroup().addTo(locationMap);
  applyMapLayers();
  viewLayer = window.L.layerGroup().addTo(locationMap);
  captureLayer = window.L.layerGroup().addTo(locationMap);
  new ResizeObserver(() => locationMap.invalidateSize()).observe(elements.mapCanvas);
}

function initializeMapillaryAoiMap() {
  if (mapillaryAoiMap) return;
  mapillaryAoiMap = window.L.map(elements.mapillaryAoiMap, { scrollWheelZoom: true, zoomControl: true }).setView([20, 0], 2);
  window.L.tileLayer("/tiles/{z}/{x}/{y}.png", {
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
  elements.mapillarySearchStatus.textContent = "Querying Mapillary...";
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
    trackJob(result.job, `Download ${datasetId}`);
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

function setMapPanelVisible(visible, remember = true) {
  if (remember && !PHONE_QUERY.matches) savePreference("map", visible);
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
    if (index === currentIndex || (!layers.captures && !layers.coverage)) return;
    const coordinates = coordinatesOf(image);
    if (!coordinates) return;
    const [longitude, latitude] = coordinates;
    const status = coverageStatus[String(image.id)];
    const [color, fillColor] = layers.coverage && status ? COVERAGE_COLORS[status] : ["#7d231c", "#d95045"];
    window.L.circleMarker([latitude, longitude], {
      color, fillColor, fillOpacity: layers.coverage ? 0.8 : 0.35, radius: 5, weight: 2,
    }).addTo(captureLayer).bindTooltip(`Panorama ${index + 1}${layers.coverage && status ? ` · analysis ${status}` : ""}`, { direction: "top" }).on("click", () => selectImage(index));
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

function attributeColorMode() {
  // The presentation colors by structure without changing the viewer's saved choice.
  if (presentationRevealed) return presentationColorMode;
  const mode = preferences().attributeColors;
  return mode in COLOR_MODES ? mode : "none";
}

function attributeStyle(feature, selected) {
  const mode = attributeColorMode();
  if (mode === "none") return null;
  const osmId = feature.properties?.osm_id;
  const visible = !presentationRevealed || presentationRevealed.has(osmId);
  const group = visible ? attributeClass(mode, attributeSummary?.buildings?.[osmId]) : null;
  // Buildings the survey has not reached stay faint, so colored ones read as the result.
  if (!group) return { color: selected ? "#fff" : "#898781", fillColor: "#898781", fillOpacity: 0.06, weight: selected ? 3 : 1 };
  return { color: selected ? "#fff" : "#3d3d3a", fillColor: group.color, fillOpacity: group.key === "undetermined" ? 0.3 : 0.72, weight: selected ? 3 : 1 };
}

function buildingStyle(feature) {
  const selected = feature.properties?.osm_id === selectedBuildingId;
  const directlyVisible = !currentAnalysis || analysisHasBuilding(feature.properties?.osm_id);
  const style = attributeStyle(feature, selected) || (selected
    ? { color: "#fff", fillColor: "#ff8a2a", fillOpacity: 0.58, weight: 3 }
    : !directlyVisible
      ? { color: "#59685f", fillColor: "#59685f", fillOpacity: 0.08, weight: 1 }
      : { color: "#d47a36", fillColor: "#ecba74", fillOpacity: 0.28, weight: 1 });
  // Dashed outlines mark geometry that differs from what OpenStreetMap serves.
  if (feature.properties?.footprint_correction) return { ...style, dashArray: "6 4", weight: Math.max(style.weight, 2) };
  if (feature.properties?.footprint_correction_stale) return { ...style, color: "#c0392b", dashArray: "2 4", weight: 2 };
  return style;
}

function refreshBuildingStyles() {
  const mode = attributeColorMode();
  buildingLayer?.eachLayer((layer) => {
    layer.setStyle(buildingStyle(layer.feature));
    // Colors are below 3:1 on the tiles, so the value is also written: hover shows the GEM string.
    const osmId = layer.feature?.properties?.osm_id;
    if (mode === "none" || !osmId) layer.unbindTooltip();
    else if (presentationRevealed && !presentationRevealed.has(osmId)) layer.unbindTooltip();
    else layer.bindTooltip(attributeLabel(mode, osmId, attributeSummary?.buildings?.[osmId]), { sticky: true, direction: "top" });
  });
}

function renderMapLegend() {
  const mode = attributeColorMode();
  // During a presentation the legend grows with the buildings the assistant has saved.
  const shown = presentationRevealed && attributeSummary
    ? { ...attributeSummary, buildings: Object.fromEntries(Object.entries(attributeSummary.buildings).filter(([osmId]) => presentationRevealed.has(osmId))) }
    : attributeSummary;
  const items = legendItems(mode, shown);
  elements.mapLegend.hidden = mode === "none";
  if (mode === "none") return;
  const counts = presentationRevealed ? { buildings: presentationRevealed.size, reviewed: 0 } : attributeSummary?.counts;
  elements.mapLegendTitle.textContent = counts
    ? `${COLOR_MODES[mode]} · ${counts.buildings} ${presentationRevealed ? "saved" : "surveyed"}${counts.reviewed ? `, ${counts.reviewed} reviewed` : ""}`
    : `${COLOR_MODES[mode]} · loading...`;
  const empty = presentationRevealed ? "Colors appear as buildings are saved" : "No building surveyed yet: run city-analyser-survey";
  elements.mapLegendItems.replaceChildren(...(items.length ? items : [{ label: counts ? empty : "", color: null }])
    .filter((item) => item.label)
    .map((item) => {
      const entry = document.createElement("li");
      if (item.color) {
        const swatch = document.createElement("span");
        swatch.className = "legend-swatch";
        swatch.style.background = item.color;
        entry.append(swatch);
      }
      const label = document.createElement("span");
      label.textContent = item.count ? `${item.label} (${item.count})` : item.label;
      entry.append(label);
      return entry;
    }));
}

async function loadAttributeSummary() {
  window.clearTimeout(attributeSummaryTimer);
  const requestId = ++attributeSummaryRequest;
  const requestedDataset = datasetId;
  if (!requestedDataset) return;
  try {
    const response = await fetch(`/api/datasets/${encodeURIComponent(requestedDataset)}/building-attributes`);
    if (!response.ok) throw new Error(`status ${response.status}`);
    const summary = await response.json();
    if (requestId !== attributeSummaryRequest || requestedDataset !== datasetId) return;
    attributeSummary = summary;
  } catch (error) {
    if (requestId === attributeSummaryRequest) console.warn(`Attribute summary unavailable: ${error.message}`);
  }
  if (requestId !== attributeSummaryRequest) return;
  refreshBuildingStyles();
  renderMapLegend();
  rebuildReviewQueue();
  if (attributeColorMode() !== "none") attributeSummaryTimer = window.setTimeout(loadAttributeSummary, ATTRIBUTE_SUMMARY_REFRESH_MS);
}

async function loadAssistantStatus() {
  try {
    const response = await fetch("/api/assistant");
    assistantStatus = response.ok ? await response.json() : { available: false };
  } catch {
    assistantStatus = { available: false };
  }
  renderAssistant();
}

function chatKey(osmId = selectedBuildingId) {
  return `${datasetId}|${osmId}`;
}

function logEntry(kind, title = "") {
  const item = document.createElement("li");
  item.className = `assistant-entry ${kind}`;
  if (title) {
    const heading = document.createElement("p");
    heading.className = "assistant-entry-title";
    heading.textContent = title;
    item.append(heading);
  }
  const body = document.createElement("p");
  body.className = "assistant-entry-body";
  item.append(body);
  elements.assistantLog.append(item);
  elements.assistantLog.scrollTop = elements.assistantLog.scrollHeight;
  return body;
}

// A model answer as text nodes, with **bold** shown as bold.
function setRichText(element, text) {
  element.replaceChildren(...boldSpans(text).map((span) => {
    if (!span.bold) return document.createTextNode(span.text);
    const strong = document.createElement("strong");
    strong.textContent = span.text;
    return strong;
  }));
}

function renderAssistant() {
  const available = Boolean(assistantStatus?.available);
  const busy = Boolean(assistantRequest);
  elements.assistantStatus.textContent = !assistantStatus ? "Checking the assistant..."
    : available ? `Model: ${assistantStatus.model} (${assistantStatus.backend === "openai" ? "server" : "in this process"})`
      : assistantStatus.help || "No model is configured.";
  elements.assistantStatus.classList.toggle("unavailable", Boolean(assistantStatus) && !available);
  elements.assistantBuilding.textContent = selectedBuildingId || "No building selected";
  elements.assistantAnalyzeButton.disabled = !available || !selectedBuildingId || busy;
  elements.assistantEditButton.disabled = !selectedBuildingId || !currentBuildingCard;
  elements.assistantInput.disabled = !available || !selectedBuildingId;
  elements.assistantSendButton.disabled = !available || !selectedBuildingId || busy;
  elements.assistantSendButton.textContent = busy ? "Working..." : "Send";
  if (busy) return;
  const history = chatHistories.get(chatKey()) || [];
  elements.assistantLog.replaceChildren();
  // Analyses stay in the building's log; the server reads only the user and assistant turns.
  const titles = { user: "You", assistant: "Assistant", analysis: "Analysis" };
  history.forEach((message) => {
    // An analysis transcript carries its own titles.
    setRichText(logEntry(message.role === "user" ? "user" : "assistant", message.role === "analysis" ? "" : titles[message.role]), message.content);
  });
}

async function sendChat(event) {
  event.preventDefault();
  const text = elements.assistantInput.value.trim();
  const osmId = selectedBuildingId;
  if (!text || !osmId || assistantRequest) return;
  const key = chatKey(osmId);
  const history = [...(chatHistories.get(key) || []), { role: "user", content: text }];
  chatHistories.set(key, history);
  elements.assistantInput.value = "";
  renderAssistant();
  assistantRequest = new AbortController();
  renderAssistant();
  const answer = logEntry("assistant pending", "Assistant");
  let reply = "";
  try {
    await streamEvents(`/api/datasets/${encodeURIComponent(datasetId)}/building-attributes/${osmId}/chat`, { messages: history }, (message) => {
      if (message.type === "text") {
        reply += message.text;
        setRichText(answer, reply);
        elements.assistantLog.scrollTop = elements.assistantLog.scrollHeight;
      }
      if (message.type === "error") throw new Error(message.error);
    }, { signal: assistantRequest.signal });
    chatHistories.set(key, [...history, { role: "assistant", content: reply || "(no answer)" }]);
  } catch (error) {
    answer.textContent = `The assistant could not answer: ${error.message}`;
    answer.closest("li").classList.add("failed");
  } finally {
    answer.closest("li").classList.remove("pending");
    assistantRequest = null;
    renderAssistant();
  }
}

// Turn the camera smoothly (Street View style: pan, then tilt, then zoom).
async function tweenCamera(target, durationMs = 900) {
  const start = presentationCamera();
  const delta = ((target.yawDeg - start.yawDeg + 540) % 360) - 180;
  const begin = performance.now();
  await new Promise((resolve) => {
    const frame = (now) => {
      const t = Math.min(1, (now - begin) / durationMs);
      const ease = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
      setPresentationCamera({
        yawDeg: start.yawDeg + delta * ease,
        pitchDeg: start.pitchDeg + (target.pitchDeg - start.pitchDeg) * ease,
        fov: start.fov + (target.fov - start.fov) * ease,
      });
      if (t < 1) window.requestAnimationFrame(frame);
      else resolve();
    };
    window.requestAnimationFrame(frame);
  });
}


/** Ask the model about the selected building now; the camera follows the view it reads and its notes stream in. */
async function analyzeNow() {
  const osmId = selectedBuildingId;
  if (!osmId || assistantRequest || !assistantStatus?.available) return;
  openPanelTab("assistant");
  assistantRequest = new AbortController();
  renderAssistant();
  const status = logEntry("assistant pending", "Analysis");
  status.textContent = "Preparing the views...";
  let current = null;
  let raw = "";
  const cameraMoves = [];
  const key = chatKey(osmId);
  // The status entry above is the first entry of this analysis.
  const firstEntry = elements.assistantLog.children.length - 1;
  try {
    await streamEvents(`/api/datasets/${encodeURIComponent(datasetId)}/building-attributes/${osmId}/analyze`, {}, (message) => {
      if (message.type === "start") status.textContent = `${message.model} is reading ${message.views} view${message.views === 1 ? "" : "s"}.`;
      if (message.type === "view") {
        raw = "";
        current = logEntry("assistant", `View ${message.index + 1}`);
        current.textContent = "Looking...";
        const reference = message.reference || {};
        cameraMoves.push((async () => {
          if (reference.image_id && String(reference.image_id) !== String(images[currentIndex]?.id)) await glideToImage(reference.image_id);
          const framing = framingOf(reference.view, panoramaLevel);
          if (framing) await tweenCamera(framing);
        })());
      }
      if (message.type === "text" && current) {
        raw += message.text;
        current.textContent = visibleReasoning(raw) || "Looking...";
        elements.assistantLog.scrollTop = elements.assistantLog.scrollHeight;
      }
      if (message.type === "answer" && current) {
        const line = message.error ? `No usable answer: ${message.error}` : answerLine(message.answer);
        if (line) current.textContent = `${visibleReasoning(raw) || ""}\n→ ${line}`.trim();
      }
      if (message.type === "done") {
        status.textContent = `Saved: ${message.card.gem.string}. The card now shows this answer; correct it with Edit values if needed.`;
        if (selectedBuildingId === osmId) fillBuildingCard(message.card);
        loadAttributeSummary();
      }
      if (message.type === "error") throw new Error(message.error);
    }, { signal: assistantRequest.signal });
  } catch (error) {
    status.textContent = `Analysis failed: ${error.message}`;
    status.closest("li").classList.add("failed");
  } finally {
    status.closest("li").classList.remove("pending");
    // Kept with the conversation, so the log survives the redraw and a later question can refer to it.
    const transcript = [...elements.assistantLog.children].slice(firstEntry)
      .map((entry) => [...entry.children].map((part) => part.textContent.trim()).filter(Boolean).join("\n")).join("\n\n");
    chatHistories.set(key, [...(chatHistories.get(key) || []), { role: "analysis", content: transcript }]);
    await Promise.allSettled(cameraMoves);
    assistantRequest = null;
    renderAssistant();
  }
}

function editValues() {
  if (!currentBuildingCard) return;
  showBuildingCard();
  window.setTimeout(openReviewForm, 0);
}

function presentationCamera() {
  return { yawDeg: yaw * 180 / Math.PI, pitchDeg: pitch * 180 / Math.PI, fov };
}

function setPresentationCamera(view) {
  yaw = (((view.yawDeg + 180) % 360 + 360) % 360 - 180) * Math.PI / 180;
  pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, view.pitchDeg * Math.PI / 180));
  fov = Math.max(30, Math.min(110, view.fov));
  render();
  scheduleViewIndicator();
}

// Yaw (degrees) that faces a footprint's centroid from the current panorama, when no crop framing is known.
function yawToBuilding(osmId) {
  const image = images[currentIndex];
  const centroid = geometryCentroid(featureById(osmId)?.geometry);
  const camera = coordinatesOf(image);
  const heading = recordedHeading(image);
  if (!centroid || !camera || heading === null) return null;
  return signedDegrees(bearingDegrees(camera, centroid) - heading);
}

function headingOf(image) {
  const heading = recordedHeading(image);
  return heading === null ? 0 : heading;
}

function nextTexture(timeoutMs = 10_000) {
  return new Promise((resolve) => {
    textureWaiters.push(resolve);
    window.setTimeout(resolve, timeoutMs);
  });
}

// Street View style move to another capture: the view keeps its compass direction, and the old
// panorama fades out while it grows a little, as if stepping forward into the new one.
async function glideToImage(imageId) {
  const index = images.findIndex((image) => String(image.id) === String(imageId));
  if (index < 0 || index === currentIndex) return;
  const bearing = yaw * 180 / Math.PI + headingOf(images[currentIndex]);
  render();
  const snapshot = elements.transitionCanvas;
  snapshot.width = elements.canvas.width;
  snapshot.height = elements.canvas.height;
  snapshot.getContext("2d").drawImage(elements.canvas, 0, 0);
  snapshot.style.transition = "none";
  snapshot.style.opacity = "1";
  snapshot.style.transform = "scale(1)";
  snapshot.hidden = false;
  elements.canvasWrap.classList.add("gliding");
  const loaded = nextTexture();
  yaw = signedDegrees(bearing - headingOf(images[index])) * Math.PI / 180;
  selectImage(index);
  await loaded;
  await new Promise((resolve) => window.requestAnimationFrame(() => window.requestAnimationFrame(resolve)));
  snapshot.style.transition = "opacity 800ms ease-in-out, transform 1000ms ease-out";
  snapshot.style.opacity = "0";
  snapshot.style.transform = "scale(1.15)";
  await new Promise((resolve) => window.setTimeout(resolve, 1000));
  snapshot.hidden = true;
  elements.canvasWrap.classList.remove("gliding");
}

// Yaw (degrees, relative to the current panorama) that faces another capture.
function yawToImage(imageId) {
  const target = images.find((image) => String(image.id) === String(imageId));
  const here = images[currentIndex];
  const from = coordinatesOf(here);
  const to = coordinatesOf(target);
  if (!from || !to || !target) return null;
  return signedDegrees(bearingDegrees(from, to) - headingOf(here));
}

// The side panel as it was before the presentation showed the review queue.
function restorePresentationPanel() {
  if (!presentationRestore?.panel) return;
  const { tab, kind, visible, focus } = presentationRestore.panel;
  presentationRestore.panel = null;
  elements.reviewKind.value = kind;
  rebuildReviewQueue();
  if (!PHONE_QUERY.matches) {
    setPanelTab(tab);
    setSidePanelVisible(visible, false);
    if (focus) setFocus(true, false);
  }
}

function startPresentationMode() {
  if (presentation || !datasetId) return;
  const parameters = presentationParameters;
  const speed = Number(parameters.get("speed"));
  const minScore = Number(parameters.get("min-score"));
  const presentationDatasets = [datasetId];
  presentation = startPresentation({
    host: elements.canvasWrap,
    datasetId: () => datasetId,
    // ?then=a,b continues the presentation with other datasets, in a cycle that returns to the first.
    nextDataset: async (go) => {
      const cycle = [presentationDatasets[0], ...(parameters.get("then") || "").split(",").map((name) => name.trim()).filter((name) => datasets.some((item) => item.id === name))];
      if (cycle.length < 2) return null;
      const next = cycle[(cycle.indexOf(datasetId) + 1) % cycle.length];
      if (go) {
        elements.datasetSelect.value = next;
        await loadDataset(next);
        presentationRevealed = new Set();
        renderMapLegend();
      }
      return next;
    },
    imageIds: () => images.map((image) => String(image.id)),
    currentImageId: () => String(images[currentIndex]?.id),
    showImage: (imageId) => glideToImage(imageId),
    yawToImage,
    selectBuilding: (osmId) => selectBuilding(osmId, true, false),
    fetchSummary: async () => {
      const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/building-attributes`);
      if (!response.ok) throw new Error(`attribute summary unavailable (${response.status})`);
      attributeSummary = await response.json();
      return attributeSummary;
    },
    fetchCard: async (osmId) => {
      const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/building-attributes/${osmId}${REFERENCES_QUERY}`);
      return response.ok ? response.json() : null;
    },
    // Live when a model is configured (unless the URL says live=0): the model analyzes each building now.
    liveModel: () => (assistantStatus?.available && parameters.get("live") !== "0" ? assistantStatus.model : null),
    analyzeLive: async (osmId, onEvent) => {
      presentationLive = new AbortController();
      try {
        await streamEvents(`/api/datasets/${encodeURIComponent(datasetId)}/building-attributes/${osmId}/analyze`, {}, onEvent, { signal: presentationLive.signal });
      } finally {
        presentationLive = null;
        loadAttributeSummary();
      }
    },
    cancelLive: () => presentationLive?.abort(),
    getCamera: presentationCamera,
    setCamera: setPresentationCamera,
    // Crop framings are measured in the raw panorama; the viewer shows it leveled.
    frame: (view) => framingOf(view, panoramaLevel),
    showColors: (mode) => {
      presentationColorMode = mode in COLOR_MODES && mode !== "none" ? mode : "gem_material";
      refreshBuildingStyles();
      renderMapLegend();
    },
    // The end of a walk: a taller map framed on the buildings saved in it.
    summaryView: (on) => {
      elements.appShell.classList.toggle("presenting-summary", on);
      if (!on || !locationMap) return;
      const features = [...(presentationRevealed || [])].map(featureById).filter(Boolean);
      window.setTimeout(() => {
        locationMap.invalidateSize();
        if (features.length) locationMap.fitBounds(window.L.geoJSON({ type: "FeatureCollection", features }).getBounds(), { padding: [24, 24], maxZoom: 18 });
      }, 350);
    },
    // The engineer's queue of answers to check, shown at the end of a walk; closing it restores the panel.
    showReviewQueue: (open) => {
      if (open) {
        presentationRestore.panel = { tab: currentPanelTab(), kind: elements.reviewKind.value, visible: !elements.sidePanel.hidden, focus: isFocused() };
        elements.reviewKind.value = "attributes";
        if (!PHONE_QUERY.matches) {
          setSidePanelVisible(true, false);
          setPanelTab("review");
        }
        rebuildReviewQueue();
      } else {
        restorePresentationPanel();
      }
    },
    bearingToBuilding: yawToBuilding,
    toggleFocus: () => setFocus(!isFocused()),
    revealBuilding: (osmId) => {
      presentationRevealed?.add(osmId);
      refreshBuildingStyles();
      renderMapLegend();
    },
    // The survey runs inside the normal viewer: only the building card and status line make room.
    enter: () => {
      presentationRestore = { map: !elements.mapPanel.hidden, layers: { ...layers } };
      // The audience sees the building masks the survey uses, not the raw detector output or diagnostics;
      // the reviewer's own layer choice comes back on exit.
      for (const [name, visible] of Object.entries(PRESENTATION_LAYERS)) {
        if (layers[name] !== visible) setLayer(name, visible, false);
      }
      presentationRevealed = new Set();
      presentationColorMode = "gem_material";
      elements.appShell.classList.add("presenting");
      elements.presentationButton.setAttribute("aria-pressed", "true");
      elements.presentationButton.querySelector("span").textContent = "Stop survey";
      if (!PHONE_QUERY.matches) setMapPanelVisible(true, false);
      refreshBuildingStyles();
      renderMapLegend();
    },
    exit: () => {
      restorePresentationPanel();
      elements.appShell.classList.remove("presenting-summary");
      presentation = null;
      presentationRevealed = null;
      elements.appShell.classList.remove("presenting");
      elements.presentationButton.setAttribute("aria-pressed", "false");
      elements.presentationButton.querySelector("span").textContent = "Run survey";
      if (presentationRestore) {
        setMapPanelVisible(presentationRestore.map, false);
        for (const [name, visible] of Object.entries(presentationRestore.layers || {})) {
          if (layers[name] !== visible) setLayer(name, visible, false);
        }
      }
      refreshBuildingStyles();
      renderMapLegend();
      syncViewerUrl();
    },
  }, { speed: Number.isFinite(speed) && speed > 0 ? speed : DEFAULT_SPEED, minScore: Number.isFinite(minScore) && parameters.has("min-score") ? minScore : 0.03 });
  syncViewerUrl();
}

function setAttributeColorMode(mode) {
  savePreference("attributeColors", mode in COLOR_MODES ? mode : "none");
  elements.attributeColorSelect.value = attributeColorMode();
  refreshBuildingStyles();
  renderMapLegend();
  loadAttributeSummary();
}

function applyAnalysisFootprints() {
  if (!buildingLayer) return;
  const features = mergeBuildingFootprints(currentBuildingFeatures, currentAnalysis?.building_ids);
  buildingLayer.clearLayers();
  buildingLayer.addData({ type: "FeatureCollection", features });
  refreshBuildingStyles();
  renderBuildingPanel();
  renderDiagnostics();
}

function analysisHasBuilding(osmId) {
  return Object.values(currentAnalysis?.building_ids || {}).some((building) => building.osm_id === osmId);
}

function updateReportButton() {
  elements.generateReportButton.disabled = analysisJobRunning || !selectedBuildingId || !analysisHasBuilding(selectedBuildingId);
}

function updateMaskToggle() {
  elements.layerInputs.forEach((input) => { input.checked = layers[input.dataset.layer]; });
}

function setLayer(name, visible, remember = true) {
  layers[name] = visible;
  try {
    if (remember) window.localStorage.setItem(LAYERS_KEY, JSON.stringify(layers));
  } catch {
    // Private windows may refuse storage; layers still work for this page.
  }
  updateMaskToggle();
  if (name === "instances") loadInstanceOverlay();
  applyMapLayers();
  render();
}

function applyMapLayers() {
  if (!locationMap) return;
  const toggle = (layer, visible) => {
    if (!layer) return;
    if (visible && !locationMap.hasLayer(layer)) layer.addTo(locationMap);
    if (!visible && locationMap.hasLayer(layer)) layer.remove();
  };
  toggle(buildingLayer, layers.footprints);
  toggle(captureLayer, layers.captures || layers.coverage);
  toggle(viewLayer, layers.view);
  toggle(diagnosticsLayer, layers.diagnostics);
  renderCaptureLocations();
  renderDiagnostics();
}

const COVERAGE_COLORS = { current: ["#1e6b3a", "#3fbf6c"], stale: ["#8a5a10", "#e0a339"], missing: ["#4a524d", "#9aa49e"] };

async function loadCoverageStatus() {
  if (!datasetId) return;
  try {
    const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/analysis-status`);
    if (!response.ok) return;
    coverageStatus = (await response.json()).panoramas || {};
    renderCaptureLocations();
    renderList();
  } catch (error) {
    console.warn("Analysis coverage unavailable.", error);
  }
}

// Fans toward buildings the footprints do not explain, and outlines of footprints the imagery contradicts.
function renderDiagnostics() {
  if (!diagnosticsLayer) return;
  diagnosticsLayer.clearLayers();
  const image = images[currentIndex];
  if (!layers.diagnostics || !currentAnalysis || currentAnalysis.image_id !== image?.id) return;
  const pose = currentAnalysis.refined_pose;
  const camera = Number.isFinite(pose?.latitude) ? [pose.latitude, pose.longitude] : (coordinatesOf(image) || []).slice().reverse();
  (currentAnalysis.footprint_diagnostics || []).forEach((diagnostic) => {
    if (diagnostic.type === "unexplained_instance" && diagnostic.bearings?.length === 2 && camera.length === 2) {
      const [from, to] = diagnostic.bearings;
      const span = ((to - from) % 360 + 360) % 360;
      const arc = Array.from({ length: 13 }, (_, step) => destinationPoint(camera[0], camera[1], from + span * step / 12, 30));
      window.L.polygon([camera, ...arc], { color: "#c0392b", fillColor: "#e74c3c", fillOpacity: 0.12, weight: 1.5, dashArray: "4 3" })
        .bindTooltip("A building in the imagery with no footprint", { direction: "top" }).addTo(diagnosticsLayer);
    }
    if ((diagnostic.type === "merged_footprint" || diagnostic.type === "unsupported_footprint") && diagnostic.osm_id) {
      const feature = featureById(diagnostic.osm_id);
      if (!feature) return;
      const label = diagnostic.type === "merged_footprint" ? "Several buildings in this footprint" : "No building in the imagery";
      window.L.geoJSON(feature.geometry, { style: { color: "#c0392b", weight: 2, fill: false, dashArray: "2 3" }, interactive: false })
        .addTo(diagnosticsLayer);
      const centroid = geometryCentroid(feature.geometry);
      // The marker sits on the footprint, so a click on it must still select that building.
      if (centroid) {
        window.L.circleMarker([centroid[1], centroid[0]], { color: "#fff", fillColor: "#c0392b", fillOpacity: 1, radius: 4, weight: 1 })
          .bindTooltip(label, { direction: "top" })
          .on("click", () => (footprintEditMode ? beginFootprintEdit(feature) : selectBuilding(diagnostic.osm_id, true, true)))
          .addTo(diagnosticsLayer);
      }
    }
  });
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
  selectedBuildingId = osmId || null;
  if (focus && selectedBuildingId) focusBuilding(selectedBuildingId);
  refreshBuildingStyles();
  if (selectedBuildingId && !analysisHasBuilding(selectedBuildingId)) {
    // Any footprint can be selected; without a facade here, the panel offers its best view.
    loadBuildingOverviewMask();
    showBuildingStatus(`Selected ${selectedBuildingId} · no directly visible facade in this panorama`);
  } else if (selectedBuildingId) {
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
  renderBuildingPanel();
  renderBuildingCard();
  renderAssistant();
  syncViewerUrl();
  if (broadcast) broadcastMapWindowState();
}

function setBuildingCardVisible(visible) {
  elements.buildingCard.hidden = !visible;
  elements.canvasWrap.classList.toggle("has-building-card", visible);
}

function fillBuildingCard(card) {
  currentBuildingCard = card;
  elements.assistantEditButton.disabled = false;
  const status = reviewStatus(card);
  elements.buildingCardReviewStatus.textContent = status;
  elements.buildingCardReviewStatus.hidden = !status;
  elements.buildingCardReview.hidden = false;
  elements.buildingCardReview.textContent = card.review ? "Change review" : "Review";
  if (pendingReviewForm === card.building) {
    pendingReviewForm = null;
    window.setTimeout(openReviewForm, 0);
  }
  const gem = gemRow(card);
  const rows = attributeRows(card);
  fillCardRows([...(gem ? [{ ...gem, className: "gem" }] : []), ...rows], referenceRow(card));
  const observations = observationItems(card);
  elements.buildingCardObservations.replaceChildren(...observations.map((item) => {
    const li = document.createElement("li");
    if (item.crop_url) {
      const image = document.createElement("img");
      image.src = item.crop_url;
      image.alt = Number.isInteger(item.rank) ? `View ${item.rank} of the building` : "View of the building";
      image.loading = "lazy";
      // Crops are re-exported when better views arrive; a missing one just drops its thumbnail.
      image.addEventListener("error", () => {
        image.remove();
        li.classList.add("no-image");
      });
      li.append(image);
    } else {
      li.classList.add("no-image");
    }
    const text = document.createElement("p");
    text.textContent = item.text;
    li.append(text);
    return li;
  }));
  elements.buildingCardWhy.hidden = !observations.length;
  const setAside = setAsideNote(card);
  elements.buildingCardSetAside.textContent = setAside || "";
  elements.buildingCardSetAside.hidden = !setAside;
  elements.buildingCardMessage.textContent = card.no_clean_view || rows.some((row) => row.value !== "Not surveyed") ? "" : "No attribute could be determined for this building.";
}

// Attribute rows, each with the cadastre's value next to it (marked when they disagree), then the cadastre's parcel.
function fillCardRows(rows, reference) {
  elements.buildingCardAttributes.replaceChildren(...rows.flatMap((row) => {
    const dt = document.createElement("dt");
    dt.textContent = row.label;
    const dd = document.createElement("dd");
    dd.classList.toggle("abstained", Boolean(row.abstained));
    if (row.className) dd.classList.add(row.className);
    if (row.title) dd.title = row.title;
    for (const [className, text] of [["value", row.value], ["source", row.source], ["note", row.note]]) {
      if (!text) continue;
      const span = document.createElement("span");
      span.className = className;
      span.textContent = text;
      dd.append(span);
    }
    if (row.reference) {
      const span = document.createElement("span");
      span.className = row.reference.disagrees ? "reference disagrees" : "reference";
      span.textContent = row.reference.disagrees ? `${row.reference.text} (differs)` : row.reference.text;
      if (row.reference.title) span.title = row.reference.title;
      dd.append(span);
    }
    return [dt, dd];
  }), ...(reference ? [cadastreRow(reference)] : []).flat());
}

function cadastreRow(reference) {
  const dt = document.createElement("dt");
  dt.textContent = reference.label;
  const dd = document.createElement("dd");
  dd.className = "cadastre";
  dd.title = reference.title;
  const value = document.createElement(reference.url ? "a" : "span");
  value.className = "value";
  value.textContent = reference.value;
  if (reference.url) {
    value.href = reference.url;
    value.target = "_blank";
    value.rel = "noopener";
  }
  dd.append(value);
  return [dt, dd];
}

// Attributes written by city-analyser-survey or city-analyser-attributes, for the selected building.
async function renderBuildingCard() {
  const osmId = selectedBuildingId;
  const requestId = ++buildingCardRequest;
  const key = osmId && `${datasetId}|${osmId}`;
  elements.buildingAttributesButton.hidden = !osmId || dismissedBuildingCard !== osmId;
  setBuildingCardVisible(Boolean(osmId) && dismissedBuildingCard !== osmId);
  if (!osmId) {
    renderedBuildingCard = null;
    return;
  }
  if (renderedBuildingCard !== key) {
    // Reloading the same building (another panorama) keeps its card instead of flashing a loading state.
    closeReviewForm();
    currentBuildingCard = null;
    elements.buildingCardReview.hidden = true;
    elements.buildingCardReviewStatus.hidden = true;
    elements.buildingCardTitle.textContent = osmId;
    elements.buildingCardMessage.textContent = "Loading attributes...";
    elements.buildingCardAttributes.replaceChildren();
    elements.buildingCardWhy.hidden = true;
    elements.buildingCardWhy.open = false;
  }
  try {
    const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/building-attributes/${osmId}${REFERENCES_QUERY}`);
    if (requestId !== buildingCardRequest) return;
    renderedBuildingCard = key;
    if (response.status === 404) {
      currentBuildingCard = null;
      elements.buildingCardReview.hidden = true;
      elements.buildingCardReviewStatus.hidden = true;
      elements.buildingCardWhy.hidden = true;
      // The cadastre may already know what the survey has not described yet.
      const references = (await response.json().catch(() => ({}))).references || {};
      if (requestId !== buildingCardRequest) return;
      const referenceOnly = { attributes: {}, references };
      fillCardRows(attributeRows(referenceOnly), referenceRow(referenceOnly));
      elements.buildingCardMessage.textContent = "Not surveyed yet. Run city-analyser-survey on this dataset to describe its structure, floors and use.";
      return;
    }
    if (!response.ok) throw new Error(`status ${response.status}`);
    const card = await response.json();
    if (requestId !== buildingCardRequest) return;
    fillBuildingCard(card);
  } catch (error) {
    if (requestId !== buildingCardRequest) return;
    renderedBuildingCard = null;
    elements.buildingCardMessage.textContent = `Attributes unavailable: ${error.message}`;
  }
}

function fillSelect(select, options, value) {
  const cannotTell = document.createElement("option");
  cannotTell.value = "";
  cannotTell.textContent = "Cannot tell from the imagery";
  select.replaceChildren(cannotTell, ...options.map(([code, label]) => {
    const option = document.createElement("option");
    option.value = code;
    option.textContent = label;
    return option;
  }));
  select.value = value;
}

function openReviewForm() {
  if (!currentBuildingCard) return;
  const values = reviewFormValues(currentBuildingCard);
  const options = reviewOptions(currentBuildingCard);
  fillSelect(elements.reviewGemMaterial, options.gem_material, values.gem_material);
  fillSelect(elements.reviewGemLlrs, options.gem_llrs, values.gem_llrs);
  fillSelect(elements.reviewBuildingType, options.building_type, values.building_type);
  fillSelect(elements.reviewSurfaceMaterial, options.surface_material, values.surface_material);
  elements.reviewFloors.value = values.floors;
  elements.reviewConstructionYear.value = values.construction_year;
  elements.reviewEvaluation.checked = Boolean(currentBuildingCard.review?.evaluation);
  // The earlier note stays visible in the status line; a new decision gets its own.
  elements.reviewNote.value = "";
  elements.buildingCardReviewMessage.textContent = "Starts from what the card shows; change what the imagery contradicts.";
  elements.buildingCardReviewWithdraw.hidden = !currentBuildingCard.review;
  elements.buildingCardReviewForm.hidden = false;
  elements.buildingCard.classList.add("is-reviewing");
  elements.buildingCardAttributes.hidden = true;
  elements.buildingCardReview.hidden = true;
  setBuildingCardVisible(true);
  elements.reviewGemMaterial.focus();
}

function closeReviewForm() {
  elements.buildingCardReviewForm.hidden = true;
  elements.buildingCard.classList.remove("is-reviewing");
  elements.buildingCardAttributes.hidden = false;
  if (currentBuildingCard) elements.buildingCardReview.hidden = false;
}

async function sendAttributeReview(body, osmId = selectedBuildingId) {
  if (!osmId) return null;
  const reviewer = requireReviewerName();
  if (!reviewer) {
    elements.buildingCardReviewMessage.textContent = "Enter your name in the Review tab first; it is recorded with the review.";
    return null;
  }
  const card = await footprintRequest(`/api/datasets/${encodeURIComponent(datasetId)}/building-attributes/${osmId}${REFERENCES_QUERY}`, { ...body, reviewer });
  if (selectedBuildingId === osmId) {
    closeReviewForm();
    fillBuildingCard(card);
  }
  loadAttributeSummary();
  return card;
}

async function submitAttributeReview(event) {
  event.preventDefault();
  const number = (input) => (input.value.trim() === "" ? null : Number(input.value));
  // An empty field records that the imagery cannot tell, like the model's own abstentions.
  const values = {
    gem_material: elements.reviewGemMaterial.value || null,
    gem_llrs: elements.reviewGemLlrs.value || null,
    floors: number(elements.reviewFloors),
    building_type: elements.reviewBuildingType.value || null,
    surface_material: elements.reviewSurfaceMaterial.value || null,
    construction_year: number(elements.reviewConstructionYear),
  };
  try {
    const card = await sendAttributeReview({ action: "review", values, note: elements.reviewNote.value, evaluation: elements.reviewEvaluation.checked });
    if (card) showBuildingStatus(`Review saved: ${card.gem.string}`);
  } catch (error) {
    elements.buildingCardReviewMessage.textContent = `Review not saved: ${error.message}`;
  }
}

async function withdrawAttributeReview() {
  if (!window.confirm("Withdraw this review? It stays in the building's review history.")) return;
  try {
    const card = await sendAttributeReview({ action: "withdraw", note: elements.reviewNote.value });
    if (card) showBuildingStatus("Review withdrawn; the card shows the model and OSM again.");
  } catch (error) {
    elements.buildingCardReviewMessage.textContent = `Not withdrawn: ${error.message}`;
  }
}

// Confirm the values a queue item shows, without opening the form.
async function confirmAttributeValues(osmId, values) {
  const determined = Object.fromEntries(Object.entries(values).filter(([, value]) => value !== null && value !== undefined));
  if (!Object.keys(determined).length) {
    // Nothing to confirm (a split vote): the reviewer has to choose.
    reviewCurrentAction("edit");
    return;
  }
  try {
    const card = await sendAttributeReview({ action: "review", values: determined, note: "Confirmed from the review queue" }, osmId);
    if (card) elements.footprintReviewMessage.textContent = `Confirmed ${card.building}: ${card.gem.string}`;
  } catch (error) {
    elements.footprintReviewMessage.textContent = `Not confirmed: ${error.message}`;
  }
}

function dismissBuildingCard() {
  dismissedBuildingCard = selectedBuildingId;
  setBuildingCardVisible(false);
  elements.buildingAttributesButton.hidden = !selectedBuildingId;
}

function showBuildingCard() {
  dismissedBuildingCard = null;
  renderBuildingCard();
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
  // During a survey the map glides to the next capture with the camera, instead of jumping.
  locationMap.setView([latitude, longitude], 18, presentation ? { animate: true, duration: 0.8 } : { animate: false });
}

function storedFootprintAuthor() {
  try {
    return window.localStorage.getItem(FOOTPRINT_AUTHOR_KEY) || "";
  } catch {
    return "";
  }
}

function rememberFootprintAuthor(name) {
  try {
    window.localStorage.setItem(FOOTPRINT_AUTHOR_KEY, name);
  } catch {
    // Private windows may refuse storage; the name is only a convenience.
  }
}

function setFootprintEditMode(enabled) {
  footprintEditMode = enabled;
  elements.footprintEditButton.setAttribute("aria-pressed", String(enabled));
  elements.footprintEditButton.textContent = enabled ? "Stop editing" : "Edit a footprint on the map";
  if (enabled) elements.mapMessage.textContent = "Click the footprint to correct.";
  else endFootprintEdit();
}

function reviewerName() {
  return elements.reviewerName.value.trim();
}

function requireReviewerName() {
  const name = reviewerName();
  if (name) {
    rememberFootprintAuthor(name);
    return name;
  }
  openPanelTab("review");
  elements.footprintReviewMessage.textContent = "Enter your name first; it is recorded with proposals and reviews.";
  elements.reviewerName.focus();
  return null;
}

function beginFootprintEdit(feature) {
  endFootprintEdit();
  const parent = feature.properties?.parent_osm_id;
  if (parent) {
    elements.footprintReviewMessage.textContent = `${feature.properties.osm_id} is part of a reviewed split of ${parent}. To change it, propose a new revision of ${parent}; editing single parts is not supported yet.`;
    return;
  }
  if (!editableRings(feature.geometry)) {
    elements.mapMessage.textContent = "Footprints with several parts cannot be edited here yet.";
    return;
  }
  const analyzed = Object.values(currentAnalysis?.building_ids || {}).find((building) => building.osm_id === feature.properties.osm_id);
  projection = {
    original: feature.geometry,
    edited: [feature.geometry],
    height: Number.isFinite(analyzed?.render_height_m) ? analyzed.render_height_m : wallHeight(feature.properties),
  };
  footprintEditSession = startFootprintEdit(window.L, locationMap, feature.geometry, (geometries) => {
    projection = { ...projection, edited: geometries };
    render();
    // Undo steps are committed right after the change that triggered this.
    window.setTimeout(updateUndoButtons, 0);
  });
  footprintEditFeature = feature;
  setSplitMode(false);
  updateUndoButtons();
  const version = feature.properties?.osm_version;
  elements.footprintEditTarget.textContent = `Correcting ${feature.properties.osm_id}${version ? ` (OSM version ${version})` : ""}.`;
  elements.footprintNote.value = "";
  elements.footprintEditForm.hidden = false;
}

function endFootprintEdit() {
  footprintEditSession?.stop();
  footprintEditSession = null;
  footprintEditFeature = null;
  projection = null;
  elements.footprintEditForm.hidden = true;
  if (program) render();
}

function setSplitMode(enabled) {
  elements.footprintSplitButton.setAttribute("aria-pressed", String(enabled));
  elements.footprintSplitButton.textContent = enabled ? "Done splitting" : "Split";
  footprintEditSession?.setSplitting(enabled);
  if (enabled) elements.footprintReviewMessage.textContent = "Click two vertices of one part to cut it between them. Add vertices first with the edge dots if needed.";
}

function undoFootprintEdit(redo = false) {
  if (!footprintEditSession) return;
  if (redo) footprintEditSession.redo();
  else footprintEditSession.undo();
  updateUndoButtons();
}

async function footprintRequest(url, body) {
  const response = await fetch(url, body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : undefined);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

async function submitFootprintCorrection(event) {
  event.preventDefault();
  if (!footprintEditSession || !footprintEditFeature) return;
  const author = requireReviewerName();
  if (!author) return;
  const osmId = footprintEditFeature.properties.osm_id;
  const image = images[currentIndex];
  try {
    const geometries = footprintEditSession.geometries();
    const result = await footprintRequest(`/api/footprint-corrections/${osmId}`, {
      action: "propose",
      ...(geometries.length > 1 ? { parts: geometries } : { geometry: geometries[0] }),
      author,
      note: elements.footprintNote.value.trim(),
      evidence: image ? [{ dataset: datasetId, image: image.id }] : [],
    });
    setFootprintEditMode(false);
    elements.footprintReviewMessage.textContent = `Proposed revision ${result.revision.revision} for ${osmId}; it waits for review below.`;
    await refreshFootprintReview(false);
  } catch (error) {
    elements.footprintReviewMessage.textContent = `Correction not saved: ${error.message}`;
  }
}

function previewFootprint(geometry, color = "#2f80ed") {
  if (!locationMap || !footprintPreviewLayer) return;
  footprintPreviewLayer.clearLayers();
  const layer = window.L.geoJSON(geometry, { style: { color, fillColor: color, fillOpacity: 0.15, dashArray: "5 4", weight: 2 } }).addTo(footprintPreviewLayer);
  locationMap.fitBounds(layer.getBounds(), { maxZoom: 20, padding: [24, 24] });
}

async function reviewFootprintCorrection(osmId, revision, decision) {
  const reviewer = requireReviewerName();
  if (!reviewer) return;
  try {
    const result = await footprintRequest(`/api/footprint-corrections/${osmId}`, { action: "review", revision, decision, reviewer });
    footprintPreviewLayer?.clearLayers();
    elements.footprintReviewMessage.textContent = `Revision ${revision} of ${osmId} ${decision}.`;
    const reassociation = result.reassociation;
    if (reassociation?.job) {
      trackJob(reassociation.job, "Re-associate panoramas");
      elements.footprintReviewMessage.textContent = reassociation.job.message;
      await followFootprintJob(reassociation.job.id);
    } else if (reassociation?.message) {
      elements.footprintReviewMessage.textContent = reassociation.message;
    }
    // The queue keeps the reviewer's position, which now holds the next item.
    await refreshFootprintReview(false);
    if (reviewQueue[reviewIndex]) await openReviewItem(reviewQueue[reviewIndex]);
  } catch (error) {
    elements.footprintReviewMessage.textContent = `Review not saved: ${error.message}`;
  }
}

async function followFootprintJob(jobId) {
  for (;;) {
    const job = await footprintRequest(`/api/jobs/${encodeURIComponent(jobId)}`);
    elements.footprintReviewMessage.textContent = job.message;
    if (job.status === "failed") throw new Error(job.message);
    if (job.status === "complete") break;
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
  }
  // Footprints and analyses changed on the server: drop cached copies and reload.
  buildingCache.clear();
  loadCoverageStatus();
  const image = images[currentIndex];
  if (image) await Promise.all([loadBuildingAnalysis(image), updateMap(image)]);
}

function reviewFilters() {
  return {
    kind: elements.reviewKind.value,
    dataset: elements.reviewAllDatasets.checked ? null : datasetId,
    unresolvedOnly: elements.reviewHideResolved.checked,
  };
}

async function refreshFootprintReview(announce = true) {
  if (announce) elements.footprintReviewMessage.textContent = "Loading footprint review...";
  try {
    const issuesUrl = elements.reviewAllDatasets.checked || !datasetId ? "/api/footprint-issues" : `/api/footprint-issues?dataset=${encodeURIComponent(datasetId)}`;
    const [corrections, issues] = await Promise.all([footprintRequest("/api/footprint-corrections?status=proposed"), footprintRequest(issuesUrl)]);
    reviewData = { corrections: corrections.corrections || [], issues };
    rebuildReviewQueue();
    if (announce) elements.footprintReviewMessage.textContent = "";
  } catch (error) {
    elements.footprintReviewMessage.textContent = `Footprint review unavailable: ${error.message}`;
  }
}

function rebuildReviewQueue() {
  const previous = reviewQueue[reviewIndex];
  reviewQueue = buildReviewQueue(reviewData.corrections, reviewData.issues, reviewFilters(), attributeSummary?.dataset === datasetId ? attributeSummary : null);
  reviewIndex = preservedIndex(reviewQueue, previous?.key, reviewIndex);
  const pending = reviewData.corrections.reduce((total, record) => total + record.revisions.filter((revision) => revision.status === "proposed").length, 0);
  elements.reviewCount.textContent = String(pending);
  elements.reviewCount.hidden = pending === 0;
  renderReviewQueue();
}

function renderReviewQueue() {
  elements.reviewPosition.textContent = !reviewQueue.length ? "Nothing to review"
    : reviewIndex < 0 ? `${reviewQueue.length} to review` : `${reviewIndex + 1} of ${reviewQueue.length}`;
  elements.reviewPreviousButton.disabled = reviewQueue.length === 0;
  elements.reviewNextButton.disabled = reviewQueue.length === 0;
  const current = reviewQueue[reviewIndex];
  elements.reviewCurrent.hidden = !current;
  if (current) {
    elements.reviewCurrentKind.textContent = REVIEW_KINDS[current.kind] || current.kind;
    elements.reviewCurrentLabel.textContent = current.label;
    elements.reviewCurrentDetail.textContent = current.detail;
    const structure = current.kind === "attributes";
    elements.reviewAcceptButton.hidden = current.kind !== "proposal" && !structure;
    elements.reviewAcceptButton.textContent = structure ? "Confirm" : "Accept";
    elements.reviewAcceptButton.title = structure ? "Confirm the values shown as your review" : "";
    elements.reviewRejectButton.hidden = current.kind !== "proposal";
    elements.reviewEditButton.hidden = !current.osmId;
    // Reviewing a structure needs no footprint editor, so it also works on a phone.
    elements.reviewEditButton.classList.toggle("needs-large-screen", !structure);
    elements.reviewEditButton.textContent = structure ? "Review" : "Edit footprint";
  }
  elements.reviewList.replaceChildren(...reviewQueue.map((item, index) => {
    const entry = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-current", String(index === reviewIndex));
    const kind = document.createElement("span");
    kind.className = "review-kind";
    kind.textContent = REVIEW_KINDS[item.kind] || item.kind;
    const label = document.createElement("strong");
    label.textContent = item.label;
    const detail = document.createElement("span");
    detail.textContent = item.detail;
    button.append(kind, label, detail);
    button.addEventListener("click", () => goToReviewItem(index));
    entry.append(button);
    return entry;
  }));
  elements.reviewList.querySelector('[aria-current="true"]')?.scrollIntoView({ block: "nearest" });
}

async function goToReviewItem(index) {
  if (index < 0 || !reviewQueue[index]) return;
  reviewIndex = index;
  renderReviewQueue();
  await openReviewItem(reviewQueue[index]);
  syncViewerUrl();
}

function stepReview(step) {
  return goToReviewItem(stepIndex(reviewQueue.length, reviewIndex, step));
}

function featureById(osmId) {
  let feature = null;
  buildingLayer?.eachLayer((layer) => {
    if (layer.feature?.properties?.osm_id === osmId) feature = layer.feature;
  });
  return feature;
}

// Open the panorama that shows a review item, look at it and mark it on the map.
async function openReviewItem(item) {
  if (!item) return;
  endFootprintEdit();
  if (item.dataset && item.image) await showPanorama(item.dataset, item.image);
  else if (item.kind === "attributes") {
    await goToBestView(item.osmId);
    return;
  }
  if (item.osmId) {
    selectBuilding(item.osmId, true, true);
    const feature = featureById(item.osmId);
    if (item.geometry) previewFootprint(item.geometry);
    else if (feature) previewFootprint(feature.geometry, "#c0392b");
  } else if (item.bearings) {
    footprintPreviewLayer?.clearLayers();
    lookAtBearing(bearingCenter(item.bearings));
  }
}

function lookAtBearing(bearing) {
  const image = images[currentIndex];
  if (!image) return;
  const analysisPose = currentAnalysis && currentAnalysis.image_id === image.id ? currentAnalysis.refined_pose : null;
  const heading = Number.isFinite(analysisPose?.heading_degrees) ? analysisPose.heading_degrees : recordedHeading(image);
  if (heading === null) return;
  yaw = signedDegrees(bearing - heading) * Math.PI / 180;
  render();
  scheduleViewIndicator();
}

async function showPanorama(dataset, imageId) {
  if (dataset !== datasetId) {
    if (!datasets.some((item) => item.id === dataset)) return;
    elements.datasetSelect.value = dataset;
    await loadDataset(dataset, { imageId });
    return;
  }
  const index = images.findIndex((image) => String(image.id) === String(imageId));
  if (index >= 0 && index !== currentIndex) await selectImage(index);
}

async function openLinkedReviewItem(key) {
  openPanelTab("review");
  await refreshFootprintReview(false);
  let index = reviewQueue.findIndex((item) => item.key === key);
  if (index < 0 && !elements.reviewAllDatasets.checked) {
    elements.reviewAllDatasets.checked = true;
    await refreshFootprintReview(false);
    index = reviewQueue.findIndex((item) => item.key === key);
  }
  if (index >= 0) await goToReviewItem(index);
  else elements.footprintReviewMessage.textContent = "The linked review item is resolved or no longer exists.";
}

function reviewCurrentAction(action) {
  const current = reviewQueue[reviewIndex];
  if (!current) return;
  if (current.kind === "attributes") {
    if (action === "accept") confirmAttributeValues(current.osmId, current.values);
    if (action === "edit") {
      if (currentBuildingCard?.building === current.osmId && selectedBuildingId === current.osmId) {
        openReviewForm();
      } else {
        pendingReviewForm = current.osmId;
        if (selectedBuildingId !== current.osmId) openReviewItem(current);
      }
    }
    if (action === "show") openReviewItem(current);
    return;
  }
  if (action === "accept" || action === "reject") {
    if (current.kind === "proposal") reviewFootprintCorrection(current.osmId, current.revision, action === "accept" ? "accepted" : "rejected");
    return;
  }
  if (action === "edit" && current.osmId && !PHONE_QUERY.matches) {
    const feature = featureById(current.osmId);
    if (feature) {
      setFootprintEditMode(true);
      beginFootprintEdit(feature);
    }
  }
  if (action === "show") openReviewItem(current);
}

async function loadBuildingAnalysis(image) {
  const requestId = ++analysisRequest;
  const requestedDataset = datasetId;
  const requestedImageId = image.id;
  maskLoadRequest += 1;
  currentAnalysis = null;
  setBuildingAnalysisRunning(analysisJobRunning);
  clearSelectionMask();
  clearInstanceOverlay();
  renderDiagnostics();
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
    loadInstanceOverlay();
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
    loadCoverageStatus();
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

async function startBatchAnalysis(useMapView) {
  if (!datasetId) return;
  let body = {};
  if (useMapView && locationMap) {
    const bounds = locationMap.getBounds();
    body = { bbox: [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()] };
  }
  try {
    const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/analysis-batch`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `The analysis could not start (${response.status}).`);
    const job = result.job;
    if (!job.total) {
      showBuildingStatus(useMapView ? "Every panorama in the map view is already analyzed." : "Every panorama in this dataset is already analyzed.");
      return;
    }
    const requestedDataset = datasetId;
    trackJob(job, useMapView ? "Analyze map view" : `Analyze ${datasetId}`, async () => {
      if (datasetId !== requestedDataset) return;
      await loadCoverageStatus();
      if (images[currentIndex]) await loadBuildingAnalysis(images[currentIndex]);
      refreshFootprintReview(false);
    });
    openPanelTab("jobs");
    showBuildingStatus(job.message);
  } catch (error) {
    showBuildingStatus(`Analysis could not start: ${error.message}`);
  }
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
    trackJob(result.job, "Analyze buildings");
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
    trackJob(result.job, "Sparse depth (SfM)");
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
  direction = applyMatrix(panoramaLevel, rotateYaw(rotatePitch(direction, pitch), yaw));
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
    const status = coverageStatus[String(image.id)];
    if (status) {
      const dot = document.createElement("span");
      dot.className = "status-dot";
      dot.dataset.status = status;
      dot.title = { current: "Analysis current", stale: "Analysis from an older version", missing: "Not analyzed" }[status];
      indexElement.append(dot);
    }
    indexElement.append(String(index + 1).padStart(2, "0"));
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
  elements.analyzeDatasetButton.disabled = !hasImages;
  elements.analyzeMapViewButton.disabled = !hasImages;
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
  // Callers that need the building data wait for both; the texture loads independently.
  const ready = Promise.all([updateMap(image), loadBuildingAnalysis(image)]);
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
      panoramaLevel = levelMatrix(image.camera_up);
      const texture = uploadTexture(textureImage);
      render();
      textureWaiters.splice(0).forEach((resolve) => resolve());
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
  return ready;
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
    renderBuildingCard();
    attributeSummary = null;
    renderMapLegend();
    loadAttributeSummary();
    const ready = selectImage(requestedIndex);
    refreshFootprintReview(false);
    loadCoverageStatus();
    await ready;
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
  // Parts of a split have a third segment ("way/1/part-2").
  const buildingPath = selectedBuildingId.split("/").map(encodeURIComponent).join("/");
  const popup = window.open("", "_blank");
  elements.generateReportButton.disabled = true;
  elements.generateReportButton.textContent = "Generating report...";
  showBuildingStatus("Generating ranked views, annotated screenshots, and footprint assessment...");
  try {
    const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/building-reports/${buildingPath}`, { method: "POST" });
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
  const touches = new Map();
  let pinch = null;
  const pinchDistance = () => {
    const [first, second] = [...touches.values()];
    return Math.hypot(first.x - second.x, first.y - second.y);
  };
  elements.canvas.addEventListener("pointerdown", (event) => {
    touches.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (touches.size === 2) {
      // A second finger turns the drag into a pinch.
      pinch = { distance: pinchDistance(), fov };
      pointer = null;
      return;
    }
    pointer = { id: event.pointerId, moved: false, startX: event.clientX, startY: event.clientY, x: event.clientX, y: event.clientY };
    elements.canvas.setPointerCapture(event.pointerId);
    elements.canvas.classList.add("dragging");
  });
  elements.canvas.addEventListener("pointermove", (event) => {
    if (touches.has(event.pointerId)) touches.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pinch && touches.size === 2) {
      fov = Math.max(45, Math.min(110, pinch.fov * pinch.distance / Math.max(pinchDistance(), 1)));
      render();
      scheduleViewIndicator();
      return;
    }
    if (!pointer || event.pointerId !== pointer.id) return;
    // Drag the panorama itself: down looks up, right looks left.
    yaw -= (event.clientX - pointer.x) * 0.006;
    pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, pitch - (event.clientY - pointer.y) * 0.006));
    pointer = { ...pointer, moved: pointer.moved || Math.hypot(event.clientX - pointer.startX, event.clientY - pointer.startY) > 3, x: event.clientX, y: event.clientY };
    render();
    scheduleViewIndicator();
  });
  const endPointer = (event, cancelled = false) => {
    touches.delete(event.pointerId);
    if (touches.size < 2) pinch = null;
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
  // A menu closes once one of its actions is chosen.
  document.querySelectorAll(".advanced-actions-menu button").forEach((button) => {
    button.addEventListener("click", () => { button.closest("details").open = false; });
  });
  elements.analyzeDatasetButton.addEventListener("click", () => startBatchAnalysis(false));
  elements.analyzeMapViewButton.addEventListener("click", () => startBatchAnalysis(true));
  elements.generateReportButton.addEventListener("click", generateBuildingReport);
  elements.buildingCardClose.addEventListener("click", dismissBuildingCard);
  elements.buildingCardMore.addEventListener("click", () => openPanelTab("building"));
  elements.buildingAttributesButton.addEventListener("click", showBuildingCard);
  elements.buildingCardReview.addEventListener("click", openReviewForm);
  elements.assistantForm.addEventListener("submit", sendChat);
  elements.assistantInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      elements.assistantForm.requestSubmit();
    }
  });
  elements.assistantAnalyzeButton.addEventListener("click", analyzeNow);
  elements.assistantEditButton.addEventListener("click", editValues);
  elements.presentationButton.addEventListener("click", () => (presentation ? presentation.stop() : startPresentationMode()));
  elements.buildingCardReviewCancel.addEventListener("click", closeReviewForm);
  elements.buildingCardReviewWithdraw.addEventListener("click", withdrawAttributeReview);
  elements.buildingCardReviewForm.addEventListener("submit", submitAttributeReview);
  elements.attributeColorSelect.replaceChildren(...Object.entries(COLOR_MODES).map(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
  }));
  elements.attributeColorSelect.value = attributeColorMode();
  elements.attributeColorSelect.addEventListener("change", () => setAttributeColorMode(elements.attributeColorSelect.value));
  elements.footprintEditButton.addEventListener("click", () => setFootprintEditMode(!footprintEditMode));
  elements.footprintCancelButton.addEventListener("click", () => setFootprintEditMode(false));
  elements.footprintEditForm.addEventListener("submit", submitFootprintCorrection);
  elements.footprintRefreshButton.addEventListener("click", () => refreshFootprintReview());
  elements.footprintUndoButton.addEventListener("click", () => undoFootprintEdit());
  elements.footprintSplitButton.addEventListener("click", () => setSplitMode(elements.footprintSplitButton.getAttribute("aria-pressed") !== "true"));
  elements.footprintRedoButton.addEventListener("click", () => undoFootprintEdit(true));
  elements.reviewerName.value = storedFootprintAuthor();
  elements.reviewerName.addEventListener("change", () => rememberFootprintAuthor(reviewerName()));
  elements.reviewKind.addEventListener("change", rebuildReviewQueue);
  elements.reviewHideResolved.addEventListener("change", rebuildReviewQueue);
  elements.reviewAllDatasets.addEventListener("change", () => refreshFootprintReview(false));
  elements.reviewPreviousButton.addEventListener("click", () => stepReview(-1));
  elements.reviewNextButton.addEventListener("click", () => stepReview(1));
  elements.reviewAcceptButton.addEventListener("click", () => reviewCurrentAction("accept"));
  elements.reviewRejectButton.addEventListener("click", () => reviewCurrentAction("reject"));
  elements.reviewEditButton.addEventListener("click", () => reviewCurrentAction("edit"));
  elements.reviewShowButton.addEventListener("click", () => reviewCurrentAction("show"));
  elements.buildingEditButton.addEventListener("click", editSelectedFootprint);
  elements.buildingRevertButton.addEventListener("click", revertSelectedCorrection);
  elements.buildingBestViewButton.addEventListener("click", async () => {
    if (selectedBuildingId) showBuildingStatus(await goToBestView(selectedBuildingId));
  });
  elements.searchForm.addEventListener("submit", runSearch);
  elements.panelToggleButton.addEventListener("click", () => setSidePanelVisible(elements.sidePanel.hidden));
  elements.focusButton.addEventListener("click", () => setFocus(!isFocused()));
  elements.panelCloseButton.addEventListener("click", () => setSidePanelVisible(false));
  elements.tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => setPanelTab(PANEL_TABS[index]));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      event.preventDefault();
      event.stopPropagation();
      const step = event.key === "ArrowRight" ? 1 : -1;
      setPanelTab(PANEL_TABS[(index + step + PANEL_TABS.length) % PANEL_TABS.length], true);
    });
  });
  elements.layerInputs.forEach((input) => input.addEventListener("change", () => setLayer(input.dataset.layer, input.checked)));
  elements.closeMapillaryDialogButton.addEventListener("click", () => elements.mapillaryDialog.close());
  elements.helpButton.addEventListener("click", () => elements.helpDialog.showModal());
  elements.closeHelpDialogButton.addEventListener("click", () => elements.helpDialog.close());
  elements.themeSelect.addEventListener("change", () => {
    applyTheme(elements.themeSelect.value);
    savePreference("theme", elements.themeSelect.value);
  });
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
    if (footprintEditSession && (event.ctrlKey || event.metaKey) && ["z", "y"].includes(event.key.toLowerCase())) {
      event.preventDefault();
      undoFootprintEdit(event.key.toLowerCase() === "y" || event.shiftKey);
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    // Review shortcuts apply while the Review tab is open.
    if (!elements.sidePanel.hidden && currentPanelTab() === "review") {
      const action = { n: () => stepReview(1), p: () => stepReview(-1), a: () => reviewCurrentAction("accept"), x: () => reviewCurrentAction("reject"), e: () => reviewCurrentAction("edit") }[event.key.toLowerCase()];
      if (action) {
        event.preventDefault();
        action();
        return;
      }
    }
    if (event.key === "Escape" && !elements.buildingCard.hidden && !document.querySelector("dialog[open]")) {
      dismissBuildingCard();
      return;
    }
    if (event.key === "ArrowLeft") { event.preventDefault(); selectImage(currentIndex - 1); }
    if (event.key === "ArrowRight") { event.preventDefault(); selectImage(currentIndex + 1); }
    if (event.key.toLowerCase() === "r") resetView();
    if (event.key.toLowerCase() === "m") setLayer("masks", !layers.masks);
    if (event.key.toLowerCase() === "f") setFocus(!isFocused());
    if (event.key === "?") {
      event.preventDefault();
      elements.helpDialog.showModal();
    }
  });
}

async function start() {
  // Read before loading: loading a dataset rewrites the URL.
  const presentRequested = new URLSearchParams(window.location.search).get("present") === "1";
  try {
    initializeRenderer();
    bindInteraction();
    const saved = preferences();
    applyTheme(saved.theme);
    setSidePanelVisible(!OVERLAY_PANEL_QUERY.matches && saved.panel !== false, false);
    if (saved.focus === true) setFocus(true, false);
    if (PANEL_TABS.includes(saved.tab) && saved.tab !== "building") setPanelTab(saved.tab);
    updateMaskToggle();
    if (PHONE_QUERY.matches || saved.map === false) setMapPanelVisible(false, false);
    setActivity(elements.loading, true);
    await loadAnalysisBackend();
    // Awaited: a presentation started from the URL needs to know whether it can run live.
    await loadAssistantStatus();
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
    if (target.reviewKey) await openLinkedReviewItem(target.reviewKey);
    if (presentRequested) startPresentationMode();
  } catch (error) {
    showError(`${error.message} Run uv run city-analyser-viewer and open http://127.0.0.1:8765/viewer/.`);
  }
}

start();
