import { normalizeDegrees } from "./map-geometry.mjs";

export const MAX_NEARBY_PANORAMA_DISTANCE_M = 30;

const EARTH_RADIUS_M = 6_371_008.8;

export function signedDegrees(angle) {
  return ((angle + 540) % 360) - 180;
}

export function distanceMeters(first, second) {
  const [firstLongitude, firstLatitude] = first.map((coordinate) => coordinate * Math.PI / 180);
  const [secondLongitude, secondLatitude] = second.map((coordinate) => coordinate * Math.PI / 180);
  const latitudeDelta = secondLatitude - firstLatitude;
  const longitudeDelta = secondLongitude - firstLongitude;
  const halfChord = Math.sin(latitudeDelta / 2) ** 2
    + Math.cos(firstLatitude) * Math.cos(secondLatitude) * Math.sin(longitudeDelta / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.asin(Math.sqrt(halfChord));
}

export function bearingDegrees(first, second) {
  const [firstLongitude, firstLatitude] = first.map((coordinate) => coordinate * Math.PI / 180);
  const [secondLongitude, secondLatitude] = second.map((coordinate) => coordinate * Math.PI / 180);
  const longitudeDelta = secondLongitude - firstLongitude;
  return normalizeDegrees(Math.atan2(
    Math.sin(longitudeDelta) * Math.cos(secondLatitude),
    Math.cos(firstLatitude) * Math.sin(secondLatitude)
      - Math.sin(firstLatitude) * Math.cos(secondLatitude) * Math.cos(longitudeDelta),
  ) * 180 / Math.PI);
}

export function nearbyPanoramas(panoramas, currentIndex, maximumDistance = MAX_NEARBY_PANORAMA_DISTANCE_M) {
  const current = panoramas[currentIndex];
  if (!current?.coordinates) return [];
  const targets = [];
  panoramas.forEach((panorama, index) => {
    if (index === currentIndex || !panorama.coordinates) return;
    if (current.sequenceId && panorama.sequenceId && current.sequenceId !== panorama.sequenceId) return;
    const distance = distanceMeters(current.coordinates, panorama.coordinates);
    if (distance < 1 || distance > maximumDistance) return;
    const bearing = bearingDegrees(current.coordinates, panorama.coordinates);
    targets.push({ bearing, distance, index });
  });
  return targets.sort((first, second) => first.distance - second.distance || first.index - second.index);
}

export function nearestPanoramasByBearing(targets, minimumSeparation = 10) {
  const selected = [];
  [...targets].sort((first, second) => first.distance - second.distance || first.index - second.index).forEach((target) => {
    if (selected.every((other) => Math.abs(signedDegrees(target.bearing - other.bearing)) >= minimumSeparation)) {
      selected.push(target);
    }
  });
  return selected;
}

export function projectGroundPanoramaToView(bearing, distance, viewHeading, pitch, horizontalFov, aspect, cameraHeight = 2.1) {
  const relativeBearing = signedDegrees(bearing - viewHeading) * Math.PI / 180;
  const horizontalDepth = Math.cos(relativeBearing) * distance;
  const x = Math.sin(relativeBearing) * distance;
  // The manifests have no elevation or camera height. Treat each target as a
  // ground point on a flat plane below an assumed camera height.
  const y = Math.cos(pitch) * -cameraHeight + Math.sin(pitch) * horizontalDepth;
  const z = Math.sin(pitch) * cameraHeight + Math.cos(pitch) * horizontalDepth;
  if (z <= 0) return null;
  const horizontalField = Math.tan(horizontalFov * Math.PI / 360);
  const verticalField = horizontalField / aspect;
  const normalizedX = x / z / horizontalField;
  const normalizedY = y / z / verticalField;
  if (Math.abs(normalizedX) > 1 || Math.abs(normalizedY) > 1) return null;
  return { left: (normalizedX + 1) * 50, top: (1 - normalizedY) * 50 };
}
