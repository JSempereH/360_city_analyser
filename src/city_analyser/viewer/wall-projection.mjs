// Project footprint walls into the current panorama view, so an edited
// outline can be aligned with the real facade while it is dragged.
//
// The view math is the inverse of the panorama shader in app.js: the shader
// turns a screen ray into a panorama direction by pitching, then yawing; here a
// world direction is yawed back, pitched back and divided by depth.

const EARTH_RADIUS_M = 6371008.8;
// Matches building_analysis.CAMERA_HEIGHT_M, the height the analysis projects from.
export const CAMERA_HEIGHT_M = 1.7;
export const DEFAULT_WALL_HEIGHT_M = 10;
const SAMPLES_PER_EDGE = 16;

// East/north meters of a [longitude, latitude] point from the camera.
export function localMeters(camera, point) {
  const latitude = camera[1] * Math.PI / 180;
  return [
    (point[0] - camera[0]) * Math.PI / 180 * Math.cos(latitude) * EARTH_RADIUS_M,
    (point[1] - camera[1]) * Math.PI / 180 * EARTH_RADIUS_M,
  ];
}

// Screen position (pixels) of a point at ``height`` meters above the ground,
// or null when it is behind the viewer. ``view`` holds the panorama heading
// (degrees of the image center), yaw and pitch (radians, as in app.js),
// horizontal fov (degrees) and the canvas size.
export function projectPoint(camera, point, height, view) {
  const [east, north] = localMeters(camera, point);
  const distance = Math.hypot(east, north);
  const azimuth = Math.atan2(east, north) - view.headingDeg * Math.PI / 180;
  const elevation = Math.atan2(height - CAMERA_HEIGHT_M, distance);
  // Panorama frame: z forward (image center), x right, y up.
  let x = Math.cos(elevation) * Math.sin(azimuth);
  let y = Math.sin(elevation);
  let z = Math.cos(elevation) * Math.cos(azimuth);
  // Undo the shader's yaw, then its pitch.
  const cosYaw = Math.cos(-view.yaw);
  const sinYaw = Math.sin(-view.yaw);
  [x, z] = [cosYaw * x + sinYaw * z, -sinYaw * x + cosYaw * z];
  const cosPitch = Math.cos(-view.pitch);
  const sinPitch = Math.sin(-view.pitch);
  [y, z] = [cosPitch * y - sinPitch * z, sinPitch * y + cosPitch * z];
  if (z <= 1e-3) return null;
  const aspect = view.width / view.height;
  const field = Math.tan(view.fovDeg * Math.PI / 360) / aspect;
  const screenX = x / z / (aspect * field);
  const screenY = y / z / field;
  return [(screenX + 1) / 2 * view.width, (1 - screenY) / 2 * view.height];
}

// Walls whose outer face looks toward the camera; the others are hidden
// behind the building and would only clutter the guide.
export function facingWalls(camera, points) {
  const local = points.map((point) => localMeters(camera, point));
  const area = local.reduce((total, [x, y], index) => {
    const [nextX, nextY] = local[(index + 1) % local.length];
    return total + x * nextY - nextX * y;
  }, 0);
  return local.map(([ax, ay], index) => {
    const [bx, by] = local[(index + 1) % local.length];
    // Camera (the origin) to the right of the edge means outside a counter-clockwise ring.
    const cross = (bx - ax) * (0 - ay) - (by - ay) * (0 - ax);
    return area > 0 ? cross < 0 : cross > 0;
  });
}

// Polylines (lists of screen points) for the base, top and corners of every
// camera-facing wall of a ring, split wherever a wall passes behind the viewer.
export function wallPolylines(camera, ring, height, view) {
  const lines = [];
  const points = ring.length > 1 && ring[0][0] === ring.at(-1)[0] && ring[0][1] === ring.at(-1)[1] ? ring.slice(0, -1) : ring;
  const facing = facingWalls(camera, points);
  points.forEach((start, index) => {
    if (!facing[index]) return;
    const end = points[(index + 1) % points.length];
    for (const level of [0, height]) {
      let current = [];
      for (let step = 0; step <= SAMPLES_PER_EDGE; step += 1) {
        const fraction = step / SAMPLES_PER_EDGE;
        const projected = projectPoint(camera, [start[0] + (end[0] - start[0]) * fraction, start[1] + (end[1] - start[1]) * fraction], level, view);
        if (projected) current.push(projected);
        else if (current.length) {
          if (current.length > 1) lines.push(current);
          current = [];
        }
      }
      if (current.length > 1) lines.push(current);
    }
    for (const corner of [start, end]) {
      const base = projectPoint(camera, corner, 0, view);
      const top = projectPoint(camera, corner, height, view);
      if (base && top) lines.push([base, top]);
    }
  });
  return lines;
}

// Wall height the analysis would use for a footprint's OSM tags.
export function wallHeight(properties) {
  const height = Number.parseFloat(String(properties?.height ?? "").replace(",", "."));
  if (Number.isFinite(height) && height > 0) return height;
  const levels = Number.parseFloat(String(properties?.["building:levels"] ?? ""));
  if (Number.isFinite(levels) && levels > 0) return Math.max(3, levels * 3.2);
  return DEFAULT_WALL_HEIGHT_M;
}
