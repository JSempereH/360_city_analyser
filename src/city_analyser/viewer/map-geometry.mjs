const EARTH_RADIUS_M = 6_371_008.8;
// OSM elements and parts of a reviewed split ("way/123/part-2").
const OSM_BUILDING_ID = /^(?:node|way|relation)\/[1-9][0-9]*(?:\/part-[1-9][0-9]*)?$/;

export function coordinatesOf(image) {
  return image?.selected_geometry?.coordinates || image?.computed_geometry?.coordinates || image?.geometry?.coordinates || null;
}

export function outerRingsOf(geometry) {
  if (geometry?.type === "Polygon") {
    return Array.isArray(geometry.coordinates?.[0]) ? [geometry.coordinates[0]] : [];
  }
  if (geometry?.type === "MultiPolygon") {
    return (Array.isArray(geometry.coordinates) ? geometry.coordinates : [])
      .map((polygon) => polygon?.[0])
      .filter(Array.isArray);
  }
  return [];
}

export function mergeBuildingFootprints(baseFeatures, analyzedBuildings) {
  const merged = new Map();
  (Array.isArray(baseFeatures) ? baseFeatures : []).forEach((feature) => {
    const identifier = feature?.properties?.osm_id;
    if (typeof identifier === "string") merged.set(identifier, feature);
  });
  const buildings = Array.isArray(analyzedBuildings) ? analyzedBuildings : Object.values(analyzedBuildings || {});
  buildings.forEach((building) => {
    const identifier = building?.osm_id;
    const geometry = building?.footprint_geometry;
    const outerRings = outerRingsOf(geometry);
    if (!OSM_BUILDING_ID.test(identifier) || !outerRings.length || outerRings.some((ring) => ring.length < 4)) return;
    const existing = merged.get(identifier);
    const properties = {
      ...(existing?.properties || {}),
      osm_id: identifier,
      footprint_source: building.footprint_source || existing?.properties?.footprint_source || "osm",
    };
    // The analysis geometry wins, so its correction state must win too.
    if ("footprint_correction" in building) {
      if (building.footprint_correction) properties.footprint_correction = building.footprint_correction;
      else delete properties.footprint_correction;
    }
    merged.set(identifier, { type: "Feature", properties, geometry });
  });
  return [...merged.values()];
}

export function normalizeDegrees(angle) {
  return ((angle % 360) + 360) % 360;
}

export function destinationPoint(latitude, longitude, bearingDegrees, distanceMeters) {
  const bearing = bearingDegrees * Math.PI / 180;
  const latitudeRadians = latitude * Math.PI / 180;
  const longitudeRadians = longitude * Math.PI / 180;
  const distance = distanceMeters / EARTH_RADIUS_M;
  const targetLatitude = Math.asin(Math.sin(latitudeRadians) * Math.cos(distance) + Math.cos(latitudeRadians) * Math.sin(distance) * Math.cos(bearing));
  const targetLongitude = longitudeRadians + Math.atan2(Math.sin(bearing) * Math.sin(distance) * Math.cos(latitudeRadians), Math.cos(distance) - Math.sin(latitudeRadians) * Math.sin(targetLatitude));
  return [targetLatitude * 180 / Math.PI, targetLongitude * 180 / Math.PI];
}

export function segmentIntersectionFraction(start, end, first, second) {
  const latitudeScale = Math.cos(start[0] * Math.PI / 180);
  const point = ([latitude, longitude]) => [longitude * latitudeScale, latitude];
  const [startX, startY] = point(start);
  const [endX, endY] = point(end);
  const [firstX, firstY] = point(first);
  const [secondX, secondY] = point(second);
  const rayX = endX - startX;
  const rayY = endY - startY;
  const edgeX = secondX - firstX;
  const edgeY = secondY - firstY;
  const denominator = rayX * edgeY - rayY * edgeX;
  if (Math.abs(denominator) < 1e-12) return null;
  const offsetX = firstX - startX;
  const offsetY = firstY - startY;
  const rayFraction = (offsetX * edgeY - offsetY * edgeX) / denominator;
  const edgeFraction = (offsetX * rayY - offsetY * rayX) / denominator;
  return rayFraction > 0.02 && rayFraction < 1 && edgeFraction >= 0 && edgeFraction <= 1 ? rayFraction : null;
}

export function sortedLocalPanoramas(manifest) {
  return [...(manifest?.images || [])].filter((image) => image.is_pano && image.local_file).sort((first, second) => {
    const byCapture = (first.captured_at_ms || 0) - (second.captured_at_ms || 0);
    return byCapture || String(first.id).localeCompare(String(second.id));
  });
}

// Index of the panorama nearest to a [longitude, latitude] point, or -1.
export function nearestImageIndex(images, point, distance) {
  let best = -1;
  let bestDistance = Infinity;
  images.forEach((image, index) => {
    const coordinates = coordinatesOf(image);
    if (!coordinates) return;
    const meters = distance(coordinates, point);
    if (meters < bestDistance) {
      best = index;
      bestDistance = meters;
    }
  });
  return best;
}

export function geometryCentroid(geometry) {
  const vertices = outerRingsOf(geometry).flatMap((ring) => ring.slice(0, -1));
  if (!vertices.length) return null;
  const [longitude, latitude] = vertices.reduce((total, point) => [total[0] + point[0], total[1] + point[1]], [0, 0]);
  return [longitude / vertices.length, latitude / vertices.length];
}
