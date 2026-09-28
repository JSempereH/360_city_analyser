const EARTH_RADIUS_M = 6_371_008.8;
const OSM_BUILDING_ID = /^(?:node|way|relation)\/[1-9][0-9]*$/;

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
    merged.set(identifier, {
      type: "Feature",
      properties: {
        ...(existing?.properties || {}),
        osm_id: identifier,
        footprint_source: building.footprint_source || existing?.properties?.footprint_source || "osm",
      },
      geometry,
    });
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
