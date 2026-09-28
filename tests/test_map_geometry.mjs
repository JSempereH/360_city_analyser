import assert from "node:assert/strict";
import test from "node:test";

import { coordinatesOf, mergeBuildingFootprints, outerRingsOf } from "../src/city_analyser/viewer/map-geometry.mjs";

test("uses the recorded Mapillary coordinates", () => {
  const image = {
    selected_geometry: { coordinates: [4.9, 52.3] },
    computed_geometry: { coordinates: [4.8, 52.2] },
  };

  assert.deepEqual(coordinatesOf(image), [4.9, 52.3]);
});

test("adds analyzed footprints that are missing from the base map", () => {
  const base = [{
    type: "Feature",
    properties: { osm_id: "way/1", name: "OSM building" },
    geometry: { type: "Polygon", coordinates: [[[0, 0], [1, 0], [1, 1], [0, 0]]] },
  }];
  const analyzedGeometry = { type: "Polygon", coordinates: [[[2, 2], [3, 2], [3, 3], [2, 2]]] };

  const merged = mergeBuildingFootprints(base, {
    1: { osm_id: "way/2", footprint_source: "osm", footprint_geometry: analyzedGeometry },
  });

  assert.equal(merged.length, 2);
  assert.deepEqual(merged[1], {
    type: "Feature",
    properties: { osm_id: "way/2", footprint_source: "osm" },
    geometry: analyzedGeometry,
  });
});

test("does not add non-OSM analyzed footprints", () => {
  const merged = mergeBuildingFootprints([], {
    1: {
      osm_id: "microsoft/120202110-84625",
      footprint_geometry: { type: "Polygon", coordinates: [[[0, 0], [1, 0], [1, 1], [0, 0]]] },
    },
  });

  assert.deepEqual(merged, []);
});

test("uses the exact analyzed geometry for an existing footprint", () => {
  const base = [{
    type: "Feature",
    properties: { osm_id: "way/1", name: "OSM building" },
    geometry: { type: "Polygon", coordinates: [[[0, 0], [1, 0], [1, 1], [0, 0]]] },
  }];
  const analyzedGeometry = { type: "Polygon", coordinates: [[[0, 0], [2, 0], [2, 2], [0, 0]]] };

  const [merged] = mergeBuildingFootprints(base, {
    1: { osm_id: "way/1", footprint_source: "osm", footprint_geometry: analyzedGeometry },
  });

  assert.equal(merged.properties.name, "OSM building");
  assert.equal(merged.properties.footprint_source, "osm");
  assert.equal(merged.geometry, analyzedGeometry);
});

test("returns every exterior ring without including polygon holes", () => {
  const firstOuter = [[0, 0], [2, 0], [2, 2], [0, 0]];
  const firstHole = [[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 0.5]];
  const secondOuter = [[4, 4], [5, 4], [5, 5], [4, 4]];
  const geometry = {
    type: "MultiPolygon",
    coordinates: [[firstOuter, firstHole], [secondOuter]],
  };

  assert.deepEqual(outerRingsOf({ type: "Polygon", coordinates: [firstOuter, firstHole] }), [firstOuter]);
  assert.deepEqual(outerRingsOf(geometry), [firstOuter, secondOuter]);
});

test("keeps a relation MultiPolygon and all of its holes", () => {
  const geometry = {
    type: "MultiPolygon",
    coordinates: [
      [
        [[0, 0], [3, 0], [3, 3], [0, 0]],
        [[1, 1], [2, 1], [2, 2], [1, 1]],
      ],
      [[[5, 5], [6, 5], [6, 6], [5, 5]]],
    ],
  };

  const [merged] = mergeBuildingFootprints([], {
    building: { osm_id: "relation/42", footprint_source: "osm", footprint_geometry: geometry },
  });

  assert.equal(merged.properties.osm_id, "relation/42");
  assert.equal(merged.geometry, geometry);
  assert.deepEqual(merged.geometry.coordinates, geometry.coordinates);
});
