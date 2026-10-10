// Footprint editing on the mini-map: drag vertices, move the whole polygon,
// insert a vertex from an edge midpoint, double-click a vertex to remove it.
// Validity (self-intersection, size) is checked by the server, which owns the
// correction store; this module only produces a GeoJSON polygon.

export const MINIMUM_VERTICES = 3;

// The outer ring and holes of a footprint as open [latitude, longitude] lists,
// or null for footprints this editor cannot change (multi-part polygons).
export function editableRings(geometry) {
  let rings = null;
  if (geometry?.type === "Polygon") rings = geometry.coordinates;
  else if (geometry?.type === "MultiPolygon" && geometry.coordinates.length === 1) rings = geometry.coordinates[0];
  if (!Array.isArray(rings) || !rings.length) return null;
  const open = rings.map((ring) => {
    const points = ring.map(([longitude, latitude]) => [latitude, longitude]);
    const [first, last] = [points[0], points.at(-1)];
    return first && last && first[0] === last[0] && first[1] === last[1] ? points.slice(0, -1) : points;
  });
  return open[0].length >= MINIMUM_VERTICES ? { outer: open[0], holes: open.slice(1) } : null;
}

export function polygonFromRings({ outer, holes }) {
  const close = (ring) => [...ring, ring[0]].map(([latitude, longitude]) => [longitude, latitude]);
  return { type: "Polygon", coordinates: [close(outer), ...holes.map(close)] };
}

export function translatedRings({ outer, holes }, latitudeDelta, longitudeDelta) {
  const move = (ring) => ring.map(([latitude, longitude]) => [latitude + latitudeDelta, longitude + longitudeDelta]);
  return { outer: move(outer), holes: holes.map(move) };
}

export function edgeMidpoints(outer) {
  return outer.map((point, index) => {
    const next = outer[(index + 1) % outer.length];
    return [(point[0] + next[0]) / 2, (point[1] + next[1]) / 2];
  });
}

export function withInsertedVertex(outer, edgeIndex, point) {
  return [...outer.slice(0, edgeIndex + 1), point, ...outer.slice(edgeIndex + 1)];
}

export function withoutVertex(outer, index) {
  return outer.length > MINIMUM_VERTICES ? outer.filter((_, position) => position !== index) : outer;
}

export function ringCentroid(outer) {
  const [latitude, longitude] = outer.reduce((total, point) => [total[0] + point[0], total[1] + point[1]], [0, 0]);
  return [latitude / outer.length, longitude / outer.length];
}

// Undo/redo over immutable states; ``commit`` drops any redo branch.
export function createHistory(initial) {
  const past = [];
  const future = [];
  let present = initial;
  return {
    current: () => present,
    commit(state) {
      past.push(present);
      present = state;
      future.length = 0;
    },
    undo() {
      if (past.length) {
        future.push(present);
        present = past.pop();
      }
      return present;
    },
    redo() {
      if (future.length) {
        past.push(present);
        present = future.pop();
      }
      return present;
    },
    canUndo: () => past.length > 0,
    canRedo: () => future.length > 0,
  };
}

// Cut an outer ring along the chord between two of its vertices. Returns the
// two new rings, or null when the vertices are the same or neighbors.
export function splitRing(outer, first, second) {
  const [low, high] = first < second ? [first, second] : [second, first];
  if (low === high || high - low === 1 || (low === 0 && high === outer.length - 1)) return null;
  return [outer.slice(low, high + 1), [...outer.slice(high), ...outer.slice(0, low + 1)]];
}

// Start an editing session on a Leaflet map. ``onChange(geometries)`` runs
// while an outline moves, with one GeoJSON polygon per part, so a live
// projection can follow it. Returns { geometries(), split(), undo(), redo(),
// canUndo(), canRedo(), stop() }.
export function startFootprintEdit(L, map, geometry, onChange = () => {}) {
  const initial = editableRings(geometry);
  if (!initial) return null;
  // State: one {outer, holes} per part; a split adds parts.
  let parts = [initial];
  const history = createHistory(parts);
  const layer = L.layerGroup().addTo(map);
  // The OSM outline stays visible underneath, as the reference being corrected.
  L.polygon([initial.outer, ...initial.holes], { color: "#9aa49e", weight: 1.5, dashArray: "2 4", fill: false, interactive: false }).addTo(layer);
  const outlines = L.layerGroup().addTo(layer);
  const handles = L.layerGroup().addTo(layer);
  const icon = (className) => L.divIcon({ className, iconSize: [12, 12] });
  let splitting = false;
  let splitStart = null;

  const geometries = () => parts.map(polygonFromRings);

  function redraw() {
    outlines.clearLayers();
    parts.forEach((part) => {
      L.polygon([part.outer, ...part.holes], { color: "#2f80ed", fillColor: "#2f80ed", fillOpacity: 0.18, weight: 2, dashArray: "5 4", interactive: false }).addTo(outlines);
    });
    onChange(geometries());
  }

  // A finished gesture becomes one undo step.
  function commit() {
    if (history.current() !== parts) history.commit(parts);
    rebuildHandles();
    onChange(geometries());
  }

  function restore(state) {
    parts = state;
    splitStart = null;
    redraw();
    rebuildHandles();
  }

  function updatePart(index, part) {
    parts = parts.map((current, position) => (position === index ? part : current));
  }

  function chooseSplitVertex(partIndex, vertexIndex) {
    if (!splitStart || splitStart.part !== partIndex) {
      splitStart = { part: partIndex, vertex: vertexIndex };
      rebuildHandles();
      return;
    }
    const part = parts[partIndex];
    const pieces = part.holes.length ? null : splitRing(part.outer, splitStart.vertex, vertexIndex);
    splitStart = null;
    if (pieces) {
      parts = [...parts.slice(0, partIndex), ...pieces.map((outer) => ({ outer, holes: [] })), ...parts.slice(partIndex + 1)];
      redraw();
      commit();
    } else {
      rebuildHandles();
    }
  }

  function rebuildHandles() {
    handles.clearLayers();
    parts.forEach((part, partIndex) => {
      part.outer.forEach((point, index) => {
        const chosen = splitStart?.part === partIndex && splitStart.vertex === index;
        const vertex = L.marker(point, {
          draggable: !splitting,
          icon: icon(chosen ? "footprint-vertex chosen" : "footprint-vertex"),
          title: splitting ? "Click two vertices of one part to cut it between them" : "Drag to move; double-click to remove",
        }).addTo(handles);
        if (splitting) {
          vertex.on("click", (event) => {
            L.DomEvent.stop(event);
            chooseSplitVertex(partIndex, index);
          });
          return;
        }
        vertex.on("drag", (event) => {
          const { lat, lng } = event.target.getLatLng();
          updatePart(partIndex, { ...part, outer: part.outer.map((current, position) => (position === index ? [lat, lng] : current)) });
          redraw();
        });
        vertex.on("dragend", commit);
        vertex.on("dblclick", (event) => {
          L.DomEvent.stop(event);
          updatePart(partIndex, { ...part, outer: withoutVertex(part.outer, index) });
          redraw();
          commit();
        });
      });
      if (splitting) return;
      edgeMidpoints(part.outer).forEach((point, edgeIndex) => {
        L.marker(point, { icon: icon("footprint-midpoint"), title: "Click to add a vertex" }).addTo(handles).on("click", (event) => {
          L.DomEvent.stop(event);
          updatePart(partIndex, { ...part, outer: withInsertedVertex(part.outer, edgeIndex, point) });
          redraw();
          commit();
        });
      });
      let origin = null;
      let start = null;
      const mover = L.marker(ringCentroid(part.outer), { draggable: true, icon: icon("footprint-mover"), title: "Drag to move this part" }).addTo(handles);
      mover.on("dragstart", (event) => {
        origin = part;
        start = event.target.getLatLng();
      });
      mover.on("drag", (event) => {
        const { lat, lng } = event.target.getLatLng();
        updatePart(partIndex, translatedRings(origin, lat - start.lat, lng - start.lng));
        redraw();
      });
      mover.on("dragend", commit);
    });
  }

  redraw();
  rebuildHandles();
  map.fitBounds(L.polygon(initial.outer).getBounds(), { maxZoom: 20, padding: [24, 24] });
  return {
    geometries,
    partCount: () => parts.length,
    setSplitting(enabled) {
      splitting = enabled;
      splitStart = null;
      rebuildHandles();
    },
    undo: () => restore(history.undo()),
    redo: () => restore(history.redo()),
    canUndo: history.canUndo,
    canRedo: history.canRedo,
    stop: () => layer.remove(),
  };
}
