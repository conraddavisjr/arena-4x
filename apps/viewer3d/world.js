/**
 * The world: terrain, settlements, armies and wildlife, built as 3D geometry.
 *
 * **Everything is instanced.** The board is ~1000 tiles and a late-game turn
 * carries 80+ units; one mesh per object would be a thousand draw calls and the
 * camera would stutter the moment you tried to orbit. Instead there is one
 * InstancedMesh per *kind* of thing - one for the entire hex field, one per tree
 * species, one per unit type - which is roughly 25 draw calls for the whole
 * scene regardless of how big the empire grows.
 *
 * Geometry is procedural rather than loaded from model files. That keeps the
 * art direction consistent, avoids a licence question on a page that gets
 * published, and means the whole viewer is still text in a repo. Swapping in
 * real GLTF models later replaces `unitGeometry` and nothing else.
 */

import * as THREE from "three";
import { mergeGeometries } from "three/BufferGeometryUtils.js";

export const HEX = 1;
// Relief. The steps between bands are deliberately large: at a low camera
// angle the side wall of each column is what reads as a cliff or a shoreline,
// and small differences flatten into nothing.
export const HEIGHT = {
  ocean: 0.10, coast: 0.28, desert: 0.60, plains: 0.62,
  grassland: 0.64, forest: 0.66, hills: 1.05, mountains: 1.95,
};
// Ground colours, pushed apart so terrain type reads instantly at a distance -
// the thing a flat palette loses first. Sand is genuinely pale, ocean genuinely
// deep, grassland and forest clearly different greens rather than two shades.
export const GROUND = {
  ocean: 0x11395f, coast: 0x2f86ad, grassland: 0x54963a, plains: 0xc0aa5a,
  forest: 0x2f6b34, hills: 0x8a7346, desert: 0xe4cf90, mountains: 0x74737c,
};
export const CIV_COLOURS = [0x4ade80, 0xfbbf24, 0x38bdf8, 0xf472b6];
export const WILD_COLOUR = 0x9aa0aa;

/** Axial hex to world position. Pointy-top, flat on the XZ plane. */
export function axialToWorld(q, r) {
  return [HEX * Math.sqrt(3) * (q + r / 2), 0, HEX * 1.5 * r];
}

const cyl = (rt, rb, h, seg = 6) => new THREE.CylinderGeometry(rt, rb, h, seg);

// ---------------------------------------------------------------------------
// Terrain
// ---------------------------------------------------------------------------

export function buildTerrain(tiles, terrain) {
  // A single instanced hex column for the entire board. Height and colour come
  // from the per-instance matrix and colour buffer, so eight terrain types cost
  // one draw call rather than eight.
  //
  // Radius exactly HEX and *no* rotation, which is what makes the board a solid
  // surface instead of scattered tiles. three.js puts a cylinder's first vertex
  // at +Z, so a six-sided cylinder is already pointy-top - the orientation this
  // axial layout is spaced for. The earlier `rotateY(PI/6)` turned every column
  // flat-top while the spacing stayed pointy-top, so neighbours overlapped
  // along one axis and left gaps along the others. That mismatch, not the
  // radius, is what produced the broken-honeycomb look with dark voids between
  // tiles. At radius HEX with no rotation the seam is exactly zero.
  const geo = cyl(HEX, HEX, 1, 6);
  const mesh = new THREE.InstancedMesh(
    geo,
    new THREE.MeshLambertMaterial({ flatShading: true }),
    tiles.length
  );
  const m = new THREE.Matrix4();
  const colour = new THREE.Color();

  tiles.forEach((key, i) => {
    const [q, r] = key.split(",").map(Number);
    const [x, , z] = axialToWorld(q, r);
    const h = HEIGHT[terrain[i]] ?? 0.36;
    m.makeScale(1, h, 1);
    m.setPosition(x, h / 2, z);
    mesh.setMatrixAt(i, m);
    // Slight per-tile value jitter, keyed to the index so it never shimmers
    // between turns. Without it a large grassland reads as one flat sheet.
    colour.setHex(GROUND[terrain[i]] ?? 0x777777);
    const jitter = (((i * 2654435761) % 1000) / 1000 - 0.5) * 0.08;
    colour.offsetHSL(0, 0, jitter);
    mesh.setColorAt(i, colour);
  });
  mesh.instanceMatrix.needsUpdate = true;
  mesh.receiveShadow = true;
  mesh.castShadow = true;
  return mesh;
}

/** Trees, peaks and scrub standing on the terrain. */
export function buildScatter(tiles, terrain) {
  const group = new THREE.Group();

  const pine = mergeGeometries([
    cyl(0, 0.26, 0.62, 7).translate(0, 0.46, 0),
    cyl(0, 0.19, 0.42, 7).translate(0, 0.78, 0),
    cyl(0.045, 0.055, 0.2, 5).translate(0, 0.1, 0),
  ]);
  const peak = cyl(0.02, 0.42, 0.7, 6);
  const scrub = cyl(0, 0.13, 0.2, 5);

  const kinds = [
    { geo: pine, colour: 0x2f5f34, on: "forest", per: 3, spread: 0.42 },
    { geo: peak, colour: 0xbfc4cd, on: "mountains", per: 1, spread: 0.12 },
    { geo: scrub, colour: 0x7d9450, on: "grassland", per: 2, spread: 0.5 },
    { geo: scrub, colour: 0x9c8f5a, on: "desert", per: 1, spread: 0.5 },
  ];

  for (const kind of kinds) {
    const targets = [];
    tiles.forEach((key, i) => {
      if (terrain[i] === kind.on) targets.push(i);
    });
    if (!targets.length) continue;

    const mesh = new THREE.InstancedMesh(
      kind.geo,
      new THREE.MeshLambertMaterial({ color: kind.colour, flatShading: true }),
      targets.length * kind.per
    );
    const m = new THREE.Matrix4();
    let n = 0;
    for (const i of targets) {
      const [q, r] = tiles[i].split(",").map(Number);
      const [x, , z] = axialToWorld(q, r);
      const top = HEIGHT[terrain[i]];
      for (let k = 0; k < kind.per; k++) {
        // Offsets derived from the tile index, so a forest looks scattered but
        // is identical on every replay of the same match.
        const a = ((i * 97 + k * 211) % 360) * (Math.PI / 180);
        const d = (((i * 57 + k * 131) % 100) / 100) * kind.spread;
        const s = 0.8 + (((i * 31 + k * 17) % 40) / 100);
        m.makeScale(s, s, s);
        m.setPosition(x + Math.cos(a) * d, top, z + Math.sin(a) * d);
        mesh.setMatrixAt(n++, m);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.castShadow = true;
    group.add(mesh);
  }
  return group;
}

// ---------------------------------------------------------------------------
// Units
// ---------------------------------------------------------------------------

const PAWN = () =>
  mergeGeometries([
    cyl(0.11, 0.17, 0.34, 8).translate(0, 0.17, 0),
    new THREE.SphereGeometry(0.115, 8, 6).translate(0, 0.42, 0),
  ]);

/**
 * One merged low-poly model per unit type.
 *
 * Silhouette carries the meaning: at the zoom a spectator actually watches
 * from, a spear above the head or a hull below the waterline reads far faster
 * than any colour or icon could.
 */
export function unitGeometry(type) {
  const box = (w, h, d) => new THREE.BoxGeometry(w, h, d);
  switch (type) {
    case "settler":
      return mergeGeometries([
        box(0.42, 0.16, 0.26).translate(0, 0.16, 0),
        cyl(0.16, 0.16, 0.4, 8, 1).rotateZ(Math.PI / 2).translate(0, 0.3, 0),
        cyl(0.07, 0.07, 0.05, 8).rotateX(Math.PI / 2).translate(-0.16, 0.09, 0.14),
        cyl(0.07, 0.07, 0.05, 8).rotateX(Math.PI / 2).translate(0.16, 0.09, 0.14),
      ]);
    case "worker":
      return mergeGeometries([PAWN(), cyl(0.02, 0.02, 0.4, 5).rotateZ(0.5).translate(0.17, 0.3, 0)]);
    case "scout":
      return mergeGeometries([PAWN().scale(0.85, 1.05, 0.85)]);
    case "warrior":
      return mergeGeometries([PAWN(), box(0.05, 0.24, 0.16).translate(-0.19, 0.28, 0)]);
    case "archer":
      return mergeGeometries([
        PAWN(),
        new THREE.TorusGeometry(0.17, 0.018, 4, 10, Math.PI).rotateY(Math.PI / 2).translate(0.18, 0.32, 0),
      ]);
    case "spearman":
      return mergeGeometries([
        PAWN(),
        cyl(0.018, 0.018, 0.6, 5).translate(0.18, 0.36, 0),
        cyl(0, 0.05, 0.12, 4).translate(0.18, 0.68, 0),
      ]);
    case "swordsman":
      return mergeGeometries([
        PAWN().scale(1.1, 1, 1.1),
        box(0.04, 0.3, 0.06).translate(0.18, 0.38, 0),
        box(0.14, 0.03, 0.05).translate(0.18, 0.25, 0),
      ]);
    case "horseman":
      return mergeGeometries([
        box(0.46, 0.2, 0.2).translate(0, 0.28, 0),
        box(0.16, 0.22, 0.16).translate(0.24, 0.4, 0),
        cyl(0.04, 0.04, 0.26).translate(-0.16, 0.13, 0.07),
        cyl(0.04, 0.04, 0.26).translate(0.16, 0.13, 0.07),
        cyl(0.04, 0.04, 0.26).translate(-0.16, 0.13, -0.07),
        cyl(0.04, 0.04, 0.26).translate(0.16, 0.13, -0.07),
        new THREE.SphereGeometry(0.1, 8, 6).translate(0, 0.52, 0),
      ]);
    case "catapult":
      return mergeGeometries([
        box(0.4, 0.1, 0.24).translate(0, 0.14, 0),
        cyl(0.09, 0.09, 0.05, 10).rotateX(Math.PI / 2).translate(-0.15, 0.09, 0.13),
        cyl(0.09, 0.09, 0.05, 10).rotateX(Math.PI / 2).translate(0.15, 0.09, 0.13),
        cyl(0.025, 0.025, 0.45, 5).rotateZ(-0.9).translate(0.06, 0.34, 0),
        new THREE.SphereGeometry(0.09, 7, 5).translate(0.24, 0.5, 0),
      ]);
    case "trireme":
      return mergeGeometries([
        cyl(0.14, 0.1, 0.62, 6).rotateZ(Math.PI / 2).translate(0, 0.12, 0),
        cyl(0.02, 0.02, 0.52, 5).translate(0, 0.4, 0),
        box(0.01, 0.3, 0.3).translate(0.02, 0.44, 0),
      ]);
    case "wolf":
      return mergeGeometries([
        box(0.34, 0.14, 0.14).translate(0, 0.2, 0),
        box(0.14, 0.12, 0.12).translate(0.21, 0.24, 0),
        cyl(0.03, 0.03, 0.18).translate(-0.11, 0.09, 0.05),
        cyl(0.03, 0.03, 0.18).translate(0.11, 0.09, 0.05),
        cyl(0.03, 0.03, 0.18).translate(-0.11, 0.09, -0.05),
        cyl(0.03, 0.03, 0.18).translate(0.11, 0.09, -0.05),
        cyl(0, 0.04, 0.14, 4).rotateZ(-0.7).translate(-0.2, 0.28, 0),
      ]);
    case "barbarian":
      return mergeGeometries([
        PAWN().scale(1.05, 1.05, 1.05),
        cyl(0, 0.09, 0.2, 4).translate(0.19, 0.46, 0),
        cyl(0.02, 0.02, 0.42, 5).translate(0.19, 0.28, 0),
      ]);
    default:
      return PAWN();
  }
}

/** A ring marking which civ a unit belongs to, read at any camera angle. */
export function baseGeometry() {
  return new THREE.CylinderGeometry(0.26, 0.26, 0.035, 12);
}

// ---------------------------------------------------------------------------
// Settlements
// ---------------------------------------------------------------------------

export function houseGeometry() {
  return mergeGeometries([
    new THREE.BoxGeometry(0.24, 0.2, 0.24).translate(0, 0.1, 0),
    cyl(0, 0.2, 0.16, 4).rotateY(Math.PI / 4).translate(0, 0.28, 0),
  ]);
}

export function towerGeometry() {
  return mergeGeometries([
    cyl(0.09, 0.11, 0.52, 8).translate(0, 0.26, 0),
    cyl(0, 0.14, 0.16, 8).translate(0, 0.6, 0),
  ]);
}

export function wallGeometry() {
  // A low ring around the settlement. Visibly a fortification from above and
  // from the side, which is the point of showing what a civ chose to build.
  return new THREE.CylinderGeometry(0.82, 0.86, 0.24, 12, 1, true);
}

/**
 * Where the buildings of a city sit, given its population.
 *
 * Deterministic: the same city has the same street plan on every replay, so
 * scrubbing back and forth does not rearrange the town.
 */
export function cityLayout(population, seed) {
  const count = Math.min(7, 1 + Math.floor(population / 1.6));
  const spots = [];
  for (let k = 0; k < count; k++) {
    if (k === 0) {
      spots.push([0, 0, 0]);
      continue;
    }
    const a = ((seed * 53 + k * 137) % 360) * (Math.PI / 180);
    const d = 0.28 + ((seed * 17 + k * 41) % 30) / 100;
    spots.push([Math.cos(a) * d, ((seed + k) % 3) * 0.02, Math.sin(a) * d]);
  }
  return spots;
}
