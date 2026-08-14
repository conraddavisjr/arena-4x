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
// Plains sit on the green side of gold rather than the sand side. As a tan they
// were a second desert, and with the shoreline now painting sand of its own the
// board read as far more arid than the map generator actually made it.
export const GROUND = {
  ocean: 0x11395f, coast: 0x2f86ad, grassland: 0x54963a, plains: 0xa9ac55,
  forest: 0x2f6b34, hills: 0x8a7346, desert: 0xe4cf90, mountains: 0x74737c,
};
const SAND = 0xd9c288;
const WATER = new Set(["ocean", "coast"]);
export const CIV_COLOURS = [0x4ade80, 0xfbbf24, 0x38bdf8, 0xf472b6];
export const WILD_COLOUR = 0x9aa0aa;

/** Axial hex to world position. Pointy-top, flat on the XZ plane. */
export function axialToWorld(q, r) {
  return [HEX * Math.sqrt(3) * (q + r / 2), 0, HEX * 1.5 * r];
}

// The six axial neighbours, and for each the pair of hex corners that form the
// edge facing it. three.js lays a six-sided cylinder's corner k at
// (sin(k*60deg), cos(k*60deg)), so corner 0 is at +Z; the pairings below were
// derived from that and are what lets a border be drawn on one *side* of a tile
// rather than over the whole of it.
const DIRS = [[0, 1], [1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1]];
const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]];
const corner = (k, radius) => [
  Math.sin((k * Math.PI) / 3) * radius,
  Math.cos((k * Math.PI) / 3) * radius,
];

const cyl = (rt, rb, h, seg = 6) => new THREE.CylinderGeometry(rt, rb, h, seg);
const scratch = new THREE.Color();

/** Deterministic value noise in [0,1). Same texture on every machine and run. */
function rand(seed) {
  let t = (seed + 0x6d2b79f5) | 0;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

// ---------------------------------------------------------------------------
// Terrain
// ---------------------------------------------------------------------------

/**
 * A surface pattern per terrain type, drawn once into a canvas.
 *
 * Terrain in Freeciv reads by *pattern* before it reads by hue - dune ripples,
 * wave lines, grass speckle - and that is most of what a flat-coloured board
 * loses. Each texture is near-white with darker marks so it multiplies against
 * the per-instance colour: the texture supplies the detail, the instance colour
 * supplies the hue and the per-tile variation.
 *
 * Generated rather than loaded, for the same reason the geometry is: no asset
 * licence question on a page that gets published, and the viewer stays text.
 */
const textures = new Map();

function terrainTexture(kind) {
  if (textures.has(kind)) return textures.get(kind);
  const n = 256;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = n;
  const g = canvas.getContext("2d");
  g.fillStyle = "#ffffff";
  g.fillRect(0, 0, n, n);

  let seed = 0;
  for (const ch of kind) seed = (seed * 131 + ch.charCodeAt(0)) | 0;

  // Scattered marks: grass tufts, rock chips, sand grain.
  //
  // Ellipses at a random angle, not rectangles. Axis-aligned rects all share
  // two edge directions, and a few hundred of them line up into a visible
  // checkerboard the moment a tile fills the screen - which is exactly the
  // artifact that reads as "low-resolution bug" rather than as ground.
  const fleck = (count, size, dark, alpha) => {
    for (let i = 0; i < count; i++) {
      const v = Math.round(255 - dark * (0.5 + rand(seed + i * 7)));
      g.fillStyle = `rgba(${v},${v},${v},${alpha})`;
      const rx = size * (0.3 + rand(seed + i * 41) * 0.7);
      g.beginPath();
      g.ellipse(
        rand(seed + i * 13) * n, rand(seed + i * 29) * n,
        rx, rx * (0.35 + rand(seed + i * 53)),
        rand(seed + i * 67) * Math.PI, 0, Math.PI * 2
      );
      g.fill();
    }
  };
  // Wavering horizontal lines: dunes on desert, swell on water.
  const ripple = (count, dark, amp, width) => {
    g.lineWidth = width;
    for (let i = 0; i < count; i++) {
      const v = Math.round(255 - dark);
      g.strokeStyle = `rgba(${v},${v},${v},0.5)`;
      const y = ((i + 0.5) / count) * n;
      g.beginPath();
      for (let x = 0; x <= n; x += 4) {
        const dy = Math.sin(x / 11 + i * 1.7 + rand(seed + i) * 6) * amp;
        x === 0 ? g.moveTo(x, y + dy) : g.lineTo(x, y + dy);
      }
      g.stroke();
    }
  };

  switch (kind) {
    case "grassland": fleck(1200, 5, 95, 0.42); break;
    case "plains": ripple(9, 34, 3, 3); fleck(650, 6, 78, 0.38); break;
    case "forest": fleck(400, 13, 120, 0.44); break;
    case "hills": fleck(480, 10, 105, 0.44); break;
    case "mountains": ripple(6, 60, 12, 5); fleck(320, 12, 120, 0.44); break;
    case "desert": ripple(14, 38, 6, 4); fleck(700, 4, 42, 0.36); break;
    case "coast": ripple(11, 50, 8, 4); break;
    case "ocean": ripple(7, 44, 12, 6); break;
    default: fleck(400, 5, 70, 0.36);
  }

  // One blur pass over the finished pattern. Marks drawn at full contrast read
  // as speckle from above but as pixels at close zoom; softening them once is
  // cheaper than tuning every case for both distances.
  g.filter = "blur(1px)";
  g.drawImage(canvas, 0, 0);
  g.filter = "none";

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  textures.set(kind, tex);
  return tex;
}

/** Which land tiles touch water, and therefore get a sand shoreline. */
function shoreline(tiles, terrain) {
  const index = new Map(tiles.map((key, i) => [key, i]));
  const shore = new Set();
  tiles.forEach((key, i) => {
    if (WATER.has(terrain[i]) || terrain[i] === "mountains") return;
    const [q, r] = key.split(",").map(Number);
    for (const [dq, dr] of DIRS) {
      const j = index.get(`${q + dq},${r + dr}`);
      if (j !== undefined && WATER.has(terrain[j])) return shore.add(i);
    }
  });
  return shore;
}

export function buildTerrain(tiles, terrain) {
  // One instanced hex column per terrain *type* - eight draw calls for the
  // whole board - because each type now carries its own surface texture and a
  // single InstancedMesh can only hold one.
  //
  // Radius exactly HEX and *no* rotation, which is what makes the board a solid
  // surface instead of scattered tiles. three.js puts a cylinder's first vertex
  // at +Z, so a six-sided cylinder is already pointy-top - the orientation this
  // axial layout is spaced for. The earlier `rotateY(PI/6)` turned every column
  // flat-top while the spacing stayed pointy-top, so neighbours overlapped
  // along one axis and left gaps along the others. That mismatch, not the
  // radius, is what produced the broken-honeycomb look with dark voids between
  // tiles. At radius HEX with no rotation the seam is exactly zero.
  const group = new THREE.Group();
  const shore = shoreline(tiles, terrain);
  const sand = new THREE.Color(SAND);
  const byKind = new Map();
  terrain.forEach((kind, i) => {
    if (!byKind.has(kind)) byKind.set(kind, []);
    byKind.get(kind).push(i);
  });

  const m = new THREE.Matrix4();
  for (const [kind, indices] of byKind) {
    const mesh = new THREE.InstancedMesh(
      cyl(HEX, HEX, 1, 6),
      new THREE.MeshLambertMaterial({ flatShading: true, map: terrainTexture(kind) }),
      indices.length
    );
    const base = [];
    indices.forEach((i, n) => {
      const [q, r] = tiles[i].split(",").map(Number);
      const [x, , z] = axialToWorld(q, r);
      const h = HEIGHT[kind] ?? 0.36;
      m.makeScale(1, h, 1);
      m.setPosition(x, h / 2, z);
      mesh.setMatrixAt(n, m);
      // Slight per-tile value jitter, keyed to the index so it never shimmers
      // between turns. Without it a large grassland reads as one flat sheet.
      const colour = new THREE.Color(GROUND[kind] ?? 0x777777);
      colour.offsetHSL(0, 0, (((i * 2654435761) % 1000) / 1000 - 0.5) * 0.07);
      // The shoreline. Ringing every landmass in sand is the thing that makes a
      // coast legible at a glance, and it is doing the work of a whole
      // transition-tile system for the cost of one lerp.
      if (shore.has(i)) colour.lerp(sand, 0.42);
      base.push(colour);
      mesh.setColorAt(n, colour);
    });
    mesh.instanceMatrix.needsUpdate = true;
    mesh.receiveShadow = true;
    mesh.castShadow = true;
    mesh.userData = { indices, base };
    group.add(mesh);
  }
  group.userData.paint = (shade) => paint(group, shade);
  return group;
}

/**
 * Re-tint instances by tile, which is how fog of war is applied.
 *
 * Rebuilding the board on every focus change would drop frames on a 1000-tile
 * map; multiplying the existing colour buffer is close to free.
 */
function paint(group, shade) {
  for (const mesh of group.children) {
    const { indices, base } = mesh.userData;
    indices.forEach((tile, n) => {
      mesh.setColorAt(n, scratch.copy(base[n]).multiplyScalar(shade(tile)));
    });
    mesh.instanceColor.needsUpdate = true;
  }
}

/** Trees, peaks and scrub standing on the terrain. */
export function buildScatter(tiles, terrain) {
  const group = new THREE.Group();

  const pine = mergeGeometries([
    cyl(0, 0.26, 0.62, 7).translate(0, 0.46, 0),
    cyl(0, 0.19, 0.42, 7).translate(0, 0.78, 0),
    cyl(0.045, 0.055, 0.2, 5).translate(0, 0.1, 0),
  ]);
  // Translated up by half their height, because a cylinder is built centred on
  // the origin: an untranslated cone sits half-buried in the tile it stands on.
  // That is what made mountains read as pebbles scattered on a grey plateau
  // rather than as a range - only the top half of an already-short cone showed.
  const peak = cyl(0.03, 0.72, 1.4, 6).translate(0, 0.7, 0);
  const scrub = cyl(0, 0.13, 0.2, 5).translate(0, 0.1, 0);

  const kinds = [
    { geo: pine, colour: 0x2f5f34, on: "forest", per: 3, spread: 0.42 },
    // Two overlapping peaks per tile at different scales, so a mountain range
    // has a broken skyline instead of a row of identical cones.
    { geo: peak, colour: 0xbfc4cd, on: "mountains", per: 2, spread: 0.34 },
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
      new THREE.MeshLambertMaterial({ flatShading: true }),
      targets.length * kind.per
    );
    const indices = [];
    const base = [];
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
        mesh.setColorAt(n, scratch.setHex(kind.colour));
        mesh.setMatrixAt(n++, m);
        indices.push(i);
        base.push(new THREE.Color(kind.colour));
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.castShadow = true;
    mesh.userData = { indices, base };
    group.add(mesh);
  }
  group.userData.paint = (shade) => paint(group, shade);
  return group;
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/**
 * Where the wheat, iron and fish are.
 *
 * Freeciv puts a sprite on every special tile, and it matters here for a
 * reason beyond decoration: strategic resources gate units, so "that civ has
 * the only iron on the continent" is a fact a spectator needs to be able to
 * read off the board to judge whether an agent's plan makes sense.
 *
 * Offset off-centre so a marker never sits under a unit or a city.
 */
const RESOURCE_MODELS = {
  wheat: () => ({
    colour: 0xe8c65c,
    geo: mergeGeometries([0, 1, 2].map((k) =>
      cyl(0, 0.05, 0.26, 5).rotateZ((k - 1) * 0.3).translate((k - 1) * 0.06, 0.13, 0))),
  }),
  iron: () => ({ colour: 0x7f858e, geo: new THREE.OctahedronGeometry(0.11).translate(0, 0.1, 0) }),
  gold_ore: () => ({ colour: 0xffcf3d, geo: new THREE.OctahedronGeometry(0.1).translate(0, 0.1, 0) }),
  horses: () => ({
    colour: 0x8d5a35,
    geo: mergeGeometries([
      new THREE.BoxGeometry(0.24, 0.1, 0.1).translate(0, 0.16, 0),
      new THREE.BoxGeometry(0.08, 0.14, 0.08).translate(0.11, 0.24, 0),
    ]),
  }),
  deer: () => ({
    colour: 0xa8703c,
    geo: mergeGeometries([
      new THREE.BoxGeometry(0.22, 0.1, 0.1).translate(0, 0.16, 0),
      cyl(0, 0.04, 0.14, 4).translate(0.1, 0.28, 0),
    ]),
  }),
  fish: () => ({
    colour: 0x86d8ea,
    geo: mergeGeometries([
      new THREE.SphereGeometry(0.09, 7, 5).scale(1.5, 0.7, 0.8).translate(0, 0.04, 0),
      cyl(0, 0.07, 0.12, 4).rotateZ(Math.PI / 2).translate(-0.16, 0.04, 0),
    ]),
  }),
};

export function buildResources(tiles, terrain, resources) {
  const group = new THREE.Group();
  const byKind = new Map();
  for (const [i, kind] of Object.entries(resources || {})) {
    if (!RESOURCE_MODELS[kind]) continue;
    if (!byKind.has(kind)) byKind.set(kind, []);
    byKind.get(kind).push(Number(i));
  }

  const m = new THREE.Matrix4();
  for (const [kind, targets] of byKind) {
    const { geo, colour } = RESOURCE_MODELS[kind]();
    const mesh = new THREE.InstancedMesh(
      geo,
      new THREE.MeshLambertMaterial({ flatShading: true }),
      targets.length
    );
    const base = [];
    targets.forEach((i, n) => {
      const [q, r] = tiles[i].split(",").map(Number);
      const [x, , z] = axialToWorld(q, r);
      m.makeRotationY((i % 6) * 1.05);
      m.setPosition(x - 0.34, HEIGHT[terrain[i]] ?? 0.36, z + 0.42);
      mesh.setMatrixAt(n, m);
      mesh.setColorAt(n, scratch.setHex(colour));
      base.push(new THREE.Color(colour));
    });
    mesh.instanceMatrix.needsUpdate = true;
    mesh.castShadow = true;
    mesh.userData = { indices: targets, base };
    group.add(mesh);
  }
  group.userData.paint = (shade) => paint(group, shade);
  return group;
}

// ---------------------------------------------------------------------------
// Borders
// ---------------------------------------------------------------------------

/**
 * Territory drawn as a line on the *edge* of the claim, not a wash over it.
 *
 * The earlier version laid a tinted slab across every owned tile, which meant
 * the terrain a civ had claimed was the terrain you could no longer see - and
 * with four civs on one screen it turned the middle of the board to mud.
 * Freeciv draws a thin coloured line along the boundary instead, which says the
 * same thing and costs nothing visually. This is that line, given a little
 * height so it survives a low camera angle.
 *
 * An edge is emitted when a tile's neighbour has a different owner, so interior
 * edges never draw and the ribbon count stays proportional to the perimeter.
 */
export function buildBorders(tiles, owners, heightOf, colourOf, visible) {
  const group = new THREE.Group();
  const index = new Map(tiles.map((key, i) => [key, i]));
  const byOwner = new Map();

  tiles.forEach((key, i) => {
    const who = owners[i];
    if (!who || (visible && !visible(i))) return;
    const [q, r] = key.split(",").map(Number);
    DIRS.forEach(([dq, dr], d) => {
      const j = index.get(`${q + dq},${r + dr}`);
      if (j !== undefined && owners[j] === who) return;
      if (!byOwner.has(who)) byOwner.set(who, []);
      byOwner.get(who).push([i, d]);
    });
  });

  const INSET = 0.94;
  const RISE = 0.16;
  for (const [who, edges] of byOwner) {
    const verts = new Float32Array(edges.length * 18);
    let o = 0;
    for (const [i, d] of edges) {
      const [q, r] = tiles[i].split(",").map(Number);
      const [cx, , cz] = axialToWorld(q, r);
      const y = heightOf(i) + 0.01;
      const [ax, az] = corner(EDGES[d][0], HEX * INSET);
      const [bx, bz] = corner(EDGES[d][1], HEX * INSET);
      const a = [cx + ax, y, cz + az], b = [cx + bx, y, cz + bz];
      const at = [a[0], y + RISE, a[2]], bt = [b[0], y + RISE, b[2]];
      for (const v of [a, b, bt, a, bt, at]) {
        verts.set(v, o);
        o += 3;
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    // Unlit, so a border is the same vivid colour on a shadowed slope as on a
    // sunlit plain - it is a label, not a surface.
    group.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      color: colourOf(who), side: THREE.DoubleSide, transparent: true, opacity: 0.92,
    })));
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
