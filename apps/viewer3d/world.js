/**
 * The world: terrain, settlements, armies and wildlife, built as 3D geometry.
 *
 * **Nothing is drawn one object at a time.** The board is ~1000 tiles and a
 * late-game turn carries 80+ units; a mesh per object would be a thousand draw
 * calls and the camera would stutter the moment you tried to orbit. Everything
 * is batched per *kind* of thing - one mesh for each terrain type, one per tree
 * species, two per unit type - which is a few dozen draw calls for the whole
 * scene regardless of how big the empires grow.
 *
 * Trees, units and resources are instanced, because every copy is identical.
 * The terrain is *merged* rather than instanced, because instancing forces one
 * flat colour per tile and that is what made every terrain boundary a hard
 * edge; see `buildTerrain`.
 *
 * Geometry is procedural rather than loaded from model files. That keeps the
 * art direction consistent, avoids a licence question on a page that gets
 * published, and means the whole viewer is still text in a repo. Swapping in
 * real GLTF models later replaces `unitModel` and nothing else.
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
const FOAM = 0xa8dced;
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
// Surface atlas
// ---------------------------------------------------------------------------

/**
 * One texture holding sixteen surface patterns, and the UV remapping that lets
 * a model made of a dozen merged parts sample a different one per part.
 *
 * The problem this solves: a unit is a single merged geometry drawn in one
 * call, so it gets exactly one texture - but a soldier needs cloth on his
 * tunic, mail on his helmet, wood on his spear shaft and steel on its head.
 * Packing every pattern into one atlas and rewriting each part's UVs into its
 * own cell at merge time gives per-part materials at no extra draw cost.
 *
 * Every cell is drawn near-white, because it multiplies against the vertex
 * colour that carries the hue and the instance colour that carries the civ
 * livery. The texture supplies grain; it never supplies colour. That is the
 * same division the terrain uses.
 *
 * `flipY` is off so canvas coordinates and UV coordinates agree, and each
 * cell's UVs are inset by a pixel so mip filtering cannot bleed a neighbouring
 * pattern in along the seam.
 */
const CELL = 128;
const GRID = 4;
const INSET = 1 / CELL;

export const SURFACE = {
  plain: 0, skin: 1, weave: 2, mail: 3,
  leather: 4, wood: 5, metal: 6, fur: 7,
  hide: 8, stone: 9, plaster: 10, thatch: 11,
  tile: 12, timber: 13, canvas: 14, scales: 15,
};

/** Pattern painters. Each draws into a CELL-sized square at the origin. */
const PATTERNS = {
  plain: (g, s, r) => speckle(g, s, r, 120, 3, 14, 0.25),
  skin: (g, s, r) => speckle(g, s, r, 260, 4, 20, 0.2),
  weave: (g, s, r) => {
    // Crosshatch, which is what cloth reads as once it is too small to see.
    hatch(g, s, 5, 26, 0.55, 0);
    hatch(g, s, 5, 20, 0.45, Math.PI / 2);
  },
  mail: (g, s, r) => {
    for (let y = 3; y < s; y += 7) {
      for (let x = (y % 14 ? 3 : 7); x < s; x += 7) {
        g.strokeStyle = "rgba(120,120,120,0.55)";
        g.lineWidth = 1.4;
        g.beginPath();
        g.arc(x, y, 2.4, 0, Math.PI * 2);
        g.stroke();
      }
    }
  },
  leather: (g, s, r) => {
    speckle(g, s, r, 180, 9, 40, 0.3);
    crease(g, s, r, 14, 30, 0.35);
  },
  wood: (g, s, r) => {
    // Grain runs along one axis, with a couple of knots.
    for (let i = 0; i < 26; i++) {
      const v = 255 - 30 - r(i * 7) * 55;
      g.strokeStyle = `rgba(${v},${v},${v},0.6)`;
      g.lineWidth = 0.7 + r(i * 11) * 1.8;
      g.beginPath();
      const x = r(i * 13) * s;
      for (let y = 0; y <= s; y += 6) g.lineTo(x + Math.sin(y / 26 + i) * 3.5, y);
      g.stroke();
    }
    for (let i = 0; i < 2; i++) {
      g.strokeStyle = "rgba(150,150,150,0.5)";
      g.lineWidth = 1.6;
      g.beginPath();
      g.ellipse(r(i * 3 + 1) * s, r(i * 5 + 2) * s, 4, 7, 0, 0, Math.PI * 2);
      g.stroke();
    }
  },
  metal: (g, s, r) => {
    // Brushed streaks plus one brighter band, which is what sells a blade at
    // this size far more than any amount of surface detail.
    for (let i = 0; i < 70; i++) {
      const v = 255 - r(i * 17) * 46;
      g.strokeStyle = `rgba(${v},${v},${v},0.5)`;
      g.lineWidth = 0.6 + r(i * 23) * 1.2;
      const y = r(i * 29) * s;
      g.beginPath();
      g.moveTo(0, y);
      g.lineTo(s, y + (r(i * 31) - 0.5) * 5);
      g.stroke();
    }
    const band = g.createLinearGradient(0, 0, 0, s);
    band.addColorStop(0, "rgba(255,255,255,0)");
    band.addColorStop(0.45, "rgba(255,255,255,0.85)");
    band.addColorStop(1, "rgba(160,160,160,0.35)");
    g.fillStyle = band;
    g.fillRect(0, 0, s, s);
  },
  fur: (g, s, r) => strokes(g, s, r, 900, 7, 70, -0.5),
  hide: (g, s, r) => {
    strokes(g, s, r, 400, 5, 40, -0.4);
    // Dappling, which is the one marking that says deer rather than dog.
    for (let i = 0; i < 26; i++) {
      g.fillStyle = "rgba(255,255,255,0.75)";
      g.beginPath();
      g.ellipse(r(i * 19) * s, r(i * 37) * s, 3.5, 2.6, r(i) * 3, 0, Math.PI * 2);
      g.fill();
    }
  },
  stone: (g, s, r) => {
    // Courses of blocks with mortar between them.
    g.strokeStyle = "rgba(105,105,105,0.6)";
    g.lineWidth = 1.6;
    const rows = 6;
    for (let row = 0; row < rows; row++) {
      const y = (row / rows) * s;
      g.beginPath();
      g.moveTo(0, y);
      g.lineTo(s, y);
      g.stroke();
      const step = s / 4;
      for (let x = (row % 2 ? step / 2 : 0); x < s; x += step) {
        g.beginPath();
        g.moveTo(x, y);
        g.lineTo(x, y + s / rows);
        g.stroke();
      }
    }
    speckle(g, s, r, 300, 4, 26, 0.22);
  },
  plaster: (g, s, r) => speckle(g, s, r, 340, 7, 26, 0.22),
  thatch: (g, s, r) => strokes(g, s, r, 700, 12, 62, 0.9),
  tile: (g, s, r) => {
    // Overlapping scallops in courses, read as a tiled roof from above.
    const step = s / 8;
    for (let row = 0; row < 8; row++) {
      for (let col = -1; col < 9; col++) {
        const x = col * step + (row % 2 ? step / 2 : 0);
        const y = row * step;
        g.strokeStyle = "rgba(110,110,110,0.6)";
        g.lineWidth = 1.5;
        g.beginPath();
        g.arc(x + step / 2, y, step / 2, 0, Math.PI);
        g.stroke();
      }
    }
  },
  timber: (g, s, r) => {
    PATTERNS.plaster(g, s, r);
    // Half-timbering: dark posts and a rail over pale infill.
    g.fillStyle = "rgba(90,90,90,0.7)";
    for (const x of [0.06, 0.47, 0.88]) g.fillRect(x * s, 0, s * 0.06, s);
    g.fillRect(0, s * 0.46, s, s * 0.06);
  },
  canvas: (g, s, r) => {
    PATTERNS.weave(g, s, r);
    // Ribs, which is what makes a wagon tilt or a sail read as stretched.
    g.strokeStyle = "rgba(120,120,120,0.5)";
    g.lineWidth = 2.4;
    for (let x = s / 8; x < s; x += s / 4) {
      g.beginPath();
      g.moveTo(x, 0);
      g.lineTo(x, s);
      g.stroke();
    }
  },
  scales: (g, s, r) => {
    const step = s / 10;
    for (let row = 0; row < 11; row++) {
      for (let col = -1; col < 11; col++) {
        g.strokeStyle = "rgba(115,115,115,0.55)";
        g.lineWidth = 1.2;
        g.beginPath();
        g.arc(col * step + (row % 2 ? step / 2 : 0) + step / 2, row * step, step / 2, 0, Math.PI);
        g.stroke();
      }
    }
  },
};

function speckle(g, s, r, count, size, dark, alpha) {
  for (let i = 0; i < count; i++) {
    const v = 255 - dark * (0.4 + r(i * 7));
    g.fillStyle = `rgba(${v},${v},${v},${alpha})`;
    const rx = size * (0.3 + r(i * 41) * 0.7);
    g.beginPath();
    g.ellipse(r(i * 13) * s, r(i * 29) * s, rx, rx * (0.4 + r(i * 53)), r(i * 67) * Math.PI,
      0, Math.PI * 2);
    g.fill();
  }
}

function strokes(g, s, r, count, len, dark, angle) {
  for (let i = 0; i < count; i++) {
    const v = 255 - dark * (0.3 + r(i * 7));
    g.strokeStyle = `rgba(${v},${v},${v},0.5)`;
    g.lineWidth = 0.8 + r(i * 3) * 0.8;
    const x = r(i * 13) * s;
    const y = r(i * 29) * s;
    const a = angle + (r(i * 43) - 0.5) * 0.6;
    g.beginPath();
    g.moveTo(x, y);
    g.lineTo(x + Math.cos(a) * len, y + Math.sin(a) * len);
    g.stroke();
  }
}

function hatch(g, s, gap, dark, alpha, angle) {
  g.save();
  g.translate(s / 2, s / 2);
  g.rotate(angle);
  g.translate(-s, -s);
  g.strokeStyle = `rgba(${255 - dark},${255 - dark},${255 - dark},${alpha})`;
  g.lineWidth = 1;
  for (let y = 0; y < s * 2; y += gap) {
    g.beginPath();
    g.moveTo(0, y);
    g.lineTo(s * 2, y);
    g.stroke();
  }
  g.restore();
}

function crease(g, s, r, count, dark, alpha) {
  g.strokeStyle = `rgba(${255 - dark},${255 - dark},${255 - dark},${alpha})`;
  g.lineWidth = 1.1;
  for (let i = 0; i < count; i++) {
    g.beginPath();
    g.moveTo(r(i * 11) * s, r(i * 17) * s);
    g.bezierCurveTo(r(i * 19) * s, r(i * 23) * s, r(i * 29) * s, r(i * 31) * s,
      r(i * 37) * s, r(i * 41) * s);
    g.stroke();
  }
}

let atlas = null;

export function surfaceAtlas() {
  if (atlas) return atlas;
  const size = CELL * GRID;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const g = canvas.getContext("2d");
  g.fillStyle = "#ffffff";
  g.fillRect(0, 0, size, size);

  for (const [name, index] of Object.entries(SURFACE)) {
    let seed = 0;
    for (const ch of name) seed = (seed * 131 + ch.charCodeAt(0)) | 0;
    g.save();
    g.beginPath();
    g.rect((index % GRID) * CELL, Math.floor(index / GRID) * CELL, CELL, CELL);
    g.clip();
    g.translate((index % GRID) * CELL, Math.floor(index / GRID) * CELL);
    PATTERNS[name](g, CELL, (n) => rand(seed + n));
    g.restore();
  }

  atlas = new THREE.CanvasTexture(canvas);
  atlas.colorSpace = THREE.SRGBColorSpace;
  atlas.flipY = false;
  atlas.wrapS = atlas.wrapT = THREE.ClampToEdgeWrapping;
  atlas.anisotropy = 4;
  return atlas;
}

/** Rewrite a part's UVs into one atlas cell. */
function surface(geo, cell) {
  const uv = geo.attributes.uv;
  const cx = cell % GRID;
  const cy = Math.floor(cell / GRID);
  const span = 1 - 2 * INSET;
  for (let i = 0; i < uv.count; i++) {
    uv.setXY(
      i,
      (cx + INSET + uv.getX(i) * span) / GRID,
      (cy + INSET + uv.getY(i) * span) / GRID
    );
  }
  return geo;
}

/** Give a part its hue and its grain: a vertex colour for the one, an atlas
 *  cell for the other. Parts prepared this way merge into a single geometry
 *  and still shade and texture independently. */
function material(geo, hex, cell) {
  const c = new THREE.Color(hex);
  const n = geo.attributes.position.count;
  const colours = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) c.toArray(colours, i * 3);
  geo.setAttribute("color", new THREE.BufferAttribute(colours, 3));
  return surface(geo, cell);
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

/** Neighbour lookup for every tile, by direction index. Built once and reused
 *  by the shoreline pass, the depth pass and the corner blending. */
function neighbours(tiles) {
  const index = new Map(tiles.map((key, i) => [key, i]));
  return tiles.map((key) => {
    const [q, r] = key.split(",").map(Number);
    return DIRS.map(([dq, dr]) => index.get(`${q + dq},${r + dr}`) ?? -1);
  });
}

/** Rings of water outward from the nearest land, so open sea reads as deeper
 *  than the shallows. Freeciv gets this from two water sprites; a distance
 *  field gives a continuous gradient for the same idea. */
function depth(tiles, terrain, near) {
  const dist = new Int16Array(tiles.length).fill(-1);
  const queue = [];
  terrain.forEach((kind, i) => {
    if (!WATER.has(kind)) return;
    if (near[i].some((j) => j >= 0 && !WATER.has(terrain[j]))) {
      dist[i] = 0;
      queue.push(i);
    }
  });
  for (let head = 0; head < queue.length; head++) {
    const i = queue[head];
    if (dist[i] >= 4) continue;
    for (const j of near[i]) {
      if (j >= 0 && WATER.has(terrain[j]) && dist[j] < 0) {
        dist[j] = dist[i] + 1;
        queue.push(j);
      }
    }
  }
  return dist;
}

/** The colour of a tile before any blending: terrain, jitter, shore, depth. */
function tileColour(i, terrain, shore, dist) {
  const kind = terrain[i];
  const colour = new THREE.Color(GROUND[kind] ?? 0x777777);
  // Slight per-tile value jitter, keyed to the index so it never shimmers
  // between turns. Without it a large grassland reads as one flat sheet.
  colour.offsetHSL(0, 0, (((i * 2654435761) % 1000) / 1000 - 0.5) * 0.07);
  // The shoreline. Ringing every landmass in sand is the thing that makes a
  // coast legible at a glance, and it does the work of a whole transition-tile
  // system for the cost of one lerp.
  if (shore.has(i)) colour.lerp(new THREE.Color(SAND), 0.42);
  if (WATER.has(kind)) {
    if (dist[i] === 0) colour.lerp(new THREE.Color(FOAM), 0.3);
    else colour.multiplyScalar(1 - Math.min(dist[i], 4) * 0.11);
  }
  return colour;
}

/**
 * The board as merged geometry, one mesh per terrain type.
 *
 * Instancing was the obvious choice and it is the wrong one here. An
 * `InstancedMesh` shares a single geometry across every copy, so a tile can
 * only be one flat colour - which is why every terrain boundary was a hard
 * edge. Merging instead gives every *vertex* its own colour, and colouring the
 * six corners of a hex by the average of the tiles meeting at that corner
 * makes the transition blend. Adjacent tiles agree on the shared corner, so
 * the gradient is continuous across the seam: a real blend, not a blend
 * texture. Freeciv needs a hand-authored transition sprite per terrain pair to
 * get the same effect.
 *
 * Still one mesh per terrain type, so each keeps its own surface texture, and
 * still ~8 draw calls. Vertex colours interpolate across a triangle regardless
 * of `flatShading`, so the lighting stays faceted while the colour is smooth.
 *
 * Geometry note that predates this: the hex is built at radius exactly HEX
 * with corner k at (sin(k*60deg), cos(k*60deg)), which is pointy-top - the
 * orientation this axial layout is spaced for. An earlier version rotated the
 * tiles 30 degrees, which made every column flat-top while the spacing stayed
 * pointy-top, so neighbours overlapped along one axis and gapped along the
 * others. At this orientation the seam is exactly zero.
 */
export function buildTerrain(tiles, terrain) {
  const group = new THREE.Group();
  const near = neighbours(tiles);
  const shore = new Set();
  terrain.forEach((kind, i) => {
    if (WATER.has(kind) || kind === "mountains") return;
    if (near[i].some((j) => j >= 0 && WATER.has(terrain[j]))) shore.add(i);
  });
  const dist = depth(tiles, terrain, near);
  const colours = tiles.map((_, i) => tileColour(i, terrain, shore, dist));

  // Corner k is shared with the neighbours across the two edges that meet
  // there - EDGES[k] and EDGES[k-1] both contain it.
  // The tile's own colour is weighted double. At an equal three-way average a
  // lone desert tile surrounded by grassland lost its identity entirely, and
  // the point of the blend is to soften the seam, not to erase what the tile is
  // - a spectator still has to be able to read terrain off the board.
  const OWN = 2;
  const cornerColour = (i, k) => {
    const mix = scratch.copy(colours[i]).multiplyScalar(OWN);
    let n = OWN;
    for (const d of [k, (k + 5) % 6]) {
      const j = near[i][d];
      if (j < 0) continue;
      mix.add(colours[j]);
      n++;
    }
    return mix.multiplyScalar(1 / n);
  };

  const byKind = new Map();
  terrain.forEach((kind, i) => {
    if (!byKind.has(kind)) byKind.set(kind, []);
    byKind.get(kind).push(i);
  });

  for (const [kind, indices] of byKind) {
    const water = WATER.has(kind);
    // 6 top triangles plus 6 side quads: 18 + 36 vertices per tile.
    const verts = indices.length * 54;
    const pos = new Float32Array(verts * 3);
    const col = new Float32Array(verts * 3);
    const uv = new Float32Array(verts * 2);
    const tileOf = new Uint32Array(verts);
    let v = 0;

    const push = (x, y, z, c, tile) => {
      pos.set([x, y, z], v * 3);
      col.set([c.r, c.g, c.b], v * 3);
      // Planar UV in world space rather than per tile, so the surface pattern
      // runs continuously across neighbouring tiles of the same terrain
      // instead of repeating once per hex.
      uv.set([(x + z * 0.5) / 2.6, (z + y) / 2.6], v * 2);
      tileOf[v] = tile;
      v++;
    };

    for (const i of indices) {
      const [q, r] = tiles[i].split(",").map(Number);
      const [cx, , cz] = axialToWorld(q, r);
      const h = HEIGHT[kind] ?? 0.36;
      const own = colours[i];
      const corners = [0, 1, 2, 3, 4, 5].map((k) => corner(k, HEX));
      const tips = [0, 1, 2, 3, 4, 5].map((k) => cornerColour(i, k).clone());
      // Cliff faces darken toward the base, which is what gives the relief
      // its depth at a low camera angle.
      const foot = own.clone().multiplyScalar(0.62);

      for (let k = 0; k < 6; k++) {
        const [ax, az] = corners[k];
        const [bx, bz] = corners[(k + 1) % 6];
        // Top face, wound anticlockwise from above so the normal points up.
        push(cx, h, cz, own, i);
        push(cx + ax, h, cz + az, tips[k], i);
        push(cx + bx, h, cz + bz, tips[(k + 1) % 6], i);
        // Side wall, wound so the normal points away from the tile centre.
        push(cx + ax, h, cz + az, tips[k], i);
        push(cx + ax, 0, cz + az, foot, i);
        push(cx + bx, 0, cz + bz, foot, i);
        push(cx + ax, h, cz + az, tips[k], i);
        push(cx + bx, 0, cz + bz, foot, i);
        push(cx + bx, h, cz + bz, tips[(k + 1) % 6], i);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    geo.computeVertexNormals();

    // Water is the one surface that has to be shiny. Lambert is purely
    // diffuse, so ocean rendered as painted clay no matter what colour or
    // texture it was given; a specular highlight is the single cue that says
    // liquid rather than solid.
    const material = water
      ? new THREE.MeshPhongMaterial({
          vertexColors: true, flatShading: true, map: terrainTexture(kind),
          shininess: 80, specular: 0x9fd6ff,
        })
      : new THREE.MeshLambertMaterial({
          vertexColors: true, flatShading: true, map: terrainTexture(kind),
        });

    const mesh = new THREE.Mesh(geo, material);
    mesh.receiveShadow = true;
    mesh.castShadow = !water;
    mesh.userData = { tileOf, base: col.slice(), paints: true };
    group.add(mesh);
  }
  group.userData.paint = (shade) => paintVertices(group, shade);
  return group;
}

/**
 * Re-tint by tile, which is how fog of war is applied.
 *
 * Rebuilding the board on every focus change would drop frames on a 1000-tile
 * map; multiplying the existing colour buffer in place is close to free.
 */
function paintVertices(group, shade) {
  for (const mesh of group.children) {
    const { tileOf, base } = mesh.userData;
    const attr = mesh.geometry.attributes.color;
    for (let v = 0; v < tileOf.length; v++) {
      const s = shade(tileOf[v]);
      attr.array[v * 3] = base[v * 3] * s;
      attr.array[v * 3 + 1] = base[v * 3 + 1] * s;
      attr.array[v * 3 + 2] = base[v * 3 + 2] * s;
    }
    attr.needsUpdate = true;
  }
}

/** The instanced version, for the scatter and resource layers. */
function paintInstances(group, shade) {
  for (const mesh of group.children) {
    const { indices, base } = mesh.userData;
    indices.forEach((tile, n) => {
      mesh.setColorAt(n, scratch.copy(base[n]).multiplyScalar(shade(tile)));
    });
    mesh.instanceColor.needsUpdate = true;
  }
}

/**
 * A mountain peak: a cone with its vertices pushed off the cone, and a snow
 * line baked into the vertex colours.
 *
 * A smooth cone reads as an ice-cream cone at any scale. What says "mountain"
 * is an irregular silhouette and a pale cap over a dark base - the two things
 * both Freeciv's sprites and Civ's models lean on. The displacement is derived
 * from vertex position, so the same peak comes out identical on every replay,
 * and the geometry is indexed, so neighbouring faces move together and the
 * surface never cracks open.
 */
function crag() {
  const geo = new THREE.CylinderGeometry(0.02, 0.7, 1.3, 7, 3);
  geo.translate(0, 0.65, 0);
  const pos = geo.attributes.position;
  const rock = new THREE.Color(0x6b6873);
  const snow = new THREE.Color(0xf1f4f9);
  const colours = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const n = rand(Math.round((x * 911 + z * 1373 + y * 577) * 64));
    pos.setXYZ(i, x * (1 + (n - 0.5) * 0.6), y * (0.92 + n * 0.16), z * (1 + (n - 0.5) * 0.6));
    const t = Math.min(1, Math.max(0, (pos.getY(i) - 0.78) / 0.4));
    scratch.copy(rock).lerp(snow, t).toArray(colours, i * 3);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(colours, 3));
  geo.computeVertexNormals();
  return geo;
}

/** Trees, peaks and scrub standing on the terrain. */
export function buildScatter(tiles, terrain) {
  const group = new THREE.Group();

  const pine = mergeGeometries([
    cyl(0, 0.26, 0.62, 7).translate(0, 0.46, 0),
    cyl(0, 0.19, 0.42, 7).translate(0, 0.78, 0),
    cyl(0.045, 0.055, 0.2, 5).translate(0, 0.1, 0),
  ]);
  const scrub = cyl(0, 0.13, 0.2, 5).translate(0, 0.1, 0);

  const kinds = [
    { geo: pine, colour: 0x2f5f34, on: "forest", per: 3, spread: 0.42 },
    // Three overlapping peaks per tile at different scales, so a range has a
    // broken skyline instead of a row of identical cones.
    { geo: crag(), colour: 0xffffff, on: "mountains", per: 3, spread: 0.36, tinted: true },
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
      // A tinted kind carries its own vertex colours - snow over rock - and
      // takes white as its instance colour so fog can still darken it.
      new THREE.MeshLambertMaterial({ flatShading: true, vertexColors: !!kind.tinted }),
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
  group.userData.paint = (shade) => paintInstances(group, shade);
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
/** A four-legged animal, which both the deer and the wild horses are built on. */
function beast({ hide, length = 0.26, height = 0.17, antlers = false }) {
  const parts = [
    tint(box(length, 0.1, 0.1).translate(0, height, 0), hide),
    tint(box(0.07, 0.13, 0.07).rotateZ(-0.4).translate(length * 0.42, height + 0.08, 0), hide),
    tint(box(0.1, 0.06, 0.065).translate(length * 0.58, height + 0.13, 0), hide),
    ...[-1, 1].flatMap((sx) => [0.04, -0.04].map((z) =>
      tint(cyl(0.018, 0.014, height, 4).translate(sx * length * 0.34, height / 2, z), hide))),
    tint(cyl(0, 0.022, 0.08, 4).rotateZ(0.6).translate(-length * 0.52, height + 0.05, 0), hide),
  ];
  if (antlers) {
    for (const z of [0.03, -0.03]) {
      parts.push(tint(cyl(0.008, 0.008, 0.11, 4).rotateZ(-0.3).translate(
        length * 0.55, height + 0.21, z), hide));
      for (const tip of [-0.03, 0.03]) {
        parts.push(tint(cyl(0, 0.012, 0.06, 4).rotateZ(tip * 12).translate(
          length * 0.55 + tip, height + 0.28, z), hide));
      }
    }
  }
  return mergeGeometries(parts);
}

const RESOURCE_MODELS = {
  wheat: () => mergeGeometries([0, 1, 2].map((k) =>
    tint(cyl(0, 0.05, 0.26, 5).rotateZ((k - 1) * 0.3).translate((k - 1) * 0.06, 0.13, 0),
      THATCH))),
  iron: () => material(new THREE.OctahedronGeometry(0.11).translate(0, 0.1, 0),
    0x7f858e, SURFACE.stone),
  gold_ore: () => material(new THREE.OctahedronGeometry(0.1).translate(0, 0.1, 0),
    0xffcf3d, SURFACE.metal),
  horses: () => beast({ hide: HORSEHIDE, length: 0.3, height: 0.19 }),
  deer: () => beast({ hide: DEERHIDE, antlers: true }),
  fish: () => mergeGeometries([
    material(new THREE.SphereGeometry(0.09, 7, 5).scale(1.5, 0.7, 0.8).translate(0, 0.04, 0),
      0x86d8ea, SURFACE.scales),
    material(cyl(0, 0.07, 0.12, 4).rotateZ(Math.PI / 2).translate(-0.16, 0.04, 0),
      0x86d8ea, SURFACE.scales),
  ]),
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
  const white = new THREE.Color(0xffffff);
  for (const [kind, targets] of byKind) {
    const mesh = new THREE.InstancedMesh(
      RESOURCE_MODELS[kind](),
      // Hue and grain are baked into the geometry, so the instance colour is
      // white and exists only so fog can darken it.
      new THREE.MeshLambertMaterial({
        flatShading: true, vertexColors: true, map: surfaceAtlas(),
      }),
      targets.length
    );
    const base = [];
    targets.forEach((i, n) => {
      const [q, r] = tiles[i].split(",").map(Number);
      const [x, , z] = axialToWorld(q, r);
      m.makeRotationY((i % 6) * 1.05);
      m.setPosition(x - 0.34, HEIGHT[terrain[i]] ?? 0.36, z + 0.42);
      mesh.setMatrixAt(n, m);
      mesh.setColorAt(n, white);
      base.push(white.clone());
    });
    mesh.instanceMatrix.needsUpdate = true;
    mesh.castShadow = true;
    mesh.userData = { indices: targets, base };
    group.add(mesh);
  }
  group.userData.paint = (shade) => paintInstances(group, shade);
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

/**
 * Units are people, not tokens.
 *
 * The previous models were chess pawns - a tapered cylinder with a sphere on
 * top - and they read as pieces on a board because that is exactly what they
 * were. Texture would not have fixed that. Three things make a unit read as a
 * character instead, and none of them is resolution:
 *
 *  1. **More than one of them.** Civ puts three to six figures on a tile. One
 *     figure is a game piece; a squad is an army.
 *  2. **Human articulation.** Head, torso, two arms, two legs, a held weapon.
 *     Even at ~120 triangles that silhouette is unmistakably a person.
 *  3. **Material separation.** Skin, cloth, leather and steel at different
 *     values. A model in one flat colour is a token no matter its shape.
 *
 * Point 3 is why every model is split in two. `InstancedMesh.setColorAt`
 * multiplies the *whole* mesh, so tinting a unit by its civ colour tinted the
 * soldier's skin and his sword blade too - which is most of why they looked
 * like counters. The `body` half carries baked vertex colours and is never
 * tinted; the `livery` half - tunics, shields, sails, wagon canvas - is the
 * only part that takes the civ colour. Two draw calls per unit type.
 */
const SKIN = 0xc08a5e;
const LEATHER = 0x8a6440;
const WOOD = 0x8a6338;
const STEEL = 0xb4bcc6;
const HORSEHIDE = 0x6f4b31;
const DEERHIDE = 0xa8763f;
const FUR = 0x6e6a63;
const EMBER = 0xff8a3d;
const STONE = 0x9c968c;
const PLASTER = 0xf2ead6;
const THATCH = 0xb59457;

const box = (w, h, d) => new THREE.BoxGeometry(w, h, d);

/**
 * Which grain goes with which material.
 *
 * A material's colour and its surface are the same fact - steel is grey *and*
 * brushed, thatch is straw-coloured *and* strawy - so the atlas cell follows
 * from the colour rather than being passed at every call site.
 */
const GRAIN = new Map([
  [SKIN, SURFACE.skin], [LEATHER, SURFACE.leather], [WOOD, SURFACE.wood],
  [STEEL, SURFACE.metal], [HORSEHIDE, SURFACE.hide], [DEERHIDE, SURFACE.hide],
  [FUR, SURFACE.fur], [STONE, SURFACE.stone], [PLASTER, SURFACE.plaster],
  [THATCH, SURFACE.thatch], [EMBER, SURFACE.plain],
]);

/** Paint every vertex of a part one colour and point it at the matching atlas
 *  cell, so parts of different materials can be merged into a single geometry
 *  and still shade and texture separately. */
function tint(geo, hex) {
  return material(geo, hex, GRAIN.get(hex) ?? SURFACE.plain);
}

/** Move a finished set of parts into position on the tile. */
function place(parts, x, z, yaw) {
  for (const g of parts) {
    if (yaw) g.rotateY(yaw);
    g.translate(x, 0, z);
  }
  return parts;
}

/**
 * One person, roughly 0.42 units tall against a hex of radius 1.
 *
 * `stride` and the arm angles are what give a squad the look of a formation
 * rather than a row of identical statues - each figure is posed slightly
 * differently by its index.
 */
function person({ stride = 0, leftArm = 0.12, rightArm = 0.12, scale = 1 } = {}) {
  const body = [
    tint(box(0.045, 0.17, 0.055).rotateX(stride).translate(-0.042, 0.085, 0), LEATHER),
    tint(box(0.045, 0.17, 0.055).rotateX(-stride).translate(0.042, 0.085, 0), LEATHER),
    tint(box(0.036, 0.15, 0.045).rotateZ(leftArm).translate(-0.086, 0.245, 0), SKIN),
    tint(box(0.036, 0.15, 0.045).rotateZ(-rightArm).translate(0.086, 0.245, 0), SKIN),
    // A faceted sphere, not a cube. A cube head is the single detail that makes
    // a low-poly figure read as a toy block rather than as a person.
    tint(new THREE.SphereGeometry(0.048, 6, 4).translate(0, 0.375, 0), SKIN),
  ];
  // A belt, which is worth its four triangles: it breaks the torso into two
  // shapes and puts a second material at the waist, and without it the tunic
  // is one flat slab and the figure reads as a sign board with legs.
  body.push(tint(box(0.112, 0.026, 0.088).translate(0, 0.185, 0), LEATHER));
  // Livery carries no vertex colour - the civ colour arrives as the instance
  // colour - so it takes an atlas cell only.
  const livery = [surface(box(0.105, 0.165, 0.082).translate(0, 0.255, 0), SURFACE.weave)];
  if (scale !== 1) for (const g of [...body, ...livery]) g.scale(scale, scale, scale);
  return { body, livery };
}

// Weapons and kit. Held at the right hand, which sits at +x.
const SPEAR = () => [
  tint(cyl(0.011, 0.011, 0.54, 4).translate(0.105, 0.3, 0.02), WOOD),
  tint(cyl(0, 0.026, 0.09, 4).translate(0.105, 0.6, 0.02), STEEL),
];
const SWORD = () => [
  tint(box(0.022, 0.21, 0.05).rotateZ(-0.25).translate(0.13, 0.34, 0.02), STEEL),
  tint(box(0.028, 0.04, 0.055).translate(0.105, 0.23, 0.02), LEATHER),
];
const CLUB = () => [
  tint(cyl(0.014, 0.014, 0.24, 4).rotateZ(-0.5).translate(0.13, 0.3, 0.02), WOOD),
  tint(box(0.06, 0.07, 0.06).translate(0.18, 0.4, 0.02), STEEL),
];
// Rotated so the arc straddles the hand rather than rising out of it - the
// half-torus spans only +Y as built, which read as a horn above the head.
const BOW = () => [
  tint(new THREE.TorusGeometry(0.105, 0.011, 3, 9, Math.PI)
    .rotateZ(-Math.PI / 2).rotateY(Math.PI / 2).translate(0.12, 0.27, 0.02), WOOD),
];
const TORCH = () => [
  tint(cyl(0.012, 0.012, 0.28, 4).translate(0.12, 0.32, 0.02), WOOD),
  tint(cyl(0, 0.04, 0.1, 5).translate(0.12, 0.5, 0.02), EMBER),
];
// Shields are livery, not body: a round disc of civ colour at chest height is
// the single most legible identity cue on a battlefield seen from above. A
// cylinder's axis is Y, so one rotateZ turns the disc to face forward (+x),
// which is the direction every figure is built looking.
const SHIELD = () => [
  surface(cyl(0.095, 0.095, 0.022, 8).rotateZ(Math.PI / 2).translate(0.02, 0.26, 0.11),
    SURFACE.scales),
];

/** Four figures in a loose block, plus a fifth at the centre for larger units.
 *  Spread wide enough to read as separate men at the zoom a match is watched
 *  from; any tighter and a squad merges into one blob. */
const FORMATION = [[-0.32, -0.26], [0.28, -0.32], [-0.26, 0.32], [0.32, 0.26], [0, 0]];

function squad(count, make) {
  const body = [];
  const livery = [];
  for (let k = 0; k < count; k++) {
    const [x, z] = FORMATION[k % FORMATION.length];
    const p = make(k);
    // Each figure is turned a little differently, so a formation looks held
    // rather than stamped.
    place(p.body, x, z, ((k * 37) % 40) / 100 - 0.2);
    place(p.livery, x, z, ((k * 37) % 40) / 100 - 0.2);
    body.push(...p.body);
    livery.push(...p.livery);
  }
  return { body, livery };
}

// A helmet is livery, and it earns its place twice: a second patch of civ
// colour above the tunic, and a rounder silhouette at the top of the figure.
const HELM = () => [
  surface(new THREE.SphereGeometry(0.054, 6, 3, 0, Math.PI * 2, 0, Math.PI / 1.9)
    .translate(0, 0.372, 0), SURFACE.mail),
];

const soldier = (k, kit, worn) => {
  const p = person({ stride: (k % 3) * 0.12 - 0.12, rightArm: 0.3 });
  p.body.push(...kit());
  p.livery.push(...HELM());
  if (worn) p.livery.push(...worn());
  return p;
};

function horse(rider) {
  const body = [
    tint(box(0.32, 0.14, 0.13).translate(0, 0.3, 0), HORSEHIDE),
    tint(box(0.09, 0.16, 0.09).rotateZ(-0.35).translate(0.16, 0.4, 0), HORSEHIDE),
    tint(box(0.13, 0.075, 0.085).translate(0.24, 0.45, 0), HORSEHIDE),
    ...[-0.11, 0.11].flatMap((x) => [0.05, -0.05].map((z) =>
      tint(cyl(0.026, 0.022, 0.26, 5).translate(x, 0.13, z), HORSEHIDE))),
  ];
  const livery = [];
  if (rider) {
    const p = person({ stride: 0.5, scale: 0.85 });
    for (const g of p.body) g.translate(-0.03, 0.34, 0);
    for (const g of p.livery) g.translate(-0.03, 0.34, 0);
    body.push(...p.body, ...SWORD().map((g) => g.translate(-0.03, 0.34, 0)));
    livery.push(...p.livery);
  }
  return { body, livery };
}

/** Everything above is authored at a person height of ~0.42 because that keeps
 *  the part offsets readable. This is the one place the whole model is scaled
 *  to the board, so the figures can be resized without retuning every limb. */
const UNIT_SCALE = 1.45;

/** Built once per type: merging a squad on every turn step would be wasteful. */
const models = new Map();

export function unitModel(type) {
  if (models.has(type)) return models.get(type);
  const model = buildUnit(type);
  const merge = (parts) => {
    if (!parts.length) return null;
    return mergeGeometries(parts).scale(UNIT_SCALE, UNIT_SCALE, UNIT_SCALE);
  };
  const merged = { body: merge(model.body), livery: merge(model.livery) };
  models.set(type, merged);
  return merged;
}

function buildUnit(type) {
  switch (type) {
    case "warrior":
      return squad(4, (k) => soldier(k, CLUB));
    case "spearman":
      return squad(4, (k) => soldier(k, SPEAR));
    case "archer":
      return squad(3, (k) => soldier(k, BOW));
    case "swordsman":
      return squad(4, (k) => soldier(k, SWORD, SHIELD));
    case "scout": {
      const p = person({ stride: 0.35, leftArm: 0.4 });
      p.body.push(tint(cyl(0.01, 0.01, 0.46, 4).translate(0.1, 0.26, 0.02), WOOD));
      return p;
    }
    case "worker": {
      const a = person({ stride: 0.1, rightArm: 0.8 });
      a.body.push(tint(cyl(0.012, 0.012, 0.3, 4).rotateZ(-1).translate(0.16, 0.26, 0), WOOD));
      a.body.push(tint(box(0.11, 0.03, 0.05).rotateZ(-1).translate(0.26, 0.34, 0), STEEL));
      const b = person({ stride: -0.2 });
      place(a.body, -0.12, 0.06, 0.3);
      place(a.livery, -0.12, 0.06, 0.3);
      place(b.body, 0.14, -0.08, -0.5);
      place(b.livery, 0.14, -0.08, -0.5);
      return { body: [...a.body, ...b.body], livery: [...a.livery, ...b.livery] };
    }
    case "settler": {
      // A wagon with a civ-coloured canvas, two walkers alongside. The canopy
      // is the largest single patch of civ colour on the board, which is what
      // makes a settler easy to track across a continent.
      const body = [
        tint(box(0.36, 0.13, 0.21).translate(0, 0.22, 0), WOOD),
        ...[-0.14, 0.14].flatMap((x) => [0.12, -0.12].map((z) =>
          tint(cyl(0.075, 0.075, 0.028, 9).rotateX(Math.PI / 2).translate(x, 0.11, z), LEATHER))),
      ];
      const livery = [surface(
        new THREE.CylinderGeometry(0.14, 0.14, 0.32, 9, 1, false, 0, Math.PI)
          .rotateZ(Math.PI / 2).translate(0, 0.3, 0), SURFACE.canvas)];
      const walker = person({ stride: 0.3, scale: 0.9 });
      place(walker.body, 0.3, 0.12, -0.4);
      place(walker.livery, 0.3, 0.12, -0.4);
      return { body: [...body, ...walker.body], livery: [...livery, ...walker.livery] };
    }
    case "horseman": {
      const body = [];
      const livery = [];
      for (const [k, [x, z]] of [[0, [-0.1, -0.1]], [1, [0.12, 0.12]]]) {
        const h = horse(true);
        place(h.body, x, z, k * 0.3 - 0.15);
        place(h.livery, x, z, k * 0.3 - 0.15);
        body.push(...h.body);
        livery.push(...h.livery);
      }
      return { body, livery };
    }
    case "catapult": {
      const body = [
        tint(box(0.38, 0.09, 0.24).translate(0, 0.16, 0), WOOD),
        ...[-0.13, 0.13].flatMap((x) => [0.13, -0.13].map((z) =>
          tint(cyl(0.085, 0.085, 0.03, 10).rotateX(Math.PI / 2).translate(x, 0.09, z), LEATHER))),
        tint(cyl(0.022, 0.022, 0.42, 5).rotateZ(-0.85).translate(0.05, 0.36, 0), WOOD),
        tint(new THREE.SphereGeometry(0.075, 7, 5).translate(0.22, 0.52, 0), STEEL),
      ];
      const crew = [0, 1].map((k) => {
        const p = person({ stride: 0.2, rightArm: 0.9, scale: 0.9 });
        place(p.body, -0.26, k ? 0.16 : -0.16, k ? 0.6 : -0.6);
        place(p.livery, -0.26, k ? 0.16 : -0.16, k ? 0.6 : -0.6);
        return p;
      });
      return {
        body: [...body, ...crew.flatMap((p) => p.body)],
        livery: crew.flatMap((p) => p.livery),
      };
    }
    case "trireme": {
      const body = [
        tint(cyl(0.15, 0.1, 0.64, 6).rotateZ(Math.PI / 2).translate(0, 0.12, 0), WOOD),
        tint(cyl(0.018, 0.018, 0.5, 5).translate(0, 0.38, 0), WOOD),
        // Oars, which are what say "ship" rather than "boat-shaped object".
        ...[-0.18, 0, 0.18].flatMap((x) => [0.13, -0.13].map((z) =>
          tint(cyl(0.009, 0.009, 0.26, 4).rotateX(z > 0 ? 0.9 : -0.9).translate(x, 0.11, z), WOOD))),
      ];
      const livery = [surface(box(0.012, 0.28, 0.3).translate(0.01, 0.42, 0), SURFACE.canvas)];
      const rower = person({ scale: 0.7 });
      place(rower.body, -0.1, 0, 0);
      place(rower.livery, -0.1, 0, 0);
      for (const g of [...rower.body, ...rower.livery]) g.translate(0, 0.1, 0);
      return { body: [...body, ...rower.body], livery: [...livery, ...rower.livery] };
    }
    case "wolf": {
      const body = [];
      for (const [k, [x, z]] of [[0, [-0.14, -0.1]], [1, [0.14, 0.12]]]) {
        const parts = [
          tint(box(0.3, 0.12, 0.12).translate(0, 0.19, 0), FUR),
          tint(box(0.11, 0.1, 0.1).translate(0.18, 0.23, 0), FUR),
          // Muzzle and ears. Without them the head is a cube and the animal
          // could be anything with four legs.
          tint(box(0.07, 0.055, 0.06).translate(0.27, 0.2, 0), FUR),
          ...[0.032, -0.032].map((ez) =>
            tint(cyl(0, 0.028, 0.06, 4).translate(0.155, 0.3, ez), FUR)),
          tint(cyl(0, 0.035, 0.15, 4).rotateZ(-0.9).translate(-0.19, 0.25, 0), FUR),
          ...[-0.1, 0.1].flatMap((bx) => [0.045, -0.045].map((bz) =>
            tint(cyl(0.024, 0.02, 0.19, 4).translate(bx, 0.09, bz), FUR))),
        ];
        body.push(...place(parts, x, z, k * 0.8 - 0.4));
      }
      return { body, livery: [] };
    }
    case "barbarian":
      return squad(3, (k) => {
        const p = person({ stride: (k % 2) * 0.3 - 0.15, rightArm: 0.5 });
        p.body.push(...(k === 1 ? TORCH() : CLUB()));
        return p;
      });
    default:
      return squad(2, (k) => soldier(k, CLUB));
  }
}

/**
 * A ring marking which civ a unit belongs to, read at any camera angle.
 *
 * An outline rather than a filled disc. As a disc it was a saucer of solid civ
 * colour wider than the men standing on it, and it took over the tile now that
 * the figures carry their own livery.
 *
 * Wide enough to enclose the formation: at the old radius the outer men of a
 * squad stood outside their own marker, which read as several separate units
 * rather than one.
 */
export function baseGeometry() {
  return new THREE.TorusGeometry(0.64, 0.022, 4, 20).rotateX(Math.PI / 2);
}

// ---------------------------------------------------------------------------
// Settlements
// ---------------------------------------------------------------------------

/**
 * Towns, not camps.
 *
 * The previous version was a scatter of pale boxes with pyramid caps, all one
 * colour because the whole building took the civ tint. That is why they read
 * as tents: a building is legible as a building precisely because its roof is
 * a different material from its walls, and flattening both to one hue removed
 * the only cue that mattered.
 *
 * So a settlement is built the same way a unit is. **Walls are body** - stone
 * footing, timbered plaster - with their colours and grain baked in and never
 * tinted. **Roofs are livery** and carry the civ colour, which turns out to be
 * the better place for it anyway: a roofline is what you see of a town from a
 * camera looking down, so an empire's colour reads from directly above without
 * a single flag or plate on the board.
 */
function building(w, h, d, roofHeight) {
  const walls = mergeGeometries([
    tint(box(w + 0.035, 0.035, d + 0.035).translate(0, 0.018, 0), STONE),
    material(box(w, h, d).translate(0, 0.035 + h / 2, 0), PLASTER, SURFACE.timber),
  ]);
  // A hip roof: a four-sided pyramid turned so its eaves run parallel to the
  // walls, sized to overhang them by a tenth, then stretched along the long axis.
  const roof = cyl(0, d * 0.78, roofHeight, 4).rotateY(Math.PI / 4);
  roof.scale(w / d, 1, 1);
  roof.translate(0, 0.035 + h + roofHeight / 2, 0);
  return { walls, roof: surface(roof, SURFACE.tile) };
}

// Walls tall, roofs shallow. The camera looks down at maybe forty degrees, so
// a roof as tall as its walls hides them completely and the town goes back to
// being a cluster of coloured cones - the two-tone read only works if there is
// enough wall left to see.
const BUILDINGS = {
  hall: () => building(0.42, 0.32, 0.26, 0.15),
  cottage: () => building(0.24, 0.25, 0.22, 0.12),
  hut: () => building(0.18, 0.19, 0.18, 0.11),
};

/** The paved ground a town stands on, which is what stops it looking like
 *  buildings dropped onto open grass. */
function plazaGeometry() {
  return material(cyl(0.8, 0.84, 0.05, 14).translate(0, 0.025, 0), STONE, SURFACE.stone);
}

/** A curtain wall with merlons. Visibly a fortification from above and from
 *  the side, which is the point of showing what a civ chose to build. */
function rampartGeometry() {
  const parts = [
    material(new THREE.CylinderGeometry(0.84, 0.9, 0.28, 16, 1, true).translate(0, 0.14, 0),
      STONE, SURFACE.stone),
  ];
  for (let k = 0; k < 16; k++) {
    const a = (k / 16) * Math.PI * 2;
    parts.push(material(
      box(0.1, 0.08, 0.06).rotateY(-a).translate(Math.sin(a) * 0.85, 0.31, Math.cos(a) * 0.85),
      STONE, SURFACE.stone));
  }
  return mergeGeometries(parts);
}

/** A temple, library or wonder: a stone keep under a civ-coloured spire. */
function keepModel() {
  return {
    walls: mergeGeometries([
      material(cyl(0.11, 0.14, 0.52, 8).translate(0, 0.26, 0), STONE, SURFACE.stone),
      material(cyl(0.17, 0.17, 0.04, 8).translate(0, 0.52, 0), STONE, SURFACE.stone),
    ]),
    roof: surface(cyl(0, 0.17, 0.26, 8).translate(0, 0.67, 0), SURFACE.tile),
  };
}

/**
 * Where the buildings of a town sit, given its population.
 *
 * Deterministic: the same city has the same street plan on every replay, so
 * scrubbing back and forth does not rearrange the town. Every building faces
 * the centre, which is what makes a cluster read as a settlement with a square
 * rather than as houses dropped at random angles.
 */
export function cityLayout(population, seed) {
  const count = Math.min(6, 1 + Math.floor(population / 1.8));
  const spots = [{ x: 0, z: 0, yaw: ((seed % 4) * Math.PI) / 2, variant: "hall" }];
  for (let k = 1; k < count; k++) {
    const a = ((seed * 53 + k * 137) % 360) * (Math.PI / 180);
    const d = 0.4 + ((seed * 17 + k * 41) % 26) / 100;
    spots.push({
      x: Math.cos(a) * d,
      z: Math.sin(a) * d,
      yaw: Math.atan2(-Math.cos(a), -Math.sin(a)),
      variant: k % 3 === 0 ? "hut" : "cottage",
    });
  }
  return spots;
}

/**
 * Every settlement on the board, batched by part rather than by city.
 *
 * A mesh per city would be one draw call per city; grouping by building type
 * keeps it to about ten for the whole map however many towns get founded.
 */
export function buildSettlements(cities, locate, colourOf) {
  const group = new THREE.Group();
  if (!cities.length) return group;

  const white = new THREE.Color(0xffffff);
  const M = new THREE.Matrix4();
  const add = (geo, rows, civ) => {
    if (!rows.length) return;
    const mesh = new THREE.InstancedMesh(
      geo,
      new THREE.MeshLambertMaterial({
        flatShading: true, vertexColors: !civ, map: surfaceAtlas(),
        side: civ ? THREE.FrontSide : THREE.DoubleSide,
      }),
      rows.length
    );
    rows.forEach(([city, x, y, z, yaw], n) => {
      M.makeRotationY(yaw).setPosition(x, y, z);
      mesh.setMatrixAt(n, M);
      mesh.setColorAt(n, civ ? scratch.setHex(colourOf(city.owner)) : white);
    });
    mesh.instanceMatrix.needsUpdate = true;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);
  };

  const plazas = [];
  const byVariant = new Map(Object.keys(BUILDINGS).map((k) => [k, []]));
  const ramparts = [];
  const keeps = [];
  const SPIRES = ["temple", "great_library", "pyramids", "apex_project"];

  cities.forEach((city, n) => {
    const [x, y, z] = locate(city.at);
    plazas.push([city, x, y, z, 0]);
    for (const spot of cityLayout(city.population, n + city.at)) {
      byVariant.get(spot.variant).push([city, x + spot.x, y, z + spot.z, spot.yaw]);
    }
    if (city.buildings.includes("walls")) ramparts.push([city, x, y, z, 0]);
    if (city.buildings.some((b) => SPIRES.includes(b))) keeps.push([city, x, y, z, 0]);
  });

  add(plazaGeometry(), plazas, false);
  for (const [variant, rows] of byVariant) {
    if (!rows.length) continue;
    const { walls, roof } = BUILDINGS[variant]();
    add(walls, rows, false);
    add(roof, rows, true);
  }
  add(rampartGeometry(), ramparts, false);
  if (keeps.length) {
    const keep = keepModel();
    add(keep.walls, keeps, false);
    add(keep.roof, keeps, true);
  }
  return group;
}

/**
 * A standing banner in each empire's territory naming who it is bound to.
 *
 * The treaty system was the hardest thing in this project to see. It was
 * invisible in the bundle for a while, then invisible in the panel, and even
 * once both were fixed it lived in a line of text in a collapsed section while
 * the board - the thing anyone actually looks at - showed four civs expanding
 * past each other with no indication that two of them had agreed not to fight.
 *
 * So it goes on the map. A pole in the civ's own colour, a banner reading PACT
 * or ALLIANCE, and a row of dots naming the partners in *their* colours, which
 * is the same encoding the panels and the board already use.
 *
 * A `Sprite` rather than a plane, because the camera orbits and a flat banner
 * spends half of every orbit edge-on and unreadable. Sprites always face the
 * viewer, which is exactly the property a label wants and exactly the property
 * a flag does not - so this reads as a standard rather than as cloth, and that
 * is the right trade for something whose whole job is to be legible.
 */
export function buildPacts(tiles, owners, bonds, locate, colourOf) {
  const group = new THREE.Group();
  if (!bonds.size) return group;

  // The banner is planted at the *medoid* of the claim, not the mean. A mean
  // lands outside its own territory whenever an empire is concave - which is
  // the normal shape for a civ that has grown around a mountain - and a pact
  // marker floating in a neighbour's land says the opposite of what it means.
  const home = new Map();
  tiles.forEach((key, i) => {
    if (!owners[i] || !bonds.has(owners[i])) return;
    if (!home.has(owners[i])) home.set(owners[i], []);
    home.get(owners[i]).push(i);
  });

  for (const [who, claim] of home) {
    const spots = claim.map((i) => locate(i));
    const cx = spots.reduce((a, p) => a + p[0], 0) / spots.length;
    const cz = spots.reduce((a, p) => a + p[2], 0) / spots.length;
    let best = spots[0];
    let bestD = Infinity;
    for (const p of spots) {
      const d = (p[0] - cx) ** 2 + (p[2] - cz) ** 2;
      if (d < bestD) { bestD = d; best = p; }
    }

    const [x, y, z] = best;
    const partners = bonds.get(who);
    const POLE = 1.5;
    const pole = new THREE.Mesh(
      new THREE.CylinderGeometry(0.035, 0.045, POLE, 5),
      new THREE.MeshLambertMaterial({ color: colourOf(who), flatShading: true })
    );
    pole.position.set(x, y + POLE / 2, z);
    group.add(pole);

    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: banner(partners, colourOf),
      // Drawn over whatever it stands on. A banner half-sunk into a hill it is
      // planted on top of looks like a bug, and it is one tile of occlusion
      // against the one label on the board that summarises the diplomacy.
      depthTest: false,
      transparent: true,
    }));
    // Sized against the hex grid, not against the texture. At 1.9 wide - about
    // two tiles - the banner was geometrically correct and 20 pixels across at
    // the default framing, which is to say invisible. A label has to be read at
    // the zoom people actually use, so it is deliberately out of scale with the
    // world: five tiles wide, floating clear of the pole.
    sprite.scale.set(4.6, 1.73, 1);
    sprite.position.set(x, y + POLE + 0.86, z);
    sprite.renderOrder = 10;
    group.add(sprite);
  }
  return group;
}

const banners = new Map();

/** The banner face: a word and a dot per partner, drawn once per combination. */
function banner(partners, colourOf) {
  const key = partners.map((p) => `${p.who}:${p.kind}`).sort().join("|");
  if (banners.has(key)) return banners.get(key);

  const W = 256;
  const H = 96;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const g = c.getContext("2d");

  const allied = partners.some((p) => p.kind === "alliance");
  g.fillStyle = "rgba(13,17,23,0.92)";
  g.strokeStyle = allied ? "#4ade80" : "#cbd5e1";
  g.lineWidth = 4;
  round(g, 3, 3, W - 6, H - 6, 12);
  g.fill();
  g.stroke();

  g.fillStyle = allied ? "#4ade80" : "#e6edf3";
  g.font = "bold 34px ui-sans-serif, system-ui, sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(allied ? "ALLIANCE" : "PACT", W / 2, 32);

  // One dot per partner, in that partner's colour - the same key the board,
  // the panels and the diplomacy threads all use, so no legend is needed.
  const R = 9;
  const gap = 26;
  const startX = W / 2 - ((partners.length - 1) * gap) / 2;
  partners.forEach((p, n) => {
    g.beginPath();
    g.arc(startX + n * gap, 68, R, 0, Math.PI * 2);
    g.fillStyle = `#${colourOf(p.who).toString(16).padStart(6, "0")}`;
    g.fill();
  });

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  banners.set(key, tex);
  return tex;
}

function round(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}
