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
const LEATHER = 0x6b4a2f;
const WOOD = 0x8a6338;
const STEEL = 0xb4bcc6;
const HORSEHIDE = 0x6f4b31;
const FUR = 0x6e6a63;
const EMBER = 0xff8a3d;

const box = (w, h, d) => new THREE.BoxGeometry(w, h, d);

/** Paint every vertex of a part one colour, so parts of different materials
 *  can be merged into a single geometry and still shade separately. */
function tint(geo, hex) {
  const c = new THREE.Color(hex);
  const n = geo.attributes.position.count;
  const arr = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) c.toArray(arr, i * 3);
  geo.setAttribute("color", new THREE.BufferAttribute(arr, 3));
  return geo;
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
  const livery = [box(0.128, 0.17, 0.085).translate(0, 0.25, 0)];
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
  cyl(0.095, 0.095, 0.022, 8).rotateZ(Math.PI / 2).translate(0.02, 0.26, 0.11),
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
  new THREE.SphereGeometry(0.054, 6, 3, 0, Math.PI * 2, 0, Math.PI / 1.9)
    .translate(0, 0.372, 0),
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
      const livery = [new THREE.CylinderGeometry(0.14, 0.14, 0.32, 9, 1, false, 0, Math.PI)
        .rotateZ(Math.PI / 2).translate(0, 0.3, 0)];
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
      const livery = [box(0.012, 0.28, 0.3).translate(0.01, 0.42, 0)];
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
          tint(box(0.13, 0.11, 0.11).translate(0.19, 0.23, 0), FUR),
          tint(cyl(0, 0.035, 0.13, 4).rotateZ(-0.7).translate(-0.18, 0.26, 0), FUR),
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
