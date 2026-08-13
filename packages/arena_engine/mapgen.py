"""Procedural map generation with provably fair starts.

**On "mirrored" starts.** A hex grid has six-fold rotational symmetry, not
four-fold, so four players cannot be placed at exact rotations of one another.
Making the whole map symmetric anyway would also flatten the thing we most want
to watch: exploration only means something when there is something unknown out
there.

So fairness is local rather than global. One starting neighbourhood is
generated, and the same neighbourhood is then *stamped by translation* around
each of the four starts. Translation needs no rotational symmetry at all, so
every civ provably begins with an identical multiset of terrain, resources, and
yields within its working radius, while the rest of the map stays varied and
worth scouting. `test_mapgen.py` asserts that identity directly.

The four starts sit on ring corners 0, 1, 3 and 4. That arrangement gives every
player the same distance profile to its rivals - one near neighbour and two far
ones - so no civ is handed a safer corner than another.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena_engine import hex as hx
from arena_engine.content import (
    RESOURCE_TERRAIN,
    TERRAIN,
    Resource,
    Terrain,
)
from arena_engine.hex import Hex
from arena_engine.rng import Stream
from arena_engine.types import Tile

# The four ring corners used for starts. See the module docstring.
START_CORNERS = (0, 1, 3, 4)

# Fraction of the map radius at which starts sit.
#
# **This is deliberately decoupled from map size, and that matters.** When it
# was 0.62, enlarging the world automatically pushed the civs apart, and a
# measured sweep at radius 18 showed exactly what that costs: first contact slid
# from turn 9 to turn 78, wars per match fell from 4.8 to 1.5, and every match
# ended on an unopposed science victory. Four civs quietly teching in isolation
# is the one outcome that would make the whole lab measure nothing.
#
# Holding the opening distance fixed while the map grows gives the good half of
# a big world - room to expand, real terrain features, a contested periphery -
# without the buffer that suppresses interaction. At radius 18 with this value
# the civs found 26 cities instead of 17 and still meet inside 20 turns.
START_RADIUS_FRACTION = 0.38
# The stamped template covers exactly the city work radius and no more.
#
# Radius 3 was the first attempt and it was a mistake: 37 tiles times four
# starts is 148 of 469, so a third of the board became four identical copies -
# visibly repetitive, and it flattened the strategic variety the rest of the
# generator exists to create. Radius 2 is 19 tiles each, 16% of the map, and it
# still covers every tile a city can actually work, which is where fairness has
# to hold.
TEMPLATE_RADIUS = 2

# Closest two starts may ever be placed. Below this their stamped templates
# overlap, the later stamp overwrites the earlier one, and the civs silently
# stop having identical openings. Enforced in `_choose_starts`.
MIN_START_SEPARATION = 2 * TEMPLATE_RADIUS + 1

# How many smoothing passes the elevation and moisture fields get. Higher means
# larger, blobbier continents; 3 is where the map stops looking like static.
SMOOTHING_PASSES = 3

# How wide the surrounding ocean is, in tiles, and how hard the falloff pulls
# elevation down. RIM_DEPTH above 1.0 guarantees the outermost ring is water no
# matter how high its raw elevation was.
#
# A *width* rather than a fraction of the radius, which is what it used to be.
# As a fraction, enlarging the map enlarged the ocean with it: going from radius
# 12 to 18 dropped land from 55% to 43% and mountains from 35 to 7, because most
# of a hex map's area lives in its outer rings and they were all being drowned.
# A fixed-width border keeps the coastline looking right at any size while the
# playable interior actually grows.
RIM_WIDTH = 5
RIM_DEPTH = 1.10


def _rim_start(radius: int) -> float:
    """Fraction of the radius at which the ocean falloff begins."""
    return max(0.35, (radius - RIM_WIDTH) / radius)


RESOURCE_DENSITY = 0.10


@dataclass(frozen=True, slots=True)
class GeneratedMap:
    tiles: dict[str, Tile]
    starts: tuple[Hex, ...]


def generate(
    seed: int,
    radius: int,
    player_count: int = 4,
    start_radius_fraction: float | None = None,
) -> GeneratedMap:
    """Build a map and pick fair starting positions.

    Raises ValueError rather than silently degrading if the radius is too small
    for the requested number of players, because a cramped map produces matches
    that end in a turn-1 scrum and tell us nothing.
    """
    if player_count > len(START_CORNERS):
        raise ValueError(f"at most {len(START_CORNERS)} players are supported, got {player_count}")
    if radius < 6:
        raise ValueError(f"radius {radius} is too small to place fair starts; use 6 or more")

    terrain_stream = Stream(seed, "terrain")
    resource_stream = Stream(seed, "resources")
    template_stream = Stream(seed, "start_template")

    coords = hx.spiral(hx.ORIGIN, radius)
    tiles = _build_terrain(terrain_stream, coords, radius)
    _scatter_resources(resource_stream, tiles)

    starts = _choose_starts(radius, player_count, start_radius_fraction)
    template = _make_start_template(template_stream)
    for start in starts:
        _stamp(tiles, start, template)

    _ensure_connected(tiles, starts)

    return GeneratedMap(
        tiles={h.to_key(): tile for h, tile in sorted(tiles.items())},
        starts=starts,
    )


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------


def _build_terrain(stream: Stream, coords: list[Hex], radius: int) -> dict[Hex, Tile]:
    """Two smoothed noise fields, elevation and moisture, mapped to terrain.

    Smoothing is a plain neighbour average rather than real Perlin noise. It is
    a fraction of the code, and at this map size the difference is invisible.
    """
    elevation = {h: stream.random() for h in coords}
    moisture = {h: stream.random() for h in coords}
    for _ in range(SMOOTHING_PASSES):
        elevation = _smooth(elevation)
        moisture = _smooth(moisture)

    # Renormalise after smoothing. Averaging with neighbours is a low-pass
    # filter, so three passes collapse the field into a narrow band around 0.5:
    # the first version of this produced 127 grassland tiles and 4 mountains,
    # because the hills and mountain thresholds were simply never reached.
    # Stretching each field back across [0, 1] makes the thresholds below mean
    # what they say.
    #
    # The statistics come from the interior only. Normalising over the whole map
    # ties the range to wherever the global peak happened to land, and when that
    # was out near the rim the falloff subtracted it away again - two seeds in
    # three still generated no mountains at all. Scaling to the region that is
    # actually played on guarantees the full terrain range appears there.
    rim_start = _rim_start(radius)
    interior = [h for h in coords if hx.distance(hx.ORIGIN, h) <= radius * rim_start]
    elevation = _normalize(elevation, interior)
    moisture = _normalize(moisture, interior)

    tiles: dict[Hex, Tile] = {}
    for h in coords:
        # Ring the landmass with water so land units, which cannot swim, are
        # bounded by the map rather than by an invisible edge.
        #
        # The ramp starts halfway out rather than at the centre. A falloff
        # applied across the whole radius drags the interior down too, which is
        # what produced a map with zero mountains: every peak was subtracted
        # away before it could cross the threshold. Leaving the inner half
        # untouched keeps the full elevation range where the game is played.
        t = (hx.distance(hx.ORIGIN, h) / radius - rim_start) / (1.0 - rim_start)
        falloff = 0.0 if t <= 0 else min(1.0, t) ** 1.5 * RIM_DEPTH
        tiles[h] = Tile(terrain=_classify(elevation[h] - falloff, moisture[h]))
    return tiles


def _normalize(field: dict[Hex, float], sample: list[Hex]) -> dict[Hex, float]:
    """Stretch a field so that `sample` spans [0, 1].

    Every tile is transformed, but only `sample` sets the scale, so tiles
    outside it may land beyond [0, 1]. That is intentional and harmless: those
    are rim tiles heading for ocean anyway.

    Guards the degenerate flat field rather than dividing by zero; a uniform map
    is useless but it should not crash the generator.
    """
    values = [field[h] for h in sample] or list(field.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return dict.fromkeys(field, 0.5)
    span = hi - lo
    return {h: (v - lo) / span for h, v in field.items()}


def _smooth(field: dict[Hex, float]) -> dict[Hex, float]:
    out: dict[Hex, float] = {}
    for h, value in field.items():
        total, count = value, 1
        for n in hx.neighbors(h):
            if n in field:
                total += field[n]
                count += 1
        out[h] = total / count
    return out


def _classify(elevation: float, moisture: float) -> Terrain:
    """Map an (elevation, moisture) pair to terrain.

    Both fields are normalised to [0, 1] and are roughly bell-shaped, so these
    cut points are percentiles in disguise: mountains are about the top 5% of
    interior elevation and hills the next 15%, which keeps mountains scarce
    enough to form chokepoints rather than walls.
    """
    if elevation < 0.08:
        return Terrain.OCEAN
    if elevation < 0.16:
        return Terrain.COAST
    if elevation > 0.78:
        return Terrain.MOUNTAINS
    if elevation > 0.62:
        return Terrain.HILLS
    if moisture > 0.66:
        return Terrain.FOREST
    if moisture > 0.44:
        return Terrain.GRASSLAND
    if moisture > 0.22:
        return Terrain.PLAINS
    return Terrain.DESERT


def _scatter_resources(stream: Stream, tiles: dict[Hex, Tile]) -> None:
    """Place resources, respecting each one's allowed terrain.

    Iterates in sorted order so the same seed places the same resources; a plain
    dict iteration would be stable in practice but not by contract.
    """
    by_terrain: dict[Terrain, list[Resource]] = {}
    for resource, terrains in RESOURCE_TERRAIN.items():
        for terrain in terrains:
            by_terrain.setdefault(terrain, []).append(resource)
    for options in by_terrain.values():
        options.sort()

    for h in sorted(tiles):
        tile = tiles[h]
        options = by_terrain.get(tile.terrain)
        if not options or not stream.chance(RESOURCE_DENSITY):
            continue
        tiles[h] = tile.model_copy(update={"resource": stream.choice(options)})


# ---------------------------------------------------------------------------
# Starts
# ---------------------------------------------------------------------------


def _choose_starts(
    radius: int, player_count: int, fraction: float | None = None
) -> tuple[Hex, ...]:
    """Place the starts, never closer than the templates can be stamped.

    The floor is load-bearing, not defensive tidiness. Two starts closer than
    `MIN_START_SEPARATION` have overlapping template footprints, so the second
    stamp overwrites part of the first and the four civs no longer begin with
    identical land. That is the one bug in this file that would corrupt every
    model comparison the lab produces while leaving the map looking perfectly
    reasonable, so it is clamped rather than trusted to configuration.
    """
    distance = max(MIN_START_SEPARATION, int(radius * (fraction or START_RADIUS_FRACTION)))
    return tuple(hx.DIRECTIONS[corner].scale(distance) for corner in START_CORNERS[:player_count])


def _make_start_template(stream: Stream) -> dict[Hex, Tile]:
    """One starting neighbourhood, as offsets from the city site.

    Hand-composed rather than sampled from the generated map, because a sampled
    neighbourhood would sometimes be a desert and sometimes a floodplain, and
    the whole point is that all four are the same. The composition is a
    deliberately balanced opening: enough food to grow, enough production to
    build, and exactly one of each strategic resource so no civ is locked out
    of horsemen or swordsmen by the map.
    """
    offsets = hx.spiral(hx.ORIGIN, TEMPLATE_RADIUS)
    template: dict[Hex, Tile] = {}

    # The city site itself. Plains gives a balanced centre tile and is always
    # settleable, which the founding rule requires.
    template[hx.ORIGIN] = Tile(terrain=Terrain.PLAINS)

    # Inner ring: the tiles a size-1 city works immediately. Fixed, not random.
    inner = hx.ring(hx.ORIGIN, 1)
    inner_plan: list[tuple[Terrain, Resource | None]] = [
        (Terrain.GRASSLAND, Resource.WHEAT),
        (Terrain.GRASSLAND, None),
        (Terrain.PLAINS, Resource.HORSES),
        (Terrain.HILLS, Resource.IRON),
        (Terrain.FOREST, None),
        (Terrain.PLAINS, None),
    ]
    for offset, (terrain, resource) in zip(inner, inner_plan, strict=True):
        template[offset] = Tile(terrain=terrain, resource=resource)

    # Outer rings get a fixed terrain budget, shuffled. Shuffling keeps the four
    # starts from looking like literal copies to a human watching the board,
    # while the multiset - and therefore the total available yield - is
    # identical, because every start stamps this same shuffled template.
    outer = [h for h in offsets if hx.distance(hx.ORIGIN, h) > 1]
    budget: list[Terrain] = []
    for terrain, share in (
        (Terrain.GRASSLAND, 0.24),
        (Terrain.PLAINS, 0.24),
        (Terrain.FOREST, 0.18),
        (Terrain.HILLS, 0.16),
        (Terrain.DESERT, 0.08),
        (Terrain.COAST, 0.06),
        (Terrain.MOUNTAINS, 0.04),
    ):
        budget.extend([terrain] * round(len(outer) * share))
    # Rounding can leave the budget a tile or two short or long.
    budget = (budget + [Terrain.PLAINS] * len(outer))[: len(outer)]

    for offset, terrain in zip(outer, stream.shuffled(budget), strict=True):
        template[offset] = Tile(terrain=terrain)

    # A second wheat and iron in the outer band, so a civ that loses its inner
    # ring to an early rush is not immediately unable to build anything.
    for offset in stream.shuffled(sorted(outer))[:2]:
        terrain = template[offset].terrain
        for resource in (Resource.WHEAT, Resource.IRON):
            if terrain in RESOURCE_TERRAIN[resource]:
                template[offset] = Tile(terrain=terrain, resource=resource)
                break

    return template


def _stamp(tiles: dict[Hex, Tile], start: Hex, template: dict[Hex, Tile]) -> None:
    """Copy the template around `start`, overwriting whatever was generated.

    Offsets that fall outside the map are dropped rather than raising: with the
    default radius they never do, but a caller experimenting with a small map
    should get a slightly clipped start rather than a crash.
    """
    for offset, tile in template.items():
        target = start + offset
        if target in tiles:
            tiles[target] = tile.model_copy()


# What an impassable tile becomes when a corridor has to be carved through it.
# Mountains become hills and water becomes plains, so a carved route still
# costs something to cross rather than turning into a highway.
_CARVE_TO: dict[Terrain, Terrain] = {
    Terrain.MOUNTAINS: Terrain.HILLS,
    Terrain.OCEAN: Terrain.PLAINS,
    Terrain.COAST: Terrain.PLAINS,
}


def _land_reachable_from(tiles: dict[Hex, Tile], origin: Hex) -> set[Hex]:
    seen = {origin}
    frontier = [origin]
    while frontier:
        current = frontier.pop()
        for n in hx.neighbors(current):
            tile = tiles.get(n)
            if n in seen or tile is None or not TERRAIN[tile.terrain].passable:
                continue
            seen.add(n)
            frontier.append(n)
    return seen


def _ensure_connected(tiles: dict[Hex, Tile], starts: tuple[Hex, ...]) -> None:
    """Guarantee every start can reach every other over land.

    Procedural generation gets this right the overwhelming majority of the
    time, but "overwhelming majority" is not good enough for a run that goes
    unattended for days: a sweep of 300 seeds turned up one (214) that walled
    two civs off behind mountains and ocean. On that map nobody can ever reach
    anybody, the match runs its full turn limit with no contact, and the log
    reports a score victory that measured nothing at all.

    So instead of hoping, carve. Any start that cannot be reached gets a
    corridor cut along the straight line towards the main landmass, converting
    only the tiles that are actually impassable. It is deterministic, it is
    minimal - typically a handful of tiles, often zero - and it turns a rare
    silent failure into a guarantee.
    """
    anchor = starts[0]
    for other in starts[1:]:
        reachable = _land_reachable_from(tiles, anchor)
        if other in reachable:
            continue
        for h in hx.line(anchor, other):
            tile = tiles.get(h)
            if tile is None or TERRAIN[tile.terrain].passable:
                continue
            tiles[h] = tile.model_copy(
                update={"terrain": _CARVE_TO.get(tile.terrain, Terrain.PLAINS)}
            )


def settleable(tile: Tile) -> bool:
    return TERRAIN[tile.terrain].settleable
