"""Map generation.

The headline property is start fairness: every civ must begin with an identical
multiset of terrain and resources inside its working radius. If that ever
breaks, every model comparison the lab produces is confounded, and it would
break silently, so it is asserted directly across many seeds.
"""

from __future__ import annotations

from collections import Counter

import pytest

from arena_engine import hex as hx
from arena_engine import mapgen
from arena_engine.content import RESOURCE_TERRAIN, TERRAIN, Resource, Terrain
from arena_engine.hex import Hex

SEEDS = [1, 2, 7, 42, 99, 1234, 20260812]

# Seed 214 generated a map where two civs were walled off behind mountains and
# ocean with no land route between them. A match on it would run its full turn
# limit with zero contact and report a score victory that measured nothing. The
# generator now carves a corridor rather than leaving this to chance; this seed
# is kept by name so the guarantee cannot silently regress.
DISCONNECTED_REGRESSION_SEED = 214


@pytest.mark.parametrize("seed", SEEDS)
def test_map_covers_exactly_the_requested_radius(seed: int) -> None:
    m = mapgen.generate(seed, radius=12)
    expected = {h.to_key() for h in hx.within(hx.ORIGIN, 12)}
    assert set(m.tiles) == expected


@pytest.mark.parametrize("seed", SEEDS)
def test_generation_is_deterministic(seed: int) -> None:
    a = mapgen.generate(seed, radius=12)
    b = mapgen.generate(seed, radius=12)
    assert a.tiles == b.tiles
    assert a.starts == b.starts


def test_different_seeds_give_different_maps() -> None:
    a = mapgen.generate(1, radius=12)
    b = mapgen.generate(2, radius=12)
    assert a.tiles != b.tiles, "seeds must actually vary the map"


@pytest.mark.parametrize("seed", SEEDS)
def test_starting_neighbourhoods_are_identical(seed: int) -> None:
    """The fairness guarantee, stated as bluntly as it can be."""
    m = mapgen.generate(seed, radius=12)
    signatures = []
    for start in m.starts:
        offsets = hx.spiral(hx.ORIGIN, mapgen.TEMPLATE_RADIUS)
        signature = tuple(
            (
                offset,
                m.tiles[(start + offset).to_key()].terrain,
                m.tiles[(start + offset).to_key()].resource,
            )
            for offset in offsets
        )
        signatures.append(signature)
    assert len(set(signatures)) == 1, "starts differ tile-for-tile"


@pytest.mark.parametrize("seed", SEEDS)
def test_starting_yields_are_identical(seed: int) -> None:
    """Terrain parity is the mechanism; equal opening yield is the point."""
    m = mapgen.generate(seed, radius=12)
    totals = []
    for start in m.starts:
        food = production = gold = 0
        for offset in hx.spiral(hx.ORIGIN, mapgen.TEMPLATE_RADIUS):
            tile = m.tiles[(start + offset).to_key()]
            y = TERRAIN[tile.terrain].yields
            food, production, gold = food + y.food, production + y.production, gold + y.gold
        totals.append((food, production, gold))
    assert len(set(totals)) == 1, f"unequal starting yields: {totals}"


@pytest.mark.parametrize("seed", SEEDS)
def test_every_start_is_settleable(seed: int) -> None:
    m = mapgen.generate(seed, radius=12)
    for start in m.starts:
        tile = m.tiles[start.to_key()]
        assert TERRAIN[tile.terrain].settleable, f"start on unsettleable {tile.terrain}"


@pytest.mark.parametrize("seed", SEEDS)
def test_every_start_has_both_strategic_resources_nearby(seed: int) -> None:
    """No civ may be locked out of horsemen or swordsmen by the map."""
    m = mapgen.generate(seed, radius=12)
    for start in m.starts:
        nearby = {
            m.tiles[(start + o).to_key()].resource
            for o in hx.spiral(hx.ORIGIN, mapgen.TEMPLATE_RADIUS)
        }
        assert Resource.IRON in nearby
        assert Resource.HORSES in nearby


@pytest.mark.parametrize("seed", SEEDS)
def test_starts_are_mutually_distant_and_symmetric(seed: int) -> None:
    """Every player should face the same distance profile to its rivals."""
    m = mapgen.generate(seed, radius=12)
    assert len(m.starts) == 4
    profiles = [tuple(sorted(hx.distance(a, b) for b in m.starts if b != a)) for a in m.starts]
    assert len(set(profiles)) == 1, f"asymmetric start placement: {profiles}"
    # Below MIN_START_SEPARATION the stamped templates overlap and the fairness
    # guarantee silently breaks, so that is the floor rather than a round number.
    assert min(min(p) for p in profiles) >= mapgen.MIN_START_SEPARATION


def test_start_templates_never_overlap() -> None:
    """The mechanism behind the fairness guarantee, asserted directly.

    Two starts closer than MIN_START_SEPARATION share template tiles, so the
    second stamp partly overwrites the first and the civs stop having identical
    openings - while the map still looks entirely plausible. This was a live bug
    when the start fraction was retuned for a larger board: at radius 12 the
    starts landed 4 apart and six seeds' worth of fairness assertions failed at
    once. Swept across radii rather than a fixed one, because the failure only
    appears at particular radius/fraction combinations.
    """
    for radius in range(6, 26):
        m = mapgen.generate(1, radius=radius)
        footprints = [
            {(start + offset) for offset in hx.spiral(hx.ORIGIN, mapgen.TEMPLATE_RADIUS)}
            for start in m.starts
        ]
        for i, a in enumerate(footprints):
            for b in footprints[i + 1 :]:
                assert not (a & b), f"radius {radius}: start templates overlap"


def test_extreme_start_fractions_are_clamped_not_obeyed() -> None:
    """A configuration that would break fairness must be corrected, not honoured."""
    m = mapgen.generate(1, radius=20, start_radius_fraction=0.01)
    closest = min(hx.distance(a, b) for i, a in enumerate(m.starts) for b in m.starts[i + 1 :])
    assert closest >= mapgen.MIN_START_SEPARATION


@pytest.mark.parametrize("seed", SEEDS)
def test_land_is_a_reasonable_share_of_the_map(seed: int) -> None:
    m = mapgen.generate(seed, radius=12)
    counts = Counter(t.terrain for t in m.tiles.values())
    land = sum(n for t, n in counts.items() if TERRAIN[t].passable)
    share = land / len(m.tiles)
    assert 0.35 < share < 0.85, f"land share {share:.2f} makes for a poor map"


@pytest.mark.parametrize("seed", SEEDS)
def test_map_rim_is_water(seed: int) -> None:
    """Land units cannot swim, so the rim being water is what bounds the board."""
    m = mapgen.generate(seed, radius=12)
    rim = hx.ring(hx.ORIGIN, 12)
    water = sum(1 for h in rim if not TERRAIN[m.tiles[h.to_key()].terrain].passable)
    assert water / len(rim) > 0.8, "map edge should be mostly impassable"


def _land_reachable(m: mapgen.GeneratedMap, origin: Hex) -> set[Hex]:
    seen = {origin}
    frontier = [origin]
    while frontier:
        current = frontier.pop()
        for n in hx.neighbors(current):
            key = n.to_key()
            if n in seen or key not in m.tiles:
                continue
            if TERRAIN[m.tiles[key].terrain].passable:
                seen.add(n)
                frontier.append(n)
    return seen


@pytest.mark.parametrize("seed", [*SEEDS, DISCONNECTED_REGRESSION_SEED])
def test_all_starts_are_mutually_reachable_over_land(seed: int) -> None:
    """A civ that cannot reach its rivals can never be attacked or attack.

    This is the property that would quietly ruin a match: everyone turtles on
    an island, nothing happens for 300 turns, and the log shows a score win
    that measured nothing.
    """
    m = mapgen.generate(seed, radius=12)
    seen = _land_reachable(m, m.starts[0])
    for other in m.starts[1:]:
        assert other in seen, f"{other} unreachable over land from {m.starts[0]}"


def test_connectivity_holds_across_a_wide_seed_sweep() -> None:
    """Connectivity is a guarantee, not a probability.

    The carving pass exists because one seed in a few hundred generated a
    walled-off map. A handful of parametrised seeds would not have caught that,
    so this sweeps a range wide enough to have found the original failure.
    """
    for seed in range(250):
        m = mapgen.generate(seed, radius=12)
        seen = _land_reachable(m, m.starts[0])
        unreachable = [s for s in m.starts[1:] if s not in seen]
        assert not unreachable, f"seed {seed}: {unreachable} unreachable over land"


def test_carving_is_minimal_on_maps_that_were_already_connected() -> None:
    """The corridor must not bulldoze maps that never needed it.

    Compares against generation with the carve pass disabled: on a map that was
    already connected the two must be byte-identical.
    """
    from unittest.mock import patch

    for seed in (1, 42, 1234):
        with patch.object(mapgen, "_ensure_connected", lambda *a: None):
            uncarved = mapgen.generate(seed, radius=12)
        carved = mapgen.generate(seed, radius=12)
        assert carved.tiles == uncarved.tiles, f"seed {seed} was carved unnecessarily"


@pytest.mark.parametrize("seed", SEEDS)
def test_resources_only_appear_on_legal_terrain(seed: int) -> None:
    m = mapgen.generate(seed, radius=12)
    for key, tile in m.tiles.items():
        if tile.resource is not None:
            assert tile.terrain in RESOURCE_TERRAIN[tile.resource], (
                f"{tile.resource} on {tile.terrain} at {key}"
            )


@pytest.mark.parametrize("seed", SEEDS)
def test_map_has_variety(seed: int) -> None:
    m = mapgen.generate(seed, radius=12)
    counts = Counter(t.terrain for t in m.tiles.values())
    assert len(counts) >= 6, f"only {len(counts)} terrain types: {counts}"
    assert any(t.resource for t in m.tiles.values()), "no resources placed at all"


def test_rejects_impossible_configurations() -> None:
    with pytest.raises(ValueError, match="at most"):
        mapgen.generate(1, radius=12, player_count=9)
    with pytest.raises(ValueError, match="too small"):
        mapgen.generate(1, radius=3)


def test_smaller_maps_still_generate() -> None:
    m = mapgen.generate(1, radius=8)
    assert len(m.tiles) == len(hx.within(hx.ORIGIN, 8))
    assert len(m.starts) == 4


def test_start_template_offsets_stay_inside_a_default_map() -> None:
    """Clipping is a safety net, not something the default config should hit."""
    m = mapgen.generate(1, radius=12)
    for start in m.starts:
        for offset in hx.spiral(hx.ORIGIN, mapgen.TEMPLATE_RADIUS):
            assert (start + offset).to_key() in m.tiles


def test_generated_tiles_are_independent_objects() -> None:
    """Stamped tiles must be copies; a shared object would alias four starts."""
    m = mapgen.generate(1, radius=12)
    a, b = m.starts[0], m.starts[1]
    assert m.tiles[a.to_key()] is not m.tiles[b.to_key()]
    ids = {id(t) for t in m.tiles.values()}
    assert len(ids) == len(m.tiles), "tiles alias each other"


def test_hex_helper_used_by_tests_matches_generator() -> None:
    assert mapgen.settleable(mapgen.Tile(terrain=Terrain.PLAINS))
    assert not mapgen.settleable(mapgen.Tile(terrain=Terrain.OCEAN))


def test_start_positions_are_distinct_hexes() -> None:
    m = mapgen.generate(1, radius=12)
    assert len(set(m.starts)) == len(m.starts)
    assert all(isinstance(s, Hex) for s in m.starts)
