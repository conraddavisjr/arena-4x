"""The wilderness: roaming wildlife and barbarian raiders.

Without this the first ~19 turns of a match contain no threat at all. Civs do
not meet until roughly turn 19, so before that "build a garrison" is a strictly
irrational move, unclaimed land is free, and scouting is risk-free. Every agent
correctly plays a pure optimisation game with no downside, which is both less
interesting to reason about and considerably less interesting to watch.

Two kinds, with different jobs:

  - **Wolves** are fast, weak and local. They threaten a lone scout, settler or
    worker and are nothing to a fortified garrison. They make expansion a
    decision rather than a formality.
  - **Barbarians** are slow, heavy, and march on cities. They make military
    spending rational before the first rival civ is ever sighted.

**Barbarians sack cities but never hold them.** A barbarian-owned city would
have to be counted somewhere in the domination percentage and the conquest
check, and whichever way that went it would be a silent distortion of the
result. Sacking sidesteps the question entirely: no barbarian city ever exists,
so `test_barbarians.py` can assert that as an invariant across whole matches.

All behaviour is derived from the match seed and unit identity, never from a
sequential generator, so a replay reproduces the wilderness exactly.
"""

from __future__ import annotations

from arena_engine import combat, movement, rng
from arena_engine import events as ev
from arena_engine import hex as hx
from arena_engine.content import (
    BARBARIAN_ID,
    BARBARIAN_NAME,
    TERRAIN,
    UNITS,
    Terrain,
    UnitType,
)
from arena_engine.events import Event
from arena_engine.hex import Hex
from arena_engine.types import Player, State, Unit

# Terrain wildlife prefers to appear on.
WILD_TERRAIN: frozenset[Terrain] = frozenset(
    {Terrain.FOREST, Terrain.HILLS, Terrain.DESERT, Terrain.GRASSLAND}
)

# No spawning within this many hexes of a city, so nothing materialises on top
# of a civ that had no chance to see it coming.
SPAWN_SAFE_RADIUS = 4

# Wolves arrive immediately and taper off; raiders start a little later.
WOLF_UNTIL_TURN = 120
WOLF_SPAWN_CHANCE = 0.30
BARBARIAN_FIRST_TURN = 12
BARBARIAN_SPAWN_CHANCE = 0.22

# Population caps, per 100 tiles of *unclaimed* land.
#
# Scaling to unclaimed rather than total land is what makes the wilderness
# self-limiting: it is thickest when the map is empty and early pressure is the
# point, and it recedes on its own as civs settle, without needing a turn-based
# taper that would have to be retuned for every map size. The first version
# scaled to total land and was brutal - a measured match ended with 11 cities
# instead of the usual 27, seven of them burned outright, and a civ eliminated
# by raiders rather than by a rival. Barbarians are meant to be a tax and a
# threat, not the strongest power on the board.
WOLVES_PER_100_WILD = 2.0
BARBARIANS_PER_100_WILD = 1.6

# How far each kind will chase.
WOLF_HUNT_RADIUS = 5
BARBARIAN_MARCH_RADIUS = 12

# A sacked city loses this much population, and is razed only when there is
# genuinely nothing left to take. Losing a developed city to raiders should be
# a disaster the owner had many turns to prevent, not a coin flip.
SACK_POPULATION_LOSS = 1


def ensure_faction(state: State) -> None:
    """Add the neutral player if it is not already present."""
    if BARBARIAN_ID not in state.players:
        state.players[BARBARIAN_ID] = Player(id=BARBARIAN_ID, civ_name=BARBARIAN_NAME, neutral=True)


def _wild_tiles(state: State) -> int:
    """Unclaimed passable land: the wilderness's own habitat."""
    return sum(1 for t in state.tiles.values() if TERRAIN[t.terrain].passable and t.owner is None)


def _count(state: State, kind: UnitType) -> int:
    return sum(1 for u in state.units.values() if u.type is kind)


def _spawn_sites(state: State) -> list[Hex]:
    """Unowned, unoccupied wild land far enough from any city.

    Sorted, because the choice is made with a derived draw over this list and an
    unordered one would make the same seed spawn differently between runs.
    """
    cities = [c.hex for _, c in sorted(state.cities.items())]
    occupied = {u.pos for u in state.units.values()}
    sites: list[Hex] = []
    for key in sorted(state.tiles):
        tile = state.tiles[key]
        if tile.owner is not None or tile.terrain not in WILD_TERRAIN:
            continue
        if key in occupied:
            continue
        h = hx.from_key(key)
        if any(hx.distance(h, c) < SPAWN_SAFE_RADIUS for c in cities):
            continue
        sites.append(h)
    return sites


def spawn(state: State, out: list[Event]) -> None:
    """Populate the wilderness for this turn."""
    ensure_faction(state)
    wild = _wild_tiles(state)
    sites: list[Hex] | None = None

    # Scaled by `MatchConfig.wilderness`, which is how the whole mechanic is
    # dialled down or off without touching the tables below.
    density = state.config.wilderness
    wolf_cap = int(wild * WOLVES_PER_100_WILD * density / 100)
    raider_cap = int(wild * BARBARIANS_PER_100_WILD * density / 100)
    if density <= 0:
        return

    plans: list[tuple[UnitType, int, float, str]] = []
    if state.turn <= WOLF_UNTIL_TURN and _count(state, UnitType.WOLF) < wolf_cap:
        plans.append((UnitType.WOLF, wolf_cap, WOLF_SPAWN_CHANCE * density, "wolves"))
    if state.turn >= BARBARIAN_FIRST_TURN and _count(state, UnitType.BARBARIAN) < raider_cap:
        plans.append((UnitType.BARBARIAN, raider_cap, BARBARIAN_SPAWN_CHANCE * density, "raiders"))

    for kind, _cap, chance, label in plans:
        if not rng.chance(state.seed, chance, "barb_spawn", state.turn, label):
            continue
        if sites is None:
            sites = _spawn_sites(state)
        if not sites:
            return
        where = rng.choice(state.seed, sites, "barb_site", state.turn, label)
        sites = [s for s in sites if s != where]
        uid, state.next_id = state.new_id("u")
        state.units[uid] = Unit(
            id=uid,
            owner=BARBARIAN_ID,
            type=kind,
            pos=where.to_key(),
            moves_left=UNITS[kind].moves,
        )
        out.append(
            ev.event(
                state.turn,
                ev.UNIT_SPAWNED,
                f"{'A wolf pack' if kind is UnitType.WOLF else 'Barbarian raiders'} "
                f"appeared at {where.to_key()}",
                actor=BARBARIAN_ID,
                unit_id=uid,
                unit_type=kind.value,
                pos=where.to_key(),
            )
        )


def take_turn(state: State, out: list[Event]) -> None:
    """Move and fight for the whole wilderness.

    Runs after every civ has acted, so raiders react to where armies actually
    ended up rather than where they started.
    """
    for unit_id in sorted(u.id for u in state.units.values() if u.owner == BARBARIAN_ID):
        unit = state.units.get(unit_id)
        if unit is None:  # died earlier in this same phase
            continue
        unit.moves_left = movement.moves_for(state, unit)
        _act(state, unit, out)


def _act(state: State, unit: Unit, out: list[Event]) -> None:
    target = _adjacent_prey(state, unit)
    if target is not None:
        _strike(state, unit, target, out)
        return

    # An undefended city is a target in its own right. Looking only at units
    # meant a raider would march up to an empty city and then stand beside it
    # indefinitely, because there was nothing there to attack - sacking could
    # only ever happen as a side effect of killing the last defender.
    open_city = _adjacent_open_city(state, unit)
    if open_city is not None:
        _maybe_sack(state, unit, open_city, out)
        return

    goal = _goal_for(state, unit)
    if goal is None:
        _roam(state, unit)
        return

    options = [
        n
        for n in hx.neighbors(unit.hex)
        if movement.entry_check(state, unit, n)[0]
        and not any(u.owner == BARBARIAN_ID for u in state.units_at(n))
    ]
    if not options:
        return
    step = min(options, key=lambda h: (hx.distance(h, goal), h.to_key()))
    movement.apply_move(state, unit, step)


def _adjacent_open_city(state: State, unit: Unit) -> Hex | None:
    """An adjacent enemy city with nothing military left to hold it."""
    if unit.type is not UnitType.BARBARIAN:
        return None
    for n in hx.neighbors(unit.hex):
        city = state.city_at(n)
        if city is None or city.owner == BARBARIAN_ID:
            continue
        if combat.city_falls(state, city):
            return n
    return None


def _adjacent_prey(state: State, unit: Unit) -> Unit | None:
    """The best adjacent thing to attack, if anything."""
    candidates: list[Unit] = []
    for n in hx.neighbors(unit.hex):
        candidates.extend(u for u in state.units_at(n) if u.owner != BARBARIAN_ID)
    if not candidates:
        return None
    # Prefer the softest target: civilians first, then the weakest defender.
    return min(
        candidates,
        key=lambda u: (not UNITS[u.type].civilian, UNITS[u.type].defense * u.hp, u.id),
    )


def _goal_for(state: State, unit: Unit) -> Hex | None:
    if unit.type is UnitType.BARBARIAN:
        cities = [c.hex for _, c in sorted(state.cities.items())]
        goal = _nearest(unit.hex, cities, BARBARIAN_MARCH_RADIUS)
        if goal is not None:
            return goal
    prey = [u.hex for _, u in sorted(state.units.items()) if u.owner != BARBARIAN_ID]
    return _nearest(unit.hex, prey, WOLF_HUNT_RADIUS)


def _nearest(origin: Hex, options: list[Hex], limit: int) -> Hex | None:
    within = [h for h in options if hx.distance(origin, h) <= limit]
    if not within:
        return None
    return min(within, key=lambda h: (hx.distance(origin, h), h.to_key()))


def _roam(state: State, unit: Unit) -> None:
    options = sorted(
        (
            n
            for n in hx.neighbors(unit.hex)
            if movement.entry_check(state, unit, n)[0]
            and not state.units_at(n)
            and state.city_at(n) is None
        ),
        key=lambda h: h.to_key(),
    )
    if not options:
        return
    movement.apply_move(
        state, unit, rng.choice(state.seed, options, "barb_roam", state.turn, unit.id)
    )


def _strike(state: State, attacker: Unit, defender: Unit, out: list[Event]) -> None:
    target_hex = defender.hex
    stack = [u for u in state.units_at(target_hex) if u.owner != BARBARIAN_ID]
    best = combat.best_defender(state, stack) or defender
    victim_owner = best.owner

    result = combat.resolve(state, attacker, best)
    attacker.moves_left = 0

    if result.captured:
        # Wildlife does not take prisoners; an unescorted civilian is simply lost.
        del state.units[best.id]
        out.append(
            ev.event(
                state.turn,
                ev.UNIT_KILLED,
                f"{best.type.value} was lost to the wilderness at {target_hex.to_key()}",
                actor=victim_owner,
                unit_id=best.id,
                killed_by=BARBARIAN_ID,
            )
        )
        _maybe_sack(state, attacker, target_hex, out)
        return

    attacker.hp -= result.attacker_damage
    best.hp -= result.defender_damage
    out.append(
        ev.event(
            state.turn,
            ev.COMBAT_RESOLVED,
            f"{attacker.type.value} attacked {best.type.value} at {target_hex.to_key()}",
            actor=BARBARIAN_ID,
            attacker=attacker.id,
            defender=best.id,
            attacker_damage=result.attacker_damage,
            defender_damage=result.defender_damage,
        )
    )
    if result.defender_died:
        del state.units[best.id]
        out.append(
            ev.event(
                state.turn,
                ev.UNIT_KILLED,
                f"{best.type.value} was killed at {target_hex.to_key()}",
                actor=victim_owner,
                unit_id=best.id,
                killed_by=BARBARIAN_ID,
            )
        )
        _maybe_sack(state, attacker, target_hex, out)
    if result.attacker_died:
        state.units.pop(attacker.id, None)
        out.append(
            ev.event(
                state.turn,
                ev.UNIT_KILLED,
                f"a {attacker.type.value} was driven off at {target_hex.to_key()}",
                actor=victim_owner,
                unit_id=attacker.id,
                owner=BARBARIAN_ID,
            )
        )


def _maybe_sack(state: State, attacker: Unit, where: Hex, out: list[Event]) -> None:
    """Sack an undefended city. Never occupy it.

    A barbarian-held city would have to be counted somewhere in the domination
    percentage and the conquest check, and either choice would silently distort
    the result. Sacking removes the question: the raiders take what they can and
    disperse, and no barbarian city ever exists.
    """
    city = state.city_at(where)
    if city is None or not combat.city_falls(state, city):
        return

    owner = city.owner
    state.units.pop(attacker.id, None)

    if city.population <= 1:
        del state.cities[city.id]
        for key in list(state.tiles):
            tile = state.tiles[key]
            if tile.owner == owner and hx.distance(hx.from_key(key), where) <= 2:
                state.tiles[key] = tile.model_copy(update={"owner": None})
        out.append(
            ev.event(
                state.turn,
                ev.CITY_RAZED,
                f"{city.name} was burned to the ground by raiders",
                actor=owner,
                city_id=city.id,
                pos=where.to_key(),
            )
        )
        return

    city.population -= SACK_POPULATION_LOSS
    city.production_stored = 0
    city.building = None
    out.append(
        ev.event(
            state.turn,
            ev.CITY_SACKED,
            f"{city.name} was sacked by raiders, falling to population {city.population}",
            actor=owner,
            city_id=city.id,
            population=city.population,
        )
    )
