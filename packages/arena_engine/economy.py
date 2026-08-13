"""Tile yields, city growth, production, and research.

All arithmetic here is integer. Floating point would be a determinism hazard
across platforms for no benefit, so percentage splits use integer division and
the remainder is dropped rather than rounded.

Tile assignment is automatic rather than an agent decision. That is a
deliberate scope choice: micromanaging which tiles a city works is exactly the
tedium that made a Freeciv turn cost hundreds of tokens, and it is not where
strategic reasoning lives. Cities work their best available tiles greedily, and
the agent's leverage is in *where to found* and *what to build*.
"""

from __future__ import annotations

from arena_engine import hex as hx
from arena_engine.content import (
    BUILDINGS,
    CITY_WORK_RADIUS,
    IMPROVEMENTS,
    RESOURCE_YIELDS,
    TERRAIN,
    UNITS,
    Yields,
    food_to_grow,
    tech_cost,
)
from arena_engine.content import UnitType as UT
from arena_engine.hex import Hex
from arena_engine.types import City, Player, State

# Each citizen eats this much per turn. Growth is the surplus over it, so a city
# on poor land stalls rather than starving outright.
FOOD_PER_CITIZEN = 2
# Every city produces this regardless of terrain, so a city on tundra is still
# worth founding and can always eventually build something.
BASE_CITY_YIELD = Yields(production=1, science=1, gold=1)


def tile_yields(state: State, h: Hex) -> Yields:
    """Terrain plus resource plus improvement."""
    tile = state.at(h)
    if tile is None:
        return Yields()
    total = TERRAIN[tile.terrain].yields
    if tile.resource is not None:
        total = total + RESOURCE_YIELDS[tile.resource]
    if tile.improvement is not None:
        total = total + IMPROVEMENTS[tile.improvement].yields
    return total


def _tile_score(y: Yields) -> int:
    """How desirable a tile is to work.

    Food and production are weighted equally and above commerce, which keeps
    cities growing and building rather than hoarding gold they cannot spend.
    """
    return y.food * 3 + y.production * 3 + y.gold * 2 + y.science * 2 + y.culture


def workable_tiles(state: State, city: City) -> list[Hex]:
    """Tiles this city may work, best first.

    Ties break on the hex key so assignment is byte-stable; a set-ordered
    tiebreak here would make two runs of the same seed diverge.
    """
    taken = {
        pos
        for _, other in sorted(state.cities.items())
        if other.id != city.id
        for pos in other.worked_tiles
    }
    centres = {c.pos for _, c in sorted(state.cities.items())}

    candidates: list[Hex] = []
    for h in hx.within(city.hex, CITY_WORK_RADIUS):
        key = h.to_key()
        tile = state.tiles.get(key)
        if tile is None or key == city.pos or key in taken or key in centres:
            continue
        # Tiles claimed by a rival civ cannot be worked.
        if tile.owner is not None and tile.owner != city.owner:
            continue
        candidates.append(h)

    return sorted(candidates, key=lambda h: (-_tile_score(tile_yields(state, h)), h.to_key()))


def assign_tiles(state: State, city: City) -> None:
    """Put every citizen on the best free tile. Mutates `city.worked_tiles`."""
    best = workable_tiles(state, city)[: city.population]
    city.worked_tiles = sorted(h.to_key() for h in best)


def city_yields(state: State, city: City) -> Yields:
    """Total output: centre tile, worked tiles, buildings, and the base."""
    total = BASE_CITY_YIELD + tile_yields(state, city.hex)
    for key in city.worked_tiles:
        total = total + tile_yields(state, hx.from_key(key))
    for name in city.buildings:
        total = total + BUILDINGS[name].yields
    return total


def city_defense(state: State, city: City) -> int:
    base = 8 + city.population
    for name in city.buildings:
        base += BUILDINGS[name].defense
    return base


def upkeep(state: State, player_id: str) -> int:
    """Gold per turn owed for buildings and units."""
    total = 0
    for _, city in sorted(state.cities.items()):
        if city.owner == player_id:
            total += sum(BUILDINGS[b].upkeep for b in city.buildings)
    for _, unit in sorted(state.units.items()):
        if unit.owner == player_id:
            total += UNITS[unit.type].upkeep
    return total


def player_output(state: State, player_id: str) -> tuple[int, int, int]:
    """Return `(gold_per_turn, science_per_turn, culture_per_turn)`.

    Commerce from tiles is split by the civ's tax slider; building science and
    gold are flat and bypass the split, so a library is worth the same whatever
    the slider says.
    """
    player = state.players[player_id]
    commerce = 0
    flat_gold = flat_science = culture = 0

    for _, city in sorted(state.cities.items()):
        if city.owner != player_id:
            continue
        y = city_yields(state, city)
        commerce += y.gold
        flat_science += y.science
        culture += y.culture

    gold = commerce * player.tax_pct // 100 + flat_gold
    science = commerce * player.science_pct // 100 + flat_science
    return gold - upkeep(state, player_id), science, culture


def can_build_unit(state: State, city: City, unit_type: UT) -> bool:
    """Tech and strategic-resource gates for a unit.

    A strategic resource must be inside the civ's borders somewhere, not
    necessarily this city's. That is what makes losing an iron hill matter at
    the empire level rather than just locally.
    """
    spec = UNITS[unit_type]
    player = state.players[city.owner]
    if spec.req_tech is not None and spec.req_tech not in player.known_techs:
        return False
    if spec.req_resource is not None:
        owned = any(
            tile.owner == city.owner and tile.resource == spec.req_resource
            for _, tile in sorted(state.tiles.items())
        )
        if not owned:
            return False
    return True


def can_build_building(state: State, city: City, name: str) -> bool:
    spec = BUILDINGS[name]
    player = state.players[city.owner]
    if name in city.buildings:
        return False
    if spec.req_tech is not None and spec.req_tech not in player.known_techs:
        return False
    # A wonder exists once in the world, so check every city, not just ours.
    already_built = any(name in c.buildings for _, c in sorted(state.cities.items()))
    return not (spec.wonder and already_built)


def buildable(state: State, city: City) -> list[str]:
    """Everything this city could start producing right now, sorted."""
    items = [u.value for u in UT if can_build_unit(state, city, u)]
    items += [b for b in sorted(BUILDINGS) if can_build_building(state, city, b)]
    return sorted(items)


# Unit and building names share one namespace in a city's `building` field, so
# they must never collide. Asserted in test_economy.py.
_UNIT_NAMES: frozenset[str] = frozenset(u.value for u in UT)


def is_unit(item: str) -> bool:
    return item in _UNIT_NAMES


def build_cost(item: str) -> int:
    """Production cost of a unit or building. Raises KeyError on an unknown item."""
    return UNITS[UT(item)].cost if is_unit(item) else BUILDINGS[item].cost


def research_progress(state: State, player: Player, science: int) -> tuple[bool, str | None]:
    """Add science and report whether the current tech completed.

    Returns `(completed, tech_name)`. The caller applies the tech and emits the
    event, because this module does not own state transitions.
    """
    if player.researching is None:
        return False, None
    player.science_stored += science
    cost = tech_cost(player.researching)
    if player.science_stored < cost:
        return False, None
    finished = player.researching
    player.science_stored -= cost
    return True, finished


def grow_city(city: City, food_surplus: int) -> str | None:
    """Apply a turn of food. Returns "grew", "shrank", or None.

    A granary keeps half the food box on growth, which is the difference
    between a city that snowballs and one that plateaus.
    """
    city.food_stored += food_surplus
    if city.food_stored < 0:
        if city.population > 1:
            city.population -= 1
            city.food_stored = 0
            return "shrank"
        city.food_stored = 0
        return None

    needed = food_to_grow(city.population)
    if city.food_stored < needed:
        return None

    city.population += 1
    kept_pct = max((BUILDINGS[b].food_kept_pct for b in city.buildings), default=0)
    city.food_stored = (city.food_stored - needed) + needed * kept_pct // 100
    return "grew"


def food_surplus(state: State, city: City) -> int:
    return city_yields(state, city).food - city.population * FOOD_PER_CITIZEN
