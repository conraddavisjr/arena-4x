"""Naval domain, embarkation, and coastal mechanics.

Built from hand-made fixtures rather than bot matches, because bots only take to
the sea when land runs out and would leave most of this untested for most seeds.

The invariant that matters throughout: `legal_actions` and the reducer must
agree exactly. The entire reason an agent is handed a legal-action list is that
anything on it will be accepted, so the two are asserted against each other
directly rather than separately.
"""

from __future__ import annotations

import pytest

from arena_engine import economy, movement
from arena_engine import hex as hx
from arena_engine.actions import Action, Attack, BuildImprovement, FoundCity, MoveUnit
from arena_engine.content import (
    EMBARK_TECH,
    EMBARKED_DEFENSE_PCT,
    TERRAIN,
    UNITS,
    Domain,
    Improvement,
    Terrain,
    UnitType,
)
from arena_engine.hex import Hex
from arena_engine.reducer import legal_actions, step
from arena_engine.types import City, MatchConfig, Player, State, Tile, Unit

LAND = Hex(0, 0)
SHORE = Hex(1, 0)
DEEP = Hex(2, 0)
FAR_SHORE = Hex(3, 0)


def sea_state(*, sailing: bool = True) -> State:
    """A two-island map: land at q<=0, water at q in {1,2}, land again at q>=3."""
    state = State(match_id="t", seed=1, config=MatchConfig(radius=6))
    for h in hx.within(hx.ORIGIN, 6):
        water = h.q in (1, 2)
        state.tiles[h.to_key()] = Tile(terrain=Terrain.COAST if water else Terrain.PLAINS)
    for pid, name in (("p1", "Aurelian Compact"), ("p2", "Iron Concord")):
        state.players[pid] = Player(
            id=pid, civ_name=name, known_techs=[EMBARK_TECH] if sailing else []
        )
    return state


def put(state: State, owner: str, kind: UnitType, at: Hex, **kw) -> Unit:
    uid, state.next_id = state.new_id("u")
    unit = Unit(
        id=uid,
        owner=owner,
        type=kind,
        pos=at.to_key(),
        moves_left=UNITS[kind].moves,
        **kw,
    )
    state.units[uid] = unit
    return unit


def act(state: State, player_id: str, *orders) -> tuple[State, list]:
    actions = {p: Action() for p in state.player_ids()}
    actions[player_id] = Action(orders=list(orders))
    return step(state, actions)


def rejections(events: list) -> list[str]:
    return [e.text for e in events if e.type == "order_rejected"]


# ---------------------------------------------------------------------------
# Entry rules
# ---------------------------------------------------------------------------


def test_land_units_cannot_enter_water_without_sailing() -> None:
    state = sea_state(sailing=False)
    unit = put(state, "p1", UnitType.WARRIOR, LAND)
    allowed, why = movement.entry_check(state, unit, SHORE)
    assert not allowed
    assert EMBARK_TECH in why


def test_sailing_unlocks_embarkation() -> None:
    state = sea_state()
    unit = put(state, "p1", UnitType.WARRIOR, LAND)
    allowed, _ = movement.entry_check(state, unit, SHORE)
    assert allowed
    assert movement.transition(state, unit, SHORE) == "embark"


def test_sea_units_cannot_walk_onto_land() -> None:
    state = sea_state()
    ship = put(state, "p1", UnitType.TRIREME, SHORE)
    allowed, why = movement.entry_check(state, ship, LAND)
    assert not allowed
    assert "navigable" in why


def test_sea_units_may_dock_in_a_coastal_city() -> None:
    """A trireme built in a port has to exist somewhere, and that tile is land."""
    state = sea_state()
    cid, state.next_id = state.new_id("c")
    state.cities[cid] = City(id=cid, owner="p1", name="Harbour", pos=LAND.to_key())
    ship = put(state, "p1", UnitType.TRIREME, SHORE)
    allowed, _ = movement.entry_check(state, ship, LAND)
    assert allowed, "a coastal city should act as a port"


def test_an_inland_city_is_not_a_port() -> None:
    state = sea_state()
    inland = Hex(-4, 0)
    cid, state.next_id = state.new_id("c")
    state.cities[cid] = City(id=cid, owner="p1", name="Inland", pos=inland.to_key())
    assert not movement.is_port(state, inland)


def test_mountains_remain_closed_to_every_domain() -> None:
    state = sea_state()
    state.tiles[SHORE.to_key()] = Tile(terrain=Terrain.MOUNTAINS)
    land_unit = put(state, "p1", UnitType.WARRIOR, LAND)
    ship = put(state, "p1", UnitType.TRIREME, DEEP)
    assert not movement.entry_check(state, land_unit, SHORE)[0]
    assert not movement.entry_check(state, ship, SHORE)[0]


# ---------------------------------------------------------------------------
# The crossing
# ---------------------------------------------------------------------------


def test_embarking_ends_the_turn_and_sets_the_flag() -> None:
    state = sea_state()
    unit = put(state, "p1", UnitType.WARRIOR, LAND)
    state, events = act(
        state, "p1", MoveUnit(action="move_unit", unit_id=unit.id, to=SHORE.to_key())
    )
    moved = state.units[unit.id]
    assert moved.embarked
    assert moved.pos == SHORE.to_key()
    assert moved.moves_left == 0, "a crossing should not also allow an advance"
    assert any(e.payload.get("transition") == "embark" for e in events if e.type == "unit_moved")


def test_landing_ends_the_turn_so_there_is_no_amphibious_blitz() -> None:
    """Crossing and assaulting in one turn would void a coastline's whole value."""
    state = sea_state()
    unit = put(state, "p1", UnitType.WARRIOR, DEEP, embarked=True)
    state, _ = act(
        state, "p1", MoveUnit(action="move_unit", unit_id=unit.id, to=FAR_SHORE.to_key())
    )
    landed = state.units[unit.id]
    assert not landed.embarked
    assert landed.moves_left == 0


def test_a_full_crossing_takes_several_turns() -> None:
    state = sea_state()
    unit = put(state, "p1", UnitType.WARRIOR, LAND)
    for target in (SHORE, DEEP, FAR_SHORE):
        state, events = act(
            state, "p1", MoveUnit(action="move_unit", unit_id=unit.id, to=target.to_key())
        )
        assert not rejections(events), rejections(events)
    final = state.units[unit.id]
    assert final.pos == FAR_SHORE.to_key()
    assert not final.embarked


def test_settlers_can_cross_and_found_on_the_far_shore() -> None:
    """Island settlement is the main reason embarkation exists."""
    state = sea_state()
    unit = put(state, "p1", UnitType.SETTLER, DEEP, embarked=True)
    state, _ = act(
        state, "p1", MoveUnit(action="move_unit", unit_id=unit.id, to=FAR_SHORE.to_key())
    )
    state, events = act(
        state, "p1", FoundCity(action="found_city", unit_id=unit.id, name="Overseas")
    )
    assert any(e.type == "city_founded" for e in events), rejections(events)


# ---------------------------------------------------------------------------
# What being at sea costs
# ---------------------------------------------------------------------------


def test_embarked_units_cannot_attack() -> None:
    state = sea_state()
    attacker = put(state, "p1", UnitType.WARRIOR, DEEP, embarked=True)
    put(state, "p2", UnitType.TRIREME, FAR_SHORE.__class__(2, -1))
    from arena_engine import diplomacy as dip
    from arena_engine.types import RelationState

    dip.set_relation(state, "p1", "p2", RelationState.WAR)
    state, events = act(state, "p1", Attack(action="attack", unit_id=attacker.id, target="2,-1"))
    assert any("at sea" in r for r in rejections(events))


def test_embarked_units_cannot_found_or_build() -> None:
    state = sea_state()
    settler = put(state, "p1", UnitType.SETTLER, DEEP, embarked=True)
    worker = put(state, "p1", UnitType.WORKER, DEEP, embarked=True)
    state, events = act(
        state,
        "p1",
        FoundCity(action="found_city", unit_id=settler.id, name="Atlantis"),
        BuildImprovement(action="build_improvement", unit_id=worker.id, improvement="farm"),
    )
    assert len(rejections(events)) == 2
    assert all("must land first" in r for r in rejections(events))


def test_embarked_units_defend_far_worse() -> None:
    from arena_engine import combat

    state = sea_state()
    ashore = put(state, "p1", UnitType.SWORDSMAN, LAND)
    at_sea = put(state, "p1", UnitType.SWORDSMAN, DEEP, embarked=True)
    raider = put(state, "p2", UnitType.TRIREME, Hex(2, -1))

    assert combat.defense_strength(state, at_sea, raider) < combat.defense_strength(
        state, ashore, raider
    )
    assert EMBARKED_DEFENSE_PCT < 100


def test_a_warship_beats_anything_it_catches_at_sea() -> None:
    """Otherwise escorting a crossing would be pointless."""
    from arena_engine import combat

    state = sea_state()
    ship = put(state, "p2", UnitType.TRIREME, Hex(2, -1))
    for kind in (UnitType.SWORDSMAN, UnitType.SPEARMAN, UnitType.ARCHER):
        prey = put(state, "p1", kind, DEEP, embarked=True)
        assert combat.attack_strength(state, ship, prey) > combat.defense_strength(
            state, prey, ship
        )


def test_embarked_units_move_at_the_sea_rate_not_their_own() -> None:
    """A scout should not sail three times faster than an army."""
    state = sea_state()
    scout = put(state, "p1", UnitType.SCOUT, DEEP, embarked=True)
    swordsman = put(state, "p1", UnitType.SWORDSMAN, DEEP, embarked=True)
    assert movement.moves_for(state, scout) == movement.moves_for(state, swordsman)


def test_domain_reflects_embarked_state() -> None:
    state = sea_state()
    unit = put(state, "p1", UnitType.WARRIOR, LAND)
    assert movement.domain_of(unit) is Domain.LAND
    unit.embarked = True
    assert movement.domain_of(unit) is Domain.SEA
    assert movement.domain_of(put(state, "p1", UnitType.TRIREME, SHORE)) is Domain.SEA


# ---------------------------------------------------------------------------
# legal_actions must agree with the reducer
# ---------------------------------------------------------------------------


def test_legal_actions_separates_embark_from_ordinary_moves() -> None:
    state = sea_state()
    unit = put(state, "p1", UnitType.WARRIOR, LAND)
    options = legal_actions(state, "p1")["units"][unit.id]
    assert SHORE.to_key() in options["move_unit_embarking"]
    assert SHORE.to_key() not in options["move_unit"], (
        "a crossing is a commitment and should not look like an ordinary step"
    )
    assert options["embarked"] is False


def test_legal_actions_offers_no_embark_without_sailing() -> None:
    state = sea_state(sailing=False)
    unit = put(state, "p1", UnitType.WARRIOR, LAND)
    options = legal_actions(state, "p1")["units"][unit.id]
    assert options["move_unit_embarking"] == []


def test_legal_actions_offers_disembark_only_when_at_sea() -> None:
    state = sea_state()
    unit = put(state, "p1", UnitType.WARRIOR, DEEP, embarked=True)
    options = legal_actions(state, "p1")["units"][unit.id]
    assert FAR_SHORE.to_key() in options["move_unit_landing"]
    assert options["embarked"] is True
    assert not options["fortify"], "a unit at sea cannot dig in"


@pytest.mark.parametrize("embarked", [False, True])
def test_every_offered_move_is_accepted_by_the_reducer(embarked: bool) -> None:
    """The contract: nothing on the legal list may be rejected."""
    state = sea_state()
    unit = put(state, "p1", UnitType.WARRIOR, DEEP if embarked else LAND, embarked=embarked)
    options = legal_actions(state, "p1")["units"][unit.id]

    for key in options["move_unit"] + options["move_unit_embarking"] + options["move_unit_landing"]:
        trial = state.model_copy(deep=True)
        trial, events = act(trial, "p1", MoveUnit(action="move_unit", unit_id=unit.id, to=key))
        assert not rejections(events), f"legal move {key} was rejected: {rejections(events)}"


# ---------------------------------------------------------------------------
# Coastal economy
# ---------------------------------------------------------------------------


def test_only_coastal_cities_can_build_ships() -> None:
    state = sea_state()
    coastal = City(id="c1", owner="p1", name="Harbour", pos=LAND.to_key())
    inland = City(id="c2", owner="p1", name="Inland", pos=Hex(-4, 0).to_key())
    state.cities["c1"], state.cities["c2"] = coastal, inland

    assert economy.is_coastal(state, coastal)
    assert not economy.is_coastal(state, inland)
    assert economy.can_build_unit(state, coastal, UnitType.TRIREME)
    assert not economy.can_build_unit(state, inland, UnitType.TRIREME)
    assert "trireme" in economy.buildable(state, coastal)
    assert "trireme" not in economy.buildable(state, inland)


def test_coastal_cities_develop_the_water_they_work() -> None:
    """Without this, half a coastal city's tiles are permanently unimprovable.

    The surrounding land is desert so the sea is genuinely the better tile to
    work. On plains the city would rationally ignore the water entirely, which
    is correct behaviour but would leave this mechanism untested.
    """
    state = sea_state()
    for h in hx.within(LAND, 2):
        tile = state.tiles[h.to_key()]
        if not TERRAIN[tile.terrain].navigable:
            state.tiles[h.to_key()] = Tile(terrain=Terrain.DESERT)

    cid, state.next_id = state.new_id("c")
    state.cities[cid] = City(id=cid, owner="p1", name="Harbour", pos=LAND.to_key(), population=3)
    for _ in range(3):
        state, _ = act(state, "p1")

    improved = [
        key for key, tile in state.tiles.items() if tile.improvement is Improvement.FISHING_BOATS
    ]
    assert improved, "a coastal city should put boats on the water it works"
    for key in improved:
        assert TERRAIN[state.tiles[key].terrain].navigable


def test_workers_cannot_be_ordered_to_build_fishing_boats() -> None:
    """They are city-placed; a worker cannot walk onto water to build one."""
    state = sea_state()
    worker = put(state, "p1", UnitType.WORKER, LAND)
    options = legal_actions(state, "p1")["units"][worker.id]
    assert "fishing_boats" not in options["build_improvement"]
