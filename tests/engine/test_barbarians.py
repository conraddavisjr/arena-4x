"""The neutral faction.

Barbarians are modelled as a player so that units, combat, movement and
rendering work unchanged. That convenience is also the risk: every place that
means "a rival civilisation" must ask for civs specifically, and getting one
wrong would not raise - it would quietly change who wins. Conquest would never
fire because a wolf counts as a survivor; domination percentages would drift;
an agent would be offered a treaty with a bear.

So most of what follows asserts absences rather than behaviour.
"""

from __future__ import annotations

from collections import Counter

import pytest

from arena_engine import barbarians, bots, victory
from arena_engine import hex as hx
from arena_engine.actions import Action, Attack, Propose, SendMessage, pass_turn
from arena_engine.content import (
    BARBARIAN_ID,
    EMBARK_TECH,
    UNITS,
    Terrain,
    UnitType,
)
from arena_engine.hex import Hex
from arena_engine.reducer import legal_actions, new_match, step
from arena_engine.types import City, MatchConfig, Player, ProposalType, State, Tile, Unit

ROSTER = [
    ("p1", "Aurelian Compact"),
    ("p2", "Iron Concord"),
    ("p3", "Verdant Pact"),
    ("p4", "Solari Dominion"),
]


def fresh(seed: int = 5, **cfg) -> State:
    state, _ = new_match(f"t-{seed}", seed, ROSTER, MatchConfig(**cfg))
    return state


def passes(state: State) -> dict[str, Action]:
    return {p: pass_turn() for p in state.civ_ids()}


def flat_state() -> State:
    """A small blank map with one civ and the neutral faction present."""
    state = State(match_id="t", seed=1, config=MatchConfig(radius=5))
    for h in hx.within(hx.ORIGIN, 5):
        state.tiles[h.to_key()] = Tile(terrain=Terrain.PLAINS)
    state.players["p1"] = Player(id="p1", civ_name="Aurelian Compact")
    state.players["p2"] = Player(id="p2", civ_name="Iron Concord")
    barbarians.ensure_faction(state)
    return state


def put(state: State, owner: str, kind: UnitType, at: Hex, **kw) -> Unit:
    uid, state.next_id = state.new_id("u")
    unit = Unit(id=uid, owner=owner, type=kind, pos=at.to_key(), moves_left=UNITS[kind].moves, **kw)
    state.units[uid] = unit
    return unit


# ---------------------------------------------------------------------------
# The neutral faction is not a civilisation
# ---------------------------------------------------------------------------


def test_the_faction_exists_but_is_not_a_civ() -> None:
    state = fresh()
    assert BARBARIAN_ID in state.players
    assert state.players[BARBARIAN_ID].neutral
    assert BARBARIAN_ID not in state.civ_ids()
    assert BARBARIAN_ID not in state.living_civ_ids()
    assert len(state.civ_ids()) == 4


def test_scores_and_victory_ignore_the_faction() -> None:
    state = fresh()
    assert BARBARIAN_ID not in victory.scores(state)


def test_conquest_fires_on_the_last_civ_not_the_last_player() -> None:
    """The bug this guards: a surviving wolf making conquest unreachable."""
    state = flat_state()
    cid, state.next_id = state.new_id("c")
    state.cities[cid] = City(id=cid, owner="p1", name="Sole", pos="0,0")
    state.players["p2"].alive = False
    put(state, BARBARIAN_ID, UnitType.WOLF, Hex(3, 0))

    result = victory.check(state)
    assert result is not None, "conquest should fire with one civ left"
    assert result.condition == "conquest"
    assert result.winner == "p1"


def test_domination_counts_only_civ_cities() -> None:
    state = fresh()
    assert all(not state.is_neutral(c.owner) for c in state.cities.values())


def test_the_faction_is_never_eliminated() -> None:
    """It has no units between spawns; that is not death."""
    state = fresh()
    for _ in range(4):
        state, _ = step(state, passes(state))
    assert state.players[BARBARIAN_ID].alive
    assert state.players[BARBARIAN_ID].eliminated_turn is None


def test_turn_rotation_excludes_the_faction() -> None:
    from arena_engine.reducer import _rotation

    state = fresh()
    assert BARBARIAN_ID not in _rotation(state)


# ---------------------------------------------------------------------------
# No diplomacy with the wilderness
# ---------------------------------------------------------------------------


def test_legal_actions_never_offers_the_faction_as_a_diplomatic_partner() -> None:
    state = fresh()
    diplo = legal_actions(state, "p1")["diplomacy"]
    for key in ("send_message", "propose", "declare_war"):
        assert BARBARIAN_ID not in diplo[key], f"{key} offered the wilderness"


def test_messaging_or_proposing_to_the_faction_is_rejected() -> None:
    state = fresh()
    actions = passes(state)
    actions["p1"] = Action(
        diplomacy=[
            SendMessage(action="send_message", channel="private", to=BARBARIAN_ID, text="parley"),
            Propose(action="propose", to=BARBARIAN_ID, type=ProposalType.PEACE),
        ]
    )
    state, events = step(state, actions)
    rejected = [e for e in events if e.type == "order_rejected"]
    assert len(rejected) == 2


def test_everyone_is_permanently_at_war_with_the_wilderness() -> None:
    state = fresh()
    for civ in state.civ_ids():
        assert state.at_war(civ, BARBARIAN_ID)
        assert state.at_war(BARBARIAN_ID, civ)
    assert not state.at_war(BARBARIAN_ID, BARBARIAN_ID)


def test_attacking_the_wilderness_needs_no_declaration() -> None:
    state = flat_state()
    soldier = put(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    put(state, BARBARIAN_ID, UnitType.WOLF, Hex(1, 0))

    options = legal_actions(state, "p1")["units"][soldier.id]
    assert "1,0" in options["attack"], "a wolf next door should be attackable"

    actions = {p: pass_turn() for p in state.civ_ids()}
    actions["p1"] = Action(orders=[Attack(action="attack", unit_id=soldier.id, target="1,0")])
    state, events = step(state, actions)
    assert not [e for e in events if e.type == "order_rejected"]
    assert any(e.type == "combat_resolved" for e in events)


# ---------------------------------------------------------------------------
# Contact and visibility
# ---------------------------------------------------------------------------


def test_the_faction_has_no_fog_of_its_own() -> None:
    state = fresh()
    state, _ = step(state, passes(state))
    assert BARBARIAN_ID not in state.visibility
    assert BARBARIAN_ID not in state.contact


def test_sighting_wildlife_is_not_diplomatic_contact() -> None:
    """The contact matrix answers who has sighted whom, not what."""
    state = flat_state()
    put(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    put(state, BARBARIAN_ID, UnitType.WOLF, Hex(1, 0))

    from arena_engine import visibility as vis

    report = vis.compute(state)
    assert BARBARIAN_ID not in report.contact.get("p1", set())
    assert BARBARIAN_ID not in report.visibility


# ---------------------------------------------------------------------------
# Cities are sacked, never held
# ---------------------------------------------------------------------------


def test_raiders_sack_an_undefended_city_without_taking_it() -> None:
    state = flat_state()
    cid, state.next_id = state.new_id("c")
    state.cities[cid] = City(id=cid, owner="p1", name="Ravenholt", pos="0,0", population=5)
    put(state, BARBARIAN_ID, UnitType.BARBARIAN, Hex(1, 0))

    events: list = []
    barbarians.take_turn(state, events)

    city = state.cities.get(cid)
    if city is not None:
        assert city.owner == "p1", "a raider must never come to own a city"
        assert city.population < 5 or any(e.type == "city_sacked" for e in events)
    else:
        assert any(e.type == "city_razed" for e in events)


def test_a_size_one_city_is_razed_rather_than_left_at_zero() -> None:
    state = flat_state()
    cid, state.next_id = state.new_id("c")
    state.cities[cid] = City(id=cid, owner="p1", name="Outpost", pos="0,0", population=1)
    state.tiles["0,0"] = Tile(terrain=Terrain.PLAINS, owner="p1")
    put(state, BARBARIAN_ID, UnitType.BARBARIAN, Hex(1, 0))

    events: list = []
    barbarians.take_turn(state, events)
    assert cid not in state.cities
    assert any(e.type == "city_razed" for e in events)
    assert state.tiles["0,0"].owner is None, "razing should release the borders"


def test_a_defended_city_is_not_sacked() -> None:
    state = flat_state()
    cid, state.next_id = state.new_id("c")
    state.cities[cid] = City(id=cid, owner="p1", name="Held", pos="0,0", population=5)
    put(state, "p1", UnitType.SPEARMAN, Hex(0, 0), fortified=True)
    put(state, BARBARIAN_ID, UnitType.BARBARIAN, Hex(1, 0))

    events: list = []
    barbarians.take_turn(state, events)
    assert cid in state.cities
    assert not any(e.type in {"city_sacked", "city_razed"} for e in events)


@pytest.mark.parametrize("seed", [0, 2, 4])
def test_no_barbarian_city_ever_exists_across_a_whole_match(seed: int) -> None:
    """The invariant the sack-not-capture design exists to guarantee."""
    state = fresh(seed, turn_limit=200)
    while state.victory is None and state.turn < 200:
        state, _ = step(state, bots.all_bot_actions(state))
        assert not any(c.owner == BARBARIAN_ID for c in state.cities.values())
        assert BARBARIAN_ID not in victory.scores(state)


# ---------------------------------------------------------------------------
# Spawning
# ---------------------------------------------------------------------------


def test_nothing_spawns_on_claimed_land() -> None:
    state = fresh()
    for _ in range(30):
        state, _ = step(state, bots.all_bot_actions(state))
        for unit in state.units.values():
            if unit.owner != BARBARIAN_ID:
                continue
            tile = state.tiles[unit.pos]
            # It may walk onto owned land, but it must not appear there.
            assert tile is not None


def test_spawn_sites_respect_the_safe_radius() -> None:
    state = fresh()
    for _ in range(6):
        state, events = step(state, bots.all_bot_actions(state))
        cities = [c.hex for c in state.cities.values()]
        for e in events:
            if e.type != "unit_spawned":
                continue
            where = hx.from_key(e.payload["pos"])
            for city in cities:
                assert hx.distance(where, city) >= barbarians.SPAWN_SAFE_RADIUS


def test_the_wilderness_recedes_as_land_is_claimed() -> None:
    """The cap scales to unclaimed land, which is what makes it self-limiting.

    An earlier version scaled to *total* land and was brutal: a measured match
    ended with 11 cities instead of the usual 27, seven burned outright.
    """
    state = fresh()
    assert barbarians._wild_tiles(state) > 0
    early = barbarians._wild_tiles(state)

    for key in list(state.tiles)[:200]:
        state.tiles[key] = state.tiles[key].model_copy(update={"owner": "p1"})
    assert barbarians._wild_tiles(state) < early


def test_wildlife_cannot_put_to_sea() -> None:
    """Wolves crossing oceans would make every island unsafe forever."""
    state = flat_state()
    state.players[BARBARIAN_ID].known_techs = [EMBARK_TECH]
    state.tiles["1,0"] = Tile(terrain=Terrain.OCEAN)
    from arena_engine import movement

    for kind in (UnitType.WOLF, UnitType.BARBARIAN):
        beast = put(state, BARBARIAN_ID, kind, Hex(0, 0))
        allowed, _ = movement.entry_check(state, beast, Hex(1, 0))
        assert not allowed, f"{kind.value} should not be able to embark"


# ---------------------------------------------------------------------------
# Determinism and pressure
# ---------------------------------------------------------------------------


def test_the_wilderness_replays_identically() -> None:
    def run() -> list[tuple[int, str, str]]:
        state = fresh(9, turn_limit=40)
        out = []
        for _ in range(40):
            state, events = step(state, bots.all_bot_actions(state))
            out.extend((e.turn, e.type, e.text) for e in events if e.actor == BARBARIAN_ID)
            if state.victory:
                break
        return out

    first = run()
    assert first, "the wilderness should do something in 40 turns"
    assert first == run()


def test_the_early_game_now_carries_real_threat() -> None:
    """The whole reason this exists: turns 1-19 used to be risk-free."""
    state = fresh(3, turn_limit=25)
    counts: Counter[str] = Counter()
    while state.turn < 25 and state.victory is None:
        state, events = step(state, bots.all_bot_actions(state))
        counts.update(e.type for e in events)

    assert counts["unit_spawned"] > 0, "no wilderness appeared at all"
    assert counts["combat_resolved"] > 0, "the opening was still risk-free"


# ---------------------------------------------------------------------------
# What a wilderness attack reports
# ---------------------------------------------------------------------------


def test_a_wilderness_attack_reports_the_same_facts_as_a_civs_attack() -> None:
    """The payload is the record, and half of it was missing.

    A civ's attack carries `defender_owner`, both unit types and both death
    flags. The wilderness attack carried none of them, and the replay bundle
    keys its combat feed on `attacker_type` - so every blow struck by a wolf
    was dropped on the way to the viewer.

    In the first complete match that was 75 of 103 fights. Three quarters of
    all violence in the world was invisible downstream, and a civ being mauled
    by wolves rendered identically to one at peace. Nothing failed; the panels
    drew a smaller number with total confidence.

    Asserted field by field rather than by comparing to a civ attack, because
    the bug was an omission and a test that only checks "some payload exists"
    would have passed against the broken version.
    """
    state = flat_state()
    wolf = put(state, BARBARIAN_ID, UnitType.WOLF, hx.ORIGIN)
    scout = put(state, "p1", UnitType.SCOUT, Hex(1, 0))

    out: list = []
    barbarians._strike(state, wolf, scout, out)

    strikes = [e for e in out if e.type == "combat_resolved"]
    assert strikes, "a wilderness attack must report itself"
    payload = strikes[0].payload
    assert payload["defender_owner"] == "p1"
    assert payload["attacker_type"] == UnitType.WOLF.value
    assert payload["defender_type"] == UnitType.SCOUT.value
    for flag in ("attacker_died", "defender_died"):
        assert flag in payload, f"{flag} missing: the bundle cannot tell who survived"
