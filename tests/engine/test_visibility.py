"""Vision, fog, and contact.

The geometry cases are built by hand rather than generated, because the whole
value of the contact layer is that it distinguishes situations a fog overlay
alone would conflate: overlapping vision with no contact, and one-way sightings.
Both are asserted explicitly.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from arena_engine import hex as hx
from arena_engine import visibility as vis
from arena_engine.content import CITY_VISION, UNITS, Terrain, UnitType
from arena_engine.hex import Hex
from arena_engine.types import City, MatchConfig, Player, State, Tile


def make_state(terrain: Terrain = Terrain.PLAINS, radius: int = 10) -> State:
    """A blank flat map with two civs and nothing on it."""
    state = State(match_id="t", seed=1, config=MatchConfig(radius=radius))
    for h in hx.within(hx.ORIGIN, radius):
        state.tiles[h.to_key()] = Tile(terrain=terrain)
    for pid, name in (("p1", "Aurelian Compact"), ("p2", "Iron Concord")):
        state.players[pid] = Player(id=pid, civ_name=name)
    return state


def put_unit(state: State, owner: str, kind: UnitType, at: Hex) -> str:
    uid, state.next_id = state.new_id("u")
    from arena_engine.types import Unit

    state.units[uid] = Unit(id=uid, owner=owner, type=kind, pos=at.to_key())
    return uid


def put_city(state: State, owner: str, at: Hex, name: str = "Ravenholt") -> str:
    cid, state.next_id = state.new_id("c")
    state.cities[cid] = City(id=cid, owner=owner, name=name, pos=at.to_key())
    return cid


def set_terrain(state: State, at: Hex, terrain: Terrain) -> None:
    state.tiles[at.to_key()] = Tile(terrain=terrain)


# ---------------------------------------------------------------------------
# The four hand-built geometry cases from the plan
# ---------------------------------------------------------------------------


def test_mutual_sight() -> None:
    state = make_state()
    put_unit(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    put_unit(state, "p2", UnitType.WARRIOR, Hex(2, 0))
    r = vis.compute(state)
    assert r.mutual("p1", "p2")
    assert not r.one_way("p1", "p2")


def test_one_way_sighting() -> None:
    """A scout outranges a warrior, so it can watch without being watched.

    This is the case a plain fog overlay cannot show, and often the most
    decision-relevant fact on the board.
    """
    state = make_state()
    put_unit(state, "p1", UnitType.SCOUT, Hex(0, 0))  # vision 3
    put_unit(state, "p2", UnitType.WARRIOR, Hex(3, 0))  # vision 2
    r = vis.compute(state)
    assert r.sees("p1", "p2")
    assert not r.sees("p2", "p1")
    assert r.one_way("p1", "p2")
    assert not r.mutual("p1", "p2")


def test_overlapping_vision_without_contact() -> None:
    """Both civs see the same ground and neither sees the other.

    Warriors have vision 2, so at four hexes apart their fields meet in the
    middle while each unit sits two hexes beyond the other's range.
    """
    state = make_state()
    put_unit(state, "p1", UnitType.WARRIOR, Hex(-2, 0))
    put_unit(state, "p2", UnitType.WARRIOR, Hex(2, 0))
    r = vis.compute(state)
    assert hx.distance(Hex(-2, 0), Hex(2, 0)) > UNITS[UnitType.WARRIOR].vision
    overlap = r.visibility["p1"] & r.visibility["p2"]
    assert overlap, "the fixture should have overlapping vision fields"
    assert not r.sees("p1", "p2")
    assert not r.sees("p2", "p1")


def test_contact_through_a_city() -> None:
    """A city is an asset like any other; seeing one establishes contact."""
    state = make_state()
    put_unit(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    put_city(state, "p2", Hex(2, 0))
    r = vis.compute(state)
    assert r.sees("p1", "p2")
    assert any(s.asset == "city" for s in r.sightings)
    # The city sees back: CITY_VISION reaches the warrior.
    assert CITY_VISION >= 2
    assert r.sees("p2", "p1")


# ---------------------------------------------------------------------------
# The core invariant
# ---------------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["p1", "p2"]),
            st.sampled_from(list(UnitType)),
            st.integers(min_value=-6, max_value=6),
            st.integers(min_value=-6, max_value=6),
        ),
        min_size=0,
        max_size=8,
    )
)
def test_contact_implies_a_visible_asset(placements: list[tuple[str, UnitType, int, int]]) -> None:
    """contact[a][b] must never be true unless one of b's assets is in visibility[a]."""
    state = make_state()
    for owner, kind, q, r in placements:
        h = Hex(q, r)
        if h.to_key() in state.tiles:
            put_unit(state, owner, kind, h)

    report = vis.compute(state)
    positions: dict[str, set[Hex]] = {}
    for _, unit in state.units.items():
        positions.setdefault(unit.owner, set()).add(unit.hex)

    for observer, seen in report.contact.items():
        for observed in seen:
            assert positions.get(observed, set()) & report.visibility[observer], (
                f"{observer} reports contact with {observed} but sees none of its assets"
            )


def test_a_civ_is_never_in_contact_with_itself() -> None:
    state = make_state()
    put_unit(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    put_unit(state, "p1", UnitType.WARRIOR, Hex(1, 0))
    r = vis.compute(state)
    assert "p1" not in r.contact["p1"]


# ---------------------------------------------------------------------------
# Terrain rules
# ---------------------------------------------------------------------------


def test_mountains_block_line_of_sight() -> None:
    state = make_state()
    set_terrain(state, Hex(1, 0), Terrain.MOUNTAINS)
    put_unit(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    put_unit(state, "p2", UnitType.WARRIOR, Hex(2, 0))
    r = vis.compute(state)
    assert not r.sees("p1", "p2"), "a mountain between them should hide the unit"
    # The ridge itself is still visible; you can see the mountain, not past it.
    assert Hex(1, 0) in r.visibility["p1"]


def test_hills_do_not_block_sight() -> None:
    state = make_state()
    set_terrain(state, Hex(1, 0), Terrain.HILLS)
    put_unit(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    put_unit(state, "p2", UnitType.WARRIOR, Hex(2, 0))
    assert vis.compute(state).sees("p1", "p2")


def test_high_ground_extends_vision() -> None:
    flat = make_state()
    put_unit(flat, "p1", UnitType.WARRIOR, Hex(0, 0))
    plain_view = len(vis.compute(flat).visibility["p1"])

    high = make_state()
    set_terrain(high, Hex(0, 0), Terrain.HILLS)
    put_unit(high, "p1", UnitType.WARRIOR, Hex(0, 0))
    assert len(vis.compute(high).visibility["p1"]) > plain_view


def test_a_unit_on_a_mountain_can_see_out_of_it() -> None:
    """Blocking applies to tiles between the ends, not to the observer's own."""
    state = make_state()
    set_terrain(state, Hex(0, 0), Terrain.MOUNTAINS)
    put_unit(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    put_unit(state, "p2", UnitType.WARRIOR, Hex(2, 0))
    assert vis.compute(state).sees("p1", "p2")


def test_vision_respects_unit_range() -> None:
    state = make_state()
    put_unit(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    seen = vis.compute(state).visibility["p1"]
    assert all(hx.distance(hx.ORIGIN, h) <= UNITS[UnitType.WARRIOR].vision for h in seen)


def test_vision_never_leaves_the_map() -> None:
    state = make_state(radius=4)
    put_unit(state, "p1", UnitType.SCOUT, Hex(4, 0))
    seen = vis.compute(state).visibility["p1"]
    assert all(h.to_key() in state.tiles for h in seen)


# ---------------------------------------------------------------------------
# Memory and state application
# ---------------------------------------------------------------------------


def test_apply_writes_sorted_state_and_memory() -> None:
    state = make_state()
    put_unit(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    state.turn = 7
    report = vis.compute(state)
    vis.apply(state, report)

    assert state.visibility["p1"] == sorted(state.visibility["p1"])
    assert state.contact["p1"] == sorted(state.contact["p1"])
    remembered = state.players["p1"].memory
    assert remembered, "seeing tiles should populate memory"
    assert all(r.last_seen_turn == 7 for r in remembered.values())


def test_memory_persists_after_the_unit_leaves_and_goes_stale() -> None:
    state = make_state()
    uid = put_unit(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    state.turn = 3
    vis.apply(state, vis.compute(state))
    assert "0,0" in state.players["p1"].memory

    # March away and re-observe on a later turn.
    state.units[uid].pos = Hex(6, 0).to_key()
    state.turn = 20
    vis.apply(state, vis.compute(state))

    stale = state.players["p1"].memory["0,0"]
    assert stale.last_seen_turn == 3, "memory of a tile no longer seen must not refresh"
    assert state.players["p1"].memory["6,0"].last_seen_turn == 20


def test_memory_records_enemy_cities_but_not_units() -> None:
    state = make_state()
    put_unit(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    put_city(state, "p2", Hex(1, 0), name="Kaldis")
    put_unit(state, "p2", UnitType.WARRIOR, Hex(2, 0))
    state.turn = 5
    vis.apply(state, vis.compute(state))

    memory = state.players["p1"].memory
    assert "Kaldis" in (memory["1,0"].note or "")
    # A unit position is transient and must not be baked into memory.
    assert memory["2,0"].note is None


def test_own_city_is_not_annotated_as_foreign() -> None:
    state = make_state()
    put_city(state, "p1", Hex(0, 0), name="Ravenholt")
    vis.apply(state, vis.compute(state))
    assert state.players["p1"].memory["0,0"].note is None


def test_dead_players_neither_see_nor_are_seen() -> None:
    state = make_state()
    put_unit(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    put_unit(state, "p2", UnitType.WARRIOR, Hex(1, 0))
    state.players["p2"].alive = False
    r = vis.compute(state)
    assert "p2" not in r.visibility
    assert not r.sees("p1", "p2"), "an eliminated civ's leftovers should not register"


# ---------------------------------------------------------------------------
# Contact transitions
# ---------------------------------------------------------------------------


def test_contact_changes_reports_gained_and_lost_directionally() -> None:
    gained, lost = vis.contact_changes(
        previous={"p1": ["p2"], "p2": ["p1"]},
        current={"p1": {"p2", "p3"}, "p2": set()},
    )
    assert ("p1", "p3") in gained
    assert ("p2", "p1") in lost
    assert ("p1", "p2") not in gained and ("p1", "p2") not in lost


def test_contact_changes_is_sorted() -> None:
    gained, lost = vis.contact_changes(
        previous={}, current={"p2": {"p4", "p1"}, "p1": {"p3", "p2"}}
    )
    assert gained == sorted(gained)
    assert lost == []


def test_first_contact_is_detected_when_units_close() -> None:
    state = make_state()
    put_unit(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    uid = put_unit(state, "p2", UnitType.WARRIOR, Hex(6, 0))
    vis.apply(state, vis.compute(state))
    assert state.contact["p1"] == []

    state.units[uid].pos = Hex(2, 0).to_key()
    report = vis.compute(state)
    gained, lost = vis.contact_changes(state.contact, report.contact)
    assert ("p1", "p2") in gained
    assert ("p2", "p1") in gained
    assert lost == []
