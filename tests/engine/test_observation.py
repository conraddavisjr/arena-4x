"""The observation translator.

Two classes of failure matter here and neither raises on its own.

**Leaking.** If the payload contains anything the civ has not observed, every
match is confounded: agents reason from an omniscient view while the log claims
they were playing under fog. Most of what follows asserts the *absence* of
information.

**Unbounded growth.** Every field is paid for by four agents on ~130 turns, so
anything that grows with exploration has to be capped deliberately. The
remembered map is the one that would run away.
"""

from __future__ import annotations

import json

import pytest

from arena_engine import bots, observation, rules
from arena_engine import hex as hx
from arena_engine.actions import Action, pass_turn
from arena_engine.content import BARBARIAN_ID, UNITS, Terrain, UnitType
from arena_engine.hex import Hex
from arena_engine.reducer import new_match, step
from arena_engine.types import MatchConfig, Player, State, Tile, Unit

ROSTER = [
    ("p1", "Aurelian Compact"),
    ("p2", "Iron Concord"),
    ("p3", "Verdant Pact"),
    ("p4", "Solari Dominion"),
]

# The payload legitimately grows with empire size - a fourteen-city civ has more
# to reason about than a one-city civ - so this is a cost ceiling, not a claim
# that it is constant.
#
# The arithmetic: at 12k tokens of fresh input per agent-turn, a 130-turn match
# with four agents costs about 12k x 130 x 4 x $5/M = $31 in observation input
# alone. That fits inside the $75 per-match cap alongside output and the cached
# system prefix. Measured peak is ~9.5k at turn 120, so this leaves headroom
# while still failing loudly if something starts growing without a bound - which
# it did: `legal_actions` spelled out empty option lists for every unit and hit
# 5,400 tokens on its own.
TOKEN_CEILING = 12_000


def fresh(seed: int = 7, **cfg) -> State:
    state, _ = new_match(f"o-{seed}", seed, ROSTER, MatchConfig(**cfg))
    return state


def advance(state: State, turns: int) -> State:
    for _ in range(turns):
        if state.victory is not None:
            break
        state, _ = step(state, bots.all_bot_actions(state))
    return state


def flat_state() -> State:
    state = State(match_id="t", seed=1, config=MatchConfig(radius=6))
    for h in hx.within(hx.ORIGIN, 6):
        state.tiles[h.to_key()] = Tile(terrain=Terrain.PLAINS)
    for pid, name in ROSTER[:2]:
        state.players[pid] = Player(id=pid, civ_name=name)
    return state


def put(state: State, owner: str, kind: UnitType, at: Hex) -> Unit:
    uid, state.next_id = state.new_id("u")
    unit = Unit(id=uid, owner=owner, type=kind, pos=at.to_key(), moves_left=UNITS[kind].moves)
    state.units[uid] = unit
    return unit


# ---------------------------------------------------------------------------
# Fog: what must NOT be in the payload
# ---------------------------------------------------------------------------


def test_only_visible_tiles_appear_in_the_visible_map() -> None:
    state = advance(fresh(), 20)
    visible = set(state.visibility["p1"])
    obs = observation.build(state, "p1")
    assert {t.pos for t in obs.map.visible} <= visible


def test_rival_units_outside_vision_are_absent_entirely() -> None:
    """The sharpest leak: an agent seeing an army it never scouted."""
    state = advance(fresh(), 25)
    obs = observation.build(state, "p1")
    visible = set(state.visibility["p1"])

    for rival in ("p2", "p3", "p4"):
        hidden = {u.pos for u in state.units_of(rival)} - visible
        payload = observation.to_json(obs)
        for pos in hidden:
            for tile in obs.map.visible:
                assert tile.pos != pos
            # And the coordinate must not appear anywhere else in the payload.
            assert f'"{pos}"' not in payload or pos in visible


def test_own_units_are_fully_described_but_rivals_are_not() -> None:
    state = flat_state()
    mine = put(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    theirs = put(state, "p2", UnitType.SWORDSMAN, Hex(1, 0))
    theirs.hp = 42
    from arena_engine import visibility as vis

    vis.apply(state, vis.compute(state))

    obs = observation.build(state, "p1")
    assert [u.id for u in obs.units] == [mine.id]
    payload = observation.to_json(obs)
    # We can see that a swordsman is there, but not its id or exact health.
    assert "swordsman" in payload
    assert theirs.id not in payload
    assert '"hp":42' not in payload.replace(" ", "")


def test_intel_reports_only_what_this_civ_has_observed() -> None:
    """City counts come from memory, not from the true board."""
    state = fresh()
    obs = observation.build(state, "p1")
    for entry in obs.intel:
        assert entry.known_cities == 0, "nothing has been scouted on turn 0"
        assert not entry.in_contact
        assert entry.military_estimate == "unknown"
        assert entry.known_score is None, "score is hidden until contact"


def test_score_and_strength_appear_only_after_contact() -> None:
    state = advance(fresh(3), 60)
    obs = observation.build(state, "p1")
    for entry in obs.intel:
        if entry.in_contact:
            assert entry.known_score is not None
            assert entry.military_estimate in {"stronger", "weaker", "comparable"}
        else:
            assert entry.known_score is None
            assert entry.military_estimate == "unknown"


def test_private_messages_to_others_never_appear() -> None:
    from arena_engine.actions import SendMessage

    state = fresh()
    actions = {p: pass_turn() for p in state.civ_ids()}
    actions["p2"] = Action(
        diplomacy=[
            SendMessage(action="send_message", channel="private", to="p3", text="SECRETPACT")
        ]
    )
    state, _ = step(state, actions)
    state, _ = step(state, {p: pass_turn() for p in state.civ_ids()})

    assert "SECRETPACT" not in observation.to_json(observation.build(state, "p1"))
    assert "SECRETPACT" in observation.to_json(observation.build(state, "p3"))


def test_the_neutral_faction_is_not_a_diplomatic_relation() -> None:
    state = advance(fresh(), 15)
    obs = observation.build(state, "p1")
    assert BARBARIAN_ID not in {r.player_id for r in obs.diplomacy.relations}
    assert BARBARIAN_ID not in {i.player_id for i in obs.intel}


def test_wildlife_is_visible_on_the_map_as_an_occupant() -> None:
    """An agent must be able to see the wolf pack beside its settler."""
    state = flat_state()
    from arena_engine import barbarians

    barbarians.ensure_faction(state)
    put(state, "p1", UnitType.WARRIOR, Hex(0, 0))
    put(state, BARBARIAN_ID, UnitType.WOLF, Hex(1, 0))
    from arena_engine import visibility as vis

    vis.apply(state, vis.compute(state))

    obs = observation.build(state, "p1")
    occupied = [t for t in obs.map.visible if t.occupant]
    wild = [t for t in occupied if t.occupant.get("wild")]
    assert wild, "the wolf should be visible"
    assert wild[0].occupant["units"] == {"wolf": 1}


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("turns", [1, 30, 80, 130])
def test_payload_stays_within_budget(turns: int) -> None:
    state = advance(fresh(11), turns)
    for player_id in state.living_civ_ids():
        obs = observation.build(state, player_id)
        tokens = observation.estimate_tokens(obs)
        assert tokens < TOKEN_CEILING, f"turn {state.turn} {player_id}: ~{tokens} tokens"


def test_the_remembered_map_does_not_grow_without_bound() -> None:
    """Exploration is unbounded; what we send back about it must not be.

    A civ remembers hundreds of tiles by mid-match, almost all of them
    "there is grass here", which changes no decision and costs real money.
    """
    state = advance(fresh(13), 90)
    obs = observation.build(state, "p1")
    assert obs.map.explored_tiles > 100, "the fixture should have explored a lot"
    assert len(obs.map.remembered) <= observation.MAX_REMEMBERED
    # Everything returned must actually carry information.
    for entry in obs.map.remembered:
        assert entry.note is not None or entry.resource is not None


def test_remembered_entries_are_stale_by_construction() -> None:
    state = advance(fresh(), 40)
    obs = observation.build(state, "p1")
    visible = set(state.visibility["p1"])
    for entry in obs.map.remembered:
        assert entry.pos not in visible, "a visible tile is not a memory"
        assert entry.last_seen_turn <= state.turn


def test_rival_cities_are_prioritised_in_memory() -> None:
    """The most decision-relevant thing a civ can remember."""
    state = advance(fresh(2), 70)
    obs = observation.build(state, "p1")
    noted = [r for r in obs.map.remembered if r.note]
    if noted:
        assert obs.map.remembered[0].note is not None, "cities should sort first"


def test_json_omits_nulls() -> None:
    payload = observation.to_json(observation.build(advance(fresh(), 10), "p1"))
    assert ":null" not in payload.replace(" ", "")


# ---------------------------------------------------------------------------
# Usefulness
# ---------------------------------------------------------------------------


def test_compacting_legal_actions_drops_nothing_actionable() -> None:
    """Compaction removes empty options; it must never remove a real one.

    The payload omits empty lists and false flags, because spelling them out for
    every unit cost over 5,000 tokens a turn late in a match. An absent key and
    an empty one say the same thing - but dropping a key that had *content*
    would silently take a legal move away from the agent.
    """
    from arena_engine.reducer import legal_actions

    state = advance(fresh(), 40)
    for player_id in state.living_civ_ids():
        raw = legal_actions(state, player_id)
        compact = observation.build(state, player_id).legal_actions

        assert set(compact) == set(raw), "a top-level section went missing"
        assert set(compact["units"]) == set(raw["units"]), "a unit went missing"

        for unit_id, options in raw["units"].items():
            actionable = {k: v for k, v in options.items() if v not in (None, [], False)}
            assert compact["units"][unit_id] == actionable

        # Nothing outside `units` is compacted, so it must match exactly.
        for key in set(raw) - {"units"}:
            assert compact[key] == raw[key]


def test_compaction_actually_saves_tokens() -> None:
    """If it stopped saving anything the complexity would not be worth keeping."""
    import json

    from arena_engine.reducer import legal_actions

    state = advance(fresh(), 60)
    raw = len(json.dumps(legal_actions(state, "p1")))
    compact = len(json.dumps(observation.build(state, "p1").legal_actions))
    assert compact < raw * 0.75, f"compaction saved only {1 - compact / raw:.0%}"


def test_every_own_unit_and_city_is_present() -> None:
    state = advance(fresh(), 45)
    obs = observation.build(state, "p1")
    assert {u.id for u in obs.units} == {u.id for u in state.units_of("p1")}
    assert {c.id for c in obs.cities} == {c.id for c in state.cities_of("p1")}


def test_cities_report_actionable_forecasts() -> None:
    state = advance(fresh(), 50)
    obs = observation.build(state, "p1")
    for city in obs.cities:
        assert city.food_to_grow > 0
        if city.building is not None and city.production_per_turn > 0:
            assert city.turns_remaining is not None and city.turns_remaining >= 1


def test_embarked_units_report_their_state() -> None:
    state = flat_state()
    state.tiles["1,0"] = Tile(terrain=Terrain.OCEAN)
    unit = put(state, "p1", UnitType.WARRIOR, Hex(1, 0))
    unit.embarked = True
    from arena_engine import visibility as vis

    vis.apply(state, vis.compute(state))
    obs = observation.build(state, "p1")
    assert obs.units[0].status == "at sea"


def test_dossier_round_trips_into_the_observation() -> None:
    from arena_engine.types import Dossier, OpponentModel, Trustworthiness

    written = Dossier(
        doctrine="Expand east.",
        opponent_models=[
            OpponentModel(
                player_id="p2",
                assessed_intent="military expansion",
                trustworthiness=Trustworthiness.LOW,
            )
        ],
    )
    state = fresh()
    actions = {p: pass_turn() for p in state.civ_ids()}
    actions["p1"] = Action(dossier=written)
    state, _ = step(state, actions)
    assert observation.build(state, "p1").your_dossier == written


def test_observation_serialises_to_valid_json() -> None:
    obs = observation.build(advance(fresh(), 30), "p1")
    parsed = json.loads(observation.to_json(obs))
    for key in ("match_id", "turn", "you", "cities", "units", "map", "legal_actions"):
        assert key in parsed


def test_observation_is_deterministic() -> None:
    state = advance(fresh(17), 25)
    assert observation.to_json(observation.build(state, "p1")) == observation.to_json(
        observation.build(state, "p1")
    )


def test_frontier_points_at_the_unexplored() -> None:
    state = advance(fresh(), 15)
    obs = observation.build(state, "p1")
    visible = set(state.visibility["p1"])
    assert obs.map.frontier, "there should be somewhere left to scout"
    for pos in obs.map.frontier:
        assert pos not in visible


# ---------------------------------------------------------------------------
# The rules reference and the cached prefix
# ---------------------------------------------------------------------------


def test_the_rules_reference_is_byte_stable() -> None:
    """It is the entire cached system prefix.

    Anthropic prompt caching is a strict prefix match, so one varying byte here
    invalidates the cache on every request for every agent and turns a ~0.1x
    read back into full price.
    """
    assert rules.reference() == rules.reference()
    assert rules.system_prompt("A", "p1") == rules.system_prompt("A", "p1")


def test_only_the_civ_identity_varies_between_agents() -> None:
    a = rules.system_prompt("Aurelian Compact", "p1")
    b = rules.system_prompt("Iron Concord", "p2")
    assert a != b
    assert rules.reference() in a and rules.reference() in b


def test_the_reference_carries_no_dynamic_content() -> None:
    """A turn number or timestamp in here would be a per-request cache miss."""
    text = rules.reference()
    for leak in ("turn 1", "2026", "match_id", "p1", "p2"):
        assert leak not in text, f"dynamic value {leak!r} leaked into the cached prefix"


def test_the_reference_clears_the_cache_minimum() -> None:
    """Below ~512 tokens Anthropic silently declines to cache at all."""
    assert len(rules.reference()) // 4 > 512


def test_the_reference_describes_the_actual_content_tables() -> None:
    """Generated, so it cannot drift - but assert the wiring works."""
    from arena_engine.content import BUILDINGS, TECHS

    text = rules.reference()
    for unit in (UnitType.SWORDSMAN, UnitType.TRIREME, UnitType.SETTLER):
        assert unit.value in text
    for tech in ("apex_theory", "sailing", "bronze_working"):
        assert tech in TECHS and tech in text
    for building in ("granary", "walls"):
        assert building in BUILDINGS and building in text


def test_the_reference_hides_the_wilderness_unit_stats() -> None:
    """Wolves are not buildable, so their line in the unit table is noise."""
    unit_table = rules.reference().split("## Units")[1].split("## Buildings")[0]
    assert "wolf" not in unit_table
    assert "barbarian" not in unit_table


def test_the_reference_explains_the_rules_agents_will_actually_hit() -> None:
    """The non-obvious rules, which an agent cannot infer from the tables."""
    text = rules.reference().lower()
    for topic in (
        "legal_actions",  # anything not listed will be rejected
        "simultaneous",  # everyone moves at once
        "next",  # messages arrive a turn later
        "embark",  # crossing water is a commitment
        "wilderness",  # permanently hostile, no declaration needed
        "individually",  # one bad order does not void the turn
    ):
        assert topic in text, f"the rules never mention {topic!r}"


# ---------------------------------------------------------------------------
# Schemas on disk stay in step with the models
# ---------------------------------------------------------------------------


def test_exported_schemas_match_the_models() -> None:
    """Regenerate and compare, so a model change cannot silently outdate them."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import export_schemas

    for model, name in (
        (__import__("arena_engine.actions", fromlist=["Action"]).Action, "action.schema.json"),
        (observation.Observation, "observation.schema.json"),
    ):
        on_disk = json.loads((export_schemas.OUT / name).read_text())
        regenerated = export_schemas.sanitize(model.model_json_schema())
        regenerated["title"] = on_disk["title"]
        regenerated["$schema"] = on_disk["$schema"]
        assert on_disk == regenerated, f"{name} is stale; run scripts/export_schemas.py"


def test_schemas_meet_the_strictest_provider_dialect() -> None:
    """Authored to Anthropic's rules so one schema serves all four vendors."""
    import export_schemas

    for name in ("action.schema.json", "observation.schema.json"):
        schema = json.loads((export_schemas.OUT / name).read_text())
        _assert_dialect(schema, name)


def _assert_dialect(node, name: str, path: str = "") -> None:
    if isinstance(node, list):
        for i, item in enumerate(node):
            _assert_dialect(item, name, f"{path}[{i}]")
        return
    if not isinstance(node, dict):
        return
    banned = export_schemas_unsupported() & set(node)
    assert not banned, f"{name}{path} uses unsupported keywords {sorted(banned)}"
    if node.get("type") == "object" or "properties" in node:
        assert node.get("additionalProperties") is False, f"{name}{path} is an open object"
    for key, value in node.items():
        _assert_dialect(value, name, f"{path}.{key}")


def export_schemas_unsupported() -> set[str]:
    """Everything that must not survive into a published schema.

    Both the keywords that get stripped and the ones that get rewritten. The
    rewrites matter just as much and are easier to miss: `oneOf` is what Pydantic
    emits for a discriminated union, and Anthropic and OpenAI both reject it
    outright. Every mocked test passed with it in place, the parity test passed,
    and a 108-turn dry match played through without complaint - because none of
    those ever send the schema to a vendor. It would have 400'd on turn one of
    the flagship run, on three of four seats at once.
    """
    import export_schemas

    return export_schemas.UNSUPPORTED | set(export_schemas.REWRITES)


def test_legal_actions_uses_the_exact_action_names_from_the_schema() -> None:
    """The observation and the schema must agree on what an action is called.

    They did not. `legal_actions` offered `move`, `build` and `research` while
    the schema demanded `move_unit`, `set_production` and `set_research`, and
    `set_rates` appeared in neither - the tax slider was the one lever with no
    name anywhere in the observation.

    The observation is the more immediate of the two: a model reads what it is
    shown this turn before it consults a schema it was handed once. Providers
    that enforce the enum in the decoder hid this completely. Gemini, which does
    not, followed the observation and emitted `research` and `set_taxes`. Those
    orders were dropped as invalid, so the only symptom was a civ quietly doing
    less each turn than it thought it had ordered.
    """
    from arena_engine.actions import _BRANCH_FIELDS
    from arena_engine.reducer import legal_actions

    state = advance(fresh(5), 12)
    legal = legal_actions(state, "p1")
    names = set(_BRANCH_FIELDS)

    offered = set(legal) - {"units", "cities", "diplomacy"}
    offered |= set(legal["diplomacy"])
    for options in legal["units"].values():
        # `embarked` is a state flag, not an action; the two move variants are
        # both `move_unit` and say so in their names.
        offered |= {k.split("_embarking")[0].split("_landing")[0] for k in options} - {"embarked"}
    for options in legal["cities"].values():
        offered |= set(options)

    unknown = offered - names
    assert not unknown, f"legal_actions offers names the schema has never heard of: {unknown}"


def test_every_action_the_schema_accepts_is_discoverable_somewhere() -> None:
    """An action with no name in any observation has to be guessed at, and a
    model that guesses invents something plausible and wrong."""
    from arena_engine.actions import _BRANCH_FIELDS
    from arena_engine.reducer import legal_actions

    state = advance(fresh(5), 25)
    seen: set[str] = set()
    for player_id in state.living_civ_ids():
        legal = legal_actions(state, player_id)
        seen |= set(legal) - {"units", "cities", "diplomacy"}
        seen |= set(legal["diplomacy"])
        for options in legal["units"].values():
            seen |= {k.split("_embarking")[0].split("_landing")[0] for k in options}
        for options in legal["cities"].values():
            seen |= set(options)

    missing = set(_BRANCH_FIELDS) - seen
    assert not missing, f"no observation ever names these actions: {missing}"
