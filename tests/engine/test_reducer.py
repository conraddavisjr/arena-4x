"""The step function, determinism, and full-match integration.

The determinism tests are the most valuable in the repo. Every claim the lab
makes rests on being able to replay a match and get the same match back; without
that, a post-game analysis is just a story about numbers that happened once.
"""

from __future__ import annotations

import pytest

from arena_engine import bots, economy
from arena_engine import hex as hx
from arena_engine.actions import (
    Action,
    Attack,
    DeclareWar,
    FoundCity,
    MoveUnit,
    Propose,
    SendMessage,
    SetRates,
    SetResearch,
    pass_turn,
)
from arena_engine.content import MIN_CITY_SPACING, NEUTRAL_UNITS, Terrain, UnitType
from arena_engine.reducer import legal_actions, new_match, step
from arena_engine.types import MatchConfig, ProposalType, RelationState, State, Terms, Unit

ROSTER = [
    ("p1", "Aurelian Compact"),
    ("p2", "Iron Concord"),
    ("p3", "Verdant Pact"),
    ("p4", "Solari Dominion"),
]


def fresh(seed: int = 42, **cfg) -> State:
    state, _ = new_match(f"t-{seed}", seed, ROSTER, MatchConfig(**cfg))
    return state


def passes() -> dict[str, Action]:
    return {p: pass_turn() for p, _ in ROSTER}


def settler_of(state: State, player_id: str) -> Unit:
    return next(u for u in state.units_of(player_id) if u.type is UnitType.SETTLER)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_identical_inputs_produce_identical_states() -> None:
    a, b = fresh(7), fresh(7)
    assert a.state_hash() == b.state_hash()
    for _ in range(12):
        a, _ = step(a, passes())
        b, _ = step(b, passes())
    assert a.state_hash() == b.state_hash()


def test_bot_match_replays_to_an_identical_hash_every_turn() -> None:
    """The property the whole observability story depends on."""
    first_hashes: list[str] = []
    state = fresh(11, turn_limit=40)
    for _ in range(40):
        state, _ = step(state, bots.all_bot_actions(state))
        first_hashes.append(state.state_hash())
        if state.victory:
            break

    replay = fresh(11, turn_limit=40)
    second_hashes: list[str] = []
    for _ in range(len(first_hashes)):
        replay, _ = step(replay, bots.all_bot_actions(replay))
        second_hashes.append(replay.state_hash())

    assert first_hashes == second_hashes


def test_recorded_orders_replay_identically() -> None:
    """Replaying the *recorded actions* must reproduce the match exactly.

    Distinct from the test above: this is the path the event log actually takes,
    where actions come off disk rather than from re-running the bot.
    """
    state = fresh(5, turn_limit=25)
    recorded: list[dict[str, Action]] = []
    hashes: list[str] = []
    for _ in range(25):
        actions = bots.all_bot_actions(state)
        recorded.append(actions)
        state, _ = step(state, actions)
        hashes.append(state.state_hash())
        if state.victory:
            break

    replay = fresh(5, turn_limit=25)
    for actions, expected in zip(recorded, hashes, strict=True):
        replay, _ = step(replay, actions)
        assert replay.state_hash() == expected


def test_events_are_identical_across_replays() -> None:
    def run() -> list[tuple[int, str, str]]:
        state = fresh(3, turn_limit=30)
        out = []
        for _ in range(30):
            state, events = step(state, bots.all_bot_actions(state))
            out.extend((e.turn, e.type, e.text) for e in events)
            if state.victory:
                break
        return out

    assert run() == run()


def test_different_seeds_diverge() -> None:
    assert fresh(1).state_hash() != fresh(2).state_hash()


def test_step_does_not_mutate_the_input_state() -> None:
    """A reducer that mutated its input would corrupt every stored snapshot."""
    state = fresh(9)
    before = state.state_hash()
    step(state, bots.all_bot_actions(state))
    assert state.state_hash() == before


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def test_new_match_gives_every_civ_the_same_opening() -> None:
    state = fresh(21)
    counts = {p: len(state.units_of(p)) for p, _ in ROSTER}
    assert len(set(counts.values())) == 1, f"unequal starting units: {counts}"
    # Civs only: the neutral faction is a player but starts with nothing.
    assert len({state.players[p].gold for p in state.civ_ids()}) == 1


def test_starting_units_are_not_all_stacked() -> None:
    """A single early attack should not be able to take a civ's whole opening."""
    state = fresh(21)
    positions = {u.pos for u in state.units_of("p1")}
    assert len(positions) > 1


def test_turn_advances_and_emits_bookends() -> None:
    state = fresh()
    state, events = step(state, passes())
    assert state.turn == 1
    kinds = [e.type for e in events]
    assert kinds[0] == "turn_started"
    assert kinds[-1] == "turn_ended"


def test_resolution_order_rotates() -> None:
    """No civ may permanently own first strike."""
    from arena_engine.reducer import _rotation

    state = fresh()
    seen = set()
    for turn in range(4):
        state.turn = turn
        seen.add(_rotation(state)[0])
    assert len(seen) == 4, f"rotation did not cycle: {seen}"


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def test_found_city_claims_borders_and_consumes_the_settler() -> None:
    state = fresh()
    settler = settler_of(state, "p1")
    actions = passes()
    actions["p1"] = Action(
        orders=[FoundCity(action="found_city", unit_id=settler.id, name="Ravenholt")]
    )
    state, events = step(state, actions)

    assert any(e.type == "city_founded" for e in events)
    assert settler.id not in state.units, "the settler should be consumed"
    city = next(c for c in state.cities_of("p1"))
    assert city.name == "Ravenholt"
    assert state.tiles[city.pos].owner == "p1"
    assert state.at(hx.from_key(city.pos) + hx.DIRECTIONS[0]).owner == "p1"


def test_cities_cannot_be_founded_too_close_together() -> None:
    state = fresh()
    settler = settler_of(state, "p1")
    actions = passes()
    actions["p1"] = Action(
        orders=[FoundCity(action="found_city", unit_id=settler.id, name="First")]
    )
    state, _ = step(state, actions)
    city = state.cities_of("p1")[0]

    second = next((u for u in state.units_of("p1") if u.type is UnitType.SETTLER), None)
    if second is None:
        pytest.skip("no second settler in this configuration")
    second.pos = (hx.from_key(city.pos) + hx.DIRECTIONS[0]).to_key()
    actions = passes()
    actions["p1"] = Action(
        orders=[FoundCity(action="found_city", unit_id=second.id, name="TooClose")]
    )
    state, events = step(state, actions)
    assert any("too close" in e.text for e in events if e.type == "order_rejected")


def test_one_bad_order_does_not_void_the_rest_of_the_turn() -> None:
    """The single most important robustness property for LLM agents."""
    state = fresh()
    settler = settler_of(state, "p1")
    actions = passes()
    actions["p1"] = Action(
        orders=[
            MoveUnit(action="move_unit", unit_id="does-not-exist", to="0,0"),
            MoveUnit(action="move_unit", unit_id=settler.id, to="99,99"),
            SetResearch(action="set_research", tech="not_a_tech"),
            SetRates(action="set_rates", tax_pct=70, science_pct=30),
        ]
    )
    state, events = step(state, actions)

    rejections = [e for e in events if e.type == "order_rejected"]
    assert len(rejections) == 3
    # The one valid order still applied.
    assert state.players["p1"].tax_pct == 70


def test_rates_must_sum_to_one_hundred() -> None:
    state = fresh()
    actions = passes()
    actions["p1"] = Action(orders=[SetRates(action="set_rates", tax_pct=70, science_pct=70)])
    state, events = step(state, actions)
    assert any(e.type == "order_rejected" for e in events)
    assert state.players["p1"].tax_pct == 50


def test_units_cannot_move_onto_impassable_terrain() -> None:
    state = fresh()
    unit = next(u for u in state.units_of("p1") if u.type is UnitType.SCOUT)
    target = next(
        (
            n
            for n in hx.neighbors(unit.hex)
            if (t := state.at(n)) is not None and t.terrain is Terrain.OCEAN
        ),
        None,
    )
    if target is None:
        pytest.skip("no adjacent ocean in this layout")
    actions = passes()
    actions["p1"] = Action(
        orders=[MoveUnit(action="move_unit", unit_id=unit.id, to=target.to_key())]
    )
    state, events = step(state, actions)
    assert any("impassable" in e.text for e in events if e.type == "order_rejected")


def test_attacking_without_declaring_war_is_rejected() -> None:
    state = fresh()
    attacker = next(u for u in state.units_of("p1") if u.type is UnitType.WARRIOR)
    victim = state.units_of("p2")[0]
    victim.pos = hx.neighbors(attacker.hex)[0].to_key()

    actions = passes()
    actions["p1"] = Action(orders=[Attack(action="attack", unit_id=attacker.id, target=victim.pos)])
    state, events = step(state, actions)
    assert any("declare war" in e.text for e in events if e.type == "order_rejected")


def test_combat_happens_once_at_war() -> None:
    state = fresh()
    attacker = next(u for u in state.units_of("p1") if u.type is UnitType.WARRIOR)
    victim = next(u for u in state.units_of("p2") if u.type is UnitType.WARRIOR)
    victim.pos = hx.neighbors(attacker.hex)[0].to_key()

    actions = passes()
    actions["p1"] = Action(
        diplomacy=[DeclareWar(action="declare_war", on="p2")],
        orders=[Attack(action="attack", unit_id=attacker.id, target=victim.pos)],
    )
    state, events = step(state, actions)
    assert any(e.type == "combat_resolved" for e in events)


# ---------------------------------------------------------------------------
# Diplomacy
# ---------------------------------------------------------------------------


def test_messages_are_delivered_the_following_turn() -> None:
    """One-turn latency is what makes negotiation legible rather than instant."""
    from arena_engine import diplomacy as dip

    state = fresh()
    actions = passes()
    actions["p1"] = Action(
        diplomacy=[
            SendMessage(action="send_message", channel="private", to="p2", text="Ally with me.")
        ]
    )
    state, _ = step(state, actions)
    assert dip.inbox_for(state, "p2", 8) == [], "same-turn delivery would be instant telepathy"

    state, _ = step(state, passes())
    inbox = dip.inbox_for(state, "p2", 8)
    assert len(inbox) == 1
    assert inbox[0].text == "Ally with me."


def test_private_messages_are_not_visible_to_third_parties() -> None:
    from arena_engine import diplomacy as dip

    state = fresh()
    actions = passes()
    actions["p1"] = Action(
        diplomacy=[SendMessage(action="send_message", channel="private", to="p2", text="secret")]
    )
    state, _ = step(state, actions)
    state, _ = step(state, passes())
    assert dip.inbox_for(state, "p3", 8) == []
    assert all("secret" not in m.text for m in dip.public_log(state, 8))


def test_accepting_a_trade_moves_gold_atomically() -> None:
    state = fresh()
    state.players["p1"].gold = 100
    state.players["p2"].gold = 10

    actions = passes()
    actions["p1"] = Action(
        diplomacy=[
            Propose(
                action="propose",
                to="p2",
                type=ProposalType.TRADE,
                terms=Terms(gold_to_them=60),
            )
        ]
    )
    state, _ = step(state, actions)
    proposal_id = next(iter(state.proposals))

    actions = passes()
    actions["p2"] = Action(
        diplomacy=[
            {"action": "respond_to_proposal", "proposal_id": proposal_id, "response": "accept"}
        ]
    )
    state, events = step(state, actions)
    assert any(e.type == "treaty_signed" for e in events)
    assert state.players["p1"].gold < 100
    assert state.players["p2"].gold > 10


def test_a_trade_that_cannot_be_paid_fails_wholesale() -> None:
    state = fresh()
    state.players["p1"].gold = 5
    actions = passes()
    actions["p1"] = Action(
        diplomacy=[
            Propose(
                action="propose",
                to="p2",
                type=ProposalType.TRADE,
                terms=Terms(gold_to_them=500),
            )
        ]
    )
    state, _ = step(state, actions)
    proposal_id = next(iter(state.proposals))
    before = state.players["p2"].gold

    actions = passes()
    actions["p2"] = Action(
        diplomacy=[
            {"action": "respond_to_proposal", "proposal_id": proposal_id, "response": "accept"}
        ]
    )
    state, events = step(state, actions)
    assert any(e.type == "proposal_failed" for e in events)
    assert state.players["p2"].gold == before, "a failed trade must move nothing"


def test_breaking_an_alliance_is_legal_and_logged() -> None:
    """Betrayal must be possible, and must be visible when it happens."""
    from arena_engine import diplomacy as dip

    state = fresh()
    dip.set_relation(state, "p1", "p2", RelationState.ALLIANCE)

    actions = passes()
    actions["p1"] = Action(
        diplomacy=[DeclareWar(action="declare_war", on="p2", casus_belli="Opportunity.")]
    )
    state, events = step(state, actions)

    assert any(e.type == "treaty_broken" for e in events), "betrayal must be recorded"
    assert any(e.type == "war_declared" for e in events), "and must actually succeed"
    assert state.at_war("p1", "p2")


def test_declaring_war_out_of_peace_is_not_a_betrayal() -> None:
    from arena_engine import diplomacy as dip

    state = fresh()
    dip.set_relation(state, "p1", "p2", RelationState.PEACE)
    actions = passes()
    actions["p1"] = Action(diplomacy=[DeclareWar(action="declare_war", on="p2")])
    state, events = step(state, actions)
    assert not any(e.type == "treaty_broken" for e in events)


def test_proposals_expire() -> None:
    state = fresh(proposal_ttl=2)
    actions = passes()
    actions["p1"] = Action(
        diplomacy=[Propose(action="propose", to="p2", type=ProposalType.PEACE, terms=Terms())]
    )
    state, _ = step(state, actions)
    assert state.proposals

    for _ in range(4):
        state, _ = step(state, passes())
    assert not state.proposals, "stale proposals must not accumulate in the payload"


# ---------------------------------------------------------------------------
# Dossier
# ---------------------------------------------------------------------------


def test_dossier_round_trips_verbatim() -> None:
    """What the agent wrote must be exactly what it reads back next turn."""
    from arena_engine.types import Dossier, OpponentModel, Trustworthiness

    written = Dossier(
        doctrine="Expand east, trade for military tech.",
        opponent_models=[
            OpponentModel(
                player_id="p2",
                assessed_intent="military expansion",
                trustworthiness=Trustworthiness.LOW,
                notes="Broke the turn-31 ceasefire.",
            )
        ],
        standing_commitments=["Non-aggression with p3 through turn 60."],
        lessons=["Stop attacking uphill."],
    )
    state = fresh()
    actions = passes()
    actions["p1"] = Action(dossier=written)
    state, _ = step(state, actions)
    assert state.players["p1"].dossier == written


def test_oversized_dossier_is_truncated_loudly() -> None:
    from arena_engine.types import Dossier

    state = fresh()
    actions = passes()
    actions["p1"] = Action(dossier=Dossier(lessons=[f"lesson {i}" for i in range(50)]))
    state, events = step(state, actions)

    assert any(e.type == "dossier_truncated" for e in events), "silent truncation hides the loss"
    assert len(state.players["p1"].dossier.lessons) <= 12


# ---------------------------------------------------------------------------
# Economy and legal actions
# ---------------------------------------------------------------------------


def test_cities_grow_and_produce_over_time() -> None:
    state = fresh()
    settler = settler_of(state, "p1")
    actions = passes()
    actions["p1"] = Action(
        orders=[FoundCity(action="found_city", unit_id=settler.id, name="Ravenholt")]
    )
    state, _ = step(state, actions)

    for _ in range(40):
        state, _ = step(state, bots.all_bot_actions(state))
    city = state.cities_of("p1")[0]
    assert city.population > 1, "a city on a fair start must be able to grow"
    assert state.players["p1"].known_techs, "science must accumulate"


def test_legal_actions_never_offers_an_order_the_reducer_rejects() -> None:
    """The contract that keeps the repair loop from firing constantly."""
    state = fresh(17)
    for _ in range(25):
        state, _ = step(state, bots.all_bot_actions(state))
        if state.victory:
            break

    for player_id in state.living_player_ids():
        legal = legal_actions(state, player_id)
        for unit_id, options in legal["units"].items():
            unit = state.units[unit_id]
            for target in options["move"]:
                assert hx.distance(unit.hex, hx.from_key(target)) == 1
                tile = state.at(hx.from_key(target))
                assert tile is not None
                assert not state.units_at(hx.from_key(target)) or all(
                    u.owner == player_id for u in state.units_at(hx.from_key(target))
                )
            if options["found_city"]:
                assert unit.type is UnitType.SETTLER
                assert all(
                    hx.distance(c.hex, unit.hex) >= MIN_CITY_SPACING
                    for _, c in state.cities.items()
                )


def test_buildable_items_are_actually_affordable_to_start() -> None:
    state = fresh()
    settler = settler_of(state, "p1")
    actions = passes()
    actions["p1"] = Action(orders=[FoundCity(action="found_city", unit_id=settler.id, name="X")])
    state, _ = step(state, actions)
    city = state.cities_of("p1")[0]
    options = economy.buildable(state, city)
    for item in options:
        assert economy.build_cost(item) > 0, f"{item} is free to build"
    # Wildlife has cost 0 and must never be offered to a civ at all.
    assert not (set(options) & {u.value for u in NEUTRAL_UNITS})


# ---------------------------------------------------------------------------
# Full-match integration: the engine's exit criterion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_a_full_bot_match_reaches_a_decisive_victory(seed: int) -> None:
    """The headline criterion from the plan.

    A match that runs the clock out on score is not a failure in itself, but if
    it happened on every seed it would mean the military and science paths were
    unreachable and the lab would measure nothing but turtling.
    """
    state = fresh(seed, turn_limit=300)
    turns = 0
    while state.victory is None and turns < 320:
        state, _ = step(state, bots.all_bot_actions(state))
        turns += 1

    assert state.victory is not None, f"seed {seed} never resolved"
    assert state.victory.winner is not None
    assert state.victory.condition != "turn_limit", (
        f"seed {seed} only resolved on the clock; the game should be decidable"
    )
    assert len(state.cities) > 6, "the map should actually get settled"


def test_matches_exercise_contact_war_and_capture() -> None:
    """A match where nobody ever meets would measure nothing."""
    from collections import Counter

    state = fresh(4, turn_limit=300)
    counts: Counter[str] = Counter()
    while state.victory is None and state.turn < 300:
        state, events = step(state, bots.all_bot_actions(state))
        counts.update(e.type for e in events)

    assert counts["first_contact"] > 0, "civs never saw each other"
    assert counts["war_declared"] > 0, "no war was ever declared"
    assert counts["combat_resolved"] > 0, "no combat ever happened"
    assert counts["city_founded"] > 4, "the map was never settled"


def test_no_state_invariant_is_violated_across_a_match() -> None:
    state = fresh(6, turn_limit=200)
    while state.victory is None and state.turn < 200:
        state, _ = step(state, bots.all_bot_actions(state))

        for player in state.players.values():
            assert player.gold >= 0, "gold went negative"
            assert player.tax_pct + player.science_pct == 100
        for unit in state.units.values():
            assert 0 < unit.hp <= 100, f"{unit.id} has hp {unit.hp}"
            assert unit.pos in state.tiles, "unit left the map"
            assert unit.owner in state.players
        for city in state.cities.values():
            assert city.population >= 1, "a city shrank out of existence"
            assert city.pos in state.tiles
            assert len(city.worked_tiles) <= city.population
            assert len(set(city.buildings)) == len(city.buildings), "duplicate building"


def test_worked_tiles_are_never_double_assigned() -> None:
    """Two cities working the same tile would silently double its yield."""
    state = fresh(8, turn_limit=120)
    while state.victory is None and state.turn < 120:
        state, _ = step(state, bots.all_bot_actions(state))
        claimed: list[str] = []
        for city in state.cities.values():
            claimed.extend(city.worked_tiles)
        assert len(claimed) == len(set(claimed)), "a tile was worked by two cities"


def test_wonders_are_globally_unique() -> None:
    state = fresh(2, turn_limit=200)
    while state.victory is None and state.turn < 200:
        state, _ = step(state, bots.all_bot_actions(state))
    from arena_engine.content import WONDERS

    built = [b for c in state.cities.values() for b in c.buildings if b in WONDERS]
    assert len(built) == len(set(built)), f"a wonder was built twice: {built}"
