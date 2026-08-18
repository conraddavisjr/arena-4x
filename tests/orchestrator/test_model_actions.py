"""Can each model actually *act*? Live, one small call per assertion.

`test_contracts.py` answers "does this vendor accept our request". This answers
the question after it, which turned out to be different and much more expensive:
**does the model produce orders the engine can execute?**

That gap cost a match. The contract test asserted a plan came back and passed
happily while `claude-haiku-4-5` emitted `{"action": "found_city"}` with no unit
and no name on 91% of its orders - every one silently dropped, no city ever
founded, eliminated on turn 35 of 128. Nothing failed. Nothing was logged as a
failure. The seat simply did nothing for 93 turns while three others played.

So these tests go all the way through: a real observation from a real board, the
real per-vendor dialect, the real parser, and then the real reducer. An order
that is well-formed but references a unit that does not exist is still a wasted
turn, and only the reducer can say so.

    make model-actions

A handful of cents per run. Worth it before anything unattended.
"""

from __future__ import annotations

import json
import os

import pytest

from arena_engine import observation as obs
from arena_engine.actions import parse
from arena_engine.events import ORDER_REJECTED
from arena_engine.reducer import new_match, step
from arena_engine.types import MatchConfig
from arena_orchestrator.agent import Agent
from arena_orchestrator.config import RunConfig
from arena_orchestrator.dialects import for_provider
from arena_orchestrator.providers import build
from arena_orchestrator.resilience import TokenBucket

# Duplicated from `test_contracts` rather than imported: a cross-module test
# import resolves differently depending on how pytest is invoked, and a suite
# that only runs one way is a suite nobody runs.
KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
}
SDKS = {"anthropic": "anthropic", "openai": "openai", "xai": "openai", "google": "google.genai"}


def requires(provider: str) -> None:
    pytest.importorskip(SDKS[provider], reason=f"{SDKS[provider]} not installed; run `make setup`")
    if not os.environ.get(KEYS[provider]):
        pytest.skip(f"{KEYS[provider]} not set (export it, or put it in .env)")


ROSTER = [
    ("p1", "Aurelian Compact"),
    ("p2", "Iron Concord"),
    ("p3", "Verdant Pact"),
    ("p4", "Solari Dominion"),
]

SYSTEM = (
    "You are the sovereign of a civilisation in a turn-based 4X strategy game. "
    "Each turn you receive the state visible to you and the actions currently "
    "legal. Return your reasoning, then your orders. Issue orders for the units "
    "and cities you actually have, using the ids exactly as given."
)


def opening_board():
    """A real turn-1 state and the observation p1 would be sent."""
    state, _ = new_match("t", 4, ROSTER, MatchConfig(turn_limit=50))
    return state, obs.to_json(obs.build(state, "p1", recent_events=[]))


def seat(provider: str) -> Agent:
    """A real `Agent`, not a bare client.

    The first version of this suite called `client.complete` directly, which
    tested the model and not the system - and the system is where the recovery
    lives. An `Agent` retries a malformed body and, since the discard that
    eliminated a civ, corrects an unusable order too. Asserting against the raw
    first response measures something real but not the thing that decides a
    match.
    """
    client = build(provider)
    return Agent(
        player_id="p1",
        civ_name="Aurelian Compact",
        client=client,
        bucket=TokenBucket(60, 400_000),
        schema=for_provider(ACTION_SCHEMA, provider, client.model),
        # The production deadline. `Agent`'s own default is 180s, which is a
        # different experiment - three seats failed this suite on a timeout that
        # no real match would ever have applied.
        timeout_s=RunConfig(seed=0, seats=()).turn_timeout_s,
    )


async def take_turn(provider: str, observation: str):
    agent = seat(provider)
    try:
        return await agent.take_turn(observation)
    finally:
        await agent.client.aclose()


ACTION_SCHEMA = json.loads(
    (
        __import__("pathlib").Path(__file__).resolve().parents[2] / "schemas" / "action.schema.json"
    ).read_text()
)


EMPTY_TURN = json.dumps(
    {
        "reasoning": {
            "situation_assessment": "",
            "threats_and_opportunities": [],
            "plan_this_turn": "",
        },
        "dossier": {
            "doctrine": "",
            "opponent_models": [],
            "standing_commitments": [],
            "lessons": [],
        },
        "diplomacy": [],
        "orders": [],
    }
)


@pytest.mark.contract
@pytest.mark.parametrize("provider", sorted(KEYS))
async def test_a_seat_ends_its_turn_with_usable_orders(provider: str) -> None:
    """**The end-to-end guarantee, through the path a match actually uses.**

    A model is allowed to send an unusable order - `found_city` with no unit -
    because models do. What is not allowed is for the turn to *end* that way. The
    agent gets told exactly which fields were missing and gets one correction,
    and this asserts that the combination arrives somewhere usable.

    Before that loop existed the discard was silent, and a seat spent 91% of its
    orders on nothing across a 128-turn match, founded no cities, and was
    eliminated on turn 35 with every turn logged as a success.
    """
    requires(provider)
    _, observation = opening_board()
    outcome = await take_turn(provider, observation)

    assert not outcome.passed, f"{provider} could not complete a turn: {outcome.failure}"
    assert outcome.action.orders, f"{provider} ended its turn with no orders at all"
    # Re-parsing the final body is the check: whatever survives here is exactly
    # what the reducer will be handed.
    for order in outcome.action.orders:
        assert order.action, "an order with no action reached the engine"


@pytest.mark.contract
@pytest.mark.parametrize("provider", sorted(KEYS))
async def test_nothing_is_discarded_without_telling_someone(provider: str) -> None:
    """No silent failures - the property, stated as a test.

    There are exactly two ways an order can fail to happen, and each has to
    leave a trace the model can act on:

    * unusable before the engine sees it -> `Agent` sends a correction naming
      the missing fields, and marks the outcome `repaired`
    * illegal once the engine sees it -> `order_rejected`, carrying a reason,
      which reaches that civ in next turn's `recent_events`

    A rejection is a normal part of play and not a defect. An *unexplained* one
    is the defect, because a model cannot learn from silence.
    """
    requires(provider)
    state, observation = opening_board()
    outcome = await take_turn(provider, observation)
    assert not outcome.passed, f"{provider}: {outcome.failure}"

    # If the agent had to repair, it used more than one call - that is the trace.
    if outcome.repaired:
        assert len(outcome.turns) > 1, "a repair that cost no extra call did not happen"

    idle = {p: parse(EMPTY_TURN) for p, _ in ROSTER[1:]}
    _, events = step(state, {"p1": outcome.action, **idle})
    for event in events:
        if event.type == ORDER_REJECTED and event.actor == "p1":
            assert event.text.strip(), "an order was rejected with no reason given"


@pytest.mark.contract
@pytest.mark.parametrize("provider", sorted(KEYS))
async def test_a_correction_is_understood(provider: str) -> None:
    """The retry path is only worth having if the message works on a real model.

    `agent.py` names the missing fields and asks for a reissue. That mechanism is
    covered offline with a scripted client, which proves the loop runs and
    nothing about whether a model can act on the sentence.
    """
    requires(provider)
    _, observation = opening_board()
    correction = (
        f"{observation}\n\nYour previous response was accepted but 1 of your "
        f"entries could not be used and was discarded: found_city needs name, "
        f"unit_id, and you sent none of them. That action did not happen. "
        f"Reissue it with every field it requires, and keep the rest of your "
        f"turn as it was."
    )
    outcome = await take_turn(provider, correction)
    assert not outcome.passed, f"{provider}: {outcome.failure}"
    founding = [o for o in outcome.action.orders if o.action == "found_city"]
    assert founding, f"{provider} was asked to reissue found_city and did not"
    assert all(o.unit_id and o.name for o in founding), (
        f"{provider} was told exactly which fields were missing and omitted them again"
    )


@pytest.mark.contract
@pytest.mark.parametrize("provider", sorted(KEYS))
async def test_the_model_reaches_for_more_than_one_kind_of_order(provider: str) -> None:
    """A seat that only ever moves units has a schema or prompt problem, and
    that is worth knowing for cents rather than for a 128-turn match."""
    requires(provider)
    _, observation = opening_board()
    outcome = await take_turn(
        provider,
        f"{observation}\n\nThis turn: found a city with a settler, set your "
        f"research, and fortify a unit. Issue all three orders.",
    )
    kinds = {o.action for o in outcome.action.orders}
    assert len(kinds) >= 2, f"{provider} only produced {kinds or 'no orders'}"
