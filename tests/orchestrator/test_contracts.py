"""Live contract tests. Skipped unless the matching key is present.

These are the only tests that spend money, and they exist to answer one
question the mocked tests cannot: *does this vendor still accept the request we
send?* Every adapter in this package was written against documentation, and all
four of these APIs changed shape between this project's design and its
implementation - OpenAI moved to the Responses API, Google moved to
`interactions.create`. The next such change will arrive without warning, and the
worst place to discover it is four hours into an unattended flagship run.

Each test is a few cents. Run them before a flagship run, not in CI:

    make contracts

The Anthropic cache assertion is the one to watch. If `cache_read_tokens` is
zero on the second identical-prefix request, something has crept into the
system block - a timestamp, an unsorted dict, a turn counter - and the match
will quietly cost several times what it should.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from arena_engine.actions import Action
from arena_orchestrator.providers import build

# Deliberately trivial and provider-neutral: the point is the request shape, not
# the model's ability. Authored to Anthropic's dialect - additionalProperties on
# every object, no numeric bounds - which is the most restrictive of the four,
# so the same schema is accepted unmodified by all of them.
PROBE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["civ_name", "aggression"],
    "properties": {
        "civ_name": {"type": "string"},
        "aggression": {"enum": ["low", "medium", "high"]},
    },
}

SYSTEM = (
    "You are the sovereign of a small civilisation in a turn-based strategy game. "
    "Answer with the requested object and nothing else."
)
USER = "Name your civilisation and state your posture toward your neighbours."

KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
}

SDKS = {
    "anthropic": "anthropic",
    "openai": "openai",
    "xai": "openai",
    "google": "google.genai",
}


def requires(provider: str) -> None:
    """Skip with the *specific* reason, not a generic one.

    Two different things stop these running - no key, and no SDK - and an
    engine-only checkout has neither. A skip that said only "not set" sent you
    hunting for a key you already had while the real problem was that
    `make setup-engine` does not install the vendor SDKs.
    """
    pytest.importorskip(SDKS[provider], reason=f"{SDKS[provider]} not installed; run `make setup`")
    if not os.environ.get(KEYS[provider]):
        pytest.skip(f"{KEYS[provider]} not set (export it, or put it in .env)")


@pytest.mark.contract
@pytest.mark.parametrize("provider", sorted(KEYS))
async def test_the_vendor_still_accepts_our_request_shape(provider: str) -> None:
    requires(provider)
    client = build(provider)
    try:
        turn = await client.complete(SYSTEM, USER, PROBE_SCHEMA)
    finally:
        await client.aclose()

    parsed = json.loads(turn.text)
    assert set(parsed) == {"civ_name", "aggression"}
    assert parsed["aggression"] in {"low", "medium", "high"}
    # A usage payload of all zeros means the field names moved, which would
    # leave the budget meter stuck at zero for a whole match.
    assert turn.usage.output_tokens > 0
    assert turn.latency_ms > 0


@pytest.mark.contract
@pytest.mark.parametrize("provider", sorted(KEYS))
async def test_the_real_action_schema_is_accepted(provider: str) -> None:
    """The probe above uses a toy schema, which proves the *call* works and
    almost nothing about whether a match will.

    The real action schema is two orders of magnitude larger, deeply nested,
    and authored to Anthropic's dialect - no numeric bounds, no string lengths,
    `additionalProperties: false` on every object - precisely so the same bytes
    are accepted by all four vendors. Whether that actually holds is the single
    assumption a flagship run rests on, and finding out on turn one of an
    unattended multi-day match is the wrong time.
    """
    requires(provider)
    schema = json.loads(
        (Path(__file__).resolve().parents[2] / "schemas" / "action.schema.json").read_text()
    )
    client = build(provider)
    try:
        turn = await client.complete(
            "You are the sovereign of a small civilisation. It is turn 1.",
            "You have one settler at 0,0 and 40 gold. Found your capital and "
            "state your opening plan. Return only the action object.",
            schema,
        )
    finally:
        await client.aclose()

    action = Action.model_validate_json(turn.text)
    assert action.reasoning.plan_this_turn, "no plan came back"


@pytest.mark.contract
async def test_anthropic_actually_caches_the_system_prefix() -> None:
    """The single highest-value contract test in the repo.

    A cache miss does not fail anything - it just multiplies the input bill by
    ten and nothing looks wrong. This is the only way to notice.
    """
    requires("anthropic")
    client = build("anthropic")
    try:
        # The prefix must clear the 512-token minimum to be cacheable at all.
        system = SYSTEM + "\n" + ("Rules reference. " * 400)
        first = await client.complete(system, USER, PROBE_SCHEMA)
        second = await client.complete(system, "Now name a rival.", PROBE_SCHEMA)
    finally:
        await client.aclose()

    written = first.usage.cache_write_tokens + first.usage.cache_read_tokens
    assert written > 0, "nothing was cached on the first request"
    assert second.usage.cache_read_tokens > 0, (
        "the second identical-prefix request did not hit the cache - "
        "something dynamic has crept into the system block"
    )
