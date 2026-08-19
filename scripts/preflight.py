#!/usr/bin/env python
"""Can every seat in this roster actually take a turn, right now?

    python scripts/preflight.py --roster shakeout --turns 300

Four tiny live calls and a cost projection, for a few hundredths of a cent. Run
it before anything unattended; `run_match.py` runs it automatically.

**Why this exists.** A 300-turn baseline reached turn 40 and then played 29 more
turns with one seat out of API credits - holding its cities, issuing no orders,
on its way to producing a four-way comparison missing a fourth. It cost about
six hours of wall clock and would have cost eighteen dollars more. The failure
was a billing page, and every piece of software involved behaved exactly as
designed.

**It probes rather than reads a balance**, which is deliberate. None of the four
vendors exposes "how much credit is left" on a normal API key - the ones that
offer it at all want an admin key this project does not hold. But every one of
them will tell you immediately if you cannot spend, because the request fails.
So the check is not "what is the balance" but "can this key buy a token right
now", which is the question that actually matters and the only one answerable.

What it cannot tell you is whether the balance covers the *whole* run. That is
what the projection is for: it prints the expected spend per seat from measured
token profiles, so the number you compare against your billing page is one you
did not have to guess.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from arena_orchestrator.pricing import UnknownModel, rate_for  # noqa: E402
from arena_orchestrator.providers import build  # noqa: E402
from arena_orchestrator.providers.base import OutOfCredits, ProviderError  # noqa: E402

# Deliberately tiny. The point is whether the account can spend, not whether the
# model can play - the real schema is exercised by `make contracts`, which costs
# real money and is a different question.
PROBE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ok"],
    "properties": {"ok": {"enum": ["yes"]}},
}

# Roughly what one turn costs each seat, in tokens, measured over completed
# matches at medium effort. Only used for the projection, so being approximate
# is fine; being absent would not be.
PER_TURN = {
    "input": 2_000,
    "output": 4_000,
    "cached": 6_500,
}


async def probe(provider: str, model: str | None) -> tuple[str, str]:
    """Returns (status, detail). Never raises - a failed seat is the finding."""
    try:
        client = build(provider, model)
    except Exception as error:  # noqa: BLE001 - a missing SDK is a finding too
        return "SDK", str(error)[:90]
    try:
        turn = await client.complete('Reply with {"ok":"yes"}.', "Ready?", PROBE)
        return "ok", f"{turn.usage.output_tokens} output tokens"
    except OutOfCredits as error:
        # The one this script exists for, named rather than lumped in with the
        # other fatals - because it is the only one a billing page fixes.
        return "NO CREDIT", str(error)[:90]
    except ProviderError as error:
        return type(error).__name__, str(error)[:90]
    except Exception as error:  # noqa: BLE001
        return "ERROR", str(error)[:90]
    finally:
        await client.aclose()


def projected(model: str | None, turns: int) -> float | None:
    try:
        rate = rate_for(model or "")
    except UnknownModel:
        return None
    return (
        (
            PER_TURN["input"] * rate.input
            + PER_TURN["output"] * rate.output
            + PER_TURN["cached"] * rate.input * rate.cache_read_multiplier
        )
        / 1_000_000
        * turns
    )


async def _cli() -> int:
    import run_match

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", default="shakeout", choices=run_match.ROSTERS)
    parser.add_argument("--turns", type=int, default=300)
    args = parser.parse_args()
    return await check(args.roster, args.turns)


async def check(roster: str, turns: int) -> int:
    """Probe every seat. Returns 0 when all can spend, 1 otherwise.

    Importable so `run_match` can call it rather than shelling out, which keeps
    one implementation of "is this roster ready" instead of two that drift.
    """
    import run_match

    seats = run_match.roster_for(roster)
    if roster == "dry":
        print("dry roster: no vendors, nothing to check")
        return 0

    print(f"preflight: {roster} roster, {turns} turns\n")
    results = await asyncio.gather(*(probe(p, m) for p, m in seats))

    total = 0.0
    unpriced = []
    broke = []
    for (provider, model), (status, detail) in zip(seats, results, strict=True):
        cost = projected(model, turns)
        if cost is None:
            unpriced.append(model)
        else:
            total += cost
        if status != "ok":
            broke.append((provider, status))
        mark = "  ok " if status == "ok" else "FAIL"
        money = f"~${cost:6.2f}" if cost is not None else "  unpriced"
        note = "" if status == "ok" else status
        print(f"  {mark}  {provider:10} {str(model):24} {money}   {note}")
        if status != "ok":
            print(f"        {detail}")

    print(f"\n  projected total for {turns} turns: ~${total:.2f}")
    if unpriced:
        print(f"  UNPRICED, so this total is an undercount: {', '.join(map(str, unpriced))}")
    if broke:
        print("\n  NOT READY:")
        for provider, status in broke:
            hint = {
                "NO CREDIT": "add credits to this account before starting",
                "SDK": "run `make setup`",
            }.get(status, "see the detail above")
            print(f"    {provider}: {status} - {hint}")
        return 1
    print("  all seats can spend. The projection is per-vendor; check it against")
    print("  each billing page, because no API here reports a remaining balance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_cli()))
