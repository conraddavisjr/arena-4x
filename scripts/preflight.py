#!/usr/bin/env python
"""Can every seat in this roster actually take a turn, and can the accounts pay?

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

**The projection used to be a guess, and it was about half of true.** Five flat
constants lived here - 2k input, 4k output, 6.5k cached per turn - and projected
$21.88 for a 300-turn shakeout that the journals put at $46.80. The prompt is
not flat, it is a ramp that grows with the board, and one output figure cannot
cover seats that differ by seven times. The measurements now live in
`arena_orchestrator.profiles` with the match each came from; see that module for
what went wrong and `make profiles` to re-derive it. The lesson is the one from
the rate card: a number nobody has checked against an artifact is not a
measurement, however many decimal places it carries.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from arena_orchestrator.profiles import (  # noqa: E402
    MEASURED_FINGERPRINT,
    observation_fingerprint,
    project,
)
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

# How close to the safety halt a projection may sit before this says so. At 0.8
# a run projected at the top of its range still has a fifth of the cap in hand
# for retries and the turns nobody has measured. Below that margin the halt is a
# live possibility rather than a backstop, and a match that stops on
# `budget_cap` at turn 250 is not a baseline, it is a truncated one.
CAP_MARGIN = 0.8


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


async def _cli() -> int:
    import run_match

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", default="shakeout", choices=run_match.ROSTERS)
    parser.add_argument("--turns", type=int, default=300)
    parser.add_argument(
        "--budget",
        type=float,
        # Importing run_match above has already loaded .env, so this is the cap
        # the match would actually run under and not a second guess at it.
        default=run_match.default_budget(),
        help="the match's dollar cap, to check the projection against",
    )
    args = parser.parse_args()
    return await check(args.roster, args.turns, budget_usd=args.budget)


async def check(roster: str, turns: int, *, budget_usd: float | None = None) -> int:
    """Probe every seat. Returns 0 when all can spend, 1 otherwise.

    Importable so `run_match` can call it rather than shelling out, which keeps
    one implementation of "is this roster ready" instead of two that drift.

    The dollar cap is a warning and never a failure. Whether a run that might
    halt early is worth starting is a judgement about the experiment, not about
    the accounts, and this function only has standing to refuse on the latter.
    """
    import run_match

    seats = run_match.roster_for(roster)
    if roster == "dry":
        print("dry roster: no vendors, nothing to check")
        return 0

    print(f"preflight: {roster} roster, {turns} turns\n")
    results = await asyncio.gather(*(probe(p, m) for p, m in seats))

    central = high = 0.0
    unpriced: list[str] = []
    unmeasured: list[str] = []
    broke: list[tuple[str, str]] = []

    print(f"  {'':4}  {'vendor':10} {'model':24} {'likely':>9} {'high':>9}")
    for (provider, model), (status, detail) in zip(seats, results, strict=True):
        projection = project(model or "", turns)
        if projection is None:
            unpriced.append(str(model))
            money = f"{'unpriced':>19}"
        else:
            central += projection.central
            high += projection.high
            money = f"~${projection.central:7.2f} ~${projection.high:7.2f}"
            if not projection.measured:
                unmeasured.append(str(model))
                money += " *"
        if status != "ok":
            broke.append((provider, status))
        mark = "  ok" if status == "ok" else "FAIL"
        note = "" if status == "ok" else status
        print(f"  {mark}  {provider:10} {str(model):24} {money}   {note}".rstrip())
        if status != "ok":
            print(f"        {detail}")

    print(f"\n  projected for {turns} turns:  likely ~${central:.2f}   high ~${high:.2f}")
    print("  Compare the HIGH column against each billing page, per vendor. The")
    print("  question is not what this will probably cost, it is whether the")
    print("  account survives it going badly.")

    if unpriced:
        print(f"\n  UNPRICED, so both totals are an undercount: {', '.join(unpriced)}")
    if unmeasured:
        print("\n  * NEVER RUN, so the shape is borrowed from the hungriest measured")
        print(f"    seat and only the rate is its own: {', '.join(unmeasured)}")
    if observation_fingerprint() != MEASURED_FINGERPRINT:
        print("\n  The observation schema has changed since these profiles were")
        print("  measured, so the prompt side of both totals is an estimate of")
        print("  unknown tightness. Re-derive with `make profiles` after this run.")
    if budget_usd is not None:
        headroom = high / budget_usd if budget_usd else float("inf")
        print(
            f"\n  safety halt: ${budget_usd:,.2f}, and the high projection is {headroom:.0%} of it"
        )
        if headroom > 1:
            print("  THE CAP WILL STOP THIS RUN in the high case. The match would end")
            print("  on `budget_cap` mid-board, which scores a position rather than")
            print("  finishing a match. Raise --budget or shorten --turns.")
        elif headroom > CAP_MARGIN:
            print("  Thin. Retries and the turns past 128 that nobody has measured")
            print("  both come out of what is left. Consider raising --budget.")

    if broke:
        print("\n  NOT READY:")
        for provider, status in broke:
            hint = {
                "NO CREDIT": "add credits to this account before starting",
                "SDK": "run `make setup`",
            }.get(status, "see the detail above")
            print(f"    {provider}: {status} - {hint}")
        return 1
    print("\n  All seats can spend. That is a probe of one token, not a balance:")
    print("  no API here reports a remaining balance, so the totals above are")
    print("  the only thing standing between you and a run that dies at turn 240.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_cli()))
