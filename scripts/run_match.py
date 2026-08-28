#!/usr/bin/env python
"""Play a match through the orchestrator.

    python scripts/run_match.py                     # free, bot-driven dry run
    python scripts/run_match.py --roster shakeout   # small models, a few dollars
    python scripts/run_match.py --roster flagship   # the real thing
    python scripts/run_match.py --resume output/run-4

The dry run is the default on purpose. Every layer above the HTTP call is
exercised by it - observation, schema, ledger, journal, bundle - so the
expensive rosters should only ever be reached by a deliberate flag.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

# Same .env loading the test suite does, so `make run --roster shakeout` does
# not fail for a key that is sitting in the file right next to it.
_ENV = Path(__file__).resolve().parents[1] / ".env"
if _ENV.exists():
    for _line in _ENV.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip("\"'"))

from arena_engine import victory  # noqa: E402
from arena_engine.types import MatchConfig  # noqa: E402
from arena_orchestrator import journal as jl  # noqa: E402
from arena_orchestrator.config import RunConfig, Seat  # noqa: E402
from arena_orchestrator.dryrun import bot_seats  # noqa: E402
from arena_orchestrator.loop import Orchestrator  # noqa: E402

# Flavour names, used only when a seat has no model to name it after - the
# bot-driven dry run.
CIVS = ["Aurelian Compact", "Iron Concord", "Verdant Pact", "Solari Dominion"]


# The family name, not the version. `claude-haiku-4-5` and `claude-opus-5` are
# both "Claude" at the table, which is what a rival would actually say.
FAMILY = {
    "anthropic": "Claude",
    "openai": "GPT",
    "google": "Gemini",
    "xai": "Grok",
}


def civ_name(index: int, provider: str, model: str | None) -> str:
    """What the agents call each other.

    The model's family name, deliberately - "Claude", "GPT", "Gemini", "Grok".
    It goes into the system prompt and into every message an agent writes, so
    "Greetings Iron Concord" becomes "Greetings Gemini" and a transcript reads
    without a decoder ring.

    The *family* rather than the full id, because the version is noise in a
    conversation. "Greetings gpt-5.4-mini" is a mouthful nobody would say, and
    when the flagship roster swaps `gpt-5.4-mini` for `gpt-5.6` the transcripts
    stay comparable. The exact model still lives in the journal, the bundle and
    the cost table, where the version actually matters.

    Worth being explicit that this is a choice with a cost: every agent knows
    which model it is and which models it faces. Whether a model plays
    differently knowing it is Claude facing Grok is a real question, and this
    setting answers it in one direction. Run with `CIVS` names instead for the
    other.
    """
    if provider == "bot":
        return CIVS[index]
    return FAMILY.get(provider) or model or provider


PROVIDERS = ("anthropic", "openai", "google", "xai")

# Which model plays each seat, overridable per roster from the environment.
#
# Model ids move faster than anything else in this project - `gemini-3.6-pro`,
# `grok-4` and `gpt-5.6-mini` were all in here and none of them exists - so they
# are configuration rather than code. Set `ARENA_FLAGSHIP_GOOGLE` in .env and
# the flagship roster uses it; leave it unset and the default below applies.
DEFAULT_MODELS = {
    # A full match for a few dollars, to shake out the loop before spending
    # real money. Caches are model-scoped, so this shares none with flagship.
    "shakeout": {
        "anthropic": "claude-haiku-4-5",
        "openai": "gpt-5.4-mini",
        "google": "gemini-3.6-flash",
        "xai": "grok-4.3",
    },
    "flagship": {
        "anthropic": "claude-opus-5",
        "openai": "gpt-5.6",
        # No Gemini pro exists above 3.1, and that one is still a preview.
        # `gemini-3.7-flash` is newer and GA if you would rather trade tier for
        # recency; both pass the contract test.
        "google": "gemini-3.1-pro-preview",
        "xai": "grok-4.6",
    },
}


def env_var(roster: str, provider: str) -> str:
    return f"ARENA_{roster.upper()}_{provider.upper()}"


def roster_for(name: str) -> list[tuple[str, str | None]]:
    """The (provider, model) pairs for a roster, environment first."""
    if name == "dry":
        # Free. Bot heuristics behind the provider seam; no key, no network.
        return [("bot", None)] * 4
    defaults = DEFAULT_MODELS[name]
    return [
        (provider, os.environ.get(env_var(name, provider)) or defaults[provider])
        for provider in PROVIDERS
    ]


ROSTERS = ("dry", "shakeout", "flagship")


def default_budget() -> float:
    """The safety halt, from .env or the built-in default.

    A function rather than an inline `os.environ.get` because preflight needs the
    same number: it now checks the projection against the cap, and a preflight
    that cleared a $75 run the match then started at $100 would be reassuring
    about something that was not going to happen.
    """
    return float(os.environ.get("MATCH_BUDGET_USD", 75))


KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
}


def build_config(args: argparse.Namespace) -> RunConfig:
    seats = tuple(
        Seat(
            player_id=f"p{i + 1}",
            civ_name=civ_name(i, provider, model),
            provider=provider,
            model=model,
        )
        for i, (provider, model) in enumerate(roster_for(args.roster))
    )
    # The throttle exists to protect a vendor account. A dry run has no vendor,
    # so leaving it at the default made a 300-turn match spend eight minutes
    # asleep waiting for a rate limit nobody was enforcing.
    unthrottled = args.roster == "dry"
    return RunConfig(
        seed=args.seed,
        seats=seats,
        # A third of the original wildlife. Set here rather than in the engine
        # default, because `MatchConfig` is inside the state hash and moving the
        # default would silently replay every past match as a different one -
        # journals written before the field existed carry no value for it.
        match=MatchConfig(turn_limit=args.turns, wilderness=args.wilderness),
        budget_usd=args.budget,
        agent_budget_awareness=args.awareness,
        requests_per_minute=1e9 if unthrottled else 50.0,
        tokens_per_minute=1e12 if unthrottled else 400_000.0,
        **({"turn_timeout_s": args.timeout} if args.timeout else {}),
    )


def check_keys(roster: str) -> None:
    missing = sorted(
        {KEYS[p] for p, _ in roster_for(roster) if p in KEYS and not os.environ.get(KEYS[p])}
    )
    if missing:
        raise SystemExit(f"missing API keys: {', '.join(missing)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", choices=ROSTERS, default="dry")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--turns", type=int, default=300)
    parser.add_argument("--budget", type=float, default=default_budget())
    parser.add_argument("--awareness", choices=["off", "tokens"], default="off")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="seconds before a turn is abandoned and the agent passes",
    )
    parser.add_argument(
        "--wilderness",
        type=float,
        default=0.34,
        help="wolf and raider density; 1.0 is the original setting, 0 empties the map",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="skip the live check that every seat can actually spend",
    )
    args = parser.parse_args()

    if args.roster != "dry":
        check_keys(args.roster)
        # A key that exists is not a key that can pay, and the difference costs
        # hours. A 300-turn baseline once reached turn 40 and played 29 more
        # with one seat out of credits - holding cities, issuing no orders -
        # because nothing had asked the question before starting. Four tiny
        # calls, a few hundredths of a cent, and it is on by default because the
        # run it protects is unattended.
        if not args.no_preflight:
            import preflight

            if await preflight.check(args.roster, args.turns, budget_usd=args.budget):
                raise SystemExit(
                    "preflight failed - fix the seats above, or pass --no-preflight "
                    "to start anyway."
                )

    root = args.resume or args.out or Path("output") / f"run-{args.roster}-{args.seed}"
    config = build_config(args)

    clients, handle = (None, None)
    if args.roster == "dry":
        clients, handle = bot_seats(seat.player_id for seat in config.seats)

    orchestrator = Orchestrator(
        config=config,
        root=root,
        clients=clients,
        after_turn=handle.observe if handle else None,
    )
    result = await orchestrator.run(resume=args.resume is not None)

    print(f"\n{root}")
    print(f"  turns        {result.state.turn}")
    print(f"  outcome      {result.reason} -> {result.winner or '-'}")
    print(f"  spent        ${result.ledger.spent_usd:,.2f} of ${config.budget_usd:,.2f}")
    print(f"  requests     {result.ledger.requests}")
    print(f"  passed turns {result.failures}")
    print(f"  {'civ':<20} {'score':>6} {'spent':>9} {'per 100k':>9}  model")
    for seat in config.seats:
        points = victory.score(result.state, seat.player_id)
        spent = result.ledger.by_agent.get(seat.player_id, 0.0)
        efficiency = result.ledger.score_per_100k(seat.player_id, points)
        print(
            f"  {seat.civ_name:<20} {points:>6} ${spent:>8.3f} {efficiency:>9.1f}  "
            f"{seat.model or seat.provider}"
        )
    # A halt is not an ending, and the operator is the only one who can tell
    # the difference between "this match is over" and "this match is waiting for
    # a billing page". Printing the exact command is the whole affordance: the
    # run that this protects is unattended, and whoever reads this scrollback
    # hours later should not have to work out that resuming is even possible.
    if result.reason in jl.HALTS:
        fix = {
            "provider_credits": "top up the account that ran dry",
            "budget_cap": f"raise the cap above ${result.ledger.spent_usd:,.2f}",
        }[result.reason]
        print(f"\n  STOPPED, NOT FINISHED. {fix}, then carry on from turn {result.state.turn}:")
        print(f"    make preflight ROSTER={args.roster} TURNS={args.turns}")
        print(
            f"    {sys.executable} scripts/run_match.py --roster {args.roster} "
            f"--seed {args.seed} --turns {args.turns} --resume {root}"
        )
        print("  Spend so far is carried forward, so the cap still counts the")
        print("  whole match rather than restarting at zero.")

    print(f"\nthen: make view3d MATCH={root}/bundle")


if __name__ == "__main__":
    asyncio.run(main())
