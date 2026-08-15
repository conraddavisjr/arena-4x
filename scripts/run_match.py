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

from arena_engine import victory  # noqa: E402
from arena_engine.types import MatchConfig  # noqa: E402
from arena_orchestrator.config import RunConfig, Seat  # noqa: E402
from arena_orchestrator.dryrun import bot_seats  # noqa: E402
from arena_orchestrator.loop import Orchestrator  # noqa: E402

CIVS = ["Aurelian Compact", "Iron Concord", "Verdant Pact", "Solari Dominion"]

ROSTERS = {
    # Free. Bot heuristics behind the provider seam; no key, no network.
    "dry": [("bot", None)] * 4,
    # A full match for a few dollars, to shake out the loop before spending
    # real money. Caches are model-scoped, so this shares none with flagship.
    "shakeout": [
        ("anthropic", "claude-haiku-4-5"),
        ("openai", "gpt-5.6-mini"),
        ("google", "gemini-3.6-flash"),
        ("xai", "grok-4-fast"),
    ],
    "flagship": [
        ("anthropic", "claude-opus-5"),
        ("openai", "gpt-5.6"),
        ("google", "gemini-3.6-pro"),
        ("xai", "grok-4"),
    ],
}

KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
}


def build_config(args: argparse.Namespace) -> RunConfig:
    seats = tuple(
        Seat(player_id=f"p{i + 1}", civ_name=CIVS[i], provider=provider, model=model)
        for i, (provider, model) in enumerate(ROSTERS[args.roster])
    )
    # The throttle exists to protect a vendor account. A dry run has no vendor,
    # so leaving it at the default made a 300-turn match spend eight minutes
    # asleep waiting for a rate limit nobody was enforcing.
    unthrottled = args.roster == "dry"
    return RunConfig(
        seed=args.seed,
        seats=seats,
        match=MatchConfig(turn_limit=args.turns),
        budget_usd=args.budget,
        agent_budget_awareness=args.awareness,
        requests_per_minute=1e9 if unthrottled else 50.0,
        tokens_per_minute=1e12 if unthrottled else 400_000.0,
    )


def check_keys(roster: str) -> None:
    missing = sorted(
        {KEYS[p] for p, _ in ROSTERS[roster] if p in KEYS and not os.environ.get(KEYS[p])}
    )
    if missing:
        raise SystemExit(f"missing API keys: {', '.join(missing)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", choices=sorted(ROSTERS), default="dry")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--turns", type=int, default=300)
    parser.add_argument(
        "--budget", type=float, default=float(os.environ.get("MATCH_BUDGET_USD", 75))
    )
    parser.add_argument("--awareness", choices=["off", "tokens"], default="off")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    if args.roster != "dry":
        check_keys(args.roster)

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
    print(f"\nthen: make view3d MATCH={root}")


if __name__ == "__main__":
    asyncio.run(main())
