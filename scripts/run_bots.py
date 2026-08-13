"""Play a full match with four scripted heuristic bots. No API keys needed.

This is the engine's exit criterion: a match that runs to a decisive victory
without deadlocking, starving, or stalling.

    python scripts/run_bots.py --seed 42
    python scripts/run_bots.py --sweep 20     # 20 seeds, summary table only
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from arena_engine import bots, victory
from arena_engine.reducer import new_match, step
from arena_engine.types import MatchConfig, State

ROSTER = [
    ("p1", "Aurelian Compact"),
    ("p2", "Iron Concord"),
    ("p3", "Verdant Pact"),
    ("p4", "Solari Dominion"),
]


def play(seed: int, config: MatchConfig | None = None, trace: bool = False) -> tuple[State, list]:
    state, log = new_match(f"bots-{seed}", seed, ROSTER, config)
    while state.victory is None:
        state, events = step(state, bots.all_bot_actions(state))
        log.extend(events)
        if trace:
            _trace(state, events)
    return state, log


def _trace(state: State, events: list) -> None:
    interesting = [
        e
        for e in events
        if e.type
        not in {"turn_started", "turn_ended", "unit_moved", "order_rejected", "unit_fortified"}
    ]
    if state.turn % 25 == 0 or interesting:
        cities = {p: len(state.cities_of(p)) for p in state.living_player_ids()}
        print(f"--- turn {state.turn:3d}  cities {cities}")
        for e in interesting[:6]:
            print(f"      {e.type:18s} {e.text}")


def summarize(state: State, log: list, seed: int, elapsed: float) -> dict:
    counts = Counter(e.type for e in log)
    scores = victory.scores(state)
    return {
        "seed": seed,
        "turns": state.turn,
        "condition": state.victory.condition if state.victory else "none",
        "winner": state.victory.winner if state.victory else None,
        "cities": len(state.cities),
        "combats": counts.get("combat_resolved", 0),
        "captures": counts.get("city_captured", 0),
        "wars": counts.get("war_declared", 0),
        "contacts": counts.get("first_contact", 0),
        "eliminated": counts.get("player_eliminated", 0),
        "rejected": counts.get("order_rejected", 0),
        "events": len(log),
        "scores": scores,
        "seconds": round(elapsed, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep", type=int, default=0, help="play N seeds and summarise")
    ap.add_argument("--turn-limit", type=int, default=300)
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    config = MatchConfig(turn_limit=args.turn_limit)
    seeds = range(args.sweep) if args.sweep else [args.seed]
    rows = []

    for seed in seeds:
        began = time.time()
        state, log = play(seed, config, trace=args.trace and not args.sweep)
        rows.append(summarize(state, log, seed, time.time() - began))

    header = (
        f"{'seed':>5} {'turns':>6} {'condition':<12} {'win':<4} "
        f"{'cities':>7} {'fights':>7} {'taken':>6} {'wars':>5} {'elim':>5} {'rej':>6} {'sec':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['seed']:>5} {r['turns']:>6} {r['condition']:<12} {str(r['winner'] or '-'):<4} "
            f"{r['cities']:>7} {r['combats']:>7} {r['captures']:>6} {r['wars']:>5} "
            f"{r['eliminated']:>5} {r['rejected']:>6} {r['seconds']:>6}"
        )

    if len(rows) > 1:
        by_condition = Counter(r["condition"] for r in rows)
        decisive = sum(n for c, n in by_condition.items() if c != "turn_limit")
        print()
        print(f"conditions: {dict(by_condition)}")
        print(f"decisive (not turn-limit): {decisive}/{len(rows)}")
        print(f"mean turns: {sum(r['turns'] for r in rows) / len(rows):.0f}")
        print(f"mean rejected orders: {sum(r['rejected'] for r in rows) / len(rows):.0f}")
    else:
        r = rows[0]
        print()
        print("final scores:", r["scores"])


if __name__ == "__main__":
    main()
