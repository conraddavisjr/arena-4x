"""Play a match and write a replay bundle.

    python scripts/export_match.py --seed 4 --out output/match-4

Uses the scripted heuristic bots, so it needs no API keys. That is deliberate:
the viewer and the whole published-match path get built and debugged against
real bundles before a cent is spent on a model.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from arena_replay import BundleWriter

from arena_engine import bots, victory
from arena_engine.reducer import new_match, step
from arena_engine.types import MatchConfig, State

ROSTER = [
    ("p1", "Aurelian Compact"),
    ("p2", "Iron Concord"),
    ("p3", "Verdant Pact"),
    ("p4", "Solari Dominion"),
]


def match_stats(state: State, tally: Counter[str], per_civ: dict[str, Counter]) -> dict:
    """The match dossier.

    Every figure here is derived from the event log rather than tracked as the
    match runs, which is what keeps it honest: a published bundle and a fresh
    replay of the same seed produce the same card.
    """
    scores = victory.scores(state)
    civs = state.civ_ids()

    def top(metric: str) -> str | None:
        ranked = sorted(civs, key=lambda p: (-per_civ[p][metric], p))
        return ranked[0] if ranked and per_civ[ranked[0]][metric] else None

    return {
        "winner": state.victory.winner if state.victory else None,
        "condition": state.victory.condition if state.victory else None,
        "turns": state.turn,
        "scores": scores,
        "totals": {
            "cities_founded": tally["city_founded"],
            "cities_captured": tally["city_captured"],
            "cities_sacked": tally["city_sacked"] + tally["city_razed"],
            "combats": tally["combat_resolved"],
            "wars_declared": tally["war_declared"],
            "treaties_signed": tally["treaty_signed"],
            "treaties_broken": tally["treaty_broken"],
            "first_contacts": tally["first_contact"],
            "messages": tally["message_sent"],
        },
        "superlatives": {
            # Treaty-breaking is the hard evidence of deception; comparing a
            # civ's private messages against its own private reasoning is the
            # softer signal, and belongs in the viewer where both are visible.
            "most_treacherous": top("treaty_broken"),
            "most_warlike": top("war_declared"),
            "most_diplomatic": top("treaty_signed"),
            "most_talkative": top("message_sent"),
            "most_expansionist": top("city_founded"),
            "most_seafaring": top("put_to_sea"),
            "biggest_spender": top("unit_disbanded"),
        },
        "per_civ": {p: dict(per_civ[p]) for p in civs},
    }


def play(seed: int, out: Path, turn_limit: int) -> tuple[State, Path, int]:
    config = MatchConfig(turn_limit=turn_limit)
    state, _ = new_match(f"match-{seed}", seed, ROSTER, config)
    writer = BundleWriter.start(out, state)

    tally: Counter[str] = Counter()
    per_civ: dict[str, Counter] = {p: Counter() for p in state.civ_ids()}

    while state.victory is None and state.turn < turn_limit:
        state, events = step(state, bots.all_bot_actions(state))
        writer.add(state, events)
        for e in events:
            tally[e.type] += 1
            if e.actor in per_civ:
                per_civ[e.actor][e.type] += 1
                if e.type == "unit_moved" and e.payload.get("transition") == "embark":
                    per_civ[e.actor]["put_to_sea"] += 1

    root = writer.finish(state, match_stats(state, tally, per_civ))
    total = sum(f.stat().st_size for f in root.rglob("*.json"))
    return state, root, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--turn-limit", type=int, default=300)
    args = ap.parse_args()

    out = args.out or Path("output") / f"match-{args.seed}"
    if out.exists():
        shutil.rmtree(out)

    state, root, total = play(args.seed, out, args.turn_limit)
    frames = sorted((root / "turns").glob("*.json"))
    biggest = max(frames, key=lambda f: f.stat().st_size)

    print(f"wrote {root}")
    print(f"  turns        {len(frames)}")
    print(
        f"  victory      {state.victory.condition if state.victory else 'none'} "
        f"-> {state.victory.winner if state.victory else '-'}"
    )
    print(f"  total size   {total / 1024:,.0f} KB")
    print(f"  match.json   {(root / 'match.json').stat().st_size / 1024:,.0f} KB")
    print(f"  mean frame   {total / max(len(frames), 1) / 1024:,.1f} KB")
    print(f"  largest      {biggest.name} at {biggest.stat().st_size / 1024:,.1f} KB")
    print(
        f"  first paint  "
        f"{((root / 'match.json').stat().st_size + biggest.stat().st_size) / 1024:,.0f} KB "
        f"(metadata + one turn)"
    )


if __name__ == "__main__":
    main()
