#!/usr/bin/env python
"""Re-measure the token profiles in `arena_orchestrator.profiles` from a journal.

    python scripts/derive_profiles.py output/baseline-300/journal.jsonl
    python scripts/derive_profiles.py output/*/journal.jsonl --compare

The profiles are hand-maintained on purpose, the same way the rate card is: a
table somebody has to edit is a table somebody has read. This script does the
arithmetic and prints an entry ready to paste, so the manual step is judgement
rather than long division.

**Read the R^2 before pasting anything.** The prompt is modelled as a straight
line, which every seat that played a whole match satisfies at 0.92 or better.
`claude-haiku-4-5` in `baseline-300` fits at 0.53, because it was eliminated on
turn 35 with one city and its observation never grew - a number that would have
under-projected that seat by 40% had anyone taken it at face value. A low R^2
here does not mean the fit is imprecise. It means the seat did not play a long
enough or normal enough match to have a growth rate, and the number should come
from a match where it did.

`--compare` runs every journal given and prints the seats side by side, which is
how `output_high` gets chosen: it is the worst per-call mean any completed match
has shown, not an average of the runs.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from arena_orchestrator.profiles import observation_fingerprint  # noqa: E402


@dataclass
class Fit:
    model: str
    calls: int
    first_turn: int
    last_turn: int
    prompt_base: float
    prompt_growth: float
    prompt_r2: float
    output_mean: float
    output_median: float
    output_r2: float
    cache_read_frac: float
    cache_write_frac: float


def least_squares(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """(intercept, slope, r_squared). Straight line, no library."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance == 0:
        return mean_y, 0.0, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / variance
    intercept = mean_y - slope * mean_x
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    total = sum((y - mean_y) ** 2 for y in ys)
    return intercept, slope, (1 - residual / total) if total else 0.0


def fits(path: Path) -> list[Fit]:
    calls = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    calls = [c for c in calls if c.get("type") == "agent_call"]
    if not calls:
        raise SystemExit(f"{path}: no agent_call rows - is this a journal?")

    out = []
    for model in sorted({c["model"] for c in calls}):
        rows = [c for c in calls if c["model"] == model]
        turns = [float(c["turn"]) for c in rows]
        # Total prompt, not `input_tokens`. The split between fresh input and
        # cache read is a property of where the cache breakpoint sits, which
        # moves between runs; the sum is a property of the board, which does not.
        prompt = [
            float(c["input_tokens"] + c["cache_read_tokens"] + c["cache_write_tokens"])
            for c in rows
        ]
        outputs = [float(c["output_tokens"]) for c in rows]
        base, growth, r2 = least_squares(turns, prompt)
        _, _, out_r2 = least_squares(turns, outputs)
        total_prompt = sum(prompt) or 1.0
        out.append(
            Fit(
                model=model,
                calls=len(rows),
                first_turn=int(min(turns)),
                last_turn=int(max(turns)),
                prompt_base=base,
                prompt_growth=growth,
                prompt_r2=r2,
                output_mean=statistics.mean(outputs),
                output_median=statistics.median(outputs),
                output_r2=out_r2,
                cache_read_frac=sum(c["cache_read_tokens"] for c in rows) / total_prompt,
                cache_write_frac=sum(c["cache_write_tokens"] for c in rows) / total_prompt,
            )
        )
    return out


def entry(fit: Fit, source: str) -> str:
    """A `Profile(...)` literal, ready to paste into `profiles.py`."""
    return f"""    "{fit.model}": Profile(
        prompt_base={fit.prompt_base:.0f},
        prompt_growth={fit.prompt_growth:.1f},
        output={fit.output_mean:.0f},
        output_high={fit.output_mean:.0f},  # raise if another match shows worse
        cache_read_frac={fit.cache_read_frac:.3f},
        cache_write_frac={fit.cache_write_frac:.3f},
        source="{source}",
        checked="YYYY-MM-DD",
        calls={fit.calls},
        fit_r2={fit.prompt_r2:.3f},
    ),"""


def report(path: Path) -> list[Fit]:
    found = fits(path)
    print(f"\n{path}")
    for fit in found:
        flag = "  <-- too low to trust, see the docstring" if fit.prompt_r2 < 0.85 else ""
        print(f"\n  {fit.model}   {fit.calls} calls, turns {fit.first_turn}-{fit.last_turn}")
        print(
            f"    prompt  {fit.prompt_base:8.0f} + {fit.prompt_growth:6.1f}/turn"
            f"   R^2 {fit.prompt_r2:.3f}{flag}"
        )
        print(
            f"    output  mean {fit.output_mean:7.0f}  median {fit.output_median:7.0f}"
            f"   R^2 {fit.output_r2:.3f} (flat is expected)"
        )
        print(
            f"    cache   read {fit.cache_read_frac:.3f} of prompt,"
            f" write {fit.cache_write_frac:.3f}"
        )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journals", nargs="+", type=Path)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="print a per-seat comparison across journals, for choosing output_high",
    )
    parser.add_argument(
        "--paste",
        action="store_true",
        help="print Profile(...) literals for the first journal given",
    )
    args = parser.parse_args()

    everything: dict[str, list[tuple[Path, Fit]]] = collections.defaultdict(list)
    first: list[Fit] = []
    for path in args.journals:
        if not path.exists():
            print(f"  skipped, not found: {path}")
            continue
        found = report(path)
        first = first or found
        for fit in found:
            everything[fit.model].append((path, fit))

    if args.compare:
        print("\n\nacross journals - output_high is the worst per-call mean, not the average")
        for model, seen in sorted(everything.items()):
            print(f"\n  {model}")
            for path, fit in seen:
                print(
                    f"    {str(path.parent.name):16} output mean {fit.output_mean:7.0f}"
                    f"   growth {fit.prompt_growth:6.1f}/turn  R^2 {fit.prompt_r2:.3f}"
                )
            worst = max(fit.output_mean for _, fit in seen)
            print(f"    -> output_high = {worst:.0f}")

    if args.paste and first:
        print("\n\npaste into arena_orchestrator.profiles.PROFILES, then fill in `checked`:")
        for fit in first:
            print(entry(fit, source=args.journals[0].parent.name))

    print(f"\nobservation schema fingerprint now: {observation_fingerprint()}")
    print("Update MEASURED_FINGERPRINT in profiles.py when you update the profiles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
