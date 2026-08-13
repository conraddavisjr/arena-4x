"""Deterministic randomness.

There are two kinds of randomness in this engine and they need different
treatment, for a reason that only shows up when you try to replay a match.

**Sequential streams** (`Stream`) advance internal state as they are drawn from.
That is fine for map generation, which runs exactly once at turn 0 and whose
generator is discarded immediately afterwards.

**Everything during play uses derived draws instead.** A sequential generator's
position would have to live inside `State` to survive a replay, and worse, it
would make every draw depend on how many draws happened before it: add one
combat roll anywhere and every subsequent outcome in the match shifts. Instead
each draw is a pure function of the match seed plus a semantic key, so a combat
between u17 and u23 on turn 47 resolves identically no matter what else the
turn contained, and no RNG state needs persisting at all.

That property is what makes the replay-determinism test in section 12 of the
plan meaningful rather than merely self-consistent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")

_U64 = (1 << 64) - 1
# 2**53 is where float64 stops representing consecutive integers exactly, so it
# is the right denominator for turning bits into a uniform [0, 1).
_MANTISSA = 1 << 53


def _digest(seed: int, parts: tuple[object, ...]) -> bytes:
    # The separator matters: without it, ("ab", "c") and ("a", "bc") would hash
    # the same, and two unrelated draws would silently correlate.
    material = "\x1f".join([str(seed), *(str(p) for p in parts)])
    return hashlib.sha256(material.encode()).digest()


def derive_u64(seed: int, *parts: object) -> int:
    """A 64-bit value derived from the seed and a semantic key.

    The key should name the decision, not its position in some sequence:
    `("combat", turn, attacker_id, defender_id)` rather than `("combat", n)`.
    """
    return int.from_bytes(_digest(seed, parts)[:8], "big") & _U64


def roll(seed: int, *parts: object) -> float:
    """A uniform float in [0, 1) for this exact decision."""
    return (derive_u64(seed, *parts) >> 11) / _MANTISSA


def randint(seed: int, lo: int, hi: int, *parts: object) -> int:
    """A uniform integer in [lo, hi]. Inclusive on both ends."""
    if hi < lo:
        raise ValueError(f"empty range: [{lo}, {hi}]")
    return lo + derive_u64(seed, *parts) % (hi - lo + 1)


def chance(seed: int, probability: float, *parts: object) -> bool:
    """True with the given probability, decided deterministically."""
    return roll(seed, *parts) < probability


def choice(seed: int, options: Sequence[T], *parts: object) -> T:
    """Pick one option. The caller is responsible for `options` being ordered.

    Passing a set here would be a determinism bug: set iteration order is not
    guaranteed across processes, so the same seed could pick differently.
    """
    if not options:
        raise ValueError("cannot choose from an empty sequence")
    return options[derive_u64(seed, *parts) % len(options)]


def shuffled(seed: int, options: Sequence[T], *parts: object) -> list[T]:
    """A deterministic permutation, via Fisher-Yates with derived swaps."""
    items = list(options)
    for i in range(len(items) - 1, 0, -1):
        j = derive_u64(seed, *parts, "shuffle", i) % (i + 1)
        items[i], items[j] = items[j], items[i]
    return items


class Stream:
    """A sequential generator, for map generation only.

    Named so that mapgen's draws cannot be perturbed by anything else: the
    terrain stream and the resource stream advance independently, so tuning
    resource density does not reshape the continents.
    """

    __slots__ = ("_gen", "name", "seed")

    def __init__(self, seed: int, name: str) -> None:
        self.seed = seed
        self.name = name
        # Fold the name into the seed so each named stream is independent.
        self._gen = np.random.Generator(np.random.PCG64(derive_u64(seed, "stream", name)))

    def random(self) -> float:
        return float(self._gen.random())

    def integers(self, lo: int, hi: int) -> int:
        """Inclusive on both ends, matching `randint` above."""
        return int(self._gen.integers(lo, hi + 1))

    def chance(self, probability: float) -> bool:
        return self.random() < probability

    def choice(self, options: Sequence[T]) -> T:
        if not options:
            raise ValueError("cannot choose from an empty sequence")
        return options[int(self._gen.integers(0, len(options)))]

    def shuffled(self, options: Sequence[T]) -> list[T]:
        items = list(options)
        self._gen.shuffle(items)  # type: ignore[arg-type]
        return items

    def noise(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._gen.random(shape)
