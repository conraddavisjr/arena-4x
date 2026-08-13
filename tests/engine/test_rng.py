"""Deterministic randomness.

The property that actually matters for replay is *order independence*: a draw
must depend only on its semantic key, never on how many draws preceded it. A
sequential generator would pass a naive "same seed, same result" test while
still failing replay the moment an extra roll is inserted anywhere, so these
tests check the stronger claim.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from arena_engine import rng

seeds = st.integers(min_value=0, max_value=2**31 - 1)


@given(seeds)
def test_same_key_same_value(seed: int) -> None:
    a = rng.roll(seed, "combat", 47, "u17", "u23")
    b = rng.roll(seed, "combat", 47, "u17", "u23")
    assert a == b


@given(seeds)
def test_draws_are_order_independent(seed: int) -> None:
    """The whole point. Inserting an unrelated draw must not shift this one."""
    expected = rng.roll(seed, "combat", 47, "u17", "u23")
    for i in range(50):
        rng.roll(seed, "unrelated", i)
    assert rng.roll(seed, "combat", 47, "u17", "u23") == expected


@given(seeds)
def test_different_keys_differ(seed: int) -> None:
    a = rng.roll(seed, "combat", 47, "u17", "u23")
    b = rng.roll(seed, "combat", 48, "u17", "u23")
    c = rng.roll(seed, "combat", 47, "u23", "u17")
    assert len({a, b, c}) == 3


@given(seeds)
def test_key_parts_are_unambiguous(seed: int) -> None:
    """("ab","c") and ("a","bc") must not collide, or unrelated draws correlate."""
    assert rng.roll(seed, "ab", "c") != rng.roll(seed, "a", "bc")


@given(seeds)
def test_different_seeds_differ(seed: int) -> None:
    assert rng.roll(seed, "x") != rng.roll(seed + 1, "x")


@given(seeds, st.integers(min_value=-50, max_value=50), st.integers(min_value=0, max_value=50))
def test_randint_is_in_range_and_inclusive(seed: int, lo: int, span: int) -> None:
    hi = lo + span
    for i in range(20):
        assert lo <= rng.randint(seed, lo, hi, "t", i) <= hi


def test_randint_rejects_empty_range() -> None:
    try:
        rng.randint(1, 5, 4)
    except ValueError:
        return
    raise AssertionError("expected ValueError on an empty range")


@given(seeds)
def test_randint_covers_its_range(seed: int) -> None:
    seen = {rng.randint(seed, 1, 6, "die", i) for i in range(300)}
    assert seen == {1, 2, 3, 4, 5, 6}, f"uneven coverage: {sorted(seen)}"


@given(seeds)
def test_roll_is_a_unit_interval(seed: int) -> None:
    for i in range(200):
        v = rng.roll(seed, "u", i)
        assert 0.0 <= v < 1.0


@given(seeds)
def test_roll_is_roughly_uniform(seed: int) -> None:
    values = [rng.roll(seed, "u", i) for i in range(2000)]
    assert 0.45 < sum(values) / len(values) < 0.55
    # Every decile should be populated; a biased hash fold would leave gaps.
    assert len({int(v * 10) for v in values}) == 10


@given(seeds)
def test_choice_and_shuffle_are_deterministic_permutations(seed: int) -> None:
    options = list("abcdefgh")
    assert rng.choice(seed, options, "k") == rng.choice(seed, options, "k")
    a = rng.shuffled(seed, options, "k")
    assert a == rng.shuffled(seed, options, "k")
    assert sorted(a) == sorted(options), "shuffle must be a permutation"


def test_choice_rejects_empty() -> None:
    """choice() has nothing to return; shuffled() legitimately returns []."""
    try:
        rng.choice(1, [], "k")
    except ValueError:
        pass
    else:
        raise AssertionError("choice should reject an empty sequence")

    # Callers shuffle candidate lists that are sometimes empty. Making that an
    # error would just push a guard into every call site.
    assert rng.shuffled(1, [], "k") == []


@given(seeds)
def test_streams_are_independent_by_name(seed: int) -> None:
    """Tuning one subsystem must not reshape another's output."""
    terrain_first = [rng.Stream(seed, "terrain").random() for _ in range(3)]
    # Draw heavily from a different stream, then re-create the terrain stream.
    noisy = rng.Stream(seed, "resources")
    for _ in range(100):
        noisy.random()
    assert [rng.Stream(seed, "terrain").random() for _ in range(3)] == terrain_first


@given(seeds)
def test_stream_is_reproducible(seed: int) -> None:
    a = rng.Stream(seed, "mapgen")
    b = rng.Stream(seed, "mapgen")
    assert [a.integers(0, 100) for _ in range(20)] == [b.integers(0, 100) for _ in range(20)]


@given(seeds)
def test_stream_integers_are_inclusive(seed: int) -> None:
    s = rng.Stream(seed, "t")
    seen = {s.integers(0, 3) for _ in range(200)}
    assert seen == {0, 1, 2, 3}
