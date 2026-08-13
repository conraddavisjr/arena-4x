"""Hex grid invariants.

These are cheap to get subtly wrong and expensive to debug later, because a
broken `distance` shows up as an inexplicable combat or vision bug forty files
away rather than as a failing coordinate test.
"""

from __future__ import annotations

from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st

from arena_engine import hex as hx
from arena_engine.hex import Hex

coords = st.integers(min_value=-40, max_value=40)
hexes = st.builds(Hex, coords, coords)


@given(hexes)
def test_cube_coordinates_sum_to_zero(h: Hex) -> None:
    assert h.q + h.r + h.s == 0


@given(hexes)
def test_key_roundtrip(h: Hex) -> None:
    assert hx.from_key(h.to_key()) == h


def test_from_key_rejects_malformed() -> None:
    for bad in ("", "3", "abc", "3;4"):
        try:
            hx.from_key(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


@given(hexes)
def test_neighbors_are_distance_one(h: Hex) -> None:
    ns = hx.neighbors(h)
    assert len(set(ns)) == 6
    assert all(hx.distance(h, n) == 1 for n in ns)


@given(hexes, hexes)
def test_distance_is_symmetric(a: Hex, b: Hex) -> None:
    assert hx.distance(a, b) == hx.distance(b, a)


@given(hexes, hexes, hexes)
def test_distance_obeys_triangle_inequality(a: Hex, b: Hex, c: Hex) -> None:
    assert hx.distance(a, c) <= hx.distance(a, b) + hx.distance(b, c)


@given(hexes, st.integers(min_value=0, max_value=8))
def test_ring_members_are_exactly_radius_away(center: Hex, radius: int) -> None:
    r = hx.ring(center, radius)
    assert all(hx.distance(center, h) == radius for h in r)
    assert len(r) == (6 * radius if radius else 1)
    assert len(set(r)) == len(r)


@given(hexes, st.integers(min_value=0, max_value=8))
def test_spiral_matches_within_and_is_unique(center: Hex, radius: int) -> None:
    s = hx.spiral(center, radius)
    assert len(set(s)) == len(s), "spiral must not repeat a hex"
    assert set(s) == hx.within(center, radius)
    # Centre-outward ordering is relied on by mapgen and vision.
    assert [hx.distance(center, h) for h in s] == sorted(hx.distance(center, h) for h in s)


@given(hexes, hexes)
def test_line_endpoints_and_length(a: Hex, b: Hex) -> None:
    ln = hx.line(a, b)
    assert ln[0] == a
    assert ln[-1] == b
    assert len(ln) == hx.distance(a, b) + 1
    # Every step along a line advances exactly one hex. pairwise() is empty for
    # a single-element line (a == b), which is the correct degenerate case.
    assert all(hx.distance(x, y) == 1 for x, y in pairwise(ln))


@given(hexes)
def test_rotation_preserves_distance_from_origin(h: Hex) -> None:
    for times in range(6):
        assert hx.distance(hx.ORIGIN, hx.rotate_60(h, times)) == hx.distance(hx.ORIGIN, h)


@given(hexes)
def test_six_rotations_is_identity(h: Hex) -> None:
    assert hx.rotate_60(h, 6) == h
