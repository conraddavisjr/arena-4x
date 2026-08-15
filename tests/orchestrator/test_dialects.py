"""Per-provider schema dialects.

The design said one schema would serve all four vendors, authored to
Anthropic's rules as the strictest. Measured against live endpoints that turned
out to be false, and Anthropic was the exception - it compiles the schema into a
grammar and enforces two limits nothing in the documentation prepares you for.

These tests pin the shape that was found by bisecting against the live API, so
the next person to touch the action models finds out here rather than four
hours into an unattended run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from arena_engine.actions import parse
from arena_orchestrator.dialects import for_provider, prune

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[2] / "schemas" / "action.schema.json").read_text()
)

# Measured against the live API by bisection. Anthropic rejects the schema above
# roughly 6KB of compiled grammar, and separately above 16 union-typed
# parameters. Both are undocumented and neither degrades gracefully.
MAX_BYTES = 6_100
MAX_UNIONS = 17


def count_unions(node: object) -> int:
    if isinstance(node, list):
        return sum(count_unions(item) for item in node)
    if not isinstance(node, dict):
        return 0
    here = sum(
        1
        for spec in (node.get("properties") or {}).values()
        if isinstance(spec, dict) and ("anyOf" in spec or isinstance(spec.get("type"), list))
    )
    return here + sum(count_unions(value) for value in node.values())


@pytest.mark.parametrize("provider", ["openai", "xai", "google", "scripted"])
def test_everyone_but_anthropic_gets_the_base_schema(provider: str) -> None:
    """Only one vendor needs reshaping, and it should stay that way - four
    dialects is four things to keep in step with the models."""
    assert for_provider(SCHEMA, provider) is SCHEMA


def test_the_anthropic_dialect_fits_both_of_its_limits() -> None:
    """The two caps that cost a flagship run if either is exceeded. Neither
    fails loudly: one is a 400 on turn one, the other is a 400 on turn one with
    a completely different message."""
    dialect = for_provider(SCHEMA, "anthropic")
    assert len(json.dumps(dialect)) < MAX_BYTES
    assert count_unions(dialect) <= MAX_UNIONS


def test_orders_are_flattened_strictly() -> None:
    """Every field required, inapplicable ones nulled.

    This is the half worth spending the union budget on. Under the loose
    treatment a model answers `{"action": "found_city"}` with no unit and no
    name - measured repeatedly, with a full observation in front of it - and
    the turn is wasted.
    """
    orders = for_provider(SCHEMA, "anthropic")["properties"]["orders"]["items"]
    assert orders["required"] == sorted(orders["properties"])
    assert "unit_id" in orders["required"]
    assert {"type": "null"} in orders["properties"]["unit_id"]["anyOf"]
    # The per-branch requirements cannot survive the merge structurally, so they
    # are carried as prose.
    assert "found_city requires name, unit_id" in orders["description"]


def test_diplomacy_is_flattened_loosely() -> None:
    """The half that gives way, because there is only room for one strict union
    and a malformed message costs a message rather than a turn."""
    diplomacy = for_provider(SCHEMA, "anthropic")["properties"]["diplomacy"]["items"]
    assert diplomacy["required"] == ["action"]


def test_every_action_survives_flattening() -> None:
    """A merged union that quietly dropped a branch would remove a move from the
    game without removing it from the rules the agent is given."""
    dialect = for_provider(SCHEMA, "anthropic")
    orders = set(dialect["properties"]["orders"]["items"]["properties"]["action"]["enum"])
    diplomacy = set(dialect["properties"]["diplomacy"]["items"]["properties"]["action"]["enum"])
    assert orders == {
        "move_unit",
        "attack",
        "fortify",
        "found_city",
        "build_improvement",
        "set_production",
        "set_research",
        "set_rates",
    }
    assert diplomacy == {"send_message", "propose", "respond_to_proposal", "declare_war"}


def test_pruning_drops_definitions_nothing_references() -> None:
    """Not tidiness. The grammar compiler walks every definition whether or not
    anything points at it, so an orphan left behind by flattening still counts
    against the size limit - which is why removing a *property* changed nothing
    until its definitions went too."""
    before = for_provider(SCHEMA, "anthropic")
    assert "MoveUnit" not in before.get("$defs", {})
    padded = dict(before)
    padded["$defs"] = {**before.get("$defs", {}), "Orphan": {"type": "object"}}
    assert "Orphan" not in prune(padded)["$defs"]


def test_the_base_schema_is_never_mutated() -> None:
    """It is loaded once and handed to every seat; a dialect that edited it in
    place would silently reshape the schema the other three vendors get."""
    original = json.dumps(SCHEMA, sort_keys=True)
    for_provider(SCHEMA, "anthropic")
    assert json.dumps(SCHEMA, sort_keys=True) == original


# ---------------------------------------------------------------------------
# The parsing side of the same bargain
# ---------------------------------------------------------------------------


def test_nulls_from_the_flattened_shape_are_stripped() -> None:
    """A strictly-flattened order carries every field, most of them null. The
    strict models reject those with `extra inputs are not permitted`, so they
    come off before validation."""
    action = parse(
        json.dumps(
            {
                "orders": [
                    {
                        "action": "found_city",
                        "unit_id": "u1",
                        "name": "Aurelia",
                        "to": None,
                        "target": None,
                        "tech": None,
                        "tax_pct": None,
                    }
                ]
            }
        )
    )
    assert action.orders[0].name == "Aurelia"


def test_fields_belonging_to_another_action_are_dropped() -> None:
    """A flattened schema advertises every branch's fields at once, so a model
    choosing `fortify` is looking at a shape that also offers `tech`."""
    action = parse(
        json.dumps(
            {"orders": [{"action": "fortify", "unit_id": "u1", "tech": "sailing", "tax_pct": 60}]}
        )
    )
    assert action.orders[0].action == "fortify"
    assert not hasattr(action.orders[0], "tech")


def test_an_entry_missing_what_it_needs_is_dropped_not_fatal() -> None:
    """The same trade the engine makes with illegal orders: discard the bad one,
    keep the rest of the turn. Failing the payload over an empty message would
    throw away the orders alongside it, and those are what move the game."""
    action = parse(
        json.dumps(
            {
                "diplomacy": [{"action": "send_message"}],
                "orders": [{"action": "fortify", "unit_id": "u1"}],
            }
        )
    )
    assert action.diplomacy == []
    assert len(action.orders) == 1


def test_an_invented_action_is_dropped_with_the_rest_of_the_turn_intact() -> None:
    """Measured on two vendors: models reach for plausible names that are not in
    the enum - `research` for `set_research`, `set_taxes` for `set_rates` - and
    not everyone enforces the enum strictly enough to stop it. Keeping one so
    the union could raise a nicer message cost the whole turn."""
    action = parse(
        json.dumps(
            {
                "orders": [
                    {"action": "research", "tech": "sailing"},
                    {"action": "fortify", "unit_id": "u1"},
                ]
            }
        )
    )
    assert [o.action for o in action.orders] == ["fortify"]


def test_a_body_that_is_not_json_still_raises() -> None:
    """Trimming drops noise, never meaning. A body that cannot be read at all is
    a real failure and has to reach the repair loop."""
    with pytest.raises(json.JSONDecodeError):
        parse("not json at all")
    with pytest.raises(ValidationError):
        parse('{"reasoning": "should be an object"}')
