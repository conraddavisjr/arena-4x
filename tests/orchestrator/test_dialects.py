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
    # Nullable, but in the compact encoding - an `anyOf` wrapper costs 19 bytes
    # a field over a type array, and this schema has spent a long time over its
    # budget.
    assert orders["properties"]["unit_id"]["type"] == ["string", "null"]
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


# ---------------------------------------------------------------------------
# Thinking costs grammar headroom
# ---------------------------------------------------------------------------


def test_every_anthropic_model_keeps_strict_order_enforcement() -> None:
    """The guarantee that must not be traded away again.

    It was, once. To fit extended thinking inside the grammar budget, pre-4.6
    models were given a variant where only `action` was required and the field
    requirements were demoted to prose. `claude-haiku-4-5` then answered
    `{"action": "found_city"}` with no unit and no name - 91% of its orders
    malformed against 0% on every other seat, because it was the only seat whose
    schema had stopped enforcing them. It could not found cities, lost the one it
    had, and was eliminated on turn 35 of a 128-turn match.

    Structured output works because the decoder enforces the shape. A
    description saying `found_city requires name, unit_id` is a suggestion, and
    this model did not take it.
    """
    for model in ("claude-haiku-4-5", "claude-opus-5", "claude-sonnet-5", None):
        orders = for_provider(SCHEMA, "anthropic", model)["properties"]["orders"]["items"]
        assert "unit_id" in orders["required"], f"{model} lost order enforcement"
        assert len(orders["required"]) > 1, f"{model} requires only {orders['required']}"


def test_thinking_is_the_thing_given_up_instead() -> None:
    """A seat that reasons but cannot act is worth nothing; a seat that acts
    without a visible trace is worth a great deal. Only pre-4.6 pays this."""
    from arena_orchestrator.dialects import cannot_think_with_schema

    assert cannot_think_with_schema("claude-haiku-4-5")
    assert not cannot_think_with_schema("claude-opus-5")
    assert not cannot_think_with_schema("claude-sonnet-5")


def test_titles_are_stripped_because_a_grammar_never_reads_them() -> None:
    """830 bytes of "City Id" and "Tax Pct" restating keys the model can see.

    Twice the budget that was missing when this whole problem started, sitting
    unclaimed in the schema the entire time. It does not solve the ceiling -
    that turned out to be total grammar complexity rather than bytes - but it is
    free headroom for every Anthropic seat.
    """
    dialect = for_provider(SCHEMA, "anthropic", "claude-opus-5")
    assert '"title"' not in json.dumps(dialect)
    assert len(json.dumps(dialect)) < len(json.dumps(SCHEMA))


def test_nullable_fields_are_encoded_compactly_and_correctly() -> None:
    """`{"type": ["string","null"]}` over an `anyOf` wrapper - 19 bytes a field.

    The enum case has to come first, and getting it backwards produces
    `{"enum": ["farm","mine","road"], "type": ["string","null"]}`, which
    announces that null is allowed and forbids it two keys later.
    """
    from arena_orchestrator.dialects import nullable

    assert nullable({"type": "string"}) == {"type": ["string", "null"]}
    assert nullable({"enum": ["farm", "mine"], "type": "string"}) == {
        "enum": ["farm", "mine", None]
    }
    # Anything it cannot compact keeps the wrapper rather than being mangled.
    assert nullable({"$ref": "#/$defs/X"}) == {"anyOf": [{"$ref": "#/$defs/X"}, {"type": "null"}]}

    orders = for_provider(SCHEMA, "anthropic", "claude-opus-5")["properties"]["orders"]["items"]
    for name, spec in orders["properties"].items():
        if name == "action":
            continue
        accepts_null = "null" in (spec.get("type") or []) or None in (spec.get("enum") or [])
        assert accepts_null or "anyOf" in spec, f"{name} is required but cannot be null"


def test_other_vendors_are_unaffected_by_the_model() -> None:
    for model in ("gpt-5.4-mini", "grok-4.3", "gemini-3.6-flash"):
        for provider in ("openai", "xai", "google"):
            assert for_provider(SCHEMA, provider, model) == SCHEMA
