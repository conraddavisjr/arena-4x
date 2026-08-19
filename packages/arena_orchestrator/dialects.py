"""One action schema, adjusted per vendor where a vendor forces it.

The design said one schema would serve all four, authored to Anthropic's rules
as the strictest. Measured against live endpoints, that is not quite true, and
the exception is Anthropic itself.

Anthropic compiles the schema into a grammar and enforces two separate limits:

  - a **grammar size** cap. The action schema as authored exceeds it - two
    arrays of discriminated unions, eight branches and four, is too much.
  - a **16 union-typed parameter** cap. Flattening those unions gets under the
    size cap, but the obvious flattening - merge every branch's fields and make
    each nullable - produces twenty, and trips this one instead.

The variant that satisfies both flattens the unions *and* declares only
`action` required, so the fields that do not apply are simply absent rather
than present-and-null. That is accepted by Anthropic and Google. It is rejected
by OpenAI, whose strict mode requires `required` to name every property - which
is precisely why this is a per-provider step rather than a change to the one
schema.

OpenAI, xAI and Google all take the base schema unchanged, so only Anthropic
gets transformed. The cost is paid on the parsing side instead: a flattened
schema invites the model to set fields that do not belong to the order it
chose, so `arena_engine.actions.parse` drops them before the strict models see
the body. See its docstring for why that is a repair and not a leniency.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

# Anthropic is the only vendor that needs the schema reshaped. Recorded as a set
# rather than an `if` so the next vendor to need one is an entry, not a branch.
NEEDS_FLATTENING = frozenset({"anthropic"})

# `orders` is flattened *strictly*: every field present, the inapplicable ones
# null. `diplomacy` is flattened loosely, requiring only `action`.
#
# That asymmetry is forced and then chosen. Forced, because the two limits leave
# room for exactly one strict union: flattening both strictly is 24 union-typed
# parameters against a cap of 16, and flattening only one leaves the grammar too
# large. Chosen, because of which one to spend it on. A loosely-flattened union
# lets a model answer `{"action": "found_city"}` with no unit and no name -
# measured, repeatedly, with a full observation in front of it - and that costs
# the turn. The same failure in diplomacy costs one unsent message, which
# `actions.parse` drops while the orders go through.
STRICT = ("orders",)
LOOSE = ("diplomacy",)

# Extended thinking shrinks the compiled-grammar budget far enough that the
# strict `orders` variant no longer fits on pre-4.6 models. That is a real
# constraint and the question is only what to give up for it.
#
# **We gave up the wrong thing once, and it cost a match.** Loosening `orders`
# so only `action` is required made the schema fit, and `claude-haiku-4-5` then
# answered `{"action": "found_city"}` with no unit and no name - 91% of its
# orders malformed against 0% on every other seat, because it was the only seat
# whose schema had stopped enforcing them. It could not found cities, lost the
# one it had, and was eliminated on turn 35 of a 128-turn match. The prose
# instruction was present and correct and the model ignored it.
#
# So the trade runs the other way now: pre-4.6 Anthropic keeps the strict schema
# and does without extended thinking. A seat that reasons but cannot act is
# worth nothing; a seat that acts without a visible trace is worth a great deal.
# `providers.anthropic_provider` reads this to decide.
#
# Measured while looking for a third option, and worth recording because it
# rules out the obvious ones. The binding constraint is total grammar
# complexity, not any single dimension:
#
#     strict, 4136B, 10 union-typed params  -> rejected
#     strict, 3950B,  7 union-typed params  -> rejected
#     strict, 3802B,  4 union-typed params  -> accepted
#     one array per action, 5199B, 0 unions -> rejected
#     loose orders, 4924B                   -> accepted
#
# Shrinking bytes does not help - `{"type": ["string","null"]}` is still a union
# - and removing unions does not help either if the structure grows instead.
NO_THINKING_BELOW = ("claude-haiku-", "claude-sonnet-4-", "claude-opus-4-5")


def cannot_think_with_schema(model: str | None) -> bool:
    return bool(model) and model.startswith(NO_THINKING_BELOW)


def for_provider(schema: dict[str, Any], provider: str, model: str | None = None) -> dict[str, Any]:
    """The action schema as this vendor, running this model, will accept it."""
    if provider not in NEEDS_FLATTENING:
        return schema
    # Descriptions come off *before* flattening, not after: the merge writes a
    # description of its own spelling out which fields each action needs, and
    # that one is load-bearing. Stripping last would have removed it.
    lean = strip_titles(strip_descriptions(schema))
    return prune(flatten_unions(lean, strict=STRICT, loose=LOOSE))


def nullable(spec: dict[str, Any]) -> dict[str, Any]:
    """Make one field accept null, in the fewest bytes that express it.

    Every byte here is compiled into a grammar against a hard ceiling, and the
    obvious encoding is the expensive one:

        {"anyOf": [{"type": "string"}, {"type": "null"}]}   47 bytes
        {"type": ["string", "null"]}                        28 bytes

    Nineteen bytes a field does not sound like an argument until you are 423
    bytes over the limit with eleven fields to spend them on. An enum collapses
    further still - `null` joins the member list rather than wrapping the whole
    node.

    Falls back to the wrapper for anything it cannot compact - a `$ref`, an
    existing union - because being correct matters more than being small, and
    the caller has no way to check.
    """
    # Enums first, and the order is load-bearing. Pydantic emits an enum *with*
    # a `type`, so checking `type` first produces
    # `{"enum": ["farm","mine","road"], "type": ["string","null"]}` - which
    # announces that null is allowed and then forbids it two keys later. The
    # value has to join the member list instead, and `type` goes, because once
    # null is a member the values are no longer all strings.
    if isinstance(spec.get("enum"), list) and "$ref" not in spec:
        out = {k: v for k, v in spec.items() if k != "type"}
        return {**out, "enum": [*spec["enum"], None]}
    if isinstance(spec.get("type"), str):
        return {**spec, "type": [spec["type"], "null"]}
    return {"anyOf": [spec, {"type": "null"}]}


def strip_titles(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop `title`, which Pydantic emits everywhere and a grammar never reads.

    830 bytes of it in the action schema - "City Id", "Unit Id", "Tax Pct" -
    restating field names the model can already see as keys. That is twice the
    budget that was missing when `claude-haiku-4-5` could not have both extended
    thinking and a schema strict enough to enforce its own order fields, and the
    workaround for that cost a live match: the loosened schema let the model
    answer `{"action": "found_city"}` with no unit and no name, 91% of its
    orders were dropped as unusable, and the civ was eliminated on turn 35.

    Which makes this the cheapest 830 bytes in the project. It was always
    removable and nobody looked, because the schema was already small enough
    for every model that mattered at the time.
    """
    if isinstance(schema, list):
        return [strip_titles(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    return {k: strip_titles(v) for k, v in schema.items() if k != "title"}


def strip_descriptions(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop field descriptions, which Anthropic alone cannot afford.

    They are worth real money everywhere else - two of four models in the first
    live match never wrote an opponent model, and the field carried no
    description, so its name was the only guidance. But every byte here is
    compiled into a grammar against a cap of about 6KB, and the descriptions
    added 1,400 of them, which pushed the dialect over.

    The same guidance goes to every vendor through the rules reference instead,
    where it sits in the cached prefix at a tenth the price and counts against
    nothing. Descriptions stay in the schema for the three vendors that can take
    them, because a field explains itself best where the field is.
    """
    if isinstance(schema, list):
        return [strip_descriptions(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    return {k: strip_descriptions(v) for k, v in schema.items() if k != "description"}


def flatten_unions(
    schema: dict[str, Any],
    *,
    strict: tuple[str, ...] = STRICT,
    loose: tuple[str, ...] = LOOSE,
) -> dict[str, Any]:
    """Collapse every array-of-union into an array of one merged object.

    The branches are mutually exclusive and discriminated by `action`, so the
    merged object keeps that as an enum of every branch's const and carries the
    union of the branches' other fields. Only `action` stays required: a field
    that does not apply to the chosen action is omitted, not nulled, which is
    what keeps the union-parameter count at zero instead of twenty.
    """
    out = copy.deepcopy(schema)
    defs = out.get("$defs", {})

    def resolve(node: dict[str, Any]) -> dict[str, Any]:
        return defs[node["$ref"].split("/")[-1]] if "$ref" in node else node

    def merge(union: dict[str, Any], strict: bool) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        actions: list[str] = []
        needs: list[str] = []
        for branch in (resolve(b) for b in union["anyOf"]):
            const = branch.get("properties", {}).get("action", {}).get("const")
            if const is None:
                continue
            actions.append(const)
            # Flattening throws away each branch's own `required` list - the
            # merged object can only require what *every* branch requires, which
            # is `action` alone. Left at that, a model answers `{"action":
            # "found_city"}` with no unit and the turn is wasted. The constraint
            # cannot be expressed structurally in a dialect this restricted, so
            # it is carried as prose instead, where models follow it well.
            required = [f for f in branch.get("required", []) if f != "action"]
            needs.append(f"{const} requires {', '.join(required)}" if required else f"{const}")
            for name, spec in branch.get("properties", {}).items():
                if name != "action":
                    # Strict: every field present, nulled where it does not
                    # apply, so the model cannot satisfy the schema by omitting
                    # the ones that matter. `actions.parse` strips the nulls.
                    properties.setdefault(name, nullable(spec) if strict else spec)
        properties["action"] = {"enum": sorted(set(actions))}
        lead = (
            "Set `action`, then the fields that action needs; null the rest."
            if strict
            else "Set `action`, then only the fields that action needs."
        )
        return {
            "type": "object",
            "additionalProperties": False,
            "description": f"{lead} " + "; ".join(sorted(needs)) + ".",
            "properties": properties,
            "required": sorted(properties) if strict else ["action"],
        }

    for key in strict:
        node = out.get("properties", {}).get(key)
        if node and isinstance(node.get("items"), dict) and "anyOf" in node["items"]:
            node["items"] = merge(node["items"], True)
    for key in loose:
        node = out.get("properties", {}).get(key)
        if node and isinstance(node.get("items"), dict) and "anyOf" in node["items"]:
            node["items"] = merge(node["items"], False)
    return out


def prune(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop `$defs` no longer reachable from the root.

    Not tidiness - the grammar compiler walks every definition whether or not
    anything references it, so an orphan left behind after flattening still
    counts against the size limit. Bisecting this was how the limit was found:
    removing a *property* changed nothing until its definitions went too.
    """
    out = copy.deepcopy(schema)
    defs = out.get("$defs", {})
    if not defs:
        return out

    reachable: set[str] = set()
    frontier = set(_refs(json.dumps({k: v for k, v in out.items() if k != "$defs"})))
    while frontier:
        name = frontier.pop()
        if name in reachable or name not in defs:
            continue
        reachable.add(name)
        frontier |= set(_refs(json.dumps(defs[name])))

    out["$defs"] = {k: v for k, v in defs.items() if k in reachable}
    return out


def _refs(blob: str) -> list[str]:
    return re.findall(r'"#/\$defs/([^"]+)"', blob)
