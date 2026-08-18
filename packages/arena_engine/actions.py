"""The action payload an agent returns each turn.

This is the Python mirror of `schemas/action.schema.json`. Both exist because
they serve different masters: the JSON Schema constrains the model at generation
time on four different vendors, and these Pydantic models validate and type the
result on the way in. `test_schema_parity.py` asserts they stay in step.

Two constraints inherited from the JSON Schema side are worth restating, since
they explain shapes that would otherwise look odd:

  - Anthropic's structured-output dialect supports neither `minimum`/`maximum`
    nor `minLength`/`maxLength`, so every bound is enforced here instead.
  - It requires `additionalProperties: false` everywhere, which is why each
    order is a closed model discriminated on `action` rather than a loose dict.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from arena_engine.types import Dossier, ProposalType, Terms

MAX_MESSAGE_CHARS = 1200
MAX_REASONING_CHARS = 4000


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Reasoning and memory
# ---------------------------------------------------------------------------


class Reasoning(Payload):
    """Captured before the action payload, per the chain-of-thought requirement.

    Free text, deliberately. The point of the lab is to read what the model
    actually thought, so imposing structure here would launder exactly the
    signal we are trying to observe.
    """

    situation_assessment: str = ""
    threats_and_opportunities: list[str] = Field(default_factory=list)
    plan_this_turn: str = ""
    confidence: float | None = None

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float | None) -> float | None:
        # Cannot be expressed as a schema bound on Anthropic, so clamp rather
        # than reject: a model that says 1.5 meant "very confident".
        return None if v is None else max(0.0, min(1.0, v))

    @field_validator("situation_assessment", "plan_this_turn")
    @classmethod
    def _truncate(cls, v: str) -> str:
        return v[:MAX_REASONING_CHARS]


# ---------------------------------------------------------------------------
# Diplomacy
# ---------------------------------------------------------------------------


class SendMessage(Payload):
    action: Literal["send_message"]
    channel: Literal["public", "private"]
    to: str | None = None
    text: str

    @field_validator("text")
    @classmethod
    def _truncate(cls, v: str) -> str:
        return v[:MAX_MESSAGE_CHARS]


class Propose(Payload):
    action: Literal["propose"]
    to: str
    type: ProposalType
    terms: Terms = Field(default_factory=Terms)
    message: str | None = None


class RespondToProposal(Payload):
    action: Literal["respond_to_proposal"]
    proposal_id: str
    response: Literal["accept", "reject"]
    message: str | None = None


class DeclareWar(Payload):
    action: Literal["declare_war"]
    on: str
    casus_belli: str | None = None


DiplomacyAction = Annotated[
    SendMessage | Propose | RespondToProposal | DeclareWar,
    Field(discriminator="action"),
]


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


class MoveUnit(Payload):
    action: Literal["move_unit"]
    unit_id: str
    to: str


class Attack(Payload):
    action: Literal["attack"]
    unit_id: str
    target: str


class Fortify(Payload):
    action: Literal["fortify"]
    unit_id: str


class FoundCity(Payload):
    action: Literal["found_city"]
    unit_id: str
    name: str

    @field_validator("name")
    @classmethod
    def _truncate(cls, v: str) -> str:
        return v[:40]


class BuildImprovement(Payload):
    action: Literal["build_improvement"]
    unit_id: str
    improvement: Literal["farm", "mine", "road"]


class SetProduction(Payload):
    action: Literal["set_production"]
    city_id: str
    item: str


class SetResearch(Payload):
    action: Literal["set_research"]
    tech: str


class SetRates(Payload):
    action: Literal["set_rates"]
    tax_pct: int
    science_pct: int


Order = Annotated[
    MoveUnit
    | Attack
    | Fortify
    | FoundCity
    | BuildImprovement
    | SetProduction
    | SetResearch
    | SetRates,
    Field(discriminator="action"),
]


class Action(Payload):
    """One agent's complete output for one turn."""

    reasoning: Reasoning = Field(default_factory=Reasoning)
    dossier: Dossier = Field(default_factory=Dossier)
    diplomacy: list[DiplomacyAction] = Field(default_factory=list)
    orders: list[Order] = Field(default_factory=list)


def pass_turn() -> Action:
    """The universal fallback.

    Every failure path in the orchestrator - timeout, refusal, exhausted
    retries, blown budget - lands here rather than raising. An agent that passes
    plays badly; an agent that raises ends a multi-day match.
    """
    return Action(
        reasoning=Reasoning(plan_this_turn="No action taken."),
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Which fields belong to each branch of the two discriminated unions, keyed by
# the `action` value that selects it. Derived from the models so a new order
# type cannot be forgotten here.
_BRANCH_FIELDS: dict[str, set[str]] = {
    branch.model_fields["action"].annotation.__args__[0]: set(branch.model_fields)
    for union in (
        (SendMessage, Propose, RespondToProposal, DeclareWar),
        (
            MoveUnit,
            Attack,
            Fortify,
            FoundCity,
            BuildImprovement,
            SetProduction,
            SetResearch,
            SetRates,
        ),
    )
    for branch in union
}


# What each branch cannot do without. A `send_message` with no text is not a
# message, and there is nothing to repair it into.
_BRANCH_REQUIRED: dict[str, set[str]] = {
    branch.model_fields["action"].annotation.__args__[0]: {
        name for name, f in branch.model_fields.items() if f.is_required() and name != "action"
    }
    for union in (
        (SendMessage, Propose, RespondToProposal, DeclareWar),
        (
            MoveUnit,
            Attack,
            Fortify,
            FoundCity,
            BuildImprovement,
            SetProduction,
            SetResearch,
            SetRates,
        ),
    )
    for branch in union
}


def _trim(item: object) -> object:
    """Keep only the fields that belong to the action this item declares.

    Needed because the schema Anthropic accepts is a *flattened* union - every
    branch's fields merged into one object - so a model choosing `fortify` is
    looking at a shape that also advertises `tech` and `tax_pct`, and will
    sometimes fill them in. Those fields are not wrong, they are irrelevant, and
    the strict models reject them with `extra inputs are not permitted`.

    Nulls go too. Some vendors emit the absent fields explicitly rather than
    omitting them, and `{"tech": null}` on a fortify order is the same
    irrelevance wearing a different hat.

    This drops noise, never meaning: a field that genuinely belongs to the
    declared action is always kept, so a real mistake - a missing `unit_id`,
    an invented action - still fails validation and still reaches the repair
    loop. See `arena_orchestrator.dialects` for why the schema is flattened.
    """
    if not isinstance(item, dict):
        return item
    allowed = _BRANCH_FIELDS.get(item.get("action"))
    if allowed is None:
        return item  # unknown action: let the union raise, with its own message
    return {k: v for k, v in item.items() if k in allowed and v is not None}


def parse(raw: str | bytes) -> Action:
    """Validate a model's response body into an `Action`.

    Use this rather than `Action.model_validate_json` anywhere a real provider
    is on the other end.
    """
    return parse_reporting_drops(raw)[0]


def parse_reporting_drops(raw: str | bytes) -> tuple[Action, list[dict[str, Any]]]:
    """`parse`, and the entries it had to throw away.

    Dropping is the right call - a `found_city` with no unit cannot be repaired
    into anything, and failing the whole payload over it would discard the
    orders that were fine. But dropping *silently* is how a civilisation died:
    `claude-haiku-4-5` sent unusable orders on 91% of its turns, every one
    discarded here, and from the outside that is indistinguishable from a model
    choosing to do nothing. It founded no cities and was eliminated on turn 35
    of a 128-turn match with nothing in any log marked as a failure.

    So the discards come back too, and `Agent._attempt` turns them into a
    correction the model can act on. The engine has always explained a rejected
    *order* to the civ that issued it; this closes the same loop one layer up,
    for an order that never reached the engine at all.
    """
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
    dropped: list[dict[str, Any]] = []
    for key in ("orders", "diplomacy"):
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        kept = []
        for item in value:
            trimmed = _trim(item)
            if _usable(trimmed):
                kept.append(trimmed)
            elif isinstance(trimmed, dict):
                dropped.append({"kind": key, **trimmed})
        payload[key] = kept
    return Action.model_validate(payload), dropped


def why_unusable(entry: dict[str, Any]) -> str:
    """A sentence a model can act on, naming the fields it left out."""
    action = entry.get("action")
    required = _BRANCH_REQUIRED.get(action)
    if required is None:
        return f"{action!r} is not one of the actions this game accepts"
    missing = sorted(required - set(entry))
    return f"{action} needs {', '.join(missing)}, and you sent none of them"


def _usable(item: object) -> bool:
    """Drop an entry that is missing something its action cannot do without.

    The same trade the engine already makes with illegal orders: discard the bad
    one, keep the rest of the turn. A flattened schema only requires `action`,
    so a model can legally answer `{"action": "send_message"}` - which is not a
    message, and which there is nothing to repair it into. Failing the whole
    payload over it would throw away the orders alongside, and those are the
    part that moves the game.

    An invented action goes the same way. Models reach for plausible names that
    are not in the enum - `research` for `set_research`, `set_taxes` for
    `set_rates` - and not every vendor's structured output enforces an enum
    strictly enough to stop them. Keeping one so the union could raise a nicer
    message cost the entire turn, including the orders that were fine.
    """
    if not isinstance(item, dict):
        return True
    required = _BRANCH_REQUIRED.get(item.get("action"))
    return required is not None and required <= set(item)
