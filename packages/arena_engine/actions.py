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

from typing import Annotated, Literal

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
