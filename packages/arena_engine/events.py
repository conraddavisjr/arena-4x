"""The event vocabulary.

Events are the source of truth for the whole system: the dashboard replays from
them, the determinism test folds them, and post-game analysis queries them. The
engine emits the gameplay half; the orchestrator adds the LLM half
(`prompt_sent`, `model_response`, `parse_failed`, and so on) around it.

Every event carries `text` alongside `payload`. The payload is for machines and
the text is for the event ticker, and having the engine write the sentence at
the moment it has full context beats reconstructing it in the frontend from
loose ids.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn: int
    type: str
    # The civ responsible, or None for engine-level events like turn_started.
    actor: str | None = None
    text: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


# Engine-emitted event types. The orchestrator's additions are declared next to
# the agent loop rather than here, so this list stays readable as "things that
# happen in the game" rather than "things that happen in the system".
TURN_STARTED = "turn_started"
TURN_ENDED = "turn_ended"
MATCH_ENDED = "match_ended"

ORDER_APPLIED = "order_applied"
ORDER_REJECTED = "order_rejected"

UNIT_MOVED = "unit_moved"
UNIT_FORTIFIED = "unit_fortified"
UNIT_KILLED = "unit_killed"
UNIT_CAPTURED = "unit_captured"
COMBAT_RESOLVED = "combat_resolved"

CITY_FOUNDED = "city_founded"
CITY_CAPTURED = "city_captured"
CITY_GREW = "city_grew"
CITY_SHRANK = "city_shrank"
BUILD_COMPLETED = "build_completed"
IMPROVEMENT_BUILT = "improvement_built"

TECH_COMPLETED = "tech_completed"
RESEARCH_SET = "research_set"

MESSAGE_SENT = "message_sent"
PROPOSAL_MADE = "proposal_made"
PROPOSAL_REJECTED = "proposal_rejected"
PROPOSAL_EXPIRED = "proposal_expired"
PROPOSAL_FAILED = "proposal_failed"
TREATY_SIGNED = "treaty_signed"
TREATY_BROKEN = "treaty_broken"
WAR_DECLARED = "war_declared"
PACT_EXPIRED = "pact_expired"

FIRST_CONTACT = "first_contact"
CONTACT_LOST = "contact_lost"

AGENT_ACTION = "agent_action"

PLAYER_ELIMINATED = "player_eliminated"
DOSSIER_TRUNCATED = "dossier_truncated"


def event(turn: int, kind: str, text: str, /, *, actor: str | None = None, **payload: Any) -> Event:
    """Terse constructor; the reducer emits a lot of these.

    `turn`, `kind` and `text` are positional-only and `actor` is keyword-only on
    purpose. With a plain signature, a caller passing a payload field that
    happens to be named `text` - which the message handler did - collides with
    the parameter and raises `TypeError: got multiple values for argument
    'text'`. That crashed the reducer on every single message send, and no bot
    match caught it because the heuristic bots never talk. The first agent to
    send a DM would have taken the whole match down with it.

    Positional-only parameters make that collision unrepresentable: any
    `text=...` keyword now lands in `payload` where it was meant to go.
    """
    return Event(turn=turn, type=kind, actor=actor, text=text, payload=payload)
