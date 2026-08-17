"""A provider that costs nothing and does what it is told.

Two jobs, both of which need doing before a single dollar is spent:

- **Dry runs.** The whole orchestrator - turn loop, validation, repair,
  budget, event writing, resume-after-crash - can play a full match with this
  in every seat. It uses the same heuristic bots the engine tests use, so the
  match is a real match and the bundle is a real bundle.
- **Fault injection.** `failures` makes it raise on cue, which is the only
  practical way to assert that a 429 storm, a refusal, or a provider outage
  leaves the match running and writes the right events. Waiting for a real
  vendor to have a bad day is not a test strategy.

It reports plausible token counts and prices against a real rate card entry, so
the budget accountant is exercised too rather than being bypassed by a zero.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

from .base import ProviderError, Turn, Usage

# Priced like Haiku, so a dry run produces a believable bill rather than $0.00
# and the budget halt is genuinely exercised.
SCRIPTED_MODEL = "claude-haiku-4-5"


class ScriptedClient:
    """Returns canned responses, optionally failing first."""

    name = "scripted"

    def __init__(
        self,
        responses: Iterable[dict[str, Any] | str] | Callable[[str], dict[str, Any] | str],
        *,
        model: str = SCRIPTED_MODEL,
        failures: Iterable[ProviderError] = (),
        latency_ms: int = 0,
        thinking: str | None = "scripted deliberation",
    ):
        self.model = model
        self._callable = responses if callable(responses) else None
        self._queue = deque(responses) if not callable(responses) else deque()
        self._failures = deque(failures)
        self._latency_ms = latency_ms
        self._thinking = thinking
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str, schema: dict[str, Any]) -> Turn:
        self.calls.append((system, user))
        # Failures are consumed before responses, so a script of two errors then
        # a body models "it failed twice and then worked" - the case the retry
        # ladder exists for.
        if self._failures:
            raise self._failures.popleft()

        if self._callable is not None:
            payload = self._callable(user)
        elif self._queue:
            payload = self._queue.popleft()
        else:
            raise AssertionError("ScriptedClient ran out of responses")

        text = payload if isinstance(payload, str) else json.dumps(payload)
        return Turn(
            text=text,
            # Roughly the shape of a real turn: a cached system prefix, a fresh
            # observation, a small structured body.
            usage=Usage(
                input_tokens=len(user) // 4,
                output_tokens=len(text) // 4,
                cache_read_tokens=len(system) // 4,
            ),
            model=self.model,
            latency_ms=self._latency_ms,
            stop_reason="end_turn",
            # A stand-in trace, so the persistence path is exercised offline.
            # Without it the only test of "is thinking stored" would be a live
            # one, which is how the storing came to be missing in the first
            # place: three adapters parsed a trace and nothing carried it.
            thinking=self._thinking,
        )

    async def aclose(self) -> None:
        return None
