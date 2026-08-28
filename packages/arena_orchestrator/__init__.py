"""The agent loop: four models, one board, and everything that can go wrong.

Layers, outermost first:

    loop         the match: build observations, call four models, resolve a turn
    agent        one seat, its guard rails, and its memory between turns
    journal      the append-only log a crashed match is rebuilt from
    budget       the out-of-fiction dollar halt and the in-fiction allowance
    pricing      the rate card behind both
    profiles     how many tokens a seat spends, so a run can be costed first
    resilience   rate limiting, retry, circuit breaking
    providers/   one narrow seam over four unrelated vendor APIs

The line worth keeping sharp runs through the middle rather than around the
outside. **`providers/` knows nothing about the game and nothing about the
engine**; it deals only in strings, schemas and token counts, which is what
lets an adapter be tested without a board and the engine be tested without an
SDK. The loop above it necessarily knows both.

The engine imports nothing from here in either direction. It is a pure reducer
with no I/O; this package is all I/O and no rules. That separation is what lets
a finished match be replayed - and a crashed one resumed - without a single
network call.
"""

from __future__ import annotations

from .agent import Agent, Outcome
from .budget import Allowance, Ledger
from .config import RunConfig, Seat
from .journal import Journal, Recovered, recover
from .loop import MatchResult, Orchestrator
from .pricing import RATES, UnknownModel, cost_of, rate_for
from .profiles import PROFILES, Profile, Projection, profile_for, project
from .providers import PROVIDERS, LLMClient, ProviderError, Turn, Usage, build
from .resilience import CircuitBreaker, RetryPolicy, TokenBucket, with_retry

__all__ = [
    "PROFILES",
    "PROVIDERS",
    "RATES",
    "Agent",
    "Allowance",
    "CircuitBreaker",
    "Journal",
    "LLMClient",
    "Ledger",
    "MatchResult",
    "Orchestrator",
    "Outcome",
    "Profile",
    "Projection",
    "ProviderError",
    "Recovered",
    "RetryPolicy",
    "RunConfig",
    "Seat",
    "TokenBucket",
    "Turn",
    "UnknownModel",
    "Usage",
    "build",
    "cost_of",
    "profile_for",
    "project",
    "rate_for",
    "recover",
    "with_retry",
]
