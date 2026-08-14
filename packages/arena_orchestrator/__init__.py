"""The agent loop: four models, one board, and everything that can go wrong.

Layers, from the outside in:

    providers/   one narrow seam over four unrelated vendor APIs
    resilience   rate limiting, retry, circuit breaking
    pricing      the rate card
    budget       the out-of-fiction dollar halt and the in-fiction allowance

Nothing here imports the engine's rules, and the engine imports nothing from
here. The engine is a pure reducer with no I/O; this package is all I/O and no
rules. Keeping that line sharp is what lets a match be replayed from its event
log without a single network call.
"""

from __future__ import annotations

from .budget import Allowance, Ledger
from .pricing import RATES, UnknownModel, cost_of, rate_for
from .providers import PROVIDERS, LLMClient, ProviderError, Turn, Usage, build
from .resilience import CircuitBreaker, RetryPolicy, TokenBucket, with_retry

__all__ = [
    "PROVIDERS",
    "RATES",
    "Allowance",
    "CircuitBreaker",
    "LLMClient",
    "Ledger",
    "ProviderError",
    "RetryPolicy",
    "TokenBucket",
    "Turn",
    "UnknownModel",
    "Usage",
    "build",
    "cost_of",
    "rate_for",
    "with_retry",
]
