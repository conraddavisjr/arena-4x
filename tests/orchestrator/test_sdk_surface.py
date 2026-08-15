"""Do the calls these adapters make still exist in the installed SDKs?

The gap this fills is specific. Mocked tests prove an adapter reads a response
correctly, but they read a response *we wrote*, so they agree with whatever we
believed when we wrote them. Live contract tests prove the vendor agrees, but
they need a key, they cost money, and nobody runs them on every commit.

This sits between: it asserts our calls against the real installed SDK, for
free, offline, on every run. It is what caught the Google adapter being wrong in
four places - written from a documentation summary, mocked against that same
misunderstanding, and green the whole time.

The failure it guards against is nastier than it looks. `google-genai` takes
`**body`, so a misspelled parameter is not a TypeError - it is silently
dropped. Sending `instructions=` instead of `system_instruction=` would have
posted the entire rules reference to nowhere and had four agents play a
multi-day match with no rules at all, at full price, with nothing in any log to
say why they played so badly.

Skips per-SDK when that SDK is absent, so an engine-only checkout stays green.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest


def sdk(name: str) -> Any:
    return pytest.importorskip(name, reason=f"{name} not installed (engine-only checkout)")


def path_exists(root: Any, dotted: str) -> bool:
    current = root
    for part in dotted.split("."):
        current = getattr(current, part, None)
        if current is None:
            return False
    return True


def accepts(fn: Any, names: list[str]) -> list[str]:
    """Which of `names` the function does not accept.

    A `**kwargs` signature accepts anything, so it is reported as accepting
    everything - which is exactly the case that needs a different check.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - C-implemented callable
        return []
    if any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
        return []
    return [n for n in names if n not in signature.parameters]


# ---------------------------------------------------------------------------


def test_anthropic_surface() -> None:
    anthropic = sdk("anthropic")
    client = anthropic.AsyncAnthropic(api_key="x")
    assert path_exists(client, "messages.create")
    assert path_exists(client, "messages.stream")
    assert not accepts(
        client.messages.create,
        ["model", "max_tokens", "system", "messages", "thinking", "output_config"],
    )
    # Every error class the translation matches on.
    for name in (
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "APIStatusError",
    ):
        assert hasattr(anthropic, name), f"anthropic.{name} is gone"


def test_openai_uses_the_responses_surface() -> None:
    openai = sdk("openai")
    client = openai.AsyncOpenAI(api_key="x")
    assert path_exists(client, "responses.create")
    assert not accepts(
        client.responses.create,
        ["model", "instructions", "input", "max_output_tokens", "text", "reasoning"],
    )
    for name in ("RateLimitError", "APITimeoutError", "APIConnectionError", "APIStatusError"):
        assert hasattr(openai, name), f"openai.{name} is gone"


def test_xai_uses_the_compatible_chat_surface() -> None:
    """xAI rides the OpenAI SDK but on the older surface. Pointing the Responses
    API at api.x.ai would 404 in the middle of a run."""
    openai = sdk("openai")
    client = openai.AsyncOpenAI(api_key="x", base_url="https://api.x.ai/v1")
    assert path_exists(client, "chat.completions.create")
    assert not accepts(
        client.chat.completions.create, ["model", "max_tokens", "messages", "response_format"]
    )


def test_google_surface() -> None:
    genai = sdk("google.genai")
    client = genai.Client(api_key="x")
    assert path_exists(client, "aio.interactions.create")


def test_google_request_field_names() -> None:
    """`interactions.create` takes `**body`, so a wrong name is dropped rather
    than rejected. These are checked against the request model instead."""
    genai = sdk("google.genai")
    from google.genai import interactions

    fields = interactions.Interaction.model_fields
    for name in (
        "system_instruction",
        "response_format",
        "response_mime_type",
        "generation_config",
    ):
        assert name in fields, f"google request field {name!r} is gone"
    # Nested under a modality, with a camelCase schema key unlike its neighbours.
    text_format = genai.types.TextResponseFormat.model_fields
    assert "jsonSchema" in text_format
    assert "mime_type" in text_format
    assert "max_output_tokens" in genai.types.GenerationConfig.model_fields


def test_google_response_field_names() -> None:
    """The counters are `total_*` on this surface. The `candidates_token_count`
    name belongs to the old generate_content response, and reading it here
    returned zero for every field - pricing a whole match at nothing and leaving
    the budget halt disarmed."""
    sdk("google.genai")
    from google.genai import interactions
    from google.genai._gaos.types.interactions.usage import Usage

    fields = interactions.Interaction.model_fields
    for name in ("output_text", "status", "usage", "errors"):
        assert name in fields, f"google response field {name!r} is gone"
    # And no finish_reason or candidates, which is why status is what we read.
    assert "finish_reason" not in fields

    for name in (
        "total_input_tokens",
        "total_output_tokens",
        "total_cached_tokens",
        "total_thought_tokens",
    ):
        assert name in Usage.model_fields, f"google usage counter {name!r} is gone"
