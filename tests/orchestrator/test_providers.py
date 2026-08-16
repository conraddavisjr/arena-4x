"""The four adapters, driven by stand-in SDK objects.

These are not contract tests. A contract test needs a live key and asserts that
the vendor still accepts our request shape; those live in `test_contracts.py`
and skip without credentials. What is tested here is the half we own: that a
vendor's response is read into a `Turn` correctly, and - more importantly - that
the failure modes are classified correctly, because a refusal misread as a crash
or a 500 misread as fatal changes what the match does next.

The response objects are hand-built duck types rather than SDK fixtures. That
is deliberate: it documents exactly which fields each adapter depends on, so
when a vendor renames one the diff shows what broke.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from arena_orchestrator.providers import build
from arena_orchestrator.providers.base import (
    FatalProviderError,
    Malformed,
    Overloaded,
    RateLimited,
    Refused,
)
from arena_orchestrator.providers.scripted import ScriptedClient

SCHEMA = {"type": "object", "additionalProperties": False, "properties": {}}


class Bag:
    """A stand-in for an SDK response object."""

    def __init__(self, **fields: Any):
        self.__dict__.update(fields)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_an_unknown_provider_raises_rather_than_defaulting() -> None:
    """Silently seating the wrong vendor in a four-way comparison would
    invalidate the match without anything looking wrong."""
    with pytest.raises(ValueError, match="unknown provider"):
        build("antropic")


def test_building_one_provider_does_not_import_the_others() -> None:
    """A match played entirely on Anthropic must not fail to start because
    google-genai is absent."""
    before = {name for name in sys.modules if name.split(".")[0] in {"openai", "google"}}
    build("scripted", responses=[{}])
    after = {name for name in sys.modules if name.split(".")[0] in {"openai", "google"}}
    assert after == before


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def fake_anthropic_sdk() -> types.ModuleType:
    """The SDK surface the adapter actually touches."""
    module = types.ModuleType("anthropic")

    class APIStatusError(Exception):
        def __init__(self, message: str, status_code: int = 500):
            super().__init__(message)
            self.status_code = status_code

    class RateLimitError(APIStatusError):
        def __init__(self, message: str, retry_after: str | None = None):
            super().__init__(message, 429)
            self.response = Bag(headers={"retry-after": retry_after} if retry_after else {})

    module.APIStatusError = APIStatusError
    module.RateLimitError = RateLimitError
    module.APITimeoutError = type("APITimeoutError", (Exception,), {})
    module.APIConnectionError = type("APIConnectionError", (Exception,), {})
    module.AsyncAnthropic = lambda **kwargs: Bag(messages=None, close=None)
    return module


@pytest.fixture()
def anthropic_client(monkeypatch: pytest.MonkeyPatch):
    sdk = fake_anthropic_sdk()
    monkeypatch.setitem(sys.modules, "anthropic", sdk)
    from arena_orchestrator.providers.anthropic_provider import AnthropicClient

    client = AnthropicClient(api_key="x")

    def respond(message: Any = None, error: Exception | None = None):
        async def create(**request):
            client.last_request = request
            if error is not None:
                raise error
            return message

        # The adapter streams above STREAM_ABOVE_MAX_TOKENS, and the default
        # max_tokens is now above it - sized so adaptive thinking and the action
        # payload both fit, since Anthropic caps them together. So the stub has
        # to offer both surfaces or it tests a path production never takes.
        class Stream:
            def __init__(self, **request):
                self._request = request

            async def __aenter__(self):
                client.last_request = self._request
                if error is not None:
                    raise error
                return self

            async def __aexit__(self, *exc):
                return False

            async def get_final_message(self):
                return message

        client._client = Bag(
            messages=Bag(create=create, stream=lambda **kw: Stream(**kw)), close=_noop
        )
        return client

    return respond


async def _noop(*args: Any, **kwargs: Any) -> None:
    return None


async def test_anthropic_reads_a_normal_response(anthropic_client) -> None:
    client = anthropic_client(
        Bag(
            stop_reason="end_turn",
            content=[
                Bag(type="thinking", thinking="weighing the north"),
                Bag(type="text", text='{"orders": []}'),
            ],
            usage=Bag(
                input_tokens=6000,
                output_tokens=1500,
                cache_read_input_tokens=5800,
                cache_creation_input_tokens=0,
            ),
        )
    )
    turn = await client.complete("system", "user", SCHEMA)
    assert turn.text == '{"orders": []}'
    assert turn.thinking == "weighing the north"
    assert turn.usage.cache_read_tokens == 5800
    assert turn.usage.output_tokens == 1500


async def test_anthropic_sends_the_shapes_this_vendor_requires(anthropic_client) -> None:
    """Four things that each return a 400 or silently degrade if wrong."""
    client = anthropic_client(
        Bag(stop_reason="end_turn", content=[Bag(type="text", text="{}")], usage=Bag())
    )
    await client.complete("system", "user", SCHEMA)
    request = client.last_request

    # Adaptive thinking: budget_tokens is removed on Opus 5 and 400s if sent.
    assert request["thinking"]["type"] == "adaptive"
    assert "budget_tokens" not in request["thinking"]
    # Thinking text is omitted unless display is asked for, and reading the
    # model's reasoning is the entire experiment.
    assert request["thinking"]["display"] == "summarized"
    # output_config.format, not the deprecated top-level output_format.
    assert request["output_config"]["format"]["schema"] is SCHEMA
    assert "output_format" not in request
    # Sampling parameters are rejected outright on Opus 5.
    assert not {"temperature", "top_p", "top_k"} & request.keys()
    # The breakpoint sits on the *system block*, not at the top level. Measured
    # side by side: top-level wrote 6,846 tokens on every single call and read
    # none, because it caches the last cacheable block and that is the user turn
    # carrying the observation. On the system block it wrote once and then read
    # 5,036 on every call after - 12.5x the input price of the prefix, and the
    # only symptom is a larger bill.
    assert "cache_control" not in request
    assert request["system"] == [
        {"type": "text", "text": "system", "cache_control": {"type": "ephemeral"}}
    ]


async def test_anthropic_checks_the_stop_reason_before_reading_content(
    anthropic_client,
) -> None:
    """A refusal is an HTTP 200 whose content array can be empty. Reading
    content[0] first turns it into an IndexError, and then into a bug report
    about a bug that is not there."""
    client = anthropic_client(
        Bag(stop_reason="refusal", content=[], stop_details=Bag(reason="policy"), usage=Bag())
    )
    with pytest.raises(Refused):
        await client.complete("system", "user", SCHEMA)


async def test_anthropic_truncation_is_malformed_not_success(anthropic_client) -> None:
    client = anthropic_client(
        Bag(stop_reason="max_tokens", content=[Bag(type="text", text='{"orders": [')], usage=Bag())
    )
    with pytest.raises(Malformed):
        await client.complete("system", "user", SCHEMA)


@pytest.mark.parametrize(
    ("status", "expected"),
    [(429, RateLimited), (500, Overloaded), (529, Overloaded), (400, FatalProviderError)],
)
async def test_anthropic_classifies_status_codes(anthropic_client, status, expected) -> None:
    sdk = sys.modules["anthropic"]
    error = (
        sdk.RateLimitError("slow down", retry_after="7")
        if status == 429
        else sdk.APIStatusError("boom", status)
    )
    client = anthropic_client(error=error)
    with pytest.raises(expected) as caught:
        await client.complete("system", "user", SCHEMA)
    if status == 429:
        assert caught.value.retry_after == 7.0


# ---------------------------------------------------------------------------
# OpenAI and xAI
# ---------------------------------------------------------------------------


def fake_openai_sdk() -> types.ModuleType:
    module = types.ModuleType("openai")

    class APIStatusError(Exception):
        def __init__(self, message: str, status_code: int = 500):
            super().__init__(message)
            self.status_code = status_code

    module.APIStatusError = APIStatusError
    module.RateLimitError = type("RateLimitError", (APIStatusError,), {})
    module.APITimeoutError = type("APITimeoutError", (Exception,), {})
    module.APIConnectionError = type("APIConnectionError", (Exception,), {})
    module.AsyncOpenAI = lambda **kwargs: Bag(responses=None, chat=None, close=_noop)
    return module


async def test_openai_uses_the_responses_api_not_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plan for this project was written against
    `response_format` on chat.completions. That is no longer the call."""
    monkeypatch.setitem(sys.modules, "openai", fake_openai_sdk())
    from arena_orchestrator.providers.openai_provider import OpenAIClient

    client = OpenAIClient(api_key="x")
    seen: dict[str, Any] = {}

    async def create(**request):
        seen.update(request)
        return Bag(
            output_text='{"orders": []}',
            status="completed",
            output=[],
            usage=Bag(
                input_tokens=6000,
                output_tokens=1500,
                input_tokens_details=Bag(cached_tokens=5800),
                output_tokens_details=Bag(reasoning_tokens=900),
            ),
        )

    client._client = Bag(responses=Bag(create=create), close=_noop)
    turn = await client.complete("system", "user", SCHEMA)

    assert seen["text"]["format"]["type"] == "json_schema"
    assert seen["text"]["format"]["strict"] is True
    assert seen["instructions"] == "system"
    assert turn.usage.cache_read_tokens == 5800
    assert turn.usage.reasoning_tokens == 900
    # Reasoning is inside the output count; it must not be added on top.
    assert turn.usage.output_tokens == 1500


async def test_openai_surfaces_a_refusal_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """On this surface a refusal is a content block, not a status."""
    monkeypatch.setitem(sys.modules, "openai", fake_openai_sdk())
    from arena_orchestrator.providers.openai_provider import OpenAIClient

    client = OpenAIClient(api_key="x")

    async def create(**request):
        return Bag(
            output_text="",
            status="completed",
            output=[Bag(content=[Bag(type="refusal", refusal="cannot help")])],
            usage=Bag(),
        )

    client._client = Bag(responses=Bag(create=create), close=_noop)
    with pytest.raises(Refused, match="cannot help"):
        await client.complete("system", "user", SCHEMA)


async def test_xai_uses_the_compatible_chat_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    """xAI is OpenAI-*compatible*, which means the older chat-completions shape.
    Pointing the Responses API at its base URL would 404 mid-match."""
    monkeypatch.setitem(sys.modules, "openai", fake_openai_sdk())
    from arena_orchestrator.providers.openai_provider import XAI_BASE_URL, XAIClient

    client = XAIClient(api_key="x")
    seen: dict[str, Any] = {}

    async def create(**request):
        seen.update(request)
        return Bag(
            choices=[
                Bag(
                    finish_reason="stop",
                    message=Bag(content='{"orders": []}', refusal=None),
                )
            ],
            usage=Bag(
                prompt_tokens=6000,
                completion_tokens=1500,
                prompt_tokens_details=Bag(cached_tokens=1000),
            ),
        )

    client._client = Bag(chat=Bag(completions=Bag(create=create)), close=_noop)
    turn = await client.complete("system", "user", SCHEMA)

    assert XAI_BASE_URL == "https://api.x.ai/v1"
    assert seen["response_format"]["json_schema"]["strict"] is True
    assert seen["messages"][0]["role"] == "system"
    # prompt_tokens is inclusive of the cached portion on this surface, so the
    # cached part is split out rather than charged at the full rate twice.
    assert turn.usage.input_tokens == 5000
    assert turn.usage.cache_read_tokens == 1000


async def test_xai_truncation_is_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", fake_openai_sdk())
    from arena_orchestrator.providers.openai_provider import XAIClient

    client = XAIClient(api_key="x")

    async def create(**request):
        return Bag(
            choices=[Bag(finish_reason="length", message=Bag(content='{"or', refusal=None))],
            usage=Bag(),
        )

    client._client = Bag(chat=Bag(completions=Bag(create=create)), close=_noop)
    with pytest.raises(Malformed):
        await client.complete("system", "user", SCHEMA)


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


def _fake_genai() -> types.ModuleType:
    genai = types.ModuleType("google.genai")
    genai.Client = lambda **kwargs: Bag(aio=None)
    package = types.ModuleType("google")
    package.genai = genai
    return package


async def test_google_reads_the_interactions_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "google", _fake_genai())
    monkeypatch.setitem(sys.modules, "google.genai", sys.modules["google"].genai)
    from arena_orchestrator.providers.google_provider import GoogleClient

    client = GoogleClient(api_key="x")
    seen: dict[str, Any] = {}

    async def create(**request):
        seen.update(request)
        return Bag(
            output_text='{"orders": []}',
            status="completed",
            usage=Bag(
                total_input_tokens=6000,
                total_output_tokens=1500,
                total_cached_tokens=4000,
                total_thought_tokens=700,
            ),
        )

    client._client = Bag(aio=Bag(interactions=Bag(create=create)))
    turn = await client.complete("system", "user", SCHEMA)

    # `system_instruction`, not `instructions`. This SDK takes **body, so a
    # misspelled parameter is not an error - it is an omission, and the agent
    # would have played the whole match with no rules reference at all.
    assert seen["system_instruction"] == "system"
    assert "instructions" not in seen
    # A list of per-modality formats, keyed by *wire alias*. `jsonSchema` is the
    # Python field name and is silently ignored on the wire - the model then
    # answers in prose and every turn fails to parse.
    assert seen["response_format"] == [{"mime_type": "application/json", "schema": SCHEMA}]
    # Not a top-level parameter on this surface.
    assert seen["generation_config"]["max_output_tokens"] > 0
    assert "max_output_tokens" not in seen

    assert turn.usage.input_tokens == 2000
    assert turn.usage.output_tokens == 1500
    assert turn.usage.cache_read_tokens == 4000
    assert turn.usage.reasoning_tokens == 700


async def test_google_usage_uses_the_interactions_counter_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`candidates_token_count` belongs to the older generate_content response.
    Reading it here returned zero for every field, which prices a match at
    nothing and leaves the budget halt disarmed for a whole run."""
    monkeypatch.setitem(sys.modules, "google", _fake_genai())
    monkeypatch.setitem(sys.modules, "google.genai", sys.modules["google"].genai)
    from arena_orchestrator.providers.google_provider import GoogleClient

    client = GoogleClient(api_key="x")

    async def create(**request):
        return Bag(
            output_text="{}",
            status="completed",
            usage=Bag(prompt_token_count=6000, candidates_token_count=1500),
        )

    client._client = Bag(aio=Bag(interactions=Bag(create=create)))
    turn = await client.complete("system", "user", SCHEMA)
    assert turn.usage.output_tokens == 0  # the old names carry nothing


async def test_google_failure_status_is_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """There is no finish_reason and no candidates array on this surface -
    a refusal is a status plus an errors list."""
    monkeypatch.setitem(sys.modules, "google", _fake_genai())
    monkeypatch.setitem(sys.modules, "google.genai", sys.modules["google"].genai)
    from arena_orchestrator.providers.google_provider import GoogleClient

    client = GoogleClient(api_key="x")

    async def create(**request):
        return Bag(output_text="", status="failed", errors=[Bag(message="blocked by safety")])

    client._client = Bag(aio=Bag(interactions=Bag(create=create)))
    with pytest.raises(Refused, match="blocked by safety"):
        await client.complete("system", "user", SCHEMA)


# ---------------------------------------------------------------------------
# The scripted provider
# ---------------------------------------------------------------------------


async def test_the_scripted_provider_returns_what_it_was_given() -> None:
    client = ScriptedClient([{"orders": [{"action": "fortify", "unit_id": "u1"}]}])
    turn = await client.complete("system", "user", SCHEMA)
    assert '"fortify"' in turn.text
    assert client.calls == [("system", "user")]


async def test_the_scripted_provider_prices_against_a_real_rate_card() -> None:
    """So a dry run produces a believable bill and the budget halt is genuinely
    exercised rather than bypassed by a zero."""
    from arena_orchestrator.pricing import cost_of

    client = ScriptedClient([{"orders": []}])
    turn = await client.complete("s" * 4000, "u" * 8000, SCHEMA)
    assert cost_of(turn.model, turn.usage) > 0


async def test_the_scripted_provider_fails_on_cue_then_succeeds() -> None:
    """Waiting for a real vendor to have a bad day is not a test strategy."""
    client = ScriptedClient(
        [{"orders": []}],
        failures=[RateLimited("429"), Overloaded("529")],
    )
    for _ in range(2):
        with pytest.raises((RateLimited, Overloaded)):
            await client.complete("s", "u", SCHEMA)
    turn = await client.complete("s", "u", SCHEMA)
    assert turn.text == '{"orders": []}'
