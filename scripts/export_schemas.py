"""Write the JSON Schemas the providers are given, from the Pydantic models.

    python scripts/export_schemas.py

Generated rather than hand-written for the same reason as the rules reference:
a schema that describes an action the engine no longer accepts would produce
perfectly valid model output that gets rejected every turn, and nothing would
fail loudly.

**Authored to Anthropic's dialect, which is the strictest of the four.** It
requires `additionalProperties: false` on every object, and supports neither
numeric bounds (`minimum`, `maximum`) nor string bounds (`minLength`,
`maxLength`) nor recursive schemas. Meeting that constraint means the same
schema works unmodified on OpenAI, Gemini and xAI, so exactly one schema is
maintained rather than four. The bounds Pydantic would have emitted are enforced
after parsing instead - see the validators in `actions.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena_engine.actions import Action
from arena_engine.observation import Observation

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "schemas"

# Keywords Anthropic's structured-output dialect does not support. Emitting them
# is not a warning, it is a rejected request, so they are stripped rather than
# trusted to be ignored.
UNSUPPORTED = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "default",
    # Pydantic emits this alongside `oneOf` for a discriminated union. It is an
    # OpenAPI keyword rather than a JSON Schema one, and it is meaningless once
    # the union below has been rewritten.
    "discriminator",
}

# Keywords that carry meaning and so are renamed rather than dropped. Kept
# separate from UNSUPPORTED because stripping these would delete the union
# itself; the dialect test asserts against both sets.
REWRITES = {"oneOf": "anyOf"}


def sanitize(node: Any) -> Any:
    """Recursively drop unsupported keywords and close every object."""
    if isinstance(node, list):
        return [sanitize(item) for item in node]
    if not isinstance(node, dict):
        return node

    out = {k: sanitize(v) for k, v in node.items() if k not in UNSUPPORTED}

    # Pydantic renders a discriminated union as `oneOf`, and neither Anthropic
    # nor OpenAI supports it: both reject the request outright with
    # "'oneOf' is not permitted". `anyOf` is accepted everywhere and means the
    # same thing here, because the branches are mutually exclusive anyway - each
    # is discriminated by a different `const` on `action`.
    #
    # This is worth the comment because of how it failed. Every mocked test
    # passed, the schema round-tripped through the parity test, and a full dry
    # match played 108 turns without complaint - the schema is never sent to a
    # vendor in any of those. It would have 400'd on turn one of the flagship
    # run, on three of four seats simultaneously.
    for old, new in REWRITES.items():
        if old in out:
            out[new] = out.pop(old)

    if out.get("type") == "object" or "properties" in out:
        out["additionalProperties"] = False
        # Anthropic strict mode wants `required` to name every property. Optional
        # fields stay optional by permitting null in their own type instead.
        properties = out.get("properties", {})
        if properties:
            out["required"] = sorted(properties)
    return out


def export(model: type, name: str, title: str) -> Path:
    schema = sanitize(model.model_json_schema())
    schema["title"] = title
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    path = OUT / name
    path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for model, name, title in (
        (Action, "action.schema.json", "ARENA-4X agent action"),
        (Observation, "observation.schema.json", "ARENA-4X observation"),
    ):
        path = export(model, name, title)
        size = len(path.read_text())
        print(f"{path.relative_to(ROOT)}  {size:,} bytes")


if __name__ == "__main__":
    main()
