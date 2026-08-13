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
}


def sanitize(node: Any) -> Any:
    """Recursively drop unsupported keywords and close every object."""
    if isinstance(node, list):
        return [sanitize(item) for item in node]
    if not isinstance(node, dict):
        return node

    out = {k: sanitize(v) for k, v in node.items() if k not in UNSUPPORTED}

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
