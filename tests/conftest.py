"""Shared test setup.

The only thing here is loading `.env`, and it earns its place: the repo ships a
`.env.example` listing four API keys, which reads as a promise that a `.env`
file works. Nothing was reading one. Following that documented workflow and then
watching every contract test skip for a missing key you had just written down is
a bad half hour, and the fix is nine lines.

Deliberately does not override a variable already in the environment - an
explicit `export` should beat a file the shell never saw - and deliberately no
dependency, because `python-dotenv` is more machinery than four `KEY=value`
lines deserve.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def pytest_configure() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
