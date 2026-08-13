# Every command you need to run this project. If a workflow is not here, it is
# not a workflow yet.

PYTHON  ?= python3
VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
RUFF    := $(VENV)/bin/ruff

.DEFAULT_GOAL := help
.PHONY: help setup setup-engine env test test-engine lint fmt clean bots map

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup: $(VENV) env  ## Create the venv and install everything
	$(PIP) install -q -e ".[engine,orchestrator,api,dev]"
	@echo "Python side ready."

setup-engine: $(VENV)  ## Engine only. No LLM SDKs, no database. Fast and offline.
	$(PIP) install -q -e ".[engine,dev]"

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -q --upgrade pip

env:
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example - add your API keys before running a match.")

# ---------------------------------------------------------------------------
# Test and lint
# ---------------------------------------------------------------------------

test:  ## Run the whole suite
	$(PYTEST) -q

test-engine:  ## Engine tests only. No network, no database.
	$(PYTEST) -q tests/engine

bots:  ## Play a full match with four scripted heuristic bots. No API keys needed.
	$(PY) scripts/run_bots.py $(ARGS)

map:  ## Render a generated map to the terminal. make map ARGS="--seed 7 --resources"
	$(PY) scripts/show_map.py $(ARGS)

lint:  ## Check formatting and lints
	$(RUFF) check packages tests scripts
	$(RUFF) format --check packages tests scripts

fmt:  ## Fix what can be fixed automatically
	$(RUFF) check --fix packages tests scripts
	$(RUFF) format packages tests scripts

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .hypothesis build dist
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
