# Every command you need to run this project. If a workflow is not here, it is
# not a workflow yet.

PYTHON  ?= python3
VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
RUFF    := $(VENV)/bin/ruff

.DEFAULT_GOAL := help
.PHONY: help setup setup-engine env test test-engine contracts lint fmt clean bots map run view view3d stage3d export

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

run:  ## Play a match through the orchestrator. make run ROSTER=dry SEED=4
	$(PY) scripts/run_match.py --roster $(or $(ROSTER),dry) --seed $(or $(SEED),4) \
		$(if $(TURNS),--turns $(TURNS),)

contracts:  ## Live provider contract tests. Spends money; skips without keys
	@echo "These hit the real vendors. Each is a few cents; run before a flagship run."
	$(PYTEST) tests/orchestrator/test_contracts.py -m contract -v

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .hypothesis build dist
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +

# `--directory` rather than `cd`: PY is a path relative to the repo root, so
# changing into the bundle first put the interpreter out of reach and every
# serve target failed with "No such file or directory".
view:  ## Serve an exported match in the browser. make view MATCH=output/match-4
	@echo "http://localhost:8123/  (ctrl-c to stop)"
	@$(PY) -m http.server 8123 --bind 127.0.0.1 --directory $(or $(MATCH),output/match-4)

export:  ## Play a match and write a replay bundle. make export SEED=4
	$(PY) scripts/export_match.py --seed $(or $(SEED),4)
	@cp apps/viewer/index.html output/match-$(or $(SEED),4)/index.html
	@echo "then: make view MATCH=output/match-$(or $(SEED),4)"

view3d:  ## Serve the 3D world viewer. make view3d MATCH=output/match-4
	@$(MAKE) -s stage3d MATCH=$(or $(MATCH),output/match-4)
	@echo "board:  http://localhost:8123/world.html"
	@echo "models: http://localhost:8123/models.html   (ctrl-c to stop)"
	@$(PY) -m http.server 8123 --bind 127.0.0.1 --directory $(or $(MATCH),output/match-4)

stage3d:
	@cp -r apps/viewer3d/vendor $(or $(MATCH),output/match-4)/vendor
	@cp apps/viewer3d/world.js $(or $(MATCH),output/match-4)/world.js
	@cp apps/viewer3d/index.html $(or $(MATCH),output/match-4)/world.html
	@cp apps/viewer3d/models.html $(or $(MATCH),output/match-4)/models.html
