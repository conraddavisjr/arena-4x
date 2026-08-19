# Every command you need to run this project. If a workflow is not here, it is
# not a workflow yet.

PORT    ?= 8123
PYTHON  ?= python3
VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
RUFF    := $(VENV)/bin/ruff

.DEFAULT_GOAL := help
.PHONY: help setup setup-engine env test test-engine contracts lint fmt clean bots map run view view3d stage3d port-free export library prices preflight

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

preflight:  ## Check every seat can actually spend. make preflight ROSTER=shakeout TURNS=300
	$(PY) scripts/preflight.py --roster $(or $(ROSTER),shakeout) --turns $(or $(TURNS),300)

prices:  ## Print the rate card with its provenance, for re-checking against vendors
	@$(PY) -c "import sys; sys.path.insert(0,'packages'); \
from arena_orchestrator.pricing import RATES, STALE_AFTER_DAYS; \
[print(f'{m:26} {r.input:6.2f} in {r.output:7.2f} out  {r.age_days:3}d old  {r.source}') \
 for m, r in RATES.items()]; \
print(); print(f'stale after {STALE_AFTER_DAYS} days. Update the rate AND its checked date.')"

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
	@$(MAKE) -s port-free GOAL=view
	@echo "http://localhost:$(PORT)/  (ctrl-c to stop)"
	@$(PY) -m http.server $(PORT) --bind 127.0.0.1 --directory $(or $(MATCH),output/match-4)

export:  ## Play a match and write a replay bundle. make export SEED=4
	$(PY) scripts/export_match.py --seed $(or $(SEED),4)
	@cp apps/viewer/index.html output/match-$(or $(SEED),4)/index.html
	@echo "then: make view MATCH=output/match-$(or $(SEED),4)"

view3d:  ## Serve the 3D world viewer. make view3d MATCH=output/match-4
	@$(MAKE) -s port-free GOAL=view3d
	@$(MAKE) -s stage3d MATCH=$(or $(MATCH),output/match-4)
	@echo "board:  http://localhost:$(PORT)/world.html"
	@echo "models: http://localhost:$(PORT)/models.html   (ctrl-c to stop)"
	@$(PY) -m http.server $(PORT) --bind 127.0.0.1 --directory $(or $(MATCH),output/match-4)

# Checked *before* the URL is printed, which is the whole point. These targets
# used to announce the address and then fail to bind, so a stale server left
# over from an earlier run kept answering on it - and the browser showed a
# different match with nothing to say so. A crash you scroll past is worse than
# a crash you read: HTTP 200 proves a server is up, not that it is yours.
port-free:
	@if lsof -nP -iTCP:$(PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
		pid=$$(lsof -nP -iTCP:$(PORT) -sTCP:LISTEN -t | head -1); \
		echo "port $(PORT) is already in use by pid $$pid."; \
		echo "it is currently serving:"; \
		curl -s --max-time 2 http://localhost:$(PORT)/match.json \
			| head -c 200 | sed 's/^/    /' || echo "    (not an ARENA-4X bundle)"; \
		echo ""; \
		echo "stop it with:  kill $$pid"; \
		echo "or pick another port:  make $(or $(GOAL),view3d) PORT=8124 MATCH=$(MATCH)"; \
		exit 1; \
	fi

library:  ## Browse every match played so far. make library
	@$(PY) scripts/build_library.py $(or $(ROOT),output)
	@$(MAKE) -s port-free GOAL=library
	@$(MAKE) -s stage3d MATCH=$(or $(ROOT),output)
	@echo "library: http://localhost:$(PORT)/world.html"
	@$(PY) -m http.server $(PORT) --bind 127.0.0.1 --directory $(or $(ROOT),output)

stage3d:
	@cp -r apps/viewer3d/vendor $(or $(MATCH),output/match-4)/vendor
	@cp apps/viewer3d/world.js $(or $(MATCH),output/match-4)/world.js
	@cp apps/viewer3d/index.html $(or $(MATCH),output/match-4)/world.html
	@cp apps/viewer3d/models.html $(or $(MATCH),output/match-4)/models.html
