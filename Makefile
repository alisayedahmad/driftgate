# kept out of the repo, a windows drive under WSL2 is slow on small files
VENV ?= $(HOME)/.venvs/driftgate
PY := $(VENV)/bin/python

.PHONY: help env test world fingerprints train serve replay

help:  ## every target, one line each
	@grep -E '^[a-z][a-z-]*:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-13s %s\n", $$1, $$2}'

env: $(VENV)/.synced  ## the pinned interpreter and requirements, via uv

# shared by every checkout, rebuilt only if its python isn't the pinned one
$(VENV)/.synced: .python-version requirements.txt
	@[ "$$($(PY) -c 'import platform; print(platform.python_version())' 2>/dev/null)" = "$$(cat .python-version)" ] \
		|| uv venv --quiet --clear $(VENV)
	uv pip sync --quiet --python $(PY) requirements.txt
	touch $@

test: env  ## run the tests
	$(PY) -m pytest -q

SEED ?= 0
KIND ?= concept

world: env  ## draw a scenario and look at it, pick it with SEED= and KIND=
	$(PY) -m world.summary --seed $(SEED) --kind $(KIND)

fingerprints: env  ## rewrite world/fingerprints.txt, only when the data is meant to change
	$(PY) -m world.fingerprint

train: env  ## train two models on a scenario, register them, rebuild them from their records
	$(PY) -m retrain.train --seed $(SEED) --kind $(KIND)

serve: env  ## serve a model trained on a scenario at http://127.0.0.1:8000, SEED= and KIND=
	$(PY) -m serving.app --seed $(SEED) --kind $(KIND)

replay: env  ## replay a scenario through the service, true accuracy against what was known
	$(PY) -m serving.replay --seed $(SEED) --kind $(KIND)
