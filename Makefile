# kept out of the repo, a windows drive under WSL2 is slow on small files
VENV ?= $(HOME)/.venvs/driftgate
PY := $(VENV)/bin/python

.PHONY: help env test world fingerprints train serve replay monitor retrain loop gate rollout sweep report

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

monitor: env  ## each monitor's false alarms against its delay, over reserved seeds, to results/monitor/
	$(PY) -m monitor.tradeoff

retrain: env  ## what each retraining policy buys over the serving model, over reserved seeds, to results/retrain/
	$(PY) -m retrain.sweep

loop: env  ## the whole pipeline on one scenario, SEED= and KIND=: watch, retrain, gate, promote
	$(PY) -m loop --seed $(SEED) --kind $(KIND)

gate: env  ## each gate's calls against the truth, on every candidate the loop makes, to results/gate/
	$(PY) -m gate.sweep

rollout: env  ## each rollout schedule's cost in accuracy, over reserved seeds, to results/rollout/
	$(PY) -m rollout

sweep: env  ## every policy on 100 seeds per kind, one record per run, to results/runs/
	$(PY) -m eval.harness

report: env  ## the seven metrics per policy from make sweep's records, to results/eval/
	$(PY) -m eval.report
