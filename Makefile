# not next to the repo: under WSL2 a windows drive is slow on small files
VENV ?= $(HOME)/.venvs/driftgate
PY := $(VENV)/bin/python

.PHONY: help env

help:  ## every target, one line each
	@grep -E '^[a-z][a-z-]*:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

env: $(VENV)/.synced  ## the pinned interpreter and requirements, via uv

$(VENV)/pyvenv.cfg: .python-version
	uv venv --quiet --clear $(VENV)

$(VENV)/.synced: $(VENV)/pyvenv.cfg requirements.txt
	uv pip sync --quiet --python $(PY) requirements.txt
	touch $@
