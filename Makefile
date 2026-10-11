# kept out of the repo, a windows drive under WSL2 is slow on small files
VENV ?= $(HOME)/.venvs/driftgate
PY := $(VENV)/bin/python

.PHONY: help env test

help:  ## every target, one line each
	@grep -E '^[a-z][a-z-]*:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

env: $(VENV)/.synced  ## the pinned interpreter and requirements, via uv

# shared by every checkout, rebuilt only if its python isn't the pinned one
$(VENV)/.synced: .python-version requirements.txt
	@[ "$$($(PY) -c 'import platform; print(platform.python_version())' 2>/dev/null)" = "$$(cat .python-version)" ] \
		|| uv venv --quiet --clear $(VENV)
	uv pip sync --quiet --python $(PY) requirements.txt
	touch $@

test: env  ## run the tests
	$(PY) -m pytest -q
