# Aletheia 澄明 — 构建与运行
# 用法: make install / make test / make eval / make serve / make offline-eval

PY ?= python3
PIP_INDEX ?= https://pypi.tuna.tsinghua.edu.cn/simple/
VENV ?= .venv
BIN := $(VENV)/bin
ifeq ($(OS),Windows_NT)
  BIN := $(VENV)/Scripts
endif

.PHONY: install test eval offline-eval serve lint clean lock-check

install:
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install -U pip
	$(BIN)/pip install -r requirements.lock.txt -i $(PIP_INDEX)

test:
	$(BIN)/python -m pytest tests/ -q

offline-eval:
	ALETHEIA_OFFLINE=1 $(BIN)/python scripts/eval.py

eval:
	$(BIN)/python scripts/eval.py

serve:
	$(BIN)/python -m aletheia.cli serve --host 0.0.0.0 --port 8000

lint:
	$(BIN)/python -m ruff check src tests

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache data/eval_report.json
