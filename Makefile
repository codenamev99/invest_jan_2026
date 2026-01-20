PYTHON ?= python3

.PHONY: install test

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest
