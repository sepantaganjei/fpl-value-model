.PHONY: help install fmt lint typecheck test check data train app clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install dependencies into the project virtualenv
	poetry install

fmt:  ## Auto-format and auto-fix with ruff
	poetry run ruff format .
	poetry run ruff check --fix .

lint:  ## Lint without modifying files
	poetry run ruff format --check .
	poetry run ruff check .

typecheck:  ## Static type check
	poetry run mypy

test:  ## Run the test suite
	poetry run pytest

check: lint typecheck test  ## Run every quality gate

data:  ## Refresh cached FPL data
	poetry run python -m fpl_value_model.data

train:  ## Fit the model and write predictions
	poetry run python -m fpl_value_model.pipeline

app:  ## Launch the Streamlit dashboard locally
	poetry run streamlit run app/streamlit_app.py

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ .coverage
