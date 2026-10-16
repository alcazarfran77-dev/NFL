PY ?= python3
export PYTHONPATH := src
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: install test smoke data translation models stats figures all ui clean

install:            ## install pinned dependencies
	$(PY) -m pip install -r requirements.txt

test:               ## unit tests (cleaning rules, movement metrics, statistics, determinism)
	$(PY) -m pytest -q

smoke:              ## whole pipeline on tiny CV/grids (CI)
	$(PY) -m cgmt.pipeline --config configs/smoke.yaml --stage all

data:               ## ingest + clean + movement metrics + manifest
	$(PY) -m cgmt.pipeline --stage data

translation:        ## drill-vs-game statistics, reliability, GLM
	$(PY) -m cgmt.pipeline --stage translation

models:             ## translation + nested-CV regression and classification + registry
	$(PY) -m cgmt.pipeline --stage models

stats:              ## recompute omnibus tests from saved fold scores (no retraining)
	$(PY) -m cgmt.pipeline --stage stats

figures:            ## regenerate reports/figures from saved metrics
	$(PY) -m cgmt.pipeline --stage figures

all:                ## full reproducible run (about 25 min on 2 cores)
	$(PY) -m cgmt.pipeline --stage all

ui:                 ## browse runs and the model registry
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:              ## remove generated data and smoke artefacts (keeps raw data and mlflow.db)
	rm -rf data/interim data/processed reports/smoke mlflow_smoke.db mlartifacts_smoke
