# Shortcuts to reproduce the project. In Colab, src/training.ipynb calls the same steps.
PY ?= python3
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: install test smoke validate verify data translation models stats figures all relocate ui clean

install:            ## installs the pinned dependencies
	$(PY) -m pip install -r requirements.txt

test:               ## unit tests (cleaning rules, movement metrics, statistics, contract, Parquet/CSV ingestion, determinism)
	$(PY) -m pytest -q

smoke:              ## the whole pipeline with minimal CV and grids (CI, ~10 min; its own output folders)
	$(PY) src/train.py --config configs/smoke.yaml --stage all

validate:           ## data contract only: raw files vs the spec -> reports/metrics/data_contract_*
	$(PY) src/data_prep.py --stage contract

verify:             ## compares each raw file's hash with data/raw/manifest.json
	$(PY) src/data_prep.py --stage verify

data:               ## contract + ingest + cleaning + movement + on-field metrics + manifest
	$(PY) src/data_prep.py --stage data

translation:        ## drill-to-game associations, reliability and GLM on the clean data
	$(PY) src/train.py --stage translation

models:             ## translation + nested CV of regression and classification + MLflow registry
	$(PY) src/train.py --stage models

stats:              ## recomputes the omnibus tests from the per-fold scores (no retraining)
	$(PY) src/train.py --stage stats

figures:            ## regenerates reports/figures from the saved metrics
	$(PY) src/train.py --stage figures

all:                ## full reproducible run (~60 min on 2 cores)
	$(PY) src/train.py --stage all

relocate:           ## once after cloning or extracting: points the MLflow store paths at this folder
	$(PY) src/train.py --stage relocate

ui:                 ## browse runs and the model registry
	mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

clean:              ## deletes the smoke-run outputs (keeps raw data, clean data and mlflow.db)
	rm -rf data/smoke reports/smoke mlflow/smoke.db mlflow/smoke_artifacts
