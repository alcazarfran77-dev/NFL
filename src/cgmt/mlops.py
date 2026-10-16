"""MLflow tracking and model-registry helpers.

Backend: a local SQLite store (mlflow.db) so the model registry works offline;
artifacts go to ./mlartifacts. Every run is tagged with the pipeline version, the
data version (hash from data/manifest.json), the git commit and the seed, which
is what makes any logged number traceable and reproducible.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from . import __version__  # noqa: E402
from .config import PROJECT_ROOT, Config  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def setup(cfg: Config) -> str:
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{PROJECT_ROOT / uri.removeprefix('sqlite:///')}"
    mlflow.set_tracking_uri(uri)
    name = cfg["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        art = (PROJECT_ROOT / cfg["mlflow"]["artifact_location"]).as_uri()
        mlflow.create_experiment(name, artifact_location=art)
    mlflow.set_experiment(name)
    return uri


def base_tags(cfg: Config) -> dict[str, str]:
    manifest_path = cfg.path("manifest")
    data_version = json.loads(manifest_path.read_text())["data_version"] if manifest_path.exists() else "unknown"
    return {
        "pipeline_version": __version__,
        "data_version": data_version,
        "git_commit": git_commit(),
        "seed": str(cfg.seed),
        "python": platform.python_version(),
    }


@contextmanager
def run(cfg: Config, name: str, tags: dict | None = None):
    with mlflow.start_run(run_name=name, tags={**base_tags(cfg), **(tags or {})}) as r:
        yield r


def log_cv_result(res, extra_params: dict | None = None) -> None:
    """Log fold metrics (as a step series), their mean/std and the OOF predictions."""
    mlflow.log_params({"model": res.model, "feature_set": res.feature_set, "task": res.task,
                       "n_train_per_fold": res.n_train, "n_test_per_fold": res.n_test, **(extra_params or {})})
    metric_cols = [c for c in res.fold_scores.columns if c not in ("fold", "repeat")]
    for _, row in res.fold_scores.iterrows():
        for m in metric_cols:
            mlflow.log_metric(f"fold_{m}", float(row[m]), step=int(row["fold"]))
    summary = res.fold_scores[metric_cols].agg(["mean", "std"])
    for m in metric_cols:
        mlflow.log_metric(f"cv_{m}_mean", float(summary.loc["mean", m]))
        mlflow.log_metric(f"cv_{m}_std", float(summary.loc["std", m]))
    mlflow.log_text(res.fold_scores.to_csv(index=False), "cv/fold_scores.csv")
    mlflow.log_text(res.oof.to_csv(index=False), "cv/oof_predictions.csv")
    mlflow.log_text(pd.DataFrame(res.best_params).to_csv(index=False), "cv/selected_hyperparameters.csv")


def relocate_store(cfg: Config) -> int:
    """Point a copied MLflow store at this checkout. Returns the number of cells updated.

    MLflow records artifact locations as absolute URIs, so a store moved to another
    folder or machine cannot find its models. This rewrites the recorded project root
    (read from the experiment's artifact location) to the current PROJECT_ROOT in every
    text column of the SQLite store. Run once after extracting the release archive.
    """
    import sqlite3

    db = PROJECT_ROOT / cfg["mlflow"]["tracking_uri"].removeprefix("sqlite:///")
    art_name = cfg["mlflow"]["artifact_location"]
    con = sqlite3.connect(db)
    try:
        row = con.execute("SELECT artifact_location FROM experiments WHERE name = ?", (cfg["mlflow"]["experiment"],)).fetchone()
        if row is None:
            return 0
        old_art = row[0].removeprefix("file://")
        old_root = old_art[: -len(art_name)].rstrip("/")
        new_root = str(PROJECT_ROOT)
        if old_root == new_root:
            return 0
        changed = 0
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        for t in tables:
            for _, col, ctype, *_ in con.execute(f'PRAGMA table_info("{t}")'):
                if "CHAR" in ctype.upper() or "TEXT" in ctype.upper():
                    cur = con.execute(f'UPDATE "{t}" SET "{col}" = REPLACE("{col}", ?, ?) WHERE "{col}" LIKE ?',
                                      (old_root, new_root, f"%{old_root}%"))
                    changed += cur.rowcount
        con.commit()
        return changed
    finally:
        con.close()


def log_and_register(model, X: pd.DataFrame, task: str, registered_name: str | None, alias: str | None = None):
    """Log a fitted sklearn pipeline with signature; optionally register and alias it."""
    example = X.head(5)
    output = model.predict_proba(example)[:, 1] if task == "classification" else model.predict(example)
    sig = infer_signature(example, output)
    info = mlflow.sklearn.log_model(
        model, name="model", signature=sig, input_example=example,
        code_paths=[str(Path(__file__).resolve().parent)],
        # The pipeline contains a project transformer (PositionMedianImputer) that skops
        # cannot serialise; cloudpickle is safe here because only our own code is loaded.
        serialization_format="cloudpickle",
        registered_model_name=registered_name,
    )
    if registered_name and alias:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{registered_name}'")
        latest = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias(registered_name, alias, latest.version)
        client.set_model_version_tag(registered_name, latest.version, "validation_status", "nested-cv-passed")
    return info
