"""Training, evaluation and MLflow registration for the NFL project.

Uses the clean data from `data_prep.py` and runs:
  4 translation     associations Combine drill -> game movement and -> on-field performance,
                    reliability, mediation and generalised linear models (GLM)
  5 regression      Combine -> Snap Burst Index (SBI) and Combine -> on-field performance
                    (position mean vs Ridge vs Random forest vs XGBoost)
  6 classification  starter (logistic regression vs Random forest vs XGBoost) x nested feature sets
  7 statistics      omnibus tests recomputed from the saved per-fold scores (no retraining)
  8 figures         the report's 21 figures in reports/figures

Every model is a scikit-learn Pipeline: position-median imputation -> scaling / one-hot -> estimator,
fitted inside each training fold. Evaluation is nested cross-validation (5 folds x 3 repeats outside;
3-fold GridSearchCV inside) with seed 42. Every run is logged to MLflow with the tags
pipeline_version, data_version, git_commit and seed; champion models are refitted on all rows and
registered with aliases (champion / challenger).

Usage
  python src/train.py                          all: data + translation + models + figures
  python src/train.py --stage models           translation + models on the clean data already saved
  python src/train.py --stage figures          figures only, from the saved metrics
  python src/train.py --stage stats            omnibus tests without retraining
  python src/train.py --stage relocate         once after cloning or extracting: fixes MLflow store paths
  python src/train.py --config configs/smoke.yaml   smoke run (CI, ~10 min)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin, clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,  # noqa: E402
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from xgboost import XGBClassifier, XGBRegressor  # noqa: E402

from data_prep import (ATHLETIC, ATHLETIC_CT, CT_FEATURES, CT_LOWER_IS_BETTER, CT_SET,  # noqa: E402
                       FEATURE_SETS, METRICS, ONFIELD_METRICS, PROJECT_ROOT, Config, __version__, _dump, _role,
                       _save, _strata, classification_table, context_adjusted_burst, estimator_bias,
                       has_combine_tracking, load_config, load_processed, metric_reliability,
                       official_vs_tracking, onfield_classification_table, onfield_regression_table,
                       projector_table, regression_table, reliability_table, rookie_stability, set_global_seed,
                       speed_curves, stage_contract, stage_data, xy)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger("nfl")


# =============================================================================
# Models: preprocessing and nested cross-validation
# =============================================================================
# Every model is an sklearn Pipeline:
#   PositionMedianImputer -> ColumnTransformer(StandardScaler | OneHotEncoder) -> model
# so imputation medians and scaling statistics are learned inside each training fold only (no
# leakage from the validation folds).
#
# Evaluation = nested cross-validation: an outer RepeatedStratifiedKFold estimates performance and an
# inner GridSearchCV picks the hyperparameters using the outer training part only.
GROUP_COL = "combine_position"


class PositionMedianImputer(BaseEstimator, TransformerMixin):
    """Fills numeric NaNs with the position-group median of the training fold.

    A group with no observed value falls back to the global training median (for example, offensive
    linemen rarely run the 3-cone in some folds). The `exclude` columns (projector labels) pass
    through unchanged.
    """

    def __init__(self, group_col: str = GROUP_COL, exclude: tuple = ()):
        self.group_col = group_col
        self.exclude = exclude

    def fit(self, X: pd.DataFrame, y=None):
        num = X.drop(columns=[self.group_col, *[c for c in self.exclude if c in X.columns]]).select_dtypes("number")
        self.numeric_cols_ = list(num.columns)
        self.global_median_ = num.median()
        self.group_median_ = num.groupby(X[self.group_col].astype(str)).median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        groups = X[self.group_col].astype(str)
        for col in self.numeric_cols_:
            miss = X[col].isna()
            if miss.any():
                fill = groups[miss].map(self.group_median_[col]) if col in self.group_median_ else np.nan
                X.loc[miss, col] = pd.Series(fill, index=X.index[miss]).fillna(self.global_median_[col])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.numeric_cols_ + [self.group_col], dtype=object)


PROJECTOR_PREFIX = "onfield_target_"


class OnFieldProjector(BaseEstimator, TransformerMixin):
    """Stacked Combine -> on-field performance projection. Usable before the draft.

    The `onfield_target_<metric>` columns carry the player's observed on-field value (NaN if not
    measured) and are used ONLY as labels: `fit` trains one Ridge per metric on the labelled rows,
    using the Combine columns. `transform` replaces each label column by `proj_<metric>`, the
    projection made from the Combine alone (0 for players outside the metric's position family).
    Training rows get out-of-fold projections (`fit_transform`), so the classifier sees the same kind
    of noisy projection it will see for new players.
    """

    def __init__(self, alpha: float = 10.0, inner_folds: int = 5, random_state: int = 0, group_col: str = GROUP_COL):
        self.alpha = alpha
        self.inner_folds = inner_folds
        self.random_state = random_state
        self.group_col = group_col

    def _split(self, X):
        self.targets_ = [c for c in X.columns if c.startswith(PROJECTOR_PREFIX)]
        self.features_ = [c for c in X.columns if c not in self.targets_ and c != self.group_col]

    def _model(self):
        return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=self.alpha))])

    def fit(self, X: pd.DataFrame, y=None):
        self._split(X)
        self.models_, self.positions_ = {}, {}
        for t in self.targets_:
            rows = X[t].notna()
            self.positions_[t] = set(X.loc[rows, self.group_col].astype(str))
            self.models_[t] = self._model().fit(X.loc[rows, self.features_], X.loc[rows, t]) if rows.sum() >= 10 else None
        return self

    def _project(self, X: pd.DataFrame, t: str, model) -> np.ndarray:
        out = np.zeros(len(X))
        if model is None:
            return out
        fam = X[self.group_col].astype(str).isin(self.positions_[t]).to_numpy()
        if fam.any():
            out[fam] = model.predict(X.loc[fam, self.features_])
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.drop(columns=self.targets_).copy()
        for t in self.targets_:
            out[t.replace(PROJECTOR_PREFIX, "proj_")] = self._project(X, t, self.models_[t])
        return out

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        self.fit(X)
        out = self.transform(X)
        for t in self.targets_:  # replace the in-sample projections of labelled rows by out-of-fold ones
            rows = np.flatnonzero(X[t].notna().to_numpy())
            if len(rows) < 2 * self.inner_folds:
                continue
            col = t.replace(PROJECTOR_PREFIX, "proj_")
            kf = KFold(self.inner_folds, shuffle=True, random_state=self.random_state)
            for tr, te in kf.split(rows):
                m = self._model().fit(X.iloc[rows[tr]][self.features_], X.iloc[rows[tr]][t])
                out.iloc[rows[te], out.columns.get_loc(col)] = m.predict(X.iloc[rows[te]][self.features_])
        return out


def make_preprocessor(numeric_cols: list[str], include_position: bool = True,
                      projector: dict | None = None) -> Pipeline:
    """Position imputation (+ projector when there are on-field labels) + scaling / one-hot."""
    targets = [c for c in numeric_cols if c.startswith(PROJECTOR_PREFIX)]
    plain = [c for c in numeric_cols if c not in targets]
    steps = [("impute", PositionMedianImputer(exclude=tuple(targets)))]
    if targets:
        steps.append(("project", OnFieldProjector(**(projector or {}))))
        plain = plain + [t.replace(PROJECTOR_PREFIX, "proj_") for t in targets]
    transformers = [("num", StandardScaler(), plain)]
    if include_position:
        transformers.append(("pos", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [GROUP_COL]))
    steps.append(("encode", ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)))
    return Pipeline(steps)


def make_estimator(name: str, task: str, seed: int, pos_weight: float = 1.0):
    """Estimator by name and task, always seeded."""
    if task == "regression":
        return {
            "position_mean": LinearRegression(),
            "ridge": Ridge(random_state=seed),
            "random_forest": RandomForestRegressor(random_state=seed, n_jobs=1),
            "xgboost": XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist", verbosity=0),
        }[name]
    return {
        "logistic_regression": LogisticRegression(solver="saga", max_iter=5000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1),
        "xgboost": XGBClassifier(random_state=seed, n_jobs=1, tree_method="hist", eval_metric="logloss",
                                 scale_pos_weight=pos_weight, verbosity=0),
    }[name]


def make_pipeline(name: str, task: str, numeric_cols: list[str], seed: int, pos_weight: float = 1.0,
                  projector: dict | None = None) -> Pipeline:
    """Full pipeline: preprocessing + estimator."""
    if name == "position_mean":  # baseline: only the position-group one-hot -> group means
        pre = make_preprocessor([], include_position=True)
    else:
        pre = make_preprocessor(numeric_cols, projector={**(projector or {}), "random_state": seed})
    return Pipeline([("prep", pre), ("model", make_estimator(name, task, seed, pos_weight))])


def projector_params(cfg) -> dict:
    c = cfg["classification"]
    return {"alpha": float(c.get("projector_alpha", 10.0)), "inner_folds": int(c.get("projector_inner_folds", 5))}


def regression_metrics(y, pred) -> dict[str, float]:
    return {"mae": mean_absolute_error(y, pred), "rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": r2_score(y, pred)}


def classification_metrics(y, proba, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "brier": brier_score_loss(y, proba),
    }


@dataclass
class CVResult:
    """Result of one nested cross-validation: per-fold scores, out-of-fold predictions and the
    hyperparameters chosen in each fold."""

    model: str
    feature_set: str
    task: str
    fold_scores: pd.DataFrame
    oof: pd.DataFrame                    # columns: row, repeat, y, pred
    best_params: list[dict] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0


def outer_splitter(cfg, seed: int) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=cfg["cv"]["outer_splits"], n_repeats=cfg["cv"]["outer_repeats"], random_state=seed)


def nested_cv(model_name: str, feature_set_name: str, task: str, X: pd.DataFrame, y: pd.Series,
              strata: pd.Series, grid: dict, cfg, seed: int, threshold: float = 0.5) -> CVResult:
    """Nested cross-validation of one model: grid search inside, evaluation outside."""
    numeric_cols = [c for c in X.columns if c != GROUP_COL]
    outer = outer_splitter(cfg, seed)
    k = cfg["cv"]["outer_splits"]
    rows, oof, params = [], [], []
    for fold, (tr, te) in enumerate(outer.split(X, strata)):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1)) if task == "classification" else 1.0
        pipe = make_pipeline(model_name, task, numeric_cols, seed, pos_weight, projector_params(cfg))
        if task == "classification":
            inner = StratifiedKFold(cfg["cv"]["inner_splits"], shuffle=True, random_state=seed + fold)
            scoring = cfg["classification"]["scoring"]
        else:
            inner = KFold(cfg["cv"]["inner_splits"], shuffle=True, random_state=seed + fold)
            scoring = "neg_root_mean_squared_error"
        if grid:
            search = GridSearchCV(pipe, grid, cv=inner, scoring=scoring, n_jobs=-1, refit=True)
            search.fit(Xtr, ytr)
            est, best = search.best_estimator_, search.best_params_
        else:
            est, best = clone(pipe).fit(Xtr, ytr), {}
        if task == "classification":
            pred = est.predict_proba(Xte)[:, 1]
            m = classification_metrics(yte, pred, threshold)
        else:
            pred = est.predict(Xte)
            m = regression_metrics(yte, pred)
        rows.append({"fold": fold, "repeat": fold // k, **m})
        oof.append(pd.DataFrame({"row": X.index[te], "repeat": fold // k, "y": yte.to_numpy(), "pred": pred}))
        params.append(best)
    return CVResult(model_name, feature_set_name, task, pd.DataFrame(rows), pd.concat(oof, ignore_index=True),
                    params, n_train=len(tr), n_test=len(te))


def refit_final(model_name: str, task: str, X: pd.DataFrame, y: pd.Series, grid: dict, cfg, seed: int):
    """Tunes on all data (5-fold inner CV) and refits: this is the model that gets registered."""
    numeric_cols = [c for c in X.columns if c != GROUP_COL]
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1)) if task == "classification" else 1.0
    pipe = make_pipeline(model_name, task, numeric_cols, seed, pos_weight, projector_params(cfg))
    if task == "classification":
        cv, scoring = StratifiedKFold(5, shuffle=True, random_state=seed), cfg["classification"]["scoring"]
    else:
        cv, scoring = KFold(5, shuffle=True, random_state=seed), "neg_root_mean_squared_error"
    if not grid:
        return clone(pipe).fit(X, y), {}, np.nan
    search = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True).fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_


def feature_names(pipe: Pipeline) -> list[str]:
    return list(pipe.named_steps["prep"].named_steps["encode"].get_feature_names_out())


# =============================================================================
# Statistical comparison
# =============================================================================
#   corrected_resampled_ttest  Nadeau and Bengio (2003): paired t-test on repeated-CV scores with the
#                              variance inflated for the overlap of the training sets (the naive
#                              paired t-test is too liberal).
#   pairwise_model_tests       the above + Wilcoxon, Holm-corrected.
#   friedman                   omnibus test that k models have the same rank across folds.
#   bootstrap_ci               percentile interval of any metric on out-of-fold predictions.
#   delong_test                DeLong (1988) for two correlated ROC-AUCs.
#   within_group_spearman      drill-game association net of the position group, bootstrap CI +
#                              Benjamini-Hochberg q.
#   partial_spearman           rank correlation of x and y after removing the linear effect of controls.
def corrected_resampled_ttest(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    j = len(d)
    var = d.var(ddof=1)
    se = np.sqrt((1 / j + n_test / n_train) * var) if var > 0 else np.nan
    t = d.mean() / se if se and se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), df=j - 1) if np.isfinite(t) else np.nan
    return {"mean_diff": d.mean(), "t": t, "df": j - 1, "p_value": p}


def holm(pvals: list[float]) -> np.ndarray:
    """Holm-adjusted p-values (step-down, monotone, capped at 1)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def pairwise_model_tests(scores: pd.DataFrame, metric: str, n_train: int, n_test: int,
                         higher_is_better: bool = True) -> pd.DataFrame:
    """scores: one column per model, one row per outer fold (the same folds for every model)."""
    rows = []
    for m1, m2 in combinations(scores.columns, 2):
        a, b = scores[m1].to_numpy(), scores[m2].to_numpy()
        ct = corrected_resampled_ttest(a, b, n_train, n_test)
        try:
            w = stats.wilcoxon(a, b, zero_method="zsplit").pvalue
        except ValueError:
            w = np.nan
        better = m1 if (ct["mean_diff"] > 0) == higher_is_better else m2
        rows.append({"metric": metric, "model_a": m1, "model_b": m2, "mean_diff_a_minus_b": ct["mean_diff"],
                     "t_corrected": ct["t"], "p_corrected_t": ct["p_value"], "p_wilcoxon": w, "favours": better})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_corrected_t_holm"] = holm(out["p_corrected_t"].fillna(1).tolist())
        out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"].fillna(1).tolist())
    return out


def friedman(scores: pd.DataFrame, higher_is_better: bool = True) -> dict:
    """Friedman test across models on the same folds; mean_ranks: 1 = best in each fold."""
    if scores.shape[1] < 3:
        return {"statistic": np.nan, "p_value": np.nan, "mean_ranks": {}}
    res = stats.friedmanchisquare(*[scores[c].to_numpy() for c in scores.columns])
    ranks = scores.rank(axis=1, ascending=not higher_is_better).mean().to_dict()
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "rank_1_is": "best", "mean_ranks": ranks}


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, fn: Callable, n_boot: int, seed: int,
                 stratify: bool = False, alpha: float = 0.05) -> tuple[float, float, float]:
    """(estimate, lower bound, upper bound) by percentile bootstrap."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        if stratify:
            idx = np.concatenate([rng.choice(np.where(y == c)[0], (y == c).sum()) for c in np.unique(y)])
        else:
            idx = rng.integers(0, n, n)
        try:
            vals.append(fn(y[idx], pred[idx]))
        except ValueError:
            continue
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(fn(y, pred)), float(lo), float(hi)


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    """Two-sided DeLong test of AUC(p1) vs AUC(p2) on the same samples."""
    y = np.asarray(y).astype(int)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v10s, v01s = [], [], []
    for p in (np.asarray(p1), np.asarray(p2)):
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(np.concatenate([p[pos], p[neg]]))
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v10s.append((tz[:m] - tx) / n)
        v01s.append(1 - (tz[m:] - ty) / m)
        aucs.append(auc)
    s10, s01 = np.cov(np.vstack(v10s)), np.cov(np.vstack(v01s))
    cov = s10 / m + s01 / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else np.nan
    p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    return {"auc_1": aucs[0], "auc_2": aucs[1], "diff": aucs[0] - aucs[1], "z": z, "p_value": p}


def mediation_bootstrap(df: pd.DataFrame, x: str, m: str, y: str, n_boot: int, seed: int) -> dict:
    """Mediation x -> m -> y by product of coefficients on z-scored variables, percentile bootstrap.

    a = slope of m on x; b = slope of y on m controlling for x; indirect = a x b.
    total = slope of y on x; direct = slope of x in y ~ x + m. Coefficients in SD units.
    """
    d = df[[x, m, y]].dropna()
    z = ((d - d.mean()) / d.std(ddof=1)).to_numpy()

    def paths(arr: np.ndarray) -> tuple[float, float, float, float]:
        xx, mm, yy = arr[:, 0], arr[:, 1], arr[:, 2]
        a = np.polyfit(xx, mm, 1)[0]
        total = np.polyfit(xx, yy, 1)[0]
        design = np.column_stack([np.ones(len(arr)), xx, mm])
        coef = np.linalg.lstsq(design, yy, rcond=None)[0]
        return a, coef[2], total, coef[1]

    a, b, total, direct = paths(z)
    rng = np.random.default_rng(seed)
    boots, directs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(z), len(z))
        ba, bb, _, bd = paths(z[idx])
        boots.append(ba * bb)
        directs.append(bd)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    dlo, dhi = np.percentile(directs, [2.5, 97.5])
    return {"n": len(d), "a_x_to_m": a, "b_m_to_y": b, "total_effect": total, "direct_effect": direct,
            "direct_ci_low": dlo, "direct_ci_high": dhi,
            "indirect_effect": a * b, "indirect_ci_low": lo, "indirect_ci_high": hi,
            "share_mediated": (a * b) / total if abs(total) > 1e-9 else np.nan}


def within_group_spearman(df: pd.DataFrame, xs: list[str], ys: list[str], group: str,
                          n_boot: int, seed: int) -> pd.DataFrame:
    """Spearman rho between x and y after removing the position-group means from both."""
    rng = np.random.default_rng(seed)
    res = df.copy()
    for c in set(xs) | set(ys):
        res[c] = df[c] - df.groupby(group, observed=True)[c].transform("mean")
    rows = []
    for x in xs:
        for y in ys:
            d = res[[x, y]].dropna()
            if len(d) < 15:
                continue
            rho, p = stats.spearmanr(d[x], d[y])
            a, b = d[x].to_numpy(), d[y].to_numpy()
            boots = []
            for _ in range(n_boot):
                idx = rng.integers(0, len(d), len(d))
                boots.append(stats.spearmanr(a[idx], b[idx])[0])
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            rows.append({"x": x, "y": y, "n": len(d), "rho": rho, "ci_low": lo, "ci_high": hi, "p_value": p})
    out = pd.DataFrame(rows, columns=["x", "y", "n", "rho", "ci_low", "ci_high", "p_value"])
    # BH over the finite p-values (a constant column gives rho = NaN: reported but not tested)
    p = out["p_value"].to_numpy(dtype=float)
    q = np.full(len(p), np.nan)
    if np.isfinite(p).any():
        q[np.isfinite(p)] = stats.false_discovery_control(p[np.isfinite(p)], method="bh")
    out["q_value"] = q
    return out


def partial_spearman(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> dict:
    """Partial Spearman correlation: Pearson r between rank residuals after regressing out the controls' ranks."""
    d = df[[x, y, *controls]].dropna()
    if len(d) < len(controls) + 10:
        return {"n": len(d), "rho": np.nan, "p_value": np.nan}
    r = d.rank()
    design = np.column_stack([np.ones(len(r)), r[controls].to_numpy()])

    def resid(v):
        return v - design @ np.linalg.lstsq(design, v, rcond=None)[0]

    rho, _ = stats.pearsonr(resid(r[x].to_numpy()), resid(r[y].to_numpy()))
    dof = len(d) - 2 - len(controls)
    t = rho * np.sqrt(dof / max(1 - rho ** 2, 1e-12))
    return {"n": len(d), "rho": float(rho), "p_value": float(2 * stats.t.sf(abs(t), dof))}


# =============================================================================
# MLflow: experiment tracking and model registry
# =============================================================================
# Local SQLite store (mlflow/mlflow.db) so the model registry works offline; artifacts go to
# mlflow/mlartifacts. Every run carries the pipeline version, the data version (manifest hash), the
# git commit and the seed, which is what makes any number traceable and reproducible.
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def mlflow_setup(cfg: Config) -> str:
    """Points MLflow at the project store and creates the experiment if it does not exist."""
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        db = PROJECT_ROOT / uri.removeprefix("sqlite:///")
        db.parent.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{db}"
    mlflow.set_tracking_uri(uri)
    name = cfg["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        art = (PROJECT_ROOT / cfg["mlflow"]["artifact_location"]).as_uri()
        mlflow.create_experiment(name, artifact_location=art)
    mlflow.set_experiment(name)
    return uri


def base_tags(cfg: Config) -> dict[str, str]:
    """Traceability tags of every run."""
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
def mlflow_run(cfg: Config, name: str, tags: dict | None = None):
    with mlflow.start_run(run_name=name, tags={**base_tags(cfg), **(tags or {})}) as r:
        yield r


def log_cv_result(res, extra_params: dict | None = None) -> None:
    """Logs the per-fold metrics (as a step series), their mean/SD and the out-of-fold predictions."""
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
    """Points a copied MLflow store at this folder. Returns the number of cells updated.

    MLflow stores artifact locations as absolute URIs, so a store moved to another folder or machine
    cannot find its models. This rewrites the recorded project root (read from the experiment's
    artifact location) to the current root in every text column of the SQLite store. Run it once after
    cloning the repository or extracting the archive.
    """
    import sqlite3

    db = PROJECT_ROOT / cfg["mlflow"]["tracking_uri"].removeprefix("sqlite:///")
    if not db.exists():  # no store yet (nothing to relocate)
        return 0
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
    """Logs a fitted pipeline with its signature; optionally registers it with an alias."""
    example = X.head(5)
    output = model.predict_proba(example)[:, 1] if task == "classification" else model.predict(example)
    sig = infer_signature(example, output)
    here = Path(__file__).resolve().parent
    info = mlflow.sklearn.log_model(
        model, name="model", signature=sig, input_example=example,
        # The project code travels with the model, so it loads on any machine with just the dependencies.
        code_paths=[str(here / "data_prep.py"), str(here / "train.py")],
        # The pipeline holds project transformers (PositionMedianImputer) that skops cannot serialise;
        # cloudpickle is safe here because only the project's own code is loaded.
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


# =============================================================================
# 4 Translation: Combine drill -> game -> on-field
# =============================================================================
TIME_DRILLS = ["ten_yd_split", "forty", "three_cone", "short_shuttle"]
TRANSLATION_X = ["ten_yd_split", "forty", "vertical", "broad_jump", "three_cone", "short_shuttle",
                 "bench_reps", "combine_weight", "ngs_athleticism_score"]
TRANSLATION_Y = [f"{m}_eb" for m in ["burst_2s", "get_off_1s", "peak_accel_2s", "top_speed", "max_decel", "max_cut_deg"]]
ONFIELD_Y = [f"{m}_eb" for m in ONFIELD_METRICS]


def _icc(df: pd.DataFrame, key: str, value: str) -> float:
    """Method-of-moments ICC of `value` grouped by `key`."""
    g = df.groupby(key)[value]
    n, mu, var = g.size(), g.mean(), g.var(ddof=1)
    if len(n) < 3:
        return np.nan
    sigma2 = np.average(var.fillna(var.median()), weights=np.maximum(n - 1, 1))
    total = mu.var(ddof=1)
    return float((total - np.mean(sigma2 / n)) / total) if total > 0 else np.nan


def stage_translation(cfg: Config, d: dict) -> dict:
    """Translation matrix, reliability, defensive-line deep dive, estimator sensitivity and GLM."""
    prospects, profiles = d["prospects"], d["profiles"]
    df = profiles.merge(prospects, on=["nfl_id", "combine_position"], how="inner")
    oriented = df.copy()
    for c in TIME_DRILLS:  # orient so that a positive rho always means "better drill -> more movement"
        oriented[c] = -oriented[c]
    nb = cfg["cv"]["bootstrap_resamples"]
    matrix = within_group_spearman(oriented, TRANSLATION_X, TRANSLATION_Y + ["start_rate"], "combine_position", nb, cfg.seed)
    matrix["x_oriented"] = matrix["x"].isin(TIME_DRILLS)
    _save(matrix, cfg, "translation_matrix.csv", key="metrics")

    # SBI reliability: mean empirical-Bayes weight. It caps how much variance any pre-draft model can
    # explain (observed R^2 <= reliability x true R^2).
    rel = profiles.groupby("combine_position", observed=True)["sbi_reliability"].agg(["mean", "median", "min", "max", "count"])
    rel.loc["ALL"] = [profiles.sbi_reliability.mean(), profiles.sbi_reliability.median(),
                      profiles.sbi_reliability.min(), profiles.sbi_reliability.max(), len(profiles)]
    _save(rel.reset_index(names="combine_position"), cfg, "sbi_reliability_by_position.csv", key="metrics")

    # Which game metrics are stable player traits, by position? (ICC + one-way ANOVA)
    reliab = metric_reliability(d["plays"], profiles)
    _save(reliab, cfg, "metric_reliability.csv", key="metrics")

    # Defensive-line deep dive: raw vs role-adjusted translation (edge DE/OLB vs interior DT/NT).
    dl = oriented[oriented.combine_position == "DL"].copy()
    dl["dl_role"] = np.where(dl["nfl_position"].isin(["DT", "NT"]), "interior", "edge")
    dl_y = ["burst_2s_eb", "get_off_1s_eb", "peak_accel_2s_eb"]
    dl_tab = pd.concat([
        within_group_spearman(dl, TRANSLATION_X, dl_y, "combine_position", nb, cfg.seed).assign(adjustment="none"),
        within_group_spearman(dl, TRANSLATION_X, dl_y, "dl_role", nb, cfg.seed).assign(adjustment="edge vs interior role"),
    ], ignore_index=True)
    _save(dl_tab, cfg, "dl_translation_role_adjusted.csv", key="metrics")

    # Game movement vs start rate, within position group
    starter_assoc = within_group_spearman(df, ["sbi"] + [f"{m}_z" for m in METRICS if m != "burst_2s"],
                                          ["start_rate"], "combine_position", nb, cfg.seed)
    _save(starter_assoc, cfg, "movement_vs_start_rate.csv", key="metrics")

    # Sensitivity: the discarded 80th-percentile capacity vs the adopted shrunk mean
    gq = df.groupby("combine_position", observed=True)["burst_2s_q80"]
    sens_df = df.assign(sbi_q80=50 + 10 * (df["burst_2s_q80"] - gq.transform("mean")) / gq.transform("std"),
                        neg_log_pick=-df["log_draft_pick"])
    sens = within_group_spearman(sens_df, ["sbi", "sbi_q80"], ["n_plays", "start_rate", "neg_log_pick"],
                                 "combine_position", nb, cfg.seed)
    _save(sens, cfg, "sbi_estimator_sensitivity.csv", key="metrics")
    bias = estimator_bias(d["plays"], "burst_2s", [3, 5, 10], 200, cfg.seed)
    _save(bias, cfg, "estimator_bias_simulation.csv", key="metrics")

    # Exploratory (not pre-registered): SBI vs start rate where draft capital says little (day 3 + undrafted)
    day3 = df[df["draft_round_filled"] >= 4]
    day12 = df[df["draft_round_filled"] < 4]
    explo = pd.concat([within_group_spearman(day3, ["sbi"], ["start_rate"], "combine_position", nb, cfg.seed).assign(subgroup="rounds 4-7 + undrafted"),
                       within_group_spearman(day12, ["sbi"], ["start_rate"], "combine_position", nb, cfg.seed).assign(subgroup="rounds 1-3")],
                      ignore_index=True)
    explo["starters"] = [int(day3.is_starter.sum()), int(day12.is_starter.sum())]
    _save(explo, cfg, "exploratory_sbi_by_draft_tier.csv", key="metrics")

    # Starter rate by SBI band (95% Wilson CI): the coach-facing table
    bands = pd.cut(df["sbi"], [-np.inf, 45, 55, np.inf], labels=["Below 45", "45-55", "Above 55"])
    tab = df.groupby(bands, observed=True)["is_starter"].agg(["sum", "count"]).reset_index(names="sbi_band")
    z = 1.96
    ph = tab["sum"] / tab["count"]
    denom = 1 + z ** 2 / tab["count"]
    centre = (ph + z ** 2 / (2 * tab["count"])) / denom
    half = z * np.sqrt(ph * (1 - ph) / tab["count"] + z ** 2 / (4 * tab["count"] ** 2)) / denom
    tab["starter_rate"], tab["ci_low"], tab["ci_high"] = ph, centre - half, centre + half
    _save(tab, cfg, "starter_rate_by_sbi_band.csv", key="metrics")

    # Logistic GLM: does the SBI add anything to position + draft capital + Combine athleticism? (likelihood ratio)
    # `is_undrafted` is left out: no tracked undrafted player started (quasi-complete separation, the
    # maximum likelihood does not converge); log pick already places them at pick 260.
    g = df.assign(sbi10=df["sbi"] / 10, athl10=df["ngs_athleticism_score"] / 10, pos=df["combine_position"].astype(str))
    base = smf.logit("is_starter ~ C(pos) + log_draft_pick + athl10", data=g).fit(disp=0, maxiter=200)
    full = smf.logit("is_starter ~ C(pos) + log_draft_pick + athl10 + sbi10", data=g).fit(disp=0, maxiter=200)
    lr = 2 * (full.llf - base.llf)
    ci = np.exp(full.conf_int().loc["sbi10"])
    ci_a = np.exp(full.conf_int().loc["athl10"])
    glm = {
        "n": int(full.nobs), "events": int(g["is_starter"].sum()), "converged": bool(full.mle_retvals["converged"]),
        "or_per_10_sbi": float(np.exp(full.params["sbi10"])), "or_sbi_ci": [float(ci.iloc[0]), float(ci.iloc[1])],
        "p_sbi_wald": float(full.pvalues["sbi10"]),
        "or_per_10_ngs_athleticism": float(np.exp(full.params["athl10"])), "or_athl_ci": [float(ci_a.iloc[0]), float(ci_a.iloc[1])],
        "p_athl_wald": float(full.pvalues["athl10"]),
        "lr_stat": float(lr), "lr_p_value": float(stats.chi2.sf(lr, 1)),
        "aic_base": float(base.aic), "aic_full": float(full.aic),
        "pseudo_r2_base": float(base.prsquared), "pseudo_r2_full": float(full.prsquared),
    }
    _dump(glm, cfg, "glm_sbi_starter.json")
    (cfg.path("metrics") / "glm_sbi_starter_summary.txt").write_text(str(full.summary()))
    log.info("translation: GLM %s", glm)
    out = {"matrix": matrix, "starter_assoc": starter_assoc, "bands": tab, "glm": glm}
    if d.get("onfield") is not None:
        out["onfield"] = stage_onfield_translation(cfg, d)
    if has_combine_tracking(prospects):
        out["combine_tracking"] = stage_combine_tracking_translation(cfg, d)
    return out


def stage_combine_tracking_translation(cfg: Config, d: dict) -> dict:
    """Drill tracking only: tracking vs official results, and drill tracking vs the game.

    Separate output files, so the earlier matrices and their FDR q-values are untouched.
    """
    nb, seed = cfg["cv"]["bootstrap_resamples"], cfg.seed
    prospects = d["prospects"]
    feats = prospects[["nfl_id"] + CT_SET]
    _save(official_vs_tracking(feats, prospects), cfg, "ct_official_validity.csv", key="metrics")
    xs = list(CT_FEATURES)

    def orient(df: pd.DataFrame) -> pd.DataFrame:
        o = df.copy()
        for c in CT_LOWER_IS_BETTER:
            o[c] = -o[c]
        return o

    game = orient(d["profiles"].merge(prospects, on=["nfl_id", "combine_position"], how="inner"))
    mat = within_group_spearman(game, xs, TRANSLATION_Y + ["start_rate"], "combine_position", nb, seed)
    mat["x_oriented"] = mat["x"].isin(CT_LOWER_IS_BETTER)
    _save(mat, cfg, "ct_translation_matrix.csv", key="metrics")
    out = {"matrix": mat}
    if d.get("onfield") is not None:
        on = orient(d["onfield"].drop(columns=["combine_position"]).merge(prospects, on="nfl_id", how="inner"))
        om = within_group_spearman(on, xs, ONFIELD_Y, "combine_position", nb, seed)
        om["x_oriented"] = om["x"].isin(CT_LOWER_IS_BETTER)
        _save(om, cfg, "ct_onfield_translation_matrix.csv", key="metrics")
        out["onfield_matrix"] = om
        # Does a drill-tracking feature add anything to the official 40 for DL get-off? (rank partials)
        dl = on[on.combine_position.astype(str) == "DL"].assign(role=lambda t: _role(t["nfl_position"]).eq("interior").astype(int))
        rows = []
        for x in xs:
            for label, ctrl in (("none", []), ("weight + role", ["combine_weight", "role"]),
                                ("weight + role + official 40", ["combine_weight", "role", "forty"])):
                r = partial_spearman(dl, x, "goe_eb", ctrl)
                rows.append({"x": x, "y": "goe_eb", "controls": label, **r})
        _save(pd.DataFrame(rows), cfg, "ct_partial_dl_getoff.csv", key="metrics")
    return out


def stage_onfield_translation(cfg: Config, d: dict) -> dict:
    """Drill -> on-field performance, reliability, the DL get-off chain and SBI validity."""
    nb, seed = cfg["cv"]["bootstrap_resamples"], cfg.seed
    prospects, onfield, fam = d["prospects"], d["onfield"], d["fam"]
    df = onfield.drop(columns=["combine_position"]).merge(prospects, on="nfl_id", how="inner", validate="1:1")
    df["neg_log_pick"] = -df["log_draft_pick"]
    df["dl_role"] = _role(df["nfl_position"])
    oriented = df.copy()
    for c in TIME_DRILLS:
        oriented[c] = -oriented[c]

    # 1. Is each on-field metric a stable player trait?
    rel = reliability_table(fam, onfield)
    _save(rel, cfg, "onfield_reliability.csv", key="metrics")
    stab = rookie_stability(fam, cfg)
    _save(stab, cfg, "onfield_stability.csv", key="metrics")

    # 2. Drill -> on-field translation (within Combine position), with draft capital as the reference.
    xs = TRANSLATION_X + ["neg_log_pick"]
    mat = within_group_spearman(oriented, xs, ONFIELD_Y, "combine_position", nb, seed)
    mat["x_oriented"] = mat["x"].isin(TIME_DRILLS)
    _save(mat, cfg, "onfield_translation_matrix.csv", key="metrics")
    dl_role = within_group_spearman(oriented[oriented.combine_position == "DL"], TRANSLATION_X,
                                    ["goe_eb", "proe_eb"], "dl_role", nb, seed)
    _save(dl_role, cfg, "onfield_dl_role_check.csv", key="metrics")

    # 3. On-field performance vs start rate (does the metric capture what teams value?)
    assoc = within_group_spearman(df, ONFIELD_Y + ["perf_index"], ["start_rate"], "combine_position", nb, seed)
    _save(assoc, cfg, "onfield_vs_start_rate.csv", key="metrics")

    # 4. The DL chain: drill -> get-off over expected -> pressure over expected (bootstrap mediation)
    dl = oriented[oriented.combine_position == "DL"]
    med = pd.DataFrame([{"x": x, **mediation_bootstrap(dl, x, "goe_eb", "proe_eb", nb, seed)}
                        for x in ["ten_yd_split", "forty", "broad_jump", "vertical", "combine_weight"]])
    _save(med, cfg, "onfield_mediation_dl.csv", key="metrics")

    # 5. Convergent validity of the tracking metrics against season-long NGS get-off
    v = d["profiles"].merge(onfield[["nfl_id", "goe_eb", "proe_eb"]], on="nfl_id").query("combine_position == 'DL'")
    rows = []
    for x in ["sbi", "get_off_1s_z", "burst_1s_z", "peak_accel_2s_z"]:
        for y in ["goe_eb", "proe_eb"]:
            t = v[[x, y]].dropna()
            r, p = stats.spearmanr(t[x], t[y])
            rows.append({"tracking_metric": x, "onfield_metric": y, "n": len(t), "rho": r, "p_value": p})
    _save(pd.DataFrame(rows), cfg, "sbi_validity.csv", key="metrics")

    # 6. Is WR/DB snap burst a trait once the play context is removed?
    ctx = pd.read_parquet(cfg.path("interim") / "play_context.parquet")
    tp = d["plays"][d["plays"].nfl_id.isin(d["profiles"].nfl_id)]
    adj = context_adjusted_burst(tp, ctx)
    cr = pd.DataFrame([{"combine_position": pos, "players": g.nfl_id.nunique(), "plays": len(g),
                        "icc_raw": _icc(g, "nfl_id", "burst_2s"), "icc_context_adjusted": _icc(g, "nfl_id", "burst_2s_ctx"),
                        "context_matched_share": float(g["lined_up_position"].notna().mean())}
                       for pos, g in adj.groupby(adj.combine_position.astype(str))])
    _save(cr, cfg, "tracking_context_reliability.csv", key="metrics")

    # 7. Logistic GLM: does on-field performance add anything to position + draft capital? (diagnostic, post-draft)
    sub = df.dropna(subset=["perf_index"]).assign(pos=lambda x: x["combine_position"].astype(str))
    base = smf.logit("is_starter ~ C(pos) + log_draft_pick", data=sub).fit(disp=0, maxiter=200)
    full = smf.logit("is_starter ~ C(pos) + log_draft_pick + perf_index", data=sub).fit(disp=0, maxiter=200)
    lr = 2 * (full.llf - base.llf)
    ci = np.exp(full.conf_int().loc["perf_index"])
    glm = {"n": int(full.nobs), "events": int(sub.is_starter.sum()), "converged": bool(full.mle_retvals["converged"]),
           "or_per_sd_perf": float(np.exp(full.params["perf_index"])), "or_ci": [float(ci.iloc[0]), float(ci.iloc[1])],
           "p_wald": float(full.pvalues["perf_index"]), "lr_stat": float(lr), "lr_p_value": float(stats.chi2.sf(lr, 1)),
           "aic_base": float(base.aic), "aic_full": float(full.aic),
           "pseudo_r2_base": float(base.prsquared), "pseudo_r2_full": float(full.prsquared)}
    _dump(glm, cfg, "glm_onfield_starter.json")
    log.info("on-field translation: GLM %s", glm)
    return {"reliability": rel, "matrix": mat, "mediation": med, "glm": glm}


# =============================================================================
# 5 Regression: Combine -> SBI and Combine -> on-field performance
# =============================================================================
REG_MODELS = ["position_mean", "ridge", "random_forest", "xgboost"]


def _summarise(results: dict, metrics: list[str], cfg: Config, task: str):
    """Summary table (mean, SD, pooled R^2 with bootstrap CI, most-selected hyperparameters) and per-fold scores."""
    rows, scores = [], {m: {} for m in metrics}
    for name, res in results.items():
        row = {"model": name}
        for m in metrics:
            v = res.fold_scores[m]
            row[f"{m}_mean"], row[f"{m}_std"] = v.mean(), v.std()
            scores[m][name] = v.to_numpy()
        oof = res.oof.groupby("row").agg(y=("y", "first"), pred=("pred", "mean"))
        for m, fn in [("rmse", lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))),
                      ("r2", lambda a, b: 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))]:
            if m in metrics:
                est, lo, hi = bootstrap_ci(oof.y.to_numpy(), oof.pred.to_numpy(), fn, cfg["cv"]["bootstrap_resamples"], cfg.seed)
                row[f"oof_{m}"], row[f"oof_{m}_ci_low"], row[f"oof_{m}_ci_high"] = est, lo, hi
        params = pd.DataFrame(res.best_params)
        row["selected_params_mode"] = json.dumps({c: params[c].mode().iloc[0] for c in params.columns}, default=str) if len(params.columns) else "{}"
        rows.append(row)
    return pd.DataFrame(rows), {m: pd.DataFrame(v) for m, v in scores.items()}


def _regress(cfg: Config, pop: str, df: pd.DataFrame, target: str, strata: pd.Series, reg_name: str,
             gap_cols: list[str], extra_tags: dict | None = None, features: list[str] | None = None,
             feature_set: str = "athletic", alias: str = "champion") -> dict:
    """Nested CV of the four regressors on one population, tests, final refit and registration."""
    grids = cfg["regression"]["grids"]
    df = df.reset_index(drop=True)
    strata = strata.reset_index(drop=True)
    X, y = xy(df, features or ATHLETIC, target)
    results = {}
    for name in REG_MODELS:
        t0 = time.time()
        res = nested_cv(name, feature_set, "regression", X, y, strata, grids.get(name, {}), cfg, cfg.seed)
        with mlflow_run(cfg, f"reg-{pop}-{name}", {"stage": "cv", "task": "regression", "target": target,
                                                   "population": pop, **(extra_tags or {})}):
            log_cv_result(res, {"grid": json.dumps(grids.get(name, {})), "population": pop, "n_rows": len(df)})
        results[name] = res
        log.info("regression %s/%s: %s (%.0fs)", pop, name, res.fold_scores[["mae", "rmse", "r2"]].mean().round(3).to_dict(), time.time() - t0)

    summary, scores = _summarise(results, ["mae", "rmse", "r2"], cfg, task="regression")
    summary.insert(0, "population", pop)
    summary.insert(1, "target", target)
    summary.insert(2, "n", len(df))
    any_res = next(iter(results.values()))
    tests = pd.concat([pairwise_model_tests(scores[m], m, any_res.n_train, any_res.n_test, higher_is_better=(m == "r2"))
                       for m in ["rmse", "r2"]], ignore_index=True)
    tests.insert(0, "population", pop)
    fried = {m: friedman(scores[m], higher_is_better=(m == "r2")) for m in ["rmse", "r2"]}
    for m, sc in scores.items():
        _save(sc.reset_index(names="fold"), cfg, f"regression_fold_{pop}_{m}.csv", key="metrics")

    # Champion = lowest mean RMSE among the learned models; refitted on all rows and registered.
    champ = summary[summary.model != "position_mean"].sort_values("rmse_mean").iloc[0]["model"]
    est, best, score = refit_final(champ, "regression", X, y, grids.get(champ, {}), cfg, cfg.seed)
    with mlflow_run(cfg, f"reg-final-{pop}-{champ}", {"stage": "final", "task": "regression", "population": pop}):
        mlflow.log_params({"model": champ, "population": pop, "target": target,
                           **{k.replace("model__", ""): v for k, v in best.items()}})
        mlflow.log_metric("refit_inner_cv_rmse", -score)
        log_and_register(est, X, "regression", reg_name if cfg["mlflow"]["register_models"] else None, alias)

    # Translation gap: observed minus Combine-predicted (out-of-fold, mean over repeats).
    oof = results[champ].oof.groupby("row")["pred"].mean()
    gap = df[[c for c in dict.fromkeys(gap_cols + [target]) if c in df.columns]].copy()
    gap["predicted"] = oof.reindex(gap.index).to_numpy()
    gap["translation_gap"] = gap[target] - gap["predicted"]
    gap["population"] = pop

    pi = permutation_importance(est, X, y, scoring="neg_root_mean_squared_error", n_repeats=20, random_state=cfg.seed)
    imp = pd.DataFrame({"population": pop, "feature": X.columns, "importance": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("importance", ascending=False)
    return {"summary": summary, "tests": tests, "friedman": fried, "gap": gap, "importance": imp,
            "scores": scores, "n_train": any_res.n_train, "n_test": any_res.n_test, "feature_set": feature_set,
            "champion": {"model": champ, "params": best, "inner_rmse": -score, "target": target,
                         "registered_as": reg_name,
                         **({} if feature_set == "athletic" else {"feature_set": feature_set, "alias": alias})}}


# Snap-burst populations: name -> (Combine position or None for all, registry suffix).
# With full-season tracking, snap burst is a stable trait on the DL (ICC 0.63) and at WR (0.62).
SBI_POPULATIONS = {"all_positions": (None, ""), "defensive_line": ("DL", "-dl"), "wide_receivers": ("WR", "-wr")}


def _sbi_population(full: pd.DataFrame, pop: str, k: int) -> tuple[pd.DataFrame, pd.Series, str]:
    """Rows, CV strata and registry suffix of one SBI population."""
    pos, suffix = SBI_POPULATIONS[pop]
    if pos is None:
        return full, full["combine_position"].astype(str), suffix
    df = full[full.combine_position.astype(str) == pos]
    strata = df["dl_role"] if pos == "DL" else _strata(_role(df["nfl_position"]), k)
    return df, strata, suffix


def stage_regression(cfg: Config, d: dict) -> dict:
    """Combine -> Snap Burst Index and Combine -> on-field performance.

    SBI: all positions, defensive line only and wide receivers only (where snap burst is a reliable trait).
    On-field: one task per family with a measurable outcome (configuration: regression.onfield_tasks).
    """
    target = cfg["regression"]["target"]
    k = cfg["cv"]["outer_splits"]
    gap_cols = ["nfl_id", "display_name", "combine_position", "nfl_position", "draft_year", "draft_round",
                "start_rate", "is_starter"]
    runs = []
    full = regression_table(d["prospects"], d["profiles"], target)
    full["dl_role"] = _role(full["nfl_position"])
    for pop in cfg["regression"]["populations"]:
        df, strata, suffix = _sbi_population(full, pop, k)
        reg_name = cfg["mlflow"]["regressor_name"] + suffix
        runs.append(_regress(cfg, pop, df, target, strata, reg_name,
                             gap_cols + ["dl_role", "burst_2s_eb", "sbi_reliability", "n_plays"], {"family": "sbi"}))

    if d.get("onfield") is not None:
        for task in cfg["regression"].get("onfield_tasks", []):
            df = onfield_regression_table(d["prospects"], d["onfield"], task["target"], task["positions"])
            strata = _strata(_role(df["nfl_position"]), k)
            reg_name = f"{cfg['mlflow']['onfield_regressor_prefix']}-{task['name'].replace('_', '-')}"
            metric = task["target"].split("_")[0]
            runs.append(_regress(cfg, task["name"], df, task["target"], strata, reg_name,
                                 gap_cols + [f"{metric}_rel", f"n_{metric}"], {"family": "onfield"}))

    # Athletic vs athletic + Combine drill tracking, same rows and folds (only with the combine_tracking file)
    ct_runs, ct_rows = [], []
    if cfg["regression"].get("combine_tracking_comparison", True) and has_combine_tracking(d["prospects"]):
        for base in list(runs):
            pop = base["summary"].population.iloc[0]
            if pop == "all_positions":
                continue  # the comparison runs where snap burst is a trait (single-position populations)
            if pop in SBI_POPULATIONS:
                df, strata, _ = _sbi_population(full, pop, k)
                tgt = target
            else:
                task = next(t for t in cfg["regression"]["onfield_tasks"] if t["name"] == pop)
                df = onfield_regression_table(d["prospects"], d["onfield"], task["target"], task["positions"])
                strata, tgt = _strata(_role(df["nfl_position"]), k), task["target"]
            reg_name = base["champion"]["registered_as"] + "-ct"
            r = _regress(cfg, f"{pop}_ct", df, tgt, strata, reg_name, gap_cols, {"family": "combine_tracking"},
                         features=ATHLETIC_CT, feature_set="athletic_ct", alias="challenger")
            ct_runs.append(r)
            for model in REG_MODELS[1:]:
                for m, hib in (("rmse", False), ("r2", True)):
                    t = pairwise_model_tests(pd.DataFrame({"athletic_ct": r["scores"][m][model],
                                                           "athletic": base["scores"][m][model]}),
                                             m, r["n_train"], r["n_test"], higher_is_better=hib)
                    ct_rows.append(t.assign(population=pop, model=model))
        if ct_rows:
            _save(pd.concat(ct_rows, ignore_index=True), cfg, "regression_ct_comparison.csv", key="metrics")
    gap_runs = runs
    runs = runs + ct_runs

    summary = pd.concat([r["summary"] for r in runs], ignore_index=True)
    _save(summary, cfg, "regression_summary.csv", key="metrics")
    _save(pd.concat([r["tests"] for r in runs], ignore_index=True), cfg, "regression_pairwise_tests.csv", key="metrics")
    _dump({r["summary"].population.iloc[0]: r["friedman"] for r in runs}, cfg, "regression_friedman.json")
    _dump({r["summary"].population.iloc[0]: r["champion"] for r in runs}, cfg, "regression_final_models.json")
    sbi_pops = set(cfg["regression"]["populations"])
    gaps = [r["gap"] for r in gap_runs]
    sbi_gap = pd.concat([g for g in gaps if g.population.iloc[0] in sbi_pops], ignore_index=True)
    sbi_gap = sbi_gap.rename(columns={"predicted": "predicted_sbi"}).sort_values(["population", "translation_gap"], ascending=[True, False])
    _save(sbi_gap, cfg, "translation_gap.csv", key="metrics")
    on_gaps = [g for g in gaps if g.population.iloc[0] not in sbi_pops]
    if on_gaps:
        _save(pd.concat(on_gaps, ignore_index=True).sort_values(["population", "translation_gap"], ascending=[True, False]),
              cfg, "onfield_translation_gap.csv", key="metrics")
    _save(pd.concat([r["importance"] for r in runs], ignore_index=True), cfg, "regression_permutation_importance.csv", key="metrics")
    return {"summary": summary}


# =============================================================================
# 6 Classification: who becomes a starter?
# =============================================================================
CLF_MODELS = ["logistic_regression", "random_forest", "xgboost"]


def _classification_experiments(d: dict) -> dict:
    """name -> (table, feature sets, feature-set comparisons (new, reference))."""
    exps = {}
    if d.get("onfield") is not None:
        exps["all_prospects"] = (projector_table(d["prospects"], d["onfield"]),
                                 ["combine", "combine_draft", "combine_draft_proj"],
                                 [("combine_draft", "combine"), ("combine_draft_proj", "combine_draft")])
    else:
        exps["all_prospects"] = (classification_table(d["prospects"]), ["combine", "combine_draft"],
                                 [("combine_draft", "combine")])
    if has_combine_tracking(d["prospects"]):  # pre-draft
        exps["all_prospects"][1].append("combine_draft_ct")
        exps["all_prospects"][2].append(("combine_draft_ct", "combine_draft"))
    exps["tracked_subset"] = (classification_table(d["prospects"], d["profiles"]),
                              ["combine_draft", "combine_draft_mov"], [("combine_draft_mov", "combine_draft")])
    if d.get("onfield") is not None:
        exps["onfield_subset"] = (onfield_classification_table(d["prospects"], d["onfield"]),
                                  ["combine_draft", "combine_draft_perf"], [("combine_draft_perf", "combine_draft")])
    return exps


def stage_classification(cfg: Config, d: dict) -> dict:
    """Nested CV of the three classifiers x feature sets, tests, DeLong and registration."""
    target = cfg["classification"]["target"]
    grids = cfg["classification"]["grids"]
    thr = cfg["classification"]["decision_threshold"]
    experiments = _classification_experiments(d)
    results, summaries, tests_all, fried_all, delong_rows = {}, [], [], {}, []
    for exp, (df, fsets, comparisons) in experiments.items():
        df = df.reset_index(drop=True)
        for fs in fsets:
            X, y = xy(df, FEATURE_SETS[fs], target)
            for name in CLF_MODELS:
                t0 = time.time()
                res = nested_cv(name, fs, "classification", X, y, y, grids[name], cfg, cfg.seed, thr)
                with mlflow_run(cfg, f"clf-{exp}-{fs}-{name}", {"stage": "cv", "task": "classification", "population": exp}):
                    log_cv_result(res, {"population": exp, "n_rows": len(df), "positives": int(y.sum())})
                results[(exp, fs, name)] = res
                log.info("classification %s/%s/%s AUC=%.3f (%.0fs)", exp, fs, name, res.fold_scores["roc_auc"].mean(), time.time() - t0)

        sub = {k: v for k, v in results.items() if k[0] == exp}
        for (e, fs, name), res in sub.items():
            fsum = res.fold_scores.drop(columns=["fold", "repeat"]).agg(["mean", "std"])
            row = {"population": e, "feature_set": fs, "model": name, "n": len(df), "positives": int(df[target].sum())}
            oof = res.oof.groupby("row").agg(y=("y", "first"), pred=("pred", "mean"))
            auc, lo, hi = bootstrap_ci(oof.y.to_numpy(), oof.pred.to_numpy(), roc_auc_score, cfg["cv"]["bootstrap_resamples"], cfg.seed, stratify=True)
            for m in fsum.columns:
                row[f"{m}_mean"], row[f"{m}_std"] = fsum.loc["mean", m], fsum.loc["std", m]
            row.update({"oof_auc": auc, "oof_auc_ci_low": lo, "oof_auc_ci_high": hi})
            summaries.append(row)
        any_res = next(iter(sub.values()))
        for fs in fsets:
            sc = pd.DataFrame({name: results[(exp, fs, name)].fold_scores["roc_auc"].to_numpy() for name in CLF_MODELS})
            t = pairwise_model_tests(sc, "roc_auc", any_res.n_train, any_res.n_test)
            t.insert(0, "comparison", "model_vs_model")
            t.insert(1, "population", exp)
            t.insert(2, "feature_set", fs)
            tests_all.append(t)
            fried_all[f"{exp}/{fs}"] = friedman(sc)
        for fs_b, fs_a in comparisons:
            for name in CLF_MODELS:
                a = results[(exp, fs_b, name)].fold_scores["roc_auc"].to_numpy()
                b = results[(exp, fs_a, name)].fold_scores["roc_auc"].to_numpy()
                t = pairwise_model_tests(pd.DataFrame({fs_b: a, fs_a: b}), "roc_auc", any_res.n_train, any_res.n_test)
                t.insert(0, "comparison", "feature_set_vs_feature_set")
                t.insert(1, "population", exp)
                t.insert(2, "feature_set", f"{fs_b} vs {fs_a} ({name})")
                tests_all.append(t)
                oa = results[(exp, fs_b, name)].oof.groupby("row").agg(y=("y", "first"), p=("pred", "mean"))
                ob = results[(exp, fs_a, name)].oof.groupby("row").agg(p=("pred", "mean"))
                dl = delong_test(oa.y.to_numpy(), oa.p.to_numpy(), ob.p.reindex(oa.index).to_numpy())
                delong_rows.append({"population": exp, "model": name, "feature_set_1": fs_b, "feature_set_2": fs_a, **dl})

    summary = pd.DataFrame(summaries)
    tests = pd.concat(tests_all, ignore_index=True)
    delong = pd.DataFrame(delong_rows)
    _save(summary, cfg, "classification_summary.csv", key="metrics")
    _save(tests, cfg, "classification_pairwise_tests.csv", key="metrics")
    _save(delong, cfg, "classification_delong.csv", key="metrics")
    _dump(fried_all, cfg, "classification_friedman.json")
    fold_long = pd.concat([r.fold_scores.assign(population=k[0], feature_set=k[1], model=k[2]) for k, r in results.items()], ignore_index=True)
    _save(fold_long, cfg, "classification_fold_scores.csv", key="metrics")
    oof_long = pd.concat([r.oof.assign(population=k[0], feature_set=k[1], model=k[2]) for k, r in results.items()], ignore_index=True)
    _save(oof_long, cfg, "classification_oof.parquet", key="metrics")

    # Champion (pre-draft): a candidate set (projector or drill tracking) replaces Combine + draft only if it
    # wins on mean AUC AND its DeLong test against Combine + draft (same model) gives p < 0.05.
    champ_fs, champ_auc = "combine_draft", -np.inf
    sb = summary[(summary.population == "all_prospects") & (summary.feature_set == "combine_draft")].sort_values("roc_auc_mean").iloc[-1]
    for cand in ("combine_draft_proj", "combine_draft_ct"):
        if cand not in experiments["all_prospects"][1]:
            continue
        sp = summary[(summary.population == "all_prospects") & (summary.feature_set == cand)].sort_values("roc_auc_mean").iloc[-1]
        dlr = delong[(delong.population == "all_prospects") & (delong.feature_set_1 == cand) & (delong.model == sp.model)]
        if sp.roc_auc_mean > sb.roc_auc_mean and len(dlr) and dlr.p_value.iloc[0] < 0.05 and sp.roc_auc_mean > champ_auc:
            champ_fs, champ_auc = cand, sp.roc_auc_mean
    registry = [("all_prospects", champ_fs, "champion", cfg["mlflow"]["classifier_name"]),
                ("tracked_subset", "combine_draft_mov", "challenger", cfg["mlflow"]["classifier_name"] + "-movement")]
    if "combine_draft_ct" in experiments["all_prospects"][1] and champ_fs != "combine_draft_ct":
        registry.append(("all_prospects", "combine_draft_ct", "challenger", cfg["mlflow"]["classifier_name"] + "-combine-tracking"))
    if "onfield_subset" in experiments:
        registry.append(("onfield_subset", "combine_draft_perf", "challenger", cfg["mlflow"]["classifier_name"] + "-onfield"))
    finals = {}
    for exp, fs, alias, reg in registry:
        s = summary[(summary.population == exp) & (summary.feature_set == fs)].sort_values("roc_auc_mean", ascending=False)
        best_model = s.iloc[0]["model"]
        df = experiments[exp][0]
        X, y = xy(df.reset_index(drop=True), FEATURE_SETS[fs], target)
        est, best, score = refit_final(best_model, "classification", X, y, grids[best_model], cfg, cfg.seed)
        with mlflow_run(cfg, f"clf-final-{exp}-{fs}-{best_model}", {"stage": "final", "task": "classification", "alias": alias}):
            mlflow.log_params({"model": best_model, "population": exp, "feature_set": fs,
                               **{k.replace("model__", ""): v for k, v in best.items()}})
            mlflow.log_metric("refit_inner_cv_auc", score)
            log_and_register(est, X, "classification", reg if cfg["mlflow"]["register_models"] else None, alias)
        pi = permutation_importance(est, X, y, scoring="roc_auc", n_repeats=20, random_state=cfg.seed)
        imp = pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean, "std": pi.importances_std}).sort_values("importance", ascending=False)
        _save(imp, cfg, f"classification_importance_{exp}_{fs}.csv", key="metrics")
        finals[f"{alias}:{reg}"] = {"population": exp, "feature_set": fs, "model": best_model, "params": best,
                                    "inner_auc": score, "registered_as": f"{reg}@{alias}"}
    _dump(finals, cfg, "classification_final_models.json")
    return {"summary": summary, "tests": tests, "delong": delong, "finals": finals}


# =============================================================================
# 7 Statistics without retraining
# =============================================================================
def ct_best_vs_best(cfg: Config) -> pd.DataFrame | None:
    """Does drill tracking improve the best pipeline? Best model with vs without, on the same folds.

    Regression: champion = lowest mean RMSE among the learned models of each arm (as the registry picks
    it); paired corrected resampled t-test on the outer-fold R² and RMSE (fold sizes n(k-1)/k and n/k).
    Classification (all prospects): best combine_draft_ct vs best combine_draft by mean AUC; corrected t
    on the per-fold AUCs and DeLong on the out-of-fold predictions. Each arm picks the best of the same
    candidates, so selection favours neither side.
    """
    m = cfg.path("metrics")
    if not (m / "regression_ct_comparison.csv").exists():
        return None
    k = cfg["cv"]["outer_splits"]
    reg = pd.read_csv(m / "regression_summary.csv")
    rows = []
    for pop in [p for p in reg.population.unique() if f"{p}_ct" in set(reg.population)]:
        arms = {}
        for arm, name in (("without", pop), ("with", f"{pop}_ct")):
            s = reg[(reg.population == name) & (reg.model != "position_mean")].sort_values("rmse_mean").iloc[0]
            arms[arm] = (s.model, {mt: pd.read_csv(m / f"regression_fold_{name}_{mt}.csv")[s.model].to_numpy() for mt in ("r2", "rmse")})
        n = int(reg[reg.population == pop].n.iloc[0])
        n_test = round(n / k)
        row = {"task": pop, "kind": "regression", "n": n, "model_without": arms["without"][0], "model_with": arms["with"][0]}
        for mt, hib in (("r2", True), ("rmse", False)):
            a, b = arms["with"][1][mt], arms["without"][1][mt]
            t = pairwise_model_tests(pd.DataFrame({"with": a, "without": b}), mt, n - n_test, n_test, higher_is_better=hib).iloc[0]
            se = abs(t.mean_diff_a_minus_b / t.t_corrected) if t.t_corrected else np.nan
            half = se * stats.t.ppf(0.975, len(a) - 1)
            row[f"{mt}_folds_better"] = int(((a > b) if hib else (a < b)).sum())
            row.update({f"{mt}_without": b.mean(), f"{mt}_with": a.mean(), f"{mt}_diff": t.mean_diff_a_minus_b,
                        f"{mt}_diff_ci_low": t.mean_diff_a_minus_b - half, f"{mt}_diff_ci_high": t.mean_diff_a_minus_b + half,
                        f"{mt}_p_corrected_t": t.p_corrected_t, f"{mt}_p_wilcoxon": t.p_wilcoxon})
        rows.append(row)
    clf_path = m / "classification_summary.csv"
    if clf_path.exists():
        clf = pd.read_csv(clf_path)
        a_ = clf[(clf.population == "all_prospects")]
        if "combine_draft_ct" in set(a_.feature_set):
            folds = pd.read_csv(m / "classification_fold_scores.csv")
            oof = pd.read_parquet(m / "classification_oof.parquet")
            best = {fs: a_[a_.feature_set == fs].sort_values("roc_auc_mean").iloc[-1].model for fs in ("combine_draft", "combine_draft_ct")}

            def fold(fs):
                return folds[(folds.population == "all_prospects") & (folds.feature_set == fs) & (folds.model == best[fs])].roc_auc.to_numpy()

            def oofp(fs):
                o = oof[(oof.population == "all_prospects") & (oof.feature_set == fs) & (oof.model == best[fs])]
                return o.groupby("row").agg(y=("y", "first"), p=("pred", "mean"))

            n = int(a_.n.iloc[0])
            n_test = round(n / k)
            a, b = fold("combine_draft_ct"), fold("combine_draft")
            t = pairwise_model_tests(pd.DataFrame({"with": a, "without": b}), "roc_auc", n - n_test, n_test).iloc[0]
            se = abs(t.mean_diff_a_minus_b / t.t_corrected) if t.t_corrected else np.nan
            half = se * stats.t.ppf(0.975, len(a) - 1)
            ow, ob = oofp("combine_draft_ct"), oofp("combine_draft")
            dl = delong_test(ow.y.to_numpy(), ow.p.to_numpy(), ob.p.reindex(ow.index).to_numpy())
            rows.append({"task": "starter_all_prospects", "kind": "classification", "n": n,
                         "model_without": best["combine_draft"], "model_with": best["combine_draft_ct"],
                         "roc_auc_folds_better": int((a > b).sum()),
                         "roc_auc_without": b.mean(), "roc_auc_with": a.mean(), "roc_auc_diff": t.mean_diff_a_minus_b,
                         "roc_auc_diff_ci_low": t.mean_diff_a_minus_b - half, "roc_auc_diff_ci_high": t.mean_diff_a_minus_b + half,
                         "roc_auc_p_corrected_t": t.p_corrected_t, "roc_auc_p_wilcoxon": t.p_wilcoxon,
                         "oof_auc_without": dl["auc_2"], "oof_auc_with": dl["auc_1"], "p_delong": dl["p_value"]})
    out = pd.DataFrame(rows)
    _save(out, cfg, "ct_best_vs_best.csv", key="metrics")
    return out


def stage_stats(cfg: Config) -> None:
    """Recomputes the rank-based omnibus tests from the saved per-fold scores.

    Lets a statistics fix be applied to a release run without retraining: the per-fold scores are
    outputs of that same run, so the tests stay tied to it.
    """
    m = cfg.path("metrics")
    reg = pd.read_csv(m / "regression_summary.csv")
    fried = {}
    for pop in reg.population.unique():
        fried[pop] = {}
        for metric in ["rmse", "r2"]:
            f = pd.read_csv(m / f"regression_fold_{pop}_{metric}.csv").drop(columns=["fold"])
            fried[pop][metric] = friedman(f, higher_is_better=(metric == "r2"))
    _dump(fried, cfg, "regression_friedman.json")
    folds = pd.read_csv(m / "classification_fold_scores.csv")
    cf = {}
    for (pop, fs), g in folds.groupby(["population", "feature_set"], sort=False):
        wide = g.pivot(index="fold", columns="model", values="roc_auc")
        cf[f"{pop}/{fs}"] = friedman(wide, higher_is_better=True)
    _dump(cf, cfg, "classification_friedman.json")
    if ct_best_vs_best(cfg) is not None:
        log.info("stats: drill-tracking best-vs-best comparison written")
    log.info("stats: Friedman tests recomputed from the per-fold scores")


# =============================================================================
# 8 Report figures (PNG, 200 dpi) in reports/figures
# =============================================================================
# Style rules: one quiet grey for the data and one blue accent for what the title is about; diverging
# maps go from blue to orange around a neutral grey; thin, recessive grid; every title states the
# finding with its number, computed from the data rather than typed.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ACCENT, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED = "#b4b2ab"
SURFACE = "#fcfcfb"
POS_ORDER = ["DB", "DL", "OL", "TE", "WR"]
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_orange", [ORANGE, "#f6c2ab", "#efeeea", "#9ec5f4", ACCENT])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#eef4fc", "#9ec5f4", "#3987e5", "#184f95"])

LABELS = {
    "ten_yd_split": "10-yd split", "forty": "40-yd dash", "vertical": "Vertical jump", "broad_jump": "Broad jump",
    "three_cone": "3-cone", "short_shuttle": "Short shuttle", "bench_reps": "Bench reps",
    "combine_weight": "Body weight", "ngs_athleticism_score": "NGS athleticism score",
    "burst_2s_eb": "Snap burst\n(2-s distance)", "get_off_1s_eb": "Get-off\n(speed at 1 s)",
    "peak_accel_2s_eb": "Peak accel\n(first 2 s)", "top_speed_eb": "Top speed", "max_decel_eb": "Max braking",
    "max_cut_deg_eb": "Sharpest cut", "start_rate": "Start rate",
    "position_mean": "Position mean\n(baseline)", "ridge": "Ridge", "random_forest": "Random forest", "xgboost": "XGBoost",
    "logistic_regression": "Logistic reg.",
    "ten_split_momentum": "10-yd momentum (weight / split)", "speed_score": "Speed Score",
    "ten_split_share": "10-yd share of the 40",
    "ngs_college_production_score": "NGS production score", "age_at_draft": "Age at draft", "combine_position": "Position group",
    "log_draft_pick": "Draft pick (log)", "is_undrafted": "Undrafted", "wing_span": "Wingspan", "hand_size": "Hand size",
    "arm_length": "Arm length", "combine_height": "Height", "power_conference": "Power conference", "n_drills_run": "Drills run",
    "sbi": "SBI", "top_speed_z": "Top speed (game)", "max_cut_deg_z": "Sharpest cut (game)",
    "burst_1s_z": "Distance at 1 s (game)", "get_off_1s_z": "Get-off (game)", "peak_accel_2s_z": "Peak accel (game)",
    "max_decel_z": "Max braking (game)",
    "goe_eb": "Get-off over\nexpected (DL)", "proe_eb": "Pressure rate\nover exp. (DL)",
    "ppoe_eb": "Pressure prob.\nover exp. (OL)", "paoe_eb": "Pressures allowed\nover exp. (OL)",
    "soe_eb": "Separation\nover exp. (WR/TE)", "neg_log_pick": "Draft capital (earlier pick)",
    "goe_z": "Get-off over exp. (game)", "proe_z": "Pressure over exp. (game)", "ppoe_z": "Protection over exp. (game)",
    "soe_z": "Separation over exp. (game)", "proj_goe": "Projected get-off", "proj_proe": "Projected pressure",
    "proj_ppoe": "Projected protection", "proj_soe": "Projected separation",
    "onfield_target_goe": "Projected get-off", "onfield_target_proe": "Projected pressure",
    "onfield_target_ppoe": "Projected protection", "onfield_target_soe": "Projected separation",
    "combine_draft_proj": "Combine + draft\n+ projected on-field", "combine_draft_perf": "Combine + draft\n+ on-field performance",
    "combine": "Combine only", "combine_draft": "Combine + draft", "combine_draft_mov": "Combine + draft\n+ game movement",
    "combine_draft_ct": "Combine + draft\n+ drill tracking",
    "ct_forty_split_5_10_s": "40: 5-10 yd split (tracking)", "ct_forty_top_speed": "40: top speed (tracking)",
    "ct_shuttle_plant_speed_ratio": "Shuttle: speed kept in turns", "ct_skill_speed_z": "Position drills: top speed",
    "ct_skill_accel_z": "Position drills: acceleration",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlelocation": "left", "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "legend.frameon": False,
    })


def _fit(fig, t, x0: float = 0.012, x1: float = 0.99) -> None:
    """Shrinks the font size of a figure text until it fits between x0 and x1 (fraction of the width)."""
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    avail = fig.bbox.width * (x1 - x0)
    if bb.width > avail:
        t.set_fontsize(t.get_fontsize() * avail / bb.width)


def _title(fig, title: str, subtitle: str = "") -> None:
    _fit(fig, fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK))
    if subtitle:
        _fit(fig, fig.text(0.012, 0.935, subtitle, ha="left", va="top", fontsize=9.5, color=INK2))


def _save_fig(fig, cfg: Config, name: str, source: str) -> None:
    fig.text(0.012, 0.008, f"Source: {source}", ha="left", va="bottom", fontsize=7.5, color=INK2)
    fig.savefig(cfg.path("figures") / name, dpi=200)
    plt.close(fig)


def _mean_ci(v: np.ndarray) -> tuple[float, float]:
    v = np.asarray(v, float)
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def fig_drill_coverage(cfg, prospects):
    drills = ["forty", "ten_yd_split", "vertical", "broad_jump", "short_shuttle", "three_cone", "bench_reps"]
    cov = prospects.groupby("combine_position", observed=True)[drills].apply(lambda g: g.notna().mean()).reindex(POS_ORDER)
    overall = prospects[drills].notna().mean()
    fig, ax = plt.subplots(figsize=(9, 3.9))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.17)
    ax.imshow(cov.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            v = cov.iat[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(drills)), [f"{LABELS[d]}\n({overall[d]:.0%} overall)" for d in drills], fontsize=8.5)
    ax.set_yticks(range(len(POS_ORDER)), POS_ORDER)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    agil = prospects[["short_shuttle", "three_cone"]].notna().any(axis=1).mean()
    _title(fig, f"Agility drills are the blind spot: only {agil:.0%} of prospects ran the shuttle or 3-cone",
           f"Share of prospects with a recorded result, by Combine position group (n = {len(prospects)})")
    _save_fig(fig, cfg, "fig01_drill_coverage.png", "combine_results, 2023-2025 classes")


def fig_drill_correlations(cfg, prospects):
    cols = ["ten_yd_split", "forty", "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps", "combine_weight", "ngs_athleticism_score"]
    df = prospects[cols + ["combine_position"]].copy()
    for c in cols:
        df[c] = df[c] - df.groupby("combine_position", observed=True)[c].transform("mean")
    for c in ["ten_yd_split", "forty", "three_cone", "short_shuttle"]:
        df[c] = -df[c]
    corr = df[cols].corr(method="spearman", min_periods=30)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.86, bottom=0.2)
    shown = corr.mask(mask)
    ax.imshow(shown.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(len(cols)):
        for j in range(i):
            v = corr.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.6 else INK)
    ax.set_xticks(range(len(cols)), [LABELS[c] for c in cols], rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(cols)), [LABELS[c] for c in cols], fontsize=8.5)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    r = corr.loc["forty", "ten_yd_split"]
    _title(fig, f"Sprint drills are near-duplicates (40 vs 10-yd split rho = {r:.2f}); bench press stands apart",
           "Within-position Spearman correlation; times sign-flipped so that + means 'both better'")
    _save_fig(fig, cfg, "fig02_drill_correlations.png", "combine_results, position-group means removed")


def fig_burst_distribution(cfg, plays, anomalies):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.12)
    rng = np.random.default_rng(cfg.seed)
    flagged = anomalies[anomalies.metric == "burst_2s"].set_index(["game_id", "play_id", "nfl_id"]).index
    key = plays.set_index(["game_id", "play_id", "nfl_id"]).index
    plays = plays.assign(flag=key.isin(flagged))
    data = [plays.loc[plays.combine_position == p, "burst_2s"].to_numpy() for p in POS_ORDER]
    ax.boxplot(data, positions=range(len(POS_ORDER)), widths=0.5, showfliers=False,
               medianprops={"color": INK, "linewidth": 1.4}, boxprops={"color": INK2}, whiskerprops={"color": INK2}, capprops={"color": INK2})
    for i, p in enumerate(POS_ORDER):
        sub = plays[plays.combine_position == p]
        jit = rng.uniform(-0.18, 0.18, len(sub))
        ok = ~sub.flag.to_numpy()
        ax.scatter(i + jit[ok], sub.burst_2s[ok], s=5, color=MUTED, alpha=0.45, linewidths=0)
        ax.scatter(i + jit[~ok], sub.burst_2s[~ok], s=16, color=ACCENT, linewidths=0, label="Flagged outlier (|robust z| > 3.5)" if i == 0 else None)
        ax.text(i, -0.9, f"n = {len(sub):,}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(POS_ORDER)), POS_ORDER)
    ax.set_ylabel("Distance from snap spot at t = 2.0 s (yd)")
    ax.set_ylim(-1.4, plays.burst_2s.max() + 0.8)
    ax.legend(loc="upper left", fontsize=8.5)
    med = plays.groupby("combine_position", observed=True).burst_2s.median()
    _title(fig, f"WRs cover {med['WR']:.1f} yd in the first 2 s, OL {med['OL']:.1f} yd; {int(plays.flag.sum())} plays flagged as outliers",
           "2-second snap burst per play and position group; box = median and interquartile range, whiskers 1.5 IQR")
    _save_fig(fig, cfg, "fig03_burst_distribution.png", f"game_tracking, regular season + playoffs, {len(plays):,} player-plays")


def fig_speed_traces(cfg, frames, plays, profiles, prospects):
    roles = prospects.set_index("nfl_id").nfl_position.astype(str)
    edge = profiles[(profiles.combine_position == "DL") & (profiles.n_plays >= 8)
                    & profiles.nfl_id.map(roles).isin(["DE", "OLB"])].sort_values("sbi")
    lo, hi = edge.iloc[0], edge.iloc[-1]
    curves = speed_curves(frames, plays, [int(lo.nfl_id), int(hi.nfl_id)], 4.0)
    names = prospects.set_index("nfl_id").display_name
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.12)
    peaks = {}
    for pid, color, lab in [(int(lo.nfl_id), MUTED, "low"), (int(hi.nfl_id), ACCENT, "high")]:
        c = curves[curves.nfl_id == pid]
        for _, g in c.groupby(["game_id", "play_id"]):
            ax.plot(g.t, g.speed, color=color, alpha=0.25, linewidth=0.8)
        med = c.groupby("t").speed.median()
        ax.plot(med.index, med.values, color=color if lab == "high" else INK2, linewidth=2.2)
        p = profiles.set_index("nfl_id").loc[pid]
        early = med[med.index <= 2.0]
        peaks[lab] = early.max()
        tx = 0.75 if lab == "high" else 0.95
        ty = early.max() + 0.35 if lab == "high" else float(med.loc[min(med.index, key=lambda t: abs(t - 1.0))]) - 0.75
        ax.text(tx, ty, f"{names[pid]} ({roles[pid]}): SBI {p.sbi:.0f}, {int(p.n_plays)} plays",
                fontsize=8.5, color=INK, fontweight="bold" if lab == "high" else "normal")
    ax.axvline(2.0, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.text(2.04, 0.2, "SBI measured here (t = 2.0 s)", fontsize=8, color=INK2)
    ax.set_xlabel("Seconds after the snap")
    ax.set_ylabel("Speed (yd/s)")
    ax.set_xlim(0, 4)
    _title(fig, f"Two edge rushers: one peaks at {peaks['high']:.1f} yd/s inside 2 s, the other at {peaks['low']:.1f}",
           "Post-snap speed on every tracked play (thin) and the median trace (thick); highest vs lowest SBI among DE/OLB with 8+ plays")
    _save_fig(fig, cfg, "fig04_speed_traces.png", "game_tracking, regular season + playoffs")


def fig_estimator_bias(cfg, bias, sens):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.14, wspace=0.35)
    colors = {"q80": ORANGE, "median": MUTED, "mean": ACCENT}
    names = {"q80": "80th percentile (rejected)", "median": "Median", "mean": "Mean (adopted)"}
    for est in ["q80", "median", "mean"]:
        b = bias[bias.estimator == est].sort_values("n_plays")
        a1.plot(b.n_plays, b.bias, marker="o", color=colors[est], linewidth=2, markersize=6)
        dy = {"mean": 0.022, "median": -0.022, "q80": 0.0}[est]
        a1.text(b.n_plays.iloc[-1] + 0.4, b.bias.iloc[-1] + dy, names[est], va="center", fontsize=8.5, color=INK)
    a1.axhline(0, color=INK2, linewidth=0.8)
    a1.set_xlim(2, 15.5)
    a1.set_xticks([3, 5, 10])
    a1.set_xlabel("Plays observed for the player")
    a1.set_ylabel("Bias vs full-sample value (yd)")
    a1.set_title("A. Estimate bias when a player is seen on few plays")
    s = sens.set_index(["x", "y"])
    ys = [("n_plays", "Plays\nobserved"), ("start_rate", "Start\nrate"), ("neg_log_pick", "Draft\ncapital")]
    w = 0.36
    for k, (est, col) in enumerate([("sbi_q80", ORANGE), ("sbi", ACCENT)]):
        vals = [s.loc[(est, y), "rho"] for y, _ in ys]
        lo = [s.loc[(est, y), "rho"] - s.loc[(est, y), "ci_low"] for y, _ in ys]
        hi = [s.loc[(est, y), "ci_high"] - s.loc[(est, y), "rho"] for y, _ in ys]
        x = np.arange(len(ys)) + (k - 0.5) * w
        a2.bar(x, vals, width=w - 0.04, color=col, alpha=0.85, label=names["q80" if est == "sbi_q80" else "mean"])
        a2.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    a2.axhline(0, color=INK2, linewidth=0.8)
    a2.set_xticks(range(len(ys)), [l for _, l in ys])
    a2.set_ylabel("Within-position Spearman rho (95% CI)")
    a2.set_title("B. What the SBI correlates with")
    a2.legend(fontsize=8, loc="upper right")
    q = bias[(bias.estimator == "q80") & (bias.n_plays == 3)].bias.iloc[0]
    _title(fig, f"Top-percentile capacity reads {abs(q):.2f} yd low at 3 plays, which leaks playing time into the metric",
           "Subsampling of players with 20+ plays; panel B compares the SBI built with each estimator")
    _save_fig(fig, cfg, "fig05_estimator_bias.png", "play_metrics + careers; 200 subsamples per player and n")


def fig_translation_matrix(cfg, matrix):
    xs = ["ten_yd_split", "forty", "broad_jump", "vertical", "short_shuttle", "three_cone", "bench_reps", "combine_weight", "ngs_athleticism_score"]
    ys = ["burst_2s_eb", "get_off_1s_eb", "peak_accel_2s_eb", "top_speed_eb", "max_decel_eb", "max_cut_deg_eb", "start_rate"]
    m = matrix.set_index(["x", "y"])
    rho = np.array([[m.loc[(x, y), "rho"] for y in ys] for x in xs])
    q = np.array([[m.loc[(x, y), "q_value"] for y in ys] for x in xs])
    n = [int(m.loc[(x, ys[0]), "n"]) for x in xs]
    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.subplots_adjust(left=0.24, right=0.98, top=0.83, bottom=0.14)
    ax.imshow(rho, cmap=DIVERGING, vmin=-0.4, vmax=0.4, aspect="auto")
    for i in range(len(xs)):
        for j in range(len(ys)):
            sig = q[i, j] < 0.05
            ax.text(j, i, f"{rho[i, j]:+.2f}{'*' if sig else ''}", ha="center", va="center", fontsize=9,
                    color=INK, fontweight="bold" if sig else "normal")
    ax.axvline(len(ys) - 1.5, color=SURFACE, linewidth=4)
    ax.set_xticks(range(len(ys)), [LABELS[y] for y in ys], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(xs)), [f"{LABELS[x]}  (n={k})" for x, k in zip(xs, n)], fontsize=9)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    best = matrix[matrix.y == "burst_2s_eb"].sort_values("rho", ascending=False).iloc[0]
    _title(fig, "Sprint and jump drills carry into game movement; the NGS composite and bench press do not",
           f"Within-position Spearman rho (* = FDR q < 0.05). Best for snap burst: {LABELS[best.x]} ({best.rho:+.2f})")
    _fit(fig, fig.text(0.24, 0.07, "Blue = better test goes with more game movement; orange = the reverse. Body weight is shown as measured (heavier = more).",
                       fontsize=8, color=INK2), x0=0.24)
    _save_fig(fig, cfg, "fig07_translation_matrix.png", f"combine_results + game_tracking, {int(matrix.n.max())} players with 3+ plays")


def fig_sbi_vs_start(cfg, profiles, prospects, starter_assoc):
    df = profiles.merge(prospects, on=["nfl_id", "combine_position"])
    rho = starter_assoc.set_index("x").loc["sbi"]
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.8), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.74, bottom=0.16, wspace=0.12)
    for ax, p in zip(axes, POS_ORDER):
        sub = df[df.combine_position == p]
        ax.scatter(sub.sbi, sub.start_rate, s=14, color=MUTED, linewidths=0)
        if len(sub) > 5:
            b, a = np.polyfit(sub.sbi, sub.start_rate, 1)
            xs = np.linspace(sub.sbi.min(), sub.sbi.max(), 10)
            ax.plot(xs, a + b * xs, color=ACCENT, linewidth=2)
        ax.set_title(f"{p}  (n = {len(sub)})", fontsize=9.5)
        ax.set_xlabel("SBI")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Share of games started")
    _title(fig, f"Game snap burst barely tracks starting within position (rho = {rho.rho:+.2f}, 95% CI {rho.ci_low:+.2f} to {rho.ci_high:+.2f})",
           "Each dot is a player with 3+ tracked plays; blue line = least-squares fit within the position group")
    _save_fig(fig, cfg, "fig09_sbi_vs_start_rate.png", "game_tracking + player_career_successes")


def fig_reliability(cfg, rel):
    metrics = ["burst_1s", "burst_2s", "get_off_1s", "peak_accel_2s", "top_speed", "max_decel", "max_cut_deg"]
    names = {"burst_1s": "Distance at 1 s", "burst_2s": "Snap burst\n(distance at 2 s)", "get_off_1s": "Get-off\n(speed at 1 s)",
             "peak_accel_2s": "Peak accel\n(first 2 s)", "top_speed": "Top speed", "max_decel": "Max braking",
             "max_cut_deg": "Sharpest cut"}
    r = rel.set_index(["combine_position", "metric"])
    icc = np.array([[max(r.loc[(p, m), "icc"], 0) for m in metrics] for p in POS_ORDER])
    pv = np.array([[r.loc[(p, m), "anova_p"] for m in metrics] for p in POS_ORDER])
    n = [int(r.loc[(p, metrics[0]), "players"]) for p in POS_ORDER]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.74, bottom=0.08)
    ax.imshow(icc, cmap=SEQUENTIAL, vmin=0, vmax=0.8, aspect="auto")
    for i in range(len(POS_ORDER)):
        for j in range(len(metrics)):
            v = icc[i, j]
            raw = r.loc[(POS_ORDER[i], metrics[j]), "icc"]
            txt = "nil" if raw <= 0 else f"{v:.2f}"
            sig = np.isfinite(pv[i, j]) and pv[i, j] < 0.01
            ax.text(j, i, txt + ("*" if sig else ""), ha="center", va="center", fontsize=9,
                    color="white" if v > 0.5 else INK, fontweight="bold" if sig else "normal")
    ax.set_xticks(range(len(metrics)), [names[m] for m in metrics], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(POS_ORDER)), [f"{p} ({k})" for p, k in zip(POS_ORDER, n)])
    ax.grid(False)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    b = {p: r.loc[(p, "burst_2s"), "icc"] for p in POS_ORDER}
    stable = [p for p in POS_ORDER if b[p] >= 0.5]
    weak = [p for p in POS_ORDER if b[p] < 0.5]
    _title(fig, f"Snap burst is a stable player trait for {' and '.join(f'{p} ({b[p]:.2f})' for p in stable)}; "
                f"weaker for {', '.join(weak)}" if stable else "Snap burst is not a stable trait at any position",
           "Share of between-player variance that is real difference (ICC); * = ANOVA p < 0.01; 'nil' = not detectable")
    _save_fig(fig, cfg, "fig06_metric_reliability.png", f"game_tracking, {sum(n)} players with 3+ plays")


def fig_dl_role(cfg, dl_tab):
    xs = ["ten_yd_split", "forty", "broad_jump", "vertical", "combine_weight", "ngs_athleticism_score"]
    y = "get_off_1s_eb"
    t = dl_tab[dl_tab.y == y].set_index(["adjustment", "x"])
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    fig.subplots_adjust(left=0.24, right=0.97, top=0.78, bottom=0.14)
    for i, x in enumerate(xs):
        a, b = t.loc[("none", x)], t.loc[("edge vs interior role", x)]
        ax.plot([a.rho, b.rho], [i, i], color=MUTED, linewidth=2, zorder=1)
        ax.errorbar(b.rho, i, xerr=[[b.rho - b.ci_low], [b.ci_high - b.rho]], fmt="none", ecolor=ACCENT, elinewidth=1, alpha=0.5, capsize=3)
        ax.scatter(a.rho, i, s=55, color=INK2, zorder=3, label="Raw (all DL)" if i == 0 else None)
        ax.scatter(b.rho, i, s=55, color=ACCENT, zorder=3, label="Within role (edge vs interior), 95% CI" if i == 0 else None)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(xs)), [f"{LABELS[x]} (n={int(t.loc[('none', x), 'n'])})" for x in xs])
    ax.invert_yaxis()
    ax.set_xlabel("Spearman rho with in-game get-off (speed 1 s after the snap); times sign-flipped")
    ax.legend(loc="lower left", fontsize=8.5)
    raw = t.loc[("none", "ten_yd_split"), "rho"]
    adj = t.loc[("edge vs interior role", "ten_yd_split"), "rho"]
    _title(fig, f"On the DL, half the drill-to-game link is role: 10-yd split {raw:+.2f} raw, {adj:+.2f} within role",
           "Edge = DE/OLB, interior = DT/NT; heavier interior linemen run slower and also get off slower because of their assignment")
    _save_fig(fig, cfg, "fig08_dl_translation_by_role.png", "defensive linemen with 3+ tracked plays")


def fig_regression_cv(cfg, m_dir, summary):
    order = ["position_mean", "ridge", "random_forest", "xgboost"]
    titles = {"all_positions": "All positions", "defensive_line": "Defensive line", "wide_receivers": "Wide receivers"}
    pops = [p for p in titles if p in set(summary.population)]
    fig, axes = plt.subplots(1, len(pops), figsize=(5.4 * len(pops), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.17, wspace=0.12)
    rng = np.random.default_rng(cfg.seed)
    best = {}
    for k, (ax, pop) in enumerate(zip(axes, pops)):
        folds = pd.read_csv(m_dir / f"regression_fold_{pop}_r2.csv")
        s = summary[summary.population == pop]
        champ = s[s.model != "position_mean"].sort_values("rmse_mean").iloc[0].model
        best[pop] = s.set_index("model").loc[champ]
        for i, m in enumerate(order):
            v = folds[m].to_numpy()
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=12, color=MUTED, linewidths=0)
            mu, half = _mean_ci(v)
            col = ACCENT if m == champ else INK
            ax.errorbar(i + 0.28, mu, yerr=half, fmt="o", color=col, ecolor=col, capsize=4, markersize=6, elinewidth=1.6)
            ax.text(i + 0.38, mu, f"{mu:+.2f}", va="center", fontsize=8.5, color=col)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks(range(len(order)), [LABELS[m] for m in order], fontsize=8.5)
        ax.set_title(f"{chr(65 + k)}. {titles[pop]} (n = {int(s.n.iloc[0])})")
        ax.set_xlim(-0.5, len(order) - 0.2)
        ax.set_ylim(-0.8, 0.9)
    axes[0].set_ylabel("Outer-fold R² of the SBI (clipped at -0.8)")
    short = {"all_positions": "all positions", "defensive_line": "DL", "wide_receivers": "WRs"}
    cut = {p: 1 - best[p].rmse_mean / summary[(summary.population == p) & (summary.model == "position_mean")].rmse_mean.iloc[0]
           for p in pops}
    _title(fig, "Combine results cut snap-burst error vs position alone by " +
           ", ".join(f"{cut[p]:.0%} ({short[p]})" for p in sorted(pops, key=lambda p: -cut[p])),
           "Nested CV, 5 folds x 3 repeats; dots = outer folds, marker = mean with 95% CI; blue = lowest-RMSE learned model")
    _save_fig(fig, cfg, "fig10_regression_cv.png", "Snap Burst Index from Combine measurables and drills")


EXPERIMENT_TITLES = {"all_prospects": "All prospects", "tracked_subset": "Players with tracking",
                     "onfield_subset": "Players with on-field metrics"}
EXPERIMENT_SETS = {"all_prospects": ["combine", "combine_draft", "combine_draft_proj", "combine_draft_ct"],
                   "tracked_subset": ["combine_draft", "combine_draft_mov"],
                   "onfield_subset": ["combine_draft", "combine_draft_perf"]}
SET_COLORS = {"combine": MUTED, "combine_draft": INK2, "combine_draft_proj": ORANGE, "combine_draft_mov": ACCENT,
              "combine_draft_perf": ACCENT, "combine_draft_ct": AQUA}


def _experiments(folds):
    return [(e, [f for f in EXPERIMENT_SETS[e] if f in set(folds[folds.population == e].feature_set)])
            for e in EXPERIMENT_SETS if e in set(folds.population)]


def fig_classification_auc(cfg, folds, summary):
    models = ["logistic_regression", "random_forest", "xgboost"]
    exps = _experiments(folds)
    fig, axes = plt.subplots(1, len(exps), figsize=(5.2 * len(exps), 4.9), sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.24, wspace=0.08)
    for k, (ax, (pop, fsets)) in enumerate(zip(axes, exps)):
        width = 0.7 / len(fsets)
        for j, fs in enumerate(fsets):
            for i, m in enumerate(models):
                v = folds[(folds.population == pop) & (folds.feature_set == fs) & (folds.model == m)].roc_auc.to_numpy()
                mu, half = _mean_ci(v)
                x = i + (j - (len(fsets) - 1) / 2) * width
                col = SET_COLORS[fs]
                ax.errorbar(x, mu, yerr=half, fmt="o", color=col, ecolor=col, capsize=3, markersize=6, elinewidth=1.6,
                            label=LABELS[fs].replace("\n", " ") if i == 0 else None)
                ax.text(x, min(mu + half + 0.01, 0.985), f"{mu:.2f}", ha="center", fontsize=7.5, color=INK)
        n = int(summary[summary.population == pop].n.iloc[0])
        ax.set_xticks(range(len(models)), [LABELS[m] for m in models], fontsize=8.5)
        ax.set_title(f"{chr(65 + k)}. {EXPERIMENT_TITLES[pop]} (n = {n})", fontsize=9.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7.5, ncol=1)
        ax.set_ylim(0.6, 1.0)
    axes[0].set_ylabel("ROC-AUC, outer folds (mean, 95% CI)")
    g = folds.groupby(["population", "feature_set"]).roc_auc.mean()
    t = f"Draft capital lifts AUC from {g[('all_prospects', 'combine')]:.2f} to {g[('all_prospects', 'combine_draft')]:.2f}"
    if ("all_prospects", "combine_draft_ct") in g.index:
        t += f"; drill tracking adds {g[('all_prospects', 'combine_draft_ct')] - g[('all_prospects', 'combine_draft')]:+.3f}"
    if ("onfield_subset", "combine_draft_perf") in g.index:
        t += f"; on-field performance {g[('onfield_subset', 'combine_draft_perf')] - g[('onfield_subset', 'combine_draft')]:+.3f}"
    _title(fig, t, "Predicting a starter (started 50%+ of available games); nested CV, 5 folds x 3 repeats; same folds within each panel")
    _save_fig(fig, cfg, "fig11_classification_auc.png", "players + combine_results + combine_tracking + player_career_successes + player_play")


def fig_roc(cfg, oof, summary):
    exps = _experiments(summary)
    fig, axes = plt.subplots(1, len(exps), figsize=(4.9 * len(exps), 4.9))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.8, bottom=0.12, wspace=0.25)
    for k, (ax, (pop, fsets)) in enumerate(zip(axes, exps)):
        models = []
        for fs in fsets:
            s = summary[(summary.population == pop) & (summary.feature_set == fs)].sort_values("roc_auc_mean", ascending=False).iloc[0]
            o = oof[(oof.population == pop) & (oof.feature_set == fs) & (oof.model == s.model)]
            o = o.groupby("row").agg(y=("y", "first"), p=("pred", "mean"))
            fpr, tpr, _ = roc_curve(o.y, o.p)
            models.append(LABELS[s.model])
            ax.plot(fpr, tpr, color=SET_COLORS[fs], linewidth=1.8,
                    label=f"{LABELS[fs].replace(chr(10), ' ').replace('Combine + draft + ', '+ ')}: {s.oof_auc:.2f} [{s.oof_auc_ci_low:.2f}, {s.oof_auc_ci_high:.2f}]")
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate" if k == 0 else "")
        ax.set_title(f"{chr(65 + k)}. {EXPERIMENT_TITLES[pop]} ({' / '.join(dict.fromkeys(models))})", fontsize=9.5)
        ax.legend(loc="lower right", fontsize=7, title="ROC-AUC [95% CI]", title_fontsize=7)
    _title(fig, "Out-of-fold ROC curves of the best model per feature set",
           "Predictions averaged over the 3 CV repeats; brackets = bootstrap 95% CI (2,000 stratified resamples)")
    _save_fig(fig, cfg, "fig12_roc_curves.png", "nested CV out-of-fold predictions")


def _nice(f):
    if f.endswith("_missing"):
        return f"{LABELS.get(f[:-8], f[:-8]).replace(chr(10), ' ')} not run"
    return LABELS.get(f, f).replace("\n", " ").replace("_", " ")


def fig_importance(cfg, imp_clf, imp_reg, champ_clf_model, reg_champ, reg_pop="dl_getoff", reg_label="DL get-off",
                   unit="rise in RMSE (s)"):
    imp_reg = imp_reg[imp_reg.population == reg_pop]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.8, bottom=0.15, wspace=0.75)
    for ax, imp, title, u in [(a1, imp_clf, f"A. Starter classifier ({LABELS[champ_clf_model]})", "drop in ROC-AUC"),
                              (a2, imp_reg, f"B. {reg_label[0].upper() + reg_label[1:]} ({LABELS[reg_champ]})", unit)]:
        top = imp.head(10).iloc[::-1]
        cols = [ACCENT if i == len(top) - 1 else MUTED for i in range(len(top))]
        ax.barh(range(len(top)), top.importance, xerr=top["std"], color=cols, ecolor=INK2, capsize=2, height=0.65)
        ax.set_yticks(range(len(top)), [_nice(f) for f in top.feature], fontsize=8.5)
        ax.set_xlabel(f"Permutation importance ({u})")
        ax.set_title(title)
        ax.axvline(0, color=INK2, linewidth=0.8)
    lead = _nice(imp_reg.iloc[0].feature)
    lead = lead[0].lower() + lead[1:] if len(lead) > 1 and lead[1].islower() else lead
    _title(fig, f"Draft pick dominates who starts; {lead} leads the {reg_label} model",
           "Permutation importance on the refitted champion models (in-sample, descriptive), 20 shuffles; bars = mean, lines = SD")
    _save_fig(fig, cfg, "fig13_feature_importance.png", "registered champion models")


def fig_translation_gap(cfg, gap):
    g = gap[(gap.population == "defensive_line") & (gap.n_plays >= 10)].dropna(subset=["translation_gap"]).sort_values("translation_gap")
    pick = pd.concat([g.head(6), g.tail(6)])
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    fig.subplots_adjust(left=0.34, right=0.95, top=0.85, bottom=0.1)
    cols = [ACCENT if v > 0 else ORANGE for v in pick.translation_gap]
    ax.barh(range(len(pick)), pick.translation_gap, color=cols, height=0.7, alpha=0.9)
    ax.set_ylim(-0.7, len(pick) + 0.2)
    ax.set_yticks(range(len(pick)), [f"{r.display_name} ({r.nfl_position}, {int(r.draft_year)}, {r.n_plays} plays)" for r in pick.itertuples()], fontsize=8.5)
    for i, r in enumerate(pick.itertuples()):
        ax.text(r.translation_gap + (0.6 if r.translation_gap > 0 else -0.6), i, f"{r.sbi:.0f} vs {r.predicted_sbi:.0f}",
                va="center", ha="left" if r.translation_gap > 0 else "right", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, linewidth=0.9)
    ax.set_xlabel("Translation gap = game SBI - Combine-predicted SBI (points)")
    lim = np.abs(pick.translation_gap).max() + 6
    ax.set_xlim(-lim, lim)
    ax.text(lim * 0.97, len(pick) - 0.4, "Plays faster than he tested", ha="right", fontsize=8.5, color=ACCENT, fontweight="bold")
    ax.text(-lim * 0.97, len(pick) - 0.4, "Tested faster than he plays", ha="left", fontsize=8.5, color=ORANGE, fontweight="bold")
    _title(fig, "Defensive linemen whose snap burst departs most from what their testing predicted",
           "Six largest positive and negative gaps among linemen with 10+ plays; labels = game SBI vs predicted SBI (out-of-fold)")
    _save_fig(fig, cfg, "fig14_translation_gap.png", "out-of-fold predictions of the champion regressor (nested CV)")


ONFIELD_NAMES = {"goe": "DL get-off over expected", "proe": "DL pressure rate over expected",
                 "ppoe": "OL pressure prob. over expected", "paoe": "OL pressures allowed over exp.",
                 "soe": "Separation over expected"}


def fig_onfield_reliability(cfg, rel, track_rel, stab):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=0.24, right=0.97, top=0.8, bottom=0.13, wspace=0.6)
    rows = [(f"{ONFIELD_NAMES[r.metric]} ({r.combine_position})" if r.metric == "soe" else ONFIELD_NAMES[r.metric],
             max(r.icc, 0), int(r.players), "on-field") for r in rel.itertuples()]
    tb = track_rel[track_rel.metric == "burst_2s"]
    rows += [(f"Tracking snap burst ({r.combine_position})", max(r.icc, 0), int(r.players), "tracking") for r in tb.itertuples()]
    rows.sort(key=lambda t: t[1])
    for i, (name, v, n, kind) in enumerate(rows):
        a1.barh(i, v, color=ACCENT if kind == "on-field" else MUTED, height=0.68)
        a1.text(v + 0.015, i, f"{v:.2f}  (n={n})", va="center", fontsize=8, color=INK)
    a1.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    a1.set_xlim(0, 1.12)
    a1.set_xlabel("ICC: share of variance that is real player difference")
    a1.set_title("A. Stable trait? Blue = on-field (season), grey = tracking SBI", fontsize=9.5)
    st = stab.sort_values("rho")
    a2.barh(range(len(st)), st.rho, color=[ACCENT if p < 0.05 else MUTED for p in st.p_value], height=0.6)
    for i, r in enumerate(st.itertuples()):
        a2.text(r.rho + 0.02, i, f"{r.rho:+.2f} (n={r.players})", va="center", fontsize=8, color=INK)
    a2.set_yticks(range(len(st)), [ONFIELD_NAMES[m].replace(" over expected", "").replace(" over exp.", "") for m in st.metric], fontsize=8.5)
    a2.set_xlim(min(0, st.rho.min()) - 0.05, 1.0)
    a2.axvline(0, color=INK2, linewidth=0.8)
    a2.set_xlabel("Spearman rho, rookie vs later seasons")
    a2.set_title("B. Rookie year vs later seasons", fontsize=9.5)
    g = rel.set_index(["metric", "combine_position"]).icc
    tr = [r.combine_position for r in tb.itertuples() if r.icc >= 0.5]
    _title(fig, f"Season-long on-field metrics are stable traits (DL get-off ICC {g[('goe', 'DL')]:.2f}); "
                f"2-s tracking burst is one for {' and '.join(tr) if tr else 'no position'}",
           "ICC by method of moments among players meeting the play minimum; panel B: 2023-2024 classes, blue = p < 0.05")
    _save_fig(fig, cfg, "fig15_onfield_reliability.png", "player_play regular season 2023-2025 + game_tracking")


def fig_onfield_translation(cfg, mat):
    xs = ["ten_yd_split", "forty", "broad_jump", "vertical", "short_shuttle", "three_cone", "bench_reps", "combine_weight",
          "ngs_athleticism_score", "neg_log_pick"]
    ys = ["goe_eb", "proe_eb", "ppoe_eb", "paoe_eb", "soe_eb"]
    m = mat.set_index(["x", "y"])
    rho = np.array([[m.loc[(x, y), "rho"] if (x, y) in m.index else np.nan for y in ys] for x in xs])
    q = np.array([[m.loc[(x, y), "q_value"] if (x, y) in m.index else np.nan for y in ys] for x in xs])
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    fig.subplots_adjust(left=0.27, right=0.98, top=0.8, bottom=0.12)
    ax.imshow(rho, cmap=DIVERGING, vmin=-0.45, vmax=0.45, aspect="auto")
    for i in range(len(xs)):
        for j in range(len(ys)):
            if np.isfinite(rho[i, j]):
                sig = q[i, j] < 0.05
                ax.text(j, i, f"{rho[i, j]:+.2f}{'*' if sig else ''}", ha="center", va="center", fontsize=9,
                        color=INK, fontweight="bold" if sig else "normal")
    ax.axhline(len(xs) - 1.5, color=SURFACE, linewidth=4)
    ax.set_xticks(range(len(ys)), [LABELS[y] for y in ys], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(xs)), [LABELS[x] for x in xs], fontsize=9)
    ax.grid(False)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    g = mat[mat.y == "goe_eb"].sort_values("rho", ascending=False).iloc[0]
    _title(fig, f"Drills predict DL get-off ({LABELS[g.x].lower()} rho {g.rho:+.2f}); OL protection tracks draft capital",
           "Within-position Spearman rho, Combine result vs season-long on-field metric (* = FDR q < 0.05); times sign-flipped")
    _fit(fig, fig.text(0.27, 0.05, "Blue = better test (or earlier pick) goes with better on-field performance; orange = the reverse. Body weight as measured.",
                       fontsize=8, color=INK2), x0=0.27)
    _save_fig(fig, cfg, "fig16_onfield_translation.png", "combine_results + player_play regular season 2023-2025")


def fig_dl_chain(cfg, onfield, prospects, med):
    df = onfield.drop(columns=["combine_position"]).merge(prospects, on="nfl_id")
    df = df[(df.combine_position.astype(str) == "DL") & df.goe_eb.notna()]
    role = df.nfl_position.astype(str).isin(["DT", "NT"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.14, wspace=0.25)
    for ax, x, y, xl, yl in [(a1, "ten_yd_split", "goe_eb", "10-yd split (s)", "Get-off over expected (s; + = quicker)"),
                             (a2, "goe_eb", "proe_eb", "Get-off over expected (s; + = quicker)", "Pressure rate over expected")]:
        t = df[[x, y]].dropna()
        for mask, col, lab in [(~role, ACCENT, "Edge (DE/OLB)"), (role, INK2, "Interior (DT/NT)")]:
            tt = df[mask][[x, y]].dropna()
            ax.scatter(tt[x], tt[y], s=22, color=col, alpha=0.75, linewidths=0, label=lab)
        b, a = np.polyfit(t[x], t[y], 1)
        xs = np.linspace(t[x].min(), t[x].max(), 10)
        ax.plot(xs, a + b * xs, color=INK, linewidth=1.4)
        r, p = stats.spearmanr(t[x], t[y])
        ax.text(0.02, 0.96, f"rho = {r:+.2f} (n = {len(t)})", transform=ax.transAxes, va="top", fontsize=9, color=INK)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.axhline(0, color=GRID, linewidth=0.8)
    a1.set_title("A. Faster 10-yd split, quicker get-off in games")
    a2.set_title("B. Quicker get-off, more pressure")
    a1.legend(loc="lower left", fontsize=8)
    r = med.set_index("x").loc["ten_yd_split"]
    _title(fig, f"About {r.share_mediated:.0%} of the 10-yd split's link to pressure runs through in-game get-off",
           f"Mediation on z-scores: indirect effect {abs(r.indirect_effect):.2f} SD [{abs(r.indirect_ci_high):.2f}, {abs(r.indirect_ci_low):.2f}] (bootstrap 95% CI); "
           "defensive linemen with 20+ pass-rush snaps" if r.indirect_effect < 0 else
           f"Mediation on z-scores: indirect effect {r.indirect_effect:.2f} SD [{r.indirect_ci_low:.2f}, {r.indirect_ci_high:.2f}] (bootstrap 95% CI); "
           "defensive linemen with 20+ pass-rush snaps")
    _save_fig(fig, cfg, "fig17_dl_chain.png", "combine_results + player_play regular season 2023-2025")


def fig_onfield_regression(cfg, m_dir, summary, tasks):
    order = ["position_mean", "ridge", "random_forest", "xgboost"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.3 * len(tasks), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.76, bottom=0.2, wspace=0.08)
    rng = np.random.default_rng(cfg.seed)
    best = {}
    for k, (ax, task) in enumerate(zip(axes, tasks)):
        pop = task["name"]
        folds = pd.read_csv(m_dir / f"regression_fold_{pop}_r2.csv")
        s = summary[summary.population == pop]
        champ = s[s.model != "position_mean"].sort_values("rmse_mean").iloc[0].model
        best[pop] = s.set_index("model").loc[champ]
        for i, m in enumerate(order):
            v = folds[m].to_numpy()
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=9, color=MUTED, linewidths=0)
            mu, half = _mean_ci(v)
            col = ACCENT if m == champ else INK
            ax.errorbar(i + 0.25, mu, yerr=half, fmt="o", color=col, ecolor=col, capsize=3, markersize=5, elinewidth=1.4)
            ax.text(i + 0.33, float(np.clip(mu, -0.76, 0.76)), f"{mu:+.2f}", va="center", fontsize=7.5, color=col)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks(range(len(order)), [LABELS[m].replace("\n(baseline)", "") for m in order], fontsize=7.5, rotation=30, ha="right")
        ax.set_title(f"{chr(65 + k)}. {task['label']}\n(n = {int(s.n.iloc[0])})", fontsize=8.5)
        ax.set_xlim(-0.5, len(order) - 0.1)
        ax.set_ylim(-0.8, 0.8)
    axes[0].set_ylabel("Outer-fold R² (clipped at -0.8)")
    short = {"dl_getoff": "DL get-off", "dl_pressure": "DL pressure", "ol_protection": "OL protection",
             "wrte_separation": "WR/TE separation"}
    yes = [f"{short.get(k, k)} (R² {v.r2_mean:+.2f})" for k, v in best.items() if v.r2_mean > 0.05]
    no = [short.get(k, k) for k, v in best.items() if v.r2_mean <= 0.05]
    _title(fig, f"Combine testing predicts {' and '.join(yes)}; not {' or '.join(no)}" if yes and no else
           "Combine testing vs on-field performance by family",
           "Nested CV, 5 folds x 3 repeats; dots = outer folds, marker = mean with 95% CI; blue = lowest-RMSE learned model")
    _save_fig(fig, cfg, "fig18_onfield_regression.png", "Combine -> season-long on-field performance (player_play)")


CT_GROUP_TITLES = {"forty": "40-yard dash", "3cone": "3-cone", "shuttle": "Short shuttle", "skill": "Position drills"}
CT_METRIC_NAMES = {"split_0_5_s": "0-5 yd split", "split_5_10_s": "5-10 yd split", "split_10_20_s": "10-20 yd split",
                   "split_10yd_s": "10-yd split from the start", "top_speed": "Top speed", "peak_accel": "Peak acceleration",
                   "burst_1s": "Distance at 1 s", "burst_2s": "Distance at 2 s", "t90_top_s": "Time to 90% of top speed",
                   "duration_s": "Drill time", "peak_decel": "Peak braking", "plant_speed_ratio": "Speed kept in turns",
                   "max_cut_deg": "Sharpest cut"}


def fig_ct_reliability(cfg, rel):
    """Which drill-tracking metrics repeat within a player (the gate that picks the features)."""
    order = ["forty", "skill", "shuttle", "3cone"]
    rel = rel.copy()
    rel["g"] = rel.drill_group.map({g: i for i, g in enumerate(order)})
    rel = rel.sort_values(["g", "reliability"], ascending=[True, True])
    rel["feature"] = rel["feature"].fillna("").astype(str)
    fig, ax = plt.subplots(figsize=(10.5, 8.6))
    fig.subplots_adjust(left=0.33, right=0.95, top=0.83, bottom=0.08)
    y, ticks, labels, seps = 0, [], [], []
    for g in order[::-1]:
        sub = rel[rel.drill_group == g]
        for r in sub.itertuples():
            col = ACCENT if r.feature else (INK2 if bool(r.passes_gate) else MUTED)
            ax.barh(y, max(r.reliability, 0), color=col, height=0.7)
            lab = f"{r.reliability:.3f}" if abs(r.reliability - 0.5) < 0.005 else f"{r.reliability:.2f}"
            ax.text(max(r.reliability, 0) + 0.015, y, lab, va="center", fontsize=8, color=INK)
            ticks.append(y)
            labels.append(f"{CT_METRIC_NAMES.get(r.metric, r.metric)}")
            y += 1
        ax.text(-0.02, y - 0.35, f"{CT_GROUP_TITLES[g]} (n = {int(sub.n_players.max())} players)", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8.5, fontweight="bold", color=INK)
        seps.append(y - 0.5)
        y += 1.2
    ax.axvline(0.5, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.505, y - 0.6, "gate r = 0.5", fontsize=8, color=INK2, va="top")
    ax.set_yticks(ticks, labels, fontsize=8.5)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.8, y - 0.4)
    ax.set_xlabel("Reliability (test-retest r; split-half with Spearman-Brown for position drills)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=ACCENT, label="Model feature"), Patch(color=INK2, label="Repeats, but another metric covers it"),
                        Patch(color=MUTED, label="Below the gate: not used")], loc="upper left", bbox_to_anchor=(0.33, 0.885),
               ncol=3, fontsize=8.5, frameon=False)
    f = rel.set_index(["drill_group", "metric"])
    _title(fig, f"Splits between distance marks repeat (5-10 yd r = {f.loc[('forty', 'split_5_10_s'), 'reliability']:.2f}); "
                f"the tracked start does not (0-5 yd r = {f.loc[('forty', 'split_0_5_s'), 'reliability']:.2f})",
           "How well each drill-tracking metric repeats within a player; a metric at r >= 0.5 can become a feature, one per construct")
    _save_fig(fig, cfg, "fig19_ct_reliability.png",
              f"combine_tracking: {int(rel.loc[rel.drill_group == 'forty', 'n_players'].max())} players with two tracked 40s")


def _ci_from_t(mean_diff, t, df, level=0.95):
    se = abs(mean_diff / t) if t and np.isfinite(t) and t != 0 else np.nan
    return se * stats.t.ppf(0.5 + level / 2, df)


def fig_ct_value(cfg, m_dir, clf_tests, n_folds):
    """Does drill tracking improve out-of-sample prediction? Same rows and folds, with vs without."""
    bvb = pd.read_csv(m_dir / "ct_best_vs_best.csv")
    names = {"defensive_line": "DL snap burst (SBI)", "wide_receivers": "WR snap burst (SBI)", "dl_getoff": "DL get-off over expected",
             "dl_pressure": "DL pressure over expected", "ol_protection": "OL protection over expected",
             "wrte_separation": "WR/TE separation over expected"}
    r1 = pd.DataFrame([{"label": f"{names[r.task]}\n{LABELS[r.model_without]} -> {LABELS[r.model_with]}",
                        "d": r.r2_diff, "lo": r.r2_diff_ci_low, "hi": r.r2_diff_ci_high, "p": r.r2_p_corrected_t,
                        "base": r.r2_without, "new": r.r2_with, "wins": r.r2_folds_better, "p_w": r.r2_p_wilcoxon}
                       for r in bvb[bvb.kind == "regression"].itertuples() if r.task in names])
    t = clf_tests[(clf_tests.comparison == "feature_set_vs_feature_set") & clf_tests.feature_set.str.startswith("combine_draft_ct vs combine_draft")]
    rows = []
    for m in ["logistic_regression", "random_forest", "xgboost"]:
        c = t[t.feature_set.str.contains(f"({m})", regex=False)].iloc[0]
        half = _ci_from_t(c.mean_diff_a_minus_b, c.t_corrected, n_folds - 1)
        rows.append({"label": LABELS[m], "d": c.mean_diff_a_minus_b, "lo": c.mean_diff_a_minus_b - half,
                     "hi": c.mean_diff_a_minus_b + half, "p": c.p_corrected_t})
    cb = bvb[bvb.kind == "classification"]
    if len(cb):
        c = cb.iloc[0]
        rows.append({"label": f"Best vs best\n{LABELS[c.model_without]} -> {LABELS[c.model_with]}", "d": c.roc_auc_diff,
                     "lo": c.roc_auc_diff_ci_low, "hi": c.roc_auc_diff_ci_high, "p": c.roc_auc_p_corrected_t})
    r2 = pd.DataFrame(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=0.2, right=0.97, top=0.78, bottom=0.14, wspace=0.5)
    for ax, r, xl, title, fmt in [(a1, r1, "Change in outer-fold R² (with - without drill tracking)",
                                   "A. Combine -> game metrics (best model in each arm)", "{:+.3f}"),
                                  (a2, r2, "Change in outer-fold ROC-AUC", "B. Starter classifier, Combine + draft", "{:+.3f}")]:
        yy = np.arange(len(r))[::-1]
        for yi, row in zip(yy, r.itertuples()):
            sig = row.p < 0.05
            col = ACCENT if sig and row.d > 0 else (ORANGE if sig else INK2)
            ax.errorbar(row.d, yi, xerr=[[row.d - row.lo], [row.hi - row.d]], fmt="o", color=col, ecolor=col, capsize=3,
                        markersize=7, elinewidth=1.6)
            extra = f"  R² {row.base:+.2f} -> {row.new:+.2f}" if hasattr(row, "base") else ""
            ax.text(row.d, yi + 0.24, fmt.format(row.d) + f" (p = {row.p:.2f})" + extra, ha="center", fontsize=7.6, color=INK)
        ax.axvline(0, color=INK2, linewidth=0.9)
        ax.set_yticks(yy, r.label, fontsize=8)
        ax.set_ylim(-0.6, len(r) - 0.25)
        ax.set_xlabel(xl, fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        lim = float(np.nanmax(np.abs(np.r_[r.lo.to_numpy(), r.hi.to_numpy()]))) * 1.1
        ax.set_xlim(-lim, lim)
        ax.grid(axis="y", visible=False)
    up = [names[r.task] for r in bvb[bvb.kind == "regression"].itertuples() if r.task in names and r.r2_p_corrected_t < 0.05 and r.r2_diff > 0]
    best_up = r1.sort_values("d").iloc[-1]
    cbest = r2.iloc[-1]
    if up:
        head = f"Drill tracking improves {', '.join(up)}; nothing else improves"
    elif best_up.d > 0.01:
        head = (f"Drill tracking lifts {best_up.label.split(chr(10))[0].replace(' over expected', '')} R² "
                f"{best_up.base:+.2f} -> {best_up.new:+.2f} ({int(best_up.wins)}/{n_folds} folds, p = {best_up.p:.2f}); nothing else improves")
    else:
        head = (f"Drill tracking improves no model: R² change {r1.d.min():+.3f} to {r1.d.max():+.3f}, "
                f"starter AUC {cbest.d:+.3f}")
    _title(fig, head,
           f"Same players and folds with and without the 5 drill-tracking features; mean over {n_folds} outer folds, "
           "95% CI from the corrected resampled t-test; blue = significant gain")
    _save_fig(fig, cfg, "fig20_ct_value.png", "nested CV per-fold scores: ct_best_vs_best.csv, classification_pairwise_tests.csv")


def fig_ct_translation(cfg, mat):
    ys = ["goe_eb", "proe_eb", "ppoe_eb", "paoe_eb", "soe_eb"]
    m = mat.set_index(["x", "y"])
    rho = np.array([[m.loc[(x, y), "rho"] if (x, y) in m.index else np.nan for y in ys] for x in CT_FEATURES])
    q = np.array([[m.loc[(x, y), "q_value"] if (x, y) in m.index else np.nan for y in ys] for x in CT_FEATURES])
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    fig.subplots_adjust(left=0.27, right=0.98, top=0.7, bottom=0.17)
    ax.imshow(rho, cmap=DIVERGING, vmin=-0.45, vmax=0.45, aspect="auto")
    for i in range(len(CT_FEATURES)):
        for j in range(len(ys)):
            if np.isfinite(rho[i, j]):
                sig = q[i, j] < 0.05
                ax.text(j, i, f"{rho[i, j]:+.2f}{'*' if sig else ''}", ha="center", va="center", fontsize=9,
                        color=INK, fontweight="bold" if sig else "normal")
    ax.set_xticks(range(len(ys)), [LABELS[y] for y in ys], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(CT_FEATURES)), [LABELS[x] for x in CT_FEATURES], fontsize=9)
    ax.grid(False)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    short_x = {"ct_forty_split_5_10_s": "40 acceleration (5-10 yd)", "ct_forty_top_speed": "40 top speed",
               "ct_shuttle_plant_speed_ratio": "shuttle turns", "ct_skill_speed_z": "position-drill top speed",
               "ct_skill_accel_z": "position-drill acceleration"}
    short_y = {"goe_eb": "DL get-off", "proe_eb": "DL pressure", "ppoe_eb": "OL protection", "paoe_eb": "OL pressures allowed",
               "soe_eb": "WR/TE separation"}
    sig = mat[(mat.q_value < 0.05) & mat.x.isin(CT_FEATURES) & mat.y.isin(ys)].sort_values("rho", key=np.abs, ascending=False)
    if len(sig):
        g = sig.iloc[0]
        sx = short_x[g.x]
        t = f"{sx[0].upper() + sx[1:]} links to {short_y[g.y]} (rho {g.rho:+.2f})"
        t += f"; {len(sig) - 1} other cells pass FDR" if len(sig) > 2 else ("; one other cell passes FDR" if len(sig) == 2 else "; no other cell passes FDR")
    else:
        t = "No drill-tracking feature links to on-field performance after FDR correction"
    _title(fig, t, "Within-position Spearman rho, drill-tracking feature vs season-long on-field metric (* = FDR q < 0.05); split time sign-flipped")
    _fit(fig, fig.text(0.27, 0.06, "Blue = better drill movement goes with better on-field performance; orange = the reverse.", fontsize=8, color=INK2), x0=0.27)
    _save_fig(fig, cfg, "fig21_ct_onfield_translation.png", "combine_tracking + player_play regular season 2023-2025")


def make_all_figures(cfg: Config) -> None:
    """Draws every available figure from the clean data and the saved metrics."""
    _style()
    p, m, i = cfg.path("processed"), cfg.path("metrics"), cfg.path("interim")
    prospects = pd.read_parquet(p / "prospects.parquet")
    plays = pd.read_parquet(p / "play_metrics.parquet")
    profiles = pd.read_parquet(p / "player_movement_profiles.parquet")
    # Only the frames the speed-trace figure uses (DL players), so the full tracking never enters memory.
    dl_ids = profiles.loc[profiles.combine_position == "DL", "nfl_id"].astype(int).tolist()
    frames = pd.read_parquet(i / "tracking_frames.parquet", columns=["game_id", "play_id", "nfl_id", "time", "s", "event"],
                             filters=[("nfl_id", "in", dl_ids)])
    fig_drill_coverage(cfg, prospects)
    fig_drill_correlations(cfg, prospects)
    fig_burst_distribution(cfg, plays, pd.read_csv(m / "play_metric_anomalies.csv"))
    fig_speed_traces(cfg, frames, plays, profiles, prospects)
    fig_estimator_bias(cfg, pd.read_csv(m / "estimator_bias_simulation.csv"), pd.read_csv(m / "sbi_estimator_sensitivity.csv"))
    fig_reliability(cfg, pd.read_csv(m / "metric_reliability.csv"))
    fig_translation_matrix(cfg, pd.read_csv(m / "translation_matrix.csv"))
    fig_dl_role(cfg, pd.read_csv(m / "dl_translation_role_adjusted.csv"))
    fig_sbi_vs_start(cfg, profiles, prospects, pd.read_csv(m / "movement_vs_start_rate.csv"))
    if (m / "regression_summary.csv").exists():
        reg = pd.read_csv(m / "regression_summary.csv")
        fig_regression_cv(cfg, m, reg)
        fig_translation_gap(cfg, pd.read_csv(m / "translation_gap.csv"))
    onfield_ready = (p / "player_onfield_profiles.parquet").exists() and (m / "onfield_reliability.csv").exists()
    if onfield_ready:
        onfield = pd.read_parquet(p / "player_onfield_profiles.parquet")
        fig_onfield_reliability(cfg, pd.read_csv(m / "onfield_reliability.csv"), pd.read_csv(m / "metric_reliability.csv"),
                                pd.read_csv(m / "onfield_stability.csv"))
        fig_onfield_translation(cfg, pd.read_csv(m / "onfield_translation_matrix.csv"))
        fig_dl_chain(cfg, onfield, prospects, pd.read_csv(m / "onfield_mediation_dl.csv"))
    if (m / "regression_summary.csv").exists() and onfield_ready:
        reg = pd.read_csv(m / "regression_summary.csv")
        tasks = [t for t in cfg["regression"].get("onfield_tasks", []) if t["name"] in set(reg.population)]
        if tasks:
            fig_onfield_regression(cfg, m, reg, tasks)
    if (m / "ct_reliability.csv").exists():
        fig_ct_reliability(cfg, pd.read_csv(m / "ct_reliability.csv"))
    if (m / "ct_onfield_translation_matrix.csv").exists():
        fig_ct_translation(cfg, pd.read_csv(m / "ct_onfield_translation_matrix.csv"))
    if (m / "ct_best_vs_best.csv").exists() and (m / "classification_pairwise_tests.csv").exists():
        tests = pd.read_csv(m / "classification_pairwise_tests.csv")
        if tests.feature_set.astype(str).str.startswith("combine_draft_ct vs").any():
            fig_ct_value(cfg, m, tests, cfg["cv"]["outer_splits"] * cfg["cv"]["outer_repeats"])
    if (m / "classification_summary.csv").exists():
        clf = pd.read_csv(m / "classification_summary.csv")
        fig_classification_auc(cfg, pd.read_csv(m / "classification_fold_scores.csv"), clf)
        fig_roc(cfg, pd.read_parquet(m / "classification_oof.parquet"), clf)
        finals = json.loads((m / "classification_final_models.json").read_text())
        champ_entry = next(v for k, v in finals.items() if k.startswith("champion"))
        reg = pd.read_csv(m / "regression_summary.csv")
        if "dl_getoff" in set(reg.population):
            pop, label, unit = "dl_getoff", "DL get-off", "rise in RMSE (s)"
        else:
            pop, label, unit = "defensive_line", "DL snap burst", "rise in RMSE (SBI points)"
        rc = reg[(reg.population == pop) & (reg.model != "position_mean")].sort_values("rmse_mean").iloc[0].model
        fig_importance(cfg, pd.read_csv(m / f"classification_importance_all_prospects_{champ_entry['feature_set']}.csv"),
                       pd.read_csv(m / "regression_permutation_importance.csv"), champ_entry["model"], rc, pop, label, unit)


# =============================================================================
# Command line
# =============================================================================
STAGES = ["all", "contract", "data", "translation", "models", "stats", "figures", "relocate"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Training, evaluation and MLflow registration for the NFL project")
    ap.add_argument("--config", default=None, help="configuration file (default configs/config.yaml)")
    ap.add_argument("--stage", default="all", choices=STAGES,
                    help="all = data + translation + models + figures; models = translation + models on the clean data")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    if args.stage == "relocate":
        log.info("relocate: %d MLflow store cells now point at this folder", relocate_store(cfg))
        return
    if args.stage == "contract":
        stage_contract(cfg)
        return
    mlflow_setup(cfg)
    log.info("NFL v%s seed=%d", __version__, cfg.seed)

    if args.stage == "stats":
        stage_stats(cfg)
        return
    if args.stage == "figures":
        make_all_figures(cfg)
        log.info("done")
        return
    d = stage_data(cfg) if args.stage in ("all", "data") else load_processed(cfg)
    if args.stage == "data":
        return
    if args.stage in ("all", "translation", "models"):
        stage_translation(cfg, d)
    if args.stage in ("all", "models"):
        stage_regression(cfg, d)
        stage_classification(cfg, d)
        ct_best_vs_best(cfg)
    if args.stage == "all":
        make_all_figures(cfg)
    log.info("done")


if __name__ == "__main__":
    # Runs through the imported `train` module so the pipeline classes (PositionMedianImputer,
    # OnFieldProjector) are serialised as `train.<Class>` rather than `__main__.<Class>`: that way the
    # registered models load from the notebook or any script.
    import train

    train.main()
