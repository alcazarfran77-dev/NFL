"""Model zoo, preprocessing and nested cross-validation.

Every model is an sklearn Pipeline:
  PositionMedianImputer -> ColumnTransformer(StandardScaler | OneHotEncoder) -> model
so imputation medians and scaling statistics are learned inside each training
fold only (no leakage from validation folds).

Evaluation = nested CV: an outer RepeatedStratifiedKFold estimates performance,
an inner GridSearchCV picks hyperparameters using the outer-training part only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

GROUP_COL = "combine_position"


class PositionMedianImputer(BaseEstimator, TransformerMixin):
    """Fill numeric NaNs with the training-fold median of the player's position group.

    Falls back to the global training median when a group has no observed value
    (e.g. offensive linemen rarely run the 3-cone in some folds). Columns in `exclude`
    (projector labels) pass through untouched.
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
    """Stacked Combine -> on-field projection (v2.0.0). Pre-draft safe.

    Input columns named `onfield_target_<metric>` hold a player's observed on-field value
    (NaN when not measured). They are used ONLY as labels: `fit` trains one Ridge model per
    metric on the training rows that have a label, using the Combine columns. `transform`
    replaces every label column by `proj_<metric>`, the projection from Combine columns alone
    (0 for players outside the metric's position family). Training rows get out-of-fold
    projections (`fit_transform`), so the downstream classifier sees the same kind of
    noisy projection it will see on new players.
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
        for t in self.targets_:  # replace in-sample projections of labelled rows by out-of-fold ones
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
    if name == "position_mean":  # baseline: position group one-hot only -> group means
        pre = make_preprocessor([], include_position=True)
    else:
        pre = make_preprocessor(numeric_cols, projector={**(projector or {}), "random_state": seed})
    return Pipeline([("prep", pre), ("model", make_estimator(name, task, seed, pos_weight))])


def projector_params(cfg) -> dict:
    c = cfg["classification"]
    return {"alpha": float(c.get("projector_alpha", 10.0)), "inner_folds": int(c.get("projector_inner_folds", 5))}


# ----------------------------------------------------------------------------- metrics
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


# ----------------------------------------------------------------------------- nested CV
@dataclass
class CVResult:
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
    """Tune on all data (5-fold inner CV) and refit: the model that gets registered."""
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
