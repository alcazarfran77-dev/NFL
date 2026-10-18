"""Entrenamiento, evaluación y registro en MLflow del proyecto NFL.

Usa los datos limpios de `datos_prep.py` y ejecuta:
  4 traducción    asociaciones prueba del Combine -> movimiento en partido y -> rendimiento en el campo,
                  fiabilidad, mediación y modelos lineales generalizados (GLM)
  5 regresión     Combine -> Snap Burst Index (SBI) y Combine -> rendimiento en el campo
                  (media por posición vs Ridge vs Random forest vs XGBoost)
  6 clasificación titular (regresión logística vs Random forest vs XGBoost) x conjuntos de variables anidados
  7 estadística   pruebas ómnibus recalculadas desde los puntajes por pliegue guardados (sin reentrenar)
  8 figuras       21 figuras del reporte en reportes/figuras

Cada modelo es un Pipeline de scikit-learn: imputación por mediana de posición -> escalado / one-hot ->
estimador, ajustado dentro de cada pliegue de entrenamiento. La evaluación es validación cruzada anidada
(5 pliegues x 3 repeticiones por fuera; GridSearchCV de 3 pliegues por dentro) con semilla 42. Cada
corrida queda en MLflow con las etiquetas pipeline_version, data_version, git_commit y seed; los
modelos campeones se reentrenan con todas las filas y se registran con alias (champion / challenger).

Uso
  python fuentes/train.py                         todo: datos + traducción + modelos + figuras
  python fuentes/train.py --etapa modelos         traducción + modelos con los datos limpios ya guardados
  python fuentes/train.py --etapa figuras         solo figuras, desde las métricas guardadas
  python fuentes/train.py --etapa estadistica     pruebas ómnibus sin reentrenar
  python fuentes/train.py --etapa reubicar        una vez tras clonar o extraer: corrige rutas del almacén MLflow
  python fuentes/train.py --config configuracion/rapida.yaml   corrida rápida (CI, ~10 min)
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

from datos_prep import (ATHLETIC, ATHLETIC_CT, CT_FEATURES, CT_LOWER_IS_BETTER, CT_SET,  # noqa: E402
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
# Modelos: preprocesamiento y validación cruzada anidada
# =============================================================================
# Cada modelo es un Pipeline de sklearn:
#   PositionMedianImputer -> ColumnTransformer(StandardScaler | OneHotEncoder) -> modelo
# así las medianas de imputación y las estadísticas de escalado se aprenden solo dentro de cada
# pliegue de entrenamiento (sin fuga de información desde los pliegues de validación).
#
# Evaluación = validación cruzada anidada: un RepeatedStratifiedKFold externo estima el rendimiento y
# un GridSearchCV interno elige los hiperparámetros usando solo la parte de entrenamiento externa.
GROUP_COL = "combine_position"


class PositionMedianImputer(BaseEstimator, TransformerMixin):
    """Rellena los NaN numéricos con la mediana del grupo de posición en el pliegue de entrenamiento.

    Si un grupo no tiene ningún valor observado usa la mediana global de entrenamiento (por ejemplo,
    los linieros ofensivos casi no corren el 3 conos en algunos pliegues). Las columnas de `exclude`
    (etiquetas del proyector) pasan sin cambios.
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
    """Proyección apilada Combine -> rendimiento en el campo. Se puede usar antes del draft.

    Las columnas `onfield_target_<métrica>` traen el valor observado del jugador en el campo (NaN si no
    se midió) y se usan SOLO como etiquetas: `fit` entrena un Ridge por métrica con las filas que tienen
    etiqueta, usando las columnas del Combine. `transform` reemplaza cada columna de etiqueta por
    `proj_<métrica>`, la proyección hecha solo con el Combine (0 para jugadores fuera de la familia de
    posiciones de la métrica). Las filas de entrenamiento reciben proyecciones fuera de pliegue
    (`fit_transform`), para que el clasificador vea el mismo tipo de proyección ruidosa que verá en
    jugadores nuevos.
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
        for t in self.targets_:  # reemplaza las proyecciones dentro de muestra de las filas etiquetadas por fuera de pliegue
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
    """Imputación por posición (+ proyector si hay etiquetas de campo) + escalado / one-hot."""
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
    """Estimador por nombre y tarea, siempre con la semilla fija."""
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
    """Pipeline completo: preprocesamiento + estimador."""
    if name == "position_mean":  # referencia: solo el one-hot del grupo de posición -> medias por grupo
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
    """Resultado de una validación cruzada anidada: puntajes por pliegue, predicciones fuera de pliegue
    e hiperparámetros elegidos en cada pliegue."""

    model: str
    feature_set: str
    task: str
    fold_scores: pd.DataFrame
    oof: pd.DataFrame                    # columnas: row, repeat, y, pred
    best_params: list[dict] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0


def outer_splitter(cfg, seed: int) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=cfg["cv"]["outer_splits"], n_repeats=cfg["cv"]["outer_repeats"], random_state=seed)


def nested_cv(model_name: str, feature_set_name: str, task: str, X: pd.DataFrame, y: pd.Series,
              strata: pd.Series, grid: dict, cfg, seed: int, threshold: float = 0.5) -> CVResult:
    """Validación cruzada anidada de un modelo: búsqueda en rejilla por dentro, evaluación por fuera."""
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
    """Ajusta con todos los datos (CV interna de 5 pliegues) y reentrena: es el modelo que se registra."""
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
# Comparación estadística
# =============================================================================
#   corrected_resampled_ttest  Nadeau y Bengio (2003): prueba t pareada sobre los puntajes de CV repetida
#                              con la varianza inflada por el solapamiento de los conjuntos de entrenamiento
#                              (la prueba t pareada ingenua es demasiado liberal).
#   pairwise_model_tests       la anterior + Wilcoxon, con corrección de Holm.
#   friedman                   prueba ómnibus de que k modelos tienen el mismo rango en los pliegues.
#   bootstrap_ci               intervalo percentil de cualquier métrica sobre predicciones fuera de pliegue.
#   delong_test                DeLong (1988) para dos ROC-AUC correlacionados.
#   within_group_spearman      asociación prueba-partido neta del grupo de posición, IC bootstrap + q de
#                              Benjamini-Hochberg.
#   partial_spearman           correlación de rangos de x e y tras quitar el efecto lineal de controles.
def corrected_resampled_ttest(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    j = len(d)
    var = d.var(ddof=1)
    se = np.sqrt((1 / j + n_test / n_train) * var) if var > 0 else np.nan
    t = d.mean() / se if se and se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), df=j - 1) if np.isfinite(t) else np.nan
    return {"mean_diff": d.mean(), "t": t, "df": j - 1, "p_value": p}


def holm(pvals: list[float]) -> np.ndarray:
    """Valores p ajustados por Holm (escalonado, monótono, con tope en 1)."""
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
    """scores: una columna por modelo, una fila por pliegue externo (los mismos pliegues para todos)."""
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
    """Prueba de Friedman entre modelos en los mismos pliegues; mean_ranks: 1 = el mejor en cada pliegue."""
    if scores.shape[1] < 3:
        return {"statistic": np.nan, "p_value": np.nan, "mean_ranks": {}}
    res = stats.friedmanchisquare(*[scores[c].to_numpy() for c in scores.columns])
    ranks = scores.rank(axis=1, ascending=not higher_is_better).mean().to_dict()
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "rank_1_is": "best", "mean_ranks": ranks}


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, fn: Callable, n_boot: int, seed: int,
                 stratify: bool = False, alpha: float = 0.05) -> tuple[float, float, float]:
    """(estimación, límite inferior, límite superior) por bootstrap percentil."""
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
    """Prueba de DeLong bilateral para AUC(p1) frente a AUC(p2) en las mismas muestras."""
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
    """Mediación x -> m -> y por producto de coeficientes sobre variables en z, bootstrap percentil.

    a = pendiente de m sobre x; b = pendiente de y sobre m controlando x; indirecto = a x b.
    total = pendiente de y sobre x; directo = pendiente de x en y ~ x + m. Coeficientes en unidades de DE.
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
    """rho de Spearman entre x e y después de quitar a ambas las medias del grupo de posición."""
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
    # BH sobre los valores p finitos (una columna constante da rho = NaN: se reporta pero no se prueba)
    p = out["p_value"].to_numpy(dtype=float)
    q = np.full(len(p), np.nan)
    if np.isfinite(p).any():
        q[np.isfinite(p)] = stats.false_discovery_control(p[np.isfinite(p)], method="bh")
    out["q_value"] = q
    return out


def partial_spearman(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> dict:
    """Correlación parcial de Spearman: r de Pearson entre residuos de rangos tras regresar los rangos de los controles."""
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
# MLflow: seguimiento de experimentos y registro de modelos
# =============================================================================
# Almacén local SQLite (mlflow/mlflow.db) para que el registro de modelos funcione sin conexión; los
# artefactos van a mlflow/mlartifacts. Cada corrida lleva la versión del flujo, la versión de datos (hash
# del manifiesto), el commit de git y la semilla, que es lo que hace rastreable y reproducible cualquier número.
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def mlflow_setup(cfg: Config) -> str:
    """Conecta MLflow al almacén del proyecto y crea el experimento si no existe."""
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
    """Etiquetas de trazabilidad de cada corrida."""
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
    """Registra las métricas por pliegue (como serie por paso), su media/DE y las predicciones fuera de pliegue."""
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
    """Apunta un almacén MLflow copiado a esta carpeta. Devuelve el número de celdas actualizadas.

    MLflow guarda las ubicaciones de los artefactos como URI absolutas, así que un almacén movido a otra
    carpeta o máquina no encuentra sus modelos. Esto reescribe la raíz del proyecto registrada (leída de
    la ubicación de artefactos del experimento) por la raíz actual en cada columna de texto del almacén
    SQLite. Se ejecuta una vez después de clonar el repositorio o extraer el archivo.
    """
    import sqlite3

    db = PROJECT_ROOT / cfg["mlflow"]["tracking_uri"].removeprefix("sqlite:///")
    if not db.exists():  # todavía no hay almacén (nada que reubicar)
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
    """Registra un pipeline ajustado con su firma; opcionalmente lo inscribe en el registro con un alias."""
    example = X.head(5)
    output = model.predict_proba(example)[:, 1] if task == "classification" else model.predict(example)
    sig = infer_signature(example, output)
    here = Path(__file__).resolve().parent
    info = mlflow.sklearn.log_model(
        model, name="model", signature=sig, input_example=example,
        # El código del proyecto viaja con el modelo, así carga en cualquier máquina con solo las dependencias.
        code_paths=[str(here / "datos_prep.py"), str(here / "train.py")],
        # El pipeline contiene transformadores propios (PositionMedianImputer) que skops no puede
        # serializar; cloudpickle es seguro aquí porque solo se carga código del propio proyecto.
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
# 4 Traducción: prueba del Combine -> partido -> campo
# =============================================================================
TIME_DRILLS = ["ten_yd_split", "forty", "three_cone", "short_shuttle"]
TRANSLATION_X = ["ten_yd_split", "forty", "vertical", "broad_jump", "three_cone", "short_shuttle",
                 "bench_reps", "combine_weight", "ngs_athleticism_score"]
TRANSLATION_Y = [f"{m}_eb" for m in ["burst_2s", "get_off_1s", "peak_accel_2s", "top_speed", "max_decel", "max_cut_deg"]]
ONFIELD_Y = [f"{m}_eb" for m in ONFIELD_METRICS]


def _icc(df: pd.DataFrame, key: str, value: str) -> float:
    """ICC por método de momentos de `value` agrupado por `key`."""
    g = df.groupby(key)[value]
    n, mu, var = g.size(), g.mean(), g.var(ddof=1)
    if len(n) < 3:
        return np.nan
    sigma2 = np.average(var.fillna(var.median()), weights=np.maximum(n - 1, 1))
    total = mu.var(ddof=1)
    return float((total - np.mean(sigma2 / n)) / total) if total > 0 else np.nan


def stage_translation(cfg: Config, d: dict) -> dict:
    """Matriz de traducción, fiabilidad, análisis de la línea defensiva, sensibilidad del estimador y GLM."""
    prospects, profiles = d["prospects"], d["profiles"]
    df = profiles.merge(prospects, on=["nfl_id", "combine_position"], how="inner")
    oriented = df.copy()
    for c in TIME_DRILLS:  # orientar para que un rho positivo siempre signifique "mejor prueba -> más movimiento"
        oriented[c] = -oriented[c]
    nb = cfg["cv"]["bootstrap_resamples"]
    matrix = within_group_spearman(oriented, TRANSLATION_X, TRANSLATION_Y + ["start_rate"], "combine_position", nb, cfg.seed)
    matrix["x_oriented"] = matrix["x"].isin(TIME_DRILLS)
    _save(matrix, cfg, "translation_matrix.csv", key="metrics")

    # Fiabilidad del SBI: peso medio de Bayes empírico. Limita cuánta varianza puede explicar cualquier
    # modelo pre-draft (R^2 observado <= fiabilidad x R^2 real).
    rel = profiles.groupby("combine_position", observed=True)["sbi_reliability"].agg(["mean", "median", "min", "max", "count"])
    rel.loc["ALL"] = [profiles.sbi_reliability.mean(), profiles.sbi_reliability.median(),
                      profiles.sbi_reliability.min(), profiles.sbi_reliability.max(), len(profiles)]
    _save(rel.reset_index(names="combine_position"), cfg, "sbi_reliability_by_position.csv", key="metrics")

    # ¿Qué métricas de partido son rasgos estables del jugador, por posición? (ICC + ANOVA de una vía)
    reliab = metric_reliability(d["plays"], profiles)
    _save(reliab, cfg, "metric_reliability.csv", key="metrics")

    # Análisis de la línea defensiva: traducción bruta vs ajustada por rol (edge DE/OLB vs interior DT/NT).
    dl = oriented[oriented.combine_position == "DL"].copy()
    dl["dl_role"] = np.where(dl["nfl_position"].isin(["DT", "NT"]), "interior", "edge")
    dl_y = ["burst_2s_eb", "get_off_1s_eb", "peak_accel_2s_eb"]
    dl_tab = pd.concat([
        within_group_spearman(dl, TRANSLATION_X, dl_y, "combine_position", nb, cfg.seed).assign(adjustment="none"),
        within_group_spearman(dl, TRANSLATION_X, dl_y, "dl_role", nb, cfg.seed).assign(adjustment="edge vs interior role"),
    ], ignore_index=True)
    _save(dl_tab, cfg, "dl_translation_role_adjusted.csv", key="metrics")

    # Movimiento en partido vs titularidad, dentro del grupo de posición
    starter_assoc = within_group_spearman(df, ["sbi"] + [f"{m}_z" for m in METRICS if m != "burst_2s"],
                                          ["start_rate"], "combine_position", nb, cfg.seed)
    _save(starter_assoc, cfg, "movement_vs_start_rate.csv", key="metrics")

    # Sensibilidad: la capacidad de percentil 80 descartada vs la media contraída adoptada
    gq = df.groupby("combine_position", observed=True)["burst_2s_q80"]
    sens_df = df.assign(sbi_q80=50 + 10 * (df["burst_2s_q80"] - gq.transform("mean")) / gq.transform("std"),
                        neg_log_pick=-df["log_draft_pick"])
    sens = within_group_spearman(sens_df, ["sbi", "sbi_q80"], ["n_plays", "start_rate", "neg_log_pick"],
                                 "combine_position", nb, cfg.seed)
    _save(sens, cfg, "sbi_estimator_sensitivity.csv", key="metrics")
    bias = estimator_bias(d["plays"], "burst_2s", [3, 5, 10], 200, cfg.seed)
    _save(bias, cfg, "estimator_bias_simulation.csv", key="metrics")

    # Exploratorio (no pre-registrado): SBI vs titularidad donde el capital de draft dice poco (día 3 + no seleccionados)
    day3 = df[df["draft_round_filled"] >= 4]
    day12 = df[df["draft_round_filled"] < 4]
    explo = pd.concat([within_group_spearman(day3, ["sbi"], ["start_rate"], "combine_position", nb, cfg.seed).assign(subgroup="rounds 4-7 + undrafted"),
                       within_group_spearman(day12, ["sbi"], ["start_rate"], "combine_position", nb, cfg.seed).assign(subgroup="rounds 1-3")],
                      ignore_index=True)
    explo["starters"] = [int(day3.is_starter.sum()), int(day12.is_starter.sum())]
    _save(explo, cfg, "exploratory_sbi_by_draft_tier.csv", key="metrics")

    # Tasa de titulares por banda de SBI (IC de Wilson al 95%): la tabla para entrenadores
    bands = pd.cut(df["sbi"], [-np.inf, 45, 55, np.inf], labels=["Below 45", "45-55", "Above 55"])
    tab = df.groupby(bands, observed=True)["is_starter"].agg(["sum", "count"]).reset_index(names="sbi_band")
    z = 1.96
    ph = tab["sum"] / tab["count"]
    denom = 1 + z ** 2 / tab["count"]
    centre = (ph + z ** 2 / (2 * tab["count"])) / denom
    half = z * np.sqrt(ph * (1 - ph) / tab["count"] + z ** 2 / (4 * tab["count"] ** 2)) / denom
    tab["starter_rate"], tab["ci_low"], tab["ci_high"] = ph, centre - half, centre + half
    _save(tab, cfg, "starter_rate_by_sbi_band.csv", key="metrics")

    # GLM logístico: ¿agrega el SBI algo a posición + capital de draft + atletismo del Combine? (razón de verosimilitud)
    # `is_undrafted` queda fuera: ningún jugador no seleccionado con tracking fue titular (separación
    # casi completa, la máxima verosimilitud no converge); log pick ya los ubica en el turno 260.
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
    log.info("traducción: GLM %s", glm)
    out = {"matrix": matrix, "starter_assoc": starter_assoc, "bands": tab, "glm": glm}
    if d.get("onfield") is not None:
        out["onfield"] = stage_onfield_translation(cfg, d)
    if has_combine_tracking(prospects):
        out["combine_tracking"] = stage_combine_tracking_translation(cfg, d)
    return out


def stage_combine_tracking_translation(cfg: Config, d: dict) -> dict:
    """Solo con el tracking de pruebas: tracking vs resultados oficiales, y tracking de pruebas vs el partido.

    Archivos de salida propios, para no tocar las matrices previas ni sus valores q de FDR.
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
        # ¿Agrega una variable del tracking de pruebas algo al 40 oficial para el arranque de la DL? (parciales de rangos)
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
    """Prueba -> rendimiento en el campo, fiabilidad, la cadena de arranque de la DL y validez del SBI."""
    nb, seed = cfg["cv"]["bootstrap_resamples"], cfg.seed
    prospects, onfield, fam = d["prospects"], d["onfield"], d["fam"]
    df = onfield.drop(columns=["combine_position"]).merge(prospects, on="nfl_id", how="inner", validate="1:1")
    df["neg_log_pick"] = -df["log_draft_pick"]
    df["dl_role"] = _role(df["nfl_position"])
    oriented = df.copy()
    for c in TIME_DRILLS:
        oriented[c] = -oriented[c]

    # 1. ¿Es cada métrica de campo un rasgo estable del jugador?
    rel = reliability_table(fam, onfield)
    _save(rel, cfg, "onfield_reliability.csv", key="metrics")
    stab = rookie_stability(fam, cfg)
    _save(stab, cfg, "onfield_stability.csv", key="metrics")

    # 2. Traducción prueba -> campo (dentro de la posición del Combine), con el capital de draft como referencia.
    xs = TRANSLATION_X + ["neg_log_pick"]
    mat = within_group_spearman(oriented, xs, ONFIELD_Y, "combine_position", nb, seed)
    mat["x_oriented"] = mat["x"].isin(TIME_DRILLS)
    _save(mat, cfg, "onfield_translation_matrix.csv", key="metrics")
    dl_role = within_group_spearman(oriented[oriented.combine_position == "DL"], TRANSLATION_X,
                                    ["goe_eb", "proe_eb"], "dl_role", nb, seed)
    _save(dl_role, cfg, "onfield_dl_role_check.csv", key="metrics")

    # 3. Rendimiento en el campo vs titularidad (¿captura la métrica lo que valoran los equipos?)
    assoc = within_group_spearman(df, ONFIELD_Y + ["perf_index"], ["start_rate"], "combine_position", nb, seed)
    _save(assoc, cfg, "onfield_vs_start_rate.csv", key="metrics")

    # 4. La cadena de la DL: prueba -> arranque sobre lo esperado -> presión sobre lo esperado (mediación bootstrap)
    dl = oriented[oriented.combine_position == "DL"]
    med = pd.DataFrame([{"x": x, **mediation_bootstrap(dl, x, "goe_eb", "proe_eb", nb, seed)}
                        for x in ["ten_yd_split", "forty", "broad_jump", "vertical", "combine_weight"]])
    _save(med, cfg, "onfield_mediation_dl.csv", key="metrics")

    # 5. Validez convergente de las métricas de tracking frente al arranque NGS de temporada completa
    v = d["profiles"].merge(onfield[["nfl_id", "goe_eb", "proe_eb"]], on="nfl_id").query("combine_position == 'DL'")
    rows = []
    for x in ["sbi", "get_off_1s_z", "burst_1s_z", "peak_accel_2s_z"]:
        for y in ["goe_eb", "proe_eb"]:
            t = v[[x, y]].dropna()
            r, p = stats.spearmanr(t[x], t[y])
            rows.append({"tracking_metric": x, "onfield_metric": y, "n": len(t), "rho": r, "p_value": p})
    _save(pd.DataFrame(rows), cfg, "sbi_validity.csv", key="metrics")

    # 6. ¿Es el snap burst de WR/DB un rasgo una vez que se quita el contexto de la jugada?
    ctx = pd.read_parquet(cfg.path("interim") / "play_context.parquet")
    tp = d["plays"][d["plays"].nfl_id.isin(d["profiles"].nfl_id)]
    adj = context_adjusted_burst(tp, ctx)
    cr = pd.DataFrame([{"combine_position": pos, "players": g.nfl_id.nunique(), "plays": len(g),
                        "icc_raw": _icc(g, "nfl_id", "burst_2s"), "icc_context_adjusted": _icc(g, "nfl_id", "burst_2s_ctx"),
                        "context_matched_share": float(g["lined_up_position"].notna().mean())}
                       for pos, g in adj.groupby(adj.combine_position.astype(str))])
    _save(cr, cfg, "tracking_context_reliability.csv", key="metrics")

    # 7. GLM logístico: ¿agrega el rendimiento en el campo algo a posición + capital de draft? (diagnóstico, post-draft)
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
    log.info("traducción al campo: GLM %s", glm)
    return {"reliability": rel, "matrix": mat, "mediation": med, "glm": glm}


# =============================================================================
# 5 Regresión: Combine -> SBI y Combine -> rendimiento en el campo
# =============================================================================
REG_MODELS = ["position_mean", "ridge", "random_forest", "xgboost"]


def _summarise(results: dict, metrics: list[str], cfg: Config, task: str):
    """Tabla resumen (media, DE, R^2 agrupado con IC bootstrap, hiperparámetros más elegidos) y puntajes por pliegue."""
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
    """CV anidada de los cuatro regresores en una población, pruebas, reentrenamiento final y registro."""
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
        log.info("regresión %s/%s: %s (%.0fs)", pop, name, res.fold_scores[["mae", "rmse", "r2"]].mean().round(3).to_dict(), time.time() - t0)

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

    # Campeón = menor RMSE medio entre los modelos aprendidos; se reentrena con todas las filas y se registra.
    champ = summary[summary.model != "position_mean"].sort_values("rmse_mean").iloc[0]["model"]
    est, best, score = refit_final(champ, "regression", X, y, grids.get(champ, {}), cfg, cfg.seed)
    with mlflow_run(cfg, f"reg-final-{pop}-{champ}", {"stage": "final", "task": "regression", "population": pop}):
        mlflow.log_params({"model": champ, "population": pop, "target": target,
                           **{k.replace("model__", ""): v for k, v in best.items()}})
        mlflow.log_metric("refit_inner_cv_rmse", -score)
        log_and_register(est, X, "regression", reg_name if cfg["mlflow"]["register_models"] else None, alias)

    # Brecha de traducción: observado menos lo predicho por el Combine (fuera de pliegue, media de repeticiones).
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


# Poblaciones del snap burst: nombre -> (posición del Combine o None para todas, sufijo en el registro).
# Con tracking de temporada completa el snap burst es un rasgo estable en DL (ICC 0.63) y WR (0.62).
SBI_POPULATIONS = {"all_positions": (None, ""), "defensive_line": ("DL", "-dl"), "wide_receivers": ("WR", "-wr")}


def _sbi_population(full: pd.DataFrame, pop: str, k: int) -> tuple[pd.DataFrame, pd.Series, str]:
    """Filas, estratos de CV y sufijo de registro de una población del SBI."""
    pos, suffix = SBI_POPULATIONS[pop]
    if pos is None:
        return full, full["combine_position"].astype(str), suffix
    df = full[full.combine_position.astype(str) == pos]
    strata = df["dl_role"] if pos == "DL" else _strata(_role(df["nfl_position"]), k)
    return df, strata, suffix


def stage_regression(cfg: Config, d: dict) -> dict:
    """Combine -> Snap Burst Index y Combine -> rendimiento en el campo.

    SBI: todas las posiciones, solo línea defensiva y solo receptores (donde el snap burst es un rasgo fiable).
    Campo: una tarea por familia con un resultado medible (configuración: regression.onfield_tasks).
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

    # Atlético vs atlético + tracking de pruebas del Combine, mismas filas y pliegues (solo con combine_tracking.csv)
    ct_runs, ct_rows = [], []
    if cfg["regression"].get("combine_tracking_comparison", True) and has_combine_tracking(d["prospects"]):
        for base in list(runs):
            pop = base["summary"].population.iloc[0]
            if pop == "all_positions":
                continue  # la comparación se hace donde el snap burst es un rasgo (poblaciones de una sola posición)
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
# 6 Clasificación: ¿quién llega a titular?
# =============================================================================
CLF_MODELS = ["logistic_regression", "random_forest", "xgboost"]


def _classification_experiments(d: dict) -> dict:
    """nombre -> (tabla, conjuntos de variables, comparaciones entre conjuntos (nuevo, referencia))."""
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
    """CV anidada de los tres clasificadores x conjuntos de variables, pruebas, DeLong y registro."""
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
                log.info("clasificación %s/%s/%s AUC=%.3f (%.0fs)", exp, fs, name, res.fold_scores["roc_auc"].mean(), time.time() - t0)

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

    # Campeón (pre-draft): un conjunto candidato (proyector o tracking de pruebas) reemplaza a Combine + draft
    # solo si gana en AUC medio Y su prueba de DeLong contra Combine + draft (mismo modelo) da p < 0.05.
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
# 7 Estadística sin reentrenar
# =============================================================================
def ct_best_vs_best(cfg: Config) -> pd.DataFrame | None:
    """¿Mejora el tracking de pruebas al mejor pipeline? Mejor modelo con vs sin, en los mismos pliegues.

    Regresión: campeón = menor RMSE medio entre los modelos aprendidos de cada brazo (como lo elige el
    registro); prueba t pareada corregida sobre el R² y el RMSE de los pliegues externos (tamaños de
    pliegue n(k-1)/k y n/k). Clasificación (todos los prospectos): mejor combine_draft_ct vs mejor
    combine_draft por AUC medio; t corregida sobre los AUC por pliegue y DeLong sobre las predicciones
    fuera de pliegue. Cada brazo elige el mejor de los mismos candidatos, así la selección no favorece a ninguno.
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
    """Recalcula las pruebas ómnibus basadas en rangos desde los puntajes por pliegue guardados.

    Permite aplicar una corrección estadística a una corrida de entrega sin reentrenar: los puntajes por
    pliegue son salidas de esa misma corrida, así las pruebas siguen ligadas a ella.
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
        log.info("estadística: comparación mejor contra mejor del tracking de pruebas escrita")
    log.info("estadística: pruebas de Friedman recalculadas desde los puntajes por pliegue")


# =============================================================================
# 8 Figuras del reporte (PNG, 200 dpi) en reportes/figuras
# =============================================================================
# Reglas de estilo: un gris tranquilo para los datos y un azul de acento para lo que dice el título;
# los mapas divergentes van de azul a naranja alrededor de un gris neutro; cuadrícula fina y discreta;
# cada título enuncia el hallazgo con su número, calculado desde los datos y no escrito a mano.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ACCENT, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED = "#b4b2ab"
SURFACE = "#fcfcfb"
POS_ORDER = ["DB", "DL", "OL", "TE", "WR"]
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_orange", [ORANGE, "#f6c2ab", "#efeeea", "#9ec5f4", ACCENT])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#eef4fc", "#9ec5f4", "#3987e5", "#184f95"])

LABELS = {
    "ten_yd_split": "Parcial de 10 yd", "forty": "40 yardas", "vertical": "Salto vertical", "broad_jump": "Salto largo",
    "three_cone": "3 conos", "short_shuttle": "Shuttle corto", "bench_reps": "Press de banca",
    "combine_weight": "Peso corporal", "ngs_athleticism_score": "Puntaje atlético NGS",
    "burst_2s_eb": "Snap burst\n(distancia a 2 s)", "get_off_1s_eb": "Arranque\n(velocidad a 1 s)",
    "peak_accel_2s_eb": "Acel. máxima\n(primeros 2 s)", "top_speed_eb": "Velocidad\nmáxima", "max_decel_eb": "Frenado\nmáximo",
    "max_cut_deg_eb": "Corte más\ncerrado", "start_rate": "Tasa de\ntitularidad",
    "position_mean": "Media por posición\n(referencia)", "ridge": "Ridge", "random_forest": "Random forest", "xgboost": "XGBoost",
    "logistic_regression": "Reg. logística",
    "ten_split_momentum": "Momento 10 yd (peso/parcial)", "speed_score": "Speed Score",
    "ten_split_share": "Fracción del 40 en 10 yd",
    "ngs_college_production_score": "Puntaje de producción NGS", "age_at_draft": "Edad al draft", "combine_position": "Grupo de posición",
    "log_draft_pick": "Turno de draft (log)", "is_undrafted": "No seleccionado", "wing_span": "Envergadura", "hand_size": "Tamaño de mano",
    "arm_length": "Largo de brazo", "combine_height": "Estatura", "power_conference": "Conferencia fuerte", "n_drills_run": "Pruebas realizadas",
    "sbi": "SBI", "top_speed_z": "Velocidad máxima (partido)", "max_cut_deg_z": "Corte más cerrado (partido)",
    "burst_1s_z": "Distancia a 1 s (partido)", "get_off_1s_z": "Arranque (partido)", "peak_accel_2s_z": "Acel. máxima (partido)",
    "max_decel_z": "Frenado máximo (partido)",
    "goe_eb": "Arranque sobre lo\nesperado (DL)", "proe_eb": "Tasa de presión\nsobre lo esp. (DL)",
    "ppoe_eb": "Prob. de presión\nsobre lo esp. (OL)", "paoe_eb": "Presiones permitidas\nsobre lo esp. (OL)",
    "soe_eb": "Separación sobre\nlo esp. (WR/TE)", "neg_log_pick": "Capital de draft (turno más alto)",
    "goe_z": "Arranque sobre lo esp. (campo)", "proe_z": "Presión sobre lo esp. (campo)", "ppoe_z": "Protección sobre lo esp. (campo)",
    "soe_z": "Separación sobre lo esp. (campo)", "proj_goe": "Arranque proyectado", "proj_proe": "Presión proyectada",
    "proj_ppoe": "Protección proyectada", "proj_soe": "Separación proyectada",
    "onfield_target_goe": "Arranque proyectado", "onfield_target_proe": "Presión proyectada",
    "onfield_target_ppoe": "Protección proyectada", "onfield_target_soe": "Separación proyectada",
    "combine_draft_proj": "Combine + draft\n+ campo proyectado", "combine_draft_perf": "Combine + draft\n+ rendimiento en el campo",
    "combine": "Solo Combine", "combine_draft": "Combine + draft", "combine_draft_mov": "Combine + draft\n+ movimiento en partido",
    "combine_draft_ct": "Combine + draft\n+ tracking de pruebas",
    "ct_forty_split_5_10_s": "40: parcial 5-10 yd (tracking)", "ct_forty_top_speed": "40: velocidad máxima (tracking)",
    "ct_shuttle_plant_speed_ratio": "Shuttle: velocidad en los giros", "ct_skill_speed_z": "Pruebas de posición: velocidad máx.",
    "ct_skill_accel_z": "Pruebas de posición: aceleración",
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
    """Reduce el tamaño de letra de un texto de la figura hasta que quepa entre x0 y x1 (fracción del ancho)."""
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    avail = fig.bbox.width * (x1 - x0)
    if bb.width > avail:
        t.set_fontsize(t.get_fontsize() * avail / bb.width)


def _title(fig, title: str, subtitle: str = "") -> None:
    _fit(fig, fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK))
    if subtitle:
        _fit(fig, fig.text(0.012, 0.935, subtitle, ha="left", va="top", fontsize=9.5, color=INK2))


def _save_fig(fig, cfg: Config, name: str, source: str) -> None:
    fig.text(0.012, 0.008, f"Fuente: {source}", ha="left", va="bottom", fontsize=7.5, color=INK2)
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
    ax.set_xticks(range(len(drills)), [f"{LABELS[d]}\n({overall[d]:.0%} en total)" for d in drills], fontsize=8.5)
    ax.set_yticks(range(len(POS_ORDER)), POS_ORDER)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    agil = prospects[["short_shuttle", "three_cone"]].notna().any(axis=1).mean()
    _title(fig, f"Agilidad, el punto ciego: solo {agil:.0%} de los prospectos hizo el shuttle o el 3 conos",
           f"Proporción de prospectos con resultado registrado, por grupo de posición del Combine (n = {len(prospects)})")
    _save_fig(fig, cfg, "fig01_drill_coverage.png", "combine_results.csv, clases 2023-2025")


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
    _title(fig, f"40 y parcial de 10 yd casi duplicados (rho = {r:.2f}); press de banca aparte",
           "Correlación de Spearman dentro de la posición; tiempos con signo invertido para que + signifique 'ambos mejores'")
    _save_fig(fig, cfg, "fig02_drill_correlations.png", "combine_results.csv, sin las medias del grupo de posición")


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
        ax.scatter(i + jit[~ok], sub.burst_2s[~ok], s=16, color=ACCENT, linewidths=0, label="Atípico marcado (|z robusto| > 3.5)" if i == 0 else None)
        ax.text(i, -0.9, f"n = {len(sub):,}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(POS_ORDER)), POS_ORDER)
    ax.set_ylabel("Distancia desde el punto del snap en t = 2.0 s (yd)")
    ax.set_ylim(-1.4, plays.burst_2s.max() + 0.8)
    ax.legend(loc="upper left", fontsize=8.5)
    med = plays.groupby("combine_position", observed=True).burst_2s.median()
    _title(fig, f"Los WR recorren {med['WR']:.1f} yd en 2 s y la OL {med['OL']:.1f} yd; {int(plays.flag.sum())} jugadas atípicas",
           "Snap burst de 2 segundos por jugada y grupo de posición; caja = mediana y rango intercuartílico, bigotes 1.5 RIC")
    _save_fig(fig, cfg, "fig03_burst_distribution.png", f"game_tracking, temporada regular + playoffs, {len(plays):,} jugador-jugadas")


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
        ax.text(tx, ty, f"{names[pid]} ({roles[pid]}): SBI {p.sbi:.0f}, {int(p.n_plays)} jugadas",
                fontsize=8.5, color=INK, fontweight="bold" if lab == "high" else "normal")
    ax.axvline(2.0, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.text(2.04, 0.2, "El SBI se mide aquí (t = 2.0 s)", fontsize=8, color=INK2)
    ax.set_xlabel("Segundos después del snap")
    ax.set_ylabel("Velocidad (yd/s)")
    ax.set_xlim(0, 4)
    _title(fig, f"Dos edge rushers: uno llega a {peaks['high']:.1f} yd/s dentro de los 2 s, el otro a {peaks['low']:.1f}",
           "Velocidad tras el snap en cada jugada registrada (fina) y la traza mediana (gruesa); mayor vs menor SBI entre DE/OLB con 8+ jugadas")
    _save_fig(fig, cfg, "fig04_speed_traces.png", "game_tracking, temporada regular + playoffs")


def fig_estimator_bias(cfg, bias, sens):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.14, wspace=0.35)
    colors = {"q80": ORANGE, "median": MUTED, "mean": ACCENT}
    names = {"q80": "Percentil 80 (descartado)", "median": "Mediana", "mean": "Media (adoptada)"}
    for est in ["q80", "median", "mean"]:
        b = bias[bias.estimator == est].sort_values("n_plays")
        a1.plot(b.n_plays, b.bias, marker="o", color=colors[est], linewidth=2, markersize=6)
        dy = {"mean": 0.022, "median": -0.022, "q80": 0.0}[est]
        a1.text(b.n_plays.iloc[-1] + 0.4, b.bias.iloc[-1] + dy, names[est], va="center", fontsize=8.5, color=INK)
    a1.axhline(0, color=INK2, linewidth=0.8)
    a1.set_xlim(2, 15.5)
    a1.set_xticks([3, 5, 10])
    a1.set_xlabel("Jugadas observadas del jugador")
    a1.set_ylabel("Sesgo vs muestra completa (yd)")
    a1.set_title("A. Sesgo con pocas jugadas por jugador")
    s = sens.set_index(["x", "y"])
    ys = [("n_plays", "Jugadas\nobservadas"), ("start_rate", "Tasa de\ntitularidad"), ("neg_log_pick", "Capital\nde draft")]
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
    a2.set_ylabel("rho de Spearman en la posición (IC 95%)")
    a2.set_title("B. Con qué se correlaciona el SBI")
    a2.legend(fontsize=8, loc="upper right")
    q = bias[(bias.estimator == "q80") & (bias.n_plays == 3)].bias.iloc[0]
    _title(fig, f"El percentil alto lee {abs(q):.2f} yd de menos con 3 jugadas y filtra el tiempo de juego a la métrica",
           "Submuestreo de jugadores con 20+ jugadas; el panel B compara el SBI construido con cada estimador")
    _save_fig(fig, cfg, "fig05_estimator_bias.png", "play_metrics + careers; 200 submuestras por jugador y n")


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
    _title(fig, "Sprint y salto se trasladan al movimiento en partido; el compuesto NGS y el press de banca no",
           f"rho de Spearman dentro de la posición (* = FDR q < 0.05). Mejor para el snap burst: {LABELS[best.x]} ({best.rho:+.2f})")
    _fit(fig, fig.text(0.24, 0.07, "Azul = mejor prueba con más movimiento en partido; naranja = lo contrario. El peso va tal como se mide (más pesado = más).",
                       fontsize=8, color=INK2), x0=0.24)
    _save_fig(fig, cfg, "fig07_translation_matrix.png", f"combine_results + game_tracking, {int(matrix.n.max())} jugadores con 3+ jugadas")


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
    axes[0].set_ylabel("Proporción de partidos como titular")
    _title(fig, f"El snap burst en partido apenas sigue a la titularidad dentro de la posición (rho = {rho.rho:+.2f}, IC 95% {rho.ci_low:+.2f} a {rho.ci_high:+.2f})",
           "Cada punto es un jugador con 3+ jugadas registradas; línea azul = ajuste de mínimos cuadrados dentro del grupo de posición")
    _save_fig(fig, cfg, "fig09_sbi_vs_start_rate.png", "game_tracking + player_career_successes")


def fig_reliability(cfg, rel):
    metrics = ["burst_1s", "burst_2s", "get_off_1s", "peak_accel_2s", "top_speed", "max_decel", "max_cut_deg"]
    names = {"burst_1s": "Distancia a 1 s", "burst_2s": "Snap burst\n(distancia a 2 s)", "get_off_1s": "Arranque\n(velocidad a 1 s)",
             "peak_accel_2s": "Acel. máxima\n(primeros 2 s)", "top_speed": "Velocidad máxima", "max_decel": "Frenado máximo",
             "max_cut_deg": "Corte más cerrado"}
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
            txt = "nulo" if raw <= 0 else f"{v:.2f}"
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
    _title(fig, f"El snap burst es un rasgo estable del jugador en {' y '.join(f'{p} ({b[p]:.2f})' for p in stable)}; "
                f"más débil en {', '.join(weak)}" if stable else "El snap burst no es un rasgo estable en ninguna posición",
           "Proporción de la varianza entre jugadores que es diferencia real (ICC); * = ANOVA p < 0.01; 'nulo' = no detectable")
    _save_fig(fig, cfg, "fig06_metric_reliability.png", f"game_tracking, {sum(n)} jugadores con 3+ jugadas")


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
        ax.scatter(a.rho, i, s=55, color=INK2, zorder=3, label="Bruto (toda la DL)" if i == 0 else None)
        ax.scatter(b.rho, i, s=55, color=ACCENT, zorder=3, label="Dentro del rol (edge vs interior), IC 95%" if i == 0 else None)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(xs)), [f"{LABELS[x]} (n={int(t.loc[('none', x), 'n'])})" for x in xs])
    ax.invert_yaxis()
    ax.set_xlabel("rho de Spearman con el arranque en partido (velocidad 1 s tras el snap); tiempos con signo invertido")
    ax.legend(loc="lower left", fontsize=8.5)
    raw = t.loc[("none", "ten_yd_split"), "rho"]
    adj = t.loc[("edge vs interior role", "ten_yd_split"), "rho"]
    _title(fig, f"En la DL, la mitad del vínculo es el rol: 10 yd {raw:+.2f} bruto, {adj:+.2f} dentro del rol",
           "Edge = DE/OLB, interior = DT/NT; los linieros interiores, más pesados, corren más lento y también arrancan más lento por su asignación")
    _save_fig(fig, cfg, "fig08_dl_translation_by_role.png", "linieros defensivos con 3+ jugadas registradas")


def fig_regression_cv(cfg, m_dir, summary):
    order = ["position_mean", "ridge", "random_forest", "xgboost"]
    titles = {"all_positions": "Todas las posiciones", "defensive_line": "Línea defensiva", "wide_receivers": "Receptores abiertos"}
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
    axes[0].set_ylabel("R² del SBI en los pliegues externos (recortado en -0.8)")
    short = {"all_positions": "todas las posiciones", "defensive_line": "DL", "wide_receivers": "WR"}
    cut = {p: 1 - best[p].rmse_mean / summary[(summary.population == p) & (summary.model == "position_mean")].rmse_mean.iloc[0]
           for p in pops}
    _title(fig, "Los resultados del Combine reducen el error del snap burst frente a solo la posición en " +
           ", ".join(f"{cut[p]:.0%} ({short[p]})" for p in sorted(pops, key=lambda p: -cut[p])),
           "CV anidada, 5 pliegues x 3 repeticiones; puntos = pliegues externos, marcador = media con IC 95%; azul = modelo aprendido de menor RMSE")
    _save_fig(fig, cfg, "fig10_regression_cv.png", "Snap Burst Index a partir de las medidas y pruebas del Combine")


EXPERIMENT_TITLES = {"all_prospects": "Todos los prospectos", "tracked_subset": "Jugadores con tracking",
                     "onfield_subset": "Jugadores con métricas de campo"}
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
    axes[0].set_ylabel("ROC-AUC, pliegues externos (media, IC 95%)")
    g = folds.groupby(["population", "feature_set"]).roc_auc.mean()
    t = f"El capital de draft sube el AUC de {g[('all_prospects', 'combine')]:.2f} a {g[('all_prospects', 'combine_draft')]:.2f}"
    if ("all_prospects", "combine_draft_ct") in g.index:
        t += f"; el tracking de pruebas suma {g[('all_prospects', 'combine_draft_ct')] - g[('all_prospects', 'combine_draft')]:+.3f}"
    if ("onfield_subset", "combine_draft_perf") in g.index:
        t += f"; el rendimiento en el campo {g[('onfield_subset', 'combine_draft_perf')] - g[('onfield_subset', 'combine_draft')]:+.3f}"
    _title(fig, t, "Predicción de titular (50%+ de los partidos disponibles como titular); CV anidada, 5 pliegues x 3 repeticiones; mismos pliegues dentro de cada panel")
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
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos" if k == 0 else "")
        ax.set_title(f"{chr(65 + k)}. {EXPERIMENT_TITLES[pop]} ({' / '.join(dict.fromkeys(models))})", fontsize=9.5)
        ax.legend(loc="lower right", fontsize=7, title="ROC-AUC [IC 95%]", title_fontsize=7)
    _title(fig, "Curvas ROC fuera de pliegue del mejor modelo de cada conjunto de variables",
           "Predicciones promediadas en las 3 repeticiones de la CV; corchetes = IC 95% bootstrap (2,000 remuestreos estratificados)")
    _save_fig(fig, cfg, "fig12_roc_curves.png", "predicciones fuera de pliegue de la CV anidada")


def _nice(f):
    if f.endswith("_missing"):
        return f"{LABELS.get(f[:-8], f[:-8]).replace(chr(10), ' ')} no realizada"
    return LABELS.get(f, f).replace("\n", " ").replace("_", " ")


def fig_importance(cfg, imp_clf, imp_reg, champ_clf_model, reg_champ, reg_pop="dl_getoff", reg_label="arranque de la DL",
                   unit="aumento del RMSE (s)"):
    imp_reg = imp_reg[imp_reg.population == reg_pop]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.8, bottom=0.15, wspace=0.75)
    for ax, imp, title, u in [(a1, imp_clf, f"A. Clasificador de titulares ({LABELS[champ_clf_model]})", "caída del ROC-AUC"),
                              (a2, imp_reg, f"B. {reg_label[0].upper() + reg_label[1:]} ({LABELS[reg_champ]})", unit)]:
        top = imp.head(10).iloc[::-1]
        cols = [ACCENT if i == len(top) - 1 else MUTED for i in range(len(top))]
        ax.barh(range(len(top)), top.importance, xerr=top["std"], color=cols, ecolor=INK2, capsize=2, height=0.65)
        ax.set_yticks(range(len(top)), [_nice(f) for f in top.feature], fontsize=8.5)
        ax.set_xlabel(f"Importancia por permutación ({u})")
        ax.set_title(title)
        ax.axvline(0, color=INK2, linewidth=0.8)
    lead = _nice(imp_reg.iloc[0].feature)
    lead = lead[0].lower() + lead[1:] if len(lead) > 1 and lead[1].islower() else lead
    _title(fig, f"El turno de draft domina quién es titular; {lead} encabeza el modelo de {reg_label}",
           "Importancia por permutación en los modelos campeones reentrenados (dentro de muestra, descriptiva), 20 permutaciones; barras = media, líneas = DE")
    _save_fig(fig, cfg, "fig13_feature_importance.png", "modelos campeones registrados")


def fig_translation_gap(cfg, gap):
    g = gap[(gap.population == "defensive_line") & (gap.n_plays >= 10)].dropna(subset=["translation_gap"]).sort_values("translation_gap")
    pick = pd.concat([g.head(6), g.tail(6)])
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    fig.subplots_adjust(left=0.34, right=0.95, top=0.85, bottom=0.1)
    cols = [ACCENT if v > 0 else ORANGE for v in pick.translation_gap]
    ax.barh(range(len(pick)), pick.translation_gap, color=cols, height=0.7, alpha=0.9)
    ax.set_ylim(-0.7, len(pick) + 0.2)
    ax.set_yticks(range(len(pick)), [f"{r.display_name} ({r.nfl_position}, {int(r.draft_year)}, {r.n_plays} jugadas)" for r in pick.itertuples()], fontsize=8.5)
    for i, r in enumerate(pick.itertuples()):
        ax.text(r.translation_gap + (0.6 if r.translation_gap > 0 else -0.6), i, f"{r.sbi:.0f} vs {r.predicted_sbi:.0f}",
                va="center", ha="left" if r.translation_gap > 0 else "right", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, linewidth=0.9)
    ax.set_xlabel("Brecha de traducción = SBI en partido - SBI predicho por el Combine (puntos)")
    lim = np.abs(pick.translation_gap).max() + 6
    ax.set_xlim(-lim, lim)
    ax.text(lim * 0.97, len(pick) - 0.4, "Juega más rápido de lo que midió", ha="right", fontsize=8.5, color=ACCENT, fontweight="bold")
    ax.text(-lim * 0.97, len(pick) - 0.4, "Midió más rápido de lo que juega", ha="left", fontsize=8.5, color=ORANGE, fontweight="bold")
    _title(fig, "Linieros defensivos cuyo snap burst más se aparta de lo que predijeron sus pruebas",
           "Las seis mayores brechas positivas y negativas entre linieros con 10+ jugadas; etiquetas = SBI en partido vs SBI predicho (fuera de pliegue)")
    _save_fig(fig, cfg, "fig14_translation_gap.png", "predicciones fuera de pliegue del regresor campeón (CV anidada)")


ONFIELD_NAMES = {"goe": "Arranque DL sobre lo esperado", "proe": "Tasa de presión DL sobre lo esperado",
                 "ppoe": "Prob. de presión OL sobre lo esperado", "paoe": "Presiones permitidas OL sobre lo esp.",
                 "soe": "Separación sobre lo esperado"}


def fig_onfield_reliability(cfg, rel, track_rel, stab):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=0.24, right=0.97, top=0.8, bottom=0.13, wspace=0.6)
    rows = [(f"{ONFIELD_NAMES[r.metric]} ({r.combine_position})" if r.metric == "soe" else ONFIELD_NAMES[r.metric],
             max(r.icc, 0), int(r.players), "on-field") for r in rel.itertuples()]
    tb = track_rel[track_rel.metric == "burst_2s"]
    rows += [(f"Snap burst de tracking ({r.combine_position})", max(r.icc, 0), int(r.players), "tracking") for r in tb.itertuples()]
    rows.sort(key=lambda t: t[1])
    for i, (name, v, n, kind) in enumerate(rows):
        a1.barh(i, v, color=ACCENT if kind == "on-field" else MUTED, height=0.68)
        a1.text(v + 0.015, i, f"{v:.2f}  (n={n})", va="center", fontsize=8, color=INK)
    a1.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    a1.set_xlim(0, 1.12)
    a1.set_xlabel("ICC: proporción de la varianza que es diferencia real entre jugadores")
    a1.set_title("A. ¿Rasgo estable? Azul = campo (temporada), gris = SBI de tracking", fontsize=9.5)
    st = stab.sort_values("rho")
    a2.barh(range(len(st)), st.rho, color=[ACCENT if p < 0.05 else MUTED for p in st.p_value], height=0.6)
    for i, r in enumerate(st.itertuples()):
        a2.text(r.rho + 0.02, i, f"{r.rho:+.2f} (n={r.players})", va="center", fontsize=8, color=INK)
    a2.set_yticks(range(len(st)), [ONFIELD_NAMES[m].replace(" sobre lo esperado", "").replace(" sobre lo esp.", "") for m in st.metric], fontsize=8.5)
    a2.set_xlim(min(0, st.rho.min()) - 0.05, 1.0)
    a2.axvline(0, color=INK2, linewidth=0.8)
    a2.set_xlabel("rho de Spearman, novato vs después")
    a2.set_title("B. Novato vs temporadas siguientes", fontsize=9.5)
    g = rel.set_index(["metric", "combine_position"]).icc
    tr = [r.combine_position for r in tb.itertuples() if r.icc >= 0.5]
    _title(fig, f"Las métricas de campo de temporada son rasgos estables (arranque DL ICC {g[('goe', 'DL')]:.2f}); "
                f"el burst de 2 s lo es en {' y '.join(tr) if tr else 'ninguna posición'}",
           "ICC por método de momentos entre jugadores con el mínimo de jugadas; panel B: clases 2023-2024, azul = p < 0.05")
    _save_fig(fig, cfg, "fig15_onfield_reliability.png", "player_play temporada regular 2023-2025 + game_tracking")


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
    _title(fig, f"Las pruebas predicen el arranque DL ({LABELS[g.x].lower()} rho {g.rho:+.2f}); la protección OL sigue al draft",
           "rho de Spearman dentro de la posición, Combine vs métrica de campo (* = FDR q < 0.05); tiempos con signo invertido")
    _fit(fig, fig.text(0.27, 0.05, "Azul = mejor prueba (o turno más alto) con mejor rendimiento en el campo; naranja = lo contrario. Peso tal como se mide.",
                       fontsize=8, color=INK2), x0=0.27)
    _save_fig(fig, cfg, "fig16_onfield_translation.png", "combine_results + player_play temporada regular 2023-2025")


def fig_dl_chain(cfg, onfield, prospects, med):
    df = onfield.drop(columns=["combine_position"]).merge(prospects, on="nfl_id")
    df = df[(df.combine_position.astype(str) == "DL") & df.goe_eb.notna()]
    role = df.nfl_position.astype(str).isin(["DT", "NT"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.14, wspace=0.25)
    for ax, x, y, xl, yl in [(a1, "ten_yd_split", "goe_eb", "Parcial de 10 yd (s)", "Arranque sobre lo esperado (s; + = más rápido)"),
                             (a2, "goe_eb", "proe_eb", "Arranque sobre lo esperado (s; + = más rápido)", "Tasa de presión sobre lo esperado")]:
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
    a1.set_title("A. Parcial de 10 yd más rápido, arranque más rápido en partido")
    a2.set_title("B. Arranque más rápido, más presión")
    a1.legend(loc="lower left", fontsize=8)
    r = med.set_index("x").loc["ten_yd_split"]
    _title(fig, f"Cerca del {r.share_mediated:.0%} del vínculo del parcial de 10 yd con la presión pasa por el arranque en partido",
           f"Mediación sobre z: efecto indirecto {abs(r.indirect_effect):.2f} DE [{abs(r.indirect_ci_high):.2f}, {abs(r.indirect_ci_low):.2f}] (IC 95% bootstrap); "
           "linieros defensivos con 20+ snaps de presión" if r.indirect_effect < 0 else
           f"Mediación sobre z: efecto indirecto {r.indirect_effect:.2f} DE [{r.indirect_ci_low:.2f}, {r.indirect_ci_high:.2f}] (IC 95% bootstrap); "
           "linieros defensivos con 20+ snaps de presión")
    _save_fig(fig, cfg, "fig17_dl_chain.png", "combine_results + player_play temporada regular 2023-2025")


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
        ax.set_xticks(range(len(order)), [LABELS[m].replace("\n(referencia)", "") for m in order], fontsize=7.5, rotation=30, ha="right")
        ax.set_title(f"{chr(65 + k)}. {task['label']}\n(n = {int(s.n.iloc[0])})", fontsize=8.5)
        ax.set_xlim(-0.5, len(order) - 0.1)
        ax.set_ylim(-0.8, 0.8)
    axes[0].set_ylabel("R² en los pliegues externos (recortado en -0.8)")
    short = {"dl_getoff": "el arranque DL", "dl_pressure": "la presión DL", "ol_protection": "la protección OL",
             "wrte_separation": "la separación WR/TE"}
    yes = [f"{short.get(k, k)} (R² {v.r2_mean:+.2f})" for k, v in best.items() if v.r2_mean > 0.05]
    no = [short.get(k, k) for k, v in best.items() if v.r2_mean <= 0.05]
    _title(fig, f"Las pruebas del Combine predicen {' y '.join(yes)}; no {' ni '.join(no)}" if yes and no else
           "Pruebas del Combine vs rendimiento en el campo por familia",
           "CV anidada, 5 pliegues x 3 repeticiones; puntos = pliegues externos, marcador = media con IC 95%; azul = modelo aprendido de menor RMSE")
    _save_fig(fig, cfg, "fig18_onfield_regression.png", "Combine -> rendimiento en el campo de temporada (player_play)")


CT_GROUP_TITLES = {"forty": "40 yardas", "3cone": "3 conos", "shuttle": "Shuttle corto", "skill": "Pruebas de posición"}
CT_METRIC_NAMES = {"split_0_5_s": "Parcial 0-5 yd", "split_5_10_s": "Parcial 5-10 yd", "split_10_20_s": "Parcial 10-20 yd",
                   "split_10yd_s": "Parcial de 10 yd desde la salida", "top_speed": "Velocidad máxima", "peak_accel": "Aceleración máxima",
                   "burst_1s": "Distancia a 1 s", "burst_2s": "Distancia a 2 s", "t90_top_s": "Tiempo al 90% de la velocidad máxima",
                   "duration_s": "Tiempo de la prueba", "peak_decel": "Frenado máximo", "plant_speed_ratio": "Velocidad conservada en los giros",
                   "max_cut_deg": "Corte más cerrado"}


def fig_ct_reliability(cfg, rel):
    """Qué métricas del tracking de pruebas se repiten dentro de un jugador (el filtro que elige las variables)."""
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
        ax.text(-0.02, y - 0.35, f"{CT_GROUP_TITLES[g]} (n = {int(sub.n_players.max())} jugadores)", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8.5, fontweight="bold", color=INK)
        seps.append(y - 0.5)
        y += 1.2
    ax.axvline(0.5, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.505, y - 0.6, "filtro r = 0.5", fontsize=8, color=INK2, va="top")
    ax.set_yticks(ticks, labels, fontsize=8.5)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.8, y - 0.4)
    ax.set_xlabel("Fiabilidad (r test-retest; mitades con Spearman-Brown en las pruebas de posición)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=ACCENT, label="Variable del modelo"), Patch(color=INK2, label="Se repite, pero otra métrica la cubre"),
                        Patch(color=MUTED, label="Bajo el filtro: no se usa")], loc="upper left", bbox_to_anchor=(0.33, 0.885),
               ncol=3, fontsize=8.5, frameon=False)
    f = rel.set_index(["drill_group", "metric"])
    _title(fig, f"Los parciales entre marcas se repiten (5-10 yd r = {f.loc[('forty', 'split_5_10_s'), 'reliability']:.2f}); "
                f"la salida registrada no (0-5 yd r = {f.loc[('forty', 'split_0_5_s'), 'reliability']:.2f})",
           "Qué tan bien se repite cada métrica del tracking de pruebas dentro de un jugador; con r >= 0.5 puede ser variable, una por constructo")
    _save_fig(fig, cfg, "fig19_ct_reliability.png",
              f"combine_tracking: {int(rel.loc[rel.drill_group == 'forty', 'n_players'].max())} jugadores con dos 40 registrados")


def _ci_from_t(mean_diff, t, df, level=0.95):
    se = abs(mean_diff / t) if t and np.isfinite(t) and t != 0 else np.nan
    return se * stats.t.ppf(0.5 + level / 2, df)


def fig_ct_value(cfg, m_dir, clf_tests, n_folds):
    """¿Mejora el tracking de pruebas la predicción fuera de muestra? Mismas filas y pliegues, con vs sin."""
    bvb = pd.read_csv(m_dir / "ct_best_vs_best.csv")
    names = {"defensive_line": "Snap burst DL (SBI)", "wide_receivers": "Snap burst WR (SBI)", "dl_getoff": "Arranque DL sobre lo esperado",
             "dl_pressure": "Presión DL sobre lo esperado", "ol_protection": "Protección OL sobre lo esperado",
             "wrte_separation": "Separación WR/TE sobre lo esperado"}
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
        rows.append({"label": f"Mejor vs mejor\n{LABELS[c.model_without]} -> {LABELS[c.model_with]}", "d": c.roc_auc_diff,
                     "lo": c.roc_auc_diff_ci_low, "hi": c.roc_auc_diff_ci_high, "p": c.roc_auc_p_corrected_t})
    r2 = pd.DataFrame(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=0.2, right=0.97, top=0.78, bottom=0.14, wspace=0.5)
    for ax, r, xl, title, fmt in [(a1, r1, "Cambio en el R² de los pliegues externos (con - sin tracking de pruebas)",
                                   "A. Combine -> métricas de juego (mejor modelo de cada brazo)", "{:+.3f}"),
                                  (a2, r2, "Cambio en el ROC-AUC de los pliegues externos", "B. Clasificador de titulares, Combine + draft", "{:+.3f}")]:
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
        head = f"El tracking de pruebas mejora {', '.join(up)}; nada más mejora"
    elif best_up.d > 0.01:
        head = (f"El tracking de pruebas sube el R² de {best_up.label.split(chr(10))[0].replace(' sobre lo esperado', '')} "
                f"{best_up.base:+.2f} -> {best_up.new:+.2f} ({int(best_up.wins)}/{n_folds} pliegues, p = {best_up.p:.2f}); nada más mejora")
    else:
        head = (f"El tracking de pruebas no mejora ningún modelo: cambio de R² {r1.d.min():+.3f} a {r1.d.max():+.3f}, "
                f"AUC de titulares {cbest.d:+.3f}")
    _title(fig, head,
           f"Mismos jugadores y pliegues con y sin las 5 variables del tracking de pruebas; media de {n_folds} pliegues externos, "
           "IC 95% de la prueba t corregida; azul = mejora significativa")
    _save_fig(fig, cfg, "fig20_ct_value.png", "puntajes por pliegue de la CV anidada: ct_best_vs_best.csv, classification_pairwise_tests.csv")


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
    short_x = {"ct_forty_split_5_10_s": "la aceleración del 40 (5-10 yd)", "ct_forty_top_speed": "la velocidad máxima del 40",
               "ct_shuttle_plant_speed_ratio": "los giros del shuttle", "ct_skill_speed_z": "la velocidad en pruebas de posición",
               "ct_skill_accel_z": "la aceleración en pruebas de posición"}
    short_y = {"goe_eb": "el arranque DL", "proe_eb": "la presión DL", "ppoe_eb": "la protección OL", "paoe_eb": "las presiones permitidas OL",
               "soe_eb": "la separación WR/TE"}
    sig = mat[(mat.q_value < 0.05) & mat.x.isin(CT_FEATURES) & mat.y.isin(ys)].sort_values("rho", key=np.abs, ascending=False)
    if len(sig):
        g = sig.iloc[0]
        sx = short_x[g.x]
        t = f"{sx[0].upper() + sx[1:]} va con {short_y[g.y]} (rho {g.rho:+.2f})"
        t += f"; otras {len(sig) - 1} celdas pasan el FDR" if len(sig) > 2 else ("; otra celda pasa el FDR" if len(sig) == 2 else "; ninguna otra celda pasa el FDR")
    else:
        t = "Ninguna variable del tracking de pruebas se asocia con el rendimiento en el campo tras corregir por FDR"
    _title(fig, t, "rho de Spearman dentro de la posición, tracking de pruebas vs métrica de campo (* = FDR q < 0.05); tiempo parcial con signo invertido")
    _fit(fig, fig.text(0.27, 0.06, "Azul = mejor movimiento en la prueba con mejor rendimiento en el campo; naranja = lo contrario.", fontsize=8, color=INK2), x0=0.27)
    _save_fig(fig, cfg, "fig21_ct_onfield_translation.png", "combine_tracking + player_play temporada regular 2023-2025")


def make_all_figures(cfg: Config) -> None:
    """Dibuja todas las figuras disponibles a partir de los datos limpios y las métricas guardadas."""
    _style()
    p, m, i = cfg.path("processed"), cfg.path("metrics"), cfg.path("interim")
    prospects = pd.read_parquet(p / "prospects.parquet")
    plays = pd.read_parquet(p / "play_metrics.parquet")
    profiles = pd.read_parquet(p / "player_movement_profiles.parquet")
    # Solo los cuadros que usa la figura de trazas de velocidad (jugadores DL), así el tracking completo no entra en memoria.
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
            pop, label, unit = "dl_getoff", "arranque de la DL", "aumento del RMSE (s)"
        else:
            pop, label, unit = "defensive_line", "snap burst de la DL", "aumento del RMSE (puntos SBI)"
        rc = reg[(reg.population == pop) & (reg.model != "position_mean")].sort_values("rmse_mean").iloc[0].model
        fig_importance(cfg, pd.read_csv(m / f"classification_importance_all_prospects_{champ_entry['feature_set']}.csv"),
                       pd.read_csv(m / "regression_permutation_importance.csv"), champ_entry["model"], rc, pop, label, unit)


# =============================================================================
# Línea de comandos
# =============================================================================
STAGES = ["todo", "contrato", "datos", "traduccion", "modelos", "estadistica", "figuras", "reubicar"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Entrenamiento, evaluación y registro en MLflow del proyecto NFL")
    ap.add_argument("--config", default=None, help="archivo de configuración (por defecto configuracion/config.yaml)")
    ap.add_argument("--etapa", default="todo", choices=STAGES,
                    help="todo = datos + traducción + modelos + figuras; modelos = traducción + modelos sobre datos limpios")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    if args.etapa == "reubicar":
        log.info("reubicar: %d celdas del almacén MLflow apuntan ahora a esta carpeta", relocate_store(cfg))
        return
    if args.etapa == "contrato":
        stage_contract(cfg)
        return
    mlflow_setup(cfg)
    log.info("NFL v%s semilla=%d", __version__, cfg.seed)

    if args.etapa == "estadistica":
        stage_stats(cfg)
        return
    if args.etapa == "figuras":
        make_all_figures(cfg)
        log.info("listo")
        return
    d = stage_data(cfg) if args.etapa in ("todo", "datos") else load_processed(cfg)
    if args.etapa == "datos":
        return
    if args.etapa in ("todo", "traduccion", "modelos"):
        stage_translation(cfg, d)
    if args.etapa in ("todo", "modelos"):
        stage_regression(cfg, d)
        stage_classification(cfg, d)
        ct_best_vs_best(cfg)
    if args.etapa == "todo":
        make_all_figures(cfg)
    log.info("listo")


if __name__ == "__main__":
    # Se ejecuta a través del módulo importado `train` para que las clases de los pipelines
    # (PositionMedianImputer, OnFieldProjector) se serialicen como `train.<Clase>` y no como
    # `__main__.<Clase>`: así los modelos registrados cargan desde el notebook o cualquier script.
    import train

    train.main()
