"""Statistical comparison utilities.

- corrected_resampled_ttest  Nadeau & Bengio (2003): paired t-test on repeated-CV
                             fold scores with the variance inflated for overlapping
                             training sets (the naive paired t-test is too liberal).
- wilcoxon_holm              non-parametric pairwise check with Holm correction.
- friedman                   omnibus test that k models rank equally across folds.
- bootstrap_ci               percentile CI of any metric on out-of-fold predictions.
- delong_test                DeLong (1988) test for two correlated ROC-AUCs.
- within_group_spearman      drill-vs-game association net of position group,
                             bootstrap CI + Benjamini-Hochberg q-values.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


def corrected_resampled_ttest(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    j = len(d)
    var = d.var(ddof=1)
    se = np.sqrt((1 / j + n_test / n_train) * var) if var > 0 else np.nan
    t = d.mean() / se if se and se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), df=j - 1) if np.isfinite(t) else np.nan
    return {"mean_diff": d.mean(), "t": t, "df": j - 1, "p_value": p}


def holm(pvals: list[float]) -> np.ndarray:
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
    """scores: one column per model, one row per outer fold (same folds for all)."""
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


# ----------------------------------------------------------------------------- DeLong
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
    """Two-sided DeLong test for AUC(p1) vs AUC(p2) on the same samples."""
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


# ----------------------------------------------------------------------------- mediation
def mediation_bootstrap(df: pd.DataFrame, x: str, m: str, y: str, n_boot: int, seed: int) -> dict:
    """Product-of-coefficients mediation x -> m -> y on z-scored variables, percentile bootstrap.

    a = slope of m on x; b = slope of y on m controlling for x; indirect = a x b.
    total = slope of y on x; direct = slope of x in y ~ x + m. Coefficients are in SD units.
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


# ----------------------------------------------------------------------------- translation
def within_group_spearman(df: pd.DataFrame, xs: list[str], ys: list[str], group: str,
                          n_boot: int, seed: int) -> pd.DataFrame:
    """Spearman rho between x and y after removing position-group means from both."""
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
    out = pd.DataFrame(rows)
    out["q_value"] = stats.false_discovery_control(out["p_value"].to_numpy(), method="bh")
    return out
