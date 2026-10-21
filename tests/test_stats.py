"""Statistical tests: corrected t, DeLong, Holm, bootstrap and partial correlation."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from train import bootstrap_ci, corrected_resampled_ttest, delong_test, holm


def test_corrected_t_is_more_conservative_than_naive():
    rng = np.random.default_rng(0)
    a = 0.80 + rng.normal(0, 0.02, 15)
    b = a - 0.01 + rng.normal(0, 0.01, 15)
    naive = stats.ttest_rel(a, b).pvalue
    corrected = corrected_resampled_ttest(a, b, n_train=400, n_test=100)["p_value"]
    assert corrected > naive


def test_delong_matches_sklearn_and_is_zero_for_identical():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p1 = y * 0.5 + rng.normal(0, 0.5, 300)
    p2 = y * 0.2 + rng.normal(0, 0.5, 300)
    res = delong_test(y, p1, p2)
    assert abs(res["auc_1"] - roc_auc_score(y, p1)) < 1e-9
    assert abs(res["auc_2"] - roc_auc_score(y, p2)) < 1e-9
    assert res["p_value"] < 0.05
    same = delong_test(y, p1, p1)
    assert same["diff"] == 0


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert np.all(adj <= 1) and np.all(adj >= np.array([0.01, 0.04, 0.03, 0.5]))
    assert adj[0] == 0.04  # smallest p x 4


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 200)
    pred = y + rng.normal(0, 0.5, 200)
    est, lo, hi = bootstrap_ci(y, pred, lambda a, b: float(np.sqrt(np.mean((a - b) ** 2))), 300, seed=3)
    assert lo <= est <= hi


def test_partial_spearman_removes_common_cause():
    from train import partial_spearman
    rng = np.random.default_rng(3)
    c = rng.normal(size=300)
    df = pd.DataFrame({"x": c + rng.normal(0, 0.3, 300), "y": c + rng.normal(0, 0.3, 300), "c": c})
    raw = partial_spearman(df, "x", "y", [])
    part = partial_spearman(df, "x", "y", ["c"])
    assert raw["rho"] > 0.7 and abs(part["rho"]) < 0.15 and part["n"] == 300
