"""Imputación dentro del pliegue, determinismo con semilla y validación de esquema."""
import numpy as np
import pandas as pd
import pytest

from datos_prep import SchemaError, _read
from train import PositionMedianImputer, make_pipeline


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.choice(["WR", "OL", "DB"], n)
    forty = np.where(pos == "OL", 5.1, 4.5) + rng.normal(0, 0.1, n)
    forty[rng.random(n) < 0.2] = np.nan
    X = pd.DataFrame({"combine_position": pd.Categorical(pos), "forty": forty, "weight": rng.normal(250, 40, n)})
    y = pd.Series((rng.random(n) < 0.3).astype(int))
    return X, y


def test_imputador_usa_solo_medianas_del_entrenamiento():
    X, _ = _toy()
    train, test = X.iloc[:80], X.iloc[80:].copy()
    imp = PositionMedianImputer().fit(train)
    out = imp.transform(test)
    assert out["forty"].notna().all()
    ol_missing = test["forty"].isna() & (test["combine_position"] == "OL")
    if ol_missing.any():
        expected = train.loc[train.combine_position == "OL", "forty"].median()
        assert np.allclose(out.loc[ol_missing, "forty"], expected)


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "xgboost"])
def test_clasificadores_deterministas_con_semilla(name):
    X, y = _toy()
    cols = ["forty", "weight"]
    p1 = make_pipeline(name, "classification", cols, seed=42).fit(X, y).predict_proba(X)[:, 1]
    p2 = make_pipeline(name, "classification", cols, seed=42).fit(X, y).predict_proba(X)[:, 1]
    assert np.array_equal(p1, p2)


def test_esquema_rechaza_columna_faltante(tmp_path):
    p = tmp_path / "games.csv"
    pd.DataFrame({"game_id": [1], "season": [2024]}).to_csv(p, index=False)
    with pytest.raises(SchemaError):
        _read(p, "games")
