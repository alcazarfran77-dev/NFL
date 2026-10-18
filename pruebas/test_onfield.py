"""Limpieza de player_play (R10-R13), métricas sobre lo esperado y el proyector apilado."""
import numpy as np
import pandas as pd

from datos_prep import clean_player_play, family_plays, player_onfield
from train import OnFieldProjector, PositionMedianImputer, make_pipeline


def _pp(n_players=4, plays_each=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_players):
        for j in range(plays_each):
            rows.append({
                "game_id": 1 + (j % 2), "play_id": j, "nfl_id": 100 + k, "team_abbr": "AAA",
                "lined_up_position": "EDGE" if k % 2 == 0 else "INTERIOR_LINE", "down": 1, "yards_to_go": 10,
                "pass_result": "C", "play_nullified_by_penalty": "Y" if j == 0 else "N", "target": "FALSE",
                "route_ran": pd.NA, "separation_at_pass_forward": np.nan, "cushion": np.nan, "dropback_duration": np.nan,
                "pass_rushers_encountered": np.nan, "peak_pressure_probability_allowed": np.nan,
                "pressure_allowed": pd.NA, "sack_allowed": np.nan,
                "player_get_off": 0.05 if j == 1 else 0.8 + 0.05 * k + rng.normal(0, 0.05),
                "sack": 0.0, "time_to_pressure": 2.5 if rng.random() < 0.1 + 0.05 * k else np.nan,
                "blitzing": "FALSE", "unblocked_pressure": "FALSE", "coverage_assignment": "NO_ASSIGNMENT",
                "coverage_assignment_at_snap": "NO_ASSIGNMENT",
            })
    pp = pd.DataFrame(rows)
    games = pd.DataFrame({"game_id": [1, 2], "season": [2024, 2024], "season_type": ["REG", "POST"], "week": [5, 19]})
    prospects = pd.DataFrame({"nfl_id": [100 + k for k in range(n_players)], "combine_position": ["DL"] * n_players,
                              "nfl_position": ["DE", "DT"] * (n_players // 2), "draft_year": [2024] * n_players})
    return pp, games, prospects


def test_reglas_de_temporada_regular_anuladas_y_arranque(cfg):
    pp, games, prospects = _pp()
    plays, context, audit = clean_player_play(pp, games, prospects, cfg)
    assert set(plays.season_type) == {"REG"}                     # R10: el partido 2 es de playoffs
    assert audit["pp_rows_not_regular_season"] == 4 * 20
    assert plays.play_nullified_by_penalty.eq("N").all()         # R11
    assert plays.player_get_off.dropna().min() >= 0.30           # R12: 0.05 s se borra
    assert len(context) == len(pp)                               # el contexto conserva todos los tipos de temporada


def test_metricas_sobre_lo_esperado_se_centran_en_el_contexto(cfg):
    pp, games, prospects = _pp(plays_each=80)
    plays, _, _ = clean_player_play(pp, games, prospects, cfg)
    dl = family_plays(plays, cfg)["DL"]
    by_al = dl.groupby("alignment")[["goe", "proe"]].mean()
    assert np.allclose(by_al.to_numpy(), 0, atol=1e-12)
    # el jugador 100 (EDGE, el más rápido) arranca antes de lo esperado -> goe positivo en promedio
    assert dl[dl.nfl_id == 100].goe.mean() > 0 > dl[dl.nfl_id == 102].goe.mean()


def test_regla_de_minimo_de_jugadas_e_indice(cfg):
    pp, games, prospects = _pp(plays_each=30)   # 15 jugadas de temporada regular, 14 tras quitar la anulada -> < 20
    plays, _, _ = clean_player_play(pp, games, prospects, cfg)
    prof = player_onfield(family_plays(plays, cfg), prospects, cfg)
    assert prof.goe_eb.isna().all() and prof.perf_index.isna().all()


def _proj_frame(n=120, seed=1):
    rng = np.random.default_rng(seed)
    pos = rng.choice(["DL", "OL", "DB"], n)
    speed = rng.normal(0, 1, n)
    target = np.where(pos == "DL", 0.5 * speed + rng.normal(0, 0.3, n), np.nan)
    return pd.DataFrame({"combine_position": pd.Categorical(pos), "speed": speed, "weight": rng.normal(0, 1, n),
                         "onfield_target_goe": target})


def test_proyector_nunca_lee_etiquetas_al_predecir():
    X = _proj_frame()
    proj = OnFieldProjector(alpha=1.0, inner_folds=5).fit(X)
    a = proj.transform(X)
    b = proj.transform(X.assign(onfield_target_goe=np.nan))
    assert "onfield_target_goe" not in a.columns
    assert np.array_equal(a["proj_goe"].to_numpy(), b["proj_goe"].to_numpy())
    assert (a.loc[X.combine_position.astype(str) != "DL", "proj_goe"] == 0).all()   # fuera de la familia


def test_filas_de_entrenamiento_del_proyector_son_fuera_de_pliegue():
    X = _proj_frame()
    proj = OnFieldProjector(alpha=1.0, inner_folds=5, random_state=0)
    fitted = proj.fit_transform(X)
    in_sample = proj.transform(X)
    lab = X.onfield_target_goe.notna()
    assert not np.allclose(fitted.loc[lab, "proj_goe"], in_sample.loc[lab, "proj_goe"])
    assert np.corrcoef(fitted.loc[lab, "proj_goe"], X.loc[lab, "speed"])[0, 1] > 0.5


def test_imputador_no_toca_las_etiquetas_del_proyector():
    X = _proj_frame()
    out = PositionMedianImputer(exclude=("onfield_target_goe",)).fit(X).transform(X)
    assert out.onfield_target_goe.isna().sum() == X.onfield_target_goe.isna().sum()


def test_pipeline_con_proyector_ajusta_y_predice():
    X = _proj_frame()
    y = pd.Series((X.speed + np.random.default_rng(2).normal(0, 1, len(X)) > 0).astype(int))
    pipe = make_pipeline("logistic_regression", "classification", ["speed", "weight", "onfield_target_goe"], seed=42,
                         projector={"alpha": 1.0, "inner_folds": 5})
    p = pipe.fit(X, y).predict_proba(X.assign(onfield_target_goe=np.nan))[:, 1]
    assert p.shape == (len(X),) and np.all((p > 0) & (p < 1))
