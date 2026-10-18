"""Métricas de movimiento sobre jugadas sintéticas con respuesta conocida (reglas R8-R9)."""
import numpy as np
import pandas as pd

from datos_prep import _shrink, clean_tracking, estimator_bias, play_metrics
from pruebas.conftest import make_play


def test_burst_coincide_con_cinematica_a_velocidad_constante(cfg):
    frames, audit = clean_tracking(make_play(speed=5.0), cfg)
    plays, _ = play_metrics(frames, cfg)
    row = plays.iloc[0]
    # 5 yd/s en los cuadros 1..20 tras el snap: 20 pasos de 0.5 yd hasta t = 2.0 s -> 10.0 yd
    assert abs(row.burst_2s - 10.0) < 1e-6
    assert abs(row.get_off_1s - 5.0) < 1e-9
    assert row.max_cut_deg < 1e-6
    assert audit["plays_with_frame_gaps"] == 0


def test_detecta_el_corte(cfg):
    frames, _ = clean_tracking(make_play(speed=5.0, turn_at=20, turn_deg=90.0), cfg)
    plays, _ = play_metrics(frames, cfg)
    assert abs(plays.iloc[0].max_cut_deg - 90.0) < 1e-6


def test_descarta_jugadas_cortas_y_sin_snap(cfg):
    short = make_play(play_id=2, n_post=10)
    nosnap = make_play(play_id=3).assign(event=pd.NA)
    frames, _ = clean_tracking(pd.concat([make_play(), short, nosnap], ignore_index=True), cfg)
    plays, audit = play_metrics(frames, cfg)
    assert list(plays.play_id) == [1]
    assert audit["plays_too_short"] == 1 and audit["plays_no_snap"] == 1


def test_falla_de_velocidad_se_topa_e_interpola(cfg):
    p = make_play(speed=5.0)
    p.loc[25, "s"] = 40.0
    frames, audit = clean_tracking(p, cfg)
    assert audit["frames_s_above_cap"] == 1
    assert frames["s"].max() <= cfg["tracking_rules"]["speed_cap"]


def test_contraccion_jala_mas_a_muestras_chicas():
    g = pd.DataFrame({"n_plays": [3, 40, 10, 10], "x_mean": [9.0, 9.0, 5.0, 5.0], "x_var": [4.0, 4.0, 4.0, 4.0]})
    s = _shrink(g, "x")
    grand = np.average(g.x_mean, weights=g.n_plays)
    assert abs(s.iloc[1] - 9.0) < abs(s.iloc[0] - 9.0)  # con 40 jugadas queda más cerca de su propia media
    assert min(s.iloc[0], 9.0) >= grand - 1e-9


def test_media_insesgada_pero_q80_no():
    rng = np.random.default_rng(0)
    plays = pd.DataFrame({"nfl_id": np.repeat(np.arange(30), 30), "burst_2s": rng.normal(5, 2, 900)})
    b = estimator_bias(plays, "burst_2s", [3], 300, seed=1).set_index("estimator").bias
    assert abs(b["mean"]) < 0.1
    assert b["q80"] < -0.2
