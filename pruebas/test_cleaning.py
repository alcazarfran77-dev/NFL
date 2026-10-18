"""Reglas de negocio R1-R7: rangos, titularidad, no seleccionados y banderas de faltantes."""
import numpy as np
import pandas as pd

from datos_prep import apply_range_rules, build_prospects


def _tables():
    combine = pd.DataFrame({
        "draft_year": [2023, 2025, 2024], "nfl_id": [1, 2, 3], "combine_position": ["WR", "OL", "DB"],
        "combine_height": [72.0, 77.0, 70.0], "combine_weight": [190.0, 310.0, 195.0], "hand_size": [9.5, 10.0, 9.0],
        "arm_length": [31.0, 34.0, 30.5], "wing_span": [76.0, 82.0, 74.0], "ten_yd_split": [1.52, 1.80, np.nan],
        "forty": [4.40, 5.10, 3.20],  # 3.20 s es imposible -> NaN
        "vertical": [36.0, 28.0, 38.0], "broad_jump": [124.0, 105.0, 126.0], "three_cone": [np.nan, 7.6, 6.9],
        "short_shuttle": [np.nan, 4.6, 4.1], "bench_reps": [np.nan, 25.0, 14.0], "ngs_athleticism_score": [80.0, 70.0, 75.0],
        "ngs_college_production_score": [70.0, 60.0, 65.0], "ngs_final_score": [75.0, 65.0, 70.0],
    })
    players = pd.DataFrame({
        "nfl_id": [1, 2, 3], "display_name": ["A", "B", "C"], "draft_year": [2023, 2025, 2024],
        "nfl_position": ["WR", "T", "CB"], "birth_date": ["2001-04-27", "2002-01-01", "2001-06-01"],
        "college_name": ["X", "Y", "Z"], "college_conference": ["Southeastern Conference", "Ivy League", "Big Ten Conference"],
        "draft_round": pd.array([1, pd.NA, 5], dtype="Int64"), "draft_pick_within_round": pd.array([3, pd.NA, 10], dtype="Int64"),
        "draft_overall_pick": pd.array([3, pd.NA, 150], dtype="Int64"),
    })
    careers = pd.DataFrame({
        "nfl_id": [1, 2, 3], "career_offensive_snaps": [3000, 100, 0], "career_defensive_snaps": [0, 0, 500],
        "career_special_teams_snaps": [10, 50, 200], "career_games_active": [51, 10, 30],
        "career_games_started": [26, 9, 16], "ap_all_pro_1st_team": [0, 0, 0], "ap_all_pro_2nd_team": [0, 0, 0],
        "pro_bowl_original_ballot": [0, 0, 0],
    })
    return {"combine": combine, "players": players, "careers": careers}


def test_regla_de_rango_borra_valores_imposibles(cfg):
    clean, audit = apply_range_rules(_tables()["combine"], cfg["business_rules"]["drill_valid_ranges"])
    assert np.isnan(clean.loc[2, "forty"])
    assert len(audit) == 1 and audit.iloc[0]["column"] == "forty"


def test_titular_usa_las_temporadas_disponibles(cfg):
    df, _ = build_prospects(_tables(), cfg)
    df = df.set_index("nfl_id")
    # clase 2023: 3 temporadas x 17 partidos = 51; 26 inicios -> 0.51 -> titular
    assert df.loc[1, "seasons_available"] == 3
    assert abs(df.loc[1, "start_rate"] - 26 / 51) < 1e-9 and df.loc[1, "is_starter"] == 1
    # clase 2025: 1 temporada; 9 inicios / 17 = 0.53 -> titular
    assert df.loc[2, "is_starter"] == 1
    # clase 2024: 16 / 34 = 0.47 -> no es titular
    assert df.loc[3, "is_starter"] == 0


def test_no_seleccionados_y_banderas_de_faltantes(cfg):
    df, _ = build_prospects(_tables(), cfg)
    df = df.set_index("nfl_id")
    assert df.loc[2, "is_undrafted"] == 1
    assert df.loc[2, "draft_pick_filled"] == cfg["business_rules"]["undrafted_pick"]
    assert df.loc[1, "three_cone_missing"] == 1 and df.loc[2, "three_cone_missing"] == 0
    assert df.loc[3, "forty_missing"] == 1  # lo borra la regla de rango y luego se marca
    assert df.loc[1, "power_conference"] == 1 and df.loc[2, "power_conference"] == 0
    assert abs(df.loc[1, "age_at_draft"] - 22.0) < 0.01
