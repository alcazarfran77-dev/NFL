"""Data contract, reader conformance with the spec, Parquet/CSV ingestion and Combine drill tracking."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from data_prep import (CT_FEATURES, Config, _parse_bool, _read, attempt_metrics, attempt_reliability,
                       clean_combine_tracking, clean_player_play, detect_format, family_plays, harmonize,
                       has_combine_tracking, load_contract, logical_name, player_features, read_raw_table,
                       resolve_files, validate_frame)
from data_prep import build_combine_tracking as build


# ----------------------------------------------------------------------------- contract
def test_contract_flags_deviations_with_the_right_severity():
    spec = load_contract()["files"]["games"]
    ok = pd.DataFrame({"game_id": [2023090700, 2023091000], "game_key": [1, 2], "season": [2023, 2023],
                       "season_type": ["REG", "POST"], "week": [1, 19 - 1], "home_team_abbr": ["KC", "NYG"],
                       "visitor_team_abbr": ["DET", "DAL"]})
    rows = pd.DataFrame(validate_frame("games", "games.csv", spec, ok))
    assert (rows.result == "pass").all()

    bad = ok.assign(season_type=["PRE", "REG"], week=[0, 3])
    bad = pd.concat([bad, bad.iloc[[0]]], ignore_index=True)          # duplicate key
    rows = pd.DataFrame(validate_frame("games", "games.csv", spec, bad)).set_index(["column", "check"])
    assert tuple(rows.loc[("season_type", "allowed_values"), ["result", "severity"]]) == ("fail", "warning")
    assert rows.loc[("week", "range[1,18]"), "n_bad"] == 2
    assert tuple(rows.loc[("game_id", "unique_key"), ["result", "severity"]]) == ("fail", "error")

    missing = pd.DataFrame(validate_frame("games", "games.csv", spec, ok.drop(columns=["week"])))
    assert missing.query("check == 'columns_present'").severity.iloc[0] == "error"


def test_contract_bool_accepts_both_encodings():
    spec = {"columns": {"flag": {"type": "bool", "nullable": True}}}
    for vals in (["TRUE", "FALSE", None], [1.0, 0.0, np.nan], ["True", "False", "1"]):
        rows = pd.DataFrame(validate_frame("t", "f", spec, pd.DataFrame({"flag": vals})))
        assert rows.query("check == 'type_bool'").result.iloc[0] == "pass", vals


def test_resolve_files_finds_spec_names_and_parts(tmp_path):
    for name in ["game_tracking_2024_2.csv", "game_tracking_2025.csv", "game_tracking_2025_2.csv", "player_play_3.csv"]:
        (tmp_path / name).write_text("x\n")
    used, superseded = resolve_files(tmp_path, "game_tracking_*.csv")
    assert [p.name for p in used] == ["game_tracking_2024_2.csv", "game_tracking_2025.csv"]
    assert [p.name for p in superseded] == ["game_tracking_2025_2.csv"]
    assert [p.name for p in resolve_files(tmp_path, "player_play*.csv")[0]] == ["player_play_3.csv"]
    assert resolve_files(tmp_path, "combine_tracking*.csv") == ([], [])


# ----------------------------------------------------------------------------- conformance
def test_parse_bool_accepts_text_and_floats():
    text = _parse_bool(pd.Series(["TRUE", "FALSE", "True", "false", None, "NA"], dtype="string"))
    num = _parse_bool(pd.Series([1.0, 0.0, np.nan, 2.0]))
    assert text.tolist()[:4] == [1.0, 0.0, 1.0, 0.0] and text.iloc[4:].isna().all()
    assert num.tolist()[:2] == [1.0, 0.0] and num.iloc[2:].isna().all()


def _pp_rows(**over):
    base = {"game_id": 1, "play_id": 1, "nfl_id": 1, "team_abbr": "KC", "lined_up_position": "EDGE", "down": 1,
            "yards_to_go": 10, "pass_result": "C", "play_nullified_by_penalty": "N", "target": "FALSE",
            "route_ran": None, "separation_at_pass_forward": np.nan, "cushion": np.nan, "dropback_duration": np.nan,
            "pass_rushers_encountered": np.nan, "peak_pressure_probability_allowed": np.nan, "pressure_allowed": None,
            "sack_allowed": np.nan, "player_get_off": 0.8, "sack": 0.0, "time_to_pressure": np.nan,
            "blitzing": "FALSE", "unblocked_pressure": None, "coverage_assignment": None,
            "coverage_assignment_at_snap": None}
    return {**base, **over}


def test_r15_two_point_tries_and_alignment_codes(cfg):
    pp = pd.DataFrame([_pp_rows(play_id=1, lined_up_position="DE"), _pp_rows(play_id=2, lined_up_position="DT"),
                       _pp_rows(play_id=3, lined_up_position="NT"), _pp_rows(play_id=4, lined_up_position="OLB"),
                       _pp_rows(play_id=5, down=0)])
    games = pd.DataFrame({"game_id": [1], "season": [2024], "season_type": ["REG"], "week": [5]})
    prospects = pd.DataFrame({"nfl_id": [1], "combine_position": ["DL"], "nfl_position": ["DE"], "draft_year": [2024]})
    plays, _, audit = clean_player_play(pp, games, prospects, cfg)
    assert audit["pp_rows_invalid_down"] == 1 and 5 not in plays.play_id.tolist()
    dl = family_plays(plays, cfg)["DL"].set_index("play_id")
    assert dl.loc[[1, 2, 3, 4], "alignment"].tolist() == ["EDGE", "INTERIOR_LINE", "INTERIOR_LINE", "OTHER"]


# ----------------------------------------------------------------------------- Combine drill tracking
def _attempt(event_id, nfl_id, drill_type, drill_name, speeds, heading=90.0, turns=(), attempt=1, year=2025):
    """Synthetic 10 Hz attempt: still for 5 frames, then the given speed trace along `heading`.

    `turns` = frame indices where the direction reverses (180 degrees), as in a shuttle.
    """
    rows, x, y, h = [], 10.0, 20.0, heading
    t0 = pd.Timestamp("2025-03-01T18:00:00")
    trace = [0.0] * 5 + list(speeds)
    for i, s in enumerate(trace):
        if i in turns:
            h = (h + 180.0) % 360
        rad = np.deg2rad(90 - h)
        x += s * 0.1 * np.cos(rad)
        y += s * 0.1 * np.sin(rad)
        a = 0.0 if i == 0 else (s - trace[i - 1]) * 10
        rows.append({"draft_year": year, "event_id": event_id, "nfl_id": nfl_id, "entity_type": "PLAYER",
                     "time": (t0 + pd.Timedelta(milliseconds=100 * i)).isoformat(), "drill_type": drill_type,
                     "drill_name": drill_name, "attempt": attempt, "x": x, "y": y, "s": s, "a": abs(a),
                     "dis": s * 0.1, "dir": h})
    rows.append({**rows[-1], "entity_type": "BALL"})  # R16 drops the ball rows
    return rows


def _cfg():
    return Config({"combine_tracking_rules": {}})


def test_forty_split_matches_constant_acceleration():
    acc, top = 6.0, 10.0          # first moving frame at 0.6 yd/s: the start point is the last still frame
    t = np.arange(1, 61) / 10
    speeds = np.minimum(acc * t, top)
    ct = pd.DataFrame(_attempt("e1", 7, "FORTY_YARD_DASH", "FORTY", speeds))
    frames, audit = clean_combine_tracking(ct, _cfg())
    assert audit["ct_frames_not_player"] == 1
    att, _ = attempt_metrics(frames, _cfg())
    row = att.iloc[0]
    # discrete (forward Euler) distance after k frames = sum of speeds; 10 yd is reached at ~2.2-2.3 s
    disp = np.concatenate([[0.0], np.cumsum(speeds * 0.1)])
    k = np.flatnonzero(disp >= 10)[0]
    expected = (k - 1 + (10 - disp[k - 1]) / (disp[k] - disp[k - 1])) / 10
    assert row.split_10yd_s == pytest.approx(expected, abs=1e-9)
    t = {m: (np.flatnonzero(disp >= m)[0] - 1 + (m - disp[np.flatnonzero(disp >= m)[0] - 1])
             / (disp[np.flatnonzero(disp >= m)[0]] - disp[np.flatnonzero(disp >= m)[0] - 1])) / 10 for m in (5, 10, 20)}
    assert row.split_5_10_s == pytest.approx(t[10] - t[5], abs=1e-9)
    assert row.split_10_20_s == pytest.approx(t[20] - t[10], abs=1e-9)
    assert row.top_speed == pytest.approx(top)
    assert row.burst_2s == pytest.approx(disp[20])
    assert np.isnan(row.n_plants)


def test_shuttle_plants_and_best_attempt_aggregation():
    up, down = list(np.linspace(1, 7, 8)), list(np.linspace(7, 1.5, 6))
    trace = up + down + up + down + up
    turn_frames = (5 + len(up) + len(down), 5 + 2 * (len(up) + len(down)))
    rows = (_attempt("s1", 7, "SHORT_SHUTTLE", "SHUTTLE", trace, turns=turn_frames)
            + _attempt("s2", 7, "SHORT_SHUTTLE", "SHUTTLE", [v * 0.8 for v in trace], turns=turn_frames, attempt=2)
            + _attempt("f1", 8, "FORTY_YARD_DASH", "FORTY", np.minimum(np.arange(1, 61) * 0.4, 9.5))
            + _attempt("k1", 7, "SKILL_DRILLS_DL", "GAUNTLET_DRILL", [3, 5, 6, 6, 6, 6, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2])
            + _attempt("k2", 8, "SKILL_DRILLS_DL", "GAUNTLET_DRILL", [2, 3, 4, 4, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
    attempts, feats, rel, audit = build(pd.DataFrame(rows), pd.DataFrame({"nfl_id": [7, 8, 9]}), _cfg())
    shuttle = attempts[attempts.drill_group == "shuttle"].set_index("event_id")
    assert (shuttle.n_plants >= 2).all() and shuttle.plant_speed_ratio.between(0, 0.5).all()
    f = feats.set_index("nfl_id")
    assert list(f.index) == [7, 8, 9] and set(CT_FEATURES) <= set(f.columns)
    # best attempt = the fastest (unscaled) shuttle
    assert f.loc[7, "ct_shuttle_plant_speed_ratio"] == pytest.approx(shuttle.plant_speed_ratio.max())
    assert f.loc[7, "ct_skill_speed_z"] > f.loc[8, "ct_skill_speed_z"]
    assert f.loc[9, "ct_missing"] == 1 and f.loc[9, "ct_attempts"] == 0
    assert np.isnan(f.loc[7, "ct_forty_split_5_10_s"]) and not np.isnan(f.loc[8, "ct_forty_split_5_10_s"])
    assert set(rel.drill_group) == {"forty", "3cone", "shuttle", "skill"}
    assert has_combine_tracking(f.reset_index())
    assert not has_combine_tracking(pd.DataFrame({"nfl_id": [1]}))


def test_reliability_gate_separates_repeatable_from_noisy_metrics():
    """A metric that repeats across a player's attempts passes the gate; pure noise does not."""
    rng = np.random.default_rng(1)
    rows = []
    for pid in range(60):
        talent = rng.normal()
        for att in (1, 2):
            rows.append({"nfl_id": pid, "attempt": att, "drill_group": "forty", "drill_type": "FORTY_YARD_DASH", "drill_name": "FORTY",
                         "split_5_10_s": 0.75 - 0.03 * talent + rng.normal(0, 0.005),   # repeatable
                         "split_0_5_s": rng.normal(1.3, 0.1)})                           # start noise
    att = pd.DataFrame(rows)
    for m in ["split_10_20_s", "split_10yd_s", "top_speed", "peak_accel", "burst_1s", "burst_2s", "t90_top_s"]:  # noise
        att[m] = rng.normal(size=len(att))
    rel = attempt_reliability(att).query("drill_group == 'forty'").set_index("metric")
    assert rel.loc["split_5_10_s", "passes_gate"] and rel.loc["split_5_10_s", "reliability"] > 0.9
    assert not rel.loc["split_0_5_s", "passes_gate"]
    assert rel.loc["split_5_10_s", "feature"] == "ct_forty_split_5_10_s"


def test_compressed_csv_equals_the_original(tmp_path):
    """A .csv.gz has the same content hash as its .csv and counts as the same logical file."""
    import gzip

    from data_prep import sha256

    raw = b"game_id,nfl_id\n1,2\n"
    (tmp_path / "game_tracking_2024_2.csv").write_bytes(raw)
    with gzip.open(tmp_path / "game_tracking_2024_2.csv.gz", "wb") as fh:
        fh.write(raw)
    with gzip.open(tmp_path / "game_tracking_2025.csv.gz", "wb") as fh:
        fh.write(raw)
    with gzip.open(tmp_path / "game_tracking_2025_2.csv.gz", "wb") as fh:
        fh.write(raw)
    used, superseded = resolve_files(tmp_path, "game_tracking_*.csv*")
    assert [p.name for p in used] == ["game_tracking_2024_2.csv", "game_tracking_2025.csv.gz"]
    assert [p.name for p in superseded] == ["game_tracking_2025_2.csv.gz"]
    assert sha256(tmp_path / "game_tracking_2024_2.csv") == sha256(tmp_path / "game_tracking_2024_2.csv.gz")
    assert logical_name(tmp_path / "game_tracking_2025.csv.gz") == "game_tracking_2025.csv"
    assert pd.read_csv(tmp_path / "game_tracking_2025.csv.gz").shape == (1, 2)


# ----------------------------------------------------------------------------- Parquet and CSV
def _games_frame():
    return pd.DataFrame({"game_id": [2023090700, 2024010700], "game_key": [1, 2], "season": [2023, 2023],
                         "season_type": ["REG", "POST"], "week": [1, 18], "home_team_abbr": ["KC", "NYG"],
                         "visitor_team_abbr": ["DET", "DAL"]})


def test_detect_format_reads_the_content_not_the_extension(tmp_path):
    df = _games_frame()
    df.to_parquet(tmp_path / "games.parquet", index=False)
    df.to_csv(tmp_path / "games.csv", index=False)
    df.to_csv(tmp_path / "games.csv.gz", index=False, compression="gzip")
    (tmp_path / "mislabelled.csv").write_bytes((tmp_path / "games.parquet").read_bytes())
    assert detect_format(tmp_path / "games.parquet") == "parquet"
    assert detect_format(tmp_path / "games.csv") == "csv"
    assert detect_format(tmp_path / "games.csv.gz") == "csv.gz"
    assert detect_format(tmp_path / "mislabelled.csv") == "parquet"
    assert read_raw_table(tmp_path / "mislabelled.csv").shape == df.shape
    assert logical_name(tmp_path / "games.parquet") == "games.parquet"
    assert logical_name(tmp_path / "games.csv.gz") == "games.csv"


def test_parquet_wins_when_a_table_comes_in_several_formats(tmp_path):
    df = _games_frame()
    df.to_csv(tmp_path / "games.csv", index=False)
    df.to_csv(tmp_path / "games.csv.gz", index=False, compression="gzip")
    assert [p.name for p in resolve_files(tmp_path, "games.*")[0]] == ["games.csv"]   # plain CSV over its .gz copy
    df.to_parquet(tmp_path / "games.parquet", index=False)
    assert [p.name for p in resolve_files(tmp_path, "games.*")[0]] == ["games.parquet"]
    # a part is superseded by the full file whatever the formats of the two
    df.to_parquet(tmp_path / "game_tracking_2025_2.parquet", index=False)
    df.to_csv(tmp_path / "game_tracking_2025.csv.gz", index=False, compression="gzip")
    used, superseded = resolve_files(tmp_path, "game_tracking_*")
    assert [p.name for p in used] == ["game_tracking_2025.csv.gz"]
    assert [p.name for p in superseded] == ["game_tracking_2025_2.parquet"]


def test_harmonize_uses_the_csv_encodings():
    df = pd.DataFrame({"flag": pd.array([True, False, None], dtype="boolean"),
                       "when": pd.to_datetime(["2025-01-11 23:31:37.5", "2025-01-11 23:31:37.6", None]),
                       "day": [dt.date(2024, 9, 5), None, dt.date(2025, 1, 1)],
                       "pos": pd.Categorical(["DL", "OL", "DL"]), "x": [1.5, np.nan, 2.0]})
    out = harmonize(df)
    assert out.flag.tolist()[:2] == ["TRUE", "FALSE"] and pd.isna(out.flag.iloc[2])
    assert out.when.tolist()[:2] == ["2025-01-11T23:31:37.500", "2025-01-11T23:31:37.600"] and pd.isna(out.when.iloc[2])
    assert out.day.iloc[0] == "2024-09-05" and out.day.iloc[2] == "2025-01-01"
    assert not isinstance(out.pos.dtype, pd.CategoricalDtype) and out.pos.tolist() == ["DL", "OL", "DL"]
    assert out.x.equals(df.x)


def test_parquet_and_csv_give_the_same_typed_table(tmp_path):
    players = pd.DataFrame({
        "nfl_id": [1, 2], "display_name": ["A", "B"], "draft_year": [2023, 2025], "nfl_position": ["WR", "T"],
        "birth_date": [dt.date(2001, 4, 27), dt.date(2002, 1, 1)], "college_name": ["X", "Y"],
        "college_conference": ["Southeastern Conference", "Ivy League"],
        "draft_round": pd.array([1, pd.NA], dtype="Int64"), "draft_pick_within_round": pd.array([3, pd.NA], dtype="Int64"),
        "draft_overall_pick": pd.array([3, pd.NA], dtype="Int64")})
    players.to_parquet(tmp_path / "players.parquet", index=False)                      # birth_date stored as date32
    players.assign(birth_date=players.birth_date.astype(str)).to_csv(tmp_path / "players.csv", index=False)
    pd.testing.assert_frame_equal(_read(tmp_path / "players.parquet", "players"), _read(tmp_path / "players.csv", "players"))


def test_contract_report_is_the_same_for_parquet_and_csv(tmp_path):
    spec = load_contract()["files"]["games"]
    df = _games_frame()
    df.to_parquet(tmp_path / "games.parquet", index=False)
    df.to_csv(tmp_path / "games.csv", index=False)
    a, b = (pd.DataFrame(validate_frame("games", "games", spec, read_raw_table(tmp_path / f))).drop(columns=["file"])
            for f in ("games.parquet", "games.csv"))
    pd.testing.assert_frame_equal(a, b)
    assert (a.result == "pass").all()
