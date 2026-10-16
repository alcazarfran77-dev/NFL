"""Raw data ingestion: schema validation, typing and a content-hash manifest.

The manifest (data/manifest.json) is the data-versioning contract: every run
records the SHA-256 of each raw file, so a result can always be traced to the
exact bytes it was trained on (a lightweight, dependency-free DVC equivalent).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Config

# Expected columns and dtypes per table (pandas nullable types where nulls are legal).
SCHEMAS: dict[str, dict[str, str]] = {
    "combine": {
        "draft_year": "int64", "nfl_id": "int64", "combine_position": "string",
        "combine_height": "float64", "combine_weight": "float64", "hand_size": "float64",
        "arm_length": "float64", "wing_span": "float64", "ten_yd_split": "float64",
        "forty": "float64", "vertical": "float64", "broad_jump": "float64",
        "three_cone": "float64", "short_shuttle": "float64", "bench_reps": "float64",
        "ngs_athleticism_score": "float64", "ngs_college_production_score": "float64",
        "ngs_final_score": "float64",
    },
    "players": {
        "nfl_id": "int64", "display_name": "string", "draft_year": "int64",
        "nfl_position": "string", "birth_date": "string", "college_name": "string",
        "college_conference": "string", "draft_round": "Int64",
        "draft_pick_within_round": "Int64", "draft_overall_pick": "Int64",
    },
    "careers": {
        "nfl_id": "int64", "career_offensive_snaps": "int64",
        "career_defensive_snaps": "int64", "career_special_teams_snaps": "int64",
        "career_games_active": "int64", "career_games_started": "int64",
        "ap_all_pro_1st_team": "int64", "ap_all_pro_2nd_team": "int64",
        "pro_bowl_original_ballot": "int64",
    },
    "games": {
        "game_id": "int64", "game_key": "int64", "season": "int64",
        "season_type": "string", "week": "int64", "home_team_abbr": "string",
        "visitor_team_abbr": "string",
    },
    "tracking": {
        "game_id": "int64", "play_id": "int64", "nfl_id": "int64", "time": "string",
        "x": "float64", "y": "float64", "s": "float64", "a": "float64",
        "dis": "float64", "o": "float64", "dir": "float64", "event": "string",
    },
    # v2.0.0. Boolean flags are stored as TRUE/FALSE/NA in the source, so they are read as
    # strings and parsed explicitly in performance.clean_player_play.
    "player_play": {
        "game_id": "int64", "play_id": "int64", "team_abbr": "string", "nfl_id": "int64",
        "lined_up_position": "string", "play_description": "string", "quarter": "int64", "down": "int64",
        "yards_to_go": "int64", "possession_team": "string", "yardline_side": "string",
        "yardline_number": "int64", "game_clock": "string", "pre_snap_home_score": "int64",
        "pre_snap_visitor_score": "int64", "play_direction": "string",
        "pre_snap_home_team_win_probability": "float64", "pre_snap_visitor_team_win_probability": "float64",
        "home_team_win_probability_added": "float64", "visitor_team_win_probility_added": "float64",
        "expected_points_added": "float64", "expected_points": "float64", "offense_formation": "string",
        "receiver_alignment": "string", "pass_result": "string", "pass_length": "float64",
        "play_nullified_by_penalty": "string", "penalty_yards": "float64", "pre_penalty_yards_gained": "int64",
        "yards_gained": "int64", "team_coverage_man_zone": "string", "team_coverage_type": "string",
        "in_motion_at_ball_snap": "string", "target": "string", "rec_yards": "float64", "rush_yards": "float64",
        "yards_after_catch": "float64", "fumble": "float64", "fumble_lost": "float64", "route_ran": "string",
        "separation_at_pass_forward": "float64", "cushion": "float64", "expected_yards_after_catch": "float64",
        "dropback_duration": "float64", "extended_sack_allowed": "float64", "pass_rushers_encountered": "float64",
        "peak_pressure_probability_allowed": "float64", "pressure_allowed": "string", "sack_allowed": "float64",
        "time_to_pressure_allowed": "float64", "quick_pressure": "string", "unblocked_pressure": "string",
        "blitzing": "string", "player_get_off": "float64", "sack": "float64", "tackle": "float64",
        "assist": "float64", "caused_forced_fumble": "float64", "recovered_fumble": "string",
        "tackle_for_loss": "string", "time_to_pressure": "float64", "time_to_qb_hurry": "float64",
        "coverage_assignment": "string", "coverage_assignment_at_snap": "string",
    },
}

# Columns of player_play the analysis uses (the full 64-column header is still validated).
PLAYER_PLAY_USECOLS = [
    "game_id", "play_id", "nfl_id", "team_abbr", "lined_up_position", "down", "yards_to_go",
    "pass_result", "play_nullified_by_penalty", "target", "route_ran", "separation_at_pass_forward",
    "cushion", "dropback_duration", "pass_rushers_encountered", "peak_pressure_probability_allowed",
    "pressure_allowed", "sack_allowed", "player_get_off", "sack", "time_to_pressure", "blitzing",
    "unblocked_pressure", "coverage_assignment", "coverage_assignment_at_snap",
]

PRIMARY_KEYS = {
    "combine": ["nfl_id"],
    "players": ["nfl_id"],
    "careers": ["nfl_id"],
    "games": ["game_id"],
    "tracking": ["game_id", "play_id", "nfl_id", "time"],
    "player_play": ["game_id", "play_id", "nfl_id"],
}


class SchemaError(ValueError):
    """Raised when a raw table does not match its expected contract."""


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _read(path: Path, table: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Validate the full header, then load (optionally a subset of) columns with declared types."""
    expected = SCHEMAS[table]
    header = list(pd.read_csv(path, nrows=0).columns)
    missing = set(expected) - set(header)
    extra = set(header) - set(expected)
    if missing:
        raise SchemaError(f"{path.name}: missing columns {sorted(missing)}")
    if extra:
        raise SchemaError(f"{path.name}: unexpected columns {sorted(extra)}")
    cols = usecols or list(expected)
    # "NA" and empty strings are the null markers used in the source files.
    df = pd.read_csv(path, na_values=["NA", ""], keep_default_na=True, usecols=cols, low_memory=False)
    try:
        df = df.astype({c: expected[c] for c in cols})
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{path.name}: type coercion failed: {exc}") from exc
    dups = df.duplicated(PRIMARY_KEYS[table]).sum()
    if dups:
        raise SchemaError(f"{path.name}: {dups} duplicate rows on key {PRIMARY_KEYS[table]}")
    return df


def load_raw(cfg: Config) -> dict[str, pd.DataFrame]:
    """Load and validate every raw table. Tracking files are concatenated."""
    raw_dir = cfg.path("raw")
    files = cfg["raw_files"]
    tables = {name: _read(raw_dir / files[name], name) for name in ("combine", "players", "careers", "games")}
    tracking = [_read(raw_dir / f, "tracking").assign(source_file=f) for f in files["tracking"]]
    tables["tracking"] = pd.concat(tracking, ignore_index=True)
    if tables["tracking"].duplicated(PRIMARY_KEYS["tracking"]).any():
        raise SchemaError("tracking files overlap on (game, play, player, time)")
    if "player_play" in files:
        tables["player_play"] = _read(raw_dir / files["player_play"], "player_play", PLAYER_PLAY_USECOLS)
    _check_referential_integrity(tables)
    return tables


def _check_referential_integrity(t: dict[str, pd.DataFrame]) -> None:
    ids = set(t["combine"].nfl_id)
    for name in ("players", "careers"):
        if set(t[name].nfl_id) != ids:
            raise SchemaError(f"{name}.nfl_id does not match combine.nfl_id")
    orphan_players = set(t["tracking"].nfl_id) - ids
    if orphan_players:
        raise SchemaError(f"tracking has {len(orphan_players)} players without combine rows")
    orphan_games = set(t["tracking"].game_id) - set(t["games"].game_id)
    if orphan_games:
        raise SchemaError(f"tracking has {len(orphan_games)} games missing from games.csv")
    if "player_play" in t:
        pp = t["player_play"]
        if set(pp.nfl_id) - ids:
            raise SchemaError(f"player_play has {len(set(pp.nfl_id) - ids)} players without combine rows")
        if set(pp.game_id) - set(t["games"].game_id):
            raise SchemaError("player_play has games missing from games.csv")


def write_manifest(cfg: Config) -> dict:
    """Hash every raw file and record shape; returns the manifest dict."""
    raw_dir = cfg.path("raw")
    files = cfg["raw_files"]
    names = [files[k] for k in ("combine", "players", "careers", "games")] + list(files["tracking"])
    if "player_play" in files:
        names.append(files["player_play"])
    entries = {}
    for name in names:
        p = raw_dir / name
        with open(p, encoding="utf-8") as fh:
            n_rows = sum(1 for _ in fh) - 1
        entries[name] = {"sha256": sha256(p), "bytes": p.stat().st_size, "rows": n_rows}
    combined = hashlib.sha256("".join(e["sha256"] for e in entries.values()).encode()).hexdigest()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_version": combined[:12],
        "files": entries,
    }
    with open(cfg.path("manifest"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest
