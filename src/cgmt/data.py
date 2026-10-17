"""Raw data ingestion: schema validation, typing and a content-hash manifest.

The manifest (data/manifest.json) is the data-versioning contract: every run
records the SHA-256 of each raw file, so a result can always be traced to the
exact bytes it was trained on (a lightweight, dependency-free DVC equivalent).

v2.1.0: file names are resolved from patterns (`raw_files` in the config), so both the names
in the provider's data structure spec (game_tracking_2024.csv, player_play.csv,
combine_tracking.csv) and the split parts delivered so far (game_tracking_2024_2.csv,
player_play_3.csv) are found. A part `<stem>_<n>.csv` is superseded when the full `<stem>.csv`
is present. Tracking is read one file at a time (`iter_tracking`) so spec-scale seasons fit in memory.
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Config

log = logging.getLogger("cgmt")

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
    # v2.1.0, spec File 3 (optional): 10 Hz tracking of Combine drill attempts. Ball rows carry
    # the nfl_id of the player in that attempt.
    "combine_tracking": {
        "draft_year": "int64", "event_id": "string", "nfl_id": "Int64", "entity_type": "string",
        "time": "string", "drill_type": "string", "drill_name": "string", "attempt": "int64",
        "x": "float64", "y": "float64", "s": "float64", "a": "float64", "dis": "float64", "dir": "float64",
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
    "combine_tracking": ["event_id", "entity_type", "time"],
}
OPTIONAL_TABLES = ("player_play", "combine_tracking")
# Data rows of a CSV saved at a spreadsheet's row limit (1,048,576 incl. header): the file was probably cut.
SPREADSHEET_ROW_LIMIT = 1_048_575
_PART = re.compile(r"^(?P<stem>.+)_\d+$")


class SchemaError(ValueError):
    """Raised when a raw table does not match its expected contract."""


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def resolve_files(raw_dir: Path, spec: str | list[str]) -> tuple[list[Path], list[Path]]:
    """Files for one table: `spec` is a name, a glob pattern or a list of either.

    Returns (used, superseded), both sorted by name. A split part `<stem>_<n>.csv` is
    superseded (not read, not hashed) when the full `<stem>.csv` is also present.
    """
    specs = [spec] if isinstance(spec, str) else list(spec)
    found = sorted({Path(p) for s_ in specs for p in glob.glob(str(Path(raw_dir) / s_))}, key=lambda p: p.name)
    names = {p.stem for p in found}
    used, superseded = [], []
    for p in found:
        m = _PART.match(p.stem)
        (superseded if m and m.group("stem") in names else used).append(p)
    return used, superseded


def table_files(cfg: Config, table: str) -> list[Path]:
    """Files used for `table` (empty for an absent optional table)."""
    spec = cfg["raw_files"].get(table)
    if spec is None:
        return []
    used, superseded = resolve_files(cfg.path("raw"), spec)
    for p in superseded:
        log.warning("%s: %s superseded by the full file, not used", table, p.name)
    if not used and table not in OPTIONAL_TABLES and table != "tracking":
        raise SchemaError(f"required table {table}: no file matching {spec}")
    return used


def _read(path: Path, table: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Validate the full header, then load (optionally a subset of) columns with declared types.

    A missing column is an error. An extra column is logged and ignored (the data-contract
    report lists it), so an additive change by the provider does not stop the pipeline.
    """
    expected = SCHEMAS[table]
    header = list(pd.read_csv(path, nrows=0).columns)
    missing = set(expected) - set(header)
    extra = set(header) - set(expected)
    if missing:
        raise SchemaError(f"{path.name}: missing columns {sorted(missing)}")
    if extra:
        log.warning("%s: extra columns ignored %s", path.name, sorted(extra))
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


def _read_parts(paths: list[Path], table: str, usecols: list[str] | None = None) -> pd.DataFrame:
    df = pd.concat([_read(p, table, usecols) for p in paths], ignore_index=True)
    if len(paths) > 1 and df.duplicated(PRIMARY_KEYS[table]).any():
        raise SchemaError(f"{table}: file parts overlap on {PRIMARY_KEYS[table]}")
    return df


def iter_tracking(cfg: Config):
    """Yield (file name, validated frames) one tracking file at a time."""
    for p in table_files(cfg, "tracking"):
        yield p.name, _read(p, "tracking").assign(source_file=p.name)


def load_raw(cfg: Config, tracking: bool = True) -> dict[str, pd.DataFrame]:
    """Load and validate every raw table.

    tracking=False skips the frames (the pipeline streams them with `iter_tracking`); the
    tracking keys are still checked against players and games, file by file.
    """
    tables = {name: _read_parts(table_files(cfg, name), name) for name in ("combine", "players", "careers", "games")}
    if tracking:
        parts = [df for _, df in iter_tracking(cfg)]
        tables["tracking"] = pd.concat(parts, ignore_index=True)
        if tables["tracking"].duplicated(PRIMARY_KEYS["tracking"]).any():
            raise SchemaError("tracking files overlap on (game, play, player, time)")
    else:
        keys = [pd.read_csv(p, usecols=["game_id", "nfl_id"]).drop_duplicates() for p in table_files(cfg, "tracking")]
        tables["tracking_keys"] = pd.concat(keys, ignore_index=True).drop_duplicates() if keys else \
            pd.DataFrame({"game_id": pd.Series(dtype="int64"), "nfl_id": pd.Series(dtype="int64")})
    if pp := table_files(cfg, "player_play"):
        tables["player_play"] = _read_parts(pp, "player_play", PLAYER_PLAY_USECOLS)
    if ct := table_files(cfg, "combine_tracking"):
        tables["combine_tracking"] = _read_parts(ct, "combine_tracking")
    _check_referential_integrity(tables)
    return tables


def _check_referential_integrity(t: dict[str, pd.DataFrame]) -> None:
    ids = set(t["combine"].nfl_id)
    for name in ("players", "careers"):
        if set(t[name].nfl_id) != ids:
            raise SchemaError(f"{name}.nfl_id does not match combine.nfl_id")
    tr = t["tracking"] if "tracking" in t else t["tracking_keys"]
    orphan_players = set(tr.nfl_id) - ids
    if orphan_players:
        raise SchemaError(f"tracking has {len(orphan_players)} players without combine rows")
    orphan_games = set(tr.game_id) - set(t["games"].game_id)
    if orphan_games:
        raise SchemaError(f"tracking has {len(orphan_games)} games missing from games.csv")
    if "player_play" in t:
        pp = t["player_play"]
        if set(pp.nfl_id) - ids:
            raise SchemaError(f"player_play has {len(set(pp.nfl_id) - ids)} players without combine rows")
        if set(pp.game_id) - set(t["games"].game_id):
            raise SchemaError("player_play has games missing from games.csv")
    if "combine_tracking" in t and set(t["combine_tracking"].nfl_id.dropna()) - ids:
        raise SchemaError("combine_tracking has players without combine rows")


def write_manifest(cfg: Config) -> dict:
    """Hash every raw file and record shape; returns the manifest dict."""
    # Order is part of data_version: core tables, tracking parts, player_play, combine_tracking.
    paths = [p for k in ("combine", "players", "careers", "games", "tracking", "player_play", "combine_tracking")
             for p in table_files(cfg, k)]
    entries = {}
    for p in paths:
        name = p.name
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
