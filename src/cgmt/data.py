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
}

PRIMARY_KEYS = {
    "combine": ["nfl_id"],
    "players": ["nfl_id"],
    "careers": ["nfl_id"],
    "games": ["game_id"],
    "tracking": ["game_id", "play_id", "nfl_id", "time"],
}


class SchemaError(ValueError):
    """Raised when a raw table does not match its expected contract."""


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _read(path: Path, table: str) -> pd.DataFrame:
    # "NA" and empty strings are the null markers used in the source files.
    df = pd.read_csv(path, na_values=["NA", ""], keep_default_na=True)
    expected = SCHEMAS[table]
    missing = set(expected) - set(df.columns)
    extra = set(df.columns) - set(expected)
    if missing:
        raise SchemaError(f"{path.name}: missing columns {sorted(missing)}")
    if extra:
        raise SchemaError(f"{path.name}: unexpected columns {sorted(extra)}")
    try:
        df = df.astype(expected)
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


def write_manifest(cfg: Config) -> dict:
    """Hash every raw file and record shape; returns the manifest dict."""
    raw_dir = cfg.path("raw")
    files = cfg["raw_files"]
    names = [files[k] for k in ("combine", "players", "careers", "games")] + list(files["tracking"])
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
