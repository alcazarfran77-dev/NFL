"""Data preparation: contract, ingestion, cleaning, and movement and performance metrics.

NFL project (CGMT, Combine-to-Game Movement Translation): does a prospect's movement in the NFL
Combine drills translate to how he moves and performs in regular-season games? This file holds
every cleaning and preparation function; `train.py` uses them to train, evaluate and register the
models in MLflow.

Stages run by this file (`python src/data_prep.py`)
  0 contract   validates the raw files (data/raw) against the data contract
               (configs/data_contract.yaml, a machine-readable copy of the provider's
               specification, docs/Data_structure_CSV_files.docx)
  1 ingest     reads and types every table, checks keys and referential integrity, and writes the
               manifest with the SHA-256 of every file (data versioning)
  2 cleaning   prospect table with business rules R1-R7
  3 movement   game tracking -> per-play metrics -> player profiles and the SBI (R8-R9)
  3b on-field  player_play -> plays over expected -> player metrics (R10-R15)
  3c Combine   Combine drill tracking -> reliable features per prospect (R16)
  Clean outputs go to data/clean (parquet) and audits to reports/metrics.

Usage
  python src/data_prep.py                        contract + ingest + cleaning (stage "data")
  python src/data_prep.py --stage contract       data contract only
  python src/data_prep.py --stage verify         checks the manifest hashes
  python src/data_prep.py --config configs/smoke.yaml   smoke configuration (CI)

Raw file formats (v3.2.0). Every raw table can be delivered as CSV (`.csv`), gzip-compressed CSV
(`.csv.gz`) or Apache Parquet (`.parquet`). The format is detected from the file's first bytes (not
only its extension). A Parquet table is harmonised to look exactly like the same table read from CSV
(dates and timestamps as ISO 8601 text, booleans as TRUE/FALSE), so both formats feed the same
typed tables and every downstream number is identical. When a table is present in more than one
format, Parquet wins over `.csv.gz`, which wins over `.csv`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import gzip
import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

__version__ = "3.2.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("nfl")


# =============================================================================
# Configuration and reproducibility
# =============================================================================
@dataclass(frozen=True)
class Config:
    """Project configuration (configs/config.yaml), accessible like a dict."""

    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def path(self, key: str) -> Path:
        """Absolute path of a `paths` entry; the folder is created if it does not exist."""
        p = PROJECT_ROOT / self.raw["paths"][key]
        (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
        return p


def _project_path(path: str | os.PathLike) -> Path:
    """Path relative to the working directory if it exists; otherwise relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else PROJECT_ROOT / p


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Reads the configuration (configs/config.yaml by default)."""
    path = _project_path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def set_global_seed(seed: int) -> None:
    """Seeds every random generator; models also receive `random_state`."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _save(df: pd.DataFrame, cfg: Config, name: str, key: str = "processed") -> None:
    """Saves a table to the `paths[key]` folder (parquet or CSV, by extension)."""
    path = cfg.path(key) / name
    if name.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _dump(obj, cfg: Config, name: str) -> None:
    """Saves a dict as JSON in reports/metrics."""
    with open(cfg.path("metrics") / name, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


# =============================================================================
# Ingestion: schemas, typing, manifest (data versioning)
# =============================================================================
# The manifest (data/raw/manifest.json) is the data-versioning contract: every run records the
# SHA-256 of every raw file, so any result can be traced back to the exact bytes it was trained on
# (a lightweight equivalent of DVC).
#
# File names are resolved from patterns (`raw_files` in the configuration), so both the spec names
# (game_tracking_2024.csv, player_play.csv, combine_tracking.csv) and the delivered parts
# (game_tracking_2024_2.csv, player_play_3.csv) are found, in any supported format. A part
# `<stem>_<n>` is ignored when the full `<stem>` file is also present. Tracking is read one file at a
# time (`iter_tracking`) so that full seasons fit in memory.

# Expected columns and types per table (pandas nullable types where the spec allows nulls).
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
    # Spec File 3 (optional): 10 Hz tracking of the Combine drill attempts. Ball rows carry the nfl_id
    # of the attempt's player.
    "combine_tracking": {
        "draft_year": "int64", "event_id": "string", "nfl_id": "Int64", "entity_type": "string",
        "time": "string", "drill_type": "string", "drill_name": "string", "attempt": "int64",
        "x": "float64", "y": "float64", "s": "float64", "a": "float64", "dis": "float64", "dir": "float64",
    },
    # Boolean flags arrive as TRUE/FALSE/NA, so they are read as text and converted explicitly in
    # clean_player_play.
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

# player_play columns used by the analysis (the full 64-column header is still validated).
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
# Data rows of a CSV saved at a spreadsheet's row limit (1,048,576 with the header): the file was
# probably cut.
SPREADSHEET_ROW_LIMIT = 1_048_575
_PART = re.compile(r"^(?P<stem>.+)_\d+$")
_RAW_EXT = re.compile(r"\.(csv(\.gz)?|parquet)$", re.IGNORECASE)
# When the same table is present in several formats, the lowest number wins: Parquet first (typed and
# compressed), then plain CSV over its .gz copy (same content and hash, as in v3.1.0).
FORMAT_PRIORITY = {"parquet": 0, "csv": 1, "csv.gz": 2}


class SchemaError(ValueError):
    """Raised when a raw table breaks its contract."""


def _stem(p: Path) -> str:
    """Name without extension or compression (`x_2.csv.gz` -> `x_2`, `x.parquet` -> `x`)."""
    return _RAW_EXT.sub("", p.name)


def detect_format(p: Path) -> str:
    """File format from its first bytes: "parquet" (PAR1), "csv.gz" (gzip) or "csv".

    The extension is only a hint: a Parquet file saved as .csv, or a gzip file without .gz, is still
    read correctly, and the mismatch is logged.
    """
    with open(p, "rb") as fh:
        head = fh.read(4)
    fmt = "parquet" if head == b"PAR1" else ("csv.gz" if head[:2] == b"\x1f\x8b" else "csv")
    by_name = "parquet" if p.name.lower().endswith(".parquet") else ("csv.gz" if p.name.lower().endswith(".gz") else "csv")
    if fmt != by_name:
        log.warning("%s: the extension says %s but the content is %s; reading it as %s", p.name, by_name, fmt, fmt)
    return fmt


def logical_name(p: Path) -> str:
    """Logical file name, independent of compression (`x.csv.gz` -> `x.csv`, `x.parquet` -> `x.parquet`)."""
    return _stem(p) + (".parquet" if detect_format(p) == "parquet" else ".csv")


def _open_raw(p: Path, mode: str = "rb"):
    """Opens a raw CSV, gzip-compressed or not."""
    kw = {"encoding": "utf-8"} if "t" in mode else {}
    return gzip.open(p, mode, **kw) if detect_format(p) == "csv.gz" else open(p, mode, **kw)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's content (decompressed for .csv.gz; the file bytes for CSV and Parquet)."""
    h = hashlib.sha256()
    with _open_raw(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def resolve_files(raw_dir: Path, spec: str | list[str]) -> tuple[list[Path], list[Path]]:
    """Files of one table: `spec` is a name, a glob pattern or a list of them.

    Returns (used, superseded), sorted by name. A part `<stem>_<n>` is superseded (not read and not
    hashed) when the full `<stem>` file is also present. `x.csv`, `x.csv.gz` and `x.parquet` are the
    same logical file: the format with the highest priority (FORMAT_PRIORITY) is used and the others
    are logged and ignored.
    """
    specs = [spec] if isinstance(spec, str) else list(spec)
    candidates: dict[str, list[Path]] = {}
    for s_ in specs:
        for p in sorted(Path(q) for q in glob.glob(str(Path(raw_dir) / s_))):
            if p.is_file() and _RAW_EXT.search(p.name) and p not in candidates.get(_stem(p), []):
                candidates.setdefault(_stem(p), []).append(p)
    found: dict[str, Path] = {}
    for stem, paths in candidates.items():
        paths = sorted(paths, key=lambda q: FORMAT_PRIORITY[detect_format(q)])
        found[stem] = paths[0]
        if len(paths) > 1:
            log.info("%s: using %s; ignoring %s", stem, paths[0].name, [q.name for q in paths[1:]])
    files = sorted(found.values(), key=_stem)
    names = set(found)
    used, superseded = [], []
    for p in files:
        m = _PART.match(_stem(p))
        (superseded if m and m.group("stem") in names else used).append(p)
    return used, superseded


def table_files(cfg: Config, table: str) -> list[Path]:
    """Files used for `table` (empty if an optional table is absent)."""
    spec = cfg["raw_files"].get(table)
    if spec is None:
        return []
    used, superseded = resolve_files(cfg.path("raw"), spec)
    for p in superseded:
        log.warning("%s: %s is superseded by the full file and is not used", table, p.name)
    if not used and table not in OPTIONAL_TABLES and table != "tracking":
        raise SchemaError(f"required table {table}: no file matches {spec}")
    return used


def read_header(path: Path) -> list[str]:
    """Column names of a raw file, without reading its rows."""
    if detect_format(path) == "parquet":
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    return list(pd.read_csv(path, nrows=0).columns)


def _format_datetime(s: pd.Series) -> pd.Series:
    """Timestamps -> the ISO 8601 text of the CSV deliveries (dates alone when every time is midnight)."""
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    v = s.dropna()
    if len(v) and (v == v.dt.normalize()).all():
        out = s.dt.strftime("%Y-%m-%d")
    else:
        out = s.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3]
    return out.where(s.notna(), None).astype(object)


def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    """Makes a Parquet table look like the same table read from CSV.

    Parquet keeps richer types than CSV text, so without this step the same data would reach the
    pipeline in two shapes. The conversions are the CSV deliveries' own encodings:
      boolean columns           -> "TRUE" / "FALSE" text (NA stays missing)
      timestamp columns         -> ISO 8601 text, e.g. 2025-01-11T23:31:37.500
      date objects (date32)     -> YYYY-MM-DD text
      categorical (dictionary)  -> their plain values
    Numeric and text columns pass through unchanged; the declared schema types them afterwards.
    """
    df = df.copy()
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype(s.cat.categories.dtype if len(s.cat.categories) else object)
            df[c] = s
        if pd.api.types.is_bool_dtype(s):
            df[c] = s.map({True: "TRUE", False: "FALSE"}).astype(object)
        elif pd.api.types.is_datetime64_any_dtype(s):
            df[c] = _format_datetime(s)
        elif s.dtype == object:
            v = s.dropna()
            if len(v) and isinstance(v.iloc[0], dt.datetime):  # datetime before date: it is a subclass
                df[c] = _format_datetime(pd.to_datetime(s, errors="coerce"))
            elif len(v) and isinstance(v.iloc[0], dt.date):
                df[c] = s.map(lambda x: x.isoformat() if isinstance(x, dt.date) else x)
            elif len(v) and isinstance(v.iloc[0], (bool, np.bool_)):
                df[c] = s.map({True: "TRUE", False: "FALSE"}).astype(object)
    return df


def read_raw_table(path: Path, columns: list[str] | None = None, harmonized: bool = True) -> pd.DataFrame:
    """Reads a raw file in any supported format, without imposing the declared types.

    CSV: "NA" and empty strings are the null markers of the deliveries. Parquet: read with its own
    types and, by default, harmonised to the CSV encodings (`harmonize`).
    """
    if detect_format(path) == "parquet":
        df = pd.read_parquet(path, columns=columns)
        return harmonize(df) if harmonized else df
    return pd.read_csv(path, na_values=["NA", ""], keep_default_na=True, usecols=columns, low_memory=False)


def _read(path: Path, table: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Validates the full header and loads (a subset of) the columns with their declared type.

    A missing column is an error. An extra column is logged and ignored (the contract report lists
    it), so an additive change by the provider does not stop the pipeline.
    """
    expected = SCHEMAS[table]
    header = read_header(path)
    missing = set(expected) - set(header)
    extra = set(header) - set(expected)
    if missing:
        raise SchemaError(f"{path.name}: missing columns {sorted(missing)}")
    if extra:
        log.warning("%s: ignoring extra columns %s", path.name, sorted(extra))
    cols = usecols or list(expected)
    df = read_raw_table(path, cols)
    try:
        df = df.astype({c: expected[c] for c in cols})
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{path.name}: type conversion failed: {exc}") from exc
    dups = df.duplicated(PRIMARY_KEYS[table]).sum()
    if dups:
        raise SchemaError(f"{path.name}: {dups} duplicate rows on key {PRIMARY_KEYS[table]}")
    return df


def _read_parts(paths: list[Path], table: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Reads and concatenates the parts of a table; parts may not repeat keys."""
    df = pd.concat([_read(p, table, usecols) for p in paths], ignore_index=True)
    if len(paths) > 1 and df.duplicated(PRIMARY_KEYS[table]).any():
        raise SchemaError(f"{table}: file parts overlap on {PRIMARY_KEYS[table]}")
    return df


def iter_tracking(cfg: Config):
    """Yields (file name, validated frames) one tracking file at a time."""
    for p in table_files(cfg, "tracking"):
        yield p.name, _read(p, "tracking").assign(source_file=p.name)


def load_raw(cfg: Config, tracking: bool = True) -> dict[str, pd.DataFrame]:
    """Loads and validates every raw table.

    tracking=False skips the frames (the pipeline processes them file by file with `iter_tracking`);
    tracking keys are still checked against players and games, file by file.
    """
    tables = {name: _read_parts(table_files(cfg, name), name) for name in ("combine", "players", "careers", "games")}
    if tracking:
        parts = [df for _, df in iter_tracking(cfg)]
        tables["tracking"] = pd.concat(parts, ignore_index=True)
        if tables["tracking"].duplicated(PRIMARY_KEYS["tracking"]).any():
            raise SchemaError("tracking files overlap on (game, play, player, time)")
    else:
        keys = [read_raw_table(p, ["game_id", "nfl_id"]).drop_duplicates() for p in table_files(cfg, "tracking")]
        tables["tracking_keys"] = pd.concat(keys, ignore_index=True).drop_duplicates() if keys else \
            pd.DataFrame({"game_id": pd.Series(dtype="int64"), "nfl_id": pd.Series(dtype="int64")})
    if pp := table_files(cfg, "player_play"):
        tables["player_play"] = _read_parts(pp, "player_play", PLAYER_PLAY_USECOLS)
    if ct := table_files(cfg, "combine_tracking"):
        tables["combine_tracking"] = _read_parts(ct, "combine_tracking")
    _check_referential_integrity(tables)
    return tables


def _check_referential_integrity(t: dict[str, pd.DataFrame]) -> None:
    """Every referenced player and game must exist in its master table."""
    ids = set(t["combine"].nfl_id)
    for name in ("players", "careers"):
        if set(t[name].nfl_id) != ids:
            raise SchemaError(f"{name}.nfl_id does not match combine.nfl_id")
    tr = t["tracking"] if "tracking" in t else t["tracking_keys"]
    orphan_players = set(tr.nfl_id) - ids
    if orphan_players:
        raise SchemaError(f"tracking has {len(orphan_players)} players with no Combine row")
    orphan_games = set(tr.game_id) - set(t["games"].game_id)
    if orphan_games:
        raise SchemaError(f"tracking has {len(orphan_games)} games that are not in the games table")
    if "player_play" in t:
        pp = t["player_play"]
        if set(pp.nfl_id) - ids:
            raise SchemaError(f"player_play has {len(set(pp.nfl_id) - ids)} players with no Combine row")
        if set(pp.game_id) - set(t["games"].game_id):
            raise SchemaError("player_play has games that are not in the games table")
    if "combine_tracking" in t and set(t["combine_tracking"].nfl_id.dropna()) - ids:
        raise SchemaError("combine_tracking has players with no Combine row")


MANIFEST_TABLES = ("combine", "players", "careers", "games", "tracking", "player_play", "combine_tracking")


def write_manifest(cfg: Config) -> dict:
    """Hashes every raw file and records its size and rows; returns the manifest.

    `data_version` = SHA-256 of the content hashes in this order: base tables, tracking parts,
    player_play, combine_tracking. A .csv.gz file is hashed decompressed, so compressing a CSV does
    not change the data version; converting a file to Parquet does (the bytes differ), and the
    manifest records each file's format.
    """
    paths = [p for k in MANIFEST_TABLES for p in table_files(cfg, k)]
    entries = {}
    for p in paths:
        fmt = detect_format(p)
        if fmt == "parquet":
            import pyarrow.parquet as pq

            n_bytes, n_rows = p.stat().st_size, int(pq.ParquetFile(p).metadata.num_rows)
        else:
            with _open_raw(p, "rb") as fh:
                n_bytes = 0
                while block := fh.read(1 << 20):
                    n_bytes += len(block)
            # Text lines minus the header (universal newlines, as in earlier versions; quoted text with
            # line breaks counts as more than one line).
            with _open_raw(p, "rt") as fh:
                n_rows = sum(1 for _ in fh) - 1
        entries[logical_name(p)] = {"sha256": sha256(p), "bytes": n_bytes, "rows": n_rows, "format": fmt,
                                    "stored_as": p.name, "stored_bytes": p.stat().st_size}
    combined = hashlib.sha256("".join(e["sha256"] for e in entries.values()).encode()).hexdigest()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_version": combined[:12],
        "files": entries,
    }
    with open(cfg.path("manifest"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def verify_manifest(cfg: Config) -> list[str]:
    """Compares every raw file's hash with the manifest; returns the files that do not match."""
    m = json.loads(cfg.path("manifest").read_text(encoding="utf-8"))
    raw = cfg.path("raw")
    bad = []
    for name, e in m["files"].items():
        p = raw / e.get("stored_as", name)
        if not p.exists() or sha256(p) != e["sha256"]:
            bad.append(name)
    return bad


# =============================================================================
# Data contract (the provider's specification)
# =============================================================================
# The contract (configs/data_contract.yaml) is a machine-readable copy of
# docs/Data_structure_CSV_files.docx. Every run validates the raw files and writes:
#   reports/metrics/data_contract_report.csv    one row per check (table, column, check, severity,
#                                               result, handling)
#   reports/metrics/data_contract_summary.json  counts, missing files, coverage against the spec
#
# Severity: a missing required file or column, a duplicate key or a broken foreign key is an
# `error` (the run stops when `contract.fail_on_error` is true). Type, null, range, allowed-value,
# pattern and row-count deviations are `warning`s: the business rules (R1-R16) decide how those rows
# are handled and the report documents how many there are. Files are read in any supported format and
# harmonised first (`read_raw_table`), so the same data gives the same report as CSV or Parquet.
TRUE_FALSE = {"TRUE", "FALSE", "T", "F", "1", "0", "1.0", "0.0"}


def load_contract(path: str | Path | None = None) -> dict:
    """Reads the data contract (configs/data_contract.yaml by default)."""
    path = _project_path(path) if path else PROJECT_ROOT / "configs" / "data_contract.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def discover(cfg: Config, table: str, spec: dict) -> tuple[list[Path], list[Path]]:
    """(used, superseded) for a contract table: the pipeline's `raw_files` entry when present (so the
    validator checks exactly what the pipeline reads); otherwise the contract's own patterns."""
    return resolve_files(cfg.path("raw"), cfg["raw_files"].get(table, spec["patterns"]))


def _row(table, file, column, check, severity, ok, n_bad=0, n=0, detail=""):
    return {"table": table, "file": file, "column": column, "check": check, "severity": severity,
            "result": "pass" if ok else "fail", "n_bad": int(n_bad), "share_bad": (n_bad / n) if n else 0.0,
            "detail": detail}


def _type_ok(s: pd.Series, kind: str) -> tuple[bool, str]:
    """Does the column have the type the contract declares? Returns (ok, detail)."""
    v = s.dropna()
    if v.empty:
        return True, "all null"
    if kind == "int":
        if pd.api.types.is_integer_dtype(v):
            return True, str(s.dtype)
        if pd.api.types.is_float_dtype(v):
            ok = bool(np.all(np.isclose(v, np.round(v))))
            return ok, "float column holding whole numbers (nullable int)" if ok else "non-integer values"
        return False, str(s.dtype)
    if kind == "float":
        return bool(pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v)), str(s.dtype)
    if kind == "bool":
        if pd.api.types.is_bool_dtype(v):
            return True, "bool"
        vals = set(v.astype(str).str.upper().str.strip())
        return vals <= TRUE_FALSE, f"codes {sorted(vals)[:6]}"
    if kind == "datetime":
        parsed = pd.to_datetime(v.head(5000), errors="coerce", format="ISO8601")
        return bool(parsed.notna().all()), "ISO 8601"
    return bool(not pd.api.types.is_numeric_dtype(v) or pd.api.types.is_object_dtype(v)), str(s.dtype)


def validate_frame(table: str, file: str, spec: dict, df: pd.DataFrame) -> list[dict]:
    """Per-column checks of one file (or part) against its contract entry."""
    rows = []
    cols = spec["columns"]
    n = len(df)
    missing = [c for c in cols if c not in df.columns]
    extra = [c for c in df.columns if c not in cols]
    rows.append(_row(table, file, "*", "columns_present", "error", not missing, len(missing), len(cols),
                     f"missing {missing}" if missing else f"{len(cols)} columns as specified"))
    if extra:
        rows.append(_row(table, file, "*", "no_extra_columns", "warning", False, len(extra), len(df.columns), f"extra {extra}"))
    for col, c in cols.items():
        if col not in df.columns:
            continue
        s = df[col]
        ok, det = _type_ok(s, c["type"])
        rows.append(_row(table, file, col, f"type_{c['type']}", "warning", ok, 0 if ok else s.notna().sum(), n, det))
        if c.get("spec_type") and c["type"] == "bool" and not pd.api.types.is_numeric_dtype(s.dropna()):
            rows.append(_row(table, file, col, "encoding", "warning", False, s.notna().sum(), n,
                             f"spec says {c['spec_type']} 1.0/0.0; file stores TRUE/FALSE (read as boolean)"))
        nulls = int(s.isna().sum())
        if not c.get("nullable", False):
            rows.append(_row(table, file, col, "not_null", "warning", nulls == 0, nulls, n, f"{nulls} nulls"))
        v = s.dropna()
        if "range" in c and len(v) and pd.api.types.is_numeric_dtype(v):
            lo, hi = c["range"]
            bad = int(((v < lo) | (v > hi)).sum())
            rows.append(_row(table, file, col, f"range[{lo},{hi}]", "warning", bad == 0, bad, n,
                             f"observed [{v.min():.4g}, {v.max():.4g}]"))
        if "values" in c and len(v):
            allowed = c["values"]
            if all(isinstance(a, (int, float)) for a in allowed) and pd.api.types.is_numeric_dtype(v):
                bad_mask = ~np.isclose(v.to_numpy()[:, None], np.asarray(allowed, float)[None, :]).any(axis=1)
                bad_vals = [x.item() if hasattr(x, "item") else x for x in sorted(pd.unique(v[bad_mask]))[:8]]
            else:
                bad_mask = ~v.astype(str).isin([str(a) for a in allowed])
                bad_vals = sorted(pd.unique(v[bad_mask].astype(str)))[:8]
            bad = int(bad_mask.sum())
            rows.append(_row(table, file, col, "allowed_values", "warning", bad == 0, bad, n,
                             f"unexpected {bad_vals}" if bad else f"within {allowed}"))
        if "pattern" in c and len(v):
            rx = re.compile(c["pattern"])
            bad = int((~v.astype(str).map(lambda x: bool(rx.match(x)))).sum())
            rows.append(_row(table, file, col, "pattern", "warning", bad == 0, bad, n, c["pattern"]))
    key = spec.get("key")
    if key and all(k in df.columns for k in key):
        dups = int(df.duplicated(key).sum())
        rows.append(_row(table, file, ",".join(key), "unique_key", "error", dups == 0, dups, n, f"{dups} duplicate keys"))
    return rows


def _read_any(path: Path) -> pd.DataFrame:
    """Reads a raw file without imposing types, so the contract sees the values as delivered
    (Parquet harmonised to the CSV encodings)."""
    return read_raw_table(path)


def validate_all(cfg: Config, contract: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Validates every contract table that is present; returns (report, summary)."""
    contract = contract or load_contract()
    rows, frames = [], {}
    summary = {"files": {}, "formats": {}, "superseded_files": {}, "missing_files": [], "coverage": {}}
    order = ["players", "combine", "careers", "games", "player_play", "tracking", "combine_tracking"]
    for table in order:
        spec = contract["files"][table]
        paths, superseded = discover(cfg, table, spec)
        if superseded:
            summary["superseded_files"][table] = [p.name for p in superseded]
            rows.append(_row(table, "+".join(p.name for p in superseded), "*", "superseded_part", "warning", False,
                             len(superseded), len(superseded) + len(paths), "part ignored: the full file is present"))
        if not paths:
            sev = "error" if spec.get("required") else "warning"
            rows.append(_row(table, spec["spec_name"], "*", "file_present", sev, False, 1, 1,
                             f"no file matches {spec['patterns']}"))
            summary["missing_files"].append({"table": table, "spec_name": spec["spec_name"], "required": spec.get("required", False)})
            continue
        summary["files"][table] = [p.name for p in paths]
        summary["formats"][table] = sorted({detect_format(p) for p in paths})
        parts = []
        for p in paths:
            df = _read_any(p)
            rows += validate_frame(table, p.name, spec, df)
            if len(df) == SPREADSHEET_ROW_LIMIT:
                rows.append(_row(table, p.name, "*", "spreadsheet_row_limit", "warning", False, 1, 1,
                                 f"exactly {SPREADSHEET_ROW_LIMIT:,} data rows: the file was probably cut in a spreadsheet"))
            keep = [c for c in set(sum([list(map(str.strip, k.split(","))) for k in spec.get("soft_foreign_keys", {})], []))
                    | set(spec.get("key", [])) | set(spec.get("foreign_keys", {})) | {"nfl_id", "game_id", "play_id", "event_id", "season", "season_type"}
                    if c in df.columns]
            parts.append(df[keep].copy())
            del df
        tab = pd.concat(parts, ignore_index=True)
        frames[table] = tab
        n = len(tab)
        files = "+".join(p.name for p in paths)
        if len(paths) > 1 and spec.get("key") and all(k in tab.columns for k in spec["key"]):
            dups = int(tab.duplicated(spec["key"]).sum())
            rows.append(_row(table, files, ",".join(spec["key"]), "unique_key_across_parts",
                             "error", dups == 0, dups, n, f"{dups} duplicate keys across parts"))
        if "expected_rows" in spec:
            exp = spec["expected_rows"]
            rows.append(_row(table, files, "*", "row_count", "warning", n == exp, abs(n - exp), exp,
                             f"{n:,} rows vs {exp:,} in the spec ({(n - exp) / exp:+.2%})"))
        if "expected_attempts" in spec and "event_id" in tab:
            got = tab["event_id"].nunique()
            rows.append(_row(table, files, "event_id", "attempt_count", "warning",
                             got == spec["expected_attempts"], abs(got - spec["expected_attempts"]), spec["expected_attempts"],
                             f"{got:,} attempts vs {spec['expected_attempts']:,} in the spec"))
        # foreign keys
        for col, ref in spec.get("foreign_keys", {}).items():
            rt, rc = ref.split(".")
            if rt in frames and col in tab and rc in frames[rt]:
                vals = tab[col].dropna()
                bad = int((~vals.isin(frames[rt][rc])).sum())
                rows.append(_row(table, files, col, f"foreign_key->{ref}", "error", bad == 0, bad, len(vals),
                                 f"{bad} rows reference missing {ref}"))
        for cols, ref in spec.get("soft_foreign_keys", {}).items():
            kc = [c.strip() for c in cols.split(",")]
            if ref in frames and all(c in tab for c in kc) and all(c in frames[ref] for c in kc):
                left = tab[kc].drop_duplicates()
                m = left.merge(frames[ref][kc].drop_duplicates(), on=kc, how="left", indicator=True)
                bad = int((m["_merge"] == "left_only").sum())
                rows.append(_row(table, files, cols, f"soft_foreign_key->{ref}", "warning", bad == 0, bad, len(left),
                                 f"{bad} of {len(left):,} {cols} combinations are not in {ref}"))
        # coverage against the spec (tracking: seasons and players)
        if table == "tracking" and "games" in frames:
            g = pd.concat([read_raw_table(p) for p in discover(cfg, "games", contract["files"]["games"])[0]])
            seasons = sorted(g[g.game_id.isin(tab.game_id.unique())].season.unique().tolist())
            types = g[g.game_id.isin(tab.game_id.unique())].season_type.value_counts().to_dict()
            exp_s = spec.get("expected_seasons", [])
            miss_s = [s for s in exp_s if s not in seasons]
            players = int(tab.nfl_id.nunique())
            games = int(tab.game_id.nunique())
            summary["coverage"]["tracking"] = {"seasons": seasons, "missing_seasons": miss_s, "games": games,
                                               "games_by_type": types, "players": players,
                                               "expected_players": spec.get("expected_players")}
            rows.append(_row(table, files, "season", "season_coverage", "warning", not miss_s,
                             len(miss_s), len(exp_s), f"seasons {seasons}; missing {miss_s}; {games} games {types}"))
            rows.append(_row(table, files, "nfl_id", "player_coverage", "warning",
                             players >= spec.get("expected_players", 0), spec.get("expected_players", 0) - players,
                             spec.get("expected_players", 0), f"{players} of {spec.get('expected_players')} prospects tracked"))
    report = pd.DataFrame(rows)
    handling = contract.get("handling", {}) or {}

    def _handle(r) -> str:
        chk = r["check"].split("[")[0].split("->")[0]
        if r["result"] == "pass":
            return ""
        return handling.get(f"{r['table']}.{r['column']}.{chk}", handling.get(f"{r['table']}.*.{chk}", ""))

    report["handling"] = report.apply(_handle, axis=1)
    summary["n_checks"] = len(report)
    summary["n_errors"] = int(((report.severity == "error") & (report.result == "fail")).sum())
    summary["n_warnings"] = int(((report.severity == "warning") & (report.result == "fail")).sum())
    summary["status"] = "fail" if summary["n_errors"] else ("pass_with_warnings" if summary["n_warnings"] else "pass")
    return report, summary


def write_report(cfg: Config, report: pd.DataFrame, summary: dict) -> None:
    """Writes the contract report and summary to reports/metrics."""
    m = cfg.path("metrics")
    report.to_csv(m / "data_contract_report.csv", index=False)
    with open(m / "data_contract_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)


# =============================================================================
# Prospect cleaning: business rules R1-R7
# =============================================================================
# Rules applied (every threshold lives in configs/config.yaml):
#   R1  Drill values outside physically plausible ranges -> NaN (measurement error).
#   R2  Missing drills are NOT imputed here: a `<drill>_missing` flag is kept and imputation
#       (position-group median) happens inside each training fold.
#   R3  Undrafted players get draft_overall_pick = 260 and is_undrafted = 1.
#   R4  Seasons available = last completed season - draft year + 1, so careers of different length
#       are compared per season.
#   R5  start_rate = games started / (17 x seasons available); starter = start_rate >= 0.5.
#   R6  Age at draft is measured against an approximate draft date (27 April).
#   R7  Conference is reduced to a binary power-conference flag (rare levels pooled).
DRILLS = ["ten_yd_split", "forty", "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps"]
MEASURABLES = ["combine_height", "combine_weight", "hand_size", "arm_length", "wing_span"]
NGS_SCORES = ["ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score"]
# Drills where a LOWER value is better (times); used to orient correlations.
LOWER_IS_BETTER = {"ten_yd_split", "forty", "three_cone", "short_shuttle"}
POSITION_GROUPS = ["DB", "DL", "OL", "TE", "WR"]


def apply_range_rules(df: pd.DataFrame, ranges: dict[str, list[float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """R1: blanks impossible values. Returns (clean table, change log)."""
    df = df.copy()
    changes = []
    for col, (lo, hi) in ranges.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        for idx in df.index[bad]:
            changes.append({"nfl_id": df.at[idx, "nfl_id"], "column": col, "value": df.at[idx, col], "rule": f"outside [{lo}, {hi}]"})
        df.loc[bad, col] = np.nan
    return df, pd.DataFrame(changes, columns=["nfl_id", "column", "value", "rule"])


def build_prospects(tables: dict[str, pd.DataFrame], cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Joins the Combine, player and career tables into one clean prospect table."""
    br = cfg["business_rules"]
    combine, audit = apply_range_rules(tables["combine"], br["drill_valid_ranges"])
    players = tables["players"].copy()
    careers = tables["careers"].copy()

    df = combine.merge(players, on=["nfl_id", "draft_year"], how="inner", validate="1:1")
    df = df.merge(careers, on="nfl_id", how="inner", validate="1:1")
    if len(df) != len(combine):
        raise ValueError("draft_year does not match between the Combine and player tables")

    # Types
    df["combine_position"] = pd.Categorical(df["combine_position"].astype(str), categories=POSITION_GROUPS)
    df["nfl_position"] = df["nfl_position"].astype("category")
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")

    # R2: missing flags (informative: prospects choose which drills to run)
    for col in DRILLS:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    df["n_drills_run"] = df[DRILLS].notna().sum(axis=1)

    # R3: draft capital
    df["is_undrafted"] = df["draft_overall_pick"].isna().astype(int)
    df["draft_pick_filled"] = df["draft_overall_pick"].astype("float").fillna(br["undrafted_pick"])
    df["log_draft_pick"] = np.log(df["draft_pick_filled"])
    df["draft_round_filled"] = df["draft_round"].astype("float").fillna(8)  # 8 = undrafted

    # R6: age at draft
    draft_date = pd.to_datetime(df["draft_year"].astype(str) + "-" + br["draft_reference_month_day"])
    df["age_at_draft"] = (draft_date - df["birth_date"]).dt.days / 365.25

    # R7: conference
    df["power_conference"] = df["college_conference"].isin(br["power_conferences"]).astype(int)

    # Derived athletic features (documented, standard in scouting analytics)
    # Speed Score (Barnwell): 40-yard time adjusted for weight.
    df["speed_score"] = df["combine_weight"] * 200 / df["forty"] ** 4
    # Acceleration share: fraction of the 40 spent in the first 10 yards (lower = faster start).
    df["ten_split_share"] = df["ten_yd_split"] / df["forty"]
    # Momentum off the line: weight moved per second over the first 10 yards (lb/s).
    df["ten_split_momentum"] = df["combine_weight"] / df["ten_yd_split"]

    # R4 + R5: career outcome normalised by opportunity
    df["seasons_available"] = br["last_completed_season"] - df["draft_year"] + 1
    possible = br["games_per_season"] * df["seasons_available"]
    df["start_rate"] = (df["career_games_started"] / possible).clip(0, 1)
    df["is_starter"] = (df["start_rate"] >= br["starter_threshold"]).astype(int)
    df["snaps_per_season"] = (df["career_offensive_snaps"] + df["career_defensive_snaps"]) / df["seasons_available"]

    return df.sort_values("nfl_id").reset_index(drop=True), audit


def missingness_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """Share of prospects missing each drill, by Combine position group."""
    return df.groupby("combine_position", observed=True)[DRILLS].apply(lambda g: g.isna().mean()).round(3)


# =============================================================================
# Game movement: 10 Hz tracking (rules R8-R9)
# =============================================================================
# Every metric is anchored to the snap, so it is comparable across plays and independent of play
# direction (no left/right normalisation is needed: everything is a speed, a distance or a change of
# angle).
#
# Per player-play metrics
#   burst_1s / burst_2s  straight-line distance from the snap point at t = 1.0 / 2.0 s (yd).
#                        burst_2s is the game analog of the Combine 10-yard split, which prospects
#                        cover in ~1.5-1.9 s.
#   get_off_1s           speed at t = 1.0 s after the snap (yd/s).
#   peak_accel_2s        peak acceleration in the first 2.0 s (yd/s^2).
#   top_speed            maximum speed after the snap (yd/s); the 40-yard analog.
#   max_decel            sharpest braking, -d(speed)/dt on the 3-frame smoothed speed while moving at
#                        >= 1 yd/s (yd/s^2); the analog of the shuttle / 3-cone plant.
#   max_cut_deg          largest change of travel direction over 0.5 s while moving at >= 2 yd/s at
#                        both ends (degrees).
#
# Player level
#   level      = mean over the player's plays. The mean is unbiased whatever the number of plays; an
#                80th-percentile "capacity" was tried first (v0.3.0) and dropped: it reads ~0.4 yd low
#                at 3 plays and ~0.2 yd low at 5, and because starters log more plays that bias leaked
#                playing time into the metric. The 80th percentile is kept as `<metric>_q80` for the
#                sensitivity analysis only.
#   shrinkage  = empirical Bayes toward the position-group mean, weight n/(n+k), so a player seen
#                on 3 plays is not ranked with the same confidence as one seen on 30.
#   Snap Burst Index (SBI) = 50 + 10 * z of the shrunk burst_2s within the position group.
PLAY_KEY = ["game_id", "play_id", "nfl_id"]
METRICS = ["burst_1s", "burst_2s", "get_off_1s", "peak_accel_2s", "top_speed", "max_decel", "max_cut_deg"]


def clean_tracking(tr: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Types, sorts and checks the tracking frames. Returns (frames, audit counts)."""
    rules = cfg["tracking_rules"]
    df = tr.copy()
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")
    df = df.sort_values(PLAY_KEY + ["time"]).reset_index(drop=True)
    audit: dict[str, int] = {"frames_in": len(df)}

    # Frame spacing must be 1 / frame rate; plays with gaps are dropped (none expected).
    step = 1.0 / rules["frame_rate_hz"]
    gaps = df.groupby(PLAY_KEY)["time"].diff().dt.total_seconds()
    bad_plays = df.loc[gaps.notna() & ~np.isclose(gaps, step), PLAY_KEY].drop_duplicates()
    audit["plays_with_frame_gaps"] = len(bad_plays)
    if len(bad_plays):
        df = df.merge(bad_plays.assign(_bad=1), on=PLAY_KEY, how="left")
        df = df[df["_bad"].isna()].drop(columns="_bad")

    # Physically impossible speed / acceleration -> NaN, then interpolation within the play.
    for col, cap in (("s", rules["speed_cap"]), ("a", rules["accel_cap"])):
        bad = df[col] > cap
        audit[f"frames_{col}_above_cap"] = int(bad.sum())
        df.loc[bad, col] = np.nan
        df[col] = df.groupby(PLAY_KEY)[col].transform(lambda s: s.interpolate(limit_direction="both"))

    # Positions far outside the field are counted (and kept: players do leave the field).
    audit["frames_out_of_field"] = int(((df.x < -10) | (df.x > 130) | (df.y < -10) | (df.y > 63.3)).sum())
    audit["frames_out"] = len(df)
    return df.reset_index(drop=True), audit


def _heading_change(dir_deg: np.ndarray, lag: int) -> np.ndarray:
    """Absolute (wrapped) change of travel direction over `lag` frames (degrees)."""
    h = np.deg2rad(dir_deg)
    return np.abs(np.rad2deg(np.angle(np.exp(1j * (h[lag:] - h[:-lag])))))


def play_metrics(frames: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """One row per player-play with the snap-anchored movement metrics."""
    rules = cfg["tracking_rules"]
    hz = rules["frame_rate_hz"]
    i1, i2 = int(rules["get_off_horizon_s"] * hz), int(rules["burst_horizon_s"] * hz)
    lag = int(rules["cut_window_s"] * hz)
    snap_events = set(rules["snap_events"])
    out, skipped_no_snap, skipped_short = [], 0, 0

    for key, d in frames.groupby(PLAY_KEY, sort=False):
        ev = d["event"].to_numpy(dtype=object)
        snap_idx = next((i for i, e in enumerate(ev) if e in snap_events), None)
        if snap_idx is None:
            skipped_no_snap += 1
            continue
        post = d.iloc[snap_idx:]
        if len(post) < rules["min_post_snap_frames"] + 1:
            skipped_short += 1
            continue
        x, y = post["x"].to_numpy(), post["y"].to_numpy()
        s, a = post["s"].to_numpy(), post["a"].to_numpy()
        disp = np.hypot(x - x[0], y - y[0])

        s_smooth = pd.Series(s).rolling(3, center=True, min_periods=1).mean().to_numpy()
        dsdt = np.gradient(s_smooth, 1.0 / hz)
        moving = s_smooth >= 1.0
        max_decel = float(-dsdt[moving].min()) if moving.any() else np.nan

        dh = _heading_change(post["dir"].to_numpy(), lag)
        fast = (s[lag:] >= rules["cut_min_speed"]) & (s[:-lag] >= rules["cut_min_speed"])
        max_cut = float(dh[fast].max()) if fast.any() else np.nan

        out.append({
            "game_id": key[0], "play_id": key[1], "nfl_id": key[2],
            "snap_event": ev[snap_idx], "post_snap_frames": len(post),
            "burst_1s": disp[i1], "burst_2s": disp[i2], "get_off_1s": s[i1],
            "peak_accel_2s": a[: i2 + 1].max(), "top_speed": s.max(),
            "max_decel": max_decel, "max_cut_deg": max_cut,
        })

    audit = {"plays_no_snap": skipped_no_snap, "plays_too_short": skipped_short, "plays_kept": len(out)}
    return pd.DataFrame(out), audit


def _shrink(group: pd.DataFrame, col: str) -> pd.Series:
    """Empirical-Bayes shrinkage of player means toward the position-group mean.

    k = sigma2_within / tau2_between, weight = n / (n + k). sigma2_within is the per-play variance
    pooled within players; tau2 is the variance of player means net of sampling noise (method of
    moments, floored at 5% of the observed variance so that no group collapses to a single point).
    """
    mean = group[f"{col}_mean"]
    w = shrink_weight(group, col)
    grand = np.average(mean, weights=group["n_plays"])
    return grand + w * (mean - grand)


def shrink_weight(group: pd.DataFrame, col: str) -> pd.Series:
    """Empirical-Bayes weight n / (n + k) on the player's own mean (= its reliability)."""
    n = group["n_plays"]
    s2w = group[f"{col}_var"].fillna(group[f"{col}_var"].median())
    sigma2 = np.average(s2w, weights=n - 1)
    total = group[f"{col}_mean"].var(ddof=1)
    tau2 = max(total - np.mean(sigma2 / n), 0.05 * total)
    return n / (n + sigma2 / tau2)


def player_profiles(plays: pd.DataFrame, prospects: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Aggregates per-play metrics to players, applies shrinkage and computes the Snap Burst Index."""
    rules = cfg["tracking_rules"]
    q = rules["capacity_quantile"]
    g = plays.groupby("nfl_id")
    prof = g[METRICS].mean().add_suffix("_mean")
    prof = prof.join(g[METRICS].quantile(q).add_suffix("_q80"))  # sensitivity analysis only
    prof = prof.join(g[METRICS].var(ddof=1).add_suffix("_var"))
    prof = prof.join(g.size().rename("n_plays"))
    prof = prof.join(g["game_id"].nunique().rename("n_games"))
    prof = prof.reset_index().merge(prospects[["nfl_id", "combine_position"]], on="nfl_id", how="left")
    prof = prof[prof["n_plays"] >= rules["min_plays_per_player"]].copy()

    for col in METRICS:
        prof[f"{col}_eb"] = prof.groupby("combine_position", observed=True, group_keys=False).apply(
            lambda grp, c=col: _shrink(grp, c)
        )
    # Reliability of each player's SBI (share of his own mean kept after shrinkage).
    prof["sbi_reliability"] = prof.groupby("combine_position", observed=True, group_keys=False).apply(
        lambda grp: shrink_weight(grp, "burst_2s")
    )
    # Snap Burst Index: position-normalised, on a 50 +/- 10 scale like other scouting grades.
    grp = prof.groupby("combine_position", observed=True)["burst_2s_eb"]
    z = (prof["burst_2s_eb"] - grp.transform("mean")) / grp.transform("std")
    prof["sbi"] = rules["sbi_center"] + rules["sbi_scale"] * z
    # The same normalisation for the other shrunk metrics (classifier features).
    for col in METRICS:
        gcol = prof.groupby("combine_position", observed=True)[f"{col}_eb"]
        prof[f"{col}_z"] = (prof[f"{col}_eb"] - gcol.transform("mean")) / gcol.transform("std")
    return prof.drop(columns=[c for c in prof.columns if c.endswith("_var")]).reset_index(drop=True)


def metric_reliability(plays: pd.DataFrame, profiles: pd.DataFrame, group: str = "combine_position") -> pd.DataFrame:
    """Is a movement metric a stable player trait? One row per (position group, metric).

    icc            share of the variance of player means net of sampling noise
                   (method of moments; <= 0 means no detectable differences between players)
    mean_weight    mean empirical-Bayes weight n/(n+k) = reliability of a player's estimate
    anova_p        one-way ANOVA p-value for "every player in the group moves the same"
    """
    keep = plays[plays.nfl_id.isin(profiles.nfl_id)]
    rows = []
    for pos, g in keep.groupby(group, observed=True):
        for m in METRICS:
            gg = g.groupby("nfl_id")[m]
            n, mu, var = gg.size(), gg.mean(), gg.var(ddof=1)
            sigma2 = np.average(var.fillna(var.median()), weights=n - 1)
            total = mu.var(ddof=1)
            tau2 = total - np.mean(sigma2 / n)
            groups = [v.dropna().to_numpy() for _, v in gg]
            groups = [v for v in groups if len(v) > 0]
            f, p = stats.f_oneway(*groups) if len(groups) > 1 else (np.nan, np.nan)
            w = n / (n + sigma2 / tau2) if tau2 > 0 else pd.Series(0.0, index=n.index)
            rows.append({group: pos, "metric": m, "players": len(n), "plays": int(n.sum()),
                         "icc": tau2 / total if total > 0 else np.nan, "mean_weight": float(w.mean()),
                         "anova_f": float(f), "anova_p": float(p)})
    return pd.DataFrame(rows)


def estimator_bias(plays: pd.DataFrame, metric: str, ns: list[int], n_sims: int, seed: int, min_plays: int = 20) -> pd.DataFrame:
    """Bias of the mean / median / 80th percentile when a player is seen on only n plays.

    Players with >= min_plays plays are the ground truth, and subsamples of n of their plays are drawn.
    """
    rng = np.random.default_rng(seed)
    big = plays.groupby("nfl_id").filter(lambda g: len(g) >= min_plays)
    est = {"mean": np.mean, "median": np.median, "q80": lambda v: np.quantile(v, 0.8)}
    rows = []
    for _, g in big.groupby("nfl_id"):
        v = g[metric].to_numpy()
        for name, fn in est.items():
            truth = fn(v)
            for n in ns:
                sims = [fn(rng.choice(v, n, replace=False)) for _ in range(n_sims)]
                rows.append({"estimator": name, "n_plays": n, "bias": float(np.mean(sims) - truth)})
    out = pd.DataFrame(rows).groupby(["estimator", "n_plays"], as_index=False)["bias"].mean()
    out["players_used"] = big.nfl_id.nunique()
    return out


def speed_curves(frames: pd.DataFrame, plays: pd.DataFrame, nfl_ids: list[int], horizon_s: float = 4.0) -> pd.DataFrame:
    """Post-snap speed traces (t, speed) of the given players, for plotting."""
    hz = 10
    sub = frames[frames.nfl_id.isin(nfl_ids)]
    rows = []
    for key, d in sub.groupby(PLAY_KEY, sort=False):
        ev = d["event"].to_numpy(dtype=object)
        idx = next((i for i, e in enumerate(ev) if e in {"ball_snap", "snap_direct"}), None)
        if idx is None:
            continue
        post = d.iloc[idx: idx + int(horizon_s * hz) + 1]
        rows.append(pd.DataFrame({"nfl_id": key[2], "game_id": key[0], "play_id": key[1],
                                  "t": np.arange(len(post)) / hz, "speed": post["s"].to_numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =============================================================================
# On-field performance: player_play (rules R10-R15)
# =============================================================================
# Every metric is measured "over expected": the play's outcome minus the mean outcome of all prospect
# plays in the same context, so a player is not rewarded for his role or his routes. Positive is
# always better.
#
# Family  Plays                              Metric (per play)                                  Context of the "expected"
# DL      pass-rush snaps (not dropping      goe  = expected get-off - get-off (s; + = quicker) alignment (EDGE / INTERIOR_LINE / other)
#         into coverage)                     proe = pressure - expected pressure                alignment
# OL      pass-block snaps                   ppoe = expected - peak pressure probability        alignment (T / G / C) x dropback quintile
#                                            paoe = expected - pressure allowed (0/1)           same
# WR/TE   routes with a separation reading   soe  = separation - expected separation (yd)       route x alignment (route only if < 30 plays)
#
# Player level: mean of valid plays, empirical-Bayes shrinkage toward the Combine position-group mean
# (`<metric>_eb`), its reliability weight (`<metric>_rel`) and a within-family z (`<metric>_z`).
#
# Business rules (configs/config.yaml -> performance_rules)
#   R10 regular season only; R11 plays nullified by penalty are removed;
#   R12 NGS get-off outside [0.30, 2.00) s is treated as missing;
#   R13 minimum valid plays per family (DL 20 rush snaps, OL 50 pass-block snaps, WR/TE 30 routes);
#   R14 DBs get no on-field metric: the file has their alignment and assignment, but not the coverage
#       outcome;
#   R15 downs outside the spec's 1-4 are removed (down 0: two-point tries).
# Boolean flags are read whether the file carries TRUE/FALSE text (as delivered), 1.0/0.0 floats (as
# the spec documents) or real booleans (Parquet); alignment cells come from
# `performance_rules.alignment_codes`, which also maps the spec's DE / DT / NT codes.
ONFIELD_METRICS = ["goe", "proe", "ppoe", "paoe", "soe"]
METRIC_FAMILY = {"goe": "DL", "proe": "DL", "ppoe": "OL", "paoe": "OL", "soe": "WRTE"}
MIN_PLAYS_KEY = {"DL": "dl_rush", "OL": "ol_pass_block", "WRTE": "wrte_route"}
BOOL_COLS = ["target", "pressure_allowed", "blitzing", "unblocked_pressure"]
KEY = ["game_id", "play_id", "nfl_id"]

_BOOL_CODES = {"TRUE": 1.0, "T": 1.0, "1": 1.0, "1.0": 1.0, "FALSE": 0.0, "F": 0.0, "0": 0.0, "0.0": 0.0}


def _parse_bool(s: pd.Series) -> pd.Series:
    """TRUE/FALSE, True/False, 1/0 and 1.0/0.0 -> 1.0/0.0; anything else (NA included) -> NaN."""
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce").astype("float64")
        return v.where(v.isin([0.0, 1.0]))
    return s.astype("string").str.strip().str.upper().map(_BOOL_CODES).astype("float64")


def _alignment(codes: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """lined_up_position -> alignment cell of the "expected" model; unmapped codes -> OTHER."""
    return codes.map(mapping).fillna("OTHER").astype(codes.dtype)


def clean_player_play(pp: pd.DataFrame, games: pd.DataFrame, prospects: pd.DataFrame,
                      cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (regular-season plays, play context for every season type, audit)."""
    rules = cfg["performance_rules"]
    df = pp.merge(games[["game_id", "season", "season_type", "week"]], on="game_id", how="left", validate="m:1")
    df = df.merge(prospects[["nfl_id", "combine_position", "nfl_position", "draft_year"]], on="nfl_id",
                  how="left", validate="m:1")
    for c in BOOL_COLS:
        df[c] = _parse_bool(df[c])
    df["is_pass"] = df["pass_result"].notna().astype(int)
    df["combine_position"] = df["combine_position"].astype(str)
    df["family"] = df["combine_position"].map(rules["families"])
    df["rookie_season"] = (df["season"] == df["draft_year"]).astype(int)
    audit = {"pp_rows_in": len(df), "pp_players": int(df.nfl_id.nunique())}

    # Context table to join with tracking (tracking also covers the playoffs).
    context = df[KEY + ["season_type", "lined_up_position", "route_ran", "is_pass",
                        "coverage_assignment_at_snap", "player_get_off"]].copy()

    keep = df["season_type"].isin(rules["season_types"])  # R10
    audit["pp_rows_not_regular_season"] = int((~keep).sum())
    if rules["exclude_nullified"]:  # R11
        nullified = df["play_nullified_by_penalty"].eq("Y")
        audit["pp_rows_nullified"] = int((keep & nullified).sum())
        keep &= ~nullified
    valid_downs = rules.get("valid_downs")  # R15
    if valid_downs:
        bad_down = ~df["down"].isin(valid_downs)
        audit["pp_rows_invalid_down"] = int((keep & bad_down).sum())
        keep &= ~bad_down
    lo, hi = rules["get_off_valid_s"]  # R12
    bad_go = df["player_get_off"].notna() & ~df["player_get_off"].between(lo, hi, inclusive="left")
    audit["pp_get_off_out_of_range"] = int((keep & bad_go).sum())
    df.loc[bad_go, "player_get_off"] = np.nan
    out = df[keep].reset_index(drop=True)
    audit["pp_rows_regular_season"] = len(out)
    return out, context, audit


def _expected(df: pd.DataFrame, value: str, keys: list[str], fallback: list[str] | None = None,
              min_n: int = 1) -> pd.Series:
    """Mean of `value` within the context `keys`; cells with < min_n plays use `fallback`."""
    grp = df.groupby(keys, dropna=False)[value]
    exp = grp.transform("mean")
    if fallback is not None:
        small = grp.transform("count") < min_n
        exp = exp.where(~small, df.groupby(fallback, dropna=False)[value].transform("mean"))
    return exp


def family_plays(plays: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Per-family play tables with the over-expected metrics."""
    rules = cfg["performance_rules"]
    codes = rules.get("alignment_codes", {
        "DL": {"EDGE": "EDGE", "INTERIOR_LINE": "INTERIOR_LINE"},
        "OL": {"T": "T", "G": "G", "C": "C"},
        "WRTE": {"WR": "WR", "SLOT_WR": "SLOT_WR", "TE": "TE"}})
    out = {}

    # DL, pass-rush snaps: a pass play on which the lineman rushed (no coverage assignment).
    dl = plays[(plays.family == "DL") & plays["sack"].notna() & (plays.is_pass == 1)
               & (plays.coverage_assignment.isna() | plays.coverage_assignment.eq("NO_ASSIGNMENT"))].copy()
    dl["alignment"] = _alignment(dl.lined_up_position, codes["DL"])
    dl["pressure"] = dl["time_to_pressure"].notna().astype(float)
    dl["goe"] = _expected(dl, "player_get_off", ["alignment"]) - dl["player_get_off"]
    dl["proe"] = dl["pressure"] - _expected(dl, "pressure", ["alignment"])
    out["DL"] = dl

    # OL, pass-block snaps.
    ol = plays[(plays.family == "OL") & plays.peak_pressure_probability_allowed.notna()].copy()
    ol["alignment"] = _alignment(ol.lined_up_position, codes["OL"])
    q = rules["dropback_quantiles"]
    ol["dropback_bin"] = pd.qcut(ol.dropback_duration, q, labels=False, duplicates="drop").fillna(-1).astype(int)
    ol["ppoe"] = _expected(ol, "peak_pressure_probability_allowed", ["alignment", "dropback_bin"]) - ol.peak_pressure_probability_allowed
    # paoe only where pressure_allowed is recorded (before: missing = no pressure; 0 rows affected here).
    ol["paoe"] = _expected(ol, "pressure_allowed", ["alignment", "dropback_bin"]) - ol["pressure_allowed"]
    out["OL"] = ol

    # WR/TE, routes with a separation reading.
    wr = plays[(plays.family == "WRTE") & plays.separation_at_pass_forward.notna() & plays.route_ran.notna()].copy()
    wr["alignment"] = _alignment(wr.lined_up_position, codes["WRTE"])
    wr["soe"] = wr.separation_at_pass_forward - _expected(wr, "separation_at_pass_forward", ["route_ran", "alignment"],
                                                          fallback=["route_ran"], min_n=30)
    out["WRTE"] = wr
    return out


def shrink_frame(table: pd.DataFrame, metric: str, group: str = "combine_position") -> pd.DataFrame:
    """Empirical-Bayes shrinkage of player means toward the group mean (same estimator as the SBI)."""
    res = []
    for _, g in table.groupby(group, observed=True):
        n, mean = g[f"n_{metric}"], g[f"{metric}_mean"]
        var = g[f"{metric}_var"].fillna(g[f"{metric}_var"].median())
        sigma2 = np.average(var, weights=np.maximum(n - 1, 1))
        total = mean.var(ddof=1) if len(g) > 1 else 0.0
        tau2 = max(total - np.mean(sigma2 / n), 0.05 * total) if total > 0 else np.nan
        w = n / (n + sigma2 / tau2) if np.isfinite(tau2) and tau2 > 0 else pd.Series(0.0, index=g.index)
        grand = np.average(mean, weights=n)
        res.append(pd.DataFrame({f"{metric}_eb": grand + w * (mean - grand), f"{metric}_rel": w}, index=g.index))
    return pd.concat(res) if res else pd.DataFrame(columns=[f"{metric}_eb", f"{metric}_rel"])


def player_onfield(fam_plays: dict[str, pd.DataFrame], prospects: pd.DataFrame, cfg: Config,
                   rookie_only: bool = False) -> pd.DataFrame:
    """One row per prospect with every valid on-field metric (NaN where not measured)."""
    rules = cfg["performance_rules"]
    base = prospects[["nfl_id", "combine_position"]].copy()
    base["combine_position"] = base["combine_position"].astype(str)
    base["family"] = base["combine_position"].map(rules["families"])
    for m in ONFIELD_METRICS:
        fam = METRIC_FAMILY[m]
        d = fam_plays[fam]
        if rookie_only:
            d = d[d.rookie_season == 1]
        d = d.dropna(subset=[m])
        g = d.groupby("nfl_id")[m]
        st = pd.DataFrame({f"n_{m}": g.size(), f"{m}_mean": g.mean(), f"{m}_var": g.var(ddof=1)}).reset_index()
        st = st[st[f"n_{m}"] >= rules["min_plays"][MIN_PLAYS_KEY[fam]]]  # R13
        st = st.merge(base[["nfl_id", "combine_position"]], on="nfl_id", how="left")
        st = st.join(shrink_frame(st, m))
        base = base.merge(st.drop(columns=["combine_position", f"{m}_var"]), on="nfl_id", how="left")
        for c in (f"n_{m}", f"{m}_mean", f"{m}_eb", f"{m}_rel"):
            base[c] = pd.to_numeric(base[c], errors="coerce").astype("float64")
        # Within-family z of the shrunk value (positive = better), for the classifiers.
        base[f"{m}_z"] = np.nan
        fam_rows = base.family == fam
        v = base.loc[fam_rows, f"{m}_eb"]
        base.loc[fam_rows, f"{m}_z"] = (v - v.mean()) / v.std(ddof=1)
    prim = rules["primary_metric"]
    base["perf_index"] = np.nan
    for fam, m in prim.items():
        rows = base.family == fam
        base.loc[rows, "perf_index"] = base.loc[rows, f"{m}_z"]
    return base


def reliability_table(fam_plays: dict[str, pd.DataFrame], profiles: pd.DataFrame) -> pd.DataFrame:
    """ICC (method of moments), mean empirical-Bayes weight and one-way ANOVA per on-field metric x position."""
    rows = []
    for m in ONFIELD_METRICS:
        fam = METRIC_FAMILY[m]
        qualified = profiles.loc[profiles[f"n_{m}"].notna(), "nfl_id"]
        d = fam_plays[fam].dropna(subset=[m])
        d = d[d.nfl_id.isin(qualified)]
        for pos, g in d.groupby("combine_position"):
            gg = g.groupby("nfl_id")[m]
            n, mu, var = gg.size(), gg.mean(), gg.var(ddof=1)
            if len(n) < 3:
                continue
            sigma2 = np.average(var.fillna(var.median()), weights=np.maximum(n - 1, 1))
            total = mu.var(ddof=1)
            tau2 = total - np.mean(sigma2 / n)
            f, p = stats.f_oneway(*[v.to_numpy() for _, v in gg])
            w = n / (n + sigma2 / tau2) if tau2 > 0 else pd.Series(0.0, index=n.index)
            rows.append({"metric": m, "family": fam, "combine_position": pos, "players": len(n),
                         "median_plays": float(n.median()), "icc": tau2 / total if total > 0 else np.nan,
                         "mean_weight": float(w.mean()), "anova_f": float(f), "anova_p": float(p)})
    return pd.DataFrame(rows)


def rookie_stability(fam_plays: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Does the rookie season predict the later ones? Spearman of the rookie mean vs the later mean."""
    rules = cfg["performance_rules"]
    rows = []
    for m in ONFIELD_METRICS:
        fam = METRIC_FAMILY[m]
        d = fam_plays[fam].dropna(subset=[m])
        d = d[d.draft_year < rules.get("stability_last_class", 2025)]
        mn = rules["min_plays"][MIN_PLAYS_KEY[fam]]
        a = d[d.rookie_season == 1].groupby("nfl_id")[m].agg(["mean", "size"])
        b = d[d.rookie_season == 0].groupby("nfl_id")[m].agg(["mean", "size"])
        j = a[a["size"] >= mn].join(b[b["size"] >= mn], lsuffix="_rookie", rsuffix="_later", how="inner")
        if len(j) >= 8:
            r, p = stats.spearmanr(j["mean_rookie"], j["mean_later"])
            rows.append({"metric": m, "family": fam, "players": len(j), "rho": r, "p_value": p})
    return pd.DataFrame(rows)


def context_adjusted_burst(track_plays: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Snap burst net of the play context, to re-check its reliability.

    WR/TE: route run; DB: coverage assignment at the snap and pass/run; OL/DL: alignment and pass/run.
    """
    d = track_plays.merge(context, on=KEY, how="left")
    pos = d["combine_position"].astype(str)
    d["context"] = np.where(pos.isin(["WR", "TE"]), d["route_ran"].fillna("NONE"),
                   np.where(pos.eq("DB"), d["coverage_assignment_at_snap"].fillna("NONE") + "|" + d["is_pass"].astype(str),
                            d["lined_up_position"].fillna("NONE") + "|" + d["is_pass"].astype(str)))
    grp = d.groupby([pos, "context"])["burst_2s"]
    small = grp.transform("count") < 5
    exp = grp.transform("mean").where(~small, d.groupby(pos)["burst_2s"].transform("mean"))
    d["burst_2s_ctx"] = d["burst_2s"] - exp
    return d


# =============================================================================
# Combine drill tracking (spec File 3, rule R16)
# =============================================================================
# The official Combine results are stopwatch times; drill tracking shows how each time was produced.
# The metrics use the same definitions as game tracking.
#
# Calibration finding (v2.2.0): **the start of a tracked attempt is not repeatable.** A tracked 40
# starts at a variable point of the launch (first-frame speed 0.2-3.2 yd/s), so anything timed from the
# first frame (distance at 1 s / 2 s, 10-yd split from the start, time to 90% of top speed) barely
# repeats between a player's two attempts (0-5 yd split: r = 0.16). Splits between distance marks do
# not depend on when the clock started and repeat well (5-10 yd 0.87, 10-20 yd 0.98, top speed 0.99).
# Features therefore pass a reliability gate, the same rule the game metrics follow (ICC before use):
# a metric can only be a model feature if it repeats at r >= 0.5, and each construct gets a single
# feature (its most reliable metric). `attempt_reliability` writes every metric, used or not and why,
# to reports/metrics/ct_reliability.csv.
#
# Per attempt (PLAYER rows; the attempt starts at the frame before the first one at >= start_speed)
#   top_speed, peak_accel, peak_decel (3-frame smoothed braking while moving at >= 1 yd/s), max_cut_deg,
#   burst_1s / burst_2s, t90_top_s, duration_s
#   40 yards: split_0_5_s, split_5_10_s, split_10_20_s (time between marks, interpolated),
#             split_10yd_s (from the start, like the official split)
#   3-cone and shuttle: n_plants and plant_speed_ratio (speed minima below 50% of the previous peak;
#             mean ratio of the speed at the plant to that peak = speed kept through the turn)
#
# Player features (CT_FEATURES; best attempt per metric, as the official Combine keeps the best)
#   construct                feature                        reliability in the delivered file
#   sprint acceleration      ct_forty_split_5_10_s          0.87 (5 -> 10 yd; lower = better)
#   sprint top speed         ct_forty_top_speed             0.99 (same construct as the 10-20 yd split, 0.98)
#   change of direction      ct_shuttle_plant_speed_ratio   0.54 (speed kept through the shuttle turns)
#   position-drill speed     ct_skill_speed_z               top speed z-scored within each drill
#                                                           (position group x drill name), player mean
#   position-drill accel     ct_skill_accel_z               peak acceleration, same scale
#
# Business rule (configs/config.yaml -> combine_tracking_rules)
#   R16 ball rows and rows without nfl_id are removed; frames must be 0.1 s apart within an attempt;
#       speed / acceleration above the game caps -> NaN and interpolation; attempts shorter than
#       `min_frames` or that never reach `start_speed` are dropped.
ATTEMPT_KEY = ["event_id"]
FORTY, THREE_CONE, SHUTTLE = "FORTY_YARD_DASH", "THREE_CONE_DRILL", "SHORT_SHUTTLE"
SKILL_PREFIX = "SKILL_DRILLS_"

CT_FEATURES = [
    "ct_forty_split_5_10_s", "ct_forty_top_speed", "ct_shuttle_plant_speed_ratio",
    "ct_skill_speed_z", "ct_skill_accel_z",
]
# For these, lower is better (times); for the rest, higher is better.
CT_LOWER_IS_BETTER = {"ct_forty_split_5_10_s"}

_BEST = {  # drill group -> {attempt metric: "max" | "min"} for the player features
    "forty": {"split_5_10_s": "min", "top_speed": "max"},
    "shuttle": {"plant_speed_ratio": "max"},
}
_SKILL_Z = {"top_speed": "speed", "peak_accel": "accel"}

# Metrics whose between-attempt reliability is reported (features or not).
RELIABILITY_METRICS = {
    "forty": ["split_0_5_s", "split_5_10_s", "split_10_20_s", "split_10yd_s", "top_speed", "peak_accel",
              "burst_1s", "burst_2s", "t90_top_s"],
    "3cone": ["duration_s", "top_speed", "peak_accel", "peak_decel", "plant_speed_ratio", "max_cut_deg"],
    "shuttle": ["duration_s", "top_speed", "peak_accel", "peak_decel", "plant_speed_ratio", "max_cut_deg"],
    "skill": ["top_speed", "peak_accel", "peak_decel", "max_cut_deg", "burst_1s"],
}
FEATURE_SOURCE = {("forty", "split_5_10_s"): "ct_forty_split_5_10_s", ("forty", "top_speed"): "ct_forty_top_speed",
                  ("shuttle", "plant_speed_ratio"): "ct_shuttle_plant_speed_ratio",
                  ("skill", "top_speed"): "ct_skill_speed_z", ("skill", "peak_accel"): "ct_skill_accel_z"}
# Why a metric that passes the reliability gate is not a feature (one feature per construct).
NOT_USED = {("forty", "split_10_20_s"): "same construct as top speed (r = -0.95), slightly less reliable",
            ("forty", "peak_accel"): "same construct as the 5-10 yd split, less reliable",
            ("3cone", "duration_s"): "duplicates the official 3-cone time",
            ("3cone", "top_speed"): "sprint speed, already measured in the 40"}

DEFAULT_RULES = {"frame_rate_hz": 10, "start_speed": 0.5, "min_frames": 15, "speed_cap": 12.5,
                 "accel_cap": 15.0, "plant_drop": 0.5, "cut_window_s": 0.5, "cut_min_speed": 2.0,
                 "min_reliability": 0.5}
DRILL_KEY = ["drill_type", "drill_name"]  # a drill as run by one position group (GAUNTLET_DRILL: WR, TE, DB)


def _rules(cfg: Config) -> dict:
    return {**DEFAULT_RULES, **(cfg.raw.get("combine_tracking_rules") or {})}


def drill_group(drill_type: pd.Series) -> pd.Series:
    """Drill type -> group: forty, 3cone, shuttle, skill (position drills) or other."""
    d = drill_type.astype(str)
    return pd.Series(np.select([d.eq(FORTY), d.eq(THREE_CONE), d.eq(SHUTTLE), d.str.startswith(SKILL_PREFIX)],
                               ["forty", "3cone", "shuttle", "skill"], "other"), index=drill_type.index)


def clean_combine_tracking(ct: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """R16: player frames only, regular 10 Hz spacing, speed / acceleration capped and interpolated."""
    r = _rules(cfg)
    audit = {"ct_frames_in": len(ct), "ct_attempts_in": int(ct["event_id"].nunique())}
    df = ct[(ct["entity_type"].astype(str) == "PLAYER") & ct["nfl_id"].notna()].copy()
    audit["ct_frames_not_player"] = audit["ct_frames_in"] - len(df)
    df["nfl_id"] = df["nfl_id"].astype("int64")
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")
    df = df.sort_values(ATTEMPT_KEY + ["time"]).reset_index(drop=True)
    gaps = df.groupby(ATTEMPT_KEY)["time"].diff().dt.total_seconds()
    bad = df.loc[gaps.notna() & ~np.isclose(gaps, 1.0 / r["frame_rate_hz"]), "event_id"].unique()
    audit["ct_attempts_frame_gaps"] = len(bad)
    df = df[~df["event_id"].isin(bad)]
    for col, cap in (("s", r["speed_cap"]), ("a", r["accel_cap"])):
        over = df[col] > cap
        audit[f"ct_frames_{col}_above_cap"] = int(over.sum())
        df.loc[over, col] = np.nan
        df[col] = df.groupby(ATTEMPT_KEY)[col].transform(lambda s: s.interpolate(limit_direction="both"))
    df["drill_group"] = drill_group(df["drill_type"])
    audit["ct_frames_out"] = len(df)
    return df.reset_index(drop=True), audit


def _time_to_distance(disp: np.ndarray, hz: int, target: float) -> float:
    """First time (s) the displacement reaches `target`, linearly interpolated between frames."""
    idx = np.flatnonzero(disp >= target)
    if not len(idx):
        return np.nan
    i = idx[0]
    if i == 0:
        return 0.0
    frac = (target - disp[i - 1]) / (disp[i] - disp[i - 1]) if disp[i] > disp[i - 1] else 0.0
    return float((i - 1 + frac) / hz)


def _plants(s_smooth: np.ndarray, drop: float) -> tuple[int, float]:
    """Speed minima below `drop` x the previous peak (the turns of the 3-cone / shuttle)."""
    n, ratios, peak = 0, [], 0.0
    for i in range(1, len(s_smooth) - 1):
        peak = max(peak, s_smooth[i])
        if s_smooth[i] <= s_smooth[i - 1] and s_smooth[i] < s_smooth[i + 1] and peak > 0 and s_smooth[i] < drop * peak:
            n += 1
            ratios.append(s_smooth[i] / peak)
            peak = 0.0
    return n, float(np.mean(ratios)) if ratios else np.nan


def attempt_metrics(frames: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """One row per attempt with the movement metrics (same definitions as in games)."""
    r = _rules(cfg)
    hz = int(r["frame_rate_hz"])
    lag = int(r["cut_window_s"] * hz)
    out, no_start, short = [], 0, 0
    for eid, d in frames.groupby("event_id", sort=False):
        s_all = d["s"].to_numpy()
        start = np.flatnonzero(s_all >= r["start_speed"])
        if not len(start):
            no_start += 1
            continue
        d = d.iloc[max(start[0] - 1, 0):]  # the frame before the first moving one = launch point
        if len(d) < r["min_frames"]:
            short += 1
            continue
        x, y, s, a = (d[c].to_numpy() for c in ("x", "y", "s", "a"))
        disp = np.hypot(x - x[0], y - y[0])
        s_sm = pd.Series(s).rolling(3, center=True, min_periods=1).mean().to_numpy()
        dsdt = np.gradient(s_sm, 1.0 / hz)
        moving = s_sm >= 1.0
        top = float(s.max())
        h = np.deg2rad(d["dir"].to_numpy())
        dh = np.abs(np.rad2deg(np.angle(np.exp(1j * (h[lag:] - h[:-lag]))))) if len(h) > lag else np.array([])
        fast = (s[lag:] >= r["cut_min_speed"]) & (s[:-lag] >= r["cut_min_speed"]) if len(h) > lag else np.array([], bool)
        grp = d["drill_group"].iloc[0]
        row = {"event_id": eid, "nfl_id": int(d["nfl_id"].iloc[0]), "draft_year": int(d["draft_year"].iloc[0]),
               "drill_type": d["drill_type"].iloc[0], "drill_name": d["drill_name"].iloc[0], "drill_group": grp,
               "attempt": int(d["attempt"].iloc[0]), "n_frames": len(d), "duration_s": (len(d) - 1) / hz,
               "burst_1s": float(disp[hz]) if len(disp) > hz else np.nan,
               "burst_2s": float(disp[2 * hz]) if len(disp) > 2 * hz else np.nan,
               "top_speed": top,
               "t90_top_s": float(np.flatnonzero(s >= 0.9 * top)[0] / hz) if top > 0 else np.nan,
               "peak_accel": float(np.nanmax(a)),
               "peak_decel": float(-dsdt[moving].min()) if moving.any() else np.nan,
               "max_cut_deg": float(dh[fast].max()) if fast.any() else np.nan,
               "split_10yd_s": np.nan, "split_0_5_s": np.nan, "split_5_10_s": np.nan, "split_10_20_s": np.nan,
               "n_plants": np.nan, "plant_speed_ratio": np.nan}
        if grp == "forty":
            t5, t10, t20 = (_time_to_distance(disp, hz, m) for m in (5.0, 10.0, 20.0))
            row.update({"split_10yd_s": t10, "split_0_5_s": t5, "split_5_10_s": t10 - t5, "split_10_20_s": t20 - t10})
        if grp in ("3cone", "shuttle"):
            row["n_plants"], row["plant_speed_ratio"] = _plants(s_sm, r["plant_drop"])
        out.append(row)
    audit = {"ct_attempts_no_start": no_start, "ct_attempts_too_short": short, "ct_attempts_kept": len(out)}
    return pd.DataFrame(out), audit


def _skill_z(attempts: pd.DataFrame, metrics) -> pd.DataFrame:
    """Position-drill attempts with each metric z-scored within its drill (position group x name).

    Drills differ by design, and the same drill name can be run by several position groups (the
    gauntlet is run by WR, TE and DB), so the reference group is the drill as run by one position group.
    """
    sk = attempts[attempts.drill_group == "skill"].copy()
    for m in metrics:
        g = sk.groupby(DRILL_KEY)[m]
        sk[f"{m}_z"] = (sk[m] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return sk


def attempt_reliability(attempts: pd.DataFrame, min_reliability: float = 0.5) -> pd.DataFrame:
    """How well each metric repeats within a player (label-free; decides which metrics are features).

    40 / 3-cone / shuttle: Pearson r between a player's first two attempts (test-retest).
    Position drills: split-half r of the mean within-drill z (alternate attempts), Spearman-Brown
    corrected to the full set of attempts, which is what the feature averages.
    """
    rows = []
    for grp in ("forty", "3cone", "shuttle"):
        a = attempts[attempts.drill_group == grp].sort_values(["nfl_id", "attempt"]).copy()
        a["k"] = a.groupby("nfl_id").cumcount()
        for m in RELIABILITY_METRICS[grp]:
            if a.empty or m not in a.columns:
                rows.append({"drill_group": grp, "metric": m, "method": "test-retest (attempts 1 vs 2)",
                             "n_players": 0, "reliability": np.nan})
                continue
            w = a[a.k < 2].pivot(index="nfl_id", columns="k", values=m).dropna()
            r = float(stats.pearsonr(w[0], w[1])[0]) if len(w) >= 10 and w.shape[1] == 2 else np.nan
            rows.append({"drill_group": grp, "metric": m, "method": "test-retest (attempts 1 vs 2)",
                         "n_players": len(w), "reliability": r})
    skill_metrics = [m for m in RELIABILITY_METRICS["skill"] if m in attempts.columns]
    sk = _skill_z(attempts, skill_metrics).sort_values(["nfl_id", *DRILL_KEY, "attempt"])
    sk["half"] = sk.groupby("nfl_id").cumcount() % 2
    for m in RELIABILITY_METRICS["skill"]:
        if sk.empty or m not in skill_metrics:
            rows.append({"drill_group": "skill", "metric": m, "method": "split-half, Spearman-Brown",
                         "n_players": 0, "reliability": np.nan})
            continue
        h = sk.groupby(["nfl_id", "half"])[f"{m}_z"].mean().unstack().dropna()
        r = float(stats.pearsonr(h[0], h[1])[0]) if len(h) >= 10 and h.shape[1] == 2 else np.nan
        rows.append({"drill_group": "skill", "metric": m, "method": "split-half, Spearman-Brown",
                     "n_players": len(h), "reliability": 2 * r / (1 + r) if np.isfinite(r) else np.nan})
    out = pd.DataFrame(rows)
    out["feature"] = [FEATURE_SOURCE.get((g, m), "") for g, m in zip(out.drill_group, out.metric)]
    out["passes_gate"] = out["reliability"] >= min_reliability
    out["note"] = [NOT_USED.get((g, m), "") if p and not f else ("" if p else "below the reliability gate")
                   for g, m, p, f in zip(out.drill_group, out.metric, out.passes_gate, out.feature)]
    return out


def player_features(attempts: pd.DataFrame, nfl_ids: pd.Series | None = None) -> pd.DataFrame:
    """Best attempt per metric (`ct_<group>_<metric>`) plus the mean within-drill z for position drills."""
    ids = pd.Index(sorted(attempts["nfl_id"].unique()) if nfl_ids is None else nfl_ids, name="nfl_id")
    feats = pd.DataFrame(index=ids)
    for grp, metrics in _BEST.items():
        sub = attempts[attempts.drill_group == grp]
        for m, how in metrics.items():
            feats[f"ct_{grp}_{m}"] = sub.groupby("nfl_id")[m].agg(how).reindex(ids)
    sk = _skill_z(attempts, list(_SKILL_Z))
    for m, name in _SKILL_Z.items():
        feats[f"ct_skill_{name}_z"] = sk.groupby("nfl_id")[f"{m}_z"].mean().reindex(ids)
    feats["ct_attempts"] = attempts.groupby("nfl_id").size().reindex(ids).fillna(0).astype(int)
    feats["ct_missing"] = feats[CT_FEATURES].isna().all(axis=1).astype(int)
    return feats[CT_FEATURES + ["ct_attempts", "ct_missing"]].reset_index()


def official_vs_tracking(features: pd.DataFrame, prospects: pd.DataFrame, group: str = "combine_position") -> pd.DataFrame:
    """Convergent validity: does tracking reproduce the official stopwatch?

    `spearman` / `pearson` pool every prospect; `within_position_spearman` first removes the
    position-group means (like every translation table), which is the comparison that matters for a
    within-position model.
    """
    pairs = [("ct_forty_split_5_10_s", "ten_yd_split"), ("ct_forty_split_5_10_s", "forty"),
             ("ct_forty_top_speed", "forty"), ("ct_forty_top_speed", "ten_yd_split"),
             ("ct_shuttle_plant_speed_ratio", "short_shuttle"), ("ct_skill_speed_z", "forty"),
             ("ct_skill_accel_z", "ten_yd_split")]
    official = prospects[["nfl_id"] + [c for c in prospects.columns if c not in features.columns]]
    df = features.merge(official, on="nfl_id", how="inner")
    rows = []
    for ct_col, off in pairs:
        if ct_col not in df or off not in df:
            continue
        t = df[[ct_col, off, group]].dropna()
        if len(t) < 10:
            continue
        rho, p = stats.spearmanr(t[ct_col], t[off])
        r, _ = stats.pearsonr(t[ct_col], t[off])
        g = t.groupby(t[group].astype(str))
        w = t[[ct_col, off]] - g[[ct_col, off]].transform("mean")
        rho_w, p_w = stats.spearmanr(w[ct_col], w[off])
        rows.append({"tracking_feature": ct_col, "official_result": off, "n": len(t), "spearman": rho,
                     "pearson": r, "p_value": p, "within_position_spearman": rho_w, "within_position_p": p_w})
    return pd.DataFrame(rows)


def build_combine_tracking(ct_raw: pd.DataFrame, prospects: pd.DataFrame, cfg: Config
                           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Raw Combine tracking -> (attempt table, features for every prospect, reliability, audit)."""
    frames, a1 = clean_combine_tracking(ct_raw, cfg)
    attempts, a2 = attempt_metrics(frames, cfg)
    if len(attempts):
        feats = player_features(attempts, prospects["nfl_id"])
        rel = attempt_reliability(attempts, _rules(cfg)["min_reliability"])
    else:
        feats = pd.DataFrame({"nfl_id": prospects["nfl_id"], **{c: np.nan for c in CT_FEATURES},
                              "ct_attempts": 0, "ct_missing": 1})
        rel = pd.DataFrame()
    audit = {**a1, **a2, "ct_players_with_features": int((feats["ct_missing"] == 0).sum()),
             **{f"ct_players_with_{c.removeprefix('ct_')}": int(feats[c].notna().sum()) for c in CT_FEATURES}}
    return attempts, feats, rel, audit


# =============================================================================
# Feature sets and modelling tables
# =============================================================================
# Feature sets are nested so that each step answers one question:
#   combine             what a scout has before the draft (athletic testing + profile)
#   combine_draft       + draft capital (what the league decided)
#   combine_draft_mov   + game movement (post-draft tracking; diagnostic, not pre-draft)
#   combine_draft_proj  + on-field performance projected from the Combine. Observed values enter only
#                         as training labels of a stacked projector inside the pipeline
#                         (train.OnFieldProjector); at prediction time only Combine data is used, so
#                         the set is usable on draft day.
#   combine_draft_perf  + observed on-field performance (post-draft; diagnostic)
#   combine_draft_ct    + Combine drill-tracking features (pre-draft)
#   ATHLETIC_CT         athletic regression features + drill tracking (with vs without tracking)
DERIVED = ["speed_score", "ten_split_share", "ten_split_momentum"]
MISSING_FLAGS = [f"{d}_missing" for d in DRILLS]
CATEGORICAL = ["combine_position"]

# Regression: pure athletic translation (no draft or production information).
ATHLETIC = MEASURABLES + DRILLS + DERIVED + MISSING_FLAGS + ["n_drills_run", "ngs_athleticism_score", "age_at_draft"]

FEATURE_SETS: dict[str, list[str]] = {
    "combine": ATHLETIC + ["ngs_college_production_score", "power_conference"],
}
FEATURE_SETS["combine_draft"] = FEATURE_SETS["combine"] + ["log_draft_pick", "is_undrafted"]
MOVEMENT_FEATURES = ["sbi"] + [f"{m}_z" for m in METRICS if m != "burst_2s"]
FEATURE_SETS["combine_draft_mov"] = FEATURE_SETS["combine_draft"] + MOVEMENT_FEATURES

# On-field performance sets. paoe is left out of both: it is the binary twin of ppoe (same family).
PROJECTED = ["goe", "proe", "ppoe", "soe"]
PROJECTOR_TARGETS = [f"onfield_target_{m}" for m in PROJECTED]
PERF_FEATURES = [f"{m}_z" for m in PROJECTED]
FEATURE_SETS["combine_draft_proj"] = FEATURE_SETS["combine_draft"] + PROJECTOR_TARGETS
FEATURE_SETS["combine_draft_perf"] = FEATURE_SETS["combine_draft"] + PERF_FEATURES

# Combine drill tracking (used only when the file is present).
CT_SET = list(CT_FEATURES)
ATHLETIC_CT = ATHLETIC + CT_SET
FEATURE_SETS["combine_draft_ct"] = FEATURE_SETS["combine_draft"] + CT_SET


def has_combine_tracking(prospects: pd.DataFrame) -> bool:
    """True when the prospect table has drill-tracking features for at least one player."""
    return all(c in prospects.columns for c in CT_SET + ["ct_missing"]) and bool((prospects["ct_missing"] == 0).any())


def regression_table(prospects: pd.DataFrame, profiles: pd.DataFrame, target: str) -> pd.DataFrame:
    """Prospects with a movement profile and the regression target (SBI)."""
    keep = list(dict.fromkeys(["nfl_id", target, "sbi", "burst_2s_eb", "sbi_reliability", "n_plays"]))
    return profiles[keep].merge(prospects, on="nfl_id", how="inner", validate="1:1")


def classification_table(prospects: pd.DataFrame, profiles: pd.DataFrame | None = None) -> pd.DataFrame:
    """Starter-classifier table; with `profiles` it adds the movement features."""
    if profiles is None:
        return prospects.copy()
    cols = ["nfl_id", "n_plays", "burst_2s_eb"] + MOVEMENT_FEATURES
    return profiles[cols].merge(prospects, on="nfl_id", how="inner", validate="1:1")


def onfield_regression_table(prospects: pd.DataFrame, onfield: pd.DataFrame, target: str,
                             positions: list[str]) -> pd.DataFrame:
    """Players of the given Combine positions with a valid `target` value."""
    metric = target.split("_")[0]
    cols = ["nfl_id", target, f"{metric}_rel", f"n_{metric}"]
    df = onfield[cols].dropna(subset=[target]).merge(prospects, on="nfl_id", how="inner", validate="1:1")
    return df[df.combine_position.astype(str).isin(positions)]


def projector_table(prospects: pd.DataFrame, onfield: pd.DataFrame) -> pd.DataFrame:
    """Every prospect + his shrunk on-field values as projector labels (NaN where not measured)."""
    lab = onfield[["nfl_id"] + [f"{m}_eb" for m in PROJECTED]].rename(
        columns={f"{m}_eb": f"onfield_target_{m}" for m in PROJECTED})
    return prospects.merge(lab, on="nfl_id", how="left", validate="1:1")


def onfield_classification_table(prospects: pd.DataFrame, onfield: pd.DataFrame) -> pd.DataFrame:
    """Players with an on-field index; metrics outside the player's family are 0 (= the family mean)."""
    df = onfield[["nfl_id", "perf_index"] + PERF_FEATURES].dropna(subset=["perf_index"])
    df = df.merge(prospects, on="nfl_id", how="inner", validate="1:1")
    df[PERF_FEATURES] = df[PERF_FEATURES].fillna(0.0)
    return df


def xy(df: pd.DataFrame, feature_set: list[str], target: str) -> tuple[pd.DataFrame, pd.Series]:
    """X matrix (position + the set's features) and target y."""
    cols = list(dict.fromkeys(CATEGORICAL + feature_set))
    return df[cols].copy(), df[target].copy()


def _role(nfl_position: pd.Series) -> pd.Series:
    """Approximate NFL role; only balances the folds and the role controls (never a feature)."""
    return nfl_position.astype(str).replace({"DE": "edge", "OLB": "edge", "DT": "interior", "NT": "interior"})


def _strata(roles: pd.Series, min_size: int) -> pd.Series:
    """Cross-validation strata: roles with fewer than `min_size` players are pooled into "other"."""
    counts = roles.value_counts()
    return roles.where(roles.map(counts) >= min_size, "other")


# =============================================================================
# Preparation stages
# =============================================================================
def stage_contract(cfg: Config) -> dict:
    """Validates the raw files against configs/data_contract.yaml (the specification)."""
    ccfg = cfg.raw.get("contract", {})
    contract = load_contract(ccfg.get("path"))
    report, summary = validate_all(cfg, contract)
    write_report(cfg, report, summary)
    log.info("data contract: %s (%d checks, %d errors, %d warnings, missing %s, formats %s)", summary["status"],
             summary["n_checks"], summary["n_errors"], summary["n_warnings"],
             [m["spec_name"] for m in summary["missing_files"]], summary["formats"])
    if summary["n_errors"] and ccfg.get("fail_on_error", True):
        bad = report[(report.severity == "error") & (report.result == "fail")]
        raise RuntimeError("data contract errors:\n" + bad[["table", "file", "column", "check", "detail"]].to_string())
    return summary


def _process_tracking(cfg: Config, games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cleans tracking and computes per-play metrics one file at a time (full seasons).

    R8b keeps the season types in `tracking_rules.season_types`. R8c: a file with exactly 1,048,575
    rows was cut at a spreadsheet's row limit, so its last player-play is dropped. A play whose
    frames are split across file parts is set aside and processed at the end. Clean frames are
    appended to data/clean/interim/tracking_frames.parquet. The output order is the play-key order,
    identical to processing the concatenated files.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rules = cfg["tracking_rules"]
    keep_games = set(games.loc[games.season_type.isin(rules.get("season_types", ["REG", "POST"])), "game_id"])
    key = ["game_id", "play_id", "nfl_id"]
    paths = table_files(cfg, "tracking")
    keysets = [read_raw_table(p, key).drop_duplicates() for p in paths]
    spanning = (pd.concat(keysets, ignore_index=True).value_counts().reset_index(name="n").query("n > 1")[key]
                if len(keysets) > 1 else pd.DataFrame(columns=key))
    audit: dict = {"tracking_files": len(paths), "frames_excluded_season_type": 0, "plays_split_across_files": len(spanning)}
    out_path = cfg.path("interim") / "tracking_frames.parquet"
    writer, plays, held = None, [], []

    def run(frames_raw: pd.DataFrame) -> None:
        nonlocal writer
        frames, fa = clean_tracking(frames_raw, cfg)
        pm, pa_ = play_metrics(frames, cfg)
        for k, v in {**fa, **pa_}.items():
            audit[k] = audit.get(k, 0) + v
        plays.append(pm)
        table = pa.Table.from_pandas(frames, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table.cast(writer.schema))

    audit["plays_dropped_cut_file"] = 0
    for name, df in iter_tracking(cfg):
        if len(df) == SPREADSHEET_ROW_LIMIT:  # R8c: file cut at the limit; its last play may be incomplete
            last = df.iloc[-1][key]
            cut = (df[key] == last.values).all(axis=1)
            audit["plays_dropped_cut_file"] += 1
            log.warning("%s has %d rows (a spreadsheet's limit): dropping its last player-play %s",
                        name, len(df), tuple(last))
            df = df[~cut]
        season_ok = df.game_id.isin(keep_games)
        audit["frames_excluded_season_type"] += int((~season_ok).sum())
        df = df[season_ok]
        if len(spanning):
            m = df.merge(spanning.assign(_span=1), on=key, how="left")["_span"].notna().to_numpy()
            held.append(df[m])
            df = df[~m]
        log.info("tracking %s: %d frames", name, len(df))
        run(df.reset_index(drop=True))
    if held:
        run(pd.concat(held, ignore_index=True))
    if writer is not None:
        writer.close()
    plays_df = pd.concat(plays, ignore_index=True) if plays else pd.DataFrame(columns=key)
    if plays_df.duplicated(key).any():
        raise RuntimeError("tracking: a play was processed twice across files")
    return plays_df.sort_values(key, kind="mergesort").reset_index(drop=True), audit


def stage_data(cfg: Config) -> dict:
    """Contract + ingest + cleaning + movement + on-field performance + manifest."""
    t0 = time.time()
    contract_summary = stage_contract(cfg)
    tables = load_raw(cfg, tracking=False)
    manifest = write_manifest(cfg)
    log.info("ingest OK: data_version=%s", manifest["data_version"])

    prospects, audit = build_prospects(tables, cfg)
    ct_audit = {}
    if "combine_tracking" in tables:  # spec File 3 (optional)
        attempts, ct_feats, ct_rel, ct_audit = build_combine_tracking(tables["combine_tracking"], prospects, cfg)
        _save(attempts, cfg, "combine_tracking_attempts.parquet")
        _save(ct_feats, cfg, "combine_tracking_features.parquet")
        _save(ct_rel, cfg, "ct_reliability.csv", key="metrics")
        prospects = prospects.merge(ct_feats, on="nfl_id", how="left", validate="1:1")
        log.info("Combine tracking: %s", ct_audit)
    _save(prospects, cfg, "prospects.parquet")
    _save(audit, cfg, "cleaning_audit.csv", key="metrics")
    miss = missingness_by_position(prospects)
    miss.to_csv(cfg.path("metrics") / "drill_missingness_by_position.csv")

    plays, tracking_audit = _process_tracking(cfg, tables["games"])
    plays = plays.merge(prospects[["nfl_id", "combine_position"]], on="nfl_id", how="left")
    _save(plays, cfg, "play_metrics.parquet")

    profiles = player_profiles(plays, prospects, cfg)
    _save(profiles, cfg, "player_movement_profiles.parquet")

    # Anomaly scan of the per-play metrics: robust z (median/MAD) within the position group.
    flags = []
    for m in ["burst_2s", "top_speed", "max_decel", "peak_accel_2s"]:
        g = plays.groupby("combine_position", observed=True)[m]
        med = g.transform("median")
        mad = g.transform(lambda s: (s - s.median()).abs().median()) * 1.4826
        rz = (plays[m] - med) / mad
        flags.append(pd.DataFrame({"game_id": plays.game_id, "play_id": plays.play_id, "nfl_id": plays.nfl_id,
                                   "metric": m, "value": plays[m], "robust_z": rz})[rz.abs() > 3.5])
    anomalies = pd.concat(flags, ignore_index=True)
    _save(anomalies, cfg, "play_metric_anomalies.csv", key="metrics")

    # 3b on-field performance
    onf_audit, onfield, onfield_rookie, fam = {}, None, None, None
    if "player_play" in tables:
        pp_plays, context, onf_audit = clean_player_play(tables["player_play"], tables["games"], prospects, cfg)
        fam = family_plays(pp_plays, cfg)
        for k, v in fam.items():
            _save(v, cfg, f"onfield_plays_{k}.parquet")
        context.to_parquet(cfg.path("interim") / "play_context.parquet", index=False)
        onfield = player_onfield(fam, prospects, cfg)
        onfield_rookie = player_onfield(fam, prospects, cfg, rookie_only=True)
        _save(onfield, cfg, "player_onfield_profiles.parquet")
        _save(onfield_rookie, cfg, "player_onfield_rookie.parquet")
        onf_audit.update({f"onfield_plays_{k}": len(v) for k, v in fam.items()})
        onf_audit.update({f"players_with_{m}": int(onfield[f"{m}_eb"].notna().sum()) for m in ONFIELD_METRICS})
        onf_audit["players_with_perf_index"] = int(onfield.perf_index.notna().sum())

    audit_all = {**tracking_audit, **ct_audit, "cleaning_rule_changes": len(audit),
                 "players_tracked": int(plays.nfl_id.nunique()), "players_profiled": len(profiles),
                 "play_metric_anomalies": len(anomalies), **onf_audit,
                 "contract_status": contract_summary["status"], "contract_warnings": contract_summary["n_warnings"],
                 "seconds": round(time.time() - t0, 1)}
    _dump(audit_all, cfg, "data_audit.json")
    log.info("data stage: %s", audit_all)
    return {"prospects": prospects, "plays": plays, "profiles": profiles, "frames": None,
            "onfield": onfield, "onfield_rookie": onfield_rookie, "fam": fam}


def load_processed(cfg: Config) -> dict:
    """Loads the clean data from data/clean (without repeating the preparation)."""
    p = cfg.path("processed")
    d = {
        "prospects": pd.read_parquet(p / "prospects.parquet"),
        "plays": pd.read_parquet(p / "play_metrics.parquet"),
        "profiles": pd.read_parquet(p / "player_movement_profiles.parquet"),
        "onfield": None, "onfield_rookie": None, "fam": None,
    }
    if (p / "player_onfield_profiles.parquet").exists():
        d["onfield"] = pd.read_parquet(p / "player_onfield_profiles.parquet")
        d["onfield_rookie"] = pd.read_parquet(p / "player_onfield_rookie.parquet")
        d["fam"] = {k: pd.read_parquet(p / f"onfield_plays_{k}.parquet") for k in ("DL", "OL", "WRTE")}
    return d


# =============================================================================
# Command line
# =============================================================================
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="NFL project data preparation (contract, ingest and cleaning)")
    ap.add_argument("--config", default=None, help="configuration file (configs/config.yaml by default)")
    ap.add_argument("--stage", default="data", choices=["data", "contract", "verify"],
                    help="data = contract + ingest + cleaning; contract = validation only; verify = manifest hashes")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    if args.stage == "contract":
        stage_contract(cfg)
    elif args.stage == "verify":
        bad = verify_manifest(cfg)
        if bad:
            raise SystemExit(f"hash differs from the manifest: {bad}")
        log.info("manifest verified: every raw file matches")
    else:
        stage_data(cfg)


if __name__ == "__main__":
    main()
