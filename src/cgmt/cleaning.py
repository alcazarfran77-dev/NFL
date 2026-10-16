"""Cleaning and business rules for the prospect-level tables.

Rules applied (all thresholds live in configs/config.yaml):
  R1  Drill values outside physically plausible ranges -> NaN (measurement error).
  R2  Missing drills are NOT imputed here; a `<drill>_missing` flag is kept, and
      imputation (position-group median) happens inside each CV training fold.
  R3  Undrafted players get draft_overall_pick = 260 and is_undrafted = 1.
  R4  Seasons available = last completed season - draft year + 1, so careers of
      different lengths are compared on a per-season basis.
  R5  start_rate = games started / (17 x seasons available); starter = start_rate >= 0.5.
  R6  Age at draft is computed against an approximate draft date (Apr 27).
  R7  Conference collapsed to a binary power-conference flag (rare levels pooled).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

DRILLS = ["ten_yd_split", "forty", "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps"]
MEASURABLES = ["combine_height", "combine_weight", "hand_size", "arm_length", "wing_span"]
NGS_SCORES = ["ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score"]
# Drills where a LOWER value is better (times); used to orient correlations.
LOWER_IS_BETTER = {"ten_yd_split", "forty", "three_cone", "short_shuttle"}
POSITION_GROUPS = ["DB", "DL", "OL", "TE", "WR"]


def apply_range_rules(df: pd.DataFrame, ranges: dict[str, list[float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """R1: blank out implausible values. Returns (clean_df, audit log of changes)."""
    df = df.copy()
    log = []
    for col, (lo, hi) in ranges.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        for idx in df.index[bad]:
            log.append({"nfl_id": df.at[idx, "nfl_id"], "column": col, "value": df.at[idx, col], "rule": f"outside [{lo}, {hi}]"})
        df.loc[bad, col] = np.nan
    return df, pd.DataFrame(log, columns=["nfl_id", "column", "value", "rule"])


def build_prospects(tables: dict[str, pd.DataFrame], cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join combine, player and career tables into one clean prospect table."""
    br = cfg["business_rules"]
    combine, audit = apply_range_rules(tables["combine"], br["drill_valid_ranges"])
    players = tables["players"].copy()
    careers = tables["careers"].copy()

    df = combine.merge(players, on=["nfl_id", "draft_year"], how="inner", validate="1:1")
    df = df.merge(careers, on="nfl_id", how="inner", validate="1:1")
    if len(df) != len(combine):
        raise ValueError("draft_year disagrees between combine and players tables")

    # Types
    df["combine_position"] = pd.Categorical(df["combine_position"].astype(str), categories=POSITION_GROUPS)
    df["nfl_position"] = df["nfl_position"].astype("category")
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")

    # R2: missingness flags (informative: prospects choose which drills to run)
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
    # Speed Score (Barnwell): weight-adjusted 40 time.
    df["speed_score"] = df["combine_weight"] * 200 / df["forty"] ** 4
    # Acceleration share: share of the 40 spent in the first 10 yards (lower = quicker start).
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
    """Share of prospects missing each drill, by combine position group."""
    return df.groupby("combine_position", observed=True)[DRILLS].apply(lambda g: g.isna().mean()).round(3)
