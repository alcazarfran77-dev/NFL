"""Feature sets and modeling tables.

Feature sets are nested so that each step answers one question:
  combine           what a scout has before the draft (athletic testing + profile)
  combine_draft     + draft capital (what the league decided)
  combine_draft_mov + in-game movement (post-draft tracking; diagnostic, not pre-draft)
"""
from __future__ import annotations

import pandas as pd

from .cleaning import DRILLS, MEASURABLES
from .movement import METRICS

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


def regression_table(prospects: pd.DataFrame, profiles: pd.DataFrame, target: str) -> pd.DataFrame:
    keep = list(dict.fromkeys(["nfl_id", target, "sbi", "burst_2s_eb", "sbi_reliability", "n_plays"]))
    return profiles[keep].merge(prospects, on="nfl_id", how="inner", validate="1:1")


def classification_table(prospects: pd.DataFrame, profiles: pd.DataFrame | None = None) -> pd.DataFrame:
    if profiles is None:
        return prospects.copy()
    cols = ["nfl_id", "n_plays", "burst_2s_eb"] + MOVEMENT_FEATURES
    return profiles[cols].merge(prospects, on="nfl_id", how="inner", validate="1:1")


def xy(df: pd.DataFrame, feature_set: list[str], target: str) -> tuple[pd.DataFrame, pd.Series]:
    cols = list(dict.fromkeys(CATEGORICAL + feature_set))
    return df[cols].copy(), df[target].copy()
