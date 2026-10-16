"""Feature sets and modeling tables.

Feature sets are nested so that each step answers one question:
  combine           what a scout has before the draft (athletic testing + profile)
  combine_draft     + draft capital (what the league decided)
  combine_draft_mov + in-game movement (post-draft tracking; diagnostic, not pre-draft)

v2.0.0 adds two sets built on player_play:
  combine_draft_proj  + Combine-projected on-field performance. The raw on-field targets enter
                        only as training labels of a stacked projector inside the pipeline
                        (models.OnFieldProjector); at prediction time only Combine data is used,
                        so the set is usable on draft day.
  combine_draft_perf  + observed on-field performance (post-draft; diagnostic, like the movement set)
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

# v2.0.0 on-field sets. paoe is left out of both: it is the binary twin of ppoe (same family).
PROJECTED = ["goe", "proe", "ppoe", "soe"]
PROJECTOR_TARGETS = [f"onfield_target_{m}" for m in PROJECTED]
PERF_FEATURES = [f"{m}_z" for m in PROJECTED]
FEATURE_SETS["combine_draft_proj"] = FEATURE_SETS["combine_draft"] + PROJECTOR_TARGETS
FEATURE_SETS["combine_draft_perf"] = FEATURE_SETS["combine_draft"] + PERF_FEATURES


def regression_table(prospects: pd.DataFrame, profiles: pd.DataFrame, target: str) -> pd.DataFrame:
    keep = list(dict.fromkeys(["nfl_id", target, "sbi", "burst_2s_eb", "sbi_reliability", "n_plays"]))
    return profiles[keep].merge(prospects, on="nfl_id", how="inner", validate="1:1")


def classification_table(prospects: pd.DataFrame, profiles: pd.DataFrame | None = None) -> pd.DataFrame:
    if profiles is None:
        return prospects.copy()
    cols = ["nfl_id", "n_plays", "burst_2s_eb"] + MOVEMENT_FEATURES
    return profiles[cols].merge(prospects, on="nfl_id", how="inner", validate="1:1")


def onfield_regression_table(prospects: pd.DataFrame, onfield: pd.DataFrame, target: str,
                             positions: list[str]) -> pd.DataFrame:
    """Players of the given Combine positions with a qualifying value of `target`."""
    metric = target.split("_")[0]
    cols = ["nfl_id", target, f"{metric}_rel", f"n_{metric}"]
    df = onfield[cols].dropna(subset=[target]).merge(prospects, on="nfl_id", how="inner", validate="1:1")
    return df[df.combine_position.astype(str).isin(positions)]


def projector_table(prospects: pd.DataFrame, onfield: pd.DataFrame) -> pd.DataFrame:
    """All prospects + their shrunk on-field values as projector labels (NaN where not measured)."""
    lab = onfield[["nfl_id"] + [f"{m}_eb" for m in PROJECTED]].rename(
        columns={f"{m}_eb": f"onfield_target_{m}" for m in PROJECTED})
    return prospects.merge(lab, on="nfl_id", how="left", validate="1:1")


def onfield_classification_table(prospects: pd.DataFrame, onfield: pd.DataFrame) -> pd.DataFrame:
    """Players with an on-field index; metrics outside a player's family are 0 (= family average)."""
    df = onfield[["nfl_id", "perf_index"] + PERF_FEATURES].dropna(subset=["perf_index"])
    df = df.merge(prospects, on="nfl_id", how="inner", validate="1:1")
    df[PERF_FEATURES] = df[PERF_FEATURES].fillna(0.0)
    return df


def xy(df: pd.DataFrame, feature_set: list[str], target: str) -> tuple[pd.DataFrame, pd.Series]:
    cols = list(dict.fromkeys(CATEGORICAL + feature_set))
    return df[cols].copy(), df[target].copy()
