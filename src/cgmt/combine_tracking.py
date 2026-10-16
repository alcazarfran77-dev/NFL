"""Combine drill tracking (spec File 3, combine_tracking.csv) -> per-prospect movement features (v2.1.0).

The official Combine results are stopwatch times and jump distances. The spec's drill tracking
(10 Hz, 6,310 attempts) shows how each time was produced. This module turns it into features that
use the same definitions as the in-game metrics in `cgmt.movement`, so the Combine and the game can be
compared on the same scale (for example 2-second burst at the Combine vs 2-second burst after the snap).

The module is optional: the pipeline uses it only when a file matching `raw_files.combine_tracking`
is present, and every v2.0.0 output is unchanged without it.

Per attempt (PLAYER rows; the attempt starts at the first frame with speed >= `start_speed`)
  burst_1s / burst_2s  straight-line distance from the start spot at t = 1.0 / 2.0 s (yd)
  top_speed            highest speed (yd/s)
  t90_top_s            time from the start to 90% of top speed (s)
  peak_accel           highest acceleration (yd/s^2)
  peak_decel           highest braking rate on a 3-frame smoothed speed trace while moving >= 1 yd/s
  max_cut_deg          largest change of direction within 0.5 s at >= 2 yd/s at both ends
  split_10yd_s / split_20yd_s   (40-yard dash only) time to cover 10 / 20 yd from the start, interpolated
  n_plants / plant_speed_ratio  (3-cone, shuttle) speed minima below 50% of the preceding peak, and the
                       mean ratio of plant speed to that peak (higher = keeps more speed through the turn)

Player features (best attempt, as the official Combine takes the best result)
  ct_forty_*    burst_1s, burst_2s, split_10yd_s, top_speed, t90_top_s, peak_accel
  ct_3cone_*    peak_decel, plant_speed_ratio
  ct_shuttle_*  peak_decel, plant_speed_ratio, peak_accel
  ct_skill_*_z  position drills differ by name, so each attempt is z-scored within its drill_name and the
                player value is his mean z across attempts (accel, decel, top speed)

Business rules (configs/config.yaml -> combine_tracking_rules)
  R16 ball rows and rows without nfl_id dropped; frames must be 0.1 s apart within an attempt;
      speed / acceleration above the in-game caps -> NaN and interpolated; attempts shorter than
      `min_frames` or that never reach `start_speed` dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

ATTEMPT_KEY = ["event_id"]
FORTY, THREE_CONE, SHUTTLE = "FORTY_YARD_DASH", "THREE_CONE_DRILL", "SHORT_SHUTTLE"
SKILL_PREFIX = "SKILL_DRILLS_"

CT_FEATURES = [
    "ct_forty_burst_1s", "ct_forty_burst_2s", "ct_forty_split_10yd_s", "ct_forty_top_speed",
    "ct_forty_t90_top_s", "ct_forty_peak_accel",
    "ct_3cone_peak_decel", "ct_3cone_plant_speed_ratio",
    "ct_shuttle_peak_decel", "ct_shuttle_plant_speed_ratio", "ct_shuttle_peak_accel",
    "ct_skill_accel_z", "ct_skill_decel_z", "ct_skill_speed_z",
]
# Lower is better for these (times); everything else: higher is better.
CT_LOWER_IS_BETTER = {"ct_forty_split_10yd_s", "ct_forty_t90_top_s"}

_BEST = {  # drill group -> {attempt metric: "max" | "min"}
    "forty": {"burst_1s": "max", "burst_2s": "max", "split_10yd_s": "min", "top_speed": "max",
              "t90_top_s": "min", "peak_accel": "max"},
    "3cone": {"peak_decel": "max", "plant_speed_ratio": "max"},
    "shuttle": {"peak_decel": "max", "plant_speed_ratio": "max", "peak_accel": "max"},
}

DEFAULT_RULES = {"frame_rate_hz": 10, "start_speed": 0.5, "min_frames": 15, "speed_cap": 12.5,
                 "accel_cap": 15.0, "plant_drop": 0.5, "cut_window_s": 0.5, "cut_min_speed": 2.0}


def _rules(cfg: Config) -> dict:
    return {**DEFAULT_RULES, **(cfg.raw.get("combine_tracking_rules") or {})}


def drill_group(drill_type: pd.Series) -> pd.Series:
    d = drill_type.astype(str)
    return pd.Series(np.select([d.eq(FORTY), d.eq(THREE_CONE), d.eq(SHUTTLE), d.str.startswith(SKILL_PREFIX)],
                               ["forty", "3cone", "shuttle", "skill"], "other"), index=drill_type.index)


def clean_combine_tracking(ct: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """R16: player frames only, regular 10 Hz spacing, capped and interpolated speed / acceleration."""
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
    """First time (s) displacement reaches `target`, linearly interpolated between frames."""
    idx = np.flatnonzero(disp >= target)
    if not len(idx):
        return np.nan
    i = idx[0]
    if i == 0:
        return 0.0
    frac = (target - disp[i - 1]) / (disp[i] - disp[i - 1]) if disp[i] > disp[i - 1] else 0.0
    return float((i - 1 + frac) / hz)


def _plants(s_smooth: np.ndarray, drop: float) -> tuple[int, float]:
    """Speed minima below `drop` x the preceding peak (the turns of the 3-cone / shuttle)."""
    n, ratios, peak = 0, [], 0.0
    for i in range(1, len(s_smooth) - 1):
        peak = max(peak, s_smooth[i])
        if s_smooth[i] <= s_smooth[i - 1] and s_smooth[i] < s_smooth[i + 1] and peak > 0 and s_smooth[i] < drop * peak:
            n += 1
            ratios.append(s_smooth[i] / peak)
            peak = 0.0
    return n, float(np.mean(ratios)) if ratios else np.nan


def attempt_metrics(frames: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """One row per drill attempt with the movement metrics (same definitions as in-game)."""
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
        d = d.iloc[max(start[0] - 1, 0):]  # the frame before the first moving frame = start spot
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
               "split_10yd_s": np.nan, "split_20yd_s": np.nan, "n_plants": np.nan, "plant_speed_ratio": np.nan}
        if grp == "forty":
            row["split_10yd_s"] = _time_to_distance(disp, hz, 10.0)
            row["split_20yd_s"] = _time_to_distance(disp, hz, 20.0)
        if grp in ("3cone", "shuttle"):
            row["n_plants"], row["plant_speed_ratio"] = _plants(s_sm, r["plant_drop"])
        out.append(row)
    audit = {"ct_attempts_no_start": no_start, "ct_attempts_too_short": short, "ct_attempts_kept": len(out)}
    return pd.DataFrame(out), audit


def player_features(attempts: pd.DataFrame, nfl_ids: pd.Series | None = None) -> pd.DataFrame:
    """Best attempt per drill group (`ct_<group>_<metric>`) plus within-drill z for skill drills."""
    ids = pd.Index(sorted(attempts["nfl_id"].unique()) if nfl_ids is None else nfl_ids, name="nfl_id")
    feats = pd.DataFrame(index=ids)
    for grp, metrics in _BEST.items():
        sub = attempts[attempts.drill_group == grp]
        for m, how in metrics.items():
            feats[f"ct_{grp}_{m}"] = sub.groupby("nfl_id")[m].agg(how).reindex(ids)
    sk = attempts[attempts.drill_group == "skill"].copy()
    for m, name in (("peak_accel", "accel"), ("peak_decel", "decel"), ("top_speed", "speed")):
        g = sk.groupby("drill_name")[m]
        sk[f"{m}_z"] = (sk[m] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        feats[f"ct_skill_{name}_z"] = sk.groupby("nfl_id")[f"{m}_z"].mean().reindex(ids)
    feats["ct_attempts"] = attempts.groupby("nfl_id").size().reindex(ids).fillna(0).astype(int)
    feats["ct_missing"] = feats[CT_FEATURES].isna().all(axis=1).astype(int)
    return feats.reset_index()


def official_vs_tracking(features: pd.DataFrame, prospects: pd.DataFrame) -> pd.DataFrame:
    """Convergent validity: does the tracking reproduce the official stopwatch? (Spearman, Pearson)."""
    from scipy import stats

    pairs = [("ct_forty_split_10yd_s", "ten_yd_split"), ("ct_forty_top_speed", "forty"),
             ("ct_forty_burst_2s", "ten_yd_split"), ("ct_3cone_peak_decel", "three_cone"),
             ("ct_shuttle_peak_decel", "short_shuttle")]
    official = prospects[["nfl_id"] + [c for c in prospects.columns if c not in features.columns]]
    df = features.merge(official, on="nfl_id", how="inner")
    rows = []
    for ct_col, off in pairs:
        t = df[[ct_col, off]].dropna()
        if len(t) < 10:
            continue
        rho, p = stats.spearmanr(t[ct_col], t[off])
        r, _ = stats.pearsonr(t[ct_col], t[off])
        rows.append({"tracking_feature": ct_col, "official_result": off, "n": len(t), "spearman": rho,
                     "pearson": r, "p_value": p})
    return pd.DataFrame(rows)


def build(ct_raw: pd.DataFrame, prospects: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Raw combine tracking -> (attempt table, player feature table for every prospect, audit)."""
    frames, a1 = clean_combine_tracking(ct_raw, cfg)
    attempts, a2 = attempt_metrics(frames, cfg)
    feats = player_features(attempts, prospects["nfl_id"]) if len(attempts) else \
        pd.DataFrame({"nfl_id": prospects["nfl_id"], **{c: np.nan for c in CT_FEATURES}, "ct_attempts": 0, "ct_missing": 1})
    audit = {**a1, **a2, "ct_players_with_features": int((feats["ct_missing"] == 0).sum())}
    return attempts, feats, audit
