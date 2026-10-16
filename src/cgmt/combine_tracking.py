"""Combine drill tracking (spec File 3, combine_tracking.csv) -> per-prospect movement features.

v2.1.0 built the module; v2.2.0 calibrated it on the delivered file (combine_tracking_6.csv:
6,310 attempts, 510 prospects). The official Combine results are stopwatch times; the drill tracking
shows how each time was produced. Metrics use the same definitions as in-game tracking (`cgmt.movement`).

Calibration finding (v2.2.0): **the start of a tracked attempt is not repeatable.** A tracked forty begins
at a variable moment of the start (first-frame speed 0.2-3.2 yd/s), so anything timed from the first frame
(1 s / 2 s burst, 10-yd split from the start, time to 90% of top speed) barely
repeats between a player's two attempts (0-5 yd split: r = 0.16). Splits between distance marks do not
depend on when the clock started and repeat well (5-10 yd 0.87, 10-20 yd 0.98, top speed 0.99).
Features are therefore gated by reliability, the same rule the game metrics follow (ICC before use):
a metric can become a model feature only if it repeats at r >= 0.5, and each construct gets one feature
(its most reliable metric). `attempt_reliability` writes every metric, kept or not and why, to
reports/metrics/ct_reliability.csv.

Per attempt (PLAYER rows; the attempt starts at the frame before the first frame with speed >= start_speed)
  top_speed, peak_accel, peak_decel (3-frame smoothed braking while moving >= 1 yd/s), max_cut_deg,
  burst_1s / burst_2s, t90_top_s, duration_s
  40-yard dash: split_0_5_s, split_5_10_s, split_10_20_s (time between distance marks, interpolated),
                split_10yd_s (from the start, as the official split)
  3-cone, shuttle: n_plants and plant_speed_ratio (speed minima below 50% of the preceding peak; mean ratio
                of plant speed to that peak = speed kept through the turn)

Player features (CT_FEATURES; best attempt per metric, as the official Combine keeps the best result)
  construct              feature                        reliability on the delivered file
  sprint acceleration    ct_forty_split_5_10_s          0.87 (5 -> 10 yd; lower = better)
  sprint top speed       ct_forty_top_speed             0.99 (same construct as the 10-20 yd split, 0.98)
  change of direction    ct_shuttle_plant_speed_ratio   0.54 (speed kept through the shuttle turns)
  position-drill speed   ct_skill_speed_z               top speed z-scored within each drill (position group x
                                                        drill name), player mean
  position-drill accel.  ct_skill_accel_z               peak acceleration, same scaling
  (reliabilities on the delivered file are in reports/metrics/ct_reliability.csv)

Business rules (configs/config.yaml -> combine_tracking_rules)
  R16 ball rows and rows without nfl_id dropped; frames must be 0.1 s apart within an attempt;
      speed / acceleration above the in-game caps -> NaN and interpolated; attempts shorter than
      `min_frames` or that never reach `start_speed` dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config

ATTEMPT_KEY = ["event_id"]
FORTY, THREE_CONE, SHUTTLE = "FORTY_YARD_DASH", "THREE_CONE_DRILL", "SHORT_SHUTTLE"
SKILL_PREFIX = "SKILL_DRILLS_"

CT_FEATURES = [
    "ct_forty_split_5_10_s", "ct_forty_top_speed", "ct_shuttle_plant_speed_ratio",
    "ct_skill_speed_z", "ct_skill_accel_z",
]
# Lower is better for these (times); everything else: higher is better.
CT_LOWER_IS_BETTER = {"ct_forty_split_5_10_s"}
CT_LABELS = {
    "ct_forty_split_5_10_s": "40: 5-10 yd split", "ct_forty_top_speed": "40: top speed",
    "ct_shuttle_plant_speed_ratio": "Shuttle: speed kept in turns",
    "ct_skill_speed_z": "Position drills: top speed", "ct_skill_accel_z": "Position drills: acceleration",
}

_BEST = {  # drill group -> {attempt metric: "max" | "min"} for the player features
    "forty": {"split_5_10_s": "min", "top_speed": "max"},
    "shuttle": {"plant_speed_ratio": "max"},
}
_SKILL_Z = {"top_speed": "speed", "peak_accel": "accel"}

# Metrics whose attempt-to-attempt reliability is reported (feature or not).
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
# Why a metric that passes the reliability gate is still not a feature (one feature per construct).
NOT_USED = {("forty", "split_10_20_s"): "same construct as top speed (r = -0.95), slightly less reliable",
            ("forty", "peak_accel"): "same construct as the 5-10 yd split, less reliable",
            ("3cone", "duration_s"): "duplicates the official 3-cone time",
            ("3cone", "top_speed"): "sprint speed, already measured in the 40"}

DEFAULT_RULES = {"frame_rate_hz": 10, "start_speed": 0.5, "min_frames": 15, "speed_cap": 12.5,
                 "accel_cap": 15.0, "plant_drop": 0.5, "cut_window_s": 0.5, "cut_min_speed": 2.0,
                 "min_reliability": 0.5}


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


DRILL_KEY = ["drill_type", "drill_name"]  # a drill as run by one position group (GAUNTLET_DRILL: WR, TE, DB)


def _skill_z(attempts: pd.DataFrame, metrics) -> pd.DataFrame:
    """Position-drill attempts with each metric z-scored within its drill (position group x drill name).

    Drills differ by design, and the same drill name can be run by several position groups (the gauntlet by
    WRs, TEs and DBs), so the reference group is the drill as run by one position group.
    """
    sk = attempts[attempts.drill_group == "skill"].copy()
    for m in metrics:
        g = sk.groupby(DRILL_KEY)[m]
        sk[f"{m}_z"] = (sk[m] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return sk


def attempt_reliability(attempts: pd.DataFrame, min_reliability: float = 0.5) -> pd.DataFrame:
    """How well each metric repeats within a player (label-free; decides which metrics become features).

    40 / 3-cone / shuttle: Pearson r between a player's first two attempts (test-retest).
    Position drills: split-half r of the mean within-drill z (alternate attempts), Spearman-Brown corrected
    to the full set of attempts, which is what the feature averages.
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
    """Best attempt per metric (`ct_<group>_<metric>`) plus mean within-drill z for position drills."""
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


def official_vs_tracking(features: pd.DataFrame, prospects: pd.DataFrame) -> pd.DataFrame:
    """Convergent validity: does the tracking reproduce the official stopwatch? (Spearman, Pearson)."""
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
        t = df[[ct_col, off]].dropna()
        if len(t) < 10:
            continue
        rho, p = stats.spearmanr(t[ct_col], t[off])
        r, _ = stats.pearsonr(t[ct_col], t[off])
        rows.append({"tracking_feature": ct_col, "official_result": off, "n": len(t), "spearman": rho,
                     "pearson": r, "p_value": p})
    return pd.DataFrame(rows)


def build(ct_raw: pd.DataFrame, prospects: pd.DataFrame, cfg: Config
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Raw combine tracking -> (attempt table, player features for every prospect, reliability, audit)."""
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
