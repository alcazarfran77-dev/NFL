"""In-game movement metrics from 10 Hz player tracking.

Every metric is anchored to the snap, so it is comparable across plays and
independent of play direction (no left/right normalisation is needed: all
quantities are speeds, distances or angle changes).

Per player-play metrics
  burst_1s / burst_2s  straight-line distance from the snap spot at t = 1.0 / 2.0 s (yd).
                       burst_2s is the in-game analog of the Combine 10-yard split,
                       which elite to slow prospects cover in ~1.5-1.9 s.
  get_off_1s           speed at t = 1.0 s after the snap (yd/s).
  peak_accel_2s        highest acceleration magnitude in the first 2.0 s (yd/s^2).
  top_speed            highest speed after the snap (yd/s); analog of the 40.
  max_decel            highest braking rate, -d(speed)/dt on a 3-frame smoothed
                       speed trace while moving >= 1 yd/s (yd/s^2); analog of
                       the shuttle / 3-cone plant.
  max_cut_deg          largest change of direction of travel within 0.5 s while
                       moving >= 2 yd/s at both ends (degrees).

Player level
  level     = mean across a player's plays. The mean is unbiased whatever the number
              of plays; an 80th-percentile "capacity" was tried first (v0.3.0) and
              rejected: it reads ~0.4 yd low at 3 plays and ~0.2 yd low at 5, and
              because starters log more plays that bias leaked starter status into
              the metric. The 80th percentile is kept as `<metric>_q80` for the
              sensitivity analysis only.
  shrinkage = empirical-Bayes pull toward the position-group mean, weight n/(n+k),
              so a player seen on 3 plays is not ranked as confidently as one seen on 30;
  Snap Burst Index (SBI) = 50 + 10 * z of shrunk burst_2s within position group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

PLAY_KEY = ["game_id", "play_id", "nfl_id"]
METRICS = ["burst_1s", "burst_2s", "get_off_1s", "peak_accel_2s", "top_speed", "max_decel", "max_cut_deg"]


def clean_tracking(tr: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Type, sort and sanity-check tracking frames. Returns (frames, audit counts)."""
    rules = cfg["tracking_rules"]
    df = tr.copy()
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")
    df = df.sort_values(PLAY_KEY + ["time"]).reset_index(drop=True)
    audit: dict[str, int] = {"frames_in": len(df)}

    # Frame spacing must be 1 / frame_rate; plays with gaps are dropped (none expected).
    step = 1.0 / rules["frame_rate_hz"]
    gaps = df.groupby(PLAY_KEY)["time"].diff().dt.total_seconds()
    bad_plays = df.loc[gaps.notna() & ~np.isclose(gaps, step), PLAY_KEY].drop_duplicates()
    audit["plays_with_frame_gaps"] = len(bad_plays)
    if len(bad_plays):
        df = df.merge(bad_plays.assign(_bad=1), on=PLAY_KEY, how="left")
        df = df[df["_bad"].isna()].drop(columns="_bad")

    # Physically implausible speed / acceleration -> NaN, then interpolate within the play.
    for col, cap in (("s", rules["speed_cap"]), ("a", rules["accel_cap"])):
        bad = df[col] > cap
        audit[f"frames_{col}_above_cap"] = int(bad.sum())
        df.loc[bad, col] = np.nan
        df[col] = df.groupby(PLAY_KEY)[col].transform(lambda s: s.interpolate(limit_direction="both"))

    # Positions far outside the field are flagged (kept: players do run out of bounds).
    audit["frames_out_of_field"] = int(((df.x < -10) | (df.x > 130) | (df.y < -10) | (df.y > 63.3)).sum())
    audit["frames_out"] = len(df)
    return df.reset_index(drop=True), audit


def _heading_change(dir_deg: np.ndarray, lag: int) -> np.ndarray:
    """Absolute wrapped change in direction of travel over `lag` frames (degrees)."""
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

    k = sigma2_within / tau2_between, weight = n / (n + k). sigma2_within is the
    pooled per-play variance inside players; tau2 is the between-player variance
    of player means net of sampling noise (method of moments, floored at 5% of the
    observed variance so no group is shrunk to a single point).
    """
    mean = group[f"{col}_mean"]
    w = shrink_weight(group, col)
    grand = np.average(mean, weights=group["n_plays"])
    return grand + w * (mean - grand)


def shrink_weight(group: pd.DataFrame, col: str) -> pd.Series:
    """Empirical-Bayes weight n / (n + k) on a player's own mean (= its reliability)."""
    n = group["n_plays"]
    s2w = group[f"{col}_var"].fillna(group[f"{col}_var"].median())
    sigma2 = np.average(s2w, weights=n - 1)
    total = group[f"{col}_mean"].var(ddof=1)
    tau2 = max(total - np.mean(sigma2 / n), 0.05 * total)
    return n / (n + sigma2 / tau2)


def player_profiles(plays: pd.DataFrame, prospects: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Aggregate play metrics to players, shrink, and compute the Snap Burst Index."""
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
    # Same normalisation for the other shrunk metrics (used as classifier features).
    for col in METRICS:
        gcol = prof.groupby("combine_position", observed=True)[f"{col}_eb"]
        prof[f"{col}_z"] = (prof[f"{col}_eb"] - gcol.transform("mean")) / gcol.transform("std")
    return prof.drop(columns=[c for c in prof.columns if c.endswith("_var")]).reset_index(drop=True)


def metric_reliability(plays: pd.DataFrame, profiles: pd.DataFrame, group: str = "combine_position") -> pd.DataFrame:
    """Is a movement metric a stable player trait? One row per (position group, metric).

    icc            share of between-player variance in player means net of sampling noise
                   (method of moments; <= 0 means no detectable player differences)
    mean_weight    average empirical-Bayes weight n/(n+k) = reliability of a player's estimate
    anova_p        one-way ANOVA p-value for "all players in the group move alike"
    """
    from scipy import stats

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
    """Bias of mean / median / 80th percentile when a player is seen on only n plays.

    Uses players with >= min_plays plays as ground truth and subsamples n of them.
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
    """Post-snap speed traces (t, speed) for the given players, for plotting."""
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
