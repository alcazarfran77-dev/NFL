"""On-field performance from player_play (added in v2.0.0).

Each metric is measured "over expected": the play's outcome minus the average outcome
of all prospect plays in the same context, so a player is not rewarded for his role or
his route tree. Positive is always better.

Family  Play set                         Metric (per play)                                Context for "expected"
DL      pass-rush snaps (rushed, did     goe  = expected get-off - get-off (s; + = quicker) alignment (EDGE / INTERIOR_LINE / other)
        not drop into coverage)          proe = pressure - expected pressure               alignment
OL      pass-block snaps                 ppoe = expected - peak pressure probability       alignment (T / G / C) x dropback quintile
                                         paoe = expected - pressure allowed (0/1)           same
WR/TE   routes with a separation reading soe  = separation - expected separation (yd)      route x alignment (route only if < 30 plays)

Player level: mean over qualifying plays, empirical-Bayes shrinkage toward the Combine
position-group mean (`<metric>_eb`), its reliability weight (`<metric>_rel`) and a
within-family z-score (`<metric>_z`).

Business rules (configs/config.yaml -> performance_rules)
  R10 regular season only; R11 plays nullified by penalty removed;
  R12 NGS get-off outside [0.30, 2.00) s treated as missing;
  R13 minimum qualifying plays per family (DL 20 rush snaps, OL 50 pass-block snaps, WR/TE 30 routes);
  R14 DBs get no on-field metric: the file holds their alignment and assignment but no coverage outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

ONFIELD_METRICS = ["goe", "proe", "ppoe", "paoe", "soe"]
METRIC_FAMILY = {"goe": "DL", "proe": "DL", "ppoe": "OL", "paoe": "OL", "soe": "WRTE"}
MIN_PLAYS_KEY = {"DL": "dl_rush", "OL": "ol_pass_block", "WRTE": "wrte_route"}
BOOL_COLS = ["target", "pressure_allowed", "blitzing", "unblocked_pressure"]
KEY = ["game_id", "play_id", "nfl_id"]


def _parse_bool(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().map({"TRUE": 1.0, "FALSE": 0.0}).astype("float64")


def clean_player_play(pp: pd.DataFrame, games: pd.DataFrame, prospects: pd.DataFrame,
                      cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (regular-season plays, play context for every season type, audit counts)."""
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

    # Context table for joining with tracking (tracking also covers the playoffs).
    context = df[KEY + ["season_type", "lined_up_position", "route_ran", "is_pass",
                        "coverage_assignment_at_snap", "player_get_off"]].copy()

    keep = df["season_type"].isin(rules["season_types"])  # R10
    audit["pp_rows_not_regular_season"] = int((~keep).sum())
    if rules["exclude_nullified"]:  # R11
        nullified = df["play_nullified_by_penalty"].eq("Y")
        audit["pp_rows_nullified"] = int((keep & nullified).sum())
        keep &= ~nullified
    lo, hi = rules["get_off_valid_s"]  # R12
    bad_go = df["player_get_off"].notna() & ~df["player_get_off"].between(lo, hi, inclusive="left")
    audit["pp_get_off_out_of_range"] = int((keep & bad_go).sum())
    df.loc[bad_go, "player_get_off"] = np.nan
    out = df[keep].reset_index(drop=True)
    audit["pp_rows_regular_season"] = len(out)
    return out, context, audit


def _expected(df: pd.DataFrame, value: str, keys: list[str], fallback: list[str] | None = None,
              min_n: int = 1) -> pd.Series:
    """Mean of `value` within context `keys`; cells with < min_n plays fall back to `fallback`."""
    grp = df.groupby(keys, dropna=False)[value]
    exp = grp.transform("mean")
    if fallback is not None:
        small = grp.transform("count") < min_n
        exp = exp.where(~small, df.groupby(fallback, dropna=False)[value].transform("mean"))
    return exp


def family_plays(plays: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Per-family play tables with the over-expected metrics."""
    rules = cfg["performance_rules"]
    out = {}

    # DL pass-rush snaps: a pass play where the lineman rushed (no coverage assignment).
    dl = plays[(plays.family == "DL") & plays["sack"].notna() & (plays.is_pass == 1)
               & (plays.coverage_assignment.isna() | plays.coverage_assignment.eq("NO_ASSIGNMENT"))].copy()
    dl["alignment"] = dl.lined_up_position.where(dl.lined_up_position.isin(["EDGE", "INTERIOR_LINE"]), "OTHER")
    dl["pressure"] = dl["time_to_pressure"].notna().astype(float)
    dl["goe"] = _expected(dl, "player_get_off", ["alignment"]) - dl["player_get_off"]
    dl["proe"] = dl["pressure"] - _expected(dl, "pressure", ["alignment"])
    out["DL"] = dl

    # OL pass-block snaps.
    ol = plays[(plays.family == "OL") & plays.peak_pressure_probability_allowed.notna()].copy()
    ol["alignment"] = ol.lined_up_position.where(ol.lined_up_position.isin(["T", "G", "C"]), "OTHER")
    q = rules["dropback_quantiles"]
    ol["dropback_bin"] = pd.qcut(ol.dropback_duration, q, labels=False, duplicates="drop").fillna(-1).astype(int)
    ol["pressure_allowed"] = ol["pressure_allowed"].fillna(0.0)
    ol["ppoe"] = _expected(ol, "peak_pressure_probability_allowed", ["alignment", "dropback_bin"]) - ol.peak_pressure_probability_allowed
    ol["paoe"] = _expected(ol, "pressure_allowed", ["alignment", "dropback_bin"]) - ol["pressure_allowed"]
    out["OL"] = ol

    # WR/TE routes with a separation reading.
    wr = plays[(plays.family == "WRTE") & plays.separation_at_pass_forward.notna() & plays.route_ran.notna()].copy()
    wr["alignment"] = wr.lined_up_position.where(wr.lined_up_position.isin(["WR", "SLOT_WR", "TE"]), "OTHER")
    wr["soe"] = wr.separation_at_pass_forward - _expected(wr, "separation_at_pass_forward", ["route_ran", "alignment"],
                                                          fallback=["route_ran"], min_n=30)
    out["WRTE"] = wr
    return out


def shrink_frame(stats: pd.DataFrame, metric: str, group: str = "combine_position") -> pd.DataFrame:
    """Empirical-Bayes shrinkage of player means toward their group mean (same estimator as the SBI)."""
    res = []
    for _, g in stats.groupby(group, observed=True):
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
    """One row per prospect with every qualifying on-field metric (NaN where not measured)."""
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
    """ICC (method of moments), mean EB weight and one-way ANOVA per on-field metric x position."""
    from scipy import stats

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
    """Does a rookie season predict the seasons after it? Spearman of rookie vs later means."""
    from scipy import stats

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
    """Snap burst net of play context, for the reliability re-check (closes v1 limitation 4).

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
