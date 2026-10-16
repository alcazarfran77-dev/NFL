"""Figures for the technical report (PNG, 200 dpi) in reports/figures.

Style rules (from the project's data-viz checklist): one quiet grey for data and
one blue accent for what the title is about; diverging maps are blue <-> orange
around a neutral grey; hairline recessive grids; every title states the finding
with its number, computed from the data rather than typed.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .config import Config  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ACCENT, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED = "#b4b2ab"
SURFACE = "#fcfcfb"
POS_ORDER = ["DB", "DL", "OL", "TE", "WR"]
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_orange", [ORANGE, "#f6c2ab", "#efeeea", "#9ec5f4", ACCENT])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#eef4fc", "#9ec5f4", "#3987e5", "#184f95"])

LABELS = {
    "ten_yd_split": "10-yd split", "forty": "40-yd dash", "vertical": "Vertical jump", "broad_jump": "Broad jump",
    "three_cone": "3-cone", "short_shuttle": "Short shuttle", "bench_reps": "Bench reps",
    "combine_weight": "Body weight", "ngs_athleticism_score": "NGS athleticism score",
    "burst_2s_eb": "Snap burst\n(2-s distance)", "get_off_1s_eb": "Get-off\n(speed at 1 s)",
    "peak_accel_2s_eb": "Peak accel\n(first 2 s)", "top_speed_eb": "Top speed", "max_decel_eb": "Max braking",
    "max_cut_deg_eb": "Sharpest cut", "start_rate": "Start rate",
    "position_mean": "Position mean\n(baseline)", "ridge": "Ridge", "random_forest": "Random forest", "xgboost": "XGBoost",
    "logistic_regression": "Logistic reg.",
    "ten_split_momentum": "10-yd momentum (weight / split)", "speed_score": "Speed Score", "ten_split_share": "10-yd share of the 40",
    "ngs_college_production_score": "NGS production score", "age_at_draft": "Age at draft", "combine_position": "Position group",
    "log_draft_pick": "Draft pick (log)", "is_undrafted": "Undrafted", "wing_span": "Wingspan", "hand_size": "Hand size",
    "arm_length": "Arm length", "combine_height": "Height", "power_conference": "Power conference", "n_drills_run": "Drills run",
    "sbi": "SBI", "top_speed_z": "Top speed (game)", "max_cut_deg_z": "Sharpest cut (game)", "burst_1s_z": "Distance at 1 s (game)",
    "get_off_1s_z": "Get-off (game)", "peak_accel_2s_z": "Peak accel (game)", "max_decel_z": "Max braking (game)",
    "combine": "Combine only", "combine_draft": "Combine + draft", "combine_draft_mov": "Combine + draft\n+ game movement",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlelocation": "left", "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "legend.frameon": False,
    })


def _title(fig, title: str, subtitle: str = "") -> None:
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.012, 0.935, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)


def _save(fig, cfg: Config, name: str, source: str) -> None:
    fig.text(0.012, 0.008, f"Source: {source}", ha="left", va="bottom", fontsize=7.5, color=INK2)
    fig.savefig(cfg.path("figures") / name, dpi=200)
    plt.close(fig)


def _mean_ci(v: np.ndarray) -> tuple[float, float]:
    from scipy import stats
    v = np.asarray(v, float)
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


# ----------------------------------------------------------------------------- EDA
def fig_drill_coverage(cfg, prospects):
    drills = ["forty", "ten_yd_split", "vertical", "broad_jump", "short_shuttle", "three_cone", "bench_reps"]
    cov = prospects.groupby("combine_position", observed=True)[drills].apply(lambda g: g.notna().mean()).reindex(POS_ORDER)
    overall = prospects[drills].notna().mean()
    fig, ax = plt.subplots(figsize=(9, 3.9))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.12)
    ax.imshow(cov.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            v = cov.iat[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(drills)), [f"{LABELS[d]}\n({overall[d]:.0%} overall)" for d in drills], fontsize=8.5)
    ax.set_yticks(range(len(POS_ORDER)), POS_ORDER)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    agil = prospects[["short_shuttle", "three_cone"]].notna().any(axis=1).mean()
    _title(fig, f"Agility drills are the blind spot: only {agil:.0%} of prospects ran the shuttle or 3-cone",
           "Share of prospects with a recorded result, by Combine position group (n = 510)")
    _save(fig, cfg, "fig01_drill_coverage.png", "combine_results.csv, 2023-2025 classes")


def fig_drill_correlations(cfg, prospects):
    cols = ["ten_yd_split", "forty", "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps", "combine_weight", "ngs_athleticism_score"]
    df = prospects[cols + ["combine_position"]].copy()
    for c in cols:
        df[c] = df[c] - df.groupby("combine_position", observed=True)[c].transform("mean")
    for c in ["ten_yd_split", "forty", "three_cone", "short_shuttle"]:
        df[c] = -df[c]
    corr = df[cols].corr(method="spearman", min_periods=30)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.86, bottom=0.2)
    shown = corr.mask(mask)
    ax.imshow(shown.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(len(cols)):
        for j in range(i):
            v = corr.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.6 else INK)
    ax.set_xticks(range(len(cols)), [LABELS[c] for c in cols], rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(cols)), [LABELS[c] for c in cols], fontsize=8.5)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    r = corr.loc["forty", "ten_yd_split"]
    _title(fig, f"Sprint drills are near-duplicates (40 vs 10-yd split rho = {r:.2f}); bench press stands apart",
           "Within-position Spearman correlation; times sign-flipped so + means 'both better'")
    _save(fig, cfg, "fig02_drill_correlations.png", "combine_results.csv, position-group means removed")


def fig_burst_distribution(cfg, plays, anomalies):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.12)
    rng = np.random.default_rng(cfg.seed)
    flagged = anomalies[anomalies.metric == "burst_2s"].set_index(["game_id", "play_id", "nfl_id"]).index
    key = plays.set_index(["game_id", "play_id", "nfl_id"]).index
    plays = plays.assign(flag=key.isin(flagged))
    data = [plays.loc[plays.combine_position == p, "burst_2s"].to_numpy() for p in POS_ORDER]
    ax.boxplot(data, positions=range(len(POS_ORDER)), widths=0.5, showfliers=False,
               medianprops={"color": INK, "linewidth": 1.4}, boxprops={"color": INK2}, whiskerprops={"color": INK2}, capprops={"color": INK2})
    for i, p in enumerate(POS_ORDER):
        sub = plays[plays.combine_position == p]
        jit = rng.uniform(-0.18, 0.18, len(sub))
        ok = ~sub.flag.to_numpy()
        ax.scatter(i + jit[ok], sub.burst_2s[ok], s=5, color=MUTED, alpha=0.45, linewidths=0)
        ax.scatter(i + jit[~ok], sub.burst_2s[~ok], s=16, color=ACCENT, linewidths=0, label="Flagged outlier (|robust z| > 3.5)" if i == 0 else None)
        ax.text(i, -0.9, f"n = {len(sub):,}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(POS_ORDER)), POS_ORDER)
    ax.set_ylabel("Distance from snap spot at t = 2.0 s (yd)")
    ax.set_ylim(-1.4, plays.burst_2s.max() + 0.8)
    ax.legend(loc="upper left", fontsize=8.5)
    med = plays.groupby("combine_position", observed=True).burst_2s.median()
    _title(fig, f"WRs cover {med['WR']:.1f} yd in the first 2 s, OL {med['OL']:.1f} yd; {int(plays.flag.sum())} plays flagged as outliers",
           "Per-play 2-second snap burst by position group; box = median and IQR, whiskers 1.5 IQR")
    _save(fig, cfg, "fig03_burst_distribution.png", "game_tracking 2024-2025, Week 18 + playoffs, 2,677 player-plays")


def fig_speed_traces(cfg, frames, plays, profiles, prospects):
    from .movement import speed_curves
    roles = prospects.set_index("nfl_id").nfl_position.astype(str)
    edge = profiles[(profiles.combine_position == "DL") & (profiles.n_plays >= 8)
                    & profiles.nfl_id.map(roles).isin(["DE", "OLB"])].sort_values("sbi")
    lo, hi = edge.iloc[0], edge.iloc[-1]
    curves = speed_curves(frames, plays, [int(lo.nfl_id), int(hi.nfl_id)], 4.0)
    names = prospects.set_index("nfl_id").display_name
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.12)
    peaks = {}
    for pid, color, lab in [(int(lo.nfl_id), MUTED, "low"), (int(hi.nfl_id), ACCENT, "high")]:
        c = curves[curves.nfl_id == pid]
        for _, g in c.groupby(["game_id", "play_id"]):
            ax.plot(g.t, g.speed, color=color, alpha=0.25, linewidth=0.8)
        med = c.groupby("t").speed.median()
        ax.plot(med.index, med.values, color=color if lab == "high" else INK2, linewidth=2.2)
        p = profiles.set_index("nfl_id").loc[pid]
        early = med[med.index <= 2.0]
        peaks[lab] = early.max()
        tx = 0.75 if lab == "high" else 0.95
        ty = early.max() + 0.35 if lab == "high" else float(med.loc[min(med.index, key=lambda t: abs(t - 1.0))]) - 0.75
        ax.text(tx, ty, f"{names[pid]} ({roles[pid]}): SBI {p.sbi:.0f}, {int(p.n_plays)} plays",
                fontsize=8.5, color=INK, fontweight="bold" if lab == "high" else "normal")
    ax.axvline(2.0, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.text(2.04, 0.2, "SBI measured here (t = 2.0 s)", fontsize=8, color=INK2)
    ax.set_xlabel("Seconds after the snap")
    ax.set_ylabel("Speed (yd/s)")
    ax.set_xlim(0, 4)
    _title(fig, f"Two edge rushers: one peaks at {peaks['high']:.1f} yd/s inside 2 s, the other at {peaks['low']:.1f}",
           "Post-snap speed on every tracked play (thin) and the median trace (thick); highest vs lowest SBI among DE/OLB with 8+ plays")
    _save(fig, cfg, "fig04_speed_traces.png", "game_tracking 2024-2025")


def fig_estimator_bias(cfg, bias, sens):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.14, wspace=0.35)
    colors = {"q80": ORANGE, "median": MUTED, "mean": ACCENT}
    names = {"q80": "80th percentile (rejected)", "median": "Median", "mean": "Mean (adopted)"}
    for est in ["q80", "median", "mean"]:
        b = bias[bias.estimator == est].sort_values("n_plays")
        a1.plot(b.n_plays, b.bias, marker="o", color=colors[est], linewidth=2, markersize=6)
        dy = {"mean": 0.022, "median": -0.022, "q80": 0.0}[est]
        a1.text(b.n_plays.iloc[-1] + 0.4, b.bias.iloc[-1] + dy, names[est], va="center", fontsize=8.5, color=INK)
    a1.axhline(0, color=INK2, linewidth=0.8)
    a1.set_xlim(2, 15.5)
    a1.set_xticks([3, 5, 10])
    a1.set_xlabel("Plays observed for the player")
    a1.set_ylabel("Bias vs full-sample value (yd)")
    a1.set_title("A. Estimate bias when a player is seen on few plays")
    s = sens.set_index(["x", "y"])
    ys = [("n_plays", "Plays observed"), ("start_rate", "Start rate"), ("neg_log_pick", "Draft capital")]
    w = 0.36
    for k, (est, col) in enumerate([("sbi_q80", ORANGE), ("sbi", ACCENT)]):
        vals = [s.loc[(est, y), "rho"] for y, _ in ys]
        lo = [s.loc[(est, y), "rho"] - s.loc[(est, y), "ci_low"] for y, _ in ys]
        hi = [s.loc[(est, y), "ci_high"] - s.loc[(est, y), "rho"] for y, _ in ys]
        x = np.arange(len(ys)) + (k - 0.5) * w
        a2.bar(x, vals, width=w - 0.04, color=col, alpha=0.85, label=names["q80" if est == "sbi_q80" else "mean"])
        a2.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    a2.axhline(0, color=INK2, linewidth=0.8)
    a2.set_xticks(range(len(ys)), [l for _, l in ys])
    a2.set_ylabel("Within-position Spearman rho (95% CI)")
    a2.set_title("B. What the SBI correlates with")
    a2.legend(fontsize=8, loc="upper right")
    q = bias[(bias.estimator == "q80") & (bias.n_plays == 3)].bias.iloc[0]
    _title(fig, f"Top-percentile capacity reads {abs(q):.2f} yd low at 3 plays, which leaks playing time into the metric",
           "Subsampling players with 20+ plays; panel B compares SBI built on each estimator")
    _save(fig, cfg, "fig05_estimator_bias.png", "play_metrics + careers; 200 subsamples per player and n")


def fig_translation_matrix(cfg, matrix):
    xs = ["ten_yd_split", "forty", "broad_jump", "vertical", "short_shuttle", "three_cone", "bench_reps", "combine_weight", "ngs_athleticism_score"]
    ys = ["burst_2s_eb", "get_off_1s_eb", "peak_accel_2s_eb", "top_speed_eb", "max_decel_eb", "max_cut_deg_eb", "start_rate"]
    m = matrix.set_index(["x", "y"])
    rho = np.array([[m.loc[(x, y), "rho"] for y in ys] for x in xs])
    q = np.array([[m.loc[(x, y), "q_value"] for y in ys] for x in xs])
    n = [int(m.loc[(x, ys[0]), "n"]) for x in xs]
    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.subplots_adjust(left=0.24, right=0.98, top=0.83, bottom=0.14)
    ax.imshow(rho, cmap=DIVERGING, vmin=-0.4, vmax=0.4, aspect="auto")
    for i in range(len(xs)):
        for j in range(len(ys)):
            sig = q[i, j] < 0.05
            ax.text(j, i, f"{rho[i, j]:+.2f}{'*' if sig else ''}", ha="center", va="center", fontsize=9,
                    color=INK, fontweight="bold" if sig else "normal")
    ax.axvline(len(ys) - 1.5, color=SURFACE, linewidth=4)
    ax.set_xticks(range(len(ys)), [LABELS[y] for y in ys], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(xs)), [f"{LABELS[x]}  (n={k})" for x, k in zip(xs, n)], fontsize=9)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    best = matrix[matrix.y == "burst_2s_eb"].sort_values("rho", ascending=False).iloc[0]
    _title(fig, "Sprint and jump drills carry into game movement; the NGS composite and bench press do not",
           f"Within-position Spearman rho, Combine result vs in-game metric (* = FDR q < 0.05). Best for snap burst: {LABELS[best.x]} ({best.rho:+.2f})")
    fig.text(0.24, 0.07, "Blue = better test goes with more game movement; orange = the reverse. Body weight is shown as measured (heavier = more).",
             fontsize=8, color=INK2)
    _save(fig, cfg, "fig07_translation_matrix.png", "combine_results + game_tracking, 221 players with 3+ plays")


def fig_sbi_vs_start(cfg, profiles, prospects, starter_assoc):
    df = profiles.merge(prospects, on=["nfl_id", "combine_position"])
    rho = starter_assoc.set_index("x").loc["sbi"]
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.8), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.74, bottom=0.16, wspace=0.12)
    for ax, p in zip(axes, POS_ORDER):
        sub = df[df.combine_position == p]
        ax.scatter(sub.sbi, sub.start_rate, s=14, color=MUTED, linewidths=0)
        if len(sub) > 5:
            b, a = np.polyfit(sub.sbi, sub.start_rate, 1)
            xs = np.linspace(sub.sbi.min(), sub.sbi.max(), 10)
            ax.plot(xs, a + b * xs, color=ACCENT, linewidth=2)
        ax.set_title(f"{p}  (n = {len(sub)})", fontsize=9.5)
        ax.set_xlabel("SBI")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Share of games started")
    _title(fig, f"Game snap burst barely tracks starting within position (rho = {rho.rho:+.2f}, 95% CI {rho.ci_low:+.2f} to {rho.ci_high:+.2f})",
           "Each dot is a player with 3+ tracked plays; blue line = least-squares fit within the position group")
    _save(fig, cfg, "fig09_sbi_vs_start_rate.png", "game_tracking + player_career_successes")


# ----------------------------------------------------------------------------- reliability
def fig_reliability(cfg, rel):
    metrics = ["burst_1s", "burst_2s", "get_off_1s", "peak_accel_2s", "top_speed", "max_decel", "max_cut_deg"]
    names = {"burst_1s": "Distance at 1 s", "burst_2s": "Snap burst\n(distance at 2 s)", "get_off_1s": "Get-off\n(speed at 1 s)",
             "peak_accel_2s": "Peak accel\n(first 2 s)", "top_speed": "Top speed", "max_decel": "Max braking", "max_cut_deg": "Sharpest cut"}
    r = rel.set_index(["combine_position", "metric"])
    icc = np.array([[max(r.loc[(p, m), "icc"], 0) for m in metrics] for p in POS_ORDER])
    pv = np.array([[r.loc[(p, m), "anova_p"] for m in metrics] for p in POS_ORDER])
    n = [int(r.loc[(p, metrics[0]), "players"]) for p in POS_ORDER]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.74, bottom=0.08)
    ax.imshow(icc, cmap=SEQUENTIAL, vmin=0, vmax=0.8, aspect="auto")
    for i in range(len(POS_ORDER)):
        for j in range(len(metrics)):
            v = icc[i, j]
            raw = r.loc[(POS_ORDER[i], metrics[j]), "icc"]
            txt = "none" if raw <= 0 else f"{v:.2f}"
            sig = np.isfinite(pv[i, j]) and pv[i, j] < 0.01
            ax.text(j, i, txt + ("*" if sig else ""), ha="center", va="center", fontsize=9,
                    color="white" if v > 0.5 else INK, fontweight="bold" if sig else "normal")
    ax.set_xticks(range(len(metrics)), [names[m] for m in metrics], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(POS_ORDER)), [f"{p} ({k})" for p, k in zip(POS_ORDER, n)])
    ax.grid(False)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    dl = r.loc[("DL", "burst_2s"), "icc"]
    _title(fig, f"Snap burst is a player trait only on the defensive line (ICC {dl:.2f}); for WR and DB the play decides it",
           "Share of variance in player averages that is real player difference (ICC, method of moments); * = ANOVA p < 0.01; 'none' = not detectable")
    _save(fig, cfg, "fig06_metric_reliability.png", "game_tracking, 221 players with 3+ plays")


def fig_dl_role(cfg, dl_tab):
    xs = ["ten_yd_split", "forty", "broad_jump", "vertical", "combine_weight", "ngs_athleticism_score"]
    y = "get_off_1s_eb"
    t = dl_tab[dl_tab.y == y].set_index(["adjustment", "x"])
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    fig.subplots_adjust(left=0.24, right=0.97, top=0.78, bottom=0.14)
    for i, x in enumerate(xs):
        a, b = t.loc[("none", x)], t.loc[("edge vs interior role", x)]
        ax.plot([a.rho, b.rho], [i, i], color=MUTED, linewidth=2, zorder=1)
        ax.errorbar(b.rho, i, xerr=[[b.rho - b.ci_low], [b.ci_high - b.rho]], fmt="none", ecolor=ACCENT, elinewidth=1, alpha=0.5, capsize=3)
        ax.scatter(a.rho, i, s=55, color=INK2, zorder=3, label="Raw (all DL)" if i == 0 else None)
        ax.scatter(b.rho, i, s=55, color=ACCENT, zorder=3, label="Within role (edge vs interior), 95% CI" if i == 0 else None)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(xs)), [f"{LABELS[x]} (n={int(t.loc[('none', x), 'n'])})" for x in xs])
    ax.invert_yaxis()
    ax.set_xlabel("Spearman rho with in-game get-off (speed 1 s after the snap); times sign-flipped")
    ax.legend(loc="lower left", fontsize=8.5)
    raw = t.loc[("none", "ten_yd_split"), "rho"]
    adj = t.loc[("edge vs interior role", "ten_yd_split"), "rho"]
    _title(fig, f"On the DL, half the drill-to-game link is role: 10-yd split {raw:+.2f} raw, {adj:+.2f} within role",
           "Edge = DE/OLB, interior = DT/NT; heavier interior linemen test slower and also get off slower by assignment")
    _save(fig, cfg, "fig08_dl_translation_by_role.png", "defensive linemen with 3+ tracked plays")


# ----------------------------------------------------------------------------- results
def fig_regression_cv(cfg, m_dir, summary):
    order = ["position_mean", "ridge", "random_forest", "xgboost"]
    pops = [("all_positions", "A. All positions (n = {n})"), ("defensive_line", "B. Defensive line only (n = {n})")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.17, wspace=0.22)
    rng = np.random.default_rng(cfg.seed)
    best = {}
    for ax, (pop, title) in zip(axes, pops):
        folds = pd.read_csv(m_dir / f"regression_fold_{pop}_r2.csv")
        s = summary[summary.population == pop]
        champ = s[s.model != "position_mean"].sort_values("rmse_mean").iloc[0].model
        best[pop] = s.set_index("model").loc[champ]
        for i, m in enumerate(order):
            v = folds[m].to_numpy()
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=12, color=MUTED, linewidths=0)
            mu, half = _mean_ci(v)
            col = ACCENT if m == champ else INK
            ax.errorbar(i + 0.28, mu, yerr=half, fmt="o", color=col, ecolor=col, capsize=4, markersize=6, elinewidth=1.6)
            ax.text(i + 0.38, mu, f"{mu:+.2f}", va="center", fontsize=8.5, color=col)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks(range(len(order)), [LABELS[m] for m in order], fontsize=8.5)
        ax.set_title(title.format(n=int(s.n.iloc[0])))
        ax.set_xlim(-0.5, len(order) - 0.2)
    axes[0].set_ylabel("Out-of-fold R² of the Snap Burst Index")
    a, b = best["all_positions"], best["defensive_line"]
    _title(fig, f"Combine results predict snap burst only where it is a trait: R² {b.r2_mean:+.2f} on the DL vs {a.r2_mean:+.2f} for all positions",
           "Nested CV, 5 folds x 3 repeats; dots = outer folds, marker = mean with 95% CI; blue = lowest-RMSE learned model")
    _save(fig, cfg, "fig10_regression_cv.png", "Snap Burst Index from Combine measurables and drills")


def fig_classification_auc(cfg, folds):
    pops = [("all_prospects", ["combine", "combine_draft"], "A. All prospects (n = 510)"),
            ("tracked_subset", ["combine_draft", "combine_draft_mov"], "B. Players with tracking (n = 221)")]
    models = ["logistic_regression", "random_forest", "xgboost"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.2, wspace=0.08)
    for ax, (pop, fsets, title) in zip(axes, pops):
        for j, fs in enumerate(fsets):
            for i, m in enumerate(models):
                v = folds[(folds.population == pop) & (folds.feature_set == fs) & (folds.model == m)].roc_auc.to_numpy()
                mu, half = _mean_ci(v)
                x = i + (j - 0.5) * 0.32
                col = ACCENT if j == 1 else MUTED
                ax.errorbar(x, mu, yerr=half, fmt="o", color=col, ecolor=col, capsize=4, markersize=7, elinewidth=1.8,
                            label=LABELS[fs].replace("\n", " ") if i == 0 else None)
                ax.text(x, mu + half + 0.012, f"{mu:.2f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(models)), [LABELS[m] for m in models])
        ax.set_title(title)
        ax.legend(loc="lower left", fontsize=8.5)
        ax.set_ylim(0.6, 1.0)
    axes[0].set_ylabel("ROC-AUC, outer folds (mean, 95% CI)")
    a = folds[(folds.population == "all_prospects")].groupby("feature_set").roc_auc.mean()
    b = folds[(folds.population == "tracked_subset")].groupby("feature_set").roc_auc.mean()
    _title(fig, f"Draft capital lifts AUC from {a['combine']:.2f} to {a['combine_draft']:.2f}; game movement adds {b['combine_draft_mov'] - b['combine_draft']:+.3f}",
           "Predicting a starter (started 50%+ of available games); nested CV, 5 folds x 3 repeats")
    _save(fig, cfg, "fig11_classification_auc.png", "players + combine_results + player_career_successes")


def fig_roc(cfg, oof, summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.12, wspace=0.25)
    pops = [("all_prospects", ["combine", "combine_draft"], "A. All prospects"),
            ("tracked_subset", ["combine_draft", "combine_draft_mov"], "B. Players with tracking")]
    for ax, (pop, fsets, title) in zip(axes, pops):
        models = []
        for j, fs in enumerate(fsets):
            s = summary[(summary.population == pop) & (summary.feature_set == fs)].sort_values("roc_auc_mean", ascending=False).iloc[0]
            models.append(LABELS[s.model])
            o = oof[(oof.population == pop) & (oof.feature_set == fs) & (oof.model == s.model)]
            o = o.groupby("row").agg(y=("y", "first"), p=("pred", "mean"))
            fpr, tpr, _ = roc_curve(o.y, o.p)
            col = ACCENT if j == 1 else INK2
            ax.plot(fpr, tpr, color=col, linewidth=2,
                    label=f"{LABELS[fs].replace(chr(10), ' ')}: {s.oof_auc:.2f} [{s.oof_auc_ci_low:.2f}, {s.oof_auc_ci_high:.2f}]")
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{title} ({' / '.join(dict.fromkeys(models))})")
        ax.legend(loc="lower right", fontsize=7.5, title="ROC-AUC [95% CI]", title_fontsize=7.5)
    _title(fig, "Out-of-fold ROC curves of the best model per feature set",
           "Predictions averaged over the 3 CV repeats; brackets = bootstrap 95% CI (2,000 stratified resamples)")
    _save(fig, cfg, "fig12_roc_curves.png", "nested CV out-of-fold predictions")


def fig_importance(cfg, imp_clf, imp_reg, finals, reg_champ):
    imp_reg = imp_reg[imp_reg.population == "defensive_line"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.8, bottom=0.15, wspace=0.75)
    def nice(f):
        if f.endswith("_missing"):
            return f"{LABELS.get(f[:-8], f[:-8]).replace(chr(10), ' ')} not run"
        return LABELS.get(f, f).replace("\n", " ").replace("_", " ")
    for ax, imp, title, unit in [(a1, imp_clf, f"A. Starter classifier ({LABELS[finals['champion']['model']]})", "drop in ROC-AUC"),
                                 (a2, imp_reg, f"B. DL snap-burst regressor ({LABELS[reg_champ]})", "rise in RMSE (SBI pts)")]:
        top = imp.head(10).iloc[::-1]
        cols = [ACCENT if i == len(top) - 1 else MUTED for i in range(len(top))]
        ax.barh(range(len(top)), top.importance, xerr=top["std"], color=cols, ecolor=INK2, capsize=2, height=0.65)
        ax.set_yticks(range(len(top)), [nice(f) for f in top.feature], fontsize=8.5)
        ax.set_xlabel(f"Permutation importance ({unit})")
        ax.set_title(title)
        ax.axvline(0, color=INK2, linewidth=0.8)
    _title(fig, f"Draft pick dominates who starts; {nice(imp_reg.iloc[0].feature)} leads for DL snap burst",
           "Permutation importance on the refit champion models (in-sample, descriptive), 20 shuffles each; bars = mean, lines = SD")
    _save(fig, cfg, "fig13_feature_importance.png", "registered champion models")


def fig_translation_gap(cfg, gap):
    g = gap[gap.population == "defensive_line"].dropna(subset=["translation_gap"]).sort_values("translation_gap")
    pick = pd.concat([g.head(6), g.tail(6)])
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    fig.subplots_adjust(left=0.3, right=0.95, top=0.85, bottom=0.1)
    cols = [ACCENT if v > 0 else ORANGE for v in pick.translation_gap]
    ax.barh(range(len(pick)), pick.translation_gap, color=cols, height=0.7, alpha=0.9)
    ax.set_ylim(-0.7, len(pick) + 0.2)
    ax.set_yticks(range(len(pick)), [f"{r.display_name} ({r.nfl_position}, {int(r.draft_year)}, {r.n_plays} plays)" for r in pick.itertuples()], fontsize=8.5)
    for i, r in enumerate(pick.itertuples()):
        ax.text(r.translation_gap + (0.6 if r.translation_gap > 0 else -0.6), i, f"{r.sbi:.0f} vs {r.predicted_sbi:.0f}",
                va="center", ha="left" if r.translation_gap > 0 else "right", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, linewidth=0.9)
    ax.set_xlabel("Translation gap = game SBI - Combine-predicted SBI (points)")
    lim = np.abs(pick.translation_gap).max() + 6
    ax.set_xlim(-lim, lim)
    ax.text(lim * 0.97, len(pick) - 0.4, "Plays faster than he tested", ha="right", fontsize=8.5, color=ACCENT, fontweight="bold")
    ax.text(-lim * 0.97, len(pick) - 0.4, "Tested faster than he plays", ha="left", fontsize=8.5, color=ORANGE, fontweight="bold")
    _title(fig, "Defensive linemen whose get-off departs most from what their testing predicted",
           "Six largest positive and negative translation gaps; labels = game SBI vs Combine-predicted SBI (out-of-fold)")
    _save(fig, cfg, "fig14_translation_gap.png", "nested-CV out-of-fold predictions of the champion regressor")


def make_all_figures(cfg: Config) -> None:
    _style()
    p, m, i = cfg.path("processed"), cfg.path("metrics"), cfg.path("interim")
    prospects = pd.read_parquet(p / "prospects.parquet")
    plays = pd.read_parquet(p / "play_metrics.parquet")
    profiles = pd.read_parquet(p / "player_movement_profiles.parquet")
    frames = pd.read_parquet(i / "tracking_frames.parquet")
    fig_drill_coverage(cfg, prospects)
    fig_drill_correlations(cfg, prospects)
    fig_burst_distribution(cfg, plays, pd.read_csv(m / "play_metric_anomalies.csv"))
    fig_speed_traces(cfg, frames, plays, profiles, prospects)
    fig_estimator_bias(cfg, pd.read_csv(m / "estimator_bias_simulation.csv"), pd.read_csv(m / "sbi_estimator_sensitivity.csv"))
    fig_reliability(cfg, pd.read_csv(m / "metric_reliability.csv"))
    fig_translation_matrix(cfg, pd.read_csv(m / "translation_matrix.csv"))
    fig_dl_role(cfg, pd.read_csv(m / "dl_translation_role_adjusted.csv"))
    fig_sbi_vs_start(cfg, profiles, prospects, pd.read_csv(m / "movement_vs_start_rate.csv"))
    if (m / "regression_summary.csv").exists():
        reg = pd.read_csv(m / "regression_summary.csv")
        fig_regression_cv(cfg, m, reg)
        dl = reg[(reg.population == "defensive_line") & (reg.model != "position_mean")]
        champ = dl.sort_values("rmse_mean").iloc[0].model
        fig_translation_gap(cfg, pd.read_csv(m / "translation_gap.csv"))
    if (m / "classification_summary.csv").exists():
        clf = pd.read_csv(m / "classification_summary.csv")
        fig_classification_auc(cfg, pd.read_csv(m / "classification_fold_scores.csv"))
        fig_roc(cfg, pd.read_parquet(m / "classification_oof.parquet"), clf)
        finals = json.loads((m / "classification_final_models.json").read_text())
        fig_importance(cfg, pd.read_csv(m / "classification_importance_all_prospects_combine_draft.csv"),
                       pd.read_csv(m / "regression_permutation_importance.csv"), finals, champ)
