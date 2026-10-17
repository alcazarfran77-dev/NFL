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
    "goe_eb": "Get-off over\nexpected (DL)", "proe_eb": "Pressure rate\nover exp. (DL)", "ppoe_eb": "Pressure prob.\nover exp. (OL)",
    "paoe_eb": "Pressures allowed\nover exp. (OL)", "soe_eb": "Separation\nover exp. (WR/TE)", "neg_log_pick": "Draft capital (earlier pick)",
    "goe_z": "Get-off over exp. (game)", "proe_z": "Pressure over exp. (game)", "ppoe_z": "Protection over exp. (game)",
    "soe_z": "Separation over exp. (game)", "proj_goe": "Projected get-off", "proj_proe": "Projected pressure",
    "proj_ppoe": "Projected protection", "proj_soe": "Projected separation",
    "onfield_target_goe": "Projected get-off", "onfield_target_proe": "Projected pressure",
    "onfield_target_ppoe": "Projected protection", "onfield_target_soe": "Projected separation",
    "combine_draft_proj": "Combine + draft\n+ projected on-field", "combine_draft_perf": "Combine + draft\n+ on-field performance",
    "combine": "Combine only", "combine_draft": "Combine + draft", "combine_draft_mov": "Combine + draft\n+ game movement",
    "combine_draft_ct": "Combine + draft\n+ drill tracking",
    "ct_forty_split_5_10_s": "40: 5-10 yd split (tracking)", "ct_forty_top_speed": "40: top speed (tracking)",
    "ct_shuttle_plant_speed_ratio": "Shuttle: speed kept in turns", "ct_skill_speed_z": "Position drills: top speed",
    "ct_skill_accel_z": "Position drills: acceleration",
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
    _save(fig, cfg, "fig03_burst_distribution.png", f"game_tracking, regular season + playoffs, {len(plays):,} player-plays")


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
    _save(fig, cfg, "fig04_speed_traces.png", "game_tracking, regular season + playoffs")


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
    _save(fig, cfg, "fig07_translation_matrix.png", f"combine_results + game_tracking, {int(matrix.n.max())} players with 3+ plays")


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
    b = {p: r.loc[(p, "burst_2s"), "icc"] for p in POS_ORDER}
    stable = [p for p in POS_ORDER if b[p] >= 0.5]
    weak = [p for p in POS_ORDER if b[p] < 0.5]
    _title(fig, f"Snap burst is a stable player trait for {' and '.join(f'{p} ({b[p]:.2f})' for p in stable)}; "
                f"weaker for {', '.join(weak)}" if stable else "Snap burst is not a stable player trait at any position",
           "Share of variance in player averages that is real player difference (ICC, method of moments); * = ANOVA p < 0.01; 'none' = not detectable")
    _save(fig, cfg, "fig06_metric_reliability.png", f"game_tracking, {sum(n)} players with 3+ plays")


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
    titles = {"all_positions": "All positions", "defensive_line": "Defensive line", "wide_receivers": "Wide receivers"}
    pops = [p for p in titles if p in set(summary.population)]
    fig, axes = plt.subplots(1, len(pops), figsize=(5.4 * len(pops), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.17, wspace=0.12)
    rng = np.random.default_rng(cfg.seed)
    best = {}
    for k, (ax, pop) in enumerate(zip(axes, pops)):
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
        ax.set_title(f"{chr(65 + k)}. {titles[pop]} (n = {int(s.n.iloc[0])})")
        ax.set_xlim(-0.5, len(order) - 0.2)
        ax.set_ylim(-0.8, 0.9)
    axes[0].set_ylabel("Outer-fold R² of the SBI (clipped at -0.8)")
    short = {"all_positions": "all positions", "defensive_line": "DL", "wide_receivers": "WRs"}
    cut = {p: 1 - best[p].rmse_mean / summary[(summary.population == p) & (summary.model == "position_mean")].rmse_mean.iloc[0]
           for p in pops}
    _title(fig, "Combine results cut snap-burst error vs position alone by " +
           ", ".join(f"{cut[p]:.0%} ({short[p]})" for p in sorted(pops, key=lambda p: -cut[p])),
           "Nested CV, 5 folds x 3 repeats; dots = outer folds, marker = mean with 95% CI; blue = lowest-RMSE learned model")
    _save(fig, cfg, "fig10_regression_cv.png", "Snap Burst Index from Combine measurables and drills")


EXPERIMENT_TITLES = {"all_prospects": "All prospects", "tracked_subset": "Players with tracking",
                     "onfield_subset": "Players with on-field metrics"}
EXPERIMENT_SETS = {"all_prospects": ["combine", "combine_draft", "combine_draft_proj", "combine_draft_ct"],
                   "tracked_subset": ["combine_draft", "combine_draft_mov"],
                   "onfield_subset": ["combine_draft", "combine_draft_perf"]}
SET_COLORS = {"combine": MUTED, "combine_draft": INK2, "combine_draft_proj": ORANGE, "combine_draft_mov": ACCENT,
              "combine_draft_perf": ACCENT, "combine_draft_ct": AQUA}


def _experiments(folds):
    return [(e, [f for f in EXPERIMENT_SETS[e] if f in set(folds[folds.population == e].feature_set)])
            for e in EXPERIMENT_SETS if e in set(folds.population)]


def fig_classification_auc(cfg, folds, summary):
    models = ["logistic_regression", "random_forest", "xgboost"]
    exps = _experiments(folds)
    fig, axes = plt.subplots(1, len(exps), figsize=(5.2 * len(exps), 4.9), sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.24, wspace=0.08)
    for k, (ax, (pop, fsets)) in enumerate(zip(axes, exps)):
        width = 0.7 / len(fsets)
        for j, fs in enumerate(fsets):
            for i, m in enumerate(models):
                v = folds[(folds.population == pop) & (folds.feature_set == fs) & (folds.model == m)].roc_auc.to_numpy()
                mu, half = _mean_ci(v)
                x = i + (j - (len(fsets) - 1) / 2) * width
                col = SET_COLORS[fs]
                ax.errorbar(x, mu, yerr=half, fmt="o", color=col, ecolor=col, capsize=3, markersize=6, elinewidth=1.6,
                            label=LABELS[fs].replace("\n", " ") if i == 0 else None)
                ax.text(x, min(mu + half + 0.01, 0.985), f"{mu:.2f}", ha="center", fontsize=7.5, color=INK)
        n = int(summary[summary.population == pop].n.iloc[0])
        ax.set_xticks(range(len(models)), [LABELS[m] for m in models], fontsize=8.5)
        ax.set_title(f"{chr(65 + k)}. {EXPERIMENT_TITLES[pop]} (n = {n})", fontsize=9.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7.5, ncol=1)
        ax.set_ylim(0.6, 1.0)
    axes[0].set_ylabel("ROC-AUC, outer folds (mean, 95% CI)")
    g = folds.groupby(["population", "feature_set"]).roc_auc.mean()
    t = f"Draft capital lifts AUC from {g[('all_prospects', 'combine')]:.2f} to {g[('all_prospects', 'combine_draft')]:.2f}"
    if ("all_prospects", "combine_draft_ct") in g.index:
        t += f"; drill tracking adds {g[('all_prospects', 'combine_draft_ct')] - g[('all_prospects', 'combine_draft')]:+.3f}"
    if ("onfield_subset", "combine_draft_perf") in g.index:
        t += f"; on-field performance {g[('onfield_subset', 'combine_draft_perf')] - g[('onfield_subset', 'combine_draft')]:+.3f}"
    _title(fig, t, "Predicting a starter (started 50%+ of available games); nested CV, 5 folds x 3 repeats; same folds within a panel")
    _save(fig, cfg, "fig11_classification_auc.png", "players + combine_results + combine_tracking + player_career_successes + player_play")


def fig_roc(cfg, oof, summary):
    exps = _experiments(summary)
    fig, axes = plt.subplots(1, len(exps), figsize=(4.9 * len(exps), 4.9))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.8, bottom=0.12, wspace=0.25)
    for k, (ax, (pop, fsets)) in enumerate(zip(axes, exps)):
        models = []
        for fs in fsets:
            s = summary[(summary.population == pop) & (summary.feature_set == fs)].sort_values("roc_auc_mean", ascending=False).iloc[0]
            o = oof[(oof.population == pop) & (oof.feature_set == fs) & (oof.model == s.model)]
            o = o.groupby("row").agg(y=("y", "first"), p=("pred", "mean"))
            fpr, tpr, _ = roc_curve(o.y, o.p)
            models.append(LABELS[s.model])
            ax.plot(fpr, tpr, color=SET_COLORS[fs], linewidth=1.8,
                    label=f"{LABELS[fs].replace(chr(10), ' ').replace('Combine + draft + ', '+ ')}: {s.oof_auc:.2f} [{s.oof_auc_ci_low:.2f}, {s.oof_auc_ci_high:.2f}]")
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate" if k == 0 else "")
        ax.set_title(f"{chr(65 + k)}. {EXPERIMENT_TITLES[pop]} ({' / '.join(dict.fromkeys(models))})", fontsize=9.5)
        ax.legend(loc="lower right", fontsize=7, title="ROC-AUC [95% CI]", title_fontsize=7)
    _title(fig, "Out-of-fold ROC curves of the best model per feature set",
           "Predictions averaged over the 3 CV repeats; brackets = bootstrap 95% CI (2,000 stratified resamples)")
    _save(fig, cfg, "fig12_roc_curves.png", "nested CV out-of-fold predictions")


def _nice(f):
    if f.endswith("_missing"):
        return f"{LABELS.get(f[:-8], f[:-8]).replace(chr(10), ' ')} not run"
    return LABELS.get(f, f).replace("\n", " ").replace("_", " ")


def fig_importance(cfg, imp_clf, imp_reg, champ_clf_model, reg_champ, reg_pop="dl_getoff", reg_label="DL get-off regressor",
                   unit="rise in RMSE (s)"):
    imp_reg = imp_reg[imp_reg.population == reg_pop]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.8, bottom=0.15, wspace=0.75)
    for ax, imp, title, u in [(a1, imp_clf, f"A. Starter classifier ({LABELS[champ_clf_model]})", "drop in ROC-AUC"),
                              (a2, imp_reg, f"B. {reg_label} ({LABELS[reg_champ]})", unit)]:
        top = imp.head(10).iloc[::-1]
        cols = [ACCENT if i == len(top) - 1 else MUTED for i in range(len(top))]
        ax.barh(range(len(top)), top.importance, xerr=top["std"], color=cols, ecolor=INK2, capsize=2, height=0.65)
        ax.set_yticks(range(len(top)), [_nice(f) for f in top.feature], fontsize=8.5)
        ax.set_xlabel(f"Permutation importance ({u})")
        ax.set_title(title)
        ax.axvline(0, color=INK2, linewidth=0.8)
    lead = _nice(imp_reg.iloc[0].feature)
    lead = lead[0].lower() + lead[1:] if len(lead) > 1 and lead[1].islower() else lead
    _title(fig, f"Draft pick dominates who starts; {lead} leads for {reg_label.replace(' regressor', '')}",
           "Permutation importance on the refit champion models (in-sample, descriptive), 20 shuffles each; bars = mean, lines = SD")
    _save(fig, cfg, "fig13_feature_importance.png", "registered champion models")


def fig_translation_gap(cfg, gap):
    g = gap[(gap.population == "defensive_line") & (gap.n_plays >= 10)].dropna(subset=["translation_gap"]).sort_values("translation_gap")
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
           "Six largest positive and negative gaps among linemen with 10+ tracked plays; labels = game SBI vs Combine-predicted SBI (out-of-fold)")
    _save(fig, cfg, "fig14_translation_gap.png", "nested-CV out-of-fold predictions of the champion regressor")


# ----------------------------------------------------------------------------- on-field (v2.0.0)
ONFIELD_NAMES = {"goe": "DL get-off over expected", "proe": "DL pressure rate over expected",
                 "ppoe": "OL pressure prob. over expected", "paoe": "OL pressures allowed over exp.",
                 "soe": "Separation over expected"}


def fig_onfield_reliability(cfg, rel, track_rel, stab):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=0.24, right=0.97, top=0.8, bottom=0.13, wspace=0.6)
    rows = [(f"{ONFIELD_NAMES[r.metric]} ({r.combine_position})" if r.metric == "soe" else ONFIELD_NAMES[r.metric],
             max(r.icc, 0), int(r.players), "on-field") for r in rel.itertuples()]
    tb = track_rel[track_rel.metric == "burst_2s"]
    rows += [(f"Tracking snap burst ({r.combine_position})", max(r.icc, 0), int(r.players), "tracking") for r in tb.itertuples()]
    rows.sort(key=lambda t: t[1])
    for i, (name, v, n, kind) in enumerate(rows):
        a1.barh(i, v, color=ACCENT if kind == "on-field" else MUTED, height=0.68)
        a1.text(v + 0.015, i, f"{v:.2f}  (n={n})", va="center", fontsize=8, color=INK)
    a1.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    a1.set_xlim(0, 1.12)
    a1.set_xlabel("ICC: share of variance that is real player difference")
    a1.set_title("A. Stable trait? Blue = on-field (season), grey = tracking SBI", fontsize=9.5)
    st = stab.sort_values("rho")
    a2.barh(range(len(st)), st.rho, color=[ACCENT if p < 0.05 else MUTED for p in st.p_value], height=0.6)
    for i, r in enumerate(st.itertuples()):
        a2.text(r.rho + 0.02, i, f"{r.rho:+.2f} (n={r.players})", va="center", fontsize=8, color=INK)
    a2.set_yticks(range(len(st)), [ONFIELD_NAMES[m].replace(" over expected", "").replace(" over exp.", "") for m in st.metric], fontsize=8.5)
    a2.set_xlim(min(0, st.rho.min()) - 0.05, 1.0)
    a2.axvline(0, color=INK2, linewidth=0.8)
    a2.set_xlabel("Spearman rho, rookie vs later seasons")
    a2.set_title("B. Rookie year vs later seasons", fontsize=9.5)
    g = rel.set_index(["metric", "combine_position"]).icc
    tr = [r.combine_position for r in tb.itertuples() if r.icc >= 0.5]
    _title(fig, f"Season-long on-field metrics are stable traits (DL get-off ICC {g[('goe', 'DL')]:.2f}); "
                f"2-s tracking burst is one for {' and '.join(tr) if tr else 'no position'}",
           "ICC by method of moments among players meeting the play minimum; panel B: 2023-2024 classes, blue = p < 0.05")
    _save(fig, cfg, "fig15_onfield_reliability.png", "player_play regular season 2023-2025 + game_tracking")


def fig_onfield_translation(cfg, mat):
    xs = ["ten_yd_split", "forty", "broad_jump", "vertical", "short_shuttle", "three_cone", "bench_reps", "combine_weight",
          "ngs_athleticism_score", "neg_log_pick"]
    ys = ["goe_eb", "proe_eb", "ppoe_eb", "paoe_eb", "soe_eb"]
    m = mat.set_index(["x", "y"])
    rho = np.array([[m.loc[(x, y), "rho"] if (x, y) in m.index else np.nan for y in ys] for x in xs])
    q = np.array([[m.loc[(x, y), "q_value"] if (x, y) in m.index else np.nan for y in ys] for x in xs])
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    fig.subplots_adjust(left=0.27, right=0.98, top=0.8, bottom=0.12)
    ax.imshow(rho, cmap=DIVERGING, vmin=-0.45, vmax=0.45, aspect="auto")
    for i in range(len(xs)):
        for j in range(len(ys)):
            if np.isfinite(rho[i, j]):
                sig = q[i, j] < 0.05
                ax.text(j, i, f"{rho[i, j]:+.2f}{'*' if sig else ''}", ha="center", va="center", fontsize=9,
                        color=INK, fontweight="bold" if sig else "normal")
    ax.axhline(len(xs) - 1.5, color=SURFACE, linewidth=4)
    ax.set_xticks(range(len(ys)), [LABELS[y] for y in ys], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(xs)), [LABELS[x] for x in xs], fontsize=9)
    ax.grid(False)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    g = mat[mat.y == "goe_eb"].sort_values("rho", ascending=False).iloc[0]
    _title(fig, f"Drills predict DL get-off ({LABELS[g.x].lower()} rho {g.rho:+.2f}); OL protection tracks draft capital",
           "Within-position Spearman rho, Combine result vs season-long on-field metric (* = FDR q < 0.05); times sign-flipped")
    fig.text(0.27, 0.05, "Blue = better test (or earlier pick) goes with better on-field performance; orange = the reverse. Body weight as measured.",
             fontsize=8, color=INK2)
    _save(fig, cfg, "fig16_onfield_translation.png", "combine_results + player_play regular season 2023-2025")


def fig_dl_chain(cfg, onfield, prospects, med):
    df = onfield.drop(columns=["combine_position"]).merge(prospects, on="nfl_id")
    df = df[(df.combine_position.astype(str) == "DL") & df.goe_eb.notna()]
    role = df.nfl_position.astype(str).isin(["DT", "NT"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.14, wspace=0.25)
    for ax, x, y, xl, yl in [(a1, "ten_yd_split", "goe_eb", "10-yd split (s)", "Get-off over expected (s; + = quicker)"),
                             (a2, "goe_eb", "proe_eb", "Get-off over expected (s; + = quicker)", "Pressure rate over expected")]:
        t = df[[x, y]].dropna()
        for mask, col, lab in [(~role, ACCENT, "Edge (DE/OLB)"), (role, INK2, "Interior (DT/NT)")]:
            tt = df[mask][[x, y]].dropna()
            ax.scatter(tt[x], tt[y], s=22, color=col, alpha=0.75, linewidths=0, label=lab)
        b, a = np.polyfit(t[x], t[y], 1)
        xs = np.linspace(t[x].min(), t[x].max(), 10)
        ax.plot(xs, a + b * xs, color=INK, linewidth=1.4)
        from scipy import stats
        r, p = stats.spearmanr(t[x], t[y])
        ax.text(0.02, 0.96, f"rho = {r:+.2f} (n = {len(t)})", transform=ax.transAxes, va="top", fontsize=9, color=INK)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.axhline(0, color=GRID, linewidth=0.8)
    a1.set_title("A. Faster 10-yd split, quicker get-off in games")
    a2.set_title("B. Quicker get-off, more pressure")
    a1.legend(loc="lower left", fontsize=8)
    r = med.set_index("x").loc["ten_yd_split"]
    _title(fig, f"About {r.share_mediated:.0%} of the 10-yd split's link to pressure runs through in-game get-off",
           f"Mediation on z-scores: indirect effect {abs(r.indirect_effect):.2f} SD [{abs(r.indirect_ci_high):.2f}, {abs(r.indirect_ci_low):.2f}] (bootstrap 95% CI); "
           "defensive linemen with 20+ pass-rush snaps" if r.indirect_effect < 0 else
           f"Mediation on z-scores: indirect effect {r.indirect_effect:.2f} SD [{r.indirect_ci_low:.2f}, {r.indirect_ci_high:.2f}] (bootstrap 95% CI); "
           "defensive linemen with 20+ pass-rush snaps")
    _save(fig, cfg, "fig17_dl_chain.png", "combine_results + player_play regular season 2023-2025")


def fig_onfield_regression(cfg, m_dir, summary, tasks):
    order = ["position_mean", "ridge", "random_forest", "xgboost"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.3 * len(tasks), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.76, bottom=0.2, wspace=0.08)
    rng = np.random.default_rng(cfg.seed)
    best = {}
    for k, (ax, task) in enumerate(zip(axes, tasks)):
        pop = task["name"]
        folds = pd.read_csv(m_dir / f"regression_fold_{pop}_r2.csv")
        s = summary[summary.population == pop]
        champ = s[s.model != "position_mean"].sort_values("rmse_mean").iloc[0].model
        best[pop] = s.set_index("model").loc[champ]
        for i, m in enumerate(order):
            v = folds[m].to_numpy()
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=9, color=MUTED, linewidths=0)
            mu, half = _mean_ci(v)
            col = ACCENT if m == champ else INK
            ax.errorbar(i + 0.25, mu, yerr=half, fmt="o", color=col, ecolor=col, capsize=3, markersize=5, elinewidth=1.4)
            ax.text(i + 0.33, float(np.clip(mu, -0.76, 0.76)), f"{mu:+.2f}", va="center", fontsize=7.5, color=col)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks(range(len(order)), [LABELS[m].replace("\n(baseline)", "") for m in order], fontsize=7.5, rotation=30, ha="right")
        ax.set_title(f"{chr(65 + k)}. {task['label']}\n(n = {int(s.n.iloc[0])})", fontsize=8.5)
        ax.set_xlim(-0.5, len(order) - 0.1)
        ax.set_ylim(-0.8, 0.8)
    axes[0].set_ylabel("Outer-fold R² (clipped at -0.8)")
    short = {"dl_getoff": "DL get-off", "dl_pressure": "DL pressure", "ol_protection": "OL protection",
             "wrte_separation": "WR/TE separation"}
    yes = [f"{short.get(k, k)} (R² {v.r2_mean:+.2f})" for k, v in best.items() if v.r2_mean > 0.05]
    no = [short.get(k, k) for k, v in best.items() if v.r2_mean <= 0.05]
    _title(fig, f"Combine testing predicts {' and '.join(yes)}; not {' or '.join(no)}" if yes and no else
           "Combine testing vs on-field performance by family",
           "Nested CV, 5 folds x 3 repeats; dots = outer folds, marker = mean with 95% CI; blue = lowest-RMSE learned model")
    _save(fig, cfg, "fig18_onfield_regression.png", "Combine -> season-long on-field performance (player_play)")


# ----------------------------------------------------------------------------- Combine drill tracking (v2.2.0)
CT_GROUP_TITLES = {"forty": "40-yard dash", "3cone": "3-cone", "shuttle": "Short shuttle", "skill": "Position drills"}
CT_METRIC_NAMES = {"split_0_5_s": "0-5 yd split", "split_5_10_s": "5-10 yd split", "split_10_20_s": "10-20 yd split",
                   "split_10yd_s": "10-yd split from the start", "top_speed": "Top speed", "peak_accel": "Peak acceleration",
                   "burst_1s": "Distance at 1 s", "burst_2s": "Distance at 2 s", "t90_top_s": "Time to 90% of top speed",
                   "duration_s": "Drill time", "peak_decel": "Peak braking", "plant_speed_ratio": "Speed kept in turns",
                   "max_cut_deg": "Sharpest cut"}


def fig_ct_reliability(cfg, rel):
    """Which drill-tracking metrics repeat within a player (the gate that picks the features)."""
    order = ["forty", "skill", "shuttle", "3cone"]
    rel = rel.copy()
    rel["g"] = rel.drill_group.map({g: i for i, g in enumerate(order)})
    rel = rel.sort_values(["g", "reliability"], ascending=[True, True])
    rel["feature"] = rel["feature"].fillna("").astype(str)
    fig, ax = plt.subplots(figsize=(10.5, 8.6))
    fig.subplots_adjust(left=0.33, right=0.95, top=0.83, bottom=0.08)
    y, ticks, labels, seps = 0, [], [], []
    for g in order[::-1]:
        sub = rel[rel.drill_group == g]
        for r in sub.itertuples():
            col = ACCENT if r.feature else (INK2 if bool(r.passes_gate) else MUTED)
            ax.barh(y, max(r.reliability, 0), color=col, height=0.7)
            lab = f"{r.reliability:.3f}" if abs(r.reliability - 0.5) < 0.005 else f"{r.reliability:.2f}"
            ax.text(max(r.reliability, 0) + 0.015, y, lab, va="center", fontsize=8, color=INK)
            ticks.append(y)
            labels.append(f"{CT_METRIC_NAMES.get(r.metric, r.metric)}")
            y += 1
        ax.text(-0.02, y - 0.35, f"{CT_GROUP_TITLES[g]} (n = {int(sub.n_players.max())} players)", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8.5, fontweight="bold", color=INK)
        seps.append(y - 0.5)
        y += 1.2
    ax.axvline(0.5, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.505, y - 0.6, "gate r = 0.5", fontsize=8, color=INK2, va="top")
    ax.set_yticks(ticks, labels, fontsize=8.5)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.8, y - 0.4)
    ax.set_xlabel("Reliability (test-retest r; split-half with Spearman-Brown for position drills)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=ACCENT, label="Model feature"), Patch(color=INK2, label="Repeats, but another metric covers it"),
                        Patch(color=MUTED, label="Below the gate: not used")], loc="upper left", bbox_to_anchor=(0.33, 0.885),
               ncol=3, fontsize=8.5, frameon=False)
    f = rel.set_index(["drill_group", "metric"])
    _title(fig, f"Splits between distance marks repeat (5-10 yd r = {f.loc[('forty', 'split_5_10_s'), 'reliability']:.2f}); "
                f"the tracked start does not (0-5 yd r = {f.loc[('forty', 'split_0_5_s'), 'reliability']:.2f})",
           "How well each drill-tracking metric repeats within a player; a metric at r >= 0.5 can become a feature, one per construct")
    _save(fig, cfg, "fig19_ct_reliability.png", "combine_tracking_6.csv: 6,310 drill attempts, 510 prospects")


def _ci_from_t(mean_diff, t, df, level=0.95):
    from scipy import stats
    se = abs(mean_diff / t) if t and np.isfinite(t) and t != 0 else np.nan
    return se * stats.t.ppf(0.5 + level / 2, df)


def fig_ct_value(cfg, m_dir, clf_tests, n_folds):
    """Does drill tracking improve out-of-sample prediction? Same rows and folds, with vs without."""
    bvb = pd.read_csv(m_dir / "ct_best_vs_best.csv")
    names = {"defensive_line": "DL snap burst (SBI)", "wide_receivers": "WR snap burst (SBI)", "dl_getoff": "DL get-off over expected",
             "dl_pressure": "DL pressure over expected", "ol_protection": "OL protection over expected",
             "wrte_separation": "WR/TE separation over expected"}
    r1 = pd.DataFrame([{"label": f"{names[r.task]}\n{LABELS[r.model_without]} -> {LABELS[r.model_with]}",
                        "d": r.r2_diff, "lo": r.r2_diff_ci_low, "hi": r.r2_diff_ci_high, "p": r.r2_p_corrected_t,
                        "base": r.r2_without, "new": r.r2_with, "wins": r.r2_folds_better, "p_w": r.r2_p_wilcoxon}
                       for r in bvb[bvb.kind == "regression"].itertuples() if r.task in names])
    t = clf_tests[(clf_tests.comparison == "feature_set_vs_feature_set") & clf_tests.feature_set.str.startswith("combine_draft_ct vs combine_draft")]
    rows = []
    for m in ["logistic_regression", "random_forest", "xgboost"]:
        c = t[t.feature_set.str.contains(f"({m})", regex=False)].iloc[0]
        half = _ci_from_t(c.mean_diff_a_minus_b, c.t_corrected, n_folds - 1)
        rows.append({"label": LABELS[m], "d": c.mean_diff_a_minus_b, "lo": c.mean_diff_a_minus_b - half,
                     "hi": c.mean_diff_a_minus_b + half, "p": c.p_corrected_t})
    cb = bvb[bvb.kind == "classification"]
    if len(cb):
        c = cb.iloc[0]
        rows.append({"label": f"Best vs best\n{LABELS[c.model_without]} -> {LABELS[c.model_with]}", "d": c.roc_auc_diff,
                     "lo": c.roc_auc_diff_ci_low, "hi": c.roc_auc_diff_ci_high, "p": c.roc_auc_p_corrected_t})
    r2 = pd.DataFrame(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=0.2, right=0.97, top=0.78, bottom=0.14, wspace=0.5)
    for ax, r, xl, title, fmt in [(a1, r1, "Change in outer-fold R² (with - without drill tracking)",
                                   "A. Combine -> game metrics (best model in each arm)", "{:+.3f}"),
                                  (a2, r2, "Change in outer-fold ROC-AUC", "B. Starter classifier, Combine + draft", "{:+.3f}")]:
        yy = np.arange(len(r))[::-1]
        for yi, row in zip(yy, r.itertuples()):
            sig = row.p < 0.05
            col = ACCENT if sig and row.d > 0 else (ORANGE if sig else INK2)
            ax.errorbar(row.d, yi, xerr=[[row.d - row.lo], [row.hi - row.d]], fmt="o", color=col, ecolor=col, capsize=3,
                        markersize=7, elinewidth=1.6)
            extra = f"  R² {row.base:+.2f} -> {row.new:+.2f}" if hasattr(row, "base") else ""
            ax.text(row.d, yi + 0.24, fmt.format(row.d) + f" (p = {row.p:.2f})" + extra, ha="center", fontsize=7.6, color=INK)
        ax.axvline(0, color=INK2, linewidth=0.9)
        ax.set_yticks(yy, r.label, fontsize=8)
        ax.set_ylim(-0.6, len(r) - 0.25)
        ax.set_xlabel(xl, fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        lim = float(np.nanmax(np.abs(np.r_[r.lo.to_numpy(), r.hi.to_numpy()]))) * 1.1
        ax.set_xlim(-lim, lim)
        ax.grid(axis="y", visible=False)
    up = [names[r.task] for r in bvb[bvb.kind == "regression"].itertuples() if r.task in names and r.r2_p_corrected_t < 0.05 and r.r2_diff > 0]
    best_up = r1.sort_values("d").iloc[-1]
    cbest = r2.iloc[-1]
    if up:
        head = f"Drill tracking improves {', '.join(up)}; nothing else improves"
    elif best_up.d > 0.01:
        head = (f"Drill tracking lifts {best_up.label.split(chr(10))[0].replace(' over expected', '')} R² "
                f"{best_up.base:+.2f} -> {best_up.new:+.2f} ({int(best_up.wins)}/{n_folds} folds, p = {best_up.p:.2f}); nothing else improves")
    else:
        head = (f"Drill tracking improves no model: R² change {r1.d.min():+.3f} to {r1.d.max():+.3f}, "
                f"starter AUC {cbest.d:+.3f}")
    _title(fig, head,
           f"Same players and folds with and without the 5 drill-tracking features; mean over {n_folds} outer folds, "
           "95% CI from the corrected resampled t-test; blue = significant gain")
    _save(fig, cfg, "fig20_ct_value.png", "nested CV fold scores: ct_best_vs_best.csv, classification_pairwise_tests.csv")


def fig_ct_translation(cfg, mat):
    from .combine_tracking import CT_FEATURES
    ys = ["goe_eb", "proe_eb", "ppoe_eb", "paoe_eb", "soe_eb"]
    m = mat.set_index(["x", "y"])
    rho = np.array([[m.loc[(x, y), "rho"] if (x, y) in m.index else np.nan for y in ys] for x in CT_FEATURES])
    q = np.array([[m.loc[(x, y), "q_value"] if (x, y) in m.index else np.nan for y in ys] for x in CT_FEATURES])
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    fig.subplots_adjust(left=0.27, right=0.98, top=0.7, bottom=0.14)
    ax.imshow(rho, cmap=DIVERGING, vmin=-0.45, vmax=0.45, aspect="auto")
    for i in range(len(CT_FEATURES)):
        for j in range(len(ys)):
            if np.isfinite(rho[i, j]):
                sig = q[i, j] < 0.05
                ax.text(j, i, f"{rho[i, j]:+.2f}{'*' if sig else ''}", ha="center", va="center", fontsize=9,
                        color=INK, fontweight="bold" if sig else "normal")
    ax.set_xticks(range(len(ys)), [LABELS[y] for y in ys], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(CT_FEATURES)), [LABELS[x] for x in CT_FEATURES], fontsize=9)
    ax.grid(False)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    short_x = {"ct_forty_split_5_10_s": "40 acceleration (5-10 yd)", "ct_forty_top_speed": "40 top speed",
               "ct_shuttle_plant_speed_ratio": "shuttle turns", "ct_skill_speed_z": "position-drill top speed",
               "ct_skill_accel_z": "position-drill acceleration"}
    short_y = {"goe_eb": "DL get-off", "proe_eb": "DL pressure", "ppoe_eb": "OL protection", "paoe_eb": "OL pressures allowed",
               "soe_eb": "WR/TE separation"}
    sig = mat[(mat.q_value < 0.05) & mat.x.isin(CT_FEATURES) & mat.y.isin(ys)].sort_values("rho", key=np.abs, ascending=False)
    if len(sig):
        g = sig.iloc[0]
        t = f"{short_x[g.x][0].upper() + short_x[g.x][1:]} links to {short_y[g.y]} (rho {g.rho:+.2f})"
        t += f"; {len(sig) - 1} other cell{'s' if len(sig) != 2 else ''} pass FDR" if len(sig) > 1 else "; no other cell passes FDR"
    else:
        t = "No drill-tracking feature links to on-field performance after FDR correction"
    _title(fig, t, "Within-position Spearman rho, drill-tracking feature vs season-long on-field metric (* = FDR q < 0.05); split time sign-flipped")
    fig.text(0.27, 0.04, "Blue = better drill movement goes with better on-field performance; orange = the reverse.", fontsize=8, color=INK2)
    _save(fig, cfg, "fig21_ct_onfield_translation.png", "combine_tracking_6.csv + player_play regular season 2023-2025")


def make_all_figures(cfg: Config) -> None:
    _style()
    p, m, i = cfg.path("processed"), cfg.path("metrics"), cfg.path("interim")
    prospects = pd.read_parquet(p / "prospects.parquet")
    plays = pd.read_parquet(p / "play_metrics.parquet")
    profiles = pd.read_parquet(p / "player_movement_profiles.parquet")
    # Only the frames the speed-trace figure can use (DL players), so spec-scale tracking stays out of memory.
    dl_ids = profiles.loc[profiles.combine_position == "DL", "nfl_id"].astype(int).tolist()
    frames = pd.read_parquet(i / "tracking_frames.parquet", columns=["game_id", "play_id", "nfl_id", "time", "s", "event"],
                             filters=[("nfl_id", "in", dl_ids)])
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
    onfield_ready = (p / "player_onfield_profiles.parquet").exists() and (m / "onfield_reliability.csv").exists()
    if onfield_ready:
        onfield = pd.read_parquet(p / "player_onfield_profiles.parquet")
        fig_onfield_reliability(cfg, pd.read_csv(m / "onfield_reliability.csv"), pd.read_csv(m / "metric_reliability.csv"),
                                pd.read_csv(m / "onfield_stability.csv"))
        fig_onfield_translation(cfg, pd.read_csv(m / "onfield_translation_matrix.csv"))
        fig_dl_chain(cfg, onfield, prospects, pd.read_csv(m / "onfield_mediation_dl.csv"))
    if (m / "regression_summary.csv").exists() and onfield_ready:
        reg = pd.read_csv(m / "regression_summary.csv")
        tasks = [t for t in cfg["regression"].get("onfield_tasks", []) if t["name"] in set(reg.population)]
        if tasks:
            fig_onfield_regression(cfg, m, reg, tasks)
    if (m / "ct_reliability.csv").exists():
        fig_ct_reliability(cfg, pd.read_csv(m / "ct_reliability.csv"))
    if (m / "ct_onfield_translation_matrix.csv").exists():
        fig_ct_translation(cfg, pd.read_csv(m / "ct_onfield_translation_matrix.csv"))
    if (m / "ct_best_vs_best.csv").exists() and (m / "classification_pairwise_tests.csv").exists():
        tests = pd.read_csv(m / "classification_pairwise_tests.csv")
        if tests.feature_set.astype(str).str.startswith("combine_draft_ct vs").any():
            fig_ct_value(cfg, m, tests, cfg["cv"]["outer_splits"] * cfg["cv"]["outer_repeats"])
    if (m / "classification_summary.csv").exists():
        clf = pd.read_csv(m / "classification_summary.csv")
        fig_classification_auc(cfg, pd.read_csv(m / "classification_fold_scores.csv"), clf)
        fig_roc(cfg, pd.read_parquet(m / "classification_oof.parquet"), clf)
        finals = json.loads((m / "classification_final_models.json").read_text())
        champ_entry = next(v for k, v in finals.items() if k.startswith("champion"))
        reg = pd.read_csv(m / "regression_summary.csv")
        if "dl_getoff" in set(reg.population):
            pop, label, unit = "dl_getoff", "DL get-off regressor", "rise in RMSE (s)"
        else:
            pop, label, unit = "defensive_line", "DL snap-burst regressor", "rise in RMSE (SBI pts)"
        rc = reg[(reg.population == pop) & (reg.model != "position_mean")].sort_values("rmse_mean").iloc[0].model
        fig_importance(cfg, pd.read_csv(m / f"classification_importance_all_prospects_{champ_entry['feature_set']}.csv"),
                       pd.read_csv(m / "regression_permutation_importance.csv"), champ_entry["model"], rc, pop, label, unit)
