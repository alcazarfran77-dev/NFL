"""End-to-end pipeline: python -m cgmt.pipeline [--stage all|data|models|figures]

Stages
  1 ingest       validate raw CSVs, write the data manifest (hashes)
  2 clean        prospect table + business rules + audit log
  3 movement     tracking -> per-play metrics -> per-player profiles + SBI
  4 translation  drill-vs-game associations, starter association, logistic GLM
  5 regression   Combine -> Snap Burst (position mean vs Ridge vs RF vs XGBoost)
  6 classify     Starter (LogReg vs RF vs XGBoost) x nested feature sets
  7 figures      EDA, anomaly and results figures
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sst
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from . import __version__, mlops
from .cleaning import build_prospects, missingness_by_position
from .config import Config, load_config, set_global_seed
from .data import load_raw, write_manifest
from .features import FEATURE_SETS, ATHLETIC, classification_table, regression_table, xy
from .models import nested_cv, refit_final
from .movement import METRICS, clean_tracking, estimator_bias, metric_reliability, play_metrics, player_profiles
from .stats import bootstrap_ci, delong_test, friedman, pairwise_model_tests, within_group_spearman

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger("cgmt")

TIME_DRILLS = ["ten_yd_split", "forty", "three_cone", "short_shuttle"]
TRANSLATION_X = ["ten_yd_split", "forty", "vertical", "broad_jump", "three_cone", "short_shuttle",
                 "bench_reps", "combine_weight", "ngs_athleticism_score"]
TRANSLATION_Y = [f"{m}_eb" for m in ["burst_2s", "get_off_1s", "peak_accel_2s", "top_speed", "max_decel", "max_cut_deg"]]


def _save(df: pd.DataFrame, cfg: Config, name: str, key: str = "processed") -> None:
    path = cfg.path(key) / name
    if name.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _dump(obj, cfg: Config, name: str) -> None:
    with open(cfg.path("metrics") / name, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


# ----------------------------------------------------------------------------- 1-3 data
def stage_data(cfg: Config) -> dict:
    t0 = time.time()
    tables = load_raw(cfg)
    manifest = write_manifest(cfg)
    log.info("ingest ok: data_version=%s", manifest["data_version"])

    prospects, audit = build_prospects(tables, cfg)
    _save(prospects, cfg, "prospects.parquet")
    _save(audit, cfg, "cleaning_audit.csv", key="metrics")
    miss = missingness_by_position(prospects)
    miss.to_csv(cfg.path("metrics") / "drill_missingness_by_position.csv")

    frames, frame_audit = clean_tracking(tables["tracking"], cfg)
    plays, play_audit = play_metrics(frames, cfg)
    plays = plays.merge(prospects[["nfl_id", "combine_position"]], on="nfl_id", how="left")
    _save(plays, cfg, "play_metrics.parquet")
    frames.to_parquet(cfg.path("interim") / "tracking_frames.parquet", index=False)

    profiles = player_profiles(plays, prospects, cfg)
    _save(profiles, cfg, "player_movement_profiles.parquet")

    # Anomaly scan on per-play metrics: robust z (median/MAD) within position group.
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

    audit_all = {**frame_audit, **play_audit, "cleaning_rule_changes": len(audit),
                 "players_tracked": int(plays.nfl_id.nunique()), "players_profiled": len(profiles),
                 "play_metric_anomalies": len(anomalies), "seconds": round(time.time() - t0, 1)}
    _dump(audit_all, cfg, "data_audit.json")
    log.info("data stage: %s", audit_all)
    return {"prospects": prospects, "plays": plays, "profiles": profiles, "frames": frames}


def load_processed(cfg: Config) -> dict:
    p = cfg.path("processed")
    return {
        "prospects": pd.read_parquet(p / "prospects.parquet"),
        "plays": pd.read_parquet(p / "play_metrics.parquet"),
        "profiles": pd.read_parquet(p / "player_movement_profiles.parquet"),
    }


# ----------------------------------------------------------------------------- 4 translation
def stage_translation(cfg: Config, d: dict) -> dict:
    prospects, profiles = d["prospects"], d["profiles"]
    df = profiles.merge(prospects, on=["nfl_id", "combine_position"], how="inner")
    oriented = df.copy()
    for c in TIME_DRILLS:  # orient so that a positive rho always means "better test -> more movement"
        oriented[c] = -oriented[c]
    nb = cfg["cv"]["bootstrap_resamples"]
    matrix = within_group_spearman(oriented, TRANSLATION_X, TRANSLATION_Y + ["start_rate"], "combine_position", nb, cfg.seed)
    matrix["x_oriented"] = matrix["x"].isin(TIME_DRILLS)
    _save(matrix, cfg, "translation_matrix.csv", key="metrics")

    # Reliability of the SBI: mean empirical-Bayes weight. It caps how much variance any
    # pre-draft model can explain (observed R^2 <= reliability x true R^2).
    rel = profiles.groupby("combine_position", observed=True)["sbi_reliability"].agg(["mean", "median", "min", "max", "count"])
    rel.loc["ALL"] = [profiles.sbi_reliability.mean(), profiles.sbi_reliability.median(),
                      profiles.sbi_reliability.min(), profiles.sbi_reliability.max(), len(profiles)]
    _save(rel.reset_index(names="combine_position"), cfg, "sbi_reliability_by_position.csv", key="metrics")

    # Which game metrics are stable player traits, by position? (ICC + one-way ANOVA)
    reliab = metric_reliability(d["plays"], profiles)
    _save(reliab, cfg, "metric_reliability.csv", key="metrics")

    # Defensive line deep dive: raw vs role-adjusted (edge DE/OLB vs interior DT/NT) translation.
    dl = oriented[oriented.combine_position == "DL"].copy()
    dl["dl_role"] = np.where(dl["nfl_position"].isin(["DT", "NT"]), "interior", "edge")
    dl_y = ["burst_2s_eb", "get_off_1s_eb", "peak_accel_2s_eb"]
    dl_tab = pd.concat([
        within_group_spearman(dl, TRANSLATION_X, dl_y, "combine_position", nb, cfg.seed).assign(adjustment="none"),
        within_group_spearman(dl, TRANSLATION_X, dl_y, "dl_role", nb, cfg.seed).assign(adjustment="edge vs interior role"),
    ], ignore_index=True)
    _save(dl_tab, cfg, "dl_translation_role_adjusted.csv", key="metrics")

    # Game movement vs starting, within position group
    starter_assoc = within_group_spearman(df, ["sbi"] + [f"{m}_z" for m in METRICS if m != "burst_2s"],
                                          ["start_rate"], "combine_position", nb, cfg.seed)
    _save(starter_assoc, cfg, "movement_vs_start_rate.csv", key="metrics")

    # Sensitivity: the rejected 80th-percentile capacity vs the adopted shrunk mean
    gq = df.groupby("combine_position", observed=True)["burst_2s_q80"]
    sens_df = df.assign(sbi_q80=50 + 10 * (df["burst_2s_q80"] - gq.transform("mean")) / gq.transform("std"),
                        neg_log_pick=-df["log_draft_pick"])
    sens = within_group_spearman(sens_df, ["sbi", "sbi_q80"], ["n_plays", "start_rate", "neg_log_pick"],
                                 "combine_position", nb, cfg.seed)
    _save(sens, cfg, "sbi_estimator_sensitivity.csv", key="metrics")
    bias = estimator_bias(d["plays"], "burst_2s", [3, 5, 10], 200, cfg.seed)
    _save(bias, cfg, "estimator_bias_simulation.csv", key="metrics")

    # Exploratory (not pre-registered): SBI vs start rate where draft capital says little (Day 3 + undrafted)
    day3 = df[df["draft_round_filled"] >= 4]
    day12 = df[df["draft_round_filled"] < 4]
    explo = pd.concat([within_group_spearman(day3, ["sbi"], ["start_rate"], "combine_position", nb, cfg.seed).assign(subgroup="rounds 4-7 + undrafted"),
                       within_group_spearman(day12, ["sbi"], ["start_rate"], "combine_position", nb, cfg.seed).assign(subgroup="rounds 1-3")],
                      ignore_index=True)
    explo["starters"] = [int(day3.is_starter.sum()), int(day12.is_starter.sum())]
    _save(explo, cfg, "exploratory_sbi_by_draft_tier.csv", key="metrics")

    # Starter rate by SBI band (Wilson 95% CI) - the coach-facing table
    bands = pd.cut(df["sbi"], [-np.inf, 45, 55, np.inf], labels=["Below 45", "45-55", "Above 55"])
    tab = df.groupby(bands, observed=True)["is_starter"].agg(["sum", "count"]).reset_index(names="sbi_band")
    z = 1.96
    ph = tab["sum"] / tab["count"]
    denom = 1 + z ** 2 / tab["count"]
    centre = (ph + z ** 2 / (2 * tab["count"])) / denom
    half = z * np.sqrt(ph * (1 - ph) / tab["count"] + z ** 2 / (4 * tab["count"] ** 2)) / denom
    tab["starter_rate"], tab["ci_low"], tab["ci_high"] = ph, centre - half, centre + half
    _save(tab, cfg, "starter_rate_by_sbi_band.csv", key="metrics")

    # Logistic GLM: does SBI add to position + draft capital + Combine athleticism? (LRT)
    # `is_undrafted` is left out: no undrafted tracked player started (quasi-complete
    # separation, the MLE does not converge); log pick already places them at pick 260.
    g = df.assign(sbi10=df["sbi"] / 10, athl10=df["ngs_athleticism_score"] / 10, pos=df["combine_position"].astype(str))
    base = smf.logit("is_starter ~ C(pos) + log_draft_pick + athl10", data=g).fit(disp=0, maxiter=200)
    full = smf.logit("is_starter ~ C(pos) + log_draft_pick + athl10 + sbi10", data=g).fit(disp=0, maxiter=200)
    lr = 2 * (full.llf - base.llf)
    ci = np.exp(full.conf_int().loc["sbi10"])
    ci_a = np.exp(full.conf_int().loc["athl10"])
    glm = {
        "n": int(full.nobs), "events": int(g["is_starter"].sum()), "converged": bool(full.mle_retvals["converged"]),
        "or_per_10_sbi": float(np.exp(full.params["sbi10"])), "or_sbi_ci": [float(ci.iloc[0]), float(ci.iloc[1])],
        "p_sbi_wald": float(full.pvalues["sbi10"]),
        "or_per_10_ngs_athleticism": float(np.exp(full.params["athl10"])), "or_athl_ci": [float(ci_a.iloc[0]), float(ci_a.iloc[1])],
        "p_athl_wald": float(full.pvalues["athl10"]),
        "lr_stat": float(lr), "lr_p_value": float(sst.chi2.sf(lr, 1)),
        "aic_base": float(base.aic), "aic_full": float(full.aic),
        "pseudo_r2_base": float(base.prsquared), "pseudo_r2_full": float(full.prsquared),
    }
    _dump(glm, cfg, "glm_sbi_starter.json")
    (cfg.path("metrics") / "glm_sbi_starter_summary.txt").write_text(str(full.summary()))
    log.info("translation: GLM %s", glm)
    return {"matrix": matrix, "starter_assoc": starter_assoc, "bands": tab, "glm": glm}


# ----------------------------------------------------------------------------- 5 regression
def stage_regression(cfg: Config, d: dict) -> dict:
    """Combine -> Snap Burst Index, for all positions and for the defensive line only.

    The defensive line is the one group where 2-second snap burst is a reliable
    player trait (see sbi_reliability / metric_reliability), so it is where a
    Combine-to-game translation model has a measurable target.
    """
    target = cfg["regression"]["target"]
    grids = cfg["regression"]["grids"]
    models = ["position_mean", "ridge", "random_forest", "xgboost"]
    full = regression_table(d["prospects"], d["profiles"], target)
    full["dl_role"] = np.where(full["nfl_position"].isin(["DT", "NT"]), "interior", "edge")
    out = {"summary": [], "tests": [], "friedman": {}, "gap": [], "importance": [], "champions": {}}
    for pop in cfg["regression"]["populations"]:
        df = (full if pop == "all_positions" else full[full.combine_position == "DL"]).reset_index(drop=True)
        X, y = xy(df, ATHLETIC, target)
        # CV strata are used only to balance the splits (never as a feature).
        strata = df["combine_position"].astype(str) if pop == "all_positions" else df["dl_role"]
        results = {}
        for name in models:
            t0 = time.time()
            res = nested_cv(name, "athletic", "regression", X, y, strata, grids.get(name, {}), cfg, cfg.seed)
            with mlops.run(cfg, f"reg-{pop}-{name}", {"stage": "cv", "task": "regression", "target": target, "population": pop}):
                mlops.log_cv_result(res, {"grid": json.dumps(grids.get(name, {})), "population": pop, "n_rows": len(df)})
            results[name] = res
            log.info("regression %s/%s: %s (%.0fs)", pop, name, res.fold_scores[["mae", "rmse", "r2"]].mean().round(3).to_dict(), time.time() - t0)

        summary, scores = _summarise(results, ["mae", "rmse", "r2"], cfg, task="regression")
        summary.insert(0, "population", pop)
        summary.insert(1, "n", len(df))
        any_res = next(iter(results.values()))
        tests = pd.concat([pairwise_model_tests(scores[m], m, any_res.n_train, any_res.n_test, higher_is_better=(m == "r2"))
                           for m in ["rmse", "r2"]], ignore_index=True)
        tests.insert(0, "population", pop)
        out["friedman"][pop] = {m: friedman(scores[m], higher_is_better=(m == "r2")) for m in ["rmse", "r2"]}
        for m, s in scores.items():
            _save(s.reset_index(names="fold"), cfg, f"regression_fold_{pop}_{m}.csv", key="metrics")

        # Champion = lowest mean RMSE among learned models; refit on all rows and register.
        champ = summary[summary.model != "position_mean"].sort_values("rmse_mean").iloc[0]["model"]
        est, best, score = refit_final(champ, "regression", X, y, grids.get(champ, {}), cfg, cfg.seed)
        reg_name = cfg["mlflow"]["regressor_name"] + ("" if pop == "all_positions" else "-dl")
        with mlops.run(cfg, f"reg-final-{pop}-{champ}", {"stage": "final", "task": "regression", "population": pop}):
            import mlflow
            mlflow.log_params({"model": champ, "population": pop, **{k.replace("model__", ""): v for k, v in best.items()}})
            mlflow.log_metric("refit_inner_cv_rmse", -score)
            mlops.log_and_register(est, X, "regression", reg_name if cfg["mlflow"]["register_models"] else None, "champion")
        out["champions"][pop] = {"model": champ, "params": best, "inner_rmse": -score}

        # Translation gap: game SBI minus Combine-predicted SBI (out-of-fold, mean over repeats).
        # Positive = "plays faster off the snap than he tested"; negative = the reverse.
        oof = results[champ].oof.groupby("row")["pred"].mean()
        cols = list(dict.fromkeys(["nfl_id", "display_name", "combine_position", "nfl_position", "dl_role", "draft_year",
                                   "draft_round", "burst_2s_eb", target, "sbi_reliability", "n_plays", "start_rate", "is_starter"]))
        gap = df[cols].copy()
        gap["predicted_sbi"] = oof.reindex(gap.index).to_numpy()
        gap["translation_gap"] = gap[target] - gap["predicted_sbi"]
        out["gap"].append(gap.assign(population=pop))

        # Permutation importance of the champion (on the rows it was refit on; descriptive)
        pi = permutation_importance(est, X, y, scoring="neg_root_mean_squared_error", n_repeats=20, random_state=cfg.seed)
        out["importance"].append(pd.DataFrame({"population": pop, "feature": X.columns, "importance": pi.importances_mean,
                                               "std": pi.importances_std}).sort_values("importance", ascending=False))
        out["summary"].append(summary)
        out["tests"].append(tests)

    summary = pd.concat(out["summary"], ignore_index=True)
    _save(summary, cfg, "regression_summary.csv", key="metrics")
    _save(pd.concat(out["tests"], ignore_index=True), cfg, "regression_pairwise_tests.csv", key="metrics")
    _dump(out["friedman"], cfg, "regression_friedman.json")
    _dump(out["champions"], cfg, "regression_final_models.json")
    gap = pd.concat(out["gap"], ignore_index=True).sort_values(["population", "translation_gap"], ascending=[True, False])
    _save(gap, cfg, "translation_gap.csv", key="metrics")
    _save(pd.concat(out["importance"], ignore_index=True), cfg, "regression_permutation_importance.csv", key="metrics")
    return {"summary": summary, "champions": out["champions"], "gap": gap}


# ----------------------------------------------------------------------------- 6 classification
def stage_classification(cfg: Config, d: dict) -> dict:
    target = cfg["classification"]["target"]
    grids = cfg["classification"]["grids"]
    thr = cfg["classification"]["decision_threshold"]
    models = ["logistic_regression", "random_forest", "xgboost"]
    experiments = {
        "all_prospects": (classification_table(d["prospects"]), ["combine", "combine_draft"]),
        "tracked_subset": (classification_table(d["prospects"], d["profiles"]), ["combine_draft", "combine_draft_mov"]),
    }
    results, summaries, tests_all, fried_all, delong_rows = {}, [], [], {}, []
    for exp, (df, fsets) in experiments.items():
        df = df.reset_index(drop=True)
        for fs in fsets:
            X, y = xy(df, FEATURE_SETS[fs], target)
            for name in models:
                t0 = time.time()
                res = nested_cv(name, fs, "classification", X, y, y, grids[name], cfg, cfg.seed, thr)
                with mlops.run(cfg, f"clf-{exp}-{fs}-{name}", {"stage": "cv", "task": "classification", "population": exp}):
                    mlops.log_cv_result(res, {"population": exp, "n_rows": len(df), "positives": int(y.sum())})
                results[(exp, fs, name)] = res
                log.info("clf %s/%s/%s AUC=%.3f (%.0fs)", exp, fs, name, res.fold_scores["roc_auc"].mean(), time.time() - t0)

        # Summary + model comparison within each feature set, and feature-set comparison within each model
        sub = {k: v for k, v in results.items() if k[0] == exp}
        for (e, fs, name), res in sub.items():
            fsum = res.fold_scores.drop(columns=["fold", "repeat"]).agg(["mean", "std"])
            row = {"population": e, "feature_set": fs, "model": name, "n": len(df), "positives": int(df[target].sum())}
            oof = res.oof.groupby("row").agg(y=("y", "first"), pred=("pred", "mean"))
            auc, lo, hi = bootstrap_ci(oof.y.to_numpy(), oof.pred.to_numpy(), roc_auc_score, cfg["cv"]["bootstrap_resamples"], cfg.seed, stratify=True)
            for m in fsum.columns:
                row[f"{m}_mean"], row[f"{m}_std"] = fsum.loc["mean", m], fsum.loc["std", m]
            row.update({"oof_auc": auc, "oof_auc_ci_low": lo, "oof_auc_ci_high": hi})
            summaries.append(row)
        any_res = next(iter(sub.values()))
        for fs in fsets:
            sc = pd.DataFrame({name: results[(exp, fs, name)].fold_scores["roc_auc"].to_numpy() for name in models})
            t = pairwise_model_tests(sc, "roc_auc", any_res.n_train, any_res.n_test)
            t.insert(0, "comparison", "model_vs_model")
            t.insert(1, "population", exp)
            t.insert(2, "feature_set", fs)
            tests_all.append(t)
            fried_all[f"{exp}/{fs}"] = friedman(sc)
        fs_a, fs_b = fsets
        for name in models:
            a = results[(exp, fs_b, name)].fold_scores["roc_auc"].to_numpy()
            b = results[(exp, fs_a, name)].fold_scores["roc_auc"].to_numpy()
            t = pairwise_model_tests(pd.DataFrame({fs_b: a, fs_a: b}), "roc_auc", any_res.n_train, any_res.n_test)
            t.insert(0, "comparison", "feature_set_vs_feature_set")
            t.insert(1, "population", exp)
            t.insert(2, "feature_set", f"{fs_b} vs {fs_a} ({name})")
            tests_all.append(t)
            oa = results[(exp, fs_b, name)].oof.groupby("row").agg(y=("y", "first"), p=("pred", "mean"))
            ob = results[(exp, fs_a, name)].oof.groupby("row").agg(p=("pred", "mean"))
            dl = delong_test(oa.y.to_numpy(), oa.p.to_numpy(), ob.p.reindex(oa.index).to_numpy())
            delong_rows.append({"population": exp, "model": name, "feature_set_1": fs_b, "feature_set_2": fs_a, **dl})

    summary = pd.DataFrame(summaries)
    tests = pd.concat(tests_all, ignore_index=True)
    delong = pd.DataFrame(delong_rows)
    _save(summary, cfg, "classification_summary.csv", key="metrics")
    _save(tests, cfg, "classification_pairwise_tests.csv", key="metrics")
    _save(delong, cfg, "classification_delong.csv", key="metrics")
    _dump(fried_all, cfg, "classification_friedman.json")
    fold_long = pd.concat([r.fold_scores.assign(population=k[0], feature_set=k[1], model=k[2]) for k, r in results.items()], ignore_index=True)
    _save(fold_long, cfg, "classification_fold_scores.csv", key="metrics")
    oof_long = pd.concat([r.oof.assign(population=k[0], feature_set=k[1], model=k[2]) for k, r in results.items()], ignore_index=True)
    _save(oof_long, cfg, "classification_oof.parquet", key="metrics")

    # Register champion (pre-draft usable: all prospects, combine + draft) and the movement challenger.
    finals = {}
    for exp, fs, alias, reg in [("all_prospects", "combine_draft", "champion", cfg["mlflow"]["classifier_name"]),
                                ("tracked_subset", "combine_draft_mov", "challenger", cfg["mlflow"]["classifier_name"] + "-movement")]:
        s = summary[(summary.population == exp) & (summary.feature_set == fs)].sort_values("roc_auc_mean", ascending=False)
        best_model = s.iloc[0]["model"]
        df, _ = experiments[exp]
        X, y = xy(df.reset_index(drop=True), FEATURE_SETS[fs], target)
        est, best, score = refit_final(best_model, "classification", X, y, grids[best_model], cfg, cfg.seed)
        with mlops.run(cfg, f"clf-final-{exp}-{fs}-{best_model}", {"stage": "final", "task": "classification", "alias": alias}):
            import mlflow
            mlflow.log_params({"model": best_model, "population": exp, "feature_set": fs,
                               **{k.replace("model__", ""): v for k, v in best.items()}})
            mlflow.log_metric("refit_inner_cv_auc", score)
            mlops.log_and_register(est, X, "classification", reg if cfg["mlflow"]["register_models"] else None, alias)
        pi = permutation_importance(est, X, y, scoring="roc_auc", n_repeats=20, random_state=cfg.seed)
        imp = pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean, "std": pi.importances_std}).sort_values("importance", ascending=False)
        _save(imp, cfg, f"classification_importance_{exp}_{fs}.csv", key="metrics")
        finals[alias] = {"population": exp, "feature_set": fs, "model": best_model, "params": best, "inner_auc": score}
    _dump(finals, cfg, "classification_final_models.json")
    return {"results": results, "summary": summary, "tests": tests, "delong": delong, "friedman": fried_all, "finals": finals}


def _summarise(results: dict, metrics: list[str], cfg: Config, task: str):
    rows, scores = [], {m: {} for m in metrics}
    for name, res in results.items():
        row = {"model": name}
        for m in metrics:
            v = res.fold_scores[m]
            row[f"{m}_mean"], row[f"{m}_std"] = v.mean(), v.std()
            scores[m][name] = v.to_numpy()
        oof = res.oof.groupby("row").agg(y=("y", "first"), pred=("pred", "mean"))
        for m, fn in [("rmse", lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))),
                      ("r2", lambda a, b: 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))]:
            if m in metrics:
                est, lo, hi = bootstrap_ci(oof.y.to_numpy(), oof.pred.to_numpy(), fn, cfg["cv"]["bootstrap_resamples"], cfg.seed)
                row[f"oof_{m}"], row[f"oof_{m}_ci_low"], row[f"oof_{m}_ci_high"] = est, lo, hi
        params = pd.DataFrame(res.best_params)
        row["selected_params_mode"] = json.dumps({c: params[c].mode().iloc[0] for c in params.columns}, default=str) if len(params.columns) else "{}"
        rows.append(row)
    return pd.DataFrame(rows), {m: pd.DataFrame(v) for m, v in scores.items()}


# ----------------------------------------------------------------------------- stats (no retraining)
def stage_stats(cfg: Config) -> None:
    """Recompute the rank-based omnibus tests from saved outer-fold scores.

    Lets a statistics fix be applied to a release run without retraining: the fold
    scores are the run's own outputs, so the tests stay tied to that run.
    """
    m = cfg.path("metrics")
    reg = pd.read_csv(m / "regression_summary.csv")
    fried = {}
    for pop in reg.population.unique():
        fried[pop] = {}
        for metric in ["rmse", "r2"]:
            f = pd.read_csv(m / f"regression_fold_{pop}_{metric}.csv").drop(columns=["fold"])
            fried[pop][metric] = friedman(f, higher_is_better=(metric == "r2"))
    _dump(fried, cfg, "regression_friedman.json")
    folds = pd.read_csv(m / "classification_fold_scores.csv")
    cf = {}
    for (pop, fs), g in folds.groupby(["population", "feature_set"], sort=False):
        wide = g.pivot(index="fold", columns="model", values="roc_auc")
        cf[f"{pop}/{fs}"] = friedman(wide, higher_is_better=True)
    _dump(cf, cfg, "classification_friedman.json")
    log.info("stats: recomputed Friedman tests from saved fold scores")


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="CGMT pipeline")
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage", default="all", choices=["all", "data", "translation", "models", "stats", "figures"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    mlops.setup(cfg)
    log.info("CGMT v%s seed=%d", __version__, cfg.seed)

    if args.stage == "stats":
        stage_stats(cfg)
        return
    d = stage_data(cfg) if args.stage in ("all", "data") else load_processed(cfg)
    if args.stage == "data":
        return
    out = {}
    if args.stage in ("all", "translation", "models"):
        out["translation"] = stage_translation(cfg, d)
    if args.stage in ("all", "models"):
        out["regression"] = stage_regression(cfg, d)
        out["classification"] = stage_classification(cfg, d)
    if args.stage in ("all", "figures"):
        from .viz import make_all_figures
        make_all_figures(cfg)
    log.info("done")


if __name__ == "__main__":
    main()
