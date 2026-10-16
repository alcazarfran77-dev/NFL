# Model card — CGMT v1.0.0

Release run: clean MLflow store `mlflow.db`, experiment `cgmt-combine-translation`,
data_version `263d3909c68b`, seed 42. All scores are nested-CV out-of-sample
(5 folds x 3 repeats) unless stated.

## Registered models

| Registry name @ alias | Task | Data | Model | Final hyperparameters | Out-of-sample score |
|---|---|---|---|---|---|
| `cgmt-starter-classifier@champion` | P(starter) | 510 prospects, Combine + draft capital | XGBoost | 200 trees, depth 2, lr 0.03, min child weight 3, subsample/colsample 0.8 | ROC-AUC 0.915 +/- 0.025; F1 0.646; recall 0.80; precision 0.55 |
| `cgmt-starter-classifier-movement@challenger` | P(starter) | 221 tracked players, + game movement | XGBoost | 400 trees, depth 3, lr 0.03, min child weight 3 | ROC-AUC 0.876 +/- 0.044 (no gain over 0.880 without movement) |
| `cgmt-snap-burst-regressor@champion` | Snap Burst Index | 221 players, all positions | Ridge | alpha = 1000 | RMSE 9.88 vs 10.05 baseline; R^2 ~ 0 |
| `cgmt-snap-burst-regressor-dl@champion` | Snap Burst Index | 54 defensive linemen | XGBoost | 200 trees, depth 3, lr 0.1, min child weight 3 | RMSE 8.54 vs 10.05 baseline; pooled R^2 0.29 [-0.02, 0.45] |

Input: a pandas DataFrame with `combine_position` plus the feature columns of the
chosen feature set (`cgmt.features.FEATURE_SETS`); the logged signature and input
example define the exact schema. Missing drills are allowed (imputed in-pipeline).

## Intended use

- **Starter champion:** a pre-draft prior on whether a prospect becomes a regular starter,
  given his Combine profile and an expected draft slot. Decision support only.
- **DL regressor:** projects a defensive lineman's in-game get-off (SBI) from his Combine
  profile; the gap between projected and observed SBI flags players to review on film.
- **Not intended:** ranking WRs or DBs by snap burst (the metric is not a stable trait for
  them in this data); any use of the movement challenger for draft decisions (its
  features exist only after the draft).

## Training data and target

- 510 prospects from the 2023-2025 classes; careers through the 2025 season.
- Starter = started >= 50% of available regular-season games (17 x seasons in league): 103 positives.
- Game movement from 10 Hz tracking, Week 18 + playoffs of 2024 and 2025; 221 players with 3+ plays.

## Known limitations and risks

- Draft capital dominates the starter model (permutation importance 0.29 AUC vs < 0.01 for any
  other feature); it partly encodes team investment, so it predicts opportunity as much as ability.
- Small samples: 54 DL players, 14 DL starters; DL regression gains are consistent across folds
  (Wilcoxon, Holm p = 0.02) but not significant under the corrected resampled t-test (Holm p = 0.45).
- Tracking covers late-season and playoff games only, with no opponents or play context.
- Pickled with cloudpickle (the pipeline contains a project transformer): load only from this
  trusted registry.

## Fairness and ethics

Features are athletic measurements, age, conference and draft slot; no protected attributes are
used. Outputs concern professional athletic performance and should support, not replace, scout
and coach judgment.
