# Model card — CGMT v3.0.0

Release run: clean MLflow store `mlflow.db`, experiment `cgmt-combine-translation`,
data_version `a75ba51ae4b9` (ten raw files, incl. full-season game tracking for 2023 and 2024), seed 42,
93 runs at git commit `aad6177`, 17 registered models. All scores are nested-CV out-of-sample
(5 folds x 3 repeats) unless stated. v3.0.0 retrains every model that uses game tracking (snap burst and
the movement challenger) and adds the wide-receiver snap-burst model; every other model and score
reproduces v2.2.0 exactly.

The raw files are checked against the provider's data structure spec on every run (data contract:
435 checks, 0 errors, 32 warnings; see the README).

## Registered models

| Registry name @ alias | Task | Data | Model | Final hyperparameters | Out-of-sample score |
|---|---|---|---|---|---|
| `cgmt-starter-classifier@champion` | P(starter) | 510 prospects, Combine + draft capital | XGBoost | 200 trees, depth 2, lr 0.03, min child weight 3, subsample/colsample 0.8 | ROC-AUC 0.915 +/- 0.025; F1 0.646; recall 0.80; precision 0.55 |
| `cgmt-starter-classifier-movement@challenger` | P(starter) | 299 tracked players (93 starters), + game movement | XGBoost | 200 trees, depth 2, lr 0.1, min child weight 3 | ROC-AUC 0.884 +/- 0.038 (no gain over 0.884 without movement; DeLong p = 0.81) |
| `cgmt-snap-burst-regressor@champion` | Snap Burst Index | 299 players, all positions | Random forest | 200 trees, depth 6, max features 0.33, min leaf 3 | RMSE 9.57 vs 10.06 baseline (-5%); pooled R^2 0.07 [0.00, 0.12] |
| `cgmt-snap-burst-regressor-dl@champion` | Snap Burst Index | 72 defensive linemen | Random forest | 200 trees, depth 3, max features 0.66, min leaf 10 | RMSE 7.85 vs 9.99 baseline (-21%); pooled R^2 0.37 [0.17, 0.50]; corrected t Holm p = 0.001 |
| `cgmt-snap-burst-regressor-wr@champion` (v3.0) | Snap Burst Index | 63 wide receivers | Ridge | alpha = 300 | RMSE 8.45 vs 9.38 baseline (-10%); pooled R^2 0.22 [-0.01, 0.31]; corrected t p = 0.30 |
| `cgmt-starter-classifier-onfield@challenger` (v2.0) | P(starter) | 255 players with an on-field index, + on-field performance | Random forest | 200 trees, depth 3, max features 0.33, min leaf 10 | ROC-AUC 0.874 +/- 0.057 (vs 0.867 without on-field) |
| `cgmt-onfield-dl-getoff@champion` (v2.0) | DL get-off over expected (s) | 88 DL | Random forest | 200 trees, depth 3, max features 0.66, min leaf 3 | RMSE 0.060 vs 0.064 baseline; pooled R^2 0.13 [-0.02, 0.25] |
| `cgmt-onfield-dl-pressure@champion` (v2.0) | DL pressure rate over expected | 89 DL | Random forest | 200 trees, depth 6, max features 0.33, min leaf 3 | No better than baseline (pooled R^2 0.05 [-0.13, 0.17]) |
| `cgmt-onfield-ol-protection@champion` (v2.0) | OL pressure probability over expected | 70 OL | Random forest | 200 trees, depth 3, max features 0.33, min leaf 10 | No better than baseline (pooled R^2 -0.01) |
| `cgmt-onfield-wrte-separation@champion` (v2.0) | WR/TE separation over expected (yd) | 96 WR/TE | Ridge | alpha = 100 | RMSE 0.169 vs 0.186 baseline; pooled R^2 0.19 [-0.01, 0.30] |
| `cgmt-onfield-dl-getoff-ct@challenger` (v2.2) | DL get-off over expected (s) | 88 DL, + drill tracking | Random forest | 200 trees, depth 3, max features 0.66, min leaf 3 | RMSE 0.060, same as without tracking; R^2 change -0.004 (p = 0.92) |
| `cgmt-starter-classifier-combine-tracking@challenger` (v2.2) | P(starter) | 510 prospects, Combine + draft + drill tracking | XGBoost | 200 trees, depth 2, lr 0.03, min child weight 3 | ROC-AUC 0.913 vs 0.915 without tracking (DeLong p = 0.30) |
| `cgmt-snap-burst-regressor-{dl,wr}-ct`, `cgmt-onfield-{dl-pressure,ol-protection,wrte-separation}-ct` @challenger (v2.2, WR v3.0) | as their champions | + drill tracking | best of the four per task | see `regression_final_models.json` | No gain over the champion (R^2 change +0.004 DL snap burst, -0.007 WR, -0.03 to -0.05 on-field) |

Input: a pandas DataFrame with `combine_position` plus the feature columns of the
chosen feature set (`cgmt.features.FEATURE_SETS`); the logged signature and input
example define the exact schema. Missing drills are allowed (imputed in-pipeline).

## Intended use

- **Starter champion:** a pre-draft prior on whether a prospect becomes a regular starter,
  given his Combine profile and an expected draft slot. Decision support only.
- **DL and WR regressors:** project a defensive lineman's or wide receiver's in-game snap burst (SBI)
  from his Combine profile; the gap between projected and observed SBI flags players to review on film.
  The DL model is the stronger one (significant gain over position alone); the WR model's gain is
  consistent across folds but not significant under the corrected t-test.
- **On-field regressors (v2.0):** project a prospect's season-long on-field performance from his
  Combine profile; useful for DL get-off and WR/TE separation only. The DL-pressure and
  OL-protection models are registered for completeness and do not beat the position mean.
- **Drill-tracking challengers (v2.2):** registered so the comparison is reproducible; none beats its
  champion. They need the five `ct_*` features from `cgmt.combine_tracking` at inference.
- **Not intended:** ranking DBs, OL or TEs by snap burst (ICC 0.26, 0.40 and 0.08; not stable enough to
  rank players), or reading a DL's SBI as individual skill rather than role (with play context removed
  the DL ICC falls from 0.63 to 0.14); any use of the movement or on-field challengers for draft decisions
  (their features exist only after the draft).
- **Projector note:** the champion pipeline is unchanged by v2.0. The tested `combine_draft_proj`
  set (stacked `OnFieldProjector`) did not beat Combine + draft (DeLong p >= 0.05), so it was not
  promoted. In any pipeline that contains the projector, `onfield_target_*` inputs are training
  labels only and may be passed as NaN at inference.

## Training data and target

- 510 prospects from the 2023-2025 classes; careers through the 2025 season.
- Starter = started >= 50% of available regular-season games (17 x seasons in league): 103 positives.
- Game movement from 10 Hz tracking: 2023 season, 2024 weeks 0-17, Week 18 + playoffs of 2024 and 2025
  (regular season + playoffs, preseason removed); 299 players with 3+ plays, median 39 plays.

## Known limitations and risks

- Draft capital dominates the starter model (permutation importance 0.29 AUC vs < 0.01 for any
  other feature); it partly encodes team investment, so it predicts opportunity as much as ability.
- Small samples: 72 DL and 63 WR in the snap-burst models. The DL gain passes the corrected resampled
  t-test (Holm p = 0.001); the WR gain does not (p = 0.30) and its fold R^2 varies widely (sd 0.56).
- Tracking has no 2025 weeks 1-17, and `game_tracking_2023_5.csv` stops at exactly 1,048,575 rows (a
  spreadsheet limit), so part of 2023 is missing at random; there are no opponents or ball in the tracking. On-field metrics cover
  the 2023-2025 regular seasons, with "expected" computed from prospects' plays, not the league.
- Drill tracking: the start of a tracked attempt is not repeatable, so start-timed drill metrics are not used.
  Shuttle and 3-cone reliability rests on about 40 players with two attempts.
- The on-field challenger is circular by design (performance and starts share seasons).
- Pickled with cloudpickle (the pipeline contains a project transformer): load only from this
  trusted registry.

## Fairness and ethics

Features are athletic measurements, age, conference and draft slot; no protected attributes are
used. Outputs concern professional athletic performance and should support, not replace, scout
and coach judgment.
