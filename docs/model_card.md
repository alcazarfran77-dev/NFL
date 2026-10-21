# Model card — NFL project v3.2.0

Release run: clean MLflow store `mlflow/mlflow.db`, experiment `cgmt-combine-translation`,
`data_version = 261a80510390` (ten raw files: four Parquet tables and six compressed CSV, with full-season game
tracking for 2023 and 2024), seed 42, 93 runs at commit `878be03`, 17 registered models. Every score is
out-of-sample from nested cross-validation (5 folds × 3 repeats) unless stated otherwise. Version 3.2.0 adds
Parquet ingestion and translates the project to English; the models were retrained from the Parquet inputs and
reproduce those of v3.1.0 and v3.0.0.

The raw files are checked against the provider's data spec on every run (data contract: 435 checks, 0 errors,
32 warnings; see the [README](../README.md#data)), whatever their format: Parquet columns are harmonised to the
CSV encodings before the checks.

## Registered models

| Registry name @ alias | Task | Data | Model | Final hyperparameters | Out-of-sample score |
| --- | --- | --- | --- | --- | --- |
| `cgmt-starter-classifier@champion` | P(starter) | 510 prospects, Combine + draft capital | XGBoost | 200 trees, depth 2, learning rate 0.03, min child weight 3, row and column subsampling 0.8 | ROC-AUC 0.915 ± 0.025; F1 0.646; recall 0.80; precision 0.55 |
| `cgmt-starter-classifier-movement@challenger` | P(starter) | 299 tracked players (93 starters), + game movement | XGBoost | 200 trees, depth 2, learning rate 0.1, min child weight 3 | ROC-AUC 0.884 ± 0.038 (no gain over 0.884 without movement; DeLong p = 0.81) |
| `cgmt-snap-burst-regressor@champion` | Snap Burst Index | 299 players, all positions | Random forest | 200 trees, depth 6, max features 0.33, min leaf 3 | RMSE 9.57 vs 10.06 for the baseline (−5%); pooled R² 0.07 [0.00, 0.12] |
| `cgmt-snap-burst-regressor-dl@champion` | Snap Burst Index | 72 defensive linemen | Random forest | 200 trees, depth 3, max features 0.66, min leaf 10 | RMSE 7.85 vs 9.99 for the baseline (−21%); pooled R² 0.37 [0.17, 0.50]; corrected t, Holm p = 0.001 |
| `cgmt-snap-burst-regressor-wr@champion` (v3.0) | Snap Burst Index | 63 wide receivers | Ridge | alpha = 300 | RMSE 8.45 vs 9.38 for the baseline (−10%); pooled R² 0.22 [−0.01, 0.31]; corrected t p = 0.30 |
| `cgmt-starter-classifier-onfield@challenger` (v2.0) | P(starter) | 255 players with an on-field index, + on-field performance | Random forest | 200 trees, depth 3, max features 0.33, min leaf 10 | ROC-AUC 0.874 ± 0.057 (vs 0.867 without on-field data) |
| `cgmt-onfield-dl-getoff@champion` (v2.0) | DL get-off over expected (s) | 88 DL | Random forest | 200 trees, depth 3, max features 0.66, min leaf 3 | RMSE 0.060 vs 0.064 for the baseline; pooled R² 0.13 [−0.02, 0.25] |
| `cgmt-onfield-dl-pressure@champion` (v2.0) | DL pressure rate over expected | 89 DL | Random forest | 200 trees, depth 6, max features 0.33, min leaf 3 | No better than the baseline (pooled R² 0.05 [−0.13, 0.17]) |
| `cgmt-onfield-ol-protection@champion` (v2.0) | OL pressure prob. over expected | 70 OL | Random forest | 200 trees, depth 3, max features 0.33, min leaf 10 | No better than the baseline (pooled R² −0.01) |
| `cgmt-onfield-wrte-separation@champion` (v2.0) | WR/TE separation over expected (yd) | 96 WR/TE | Ridge | alpha = 100 | RMSE 0.169 vs 0.186 for the baseline; pooled R² 0.19 [−0.01, 0.30] |
| `cgmt-onfield-dl-getoff-ct@challenger` (v2.2) | DL get-off over expected (s) | 88 DL, + drill tracking | Random forest | 200 trees, depth 3, max features 0.66, min leaf 3 | RMSE 0.060, same as without tracking; R² change −0.004 (p = 0.92) |
| `cgmt-starter-classifier-combine-tracking@challenger` (v2.2) | P(starter) | 510 prospects, Combine + draft + drill tracking | XGBoost | 200 trees, depth 2, learning rate 0.03, min child weight 3 | ROC-AUC 0.913 vs 0.915 without tracking (DeLong p = 0.30) |
| `cgmt-snap-burst-regressor-{dl,wr}-ct`, `cgmt-onfield-{dl-pressure,ol-protection,wrte-separation}-ct` @challenger (v2.2; WR v3.0) | Those of their champions | + drill tracking | The best of the four per task | See `reports/metrics/regression_final_models.json` | No gain over the champion (R² change +0.004 DL snap burst, −0.007 WR, −0.03 to −0.05 on-field) |

**Input.** A pandas DataFrame with `combine_position` plus the columns of the chosen feature set
(`FEATURE_SETS` in `src/data_prep.py`); the registered signature and input example define the exact schema.
Missing drills are allowed (they are imputed inside the pipeline). Each model is stored together with
`data_prep.py` and `train.py`, so it loads without installing the project:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
clf = mlflow.sklearn.load_model("models:/cgmt-starter-classifier@champion")
```

The models expect the same feature values whatever format the raw tables came in: the Parquet and CSV readers
produce identical clean data.

## Intended use

- **Starter champion:** a pre-draft estimate of whether a prospect will become a regular starter, given his
  Combine profile and an expected draft pick. Decision support only.
- **DL and WR regressors:** project the in-game snap burst (SBI) of a defensive lineman or a receiver from his
  Combine profile; the gap between projected and observed SBI flags players for film review. The DL model is
  the strongest (significant gain over position alone); the receiver model's gain is consistent across folds
  but not significant under the corrected t.
- **On-field regressors (v2.0):** project a prospect's season-long on-field performance from his Combine
  profile; useful only for DL get-off and WR/TE separation. The DL pressure and OL protection models are
  registered for completeness and do not beat the position mean.
- **Drill-tracking challengers (v2.2):** registered so the comparison is reproducible; none beats its
  champion. They need the five `ct_*` drill-tracking features at prediction time.
- **Not intended:** ranking DBs, OL or TEs by snap burst (ICC 0.26, 0.40 and 0.08; not stable enough to rank
  players), or reading a defensive lineman's SBI as individual skill rather than role (without play context the
  DL ICC drops from 0.63 to 0.14); nor using the movement or on-field challengers for draft decisions (their
  features only exist after the draft).
- **Note on the projector:** the champion pipeline did not change in v2.0. The `combine_draft_proj` set (with
  the stacked `OnFieldProjector`) did not beat Combine + draft (DeLong p ≥ 0.05), so it was not promoted. In any
  pipeline with the projector, the `onfield_target_*` inputs are training labels and can be passed as NaN at
  prediction time.

## Training data and target

- 510 prospects from the 2023–2025 classes; careers through the 2025 season.
- Starter = started 50% or more of the available regular-season games (17 × seasons in the league):
  103 positives.
- In-game movement from the 10 Hz tracking: 2023 season, 2024 weeks 0–17, week 18 and playoffs of 2024 and
  2025 (regular season and playoffs, no preseason); 299 players with 3+ plays, median 39 plays.

## Limitations and known risks

- Draft capital dominates the starter model (permutation importance 0.29 of AUC against less than 0.01 for
  any other feature); it partly reflects the team's investment, so it forecasts opportunity as much as ability.
- Small samples: 72 DL and 63 WR in the snap-burst models. The DL gain passes the corrected resampled t-test
  (Holm p = 0.001); the WR gain does not (p = 0.30) and its per-fold R² varies a lot (SD 0.56).
- The tracking lacks 2025 weeks 1–17, and `game_tracking_2023_5.csv` stops at exactly 1,048,575 rows (a
  spreadsheet's limit), so a random part of 2023 is missing; the tracking carries no opponents or ball. The
  on-field metrics cover the 2023–2025 regular seasons, with "expected" computed from the prospects' plays, not
  the whole league.
- Drill tracking: the tracked start of an attempt does not repeat, so no metric timed from the start is used.
  Shuttle and 3-cone reliability rests on about 40 players with two attempts.
- The on-field challenger is circular by design (performance and starts share seasons).
- Models are serialised with cloudpickle (the pipeline holds a project transformer): load them only from this
  trusted registry.

## Fairness and ethics

The features are athletic measurements, age, conference and draft pick; no protected attribute is used. The
results concern professional athletic performance and should support, not replace, the judgement of scouts and
coaches.
