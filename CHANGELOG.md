# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/):

- **MAJOR**: a change that alters a published metric or target definition (results are not comparable across it).
- **MINOR**: new analysis, model, feature set or stage; existing results stay comparable.
- **PATCH**: fixes and refactors that do not change any number in `reports/metrics/`.

Every MLflow run is tagged with `pipeline_version` (this file), `data_version`
(hash of the raw files, `data/manifest.json`) and `git_commit`, so each number in
the report can be traced to the exact code and data that produced it.
Versions 0.1.0–0.3.0 were development steps made before the repository was initialised
(first commit = 0.4.0); they are recorded so the reasoning behind each decision is not lost.

## [2.1.0] - 2026-10-16

MINOR: the provider's data structure spec (`docs/Data_structure_CSV_files.docx`) becomes a
machine-readable data contract that every run checks, and the pipeline is made conformant
with the spec. **No number in `reports/metrics/` changes:** `data_version` stays
`dfc567c4ffe3`, and the data and translation stages were re-run and diffed against the v2.0.0
outputs: all 67 processed, interim and metric files are identical. The models were not retrained
because their inputs and code path are unchanged, so the v2.0.0 MLflow runs and registry stand.
`data_audit.json` gains keys, and `pp_rows_regular_season` goes from 240,412 to 239,673: R15 removes
739 down-0 rows, none of which was a DL, OL or WR/TE family play.

### Added
- **Data contract** `configs/data_contract.yaml`: the spec's nine files. Per column it records type,
  nullability, range, allowed values and pattern, plus keys, foreign keys, expected rows and
  tracking coverage. It also documents how the pipeline handles each known deviation.
- `cgmt.contract`: runs first in the data stage, also as `--stage contract` / `make validate`.
  Writes `reports/metrics/data_contract_report.csv` (one row per check, with severity and handling)
  and `data_contract_summary.json`. Errors (missing required file or column, duplicate key, broken
  foreign key) stop the run. Deviations are warnings, handled by the business rules.
- Result on the delivered files: 332 checks, **0 errors, 23 warnings**. `combine_tracking.csv` (spec File 3)
  is missing. Tracking covers 2024-2025 Week 18 + playoffs only (no 2023; 279 of 510 prospects).
  `player_play` has 2,141 fewer rows than the spec (-0.68%). Preseason games and playoff weeks
  19-23 sit outside the spec's REG/POST and 1-18. There are 961 down-0 rows (two-point tries).
  Four flags are stored as TRUE/FALSE where the spec documents 1.0/null. A few out-of-field
  tracking positions and minor nulls in unused columns complete the list.
- **`cgmt.combine_tracking`** (spec File 3, optional): business rule R16. Per-attempt metrics use the
  same definitions as in-game tracking: 1 s / 2 s burst, top speed, time to 90% of top speed, peak
  acceleration and braking, sharpest cut, tracking 10/20-yd splits, and plants and speed kept
  through the turns of the 3-cone and shuttle. Player features (`ct_*`) take the best attempt, and
  skill drills are z-scored within each drill name. When the file is present the pipeline adds:
  tracking-vs-official validity (`ct_official_validity.csv`); drill tracking vs game movement and
  on-field performance (`ct_translation_matrix.csv` and `ct_onfield_translation_matrix.csv`, kept
  as separate files so the v2.0.0 FDR q-values do not move); athletic vs athletic + tracking
  regressions on the same folds (`regression_ct_comparison.csv`, registered `<name>-ct@challenger`);
  and the pre-draft classifier set `combine_draft_ct`. The champion rule extends to that set:
  it replaces Combine + draft only with a higher mean AUC and DeLong p < 0.05. Tested end to end on
  a simulated file. It is inactive on the delivered data.
- 7 tests (`tests/test_contract.py`, 32 in total): contract severities, boolean encodings, file
  resolution for spec names and split parts, R15 and spec alignment codes, and the 40-yd split
  against kinematics, shuttle plants and best-attempt aggregation.

### Changed
- `raw_files` are names or glob patterns, so both the spec names (`game_tracking_2024.csv`,
  `player_play.csv`, `combine_tracking.csv`) and the delivered split parts (`game_tracking_2024_2.csv`,
  `player_play_3.csv`) are found. A part `<stem>_<n>.csv` is ignored when the full `<stem>.csv` is
  present. File order in the manifest is unchanged, so `data_version` is unchanged.
- Loader: an extra column is logged and ignored instead of stopping the run (the contract report
  lists it), while a missing column is still an error. Tracking is streamed one file at a time:
  rule R8b keeps `tracking_rules.season_types` (REG, POST; the spec's season files also hold
  preseason). A play split across file parts is processed once, after all parts are read. Frames
  are appended to `tracking_frames.parquet` and the figures read only the defensive-line frames
  they plot. Together these keep spec-scale seasons within memory.
- `performance.py`: the boolean parser accepts TRUE/FALSE, True/False, 1/0 and 1.0/0.0. Alignment
  cells come from `performance_rules.alignment_codes`, which adds the spec's DE -> EDGE and
  DT/NT -> INTERIOR_LINE (0 plays affected; OLB stays OTHER). New rule R15 (`valid_downs`) keeps
  downs 1-4. OL pressures allowed over expected are computed only where `pressure_allowed` is
  recorded (it was previously read as "no pressure"; 0 rows affected).
- `within_group_spearman`: Benjamini-Hochberg runs over finite p-values only (a constant column
  yields NaN instead of an error). The results are identical when every p-value is finite.

### Considered and not adopted
- Removing unblocked pressures from the DL pressure metric (spec: `unblocked_pressure`): the
  reliability of pressure over expected fell from ICC 0.25 to 0.22, so the metric stays as it was.
- Adding `cushion` and team coverage to the WR/TE separation expectation: WR ICC fell from 0.75 to
  0.65. Cushion is the defender's response to the receiver (a post-treatment variable), so
  conditioning on it removes part of the skill being measured.

## [2.0.0] - 2026-10-16

MAJOR: a seventh raw file and new on-field targets. `data_version` changes, and the
regression and classification tables gain rows. Every v1 number is reproduced unchanged
(same SBI, same folds, same seed). The v1 rows are a subset of the v2 outputs.

### Added
- **`player_play_3.csv`** (314,197 player-plays, 497 prospects, 2023-2025 incl. preseason and
  playoffs): schema-validated on the full 64-column header, loaded on 25 used columns,
  hashed in the manifest, referential integrity checked against `combine_results` and `games`.
- Business rules R10-R14 (`performance_rules` in the config): regular season only; plays
  nullified by penalty removed; NGS get-off outside [0.30, 2.00) s blanked (21 plays);
  minimum plays per family (DL 20 rush snaps, OL 50 pass-block snaps, WR/TE 30 routes);
  DBs get no on-field metric (the file holds no coverage outcome).
- `performance.py`: on-field metrics measured over expected for the play's context:
  DL get-off and pressure rate (vs alignment), OL peak pressure probability and pressures
  allowed (vs alignment x dropback quintile), WR/TE separation at the throw (vs route x alignment).
  Empirical-Bayes shrinkage, reliability weights, within-family z-scores, a per-family index.
- Translation analyses: on-field reliability (ICC), rookie-to-later stability, drill vs on-field
  matrix with draft capital as reference, DL role check, mediation drill -> get-off -> pressure,
  convergent validity of the tracking SBI against season-long get-off, context-adjusted tracking
  reliability (closes v1 limitation 4), starter GLM with the on-field index.
- Regression: four Combine -> on-field tasks (`regression.onfield_tasks`), each with the same
  four models, nested CV, tests and a registered champion (`cgmt-onfield-<task>@champion`).
- Classification: `combine_draft_proj` (pre-draft safe: a stacked `OnFieldProjector` learns
  Combine -> on-field inside each training fold, out-of-fold for training rows, labels never
  read at prediction time) and `combine_draft_perf` (observed on-field performance, diagnostic),
  registered as `cgmt-starter-classifier-onfield@challenger`. The champion switches to the
  projector set only if it beats Combine + draft on mean AUC with DeLong p < 0.05.
- Figures 15-18; figures 11-13 generalized to any set of experiments.
- 7 tests (`tests/test_onfield.py`): R10-R13, over-expected centering, projector never reading
  labels at prediction time, out-of-fold training projections, imputer exclusion, pipeline fit.

### Release notes
- Trained from a clean MLflow store: 54 runs tagged `git_commit = 5f30906`, `pipeline_version = 2.0.0`,
  `data_version = dfc567c4ffe3`; 9 registered models (4 new on-field regressors, 1 new challenger).
- After training: `mediation_bootstrap` also reports a bootstrap CI for the direct path (same
  resamples, so the indirect CIs are unchanged) and the translation stage was re-run; figure
  titles/legends were fixed (fig12 legend, fig13 wording, fig18 title now data-driven) and re-rendered.
- Verified: DL get-off random forest, projector logistic and on-field logistic re-runs reproduce
  their saved outer-fold scores exactly; every v1 number reproduces unchanged.
- Results: drills forecast DL get-off (rho 0.36-0.43) and get-off mediates ~half of the drill ->
  pressure link; Combine models reach pooled R^2 0.13 (DL get-off) and 0.19 (WR/TE separation),
  none for DL pressure or OL protection; the projector does not improve the starter champion
  (DeLong p >= 0.05) so it is not promoted; observed on-field performance adds +0.004 to +0.009 AUC.

### Changed
- Smoke config writes to `data/smoke/` and `reports/smoke/`, so it can never overwrite release outputs.
- `Config.path` creates the parent folder of file paths (manifest).
- Regression stage refactored into one `_regress` routine shared by SBI and on-field tasks;
  `regression_summary.csv` gains a `target` column; the SBI translation gap keeps `predicted_sbi`.

## [1.0.1] - 2026-10-15

### Added
- `make relocate` (`--stage relocate`): rewrites the absolute artifact paths MLflow stores in
  `mlflow.db` so the registry works after the archive is extracted anywhere. Idempotent.
  No number in `reports/metrics/` changes.

## [1.0.0] - 2026-10-15

Note: the release run's MLflow runs carry `git_commit = e8e0b91` (tag v0.9.0), the code that
trained every model. Changes in 1.0.0 after that run (below) touch only the GLM, the Friedman
rank labels and the figures, which were regenerated with `--stage translation`, `--stage stats`
and `--stage figures`.

### Added
- Technical report, README, model card, CI workflow (pytest + smoke pipeline).
- Release run from a clean MLflow store (`mlflow.db`), models registered with aliases:
  `cgmt-starter-classifier@champion`, `cgmt-starter-classifier-movement@challenger`,
  `cgmt-snap-burst-regressor@champion`, `cgmt-snap-burst-regressor-dl@champion`.

- `--stage stats` (`make` target via the CLI): recomputes the Friedman omnibus tests from the
  saved outer-fold scores, so a statistics fix can be applied to a release run without retraining.
- `MODEL_CARD.md`.

### Fixed
- Friedman mean ranks were computed "higher is better" for every metric, which inverted the
  ranks for RMSE (chi-square and p-values were unaffected). Ranks now use 1 = best per metric.
- Starter GLM did not converge: no undrafted tracked player started (quasi-complete separation
  on `is_undrafted`). The flag is dropped from the GLM (log pick already places undrafted
  players at pick 260); SBI result unchanged (OR 0.91 per 10 points, p = 0.62).

### Changed
- Figures: layout fixes (labels no longer overlap, titles fit the frame); speed-trace
  example restricted to edge rushers (DE/OLB) so the comparison is like-for-like; readable
  feature names in the importance chart; shorter ROC legends.

### Verified
- Two release experiments re-run from the committed config reproduce the saved outer-fold
  scores exactly (DL XGBoost RMSE, all-prospect logistic regression AUC).

## [0.9.0] - 2026-10-15

### Added
- `metric_reliability()`: ICC (method of moments) and one-way ANOVA per metric x position.
  Finding: 2-second snap burst is a player trait only on the defensive line (ICC 0.62).
- Defensive-line deep dive: drill-to-game correlations raw vs adjusted for edge/interior role.
- Regression now runs for two populations (`all_positions`, `defensive_line`).
- `sbi_reliability` per player (empirical-Bayes weight) and a reliability summary by position.
- Smoke configuration (`configs/smoke.yaml`) for CI: same code path, tiny CV and grids.
- 14 figures (`reports/figures/`), pytest suite (18 tests).

### Changed
- Regression target changed from shrunk burst in yards (`burst_2s_eb`) to the
  position-normalised SBI. With the yard target the position-mean baseline scored
  R^2 = 0.94 because shrinkage pulls values to group means, hiding the within-position question.
- Random-forest grid trimmed (200 trees, `max_features` 0.33/0.66) to fit a 2-core budget.
- Ridge alpha grid widened to [1, 1000] (alpha 0.1 overfit at n = 221).
- MLflow model serialisation set to cloudpickle (skops cannot serialise the project transformer).

## [0.5.0] - 2026-10-15

### Added
- Nested cross-validation (outer RepeatedStratifiedKFold 5x3, inner GridSearchCV 3-fold).
- Regression: position-mean baseline vs Ridge vs Random Forest vs XGBoost.
- Classification: Logistic Regression (L1/L2) vs Random Forest vs XGBoost over nested
  feature sets (combine, combine + draft, combine + draft + game movement).
- Statistics: corrected resampled t-test (Nadeau-Bengio), Wilcoxon + Holm, Friedman,
  DeLong, bootstrap CIs; logistic GLM likelihood-ratio test for SBI.
- MLflow tracking (SQLite backend) and model registry.

## [0.4.0] - 2026-10-15 - first commit

### Changed
- **SBI redefined (breaking):** player level = mean over plays with empirical-Bayes
  shrinkage, replacing the 80th-percentile "capacity". Subsampling showed the 80th
  percentile reads 0.43 yd low at 3 plays and 0.21 yd low at 5; because starters log
  more plays, the bias created a spurious SBI-start-rate correlation (0.23 -> 0.08 after the fix).
  The 80th percentile is kept as `<metric>_q80` for the sensitivity analysis only.

## [0.3.0] - 2026-10-15 (pre-repository)

### Added
- Snap-anchored movement metrics: burst at 1 s / 2 s, get-off, peak acceleration, top speed,
  max braking, sharpest cut. Player capacity = 80th percentile across plays (superseded in 0.4.0).
- Snap Burst Index on a 50 +/- 10 scale within position group.
- Within-position drill-vs-game translation matrix with bootstrap CIs and BH-FDR.

## [0.2.0] - 2026-10-15 (pre-repository)

### Added
- Prospect cleaning and business rules R1-R7 (ranges, missing flags, undrafted pick 260,
  seasons available, start rate and starter >= 0.5, age at draft, power conference).
- Tracking rules R8-R9 (snap anchoring, 2.0 s minimum, physical caps, 3+ plays per player).

## [0.1.0] - 2026-10-15 (pre-repository)

### Added
- Schema-validated loaders for the six CSVs, referential-integrity checks.
- `data/manifest.json` with SHA-256 per raw file and a combined `data_version`.
