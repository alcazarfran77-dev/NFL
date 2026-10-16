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
