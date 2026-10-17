# CGMT — Combine-to-Game Movement Translation (v3.0.0)

Do NFL Combine drills predict how a prospect moves and performs in regular-season
games, and does that tell us who becomes a starter? This project answers it for one
narrow slice of movement, **the first seconds after the snap**, measured two ways:

- the **Snap Burst Index (SBI)**: distance covered 2.0 s after the snap from 10 Hz
  tracking (the game analog of the 10-yard split), averaged over a player's plays,
  shrunk toward his position group by empirical Bayes and scaled 50 ± 10 within position;
- **season-long on-field metrics over expected** (v2.0, from `player_play`): DL get-off
  and pressure rate, OL pressure probability and pressures allowed, WR/TE separation at
  the throw, each scored against the average prospect play in the same context
  (alignment, dropback length, route).

The technical report (methods, figures, metrics, statistical tests and
insights) is published as a Claude doc; `reports/` holds every number and
figure it cites.

## Data contract (v2.1)

The provider's data structure spec (`docs/Data_structure_CSV_files.docx`) is encoded in
`configs/data_contract.yaml` and checked on every run (`make validate`, also the first step of
`make data`). The report `reports/metrics/data_contract_report.csv` has one row per check, with
severity and the handling the pipeline applies. Errors (missing required file or column,
duplicate key, broken foreign key) stop the run. Deviations are warnings, handled by business rules.

Against the files delivered: **0 errors, 32 warnings** (435 checks). What differs from the spec:

| Spec | Delivered | Effect |
|---|---|---|
| `combine_tracking.csv`, 6,310 drill attempts | `combine_tracking_6.csv`: 6,310 attempts, 510 prospects (v2.2) | 40 lane outside the 0-53.3 yd frame; ball rows exceed player ranges and are dropped (R16) |
| `game_tracking_2023/2024/2025.csv`, full seasons, all 510 prospects | v3.0: `game_tracking_2023_5.csv` (2023 season), `_2024_1_5.csv` (2024 weeks 0-17), `_2024_2.csv` and `_2025_2.csv` (Week 18 + playoffs); no 2025 weeks 1-17 | 343 prospects tracked after R8b; the SBI uses 299 players with 3+ plays (24,828 player-plays). Split parts are ignored when the full file is present |
| complete files | `game_tracking_2023_5.csv` has exactly 1,048,575 data rows, a spreadsheet's row limit | almost certainly cut when saved; R8c drops the file's last (possibly partial) player-play and the contract warns. The uncut file would add 2023 plays |
| `player_play.csv`, 316,338 rows | `player_play_3.csv`, 314,197 rows (-0.68%) | results use the rows delivered |
| games: REG/POST, weeks 1-18 | also PRE (147 games, weeks 0-3), playoffs weeks 19-23 | R10 on-field = REG; R8b tracking = REG + POST |
| `down` 1-4 | 961 rows with down 0 (two-point tries) | R15 removes them (none was a family play) |
| quick/unblocked pressure, recovered fumble, TFL as 1.0/null | TRUE/FALSE text | the boolean parser reads both |

## Combine drill tracking (v2.2)

1. **The start of a tracked drill is not repeatable; distance splits are.** Across a player's two 40s the
   0-5 yd split repeats at r = 0.16, the 5-10 yd split at 0.87 and top speed at 0.99. Features are gated at
   r >= 0.5, one per construct: 40 acceleration (5-10 yd), 40 top speed, speed kept through the shuttle turns,
   and position-drill top speed and acceleration (z-scored within each drill as run by one position group).
2. **The tracking reproduces the stopwatch.** Tracked 40 top speed vs official 40: rho -0.97.
3. **It improves no model.** Position-drill top speed correlates +0.37 with DL get-off (q = 0.009). Within
   the DL it tracks the 40 (rho -0.76), and with weight, role and the 40 held fixed the link falls to +0.13
   (p = 0.31). Same folds with and without drill tracking: R² change +0.004 for DL snap burst, -0.007 for WR
   snap burst (v3.0), -0.004 for DL get-off and -0.03 to -0.05 elsewhere. Starter AUC is 0.915 -> 0.913
   (DeLong p = 0.30). Champions are unchanged; the drill-tracking models are registered as challengers.

## Headline findings (v3.0, full-season game tracking)

With the 2023 season and 2024 weeks 0-17 added, the median tracked player has 39 plays instead of 8.

1. **Snap burst is a stable trait for defensive linemen and wide receivers.** ICC: DL 0.63, WR 0.62
   (was -0.34 with Week 18 only), OL 0.40, DB 0.26. With play context removed WRs stay at 0.62 and the
   DL falls to 0.14: a lineman's burst is mostly his role (edge vs interior), a receiver's is his own.
2. **Combine now predicts DL snap burst significantly.** Random forest RMSE 7.85 vs 9.99 for position alone
   (-21%), pooled out-of-fold R^2 0.37 [0.17, 0.50], corrected resampled t-test Holm p = 0.001 (v2.2: 0.45).
   Wide receivers (new): ridge -10%, pooled R^2 0.22 [-0.01, 0.31], better in most folds (Wilcoxon p = 0.03)
   but not significant under the corrected t-test (p = 0.30). All positions: -5%, R^2 0.07.
3. **Sprint and jump drills carry into game movement; the NGS composite and bench press do not.** 35 of 63
   within-position drill x movement cells pass FDR (25 in v2.2); the 40 vs in-game top speed rho 0.43.
4. **Game movement adds nothing to predicting starters once draft capital is known.** Draft pick lifts ROC-AUC
   from 0.84 to 0.91 (510 prospects); among 299 tracked players movement changes AUC by -0.003 to 0.000
   (DeLong p >= 0.47); GLM odds ratio per 10 SBI points 0.87 [0.61, 1.24], p = 0.45.
5. **A methods trap:** a metric built from a player's best plays (80th percentile) reads 0.50 yd low at 3
   plays and correlates more with start rate than the unbiased mean (rho 0.12 vs 0.07).

## On-field findings (v2.0, unchanged)

1. **Drills forecast DL get-off, and get-off drives pressure.** Within the defensive line the
   vertical, 40, 10-yd split and broad jump correlate 0.36-0.43 with season-long get-off over
   expected (FDR q < 0.05), also within edge/interior role. About half of each drill's link to
   pressure runs through get-off (10-yd split indirect effect 0.15 SD, 95% CI 0.04-0.30).
2. **Combine models predict DL get-off and WR/TE separation, not pressure or protection.**
   Pooled out-of-fold R^2 0.13 (get-off, random forest) and 0.19 (separation, ridge; size-led);
   DL pressure and OL protection do not beat position alone. OL protection tracks draft capital.
3. **Season-long metrics are reliable traits** (ICC: get-off 0.89, WR separation 0.75, OL 0.74),
   and the tracking SBI is a valid get-off proxy (v3.0: rho 0.45 with season-long get-off for 69 DL, 0.28 with pressure).
4. **On-field performance adds a little to starter prediction; projections add nothing.** Observed
   on-field performance +0.004 to +0.009 AUC (GLM OR 1.47 per SD beyond draft pick, p = 0.04);
   a stacked Combine -> on-field projector changes AUC by -0.002 to +0.0004.

## Layout

```
configs/config.yaml        single source of truth: paths, business rules, CV, grids
configs/smoke.yaml         same pipeline on tiny CV/grids (CI)
configs/data_contract.yaml the provider's data structure spec as a checked contract (v2.1)
docs/                      Data_structure_CSV_files.docx (the spec)
data/raw/                  the ten input CSVs (not in git; hashed in data/manifest.json)
data/processed/            prospects, tracking play metrics, movement profiles, on-field plays + profiles (parquet)
src/cgmt/
  contract.py              data-contract validation and report (v2.1)
  data.py                  file resolution (spec names + split parts), schemas, integrity, manifest
  cleaning.py              business rules R1-R7, starter target
  movement.py              tracking cleaning (R8-R9), per-play metrics, SBI, reliability
  performance.py           player_play rules R10-R15, on-field metrics over expected, stability (v2.0)
  combine_tracking.py      Combine drill tracking (spec File 3, R16): attempt metrics, reliability gate, features (v2.2)
  features.py              nested feature sets
  models.py                PositionMedianImputer, OnFieldProjector, pipelines, nested CV, final refit
  stats.py                 corrected t-test, Wilcoxon-Holm, Friedman, DeLong, bootstrap, mediation
  mlops.py                 MLflow tracking, tags, model registry
  viz.py                   21 report figures
  pipeline.py              CLI orchestrator
tests/                     34 pytest tests
reports/metrics/           every table behind the report (CSV/JSON)
reports/figures/           fig01 ... fig21
mlflow.db, mlartifacts/    experiment store and model registry (in the release archive)
```

## Reproduce

```bash
python -m pip install -r requirements.txt     # Python 3.13, pinned versions
make relocate                                 # once, after extracting the archive (repoints MLflow paths)
make validate                                 # data contract only, ~10 s
make test                                     # 34 unit tests, ~5 s
make smoke                                    # whole pipeline on tiny grids, ~5 min (own output folders)
make all                                      # release run, ~90 min on 2 cores
make ui                                       # MLflow UI: runs + registry
```

Stages can be run on their own: `make validate`, `make data`, `make translation`, `make models`,
`make stats`, `make figures`. Everything is seeded (`seed: 42`); a test asserts that two fits
with the same seed give identical predictions.

Load a registered model:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
clf = mlflow.sklearn.load_model("models:/cgmt-starter-classifier@champion")
```

## Versioning strategy

- **Code:** git + semantic version tags; every change is described in
  [CHANGELOG.md](CHANGELOG.md). MAJOR = a metric/target definition changed,
  MINOR = new analysis or model, PATCH = no number in `reports/metrics/` changes.
- **Data:** `data/manifest.json` stores SHA-256, bytes and rows per raw file and a
  combined `data_version`; CI fails on a hash mismatch. The data contract
  (`configs/data_contract.yaml`) is versioned with the code: a change in the provider's spec
  is a contract change recorded in the changelog.
- **Experiments:** each MLflow run is tagged with `pipeline_version`,
  `data_version`, `git_commit` and `seed`.
- **Models:** registered with aliases (`champion`, `challenger`) and a
  `validation_status` tag.

## Data notes

- Combine drills are measured two ways: official results (times, distances) and, since v2.2, 10 Hz drill
  tracking (`combine_tracking_6.csv`). Only tracking metrics that repeat within a player are used.
- Game tracking (v3.0) covers the 2023 season, 2024 weeks 0-17, and Week 18 + playoffs of 2024 and 2025
  (2025 weeks 1-17 were not delivered). Preseason games are removed (R8b). There are 1-4 tracked players
  per play and no ball or opponents, so the SBI uses individual kinematics.
- `player_play` (v2.0) covers every game of 2023-2025 for 497 prospects; the on-field metrics
  use regular-season, non-nullified plays on downs 1-4 only (239,673 rows). DBs get no on-field metric (the
  file has no coverage outcome). "Expected" is the average of prospect plays in the same context.
- The careers table covers seasons through 2025; start rate is normalised by the
  seasons each class has been in the league.
