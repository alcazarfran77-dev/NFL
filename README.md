# CGMT — Combine-to-Game Movement Translation (v2.1.0)

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

Against the files delivered: **0 errors, 23 warnings**, and no number changed (v2.1.0 in the changelog).
What differs from the spec:

| Spec | Delivered | Effect |
|---|---|---|
| `combine_tracking.csv`, 6,310 drill attempts | not delivered | `cgmt.combine_tracking` (tested) switches on when the file is added |
| `game_tracking_2023/2024/2025.csv`, all 510 prospects | `game_tracking_2024_2.csv`, `_2025_2.csv`: Week 18 + playoffs, 279 prospects | SBI uses 221 players with 3+ plays; full files are read as is (split parts are ignored when the full file is present) |
| `player_play.csv`, 316,338 rows | `player_play_3.csv`, 314,197 rows (-0.68%) | results use the rows delivered |
| games: REG/POST, weeks 1-18 | also PRE (147 games, weeks 0-3), playoffs weeks 19-23 | R10 on-field = REG; R8b tracking = REG + POST |
| `down` 1-4 | 961 rows with down 0 (two-point tries) | R15 removes them (none was a family play) |
| quick/unblocked pressure, recovered fumble, TFL as 1.0/null | TRUE/FALSE text | the boolean parser reads both |

## Headline findings (v2.0)

1. **Drills forecast DL get-off, and get-off drives pressure.** Within the defensive line the
   vertical, 40, 10-yd split and broad jump correlate 0.36-0.43 with season-long get-off over
   expected (FDR q < 0.05), also within edge/interior role. About half of each drill's link to
   pressure runs through get-off (10-yd split indirect effect 0.15 SD, 95% CI 0.04-0.30).
2. **Combine models predict DL get-off and WR/TE separation, not pressure or protection.**
   Pooled out-of-fold R^2 0.13 (get-off, random forest) and 0.19 (separation, ridge; size-led);
   DL pressure and OL protection do not beat position alone. OL protection tracks draft capital.
3. **Season-long metrics are reliable traits** (ICC: get-off 0.89, WR separation 0.75, OL 0.74),
   and the tracking SBI is a valid get-off proxy (rho 0.39 with season-long get-off, 0.49 with pressure).
4. **On-field performance adds a little to starter prediction; projections add nothing.** Observed
   on-field performance +0.004 to +0.009 AUC (GLM OR 1.47 per SD beyond draft pick, p = 0.04);
   a stacked Combine -> on-field projector changes AUC by -0.002 to +0.0004.

### Version 1 findings (all reproduced unchanged)

1. **Check reliability before you trust a tracking metric.** Snap burst is a
   stable player trait only on the defensive line (ICC 0.62). For WRs and DBs
   no player differences are detectable from these plays: the play call decides
   how far they move in 2 s. Peak acceleration is the most portable trait.
2. **Sprint and jump drills carry into game movement; the NGS athleticism
   composite and bench press do not.** Within position, the 40, 10-yd split and
   broad jump correlate 0.2–0.35 with in-game burst, acceleration and top speed
   (FDR q < 0.05). On the DL about half of that link is role (edge vs interior).
3. **Combine predicts snap burst only where it is a trait.** On the defensive line
   XGBoost cuts RMSE from 10.05 to 8.54 SBI points (pooled R^2 0.29); across all
   positions no model beats position alone by a useful margin.
4. **Game movement adds nothing to predicting starters once draft capital is known.**
   Draft pick lifts ROC-AUC from 0.84 to 0.91 (p < 0.003); game movement then adds
   -0.005 to 0 (corrected t p >= 0.66, DeLong p >= 0.18).
5. **A methods trap:** building a player's metric from his best plays (80th
   percentile) biases low-exposure players down and manufactures a link between
   movement and playing time (rho 0.23 vs 0.08 with an unbiased mean).

## Layout

```
configs/config.yaml        single source of truth: paths, business rules, CV, grids
configs/smoke.yaml         same pipeline on tiny CV/grids (CI)
configs/data_contract.yaml the provider's data structure spec as a checked contract (v2.1)
docs/                      Data_structure_CSV_files.docx (the spec)
data/raw/                  the seven input CSVs (not in git; hashed in data/manifest.json)
data/processed/            prospects, tracking play metrics, movement profiles, on-field plays + profiles (parquet)
src/cgmt/
  contract.py              data-contract validation and report (v2.1)
  data.py                  file resolution (spec names + split parts), schemas, integrity, manifest
  cleaning.py              business rules R1-R7, starter target
  movement.py              tracking cleaning (R8-R9), per-play metrics, SBI, reliability
  performance.py           player_play rules R10-R15, on-field metrics over expected, stability (v2.0)
  combine_tracking.py      Combine drill tracking (spec File 3, R16): drill features; on when the file exists (v2.1)
  features.py              nested feature sets
  models.py                PositionMedianImputer, OnFieldProjector, pipelines, nested CV, final refit
  stats.py                 corrected t-test, Wilcoxon-Holm, Friedman, DeLong, bootstrap, mediation
  mlops.py                 MLflow tracking, tags, model registry
  viz.py                   18 report figures
  pipeline.py              CLI orchestrator
tests/                     32 pytest tests
reports/metrics/           every table behind the report (CSV/JSON)
reports/figures/           fig01 ... fig18
mlflow.db, mlartifacts/    experiment store and model registry (in the release archive)
```

## Reproduce

```bash
python -m pip install -r requirements.txt     # Python 3.13, pinned versions
make relocate                                 # once, after extracting the archive (repoints MLflow paths)
make validate                                 # data contract only, ~10 s
make test                                     # 32 unit tests, ~5 s
make smoke                                    # whole pipeline on tiny grids, ~5 min (own output folders)
make all                                      # release run, ~45 min on 2 cores
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

- The Combine file holds drill **results**. The spec also lists drill tracking (`combine_tracking.csv`),
  which was not delivered, so "movement during drills" is measured by times and distances until it is.
- Tracking covers Week 18 and the playoffs of the 2024 and 2025 seasons, with 1–4
  tracked players per play and no ball or opponents, so the SBI uses individual kinematics.
- `player_play` (v2.0) covers every game of 2023-2025 for 497 prospects; the on-field metrics
  use regular-season, non-nullified plays on downs 1-4 only (239,673 rows). DBs get no on-field metric (the
  file has no coverage outcome). "Expected" is the average of prospect plays in the same context.
- The careers table covers seasons through 2025; start rate is normalised by the
  seasons each class has been in the league.
