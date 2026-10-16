# CGMT — Combine-to-Game Movement Translation (v1.0.1)

Do NFL Combine drills predict how a prospect moves in games, and does in-game
movement tell us who becomes a starter? This project answers that for one
narrow slice of movement: **the first two seconds after the snap**.

It introduces the **Snap Burst Index (SBI)**: distance covered 2.0 s after the
snap (the game analog of the 10-yard split), averaged over a player's plays,
shrunk toward his position group by empirical Bayes and scaled 50 ± 10 within
position, like other scouting grades.

The technical report (methods, figures, metrics, statistical tests and
insights) is published as a Claude doc; `reports/` holds every number and
figure it cites.

## Headline findings

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
data/raw/                  the six input CSVs (not in git; hashed in data/manifest.json)
data/processed/            prospects, per-play metrics, player movement profiles (parquet)
src/cgmt/
  data.py                  schema validation, referential integrity, SHA-256 manifest
  cleaning.py              business rules R1-R7, starter target
  movement.py              tracking cleaning (R8-R9), per-play metrics, SBI, reliability
  features.py              nested feature sets
  models.py                PositionMedianImputer, pipelines, nested CV, final refit
  stats.py                 corrected t-test, Wilcoxon-Holm, Friedman, DeLong, bootstrap
  mlops.py                 MLflow tracking, tags, model registry
  viz.py                   14 report figures
  pipeline.py              CLI orchestrator
tests/                     18 pytest tests
reports/metrics/           every table behind the report (CSV/JSON)
reports/figures/           fig01 ... fig14
mlflow.db, mlartifacts/    experiment store and model registry (in the release archive)
```

## Reproduce

```bash
python -m pip install -r requirements.txt     # Python 3.13, pinned versions
make relocate                                 # once, after extracting the archive (repoints MLflow paths)
make test                                     # 18 unit tests, ~5 s
make smoke                                    # whole pipeline on tiny grids, ~3 min
make all                                      # release run, ~30 min on 2 cores
make ui                                       # MLflow UI: runs + registry
```

Stages can be run on their own: `make data`, `make translation`, `make models`,
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
  combined `data_version`; CI fails on a hash mismatch.
- **Experiments:** each MLflow run is tagged with `pipeline_version`,
  `data_version`, `git_commit` and `seed`.
- **Models:** registered with aliases (`champion`, `challenger`) and a
  `validation_status` tag.

## Data notes

- The Combine file holds drill **results**, not drill tracking, so "movement
  during drills" is measured by times and distances.
- Tracking covers Week 18 and the playoffs of the 2024 and 2025 seasons, with 1–4
  tracked players per play and no ball or opponents: separation and pass-rush win
  rates cannot be computed, so the study uses individual kinematics.
- The careers table covers seasons through 2025; start rate is normalised by the
  seasons each class has been in the league.
