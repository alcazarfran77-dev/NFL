# NFL project: from the Combine to the game (v3.2.0)

Do NFL Combine drills forecast how a prospect moves and performs in regular-season games, and does that say
anything about who becomes a starter? This project answers it for one specific slice of movement, **the first
seconds after the snap**, measured in two ways:

- the **Snap Burst Index (SBI)**: the distance covered 2.0 s after the snap in the 10 Hz tracking (the in-game
  analogue of the 10-yard split), averaged over each player's plays, shrunk towards his position group by
  empirical Bayes and scaled to 50 ± 10 within position;
- **season-long on-field metrics over expected** (from `player_play`): defensive-line get-off and pressure
  rate, offensive-line pressure probability and pressures allowed, and receiver and tight-end separation at the
  throw, each compared with the average prospect play in the same context (alignment, dropback duration,
  route).

**Short answer.** Sprint and jump drills do carry into the game, most clearly on the defensive line: the
Combine forecasts a lineman's snap burst with 21% less error than position alone, and his in-game get-off,
which is the path through which athleticism becomes pass-rush pressure. But once the draft pick is known,
neither game movement nor drill tracking improves the prediction of who will start.

The **final technical report** (methods, figures, metrics, statistical tests and conclusions) and the
**version history** are in [CHANGELOG.md](CHANGELOG.md). This README is the guide to reproducing the project
and explains the design decisions.

**What is new in 3.2.0.** The raw tables can now be delivered as **Parquet** as well as CSV or compressed CSV:
the pipeline detects each file's format from its content, converts Parquet columns to the CSV encodings and
reprocesses everything with the same rules. Four tables (`players`, `combine_results`,
`player_career_successes`, `games`) now arrive as Parquet; the clean data and every metric are identical to
v3.1.0. The whole project (code, configuration, figures, documentation and folder names) is in English on this
`english` branch; `main` keeps the Spanish v3.1.0.

## Repository structure

The structure follows the `Estructura_Gitub_NFL.docx` document, with its folders in English. Folders and files
marked **add-on** were not in that document; they were added because the project needs them to be reproducible
(configuration, tests, results, model registry and continuous integration).

```
NFL/
│── data/
│   ├── raw/                  # original data: 4 Parquet + 6 compressed CSV (.csv.gz) + manifest.json
│   └── clean/                # clean, model-ready data (parquet)
│       └── interim/          #   cleaned tracking frames and play context
│── src/
│   ├── training.ipynb        # Colab notebook (training and evaluation)
│   ├── data_prep.py          # data ingestion, cleaning and preparation functions
│   └── train.py              # training, evaluation and MLflow registration script
│── README.md                 # reproduction guide and decisions (this file)
│── CHANGELOG.md              # final report + version history
│
│   Add-ons
│── configs/                  # add-on: pipeline parameters
│   ├── config.yaml           #   paths, business rules, cross-validation and grids (release run)
│   ├── smoke.yaml            #   the same pipeline with minimal grids (CI, about 10 min)
│   └── data_contract.yaml    #   the provider's spec as a contract checked on every run
│── tests/                    # add-on: 40 unit tests (pytest)
│── reports/                  # add-on: results
│   ├── figures/              #   the report's 21 figures (PNG)
│   └── metrics/              #   every table behind the report (CSV/JSON)
│── docs/                     # add-on
│   ├── Data_structure_CSV_files.docx   # the provider's data specification
│   └── model_card.md         #   card of the registered models
│── mlflow/                   # add-on: experiment store and model registry
│   ├── mlflow.db             #   runs, metrics, tags and registered models (SQLite)
│   └── mlartifacts/          #   artifacts: models, per-fold scores, predictions
│── .github/workflows/ci.yml  # add-on: continuous integration (tests, hashes, contract, smoke run)
│── requirements.txt          # add-on: pinned dependencies (Python 3.13)
│── pyproject.toml            # add-on: pytest configuration
│── Makefile                  # add-on: shortcuts for every stage
└── .gitignore
```

**What each file in `src/` holds.**

| File | Contents |
| --- | --- |
| `data_prep.py` | Configuration and seed; format detection (Parquet, CSV, CSV.gz), file resolution and manifest (data versioning); data contract; prospect cleaning (rules R1–R7); game tracking → per-play metrics → profiles and SBI (R8–R9); `player_play` → on-field metrics over expected (R10–R15); Combine drill tracking (R16); feature sets and modelling tables. |
| `train.py` | Preprocessing and nested CV (position imputer, stacked projector, pipelines); statistical tests (corrected t, Wilcoxon-Holm, Friedman, DeLong, bootstrap, mediation); MLflow (tracking, tags, registry); translation analysis; regression and classification; the 21 figures; the command line. |
| `training.ipynb` | The whole pipeline in Google Colab, step by step, with the evaluation tables and figures and the loading of the registered models. |

## Data

The ten original files are in `data/raw/`. The four small tables now arrive as Parquet; the six large ones
stay as gzip-compressed CSV because `player_play_3.csv` weighs 147 MB and GitHub does not accept files over
100 MB (compressed, the folder takes about 100 MB instead of 427 MB).

| File | Contents | Records |
| --- | --- | --- |
| `combine_results.parquet` | Official Combine results 2023–2025 | 510 prospects |
| `players.parquet` | Biographical and draft data | 510 |
| `player_career_successes.parquet` | Games played and started through 2025 | 510 |
| `games.parquet` | Schedule and game type | 1,002 games |
| `game_tracking_2023_5.csv.gz` | 10 Hz tracking, 2023 season (cut at a spreadsheet's row limit) | 1,048,575 |
| `game_tracking_2024_1_5.csv.gz` | Tracking, 2024 weeks 0–17 | 951,733 |
| `game_tracking_2024_2.csv.gz`, `game_tracking_2025_2.csv.gz` | Tracking, week 18 and playoffs of 2024 and 2025 | 96,842 and 97,147 |
| `player_play_3.csv.gz` | One row per player and play, 2023–2025 | 314,197 player-plays |
| `combine_tracking_6.csv.gz` | 10 Hz tracking of the Combine drills | 463,189 frames |

**Parquet, CSV or compressed CSV.** Every table can be delivered in any of the three formats, with no code
change:

- **Detection by content.** The reader looks at each file's first bytes (`PAR1` = Parquet, gzip header =
  compressed CSV, anything else = CSV), so a file with the wrong extension is still read correctly and the
  mismatch is logged.
- **Conversion to the CSV encodings.** Parquet keeps richer types than CSV text, so its columns are harmonised
  before anything else: booleans become `TRUE`/`FALSE`, dates and timestamps become ISO 8601 text and
  categorical columns become their plain values. After that step the data contract, the declared schema and
  the cleaning rules see exactly the same table whatever the format.
- **One file per table.** The config patterns (`players.*`, `game_tracking_*`, …) find any format. If a table
  is present twice, Parquet wins, then plain CSV, then its `.gz` copy; the others are logged and ignored. A
  split part (`<stem>_<n>`) is still ignored when the full file is present, whatever the formats.
- **Same results.** With the four tables in Parquet, the 12 clean data files are byte-identical to the ones
  v3.1.0 produced from CSV, and the 435 contract checks give the same results.

**Manifest.** `data/raw/manifest.json` records each file's SHA-256, size, rows and format, plus a combined
version `data_version = 261a80510390`. A `.csv.gz` is hashed decompressed, so compressing a CSV does not change
the data version; converting a table to Parquet does, because the bytes differ (v3.1.0 was `a75ba51ae4b9`).
`make verify` recomputes the hashes and fails on any difference.

**Data contract.** The provider's spec (`docs/Data_structure_CSV_files.docx`) is encoded in
`configs/data_contract.yaml` and checked on every run (`make validate`, also the first step of the data
stage). The report `reports/metrics/data_contract_report.csv` has one row per check, with its severity and the
handling the pipeline applies. Errors (missing required file or column, duplicate key, broken foreign key) stop
the run; deviations are warnings handled by the business rules. With the delivered files: **435 checks,
0 errors, 32 warnings.**

| Spec | Delivered | Effect |
| --- | --- | --- |
| `combine_tracking.csv`, 6,310 attempts | `combine_tracking_6.csv`: 6,310 attempts, 510 prospects | The 40 lane is outside the 0–53.3 yd frame; ball rows exceed a player's ranges and are dropped (R16) |
| `game_tracking_2023/2024/2025.csv`, full seasons | 2023 season, 2024 weeks 0–17, week 18 and playoffs of 2024 and 2025; 2025 weeks 1–17 are missing | 343 tracked prospects after R8b; the SBI uses 299 players with 3+ plays (24,828 player-plays) |
| Complete files | `game_tracking_2023_5.csv` has exactly 1,048,575 rows, a spreadsheet's limit | Almost certainly cut when saved; R8c drops the last (possibly partial) player-play and the contract warns |
| `player_play.csv`, 316,338 rows | `player_play_3.csv`, 314,197 rows (−0.68%) | Results use the rows delivered |
| REG/POST games, weeks 1–18 | Also PRE (147 games, weeks 0–3) and playoffs in weeks 19–23 | R10: on-field uses REG; R8b: tracking uses REG + POST |
| `down` 1–4 | 961 rows with down 0 (two-point tries) | R15 removes them (none belonged to a measured family) |
| 1.0/null flags | TRUE/FALSE text | The boolean reader accepts both |

## How to reproduce it

### Option 1: Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alcazarfran77-dev/NFL/blob/english/src/training.ipynb)

1. Open the notebook with the button above (or in colab.research.google.com: *File → Open notebook → GitHub*,
   paste `alcazarfran77-dev/NFL` and pick the `english` branch).
2. In the first cell, `REPO_URL` and `BRANCH` already point at this branch; choose `MODE = "smoke"` (about
   10 min, minimal grids, outputs in their own folders), `MODE = "full"` (the release run, 60 to 90 min on
   Colab's 2 cores) or `MODE = "load_only"` (shows the results and loads the models, 1–2 min).
3. Run everything (*Runtime → Run all*). The notebook clones the repository, installs the dependencies,
   validates the data, prepares the clean data, trains and evaluates, shows the tables and figures, and loads
   the registered models.

Without training anything, the notebook can also load the 17 models of the release run stored in `mlflow/`.

### Option 2: on your computer

Requires Python 3.13 (versions pinned in `requirements.txt`) and `make`.

```bash
git clone -b english https://github.com/alcazarfran77-dev/NFL.git && cd NFL
python -m pip install -r requirements.txt
make relocate     # once after cloning: points the MLflow store paths at this folder
make verify       # raw-data hashes vs the manifest, about 5 s
make validate     # data contract only, about 10 s
make test         # 40 unit tests, about 15 s
make smoke        # the whole pipeline with minimal grids, about 10 min (own folders, leaves the release alone)
make all          # release run: data, translation, models, statistics and figures, 60–90 min
make ui           # MLflow interface: runs, metrics and registered models
```

Each stage also runs on its own, in this order:

| Stage | Command | What it does | Writes |
| --- | --- | --- | --- |
| contract | `make validate` | Checks the ten raw files against the spec | `reports/metrics/data_contract_*` |
| data | `make data` | Contract, ingestion, manifest, cleaning (R1–R16), movement and on-field metrics | `data/clean/`, audits in `reports/metrics/` |
| translation | `make translation` | Drill-to-game and drill-to-field correlations, reliability, mediation, starter GLM | `reports/metrics/` |
| models | `make models` | Translation + nested CV of regression and classification, tests, final refit and registration | `reports/metrics/`, `mlflow/` |
| stats | `make stats` | Recomputes the omnibus tests from the per-fold scores, without retraining | `reports/metrics/` |
| figures | `make figures` | Draws the 21 figures from the saved metrics | `reports/figures/` |

`make relocate` rewrites the paths stored inside `mlflow/mlflow.db`, so git will show that file as modified.
It is a local change that does not need to be pushed; `git checkout mlflow/mlflow.db` reverts it.

Without `make`: `python src/train.py --stage all` (or `contract`, `data`, `translation`, `models`, `stats`,
`figures`, `relocate`), and `python src/data_prep.py --stage data|contract|verify`. Both accept
`--config configs/smoke.yaml`.

Everything is seeded (`seed: 42`); a test checks that two fits with the same seed give identical predictions,
and the full run reproduces the same numbers.

### Loading a registered model

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")       # from the repository root
clf = mlflow.sklearn.load_model("models:/cgmt-starter-classifier@champion")
info = mlflow.models.get_model_info("models:/cgmt-starter-classifier@champion")
print(info.signature)                                         # expected input columns
```

Models are stored together with `data_prep.py` and `train.py`, so they load on any machine with nothing more
than `requirements.txt`. The input is a DataFrame with `combine_position` plus the columns of the model's
feature set; missing drills are allowed (they are imputed inside the pipeline). The
[model card](docs/model_card.md) describes each of the 17 registered models.

## Design decisions

Each decision answers a problem that showed up in the data; the full reasoning is in the
[report](CHANGELOG.md) and the version history.

1. **The SBI is a shrunk mean, not a high percentile.** The first version used the 80th percentile of each
   player's plays ("capacity"). With few plays that percentile reads low, and since starters play more, the
   bias manufactured a correlation with starting (0.23 → 0.08 once corrected). The empirical-Bayes shrunk mean
   has no such bias; the 80th percentile remains only as a sensitivity analysis.
2. **The target is normalised within position.** With burst in yards, the position mean gave R² = 0.94 and hid
   the question that matters: within a position, does the Combine tell who moves better? The SBI (50 ± 10
   within position) forces that question.
3. **"Over expected" for the on-field metrics.** Pressure, separation and get-off depend on alignment, route and
   dropback duration. Each play is compared with the average of prospect plays in the same context, then
   shrunk by empirical Bayes.
4. **Nested cross-validation and conservative tests.** 5 folds × 3 repeats outside and a 3-fold GridSearchCV
   inside: hyperparameters never see the test fold. Because folds share players, the Nadeau and Bengio
   corrected t-test is used, with Wilcoxon, Holm, Friedman and DeLong as checks. A gain is only called
   significant if it passes the corrected test.
5. **No information leakage.** Position-median imputation and scaling are learned inside each fold; movement
   features are per-play averages, never counts (counts follow playing time); the NFL position is never a
   feature because it is decided after the draft.
6. **Champion and challenger.** The starter champion may only use what is known before the draft (Combine +
   draft capital). A new feature set replaces it only with a higher mean AUC and DeLong p < 0.05. Models with
   post-draft information (movement, on-field performance) stay as diagnostic challengers.
7. **Drill tracking is filtered by reliability.** Only metrics that repeat across a player's attempts
   (r ≥ 0.5) are used, one per construct; the tracked start of an attempt does not repeat and is dropped.
8. **An explicit data contract.** The provider's spec is checked on every run, so a different delivery (split
   files, extra columns, a cut file, a new format) is caught before it reaches a model and its handling is
   documented.
9. **Format-agnostic ingestion.** The pipeline reads Parquet, CSV and compressed CSV through one reader that
   detects the format from the content and harmonises Parquet to the CSV encodings, so a provider switching
   formats changes the data version but not a single cleaned value.
10. **Two modules and one configuration.** All preparation lives in `data_prep.py` and all modelling in
    `train.py`, as the structure asks; thresholds and grids live in `configs/config.yaml`, never in the code.
11. **Local MLflow on SQLite.** The model registry works with no server or connection and travels with the
    repository; `make relocate` fixes the absolute paths MLflow stores.

## Main results

| Question | Result |
| --- | --- |
| Does a player's snap burst repeat across plays? | Yes on the defensive line (ICC 0.63, mostly role) and at receiver (0.62, the player's own); weak for defensive backs (0.26) |
| Does the Combine forecast the SBI? | Defensive line: RMSE 7.85 vs 9.99 for position alone (−21%), R² 0.37, Holm p = 0.001. Receivers: −10%, moderate evidence. All positions: −5% |
| Which drills carry over? | The 40, the 10-yd split, the broad jump and the vertical (ρ 0.25–0.43 within position); the NGS composite and the bench press do not |
| Does the Combine forecast on-field performance? | DL get-off (R² 0.13) and WR/TE separation (0.19, through size); not pressure or pass protection |
| Does movement help predict starters? | Not after the draft: AUC 0.84 → 0.91 with draft capital; movement adds −0.003 to 0 (DeLong p ≥ 0.47) |
| Does drill tracking add anything? | No: it reproduces the stopwatch (ρ −0.97 with the official 40) and improves no model |

The conclusions for scouts, general managers and coaches, with the translation-gap tables, are in section 10 of
the [report](CHANGELOG.md#10-actionable-conclusions).

## Versioning and MLOps

- **Code:** git with semantic-version tags and every change described in [CHANGELOG.md](CHANGELOG.md).
  MAJOR = a metric or target definition changed; MINOR = new analysis, model or capability; PATCH = no number in
  `reports/metrics/` changes.
- **Data:** the manifest records each raw file's hash and format and the combined `data_version`; CI fails if a
  hash does not match. The data contract is versioned with the code.
- **Experiments:** every MLflow run carries `pipeline_version`, `data_version`, `git_commit` and `seed`, the
  per-fold scores, the out-of-fold predictions and the selected hyperparameters.
- **Models:** registered with aliases (`champion`, `challenger`), signature, input example and the tag
  `validation_status = nested-cv-passed`.
- **Continuous integration:** `.github/workflows/ci.yml` runs the tests, the hashes, the contract and the smoke
  run on every push to `main` or `english`.

## Limitations and pending data

- 2025 weeks 1–17 are missing from the game tracking, and `game_tracking_2023_5.csv` stops at 1,048,575 rows,
  so a random part of 2023 is missing. Delivering the uncut 2023 file and the 2025 regular season would sharpen
  the SBI models; the pipeline finds new files by pattern, in Parquet or CSV, and validates them with no code
  change.
- The tracking carries 1 to 4 players per play, with no opponents or ball, so movement is judged from the
  player's own kinematics.
- Samples per position are small (72 defensive linemen and 63 receivers in the SBI models).
- Defensive backs have no on-field metric: `player_play` carries no coverage outcomes.
- "Expected" is computed from the prospects' own plays, not from the whole league.

## Repository on GitHub

The project is published at https://github.com/alcazarfran77-dev/NFL with its full history: `main` holds the
Spanish v3.1.0 and `english` this version. To push changes from a local copy:

```bash
git add -A
git commit -m "description of the change"
git push origin english
```

No file exceeds GitHub's 100 MB limit; the largest is `player_play_3.csv.gz` at 41 MB. If the repository is
going to be public, first confirm you may publish the provider's data; if not, remove it from version control
(`git rm -r --cached data/raw`) and share it another way: the pipeline finds it the same way once it is copied
into `data/raw/`.
