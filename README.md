# MLOps staged training pipeline

This project prepares Oracle features and populations, selects Top-100 features with an ephemeral pretraining cohort, and trains an XGBoost model with temporal validation and test periods.

## Setup

Requirements: Python 3.12+, `uv`, and Oracle Instant Client. Copy `configs/.env.example` to `configs/.env`, supply the DS_MASK password, then run:

```powershell
uv sync
```

`DS_MASK` is the only runtime database identity. It reads and writes cross-schema objects through grants: the full population is fixed at `S_IANLEONG.MLOPS_POPULATION`, features are under `DS_SEC`, and the training-population snapshot currently uses `S_IANLEONG.MLOPS_POPULATION_SAMPLING_TEST`.

## Configuration

Non-secret settings are stored in three validated YAML files:

- `configs/config.yaml`: Oracle schemas, local paths, shared model settings, and feature metadata.
- `configs/products.yaml`: product catalog, XGBoost profiles, training schedules, sample sizes, bins, and acceptance thresholds.
- `configs/scripts.yaml`: inputs for each independently executable script.

`src/common/config.py` and `src/products/config.py` read these files and calculate runtime values such as absolute paths, source-qualified feature names, product columns, and date periods.

Credentials and tokens must not be added to YAML or Python. Keep the DS_MASK credentials in `configs/.env`, CI/CD secrets, or a secret manager. The sampling schema and table name are non-secret settings in `configs/config.yaml`; changing them moves every sampling read and write together.

Sampling and prediction read `S_IANLEONG.MLOPS_POPULATION`. Pretraining and formal training read the configured sampling snapshot table. Every runtime SELECT, INSERT, UPDATE, and DELETE uses DS_MASK.

## Independent stages

Table-owner DDL and cross-schema grants are administered separately from application runtime commands. `src/queries/SQLs/setup_population_table.sql` remains a schema-definition reference and must not be run through DS_MASK.

DS_MASK requires direct `SELECT, INSERT, DELETE` grants on both `S_IANLEONG.MLOPS_POPULATION` and the configured sampling table. See `src/queries/SQLs/README.md` for owner/DBA statements. `ALTER` alone is insufficient for the transactional replace workflows.

## Architecture

- `configs`: validated environment, product, and training configuration.
- `src/common`: shared Oracle infrastructure.
- `src/resources`: feature, population, sampling, dataset-loading workflows, query builders, and SQL assets.
- `src/models`: preprocessing, training, evaluation, prediction, and artifact persistence.
- `src/pipelines`: retraining and prediction orchestration.
- `scripts`: thin, YAML-configured executable entry points.

Application modules under `src` use the `src.*` namespace. Root-level packages use
`configs.*` and `scripts.*`.

Set `refresh_features` in `configs/scripts.yaml`, then refresh all DS_SEC feature groups. Set
`groups` to a list to refresh only selected groups:

```powershell
uv run python -m scripts.refresh_features
```

Validate the sampled population/features join and preprocessing immediately before
pretrain XGBoost, without fitting or writing a model:

```powershell
uv run python -m scripts.validate_pretrain_readiness
```

This command also creates `pretrain_reports/{product}/{population}/{yyyymm}/`. Open
`README.md` in that directory first. The report records the concrete Python class types and
runtime values, column-level dtypes/NULL counts/examples, the preprocessing vocabulary, and
five real rows both before and after preprocessing. Change `sample_rows` or `output_dir` in the
corresponding YAML section when needed. The sample CSVs contain customer
IDs and feature values, so `pretrain_reports/` is ignored by Git.

Run only the ephemeral pretrain feature-selection stage (the OOT month is derived but excluded):

```powershell
uv run python -m scripts.run_pretrain
```

The output contains `selected_features.json`, the complete `feature_importance.csv`,
`preprocessing_schema.json`, and `pretrain_summary.json`. The pretrain estimator itself is
not persisted.
`rows_per_month` is an optional development limit so the complete flow can run on a small
machine; rankings from a limited run are illustrative. Omit it in the production deployment
to use the complete configured pretrain cohort.

Refresh a mature full-population snapshot without changing the approved business SQL.
The safe form takes the latest complete label-data month and derives the snapshot using the
selected product's configured label horizon:

```powershell
uv run python -m scripts.refresh_population

# Churn uses its configured 12-month label horizon, so this derives snapshot 202506.
uv run python -m scripts.refresh_population
```

Both `as_of_ym` and `product` are required in the YAML section. The snapshot
offset comes from the product's validated configuration, and the command always atomically
replaces that snapshot month. A snapshot row can contain columns for several products, but
only product labels whose full observation window ends by the as-of month are mature.

Replace the sampling table with every deterministic training-population snapshot needed for
model training. The command derives each product's training and validation months from its
configured schedule, then samples every configured product and population:

```powershell
uv run python -m scripts.refresh_training_population
```

To validate one product first:

```powershell
uv run python -m scripts.refresh_training_population
```

`ym` is the latest schedule anchor. It is used to derive the required months but is not
inserted into the training population. Rebuilding is atomic: the existing
The configured sampling table rows are deleted and all required snapshots are appended in the
same transaction. A failure rolls the entire replacement back.

`MLOPS_POPULATION.SEGMENT` is the physical source classification and contains only `潛客` or `非潛客`. A model `population` is configured as a set of those source segments: `不分潛客` contains both. The sampling snapshot stores `SOURCE_SEGMENT`, `Y`, and `ELIGIBILITY_TAG`; it does not store the model population name.

By default, `run_retrain` uses `product: null` and `population: null` in
`configs/scripts.yaml`. This runs every configured population for every product, deriving
each product's periods from its own training schedule:

```powershell
uv run python -m scripts.run_retrain
```

For a test run, set `product` and `population` to one name. Either selector also accepts a
YAML list for a subset. Explicit `train_yms` and `backtest_ym` apply to every selected job;
leave them null for product-specific schedule calculation.

Alternatively, set `as_of_ym` and leave `train_yms`/`backtest_ym` null to derive the product's model-building and OOT backtest months. When `as_of_ym` is null, the latest complete data month defaults to the previous calendar month.

Set `write_db: true` only when the existing configured-schema retrain/model log tables should be updated.

The pretraining cohort is not persisted. For each training month it keeps every `y=1` row and deterministically samples `y=0` rows up to the configured 100,000-row target. The OOT backtest month never participates in Top-100 selection or model fitting.

Create the isolated test snapshot with `src/queries/SQLs/create_population_sampling_test.sql` as `S_IANLEONG`, then grant DS_MASK as shown in that file. Existing legacy tables are not modified. Run `src/queries/SQLs/ensure_feature_uniqueness.sql` once as the `DS_SEC` owner after resolving any historical duplicate keys.

## Artifacts

Each `model_store/{product}/{population}/{edition}/` directory contains:

- `model.pickle`
- `features.json`
- `feature_importance.csv`
- `preprocessing.json`
- `training_summary.json`

Prediction reads this contract and queries only the selected features:

```powershell
uv run python -m scripts.run_prediction
```

## Feature handling

- Feature names are source-qualified as `{table}__{column}`.
- Numeric NULL values remain missing; configured categorical NULL and unknown values are distinct.

Run local checks with:

```powershell
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run pyright
```
