# Separate persisted populations from ephemeral pretraining

The pipeline persists the full population in `S_IANLEONG.MLOPS_POPULATION` and the deterministic monthly training population in a configurable sampling table, currently `S_IANLEONG.MLOPS_POPULATION_SAMPLING_TEST`. The 100,000-row pretraining cohort is recreated deterministically during retraining. Each training-population row snapshots `CUSTOMER_ID`, `YYYYMM`, `PRODUCT`, `SOURCE_SEGMENT`, `Y`, and `ELIGIBILITY_TAG`, so retraining never rejoins mutable population labels. The model population name is configuration, not persisted row data.

All runtime reads and writes use the single `DS_MASK` database identity. Cross-schema grants provide access to `S_IANLEONG` population tables and `DS_SEC` feature tables. The sampling table's schema and name are configuration, allowing isolated test and production snapshots without changing credentials or query logic.

## Consequences

Schema DDL is performed separately by table owners; application runtime commands do not create or grant cross-schema objects. Full-population refresh, training-population sampling, and retraining remain separate commands. Pretraining remains ephemeral, feature tables are left-joined to preserve cohort rows, and retraining fails when required upstream months are absent.
