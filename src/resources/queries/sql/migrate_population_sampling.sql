-- The old four-column rows cannot be backfilled reliably because Y and the
-- eligibility tag were not captured when sampling occurred. Regenerate the
-- required product/month snapshots after running this one-time migration.

-- Reference SQL for a table owner migrating an existing legacy production
-- table. The isolated TEST table should instead be created from
-- create_population_sampling_test.sql.

DELETE FROM MLOPS_POPULATION_SAMPLING;
COMMIT;

ALTER TABLE MLOPS_POPULATION_SAMPLING
    DROP CONSTRAINT UQ_MLOPS_POP_SAMPLING KEEP INDEX;

ALTER TABLE MLOPS_POPULATION_SAMPLING DROP COLUMN POPULATION;

ALTER TABLE MLOPS_POPULATION_SAMPLING ADD (
    SOURCE_SEGMENT VARCHAR2(32) NOT NULL,
    Y NUMBER(1) NOT NULL,
    ELIGIBILITY_TAG NUMBER(1) NOT NULL
);

ALTER TABLE MLOPS_POPULATION_SAMPLING
    ADD CONSTRAINT CK_MLOPS_SAMPLING_SEGMENT
    CHECK (SOURCE_SEGMENT IN ('潛客', '非潛客', '不分潛客'));

-- Run this final index-backed constraint as the table owner/DBA. A cross-schema
-- object-level ALTER grant does not allow DS_MASK to create the backing index.
ALTER TABLE MLOPS_POPULATION_SAMPLING
    ADD CONSTRAINT UQ_MLOPS_POP_SAMPLING
    UNIQUE (YYYYMM, PRODUCT, SOURCE_SEGMENT, CUSTOMER_ID)
    USING INDEX COMPRESS 3;
