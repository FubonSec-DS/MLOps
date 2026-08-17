-- Run as S_IANLEONG for the existing isolated sampling test table.
-- This block makes the migration safe to run again after a partial execution.
BEGIN
    FOR existing_constraint IN (
        SELECT CONSTRAINT_NAME
        FROM USER_CONSTRAINTS
        WHERE TABLE_NAME = 'MLOPS_POPULATION_SAMPLING_TEST'
          AND CONSTRAINT_NAME IN (
              'CK_MLOPS_SAMPLING_TEST_SEGMENT',
              'CK_MLOPS_SAMP_TEST_SEGMENT_V2'
          )
    ) LOOP
        EXECUTE IMMEDIATE
            'ALTER TABLE MLOPS_POPULATION_SAMPLING_TEST DROP CONSTRAINT '
            || existing_constraint.CONSTRAINT_NAME;
    END LOOP;
END;
/

ALTER TABLE MLOPS_POPULATION_SAMPLING_TEST
    ADD CONSTRAINT CK_MLOPS_SAMP_TEST_SEGMENT_V2
    CHECK (SOURCE_SEGMENT IN ('潛客', '非潛客', '不分潛客'));
