-- Run once as the DS_SEC owner after resolving any duplicate keys reported
-- by this script. Monthly feature refreshes replace the month before insert;
-- these constraints protect the one-row-per-customer-month contract.

DECLARE
    duplicate_count NUMBER;
    constraint_count NUMBER;
    desired_constraint_name VARCHAR2(128);
BEGIN
    FOR feature_table IN (
        SELECT COLUMN_VALUE AS table_name
        FROM TABLE(SYS.ODCIVARCHAR2LIST(
            'CF_CUSTID',
            'CF_TXN_STST', 'CF_TXN_FS', 'CF_TXN_STMT', 'CF_TXN_STSS',
            'CF_TXN_SIP', 'CF_TXN_STDT', 'CF_TXN_LOAN', 'CF_TXN_BSBL',
            'CF_TXN_FU', 'CF_TXN_INSURANCE', 'CF_TXN_SN', 'CF_TXN_FB',
            'CF_TXN_FD', 'CF_TXN_AP', 'CF_AUM', 'CF_PROFILE',
            'CF_TXN_CURRENCY', 'CF_TXN_MAEC', 'CF_TXN_STYPE', 'CF_TXN_FPBR',
            'CF_PROFOLIO1', 'CF_PROFOLIO2', 'CF_PROFOLIO3', 'CF_PROFOLIO4',
            'CF_PROFOLIO5', 'CF_PROFOLIO6', 'CF_PROFOLIO7',
            'CF_ASSET1', 'CF_ASSET2', 'CF_ASSET3', 'CF_ASSET4',
            'CF_ACTUBNF', 'CF_ACTUBNFROI', 'CF_INVBNF', 'CF_INVBNFROI',
            'CF_DGT1', 'CF_DGT2',
            'CF_INTERACT_ECDAY', 'CF_INTERACT_ECPROD', 'CF_INTERACT_EDM',
            'CF_INTERACT_LINE', 'CF_CONTRACT', 'CF_CONTRACT_DUE', 'CF_EVENT',
            'CF_KYCQA', 'CF_JCI'
        ))
    ) LOOP
        EXECUTE IMMEDIATE
            'SELECT COUNT(*) FROM (' ||
            'SELECT 1 FROM ' || feature_table.table_name ||
            ' GROUP BY CUSTOMER_ID, YYYYMM HAVING COUNT(*) > 1)'
            INTO duplicate_count;

        IF duplicate_count > 0 THEN
            RAISE_APPLICATION_ERROR(
                -20001,
                feature_table.table_name || ' contains ' || duplicate_count ||
                ' duplicate CUSTOMER_ID/ YYYYMM keys'
            );
        END IF;

        desired_constraint_name := 'UQ_' || feature_table.table_name;
        SELECT COUNT(*)
        INTO constraint_count
        FROM USER_CONSTRAINTS
        WHERE CONSTRAINT_NAME = desired_constraint_name;

        IF constraint_count = 0 THEN
            EXECUTE IMMEDIATE
                'ALTER TABLE ' || feature_table.table_name ||
                ' ADD CONSTRAINT ' || desired_constraint_name ||
                ' UNIQUE (CUSTOMER_ID, YYYYMM)';
        END IF;
    END LOOP;
END;
/
