select round(max_bytes/1024/1024/1024, 2) as個人可使用空間_MB , round(bytes/1024/1024/1024, 2) as 個人已使用空間比例 from user_ts_quotas;
SELECT BYTES / 1024 /1024/1024,A.* FROM USER_SEGMENTS A ORDER BY BYTES DESC;

drop table "TEMP_EXCEL_DATA"  purge;

select * from PRODUCT_CATEGORY_PARTYID_2502;

select * from MLOPS_ID_LIST_DOUBLE;

SELECT
    YYYYMM, SOURCE_SEGMENT,
    COUNT(*) AS ROW_COUNT
;
select distinct YYYYMM
FROM S_IANLEONG.MLOPS_POPULATION_SAMPLING_TEST
WHERE PRODUCT = '不限用途'
order by yyyymm
GROUP BY YYYYMM, SOURCE_SEGMENT
ORDER BY YYYYMM, SOURCE_SEGMENT;


-- Run once as S_IANLEONG for the existing isolated sampling test table.
ALTER TABLE MLOPS_POPULATION_SAMPLING_TEST
    DROP CONSTRAINT CK_MLOPS_SAMPLING_TEST_SEGMENT;

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

ALTER TABLE MLOPS_POPULATION_SAMPLING_TEST
    ADD CONSTRAINT CK_MLOPS_SAMP_TEST_SEGMENT_V2
    CHECK (SOURCE_SEGMENT IN ('潛客', '非潛客', '不分潛客'));
;

select * 
from MLOPS_POPULATION
where 潛在高價值客戶前季高交易量客戶P is not null;

select distinct yyyymm 
FROM S_IANLEONG.MLOPS_POPULATION p
order by yyyymm
where 流失預警Y = 0 and yyyymm = '202603';

select 0 as 流失預警Y
from DM_S_VIEW.T_DR_CUST_PROD_TXN_ST A
where snap_yyyymm between '202604' and '202703'
    and TXN_AMT_TWD>0
    and substr(PROD_ID,1,2) = '11';