-- Backfill two overseas-stock churn columns for the existing population months.
-- Source logic matches INSERTPOPULATION.sql (read as Big5/CP950).
-- Run ALTER only when the columns do not exist yet.

ALTER TABLE S_IANLEONG.MLOPS_POPULATION ADD (
    "海外股流失預警近一年實動" NUMBER,
    "海外股流失預警Y" NUMBER
);

MERGE INTO S_IANLEONG.MLOPS_POPULATION target
USING (
    WITH months (yyyymm) AS (
        SELECT '202305' FROM dual UNION ALL
        SELECT '202306' FROM dual UNION ALL
        SELECT '202307' FROM dual UNION ALL
        SELECT '202308' FROM dual UNION ALL
        SELECT '202309' FROM dual UNION ALL
        SELECT '202310' FROM dual UNION ALL
        SELECT '202311' FROM dual UNION ALL
        SELECT '202312' FROM dual UNION ALL
        SELECT '202401' FROM dual UNION ALL
        SELECT '202402' FROM dual UNION ALL
        SELECT '202403' FROM dual UNION ALL
        SELECT '202404' FROM dual UNION ALL
        SELECT '202405' FROM dual UNION ALL
        SELECT '202406' FROM dual UNION ALL
        SELECT '202407' FROM dual UNION ALL
        SELECT '202408' FROM dual UNION ALL
        SELECT '202409' FROM dual UNION ALL
        SELECT '202410' FROM dual UNION ALL
        SELECT '202411' FROM dual UNION ALL
        SELECT '202412' FROM dual UNION ALL
        SELECT '202501' FROM dual UNION ALL
        SELECT '202502' FROM dual UNION ALL
        SELECT '202503' FROM dual UNION ALL
        SELECT '202504' FROM dual UNION ALL
        SELECT '202505' FROM dual UNION ALL
        SELECT '202506' FROM dual UNION ALL
        SELECT '202507' FROM dual UNION ALL
        SELECT '202508' FROM dual UNION ALL
        SELECT '202509' FROM dual UNION ALL
        SELECT '202510' FROM dual UNION ALL
        SELECT '202511' FROM dual UNION ALL
        SELECT '202512' FROM dual UNION ALL
        SELECT '202601' FROM dual UNION ALL
        SELECT '202602' FROM dual UNION ALL
        SELECT '202603' FROM dual UNION ALL
        SELECT '202604' FROM dual
    ),
    account_map AS (
        SELECT DISTINCT party_id, party_id_mask
        FROM DM_S_VIEW.M_AC_ACCOUNT
    ),
    activity AS (
        SELECT /*+ MATERIALIZE */ DISTINCT
               account_map.party_id_mask AS customer_id,
               txn.snap_yyyymm
        FROM DM_S_VIEW.T_DR_CUST_PROD_TXN_ST txn
        INNER JOIN account_map
            ON txn.party_id = account_map.party_id
        WHERE txn.snap_yyyymm BETWEEN '202206' AND '202704'
          AND txn.txn_amt_twd > 0
          AND SUBSTR(txn.prod_id, 1, 2) = '21'
    )
    SELECT activity.customer_id,
           months.yyyymm,
           MAX(
               CASE
                   WHEN activity.snap_yyyymm BETWEEN
                        TO_CHAR(ADD_MONTHS(TO_DATE(months.yyyymm || '01', 'YYYYMMDD'), -11), 'YYYYMM')
                        AND months.yyyymm
                   THEN 1
               END
           ) AS active_in_previous_year,
           MIN(
               CASE
                   WHEN activity.snap_yyyymm BETWEEN
                        TO_CHAR(ADD_MONTHS(TO_DATE(months.yyyymm || '01', 'YYYYMMDD'), 1), 'YYYYMM')
                        AND TO_CHAR(ADD_MONTHS(TO_DATE(months.yyyymm || '01', 'YYYYMMDD'), 12), 'YYYYMM')
                   THEN 0
               END
           ) AS active_in_next_year
    FROM months
    INNER JOIN activity
        ON activity.snap_yyyymm BETWEEN
           TO_CHAR(ADD_MONTHS(TO_DATE(months.yyyymm || '01', 'YYYYMMDD'), -11), 'YYYYMM')
           AND TO_CHAR(ADD_MONTHS(TO_DATE(months.yyyymm || '01', 'YYYYMMDD'), 12), 'YYYYMM')
    GROUP BY activity.customer_id, months.yyyymm
) source
ON (
    target.customer_id = source.customer_id
    AND target.yyyymm = source.yyyymm
)
WHEN MATCHED THEN UPDATE SET
    target."海外股流失預警近一年實動" = source.active_in_previous_year,
    target."海外股流失預警Y" = source.active_in_next_year;

COMMIT;
