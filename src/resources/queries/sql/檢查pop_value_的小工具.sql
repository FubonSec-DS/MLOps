WITH params AS (
    SELECT '202603' AS ym FROM dual
),
base AS (
    SELECT p.*
    FROM S_IANLEONG.MLOPS_POPULATION p
    CROSS JOIN params x
    WHERE p.YYYYMM = x.ym
)

,
checks AS (
    SELECT
        '流失預警' AS product,
        'TRAIN' AS stage,
        "流失預警近一年實動" AS eligibility_value,
        "流失預警Y" AS target_value
    FROM base

    UNION ALL

    SELECT
        '海外股流失預警',
        'TRAIN',
        "海外股流失預警近一年實動",
        "海外股流失預警Y"
    FROM base

    UNION ALL

    SELECT
        '客群上送',
        'TRAIN_R',
        "客群上送前季高交易量客戶R",
        "客群上送Y"
    FROM base

    UNION ALL

    SELECT
        '客群上送',
        'PREDICT_P',
        "客群上送前季高交易量客戶P",
        "客群上送Y"
    FROM base

    UNION ALL

    SELECT
        '潛在高價值客戶',
        'TRAIN_R',
        "潛在高價值客戶前季高交易量客戶R",
        "潛在高價值客戶Y"
    FROM base

    UNION ALL

    SELECT
        '潛在高價值客戶',
        'PREDICT_P',
        "潛在高價值客戶前季高交易量客戶P",
        "潛在高價值客戶Y"
    FROM base
),
summary AS (
    SELECT
        product,
        stage,
        NVL(TO_CHAR(eligibility_value), '(NULL)') AS eligibility_value,
        NVL(TO_CHAR(target_value), '(NULL)') AS target_value,
        COUNT(*) AS row_count
    FROM checks
    GROUP BY
        product,
        stage,
        NVL(TO_CHAR(eligibility_value), '(NULL)'),
        NVL(TO_CHAR(target_value), '(NULL)')
)
SELECT
    product,
    stage,
    eligibility_value,
    target_value,
    row_count,
    ROUND(
        row_count / SUM(row_count) OVER (PARTITION BY product, stage),
        6
    ) AS row_ratio
FROM summary
ORDER BY
    product,
    stage,
    eligibility_value,
    target_value;
