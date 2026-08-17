-- Schema-definition reference only. Run equivalent DDL as the relevant table owner/DBA.
-- Application runtime uses DS_MASK and must not execute this file.
DECLARE
    table_count NUMBER;
    query_user VARCHAR2(128) := DBMS_ASSERT.SIMPLE_SQL_NAME(:query_user);
BEGIN
    SELECT COUNT(*)
    INTO table_count
    FROM USER_TABLES
    WHERE TABLE_NAME = 'MLOPS_POPULATION';

    IF table_count = 0 THEN
        EXECUTE IMMEDIATE q'~
            CREATE TABLE MLOPS_POPULATION (
                CUSTOMER_ID VARCHAR2(32),
                YYYYMM VARCHAR2(6),
                SEGMENT VARCHAR2(32) NOT NULL,
                "98戶" NUMBER(1),
                "不限用途近一年舊戶" NUMBER(1),
                "不限用途Y" NUMBER(1),
                "保險商品近一年舊戶" NUMBER(1),
                "保險商品Y" NUMBER(1),
                "債券型基金近一年舊戶" NUMBER(1),
                "債券型基金Y" NUMBER(1),
                "儲蓄型保險商品近一年舊戶" NUMBER(1),
                "儲蓄型保險商品Y" NUMBER(1),
                "台股信用交易近一年舊戶" NUMBER(1),
                "台股信用交易Y" NUMBER(1),
                "台股定期定額近一年舊戶" NUMBER(1),
                "台股定期定額Y" NUMBER(1),
                "基金近一年舊戶" NUMBER(1),
                "基金Y" NUMBER(1),
                "基金定期定額近一年舊戶" NUMBER(1),
                "基金定期定額Y" NUMBER(1),
                "境內結構型近三年舊戶" NUMBER(1),
                "境內結構型Y" NUMBER(1),
                "境外結構型近三年舊戶" NUMBER(1),
                "境外結構型Y" NUMBER(1),
                "客群上送前季高交易量客戶R" NUMBER(1),
                "客群上送前季高交易量客戶P" NUMBER(1),
                "客群上送Y" NUMBER(1),
                "平衡型基金近一年舊戶" NUMBER(1),
                "平衡型基金Y" NUMBER(1),
                "投資型保險商品近一年舊戶" NUMBER(1),
                "投資型保險商品Y" NUMBER(1),
                "期貨近一年舊戶" NUMBER(1),
                "期貨Y" NUMBER(1),
                "流失預警近一年實動" NUMBER(1),
                "流失預警Y" NUMBER(1),
                "海外股流失預警近一年實動" NUMBER(1),
                "海外股流失預警Y" NUMBER(1),
                "海外債近一年舊戶" NUMBER(1),
                "海外債Y" NUMBER(1),
                "海外股票近半年舊戶" NUMBER(1),
                "海外股票Y" NUMBER(1),
                "海外股票定期定額近一年舊戶" NUMBER(1),
                "海外股票定期定額Y" NUMBER(1),
                "潛在高價值客戶前季高交易量客戶R" NUMBER(1),
                "潛在高價值客戶前季高交易量客戶P" NUMBER(1),
                "潛在高價值客戶Y" NUMBER(1),
                "結構型商品近三年舊戶" NUMBER(1),
                "結構型商品Y" NUMBER(1),
                "股票型基金近一年舊戶" NUMBER(1),
                "股票型基金Y" NUMBER(1),
                "財管商品近一年舊戶" NUMBER(1),
                "財管商品Y" NUMBER(1),
                "雙向借券近一年舊戶" NUMBER(1),
                "雙向借券Y" NUMBER(1),
                CONSTRAINT CK_MLOPS_POP_SEGMENT CHECK (SEGMENT IN ('潛客', '非潛客'))
            )
        ~';
    END IF;

    SELECT COUNT(*)
    INTO table_count
    FROM USER_TABLES
    WHERE TABLE_NAME = 'MLOPS_POPULATION_SAMPLING';

    IF table_count = 0 THEN
        EXECUTE IMMEDIATE q'~
            CREATE TABLE MLOPS_POPULATION_SAMPLING (
                CUSTOMER_ID VARCHAR2(32) NOT NULL,
                YYYYMM VARCHAR2(6) NOT NULL,
                PRODUCT VARCHAR2(128) NOT NULL,
                SOURCE_SEGMENT VARCHAR2(32) NOT NULL,
                Y NUMBER(1) NOT NULL,
                ELIGIBILITY_TAG NUMBER(1) NOT NULL,
                CONSTRAINT CK_MLOPS_SAMPLING_SEGMENT
                    CHECK (SOURCE_SEGMENT IN ('潛客', '非潛客', '不分潛客')),
                CONSTRAINT UQ_MLOPS_POP_SAMPLING
                    UNIQUE (YYYYMM, PRODUCT, SOURCE_SEGMENT, CUSTOMER_ID)
            )
        ~';
    END IF;

    EXECUTE IMMEDIATE 'GRANT SELECT, INSERT, DELETE ON MLOPS_POPULATION TO ' || query_user;
    EXECUTE IMMEDIATE 'GRANT SELECT, INSERT, DELETE ON MLOPS_POPULATION_SAMPLING TO ' || query_user;
END;
