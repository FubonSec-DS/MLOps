# 資料表建置指南

## 概述

本專案需要建立以下兩類資料表：
1. **特徵表 (CF_*)** - 客戶特徵資料，共 40+ 張表
2. **母體表 (MLOPS_POPULATION)** - 模型訓練/預測的目標客戶名單

特徵表位於 `DS_SEC`，完整母體固定於 `S_IANLEONG`，目前測試抽樣表為 `S_IANLEONG.MLOPS_POPULATION_SAMPLING_TEST`。應用程式所有查詢與寫入都使用已取得跨 schema 權限的 `DS_MASK`。

---

## 執行順序

### Step 1: 建立所有資料表結構

特徵表 DDL 由 SQL Developer 執行：

```sql
@createtable.sql
@ensure_feature_uniqueness.sql
```

`setup_population_table.sql` 僅保留為表格結構參考。建表、constraint 與跨 schema grant 必須由各表 owner／DBA 另外執行，不由 DS_MASK runtime 指令處理。

由 S_IANLEONG owner／DBA 執行 `create_population_sampling_test.sql` 建立隔離測試表。該檔案同時提供 DS_MASK 以下 runtime 權限：

```sql
GRANT SELECT, INSERT
ON S_IANLEONG.MLOPS_POPULATION_SAMPLING_TEST TO DS_MASK;

GRANT SELECT, INSERT, DELETE
ON S_IANLEONG.MLOPS_POPULATION TO DS_MASK;
```

若測試表已經存在，先由 S_IANLEONG owner 執行 `allow_combined_sampling_segment.sql`，讓特殊商品可將客群標記保存為「不分潛客」。

完整母體仍採月份 replace，所以需要 `DELETE`。訓練抽樣表採 append-only，只需要 `SELECT, INSERT`；既有快照不會由 runtime 修改或刪除。

若未來要沿用既有四欄正式表，才需要由 table owner 執行：

```sql
@migrate_population_sampling.sql
```

舊資料無法可靠補回 Y 與資格標籤，遷移後必須重新建立需要的產品月份快照。

### Step 2: 載入特徵資料 (每月執行)

**執行時間**: 每月 5 號  
**執行順序**: 必須按以下順序執行！

| 順序 | 腳本 | 說明 |
|------|------|------|
| 1 | `insert_txn.sql` | 交易特徵 (必須第一個執行) |
| 2 | `insert_txn2.sql` | 交易特徵 2 |
| 3 | `insert_profolio.sql` | 投資組合特徵 |
| 4 | `insert_bnf.sql` | 收益特徵 |
| 5 | `insert_dgt.sql` | 數位互動特徵 |
| 6 | `insert_interact.sql` | 互動行為特徵 |

這些 SQL 使用 `:ym` bind variable；每次執行都會先刪除同月份資料再重新寫入：

```powershell
uv run python -m scripts.refresh_features --ym 202607
```

### Step 3: 載入母體資料 (每月執行)

`refresh_population_table.sql` 使用 `:ym` bind variable，月份衍生運算由 Oracle SQL 完成。
透過以下指令執行：

```powershell
uv run python -m scripts.refresh_population --ym 202607
uv run python -m scripts.refresh_population --ym 202607 --replace
```

### Step 4: 驗證資料

```sql
@驗證.sql
```

---

## 資料表清單

### 特徵表 (共 47 張)

| 類別 | 資料表名稱 | 說明 |
|------|------------|------|
| 客戶ID | CF_CUSTID | 客戶基本資訊 |
| 交易 | CF_TXN_STST | 台股交易 |
| 交易 | CF_TXN_FS | 基金交易 |
| 交易 | CF_TXN_STMT | 台股融資融券 |
| 交易 | CF_TXN_STSS | 台股當沖 |
| 交易 | CF_TXN_SIP | 定期定額 |
| 交易 | CF_TXN_STDT | 台股信用交易 |
| 交易 | CF_TXN_LOAN | 貸款 |
| 交易 | CF_TXN_BSBL | 借貸 |
| 交易 | CF_TXN_FU | 期貨 |
| 交易 | CF_TXN_INSURANCE | 保險 |
| 交易 | CF_TXN_SN | 結構型商品 |
| 交易 | CF_TXN_FB | 債券基金 |
| 交易 | CF_TXN_FD | 定存 |
| 交易 | CF_TXN_AP | 全產品 |
| 交易 | CF_TXN_CURRENCY | 外幣 |
| 交易 | CF_TXN_MAEC | 複委託 |
| 交易 | CF_TXN_STYPE | 交易類型 |
| 交易 | CF_TXN_FPBR | 複委託收益 |
| 資產 | CF_AUM | 資產規模 |
| 資產 | CF_ASSET1~4 | 資產配置 |
| 收益 | CF_ACTUBNF | 實際收益 |
| 收益 | CF_ACTUBNFROI | 實際報酬率 |
| 收益 | CF_INVBNF | 投資收益 |
| 收益 | CF_INVBNFROI | 投資報酬率 |
| 客戶屬性 | CF_PROFILE | 客戶輪廓 |
| 投資組合 | CF_PROFOLIO1~7 | 投資組合特徵 |
| 互動 | CF_INTERACT_ECDAY | EC 互動天數 |
| 互動 | CF_INTERACT_ECPROD | EC 產品互動 |
| 互動 | CF_INTERACT_EDM | EDM 互動 |
| 互動 | CF_INTERACT_LINE | LINE 互動 |
| 合約 | CF_CONTRACT | 合約 |
| 合約 | CF_CONTRACT_DUE | 到期合約 |
| 事件 | CF_EVENT | 客戶事件 |
| KYC | CF_KYCQA | KYC 問卷 |
| 營業員 | CF_JCI | 營業員關係 |
| 數位 | CF_DGT1, CF_DGT2 | 數位行為 |

### 母體表

| 資料表名稱 | 說明 |
|------------|------|
| MLOPS_POPULATION | 模型訓練/預測母體 |
| MLOPS_POPULATION_SAMPLING_TEST | 隔離測試用、含來源客群、Y 與資格標籤的訓練母體快照 |

---

## 資料來源

特徵資料來自以下 View (需確認權限)：

- `DM_S_VIEW.M_PT_CUSTOMER` - 客戶主檔
- `DM_S_VIEW.M_AC_ACCOUNT` - 帳戶資料
- `DM_S_VIEW.M_AT_STOCK_TXN` - 股票交易
- `DM_S_VIEW.M_AT_FUND_TXN` - 基金交易
- `DM_S_VIEW.M_AT_INSURANCE_TXN` - 保險交易
- `DM_S_VIEW.M_AT_LOAN_TXN` - 貸款交易
- `DM_S_VIEW.T_DR_ACCT_SEG` - 客戶分群
- 等等...

---

## 注意事項

1. **編碼問題**: SQL 腳本使用 Big5 編碼，若看到亂碼請用正確編碼開啟
2. **Schema 修改**: 執行前需將腳本中的 `DS_SEC` 改為 `DS_MASK`
3. **權限**: 確認對 `DM_S_VIEW.*` 有 SELECT 權限
4. **分區**: 特徵表使用月份分區 (PARTITION)，格式為 `P_YYYYMM`

---

## 快速開始

```sql
-- 1. 先確認你有權限存取資料來源
SELECT COUNT(*) FROM DM_S_VIEW.M_PT_CUSTOMER WHERE ROWNUM <= 1;

-- 2. 建立表結構
@createtable.sql

-- 3. 載入當月資料
@insert_txn.sql
-- ... 依序執行其他 insert 腳本

-- 4. 驗證
SELECT table_name, num_rows 
FROM user_tables 
WHERE table_name LIKE 'CF_%';
```
