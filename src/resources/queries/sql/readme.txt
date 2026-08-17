每月5號定期執行
將建置當月客戶特徵入庫(ds_sec)

特徵以PARTY_ID_MASK儲存

共6支程式執行
位置:桌面/新特徵排程

執行順序如下
insert_txn(一定要第一個執行)
insert_txn2
insert_profolio
insert_bnf
insert_dgt

另外執行
insert_interact



全部跑完後跑驗證看人數
此流程需在當月模型預測前完成


