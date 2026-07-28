#!/usr/bin/env python

# In[1]:


# DB parameter
#     IP = '10.81.25.191' # IP for ods
#     PORT = '5211'
#     DATABASE = 'fbsods'

HOST = '10.81.126.24' # IP for ods
PORT = '5211'
SEVICE_NAME = 'SNDMP_USER_DS'


# In[2]:


# config parameter

#account, pwd = 'S_ALEXWCCHUNG', '@Fubon123' #sql
account, pwd = 'S_KRISYJCHEN', '!Fubon003'
account_yichieh, pwd_yichieh = 'S_LOUISYTLU', '2025Lu$000'
account_kris, pwd_kris = 'S_KRISYJCHEN', '!Fubon003'
iaccount , ipwd= "S_IANLEONG" , "Fubon#3333"

feature_file_path_fubon = '/home/cdsw/Features/new_features'
feature_file_path_jihsun = '/home/cdsw/Features/new_features_js'
feature_file_path_mlops = '/home/cdsw/Features/new_features_mlops'
feature_file_path_ntb = '/home/cdsw/Features/new_features_ntb'

feature_trans_path = '/home/cdsw/Features/new_features/feature_translate.csv'

dump_feature_log_path = '/home/cdsw/Features/new_features_log'



object_list = ['customer_id','yyyymm']

table_orginal_list = ['CUST', 'AP', 'BC', 'FD', 'FDEBT', 'FS', 'FU', 'LIFEINS'
                     , 'SIP', 'STDT', 'STMT', 'STRUPRO', 'STSS', 'STST', 'UU'
                     , 'INVBNF', 'BNF', 'AUM'
                       ,'FTBD_profile'
                     ]

table_append_list = ['FTBD_PROFOLIO1', 'FTBD_PROFOLIO2', 'FTBD_PROFOLIO3'
                     ,'FTBD_ACTUBNF', 'FTBD_INVBNF'
                     ,'EDM_INTERACT', 'FTBD_INTERACT', 'LINE_INTERACT'
                     ,'TRANS_CATE1', 'TRANS_CATE2', 'TRANS_CATE3', 'TRANS_CATE31', 'TRANS_CATE32', 'TRANS_CATE4','TRANS_CATE5'
                     ,'SAFETY_STOCK', 'SAFETY_STOCK_1', 'SAFETY_STOCK_2'
                     ]


table_append_list_2024 = [ 'CUST', 'AP', 'BC', 'FD', 'FDEBT', 'FS', 'FU', 'LIFEINS'
                          , 'SIP', 'STDT', 'STMT', 'STRUPRO', 'STSS', 'STST', 'UU'
                          , 'AUM'
                          ,'FTBD_profile'
                          , 'FTBD_PROFOLIO1', 'FTBD_PROFOLIO2', 'FTBD_PROFOLIO3'
                          , 'FTBD_ACTUBNF'
                          , 'SAFETY_STOCK_1', 'SAFETY_STOCK_2'
                          , 'TRANS_CATE1', 'TRANS_CATE2', 'TRANS_CATE31', 'TRANS_CATE32', 'TRANS_CATE4', 'TRANS_CATE5'
                          , 'LINE_INTERACT','EDM_INTERACT','FTBD_INTERACT'
                          , 'FTBD_CONTRACT1','FTBD_CONTRACT2','FTBD_EVENT','FTBD_KYCQA','FTBD_INVBNF_PLUS'
                          , 'FTBD_JCI','FTBD_DGT1'
                          ]

table_append_list_2025 = ['CF_CUSTID','CF_ACTUBNF', 'CF_ACTUBNFROI', 'CF_ASSET1', 'CF_ASSET2', 'CF_ASSET3', 'CF_ASSET4', 'CF_AUM',
                         'CF_CONTRACT', 'CF_CONTRACT_DUE', 'CF_DGT1', 'CF_DGT2', 'CF_EVENT',
                         'CF_INTERACT_ECDAY', 'CF_INTERACT_ECPROD', 'CF_INTERACT_EDM', 'CF_INTERACT_LINE',
                         'CF_INVBNF', 'CF_INVBNFROI', 'CF_JCI', 'CF_KYCQA', 'CF_PROFILE',
                         'CF_PROFOLIO1', 'CF_PROFOLIO2', 'CF_PROFOLIO3', 'CF_PROFOLIO4', 'CF_PROFOLIO5', 'CF_PROFOLIO6', 'CF_PROFOLIO7',
                         'CF_TXN_AP', 'CF_TXN_BSBL', 'CF_TXN_CURRENCY', 'CF_TXN_FB', 'CF_TXN_FD', 'CF_TXN_FPBR',
                         'CF_TXN_FS', 'CF_TXN_FU', 'CF_TXN_INSURANCE', 'CF_TXN_LOAN', 'CF_TXN_MAEC', 'CF_TXN_SIP',
                         'CF_TXN_SN', 'CF_TXN_STDT', 'CF_TXN_STMT', 'CF_TXN_STSS', 'CF_TXN_STST', 'CF_TXN_STYPE']

table_check_list = [ 'CUST', 'AP', 'BC', 'FD', 'FDEBT', 'FS', 'FU', 'LIFEINS'
                    , 'SIP', 'STDT', 'STMT', 'STRUPRO', 'STSS', 'STST', 'UU'
                    , 'AUM'
                    , 'FTBD_profile'
                    , 'FTBD_PROFOLIO1', 'FTBD_PROFOLIO2', 'FTBD_PROFOLIO3'
                    , 'FTBD_ACTUBNF'
                    , 'SAFETY_STOCK_1', 'SAFETY_STOCK_2'
                    , 'TRANS_CATE1', 'TRANS_CATE2', 'TRANS_CATE31', 'TRANS_CATE32', 'TRANS_CATE4', 'TRANS_CATE5'
                    , 'LINE_INTERACT','EDM_INTERACT','FTBD_INTERACT'
                    , 'FTBD_CONTRACT1','FTBD_CONTRACT2','FTBD_EVENT','FTBD_KYCQA','FTBD_INVBNF_PLUS'
                    , 'FTBD_JCI','FTBD_DGT1'
                    ]


table_not_fixed_column_list = ['TRANS_CATE1', 'TRANS_CATE2', 'TRANS_CATE3', 'TRANS_CATE31', 'TRANS_CATE32', 'TRANS_CATE4','TRANS_CATE5'
                               ,'SAFETY_STOCK', 'SAFETY_STOCK_1', 'SAFETY_STOCK_2']

colname_object_list = ['customer_id','yyyymm']
colname_catgory_list_1 = ['gender_code','edu_desc','postal_rsd_code','postal_hh_code','use_fubon_acct'
                          ,'aml_highrisk_flag','nw_segment_desc','segment_desc','valid_flag','e_acct_valid_flag'
                          ,'emp_flag','pro_invest_flag','aml_risk_desc','kyc_risk_desc','email_flag','line_bc_flag'
                         ]
colname_catgory_list_2 = ['c_acct_valid_flag', 'subbrok_acct_open_flag','future_acct_open_flag','trust_acct_open_flag']

colname_catgory_list_3 = ['kycqa_q5', 'kycqa_q6','kycqa_q7','kycqa_q8','kycqa_q10', 'kycqa_q11','kycqa_q12','kycqa_q13'
                         ,'kycqa_q91', 'kycqa_q92','kycqa_q93','kycqa_q94','kycqa_q95', 'kycqa_q96','kycqa_q97','kycqa_q98']
colname_catgory_list_4 = ['histy_segment_desc','compare_histy_segment_desc','once_be_pi_customer']
colname_catgory_list_5 = ['agent_flag']
colname_catgory_list_6 = ['dgt_acct_freqtfct','dgt_stafth_freqtfct','dgt_smtstp_freqtfct','dgt_afthif_freqtfct','dgt_stqt_fqtps'
                          ,'dgt_stdtqt_fqtps','dgt_finnews_fqtps','dgt_stcqt_fqtps','dgt_wa_fqtps','dgt_intf_fqtps']

colname_catgory_list_202503 = ['gender_code','edu_desc','postal_rsd_code','postal_hh_code','use_fubon_acct'
                        ,'aml_highrisk_flag','nw_segment_desc','segment_desc','valid_flag','e_acct_valid_flag'
                        ,'emp_flag','pro_invest_flag','aml_risk_desc','kyc_risk_desc','email_flag','line_bc_flag'
                        ,'c_acct_valid_flag', 'subbrok_acct_open_flag','future_acct_open_flag','trust_acct_open_flag'
                        ,'kycqa_q5', 'kycqa_q6','kycqa_q7','kycqa_q9','kycqa_q10', 'kycqa_q11','kycqa_q12','kycqa_q13'
                        ,'kycqa_q81', 'kycqa_q82','kycqa_q83','kycqa_q84','kycqa_q85', 'kycqa_q86','kycqa_q87','kycqa_q88'
                        ,'histy_segment_desc','compare_histy_segment_desc','once_be_pi_customer'
                        ,'agent_flag','dgt_acct_freqtfct','dgt_stafth_freqtfct','dgt_smtstp_freqtfct','dgt_afthif_freqtfct','dgt_stqt_fqtps_m'
                        ,'dgt_stdtqt_fqtps_m','dgt_finnews_fqtps_m','dgt_stcqt_fqtps_m','dgt_wa_fqtps_m','dgt_intf_fqtps_m']
colname_catgory_list_202503_1 = ['gender_code','edu_desc','postal_rsd_code','postal_hh_code','use_fubon_acct'
                        ,'aml_highrisk_flag','nw_segment_desc','segment_desc','valid_flag','e_acct_valid_flag'
                        ,'emp_flag','pro_invest_flag','aml_risk_desc','kyc_risk_desc','email_flag','line_bc_flag'
                        ,'c_acct_valid_flag', 'subbrok_acct_open_flag','future_acct_open_flag','trust_acct_open_flag']
colname_catgory_list_202503_2 = ['kycqa_q5', 'kycqa_q6','kycqa_q7','kycqa_q9','kycqa_q10', 'kycqa_q11','kycqa_q12','kycqa_q13'
                        ,'kycqa_q81', 'kycqa_q82','kycqa_q83','kycqa_q84','kycqa_q85', 'kycqa_q86','kycqa_q87','kycqa_q88'
                        ,'histy_segment_desc','compare_histy_segment_desc','once_be_pi_customer'
                        ,'agent_flag','dgt_acct_freqtfct','dgt_stafth_freqtfct','dgt_smtstp_freqtfct','dgt_afthif_freqtfct','dgt_stqt_fqtps_m'
                        ,'dgt_stdtqt_fqtps_m','dgt_finnews_fqtps_m','dgt_stcqt_fqtps_m','dgt_wa_fqtps_m','dgt_intf_fqtps_m']


# In[3]:


# import sys
# import pandas as pd
# sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
# from Sql_module import get_SQL_raw_data
# # 測試DJ
# test_df_dj = get_SQL_raw_data('select * from s_taichiehfan.mlops_ref_info_double', account=account, pwd=pwd)
# if isinstance(test_df_dj, pd.DataFrame):
#     print('DJ OK!')
# else:
#     print('DJ 密碼有問題')
# # 測試EJ
# test_df_dj = get_SQL_raw_data('select * from s_taichiehfan.mlops_ref_info_double', account=account_yichieh, pwd=pwd_yichieh)
# if isinstance(test_df_dj, pd.DataFrame):
#     print('EJ OK!')
# else:
#     print('EJ 密碼有問題')
# # 測試PS
# test_df_dj = get_SQL_raw_data('select * from s_taichiehfan.mlops_ref_info_double', account=account_paoshiang, pwd=pwd_paoshiang)
# if isinstance(test_df_dj, pd.DataFrame):
#     print('PS OK!')
# else:
#     print('PS 密碼有問題')


# In[4]:


# 模型間不需修改
#MLOPS
algorithm = 'xgboost'
do_ym_list_pd_3m = ['202412','202503','202506','202509','202512']
do_ym_list_pd_3m_detail = ['202412','202501','202502','202503','202504','202505','202506','202507','202508','202509','202512']
do_ym_list_pd_12m = ['202403','202406','202409','202412','202503']
ym = '202603'
mlops_retrain_day = '20260327'



ym_ntb_retrain = '20260505'
ym_ntb_predict = '20260405'
do_ym_list_ntb = ['20251116','20251130','20251214','20251228','20260111','20260125','20260208','20260222','20260308','20260322','20260405']
write_db_Y_N = True
limit_size_select = 100000
limit_size_build= 400000
frequency = '批次'
edition_detail = '新資料retrain'
edition_detail_ntb = '跳10週+雙週y'
write_feature_Y_N = False


# In[5]:


# Model parameter

#潛客參數
max_depth_a = 3
scale_pos_weight_a = 4
n_estimator_a = 100

#非潛客參數
max_depth_b = 3
scale_pos_weight_b = 4
n_estimator_b = 100


# In[6]:


get_ipython().system('jupyter nbconvert --to script config.ipynb')


# In[ ]:




