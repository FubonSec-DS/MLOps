#!/usr/bin/env python

# In[1]:


# loading parameter
import sys

sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
import config

account, pwd = config.account, config.pwd

feature_file_path_mlops = config.feature_file_path_mlops
feature_file_path_jihsun = config.feature_file_path_jihsun
feature_file_path_ntb = config.feature_file_path_ntb
table_orginal_list = config.table_orginal_list
table_append_list = config.table_append_list
table_append_list_2024 = config.table_append_list_2024
table_append_list_2025 = config.table_append_list_2025
table_check_list = config.table_check_list
table_not_fixed_column_list = config.table_not_fixed_column_list
table_fin_table_path =  '/home/cdsw/Features/new_features/df_fin_feats.csv'
compare_column_ym = '202302'
# colname_catgory_list_1 = config.colname_catgory_list_1
# colname_catgory_list_2 = config.colname_catgory_list_2
colname_object_list = config.colname_object_list
colname_catgory_list_202503 = config.colname_catgory_list_202503
colname_catgory_list_202503_1 = config.colname_catgory_list_202503_1
colname_catgory_list_202503_2 = config.colname_catgory_list_202503_2


# In[2]:


#feature_file_path_jihsun


# In[3]:


# import packages
import os
import pickle
import random
import time
from os import listdir

import numpy as np
import pandas as pd
from sklearn.utils import resample

# In[4]:


# define function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df


# In[ ]:


# def read_feature_by_sampling(table_name,ym,this_prod,mother):
#     from Sql_module import get_SQL_raw_data
#     query = f"""
#         SELECT A.* FROM DS_SEC.{table_name} A
#         INNER JOIN (SELECT CUSTOMER_ID
#             FROM MLOPS_POPULATION_SAMPLING
#             WHERE YYYYMM = '{ym}' AND PRODUCT = '{this_prod}' AND POPULATION = '{mother}')B
#         ON A.CUSTOMER_ID = B.CUSTOMER_ID
#         WHERE A.YYYYMM = '{ym}'
#     """

#     try:
#         df = get_SQL_raw_data2(query=query)
#         return df
#     except Exception as e:
#         print(f"查詢失敗:{table_name}-{e}")
#         return None


# In[ ]:


# def read_feature_by_all(table_name,ym,this_prod,mother):
#     from Sql_module import get_SQL_raw_data
#     query1 = f"""
#         SELECT A.* FROM DS_SEC.{table_name} A
#         INNER JOIN (SELECT CUSTOMER_ID
#             FROM MLOPS_POPULATION
#             WHERE YYYYMM = '{ym}' AND SEGMENT = '{mother}')B
#         ON A.CUSTOMER_ID = B.CUSTOMER_ID
#         WHERE A.YYYYMM = '{ym}'
#     """
#     query2 = f"""
#     SELECT A.* FROM DS_SEC.{table_name} A
#     INNER JOIN (SELECT CUSTOMER_ID
#         FROM MLOPS_POPULATION
#         WHERE YYYYMM = '{ym}')B
#     ON A.CUSTOMER_ID = B.CUSTOMER_ID
#     WHERE A.YYYYMM = '{ym}'
#     """

#     try:
#         if mother == '不分潛客':
#             df = get_SQL_raw_data2(query=query2)
#         else:
#             df = get_SQL_raw_data2(query=query1)
#         return df
#     except Exception as e:
#         print(f"查詢失敗:{table_name}-{e}")
#         return None


# In[ ]:


# def build_sampled_feature(ym,this_prod,mother, papu_and_y = pd.DataFrame()):
#     merge_df = None
#     for i, table in enumerate(table_append_list_2025):
#         print(f"讀取中:{table}")
#         if len(papu_and_y)>0:
#             df = read_feature_by_all(table,ym,this_prod,mother)
#             df = df[df['customer_id'].isin(papu_and_y['customer_id'])]
#         else:
#             df = read_feature_by_sampling(table,ym,this_prod,mother)


#         if df is None:
#             print(f"{table_name}-未撈到資料")
#             continue
#         df.sort_values(["customer_id","yyyymm"], inplace=True)
#         df = df.reset_index(drop = True)

#         if merge_df is None:
#             merge_df = df
#         else:
#             df = df.drop(columns = ["customer_id","yyyymm"],errors = "ignore")
#             merge_df = pd.concat([merge_df,df],axis = 1, join='outer')
# #         print(f"每合併一次空值:{merge_df['yyyymm'].isnull().sum()}")
#         print(f"目前資料數:{len(merge_df)}-{len(merge_df.columns)}")

#     return merge_df



# In[ ]:


def read_feature_by_sampling(ym,this_prod,mother,drop_key_word,Fill_zero):
    from Sql_module import get_SQL_raw_data2

    query1 = f"""
    select
    /*+ PARALLEL(a 4) PARALLEL(inn 4) PARALLEL(b 4) PARALLEL(c 4) PARALLEL(d 4) PARALLEL(e 4) PARALLEL(f 4) PARALLEL(g 4) PARALLEL(h 4) PARALLEL(i 4)*/
    *
    from (select * from ds_sec.cf_custid PARTITION ("P_{ym}")) a
    join(select customer_id,yyyymm from s_ianleong.mlops_population_sampling where product = '{this_prod}' and population = '{mother}' and yyyymm = '{ym}') inn
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_asset1 PARTITION ("P_{ym}")) b
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_asset2 PARTITION ("P_{ym}")) c
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_asset3 PARTITION ("P_{ym}")) d
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_asset4 PARTITION ("P_{ym}")) e
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_actubnf PARTITION ("P_{ym}")) f
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_actubnfroi PARTITION ("P_{ym}")) g
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_invbnf PARTITION ("P_{ym}")) h
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_invbnfroi PARTITION ("P_{ym}")) i
    using (customer_id ,yyyymm)
    """

    query2 = f"""
    select
    /*+ PARALLEL(a 4) PARALLEL(inn 4) PARALLEL(b 4) PARALLEL(c 4) PARALLEL(d 4) PARALLEL(e 4) PARALLEL(f 4) PARALLEL(g 4) PARALLEL(h 4) PARALLEL(i 4) PARALLEL(j 4)*/
    *
    from (select * from ds_sec.cf_custid PARTITION ("P_{ym}")) a
    join(select customer_id,yyyymm from s_ianleong.mlops_population_sampling where product = '{this_prod}' and population = '{mother}' and yyyymm = '{ym}')inn
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_AP PARTITION ("P_{ym}")) b
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_FD PARTITION ("P_{ym}")) c
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_FB PARTITION ("P_{ym}")) d
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_FS PARTITION ("P_{ym}")) e
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_FU PARTITION ("P_{ym}")) f
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_STMT PARTITION ("P_{ym}")) g
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_SN PARTITION ("P_{ym}")) h
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_STSS PARTITION ("P_{ym}")) i
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_STST PARTITION ("P_{ym}")) j
    using (customer_id ,yyyymm)
    """

    query3 = f"""
    select
    /*+ PARALLEL(a 4) PARALLEL(inn 4) PARALLEL(b 4) PARALLEL(c 4) PARALLEL(d 4) PARALLEL(e 4) PARALLEL(f 4) PARALLEL(g 4) PARALLEL(h 4) PARALLEL(i 4) PARALLEL(j 4) PARALLEL(k 4)*/
    *
    from (select * from ds_sec.cf_custid PARTITION ("P_{ym}")) a
    join(select customer_id,yyyymm from s_ianleong.mlops_population_sampling where product = '{this_prod}' and population = '{mother}' and yyyymm = '{ym}')inn
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_BSBL PARTITION ("P_{ym}")) b
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_INSURANCE PARTITION ("P_{ym}")) c
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_SIP PARTITION ("P_{ym}")) d
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_STDT PARTITION ("P_{ym}")) e
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_LOAN PARTITION ("P_{ym}")) f
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_STYPE PARTITION ("P_{ym}")) g
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_MAEC PARTITION ("P_{ym}")) h
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_CURRENCY PARTITION ("P_{ym}")) i
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_FPBR PARTITION ("P_{ym}")) j
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_AUM PARTITION ("P_{ym}")) k
    using (customer_id ,yyyymm)
    """

    query4 = f"""
    select
    /*+ PARALLEL(a 4) PARALLEL(inn 4) PARALLEL(b 4) PARALLEL(c 4) PARALLEL(d 4) PARALLEL(e 4) PARALLEL(f 4) PARALLEL(g 4) PARALLEL(h 4) PARALLEL(i 4)*/
    *
    from (select * from ds_sec.cf_custid PARTITION ("P_{ym}")) a
    join(select customer_id,yyyymm from s_ianleong.mlops_population_sampling where product = '{this_prod}' and population = '{mother}' and yyyymm = '{ym}')inn
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFILE PARTITION ("P_{ym}")) b
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO1 PARTITION ("P_{ym}")) c
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO2 PARTITION ("P_{ym}")) d
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO3 PARTITION ("P_{ym}")) e
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO4 PARTITION ("P_{ym}")) f
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO5 PARTITION ("P_{ym}")) g
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO6 PARTITION ("P_{ym}")) h
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO7 PARTITION ("P_{ym}")) i
    using (customer_id ,yyyymm)
    """

    query5 = f"""
    select
    /*+ PARALLEL(a 4) PARALLEL(inn 4) PARALLEL(b 4) PARALLEL(c 4) PARALLEL(d 4) PARALLEL(e 4) PARALLEL(f 4) PARALLEL(g 4) PARALLEL(h 4) PARALLEL(i 4) PARALLEL(j 4) PARALLEL(k 4) PARALLEL(l 4)*/
    *
    from (select * from ds_sec.cf_custid PARTITION ("P_{ym}")) a
    join(select customer_id,yyyymm from s_ianleong.mlops_population_sampling where product = '{this_prod}' and population = '{mother}' and yyyymm = '{ym}')inn
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_INTERACT_ECDAY PARTITION ("P_{ym}")) b
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_INTERACT_ECPROD PARTITION ("P_{ym}")) c
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_INTERACT_EDM PARTITION ("P_{ym}")) d
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_INTERACT_LINE PARTITION ("P_{ym}")) e
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_CONTRACT_DUE PARTITION ("P_{ym}")) f
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_CONTRACT PARTITION ("P_{ym}")) g
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_EVENT PARTITION ("P_{ym}")) h
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_KYCQA PARTITION ("P_{ym}")) i
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_JCI PARTITION ("P_{ym}")) j
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_DGT1 PARTITION ("P_{ym}")) k
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_DGT2 PARTITION ("P_{ym}")) l
    using (customer_id ,yyyymm)
    """

    try:
        df1 = get_SQL_raw_data2(query=query1)
        if df1 is None:
            print("特徵1-未撈到資料")
        else:
            print(f"特徵1:{len(df1)}-{len(df1.columns)}")
        df2 = get_SQL_raw_data2(query=query2)
        if df2 is None:
            print("特徵2-未撈到資料")
        else:
            print(f"特徵2:{len(df2)}-{len(df2.columns)}")
        df3 = get_SQL_raw_data2(query=query3)
        if df3 is None:
            print("特徵3-未撈到資料")
        else:
            print(f"特徵3:{len(df3)}-{len(df3.columns)}")
        df4 = get_SQL_raw_data2(query=query4)
        if df4 is None:
            print("特徵4-未撈到資料")
        else:
            print(f"特徵4:{len(df4)}-{len(df4.columns)}")
        df5 = get_SQL_raw_data2(query=query5)
        if df5 is None:
            print("特徵5-未撈到資料")
        else:
            print(f"特徵5:{len(df5)}-{len(df5.columns)}")

        print("欄位轉型態")

        df4[colname_catgory_list_202503_1] = df4[colname_catgory_list_202503_1].astype('category')
        df5[colname_catgory_list_202503_2] = df5[colname_catgory_list_202503_2].astype('category')
        cols1 = df1.select_dtypes('object').drop(['customer_id','yyyymm'],axis=1).columns
        if cols1.tolist() != []:
            df1[cols1.tolist()] = df1[cols1.tolist()].astype('float')
        cols2 = df2.select_dtypes('object').drop(['customer_id','yyyymm'],axis=1).columns
        if cols2.tolist() != []:
            df2[cols2.tolist()] = df2[cols2.tolist()].astype('float')
        cols3 = df3.select_dtypes('object').drop(['customer_id','yyyymm'],axis=1).columns
        if cols3.tolist() != []:
            df3[cols3.tolist()] = df3[cols3.tolist()].astype('float')
        cols4 = df4.select_dtypes('object').drop(['customer_id','yyyymm'],axis=1).columns
        if cols4.tolist() != []:
            df4[cols4.tolist()] = df4[cols4.tolist()].astype('float')
        cols5 = df5.select_dtypes('object').drop(['customer_id','yyyymm'],axis=1).columns
        if cols5.tolist() != []:
            df5[cols5.tolist()] = df5[cols5.tolist()].astype('float')

        print("排除不要的特徵")
        for drop_kw in drop_key_word:
            df1.drop([a for a in df1.columns if (drop_kw in a)], axis=1, inplace=True)
        for drop_kw in drop_key_word:
            df2.drop([a for a in df2.columns if (drop_kw in a)], axis=1, inplace=True)
        for drop_kw in drop_key_word:
            df3.drop([a for a in df3.columns if (drop_kw in a)], axis=1, inplace=True)
        for drop_kw in drop_key_word:
            df4.drop([a for a in df4.columns if (drop_kw in a)], axis=1, inplace=True)
        for drop_kw in drop_key_word:
            df5.drop([a for a in df5.columns if (drop_kw in a)], axis=1, inplace=True)


        print("NA值補0")
        if Fill_zero:
            feature_list = list(df1.select_dtypes(exclude = ['category','object']))
            df1[feature_list] = df1[feature_list].fillna(0)
#             print('df1_NA補0完畢!!!')
        if Fill_zero:
            feature_list = list(df2.select_dtypes(exclude = ['category','object']))
            df2[feature_list] = df2[feature_list].fillna(0)
#             print('df2_NA補0完畢!!!')
        if Fill_zero:
            feature_list = list(df3.select_dtypes(exclude = ['category','object']))
            df3[feature_list] = df3[feature_list].fillna(0)
#             print('df3_NA補0完畢!!!')
        if Fill_zero:
            feature_list = list(df4.select_dtypes(exclude = ['category','object']))
            df4[feature_list] = df4[feature_list].fillna(0)
#             print('df4_NA補0完畢!!!')
        if Fill_zero:
            feature_list = list(df5.select_dtypes(exclude = ['category','object']))
            df5[feature_list] = df5[feature_list].fillna(0)
#             print('df5_NA補0完畢!!!')

        df1.sort_values(["customer_id","yyyymm"], inplace=True)
        df1 = df1.reset_index(drop = True)
        df2.sort_values(["customer_id","yyyymm"], inplace=True)
        df2 = df2.reset_index(drop = True)
        df3.sort_values(["customer_id","yyyymm"], inplace=True)
        df3 = df3.reset_index(drop = True)
        df4.sort_values(["customer_id","yyyymm"], inplace=True)
        df4 = df4.reset_index(drop = True)
        df5.sort_values(["customer_id","yyyymm"], inplace=True)
        df5 = df5.reset_index(drop = True)

        df2 = df2.drop(columns = ["customer_id","yyyymm"],errors = "ignore")
        df3 = df3.drop(columns = ["customer_id","yyyymm"],errors = "ignore")
        df4 = df4.drop(columns = ["customer_id","yyyymm"],errors = "ignore")
        df5 = df5.drop(columns = ["customer_id","yyyymm"],errors = "ignore")

        merge_df = pd.concat([df1,df2,df3,df4,df5],axis = 1,copy = False).copy()

        return merge_df
    except Exception as e:
        print(f"查詢失敗:-{e}")
        return None


# In[ ]:


def read_feature_by_all(ym,this_prod,mother,drop_key_word,Fill_zero):
    from Sql_module import get_SQL_raw_data2

    query1 = f"""
    select
    /*+ PARALLEL(a 4) PARALLEL(inn 4) PARALLEL(b 4) PARALLEL(c 4) PARALLEL(d 4) PARALLEL(e 4) PARALLEL(f 4) PARALLEL(g 4) PARALLEL(h 4) PARALLEL(i 4)*/
    *
    from (select * from ds_sec.cf_custid PARTITION ("P_{ym}")) a
    join(select customer_id,yyyymm FROM s_ianleong.MLOPS_POPULATION WHERE YYYYMM = '{ym}' AND SEGMENT = '{mother}')inn
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_asset1 PARTITION ("P_{ym}")) b
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_asset2 PARTITION ("P_{ym}")) c
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_asset3 PARTITION ("P_{ym}")) d
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_asset4 PARTITION ("P_{ym}")) e
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_actubnf PARTITION ("P_{ym}")) f
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_actubnfroi PARTITION ("P_{ym}")) g
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_invbnf PARTITION ("P_{ym}")) h
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_invbnfroi PARTITION ("P_{ym}")) i
    using (customer_id ,yyyymm)
    """

    query2 = f"""
    select
    /*+ PARALLEL(a 4) PARALLEL(inn 4) PARALLEL(b 4) PARALLEL(c 4) PARALLEL(d 4) PARALLEL(e 4) PARALLEL(f 4) PARALLEL(g 4) PARALLEL(h 4) PARALLEL(i 4) PARALLEL(j 4)*/
    *
    from (select * from ds_sec.cf_custid PARTITION ("P_{ym}")) a
    join(select customer_id,yyyymm FROM s_ianleong.MLOPS_POPULATION WHERE YYYYMM = '{ym}' AND SEGMENT = '{mother}')inn
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_AP PARTITION ("P_{ym}")) b
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_FD PARTITION ("P_{ym}")) c
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_FB PARTITION ("P_{ym}")) d
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_FS PARTITION ("P_{ym}")) e
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_FU PARTITION ("P_{ym}")) f
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_STMT PARTITION ("P_{ym}")) g
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_SN PARTITION ("P_{ym}")) h
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_STSS PARTITION ("P_{ym}")) i
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_STST PARTITION ("P_{ym}")) j
    using (customer_id ,yyyymm)
    """

    query3 = f"""
    select
    /*+ PARALLEL(a 4) PARALLEL(inn 4) PARALLEL(b 4) PARALLEL(c 4) PARALLEL(d 4) PARALLEL(e 4) PARALLEL(f 4) PARALLEL(g 4) PARALLEL(h 4) PARALLEL(i 4) PARALLEL(j 4) PARALLEL(k 4)*/
    *
    from (select * from ds_sec.cf_custid PARTITION ("P_{ym}")) a
    join(select customer_id,yyyymm FROM s_ianleong.MLOPS_POPULATION WHERE YYYYMM = '{ym}' AND SEGMENT = '{mother}')inn
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_BSBL PARTITION ("P_{ym}")) b
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_INSURANCE PARTITION ("P_{ym}")) c
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_SIP PARTITION ("P_{ym}")) d
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_STDT PARTITION ("P_{ym}")) e
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_LOAN PARTITION ("P_{ym}")) f
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_STYPE PARTITION ("P_{ym}")) g
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_MAEC PARTITION ("P_{ym}")) h
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_CURRENCY PARTITION ("P_{ym}")) i
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_FPBR PARTITION ("P_{ym}")) j
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_AUM PARTITION ("P_{ym}")) k
    using (customer_id ,yyyymm)
    """

    query4 = f"""
    select
    /*+ PARALLEL(a 4) PARALLEL(inn 4) PARALLEL(b 4) PARALLEL(c 4) PARALLEL(d 4) PARALLEL(e 4) PARALLEL(f 4) PARALLEL(g 4) PARALLEL(h 4) PARALLEL(i 4)*/
    *
    from (select * from ds_sec.cf_custid PARTITION ("P_{ym}")) a
    join(select customer_id,yyyymm FROM s_ianleong.MLOPS_POPULATION WHERE YYYYMM = '{ym}' AND SEGMENT = '{mother}')inn
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFILE PARTITION ("P_{ym}")) b
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO1 PARTITION ("P_{ym}")) c
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO2 PARTITION ("P_{ym}")) d
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO3 PARTITION ("P_{ym}")) e
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO4 PARTITION ("P_{ym}")) f
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO5 PARTITION ("P_{ym}")) g
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO6 PARTITION ("P_{ym}")) h
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO7 PARTITION ("P_{ym}")) i
    using (customer_id ,yyyymm)
    """

    query5 = f"""
    select
    /*+ PARALLEL(a 4) PARALLEL(inn 4) PARALLEL(b 4) PARALLEL(c 4) PARALLEL(d 4) PARALLEL(e 4) PARALLEL(f 4) PARALLEL(g 4) PARALLEL(h 4) PARALLEL(i 4) PARALLEL(j 4) PARALLEL(k 4) PARALLEL(l 4)*/
    *
    from (select * from ds_sec.cf_custid PARTITION ("P_{ym}")) a
    join(select customer_id,yyyymm FROM s_ianleong.MLOPS_POPULATION WHERE YYYYMM = '{ym}' AND SEGMENT = '{mother}')inn
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_INTERACT_ECDAY PARTITION ("P_{ym}")) b
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_INTERACT_ECPROD PARTITION ("P_{ym}")) c
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_INTERACT_EDM PARTITION ("P_{ym}")) d
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_INTERACT_LINE PARTITION ("P_{ym}")) e
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_CONTRACT_DUE PARTITION ("P_{ym}")) f
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_CONTRACT PARTITION ("P_{ym}")) g
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_EVENT PARTITION ("P_{ym}")) h
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_KYCQA PARTITION ("P_{ym}")) i
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_JCI PARTITION ("P_{ym}")) j
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_DGT1 PARTITION ("P_{ym}")) k
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_DGT2 PARTITION ("P_{ym}")) l
    using (customer_id ,yyyymm)
    """

    query6 = f"""
    select
    /*+ PARALLEL(a 4) PARALLEL(inn 4) PARALLEL(b 4) PARALLEL(c 4) PARALLEL(d 4) PARALLEL(e 4) PARALLEL(f 4) PARALLEL(g 4) PARALLEL(h 4) PARALLEL(i 4)*/
    *
    from (select * from ds_sec.cf_custid PARTITION ("P_{ym}")) a
    join(select customer_id,yyyymm FROM s_ianleong.MLOPS_POPULATION WHERE YYYYMM = '{ym}')inn
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_asset1 PARTITION ("P_{ym}")) b
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_asset2 PARTITION ("P_{ym}")) c
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_asset3 PARTITION ("P_{ym}")) d
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_asset4 PARTITION ("P_{ym}")) e
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_actubnf PARTITION ("P_{ym}")) f
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_actubnfroi PARTITION ("P_{ym}")) g
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_invbnf PARTITION ("P_{ym}")) h
    using (customer_id ,yyyymm)
    join (select * from ds_sec.cf_invbnfroi PARTITION ("P_{ym}")) i
    using (customer_id ,yyyymm)
    """

    query7 = f"""
    select
    /*+ PARALLEL(a 4) PARALLEL(inn 4) PARALLEL(b 4) PARALLEL(c 4) PARALLEL(d 4) PARALLEL(e 4) PARALLEL(f 4) PARALLEL(g 4) PARALLEL(h 4) PARALLEL(i 4) PARALLEL(j 4)*/
    *
    from (select * from ds_sec.cf_custid PARTITION ("P_{ym}")) a
    join(select customer_id,yyyymm FROM s_ianleong.MLOPS_POPULATION WHERE YYYYMM = '{ym}')inn
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_AP PARTITION ("P_{ym}")) b
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_FD PARTITION ("P_{ym}")) c
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_FB PARTITION ("P_{ym}")) d
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_FS PARTITION ("P_{ym}")) e
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_FU PARTITION ("P_{ym}")) f
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_STMT PARTITION ("P_{ym}")) g
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_SN PARTITION ("P_{ym}")) h
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_STSS PARTITION ("P_{ym}")) i
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_STST PARTITION ("P_{ym}")) j
    using (customer_id ,yyyymm)
    """

    query8 = f"""
    select
    /*+ PARALLEL(a 4) PARALLEL(inn 4) PARALLEL(b 4) PARALLEL(c 4) PARALLEL(d 4) PARALLEL(e 4) PARALLEL(f 4) PARALLEL(g 4) PARALLEL(h 4) PARALLEL(i 4) PARALLEL(j 4) PARALLEL(k 4)*/
    *
    from (select * from ds_sec.cf_custid PARTITION ("P_{ym}")) a
    join(select customer_id,yyyymm FROM s_ianleong.MLOPS_POPULATION WHERE YYYYMM = '{ym}')inn
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_BSBL PARTITION ("P_{ym}")) b
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_INSURANCE PARTITION ("P_{ym}")) c
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_SIP PARTITION ("P_{ym}")) d
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_STDT PARTITION ("P_{ym}")) e
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_LOAN PARTITION ("P_{ym}")) f
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_STYPE PARTITION ("P_{ym}")) g
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_MAEC PARTITION ("P_{ym}")) h
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_CURRENCY PARTITION ("P_{ym}")) i
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_TXN_FPBR PARTITION ("P_{ym}")) j
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_AUM PARTITION ("P_{ym}")) k
    using (customer_id ,yyyymm)
    """

    query9 = f"""
    select
    /*+ PARALLEL(a 4) PARALLEL(inn 4) PARALLEL(b 4) PARALLEL(c 4) PARALLEL(d 4) PARALLEL(e 4) PARALLEL(f 4) PARALLEL(g 4) PARALLEL(h 4) PARALLEL(i 4)*/
    *
    from (select * from ds_sec.cf_custid PARTITION ("P_{ym}")) a
    join(select customer_id,yyyymm FROM s_ianleong.MLOPS_POPULATION WHERE YYYYMM = '{ym}')inn
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFILE PARTITION ("P_{ym}")) b
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO1 PARTITION ("P_{ym}")) c
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO2 PARTITION ("P_{ym}")) d
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO3 PARTITION ("P_{ym}")) e
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO4 PARTITION ("P_{ym}")) f
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO5 PARTITION ("P_{ym}")) g
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO6 PARTITION ("P_{ym}")) h
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_PROFOLIO7 PARTITION ("P_{ym}")) i
    using (customer_id ,yyyymm)
    """

    query10 = f"""
    select
    /*+ PARALLEL(a 4) PARALLEL(inn 4) PARALLEL(b 4) PARALLEL(c 4) PARALLEL(d 4) PARALLEL(e 4) PARALLEL(f 4) PARALLEL(g 4) PARALLEL(h 4) PARALLEL(i 4) PARALLEL(j 4) PARALLEL(k 4) PARALLEL(l 4)*/
    *
    from (select * from ds_sec.cf_custid PARTITION ("P_{ym}")) a
    join(select customer_id,yyyymm FROM s_ianleong.MLOPS_POPULATION WHERE YYYYMM = '{ym}')inn
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_INTERACT_ECDAY PARTITION ("P_{ym}")) b
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_INTERACT_ECPROD PARTITION ("P_{ym}")) c
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_INTERACT_EDM PARTITION ("P_{ym}")) d
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_INTERACT_LINE PARTITION ("P_{ym}")) e
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_CONTRACT_DUE PARTITION ("P_{ym}")) f
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_CONTRACT PARTITION ("P_{ym}")) g
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_EVENT PARTITION ("P_{ym}")) h
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_KYCQA PARTITION ("P_{ym}")) i
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_JCI PARTITION ("P_{ym}")) j
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_DGT1 PARTITION ("P_{ym}")) k
    using (customer_id ,yyyymm)
    join (select * from ds_sec.CF_DGT2 PARTITION ("P_{ym}")) l
    using (customer_id ,yyyymm)
    """

    try:
        if mother == '不分潛客':
            df1 = get_SQL_raw_data2(query=query6)
            if df1 is None:
                print("特徵1-未撈到資料")
            else:
                print(f"特徵1:{len(df1)}-{len(df1.columns)}")
            df2 = get_SQL_raw_data2(query=query7)
            if df2 is None:
                print("特徵2-未撈到資料")
            else:
                print(f"特徵2:{len(df2)}-{len(df2.columns)}")
            df3 = get_SQL_raw_data2(query=query8)
            if df3 is None:
                print("特徵3-未撈到資料")
            else:
                print(f"特徵3:{len(df3)}-{len(df3.columns)}")
            df4 = get_SQL_raw_data2(query=query9)
            if df4 is None:
                print("特徵4-未撈到資料")
            else:
                print(f"特徵4:{len(df4)}-{len(df4.columns)}")
            df5 = get_SQL_raw_data2(query=query10)
            if df5 is None:
                print("特徵5-未撈到資料")
            else:
                print(f"特徵5:{len(df5)}-{len(df5.columns)}")
        else:
            df1 = get_SQL_raw_data2(query=query1)
            if df1 is None:
                print("特徵1-未撈到資料")
            else:
                print(f"特徵1:{len(df1)}-{len(df1.columns)}")
            df2 = get_SQL_raw_data2(query=query2)
            if df2 is None:
                print("特徵2-未撈到資料")
            else:
                print(f"特徵2:{len(df2)}-{len(df2.columns)}")
            df3 = get_SQL_raw_data2(query=query3)
            if df3 is None:
                print("特徵3-未撈到資料")
            else:
                print(f"特徵3:{len(df3)}-{len(df3.columns)}")
            df4 = get_SQL_raw_data2(query=query4)
            if df4 is None:
                print("特徵4-未撈到資料")
            else:
                print(f"特徵4:{len(df4)}-{len(df4.columns)}")
            df5 = get_SQL_raw_data2(query=query5)
            if df5 is None:
                print("特徵5-未撈到資料")
            else:
                print(f"特徵5:{len(df5)}-{len(df5.columns)}")

        print("欄位轉型態")
        df4[colname_catgory_list_202503_1] = df4[colname_catgory_list_202503_1].astype('category')
        df5[colname_catgory_list_202503_2] = df5[colname_catgory_list_202503_2].astype('category')
        cols1 = df1.select_dtypes('object').drop(['customer_id','yyyymm'],axis=1).columns
        if cols1.tolist() != []:
            df1[cols1.tolist()] = df1[cols1.tolist()].astype('float')
        cols2 = df2.select_dtypes('object').drop(['customer_id','yyyymm'],axis=1).columns
        if cols2.tolist() != []:
            df2[cols2.tolist()] = df2[cols2.tolist()].astype('float')
        cols3 = df3.select_dtypes('object').drop(['customer_id','yyyymm'],axis=1).columns
        if cols3.tolist() != []:
            df3[cols3.tolist()] = df3[cols3.tolist()].astype('float')
        cols4 = df4.select_dtypes('object').drop(['customer_id','yyyymm'],axis=1).columns
        if cols4.tolist() != []:
            df4[cols4.tolist()] = df4[cols4.tolist()].astype('float')
        cols5 = df5.select_dtypes('object').drop(['customer_id','yyyymm'],axis=1).columns
        if cols5.tolist() != []:
            df5[cols5.tolist()] = df5[cols5.tolist()].astype('float')
        print("欄位轉型態完成")

        print("排除不要的特徵")
        for drop_kw in drop_key_word:
            df1.drop([a for a in df1.columns if (drop_kw in a)], axis=1, inplace=True)
        for drop_kw in drop_key_word:
            df2.drop([a for a in df2.columns if (drop_kw in a)], axis=1, inplace=True)
        for drop_kw in drop_key_word:
            df3.drop([a for a in df3.columns if (drop_kw in a)], axis=1, inplace=True)
        for drop_kw in drop_key_word:
            df4.drop([a for a in df4.columns if (drop_kw in a)], axis=1, inplace=True)
        for drop_kw in drop_key_word:
            df5.drop([a for a in df5.columns if (drop_kw in a)], axis=1, inplace=True)
        print("排除特徵完成")

        print("NA值補0")
        if Fill_zero:
            feature_list = list(df1.select_dtypes(exclude = ['category','object']))
            df1[feature_list] = df1[feature_list].fillna(0)
#             print('df1_NA補0完畢!!!')
        if Fill_zero:
            feature_list = list(df2.select_dtypes(exclude = ['category','object']))
            df2[feature_list] = df2[feature_list].fillna(0)
#             print('df2_NA補0完畢!!!')
        if Fill_zero:
            feature_list = list(df3.select_dtypes(exclude = ['category','object']))
            df3[feature_list] = df3[feature_list].fillna(0)
#             print('df3_NA補0完畢!!!')
        if Fill_zero:
            feature_list = list(df4.select_dtypes(exclude = ['category','object']))
            df4[feature_list] = df4[feature_list].fillna(0)
#             print('df4_NA補0完畢!!!')
        if Fill_zero:
            feature_list = list(df5.select_dtypes(exclude = ['category','object']))
            df5[feature_list] = df5[feature_list].fillna(0)
#             print('df5_NA補0完畢!!!')
        print("NA補0完成")

        print("id排序")
        df1.sort_values(["customer_id","yyyymm"], inplace=True)
        df1 = df1.reset_index(drop = True)
        df2.sort_values(["customer_id","yyyymm"], inplace=True)
        df2 = df2.reset_index(drop = True)
        df3.sort_values(["customer_id","yyyymm"], inplace=True)
        df3 = df3.reset_index(drop = True)
        df4.sort_values(["customer_id","yyyymm"], inplace=True)
        df4 = df4.reset_index(drop = True)
        df5.sort_values(["customer_id","yyyymm"], inplace=True)
        df5 = df5.reset_index(drop = True)

        print("刪欄位")
        df2 = df2.drop(columns = ["customer_id","yyyymm"],errors = "ignore")
        df3 = df3.drop(columns = ["customer_id","yyyymm"],errors = "ignore")
        df4 = df4.drop(columns = ["customer_id","yyyymm"],errors = "ignore")
        df5 = df5.drop(columns = ["customer_id","yyyymm"],errors = "ignore")

        merge_df = pd.concat([df1,df2,df3,df4,df5],axis = 1,copy = False).copy()
        return merge_df
    except Exception as e:
        print(f"查詢失敗:-{e}")
        return None


# In[ ]:


def build_sampled_feature(ym,this_prod,mother,drop_key_word,Fill_zero, papu_and_y = pd.DataFrame()):
    merge_df = None
    try:
        if len(papu_and_y)>0:
            df = read_feature_by_all(ym,this_prod,mother,drop_key_word,Fill_zero)
#             print(type(df))
#             print(type(papu_and_y))
            df = df[df['customer_id'].isin(papu_and_y['customer_id'])]
            df.sort_values(["customer_id","yyyymm"], inplace=True)
            df = df.reset_index(drop = True)
        else:
            df = read_feature_by_sampling(ym,this_prod,mother,drop_key_word,Fill_zero)

        print(f"目前資料數:{len(df)}-{len(df.columns)}")

        return df
    except Exception as e:
        print(f"誰的問題:-{e}")
        return None


# In[ ]:


def get_feature_by_SOP_202503(do_ym_lst, mother = None, drop_key_word = [], papu_and_y = pd.DataFrame(),
                       this_prod = None, Fill_zero=False):
    st_time = time.time()
    concat_df_outcome = {}
    for ym in do_ym_lst:
    #抓取母體跟Y
        if (not papu_and_y.empty) :
            papu_and_y_yyyymm = papu_and_y[papu_and_y['yyyymm'] == ym]
            if not papu_and_y_yyyymm.empty:
                print(f"年月:{ym} 筆數為{len(papu_and_y_yyyymm)}")
            else:
                raise ValueError(f"年月:{ym} 筆數為0")
        elif just_for_check:
            print('允許空的dataframe用於check特徵完成情形!')
        else:
            raise ValueError("papu_and_y筆數為0")
#         print(f"母體空值:{papu_and_y_yyyymm['yyyymm'].isnull().sum()}")

    #逐月撈取特徵
        st_time1 = time.time()
        print(f' @@@@@@@@@@@@@@@@ Year of {ym} combine info @@@@@@@@@@@@@@@@ ')
        if ym in ym in sorted(do_ym_lst)[0:-1]:
            feature_sample = build_sampled_feature(ym,this_prod,mother,drop_key_word,Fill_zero)
        else:
            feature_sample = build_sampled_feature(ym,this_prod,mother,drop_key_word,Fill_zero,papu_and_y_yyyymm)

        end_time1 = time.time()
        print(f'全撈取特徵,  runtime: {round((end_time1-st_time1)/60)} minutes')

        st_time2 = time.time()
        print("母體降維合併")
        papu_and_y_yyyymm = papu_and_y_yyyymm[papu_and_y_yyyymm['customer_id'].isin(feature_sample['customer_id'])]

        if len(papu_and_y_yyyymm) != len(feature_sample):
            print("母體特徵數量不對，互減對齊")
            feature_sample = feature_sample[feature_sample['customer_id'].isin(papu_and_y_yyyymm['customer_id'])]
            feature_sample.sort_values(["customer_id","yyyymm"], inplace=True)
            feature_sample = feature_sample.reset_index(drop = True)

        papu_and_y_yyyymm.sort_values(["customer_id","yyyymm"], inplace=True)
        papu_and_y_yyyymm = papu_and_y_yyyymm.reset_index(drop = True)
        papu_and_y_yyyymm = papu_and_y_yyyymm.drop(columns = ["customer_id","yyyymm"],errors = "ignore")
        print(f"母體共{len(papu_and_y_yyyymm)}筆")
        print(f"特徵共{len(feature_sample)}筆")

        concat_df = pd.concat([feature_sample,papu_and_y_yyyymm],axis = 1)
#         concat_df = pd.merge(feature_sample, papu_and_y_yyyymm, how='left', on=['customer_id','yyyymm'])
        end_time2 = time.time()
        print(f'母體合併,  runtime: {round((end_time2-st_time2)/60)} minutes')
        print(f"特徵空值:{concat_df['yyyymm'].isnull().sum()}")
        concat_df['yyyymm'] = concat_df['yyyymm'].astype('int')
    #抓取金融指標
        st_time3 = time.time()
        print(f"撈取{ym}金融指標")
        if os.path.exists(table_fin_table_path):
            fin = pd.read_csv(table_fin_table_path)
            fin['ym'] = fin['ym'].astype('int')
            fin = fin.rename({'ym': 'yyyymm'}, axis=1)
            concat_df = pd.merge(concat_df, fin, how='left', on=['yyyymm'])
        end_time3 = time.time()
        print(f'抓取金融指標,  runtime: {round((end_time3-st_time3)/60)} minutes')
    #排除不要的特徵
#         print(f"排除不要的特徵")
#         for drop_kw in drop_key_word:
#             concat_df.drop([a for a in concat_df.columns if (drop_kw in a)], axis=1, inplace=True)
    # NA補0
#         if Fill_zero:
#             feature_list = list(concat_df.select_dtypes(exclude = ['category','object']))
#             na_totally_num = sum(concat_df[feature_list].isna().sum(axis=1))
#             print(f'補0前NA數量:{na_totally_num}')
#             concat_df[feature_list] = concat_df[feature_list].fillna(0)
#             print('NA補0完畢!!!')
#             na_totally_num = sum(concat_df[feature_list].isna().sum(axis=1))
#             print(f'補0後NA數量:{na_totally_num}')
#         print(f"合併前空值:{concat_df['yyyymm'].isnull().sum()}")


        mask1 = len(concat_df)==len(feature_sample)
        mask2 = len(concat_df) >= 1000
        status = ( mask1 and mask2)
        concat_df_outcome[str(ym)+'_status'] = status
        concat_df_outcome[str(ym)+'_df'] = concat_df

    end_time = time.time()
    print(f'get_feature_by_SOP,  runtime: {round((end_time-st_time)/60)} minutes')
    return concat_df_outcome


# In[6]:


def check_undo_feature(do_concat_ym_lst, feature_file_path):
    outcome = {}
    for ym in do_concat_ym_lst:
        #參數與宣告
        check_path = feature_file_path + '/' + f'{ym}'
        exist_list = listdir(check_path)
        notexist_list = []
        undo_list = []
        print('-----Check [undo_feature] info------')
        print(f'年月: {ym} ')
        print('未執行清單:')

        #抓出不存在TABLE
        for table in table_append_list_2024:
            check_file_name = table + f'_{ym}.pickle'
            if check_file_name not in exist_list :
                notexist_list.append(table)
        ############################

        #判斷是否有特殊情況
        for not_exist_table in notexist_list:
            if not_exist_table not in ['TRANS_CATE3','TRANS_CATE31','TRANS_CATE32','SAFETY_STOCK', 'SAFETY_STOCK_1', 'SAFETY_STOCK_2'] or not_exist_table == 'TRANS_CATE3' and 'TRANS_CATE31' in notexist_list and 'TRANS_CATE32' in notexist_list or (not_exist_table == 'TRANS_CATE31' or not_exist_table == 'TRANS_CATE32')  and 'TRANS_CATE3' in notexist_list or not_exist_table == 'SAFETY_STOCK' and 'SAFETY_STOCK_1' in notexist_list and 'SAFETY_STOCK_2' in notexist_list or (not_exist_table == 'SAFETY_STOCK_1' or not_exist_table == 'SAFETY_STOCK_2')  and 'SAFETY_STOCK' in notexist_list:
                print(not_exist_table); undo_list.append(not_exist_table)
        ############################

        #存入確認結果
        status = (undo_list==[])
        outcome[str(ym)+'_status'] = status
        outcome[str(ym)+'_undo_list'] = undo_list
        if status : print(f'year: {ym} feature is no problem!!!!!!!!!')
    return outcome


# In[7]:


def check_undo_feature_2024test(do_concat_ym_lst, feature_file_path):
    outcome = {}
    for ym in do_concat_ym_lst:
        #參數與宣告
        check_path = feature_file_path + '/' + f'{ym}'
        exist_list = listdir(check_path)
        notexist_list = []
        undo_list = []
        print('-----Check [undo_feature] info------')
        print(f'年月: {ym} ')
        print('未執行清單:')

        #抓出不存在TABLE
        for table in table_check_list:
            check_file_name = table + f'_{ym}.pickle'
            if check_file_name not in exist_list :
                notexist_list.append(table)
        ############################

        #判斷是否有特殊情況
        for not_exist_table in notexist_list:
            if not_exist_table not in ['TRANS_CATE3','TRANS_CATE31','TRANS_CATE32','SAFETY_STOCK', 'SAFETY_STOCK_1', 'SAFETY_STOCK_2'] or not_exist_table == 'TRANS_CATE3' and 'TRANS_CATE31' in notexist_list and 'TRANS_CATE32' in notexist_list or (not_exist_table == 'TRANS_CATE31' or not_exist_table == 'TRANS_CATE32')  and 'TRANS_CATE3' in notexist_list or not_exist_table == 'SAFETY_STOCK' and 'SAFETY_STOCK_1' in notexist_list and 'SAFETY_STOCK_2' in notexist_list or (not_exist_table == 'SAFETY_STOCK_1' or not_exist_table == 'SAFETY_STOCK_2')  and 'SAFETY_STOCK' in notexist_list:
                print(not_exist_table); undo_list.append(not_exist_table)
        ############################

        #存入確認結果
        status = (undo_list==[])
        outcome[str(ym)+'_status'] = status
        outcome[str(ym)+'_undo_list'] = undo_list
        if status : print(f'year: {ym} feature is no problem!!!!!!!!!')
    return outcome


# In[8]:


# outcome = check_undo_feature(['202112'])


# In[9]:


def check_length_and_index(do_concat_ym_lst,feature_file_path , number_of_check=2000):
    outcome = {}
    for ym in do_concat_ym_lst:
        #參數與宣告
        len_error_table = []
        idx_error_table = []
        col_error_table = []
        check_path = feature_file_path + '/' + f'{ym}'
        exist_list = listdir(check_path)
        df = pickle.load(open(check_path + '/' + f'CUST_{ym}.pickle', 'rb'))
        df_len = len(df)
        check_index_list = [random.randint(0,df_len-1) for n in range(number_of_check)]
        check_id_df = df['customer_id'][check_index_list]
        print('-----Check [length and index] info------')
        print(f'Year: {ym} Row Number: {df_len}')
        print('Check df:')
        print(check_id_df)

        for exist_file in exist_list:
            if 'ipynb_checkpoints' not in exist_file and 'CUST' not in exist_file:
                df = pickle.load(open(check_path + '/' + exist_file, 'rb'))
                df_compare = pickle.load(open(
                    feature_file_path + f'/{compare_column_ym}/'
                    + exist_file.split('.')[0][:-6] + f'{compare_column_ym}.pickle', 'rb'))
                #確認資料筆數
                if len(df) != df_len:
                    print(f'table: {exist_file} length not match!!!!!!!!!!!')
                    len_error_table.append(exist_file)
                #確認INDEX
                if sum(df['customer_id'][check_index_list] != check_id_df) == number_of_check:
                    print(f'table: {exist_file} ID not match!!!!!!!!!!!')
                    idx_error_table.append(exist_file)
                #確認欄位長度是否一致
                if exist_file[:-14] not in table_not_fixed_column_list  and len(df.columns) != len(df_compare.columns):
                    print(f'table: {exist_file} Columns size not match!!!!!!!!!!!')
                    col_error_table.append(exist_file)
        #存入確認結果
        status = (len_error_table==[] and idx_error_table==[] and col_error_table==[])
        outcome[str(ym)+'_status'] = status
        outcome[str(ym)+'_len_error_table'] = len_error_table
        outcome[str(ym)+'_idx_error_table'] = idx_error_table
        if status: print(f'year: {ym} length and index are no problem!!!!!!!!!!')
    return outcome


# In[10]:


# number_of_check=100
# check_length_and_index(do_concat_ym_lst, number_of_check)


# In[11]:


def concat_all_feature_df(do_concat_ym_lst, mother,feature_file_path , outcome = {}, writing_path = None,
                          drop_key_word = [], papu_and_y = pd.DataFrame(), just_for_check=False, Fill_zero=False,
                          if_old = False):
    def rename_dup_col(df):
        if sum(df.columns.duplicated(keep='first')) > 0 :
            dup_col_first = [i+'_x' for i in list(df.columns.values[df.columns.duplicated(keep='first')])]
            dup_col_last = [i+'_y' for i in list(df.columns.values[df.columns.duplicated(keep='last')])]
            dup_first_index = df.columns.duplicated(keep='first')
            dup_last_index = df.columns.duplicated(keep='last')
            df.columns.values[dup_first_index] = dup_col_first
            df.columns.values[dup_last_index] = dup_col_last
        return df

    for ym in do_concat_ym_lst:
        st_each_year_time = time.time()
        print('-----Concat datafram info------')
        #參數與宣告
        check_path = feature_file_path + '/' + f'{ym}'
        exist_list = listdir(check_path)
        #先抓客戶檔
        df = pickle.load(open(check_path + '/' + f'CUST_{ym}.pickle', 'rb'))
        df['yyyymm'] = df['yyyymm'].astype('int')
        print(f'CUST_{ym}.pickle loading finished')
        #抓取母體跟Y
        if (not papu_and_y.empty) :
#         if (not papu_and_y[papu_and_y['yyyymm']==int(ym)].empty) :
            papu_and_y['yyyymm'] = papu_and_y['yyyymm'].astype('int')
            papu_and_y_yyyymm = papu_and_y[papu_and_y['yyyymm']==int(ym)]
            if not papu_and_y_yyyymm.empty:
                df_papu_and_y = pd.merge(df, papu_and_y_yyyymm, how='left', on=['customer_id','yyyymm'])
                df_papu_and_y.drop(['customer_id','yyyymm'], axis=1, inplace=True)
                mask_papulation = df_papu_and_y['y'].notna()
                df_papu_and_y = df_papu_and_y[mask_papulation]
                df_papu_and_y['y'] = df_papu_and_y['y'].astype('int')
                print(f'papu_and_y loading finished with {len(df_papu_and_y.columns)} columns and {len(df_papu_and_y)} length ')
            else:
                raise ValueError(f"年月:{ym} 筆數為0")
        elif just_for_check:
            print('允許空的dataframe用於check特徵完成情形!')
        else:
            raise ValueError("papu_and_y筆數為0")

        if os.path.exists(table_fin_table_path):
            fin = pd.read_csv(table_fin_table_path)
            fin['ym'] = fin['ym'].astype('int')
            fin = fin.rename({'ym': 'yyyymm'}, axis=1)
            df_fin = pd.merge(df, fin, how='left', on=['yyyymm'])
            if (not papu_and_y.empty) :
                    #透過母體調整DF
                    df_fin = df_fin[mask_papulation]
                    df = df[mask_papulation]
            df_fin.drop(['customer_id','yyyymm'], axis=1, inplace=True)
            print(f'df_fin loading finished with {len(df_fin.columns)} columns and {len(df_fin)} length ')

        df_length = len(df) # for check length
        #放入需合併DF
        if (not papu_and_y.empty) :
            combine_table_list = [df, df_papu_and_y,df_fin]
        else:
            combine_table_list = [df,df_fin]
        if if_old:
            table_all_list = table_orginal_list + table_append_list
        else:
            table_all_list = table_append_list_2024
        for table in table_all_list:
            filename = table + f'_{ym}.pickle'
            if table != 'CUST' and filename in exist_list:
                df_temp = pickle.load(open(check_path + '/' + filename, 'rb'))
                if (not papu_and_y.empty) :
                    df_temp = df_temp[mask_papulation]
                #排除不要的特徵
                for drop_kw in drop_key_word:
                    df_temp.drop([a for a in df_temp.columns if (drop_kw in a)], axis=1, inplace=True)
                #拿掉customer_id還有yyyymmm
                df_temp.drop(['customer_id'], axis=1, inplace=True)
                if 'yyyymm' in df_temp.columns: df_temp.drop(['yyyymm'], axis=1, inplace=True)
                print(f'{table}_{ym}.pickle loading finished with {len(df_temp.columns)} columns and {len(df_temp)} length ')

                # NA補0
                if Fill_zero:
                    feature_list = list(df_temp.select_dtypes(exclude = ['category','object']))
                    na_totally_num = sum(df_temp[feature_list].isna().sum(axis=1))
                    print(f'補0前NA數量:{na_totally_num}')
                    df_temp[feature_list] = df_temp[feature_list].fillna(0)
                    print('NA補0完畢!!!')
                    na_totally_num = sum(df_temp[feature_list].isna().sum(axis=1))
                    print(f'補0後NA數量:{na_totally_num}')

                var_name = 'df'+ table
                locals()[var_name] =  df_temp
                combine_table_list.append(locals()[var_name])
        #合併TABLE
        df_combined = pd.concat(combine_table_list, axis= 1, join='outer')
        mask1 = len(df_combined)==df_length
        mask2 = len(df_combined) >= 1000
        print(f'{table}_{ym}.pickle concat finished , now is {len(df_combined.columns)} columns ')
        print(f'Concat df of {ym}, runtime: {(time.time()-st_each_year_time)/60} minutes')
        #存入確認結果
        status = ( mask1 and mask2)
        outcome[str(ym)+'_status'] = status
        df_combined = rename_dup_col(df_combined)
        outcome[str(ym)+'_df'] = df_combined
        if writing_path != None:
            st_dumping_time = time.time()
            table_filename = f'concat_feature_{ym}.pickle'
            col_filename = f'colname_{ym}.pickle'
            writing_path_fold = writing_path +'/'+ mother

            if not os.path.exists(writing_path_fold):
                os.mkdir(writing_path_fold)

            pickle.dump(df_combined, open(writing_path_fold +'/' +table_filename, 'wb'), protocol = 4)
            pickle.dump(df_combined.columns, open(writing_path_fold +'/' + col_filename, 'wb'), protocol = 4)
            print(f'all_fature{ym} dumping finished!!!!!!!!!!!!!!')
            print(f'dumping df of {ym}, runtime: {(time.time()-st_dumping_time)/60} minutes')

    return outcome


# In[12]:


def concat_all_feature_df_2024test(do_concat_ym_lst, mother,feature_file_path , outcome = {}, writing_path = None,
                          drop_key_word = [], papu_and_y = pd.DataFrame(), just_for_check=False, Fill_zero=False):
    def rename_dup_col(df):
        if sum(df.columns.duplicated(keep='first')) > 0 :
            dup_col_first = [i+'_x' for i in list(df.columns.values[df.columns.duplicated(keep='first')])]
            dup_col_last = [i+'_y' for i in list(df.columns.values[df.columns.duplicated(keep='last')])]
            dup_first_index = df.columns.duplicated(keep='first')
            dup_last_index = df.columns.duplicated(keep='last')
            df.columns.values[dup_first_index] = dup_col_first
            df.columns.values[dup_last_index] = dup_col_last
        return df

    for ym in do_concat_ym_lst:
        st_each_year_time = time.time()
        print('-----Concat datafram info------')
        #參數與宣告
        check_path = feature_file_path + '/' + f'{ym}'
        exist_list = listdir(check_path)
        #先抓客戶檔
        df = pickle.load(open(check_path + '/' + f'CUST_{ym}.pickle', 'rb'))
        df['yyyymm'] = df['yyyymm'].astype('int')
        print(f'CUST_{ym}.pickle loading finished')
        #抓取母體跟Y
        if (not papu_and_y.empty) :
#         if (not papu_and_y[papu_and_y['yyyymm']==int(ym)].empty) :
            papu_and_y['yyyymm'] = papu_and_y['yyyymm'].astype('int')
            papu_and_y_yyyymm = papu_and_y[papu_and_y['yyyymm']==int(ym)]
            if not papu_and_y_yyyymm.empty:
                df_papu_and_y = pd.merge(df, papu_and_y_yyyymm, how='left', on=['customer_id','yyyymm'])
                df_papu_and_y.drop(['customer_id','yyyymm'], axis=1, inplace=True)
                mask_papulation = df_papu_and_y['y'].notna()
                df_papu_and_y = df_papu_and_y[mask_papulation]
                df_papu_and_y['y'] = df_papu_and_y['y'].astype('int')
                print(f'papu_and_y loading finished with {len(df_papu_and_y.columns)} columns and {len(df_papu_and_y)} length ')
            else:
                raise ValueError(f"年月:{ym} 筆數為0")
        elif just_for_check:
            print('允許空的dataframe用於check特徵完成情形!')
        else:
            raise ValueError("papu_and_y筆數為0")

        if os.path.exists(table_fin_table_path):
            fin = pd.read_csv(table_fin_table_path)
            fin['ym'] = fin['ym'].astype('int')
            fin = fin.rename({'ym': 'yyyymm'}, axis=1)
            df_fin = pd.merge(df, fin, how='left', on=['yyyymm'])
            if (not papu_and_y.empty) :
                    #透過母體調整DF
                    df_fin = df_fin[mask_papulation]
                    df = df[mask_papulation]
            df_fin.drop(['customer_id','yyyymm'], axis=1, inplace=True)
            print(f'df_fin loading finished with {len(df_fin.columns)} columns and {len(df_fin)} length ')

        df_length = len(df) # for check length
        #放入需合併DF
        if (not papu_and_y.empty) :
            combine_table_list = [df, df_papu_and_y,df_fin]
        else:
            combine_table_list = [df,df_fin]
        for table in table_check_list:
            filename = table + f'_{ym}.pickle'
            if table != 'CUST' and filename in exist_list:
                df_temp = pickle.load(open(check_path + '/' + filename, 'rb'))
                if (not papu_and_y.empty) :
                    df_temp = df_temp[mask_papulation]
                #排除不要的特徵
                for drop_kw in drop_key_word:
                    df_temp.drop([a for a in df_temp.columns if (drop_kw in a)], axis=1, inplace=True)
                #拿掉customer_id還有yyyymmm
                df_temp.drop(['customer_id'], axis=1, inplace=True)
                if 'yyyymm' in df_temp.columns: df_temp.drop(['yyyymm'], axis=1, inplace=True)
                print(f'{table}_{ym}.pickle loading finished with {len(df_temp.columns)} columns and {len(df_temp)} length ')

                # NA補0
                if Fill_zero:
                    feature_list = list(df_temp.select_dtypes(exclude = ['category','object']))
                    na_totally_num = sum(df_temp[feature_list].isna().sum(axis=1))
                    print(f'補0前NA數量:{na_totally_num}')
                    df_temp[feature_list] = df_temp[feature_list].fillna(0)
                    print('NA補0完畢!!!')
                    na_totally_num = sum(df_temp[feature_list].isna().sum(axis=1))
                    print(f'補0後NA數量:{na_totally_num}')

                var_name = 'df'+ table
                locals()[var_name] =  df_temp
                combine_table_list.append(locals()[var_name])
        #合併TABLE
        df_combined = pd.concat(combine_table_list, axis= 1, join='outer')
        mask1 = len(df_combined)==df_length
        mask2 = len(df_combined) >= 1000
        print(f'{table}_{ym}.pickle concat finished , now is {len(df_combined.columns)} columns ')
        print(f'Concat df of {ym}, runtime: {(time.time()-st_each_year_time)/60} minutes')
        #存入確認結果
        status = ( mask1 and mask2)
        outcome[str(ym)+'_status'] = status
        df_combined = rename_dup_col(df_combined)
        outcome[str(ym)+'_df'] = df_combined
        if writing_path != None:
            st_dumping_time = time.time()
            table_filename = f'concat_feature_{ym}.pickle'
            col_filename = f'colname_{ym}.pickle'
            writing_path_fold = writing_path +'/'+ mother

            if not os.path.exists(writing_path_fold):
                os.mkdir(writing_path_fold)

            pickle.dump(df_combined, open(writing_path_fold +'/' +table_filename, 'wb'), protocol = 4)
            pickle.dump(df_combined.columns, open(writing_path_fold +'/' + col_filename, 'wb'), protocol = 4)
            print(f'all_fature{ym} dumping finished!!!!!!!!!!!!!!')
            print(f'dumping df of {ym}, runtime: {(time.time()-st_dumping_time)/60} minutes')

    return outcome


# In[13]:


# concat_all_feature_df(do_concat_ym_lst)


# In[14]:


def get_feature_by_SOP(do_ym_lst, mother = None, writing_path = None, drop_key_word = [], papu_and_y = pd.DataFrame(),
                       feature_file_path = feature_file_path_mlops, just_for_check=False, Fill_zero=False, if_old = False):
    st_time = time.time()
    concat_df_outcome = {}
    for ym in do_ym_lst:

        writing_path_fold = str(writing_path) +'/'+ str(mother)
        table_filename = f'concat_feature_{ym}.pickle'
        writing_path_file = writing_path_fold + '/' + table_filename

        if writing_path != None and os.path.exists(writing_path_file):
            print(f' year of {ym} df had been existed!!!!!')
            print(f' @@@@@@@@@@@@@@@@ Year of {ym} load info @@@@@@@@@@@@@@@@ ')
            print(f' year of {ym} df is loading....')
            df = pickle.load(open(writing_path_file, 'rb'))
            concat_df_outcome[str(ym)+'_df'] = df
            print(f' year of {ym} df was be finished ')
        else:
            print(f' @@@@@@@@@@@@@@@@ Year of {ym} combine info @@@@@@@@@@@@@@@@ ')
            ym_status = str(ym)+'_status'
            check_undo_outcome = check_undo_feature([ym], feature_file_path)
            print(check_undo_outcome)
            if check_undo_outcome[ym_status] :
                check_index_outcome = check_length_and_index([ym], feature_file_path)
                if check_index_outcome[ym_status] :
                    concat_df_outcome =                     concat_all_feature_df([ym], mother,feature_file_path, concat_df_outcome, writing_path,
                                          drop_key_word, papu_and_y, just_for_check, Fill_zero, if_old)
                    if concat_df_outcome[ym_status] :
                        print(f'Succssed !! df: featue_{ym} already concated')
                    else:
                        print(f'Concat failed, feature of {ym} !!' )
                else: print(f'Index or length had problem !! feature of {ym} !')
            else: print(f'Some table is unfinished !! feature of {ym} !')
    end_time = time.time()
    print(f'get_feature_by_SOP,  runtime: {round((end_time-st_time)/60)} minutes')
    return concat_df_outcome


# In[15]:


def get_feature_by_SOP_2024test(do_ym_lst, mother = None, writing_path = None, drop_key_word = [], papu_and_y = pd.DataFrame(),
                       feature_file_path = feature_file_path_mlops, just_for_check=False, Fill_zero=False):
    st_time = time.time()
    concat_df_outcome = {}
    for ym in do_ym_lst:

        writing_path_fold = str(writing_path) +'/'+ str(mother)
        table_filename = f'concat_feature_{ym}.pickle'
        writing_path_file = writing_path_fold + '/' + table_filename

        if writing_path != None and os.path.exists(writing_path_file):
            print(f' year of {ym} df had been existed!!!!!')
            print(f' @@@@@@@@@@@@@@@@ Year of {ym} load info @@@@@@@@@@@@@@@@ ')
            print(f' year of {ym} df is loading....')
            df = pickle.load(open(writing_path_file, 'rb'))
            concat_df_outcome[str(ym)+'_df'] = df
            print(f' year of {ym} df was be finished ')
        else:
            print(f' @@@@@@@@@@@@@@@@ Year of {ym} combine info @@@@@@@@@@@@@@@@ ')
            ym_status = str(ym)+'_status'
            check_undo_outcome = check_undo_feature([ym], feature_file_path)
            print(check_undo_outcome)
            if check_undo_outcome[ym_status] :
                check_index_outcome = check_length_and_index([ym], feature_file_path)
                if check_index_outcome[ym_status] :
                    concat_df_outcome =                     concat_all_feature_df_2024test([ym], mother,feature_file_path, concat_df_outcome, writing_path,
                                          drop_key_word, papu_and_y, just_for_check, Fill_zero)
                    if concat_df_outcome[ym_status] :
                        print(f'Succssed !! df: featue_{ym} already concated')
                    else:
                        print(f'Concat failed, feature of {ym} !!' )
                else: print(f'Index or length had problem !! feature of {ym} !')
            else: print(f'Some table is unfinished !! feature of {ym} !')
    end_time = time.time()
    print(f'get_feature_by_SOP,  runtime: {round((end_time-st_time)/60)} minutes')
    return concat_df_outcome


# In[16]:


def get_feature_by_SOP_jihsun(do_ym_lst, mother = None, writing_path = None, drop_key_word = [], papu_and_y = pd.DataFrame(),
                              feature_file_path = feature_file_path_jihsun, just_for_check=False, Fill_zero=False, if_old = False ):
    st_time = time.time()
    concat_df_outcome = {}
    for ym in do_ym_lst:

        writing_path_fold = str(writing_path) +'/'+ str(mother)
        table_filename = f'concat_feature_{ym}.pickle'
        writing_path_file = writing_path_fold + '/' + table_filename

        if writing_path != None and os.path.exists(writing_path_file):
            print(f' year of {ym} df had been existed!!!!!')
            print(f' @@@@@@@@@@@@@@@@ Year of {ym} load info @@@@@@@@@@@@@@@@ ')
            print(f' year of {ym} df is loading....')
            df = pickle.load(open(writing_path_file, 'rb'))
            concat_df_outcome[str(ym)+'_df'] = df
            print(f' year of {ym} df was be finished ')
        else:
            print(f' @@@@@@@@@@@@@@@@ Year of {ym} combine info @@@@@@@@@@@@@@@@ ')
            ym_status = str(ym)+'_status'
            check_undo_outcome = check_undo_feature([ym], feature_file_path)
            print(check_undo_outcome)
            if check_undo_outcome[ym_status] :
                check_index_outcome = check_length_and_index([ym], feature_file_path)
                if check_index_outcome[ym_status] :
                    concat_df_outcome =                     concat_all_feature_df([ym], mother,feature_file_path, concat_df_outcome, writing_path,
                                          drop_key_word, papu_and_y, just_for_check, Fill_zero, if_old)
                    if concat_df_outcome[ym_status] :
                        print(f'Succssed !! df: featue_{ym} already concated')
                    else:
                        print(f'Concat failed, feature of {ym} !!' )
                else: print(f'Index or length had problem !! feature of {ym} !')
            else: print(f'Some table is unfinished !! feature of {ym} !')
    end_time = time.time()
    print(f'get_feature_by_SOP,  runtime: {round((end_time-st_time)/60)} minutes')
    return concat_df_outcome


# In[17]:


def get_feature_by_SOP_jihsun_2024test(do_ym_lst, mother = None, writing_path = None, drop_key_word = [], papu_and_y = pd.DataFrame(),
                              feature_file_path = feature_file_path_jihsun, just_for_check=False, Fill_zero=False ):
    st_time = time.time()
    concat_df_outcome = {}
    for ym in do_ym_lst:

        writing_path_fold = str(writing_path) +'/'+ str(mother)
        table_filename = f'concat_feature_{ym}.pickle'
        writing_path_file = writing_path_fold + '/' + table_filename

        if writing_path != None and os.path.exists(writing_path_file):
            print(f' year of {ym} df had been existed!!!!!')
            print(f' @@@@@@@@@@@@@@@@ Year of {ym} load info @@@@@@@@@@@@@@@@ ')
            print(f' year of {ym} df is loading....')
            df = pickle.load(open(writing_path_file, 'rb'))
            concat_df_outcome[str(ym)+'_df'] = df
            print(f' year of {ym} df was be finished ')
        else:
            print(f' @@@@@@@@@@@@@@@@ Year of {ym} combine info @@@@@@@@@@@@@@@@ ')
            ym_status = str(ym)+'_status'
            check_undo_outcome = check_undo_feature([ym], feature_file_path)
            print(check_undo_outcome)
            if check_undo_outcome[ym_status] :
                check_index_outcome = check_length_and_index([ym], feature_file_path)
                if check_index_outcome[ym_status] :
                    concat_df_outcome =                     concat_all_feature_df_2024test([ym], mother,feature_file_path, concat_df_outcome, writing_path,
                                          drop_key_word, papu_and_y, just_for_check, Fill_zero)
                    if concat_df_outcome[ym_status] :
                        print(f'Succssed !! df: featue_{ym} already concated')
                    else:
                        print(f'Concat failed, feature of {ym} !!' )
                else: print(f'Index or length had problem !! feature of {ym} !')
            else: print(f'Some table is unfinished !! feature of {ym} !')
    end_time = time.time()
    print(f'get_feature_by_SOP,  runtime: {round((end_time-st_time)/60)} minutes')
    return concat_df_outcome


# In[18]:


def combine_multi_year_df(do_ym_lst, concat_df_outcome):
    colname_initial = concat_df_outcome[str(do_ym_lst[0])+'_df'].columns
    col_match_status = []
    for ym in do_ym_lst[1:] :
        colname = concat_df_outcome[str(ym)+'_df'].columns
        if len(colname) == len(colname_initial):
            if sum(colname == colname_initial) == len(colname):
                print(f'{ym} concat_df : colnames are no problemed')
                col_match_status.append(True)
            else:
                print(f'{ym} concat_df : colnames are not matched!!!!')
                col_match_status.append(False)
        else:
            print(f'{ym} concat_df : colsize is not matched!!!! ')
            col_match_status.append(False)
    if sum(col_match_status)+1 == len(do_ym_lst):
        st_time = time.time()
        combine_ym_table_list = []
        for ym in do_ym_lst :
            combine_ym_table_list.append(concat_df_outcome[str(ym)+'_df'])
        #合併TABLE
        print(f'{do_ym_lst} year will be concated ......')
        df_combine_ym = pd.concat(combine_ym_table_list, axis = 0, ignore_index=True)
        end_time = time.time()
        print(f'combine_multi_year_df,  runtime: {round((end_time-st_time)/60)} minutes')
        # 確認所有類別變數的型態
        print('Confirm categorical feature ...')
        st_time = time.time()
        obj_cols = df_combine_ym.select_dtypes('object').drop(['customer_id'],axis=1).columns
        obj_cols_list = obj_cols.tolist()
        if len(obj_cols_list) > 0 :
            df_combine_ym[obj_cols_list] = df_combine_ym[obj_cols_list].astype('category')
#             for obj_col in obj_cols_list:
#                 df_combine_ym[obj_col] = df_combine_ym[obj_col].astype('category')
        else:
            print('don"t have any object col for trans to categorical feature')
        print(f'trans to categorical feature,  runtime: {round((end_time-st_time)/60)} minutes')
        return df_combine_ym
    else:
        print('col_match_status is false')
        raise ValueError('col_match_status is false')



# In[19]:


def if_to_large_down_sampling(do_ym_lst, concat_df_outcome, limit_size=500000):
    st_time = time.time()
    for ym in sorted(do_ym_lst)[0:-1] :
        df = concat_df_outcome[str(ym)+'_df']
        if len(df) > limit_size:
            print(f'starting down sampling (year:{ym}) ...')
            df_y0 = df[df['y']==0]
            df_y1 = df[df['y']==1]
#             df_y0_downsampled = resample(df_y0, random_state=42, n_samples=limit_size-len(df_y1), replace=True)
            n_y0_samples = limit_size - len(df_y1)
            if n_y0_samples <= 0 :
                print(f'warning , the amount of Y =1 {len(df_y1)} exceed the limit size{limit_size} , so the amount of Y =1 and Y = 0 will become {limit_size//2}')
                df_y1 = resample(df_y1 , random_state=42 ,n_samples = limit_size//2, replace = len(df_y1) < limit_size//2)
                n_y0_samples = limit_size//2
            df_y0_downsampled = resample(df_y0 , random_state=42 ,n_samples = n_y0_samples,replace = True)
            #concat
            df_downsampled = pd.concat([df_y0_downsampled, df_y1])
            print(f'Successed!! down sampling (year:{ym})/ size:{len(df_downsampled)}) ...')
            concat_df_outcome[str(ym)+'_df'] = df_downsampled
        else:
            print(f'don"t need down sampling (year:{ym}/ size:{len(df)}) ...')
    end_time = time.time()
    print(f'if_to_large_down_sampling,  runtime: {round((end_time-st_time)/60)} minutes')
    return concat_df_outcome


# In[20]:


def if_large_down_sampling_from_papu_and_y(do_ym_lst, papu_and_y, cust_source, limit_size=500000):
    st_time = time.time()
    papu_and_y['yyyymm'] = papu_and_y['yyyymm'].astype('int')
    reduce_papu_and_y = papu_and_y[papu_and_y['yyyymm']==int(sorted(do_ym_lst)[-1])]

    if cust_source == 'Fubon':
        feature_file_path = config.feature_file_path_mlops
    elif cust_source == 'Jihsun':
        feature_file_path = config.feature_file_path_jihsun

    for ym in sorted(do_ym_lst)[0:-1] :
        print(f'-----Mapping CUST_{ym} info------')
        #參數與宣告
        check_path = feature_file_path + '/' + f'{ym}'
        print(f'PATH: {check_path}')
        exist_list = listdir(check_path)
        #先抓客戶檔
        df = pickle.load(open(check_path + '/' + f'CUST_{ym}.pickle', 'rb'))
        df['yyyymm'] = df['yyyymm'].astype('int')
        print(f'CUST_{ym}.pickle loading finished')

        if (not papu_and_y[papu_and_y['yyyymm']==int(ym)].empty) :
            # 抓取該yyyymm資料
            papu_and_y_yyyymm = papu_and_y[papu_and_y['yyyymm']==int(ym)]

            df_papu_and_y = pd.merge(df, papu_and_y_yyyymm, how='left', on=['customer_id','yyyymm'])
            mask_papulation = df_papu_and_y['y'].notna()
            papu_and_y_yyyymm = df_papu_and_y[mask_papulation]
            papu_and_y_yyyymm['y'] = papu_and_y_yyyymm['y'].astype('int')
            papu_and_y_yyyymm.reset_index(drop=True, inplace=True)
            print(f'papu_and_y mapping finished with {len(papu_and_y_yyyymm.columns)} columns and {len(papu_and_y_yyyymm)} length ')

            if len(papu_and_y_yyyymm) > limit_size:
                print(f'starting down sampling (year:{ym}) ...')
                papu_and_y_yyyymm_y0 = papu_and_y_yyyymm[papu_and_y_yyyymm['y']==0]
                papu_and_y_yyyymm_y1 = papu_and_y_yyyymm[papu_and_y_yyyymm['y']==1]
#                 papu_and_y_yyyymm_y0_downsampled = resample(papu_and_y_yyyymm_y0, random_state=42, n_samples=limit_size-len(papu_and_y_yyyymm_y1), replace=False)
                #concat
                n_y0_samples = limit_size - len(papu_and_y_yyyymm_y1)
                if n_y0_samples <= 0 :
                    print(f'warning , the amount of Y =1 {len(papu_and_y_yyyymm_y1)} exceed the limit size{limit_size} , so the amount of Y =1 and Y = 0 will become {limit_size//2}')
                    papu_and_y_yyyymm_y1 = resample(papu_and_y_yyyymm_y1 , random_state=42 ,n_samples = limit_size//2, replace = len(papu_and_y_yyyymm_y1) < limit_size//2)
                    n_y0_samples = limit_size//2
                papu_and_y_yyyymm_y0_downsampled = resample(papu_and_y_yyyymm_y0 , random_state=42 ,n_samples = n_y0_samples,replace = True)
                papu_and_y_yyyymm_downsampled = pd.concat([papu_and_y_yyyymm_y0_downsampled, papu_and_y_yyyymm_y1])
                print(f'Successed!! down sampling (year:{ym})/ size:{len(papu_and_y_yyyymm_downsampled)}) ...')
                print(papu_and_y_yyyymm_downsampled)
                reduce_papu_and_y = reduce_papu_and_y.append(papu_and_y_yyyymm_downsampled)
            else:
                print(f'don"t need down sampling (year:{ym}/ size:{len(papu_and_y_yyyymm)}) ...')
                reduce_papu_and_y = reduce_papu_and_y.append(papu_and_y_yyyymm)
        else:
            raise ValueError(f"年月:{ym} 筆數為0")
    #最新月份
    reduce_papu_and_y.reset_index(drop=True,inplace=True)
    print("Running time : %.0f sec" %(time.time() - st_time))

    return reduce_papu_and_y


# In[21]:


#固定母體 如果已經抓過了 就不送 query
def get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
                   writing_popu_path,mother,
                   account1=config.account , pwd1=config.pwd,
                   account2=config.account_kris, pwd2=config.pwd_kris, failed_limit_times = 4,
                   y_type_name = ''):
    from datetime import date, datetime

    from Model import monthdelta
    from Sql_module import get_SQL_raw_data
    papu_and_y = pd.DataFrame()
    chage_acct = True
    failed_times = 0
    for ym in do_ym_list:
        date =  datetime.date(datetime.strptime(ym,'%Y%m'))
        next_month = monthdelta(date, 1)
        print(next_month)

        writing_path_fold = str(writing_popu_path) +'/'+ str(mother)
        if writing_popu_path != None and not os.path.exists(writing_path_fold):
            os.mkdir(writing_path_fold)
        table_filename = 'papu_and_y' + str(y_type_name) + f'_{ym}.pickle'
        writing_path_file = writing_path_fold + '/' + table_filename
        if writing_popu_path != None and os.path.exists(writing_path_file):
            print(f' year of {ym} papu_and_y had been existed!!!!!')

            df_temp = pickle.load(open(writing_path_file, 'rb'))
            # 抓target y的欄位名稱
            for i, col in enumerate(df_temp.columns):
                if 'y' in col and 'yyyy' not in col:
                    y_colname = col
            size = len(df_temp)
            y_number = sum(df_temp[y_colname]==1)

            print(f'yyyymm: {ym} propotion: {y_number/size}')
            print(f'size of df {size} and number of y is {y_number}')

            print(f' year of {ym} papu_and_y loading finished ')
        else:
            # 判斷query_function是回傳
            if y_type_name == '':
                query = query_function(next_month)
            elif y_type_name == '2':
                query = query_function(next_month, int(y_type_name))
            if type(query) == str:
                print('該Query function是傳回SQL語法')
                if chage_acct:
                    df_temp = get_SQL_raw_data(query, account1, pwd1)
                    time.sleep(10)
                else:
                    df_temp = get_SQL_raw_data(query, account2, pwd2)
                    time.sleep(10)
                chage_acct = not chage_acct
                time.sleep(10)
            elif isinstance(query,pd.DataFrame):
                print('該Query function是傳回DataFrame')
                df_temp = query

            if isinstance(df_temp,pd.DataFrame) and len(df_temp) > 0:
                print(f'此月份筆數為: {len(df_temp)}')
                pickle.dump(df_temp, open(writing_path_file, 'wb'), protocol = 4)
            elif isinstance(df_temp,pd.DataFrame) and len(df_temp) == 0:
                print('此月份筆數為: 0 ')
                failed_times = failed_times+1
            elif not isinstance(df_temp,pd.DataFrame):
                print(f'SQL Query 錯誤LOG: \n {df_temp}')

            if failed_times >= failed_limit_times:
                raise ValueError("已經兩次抓不到papu_and_y")

        if isinstance(df_temp,pd.DataFrame):
            papu_and_y = papu_and_y.append(df_temp)
        #print(papu_and_y)
    return papu_and_y


# In[ ]:


# mlops撈資料庫現成母體
def get_papu_and_y_202503(this_prod, do_ym_list, query_function, papulation_colname, papulation_train_value,mother,
                account1=config.account , pwd1=config.pwd,
                account2=config.account_kris, pwd2=config.pwd_kris,
                failed_limit_times = 4, y_type_name = ''):
    from datetime import date, datetime

    from Sql_module import get_SQL_raw_data2
    papu_and_y = pd.DataFrame()
    chage_acct = True
    failed_times = 0
    for ym in do_ym_list:
        date =  datetime.date(datetime.strptime(ym,'%Y%m')).strftime('%Y%m')
        print(date)

        query = query_function(this_prod , mother , date, papulation_colname)

        if type(query) == str:
            print('該Query function是傳回SQL語法')
            if chage_acct:
                df_temp = get_SQL_raw_data2(query, account1, pwd1)
                print('y')
                time.sleep(5)
            else:
                df_temp = get_SQL_raw_data2(query, account2, pwd2)
                print('N')
                time.sleep(5)
            chage_acct = not chage_acct
            time.sleep(5)
        elif isinstance(query,pd.DataFrame):
            print('該Query function是傳回DataFrame')
            df_temp = query

        if isinstance(df_temp,pd.DataFrame) and len(df_temp) > 0:
            col = 'y'
            print(f'此月份筆數為: {len(df_temp)}')
            print(f'此月份Y有: {df_temp[col].sum()}')
        elif isinstance(df_temp,pd.DataFrame) and len(df_temp) == 0:
            print('此月份筆數為: 0 ')
            failed_times = failed_times+1
        elif not isinstance(df_temp,pd.DataFrame):
            print(f'SQL Query 錯誤LOG: \n {df_temp}')

        if failed_times >= failed_limit_times:
            raise ValueError("已經多次抓不到papu_and_y")

        if isinstance(df_temp,pd.DataFrame):
            papu_and_y = papu_and_y.append(df_temp)
        #print(papu_and_y)
    return papu_and_y


# In[22]:


#固定母體 如果已經抓過了 就不送 query
def get_papu_and_y_local(do_ym_list, query_function, papulation_colname, papulation_train_value,
                   writing_popu_path,mother,
                   account1=config.account, pwd1=config.pwd,
                   account2=config.account_kris, pwd2=config.pwd_kris, failed_limit_times = 4,
                   y_type_name = ''):
    from datetime import date, datetime

    from Model import monthdelta
    papu_and_y = pd.DataFrame()
    chage_acct = True
    failed_times = 0
    for ym in do_ym_list:
        date =  datetime.date(datetime.strptime(ym,'%Y%m'))
        next_month = monthdelta(date, 1)
        print(next_month)

        writing_path_fold = str(writing_popu_path) +'/'+ str(mother)
        if writing_popu_path != None and not os.path.exists(writing_path_fold):
            os.mkdir(writing_path_fold)
        table_filename = 'papu_and_y' + str(y_type_name) + f'_{ym}.pickle'
        writing_path_file = writing_path_fold + '/' + table_filename
        if writing_popu_path != None and os.path.exists(writing_path_file):
            print(f' year of {ym} papu_and_y had been existed!!!!!')

            df_temp = pickle.load(open(writing_path_file, 'rb'))
            # 抓target y的欄位名稱
            for i, col in enumerate(df_temp.columns):
                if 'y' in col and 'yyyy' not in col:
                    y_colname = col
            size = len(df_temp)
            y_number = sum(df_temp[y_colname]==1)

            print(f'yyyymm: {ym} propotion: {y_number/size}')
            print(f'size of df {size} and number of y is {y_number}')

            print(f' year of {ym} papu_and_y loading finished ')
        else:
            raise ValueError(f'yyyymm: {ym} 沒有先跑好母體')
        if isinstance(df_temp,pd.DataFrame):
            papu_and_y = papu_and_y.append(df_temp)
        #print(papu_and_y)
    return papu_and_y


# In[23]:


#固定母體 如果已經抓過了 就不送 query
def ntb_get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
                   writing_popu_path,mother,
                   account1=config.account_kris , pwd1=config.pwd_kris,
                   account2=config.account_kris, pwd2=config.pwd_kris, failed_limit_times = 4,predict_dump=False):
    from datetime import date, datetime

    from Sql_module import get_SQL_raw_data
    papu_and_y = pd.DataFrame()
    chage_acct = True
    failed_times = 0
    for ym in do_ym_list:
        date =  datetime.date(datetime.strptime(ym,'%Y%m%d'))

        writing_path_fold = str(writing_popu_path) +'/'+ str(mother)
        if writing_popu_path != None and not os.path.exists(writing_path_fold):
            os.mkdir(writing_path_fold)
        table_filename = f'papu_and_y_{ym}.pickle'
        writing_path_file = writing_path_fold + '/' + table_filename
        if writing_popu_path != None and os.path.exists(writing_path_file):
            print(f' year of {ym} papu_and_y had been existed!!!!!')

            df_temp = pickle.load(open(writing_path_file, 'rb'))

            size = len(df_temp)
            y_number = sum(df_temp['y']==1)

            print(f'yyyymm: {ym} propotion: {y_number/size}')
            print(f'size of df {size} and number of y is {y_number}')

            print(f' year of {ym} papu_and_y loading finished ')
        else:
            # 判斷query_function是回傳
            query = query_function(date)
            if type(query) == str:
                print('該Query function是傳回SQL語法')
                if chage_acct:
                    df_temp = get_SQL_raw_data(query, account1, pwd1)
                    time.sleep(120)
                else:
                    df_temp = get_SQL_raw_data(query, account2, pwd2)
                    time.sleep(120)
                chage_acct = not chage_acct
                time.sleep(30)
            elif isinstance(query,pd.DataFrame):
                print('該Query function是傳回DataFrame')
                df_temp = query

            if isinstance(df_temp,pd.DataFrame) and len(df_temp) > 0:
                print(f'此月份筆數為: {len(df_temp)}')
                if predict_dump:
                    print("預測不儲存")
                else:
                    pickle.dump(df_temp, open(writing_path_file, 'wb'), protocol = 4)
            elif isinstance(df_temp,pd.DataFrame) and len(df_temp) == 0:
                print('此月份筆數為: 0 ')
                failed_times = failed_times+1
            elif not isinstance(df_temp,pd.DataFrame):
                print(f'SQL Query 錯誤LOG: \n {df_temp}')

            if failed_times >= failed_limit_times:
                raise ValueError("已經兩次抓不到papu_and_y")

        if isinstance(df_temp,pd.DataFrame):
            papu_and_y = papu_and_y.append(df_temp)
        #print(papu_and_y)
    return papu_and_y


# In[24]:


def ntb_concat_all_feature(do_concat_ym_lst, mother,feature_file_path = feature_file_path_ntb , writing_path = None,
                          drop_key_word = [], papu_and_y = pd.DataFrame(), Fill_zero=False):


    combine_table_list = []

    for ym in do_concat_ym_lst:
        st_each_year_time = time.time()

        writing_path_fold = str(writing_path) +'/'+ str(mother)
        table_filename = f'concat_feature_{ym}.pickle'
        writing_path_file = writing_path_fold + '/' + table_filename

        print('-----Concat datafram info------')
        #參數與宣告
        check_path = feature_file_path + '/' + f'{ym}'
        exist_list = listdir(check_path)
        #先抓軌跡檔
        df_digital = pickle.load(open(check_path + f'/軌跡{ym}.pickle', 'rb'))
        print(f'軌跡{ym}.pickle loading finished')
        #抓取母體跟Y
        if (not papu_and_y.empty) :
#         if (not papu_and_y[papu_and_y['yyyymm']==int(ym)].empty) :
            papu_and_y['yyyymm'] = papu_and_y['yyyymm'].astype('int')
            papu_and_y_yyyymm = papu_and_y[papu_and_y['yyyymm']==int(ym)]
            if not papu_and_y_yyyymm.empty:
                df_papu_and_y = pd.merge(papu_and_y_yyyymm,df_digital , how='inner', left_on=['customer_id'],right_on=['uid_']).drop('uid_',axis = 1)
                print(f'papu_and_y loading finished with {len(df_papu_and_y.columns)} columns and {len(df_papu_and_y)} length ')
            else:
                raise ValueError(f"年月:{ym} 筆數為0")
        else:
            raise ValueError("papu_and_y筆數為0")


        df_score = pickle.load(open(check_path + '/分數_'+ mother +f'_{ym}.pickle', 'rb'))
        print(f'分數: {len(df_score)}')
        df_combined = pd.merge(df_papu_and_y, df_score, how='left', left_on=['customer_id'],right_on=['party_id']).drop('party_id',axis = 1)

        # NA補0
        if Fill_zero:
            df_combined = df_combined.fillna(0)
            print('NA補0完畢!!!')

#        pickle.dump(df_combined, open(writing_path_file, 'wb'), protocol = 4)
        locals()[ym] = df_combined
        combine_table_list.append(locals()[ym])


    outcome = pd.concat(combine_table_list, join='outer')

    return outcome


# In[ ]:


def get_recommded_build_size(df, mother, test_propotion=0.1 ):
    ##################################################
    # 樣本建議比例是透過同月份下各商品去執行不同的樣本比例不斷Retrain
    # 會獲得各個回測的AUC以及萬人覆蓋率，這兩項相加用於判斷哪個樣本比例是最佳解後
    # 將Y=1的數量以 1~2000 / 2000~10000 / >10000 進行分組
    # 分組後將樣本比例取平均作為建議的樣本比例
    # !!!!!!! 補充 價值以上跟潛客為分開給予建議樣本比例
    #
    ###############################################
    # df 需要 yyyymm & y
    #####################################
    r1 = df['yyyymm'] != max(df['yyyymm'])
    r2 = df['y'] == 1
    y1_num = len(df[r1 & r2])
    if mother == '潛客':
        if y1_num <= 2000:
            y1_propotion = 0.001477
        elif y1_num <= 10000:
            y1_propotion = 0.005346
        elif y1_num > 10000:
            y1_propotion = 0.03583
    elif mother == '非潛客':
        if y1_num <= 2000:
            y1_propotion = 0.001049
        elif y1_num <= 10000:
            y1_propotion = 0.005516
        elif y1_num > 10000:
            y1_propotion = 0.019517
    build_size = int(y1_num / y1_propotion/ (1-test_propotion) )

    if len(df[r1]) <= build_size:
        need_adjust_sample = False
    else:
        need_adjust_sample = True
    return need_adjust_sample, build_size


# In[ ]:


def down_sampling_from_df(df, recommded_build_size):
    ############################
    # 具有特徵的df 需要 yyyymm & y
    ###############################
    # 先獨立回測年月資料
    r1 = df['yyyymm'] != max(df['yyyymm'])
    backtest_ym_df = df[~r1]

    # need_downsample的資料
    df_train_valid = df[r1]

    # 設定每個年月資料大小
    train_months_n = (len(set(df['yyyymm'])) - 1 )
    target_size = int(recommded_build_size / train_months_n)
    print(f'recommded_build_size = {recommded_build_size}')
    print(f'train_months_n = {train_months_n}')
    print(f'target_size = {target_size}')

    def downsample(group, target_size):
        y_1_data = group[group['y'] == 1]
        y_0_data = group[group['y'] == 0]

        if target_size < len(y_1_data):
            raise ValueError(f"target_size ({target_size}) 小於 y=1 資料量 ({len(y_1_data)})")
        # 計算所需 y = 0資料數量
        required_y_0 = target_size - len(y_1_data)
        # 抽樣
        y_0_sampled = y_0_data.sample(n=min(required_y_0, len(y_0_data)), random_state=42)

        return pd.concat([y_1_data, y_0_sampled])

    # 按yyyymm分組並處理
    downsample_df = df_train_valid.groupby('yyyymm').apply(lambda group: downsample(group, target_size)).reset_index(drop=True)

    return pd.concat([downsample_df, backtest_ym_df])


# In[1]:


# !jupyter nbconvert --to script Pretreatment.ipynb


# In[ ]:




