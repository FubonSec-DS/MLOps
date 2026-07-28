#!/usr/bin/env python

# In[1]:


# loading parameter
import sys

sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
import config

# feature_file_path_fubon = config.feature_file_path_fubon
feature_file_path_mlops = config.feature_file_path_mlops
feature_file_path_ntb = config.feature_file_path_ntb
dump_feature_log_path = config.dump_feature_log_path
account, pwd = config.account, config.pwd
table_orginal_list = config.table_orginal_list
table_append_list = config.table_append_list
colname_object_list = config.colname_object_list
colname_catgory_list_1 = config.colname_catgory_list_1
colname_catgory_list_2 = config.colname_catgory_list_2
colname_catgory_list_3 = config.colname_catgory_list_3
colname_catgory_list_4 = config.colname_catgory_list_4
colname_catgory_list_5 = config.colname_catgory_list_5
colname_catgory_list_6 = config.colname_catgory_list_6


# In[2]:


import os
import pickle
import time
from datetime import datetime

import cx_Oracle
import numpy as np
import pandas as pd
from sqlalchemy.types import Boolean, DateTime, Float, Integer, String

# In[3]:


def create_feature_dir(input_ym ,feature_file_path = feature_file_path_mlops):
    path = feature_file_path + '/' + input_ym + '/'
    if not os.path.exists(path):
        os.mkdir(path)
        print(path, 'directory has been created')
    else:
        print('directory had already existed')


# In[4]:


HOST = config.HOST # IP for ods
PORT =  config.PORT
SEVICE_NAME = config.SEVICE_NAME


# In[5]:


def get_SQL_raw_data(query, account=account, pwd=pwd, table=''):
    from sqlalchemy import create_engine

    combined = query.lower() + ' ' + table.lower()
    if 'ntb' in combined:
        account = config.account
        pwd = config.pwd
    elif 'mlops' in combined:
        account = config.iaccount
        pwd = config.ipwd


    ACCOUNT = account # user name for ods
    PASSWORD = pwd # password

    #     engine = create_engine(f'oracle+cx_oracle://{ACCOUNT}:{PASSWORD}@{IP}:{PORT}/{DATABASE}')
    oracle_connection_string_fmt = (
        'oracle+cx_oracle://{ACCOUNT}:{PASSWORD}@' +
        cx_Oracle.makedsn('{HOST}','{PORT}', service_name = '{SEVICE_NAME}')
    )
    url = oracle_connection_string_fmt.format(ACCOUNT = account,
                                       PASSWORD = PASSWORD,
                                       HOST = HOST, PORT = PORT,
                                       SEVICE_NAME = SEVICE_NAME)
    engine = create_engine(url)


    t0 = time.time()

    query_status = False


    try:
        sql_query = query # sql command
#         with engine.begin() as conn:
        df = pd.read_sql(sql_query, engine) # DB table in python
#         conn.close()
        t1 = time.time()
        print("Running time of %s: %.0f sec" %(table, t1 - t0))
        print(table, 'loading completed')
        query_status = True
        engine.dispose()
        if len(df) > 0:
            return df
        else:
            print('length of dataframme is 0 !!')
            return pd.DataFrame()
    except Exception as e:
        print(f'[使用帳號] {account}')
        print(table, str(e))
        engine.dispose()
        return str(e)

    engine.dispose()

#     if query_status:
#         if len(df) > 0:
#             return df
#         else:
#             print('length of dataframme is 0 !!')
#             return pd.DataFrame()
#     else:
#         print('Query Failed !!!!')
#         return pd.DataFrame()


# In[6]:


def get_SQL_raw_data2(query, account=account, pwd=pwd, table=''):
    combined = query.lower() + ' ' + table.lower()
    if 'ntb' in combined:
        account = config.account
        pwd = config.pwd
    elif 'mlops' in combined:
        account = config.iaccount
        pwd = config.ipwd

    ACCOUNT = account # user name for ods
    PASSWORD = pwd # password

    dsn = f"{HOST}:{PORT}/{SEVICE_NAME}"
    conn = cx_Oracle.connect(ACCOUNT,PASSWORD,dsn)
    cursor = conn.cursor()
    cursor.arraysize = 10000


    t0 = time.time()

    query_status = False


    try:
        print("執行查詢")
        cursor.execute(query)
        column_names = [item.lower() if isinstance(item,str) else item for item in [desc[0] for desc in cursor.description]]
        data = cursor.fetchall()
        t1 = time.time()
#         print("取回資料 : %.0f sec" %( t1 - t0))
        try:
            cursor.close()
            conn.close()
        except:
            print(str(e))

        df = pd.DataFrame(data,columns = column_names)
        t2 = time.time()
#         print("建arrow+轉pandas : %.0f sec" %( t2 - t1))

        print("Running time of : %.0f sec" %( t2 - t0))
        print('loading completed')
        query_status = True
        if len(df) > 0:
            return df
        else:
            print('length of dataframme is 0 !!')
            return pd.DataFrame()
    except Exception as e:
        print(f'[使用帳號] {account}')
        print( str(e))
        cursor.close()
        conn.close()
        return str(e)

    engine.dispose()


# In[7]:


def execute_sql(query, account=account, pwd=pwd, table=''):
    from sqlalchemy import create_engine
    ACCOUNT = account # user name for ods
    PASSWORD = pwd # password

    #     engine = create_engine(f'oracle+cx_oracle://{ACCOUNT}:{PASSWORD}@{IP}:{PORT}/{DATABASE}')
    oracle_connection_string_fmt = (
        'oracle+cx_oracle://{ACCOUNT}:{PASSWORD}@' +
        cx_Oracle.makedsn('{HOST}','{PORT}', service_name = '{SEVICE_NAME}')
    )
    url = oracle_connection_string_fmt.format(ACCOUNT = account,
                                       PASSWORD = PASSWORD,
                                       HOST = HOST, PORT = PORT,
                                       SEVICE_NAME = SEVICE_NAME)
    engine = create_engine(url)

    t0 = time.time()
    try:
        sql_query = query # sql command
        with engine.begin() as conn:
            conn.execute(sql_query)
        conn.close()
        t1 = time.time()
        print("Running time of %s: %.0f sec" %(table, t1 - t0))
        print(table, 'executing completed')
    except Exception as e:
        print(table, str(e))

    engine.dispose()


# In[8]:


def drop_SQL_raw_data(account, pwd, table_name):
    from sqlalchemy import MetaData, create_engine
    from sqlalchemy.ext.declarative import declarative_base

    ACCOUNT = account # user name for ods
    PASSWORD = pwd # password


    #     engine = create_engine(f'oracle+cx_oracle://{ACCOUNT}:{PASSWORD}@{IP}:{PORT}/{DATABASE}')
    oracle_connection_string_fmt = (
        'oracle+cx_oracle://{ACCOUNT}:{PASSWORD}@' +
        cx_Oracle.makedsn('{HOST}','{PORT}', service_name = '{SEVICE_NAME}')
    )
    url = oracle_connection_string_fmt.format(ACCOUNT = account,
                                       PASSWORD = PASSWORD,
                                       HOST = HOST, PORT = PORT,
                                       SEVICE_NAME = SEVICE_NAME)
    engine = create_engine(url)


    try:
        base = declarative_base()
        metadata = MetaData(engine, reflect=True)
        table = metadata.tables.get(table_name)
        if table is not None:
            query = 'drop table '+table_name+' purge'
            with engine.begin() as conn:
                conn.execute(query)
            t1 = time.time()
            conn.close()
            print(table_name, 'deleting')
        else:
            print(table_name, 'did not exist')
    except Exception as e:
        print(table, str(e))

    engine.dispose()


# In[9]:


def write_data_to_SQL(table_name, df, account=account, pwd=pwd, exist_action='append', chunk_size=10000, col_types=None):
#     col_types = {
#     '模型名稱' : String(30),
#     '母體' : String(30),
#     'retrain日期' : String(30),
#     'train_period' : String(250),
#     'valid_period': String(250),
#     'test_period': String(30),
#     'valid_auc': Float(),
#     'test_auc': Float(),
#     '訓練母體人數': Integer(),
#     '訓練母體y=1人數': Integer(),
#     'VALID_人數' :String(200),
#     'VALID_購買率' :String(200),
#     'TEST_人數' :String(200),
#     'TEST_購買率' :String(200),
#     '是否通過標準': String(30),
#     '版本':String(10),
#     'train_auc': Float() }
    from sqlalchemy import create_engine

    if 'ntb' in table_name.lower():
        account = config.account
        pwd = config.pwd

    elif 'mlops' in table_name.lower():
        account = config.iaccount
        pwd = config.ipwd
    ACCOUNT = account # user name for ods
    PASSWORD = pwd # password

    #     engine = create_engine(f'oracle+cx_oracle://{ACCOUNT}:{PASSWORD}@{IP}:{PORT}/{DATABASE}')
    oracle_connection_string_fmt = (
        'oracle+cx_oracle://{ACCOUNT}:{PASSWORD}@' +
        cx_Oracle.makedsn('{HOST}','{PORT}', service_name = '{SEVICE_NAME}')
    )
    url = oracle_connection_string_fmt.format(ACCOUNT = account,
                                       PASSWORD = PASSWORD,
                                       HOST = HOST, PORT = PORT,
                                       SEVICE_NAME = SEVICE_NAME)
    print('use this:',ACCOUNT,PASSWORD)
    engine = create_engine(url)

    table_name = table_name.upper()
    if '.' in table_name :
        schema_name, table_name =  table_name.split(".",1)
    elif 'NTB'  in table_name or 'MLOPS' in table_name:
        schema_name = config.iaccount.upper()
    else:
        schema_name = None

    print(f"ready to write down to table name:{table_name}")
    try:
        t0 = time.time()
        with engine.begin() as conn:
            df.to_sql(table_name, engine,schema = schema_name, if_exists=exist_action, index=False, chunksize=chunk_size, dtype=col_types)
        conn.close()
        t1 = time.time()
        print("[Success writing down to db] Running time : %.0f sec" %(t1 - t0))
    except Exception as e:
        print(table_name, str(e))
        print("[Failed writing down to db]" )

    engine.dispose()


# In[10]:


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


# In[11]:


def dump_table_log(table_name, feature_date, status, import_date):
    log_path = dump_feature_log_path + f'/table_log_{feature_date}.pickle'
    if not os.path.exists(log_path):
        t_log = {'0':
                    {
                    'table_name' : table_name,
                    'feature_date' : feature_date,
                    'status': status,
                    'import_date': import_date
                    }
                }
        with open(log_path, 'wb') as handler:
            pickle.dump(t_log, handler, protocol=pickle.HIGHEST_PROTOCOL)
        print('table_log on', feature_date, 'has been inserted')
    else:
        t_log = pd.read_pickle(log_path)
        t_log.update({str(len(t_log)):
                       {
                        'table_name' : table_name,
                        'feature_date' : feature_date,
                        'status': status,
                        'import_date': import_date
                        }
                     })
        with open(log_path, 'wb') as handler:
            pickle.dump(t_log, handler, protocol=pickle.HIGHEST_PROTOCOL)
        print('table_log on year:', feature_date, 'has been updated')


# In[12]:


len(table_orginal_list + table_append_list )
table_orginal_list + table_append_list


# In[13]:


def feature_dump(yyyymm_lst, sql_table_name_list, feature_file_path,account=account, pwd=pwd):
    account = account # user name for ods
    pwd = pwd # password
    for ym in yyyymm_lst:
        # 確認是否有存在資料夾,若沒有則創建
        create_feature_dir(input_ym=ym)
        # LOOP
        for table in sql_table_name_list:
            # filename by case
            if str(table)[-1] != '_':
                file_path = feature_file_path + '/' + ym + '/' + table + '_' + ym + '.pickle'
            else:
                file_path = feature_file_path + '/' + ym + '/' + table + ym + '.pickle'
            # if pickle files existed, ignored them to prevent overwritting
            if os.path.exists(file_path):
                print(table, 'already existed')
                delete_table = table.lower()
                if table != 'CUST_':
                    drop_SQL_raw_data(account, pwd, delete_table)
            else:
                print(table, 'starting query')
                query = f"""SELECT * FROM {table}"""
                df = get_SQL_raw_data(query=query, account=account, pwd=pwd, table=table)

#                 time.sleep(10)
                if df is not None:
#                     if table != 'CUST_':
#                         IDFOLLOW = pickle.load(open(feature_file_path + '/' + ym + '/CUST_' + ym + '.pickle', 'rb'))['customer_id']
#                         df = pd.merge(IDFOLLOW, df, how='left', on=['customer_id'])
                    stime = time.time()
                    df = reduce_mem_usage(df)
                    if len(df) > 0:
                        #將 id,ym 轉成 object type
                        for col in colname_object_list:
                            if col in list(df.columns):
                                df[col] = df[col].astype('object')
                        #將 profile 內的類別型變數轉換成 category type
                        if table == 'FTBD_profile_':
                            df[colname_catgory_list_1] = df[colname_catgory_list_1].astype('category')
                            print(colname_catgory_list_1)
                            if 'c_acct_valid_flag' in df.columns:
                                df[colname_catgory_list_2] = df[colname_catgory_list_2].astype('category')
                                print(colname_catgory_list_2)
                        if table == 'FTBD_KYCQA_':
                            df[colname_catgory_list_3] = df[colname_catgory_list_3].astype('category')
                            print(colname_catgory_list_3)
                        if table == 'FTBD_EVENT_':
                            df[colname_catgory_list_4] = df[colname_catgory_list_4].astype('category')
                            print(colname_catgory_list_4)
                        if table == 'FTBD_JCI_':
                            df[colname_catgory_list_5] = df[colname_catgory_list_5].astype('category')
                            print(colname_catgory_list_5)
                        if table == 'FTBD_DGT1_':
                            df[colname_catgory_list_6] = df[colname_catgory_list_6].astype('category')
                            print(colname_catgory_list_6)
                        #將剩餘 object 特徵轉成 float type
                        if 'yyyymm' in list(df.columns):
                            cols = df.select_dtypes('object').drop(['customer_id','yyyymm'],axis=1).columns
                        else:
                            cols = df.select_dtypes('object').drop(['customer_id'],axis=1).columns
                        if cols.tolist() != []:
                            print('trans to float: ')
                            df[cols.tolist()] = df[cols.tolist()].astype('float')
                        #轉完型態後寫入CDSW
                        pickle.dump(df, open(file_path, 'wb'), protocol = 4)
                        dump_table_log(table_name=table, feature_date=ym, status='Y', import_date=datetime.now().strftime('%Y%m%d %H:%M:%S'))
                        print('save', table, 'completed')
                        #drop the table from sql
                        time.sleep(1)
                        delete_table = table.lower()
                        if table != 'CUST_':
                            drop_SQL_raw_data(account, pwd, delete_table)
                    else:
                        dump_table_log(table_name=table, feature_date=ym, status='N', import_date=datetime.now().strftime('%Y%m%d %H:%M:%S'))
                        print(table, 'did not have data')
                    etime = time.time() - stime
                    print('saving table', table, 'requires', etime, 'sec')
                else:
                    print(table, 'no data on SQL')
#             time.sleep(10)


# In[14]:


def feature_dump_202504(yyyymm_lst, sql_table_name_list, feature_file_path,account=account, pwd=pwd):
    account = account # user name for ods
    pwd = pwd # password
    for ym in yyyymm_lst:
        # 確認是否有存在資料夾,若沒有則創建
        create_feature_dir(input_ym=ym)
        # LOOP
        for table in sql_table_name_list:
            # filename by case
            if str(table)[-1] != '_':
                file_path = feature_file_path + '/' + ym + '/' + table + '_' + ym + '.pickle'
            else:
                file_path = feature_file_path + '/' + ym + '/' + table + ym + '.pickle'
            # if pickle files existed, ignored them to prevent overwritting
            if os.path.exists(file_path):
                print(table, 'already existed')
                delete_table = table.lower()
                if table != 'CUST_':
                    drop_SQL_raw_data(account, pwd, delete_table)
            else:
                print(table, 'starting query')
                query = f"""SELECT * FROM {table}"""
                df = get_SQL_raw_data(query=query, account=account, pwd=pwd, table=table)

#                 time.sleep(10)
                if df is not None:
#                     if table != 'CUST_':
#                         IDFOLLOW = pickle.load(open(feature_file_path + '/' + ym + '/CUST_' + ym + '.pickle', 'rb'))['customer_id']
#                         df = pd.merge(IDFOLLOW, df, how='left', on=['customer_id'])
                    stime = time.time()
                    df = reduce_mem_usage(df)
                    if len(df) > 0:
                        #將 id,ym 轉成 object type
                        for col in colname_object_list:
                            if col in list(df.columns):
                                df[col] = df[col].astype('object')
                        #將 profile 內的類別型變數轉換成 category type
                        if table == 'FTBD_profile_':
                            df[colname_catgory_list_1] = df[colname_catgory_list_1].astype('category')
                            print(colname_catgory_list_1)
                            if 'c_acct_valid_flag' in df.columns:
                                df[colname_catgory_list_2] = df[colname_catgory_list_2].astype('category')
                                print(colname_catgory_list_2)
                        if table == 'FTBD_KYCQA_':
                            df[colname_catgory_list_3] = df[colname_catgory_list_3].astype('category')
                            print(colname_catgory_list_3)
                        if table == 'FTBD_EVENT_':
                            df[colname_catgory_list_4] = df[colname_catgory_list_4].astype('category')
                            print(colname_catgory_list_4)
                        if table == 'FTBD_JCI_':
                            df[colname_catgory_list_5] = df[colname_catgory_list_5].astype('category')
                            print(colname_catgory_list_5)
                        if table == 'FTBD_DGT1_':
                            df[colname_catgory_list_6] = df[colname_catgory_list_6].astype('category')
                            print(colname_catgory_list_6)
                        #將剩餘 object 特徵轉成 float type
                        if 'yyyymm' in list(df.columns):
                            cols = df.select_dtypes('object').drop(['customer_id','yyyymm'],axis=1).columns
                        else:
                            cols = df.select_dtypes('object').drop(['customer_id'],axis=1).columns
                        if cols.tolist() != []:
                            print('trans to float: ')
                            df[cols.tolist()] = df[cols.tolist()].astype('float')
                        #轉完型態後寫入CDSW
                        pickle.dump(df, open(file_path, 'wb'), protocol = 4)
                        dump_table_log(table_name=table, feature_date=ym, status='Y', import_date=datetime.now().strftime('%Y%m%d %H:%M:%S'))
                        print('save', table, 'completed')
                        #drop the table from sql
                        time.sleep(1)
                        delete_table = table.lower()
                        if table != 'CUST_':
                            drop_SQL_raw_data(account, pwd, delete_table)
                    else:
                        dump_table_log(table_name=table, feature_date=ym, status='N', import_date=datetime.now().strftime('%Y%m%d %H:%M:%S'))
                        print(table, 'did not have data')
                    etime = time.time() - stime
                    print('saving table', table, 'requires', etime, 'sec')
                else:
                    print(table, 'no data on SQL')
#             time.sleep(10)


# In[15]:


def send_table_to_sql(targ, table_name, account, pwd, exist_action = 'append'):
    dtype_dist = {}
    for col,dtype in targ.dtypes.items():
        dtype_name = str(dtype)
        if dtype_name == "object" or dtype_name == "category" :
            max_length = targ[col].apply(lambda x:len(str(x).encode('utf-8'))).max()
            string_length = int(max_length*1.2)
            dtype_dist[col] = String(length = string_length)
        elif "float" in dtype_name:
            dtype_dist[col] = Float
        elif "int" in dtype_name:
            dtype_dist[col] = Integer
        elif "datetime" in dtype_name:
            dtype_dist[col] = DateTime
        elif "bool" in dtype_name:
            dtype_dist[col] = Boolean

    write_data_to_SQL(table_name=table_name, df = targ, account = account, pwd = pwd,col_types=dtype_dist, exist_action=exist_action)


# In[16]:


get_ipython().system('jupyter nbconvert --to script Sql_module.ipynb')


# In[ ]:




