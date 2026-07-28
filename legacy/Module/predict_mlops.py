#!/usr/bin/env python

# In[1]:


# loading parameter
import sys

sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
import calendar

import config

account, pwd = config.account, config.pwd
account_kris, pwd_kris = config.account_kris, config.pwd_kris


# In[2]:


#前版本predict_mlops_np_and_p_double_OPT_market_flag
def predict_mlops_202503(this_file_path,target,papulation_colname,papulation_except_value,mother_list,query_list,ym,
                           algorithm,write_db_Y_N,drop_key_word,frequency,edition_detail, market_flag_Y_N):
    import sys
    sys.path.append('/home/cdsw/Tony/Mlops_new/Module')

    import config
    from Model import (
        confirm_retrain_log,
        get_bin_with_dif_mother_conversion,
        get_bin_with_dif_mother_version230417,
        get_max_model_retrain_version,
        monthdelta,
        predict_df_v230517,
        save_db_id_list_mlops,
        save_db_ref_info_mlops,
        whether_done_next_population_predict,
    )
    from Pretreatment import get_feature_by_SOP_202503
    from Sql_module import get_SQL_raw_data2
    #該檔案路徑
    writing_path = this_file_path
    this_prod = this_file_path.split('/')[-1]

    #     from IPython.display import display
    from datetime import date, datetime

    import numpy as np
    import pandas as pd

    year=ym[0:4]
    month=ym[4:6]

    end_day = calendar.monthrange(int(year),int(month))[1]
    snap_date = year+'/'+month+'/'+str(end_day)
    # 確認母體跟商品下的最新retrain結果
    confirm_retrain_log(this_prod, mother_list, table_name='mlops_retrain_log_double')

    # 最新版
    new_edition_list = []
    for index, mother in enumerate(mother_list):
    #         new_edition = get_next_model_version(this_prod,target,mother,frequency,edition_detail, table_name = 'mlops_ref_info_double')
        new_edition = get_max_model_retrain_version(config.mlops_retrain_day[0:6], 'mlops_retrain_log_double', this_prod, mother)
        new_edition_list.append(new_edition)

    if market_flag_Y_N :
        # 抓取對應母體的bin
        bins_list, bins_label_list = get_bin_with_dif_mother_conversion(this_prod, mother_list, new_edition_list, table_name='mlops_retrain_log_double')
        # 轉換率對應的真實情形
        label_to_rank_list = get_bin_with_dif_mother_version230417(this_prod, mother_list, table_name='mlops_model_log_double')

    else:
        # 抓取對應母體的bin
        bins_list = get_bin_with_dif_mother_version230417(this_prod, mother_list, table_name='mlops_model_log_double')

    # 判斷是設定有無問題
    if len(mother_list) == len(bins_list) :
        print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
    else:
        raise Exception('mother_list、bins_list 長度不一致')

    query_function = query_list[0]
    print(f'query_function  : {query_function}')
    # 開始執行迴圈
    for index, mother in enumerate(mother_list):
        print(f'mother : {mother}')

        bins = bins_list[index]
        print(f'bins  : {bins}')

        # BIN的類型
        print(f'type(bins[1]) : {type(bins[1])}')

        if market_flag_Y_N :
            # bins_label_list
            bins_label = bins_label_list[index]
            print(f'bins_label: {bins_label}')

            label_to_rank = label_to_rank_list[index]
            print(f'真實轉換等級的label_to_rank: {label_to_rank}')
        # 版本
        new_edition = new_edition_list[index]
        print(f'new_edition : {new_edition}')

        if not whether_done_next_population_predict(this_file_path, this_prod, mother, target, algorithm, snap_date,
                                        db_table_rf='mlops_ref_info_double', db_table_IL='mlops_id_list_double',
                                        account=config.account_kris, pwd=config.pwd_kris):

            def get_df_combined():
                # 抓取母體
                date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#                 next_month = monthdelta(date, 1)
#                 print(next_month)
                query = query_function(this_prod , mother , ym, papulation_colname)

                if type(query) == str:
                    df_papu = get_SQL_raw_data2(query, account=config.account_kris, pwd=config.pwd_kris)
                else:
                    df_papu = query

                today = datetime.now().date()
                if today < monthdelta(date, 3):
                    df_papu['y'] = '999'
                    print(f"today: {today} < {ym} + 3 months, so don't have true y (all is 999)")
                else:
                    print(f"today: {today} >= {ym} + 3 months, so have true y")
                print('df_papu : ')
                print(df_papu.head())


                # 將母體與特徵左右拼接
                writing_feature_file_path = None
                concat_df_outcome = get_feature_by_SOP_202503([ym], mother, drop_key_word, df_papu, this_prod, Fill_zero=True)

                key = str(ym)+'_df'
                df_combined =concat_df_outcome[key]
                print('df_combined : ')
                print(len(df_combined))
#                 print(df_combined.head().to_string(index = False))

                return df_combined

            # 拉取母體
            df_combined = get_df_combined()


            # 進行預測，產出各等級人數、預測後等級與特徵、DB名單檔、DB參照資訊檔
            df_level_with_bin_num, df_pred_and_features, id_list_to_db, ref_info_to_db =             predict_df_v230517(df_combined, [ym], writing_path, this_prod, mother, algorithm,
                               bins,papulation_colname, papulation_except_value,
                               target, frequency, edition_detail,new_edition)
            if market_flag_Y_N:
                # 轉換為 轉換率等級
                bins_label.sort(reverse=True)
                rank_list = list(set(id_list_to_db['rank'][id_list_to_db['rank'] != papulation_colname]))
                rank_list.sort()

                for ri, rk in enumerate(rank_list):
                    binlabel = bins_label[ri]
                    print(f'將等級{rk}轉換成{binlabel}')
                    df_pred_and_features['model_level'][df_pred_and_features['model_level']==str(rk)] = binlabel
                    id_list_to_db['rank'][id_list_to_db['rank']==str(rk)] = binlabel

                # 用真實轉換率標籤再轉回等級
                for lbri, lbrk in enumerate(label_to_rank):
                    label_rank = label_to_rank[lbri]
                    rank_label = str(lbri+1)
                    print(f'將轉換率標籤{lbrk}轉換成等級{rank_label}')
                    df_pred_and_features['model_level'][df_pred_and_features['model_level']==lbrk] = rank_label
                    id_list_to_db['rank'][id_list_to_db['rank']==lbrk] = rank_label


            print('df_level_with_bin_num : ')
            print(df_level_with_bin_num.to_string(index = False))

            print('df_pred_and_features : ')
            print(df_pred_and_features.head().to_string(index = False))

            print('id_list_to_db  : ')
            print(id_list_to_db.head().to_string(index = False))
            print('id_list_to_db (except) : ')
            print(id_list_to_db[np.isnan(id_list_to_db['probability'])].head().to_string(index = False))

            print('ref_info_to_db : ')
            print(ref_info_to_db.to_string(index = False))

            # 寫入DB(名單檔)
            save_db_id_list_mlops(write_db_Y_N, id_list_to_db, table_name='s_ianleong.mlops_id_list_double')
            # 寫入DB(參照資訊檔)
            save_db_ref_info_mlops(write_db_Y_N, ref_info_to_db, table_name='s_ianleong.mlops_ref_info_double')

            print(f'{this_prod}_{mother}_{snap_date} 名單產出完畢，先跳出迴圈!')
            break



        else:
            print(f'{this_prod}_{mother}_{snap_date} 先前已執行過了，執行下個母體!')


    need_other_tag = False
    untrain_mother = '潛客'
    if '非潛客' in mother_list and untrain_mother not in mother_list:
        if not whether_done_next_population_predict(this_file_path, this_prod, untrain_mother, target, algorithm, snap_date,
                            db_table_rf='mlops_ref_info_double', db_table_IL='mlops_id_list_double',
                            account=config.account_kris, pwd=config.pwd_kris):
            need_other_tag = True
            # 宣告沒有建模的名單參數
            mother = untrain_mother
            if untrain_mother =='潛客': show_name = '潛力客群'
            tag_name = show_name + '註記'
            rank = show_name
            print(f'{this_prod}沒有對{untrain_mother}進行建模,用{show_name}註記')
            date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#             next_month = monthdelta(date, 1)
#             print(ym)
            query = query_function(this_prod , mother , ym, papulation_colname)
            df_papu_untrain_mother = get_SQL_raw_data2(query)
            today = datetime.now().date()
            if today < monthdelta(date, 3):
                df_papu_untrain_mother['y'] = '999'
                print(f"today: {today} < {ym} + 3 months, so don't have true y (all is 999)")
            else:
                print(f"today: {today} >= {ym} + 3 months, so have true y")


            query = f"""
                select * from ds_sec.cf_custid where yyyymm = '{ym}'
            """
            df = get_SQL_raw_data2(query, account=config.account_kris, pwd=config.pwd_kris)
            df['yyyymm'] = df['yyyymm'].astype('int')

            df_papu_untrain_mother['yyyymm'] = df_papu_untrain_mother['yyyymm'].astype('int')
            papu_and_y_yyyymm = df_papu_untrain_mother[df_papu_untrain_mother['yyyymm']==int(ym)]

            df_untrain_mother = pd.merge(df, papu_and_y_yyyymm, how='left', on=['customer_id','yyyymm'])
            mask_papulation = df_untrain_mother['y'].notna()
            df_untrain_mother = df_untrain_mother[mask_papulation]
            df_untrain_mother['y'] = df_untrain_mother['y'].astype('int')
        else:
            print(f'{this_prod}_{mother}_{snap_date} 先前已執行過了，執行下個母體!')
    if need_other_tag:
        # 預測時間
        pred_date = datetime.today().strftime('%Y/%m/%d %H:%M')

        print('df_papu_untrain_mother : ')
        print(df_papu_untrain_mother.head().to_string(index = False))

        id_list_to_db = pd.DataFrame(data={
                                            'product': this_prod,
                                            'target': target ,
                                            'population': mother,
                                            'frequency': frequency,
                                            'snap_date': snap_date,
                                            'party_id': df_untrain_mother['customer_id'],
                                            'tag_name': tag_name,
                                            'probability': np.nan,
                                            'rank': rank,
                                            'pred_date': pred_date})
        print('id_list_to_db_untrain : ')
        print(id_list_to_db.head().to_string(index = False))


        ref_info_to_db = pd.DataFrame(data={
                                            'product': this_prod,
                                            'target': target ,
                                            'population': mother,
                                            'frequency': frequency,
                                            'edition': np.nan,
                                            'snap_date': snap_date ,
                                            'model_valid_falg': np.nan,
                                            'tag_name':  tag_name,
                                            'exception': '' ,
                                            'edition_detail': '沒有進行建模純註記',
                                            'pred_date': pred_date


                                        },index=[0])
        print('ref_info_to_db_untrain : ')
        print(ref_info_to_db.to_string(index = False))

        save_db_id_list_mlops(write_db_Y_N, id_list_to_db, table_name='s_ianleong.mlops_id_list_double')
        save_db_ref_info_mlops(write_db_Y_N, ref_info_to_db, table_name='s_ianleong.mlops_ref_info_double')


# In[3]:


# def predict_mlops_np_and_p_double_OPT_market_flag(this_file_path,target,papulation_colname,papulation_except_value,mother_list,query_list,ym,
#                            algorithm,write_db_Y_N,drop_key_word,frequency,edition_detail, market_flag_Y_N):
#     import sys
#     sys.path.append(this_file_path)
#     try:
#         from Papulation import make_query_non_potential, make_query_potential
#     except:
#         print(f'沒有潛客/非潛客 query function')
#         pass
#     try:
#         from Papulation import make_query
#     except:
#         print(f'沒有不分潛客 query function')
#         pass
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     from Model import monthdelta, confirm_retrain_log, get_bin_with_dif_mother,get_bin_with_dif_mother_version230417, \
#     predict_df_v230517, predict_log, save_db_mlops,save_db_ref_info_mlops,save_db_id_list_mlops,get_next_model_version,whether_done_next_population_predict, \
#     get_max_model_retrain_version, get_bin_with_dif_mother_conversion
#     from Pretreatment import get_feature_by_SOP,get_feature_by_SOP_jihsun
#     from Sql_module import get_SQL_raw_data
#     import config
#     import os
#     #該檔案路徑
#     writing_path = this_file_path
#     project_name = this_file_path.split('/')[-1]

#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from IPython.display import display
#     import gc

#     year=ym[0:4]
#     month=ym[4:6]

#     end_day = calendar.monthrange(int(year),int(month))[1]
#     snap_date = year+'/'+month+'/'+str(end_day)
#     # 確認母體跟商品下的最新retrain結果
#     confirm_retrain_log(project_name, mother_list, table_name='mlops_retrain_log_double')

#     # 最新版
#     new_edition_list = []
#     for index, mother in enumerate(mother_list):
#     #         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'mlops_ref_info_double')
#         new_edition = get_max_model_retrain_version(ym, 'mlops_retrain_log_double', project_name, mother)
#         new_edition_list.append(new_edition)

#     if market_flag_Y_N :
#         # 抓取對應母體的bin
#         bins_list, bins_label_list = get_bin_with_dif_mother_conversion(project_name, mother_list, new_edition_list, table_name='mlops_retrain_log_double')
#         # 轉換率對應的真實情形
#         label_to_rank_list = get_bin_with_dif_mother_version230417(project_name, mother_list, table_name='mlops_model_log_double')

#     else:
#         # 抓取對應母體的bin
#         bins_list = get_bin_with_dif_mother_version230417(project_name, mother_list, table_name='mlops_model_log_double')

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')

#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):
#         print(f'mother : {mother}')

#         query_function = query_list[index]
#         print(f'query_function  : {query_function}')

#         bins = bins_list[index]
#         print(f'bins  : {bins}')

#         # BIN的類型
#         print(f'type(bins[1]) : {type(bins[1])}')

#         if market_flag_Y_N :
#             # bins_label_list
#             bins_label = bins_label_list[index]
#             print(f'bins_label: {bins_label}')

#             label_to_rank = label_to_rank_list[index]
#             print(f'真實轉換等級的label_to_rank: {label_to_rank}')
#         # 版本
#         new_edition = new_edition_list[index]
#         print(f'new_edition : {new_edition}')

#         if not whether_done_next_population_predict(this_file_path, project_name, mother, target, algorithm, snap_date,
#                                         db_table_rf='mlops_ref_info_double', db_table_IL='mlops_id_list_double',
#                                         account=config.account_yt, pwd=config.pwd_yt):

#             def get_df_combined(customer_source):
#                 # 看要抓雙證哪個
#                 if customer_source == 'Fubon':
#                     print('---------------------------------------------')
#                     print('-------------富邦客戶-------------------------')
#                     print('---------------------------------------------')
#                     get_feature_function = get_feature_by_SOP

#                 elif customer_source == 'Jihsun':
#                     print('---------------------------------------------')
#                     print('-------------日盛客戶-------------------------')
#                     print('---------------------------------------------')
#                     get_feature_function = get_feature_by_SOP_jihsun
#                 # 抓取母體
#                 date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#                 next_month = monthdelta(date, 1)
#                 print(next_month)
#                 query = query_function(next_month)

#                 if type(query) == str:
#                     df_papu = get_SQL_raw_data(query, account=config.account_yt, pwd=config.pwd_yt)
#                 else:
#                     df_papu = query

#                 today = datetime.now().date()
#                 if today < monthdelta(date, 4):
#                     df_papu['y'] = '999'
#                     print(f"today: {today} < {ym} + 4 months, so don't have true y (all is 999)")
#                 else:
#                     print(f"today: {today} >= {ym} + 4 months, so have true y")
#                 print('df_papu : ')
#                 display(df_papu.head())

#                 # 將母體與特徵左右拼接
#                 writing_feature_file_path = None
#                 concat_df_outcome = get_feature_function([ym], mother,writing_feature_file_path, drop_key_word, df_papu,
#                                                          just_for_check=False, Fill_zero=True)
#                 key = str(ym)+'_df'
#                 df_combined =concat_df_outcome[key]
#                 print('df_combined : ')
#                 display(df_combined.head())
#                 return df_combined

#             # 拉取富邦日盛母體
#             df_combined_fubon = get_df_combined('Fubon')
#             df_combined_jihsun = get_df_combined('Jihsun')

#             ###############################################################################################################
#             # 開始合併富證日盛客戶
#             ##############################################################################################################
#             print('!!!!開始合併富邦日盛資料')
#             # 資料: [build_set]
#             st_concat = time.time()
#             df_combined = pd.concat([df_combined_fubon,df_combined_jihsun],axis = 0)
#             print(f'富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#             # Concat後categorical會變objective,要轉回來
#             st_concat_astype = time.time()
#             obj_cols = df_combined.select_dtypes('object').drop(['customer_id'],axis=1).columns
#             df_combined[obj_cols.tolist()] = df_combined[obj_cols.tolist()].astype('category')
#             print(f'富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#             #移除雙證重複
#             remove_dupli_id_time = time.time()
#             df_combined['status'] = df_combined.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#             print(f'原始雙證人數: {len(df_combined)}')
#             df_combined = df_combined[df_combined['status']==1]
#             print(f'移除重複後雙證人數: {len(df_combined)}')
#             print(f'富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')


#             # 進行預測，產出各等級人數、預測後等級與特徵、DB名單檔、DB參照資訊檔
#             df_level_with_bin_num, df_pred_and_features, id_list_to_db, ref_info_to_db = \
#             predict_df_v230517(df_combined, [ym], writing_path, project_name, mother, algorithm,
#                                bins,papulation_colname, papulation_except_value,
#                                target, frequency, edition_detail,new_edition)
#             if market_flag_Y_N:
#                 # 轉換為 轉換率等級
#                 bins_label.sort(reverse=True)
#                 rank_list = list(set(id_list_to_db['rank'][id_list_to_db['rank'] != papulation_colname]))
#                 rank_list.sort()

#                 for ri, rk in enumerate(rank_list):
#                     binlabel = bins_label[ri]
#                     print(f'將等級{rk}轉換成{binlabel}')
#                     df_pred_and_features['model_level'][df_pred_and_features['model_level']==str(rk)] = binlabel
#                     id_list_to_db['rank'][id_list_to_db['rank']==str(rk)] = binlabel

#                 # 用真實轉換率標籤再轉回等級
#                 for lbri, lbrk in enumerate(label_to_rank):
#                     label_rank = label_to_rank[lbri]
#                     rank_label = str(lbri+1)
#                     print(f'將轉換率標籤{lbrk}轉換成等級{rank_label}')
#                     df_pred_and_features['model_level'][df_pred_and_features['model_level']==lbrk] = rank_label
#                     id_list_to_db['rank'][id_list_to_db['rank']==lbrk] = rank_label


#             print('df_level_with_bin_num : ')
#             display(df_level_with_bin_num)

#             print('df_pred_and_features : ')
#             display(df_pred_and_features.head())

#             print('id_list_to_db  : ')
#             display(id_list_to_db.head())
#             print('id_list_to_db (except) : ')
#             display(id_list_to_db[np.isnan(id_list_to_db['probability'])].head())

#             print('ref_info_to_db : ')
#             display(ref_info_to_db)

#             # 寫入DB(名單檔)
#             save_db_id_list_mlops(write_db_Y_N, id_list_to_db, table_name='mlops_id_list_double')
#             # 寫入DB(參照資訊檔)
#             save_db_ref_info_mlops(write_db_Y_N, ref_info_to_db, table_name='mlops_ref_info_double')

#             print(f'{project_name}_{mother}_{snap_date} 名單產出完畢，先跳出迴圈!')
#             break



#         else:
#             print(f'{project_name}_{mother}_{snap_date} 先前已執行過了，執行下個母體!')


#     need_other_tag = False
#     untrain_mother = '潛客'
#     if '非潛客' in mother_list and untrain_mother not in mother_list:
#         if not whether_done_next_population_predict(this_file_path, project_name, untrain_mother, target, algorithm, snap_date,
#                             db_table_rf='mlops_ref_info_double', db_table_IL='mlops_id_list_double',
#                             account=config.account_yt, pwd=config.pwd_yt):
#             need_other_tag = True
#             # 宣告沒有建模的名單參數
#             mother = untrain_mother
#             if untrain_mother =='潛客': show_name = '潛力客群'
#             tag_name = show_name + '註記'
#             rank = show_name
#             print(f'{project_name}沒有對{untrain_mother}進行建模,用{show_name}註記')
#             date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#             next_month = monthdelta(date, 1)
#             print(next_month)
#             query = make_query_potential(next_month)
#             df_papu_untrain_mother = get_SQL_raw_data(query)
#             today = datetime.now().date()
#             if today < monthdelta(date, 4):
#                 df_papu_untrain_mother['y'] = '999'
#                 print(f"today: {today} < {ym} + 4 months, so don't have true y (all is 999)")
#             else:
#                 print(f"today: {today} >= {ym} + 4 months, so have true y")

#             # FUBON
#             feature_ym_path_fubon = config.feature_file_path_fubon + '/' + '{}'.format(ym)
#             df_fubon = pickle.load(open(feature_ym_path_fubon + '/' + 'CUST_{}.pickle'.format(ym), 'rb'))
#             # JIHSUN
#             feature_ym_path_jihsun = config.feature_file_path_jihsun + '/' + '{}'.format(ym)
#             df_jihsun = pickle.load(open(feature_ym_path_jihsun + '/' + 'CUST_{}.pickle'.format(ym), 'rb'))
#             # 合併排除重複
#             df = pd.concat([df_fubon,df_jihsun])
#             df.drop_duplicates(inplace=True)
#             df['yyyymm'] = df['yyyymm'].astype('int')

#             df_papu_untrain_mother['yyyymm'] = df_papu_untrain_mother['yyyymm'].astype('int')
#             papu_and_y_yyyymm = df_papu_untrain_mother[df_papu_untrain_mother['yyyymm']==int(ym)]

#             df_untrain_mother = pd.merge(df, papu_and_y_yyyymm, how='left', on=['customer_id','yyyymm'])
#             mask_papulation = df_untrain_mother['y'].notna()
#             df_untrain_mother = df_untrain_mother[mask_papulation]
#             df_untrain_mother['y'] = df_untrain_mother['y'].astype('int')
#         else:
#             print(f'{project_name}_{mother}_{snap_date} 先前已執行過了，執行下個母體!')
#     if need_other_tag:
#         # 預測時間
#         pred_date = datetime.today().strftime('%Y/%m/%d %H:%M')

#         print('df_papu_untrain_mother : ')
#         display(df_papu_untrain_mother.head())

#         id_list_to_db = pd.DataFrame(data={
#                                             'product': project_name,
#                                             'target': target ,
#                                             'population': mother,
#                                             'frequency': frequency,
#                                             'snap_date': snap_date,
#                                             'party_id': df_untrain_mother['customer_id'],
#                                             'tag_name': tag_name,
#                                             'probability': np.nan,
#                                             'rank': rank,
#                                             'pred_date': pred_date})
#         print('id_list_to_db_untrain : ')
#         display(id_list_to_db.head())


#         ref_info_to_db = pd.DataFrame(data={
#                                             'product': project_name,
#                                             'target': target ,
#                                             'population': mother,
#                                             'frequency': frequency,
#                                             'edition': np.nan,
#                                             'snap_date': snap_date ,
#                                             'model_valid_falg': np.nan,
#                                             'tag_name':  tag_name,
#                                             'exception': '' ,
#                                             'edition_detail': '沒有進行建模純註記',
#                                             'pred_date': pred_date


#                                         },index=[0])
#         print('ref_info_to_db_untrain : ')
#         display(ref_info_to_db)

#         save_db_id_list_mlops(write_db_Y_N, id_list_to_db, table_name='mlops_id_list_double')
#         save_db_ref_info_mlops(write_db_Y_N, ref_info_to_db, table_name='mlops_ref_info_double')


# In[4]:


# def predict_mlops_np_and_p_double_OPT(this_file_path,target,papulation_colname,papulation_except_value,mother_list,query_list,ym,
#                            algorithm,write_db_Y_N,drop_key_word,frequency,edition_detail):
#     import sys
#     sys.path.append(this_file_path)
#     try:
#         from Papulation import make_query_non_potential, make_query_potential
#     except:
#         print(f'沒有潛客/非潛客 query function')
#         pass
#     try:
#         from Papulation import make_query
#     except:
#         print(f'沒有不分潛客 query function')
#         pass
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     from Model import monthdelta, confirm_retrain_log, get_bin_with_dif_mother,get_bin_with_dif_mother_version230417, \
#     predict_df_v230517, predict_log, save_db_mlops,save_db_ref_info_mlops,save_db_id_list_mlops,get_next_model_version,whether_done_next_population_predict, \
#     get_max_model_retrain_version
#     from Pretreatment import get_feature_by_SOP,get_feature_by_SOP_jihsun
#     from Sql_module import get_SQL_raw_data
#     import config
#     import os
#     #該檔案路徑
#     writing_path = this_file_path
#     project_name = this_file_path.split('/')[-1]

#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from IPython.display import display
#     import gc

#     year=ym[0:4]
#     month=ym[4:6]

#     end_day = calendar.monthrange(int(year),int(month))[1]
#     snap_date = year+'/'+month+'/'+str(end_day)
#     # 確認母體跟商品下的最新retrain結果
#     confirm_retrain_log(project_name, mother_list, table_name='mlops_retrain_log_double')

#     # 抓取對應母體的bin
#     bins_list = get_bin_with_dif_mother_version230417(project_name, mother_list, table_name='mlops_model_log_double')

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')

#     # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#     new_edition_list = []
#     for index, mother in enumerate(mother_list):
# #         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'mlops_ref_info_double')
#         new_edition = get_max_model_retrain_version(ym, 'mlops_retrain_log_double', project_name, mother)
#         new_edition_list.append(new_edition)


#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):
#         print(f'mother : {mother}')

#         query_function = query_list[index]
#         print(f'query_function  : {query_function}')

#         bins = bins_list[index]
#         print(f'bins  : {bins}')

#         # BIN的類型
#         print(f'type(bins[1]) : {type(bins[1])}')

#         # 版本
#         new_edition = new_edition_list[index]
#         print(f'new_edition : {new_edition}')

#         if not whether_done_next_population_predict(this_file_path, project_name, mother, target, algorithm, snap_date,
#                                         db_table_rf='mlops_ref_info_double', db_table_IL='mlops_id_list_double',
#                                         account=config.account_yt, pwd=config.pwd_yt):

#             def get_df_combined(customer_source):
#                 # 看要抓雙證哪個
#                 if customer_source == 'Fubon':
#                     print('---------------------------------------------')
#                     print('-------------富邦客戶-------------------------')
#                     print('---------------------------------------------')
#                     get_feature_function = get_feature_by_SOP

#                 elif customer_source == 'Jihsun':
#                     print('---------------------------------------------')
#                     print('-------------日盛客戶-------------------------')
#                     print('---------------------------------------------')
#                     get_feature_function = get_feature_by_SOP_jihsun
#                 # 抓取母體
#                 date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#                 next_month = monthdelta(date, 1)
#                 print(next_month)
#                 query = query_function(next_month)

#                 if type(query) == str:
#                     df_papu = get_SQL_raw_data(query, account=account_yt, pwd=pwd_yt)
#                 else:
#                     df_papu = query

#                 today = datetime.now().date()
#                 if today < monthdelta(date, 4):
#                     df_papu['y'] = '999'
#                     print(f"today: {today} < {ym} + 4 months, so don't have true y (all is 999)")
#                 else:
#                     print(f"today: {today} >= {ym} + 4 months, so have true y")
#                 print('df_papu : ')
#                 display(df_papu.head())

#                 # 將母體與特徵左右拼接
#                 writing_feature_file_path = None
#                 concat_df_outcome = get_feature_function([ym], mother,writing_feature_file_path, drop_key_word, df_papu,
#                                                          just_for_check=False, Fill_zero=True)
#                 key = str(ym)+'_df'
#                 df_combined =concat_df_outcome[key]
#                 print('df_combined : ')
#                 display(df_combined.head())
#                 return df_combined

#             # 拉取富邦日盛母體
#             df_combined_fubon = get_df_combined('Fubon')
#             df_combined_jihsun = get_df_combined('Jihsun')

#             ###############################################################################################################
#             # 開始合併富證日盛客戶
#             ##############################################################################################################
#             print('!!!!開始合併富邦日盛資料')
#             # 資料: [build_set]
#             st_concat = time.time()
#             df_combined = pd.concat([df_combined_fubon,df_combined_jihsun],axis = 0)
#             print(f'富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#             # Concat後categorical會變objective,要轉回來
#             st_concat_astype = time.time()
#             obj_cols = df_combined.select_dtypes('object').drop(['customer_id'],axis=1).columns
#             df_combined[obj_cols.tolist()] = df_combined[obj_cols.tolist()].astype('category')
#             print(f'富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#             #移除雙證重複
#             remove_dupli_id_time = time.time()
#             df_combined['status'] = df_combined.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#             print(f'原始雙證人數: {len(df_combined)}')
#             df_combined = df_combined[df_combined['status']==1]
#             print(f'移除重複後雙證人數: {len(df_combined)}')
#             print(f'富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')


#             # 進行預測，產出各等級人數、預測後等級與特徵、DB名單檔、DB參照資訊檔
#             df_level_with_bin_num, df_pred_and_features, id_list_to_db, ref_info_to_db = \
#             predict_df_v230517(df_combined, [ym], writing_path, project_name, mother, algorithm,
#                                bins,papulation_colname, papulation_except_value,
#                                target, frequency, edition_detail,new_edition)

#             print('df_level_with_bin_num : ')
#             display(df_level_with_bin_num)

#             print('df_pred_and_features : ')
#             display(df_pred_and_features.head())

#             print('id_list_to_db  : ')
#             display(id_list_to_db.head())
#             print('id_list_to_db (except) : ')
#             display(id_list_to_db[np.isnan(id_list_to_db['probability'])].head())

#             print('ref_info_to_db : ')
#             display(ref_info_to_db)

#             # 寫入DB(名單檔)
#             save_db_id_list_mlops(write_db_Y_N, id_list_to_db, table_name='mlops_id_list_double')
#             # 寫入DB(參照資訊檔)
#             save_db_ref_info_mlops(write_db_Y_N, ref_info_to_db, table_name='mlops_ref_info_double')

#             print(f'{project_name}_{mother}_{snap_date} 名單產出完畢，先跳出迴圈!')
#             break
#         else:
#             print(f'{project_name}_{mother}_{snap_date} 先前已執行過了，執行下個母體!')


#     need_other_tag = False
#     untrain_mother = '潛客'
#     if '非潛客' in mother_list and untrain_mother not in mother_list:
#         if not whether_done_next_population_predict(this_file_path, project_name, untrain_mother, target, algorithm, snap_date,
#                             db_table_rf='mlops_ref_info_double', db_table_IL='mlops_id_list_double',
#                             account=config.account_yt, pwd=config.pwd_yt):
#             need_other_tag = True
#             # 宣告沒有建模的名單參數
#             mother = untrain_mother
#             if untrain_mother =='潛客': show_name = '潛力客群'
#             tag_name = show_name + '註記'
#             rank = show_name
#             print(f'{project_name}沒有對{untrain_mother}進行建模,用{show_name}註記')
#             date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#             next_month = monthdelta(date, 1)
#             print(next_month)
#             query = make_query_potential(next_month)
#             df_papu_untrain_mother = get_SQL_raw_data(query)
#             today = datetime.now().date()
#             if today < monthdelta(date, 4):
#                 df_papu_untrain_mother['y'] = '999'
#                 print(f"today: {today} < {ym} + 4 months, so don't have true y (all is 999)")
#             else:
#                 print(f"today: {today} >= {ym} + 4 months, so have true y")

#             # FUBON
#             feature_ym_path_fubon = config.feature_file_path_fubon + '/' + '{}'.format(ym)
#             df_fubon = pickle.load(open(feature_ym_path_fubon + '/' + 'CUST_{}.pickle'.format(ym), 'rb'))
#             # JIHSUN
#             feature_ym_path_jihsun = config.feature_file_path_jihsun + '/' + '{}'.format(ym)
#             df_jihsun = pickle.load(open(feature_ym_path_jihsun + '/' + 'CUST_{}.pickle'.format(ym), 'rb'))
#             # 合併排除重複
#             df = pd.concat([df_fubon,df_jihsun])
#             df.drop_duplicates(inplace=True)
#             df['yyyymm'] = df['yyyymm'].astype('int')

#             df_papu_untrain_mother['yyyymm'] = df_papu_untrain_mother['yyyymm'].astype('int')
#             papu_and_y_yyyymm = df_papu_untrain_mother[df_papu_untrain_mother['yyyymm']==int(ym)]

#             df_untrain_mother = pd.merge(df, papu_and_y_yyyymm, how='left', on=['customer_id','yyyymm'])
#             mask_papulation = df_untrain_mother['y'].notna()
#             df_untrain_mother = df_untrain_mother[mask_papulation]
#             df_untrain_mother['y'] = df_untrain_mother['y'].astype('int')
#         else:
#             print(f'{project_name}_{mother}_{snap_date} 先前已執行過了，執行下個母體!')
#     if need_other_tag:
#         # 預測時間
#         pred_date = datetime.today().strftime('%Y/%m/%d %H:%M')

#         print('df_papu_untrain_mother : ')
#         display(df_papu_untrain_mother.head())

#         id_list_to_db = pd.DataFrame(data={
#                                             'product': project_name,
#                                             'target': target ,
#                                             'population': mother,
#                                             'frequency': frequency,
#                                             'snap_date': snap_date,
#                                             'party_id': df_untrain_mother['customer_id'],
#                                             'tag_name': tag_name,
#                                             'probability': np.nan,
#                                             'rank': rank,
#                                             'pred_date': pred_date})
#         print('id_list_to_db_untrain : ')
#         display(id_list_to_db.head())


#         ref_info_to_db = pd.DataFrame(data={
#                                             'product': project_name,
#                                             'target': target ,
#                                             'population': mother,
#                                             'frequency': frequency,
#                                             'edition': np.nan,
#                                             'snap_date': snap_date ,
#                                             'model_valid_falg': np.nan,
#                                             'tag_name':  tag_name,
#                                             'exception': '' ,
#                                             'edition_detail': '沒有進行建模純註記',
#                                             'pred_date': pred_date


#                                         },index=[0])
#         print('ref_info_to_db_untrain : ')
#         display(ref_info_to_db)

#         save_db_id_list_mlops(write_db_Y_N, id_list_to_db, table_name='mlops_id_list_double')
#         save_db_ref_info_mlops(write_db_Y_N, ref_info_to_db, table_name='mlops_ref_info_double')


# In[5]:


# def predict_mlops_np_and_p_double(this_file_path,target,papulation_colname,papulation_except_value,mother_list,query_list,ym,
#                            algorithm,write_db_Y_N,drop_key_word,frequency,edition_detail):
#     import sys
#     sys.path.append(this_file_path)
#     try:
#         from Papulation import make_query_non_potential, make_query_potential
#     except:
#         print(f'沒有潛客/非潛客 query function')
#         pass
#     try:
#         from Papulation import make_query
#     except:
#         print(f'沒有不分潛客 query function')
#         pass
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     from Model import monthdelta, confirm_retrain_log, get_bin_with_dif_mother,get_bin_with_dif_mother_version230417, \
#     predict_df_v230517, predict_log, save_db_mlops,save_db_ref_info_mlops,save_db_id_list_mlops,get_next_model_version
#     from Pretreatment import get_feature_by_SOP,get_feature_by_SOP_jihsun
#     from Sql_module import get_SQL_raw_data
#     import config
#     import os
#     #該檔案路徑
#     writing_path = this_file_path
#     project_name = this_file_path.split('/')[-1]

#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from IPython.display import display
#     import gc

#     # 確認母體跟商品下的最新retrain結果
#     confirm_retrain_log(project_name, mother_list, table_name='mlops_retrain_log_double')

#     # 抓取對應母體的bin
#     bins_list = get_bin_with_dif_mother_version230417(project_name, mother_list, table_name='mlops_model_log_double')

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')

#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):
#         print(f'mother : {mother}')

#         query_function = query_list[index]
#         print(f'query_function  : {query_function}')

#         bins = bins_list[index]
#         print(f'bins  : {bins}')

#         def get_df_combined(customer_source):
#             # 看要抓雙證哪個
#             if customer_source == 'Fubon':
#                 print('---------------------------------------------')
#                 print('-------------富邦客戶-------------------------')
#                 print('---------------------------------------------')
#                 get_feature_function = get_feature_by_SOP

#             elif customer_source == 'Jihsun':
#                 print('---------------------------------------------')
#                 print('-------------日盛客戶-------------------------')
#                 print('---------------------------------------------')
#                 get_feature_function = get_feature_by_SOP_jihsun
#             # 抓取母體
#             date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#             next_month = monthdelta(date, 1)
#             print(next_month)
#             query = query_function(next_month)

#             if type(query) == str:
#                 df_papu = get_SQL_raw_data(query, account=account_yt, pwd=pwd_yt)
#             else:
#                 df_papu = query

#             today = datetime.now().date()
#             if today < monthdelta(date, 4):
#                 df_papu['y'] = '999'
#                 print(f"today: {today} < {ym} + 4 months, so don't have true y (all is 999)")
#             else:
#                 print(f"today: {today} >= {ym} + 4 months, so have true y")
#             print('df_papu : ')
#             display(df_papu.head())

#             # 將母體與特徵左右拼接
#             writing_feature_file_path = None
#             concat_df_outcome = get_feature_function([ym], mother,writing_feature_file_path, drop_key_word, df_papu)
#             key = str(ym)+'_df'
#             df_combined =concat_df_outcome[key]
#             print('df_combined : ')
#             display(df_combined.head())
#             return df_combined

#         # 拉取富邦日盛母體
#         df_combined_fubon = get_df_combined('Fubon')
#         df_combined_jihsun = get_df_combined('Jihsun')

#         ###############################################################################################################
#         # 開始合併富證日盛客戶
#         ##############################################################################################################
#         print('!!!!開始合併富邦日盛資料')
#         # 資料: [build_set]
#         st_concat = time.time()
#         df_combined = pd.concat([df_combined_fubon,df_combined_jihsun],axis = 0)
#         print(f'富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#         # Concat後categorical會變objective,要轉回來
#         st_concat_astype = time.time()
#         obj_cols = df_combined.select_dtypes('object').drop(['customer_id'],axis=1).columns
#         df_combined[obj_cols.tolist()] = df_combined[obj_cols.tolist()].astype('category')
#         print(f'富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#         #移除雙證重複
#         remove_dupli_id_time = time.time()
#         df_combined['status'] = df_combined.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#         print(f'原始雙證人數: {len(df_combined)}')
#         df_combined = df_combined[df_combined['status']==1]
#         print(f'移除重複後雙證人數: {len(df_combined)}')
#         print(f'富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')



#         # BIN的類型
#         print(f'type(bins[1]) : {type(bins[1])}')

#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'mlops_ref_info_double')


#         # 進行預測，產出各等級人數、預測後等級與特徵、DB名單檔、DB參照資訊檔
#         df_level_with_bin_num, df_pred_and_features, id_list_to_db, ref_info_to_db = \
#         predict_df_v230517(df_combined, [ym], writing_path, project_name, mother, algorithm,
#                            bins,papulation_colname, papulation_except_value,
#                            target, frequency, edition_detail,new_edition)

#         print('df_level_with_bin_num : ')
#         display(df_level_with_bin_num)

#         print('df_pred_and_features : ')
#         display(df_pred_and_features.head())

#         print('id_list_to_db  : ')
#         display(id_list_to_db.head())
#         print('id_list_to_db (except) : ')
#         display(id_list_to_db[np.isnan(id_list_to_db['probability'])].head())

#         print('ref_info_to_db : ')
#         display(ref_info_to_db)

#         # 寫入DB(名單檔)
#         save_db_id_list_mlops(write_db_Y_N, id_list_to_db, table_name='mlops_id_list_double')
#         # 寫入DB(參照資訊檔)
#         save_db_ref_info_mlops(write_db_Y_N, ref_info_to_db, table_name='mlops_ref_info_double')


#     need_other_tag = False
#     untrain_mother = '潛客'
#     if '非潛客' in mother_list and untrain_mother not in mother_list:
#         need_other_tag = True
#         # 宣告沒有建模的名單參數
#         mother = untrain_mother
#         if untrain_mother =='潛客': show_name = '潛力客群'
#         tag_name = show_name + '註記'
#         rank = show_name
#         print(f'{project_name}沒有對{untrain_mother}進行建模,用{show_name}註記')
#         date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#         next_month = monthdelta(date, 1)
#         print(next_month)
#         query = make_query_potential(next_month)
#         df_papu_untrain_mother = get_SQL_raw_data(query)
#         today = datetime.now().date()
#         if today < monthdelta(date, 4):
#             df_papu_untrain_mother['y'] = '999'
#             print(f"today: {today} < {ym} + 4 months, so don't have true y (all is 999)")
#         else:
#             print(f"today: {today} >= {ym} + 4 months, so have true y")

#         # FUBON
#         feature_ym_path_fubon = config.feature_file_path_fubon + '/' + '{}'.format(ym)
#         df_fubon = pickle.load(open(feature_ym_path_fubon + '/' + 'CUST_{}.pickle'.format(ym), 'rb'))
#         # JIHSUN
#         feature_ym_path_jihsun = config.feature_file_path_jihsun + '/' + '{}'.format(ym)
#         df_jihsun = pickle.load(open(feature_ym_path_jihsun + '/' + 'CUST_{}.pickle'.format(ym), 'rb'))
#         # 合併排除重複
#         df = pd.concat([df_fubon,df_jihsun])
#         df.drop_duplicates(inplace=True)
#         df['yyyymm'] = df['yyyymm'].astype('int')

#         df_papu_untrain_mother['yyyymm'] = df_papu_untrain_mother['yyyymm'].astype('int')
#         papu_and_y_yyyymm = df_papu_untrain_mother[df_papu_untrain_mother['yyyymm']==int(ym)]

#         df_untrain_mother = pd.merge(df, papu_and_y_yyyymm, how='left', on=['customer_id','yyyymm'])
#         mask_papulation = df_untrain_mother['y'].notna()
#         df_untrain_mother = df_untrain_mother[mask_papulation]
#         df_untrain_mother['y'] = df_untrain_mother['y'].astype('int')

#     if need_other_tag:
#         print('df_papu_untrain_mother : ')
#         display(df_papu_untrain_mother.head())

#         id_list_to_db = pd.DataFrame(data={
#                                             'product': project_name,
#                                             'target': target ,
#                                             'population': mother,
#                                             'frequency': frequency,
#                                             'snap_date':id_list_to_db['snap_date'].unique()[0] ,
#                                             'party_id': df_untrain_mother['customer_id'],
#                                             'tag_name': tag_name,
#                                             'probability': np.nan,
#                                             'rank': rank,
#                                             'pred_date': id_list_to_db['pred_date'].unique()[0]})
#         print('id_list_to_db_untrain : ')
#         display(id_list_to_db.head())


#         ref_info_to_db = pd.DataFrame(data={
#                                             'product': project_name,
#                                             'target': target ,
#                                             'population': mother,
#                                             'frequency': frequency,
#                                             'edition': np.nan,
#                                             'snap_date': ref_info_to_db['snap_date'].unique()[0] ,
#                                             'model_valid_falg': np.nan,
#                                             'tag_name':  tag_name,
#                                             'exception': '' ,
#                                             'edition_detail': '沒有進行建模純註記',
#                                             'pred_date': ref_info_to_db['pred_date'].unique()[0]


#                                         },index=[0])
#         print('ref_info_to_db_untrain : ')
#         display(ref_info_to_db)

#         save_db_id_list_mlops(write_db_Y_N, id_list_to_db, table_name='mlops_id_list_double')
#         save_db_ref_info_mlops(write_db_Y_N, ref_info_to_db, table_name='mlops_ref_info_double')


# In[6]:


# def predict_mlops_np_and_p(this_file_path,target,papulation_colname,papulation_except_value,mother_list,query_list,ym,
#                            algorithm,write_db_Y_N,drop_key_word,frequency,edition_detail):
#     import sys
#     sys.path.append(this_file_path)
#     try:
#         from Papulation import make_query_non_potential, make_query_potential
#     except:
#         print(f'沒有潛客/非潛客 query function')
#         pass
#     try:
#         from Papulation import make_query
#     except:
#         print(f'沒有不分潛客 query function')
#         pass
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     from Model import monthdelta, confirm_retrain_log, get_bin_with_dif_mother,get_bin_with_dif_mother_version230417, \
#     predict_df_v230517, predict_log, save_db_mlops,save_db_ref_info_mlops,save_db_id_list_mlops,get_next_model_version
#     from Pretreatment import get_feature_by_SOP
#     from Sql_module import get_SQL_raw_data
#     import config
#     import os
#     #該檔案路徑
#     writing_path = this_file_path
#     project_name = this_file_path.split('/')[-1]

#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from IPython.display import display
#     import gc

#     # 確認母體跟商品下的最新retrain結果
#     confirm_retrain_log(project_name, mother_list, table_name='mlops_retrain_log')

#     # 抓取對應母體的bin
#     bins_list = get_bin_with_dif_mother_version230417(project_name, mother_list, table_name='mlops_model_log')

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')

#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):
#         print(f'mother : {mother}')

#         query_function = query_list[index]
#         print(f'query_function  : {query_function}')

#         bins = bins_list[index]
#         print(f'bins  : {bins}')

#         # 抓取母體
#         date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#         next_month = monthdelta(date, 1)
#         print(next_month)
#         query = query_function(next_month)

#         if type(query) == str:
#             df_papu = get_SQL_raw_data(query, account=account_yt , pwd=pwd_yt)
#         else:
#             df_papu = query

#         today = datetime.now().date()
#         if today < monthdelta(date, 4):
#             df_papu['y'] = '999'
#             print(f"today: {today} < {ym} + 4 months, so don't have true y (all is 999)")
#         else:
#             print(f"today: {today} >= {ym} + 4 months, so have true y")
#         print('df_papu : ')
#         display(df_papu.head())

#         # 將母體與特徵左右拼接
#         writing_feature_file_path = None
#         concat_df_outcome = get_feature_by_SOP([ym], mother,writing_feature_file_path, drop_key_word, df_papu)
#         key = str(ym)+'_df'
#         df_combined =concat_df_outcome[key]
#         print('df_combined : ')
#         display(df_combined.head())

#         # BIN的類型
#         print(f'type(bins[1]) : {type(bins[1])}')

#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'mlops_ref_info')

#         # 進行預測，產出各等級人數、預測後等級與特徵、DB名單檔、DB參照資訊檔
#         df_level_with_bin_num, df_pred_and_features, id_list_to_db, ref_info_to_db = \
#         predict_df_v230517(df_combined, [ym], writing_path, project_name, mother, algorithm,
#                            bins,papulation_colname, papulation_except_value,
#                            target, frequency, edition_detail,new_edition)

#         print('df_level_with_bin_num : ')
#         display(df_level_with_bin_num)

#         print('df_pred_and_features : ')
#         display(df_pred_and_features.head())

#         print('id_list_to_db  : ')
#         display(id_list_to_db.head())
#         print('id_list_to_db (except) : ')
#         display(id_list_to_db[np.isnan(id_list_to_db['probability'])].head())

#         print('ref_info_to_db : ')
#         display(ref_info_to_db)

#         # 寫入DB(名單檔)
#         save_db_id_list_mlops(write_db_Y_N, id_list_to_db, table_name='mlops_id_list')
#         # 寫入DB(參照資訊檔)
#         save_db_ref_info_mlops(write_db_Y_N, ref_info_to_db, table_name='mlops_ref_info')


#     need_other_tag = False
#     untrain_mother = '潛客'
#     if '非潛客' in mother_list and untrain_mother not in mother_list:
#         need_other_tag = True
#         # 宣告沒有建模的名單參數
#         mother = untrain_mother
#         if untrain_mother =='潛客': show_name = '潛力客群'
#         tag_name = show_name + '註記'
#         rank = show_name
#         print(f'{project_name}沒有對{untrain_mother}進行建模,用{show_name}註記')
#         date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#         next_month = monthdelta(date, 1)
#         print(next_month)
#         query = make_query_potential(next_month)
#         df_papu_untrain_mother = get_SQL_raw_data(query)
#         today = datetime.now().date()
#         if today < monthdelta(date, 4):
#             df_papu_untrain_mother['y'] = '999'
#             print(f"today: {today} < {ym} + 4 months, so don't have true y (all is 999)")
#         else:
#             print(f"today: {today} >= {ym} + 4 months, so have true y")

#         feature_ym = config.feature_file_path_fubon + '/' + '{}'.format(ym)
#         df = pickle.load(open(feature_ym + '/' + 'CUST_{}.pickle'.format(ym), 'rb'))
#         df['yyyymm'] = df['yyyymm'].astype('int')
#         df_papu_untrain_mother['yyyymm'] = df_papu_untrain_mother['yyyymm'].astype('int')
#         papu_and_y_yyyymm = df_papu_untrain_mother[df_papu_untrain_mother['yyyymm']==int(ym)]

#         df_untrain_mother = pd.merge(df, papu_and_y_yyyymm, how='left', on=['customer_id','yyyymm'])
#         mask_papulation = df_untrain_mother['y'].notna()
#         df_untrain_mother = df_untrain_mother[mask_papulation]
#         df_untrain_mother['y'] = df_untrain_mother['y'].astype('int')

#     if need_other_tag:
#         print('df_papu_untrain_mother : ')
#         display(df_papu_untrain_mother.head())

#         id_list_to_db = pd.DataFrame(data={
#                                             'product': project_name,
#                                             'target': target ,
#                                             'population': mother,
#                                             'frequency': frequency,
#                                             'snap_date':id_list_to_db['snap_date'].unique()[0] ,
#                                             'party_id': df_untrain_mother['customer_id'],
#                                             'tag_name': tag_name,
#                                             'probability': np.nan,
#                                             'rank': rank,
#                                             'pred_date': id_list_to_db['pred_date'].unique()[0]})
#         print('id_list_to_db_untrain : ')
#         display(id_list_to_db.head())


#         ref_info_to_db = pd.DataFrame(data={
#                                             'product': project_name,
#                                             'target': target ,
#                                             'population': mother,
#                                             'frequency': frequency,
#                                             'edition': np.nan,
#                                             'snap_date': ref_info_to_db['snap_date'].unique()[0] ,
#                                             'model_valid_falg': np.nan,
#                                             'tag_name':  tag_name,
#                                             'exception': '' ,
#                                             'edition_detail': '沒有進行建模純註記',
#                                             'pred_date': ref_info_to_db['pred_date'].unique()[0]


#                                         },index=[0])
#         print('ref_info_to_db_untrain : ')
#         display(ref_info_to_db)

#         save_db_id_list_mlops(write_db_Y_N, id_list_to_db, table_name='mlops_id_list')
#         save_db_ref_info_mlops(write_db_Y_N, ref_info_to_db, table_name='mlops_ref_info')


# In[7]:


# def predict_mlops_np_and_p_jihsun(this_file_path,target,papulation_colname,papulation_except_value,mother_list,query_list,ym,
#                            algorithm,write_db_Y_N,drop_key_word,frequency,edition_detail):
#     import sys
#     sys.path.append(this_file_path)
#     try:
#         from Papulation import make_query_non_potential, make_query_potential
#     except:
#         print(f'沒有潛客/非潛客 query function')
#         pass
#     try:
#         from Papulation import make_query
#     except:
#         print(f'沒有不分潛客 query function')
#         pass
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     from Model import monthdelta, confirm_retrain_log, get_bin_with_dif_mother,get_bin_with_dif_mother_version230417, \
#     predict_df_v230517, predict_log, save_db_mlops,save_db_ref_info_mlops,save_db_id_list_mlops,get_next_model_version
#     from Pretreatment import get_feature_by_SOP_jihsun
#     from Sql_module import get_SQL_raw_data
#     import config
#     import os
#     #該檔案路徑
#     writing_path = this_file_path
#     project_name = this_file_path.split('/')[-1]

#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from IPython.display import display
#     import gc

#     # 確認母體跟商品下的最新retrain結果
#     confirm_retrain_log(project_name, mother_list, table_name='mlops_retrain_log_jihsun')

#     # 抓取對應母體的bin
#     bins_list = get_bin_with_dif_mother_version230417(project_name, mother_list, table_name='mlops_model_log_jihsun')

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')

#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):
#         print(f'mother : {mother}')

#         query_function = query_list[index]
#         print(f'query_function  : {query_function}')

#         bins = bins_list[index]
#         print(f'bins  : {bins}')

#         # 抓取母體
#         date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#         next_month = monthdelta(date, 1)
#         print(next_month)
#         query = query_function(next_month)

#         if type(query) == str:
#             df_papu = get_SQL_raw_data(query, account=account_yt , pwd=pwd_yt)
#         else:
#             df_papu = query

#         today = datetime.now().date()
#         if today < monthdelta(date, 4):
#             df_papu['y'] = '999'
#             print(f"today: {today} < {ym} + 4 months, so don't have true y (all is 999)")
#         else:
#             print(f"today: {today} >= {ym} + 4 months, so have true y")
#         print('df_papu : ')
#         display(df_papu.head())

#         # 將母體與特徵左右拼接
#         writing_feature_file_path = None
#         concat_df_outcome = get_feature_by_SOP_jihsun([ym], mother,writing_feature_file_path, drop_key_word, df_papu)
#         key = str(ym)+'_df'
#         df_combined =concat_df_outcome[key]
#         print('df_combined : ')
#         display(df_combined.head())

#         # BIN的類型
#         print(f'type(bins[1]) : {type(bins[1])}')

#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'mlops_ref_info_jihsun')

#         # 進行預測，產出各等級人數、預測後等級與特徵、DB名單檔、DB參照資訊檔
#         df_level_with_bin_num, df_pred_and_features, id_list_to_db, ref_info_to_db = \
#         predict_df_v230517(df_combined, [ym], writing_path, project_name, mother, algorithm,
#                            bins,papulation_colname, papulation_except_value,
#                            target, frequency, edition_detail,new_edition)

#         print('df_level_with_bin_num : ')
#         display(df_level_with_bin_num)

#         print('df_pred_and_features : ')
#         display(df_pred_and_features.head())

#         print('id_list_to_db  : ')
#         display(id_list_to_db.head())
#         print('id_list_to_db (except) : ')
#         display(id_list_to_db[np.isnan(id_list_to_db['probability'])].head())

#         print('ref_info_to_db : ')
#         display(ref_info_to_db)

#         # 寫入DB(名單檔)
#         save_db_id_list_mlops(write_db_Y_N, id_list_to_db, table_name='mlops_id_list_jihsun')
#         # 寫入DB(參照資訊檔)
#         save_db_ref_info_mlops(write_db_Y_N, ref_info_to_db, table_name='mlops_ref_info_jihsun')


#     need_other_tag = False
#     untrain_mother = '潛客'
#     if '非潛客' in mother_list and untrain_mother not in mother_list:
#         need_other_tag = True
#         # 宣告沒有建模的名單參數
#         mother = untrain_mother
#         if untrain_mother =='潛客': show_name = '潛力客群'
#         tag_name = show_name + '註記'
#         rank = show_name
#         print(f'{project_name}沒有對{untrain_mother}進行建模,用{show_name}註記')
#         date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#         next_month = monthdelta(date, 1)
#         print(next_month)
#         query = make_query_potential(next_month)
#         df_papu_untrain_mother = get_SQL_raw_data(query)
#         today = datetime.now().date()
#         if today < monthdelta(date, 4):
#             df_papu_untrain_mother['y'] = '999'
#             print(f"today: {today} < {ym} + 4 months, so don't have true y (all is 999)")
#         else:
#             print(f"today: {today} >= {ym} + 4 months, so have true y")

#         feature_ym = config.feature_file_path_fubon + '/' + '{}'.format(ym)
#         df = pickle.load(open(feature_ym + '/' + 'CUST_{}.pickle'.format(ym), 'rb'))
#         df['yyyymm'] = df['yyyymm'].astype('int')
#         df_papu_untrain_mother['yyyymm'] = df_papu_untrain_mother['yyyymm'].astype('int')
#         papu_and_y_yyyymm = df_papu_untrain_mother[df_papu_untrain_mother['yyyymm']==int(ym)]

#         df_untrain_mother = pd.merge(df, papu_and_y_yyyymm, how='left', on=['customer_id','yyyymm'])
#         mask_papulation = df_untrain_mother['y'].notna()
#         df_untrain_mother = df_untrain_mother[mask_papulation]
#         df_untrain_mother['y'] = df_untrain_mother['y'].astype('int')

#     if need_other_tag:
#         print('df_papu_untrain_mother : ')
#         display(df_papu_untrain_mother.head())

#         id_list_to_db = pd.DataFrame(data={
#                                             'product': project_name,
#                                             'target': target ,
#                                             'population': mother,
#                                             'frequency': frequency,
#                                             'snap_date':id_list_to_db['snap_date'].unique()[0] ,
#                                             'party_id': df_untrain_mother['customer_id'],
#                                             'tag_name': tag_name,
#                                             'probability': np.nan,
#                                             'rank': rank,
#                                             'pred_date': id_list_to_db['pred_date'].unique()[0]})
#         print('id_list_to_db_untrain : ')
#         display(id_list_to_db.head())


#         ref_info_to_db = pd.DataFrame(data={
#                                             'product': project_name,
#                                             'target': target ,
#                                             'population': mother,
#                                             'frequency': frequency,
#                                             'edition': np.nan,
#                                             'snap_date': ref_info_to_db['snap_date'].unique()[0] ,
#                                             'model_valid_falg': np.nan,
#                                             'tag_name':  tag_name,
#                                             'exception': '' ,
#                                             'edition_detail': '沒有進行建模純註記',
#                                             'pred_date': ref_info_to_db['pred_date'].unique()[0]


#                                         },index=[0])
#         print('ref_info_to_db_untrain : ')
#         display(ref_info_to_db)

#         save_db_id_list_mlops(write_db_Y_N, id_list_to_db, table_name='mlops_id_list_jihsun')
#         save_db_ref_info_mlops(write_db_Y_N, ref_info_to_db, table_name='mlops_ref_info_jihsun')


# In[1]:


get_ipython().system('jupyter nbconvert --to script predict_mlops.ipynb')


# In[ ]:




