#!/usr/bin/env python

# In[1]:


#上個版本 retrain_mlops_np_and_p_double_opt_detailym_fillna_20241225
def retrain_mlops_np_and_p_double_opt_detailym_fillna_20250326(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
                           mother_list, bins_list, hit_rate_list, frequency, edition_detail, do_ym_list, query_function, write_db_Y_N,
                           limit_size_select, limit_size_build, write_feature_Y_N, market_flag_Y_N):
    import gc
    import sys

    from IPython.display import display
    #該檔案路徑
    this_prod = this_file_path.split('/')[-1]
    # 讀取模組設定
    sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
    import config
    # MLOPS執行參數
    algorithm = config.algorithm
    #潛客參數
    max_depth_a = config.max_depth_a
    scale_pos_weight_a = config.scale_pos_weight_a
    n_estimator_a = config.n_estimator_a
    #非潛客參數
    max_depth_b = config. max_depth_b
    scale_pos_weight_b = config.scale_pos_weight_b
    n_estimator_b = config.n_estimator_b


    import pickle
    import time

    import numpy as np
    from Model import (
        Convert_log_v240111,
        back_test,
        build_model_v230726,
        get_backtest_df_pvalue_IV,
        get_conversion_rank,
        get_next_model_version,
        model_log_v230410,
        retrain_log_v240115,
        select_feature_v230517,
        split_data,
        whether_done_next_version,
    )
    from Pretreatment import (
        combine_multi_year_df,
        down_sampling_from_df,
        get_feature_by_SOP_202503,
        get_papu_and_y_202503,
        get_recommded_build_size,
        if_to_large_down_sampling,
    )
    from Sql_module import send_table_to_sql
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

    # 判斷是設定有無問題
    if len(mother_list) == len(bins_list) :
        print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
    else:
        raise Exception('mother_list、bins_list長度不一致')

    query_function = query_function[0]
    for index, mother in enumerate(mother_list):
        totally_st_time = time.time()
        print(f'第{index}圈執行 {mother} Retrain ...')
        # 抓取對應 bins / query
        bins = bins_list[index]
        hit_rate = hit_rate_list[index]
        print(f'bin = {bins}')
        print(f'do_ym_list = {do_ym_list}')# 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
        new_edition = get_next_model_version(this_prod,target,mother,frequency,edition_detail, table_name = 'mlops_ref_info_double')
        print(f'new_edition = {new_edition}')

        if not whether_done_next_version(this_file_path, this_prod, mother, new_edition, target, algorithm,
                                    db_table_rt='mlops_retrain_log_double', db_table_md='mlops_model_log_double',
                                    account=config.account, pwd=config.pwd):
            time.sleep(5)

            #########################################################################################################
            print('!!!!開始抓取資料')
            # 先抓一次
            papu_and_y = get_papu_and_y_202503(this_prod, do_ym_list, query_function, papulation_colname, papulation_train_value,mother)
            limit_times = 1
            try_times = 1

            # 重複抓
            while set(do_ym_list)!=set(papu_and_y['yyyymm'].astype('str')) and try_times <= limit_times:
                papu_and_y = get_papu_and_y_202503(this_prod, do_ym_list, query_function, papulation_colname, papulation_train_value,mother)
                try_times = try_times+1

            print('papu_and_y :')
            display(papu_and_y.head())
            # 排除例外(例如:舊戶)
            papu_and_y = papu_and_y[papu_and_y[papulation_colname].isin(papulation_train_value)]
            papu_and_y.drop([papulation_colname], axis=1, inplace=True)

            # 將母體與特徵左右拼接
            concat_df_outcome = get_feature_by_SOP_202503(do_ym_list, mother, drop_key_word, papu_and_y, this_prod, Fill_zero=True)

            # 將各年月的DF上下拼接
            df_combined = combine_multi_year_df(do_ym_list, concat_df_outcome)
            print(f'df_combined.shape = {df_combined.shape}')

            # 額外縮減至每個年月最多10萬筆(用於select_feature)
            concat_df_outcome_redeuced = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_select)
            df_combined_redeuced = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced)
            print(f'df_combined_redeuced.shape = {df_combined_redeuced.shape}')

            # 清出空間
            del papu_and_y
            del concat_df_outcome
            del concat_df_outcome_redeuced
            gc.collect()

    #         df_combined['yyyymm'] = df_combined['yyyymm'].astype('int')
            # 若是做商品行銷類，則透過建議的母體比例調整正式建立模型的樣本比例
            if market_flag_Y_N:
                need_adjuste_sample, recommded_build_size = get_recommded_build_size(df_combined[['customer_id', 'yyyymm', 'y']], mother)

                if need_adjuste_sample:
                    print(f'現行流程富證日盛訓練資料每月抓取{limit_size_build}個樣本')
                    print(f'最後資料數為{len(df_combined)} 大於 建議的樣本數為{recommded_build_size}，故執行後續調整比例!!')
                    df_combined = down_sampling_from_df(df_combined, recommded_build_size)
                else:
                    print(f'現行流程富證日盛訓練資料每月抓取{limit_size_build}個樣本')
                    print(f'最後資料數為{len(df_combined)} 小於 建議的樣本數為{recommded_build_size}，故不執行後續調整比例!!')


            X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
            feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
                                              this_prod, mother, target, new_edition, algorithm)
            cols_top100 = feat_imp.loc[:100]['feature'].tolist()

            # 釋出空間
            del X_train_rdc
            del y_train_rdc
            del X_test_rdc
            del y_test_rdc
            del df_combined_redeuced
            gc.collect()

            # 以較大資料集，正式建模
            based_col = ['customer_id', 'yyyymm', 'y']
            X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top100],
                                                          train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
            model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top100, this_file_path,
                                                                           this_prod, mother, target, bins, new_edition, max_depth = max_depth_b,
                                                                           scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
            print('cols_impt_top100 : ')
            display(model_imp)
            print('test_level_df: ')
            display(test_level_df)

            # 以最新年月進行回測並產出效度
            backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
            print('vali_level_df : ')
            display(vali_level_df)
            print('vali_predict_df.head() : ')
            display(vali_predict_df.head())
            print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
            print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
            print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')

            # 以詳細切分方法進行最新年月回測並產出效度
            backtest_auc_dt, vali_level_df_dt, vali_predict_df_dt, df_pred_and_features_dt =             back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]],
                      bins = [0,100,200,300,400,500,600,700,800,900,
                              1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,
                              11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,
                              30000,40000,50000,60000,70000,80000,90000,100000,
                              120000,140000,160000,180000,200000,
                              1000000])
#             print(f'vali_level_df_dt : ')
            display(vali_level_df_dt)
#             print(f'vali_predict_df_dt.head() : ')
            display(vali_predict_df_dt.head())
            print(f'vali_predivali_predict_df_dtct_df.shape :  {vali_predict_df_dt.shape}\n')
            print(f'df_pred_and_features_dt.columns : \n {df_pred_and_features_dt.columns}\n')
            print(f'df_pred_and_features_dt.shape : \n {df_pred_and_features_dt.shape}\n')

            # 寫出PICKLE回測等級
            pickle.dump(vali_predict_df_dt, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_detail' + this_prod +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

            # 行銷相關才做轉換率
            if market_flag_Y_N:

                hit_rate_setting_bin = []
                for ht in sorted([0]+hit_rate, reverse=True):
                        if  ht == 0:
                            hit_rate_setting_bin.append( f"C{int(round(ht * 100)):02d}" )
                        elif  ht < 0.001:
                            hit_rate_setting_bin.append( f"C{int(round(ht * 10000)):04d}" )
                        elif  ht < 0.01:
                            hit_rate_setting_bin.append( f"C{int(round(ht * 1000)):03d}" )
                        else:
                            hit_rate_setting_bin.append( f"C{int(round(ht * 100)):02d}" )

                CHANCE_TEST_bin, CHANCE_TEST_num, CHANCE_TEST_hit_rate, CHANCE_TEST_prob = get_conversion_rank(vali_predict_df_dt, hit_rate)
                # retrain_log
                log_retrain = retrain_log_v240115(False, this_prod, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
                                                  train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
                                                  vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
                                                  CHANCE_TEST_bin=CHANCE_TEST_bin, CHANCE_TEST_num=CHANCE_TEST_num,
                                                  CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate, CHANCE_TEST_prob=CHANCE_TEST_prob)
                print('retrain_log :')
                display(log_retrain)

                # model_log
                log_model = model_log_v230410(False, this_prod, mother, target, algorithm, hit_rate_setting_bin, new_edition, table_name = 'mlops_model_log_double')
                print('model_log :')
                display(log_model)



                log_retrain_v230410 = retrain_log_v240115(write_db_Y_N, this_prod, mother, df_combined, X_train, y_train,test_level_df,
                                                          vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
                                                          vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
                                                          CHANCE_TEST_bin=CHANCE_TEST_bin, CHANCE_TEST_num=CHANCE_TEST_num,
                                                          CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate, CHANCE_TEST_prob=CHANCE_TEST_prob)
                log_model_v230410 = model_log_v230410(write_db_Y_N, this_prod, mother, target, algorithm, hit_rate_setting_bin, new_edition,
                                                      table_name = 'mlops_model_log_double')

                # 額外寫出轉換率對照表
                detail_hit_rate  = list(map(lambda x: round(x*0.0001,4), range(10)))[1:]  + list(map(lambda x: round(x*0.001,3), range(10)))[1:] + list(map(lambda x: round(x*0.01,2), range(26)))[1:]
                CHANCE_TEST_bin_dt, CHANCE_TEST_num_dt, CHANCE_TEST_hit_rate_dt, CHANCE_TEST_prob_dt = get_conversion_rank(vali_predict_df_dt, detail_hit_rate)
                conver_log = Convert_log_v240111(False, this_prod, mother, df_combined,
                                            backtest_auc, new_edition, table_name = 'mlops_convert_log_double',
                                           CHANCE_TEST_bin=CHANCE_TEST_bin_dt, CHANCE_TEST_num=CHANCE_TEST_num_dt,
                                            CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate_dt, CHANCE_TEST_prob=CHANCE_TEST_prob_dt)
                print('conver_log :')
                display(conver_log)
                conver_log = Convert_log_v240111(write_db_Y_N, this_prod, mother, df_combined,
                                            backtest_auc, new_edition, table_name = 'mlops_convert_log_double',
                                           CHANCE_TEST_bin=CHANCE_TEST_bin_dt, CHANCE_TEST_num=CHANCE_TEST_num_dt,
                                            CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate_dt, CHANCE_TEST_prob=CHANCE_TEST_prob_dt)
            else:
                # retrain_log
                log_retrain = retrain_log_v240115(False, this_prod, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
                                                  train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
                                                  vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
                                                  )
                print('retrain_log :')
                display(log_retrain)

                # model_log
                log_model = model_log_v230410(False, this_prod, mother, target, algorithm, bins,new_edition, table_name = 'mlops_model_log_double')
                print('model_log :')
                display(log_model)



                log_retrain_v230410 = retrain_log_v240115(write_db_Y_N, this_prod, mother, df_combined, X_train, y_train,test_level_df,
                                                          vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
                                                          vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
                                                          )
                log_model_v230410 = model_log_v230410(write_db_Y_N, this_prod, mother, target, algorithm, bins, new_edition,
                                                      table_name = 'mlops_model_log_double')

            ## If retrain_auc < 0.7 then print

            if test_auc < 0.7:
                print(f'{mother}retrain_auc < 0.7')


            print(f'{mother} 花了 {(time.time()-totally_st_time)/60} min')
            ############# START ##########################################
            # 額外針對回測月份計算p-value及IV值
            #
            calcu_time = time.time()
            calcuate_top_n = 20

            pvalue_col = ['y'] + list(model_imp['feature'])[0:calcuate_top_n]
            backtest_df_pvalue_IV = get_backtest_df_pvalue_IV(this_prod, target, mother, df_combined, pvalue_col)

            backtest_df_pvalue_IV = backtest_df_pvalue_IV[['prod', 'target', 'population', 'test_period', 'feature', 'feature_chinese',
                                   'dtype', 'p_value', 'IV', 'error_msg']]
             # 小數點126位限制轉換
            lower_limit = 1E-126
            def clip_float_values(value):
                if np.isfinite(value):
                    if abs(value) < lower_limit and value != 0:
                        return np.sign(value) * lower_limit
                return value

            backtest_df_pvalue_IV['p_value'] = backtest_df_pvalue_IV['p_value'].apply(clip_float_values)

            send_table_to_sql(backtest_df_pvalue_IV, 'impt_feature_statistic', account=config.account, pwd=config.pwd)
            print(f'{mother} 計算p-value&IV值花了 {(time.time()-calcu_time)/60} min')
            ######### END ###################################################

            del X_train
            del y_train
            del X_test
            del y_test
            del df_combined
            del vali_predict_df
            del df_pred_and_features
            gc.collect()

            print(f'{this_prod}_{mother}_{new_edition} 執行完畢，先跳出迴圈!')
            break
        else:
            print(f'{this_prod}_{mother}_{new_edition} 先前已執行過了，執行下個母體!')


# In[2]:


#上個版本 retrain_mlops_np_and_p_double_opt_fillna
def retrain_mlops_np_and_p_double_opt_fillna_20250326(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
                           mother_list, bins_list, frequency, edition_detail, do_ym_list, query_function, write_db_Y_N,
                           limit_size_select, limit_size_build, write_feature_Y_N):
    import gc
    import sys

    from IPython.display import display
    #該檔案路徑
    this_prod = this_file_path.split('/')[-1]
    # 讀取模組設定
    sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
    import config
    # MLOPS執行參數
    algorithm = config.algorithm
    #潛客參數
    max_depth_a = config.max_depth_a
    scale_pos_weight_a = config.scale_pos_weight_a
    n_estimator_a = config.n_estimator_a
    #非潛客參數
    max_depth_b = config. max_depth_b
    scale_pos_weight_b = config.scale_pos_weight_b
    n_estimator_b = config.n_estimator_b

    import pickle
    import time

    import numpy as np
    from Model import (
        back_test,
        build_model_v230726,
        get_backtest_df_pvalue_IV,
        get_next_model_version,
        model_log_v230410,
        retrain_log_v230726,
        select_feature_v230517,
        split_data,
        whether_done_next_version,
    )
    from Pretreatment import (
        combine_multi_year_df,
        get_feature_by_SOP_202503,
        get_papu_and_y_202503,
        if_to_large_down_sampling,
    )
    from Sql_module import send_table_to_sql
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

    # 判斷是設定有無問題
    if len(mother_list) == len(bins_list):
        print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
    else:
        raise Exception('mother_list、bins_list長度不一致')

    query_function = query_function[0]
    for index, mother in enumerate(mother_list):

        totally_st_time = time.time()
        print(f'第{index}圈執行 {mother} Retrain ...')
        bins = bins_list[index]
        print(f'bin = {bins}')
        # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
        new_edition = get_next_model_version(this_prod,target,mother,frequency,edition_detail, table_name = 'mlops_ref_info_double')

        if not whether_done_next_version(this_file_path, this_prod, mother, new_edition, target, algorithm,
                                        db_table_rt='mlops_retrain_log_double', db_table_md='mlops_model_log_double',
                                        account=config.account_kris, pwd=config.pwd_kris):
            time.sleep(15)
            print(f'do_ym_list = {do_ym_list}')

            ############################################################################################################
            print('!!!!開始抓取資料')
            # 先抓一次
            papu_and_y = get_papu_and_y_202503(this_prod, do_ym_list, query_function, papulation_colname, papulation_train_value,mother)
            limit_times = 1
            try_times = 1

            # 重複抓
            while set(do_ym_list)!=set(papu_and_y['yyyymm'].astype('str')) and try_times <= limit_times:
                papu_and_y = get_papu_and_y_202503(this_prod, do_ym_list, query_function, papulation_colname, papulation_train_value,mother)
                try_times = try_times+1

            print('papu_and_y :')
#             display(papu_and_y.head())
            # 排除例外(例如:舊戶)
            papu_and_y = papu_and_y[papu_and_y[papulation_colname].isin(papulation_train_value)]
            papu_and_y.drop([papulation_colname], axis=1, inplace=True)


            # 將母體與特徵左右拼接
            concat_df_outcome = get_feature_by_SOP_202503(do_ym_list, mother, drop_key_word, papu_and_y, this_prod, Fill_zero=True)

            # 將各年月的DF上下拼接
            df_combined = combine_multi_year_df(do_ym_list, concat_df_outcome)
            print(f'df_combined.shape = {df_combined.shape}')

            # 額外縮減至每個年月最多10萬筆(用於select_feature)
            concat_df_outcome_redeuced = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_select)
            df_combined_redeuced = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced)
            print(f'df_combined_redeuced.shape = {df_combined_redeuced.shape}')

            # 清出空間
            del papu_and_y
            del concat_df_outcome
            del concat_df_outcome_redeuced
            gc.collect()


            # 以較小資料集，建模選取重要特徵
            X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
            feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
                                              this_prod, mother, target, new_edition, algorithm)
            cols_top100 = feat_imp.loc[:100]['feature'].tolist()

            # 釋出空間
            del X_train_rdc
            del y_train_rdc
            del X_test_rdc
            del y_test_rdc
            del df_combined_redeuced
            gc.collect()

            # 以較大資料集，正式建模
            based_col = ['customer_id', 'yyyymm', 'y']
            X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top100],
                                                          train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
            model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top100, this_file_path,
                                                                           this_prod, mother, target, bins, new_edition, max_depth = max_depth_b,
                                                                           scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
            print('cols_impt_top100 : ')
            display(model_imp)
            print('test_level_df: ')
            display(test_level_df)

            # 以最新年月進行回測並產出效度
            backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
            print('vali_level_df : ')
            display(vali_level_df)
            print('vali_predict_df.head() : ')
            display(vali_predict_df.head())
            print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
            print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
            print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')

            # 以詳細切分方法進行最新年月回測並產出效度
            backtest_auc_dt, vali_level_df_dt, vali_predict_df_dt, df_pred_and_features_dt =             back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]],
                      bins = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,
                              11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,
                              30000,40000,50000,60000,70000,80000,90000,100000,
                              120000,140000,160000,180000,200000,
                              1000000])
            print('vali_level_df_dt : ')
            display(vali_level_df_dt)
            print('vali_predict_df_dt.head() : ')
            display(vali_predict_df_dt.head())
            print(f'vali_predivali_predict_df_dtct_df.shape :  {vali_predict_df_dt.shape}\n')
            print(f'df_pred_and_features_dt.columns : \n {df_pred_and_features_dt.columns}\n')
            print(f'df_pred_and_features_dt.shape : \n {df_pred_and_features_dt.shape}\n')

            # 寫出PICKLE回測等級
            pickle.dump(vali_predict_df_dt, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_detail' + this_prod +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

            # retrain_log
            log_retrain = retrain_log_v230726(False, this_prod, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
                                              train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
                                              vali_level_df_detail=vali_level_df_dt )
            print('retrain_log :')
            display(log_retrain)

            # model_log
            log_model = model_log_v230410(False, this_prod, mother, target, algorithm, bins,new_edition, table_name = 'mlops_model_log_double')
            print('model_log :')
            display(log_model)



            log_retrain_v230410 = retrain_log_v230726(write_db_Y_N, this_prod, mother, df_combined, X_train, y_train,test_level_df,
                                                      vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
                                                      vali_level_df_detail=vali_level_df_dt )
            log_model_v230410 = model_log_v230410(write_db_Y_N, this_prod, mother, target, algorithm, bins, new_edition,
                                                  table_name = 'mlops_model_log_double')

            ## If retrain_auc < 0.7 then print

            if test_auc < 0.7:
                print(f'{mother}retrain_auc < 0.7')

            print(f'{mother} 花了 {(time.time()-totally_st_time)/60} min')


            ############# START ##########################################
            # 額外針對回測月份計算p-value及IV值
            #
            calcu_time = time.time()
            calcuate_top_n = 20



            pvalue_col = ['y'] + list(model_imp['feature'])[0:calcuate_top_n]
            backtest_df_pvalue_IV = get_backtest_df_pvalue_IV(this_prod, target, mother, df_combined, pvalue_col)

            backtest_df_pvalue_IV = backtest_df_pvalue_IV[['prod', 'target', 'population', 'test_period', 'feature', 'feature_chinese',
                                   'dtype', 'p_value', 'IV', 'error_msg']]
             # 小數點126位限制轉換
            lower_limit = 1E-126
            def clip_float_values(value):
                if np.isfinite(value):
                    if abs(value) < lower_limit and value != 0:
                        return np.sign(value) * lower_limit
                return value

            backtest_df_pvalue_IV['p_value'] = backtest_df_pvalue_IV['p_value'].apply(clip_float_values)

            send_table_to_sql(backtest_df_pvalue_IV, 'impt_feature_statistic', account=config.account, pwd=config.pwd)
            print(f'{mother} 計算p-value&IV值花了 {(time.time()-calcu_time)/60} min')
            ######### END ###################################################


            del X_train
            del y_train
            del X_test
            del y_test
            del df_combined
            del vali_predict_df
            del df_pred_and_features
            gc.collect()

            print(f'{this_prod}_{mother}_{new_edition} 執行完畢，先跳出迴圈!')
            break
        else:
            print(f'{this_prod}_{mother}_{new_edition} 先前已執行過了，執行下個母體!')


# In[3]:


# def retrain_mlops_np_and_p_double_preRun_popu(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, query_list, query_list_jihsun, frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N, y_type_name = ''):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b


#     from Sql_module import get_SQL_raw_data, write_data_to_SQL
#     from Pretreatment import get_feature_by_SOP,get_feature_by_SOP_jihsun,combine_multi_year_df, if_to_large_down_sampling, get_papu_and_y
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v230726, model_log_v230410, retrain_log_v230314, get_next_model_version, whether_done_next_version
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):

#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         query_function = query_list[index]
#         query_function_jihsun = query_list_jihsun[index]
#         print(f'bin = {bins}')

#         time.sleep(15)
#         print(f'do_ym_list = {do_ym_list}')

#         #########################################################################################################
#         # 富邦DF
#         ############################################################################################################
#         print('!!!!開始抓取富邦資料')
#         # 先抓一次
#         fubon_ft_fold = this_file_path.replace('審核通過模型_雙證','審核通過模型')
#         papu_and_y_fubon = get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                writing_popu_path = fubon_ft_fold, mother = mother, y_type_name = y_type_name)
#         limit_times = 1
#         try_times = 1
#         # 重複抓
#         while set(do_ym_list)!=set(papu_and_y_fubon['yyyymm'].astype('str')) and try_times <= limit_times:
#             papu_and_y_fubon = get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                    writing_popu_path = fubon_ft_fold, mother = mother, y_type_name = y_type_name)
#             try_times = try_times+1

#         print('papu_and_y_fubon :')
#         display(papu_and_y_fubon.head())

#         ###############################################################################################################
#         # 日盛DF
#         ############################################################################################################
#         print('!!!!開始抓取日盛資料')
#         # 先抓一次
#         jihsun_ft_fold = this_file_path.replace('審核通過模型_雙證','審核通過模型_jihsun')
#         papu_and_y_jihsun = get_papu_and_y(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                writing_popu_path = jihsun_ft_fold, mother = mother, y_type_name = y_type_name)
#         limit_times = 1
#         try_times = 1
#         # 重複抓
#         while set(do_ym_list)!=set(papu_and_y_jihsun['yyyymm'].astype('str')) and try_times <= limit_times:
#             papu_and_y_jihsun = get_papu_and_y(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                    writing_popu_path = jihsun_ft_fold, mother = mother, y_type_name = y_type_name)
#             try_times = try_times+1

#         print('papu_and_y_jihsun :')
#         display(papu_and_y_jihsun.head())



# In[4]:


# def retrain_mlops_np_and_p_double_opt_detailym_fillna_20231025(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, hit_rate_list, query_list, query_list_jihsun, frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N, market_flag_Y_N):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b

#     feature_file_path_fubon = config.feature_file_path_fubon
#     feature_file_path_jihsun = config.feature_file_path_jihsun

#     from Sql_module import get_SQL_raw_data, write_data_to_SQL
#     from Pretreatment import get_feature_by_SOP,get_feature_by_SOP_jihsun,combine_multi_year_df, if_to_large_down_sampling, get_papu_and_y_local, if_large_down_sampling_from_papu_and_y
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v240115, Convert_log_v240111
#     from Model import model_log_v230410, retrain_log_v230314, get_next_model_version, whether_done_next_version, get_conversion_rank
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')
#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):

#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         hit_rate = hit_rate_list[index]
#         query_function = query_list[index]
#         query_function_jihsun = query_list_jihsun[index]
#         print(f'bin = {bins}')
#         print(f'do_ym_list = {do_ym_list}')# 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'mlops_ref_info_double')


#         if not whether_done_next_version(this_file_path, project_name, mother, new_edition, target, algorithm,
#                                         db_table_rt='mlops_retrain_log_double', db_table_md='mlops_model_log_double',
#                                         account=config.account_yt, pwd=config.pwd_yt):
#             time.sleep(15)

#             #########################################################################################################
#             # 富邦DF
#             ############################################################################################################
#             print('!!!!開始抓取富邦資料')
#             # 先抓一次
#             fubon_ft_fold = this_file_path.replace('審核通過模型_雙證','審核通過模型')
#             papu_and_y_fubon = get_papu_and_y_local(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                    writing_popu_path = fubon_ft_fold, mother = mother)
#             limit_times = 1
#             try_times = 1
#             # 重複抓
#             while set(do_ym_list)!=set(papu_and_y_fubon['yyyymm'].astype('str')) and try_times <= limit_times:
#                 papu_and_y_fubon = get_papu_and_y_local(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                        writing_popu_path = fubon_ft_fold, mother = mother)
#                 try_times = try_times+1

#             print('papu_and_y_fubon :')
#             display(papu_and_y_fubon.head())
#             # 排除例外(例如:舊戶)
#             papu_and_y_fubon = papu_and_y_fubon[papu_and_y_fubon[papulation_colname].isin(papulation_train_value)]
#             papu_and_y_fubon.drop([papulation_colname], axis=1, inplace=True)

# #             # 重複訓練集最新年月
# #             do_ym_list_sorted = sorted(do_ym_list)
# #             dupli_ym = str(do_ym_list_sorted[-2])
# #             dupli_ym_df = papu_and_y_fubon[papu_and_y_fubon['yyyymm'] == int(dupli_ym)]
# #             dupli_ym_df_y1 = dupli_ym_df[dupli_ym_df['y'] == 1]
# #             print(f'重複最新年月({dupli_ym}) Y_VALUE前Y數量為: {len(dupli_ym_df_y1)}')
# #             papu_and_y_fubon = papu_and_y_fubon.append(dupli_ym_df_y1)
# #             mask1 = (papu_and_y_fubon['yyyymm'] == int(dupli_ym))
# #             mask2 = (papu_and_y_fubon['y'] == 1)
# #             print(f'重複一次年月({dupli_ym}) Y_VALUE後Y數量為: {len(papu_and_y_fubon[mask1&mask2])}')
# #             papu_and_y_fubon = papu_and_y_fubon.append(dupli_ym_df_y1)
# #             mask1 = (papu_and_y_fubon['yyyymm'] == int(dupli_ym))
# #             mask2 = (papu_and_y_fubon['y'] == 1)
# #             print(f'重複二次年月({dupli_ym}) Y_VALUE後Y數量為: {len(papu_and_y_fubon[mask1&mask2])}')

#             # 是否存特徵, 若要存特徵則存在this_file_path
#             writing_path = None
#             if write_feature_Y_N:
#                 writing_path = this_file_path.replace('審核通過模型_雙證','審核通過模型')

#             # 每個年月最多50萬筆
#             papu_and_y_fubon_build = if_large_down_sampling_from_papu_and_y(do_ym_list, papu_and_y_fubon, cust_source='Fubon', limit_size=limit_size_build)

#             # 將母體與特徵左右拼接
#             concat_df_outcome_fubon = get_feature_by_SOP(do_ym_list, mother, writing_path, drop_key_word, papu_and_y_fubon_build,
#                                                          feature_file_path = feature_file_path_fubon, just_for_check=False, Fill_zero=True)

#             # 重複訓練集最新年月
# #             do_ym_list_sorted = sorted(do_ym_list)
# #             dupli_ym_df_key = str(do_ym_list_sorted[-2])+'_df'
# #             if dupli_ym_df_key in concat_df_outcome_fubon.keys():
# #                 df_dupli_ym = concat_df_outcome_fubon[dupli_ym_df_key].copy()
# #                 Y_NUMBER = sum(concat_df_outcome_fubon[dupli_ym_df_key]['y']==1)
# #                 print(f'重複最新年月({dupli_ym_df_key}) Y_VALUE前Y數量為: {Y_NUMBER}')

# #                 concat_df_outcome_fubon[dupli_ym_df_key] = concat_df_outcome_fubon[dupli_ym_df_key].append(df_dupli_ym[df_dupli_ym['y']==1])
# #                 Y_NUMBER = sum(concat_df_outcome_fubon[dupli_ym_df_key]['y']==1)
# #                 print(f'重複一次年月({dupli_ym_df_key}) Y_VALUE後Y數量為: {Y_NUMBER}')

# #                 concat_df_outcome_fubon[dupli_ym_df_key] = concat_df_outcome_fubon[dupli_ym_df_key].append(df_dupli_ym[df_dupli_ym['y']==1])
# #                 Y_NUMBER = sum(concat_df_outcome_fubon[dupli_ym_df_key]['y']==1)
# #                 print(f'重複二次年月({dupli_ym_df_key}) Y_VALUE後Y數量為: {Y_NUMBER}')

#            # 將各年月的DF上下拼接
#             df_combined_fubon = combine_multi_year_df(do_ym_list, concat_df_outcome_fubon)
#             print(f'df_combined_fubon.shape = {df_combined_fubon.shape}')

#             # 額外縮減至每個年月最多10萬筆(用於select_feature)
#             concat_df_outcome_redeuced_fubon = if_to_large_down_sampling(do_ym_list, concat_df_outcome_fubon, limit_size=limit_size_select)
#             df_combined_redeuced_fubon = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced_fubon)
#             print(f'df_combined_redeuced_fubon.shape = {df_combined_redeuced_fubon.shape}')

#            # 清出空間
#             del papu_and_y_fubon
#             del concat_df_outcome_fubon
#             del concat_df_outcome_redeuced_fubon
#             gc.collect()
#             ###############################################################################################################
#             # 日盛DF
#             ############################################################################################################
#             print('!!!!開始抓取日盛資料')
#             # 先抓一次
#             fubon_ft_fold = this_file_path.replace('審核通過模型_雙證','審核通過模型_jihsun')
#             papu_and_y_jihsun = get_papu_and_y_local(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                    writing_popu_path = fubon_ft_fold, mother = mother)
#             limit_times = 1
#             try_times = 1
#             # 重複抓
#             while set(do_ym_list)!=set(papu_and_y_jihsun['yyyymm'].astype('str')) and try_times <= limit_times:
#                 papu_and_y_jihsun = get_papu_and_y_local(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                        writing_popu_path = fubon_ft_fold, mother = mother)
#                 try_times = try_times+1

#             print('papu_and_y_jihsun :')
#             display(papu_and_y_jihsun.head())
#             # 排除例外(例如:舊戶)
#             papu_and_y_jihsun = papu_and_y_jihsun[papu_and_y_jihsun[papulation_colname].isin(papulation_train_value)]
#             papu_and_y_jihsun.drop([papulation_colname], axis=1, inplace=True)

# #             # 重複訓練集最新年月
# #             do_ym_list_sorted = sorted(do_ym_list)
# #             dupli_ym = str(do_ym_list_sorted[-2])
# #             dupli_ym_df = papu_and_y_jihsun[papu_and_y_jihsun['yyyymm'] == int(dupli_ym)]
# #             dupli_ym_df_y1 = dupli_ym_df[dupli_ym_df['y'] == 1]
# #             print(f'重複最新年月({dupli_ym}) Y_VALUE前Y數量為: {len(dupli_ym_df_y1)}')
# #             papu_and_y_jihsun = papu_and_y_jihsun.append(dupli_ym_df_y1)
# #             mask1 = (papu_and_y_jihsun['yyyymm'] == int(dupli_ym))
# #             mask2 = (papu_and_y_jihsun['y'] == 1)
# #             print(f'重複一次年月({dupli_ym}) Y_VALUE後Y數量為: {len(papu_and_y_jihsun[mask1&mask2])}')
# #             papu_and_y_jihsun = papu_and_y_jihsun.append(dupli_ym_df_y1)
# #             mask1 = (papu_and_y_jihsun['yyyymm'] == int(dupli_ym))
# #             mask2 = (papu_and_y_jihsun['y'] == 1)
# #             print(f'重複二次年月({dupli_ym}) Y_VALUE後Y數量為: {len(papu_and_y_jihsun[mask1&mask2])}')

#             # 是否存特徵, 若要存特徵則存在this_file_path
#             writing_path = None
#             if write_feature_Y_N:
#                 writing_path = this_file_path.replace('審核通過模型_雙證','審核通過模型_jihsun')

#             # 每個年月最多50萬筆
#             papu_and_y_jihsun_build = if_large_down_sampling_from_papu_and_y(do_ym_list, papu_and_y_jihsun, cust_source='Jihsun', limit_size=limit_size_build)

#             # 將母體與特徵左右拼接
#             concat_df_outcome_jihsun = get_feature_by_SOP_jihsun(do_ym_list, mother, writing_path, drop_key_word, papu_and_y_jihsun_build,
#                                                                  feature_file_path = feature_file_path_jihsun, just_for_check=False, Fill_zero=True)

#             # 重複訓練集最新年月
# #             do_ym_list_sorted = sorted(do_ym_list)
# #             dupli_ym_df_key = str(do_ym_list_sorted[-2])+'_df'
# #             if dupli_ym_df_key in concat_df_outcome_jihsun.keys():

# #                 df_dupli_ym = concat_df_outcome_jihsun[dupli_ym_df_key].copy()
# #                 Y_NUMBER = sum(concat_df_outcome_jihsun[dupli_ym_df_key]['y']==1)
# #                 print(f'重複最新年月({dupli_ym_df_key}) Y_VALUE前Y數量為: {Y_NUMBER}')

# #                 concat_df_outcome_jihsun[dupli_ym_df_key] = concat_df_outcome_jihsun[dupli_ym_df_key].append(df_dupli_ym[df_dupli_ym['y']==1])
# #                 Y_NUMBER = sum(concat_df_outcome_jihsun[dupli_ym_df_key]['y']==1)
# #                 print(f'重複一次年月({dupli_ym_df_key}) Y_VALUE後Y數量為: {Y_NUMBER}')

# #                 concat_df_outcome_jihsun[dupli_ym_df_key] = concat_df_outcome_jihsun[dupli_ym_df_key].append(df_dupli_ym[df_dupli_ym['y']==1])
# #                 Y_NUMBER = sum(concat_df_outcome_jihsun[dupli_ym_df_key]['y']==1)
# #                 print(f'重複二次年月({dupli_ym_df_key}) Y_VALUE後Y數量為: {Y_NUMBER}')

#             # 將各年月的DF上下拼接
#             df_combined_jihsun = combine_multi_year_df(do_ym_list, concat_df_outcome_jihsun)
#             print(f'df_combined_jihsun.shape = {df_combined_jihsun.shape}')



#             # 額外縮減至每個年月最多10萬筆(用於select_feature)
#             concat_df_outcome_redeuced_jihsun = if_to_large_down_sampling(do_ym_list, concat_df_outcome_jihsun, limit_size=limit_size_select)
#             df_combined_redeuced_jihsun = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced_jihsun)
#             print(f'df_combined_redeuced_jihsun.shape = {df_combined_redeuced_jihsun.shape}')

#             # 清出空間
#             del papu_and_y_jihsun
#             del concat_df_outcome_jihsun
#             del concat_df_outcome_redeuced_jihsun
#             gc.collect()
#             ###############################################################################################################
#             # 開始合併富證日盛客戶
#             ##############################################################################################################
#             print('!!!!開始合併富邦日盛資料')
#             # 資料: [build_set]
#             st_concat = time.time()
#             df_combined = pd.concat([df_combined_fubon,df_combined_jihsun],axis = 0)
#             print(f'[build_set] 富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#             # Concat後categorical會變objective,要轉回來
#             st_concat_astype = time.time()
#             obj_cols = df_combined.select_dtypes('object').drop(['customer_id'],axis=1).columns
#             df_combined[obj_cols.tolist()] = df_combined[obj_cols.tolist()].astype('category')
#             print(f'[build_set] 富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#             #移除雙證重複
#             remove_dupli_id_time = time.time()
#             df_combined['status'] = df_combined.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#             print(f'[build_set] 原始雙證人數: {len(df_combined)}')
#             df_combined = df_combined[df_combined['status']==1]
#             print(f'[build_set] 移除重複後雙證人數: {len(df_combined)}')
#             print(f'[build_set] 富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')


#             # 資料: [select_feature_set]
#             st_concat = time.time()
#             df_combined_redeuced = pd.concat([df_combined_redeuced_fubon,df_combined_redeuced_jihsun],axis = 0)
#             print(f'[select_feature_set] 富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#             # Concat後categorical會變objective,要轉回來
#             st_concat_astype = time.time()
#             obj_cols = df_combined_redeuced.select_dtypes('object').drop(['customer_id'],axis=1).columns
#             df_combined_redeuced[obj_cols.tolist()] = df_combined_redeuced[obj_cols.tolist()].astype('category')
#             print(f'[select_feature_set] 富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#             #移除雙證重複
#             remove_dupli_id_time = time.time()
#             df_combined_redeuced['status'] = df_combined_redeuced.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#             print(f'[select_feature_set] 原始雙證人數: {len(df_combined_redeuced)}')
#             df_combined_redeuced = df_combined_redeuced[df_combined_redeuced['status']==1]
#             print(f'[select_feature_set] 移除重複後雙證人數: {len(df_combined_redeuced)}')
#             print(f'[select_feature_set] 富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')

#             # 以較小資料集，建模選取重要特徵
#             X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
#             feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
#                                               project_name, mother, target, new_edition, algorithm)
#             cols_top100 = feat_imp.loc[:100]['feature'].tolist()

#             # 釋出空間
#             del X_train_rdc
#             del y_train_rdc
#             del X_test_rdc
#             del y_test_rdc
#             del df_combined_redeuced_fubon
#             del df_combined_redeuced_jihsun
#             del df_combined_redeuced
#             gc.collect()

#             # 以較大資料集，正式建模
#             based_col = ['customer_id', 'yyyymm', 'y']
#             X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top100],
#                                                           train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
#             model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top100, this_file_path,
#                                                                            project_name, mother, target, bins, new_edition, max_depth = max_depth_b,
#                                                                            scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
#             print('cols_impt_top100 : ')
#             display(model_imp)
#             print('test_level_df: ')
#             display(test_level_df)

#             # 以最新年月進行回測並產出效度
#             backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#             print(f'vali_level_df : ')
#             display(vali_level_df)
#             print(f'vali_predict_df.head() : ')
#             display(vali_predict_df.head())
#             print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
#             print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
#             print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')

#             # 以詳細切分方法進行最新年月回測並產出效度
#             backtest_auc_dt, vali_level_df_dt, vali_predict_df_dt, df_pred_and_features_dt = \
#             back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]],
#                       bins = [0,100,200,300,400,500,600,700,800,900,
#                               1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,
#                               11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,
#                               30000,40000,50000,60000,70000,80000,90000,100000,
#                               120000,140000,160000,180000,200000,
#                               1000000])
#             print(f'vali_level_df_dt : ')
#             display(vali_level_df_dt)
#             print(f'vali_predict_df_dt.head() : ')
#             display(vali_predict_df_dt.head())
#             print(f'vali_predivali_predict_df_dtct_df.shape :  {vali_predict_df_dt.shape}\n')
#             print(f'df_pred_and_features_dt.columns : \n {df_pred_and_features_dt.columns}\n')
#             print(f'df_pred_and_features_dt.shape : \n {df_pred_and_features_dt.shape}\n')

#             # 寫出PICKLE回測等級
#             pickle.dump(vali_predict_df_dt, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_detail' + project_name +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

#             # 行銷相關才做轉換率
#             if market_flag_Y_N:

#                 hit_rate_setting_bin = []
#                 for ht in sorted([0]+hit_rate, reverse=True):
#                         if  ht == 0:
#                             hit_rate_setting_bin.append( "C{:02d}".format(int(round(ht * 100))) )
#                         elif  ht < 0.001:
#                             hit_rate_setting_bin.append( "C{:04d}".format(int(round(ht * 10000))) )
#                         elif  ht < 0.01:
#                             hit_rate_setting_bin.append( "C{:03d}".format(int(round(ht * 1000))) )
#                         else:
#                             hit_rate_setting_bin.append( "C{:02d}".format(int(round(ht * 100))) )

#                 CHANCE_TEST_bin, CHANCE_TEST_num, CHANCE_TEST_hit_rate, CHANCE_TEST_prob = get_conversion_rank(vali_predict_df_dt, hit_rate)
#                 # retrain_log
#                 log_retrain = retrain_log_v240115(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                                   train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
#                                                   vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                                   CHANCE_TEST_bin=CHANCE_TEST_bin, CHANCE_TEST_num=CHANCE_TEST_num,
#                                                   CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate, CHANCE_TEST_prob=CHANCE_TEST_prob)
#                 print(f'retrain_log :')
#                 display(log_retrain)

#                 # model_log
#                 log_model = model_log_v230410(False, project_name, mother, target, algorithm, hit_rate_setting_bin, new_edition, table_name = 'mlops_model_log_double')
#                 print(f'model_log :')
#                 display(log_model)



#                 log_retrain_v230410 = retrain_log_v240115(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                           vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
#                                                           vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                                           CHANCE_TEST_bin=CHANCE_TEST_bin, CHANCE_TEST_num=CHANCE_TEST_num,
#                                                           CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate, CHANCE_TEST_prob=CHANCE_TEST_prob)
#                 log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, hit_rate_setting_bin, new_edition,
#                                                       table_name = 'mlops_model_log_double')

#                 # 額外寫出轉換率對照表
#                 detail_hit_rate  = list(map(lambda x: round(x*0.0001,4), range(10)))[1:]  + list(map(lambda x: round(x*0.001,3), range(10)))[1:] + list(map(lambda x: round(x*0.01,2), range(26)))[1:]
#                 CHANCE_TEST_bin_dt, CHANCE_TEST_num_dt, CHANCE_TEST_hit_rate_dt, CHANCE_TEST_prob_dt = get_conversion_rank(vali_predict_df_dt, detail_hit_rate)
#                 conver_log = Convert_log_v240111(False, project_name, mother, df_combined,
#                                             backtest_auc, new_edition, table_name = 'mlops_convert_log_double',
#                                            CHANCE_TEST_bin=CHANCE_TEST_bin_dt, CHANCE_TEST_num=CHANCE_TEST_num_dt,
#                                             CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate_dt, CHANCE_TEST_prob=CHANCE_TEST_prob_dt)
#                 print(f'conver_log :')
#                 display(conver_log)
#                 conver_log = Convert_log_v240111(write_db_Y_N, project_name, mother, df_combined,
#                                             backtest_auc, new_edition, table_name = 'mlops_convert_log_double',
#                                            CHANCE_TEST_bin=CHANCE_TEST_bin_dt, CHANCE_TEST_num=CHANCE_TEST_num_dt,
#                                             CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate_dt, CHANCE_TEST_prob=CHANCE_TEST_prob_dt)
#             else:
#                 # retrain_log
#                 log_retrain = retrain_log_v240115(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                                   train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
#                                                   vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                                   )
#                 print(f'retrain_log :')
#                 display(log_retrain)

#                 # model_log
#                 log_model = model_log_v230410(False, project_name, mother, target, algorithm, bins,new_edition, table_name = 'mlops_model_log_double')
#                 print(f'model_log :')
#                 display(log_model)



#                 log_retrain_v230410 = retrain_log_v240115(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                           vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
#                                                           vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                                           )
#                 log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition,
#                                                       table_name = 'mlops_model_log_double')

#             ## If retrain_auc < 0.7 then print

#             if test_auc < 0.7:
#                 print('{}retrain_auc < 0.7'.format(mother))


#             print('{} 花了 {} min'.format(mother,(time.time()-totally_st_time)/60))


#             del X_train
#             del y_train
#             del X_test
#             del y_test
#             del df_combined_fubon
#             del df_combined_jihsun
#             del vali_predict_df
#             del df_pred_and_features
#             gc.collect()

#             print(f'{project_name}_{mother}_{new_edition} 執行完畢，先跳出迴圈!')
#             break
#         else:
#             print(f'{project_name}_{mother}_{new_edition} 先前已執行過了，執行下個母體!')


# In[5]:


# def retrain_mlops_np_and_p_double_opt_detailym_fillna_20241225(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, hit_rate_list, query_list, query_list_jihsun, frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N, market_flag_Y_N):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b

#     feature_file_path_fubon = config.feature_file_path_fubon
#     feature_file_path_jihsun = config.feature_file_path_jihsun

#     from Sql_module import get_SQL_raw_data, write_data_to_SQL, send_table_to_sql
#     from Pretreatment import get_feature_by_SOP,get_feature_by_SOP_jihsun,combine_multi_year_df, if_to_large_down_sampling, get_papu_and_y_local, if_large_down_sampling_from_papu_and_y, get_recommded_build_size, down_sampling_from_df
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v240115, Convert_log_v240111
#     from Model import model_log_v230410, retrain_log_v230314, get_next_model_version, whether_done_next_version, get_conversion_rank, get_backtest_df_pvalue_IV
#     from Model import get_backtest_df_pvalue_IV
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')
#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):

#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         hit_rate = hit_rate_list[index]
#         query_function = query_list[index]
#         query_function_jihsun = query_list_jihsun[index]
#         print(f'bin = {bins}')
#         print(f'do_ym_list = {do_ym_list}')# 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'mlops_ref_info_double')


#         if not whether_done_next_version(this_file_path, project_name, mother, new_edition, target, algorithm,
#                                         db_table_rt='mlops_retrain_log_double', db_table_md='mlops_model_log_double',
#                                         account=config.account_yt, pwd=config.pwd_yt):
#             time.sleep(15)

#             #########################################################################################################
#             # 富邦DF
#             ############################################################################################################
#             print('!!!!開始抓取富邦資料')
#             # 先抓一次
#             fubon_ft_fold = this_file_path.replace('審核通過模型_雙證','審核通過模型')
#             papu_and_y_fubon = get_papu_and_y_local(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                    writing_popu_path = fubon_ft_fold, mother = mother)
#             limit_times = 1
#             try_times = 1
#             # 重複抓
#             while set(do_ym_list)!=set(papu_and_y_fubon['yyyymm'].astype('str')) and try_times <= limit_times:
#                 papu_and_y_fubon = get_papu_and_y_local(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                        writing_popu_path = fubon_ft_fold, mother = mother)
#                 try_times = try_times+1

#             print('papu_and_y_fubon :')
#             display(papu_and_y_fubon.head())
#             # 排除例外(例如:舊戶)
#             papu_and_y_fubon = papu_and_y_fubon[papu_and_y_fubon[papulation_colname].isin(papulation_train_value)]
#             papu_and_y_fubon.drop([papulation_colname], axis=1, inplace=True)

#             # 是否存特徵, 若要存特徵則存在this_file_path
#             writing_path = None
#             if write_feature_Y_N:
#                 writing_path = this_file_path.replace('審核通過模型_雙證','審核通過模型')

#             # 每個年月最多50萬筆
#             papu_and_y_fubon_build = if_large_down_sampling_from_papu_and_y(do_ym_list, papu_and_y_fubon, cust_source='Fubon', limit_size=limit_size_build)

#             # 將母體與特徵左右拼接
#             concat_df_outcome_fubon = get_feature_by_SOP(do_ym_list, mother, writing_path, drop_key_word, papu_and_y_fubon_build,
#                                                          feature_file_path = feature_file_path_fubon, just_for_check=False, Fill_zero=True)


#            # 將各年月的DF上下拼接
#             df_combined_fubon = combine_multi_year_df(do_ym_list, concat_df_outcome_fubon)
#             print(f'df_combined_fubon.shape = {df_combined_fubon.shape}')

#             # 額外縮減至每個年月最多10萬筆(用於select_feature)
#             concat_df_outcome_redeuced_fubon = if_to_large_down_sampling(do_ym_list, concat_df_outcome_fubon, limit_size=limit_size_select)
#             df_combined_redeuced_fubon = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced_fubon)
#             print(f'df_combined_redeuced_fubon.shape = {df_combined_redeuced_fubon.shape}')

#            # 清出空間
#             del papu_and_y_fubon
#             del concat_df_outcome_fubon
#             del concat_df_outcome_redeuced_fubon
#             gc.collect()
#             ###############################################################################################################
#             # 日盛DF
#             ############################################################################################################
#             print('!!!!開始抓取日盛資料')
#             # 先抓一次
#             fubon_ft_fold = this_file_path.replace('審核通過模型_雙證','審核通過模型_jihsun')
#             papu_and_y_jihsun = get_papu_and_y_local(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                    writing_popu_path = fubon_ft_fold, mother = mother)
#             limit_times = 1
#             try_times = 1
#             # 重複抓
#             while set(do_ym_list)!=set(papu_and_y_jihsun['yyyymm'].astype('str')) and try_times <= limit_times:
#                 papu_and_y_jihsun = get_papu_and_y_local(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                        writing_popu_path = fubon_ft_fold, mother = mother)
#                 try_times = try_times+1

#             print('papu_and_y_jihsun :')
#             display(papu_and_y_jihsun.head())
#             # 排除例外(例如:舊戶)
#             papu_and_y_jihsun = papu_and_y_jihsun[papu_and_y_jihsun[papulation_colname].isin(papulation_train_value)]
#             papu_and_y_jihsun.drop([papulation_colname], axis=1, inplace=True)


#             # 是否存特徵, 若要存特徵則存在this_file_path
#             writing_path = None
#             if write_feature_Y_N:
#                 writing_path = this_file_path.replace('審核通過模型_雙證','審核通過模型_jihsun')

#             # 每個年月最多50萬筆
#             papu_and_y_jihsun_build = if_large_down_sampling_from_papu_and_y(do_ym_list, papu_and_y_jihsun, cust_source='Jihsun', limit_size=limit_size_build)

#             # 將母體與特徵左右拼接
#             concat_df_outcome_jihsun = get_feature_by_SOP_jihsun(do_ym_list, mother, writing_path, drop_key_word, papu_and_y_jihsun_build,
#                                                                  feature_file_path = feature_file_path_jihsun, just_for_check=False, Fill_zero=True)

#             # 將各年月的DF上下拼接
#             df_combined_jihsun = combine_multi_year_df(do_ym_list, concat_df_outcome_jihsun)
#             print(f'df_combined_jihsun.shape = {df_combined_jihsun.shape}')



#             # 額外縮減至每個年月最多10萬筆(用於select_feature)
#             concat_df_outcome_redeuced_jihsun = if_to_large_down_sampling(do_ym_list, concat_df_outcome_jihsun, limit_size=limit_size_select)
#             df_combined_redeuced_jihsun = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced_jihsun)
#             print(f'df_combined_redeuced_jihsun.shape = {df_combined_redeuced_jihsun.shape}')

#             # 清出空間
#             del papu_and_y_jihsun
#             del concat_df_outcome_jihsun
#             del concat_df_outcome_redeuced_jihsun
#             gc.collect()
#             ###############################################################################################################
#             # 開始合併富證日盛客戶
#             ##############################################################################################################
#             print('!!!!開始合併富邦日盛資料')
#             # 資料: [build_set]
#             st_concat = time.time()
#             df_combined = pd.concat([df_combined_fubon,df_combined_jihsun],axis = 0)
#             print(f'[build_set] 富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#             # Concat後categorical會變objective,要轉回來
#             st_concat_astype = time.time()
#             obj_cols = df_combined.select_dtypes('object').drop(['customer_id'],axis=1).columns
#             df_combined[obj_cols.tolist()] = df_combined[obj_cols.tolist()].astype('category')
#             print(f'[build_set] 富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#             #移除雙證重複
#             remove_dupli_id_time = time.time()
#             df_combined['status'] = df_combined.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#             print(f'[build_set] 原始雙證人數: {len(df_combined)}')
#             df_combined = df_combined[df_combined['status']==1]
#             print(f'[build_set] 移除重複後雙證人數: {len(df_combined)}')
#             print(f'[build_set] 富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')

#             # 若是做商品行銷類，則透過建議的母體比例調整正式建立模型的樣本比例
#             if market_flag_Y_N:
#                 need_adjuste_sample, recommded_build_size = get_recommded_build_size(df_combined[['customer_id', 'yyyymm', 'y']], mother)

#                 if need_adjuste_sample:
#                     print(f'現行流程富證日盛訓練資料每月抓取{limit_size_build}個樣本')
#                     print(f'最後資料數為{len(df_combined)} 大於 建議的樣本數為{recommded_build_size}，故執行後續調整比例!!')
#                     df_combined = down_sampling_from_df(df_combined, recommded_build_size)
#                 else:
#                     print(f'現行流程富證日盛訓練資料每月抓取{limit_size_build}個樣本')
#                     print(f'最後資料數為{len(df_combined)} 小於 建議的樣本數為{recommded_build_size}，故不執行後續調整比例!!')

#             # 資料: [select_feature_set]
#             st_concat = time.time()
#             df_combined_redeuced = pd.concat([df_combined_redeuced_fubon,df_combined_redeuced_jihsun],axis = 0)
#             print(f'[select_feature_set] 富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#             # Concat後categorical會變objective,要轉回來
#             st_concat_astype = time.time()
#             obj_cols = df_combined_redeuced.select_dtypes('object').drop(['customer_id'],axis=1).columns
#             df_combined_redeuced[obj_cols.tolist()] = df_combined_redeuced[obj_cols.tolist()].astype('category')
#             print(f'[select_feature_set] 富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#             #移除雙證重複
#             remove_dupli_id_time = time.time()
#             df_combined_redeuced['status'] = df_combined_redeuced.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#             print(f'[select_feature_set] 原始雙證人數: {len(df_combined_redeuced)}')
#             df_combined_redeuced = df_combined_redeuced[df_combined_redeuced['status']==1]
#             print(f'[select_feature_set] 移除重複後雙證人數: {len(df_combined_redeuced)}')
#             print(f'[select_feature_set] 富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')

#             # 以較小資料集，建模選取重要特徵
#             X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
#             feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
#                                               project_name, mother, target, new_edition, algorithm)
#             cols_top100 = feat_imp.loc[:100]['feature'].tolist()

#             # 釋出空間
#             del X_train_rdc
#             del y_train_rdc
#             del X_test_rdc
#             del y_test_rdc
#             del df_combined_redeuced_fubon
#             del df_combined_redeuced_jihsun
#             del df_combined_redeuced
#             gc.collect()

#             # 以較大資料集，正式建模
#             based_col = ['customer_id', 'yyyymm', 'y']
#             X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top100],
#                                                           train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
#             model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top100, this_file_path,
#                                                                            project_name, mother, target, bins, new_edition, max_depth = max_depth_b,
#                                                                            scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
#             print('cols_impt_top100 : ')
#             display(model_imp)
#             print('test_level_df: ')
#             display(test_level_df)

#             # 以最新年月進行回測並產出效度
#             backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#             print(f'vali_level_df : ')
#             display(vali_level_df)
#             print(f'vali_predict_df.head() : ')
#             display(vali_predict_df.head())
#             print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
#             print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
#             print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')

#             # 以詳細切分方法進行最新年月回測並產出效度
#             backtest_auc_dt, vali_level_df_dt, vali_predict_df_dt, df_pred_and_features_dt = \
#             back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]],
#                       bins = [0,100,200,300,400,500,600,700,800,900,
#                               1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,
#                               11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,
#                               30000,40000,50000,60000,70000,80000,90000,100000,
#                               120000,140000,160000,180000,200000,
#                               1000000])
#             print(f'vali_level_df_dt : ')
#             display(vali_level_df_dt)
#             print(f'vali_predict_df_dt.head() : ')
#             display(vali_predict_df_dt.head())
#             print(f'vali_predivali_predict_df_dtct_df.shape :  {vali_predict_df_dt.shape}\n')
#             print(f'df_pred_and_features_dt.columns : \n {df_pred_and_features_dt.columns}\n')
#             print(f'df_pred_and_features_dt.shape : \n {df_pred_and_features_dt.shape}\n')

#             # 寫出PICKLE回測等級
#             pickle.dump(vali_predict_df_dt, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_detail' + project_name +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

#             # 行銷相關才做轉換率
#             if market_flag_Y_N:

#                 hit_rate_setting_bin = []
#                 for ht in sorted([0]+hit_rate, reverse=True):
#                         if  ht == 0:
#                             hit_rate_setting_bin.append( "C{:02d}".format(int(round(ht * 100))) )
#                         elif  ht < 0.001:
#                             hit_rate_setting_bin.append( "C{:04d}".format(int(round(ht * 10000))) )
#                         elif  ht < 0.01:
#                             hit_rate_setting_bin.append( "C{:03d}".format(int(round(ht * 1000))) )
#                         else:
#                             hit_rate_setting_bin.append( "C{:02d}".format(int(round(ht * 100))) )

#                 CHANCE_TEST_bin, CHANCE_TEST_num, CHANCE_TEST_hit_rate, CHANCE_TEST_prob = get_conversion_rank(vali_predict_df_dt, hit_rate)
#                 # retrain_log
#                 log_retrain = retrain_log_v240115(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                                   train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
#                                                   vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                                   CHANCE_TEST_bin=CHANCE_TEST_bin, CHANCE_TEST_num=CHANCE_TEST_num,
#                                                   CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate, CHANCE_TEST_prob=CHANCE_TEST_prob)
#                 print(f'retrain_log :')
#                 display(log_retrain)

#                 # model_log
#                 log_model = model_log_v230410(False, project_name, mother, target, algorithm, hit_rate_setting_bin, new_edition, table_name = 'mlops_model_log_double')
#                 print(f'model_log :')
#                 display(log_model)



#                 log_retrain_v230410 = retrain_log_v240115(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                           vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
#                                                           vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                                           CHANCE_TEST_bin=CHANCE_TEST_bin, CHANCE_TEST_num=CHANCE_TEST_num,
#                                                           CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate, CHANCE_TEST_prob=CHANCE_TEST_prob)
#                 log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, hit_rate_setting_bin, new_edition,
#                                                       table_name = 'mlops_model_log_double')

#                 # 額外寫出轉換率對照表
#                 detail_hit_rate  = list(map(lambda x: round(x*0.0001,4), range(10)))[1:]  + list(map(lambda x: round(x*0.001,3), range(10)))[1:] + list(map(lambda x: round(x*0.01,2), range(26)))[1:]
#                 CHANCE_TEST_bin_dt, CHANCE_TEST_num_dt, CHANCE_TEST_hit_rate_dt, CHANCE_TEST_prob_dt = get_conversion_rank(vali_predict_df_dt, detail_hit_rate)
#                 conver_log = Convert_log_v240111(False, project_name, mother, df_combined,
#                                             backtest_auc, new_edition, table_name = 'mlops_convert_log_double',
#                                            CHANCE_TEST_bin=CHANCE_TEST_bin_dt, CHANCE_TEST_num=CHANCE_TEST_num_dt,
#                                             CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate_dt, CHANCE_TEST_prob=CHANCE_TEST_prob_dt)
#                 print(f'conver_log :')
#                 display(conver_log)
#                 conver_log = Convert_log_v240111(write_db_Y_N, project_name, mother, df_combined,
#                                             backtest_auc, new_edition, table_name = 'mlops_convert_log_double',
#                                            CHANCE_TEST_bin=CHANCE_TEST_bin_dt, CHANCE_TEST_num=CHANCE_TEST_num_dt,
#                                             CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate_dt, CHANCE_TEST_prob=CHANCE_TEST_prob_dt)
#             else:
#                 # retrain_log
#                 log_retrain = retrain_log_v240115(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                                   train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
#                                                   vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                                   )
#                 print(f'retrain_log :')
#                 display(log_retrain)

#                 # model_log
#                 log_model = model_log_v230410(False, project_name, mother, target, algorithm, bins,new_edition, table_name = 'mlops_model_log_double')
#                 print(f'model_log :')
#                 display(log_model)



#                 log_retrain_v230410 = retrain_log_v240115(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                           vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
#                                                           vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                                           )
#                 log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition,
#                                                       table_name = 'mlops_model_log_double')

#             ## If retrain_auc < 0.7 then print

#             if test_auc < 0.7:
#                 print('{}retrain_auc < 0.7'.format(mother))


#             print('{} 花了 {} min'.format(mother,(time.time()-totally_st_time)/60))
#             ############# START ##########################################
#             # 額外針對回測月份計算p-value及IV值
#             #
#             calcu_time = time.time()
#             calcuate_top_n = 20

#             pvalue_col = ['y'] + list(model_imp['feature'])[0:calcuate_top_n]
#             backtest_df_pvalue_IV = get_backtest_df_pvalue_IV(project_name, target, mother, df_combined, pvalue_col)

#             backtest_df_pvalue_IV = backtest_df_pvalue_IV[['prod', 'target', 'population', 'test_period', 'feature', 'feature_chinese',
#                                    'dtype', 'p_value', 'IV', 'error_msg']]
#              # 小數點126位限制轉換
#             lower_limit = 1E-126
#             def clip_float_values(value):
#                 if np.isfinite(value):
#                     if abs(value) < lower_limit and value != 0:
#                         return np.sign(value) * lower_limit
#                 return value

#             backtest_df_pvalue_IV['p_value'] = backtest_df_pvalue_IV['p_value'].apply(clip_float_values)

#             send_table_to_sql(backtest_df_pvalue_IV, 'impt_feature_statistic', account=config.account, pwd=config.pwd)
#             print('{} 計算p-value&IV值花了 {} min'.format(mother,(time.time()-calcu_time)/60))
#             ######### END ###################################################

#             del X_train
#             del y_train
#             del X_test
#             del y_test
#             del df_combined_fubon
#             del df_combined_jihsun
#             del vali_predict_df
#             del df_pred_and_features
#             gc.collect()

#             print(f'{project_name}_{mother}_{new_edition} 執行完畢，先跳出迴圈!')
#             break
#         else:
#             print(f'{project_name}_{mother}_{new_edition} 先前已執行過了，執行下個母體!')


# In[6]:


# def retrain_mlops_np_and_p_double_opt_detailym_fillna(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, query_list, query_list_jihsun, frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b

#     feature_file_path_fubon = config.feature_file_path_fubon
#     feature_file_path_jihsun = config.feature_file_path_jihsun

#     from Sql_module import get_SQL_raw_data, write_data_to_SQL
#     from Pretreatment import get_feature_by_SOP,get_feature_by_SOP_jihsun,combine_multi_year_df, if_to_large_down_sampling, get_papu_and_y_local
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v230726, model_log_v230410, retrain_log_v230314, get_next_model_version, whether_done_next_version
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')
#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):

#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         query_function = query_list[index]
#         query_function_jihsun = query_list_jihsun[index]
#         print(f'bin = {bins}')
#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'mlops_ref_info_double')

#         if not whether_done_next_version(this_file_path, project_name, mother, new_edition, target, algorithm,
#                                         db_table_rt='mlops_retrain_log_double', db_table_md='mlops_model_log_double',
#                                         account=config.account_yt, pwd=config.pwd_yt):
#             time.sleep(15)
#             print(f'do_ym_list = {do_ym_list}')

#             #########################################################################################################
#             # 富邦DF
#             ############################################################################################################
#             print('!!!!開始抓取富邦資料')
#             # 先抓一次
#             fubon_ft_fold = this_file_path.replace('審核通過模型_雙證','審核通過模型')
#             papu_and_y_fubon = get_papu_and_y_local(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                    writing_popu_path = fubon_ft_fold, mother = mother)
#             limit_times = 1
#             try_times = 1
#             # 重複抓
#             while set(do_ym_list)!=set(papu_and_y_fubon['yyyymm'].astype('str')) and try_times <= limit_times:
#                 papu_and_y_fubon = get_papu_and_y_local(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                        writing_popu_path = fubon_ft_fold, mother = mother)
#                 try_times = try_times+1

#             print('papu_and_y_fubon :')
#             display(papu_and_y_fubon.head())
#             # 排除例外(例如:舊戶)
#             papu_and_y_fubon = papu_and_y_fubon[papu_and_y_fubon[papulation_colname].isin(papulation_train_value)]
#             papu_and_y_fubon.drop([papulation_colname], axis=1, inplace=True)
#             # 是否存特徵, 若要存特徵則存在this_file_path
#             writing_path = None
#             if write_feature_Y_N:
#                 writing_path = this_file_path.replace('審核通過模型_雙證','審核通過模型')
#             # 將母體與特徵左右拼接
#             concat_df_outcome_fubon = get_feature_by_SOP(do_ym_list, mother, writing_path, drop_key_word, papu_and_y_fubon,
#                                                          feature_file_path = feature_file_path_fubon, just_for_check=False, Fill_zero=True)
#             do_ym_list_sorted = sorted(do_ym_list)

#             # 重複訓練集最新年月
#             dupli_ym_df_key = str(do_ym_list_sorted[-2])+'_df'
#             if dupli_ym_df_key in concat_df_outcome_fubon.keys():

#                 df_dupli_ym = concat_df_outcome_fubon[dupli_ym_df_key].copy()
#                 Y_NUMBER = sum(concat_df_outcome_fubon[dupli_ym_df_key]['y']==1)
#                 print(f'重複最新年月({dupli_ym_df_key}) Y_VALUE前Y數量為: {Y_NUMBER}')

#                 concat_df_outcome_fubon[dupli_ym_df_key] = concat_df_outcome_fubon[dupli_ym_df_key].append(df_dupli_ym[df_dupli_ym['y']==1])
#                 Y_NUMBER = sum(concat_df_outcome_fubon[dupli_ym_df_key]['y']==1)
#                 print(f'重複一次年月({dupli_ym_df_key}) Y_VALUE後Y數量為: {Y_NUMBER}')

#                 concat_df_outcome_fubon[dupli_ym_df_key] = concat_df_outcome_fubon[dupli_ym_df_key].append(df_dupli_ym[df_dupli_ym['y']==1])
#                 Y_NUMBER = sum(concat_df_outcome_fubon[dupli_ym_df_key]['y']==1)
#                 print(f'重複二次年月({dupli_ym_df_key}) Y_VALUE後Y數量為: {Y_NUMBER}')

#             # 每個年月最多50萬筆
#             concat_df_outcome_fubon = if_to_large_down_sampling(do_ym_list, concat_df_outcome_fubon, limit_size=limit_size_build)
#             # 將各年月的DF上下拼接
#             df_combined_fubon = combine_multi_year_df(do_ym_list, concat_df_outcome_fubon)
#             print(f'df_combined_fubon.shape = {df_combined_fubon.shape}')

# #             # FILLNA df_combined
# #             feature_list = list(df_combined_fubon.loc[:,~df_combined_fubon.columns.isin(['customer_id','yyyymm','y'])].select_dtypes(exclude = ['category','object']))
# #             for fl in feature_list:
# #                 na_num = sum(df_combined_fubon[fl].isna())
# #                 if na_num>0:
# #                     df_combined_fubon[fl] = df_combined_fubon[fl].fillna(0)
# #                     na_num = sum(df_combined_fubon[fl].isna())
# #                     if na_num>0:
# #                         print(f'[df_combined_fubon] 補NA後特徵{fl}變為{na_num}個NA')


#             # 額外縮減至每個年月最多10萬筆(用於select_feature)
#             concat_df_outcome_redeuced_fubon = if_to_large_down_sampling(do_ym_list, concat_df_outcome_fubon, limit_size=limit_size_select)
#             df_combined_redeuced_fubon = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced_fubon)
#             print(f'df_combined_redeuced_fubon.shape = {df_combined_redeuced_fubon.shape}')

# #             # FILLNA df_combined_redeuced
# #             feature_list = list(df_combined_redeuced_fubon.loc[:,~df_combined_redeuced_fubon.columns.isin(['customer_id','yyyymm','y'])].select_dtypes(exclude = ['category','object']))
# #             for fl in feature_list:
# #                 na_num = sum(df_combined_redeuced_fubon[fl].isna())
# #                 if na_num>0:
# #                     df_combined_redeuced_fubon[fl] = df_combined_redeuced_fubon[fl].fillna(0)
# #                     na_num = sum(df_combined_redeuced_fubon[fl].isna())
# #                     if na_num>0:
# #                         print(f'[df_combined_redeuced_fubon] 補NA後特徵{fl}變為{na_num}個NA')

#             # 清出空間
#             del papu_and_y_fubon
#             del concat_df_outcome_fubon
#             del concat_df_outcome_redeuced_fubon
#             gc.collect()
#             ###############################################################################################################
#             # 日盛DF
#             ############################################################################################################
#             print('!!!!開始抓取日盛資料')
#             # 先抓一次
#             fubon_ft_fold = this_file_path.replace('審核通過模型_雙證','審核通過模型_jihsun')
#             papu_and_y_jihsun = get_papu_and_y_local(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                    writing_popu_path = fubon_ft_fold, mother = mother)
#             limit_times = 1
#             try_times = 1
#             # 重複抓
#             while set(do_ym_list)!=set(papu_and_y_jihsun['yyyymm'].astype('str')) and try_times <= limit_times:
#                 papu_and_y_jihsun = get_papu_and_y_local(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                        writing_popu_path = fubon_ft_fold, mother = mother)
#                 try_times = try_times+1

#             print('papu_and_y_jihsun :')
#             display(papu_and_y_jihsun.head())
#             # 排除例外(例如:舊戶)
#             papu_and_y_jihsun = papu_and_y_jihsun[papu_and_y_jihsun[papulation_colname].isin(papulation_train_value)]
#             papu_and_y_jihsun.drop([papulation_colname], axis=1, inplace=True)
#             # 是否存特徵, 若要存特徵則存在this_file_path
#             writing_path = None
#             if write_feature_Y_N:
#                 writing_path = this_file_path.replace('審核通過模型_雙證','審核通過模型_jihsun')

#             # 將母體與特徵左右拼接
#             concat_df_outcome_jihsun = get_feature_by_SOP_jihsun(do_ym_list, mother, writing_path, drop_key_word, papu_and_y_jihsun,
#                                                                  feature_file_path = feature_file_path_jihsun, just_for_check=False, Fill_zero=True)

#             # 重複訓練集最新年月
#             dupli_ym_df_key = str(do_ym_list_sorted[-2])+'_df'
#             if dupli_ym_df_key in concat_df_outcome_jihsun.keys():

#                 df_dupli_ym = concat_df_outcome_jihsun[dupli_ym_df_key].copy()
#                 Y_NUMBER = sum(concat_df_outcome_jihsun[dupli_ym_df_key]['y']==1)
#                 print(f'重複最新年月({dupli_ym_df_key}) Y_VALUE前Y數量為: {Y_NUMBER}')

#                 concat_df_outcome_jihsun[dupli_ym_df_key] = concat_df_outcome_jihsun[dupli_ym_df_key].append(df_dupli_ym[df_dupli_ym['y']==1])
#                 Y_NUMBER = sum(concat_df_outcome_jihsun[dupli_ym_df_key]['y']==1)
#                 print(f'重複一次年月({dupli_ym_df_key}) Y_VALUE後Y數量為: {Y_NUMBER}')

#                 concat_df_outcome_jihsun[dupli_ym_df_key] = concat_df_outcome_jihsun[dupli_ym_df_key].append(df_dupli_ym[df_dupli_ym['y']==1])
#                 Y_NUMBER = sum(concat_df_outcome_jihsun[dupli_ym_df_key]['y']==1)
#                 print(f'重複二次年月({dupli_ym_df_key}) Y_VALUE後Y數量為: {Y_NUMBER}')

#             # 每個年月最多50萬筆
#             concat_df_outcome_jihsun = if_to_large_down_sampling(do_ym_list, concat_df_outcome_jihsun, limit_size=limit_size_build)

#             # 將各年月的DF上下拼接
#             df_combined_jihsun = combine_multi_year_df(do_ym_list, concat_df_outcome_jihsun)
#             print(f'df_combined_jihsun.shape = {df_combined_jihsun.shape}')

# #             # FILLNA df_combined
# #             feature_list = list(df_combined_jihsun.loc[:,~df_combined_jihsun.columns.isin(['customer_id','yyyymm','y'])].select_dtypes(exclude = ['category','object']))
# #             for fl in feature_list:
# #                 na_num = sum(df_combined_jihsun[fl].isna())
# #                 if na_num>0:
# #                     df_combined_jihsun[fl] = df_combined_jihsun[fl].fillna(0)
# #                     na_num = sum(df_combined_jihsun[fl].isna())
# #                     if na_num>0:
# #                         print(f'[df_combined_jihsun] 補NA後特徵{fl}變為{na_num}個NA')

#             # 額外縮減至每個年月最多10萬筆(用於select_feature)
#             concat_df_outcome_redeuced_jihsun = if_to_large_down_sampling(do_ym_list, concat_df_outcome_jihsun, limit_size=limit_size_select)
#             df_combined_redeuced_jihsun = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced_jihsun)
#             print(f'df_combined_redeuced_jihsun.shape = {df_combined_redeuced_jihsun.shape}')

# #             # FILLNA df_combined_redeuced
# #             feature_list = list(df_combined_redeuced_jihsun.loc[:,~df_combined_redeuced_jihsun.columns.isin(['customer_id','yyyymm','y'])].select_dtypes(exclude = ['category','object']))
# #             for fl in feature_list:
# #                 na_num = sum(df_combined_redeuced_jihsun[fl].isna())
# #                 if na_num>0:
# #                     df_combined_redeuced_jihsun[fl] = df_combined_redeuced_jihsun[fl].fillna(0)
# #                     na_num = sum(df_combined_redeuced_jihsun[fl].isna())
# #                     if na_num>0:
# #                         print(f'[df_combined_redeuced_jihsun] 補NA後特徵{fl}變為{na_num}個NA')

#             # 清出空間
#             del papu_and_y_jihsun
#             del concat_df_outcome_jihsun
#             del concat_df_outcome_redeuced_jihsun
#             gc.collect()
#             ###############################################################################################################
#             # 開始合併富證日盛客戶
#             ##############################################################################################################
#             print('!!!!開始合併富邦日盛資料')
#             # 資料: [build_set]
#             st_concat = time.time()
#             df_combined = pd.concat([df_combined_fubon,df_combined_jihsun],axis = 0)
#             print(f'[build_set] 富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#             # Concat後categorical會變objective,要轉回來
#             st_concat_astype = time.time()
#             obj_cols = df_combined.select_dtypes('object').drop(['customer_id'],axis=1).columns
#             df_combined[obj_cols.tolist()] = df_combined[obj_cols.tolist()].astype('category')
#             print(f'[build_set] 富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#             #移除雙證重複
#             remove_dupli_id_time = time.time()
#             df_combined['status'] = df_combined.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#             print(f'[build_set] 原始雙證人數: {len(df_combined)}')
#             df_combined = df_combined[df_combined['status']==1]
#             print(f'[build_set] 移除重複後雙證人數: {len(df_combined)}')
#             print(f'[build_set] 富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')


#             # 資料: [select_feature_set]
#             st_concat = time.time()
#             df_combined_redeuced = pd.concat([df_combined_redeuced_fubon,df_combined_redeuced_jihsun],axis = 0)
#             print(f'[select_feature_set] 富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#             # Concat後categorical會變objective,要轉回來
#             st_concat_astype = time.time()
#             obj_cols = df_combined_redeuced.select_dtypes('object').drop(['customer_id'],axis=1).columns
#             df_combined_redeuced[obj_cols.tolist()] = df_combined_redeuced[obj_cols.tolist()].astype('category')
#             print(f'[select_feature_set] 富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#             #移除雙證重複
#             remove_dupli_id_time = time.time()
#             df_combined_redeuced['status'] = df_combined_redeuced.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#             print(f'[select_feature_set] 原始雙證人數: {len(df_combined_redeuced)}')
#             df_combined_redeuced = df_combined_redeuced[df_combined_redeuced['status']==1]
#             print(f'[select_feature_set] 移除重複後雙證人數: {len(df_combined_redeuced)}')
#             print(f'[select_feature_set] 富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')

#             # 以較小資料集，建模選取重要特徵
#             X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
#             feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
#                                               project_name, mother, target, new_edition, algorithm)
#             cols_top100 = feat_imp.loc[:100]['feature'].tolist()

#             # 釋出空間
#             del X_train_rdc
#             del y_train_rdc
#             del X_test_rdc
#             del y_test_rdc
#             del df_combined_redeuced_fubon
#             del df_combined_redeuced_jihsun
#             del df_combined_redeuced
#             gc.collect()

#             # 以較大資料集，正式建模
#             based_col = ['customer_id', 'yyyymm', 'y']
#             X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top100],
#                                                           train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
#             model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top100, this_file_path,
#                                                                            project_name, mother, target, bins, new_edition, max_depth = max_depth_b,
#                                                                            scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
#             print('cols_impt_top100 : ')
#             display(model_imp)
#             print('test_level_df: ')
#             display(test_level_df)

#             # 以最新年月進行回測並產出效度
#             backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#             print(f'vali_level_df : ')
#             display(vali_level_df)
#             print(f'vali_predict_df.head() : ')
#             display(vali_predict_df.head())
#             print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
#             print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
#             print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')

#             # 以詳細切分方法進行最新年月回測並產出效度
#             backtest_auc_dt, vali_level_df_dt, vali_predict_df_dt, df_pred_and_features_dt = \
#             back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]],
#                       bins = [0,100,200,300,400,500,600,700,800,900,
#                               1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,
#                               11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,
#                               30000,40000,50000,60000,70000,80000,90000,100000,
#                               120000,140000,160000,180000,200000,
#                               1000000])
#             print(f'vali_level_df_dt : ')
#             display(vali_level_df_dt)
#             print(f'vali_predict_df_dt.head() : ')
#             display(vali_predict_df_dt.head())
#             print(f'vali_predivali_predict_df_dtct_df.shape :  {vali_predict_df_dt.shape}\n')
#             print(f'df_pred_and_features_dt.columns : \n {df_pred_and_features_dt.columns}\n')
#             print(f'df_pred_and_features_dt.shape : \n {df_pred_and_features_dt.shape}\n')

#             # 寫出PICKLE回測等級
#             pickle.dump(vali_predict_df_dt, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_detail' + project_name +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

#             # retrain_log
#             log_retrain = retrain_log_v230726(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                               train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
#                                               vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt)
#             print(f'retrain_log :')
#             display(log_retrain)

#             # model_log
#             log_model = model_log_v230410(False, project_name, mother, target, algorithm, bins,new_edition, table_name = 'mlops_model_log_double')
#             print(f'model_log :')
#             display(log_model)



#             log_retrain_v230410 = retrain_log_v230726(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                       vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
#                                                       vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt)
#             log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition,
#                                                   table_name = 'mlops_model_log_double')

#             ## If retrain_auc < 0.7 then print

#             if test_auc < 0.7:
#                 print('{}retrain_auc < 0.7'.format(mother))

#             print('{} 花了 {} min'.format(mother,(time.time()-totally_st_time)/60))


#             del X_train
#             del y_train
#             del X_test
#             del y_test
#             del df_combined_fubon
#             del df_combined_jihsun
#             del vali_predict_df
#             del df_pred_and_features
#             gc.collect()

#             print(f'{project_name}_{mother}_{new_edition} 執行完畢，先跳出迴圈!')
#             break
#         else:
#             print(f'{project_name}_{mother}_{new_edition} 先前已執行過了，執行下個母體!')


# In[7]:


# def retrain_mlops_np_and_p_double_opt_fillna(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, query_list, query_list_jihsun, frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b

#     feature_file_path_fubon = config.feature_file_path_fubon
#     feature_file_path_jihsun = config.feature_file_path_jihsun

#     from Sql_module import get_SQL_raw_data, write_data_to_SQL, send_table_to_sql
#     from Pretreatment import get_feature_by_SOP,get_feature_by_SOP_jihsun,combine_multi_year_df, if_to_large_down_sampling, get_papu_and_y_local
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v230726, model_log_v230410, retrain_log_v230314, get_next_model_version, whether_done_next_version
#     from Model import get_backtest_df_pvalue_IV
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')
#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):

#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         query_function = query_list[index]
#         query_function_jihsun = query_list_jihsun[index]
#         print(f'bin = {bins}')
#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'mlops_ref_info_double')
#         if not whether_done_next_version(this_file_path, project_name, mother, new_edition, target, algorithm,
#                                         db_table_rt='mlops_retrain_log_double', db_table_md='mlops_model_log_double',
#                                         account=config.account_yt, pwd=config.pwd_yt):
#             time.sleep(15)
#             print(f'do_ym_list = {do_ym_list}')

#             #########################################################################################################
#             # 富邦DF
#             ############################################################################################################
#             print('!!!!開始抓取富邦資料')
#             # 先抓一次
#             fubon_ft_fold = this_file_path.replace('審核通過模型_雙證','審核通過模型')
#             papu_and_y_fubon = get_papu_and_y_local(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                    writing_popu_path = fubon_ft_fold, mother = mother)
#             limit_times = 1
#             try_times = 1
#             # 重複抓
#             while set(do_ym_list)!=set(papu_and_y_fubon['yyyymm'].astype('str')) and try_times <= limit_times:
#                 papu_and_y_fubon = get_papu_and_y_local(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                        writing_popu_path = fubon_ft_fold, mother = mother)
#                 try_times = try_times+1

#             print('papu_and_y_fubon :')
#             display(papu_and_y_fubon.head())
#             # 排除例外(例如:舊戶)
#             papu_and_y_fubon = papu_and_y_fubon[papu_and_y_fubon[papulation_colname].isin(papulation_train_value)]
#             papu_and_y_fubon.drop([papulation_colname], axis=1, inplace=True)
#             # 是否存特徵, 若要存特徵則存在this_file_path
#             writing_path = None
#             if write_feature_Y_N:
#                 writing_path = this_file_path.replace('審核通過模型_雙證','審核通過模型')
#             # 將母體與特徵左右拼接
#             concat_df_outcome_fubon = get_feature_by_SOP(do_ym_list, mother, writing_path, drop_key_word, papu_and_y_fubon,
#                                                          feature_file_path = feature_file_path_fubon, just_for_check=False, Fill_zero=True)
#             # 每個年月最多50萬筆
#             concat_df_outcome_fubon = if_to_large_down_sampling(do_ym_list, concat_df_outcome_fubon, limit_size=limit_size_build)
#             # 將各年月的DF上下拼接
#             df_combined_fubon = combine_multi_year_df(do_ym_list, concat_df_outcome_fubon)
#             print(f'df_combined_fubon.shape = {df_combined_fubon.shape}')

# #             # FILLNA df_combined
# #             feature_list = list(df_combined_fubon.loc[:,~df_combined_fubon.columns.isin(['customer_id','yyyymm','y'])].select_dtypes(exclude = ['category','object']))
# #             for fl in feature_list:
# #                 na_num = sum(df_combined_fubon[fl].isna())
# #                 if na_num>0:
# #                     df_combined_fubon[fl] = df_combined_fubon[fl].fillna(0)
# #                     na_num = sum(df_combined_fubon[fl].isna())
# #                     if na_num>0:
# #                         print(f'[df_combined_fubon] 補NA後特徵{fl}變為{na_num}個NA')


#             # 額外縮減至每個年月最多10萬筆(用於select_feature)
#             concat_df_outcome_redeuced_fubon = if_to_large_down_sampling(do_ym_list, concat_df_outcome_fubon, limit_size=limit_size_select)
#             df_combined_redeuced_fubon = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced_fubon)
#             print(f'df_combined_redeuced_fubon.shape = {df_combined_redeuced_fubon.shape}')

# #             # FILLNA df_combined_redeuced
# #             feature_list = list(df_combined_redeuced_fubon.loc[:,~df_combined_redeuced_fubon.columns.isin(['customer_id','yyyymm','y'])].select_dtypes(exclude = ['category','object']))
# #             for fl in feature_list:
# #                 na_num = sum(df_combined_redeuced_fubon[fl].isna())
# #                 if na_num>0:
# #                     df_combined_redeuced_fubon[fl] = df_combined_redeuced_fubon[fl].fillna(0)
# #                     na_num = sum(df_combined_redeuced_fubon[fl].isna())
# #                     if na_num>0:
# #                         print(f'[df_combined_redeuced_fubon] 補NA後特徵{fl}變為{na_num}個NA')


#             ###############################################################################################################
#             # 日盛DF
#             ############################################################################################################
#             print('!!!!開始抓取日盛資料')
#             # 先抓一次
#             fubon_ft_fold = this_file_path.replace('審核通過模型_雙證','審核通過模型_jihsun')
#             papu_and_y_jihsun = get_papu_and_y_local(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                    writing_popu_path = fubon_ft_fold, mother = mother)
#             limit_times = 1
#             try_times = 1
#             # 重複抓
#             while set(do_ym_list)!=set(papu_and_y_jihsun['yyyymm'].astype('str')) and try_times <= limit_times:
#                 papu_and_y_jihsun = get_papu_and_y_local(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                        writing_popu_path = fubon_ft_fold, mother = mother)
#                 try_times = try_times+1

#             print('papu_and_y_jihsun :')
#             display(papu_and_y_jihsun.head())
#             # 排除例外(例如:舊戶)
#             papu_and_y_jihsun = papu_and_y_jihsun[papu_and_y_jihsun[papulation_colname].isin(papulation_train_value)]
#             papu_and_y_jihsun.drop([papulation_colname], axis=1, inplace=True)
#             # 是否存特徵, 若要存特徵則存在this_file_path
#             writing_path = None
#             if write_feature_Y_N:
#                 writing_path = this_file_path.replace('審核通過模型_雙證','審核通過模型_jihsun')

#             # 將母體與特徵左右拼接
#             concat_df_outcome_jihsun = get_feature_by_SOP_jihsun(do_ym_list, mother, writing_path, drop_key_word, papu_and_y_jihsun,
#                                                                 feature_file_path = feature_file_path_jihsun, just_for_check=False, Fill_zero=True)
#             # 每個年月最多50萬筆
#             concat_df_outcome_jihsun = if_to_large_down_sampling(do_ym_list, concat_df_outcome_jihsun, limit_size=limit_size_build)
#             # 將各年月的DF上下拼接
#             df_combined_jihsun = combine_multi_year_df(do_ym_list, concat_df_outcome_jihsun)
#             print(f'df_combined_jihsun.shape = {df_combined_jihsun.shape}')

# #             # FILLNA df_combined
# #             feature_list = list(df_combined_jihsun.loc[:,~df_combined_jihsun.columns.isin(['customer_id','yyyymm','y'])].select_dtypes(exclude = ['category','object']))
# #             for fl in feature_list:
# #                 na_num = sum(df_combined_jihsun[fl].isna())
# #                 if na_num>0:
# #                     df_combined_jihsun[fl] = df_combined_jihsun[fl].fillna(0)
# #                     na_num = sum(df_combined_jihsun[fl].isna())
# #                     if na_num>0:
# #                         print(f'[df_combined_jihsun] 補NA後特徵{fl}變為{na_num}個NA')

#             # 額外縮減至每個年月最多10萬筆(用於select_feature)
#             concat_df_outcome_redeuced_jihsun = if_to_large_down_sampling(do_ym_list, concat_df_outcome_jihsun, limit_size=limit_size_select)
#             df_combined_redeuced_jihsun = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced_jihsun)
#             print(f'df_combined_redeuced_jihsun.shape = {df_combined_redeuced_jihsun.shape}')

# #             # FILLNA df_combined_redeuced
# #             feature_list = list(df_combined_redeuced_jihsun.loc[:,~df_combined_redeuced_jihsun.columns.isin(['customer_id','yyyymm','y'])].select_dtypes(exclude = ['category','object']))
# #             for fl in feature_list:
# #                 na_num = sum(df_combined_redeuced_jihsun[fl].isna())
# #                 if na_num>0:
# #                     df_combined_redeuced_jihsun[fl] = df_combined_redeuced_jihsun[fl].fillna(0)
# #                     na_num = sum(df_combined_redeuced_jihsun[fl].isna())
# #                     if na_num>0:
# #                         print(f'[df_combined_redeuced_jihsun] 補NA後特徵{fl}變為{na_num}個NA')

#             ###############################################################################################################
#             # 開始合併富證日盛客戶
#             ##############################################################################################################
#             print('!!!!開始合併富邦日盛資料')
#             # 資料: [build_set]
#             st_concat = time.time()
#             df_combined = pd.concat([df_combined_fubon,df_combined_jihsun],axis = 0)
#             print(f'[build_set] 富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#             # Concat後categorical會變objective,要轉回來
#             st_concat_astype = time.time()
#             obj_cols = df_combined.select_dtypes('object').drop(['customer_id'],axis=1).columns
#             df_combined[obj_cols.tolist()] = df_combined[obj_cols.tolist()].astype('category')
#             print(f'[build_set] 富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#             #移除雙證重複
#             remove_dupli_id_time = time.time()
#             df_combined['status'] = df_combined.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#             print(f'[build_set] 原始雙證人數: {len(df_combined)}')
#             df_combined = df_combined[df_combined['status']==1]
#             print(f'[build_set] 移除重複後雙證人數: {len(df_combined)}')
#             print(f'[build_set] 富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')

#             # 資料: [select_feature_set]
#             st_concat = time.time()
#             df_combined_redeuced = pd.concat([df_combined_redeuced_fubon,df_combined_redeuced_jihsun],axis = 0)
#             print(f'[select_feature_set] 富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#             # Concat後categorical會變objective,要轉回來
#             st_concat_astype = time.time()
#             obj_cols = df_combined_redeuced.select_dtypes('object').drop(['customer_id'],axis=1).columns
#             df_combined_redeuced[obj_cols.tolist()] = df_combined_redeuced[obj_cols.tolist()].astype('category')
#             print(f'[select_feature_set] 富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#             #移除雙證重複
#             remove_dupli_id_time = time.time()
#             df_combined_redeuced['status'] = df_combined_redeuced.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#             print(f'[select_feature_set] 原始雙證人數: {len(df_combined_redeuced)}')
#             df_combined_redeuced = df_combined_redeuced[df_combined_redeuced['status']==1]
#             print(f'[select_feature_set] 移除重複後雙證人數: {len(df_combined_redeuced)}')
#             print(f'[select_feature_set] 富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')

#             # 以較小資料集，建模選取重要特徵
#             X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
#             feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
#                                               project_name, mother, target, new_edition, algorithm)
#             cols_top100 = feat_imp.loc[:100]['feature'].tolist()

#             # 釋出空間
#             del X_train_rdc
#             del y_train_rdc
#             del X_test_rdc
#             del y_test_rdc
#             del df_combined_redeuced_fubon
#             del df_combined_redeuced_jihsun
#             del df_combined_redeuced
#             del concat_df_outcome_redeuced_fubon
#             del concat_df_outcome_redeuced_jihsun
#             gc.collect()

#             # 以較大資料集，正式建模
#             based_col = ['customer_id', 'yyyymm', 'y']
#             X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top100],
#                                                           train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
#             model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top100, this_file_path,
#                                                                            project_name, mother, target, bins, new_edition, max_depth = max_depth_b,
#                                                                            scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
#             print('cols_impt_top100 : ')
#             display(model_imp)
#             print('test_level_df: ')
#             display(test_level_df)

#             # 以最新年月進行回測並產出效度
#             backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#             print(f'vali_level_df : ')
#             display(vali_level_df)
#             print(f'vali_predict_df.head() : ')
#             display(vali_predict_df.head())
#             print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
#             print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
#             print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')

#             # 以詳細切分方法進行最新年月回測並產出效度
#             backtest_auc_dt, vali_level_df_dt, vali_predict_df_dt, df_pred_and_features_dt = \
#             back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]],
#                       bins = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,
#                               11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,
#                               30000,40000,50000,60000,70000,80000,90000,100000,
#                               120000,140000,160000,180000,200000,
#                               1000000])
#             print(f'vali_level_df_dt : ')
#             display(vali_level_df_dt)
#             print(f'vali_predict_df_dt.head() : ')
#             display(vali_predict_df_dt.head())
#             print(f'vali_predivali_predict_df_dtct_df.shape :  {vali_predict_df_dt.shape}\n')
#             print(f'df_pred_and_features_dt.columns : \n {df_pred_and_features_dt.columns}\n')
#             print(f'df_pred_and_features_dt.shape : \n {df_pred_and_features_dt.shape}\n')

#             # 寫出PICKLE回測等級
#             pickle.dump(vali_predict_df_dt, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_detail' + project_name +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

#             # retrain_log
#             log_retrain = retrain_log_v230726(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                               train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
#                                               vali_level_df_detail=vali_level_df_dt )
#             print(f'retrain_log :')
#             display(log_retrain)

#             # model_log
#             log_model = model_log_v230410(False, project_name, mother, target, algorithm, bins,new_edition, table_name = 'mlops_model_log_double')
#             print(f'model_log :')
#             display(log_model)



#             log_retrain_v230410 = retrain_log_v230726(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                       vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
#                                                       vali_level_df_detail=vali_level_df_dt )
#             log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition,
#                                                   table_name = 'mlops_model_log_double')

#             ## If retrain_auc < 0.7 then print

#             if test_auc < 0.7:
#                 print('{}retrain_auc < 0.7'.format(mother))

#             print('{} 花了 {} min'.format(mother,(time.time()-totally_st_time)/60))


#             ############# START ##########################################
#             # 額外針對回測月份計算p-value及IV值
#             #
#             calcu_time = time.time()
#             calcuate_top_n = 20



#             pvalue_col = ['y'] + list(model_imp['feature'])[0:calcuate_top_n]
#             backtest_df_pvalue_IV = get_backtest_df_pvalue_IV(project_name, target, mother, df_combined, pvalue_col)

#             backtest_df_pvalue_IV = backtest_df_pvalue_IV[['prod', 'target', 'population', 'test_period', 'feature', 'feature_chinese',
#                                    'dtype', 'p_value', 'IV', 'error_msg']]
#              # 小數點126位限制轉換
#             lower_limit = 1E-126
#             def clip_float_values(value):
#                 if np.isfinite(value):
#                     if abs(value) < lower_limit and value != 0:
#                         return np.sign(value) * lower_limit
#                 return value

#             backtest_df_pvalue_IV['p_value'] = backtest_df_pvalue_IV['p_value'].apply(clip_float_values)

#             send_table_to_sql(backtest_df_pvalue_IV, 'impt_feature_statistic', account=config.account, pwd=config.pwd)
#             print('{} 計算p-value&IV值花了 {} min'.format(mother,(time.time()-calcu_time)/60))
#             ######### END ###################################################

#             del concat_df_outcome_fubon
#             del concat_df_outcome_jihsun
#             del X_train
#             del y_train
#             del X_test
#             del y_test
#             del df_combined_fubon
#             del df_combined_jihsun
#             del papu_and_y_fubon
#             del papu_and_y_jihsun
#             del vali_predict_df
#             del df_pred_and_features
#             gc.collect()

#             print(f'{project_name}_{mother}_{new_edition} 執行完畢，先跳出迴圈!')
#             break
#         else:
#             print(f'{project_name}_{mother}_{new_edition} 先前已執行過了，執行下個母體!')


# In[8]:


# def retrain_mlops_np_and_p_double(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, query_list, query_list_jihsun , frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b


#     from Sql_module import get_SQL_raw_data, write_data_to_SQL
#     from Pretreatment import get_feature_by_SOP,get_feature_by_SOP_jihsun,combine_multi_year_df, if_to_large_down_sampling, get_papu_and_y_local
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v230726, model_log_v230410, retrain_log_v230314, get_next_model_version, whether_done_next_version
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')
#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):

#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         query_function = query_list[index]
#         query_function_jihsun = query_list_jihsun[index]
#         print(f'bin = {bins}')
#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'mlops_ref_info_double')

#         if not whether_done_next_version(this_file_path, project_name, mother, new_edition, target, algorithm,
#                                         db_table_rt='mlops_retrain_log_double', db_table_md='mlops_model_log_double',
#                                         account=config.account_yt, pwd=config.pwd_yt):
#             time.sleep(15)
#             print(f'do_ym_list = {do_ym_list}')

#             #########################################################################################################
#             # 富邦DF
#             ############################################################################################################
#             print('!!!!開始抓取富邦資料')
#             # 先抓一次
#             fubon_ft_fold = this_file_path.replace('審核通過模型_雙證','審核通過模型')
#             papu_and_y_fubon = get_papu_and_y_local(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                    writing_popu_path = fubon_ft_fold, mother = mother)
#             limit_times = 1
#             try_times = 1
#             # 重複抓
#             while set(do_ym_list)!=set(papu_and_y_fubon['yyyymm'].astype('str')) and try_times <= limit_times:
#                 papu_and_y_fubon = get_papu_and_y_local(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                        writing_popu_path = fubon_ft_fold, mother = mother)
#                 try_times = try_times+1

#             print('papu_and_y_fubon :')
#             display(papu_and_y_fubon.head())
#             # 排除例外(例如:舊戶)
#             papu_and_y_fubon = papu_and_y_fubon[papu_and_y_fubon[papulation_colname].isin(papulation_train_value)]
#             papu_and_y_fubon.drop([papulation_colname], axis=1, inplace=True)
#             # 是否存特徵, 若要存特徵則存在this_file_path
#             writing_path = None
#             if write_feature_Y_N:
#                 writing_path = this_file_path.replace('審核通過模型_雙證','審核通過模型')
#             # 將母體與特徵左右拼接
#             concat_df_outcome_fubon = get_feature_by_SOP(do_ym_list, mother, writing_path, drop_key_word, papu_and_y_fubon)
#             # 每個年月最多50萬筆
#             concat_df_outcome_fubon = if_to_large_down_sampling(do_ym_list, concat_df_outcome_fubon, limit_size=limit_size_build)
#             # 將各年月的DF上下拼接
#             df_combined_fubon = combine_multi_year_df(do_ym_list, concat_df_outcome_fubon)
#             print(f'df_combined_fubon.shape = {df_combined_fubon.shape}')
#             # 額外縮減至每個年月最多10萬筆(用於select_feature)
#             concat_df_outcome_redeuced_fubon = if_to_large_down_sampling(do_ym_list, concat_df_outcome_fubon, limit_size=limit_size_select)
#             df_combined_redeuced_fubon = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced_fubon)
#             print(f'df_combined_redeuced_fubon.shape = {df_combined_redeuced_fubon.shape}')

#             # 清出空間
#             del papu_and_y_fubon
#             del concat_df_outcome_fubon
#             del concat_df_outcome_redeuced_fubon
#             gc.collect()
#             ###############################################################################################################
#             # 日盛DF
#             ############################################################################################################
#             print('!!!!開始抓取日盛資料')
#             # 先抓一次
#             fubon_ft_fold = this_file_path.replace('審核通過模型_雙證','審核通過模型_jihsun')
#             papu_and_y_jihsun = get_papu_and_y_local(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                    writing_popu_path = fubon_ft_fold, mother = mother)
#             limit_times = 1
#             try_times = 1
#             # 重複抓
#             while set(do_ym_list)!=set(papu_and_y_jihsun['yyyymm'].astype('str')) and try_times <= limit_times:
#                 papu_and_y_jihsun = get_papu_and_y_local(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                        writing_popu_path = fubon_ft_fold, mother = mother)
#                 try_times = try_times+1

#             print('papu_and_y_jihsun :')
#             display(papu_and_y_jihsun.head())
#             # 排除例外(例如:舊戶)
#             papu_and_y_jihsun = papu_and_y_jihsun[papu_and_y_jihsun[papulation_colname].isin(papulation_train_value)]
#             papu_and_y_jihsun.drop([papulation_colname], axis=1, inplace=True)
#             # 是否存特徵, 若要存特徵則存在this_file_path
#             writing_path = None
#             if write_feature_Y_N:
#                 writing_path = this_file_path.replace('審核通過模型_雙證','審核通過模型_jihsun')

#             # 將母體與特徵左右拼接
#             concat_df_outcome_jihsun = get_feature_by_SOP_jihsun(do_ym_list, mother, writing_path, drop_key_word, papu_and_y_jihsun)
#             # 每個年月最多50萬筆
#             concat_df_outcome_jihsun = if_to_large_down_sampling(do_ym_list, concat_df_outcome_jihsun, limit_size=limit_size_build)
#             # 將各年月的DF上下拼接
#             df_combined_jihsun = combine_multi_year_df(do_ym_list, concat_df_outcome_jihsun)
#             print(f'df_combined_jihsun.shape = {df_combined_jihsun.shape}')
#             # 額外縮減至每個年月最多10萬筆(用於select_feature)
#             concat_df_outcome_redeuced_jihsun = if_to_large_down_sampling(do_ym_list, concat_df_outcome_jihsun, limit_size=limit_size_select)
#             df_combined_redeuced_jihsun = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced_jihsun)
#             print(f'df_combined_redeuced_jihsun.shape = {df_combined_redeuced_jihsun.shape}')

#             # 清出空間
#             del papu_and_y_jihsun
#             del concat_df_outcome_jihsun
#             del concat_df_outcome_redeuced_jihsun
#             gc.collect()
#             ###############################################################################################################
#             # 開始合併富證日盛客戶
#             ##############################################################################################################
#             print('!!!!開始合併富邦日盛資料')
#             # 資料: [build_set]
#             st_concat = time.time()
#             df_combined = pd.concat([df_combined_fubon,df_combined_jihsun],axis = 0)
#             print(f'[build_set] 富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#             # Concat後categorical會變objective,要轉回來
#             st_concat_astype = time.time()
#             obj_cols = df_combined.select_dtypes('object').drop(['customer_id'],axis=1).columns
#             df_combined[obj_cols.tolist()] = df_combined[obj_cols.tolist()].astype('category')
#             print(f'[build_set] 富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#             #移除雙證重複
#             remove_dupli_id_time = time.time()
#             df_combined['status'] = df_combined.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#             print(f'[build_set] 原始雙證人數: {len(df_combined)}')
#             df_combined = df_combined[df_combined['status']==1]
#             print(f'[build_set] 移除重複後雙證人數: {len(df_combined)}')
#             print(f'[build_set] 富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')

#             # 資料: [select_feature_set]
#             st_concat = time.time()
#             df_combined_redeuced = pd.concat([df_combined_redeuced_fubon,df_combined_redeuced_jihsun],axis = 0)
#             print(f'[select_feature_set] 富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#             # Concat後categorical會變objective,要轉回來
#             st_concat_astype = time.time()
#             obj_cols = df_combined_redeuced.select_dtypes('object').drop(['customer_id'],axis=1).columns
#             df_combined_redeuced[obj_cols.tolist()] = df_combined_redeuced[obj_cols.tolist()].astype('category')
#             print(f'[select_feature_set] 富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#             #移除雙證重複
#             remove_dupli_id_time = time.time()
#             df_combined_redeuced['status'] = df_combined_redeuced.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#             print(f'[select_feature_set] 原始雙證人數: {len(df_combined_redeuced)}')
#             df_combined_redeuced = df_combined_redeuced[df_combined_redeuced['status']==1]
#             print(f'[select_feature_set] 移除重複後雙證人數: {len(df_combined_redeuced)}')
#             print(f'[select_feature_set] 富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')

#             # 以較小資料集，建模選取重要特徵
#             X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
#             feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
#                                               project_name, mother, target, new_edition, algorithm)
#             cols_top100 = feat_imp.loc[:100]['feature'].tolist()

#             # 釋出空間
#             del X_train_rdc
#             del y_train_rdc
#             del X_test_rdc
#             del y_test_rdc
#             del df_combined_redeuced_fubon
#             del df_combined_redeuced_jihsun
#             del df_combined_redeuced
#             gc.collect()

#             # 以較大資料集，正式建模
#             based_col = ['customer_id', 'yyyymm', 'y']
#             X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top100],
#                                                           train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
#             model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top100, this_file_path,
#                                                                            project_name, mother, target, bins, new_edition, max_depth = max_depth_b,
#                                                                            scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
#             print('cols_impt_top100 : ')
#             display(model_imp)
#             print('test_level_df: ')
#             display(test_level_df)

#             # 以最新年月進行回測並產出效度
#             backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#             print(f'vali_level_df : ')
#             display(vali_level_df)
#             print(f'vali_predict_df.head() : ')
#             display(vali_predict_df.head())
#             print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
#             print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
#             print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')

#             # 以詳細切分方法進行最新年月回測並產出效度
#             backtest_auc_dt, vali_level_df_dt, vali_predict_df_dt, df_pred_and_features_dt = \
#             back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]],
#                       bins = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,
#                               11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,
#                               30000,40000,50000,60000,70000,80000,90000,100000,
#                               120000,140000,160000,180000,200000,
#                               1000000])
#             print(f'vali_level_df_dt : ')
#             display(vali_level_df_dt)
#             print(f'vali_predict_df_dt.head() : ')
#             display(vali_predict_df_dt.head())
#             print(f'vali_predivali_predict_df_dtct_df.shape :  {vali_predict_df_dt.shape}\n')
#             print(f'df_pred_and_features_dt.columns : \n {df_pred_and_features_dt.columns}\n')
#             print(f'df_pred_and_features_dt.shape : \n {df_pred_and_features_dt.shape}\n')

#             # 寫出PICKLE回測等級
#             pickle.dump(vali_predict_df_dt, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_detail' + project_name +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

#             # retrain_log
#             log_retrain = retrain_log_v230726(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                               train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
#                                               vali_level_df_detail=vali_level_df_dt )
#             print(f'retrain_log :')
#             display(log_retrain)

#             # model_log
#             log_model = model_log_v230410(False, project_name, mother, target, algorithm, bins,new_edition, table_name = 'mlops_model_log_double')
#             print(f'model_log :')
#             display(log_model)



#             log_retrain_v230410 = retrain_log_v230726(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                       vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_double',
#                                                       vali_level_df_detail=vali_level_df_dt )
#             log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition,
#                                                   table_name = 'mlops_model_log_double')

#             ## If retrain_auc < 0.7 then print

#             if test_auc < 0.7:
#                 print('{}retrain_auc < 0.7'.format(mother))

#             print('{} 花了 {} min'.format(mother,(time.time()-totally_st_time)/60))



#             del X_train
#             del y_train
#             del X_test
#             del y_test
#             del df_combined_fubon
#             del df_combined_jihsun
#             del vali_predict_df
#             del df_pred_and_features
#             gc.collect()

#             print(f'{project_name}_{mother}_{new_edition} 執行完畢，先跳出迴圈!')
#             break
#         else:
#             print(f'{project_name}_{mother}_{new_edition} 先前已執行過了，執行下個母體!')


# In[9]:


# def retrain_mlops_np_and_p(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, query_list , frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b

#     # 若要存特徵則存在檔案位置
#     writing_path = None
#     if write_feature_Y_N:
#         writing_path = this_file_path

#     from Sql_module import get_SQL_raw_data, write_data_to_SQL
#     from Pretreatment import get_feature_by_SOP,combine_multi_year_df, if_to_large_down_sampling, get_papu_and_y
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v230726, model_log_v230410, retrain_log_v230314, get_next_model_version, whether_done_next_version
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')
#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):

#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         query_function = query_list[index]
#         print(f'bin = {bins}')
#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail,
#                                              table_name = 'mlops_ref_info', account=config.account_yichieh, pwd=config.pwd_yichieh)
#         if not whether_done_next_version(this_file_path, project_name, mother, new_edition, target, algorithm,
#                                         db_table_rt='mlops_retrain_log', db_table_md='mlops_model_log',
#                                         account=config.account_yt, pwd=config.pwd_yt):
#             time.sleep(30)
#             print(f'do_ym_list = {do_ym_list}')
#             # 抓取每個年月的母體
#             papu_and_y = get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                 writing_popu_path=this_file_path, mother=mother)
#             limit_times = 1
#             try_times = 1
#             while set(do_ym_list)!=set(papu_and_y['yyyymm'].astype('str')) and try_times <= limit_times:
#                 papu_and_y = get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                             writing_popu_path=this_file_path, mother=mother)
#                 try_times = try_times+1

#     #         # 抓取每個年月的母體
#     #         papu_and_y = pd.DataFrame()
#     #         chage_acct = True
#     #         for ym in do_ym_list:
#     #             date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#     #             next_month = monthdelta(date, 1)
#     #             print(next_month)
#     #             query = query_function(next_month)
#     #             # 判斷query_function是回傳
#     #             if type(query) == str:
#     #                 if chage_acct:
#     #                     df_temp = get_SQL_raw_data(query, account=config.account_yichieh, pwd=config.pwd_yichieh)
#     #                 else:
#     #                     df_temp = get_SQL_raw_data(query)
#     #                 papu_and_y = papu_and_y.append(df_temp)
#     #                 time.sleep(30)
#     #                 chage_acct = not chage_acct
#     #             else:
#     #                 papu_and_y = query

#             print('papu_and_y :')
#             display(papu_and_y.head())
#             # 排除例外(例如:舊戶)
#             papu_and_y = papu_and_y[papu_and_y[papulation_colname].isin(papulation_train_value)]
#             papu_and_y.drop([papulation_colname], axis=1, inplace=True)


#             # 將母體與特徵左右拼接
#             concat_df_outcome = get_feature_by_SOP(do_ym_list, mother, writing_path, drop_key_word, papu_and_y)
#             # 每個年月最多50萬筆
#             concat_df_outcome = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_build)

#             # 將各年月的DF上下拼接
#             df_combined = combine_multi_year_df(do_ym_list, concat_df_outcome)
#             print(f'df_combined.shape = {df_combined.shape}')
#             # 額外縮減至每個年月最多10萬筆(用於select_feature)
#             concat_df_outcome_redeuced = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_select)
#             df_combined_redeuced = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced)
#             print(f'df_combined_redeuced.shape = {df_combined_redeuced.shape}')

#             # 以較小資料集，建模選取重要特徵
#             X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
#             feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
#                                               project_name, mother, target, new_edition, algorithm)
#             cols_top100 = feat_imp.loc[:100]['feature'].tolist()

#             # 釋出空間
#             del X_train_rdc
#             del y_train_rdc
#             del X_test_rdc
#             del y_test_rdc
#             del df_combined_redeuced
#             del concat_df_outcome_redeuced
#             gc.collect()

#             # 以較大資料集，正式建模
#             based_col = ['customer_id', 'yyyymm', 'y']
#             X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top100],
#                                                           train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
#             model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top100, this_file_path,
#                                                                            project_name, mother, target, bins, new_edition, max_depth = max_depth_b,
#                                                                            scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
#             print('cols_impt_top100 : ')
#             display(model_imp)
#             print('test_level_df: ')
#             display(test_level_df)

#             # 以最新年月進行回測並產出效度
#             backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#             print(f'vali_level_df : ')
#             display(vali_level_df)
#             print(f'vali_predict_df.head() : ')
#             display(vali_predict_df.head())
#             print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
#             print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
#             print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')

#             # 以詳細切分方法進行最新年月回測並產出效度
#             backtest_auc_dt, vali_level_df_dt, vali_predict_df_dt, df_pred_and_features_dt = \
#             back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]],
#                       bins = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,
#                               11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,
#                               30000,40000,50000,60000,70000,80000,90000,100000,
#                               120000,140000,160000,180000,200000,
#                               1000000])
#             print(f'vali_level_df_dt : ')
#             display(vali_level_df_dt)
#             print(f'vali_predict_df_dt.head() : ')
#             display(vali_predict_df_dt.head())
#             print(f'vali_predivali_predict_df_dtct_df.shape :  {vali_predict_df_dt.shape}\n')
#             print(f'df_pred_and_features_dt.columns : \n {df_pred_and_features_dt.columns}\n')
#             print(f'df_pred_and_features_dt.shape : \n {df_pred_and_features_dt.shape}\n')

#             # 寫出PICKLE回測等級
#             pickle.dump(vali_predict_df_dt, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_detail' + project_name +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

#             # retrain_log
#             log_retrain = retrain_log_v230726(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                               train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log',
#                                               vali_level_df_detail=vali_level_df_dt )
#             print(f'retrain_log :')
#             display(log_retrain)

#             # model_log
#             log_model = model_log_v230410(False, project_name, mother, target, algorithm, bins,new_edition, table_name = 'mlops_model_log')
#             print(f'model_log :')
#             display(log_model)



#             log_retrain_v230410 = retrain_log_v230726(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                       vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log',
#                                                       vali_level_df_detail=vali_level_df_dt )
#             log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition,
#                                                   table_name = 'mlops_model_log')

#             ## If retrain_auc < 0.7 then print

#             if test_auc < 0.7:
#                 print('{}retrain_auc < 0.7'.format(mother))

#             print('{} 花了 {} min'.format(mother,(time.time()-totally_st_time)/60))


#             del concat_df_outcome
#             del X_train
#             del y_train
#             del X_test
#             del y_test
#             del df_combined
#             del papu_and_y
#             del vali_predict_df
#             del df_pred_and_features

#             print(f'{project_name}_{mother}_{new_edition} 執行完畢，先跳出迴圈!')
#             break
#         else:
#             print(f'{project_name}_{mother}_{new_edition} 先前已執行過了，執行下個母體!')



# In[10]:


# def retrain_mlops_np_and_p_fillna(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, query_list , frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b

#     # 若要存特徵則存在檔案位置
#     writing_path = None
#     if write_feature_Y_N:
#         writing_path = this_file_path

#     from Sql_module import get_SQL_raw_data, write_data_to_SQL
#     from Pretreatment import get_feature_by_SOP,combine_multi_year_df, if_to_large_down_sampling, get_papu_and_y
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v230726, model_log_v230410, retrain_log_v230314, get_next_model_version
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')
#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):
#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         query_function = query_list[index]
#         print(f'bin = {bins}')
#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail,
#                                              table_name = 'mlops_ref_info', account=config.account_yichieh, pwd=config.pwd_yichieh)
#         time.sleep(30)
#         print(f'do_ym_list = {do_ym_list}')
#         # 抓取每個年月的母體
#         papu_and_y = get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                             writing_popu_path=this_file_path, mother=mother)
#         limit_times = 1
#         try_times = 1
#         while set(do_ym_list)!=set(papu_and_y['yyyymm'].astype('str')) and try_times <= limit_times:
#             papu_and_y = get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                         writing_popu_path=this_file_path, mother=mother)
#             try_times = try_times+1

# #         # 抓取每個年月的母體
# #         papu_and_y = pd.DataFrame()
# #         chage_acct = True
# #         for ym in do_ym_list:
# #             date =  datetime.date(datetime.strptime(ym,'%Y%m'))
# #             next_month = monthdelta(date, 1)
# #             print(next_month)
# #             query = query_function(next_month)
# #             # 判斷query_function是回傳
# #             if type(query) == str:
# #                 if chage_acct:
# #                     df_temp = get_SQL_raw_data(query, account=config.account_yichieh, pwd=config.pwd_yichieh)
# #                 else:
# #                     df_temp = get_SQL_raw_data(query)
# #                 papu_and_y = papu_and_y.append(df_temp)
# #                 time.sleep(30)
# #                 chage_acct = not chage_acct
# #             else:
# #                 papu_and_y = query

#         print('papu_and_y :')
#         display(papu_and_y.head())
#         # 排除例外(例如:舊戶)
#         papu_and_y = papu_and_y[papu_and_y[papulation_colname].isin(papulation_train_value)]
#         papu_and_y.drop([papulation_colname], axis=1, inplace=True)


#         # 將母體與特徵左右拼接
#         concat_df_outcome = get_feature_by_SOP(do_ym_list, mother, writing_path, drop_key_word, papu_and_y)
#         # 每個年月最多50萬筆
#         concat_df_outcome = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_build)

#         # 將各年月的DF上下拼接
#         df_combined = combine_multi_year_df(do_ym_list, concat_df_outcome)

#         # FILLNA df_combined
#         feature_list = list(df_combined.loc[:,~df_combined.columns.isin(['customer_id','yyyymm','y'])].select_dtypes(exclude = ['category','object']))
#         df_combined[feature_list].fillna(0, inplace=True) #categorical用replace會變成objective

#         print(f'df_combined.shape = {df_combined.shape}')
#         # 額外縮減至每個年月最多10萬筆(用於select_feature)
#         concat_df_outcome_redeuced = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_select)
#         df_combined_redeuced = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced)

#         # FILLNA df_combined_redeuced
#         feature_list = list(df_combined_redeuced.loc[:,~df_combined_redeuced.columns.isin(['customer_id','yyyymm','y'])].select_dtypes(exclude = ['category','object']))
#         df_combined_redeuced[feature_list].fillna(0, inplace=True) #categorical用replace會變成objective

#         print(f'df_combined_redeuced.shape = {df_combined_redeuced.shape}')


#         # 以較小資料集，建模選取重要特徵
#         X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
#         feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
#                                           project_name, mother, target, new_edition, algorithm)
#         cols_top100 = feat_imp.loc[:100]['feature'].tolist()

#         # 釋出空間
#         del X_train_rdc
#         del y_train_rdc
#         del X_test_rdc
#         del y_test_rdc
#         del df_combined_redeuced
#         del concat_df_outcome_redeuced
#         gc.collect()

#         # 以較大資料集，正式建模
#         based_col = ['customer_id', 'yyyymm', 'y']
#         X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top100],
#                                                       train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
#         model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top100, this_file_path,
#                                                                        project_name, mother, target, bins, new_edition, max_depth = max_depth_b,
#                                                                        scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
#         print('cols_impt_top100 : ')
#         display(model_imp)
#         print('test_level_df: ')
#         display(test_level_df)

#         # 以最新年月進行回測並產出效度
#         backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#         print(f'vali_level_df : ')
#         display(vali_level_df)
#         print(f'vali_predict_df.head() : ')
#         display(vali_predict_df.head())
#         print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
#         print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
#         print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')
#         # 寫出PICKLE回測等級
#         pickle.dump(vali_predict_df, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_' + project_name +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

#         # retrain_log
#         log_retrain = retrain_log_v230726(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                           train_auc, test_auc, backtest_auc, new_edition, table_name = 'opt_2_fill_retrain_log')
#         print(f'retrain_log :')
#         display(log_retrain)

#         # model_log
#         log_model = model_log_v230410(False, project_name, mother, target, algorithm, bins,new_edition, table_name = 'opt_2_fill_model_log')
#         print(f'model_log :')
#         display(log_model)



#         log_retrain_v230410 = retrain_log_v230726(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                   vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'opt_2_fill_retrain_log')
#         log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition,
#                                               table_name = 'opt_2_fill_model_log')

#         ## If retrain_auc < 0.7 then print

#         if test_auc < 0.7:
#             print('{}retrain_auc < 0.7'.format(mother))

#         print('{} 花了 {} min'.format(mother,(time.time()-totally_st_time)/60))

#         # 另外執行 bin = [0,10000,40000,80000,140000,10000000]
#         if bins != [0,10000,40000,80000,140000,10000000]:
#             bins = [0,10000,40000,80000,140000,10000000]
#             backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#             log_retrain_diff_bin = retrain_log_v230726(write_db_Y_N, project_name, mother, df_combined, X_train,y_train,
#                                                        test_level_df, vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_ori_bin')
#             log_model_diff_bin = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins,
#                                                    new_edition, table_name = 'mlops_model_log_ori_bin')
#             print(f'retrain_log :')
#             display(log_retrain_diff_bin)
#             print(f'model_log :')
#             display(log_model_diff_bin)

#         del concat_df_outcome
#         del X_train
#         del y_train
#         del X_test
#         del y_test
#         del df_combined
#         del papu_and_y
#         del vali_predict_df
#         del df_pred_and_features



# In[11]:


# def retrain_mlops_np_and_p_jihsun(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, query_list , frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b

#     # 若要存特徵則存在檔案位置
#     writing_path = None
#     if write_feature_Y_N:
#         writing_path = this_file_path

#     from Sql_module import get_SQL_raw_data, write_data_to_SQL
#     from Pretreatment import get_feature_by_SOP_jihsun,combine_multi_year_df, if_to_large_down_sampling, get_papu_and_y
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v230726, model_log_v230410, retrain_log_v230314, get_next_model_version, whether_done_next_version
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')
#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):
#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         query_function = query_list[index]
#         print(f'bin = {bins}')
#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail,
#                                              table_name = 'mlops_ref_info_jihsun', account=config.account_yichieh, pwd=config.pwd_yichieh)
#         if not whether_done_next_version(this_file_path, project_name, mother, new_edition, target, algorithm,
#                                          db_table_rt='mlops_retrain_log_jihsun', db_table_md='mlops_model_log_jihsun',
#                                          account=config.account_yt, pwd=config.pwd_yt):
#             time.sleep(30)
#             print(f'do_ym_list = {do_ym_list}')
#             # 抓取每個年月的母體
#             papu_and_y = get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                 writing_popu_path=this_file_path, mother=mother)
#             limit_times = 1
#             try_times = 1
#             while set(do_ym_list)!=set(papu_and_y['yyyymm'].astype('str')) and try_times <= limit_times:
#                 papu_and_y = get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                             writing_popu_path=this_file_path, mother=mother)
#                 try_times = try_times+1

#     #         # 抓取每個年月的母體
#     #         papu_and_y = pd.DataFrame()
#     #         for ym in do_ym_list:
#     #             date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#     #             next_month = monthdelta(date, 1)
#     #             print(next_month)
#     #             query = query_function(next_month)
#     #             df_temp = get_SQL_raw_data(query, account=config.account_yichieh, pwd=config.pwd_yichieh)
#     #             papu_and_y = papu_and_y.append(df_temp)
#     #             time.sleep(30)
#             print('papu_and_y :')
#             display(papu_and_y.head())

#             # 排除例外(例如:舊戶)
#             papu_and_y = papu_and_y[papu_and_y[papulation_colname].isin(papulation_train_value)]
#             papu_and_y.drop([papulation_colname], axis=1, inplace=True)


#             # 將母體與特徵左右拼接
#             concat_df_outcome = get_feature_by_SOP_jihsun(do_ym_list, mother, writing_path, drop_key_word, papu_and_y)
#             # 每個年月最多50萬筆
#             concat_df_outcome = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_build)

#             # 將各年月的DF上下拼接
#             df_combined = combine_multi_year_df(do_ym_list, concat_df_outcome)
#             print(f'df_combined.shape = {df_combined.shape}')
#             # 額外縮減至每個年月最多10萬筆(用於select_feature)
#             concat_df_outcome_redeuced = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_select)
#             df_combined_redeuced = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced)
#             print(f'df_combined_redeuced.shape = {df_combined_redeuced.shape}')

#             # 以較小資料集，建模選取重要特徵
#             X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
#             feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
#                                               project_name, mother, target, new_edition, algorithm)
#             cols_top100 = feat_imp.loc[:100]['feature'].tolist()

#             # 釋出空間
#             del X_train_rdc
#             del y_train_rdc
#             del X_test_rdc
#             del y_test_rdc
#             del df_combined_redeuced
#             del concat_df_outcome_redeuced
#             gc.collect()

#             # 以較大資料集，正式建模
#             based_col = ['customer_id', 'yyyymm', 'y']
#             X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top100],
#                                                           train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
#             model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top100, this_file_path,
#                                                                            project_name, mother, target, bins, new_edition, max_depth = max_depth_b,
#                                                                            scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
#             print('cols_impt_top100 : ')
#             display(model_imp)
#             print('test_level_df: ')
#             display(test_level_df)

#             # 以最新年月進行回測並產出效度
#             backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#             print(f'vali_level_df : ')
#             display(vali_level_df)
#             print(f'vali_predict_df.head() : ')
#             display(vali_predict_df.head())
#             print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
#             print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
#             print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')

#             # 以詳細切分方法進行最新年月回測並產出效度
#             backtest_auc_dt, vali_level_df_dt, vali_predict_df_dt, df_pred_and_features_dt = \
#             back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]],
#                       bins = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,
#                               11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,
#                               30000,40000,50000,60000,70000,80000,90000,100000,
#                               120000,140000,160000,180000,200000,
#                               1000000])
#             print(f'vali_level_df_dt : ')
#             display(vali_level_df_dt)
#             print(f'vali_predict_df_dt.head() : ')
#             display(vali_predict_df_dt.head())
#             print(f'vali_predivali_predict_df_dtct_df.shape :  {vali_predict_df_dt.shape}\n')
#             print(f'df_pred_and_features_dt.columns : \n {df_pred_and_features_dt.columns}\n')
#             print(f'df_pred_and_features_dt.shape : \n {df_pred_and_features_dt.shape}\n')

#             # 寫出PICKLE回測等級
#             pickle.dump(vali_predict_df_dt, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_detail' + project_name +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

#             # retrain_log
#             log_retrain = retrain_log_v230726(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                               train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_jihsun',
#                                               vali_level_df_detail=vali_level_df_dt )
#             print(f'retrain_log :')
#             display(log_retrain)

#             # model_log
#             log_model = model_log_v230410(False, project_name, mother, target, algorithm, bins,new_edition, table_name = 'mlops_model_log_jihsun')
#             print(f'model_log :')
#             display(log_model)



#             log_retrain_v230410 = retrain_log_v230726(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                       vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_jihsun',
#                                                       vali_level_df_detail=vali_level_df_dt )
#             log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition,
#                                                   table_name = 'mlops_model_log_jihsun')

#             ## If retrain_auc < 0.7 then print

#             if test_auc < 0.7:
#                 print('{}retrain_auc < 0.7'.format(mother))

#             print('{} 花了 {} min'.format(mother,(time.time()-totally_st_time)/60))


#             del concat_df_outcome
#             del X_train
#             del y_train
#             del X_test
#             del y_test
#             del df_combined
#             del papu_and_y
#             del vali_predict_df
#             del df_pred_and_features

#             print(f'{project_name}_{mother}_{new_edition} 執行完畢，先跳出迴圈!')
#             break
#         else:
#             print(f'{project_name}_{mother}_{new_edition} 先前已執行過了，執行下個母體!')


# In[12]:


# def retrain_mlops_np_and_p_optimal(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, query_list , frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b

#     # 若要存特徵則存在檔案位置
#     writing_path = None
#     if write_feature_Y_N:
#         writing_path = this_file_path

#     from Sql_module import get_SQL_raw_data, write_data_to_SQL
#     from Pretreatment import get_feature_by_SOP,combine_multi_year_df, if_to_large_down_sampling
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v230726, model_log_v230410, retrain_log_v230314, get_next_model_version
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')
#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):
#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         query_function = query_list[index]
#         print(f'bin = {bins}')
#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'opt_ref_info')
#         time.sleep(60)
#         print(f'do_ym_list = {do_ym_list}')
#         # 抓取每個年月的母體
#         papu_and_y = pd.DataFrame()
#         chage_acct = True
#         for ym in do_ym_list:
#             date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#             next_month = monthdelta(date, 1)
#             print(next_month)
#             query = query_function(next_month)
#             if chage_acct:
#                 df_temp = get_SQL_raw_data(query, account=config.account_yichieh, pwd=config.pwd_yichieh)
#             else:
#                 df_temp = get_SQL_raw_data(query)
#             papu_and_y = papu_and_y.append(df_temp)
#             time.sleep(30)
#             chage_acct = not chage_acct
#         print('papu_and_y :')
#         display(papu_and_y.head())
#         # 排除例外(例如:舊戶)
#         papu_and_y = papu_and_y[papu_and_y[papulation_colname].isin(papulation_train_value)]
#         papu_and_y.drop([papulation_colname], axis=1, inplace=True)


#         # 將母體與特徵左右拼接
#         concat_df_outcome = get_feature_by_SOP(do_ym_list, mother, writing_path, drop_key_word, papu_and_y)
#         # 每個年月最多50萬筆
#         concat_df_outcome = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_build)

#         # 將各年月的DF上下拼接
#         df_combined = combine_multi_year_df(do_ym_list, concat_df_outcome)
#         print(f'df_combined.shape = {df_combined.shape}')
#         # 額外縮減至每個年月最多10萬筆(用於select_feature)
#         concat_df_outcome_redeuced = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_select)
#         df_combined_redeuced = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced)
#         print(f'df_combined_redeuced.shape = {df_combined_redeuced.shape}')

#         # 以較小資料集，建模選取重要特徵
#         X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
#         feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
#                                           project_name, mother, target, new_edition, algorithm)
#         cols_top100 = feat_imp.loc[:100]['feature'].tolist()

#         # 釋出空間
#         del X_train_rdc
#         del y_train_rdc
#         del X_test_rdc
#         del y_test_rdc
#         del df_combined_redeuced
#         del concat_df_outcome_redeuced
#         gc.collect()

#         # 以較大資料集，正式建模
#         based_col = ['customer_id', 'yyyymm', 'y']
#         X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top100],
#                                                       train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
#         model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top100, this_file_path,
#                                                                        project_name, mother, target, bins, new_edition, max_depth = max_depth_b,
#                                                                        scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
#         print('cols_impt_top100 : ')
#         display(model_imp)
#         print('test_level_df: ')
#         display(test_level_df)

#         # 以最新年月進行回測並產出效度
#         backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#         print(f'vali_level_df : ')
#         display(vali_level_df)
#         print(f'vali_predict_df.head() : ')
#         display(vali_predict_df.head())
#         print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
#         print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
#         print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')
#         # 寫出PICKLE回測等級
#         pickle.dump(vali_predict_df, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_' + project_name +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

#         # retrain_log
#         log_retrain = retrain_log_v230726(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                           train_auc, test_auc, backtest_auc, new_edition, table_name = 'opt_retrain_log')
#         print(f'retrain_log :')
#         display(log_retrain)

#         # model_log
#         log_model = model_log_v230410(False, project_name, mother, target, algorithm, bins,new_edition, table_name = 'opt_model_log')
#         print(f'model_log :')
#         display(log_model)



#         log_retrain_v230410 = retrain_log_v230726(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                   vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'opt_retrain_log')
#         log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition,
#                                               table_name = 'opt_model_log')

#         ## If retrain_auc < 0.7 then print

#         if test_auc < 0.7:
#             print('{}retrain_auc < 0.7'.format(mother))

#         print('{} 花了 {} min'.format(mother,(time.time()-totally_st_time)/60))


#         del concat_df_outcome
#         del X_train
#         del y_train
#         del X_test
#         del y_test
#         del df_combined
#         del papu_and_y
#         del vali_predict_df
#         del df_pred_and_features



# In[13]:


# # 詳細資料補_後面月份
# def retrain_mlops_np_and_p_optimal_2(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, query_list , frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b

#     # 若要存特徵則存在檔案位置
#     writing_path = None
#     if write_feature_Y_N:
#         writing_path = this_file_path

#     from Sql_module import get_SQL_raw_data, write_data_to_SQL
#     from Pretreatment import get_feature_by_SOP,combine_multi_year_df, if_to_large_down_sampling,get_papu_and_y
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v230726, model_log_v230410, retrain_log_v230314, get_next_model_version
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')
#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):
#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         query_function = query_list[index]
#         print(f'bin = {bins}')
#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'opt_2_ref_info')
#         time.sleep(60)
#         print(f'do_ym_list = {do_ym_list}')
#         # 抓取每個年月的母體
#         papu_and_y = get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                             writing_popu_path=this_file_path, mother=mother)
#         limit_times = 1
#         try_times = 1
#         while set(do_ym_list)!=set(papu_and_y['yyyymm'].astype('str')) and try_times <= limit_times:
#             papu_and_y = get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                         writing_popu_path=this_file_path, mother=mother)
#             try_times = try_times+1
# #         papu_and_y = pd.DataFrame()
# #         chage_acct = True
# #         for ym in do_ym_list:
# #             date =  datetime.date(datetime.strptime(ym,'%Y%m'))
# #             next_month = monthdelta(date, 1)
# #             print(next_month)
# #             query = query_function(next_month)
# #             if chage_acct:
# #                 df_temp = get_SQL_raw_data(query, account=config.account_yichieh, pwd=config.pwd_yichieh)
# #             else:
# #                 df_temp = get_SQL_raw_data(query)
# #             papu_and_y = papu_and_y.append(df_temp)
# #             time.sleep(60)
# #             chage_acct = not chage_acct

#         print('papu_and_y :')
#         display(papu_and_y.head())
#         # 排除例外(例如:舊戶)
#         papu_and_y = papu_and_y[papu_and_y[papulation_colname].isin(papulation_train_value)]
#         papu_and_y.drop([papulation_colname], axis=1, inplace=True)


#         # 將母體與特徵左右拼接
#         concat_df_outcome = get_feature_by_SOP(do_ym_list, mother, writing_path, drop_key_word, papu_and_y)

#         if '202209_df' in concat_df_outcome.keys():
#             df_202209 = concat_df_outcome['202209_df'].copy()
#             print(sum( concat_df_outcome['202209_df']['y']==1 ))
#             concat_df_outcome['202209_df'] = concat_df_outcome['202209_df'].append(df_202209[df_202209['y']==1])
#             print(sum( concat_df_outcome['202209_df']['y']==1 ))
#             concat_df_outcome['202209_df'] = concat_df_outcome['202209_df'].append(df_202209[df_202209['y']==1])
#             print(sum( concat_df_outcome['202209_df']['y']==1 ))
#         # 每個年月最多50萬筆
#         concat_df_outcome = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_build)

#         # 將各年月的DF上下拼接
#         df_combined = combine_multi_year_df(do_ym_list, concat_df_outcome)
#         print(f'df_combined.shape = {df_combined.shape}')
#         # 額外縮減至每個年月最多10萬筆(用於select_feature)
#         concat_df_outcome_redeuced = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_select)
#         df_combined_redeuced = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced)
#         print(f'df_combined_redeuced.shape = {df_combined_redeuced.shape}')

#         # 以較小資料集，建模選取重要特徵
#         X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
#         feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
#                                           project_name, mother, target, new_edition, algorithm)
#         cols_top100 = feat_imp.loc[:100]['feature'].tolist()

#         # 釋出空間
#         del X_train_rdc
#         del y_train_rdc
#         del X_test_rdc
#         del y_test_rdc
#         del df_combined_redeuced
#         del concat_df_outcome_redeuced
#         gc.collect()

#         # 以較大資料集，正式建模
#         based_col = ['customer_id', 'yyyymm', 'y']
#         X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top100],
#                                                       train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
#         model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top100, this_file_path,
#                                                                        project_name, mother, target, bins, new_edition, max_depth = max_depth_b,
#                                                                        scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
#         print('cols_impt_top100 : ')
#         display(model_imp)
#         print('test_level_df: ')
#         display(test_level_df)

#         # 以最新年月進行回測並產出效度
#         backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#         print(f'vali_level_df : ')
#         display(vali_level_df)
#         print(f'vali_predict_df.head() : ')
#         display(vali_predict_df.head())
#         print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
#         print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
#         print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')
#         # 寫出PICKLE回測等級
#         pickle.dump(vali_predict_df, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_' + project_name +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

#         # retrain_log
#         log_retrain = retrain_log_v230726(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                           train_auc, test_auc, backtest_auc, new_edition, table_name = 'opt_2_retrain_log')
#         print(f'retrain_log :')
#         display(log_retrain)

#         # model_log
#         log_model = model_log_v230410(False, project_name, mother, target, algorithm, bins,new_edition, table_name = 'opt_2_model_log')
#         print(f'model_log :')
#         display(log_model)



#         log_retrain_v230410 = retrain_log_v230726(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                   vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'opt_2_retrain_log')
#         log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition,
#                                               table_name = 'opt_2_model_log')

#         ## If retrain_auc < 0.7 then print

#         if test_auc < 0.7:
#             print('{}retrain_auc < 0.7'.format(mother))

#         print('{} 花了 {} min'.format(mother,(time.time()-totally_st_time)/60))


#         del concat_df_outcome
#         del X_train
#         del y_train
#         del X_test
#         del y_test
#         del df_combined
#         del papu_and_y
#         del vali_predict_df
#         del df_pred_and_features


# In[14]:


# # 詳細資料補_後面月份
# def retrain_mlops_np_and_p_optimal_2_fillna(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, query_list , frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b

#     # 若要存特徵則存在檔案位置
#     writing_path = None
#     if write_feature_Y_N:
#         writing_path = this_file_path

#     from Sql_module import get_SQL_raw_data, write_data_to_SQL
#     from Pretreatment import get_feature_by_SOP,combine_multi_year_df, if_to_large_down_sampling,get_papu_and_y
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v230726, model_log_v230410, retrain_log_v230314, get_next_model_version
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')
#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):
#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         query_function = query_list[index]
#         print(f'bin = {bins}')
#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'opt_2_ref_info')
#         time.sleep(60)
#         print(f'do_ym_list = {do_ym_list}')
#         # 抓取每個年月的母體
#         papu_and_y = get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                             writing_popu_path=this_file_path, mother=mother)
#         limit_times = 1
#         try_times = 1
#         while set(do_ym_list)!=set(papu_and_y['yyyymm'].astype('str')) and try_times <= limit_times:
#             papu_and_y = get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                         writing_popu_path=this_file_path, mother=mother)
#             try_times = try_times+1
# #         papu_and_y = pd.DataFrame()
# #         chage_acct = True
# #         for ym in do_ym_list:
# #             date =  datetime.date(datetime.strptime(ym,'%Y%m'))
# #             next_month = monthdelta(date, 1)
# #             print(next_month)
# #             query = query_function(next_month)
# #             if chage_acct:
# #                 df_temp = get_SQL_raw_data(query, account=config.account_yichieh, pwd=config.pwd_yichieh)
# #             else:
# #                 df_temp = get_SQL_raw_data(query)
# #             papu_and_y = papu_and_y.append(df_temp)
# #             time.sleep(60)
# #             chage_acct = not chage_acct

#         print('papu_and_y :')
#         display(papu_and_y.head())
#         # 排除例外(例如:舊戶)
#         papu_and_y = papu_and_y[papu_and_y[papulation_colname].isin(papulation_train_value)]
#         papu_and_y.drop([papulation_colname], axis=1, inplace=True)


#         # 將母體與特徵左右拼接
#         concat_df_outcome = get_feature_by_SOP(do_ym_list, mother, writing_path, drop_key_word, papu_and_y)

#         if '202209_df' in concat_df_outcome.keys():
#             df_202209 = concat_df_outcome['202209_df'].copy()
#             print(sum( concat_df_outcome['202209_df']['y']==1 ))
#             concat_df_outcome['202209_df'] = concat_df_outcome['202209_df'].append(df_202209[df_202209['y']==1])
#             print(sum( concat_df_outcome['202209_df']['y']==1 ))
#             concat_df_outcome['202209_df'] = concat_df_outcome['202209_df'].append(df_202209[df_202209['y']==1])
#             print(sum( concat_df_outcome['202209_df']['y']==1 ))
#         # 每個年月最多50萬筆
#         concat_df_outcome = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_build)

#         # 將各年月的DF上下拼接
#         df_combined = combine_multi_year_df(do_ym_list, concat_df_outcome)

#         # 缺失值補0
#         feature_list = list(df_combined.loc[:,~df_combined.columns.isin(['customer_id','yyyymm','y'])].select_dtypes(exclude = ['category','object']))
#         df_combined[feature_list].fillna(0, inplace=True) #categorical用replace會變成objective
#         print(f'df_combined.shape = {df_combined.shape}')

#         # 額外縮減至每個年月最多10萬筆(用於select_feature)
#         concat_df_outcome_redeuced = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_select)
#         df_combined_redeuced = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced)

#         # 缺失值補0
#         feature_list = list(df_combined_redeuced.loc[:,~df_combined_redeuced.columns.isin(['customer_id','yyyymm','y'])].select_dtypes(exclude = ['category','object']))
#         df_combined_redeuced[feature_list].fillna(0, inplace=True) #categorical用replace會變成objective
#         print(f'df_combined_redeuced.shape = {df_combined_redeuced.shape}')

#         # 以較小資料集，建模選取重要特徵
#         X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
#         feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
#                                           project_name, mother, target, new_edition, algorithm)
#         cols_top100 = feat_imp.loc[:100]['feature'].tolist()

#         # 釋出空間
#         del X_train_rdc
#         del y_train_rdc
#         del X_test_rdc
#         del y_test_rdc
#         del df_combined_redeuced
#         del concat_df_outcome_redeuced
#         gc.collect()

#         # 以較大資料集，正式建模
#         based_col = ['customer_id', 'yyyymm', 'y']
#         X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top100],
#                                                       train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
#         model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top100, this_file_path,
#                                                                        project_name, mother, target, bins, new_edition, max_depth = max_depth_b,
#                                                                        scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
#         print('cols_impt_top100 : ')
#         display(model_imp)
#         print('test_level_df: ')
#         display(test_level_df)

#         # 以最新年月進行回測並產出效度
#         backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#         print(f'vali_level_df : ')
#         display(vali_level_df)
#         print(f'vali_predict_df.head() : ')
#         display(vali_predict_df.head())
#         print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
#         print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
#         print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')
#         # 寫出PICKLE回測等級
#         pickle.dump(vali_predict_df, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_' + project_name +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

#         # retrain_log
#         log_retrain = retrain_log_v230726(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                           train_auc, test_auc, backtest_auc, new_edition, table_name = 'opt_2_fill_retrain_log')
#         print(f'retrain_log :')
#         display(log_retrain)

#         # model_log
#         log_model = model_log_v230410(False, project_name, mother, target, algorithm, bins,new_edition, table_name = 'opt_2_fill_model_log')
#         print(f'model_log :')
#         display(log_model)



#         log_retrain_v230410 = retrain_log_v230726(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                   vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'opt_2_fill_retrain_log')
#         log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition,
#                                               table_name = 'opt_2_fill_model_log')

#         ## If retrain_auc < 0.7 then print

#         if test_auc < 0.7:
#             print('{}retrain_auc < 0.7'.format(mother))

#         print('{} 花了 {} min'.format(mother,(time.time()-totally_st_time)/60))


#         del concat_df_outcome
#         del X_train
#         del y_train
#         del X_test
#         del y_test
#         del df_combined
#         del papu_and_y
#         del vali_predict_df
#         del df_pred_and_features


# In[15]:


# # 詳細資料補_後面月份
# def retrain_mlops_np_and_p_ffill_fillna(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, query_list , frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b

#     # 若要存特徵則存在檔案位置
#     writing_path = None
#     if write_feature_Y_N:
#         writing_path = this_file_path

#     from Sql_module import get_SQL_raw_data, write_data_to_SQL
#     from Pretreatment import get_feature_by_SOP,combine_multi_year_df, if_to_large_down_sampling,get_papu_and_y
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v230726, model_log_v230410, retrain_log_v230314, get_next_model_version
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')
#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):
#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         query_function = query_list[index]
#         print(f'bin = {bins}')
#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'dev_ref_info')
#         time.sleep(60)
#         print(f'do_ym_list = {do_ym_list}')
#         # 抓取每個年月的母體
#         papu_and_y = get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                             writing_popu_path=this_file_path, mother=mother)
#         limit_times = 1
#         try_times = 1
#         while set(do_ym_list)!=set(papu_and_y['yyyymm'].astype('str')) and try_times <= limit_times:
#             papu_and_y = get_papu_and_y(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                         writing_popu_path=this_file_path, mother=mother)
#             try_times = try_times+1
# #         papu_and_y = pd.DataFrame()
# #         chage_acct = True
# #         for ym in do_ym_list:
# #             date =  datetime.date(datetime.strptime(ym,'%Y%m'))
# #             next_month = monthdelta(date, 1)
# #             print(next_month)
# #             query = query_function(next_month)
# #             if chage_acct:
# #                 df_temp = get_SQL_raw_data(query, account=config.account_yichieh, pwd=config.pwd_yichieh)
# #             else:
# #                 df_temp = get_SQL_raw_data(query)
# #             papu_and_y = papu_and_y.append(df_temp)
# #             time.sleep(60)
# #             chage_acct = not chage_acct

#         print('papu_and_y :')
#         display(papu_and_y.head())
#         # 排除例外(例如:舊戶)
#         papu_and_y = papu_and_y[papu_and_y[papulation_colname].isin(papulation_train_value)]
#         papu_and_y.drop([papulation_colname], axis=1, inplace=True)


#         # 將母體與特徵左右拼接
#         concat_df_outcome = get_feature_by_SOP(do_ym_list, mother, writing_path, drop_key_word, papu_and_y)

#         # 每個年月最多50萬筆
#         concat_df_outcome = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_build)

#         # 將各年月的DF上下拼接
#         df_combined = combine_multi_year_df(do_ym_list, concat_df_outcome)

#         # 缺失值FFILL
#         st_ffill = time.time()
#         float16_col = df_combined.select_dtypes(include=['float16']).columns
#         df_combined[float16_col] = df_combined[float16_col].astype('float32')
#         df_combined = df_combined.sort_values(by=['customer_id','yyyymm'])
#         grouped = df_combined.groupby('customer_id')
#         df_combined = grouped.apply(lambda group: group.fillna(method='ffill'))
#         print(f'FFILL花了{round(time.time()-st_ffill,2)} sec ')

#         # 缺失值補0
#         st_fill_zero = time.time()
#         feature_list = list(df_combined.loc[:,~df_combined.columns.isin(['customer_id','yyyymm','y'])].select_dtypes(exclude = ['category','object']))
#         df_combined[feature_list].fillna(0, inplace=True) #categorical用replace會變成objective
#         print(f'FILL 0 花了{round(time.time()-st_fill_zero,2)} sec ')

#         print(f'df_combined.shape = {df_combined.shape}')

#         # 額外縮減至每個年月最多10萬筆(用於select_feature)
#         concat_df_outcome_redeuced = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_select)
#         df_combined_redeuced = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced)

#         # 缺失值FFILL
#         st_ffill = time.time()
#         float16_col = df_combined_redeuced.select_dtypes(include=['float16']).columns
#         df_combined_redeuced[float16_col] = df_combined_redeuced[float16_col].astype('float32')
#         df_combined_redeuced = df_combined_redeuced.sort_values(by=['customer_id','yyyymm'])
#         grouped = df_combined_redeuced.groupby('customer_id')
#         df_combined_redeuced = grouped.apply(lambda group: group.fillna(method='ffill'))
#         print(f'FFILL花了{round(time.time()-st_ffill,2)} sec ')

#         # 缺失值補0
#         st_fill_zero = time.time()
#         feature_list = list(df_combined_redeuced.loc[:,~df_combined_redeuced.columns.isin(['customer_id','yyyymm','y'])].select_dtypes(exclude = ['category','object']))
#         df_combined_redeuced[feature_list].fillna(0, inplace=True) #categorical用replace會變成objective
#         print(f'FILL 0 花了{round(time.time()-st_fill_zero,2)} sec ')

#         print(f'df_combined_redeuced.shape = {df_combined_redeuced.shape}')

#         # 以較小資料集，建模選取重要特徵
#         X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
#         feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
#                                           project_name, mother, target, new_edition, algorithm)
#         cols_top100 = feat_imp.loc[:100]['feature'].tolist()

#         # 釋出空間
#         del X_train_rdc
#         del y_train_rdc
#         del X_test_rdc
#         del y_test_rdc
#         del df_combined_redeuced
#         del concat_df_outcome_redeuced
#         gc.collect()

#         # 以較大資料集，正式建模
#         based_col = ['customer_id', 'yyyymm', 'y']
#         X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top100],
#                                                       train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
#         model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top100, this_file_path,
#                                                                        project_name, mother, target, bins, new_edition, max_depth = max_depth_b,
#                                                                        scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
#         print('cols_impt_top100 : ')
#         display(model_imp)
#         print('test_level_df: ')
#         display(test_level_df)

#         # 以最新年月進行回測並產出效度
#         backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#         print(f'vali_level_df : ')
#         display(vali_level_df)
#         print(f'vali_predict_df.head() : ')
#         display(vali_predict_df.head())
#         print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
#         print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
#         print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')
#         # 寫出PICKLE回測等級
#         pickle.dump(vali_predict_df, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_' + project_name +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

#         # retrain_log
#         log_retrain = retrain_log_v230726(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                           train_auc, test_auc, backtest_auc, new_edition, table_name = 'opt_3_ffill_retrain_log')
#         print(f'retrain_log :')
#         display(log_retrain)

#         # model_log
#         log_model = model_log_v230410(False, project_name, mother, target, algorithm, bins,new_edition, table_name = 'opt_3_ffill_model_log')
#         print(f'model_log :')
#         display(log_model)



#         log_retrain_v230410 = retrain_log_v230726(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                   vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'opt_3_ffill_retrain_log')
#         log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition,
#                                               table_name = 'opt_3_ffill_model_log')

#         ## If retrain_auc < 0.7 then print

#         if test_auc < 0.7:
#             print('{}retrain_auc < 0.7'.format(mother))

#         print('{} 花了 {} min'.format(mother,(time.time()-totally_st_time)/60))


#         del concat_df_outcome
#         del X_train
#         del y_train
#         del X_test
#         del y_test
#         del df_combined
#         del papu_and_y
#         del vali_predict_df
#         del df_pred_and_features


# In[16]:


# def retrain_mlops_np_and_p_no_append(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, query_list , frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b

#     # 若要存特徵則存在檔案位置
#     writing_path = None
#     if write_feature_Y_N:
#         writing_path = this_file_path

#     from Sql_module import get_SQL_raw_data, write_data_to_SQL
#     from Pretreatment_no_append_table import get_feature_by_SOP,combine_multi_year_df, if_to_large_down_sampling
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v230726, model_log_v230410, retrain_log_v230314, get_next_model_version
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')
#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):
#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         query_function = query_list[index]
#         print(f'bin = {bins}')
#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'ej_ref_info')
#         time.sleep(30)
#         print(f'do_ym_list = {do_ym_list}')
#         # 抓取每個年月的母體
#         papu_and_y = pd.DataFrame()
#         for ym in do_ym_list:
#             date =  datetime.date(datetime.strptime(ym,'%Y%m'))
#             next_month = monthdelta(date, 1)
#             print(next_month)
#             query = query_function(next_month)
#             df_temp = get_SQL_raw_data(query)
#             papu_and_y = papu_and_y.append(df_temp)
#             time.sleep(30)
#         print('papu_and_y :')
#         display(papu_and_y.head())
#         # 排除例外(例如:舊戶)
#         papu_and_y = papu_and_y[papu_and_y[papulation_colname].isin(papulation_train_value)]
#         papu_and_y.drop([papulation_colname], axis=1, inplace=True)


#         # 將母體與特徵左右拼接
#         concat_df_outcome = get_feature_by_SOP(do_ym_list, mother, writing_path, drop_key_word, papu_and_y)
#         # 每個年月最多50萬筆
#         concat_df_outcome = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_build)

#         # 將各年月的DF上下拼接
#         df_combined = combine_multi_year_df(do_ym_list, concat_df_outcome)
#         print(f'df_combined.shape = {df_combined.shape}')
#         # 額外縮減至每個年月最多10萬筆(用於select_feature)
#         concat_df_outcome_redeuced = if_to_large_down_sampling(do_ym_list, concat_df_outcome, limit_size=limit_size_select)
#         df_combined_redeuced = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced)
#         print(f'df_combined_redeuced.shape = {df_combined_redeuced.shape}')

#         # 以較小資料集，建模選取重要特徵
#         X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
#         feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
#                                           project_name, mother, target, new_edition, algorithm)
#         cols_top100 = feat_imp.loc[:100]['feature'].tolist()

#         # 釋出空間
#         del X_train_rdc
#         del y_train_rdc
#         del X_test_rdc
#         del y_test_rdc
#         del df_combined_redeuced
#         del concat_df_outcome_redeuced
#         gc.collect()

#         # 以較大資料集，正式建模
#         based_col = ['customer_id', 'yyyymm', 'y']
#         X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top100],
#                                                       train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
#         model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top100, this_file_path,
#                                                                        project_name, mother, target, bins, new_edition, max_depth = max_depth_b,
#                                                                        scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
#         print('cols_impt_top100 : ')
#         display(model_imp)
#         print('test_level_df: ')
#         display(test_level_df)

#         # 以最新年月進行回測並產出效度
#         backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#         print(f'vali_level_df : ')
#         display(vali_level_df)
#         print(f'vali_predict_df.head() : ')
#         display(vali_predict_df.head())
#         print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
#         print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
#         print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')
#         # 寫出PICKLE回測等級
#         pickle.dump(vali_predict_df, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_' + project_name +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

#         # retrain_log
#         log_retrain = retrain_log_v230726(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                           train_auc, test_auc, backtest_auc, new_edition, table_name = 'ej_retrain_log')
#         print(f'retrain_log :')
#         display(log_retrain)

#         # model_log
#         log_model = model_log_v230410(False, project_name, mother, target, algorithm, bins,new_edition, table_name = 'ej_model_log')
#         print(f'model_log :')
#         display(log_model)



#         log_retrain_v230410 = retrain_log_v230726(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                   vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'ej_retrain_log')
#         log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition,
#                                               table_name = 'ej_model_log')

#         ## If retrain_auc < 0.7 then print

#         if test_auc < 0.7:
#             print('{}retrain_auc < 0.7'.format(mother))

#         print('{} 花了 {} min'.format(mother,(time.time()-totally_st_time)/60))


#         del concat_df_outcome
#         del X_train
#         del y_train
#         del X_test
#         del y_test
#         del df_combined
#         del papu_and_y
#         del vali_predict_df
#         del df_pred_and_features



# In[17]:


# def retrain_mlops_np_and_p_double_opt_detailym_fillna_2024OPT(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, hit_rate_list, query_list, query_list_jihsun, frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N, market_flag_Y_N, n_feature_list):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b

#     feature_file_path_fubon = config.feature_file_path_fubon
#     feature_file_path_jihsun = config.feature_file_path_jihsun

#     from Sql_module import get_SQL_raw_data, write_data_to_SQL
#     from Pretreatment import get_feature_by_SOP,get_feature_by_SOP_jihsun,combine_multi_year_df, if_to_large_down_sampling, get_papu_and_y_local, if_large_down_sampling_from_papu_and_y
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v240115, Convert_log_v240111
#     from Model import model_log_v230410, retrain_log_v230314, get_next_model_version, whether_done_next_version, get_conversion_rank
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')
#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):

#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         n_feature = n_feature_list[index]
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         hit_rate = hit_rate_list[index]
#         query_function = query_list[index]
#         query_function_jihsun = query_list_jihsun[index]
#         print(f'bin = {bins}')
#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱
#         new_edition = get_next_model_version(project_name,target,mother,frequency,edition_detail, table_name = 'mlops_ref_info_double',
#                                              account=config.account_yichieh, pwd=config.pwd_yichieh)

#         time.sleep(15)
#         print(f'do_ym_list = {do_ym_list}')

#         #########################################################################################################
#         # 富邦DF
#         ############################################################################################################
#         print('!!!!開始抓取富邦資料')
#         # 先抓一次
#         fubon_ft_fold = this_file_path.replace('審核通過模型_雙證_2024opt','審核通過模型')
#         papu_and_y_fubon = get_papu_and_y_local(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                writing_popu_path = fubon_ft_fold, mother = mother)
#         limit_times = 1
#         try_times = 1
#         # 重複抓
#         while set(do_ym_list)!=set(papu_and_y_fubon['yyyymm'].astype('str')) and try_times <= limit_times:
#             papu_and_y_fubon = get_papu_and_y_local(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                    writing_popu_path = fubon_ft_fold, mother = mother)
#             try_times = try_times+1

#         print('papu_and_y_fubon :')
#         display(papu_and_y_fubon.head())
#         # 排除例外(例如:舊戶)
#         papu_and_y_fubon = papu_and_y_fubon[papu_and_y_fubon[papulation_colname].isin(papulation_train_value)]
#         papu_and_y_fubon.drop([papulation_colname], axis=1, inplace=True)

# #             # 重複訓練集最新年月
# #             do_ym_list_sorted = sorted(do_ym_list)
# #             dupli_ym = str(do_ym_list_sorted[-2])
# #             dupli_ym_df = papu_and_y_fubon[papu_and_y_fubon['yyyymm'] == int(dupli_ym)]
# #             dupli_ym_df_y1 = dupli_ym_df[dupli_ym_df['y'] == 1]
# #             print(f'重複最新年月({dupli_ym}) Y_VALUE前Y數量為: {len(dupli_ym_df_y1)}')
# #             papu_and_y_fubon = papu_and_y_fubon.append(dupli_ym_df_y1)
# #             mask1 = (papu_and_y_fubon['yyyymm'] == int(dupli_ym))
# #             mask2 = (papu_and_y_fubon['y'] == 1)
# #             print(f'重複一次年月({dupli_ym}) Y_VALUE後Y數量為: {len(papu_and_y_fubon[mask1&mask2])}')
# #             papu_and_y_fubon = papu_and_y_fubon.append(dupli_ym_df_y1)
# #             mask1 = (papu_and_y_fubon['yyyymm'] == int(dupli_ym))
# #             mask2 = (papu_and_y_fubon['y'] == 1)
# #             print(f'重複二次年月({dupli_ym}) Y_VALUE後Y數量為: {len(papu_and_y_fubon[mask1&mask2])}')

#         # 是否存特徵, 若要存特徵則存在this_file_path
#         writing_path = None
#         if write_feature_Y_N:
#             writing_path = this_file_path.replace('審核通過模型_雙證_2024opt','審核通過模型')

#         # 每個年月最多50萬筆
#         papu_and_y_fubon_build = if_large_down_sampling_from_papu_and_y(do_ym_list, papu_and_y_fubon, cust_source='Fubon', limit_size=limit_size_build)

#         # 將母體與特徵左右拼接
#         concat_df_outcome_fubon = get_feature_by_SOP(do_ym_list, mother, writing_path, drop_key_word, papu_and_y_fubon_build,
#                                                      feature_file_path = feature_file_path_fubon, just_for_check=False, Fill_zero=True)

#         # 重複訓練集最新年月
# #             do_ym_list_sorted = sorted(do_ym_list)
# #             dupli_ym_df_key = str(do_ym_list_sorted[-2])+'_df'
# #             if dupli_ym_df_key in concat_df_outcome_fubon.keys():
# #                 df_dupli_ym = concat_df_outcome_fubon[dupli_ym_df_key].copy()
# #                 Y_NUMBER = sum(concat_df_outcome_fubon[dupli_ym_df_key]['y']==1)
# #                 print(f'重複最新年月({dupli_ym_df_key}) Y_VALUE前Y數量為: {Y_NUMBER}')

# #                 concat_df_outcome_fubon[dupli_ym_df_key] = concat_df_outcome_fubon[dupli_ym_df_key].append(df_dupli_ym[df_dupli_ym['y']==1])
# #                 Y_NUMBER = sum(concat_df_outcome_fubon[dupli_ym_df_key]['y']==1)
# #                 print(f'重複一次年月({dupli_ym_df_key}) Y_VALUE後Y數量為: {Y_NUMBER}')

# #                 concat_df_outcome_fubon[dupli_ym_df_key] = concat_df_outcome_fubon[dupli_ym_df_key].append(df_dupli_ym[df_dupli_ym['y']==1])
# #                 Y_NUMBER = sum(concat_df_outcome_fubon[dupli_ym_df_key]['y']==1)
# #                 print(f'重複二次年月({dupli_ym_df_key}) Y_VALUE後Y數量為: {Y_NUMBER}')

#        # 將各年月的DF上下拼接
#         df_combined_fubon = combine_multi_year_df(do_ym_list, concat_df_outcome_fubon)
#         print(f'df_combined_fubon.shape = {df_combined_fubon.shape}')

#         # 額外縮減至每個年月最多10萬筆(用於select_feature)
#         concat_df_outcome_redeuced_fubon = if_to_large_down_sampling(do_ym_list, concat_df_outcome_fubon, limit_size=limit_size_select)
#         df_combined_redeuced_fubon = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced_fubon)
#         print(f'df_combined_redeuced_fubon.shape = {df_combined_redeuced_fubon.shape}')

#        # 清出空間
#         del papu_and_y_fubon
#         del concat_df_outcome_fubon
#         del concat_df_outcome_redeuced_fubon
#         gc.collect()
#         ###############################################################################################################
#         # 日盛DF
#         ############################################################################################################
#         print('!!!!開始抓取日盛資料')
#         # 先抓一次
#         fubon_ft_fold = this_file_path.replace('審核通過模型_雙證_2024opt','審核通過模型_jihsun')
#         papu_and_y_jihsun = get_papu_and_y_local(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                writing_popu_path = fubon_ft_fold, mother = mother)
#         limit_times = 1
#         try_times = 1
#         # 重複抓
#         while set(do_ym_list)!=set(papu_and_y_jihsun['yyyymm'].astype('str')) and try_times <= limit_times:
#             papu_and_y_jihsun = get_papu_and_y_local(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                    writing_popu_path = fubon_ft_fold, mother = mother)
#             try_times = try_times+1

#         print('papu_and_y_jihsun :')
#         display(papu_and_y_jihsun.head())
#         # 排除例外(例如:舊戶)
#         papu_and_y_jihsun = papu_and_y_jihsun[papu_and_y_jihsun[papulation_colname].isin(papulation_train_value)]
#         papu_and_y_jihsun.drop([papulation_colname], axis=1, inplace=True)

# #             # 重複訓練集最新年月
# #             do_ym_list_sorted = sorted(do_ym_list)
# #             dupli_ym = str(do_ym_list_sorted[-2])
# #             dupli_ym_df = papu_and_y_jihsun[papu_and_y_jihsun['yyyymm'] == int(dupli_ym)]
# #             dupli_ym_df_y1 = dupli_ym_df[dupli_ym_df['y'] == 1]
# #             print(f'重複最新年月({dupli_ym}) Y_VALUE前Y數量為: {len(dupli_ym_df_y1)}')
# #             papu_and_y_jihsun = papu_and_y_jihsun.append(dupli_ym_df_y1)
# #             mask1 = (papu_and_y_jihsun['yyyymm'] == int(dupli_ym))
# #             mask2 = (papu_and_y_jihsun['y'] == 1)
# #             print(f'重複一次年月({dupli_ym}) Y_VALUE後Y數量為: {len(papu_and_y_jihsun[mask1&mask2])}')
# #             papu_and_y_jihsun = papu_and_y_jihsun.append(dupli_ym_df_y1)
# #             mask1 = (papu_and_y_jihsun['yyyymm'] == int(dupli_ym))
# #             mask2 = (papu_and_y_jihsun['y'] == 1)
# #             print(f'重複二次年月({dupli_ym}) Y_VALUE後Y數量為: {len(papu_and_y_jihsun[mask1&mask2])}')

#         # 是否存特徵, 若要存特徵則存在this_file_path
#         writing_path = None
#         if write_feature_Y_N:
#             writing_path = this_file_path.replace('審核通過模型_雙證_2024opt','審核通過模型_jihsun')

#         # 每個年月最多50萬筆
#         papu_and_y_jihsun_build = if_large_down_sampling_from_papu_and_y(do_ym_list, papu_and_y_jihsun, cust_source='Jihsun', limit_size=limit_size_build)

#         # 將母體與特徵左右拼接
#         concat_df_outcome_jihsun = get_feature_by_SOP_jihsun(do_ym_list, mother, writing_path, drop_key_word, papu_and_y_jihsun_build,
#                                                              feature_file_path = feature_file_path_jihsun, just_for_check=False, Fill_zero=True)

#         # 重複訓練集最新年月
# #             do_ym_list_sorted = sorted(do_ym_list)
# #             dupli_ym_df_key = str(do_ym_list_sorted[-2])+'_df'
# #             if dupli_ym_df_key in concat_df_outcome_jihsun.keys():

# #                 df_dupli_ym = concat_df_outcome_jihsun[dupli_ym_df_key].copy()
# #                 Y_NUMBER = sum(concat_df_outcome_jihsun[dupli_ym_df_key]['y']==1)
# #                 print(f'重複最新年月({dupli_ym_df_key}) Y_VALUE前Y數量為: {Y_NUMBER}')

# #                 concat_df_outcome_jihsun[dupli_ym_df_key] = concat_df_outcome_jihsun[dupli_ym_df_key].append(df_dupli_ym[df_dupli_ym['y']==1])
# #                 Y_NUMBER = sum(concat_df_outcome_jihsun[dupli_ym_df_key]['y']==1)
# #                 print(f'重複一次年月({dupli_ym_df_key}) Y_VALUE後Y數量為: {Y_NUMBER}')

# #                 concat_df_outcome_jihsun[dupli_ym_df_key] = concat_df_outcome_jihsun[dupli_ym_df_key].append(df_dupli_ym[df_dupli_ym['y']==1])
# #                 Y_NUMBER = sum(concat_df_outcome_jihsun[dupli_ym_df_key]['y']==1)
# #                 print(f'重複二次年月({dupli_ym_df_key}) Y_VALUE後Y數量為: {Y_NUMBER}')

#         # 將各年月的DF上下拼接
#         df_combined_jihsun = combine_multi_year_df(do_ym_list, concat_df_outcome_jihsun)
#         print(f'df_combined_jihsun.shape = {df_combined_jihsun.shape}')



#         # 額外縮減至每個年月最多10萬筆(用於select_feature)
#         concat_df_outcome_redeuced_jihsun = if_to_large_down_sampling(do_ym_list, concat_df_outcome_jihsun, limit_size=limit_size_select)
#         df_combined_redeuced_jihsun = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced_jihsun)
#         print(f'df_combined_redeuced_jihsun.shape = {df_combined_redeuced_jihsun.shape}')

#         # 清出空間
#         del papu_and_y_jihsun
#         del concat_df_outcome_jihsun
#         del concat_df_outcome_redeuced_jihsun
#         gc.collect()
#         ###############################################################################################################
#         # 開始合併富證日盛客戶
#         ##############################################################################################################
#         print('!!!!開始合併富邦日盛資料')
#         # 資料: [build_set]
#         st_concat = time.time()
#         df_combined = pd.concat([df_combined_fubon,df_combined_jihsun],axis = 0)
#         print(f'[build_set] 富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#         # Concat後categorical會變objective,要轉回來
#         st_concat_astype = time.time()
#         obj_cols = df_combined.select_dtypes('object').drop(['customer_id'],axis=1).columns
#         df_combined[obj_cols.tolist()] = df_combined[obj_cols.tolist()].astype('category')
#         print(f'[build_set] 富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#         #移除雙證重複
#         remove_dupli_id_time = time.time()
#         df_combined['status'] = df_combined.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#         print(f'[build_set] 原始雙證人數: {len(df_combined)}')
#         df_combined = df_combined[df_combined['status']==1]
#         print(f'[build_set] 移除重複後雙證人數: {len(df_combined)}')
#         print(f'[build_set] 富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')


#         # 資料: [select_feature_set]
#         st_concat = time.time()
#         df_combined_redeuced = pd.concat([df_combined_redeuced_fubon,df_combined_redeuced_jihsun],axis = 0)
#         print(f'[select_feature_set] 富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#         # Concat後categorical會變objective,要轉回來
#         st_concat_astype = time.time()
#         obj_cols = df_combined_redeuced.select_dtypes('object').drop(['customer_id'],axis=1).columns
#         df_combined_redeuced[obj_cols.tolist()] = df_combined_redeuced[obj_cols.tolist()].astype('category')
#         print(f'[select_feature_set] 富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#         #移除雙證重複
#         remove_dupli_id_time = time.time()
#         df_combined_redeuced['status'] = df_combined_redeuced.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#         print(f'[select_feature_set] 原始雙證人數: {len(df_combined_redeuced)}')
#         df_combined_redeuced = df_combined_redeuced[df_combined_redeuced['status']==1]
#         print(f'[select_feature_set] 移除重複後雙證人數: {len(df_combined_redeuced)}')
#         print(f'[select_feature_set] 富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')

#         # 以較小資料集，建模選取重要特徵
#         X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
#         feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
#                                           project_name, mother, target, new_edition, algorithm)
#         cols_top = feat_imp.loc[:n_feature]['feature'].tolist()

#         # 釋出空間
#         del X_train_rdc
#         del y_train_rdc
#         del X_test_rdc
#         del y_test_rdc
#         del df_combined_redeuced_fubon
#         del df_combined_redeuced_jihsun
#         del df_combined_redeuced
#         gc.collect()

#         # 以較大資料集，正式建模
#         based_col = ['customer_id', 'yyyymm', 'y']
#         X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top],
#                                                       train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
#         model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top, this_file_path,
#                                                                        project_name, mother, target, bins, new_edition, max_depth = max_depth_b,
#                                                                        scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
#         print(f'cols_impt_top{n_feature} : ')
#         display(model_imp)
#         print('test_level_df: ')
#         display(test_level_df)

#         # 以最新年月進行回測並產出效度
#         backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#         print(f'vali_level_df : ')
#         display(vali_level_df)
#         print(f'vali_predict_df.head() : ')
#         display(vali_predict_df.head())
#         print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
#         print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
#         print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')

#         # 以詳細切分方法進行最新年月回測並產出效度
#         backtest_auc_dt, vali_level_df_dt, vali_predict_df_dt, df_pred_and_features_dt =             back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]],
#                   bins = [0,100,200,300,400,500,600,700,800,900,
#                           1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,
#                           11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,
#                           30000,40000,50000,60000,70000,80000,90000,100000,
#                           120000,140000,160000,180000,200000,
#                           1000000])
#         print(f'vali_level_df_dt : ')
#         display(vali_level_df_dt)
#         print(f'vali_predict_df_dt.head() : ')
#         display(vali_predict_df_dt.head())
#         print(f'vali_predivali_predict_df_dtct_df.shape :  {vali_predict_df_dt.shape}\n')
#         print(f'df_pred_and_features_dt.columns : \n {df_pred_and_features_dt.columns}\n')
#         print(f'df_pred_and_features_dt.shape : \n {df_pred_and_features_dt.shape}\n')

#         # 寫出PICKLE回測等級
#         pickle.dump(vali_predict_df_dt, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_detail' + project_name +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

#         # 行銷相關才做轉換率
#         if market_flag_Y_N:

#             hit_rate_setting_bin = []
#             for ht in sorted([0]+hit_rate, reverse=True):
#                     if  ht == 0:
#                         hit_rate_setting_bin.append( "C{:02d}".format(int(round(ht * 100))) )
#                     elif  ht < 0.001:
#                         hit_rate_setting_bin.append( "C{:04d}".format(int(round(ht * 10000))) )
#                     elif  ht < 0.01:
#                         hit_rate_setting_bin.append( "C{:03d}".format(int(round(ht * 1000))) )
#                     else:
#                         hit_rate_setting_bin.append( "C{:02d}".format(int(round(ht * 100))) )

#             CHANCE_TEST_bin, CHANCE_TEST_num, CHANCE_TEST_hit_rate, CHANCE_TEST_prob = get_conversion_rank(vali_predict_df_dt, hit_rate)
#             # retrain_log
#             log_retrain = retrain_log_v240115(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                               train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_24opt_d19',
#                                               vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                               CHANCE_TEST_bin=CHANCE_TEST_bin, CHANCE_TEST_num=CHANCE_TEST_num,
#                                               CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate, CHANCE_TEST_prob=CHANCE_TEST_prob,
#                                               account=config.account_yichieh, pwd=config.pwd_yichieh)
#             print(f'retrain_log :')
#             display(log_retrain)

#             # model_log
#             log_model = model_log_v230410(False, project_name, mother, target, algorithm, hit_rate_setting_bin, new_edition, table_name = 'mlops_model_log_24opt_d19', account=config.account_yichieh, pwd=config.pwd_yichieh)
#             print(f'model_log :')
#             display(log_model)



#             log_retrain_v230410 = retrain_log_v240115(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                       vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_24opt_d19',
#                                                       vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                                       CHANCE_TEST_bin=CHANCE_TEST_bin, CHANCE_TEST_num=CHANCE_TEST_num,
#                                                       CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate, CHANCE_TEST_prob=CHANCE_TEST_prob,
#                                                       account=config.account_yichieh, pwd=config.pwd_yichieh)
#             log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, hit_rate_setting_bin, new_edition,
#                                                   table_name = 'mlops_model_log_24opt_d19', account=config.account_yichieh, pwd=config.pwd_yichieh)

#             # 額外寫出轉換率對照表
#             detail_hit_rate  = list(map(lambda x: round(x*0.0001,4), range(10)))[1:]  + list(map(lambda x: round(x*0.001,3), range(10)))[1:] + list(map(lambda x: round(x*0.01,2), range(26)))[1:]
#             CHANCE_TEST_bin_dt, CHANCE_TEST_num_dt, CHANCE_TEST_hit_rate_dt, CHANCE_TEST_prob_dt = get_conversion_rank(vali_predict_df_dt, detail_hit_rate)
#             conver_log = Convert_log_v240111(False, project_name, mother, df_combined,
#                                         backtest_auc, new_edition, table_name = 'mlops_convert_log_24opt_d19',
#                                        CHANCE_TEST_bin=CHANCE_TEST_bin_dt, CHANCE_TEST_num=CHANCE_TEST_num_dt,
#                                         CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate_dt, CHANCE_TEST_prob=CHANCE_TEST_prob_dt,
#                                              account=config.account_yichieh, pwd=config.pwd_yichieh)
#             print(f'conver_log :')
#             display(conver_log)
#             conver_log = Convert_log_v240111(write_db_Y_N, project_name, mother, df_combined,
#                                         backtest_auc, new_edition, table_name = 'mlops_convert_log_24opt_d19',
#                                        CHANCE_TEST_bin=CHANCE_TEST_bin_dt, CHANCE_TEST_num=CHANCE_TEST_num_dt,
#                                         CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate_dt, CHANCE_TEST_prob=CHANCE_TEST_prob_dt,
#                                              account=config.account_yichieh, pwd=config.pwd_yichieh)
#         else:
#             # retrain_log
#             log_retrain = retrain_log_v240115(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                               train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_24opt_d19',
#                                               vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                               account=config.account_yichieh, pwd=config.pwd_yichieh
#                                               )
#             print(f'retrain_log :')
#             display(log_retrain)

#             # model_log
#             log_model = model_log_v230410(False, project_name, mother, target, algorithm, bins,new_edition, table_name = 'mlops_model_log_24opt_d19',
#                                               account=config.account_yichieh, pwd=config.pwd_yichieh)
#             print(f'model_log :')
#             display(log_model)



#             log_retrain_v230410 = retrain_log_v240115(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                       vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_24opt_d19',
#                                                       vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                               account=config.account_yichieh, pwd=config.pwd_yichieh
#                                                       )
#             log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition,
#                                                   table_name = 'mlops_model_log_24opt_d19',
#                                               account=config.account_yichieh, pwd=config.pwd_yichieh)

#         ## If retrain_auc < 0.7 then print

#         if test_auc < 0.7:
#             print('{}retrain_auc < 0.7'.format(mother))


#         print('{} 花了 {} min'.format(mother,(time.time()-totally_st_time)/60))


#         del X_train
#         del y_train
#         del X_test
#         del y_test
#         del df_combined_fubon
#         del df_combined_jihsun
#         del vali_predict_df
#         del df_pred_and_features
#         gc.collect()

#         print(f'{project_name}_{mother}_{new_edition} 執行完畢，先跳出迴圈!')
# #         break
# #         else:
# #             print(f'{project_name}_{mother}_{new_edition} 先前已執行過了，執行下個母體!')


# In[18]:


# def retrain_mlops_np_and_p_double_opt_detailym_fillna_forNN(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
#                            mother_list, bins_list, hit_rate_list, query_list, query_list_jihsun, frequency, edition_detail, do_ym_list, write_db_Y_N,
#                            limit_size_select, limit_size_build, write_feature_Y_N, market_flag_Y_N, n_feature_list, new_edition):
#     import sys
#     import os
#     import gc
#     from IPython.display import display
#     #該檔案路徑
#     project_name = this_file_path.split('/')[-1]
#     # 讀取模組設定
#     sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
#     import config
#     # MLOPS執行參數
#     algorithm = config.algorithm
#     #潛客參數
#     max_depth_a = config.max_depth_a
#     scale_pos_weight_a = config.scale_pos_weight_a
#     n_estimator_a = config.n_estimator_a
#     #非潛客參數
#     max_depth_b = config. max_depth_b
#     scale_pos_weight_b = config.scale_pos_weight_b
#     n_estimator_b = config.n_estimator_b

#     feature_file_path_fubon = config.feature_file_path_fubon
#     feature_file_path_jihsun = config.feature_file_path_jihsun

#     from Sql_module import get_SQL_raw_data, write_data_to_SQL
#     from Pretreatment import get_feature_by_SOP,get_feature_by_SOP_jihsun,combine_multi_year_df, if_to_large_down_sampling, get_papu_and_y_local, if_large_down_sampling_from_papu_and_y
#     from Model import monthdelta, split_data, select_feature_v230517, build_model_v230726, back_test, retrain_log_v240115, Convert_log_v240111
#     from Model import model_log_v230410, retrain_log_v230314, get_next_model_version, whether_done_next_version, get_conversion_rank
#     import calendar
#     from sqlalchemy import create_engine
#     import time
#     import pandas as pd
#     import pickle
#     import numpy as np
#     from xgboost import XGBClassifier
#     from sklearn.model_selection import train_test_split
#     from sqlalchemy.types import String, Integer, Float
#     from datetime import date, timedelta , datetime
#     from sklearn.metrics import roc_auc_score
#     from os import listdir
#     from os.path import isfile, join
#     pd.set_option('display.max_rows',300)
#     pd.set_option('display.max_columns',300)

#     # 判斷是設定有無問題
#     if len(mother_list) == len(bins_list) and len(bins_list) == len(query_list):
#         print(f'將會以母體執行迴圈 {mother_list} \n切分方式為 {bins_list}\n ')
#     else:
#         raise Exception('mother_list、bins_list、query_list長度不一致')
#     # 開始執行迴圈
#     for index, mother in enumerate(mother_list):

#         totally_st_time = time.time()
#         print(f'第{index}圈執行 {mother} Retrain ...')
#         n_feature = n_feature_list[index]
#         # 抓取對應 bins / query
#         bins = bins_list[index]
#         hit_rate = hit_rate_list[index]
#         query_function = query_list[index]
#         query_function_jihsun = query_list_jihsun[index]
#         print(f'bin = {bins}')
#         # 透過DB TABLE(參照資訊檔)抓取下一個版本名稱

#         time.sleep(15)
#         print(f'do_ym_list = {do_ym_list}')

#         #########################################################################################################
#         # 富邦DF
#         ############################################################################################################
#         print('!!!!開始抓取富邦資料')
#         # 先抓一次
#         fubon_ft_fold = this_file_path.replace('審核通過模型_雙證_2024opt','審核通過模型')
#         papu_and_y_fubon = get_papu_and_y_local(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                writing_popu_path = fubon_ft_fold, mother = mother)
#         limit_times = 1
#         try_times = 1
#         # 重複抓
#         while set(do_ym_list)!=set(papu_and_y_fubon['yyyymm'].astype('str')) and try_times <= limit_times:
#             papu_and_y_fubon = get_papu_and_y_local(do_ym_list, query_function, papulation_colname, papulation_train_value,
#                                    writing_popu_path = fubon_ft_fold, mother = mother)
#             try_times = try_times+1

#         print('papu_and_y_fubon :')
#         display(papu_and_y_fubon.head())
#         # 排除例外(例如:舊戶)
#         papu_and_y_fubon = papu_and_y_fubon[papu_and_y_fubon[papulation_colname].isin(papulation_train_value)]
#         papu_and_y_fubon.drop([papulation_colname], axis=1, inplace=True)

#         # 是否存特徵, 若要存特徵則存在this_file_path
#         writing_path = None
#         if write_feature_Y_N:
#             writing_path = this_file_path.replace('審核通過模型_雙證_2024opt','審核通過模型')

#         # 每個年月最多50萬筆
#         papu_and_y_fubon_build = if_large_down_sampling_from_papu_and_y(do_ym_list, papu_and_y_fubon, cust_source='Fubon', limit_size=limit_size_build)

#         # 將母體與特徵左右拼接
#         concat_df_outcome_fubon = get_feature_by_SOP(do_ym_list, mother, writing_path, drop_key_word, papu_and_y_fubon_build,
#                                                      feature_file_path = feature_file_path_fubon, just_for_check=False, Fill_zero=True)

#        # 將各年月的DF上下拼接
#         df_combined_fubon = combine_multi_year_df(do_ym_list, concat_df_outcome_fubon)
#         print(f'df_combined_fubon.shape = {df_combined_fubon.shape}')

#         # 額外縮減至每個年月最多10萬筆(用於select_feature)
#         concat_df_outcome_redeuced_fubon = if_to_large_down_sampling(do_ym_list, concat_df_outcome_fubon, limit_size=limit_size_select)
#         df_combined_redeuced_fubon = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced_fubon)
#         print(f'df_combined_redeuced_fubon.shape = {df_combined_redeuced_fubon.shape}')

#        # 清出空間
#         del papu_and_y_fubon
#         del concat_df_outcome_fubon
#         del concat_df_outcome_redeuced_fubon
#         gc.collect()
#         ###############################################################################################################
#         # 日盛DF
#         ############################################################################################################
#         print('!!!!開始抓取日盛資料')
#         # 先抓一次
#         fubon_ft_fold = this_file_path.replace('審核通過模型_雙證_2024opt','審核通過模型_jihsun')
#         papu_and_y_jihsun = get_papu_and_y_local(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                writing_popu_path = fubon_ft_fold, mother = mother)
#         limit_times = 1
#         try_times = 1
#         # 重複抓
#         while set(do_ym_list)!=set(papu_and_y_jihsun['yyyymm'].astype('str')) and try_times <= limit_times:
#             papu_and_y_jihsun = get_papu_and_y_local(do_ym_list, query_function_jihsun, papulation_colname, papulation_train_value,
#                                    writing_popu_path = fubon_ft_fold, mother = mother)
#             try_times = try_times+1

#         print('papu_and_y_jihsun :')
#         display(papu_and_y_jihsun.head())
#         # 排除例外(例如:舊戶)
#         papu_and_y_jihsun = papu_and_y_jihsun[papu_and_y_jihsun[papulation_colname].isin(papulation_train_value)]
#         papu_and_y_jihsun.drop([papulation_colname], axis=1, inplace=True)

#         # 是否存特徵, 若要存特徵則存在this_file_path
#         writing_path = None
#         if write_feature_Y_N:
#             writing_path = this_file_path.replace('審核通過模型_雙證_2024opt','審核通過模型_jihsun')

#         # 每個年月最多50萬筆
#         papu_and_y_jihsun_build = if_large_down_sampling_from_papu_and_y(do_ym_list, papu_and_y_jihsun, cust_source='Jihsun', limit_size=limit_size_build)

#         # 將母體與特徵左右拼接
#         concat_df_outcome_jihsun = get_feature_by_SOP_jihsun(do_ym_list, mother, writing_path, drop_key_word, papu_and_y_jihsun_build,
#                                                              feature_file_path = feature_file_path_jihsun, just_for_check=False, Fill_zero=True)

#         # 將各年月的DF上下拼接
#         df_combined_jihsun = combine_multi_year_df(do_ym_list, concat_df_outcome_jihsun)
#         print(f'df_combined_jihsun.shape = {df_combined_jihsun.shape}')



#         # 額外縮減至每個年月最多10萬筆(用於select_feature)
#         concat_df_outcome_redeuced_jihsun = if_to_large_down_sampling(do_ym_list, concat_df_outcome_jihsun, limit_size=limit_size_select)
#         df_combined_redeuced_jihsun = combine_multi_year_df(do_ym_list, concat_df_outcome_redeuced_jihsun)
#         print(f'df_combined_redeuced_jihsun.shape = {df_combined_redeuced_jihsun.shape}')

#         # 清出空間
#         del papu_and_y_jihsun
#         del concat_df_outcome_jihsun
#         del concat_df_outcome_redeuced_jihsun
#         gc.collect()
#         ###############################################################################################################
#         # 開始合併富證日盛客戶
#         ##############################################################################################################
#         print('!!!!開始合併富邦日盛資料')
#         # 資料: [build_set]
#         st_concat = time.time()
#         df_combined = pd.concat([df_combined_fubon,df_combined_jihsun],axis = 0)
#         print(f'[build_set] 富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#         # Concat後categorical會變objective,要轉回來
#         st_concat_astype = time.time()
#         obj_cols = df_combined.select_dtypes('object').drop(['customer_id'],axis=1).columns
#         df_combined[obj_cols.tolist()] = df_combined[obj_cols.tolist()].astype('category')
#         print(f'[build_set] 富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#         #移除雙證重複
#         remove_dupli_id_time = time.time()
#         df_combined['status'] = df_combined.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#         print(f'[build_set] 原始雙證人數: {len(df_combined)}')
#         df_combined = df_combined[df_combined['status']==1]
#         print(f'[build_set] 移除重複後雙證人數: {len(df_combined)}')
#         print(f'[build_set] 富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')


#         # 資料: [select_feature_set]
#         st_concat = time.time()
#         df_combined_redeuced = pd.concat([df_combined_redeuced_fubon,df_combined_redeuced_jihsun],axis = 0)
#         print(f'[select_feature_set] 富邦日盛合併 Runtime : {round(st_concat-time.time(),2)} sec')
#         # Concat後categorical會變objective,要轉回來
#         st_concat_astype = time.time()
#         obj_cols = df_combined_redeuced.select_dtypes('object').drop(['customer_id'],axis=1).columns
#         df_combined_redeuced[obj_cols.tolist()] = df_combined_redeuced[obj_cols.tolist()].astype('category')
#         print(f'[select_feature_set] 富邦日盛合併後轉類別 Runtime : {round(st_concat_astype-time.time(),2)} sec')
#         #移除雙證重複
#         remove_dupli_id_time = time.time()
#         df_combined_redeuced['status'] = df_combined_redeuced.groupby(['customer_id','yyyymm'])['months_from_last_txn'].rank(method = 'first', ascending = False)
#         print(f'[select_feature_set] 原始雙證人數: {len(df_combined_redeuced)}')
#         df_combined_redeuced = df_combined_redeuced[df_combined_redeuced['status']==1]
#         print(f'[select_feature_set] 移除重複後雙證人數: {len(df_combined_redeuced)}')
#         print(f'[select_feature_set] 富邦日盛移除重複 Runtime : {round(remove_dupli_id_time-time.time(),2)} sec')

#         # 以較小資料集，建模選取重要特徵
#         X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc = split_data(df_combined_redeuced, train_yyyymm=list(df_combined_redeuced['yyyymm'].sort_values().unique()[:-1]))
#         feat_imp = select_feature_v230517(X_train_rdc, y_train_rdc, X_test_rdc, y_test_rdc, this_file_path,
#                                           project_name, mother, target, new_edition, algorithm)
#         cols_top = feat_imp.loc[:n_feature]['feature'].tolist()

#         # 釋出空間
#         del X_train_rdc
#         del y_train_rdc
#         del X_test_rdc
#         del y_test_rdc
#         del df_combined_redeuced_fubon
#         del df_combined_redeuced_jihsun
#         del df_combined_redeuced
#         gc.collect()

#         # 以較大資料集，正式建模
#         based_col = ['customer_id', 'yyyymm', 'y']
#         X_train, y_train, X_test, y_test = split_data(df_combined[based_col+cols_top],
#                                                       train_yyyymm=list(df_combined['yyyymm'].sort_values().unique()[:-1]))
#         model, train_auc, test_auc,test_level_df, model_imp = build_model_v230726(X_train, y_train, X_test, y_test, cols_top, this_file_path,
#                                                                        project_name, mother, target, bins, new_edition, max_depth = max_depth_b,
#                                                                        scale_pos_weight = scale_pos_weight_b, n_estimator = n_estimator_b, algorithm= algorithm)
#         print(f'cols_impt_top{n_feature} : ')
#         display(model_imp)
#         print('test_level_df: ')
#         display(test_level_df)

#         # 以最新年月進行回測並產出效度
#         backtest_auc, vali_level_df, vali_predict_df, df_pred_and_features = back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]], bins = bins)
#         print(f'vali_level_df : ')
#         display(vali_level_df)
#         print(f'vali_predict_df.head() : ')
#         display(vali_predict_df.head())
#         print(f'vali_predict_df.shape :  {vali_predict_df.shape}\n')
#         print(f'df_pred_and_features.columns : \n {df_pred_and_features.columns}\n')
#         print(f'df_pred_and_features.shape : \n {df_pred_and_features.shape}\n')

#         # 以詳細切分方法進行最新年月回測並產出效度
#         backtest_auc_dt, vali_level_df_dt, vali_predict_df_dt, df_pred_and_features_dt =             back_test(df_combined, model, backtest_yyyymm=[df_combined['yyyymm'].sort_values().unique()[-1]],
#                   bins = [0,100,200,300,400,500,600,700,800,900,
#                           1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,
#                           11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,
#                           30000,40000,50000,60000,70000,80000,90000,100000,
#                           120000,140000,160000,180000,200000,
#                           1000000])
#         print(f'vali_level_df_dt : ')
#         display(vali_level_df_dt)
#         print(f'vali_predict_df_dt.head() : ')
#         display(vali_predict_df_dt.head())
#         print(f'vali_predivali_predict_df_dtct_df.shape :  {vali_predict_df_dt.shape}\n')
#         print(f'df_pred_and_features_dt.columns : \n {df_pred_and_features_dt.columns}\n')
#         print(f'df_pred_and_features_dt.shape : \n {df_pred_and_features_dt.shape}\n')

#         # 寫出PICKLE回測等級
#         pickle.dump(vali_predict_df_dt, open(this_file_path+'/'+mother+'/'+algorithm + '_vali_pred_df_detail' + project_name +'_' + str(new_edition) + '_' + mother + '.pickle' , 'wb'))

#         # 行銷相關才做轉換率
#         if market_flag_Y_N:

#             hit_rate_setting_bin = []
#             for ht in sorted([0]+hit_rate, reverse=True):
#                     if  ht == 0:
#                         hit_rate_setting_bin.append( "C{:02d}".format(int(round(ht * 100))) )
#                     elif  ht < 0.001:
#                         hit_rate_setting_bin.append( "C{:04d}".format(int(round(ht * 10000))) )
#                     elif  ht < 0.01:
#                         hit_rate_setting_bin.append( "C{:03d}".format(int(round(ht * 1000))) )
#                     else:
#                         hit_rate_setting_bin.append( "C{:02d}".format(int(round(ht * 100))) )

#             CHANCE_TEST_bin, CHANCE_TEST_num, CHANCE_TEST_hit_rate, CHANCE_TEST_prob = get_conversion_rank(vali_predict_df_dt, hit_rate)
#             # retrain_log
#             log_retrain = retrain_log_v240115(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                               train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_24opt_nn',
#                                               vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                               CHANCE_TEST_bin=CHANCE_TEST_bin, CHANCE_TEST_num=CHANCE_TEST_num,
#                                               CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate, CHANCE_TEST_prob=CHANCE_TEST_prob,
#                                               account=config.account_nn, pwd=config.pwd_nn)
#             print(f'retrain_log :')
#             display(log_retrain)

#             # model_log
#             log_model = model_log_v230410(False, project_name, mother, target, algorithm, hit_rate_setting_bin, new_edition, table_name = 'mlops_model_log_24opt_nn', account=config.account_nn, pwd=config.pwd_nn)
#             print(f'model_log :')
#             display(log_model)



#             log_retrain_v230410 = retrain_log_v240115(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                       vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_24opt_nn',
#                                                       vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                                       CHANCE_TEST_bin=CHANCE_TEST_bin, CHANCE_TEST_num=CHANCE_TEST_num,
#                                                       CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate, CHANCE_TEST_prob=CHANCE_TEST_prob,
#                                                       account=config.account_nn, pwd=config.pwd_nn)
#             log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, hit_rate_setting_bin, new_edition,
#                                                   table_name = 'mlops_model_log_24opt_nn', account=config.account_nn, pwd=config.pwd_nn)

#             # 額外寫出轉換率對照表
#             detail_hit_rate  = list(map(lambda x: round(x*0.0001,4), range(10)))[1:]  + list(map(lambda x: round(x*0.001,3), range(10)))[1:] + list(map(lambda x: round(x*0.01,2), range(26)))[1:]
#             CHANCE_TEST_bin_dt, CHANCE_TEST_num_dt, CHANCE_TEST_hit_rate_dt, CHANCE_TEST_prob_dt = get_conversion_rank(vali_predict_df_dt, detail_hit_rate)
#             conver_log = Convert_log_v240111(False, project_name, mother, df_combined,
#                                         backtest_auc, new_edition, table_name = 'mlops_convert_log_24opt_nn',
#                                        CHANCE_TEST_bin=CHANCE_TEST_bin_dt, CHANCE_TEST_num=CHANCE_TEST_num_dt,
#                                         CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate_dt, CHANCE_TEST_prob=CHANCE_TEST_prob_dt,
#                                              account=config.account_nn, pwd=config.pwd_nn)
#             print(f'conver_log :')
#             display(conver_log)
#             conver_log = Convert_log_v240111(write_db_Y_N, project_name, mother, df_combined,
#                                         backtest_auc, new_edition, table_name = 'mlops_convert_log_24opt_nn',
#                                        CHANCE_TEST_bin=CHANCE_TEST_bin_dt, CHANCE_TEST_num=CHANCE_TEST_num_dt,
#                                         CHANCE_TEST_hit_rate=CHANCE_TEST_hit_rate_dt, CHANCE_TEST_prob=CHANCE_TEST_prob_dt,
#                                              account=config.account_nn, pwd=config.pwd_nn)
#         else:
#             # retrain_log
#             log_retrain = retrain_log_v240115(False, project_name, mother, df_combined, X_train, y_train,test_level_df, vali_level_df,
#                                               train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_24opt_nn',
#                                               vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                               account=config.account_nn, pwd=config.pwd_nn
#                                               )
#             print(f'retrain_log :')
#             display(log_retrain)

#             # model_log
#             log_model = model_log_v230410(False, project_name, mother, target, algorithm, bins,new_edition, table_name = 'mlops_model_log_24opt_nn',
#                                               account=config.account_nn, pwd=config.pwd_nn)
#             print(f'model_log :')
#             display(log_model)



#             log_retrain_v230410 = retrain_log_v240115(write_db_Y_N, project_name, mother, df_combined, X_train, y_train,test_level_df,
#                                                       vali_level_df, train_auc, test_auc, backtest_auc, new_edition, table_name = 'mlops_retrain_log_24opt_nn',
#                                                       vali_level_df_detail=vali_level_df_dt, vali_predict_df_detail=vali_predict_df_dt,
#                                               account=config.account_nn, pwd=config.pwd_nn
#                                                       )
#             log_model_v230410 = model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition,
#                                                   table_name = 'mlops_model_log_24opt_nn',
#                                               account=config.account_nn, pwd=config.pwd_nn)

#         ## If retrain_auc < 0.7 then print

#         if test_auc < 0.7:
#             print('{}retrain_auc < 0.7'.format(mother))


#         print('{} 花了 {} min'.format(mother,(time.time()-totally_st_time)/60))


#         del X_train
#         del y_train
#         del X_test
#         del y_test
#         del df_combined_fubon
#         del df_combined_jihsun
#         del vali_predict_df
#         del df_pred_and_features
#         gc.collect()

#         print(f'{project_name}_{mother}_{new_edition} 執行完畢，先跳出迴圈!')


# In[19]:


get_ipython().system('jupyter nbconvert --to script retrain_mlops.ipynb')


# In[ ]:




