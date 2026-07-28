#!/usr/bin/env python

# In[1]:


import math
import sys

sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
import config
from Sql_module import get_SQL_raw_data

account, pwd = config.account, config.pwd
feature_trans_path = config.feature_trans_path
# In[2]:
import calendar
import os
import pickle
import time
from datetime import date, datetime
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from Sql_module import write_data_to_SQL
from sqlalchemy.types import Float, Integer, String
from xgboost import XGBClassifier

# In[3]:


def is_dataframe(var):
    return isinstance(var,pd.DataFrame)


# In[4]:


# 日期月份增減
def monthdelta(date, delta):
    m, y = (date.month+delta) % 12, date.year + ((date.month)+delta-1) // 12
    if not m:
        m = 12
    d = min(date.day, calendar.monthrange(y, m)[1])
    return date.replace(day=d,month=m, year=y)


# In[5]:


def split_data(df_combined, train_yyyymm, test_size=0.1):

    df_train_test = df_combined[df_combined['yyyymm'].isin(train_yyyymm)]

#     if df_train_test['y'].isna().sum()>0:
#         print("Y值na數",df_train_test['y'].isna().sum())
#         df_train_test['y'] = df_train_test['y'].fillna(0)

    t0 = time.time()
    print('data spliting.....')
    limit_size = 2000000
    if len(df_train_test) <= limit_size:
        print(f'split propotion of train size is {1-test_size} ')
        df_train, df_test = train_test_split(df_train_test, test_size=test_size, random_state=100, stratify=df_train_test['y'])
    else:
        test_size = 0.3
        print(f'split propotion of train size is {1-test_size} ')
        df_train, df_test = train_test_split(df_train_test, test_size=test_size, random_state=100, stratify=df_train_test['y'])


    drop_list = ['customer_id', 'yyyymm', 'y']
    X_train = df_train.drop(drop_list, axis=1)
    y_train = df_train['y']



    X_test = df_test.drop(drop_list, axis=1)
    y_test = df_test['y']

    print('train')
    print('mla', y_train.mean()*100, '%')

    print('test')
    print('mla', y_test.mean()*100, '%')
    t1 = time.time()
    print("Running time of split data: %.0f sec" %(t1 - t0))

    return X_train, y_train, X_test, y_test


# In[6]:


def model_fit(X_train, y_train, X_test, y_test, max_depth, scale_pos_weight, n_estimator,ear_stop_rounds = 200, cols = [], model = 'xgboost', gpu_use=True):

    # train model with full train dataset
    print('starting model fitting...')
    t0 = time.time()
    # 選擇模型
    if model == 'xgboost':
        if gpu_use:
            model = XGBClassifier(n_estimators=n_estimator, max_depth=max_depth, eval_metric='auc',
                                  scale_pos_weight=scale_pos_weight, colsample_bylevel = 0.8, subsample = 0.8,
                                  reg_lambda = 0.1, objective = 'reg:logistic', min_child_weight = 1,
                                  # USE CPU
                                  #tree_method='hist'
                                  # USE GPU
                                  tree_method='gpu_hist',
                                  enable_categorical=True,
                                  use_label_encoder=False)
        else:
            model = XGBClassifier(n_estimators=n_estimator, max_depth=max_depth, eval_metric='auc',
                      scale_pos_weight=scale_pos_weight, colsample_bylevel = 0.8, subsample = 0.8,
                      reg_lambda = 0.1, objective = 'reg:logistic', min_child_weight = 1,
                      # USE CPU
                      #tree_method='hist'
                      # USE GPU
                      enable_categorical=False,
                      use_label_encoder=False)
    elif model == 'catboost':
        model = CatBoostClassifier(eval_metric = 'AUC', #loss_function = 'Logloss',
                                   verbose = True)
    else:
        model = LGBMClassifier(seed=42, metric='auc', n_estimators=n_estimator, max_depth=max_depth,
                               scale_pos_weight=scale_pos_weight)
    # 若cols有框定特定特徵則挑選特定column
    if cols != []:
        X_train = X_train[cols]
        X_test = X_test[cols]
    # 建立模型
    if type(model) == CatBoostClassifier:
        cat_feat = list(X_train.select_dtypes(include=['category']).columns)
        X_train[cat_feat] = X_train[cat_feat].astype(str)
        X_test[cat_feat] = X_test[cat_feat].astype(str)
        my_model = model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
                       verbose=100, early_stopping_rounds=200, cat_features = cat_feat)
    else:
        my_model = model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
                           verbose=100, early_stopping_rounds=ear_stop_rounds)

    t1 = time.time()
    print("[model_fit] Running time of: %.0f sec" %(t1 - t0))

    return my_model


# In[7]:


def plot_feature_importances(clf, X_train, y_train, top_n=20, figsize=(8,8),
                             print_table_ascending=True, title="Feature Importances"):
    __name__ = "plot_feature_importances"
    import matplotlib.pyplot as plt
    import pandas as pd

    try:
        if not hasattr(clf, 'feature_importances_'):
            clf.fit(X_train.values, y_train.values.ravel())

            if not hasattr(clf, 'feature_importances_'):
                raise AttributeError(f"{clf.__class__.__name__} does not have feature_importances_ attribute")
    except (XGBoostError, LightGBMError, ValueError):
        clf.fit(X_train.values, y_train.values.ravel())

    feat_imp = pd.DataFrame({'importance':clf.booster_.feature_importance(importance_type = 'gain')}) if type(clf) == LGBMClassifier else pd.DataFrame({'importance':clf.feature_importances_})
    feat_imp['feature'] = X_train.columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]

    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize, color=(0.2, 0.4, 0.6))
    plt.xlabel('Feature Importance Score')
    #plt.yticks(fontproperties=myfont)
    plt.show()

    if print_table_ascending:
        #from IPython.display import display
        #print("Top {} features in descending order of importance".format(top_n))
        #display(feat_imp.sort_values(by='importance', ascending=False))
        feat_imp.sort_values(by='importance', ascending=False, inplace=True)

    return feat_imp


# In[8]:


def get_model_feature_and_imp(my_model):
    #確認模型名稱
    if type(my_model) == LGBMClassifier:
        cols = my_model.booster_.feature_name()
    elif type(my_model) == XGBClassifier:
        cols = my_model.get_booster()
        cols = cols.feature_names
    elif type(my_model) == CatBoostClassifier:
        cols = my_model.feature_names_

    #判斷模型,產生對應特徵重要性Dataframe
    if type(my_model) == XGBClassifier:
        feat_imp = pd.DataFrame({'importance':my_model.feature_importances_})
    else:
        feat_imp = pd.DataFrame({'importance':my_model.booster_.feature_importance(importance_type = 'gain')})

    feat_imp['feature'] = cols
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp.reset_index(inplace=True)
    feat_imp.drop('index',axis=1)

    return feat_imp


# In[9]:


def get_max_model_version(product,target,population,frequency,edition_detail,table_name,account=account, pwd=pwd):
    new_edition = ''

    #抓取現行版本
    query = f"""
    (SELECT edition,edition_detail
    FROM {table_name}
    WHERE product = '{product}' and target = '{target}' and population = '{population}' and frequency = '{frequency}'

    order by edition desc
    fetch first row only)
    """
    #確認該母體下是否通過
    df = get_SQL_raw_data(query=query, account=account, pwd=pwd)
    if isinstance(df, pd.DataFrame):
        if len(df) >= 1:
            max_edition = df['edition'][0]
    elif isinstance(df, str):
        raise SystemError(f'由於{df}因此中止程式')

    return max_edition


# In[10]:


def get_next_model_version(product,target,population,frequency,edition_detail,table_name,account=account, pwd=pwd):
    new_edition = ''

    #抓取現行版本 使用上次寫出的版本判斷
    query = f"""
    (SELECT edition,edition_detail
    FROM {table_name}
    WHERE product = '{product}' and target = '{target}' and population = '{population}' and frequency = '{frequency}' 

    order by pred_date desc
    fetch first row only)
    """
    #確認該母體下是否通過
    df = get_SQL_raw_data(query=query, account=account, pwd=pwd)
    if isinstance(df, pd.DataFrame):
        if len(df) >= 1:
            print(df.head())
            last_edition = df['edition'][0]
            last_edition_detail = df['edition_detail'][0]
            if edition_detail == last_edition_detail or '新資料' in edition_detail or 'retrain' in edition_detail or 'RETRAIN' in edition_detail:
                print(f'last_edition = {last_edition}')
                main_version = int(last_edition.split('.')[0])
                sub_version = int(last_edition.split('.')[1])

                new_edition = str(main_version)+"."+str(sub_version+1)
                print(f'new_edition = {new_edition}')
                print(f'已存在此版本，更新版本為{new_edition}')
            else:
                main_version = float(last_edition.split('.')[0])
                sub_version = float(last_edition.split('.')[1])
                new_edition =  str(main_version+1+(0.1))
                print(f'不存在此版本，更新版本為{new_edition}')
        elif len(df) == 0:
            print(f'設定為edition:1.1, 由於不存在此product:{product}/target:{target}/population:{population}/frequency:{frequency}')
            new_edition = '1.1'
    elif isinstance(df, str):
        raise SystemError(f'由於{df}因此中止程式')

#     try:
#     except Exception as e:
#         print('abcde')
#         print(e)
#         print(f'設定為edition:1.1, 由於不存在此product:{product}/target:{target}/population:{population}/frequency:{frequency}')
#         new_edition = '1.1'
    return new_edition


# In[11]:


def get_max_model_retrain_version(ym, retrain_table, project_name, mother,account=account, pwd=pwd):

    def quer_now_done_model(retrain_table, ym):
        return f"""
                select 模型名稱, 母體, test_period, max("版本") max_version
                from s_ianleong.{retrain_table}
                where "retrain日期" >= '{ym+'19'}'  
                group by 模型名稱, 母體, test_period

                """
#     def quer_now_done_model(retrain_table, ym):
#         return f"""
#                 select 模型名稱, 母體, test_period, max("版本") max_version
#                 from s_paohsiangwang.{retrain_table}
#                 group by 模型名稱, 母體, test_period

#                 """
    now_done_model = get_SQL_raw_data(quer_now_done_model(retrain_table, ym),account, pwd)
    print('now_done_model:')
    print(now_done_model)
    # 透過新增的now_done_model去選版本
    model_rule = now_done_model['模型名稱'] == project_name+'模型'
    mother_rule = now_done_model['母體'] == mother
    max_edition = now_done_model[model_rule & mother_rule]['max_version'].iloc[0]
    return max_edition


# In[12]:


def ntb_get_max_model_retrain_version(ym, retrain_table, project_name, mother):

    def quer_now_done_model(retrain_table, ym):
        return f"""
                select 模型名稱, 母體, test_period, max("版本") max_version
                from S_KRISYJCHEN.{retrain_table}
                where "retrain日期" >='{ym}' 
                group by 模型名稱, 母體, test_period
                order by test_period desc

                """
    now_done_model = get_SQL_raw_data(quer_now_done_model(retrain_table, ym), account = config.account_kris,pwd = config.pwd_kris)
    print('now_done_model:')
    print(now_done_model)
    # 透過新增的now_done_model去選版本
    model_rule = now_done_model['模型名稱'] == project_name+'模型'
    mother_rule = now_done_model['母體'] == mother
    max_edition = now_done_model[model_rule & mother_rule]['max_version'].iloc[0]
    return max_edition


# In[13]:


def select_feature(X_train, y_train, X_test, y_test, writing_path, project_name, mother, model='xgboost', gpu_use=True):
    cols = []
    # 設定路徑及檔名
    today = date.today().strftime('%Y%m%d')
    pre_model_fold = writing_path + '/' + mother
    pre_modal_filename = model + '_pre_' + project_name +'_' + str(today) + '_' + mother + '.pickle'
    pre_model_path = pre_model_fold +'/' + pre_modal_filename
    # 判斷資料夾是否存在
    if not os.path.exists(pre_model_fold):
        os.mkdir(pre_model_fold)
    # 今天的pre模型是否存在,若沒有則建立
    if not os.path.exists(pre_model_path):
        # 如果母體是潛力客群則抓非潛力前1000個特徵
        if '潛客' in mother and '非潛客' not in mother:
            pre_model_fold_nonpo = writing_path + '/' + '非' + mother
            pre_modal_filename_nonpo = model + '_pre_' + project_name +'_' + str(today) + '_' + '非' + mother + '.pickle'
            pre_model_path_nonpo = pre_model_fold_nonpo +'/' + pre_modal_filename_nonpo
            if os.path.exists(pre_model_path_nonpo):
                print('[select_feature] drop down to Top1000 feature by 非潛客 model')
                my_model_nonpo = pickle.load(open(pre_model_path_nonpo, 'rb'))
                feat_imp_nonpo = get_model_feature_and_imp(my_model_nonpo)
                cols = feat_imp_nonpo.loc[:1000]['feature'].tolist()
            else:
                print(f'[select_feature] didn"t have 非潛客 model by {pre_modal_filename_nonpo}')
                print('[select_feature] all feature can be selected!!!')
                cols = []
        # 不是潛力的話放全部特徵進去選擇
        else:
            print('[select_feature] all feature can be selected!!!')
            cols = []
        # 開始建模
        my_model = model_fit(X_train, y_train, X_test, y_test, max_depth=2, scale_pos_weight=4, n_estimator=100,ear_stop_rounds=50, cols = cols, model=model, gpu_use=gpu_use)
        print("[select_feature] save model and return feat_imp...")
        pickle.dump(my_model, open(pre_model_path , 'wb'))
    else:
        print('[select_feature] pre_model had been existing, loading...')
        my_model = pickle.load(open(pre_model_path, 'rb'))
        #確認模型獲取特徵
        if type(my_model) == LGBMClassifier:
            cols = my_model.booster_.feature_name()
        elif type(my_model) == XGBClassifier:
            cols = my_model.get_booster()
            cols = cols.feature_names
        elif type(my_model) == CatBoostClassifier:
            cols = my_model.feature_names_
    # 開始預測
    t0 = time.time()
    if cols != []:
        X_test = X_test[cols]
        X_train = X_train[cols]
    test_preds = my_model.predict_proba(X_test)[:,1]
    # test AUC
    print ('[select_feature] test CV-ROC=', roc_auc_score(y_test, test_preds))
    # 繪製重要特徵
    feat_imp = plot_feature_importances(my_model, X_train, y_train, top_n=20, print_table_ascending=True, title=my_model.__class__.__name__)

    # 產生特徵重要性dataframe
    feat_imp = get_model_feature_and_imp(my_model)

    t1 = time.time()
    print("[select_feature] predict_proba running time of: %.0f sec" %(t1 - t0))

    return feat_imp


# In[14]:


def get_pre_model_path(writing_path, project_name, mother, target, edition, algorithm):
    pre_model_fold = writing_path + '/' + mother
    # 判斷資料夾是否存在
    if not os.path.exists(pre_model_fold):
        os.mkdir(pre_model_fold)
    pre_modal_filename = algorithm + '_pre_' + project_name + '_' + target + '_' + mother + '_' + str(edition) + '.pickle'
    pre_model_path = pre_model_fold +'/' + pre_modal_filename
    return pre_model_path


# In[15]:


def select_feature_v230410(X_train, y_train, X_test, y_test, writing_path, project_name, mother, edition, model='xgboost', gpu_use=True):
    st_time = time.time()
    cols = []
    # 設定路徑及檔名
    pre_model_fold = writing_path + '/' + mother
    pre_modal_filename = model + '_pre_' + project_name +'_' + str(edition) + '_' + mother + '.pickle'
    pre_model_path = pre_model_fold +'/' + pre_modal_filename
    # 判斷資料夾是否存在
    if not os.path.exists(pre_model_fold):
        os.mkdir(pre_model_fold)
    # 今天的pre模型是否存在,若沒有則建立
    if not os.path.exists(pre_model_path):
#         # 如果母體是潛力客群則抓非潛力前1000個特徵
#         if '潛客' in mother and '非潛客' not in mother:
#             pre_model_fold_nonpo = writing_path + '/' + '非' + mother
#             pre_modal_filename_nonpo = model + '_pre_' + project_name +'_' + str(edition) + '_' + '非' + mother + '.pickle'
#             pre_model_path_nonpo = pre_model_fold_nonpo +'/' + pre_modal_filename_nonpo
#             if os.path.exists(pre_model_path_nonpo):
#                 print('[select_feature] drop down to Top1000 feature by 非潛客 model')
#                 my_model_nonpo = pickle.load(open(pre_model_path_nonpo, 'rb'))
#                 feat_imp_nonpo = get_model_feature_and_imp(my_model_nonpo)
#                 cols = feat_imp_nonpo.loc[:1000]['feature'].tolist()
#             else:
#                 print(f'[select_feature] didn"t have 非潛客 model by {pre_modal_filename_nonpo}')
#                 print(f'[select_feature] all feature can be selected!!!')
#                 cols = []
#         # 不是潛力的話放全部特徵進去選擇
#         else:
#             print(f'[select_feature] all feature can be selected!!!')
#             cols = []
        # 開始建模
        my_model = model_fit(X_train, y_train, X_test, y_test, max_depth=2, scale_pos_weight=4, n_estimator=100,ear_stop_rounds=50, cols = cols, model=model, gpu_use=gpu_use)
        print("[select_feature] save model and return feat_imp...")
        pickle.dump(my_model, open(pre_model_path , 'wb'))
    else:
        print('[select_feature] pre_model had been existing, loading...')
        my_model = pickle.load(open(pre_model_path, 'rb'))
        #確認模型獲取特徵
        if type(my_model) == LGBMClassifier:
            cols = my_model.booster_.feature_name()
        elif type(my_model) == XGBClassifier:
            cols = my_model.get_booster()
            cols = cols.feature_names
        elif type(my_model) == CatBoostClassifier:
            cols = my_model.feature_names_
    # 開始預測
    t0_predict_proba = time.time()
    if cols != []:
        X_test = X_test[cols]
        X_train = X_train[cols]
    test_preds = my_model.predict_proba(X_test)[:,1]
    # test AUC
    print ('[select_feature] test CV-ROC=', roc_auc_score(y_test, test_preds))
    # 繪製重要特徵
    feat_imp = plot_feature_importances(my_model, X_train, y_train, top_n=20, print_table_ascending=True, title=my_model.__class__.__name__)

    # 產生特徵重要性dataframe
    feat_imp = get_model_feature_and_imp(my_model)

    t1_predict_proba = time.time()
    print("[select_feature] predict_proba running time of: %.0f sec" %(t1_predict_proba - t0_predict_proba))

    end_time = time.time()
    print(f'[select_feature], totally runtime: {round((end_time-st_time)/60)} minutes')
    return feat_imp


# In[16]:


#TAG
def select_feature_v230517(X_train, y_train, X_test, y_test, writing_path, project_name, mother, target, edition, algorithm='xgboost', gpu_use=True):
    st_time = time.time()
    cols = []
    # 設定路徑及檔名
    pre_model_path = get_pre_model_path(writing_path, project_name, mother, target, edition, algorithm)
    # 今天的pre模型是否存在,若沒有則建立
#     if not os.path.exists(pre_model_path):
#         # 如果母體是潛力客群則抓非潛力前1000個特徵
#         if '潛客' in mother and '非潛客' not in mother:
#             pre_model_fold_nonpo = writing_path + '/' + '非' + mother
#             pre_modal_filename_nonpo = model + '_pre_' + project_name +'_' + str(edition) + '_' + '非' + mother + '.pickle'
#             pre_model_path_nonpo = pre_model_fold_nonpo +'/' + pre_modal_filename_nonpo
#             if os.path.exists(pre_model_path_nonpo):
#                 print('[select_feature] drop down to Top1000 feature by 非潛客 model')
#                 my_model_nonpo = pickle.load(open(pre_model_path_nonpo, 'rb'))
#                 feat_imp_nonpo = get_model_feature_and_imp(my_model_nonpo)
#                 cols = feat_imp_nonpo.loc[:1000]['feature'].tolist()
#             else:
#                 print(f'[select_feature] didn"t have 非潛客 model by {pre_modal_filename_nonpo}')
#                 print(f'[select_feature] all feature can be selected!!!')
#                 cols = []
#         # 不是潛力的話放全部特徵進去選擇
#         else:
#             print(f'[select_feature] all feature can be selected!!!')
#             cols = []
        # 開始建模
    my_model = model_fit(X_train, y_train, X_test, y_test, max_depth=2, scale_pos_weight=4, n_estimator=100,ear_stop_rounds=50, cols = cols, model=algorithm, gpu_use=gpu_use)
    print("[select_feature] save model and return feat_imp...")
    pickle.dump(my_model, open(pre_model_path , 'wb'))
#     else:
#         print('[select_feature] pre_model had been existing, loading...')
#         my_model = pickle.load(open(pre_model_path, 'rb'))
#         #確認模型獲取特徵
#         if type(my_model) == LGBMClassifier:
#             cols = my_model.booster_.feature_name()
#         elif type(my_model) == XGBClassifier:
#             cols = my_model.get_booster()
#             cols = cols.feature_names
#         elif type(my_model) == CatBoostClassifier:
#             cols = my_model.feature_names_
    # 開始預測
    t0_predict_proba = time.time()
    if cols != []:
        X_test = X_test[cols]
        X_train = X_train[cols]
    test_preds = my_model.predict_proba(X_test)[:,1]
    # test AUC
    print ('[select_feature] test CV-ROC=', roc_auc_score(y_test, test_preds))
    # 繪製重要特徵
    feat_imp = plot_feature_importances(my_model, X_train, y_train, top_n=20, print_table_ascending=True, title=my_model.__class__.__name__)

    # 產生特徵重要性dataframe
    feat_imp = get_model_feature_and_imp(my_model)

    t1_predict_proba = time.time()
    print("[select_feature] predict_proba running time of: %.0f sec" %(t1_predict_proba - t0_predict_proba))

    end_time = time.time()
    print(f'[select_feature], totally runtime: {round((end_time-st_time)/60)} minutes')
    return feat_imp


# In[17]:


def build_model(X_train, y_train, X_test, y_test, cols, writing_path, project_name,
                mother, bins, max_depth = 3, scale_pos_weight = 10, n_estimator = 100, model = 'xgboost', gpu_use=True):

    # 對資料進行預測
    my_model = model_fit(X_train, y_train, X_test, y_test, max_depth, scale_pos_weight, n_estimator,ear_stop_rounds=200, cols = cols, model = 'xgboost', gpu_use=gpu_use)
    # 產出效度
    train_preds = my_model.predict_proba(X_train[cols])[:,1]
    test_preds = my_model.predict_proba(X_test[cols])[:,1]
    print ('XGB train CV-ROC=', roc_auc_score(y_train, train_preds))
    print ('XGB test CV-ROC=', roc_auc_score(y_test, test_preds))
    # 產出Test的效度等級
    df_test_preds = pd.DataFrame(data={'model_pred': test_preds,
                                      'sorted_rank': test_preds,
                                      'y_true': y_test
                                      })
    df_test_preds['sorted_rank'] = df_test_preds['sorted_rank'].rank(ascending = False, method = 'first')
    df_test_preds['sorted_rank'] = df_test_preds['sorted_rank'].astype('int')
    bins_table_lst = []
    bins_table_lst.append(get_bins_table(df_test_preds, bins=bins, level_type='with_bin').groupby(['model_level']).    agg({'y_true': ['count', 'sum', 'mean']}).reset_index())
    test_level_df = pd.concat(bins_table_lst, ignore_index=True).sort_values(by='model_level',ascending = True)
    # 設置路徑及檔名
    today = date.today().strftime('%Y%m%d')
    model_fold = writing_path+ '/' + mother
    model_filename = model + '_' + project_name +'_' + str(today) + '_' + mother + '.pickle'
    # 資料夾不存在就建立
    if not os.path.exists(model_fold):
        os.mkdir(model_fold)
    # 寫入模型
    pickle.dump(my_model, open( model_fold + '/' + model_filename, 'wb'))

    # 獲取模型重要特徵by排序
    feat_imp = get_model_feature_and_imp(my_model)

    return my_model, roc_auc_score(y_test, test_preds), test_level_df, feat_imp


# In[18]:


def get_model_path(writing_path, project_name, mother, new_edition, target, algorithm):
    model_fold = writing_path+ '/' + mother
    # 資料夾不存在就建立
    if not os.path.exists(model_fold):
        os.mkdir(model_fold)
    new_model_filename = algorithm +'_' + project_name +'_' + target +'_' + mother + '_' + str(new_edition) +'.pickle'
    model_path = model_fold + '/'+ new_model_filename
    return model_path


# In[19]:


def whether_done_next_version(writing_path, project_name, mother, new_edition, target, algorithm, db_table_rt, db_table_md,
                              account=account, pwd=pwd):
    # 先抓取LOG
    from Sql_module import get_SQL_raw_data
    schema_rt = config.account.upper() if 'ntb' in db_table_rt.lower() else config.iaccount.upper()
    schema_md =config.account.upper() if 'ntb' in db_table_md.lower() else config.iaccount.upper()
    query_rt =     f"""
    select * from {schema_rt}.{db_table_rt} 
    where "模型名稱" = '{project_name+'模型'}' and "母體" = '{mother}' and "版本" = '{new_edition}'
    """
    query_md =     f"""
    select * from {schema_md}.{db_table_md} 
    where "模型名稱" = '{project_name+'模型'}' and "母體" = '{mother}' and "版本" = '{new_edition}'
    """
    df_rt = get_SQL_raw_data(query_rt, account=account, pwd=pwd)
    df_md = get_SQL_raw_data(query_md, account=account, pwd=pwd)

    # 在抓取模型PICKE
    model_path = get_model_path(writing_path, project_name, mother, new_edition, target, algorithm)
    print('----------------------------------------------------------------------------')
    print(f'判斷此({project_name}/{mother}/{new_edition})下模型資訊是否完整，完整則跳過:')
    print('----------------------------------------------------------------------------')
    if is_dataframe(df_rt) and is_dataframe(df_md):
        if len(df_rt) > 0 and  len(df_md) > 0 and os.path.exists(model_path):
            print('資料完整! 直接執行下一個母體!')
            return True
        else:
            if len(df_rt) == 0:
                print(f'{db_table_rt} 無資料')
            if len(df_md) == 0:
                print(f'{db_table_md} 無資料')
            if not os.path.exists(model_path):
                print(f'{model_path} 模型還未建立')
            print('因此繼續執行此母體!')
            return False
    else:
        print(f'[retrain檔LOG]: \n {df_rt}')
        print(f'[model檔LOG]: \n {df_md}')
        raise Exception('LOG結果抓取失敗')


# In[20]:


def whether_done_backtest(writing_path, project_name, mother, new_edition, test_period, target, algorithm, db_table_bktest,
                              account=account, pwd=pwd):
    # 先抓取LOG
    from Sql_module import get_SQL_raw_data
    query_bk =     f"""
    select * from {db_table_bktest} 
    where "模型名稱" = '{project_name+'模型'}' and "母體" = '{mother}' and "版本" = '{new_edition}' and test_period = '{test_period}'
    """

    df_bk = get_SQL_raw_data(query_bk, account=account, pwd=pwd)

    # 在抓取模型PICKE
    backtest_pickle_path = writing_path+'/'+mother+'/'+algorithm + '_vali_pred_df_detail' + project_name +'_' + str(new_edition) + '_' + mother + '含舊戶' +'.pickle'
    print('----------------------------------------------------------------------------')
    print(f'判斷此({project_name}/{mother}/{new_edition}/{test_period})下回測資訊是否完整，完整則跳過:')
    print('----------------------------------------------------------------------------')
    if is_dataframe(df_bk) :
        if len(df_bk) > 0 and os.path.exists(backtest_pickle_path):
            print('資料完整! 直接執行下一個母體!')
            return True
        else:
            if len(df_bk) == 0:
                print(f'{df_bk} 無資料')
            if not os.path.exists(backtest_pickle_path):
                print(f'{backtest_pickle_path} 模型還未建立，請先retrain')

            return False
    else:
        print(f'[backtest檔LOG]: \n {df_bk}')
        raise Exception('LOG結果抓取失敗')


# In[21]:


def whether_done_next_population_predict(writing_path, project_name, mother, target, algorithm, snap_date,
                                         db_table_rf, db_table_IL,account=account, pwd=pwd, view = 's_ianleong'):

    # 先抓取LOG
    from Sql_module import get_SQL_raw_data
    query_rf =     f"""
    select * from {view}.{db_table_rf} 
    where product = '{project_name}' and population = '{mother}' and snap_date = '{snap_date}'
    """
    query_IL =     f"""
    select count(*) from {view}.{db_table_IL} 
    where product = '{project_name}' and population = '{mother}' and snap_date = '{snap_date}'
    """
    df_rf = get_SQL_raw_data(query_rf, account=account, pwd=pwd)
    df_IL = get_SQL_raw_data(query_IL, account=account, pwd=pwd)

    print('----------------------------------------------------------------------------')
    print(f'判斷此({project_name}/{mother}/{snap_date})下名單資訊是否完整，完整則跳過:')
    print('----------------------------------------------------------------------------')
    if is_dataframe(df_rf) and is_dataframe(df_IL):
        if len(df_rf) > 0 and  len(df_IL) > 0 :
            print('資料完整! 直接執行下一個母體!')
            return True
        else:
            if len(df_rf) == 0:
                print(f'{db_table_rf} 無資料')
            if len(df_IL) == 0:
                print(f'{db_table_IL} 無資料')
            print('因此繼續執行此母體!')
            return False
    else:
        print(f'[ref_info檔LOG]: \n {df_rf}')
        print(f'[id_list檔LOG]: \n {df_IL}')
        raise Exception('LOG結果抓取失敗')


# In[22]:


def build_model_v230410(X_train, y_train, X_test, y_test, cols, writing_path, project_name,
                mother, bins, edition, max_depth = 3, scale_pos_weight = 10, n_estimator = 100, model = 'xgboost', gpu_use=True):

    st_time = time.time()
    # 對資料進行預測
    my_model = model_fit(X_train, y_train, X_test, y_test, max_depth, scale_pos_weight, n_estimator,ear_stop_rounds=200, cols = cols, model = 'xgboost', gpu_use=gpu_use)
    # 產出效度
    train_preds = my_model.predict_proba(X_train[cols])[:,1]
    test_preds = my_model.predict_proba(X_test[cols])[:,1]
    print ('XGB train CV-ROC=', roc_auc_score(y_train, train_preds))
    print ('XGB test CV-ROC=', roc_auc_score(y_test, test_preds))
    # 產出Test的效度等級
    df_test_preds = pd.DataFrame(data={'model_pred': test_preds,
                                      'sorted_rank': test_preds,
                                      'y_true': y_test
                                      })
    df_test_preds['sorted_rank'] = df_test_preds['sorted_rank'].rank(ascending = False, method = 'first')
    df_test_preds['sorted_rank'] = df_test_preds['sorted_rank'].astype('int')
    bins_table_lst = []
    bins_table_lst.append(get_bins_table(df_test_preds, bins=bins, level_type='with_bin').groupby(['model_level']).    agg({'y_true': ['count', 'sum', 'mean']}).reset_index())
    test_level_df = pd.concat(bins_table_lst, ignore_index=True).sort_values(by='model_level',ascending = True)
    # 設置路徑及檔名
    model_fold = writing_path+ '/' + mother
    model_filename = model + '_' + project_name +'_' + str(edition) + '_' + mother + '.pickle'
    # 資料夾不存在就建立
    if not os.path.exists(model_fold):
        os.mkdir(model_fold)
    # 寫入模型
    pickle.dump(my_model, open( model_fold + '/' + model_filename, 'wb'))

    # 獲取模型重要特徵by排序
    feat_imp = get_model_feature_and_imp(my_model)

    end_time = time.time()
    print(f'[build_model], totally runtime: {round((end_time-st_time)/60)} minutes')

    return my_model, roc_auc_score(y_test, test_preds), test_level_df, feat_imp


# In[23]:


#TAG
def build_model_v230517(X_train, y_train, X_test, y_test, cols, writing_path, project_name, mother, target
                        , bins, new_edition, max_depth = 3, scale_pos_weight = 10, n_estimator = 100, algorithm = 'xgboost', gpu_use=True):

    st_time = time.time()
    # 對資料進行預測
    my_model = model_fit(X_train, y_train, X_test, y_test, max_depth, scale_pos_weight, n_estimator,ear_stop_rounds=200, cols = cols, model = algorithm, gpu_use=gpu_use)
    # 產出效度
    train_preds = my_model.predict_proba(X_train[cols])[:,1]
    test_preds = my_model.predict_proba(X_test[cols])[:,1]
    print ('XGB train CV-ROC=', roc_auc_score(y_train, train_preds))
    print ('XGB test CV-ROC=', roc_auc_score(y_test, test_preds))
    # 產出Test的效度等級
    df_test_preds = pd.DataFrame(data={'model_pred': test_preds,
                                      'sorted_rank': test_preds,
                                      'y_true': y_test
                                      })
    df_test_preds['sorted_rank'] = df_test_preds['sorted_rank'].rank(ascending = False, method = 'first')
    df_test_preds['sorted_rank'] = df_test_preds['sorted_rank'].astype('int')
    bins_table_lst = []
    bins_table_lst.append(get_bins_table(df_test_preds, bins=bins, level_type='with_bin').groupby(['model_level']).    agg({'y_true': ['count', 'sum', 'mean']}).reset_index())
    test_level_df = pd.concat(bins_table_lst, ignore_index=True).sort_values(by='model_level',ascending = True)
    # 設置路徑及檔名
    model_path = get_model_path(writing_path, project_name, mother, new_edition, target, algorithm)
    # 寫入模型
    pickle.dump(my_model, open( model_path , 'wb'))

    # 繪製重要特徵
    feat_imp = plot_feature_importances(my_model, X_train, y_train, top_n=20, print_table_ascending=True, title=my_model.__class__.__name__)

    # 獲取模型重要特徵by排序
    feat_imp = get_model_feature_and_imp(my_model)
    feature_trans = pd.read_csv(feature_trans_path)
    model_imp = pd.merge(feat_imp, feature_trans, how='left', on=['feature'])
    end_time = time.time()
    print(f'[build_model], totally runtime: {round((end_time-st_time)/60)} minutes')

    return my_model, roc_auc_score(y_test, test_preds), test_level_df, feat_imp


# In[24]:


#TAG
def build_model_v230726(X_train, y_train, X_test, y_test, cols, writing_path, project_name, mother, target
                        , bins, new_edition, max_depth = 3, scale_pos_weight = 10, n_estimator = 100, algorithm = 'xgboost', gpu_use=True):

    st_time = time.time()
    # 對資料進行預測
    my_model = model_fit(X_train, y_train, X_test, y_test, max_depth, scale_pos_weight, n_estimator,ear_stop_rounds=200, cols = cols, model = algorithm, gpu_use=gpu_use)
    # 產出效度
    train_preds = my_model.predict_proba(X_train[cols])[:,1]
    test_preds = my_model.predict_proba(X_test[cols])[:,1]
    print ('XGB train CV-ROC=', roc_auc_score(y_train, train_preds))
    print ('XGB test CV-ROC=', roc_auc_score(y_test, test_preds))
    train_auc = roc_auc_score(y_train, train_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    # 產出Test的效度等級
    df_test_preds = pd.DataFrame(data={'model_pred': test_preds,
                                      'sorted_rank': test_preds,
                                      'y_true': y_test
                                      })
    df_test_preds['sorted_rank'] = df_test_preds['sorted_rank'].rank(ascending = False, method = 'first')
    df_test_preds['sorted_rank'] = df_test_preds['sorted_rank'].astype('int')
    bins_table_lst = []
    bins_table_lst.append(get_bins_table(df_test_preds, bins=bins, level_type='with_bin').groupby(['model_level']).    agg({'y_true': ['count', 'sum', 'mean']}).reset_index())
    test_level_df = pd.concat(bins_table_lst, ignore_index=True).sort_values(by='model_level',ascending = True)
    # 設置路徑及檔名
    model_path = get_model_path(writing_path, project_name, mother, new_edition, target, algorithm)
    # 寫入模型
    pickle.dump(my_model, open( model_path , 'wb'))

    # 繪製重要特徵
    feat_imp = plot_feature_importances(my_model, X_train, y_train, top_n=20, print_table_ascending=True, title=my_model.__class__.__name__)

    # 獲取模型重要特徵by排序
    feat_imp = get_model_feature_and_imp(my_model)
    feature_trans = pd.read_csv(feature_trans_path)
    model_imp = pd.merge(feat_imp, feature_trans, how='left', on=['feature'])
    end_time = time.time()
    print(f'[build_model], totally runtime: {round((end_time-st_time)/60)} minutes')

    return my_model, train_auc, test_auc, test_level_df, feat_imp


# In[25]:


def build_model_today_exist(X_train, y_train, X_test, y_test, cols, writing_path, project_name,
                mother, bins, max_depth = 3, scale_pos_weight = 10, n_estimator = 100, model = 'xgboost', gpu_use=True):

    # 設置路徑及檔名
    today = date.today().strftime('%Y%m%d')
    model_fold = writing_path+ '/' + mother
    model_filename = model + '_' + project_name +'_' + str(today) + '_' + mother + '.pickle'
    path_model_file = model_fold + '/' + model_filename
    if not os.path.exists(path_model_file):
        if not os.path.exists(model_fold):
            os.mkdir(model_fold)
        # 對資料進行預測
        my_model = model_fit(X_train, y_train, X_test, y_test, max_depth, scale_pos_weight, n_estimator,ear_stop_rounds=200, cols = cols, model = 'xgboost', gpu_use=gpu_use)
        # 寫入模型
        pickle.dump(my_model, open(path_model_file, 'wb'))
    else:
        print('[build_model_today_exist] model had been existing, loading...')
        my_model = pickle.load(open(path_model_file, 'rb'))

    # 產出效度
    train_preds = my_model.predict_proba(X_train[cols])[:,1]
    test_preds = my_model.predict_proba(X_test[cols])[:,1]
    print ('XGB train CV-ROC=', roc_auc_score(y_train, train_preds))
    print ('XGB test CV-ROC=', roc_auc_score(y_test, test_preds))
    # 產出Test的效度等級
    df_test_preds = pd.DataFrame(data={'model_pred': test_preds,
                                      'sorted_rank': test_preds,
                                      'y_true': y_test
                                      })
    df_test_preds['sorted_rank'] = df_test_preds['sorted_rank'].rank(ascending = False, method = 'first')
    df_test_preds['sorted_rank'] = df_test_preds['sorted_rank'].astype('int')
    bins_table_lst = []
    bins_table_lst.append(get_bins_table(df_test_preds, bins=bins, level_type='with_bin').groupby(['model_level']).    agg({'y_true': ['count', 'sum', 'mean']}).reset_index())
    test_level_df = pd.concat(bins_table_lst, ignore_index=True).sort_values(by='model_level',ascending = True)


    # 獲取模型重要特徵by排序
    feat_imp = get_model_feature_and_imp(my_model)

    return my_model, roc_auc_score(y_test, test_preds), test_level_df, feat_imp


# In[26]:


def get_bins_table(df, bins, level_type='level', range_scale=False):
    # 產出 model level欄位
    # level_type == 'with_bin'  ------> 等級+bins
    # level_type == 'level'  ------> 等級
    df_tmp = df.copy()
    labels = []
    if level_type == 'with_bin':
        print('[get_bins_table] this labels with bin')
        for i in range(len(bins) - 1):
            if type(bins[1]) == int or float(bins[1])%1== 0:
                labels.append('等級' + str(i + 1) + '_rank_' + str(bins[i]) + '-' + str(bins[i+1]))
            elif type(bins[1]) == float:
                labels.append('等級' + str(i + 1) + '_proba_' + str(bins[i]) + '-' + str(bins[i+1]))
            elif type(bins[1]) == str:
                labels.append('等級' + str(i + 1) + '_quant_' + str(bins[i]) + '-' + str(bins[i+1]))
    elif level_type == 'level':
        print('[get_bins_table] this labels just have level')
        for i in range(len(bins) - 1):
            labels.append(str(i + 1))
    print(f'labels: {labels}')
    if (type(bins[1]) == int or float(bins[1])%1== 0) and '.' not in str(bins[1]):
        print('[get_bins_table] get model_level by rank')
        df_tmp['sorted_rank'] = df['model_pred'].rank(ascending = False, method = 'first')
        df_tmp['sorted_rank'] = df_tmp['sorted_rank'].astype('int')
        df_tmp['model_level'] = pd.cut(df_tmp['sorted_rank'], bins=bins, labels=labels, include_lowest=True)
        df_tmp.drop(['sorted_rank'], axis=1, inplace=True)
    elif type(bins[1]) == float:
        print('[get_bins_table] get model_level by probability')
        df_tmp['model_pred_inverse'] = 1.0 - df['model_pred']
        bins_inverse=[]
        for index,bi in enumerate(bins): bins_inverse.append(1.0-bi)
        df_tmp['model_level'] = pd.cut(df_tmp['model_pred_inverse'], bins=bins_inverse, labels=labels, include_lowest=True)
        df_tmp.drop(['model_pred_inverse'], axis=1, inplace=True)
        # (0, 0.1]
    elif type(bins[1]) == str:
        print('[get_bins_table] get model_level by recall')
        df_tmp['model_pred_inverse'] = 1.0 - df['model_pred']
        bins_inverse_recall=[]
        recall_pred_list = [ round(sum(df_tmp['y_true'])*(1-float(b)) ) for b in bins]
        bins_inverse_recall.append(0)
        recall_pred_list.remove(0)
        recall_pred_list.remove(sum(df_tmp['y_true']))
        print(recall_pred_list)
        n=0
        for mp in sorted(df_tmp['model_pred_inverse']):
            if n < len(recall_pred_list):
                if sum(df_tmp['y_true'][df_tmp['model_pred_inverse']<= mp]) >= recall_pred_list[n]:
                    n=n+1
                    bins_inverse_recall.append(mp)
        bins_inverse_recall.append(1.0)
        print(bins_inverse_recall)
        df_tmp['model_level'] = pd.cut(df_tmp['model_pred_inverse'], bins=bins_inverse_recall, labels=labels, include_lowest=True)
        df_tmp.drop(['model_pred_inverse'], axis=1, inplace=True)
#     elif type(bins[1]) == str:
#         print('[get_bins_table] get model_level by recall')
#         bins_inverse_quantile=[]
#         df_tmp['model_pred_inverse'] = 1.0 - df['model_pred']
#         for index,bi in enumerate(bins): bins_inverse_quantile.append(df_tmp['model_pred_inverse'].quantile(1.0-float(bi)))
#         df_tmp['model_level'] = pd.cut(df_tmp['model_pred_inverse'], bins=bins_inverse_quantile, labels=labels, include_lowest=True)
#         df_tmp.drop(['model_pred_inverse'], axis=1, inplace=True)
    return df_tmp


# In[27]:


def model_predict(df_combined, my_model, predict_yyyymm):
    st_time = time.time()
    #確認模型
    if type(my_model) == LGBMClassifier:
        cols = my_model.booster_.feature_name()
    elif type(my_model) == XGBClassifier:
        cols = my_model.get_booster()
        cols = cols.feature_names
    elif type(my_model) == CatBoostClassifier:
        cols = my_model.feature_names_
    #篩選月份
    df_combined = df_combined[df_combined['yyyymm'].isin(predict_yyyymm)]
    X = df_combined[df_combined['yyyymm'].isin(predict_yyyymm)].drop(['customer_id', 'yyyymm', 'y'], axis=1)
    Y = df_combined['y'][df_combined['yyyymm'].isin(predict_yyyymm)]

    #進行預測
    if type(my_model) == CatBoostClassifier:
        cat_feat = list(X[cols].select_dtypes(include=['category']).columns)
        X[cat_feat] = X[cat_feat].astype(str)

    current_preds = my_model.predict_proba(X[cols])[:,1]
    end_time = time.time()
    print("[model_predict] Running time of back test: %.0f sec" %(end_time - st_time))
    return current_preds



# In[28]:


def back_test(df_combined, my_model, backtest_yyyymm, bins):
    st_time = time.time()
    # 對回測月份進行預測產出機率
    current_preds = model_predict(df_combined, my_model, backtest_yyyymm)
    df_combined_bt = df_combined[df_combined['yyyymm'].isin(backtest_yyyymm)]
    # Valiation auc
    print ('XGB Valid CV-ROC=', roc_auc_score(df_combined_bt['y'], current_preds))

    # 獲取模型重要特徵by排序
    feat_imp = get_model_feature_and_imp(my_model)
    cols = feat_imp['feature'].tolist()

    # 模型預測結果
    df_current_preds = pd.DataFrame(data={'yyyymm': df_combined_bt['yyyymm'],
                                          'customer_id': df_combined_bt['customer_id'],
                                          'model_pred': current_preds,
                                          'y_true': df_combined_bt['y']
                                         })
    df_current_preds = get_bins_table(df_current_preds, bins=bins ,level_type = 'with_bin')
    print(bins)
    # 回測Dataframe + 特徵TOP
    show_columns = ['customer_id', 'yyyymm', 'y']
    show_columns = show_columns + cols
    df_pred_and_features = df_combined_bt[show_columns]
    df_pred_and_features = df_pred_and_features.rename({'y': 'y_true'}, axis=1)
    df_pred_and_features['model_pred'] = df_current_preds['model_pred']
    df_pred_and_features['model_level'] = df_current_preds['model_level']

    # 等級效度dataframe
    bins_table_lst = []
    bins_table_lst.append(get_bins_table(df_current_preds, bins=bins, level_type='with_bin').groupby(['yyyymm', 'model_level']).    agg({'y_true': ['count', 'sum', 'mean']}).reset_index())
    print(bins)
    level_df = pd.concat(bins_table_lst, ignore_index=True).sort_values(by='model_level',ascending = True)
    #測試鎖月份
    level_df = level_df[level_df['yyyymm'].isin(backtest_yyyymm)]
    end_time = time.time()
    print("[back_test] Running time of back test: %.0f sec" %(end_time - st_time))
    return roc_auc_score(df_combined_bt['y'], current_preds), level_df, df_current_preds, df_pred_and_features


# In[29]:


def retrain_log(write_db_Y_N, project_name, mother, df_combined, X_train, y_train, level_df, test_auc, backtest_auc):
    #retrain_log
    retrain_log = pd.DataFrame({
                            '模型名稱' :project_name + '模型',
                            '母體' : mother,
                            'retrain日期' :datetime.today().strftime('%Y%m%d %H:%M:%S'),
                            '訓練集區間' :str(list(df_combined['yyyymm'].sort_values().unique()[:-1])),
                            '測試集區間' :str(list(df_combined['yyyymm'].sort_values().unique()[:-1])),
                            '驗證集區間' :str(df_combined['yyyymm'].sort_values().unique()[-1]),
                            '測試集auc' :test_auc,
                            '驗證集auc' :backtest_auc,
                            '母體人數' :int(X_train.shape[0]),
                            'y=1人數' :int(sum(y_train)),
                            '等級1人數' :level_df.reset_index().loc[0].y_true['count'],
                            '等級1購買率' :level_df.reset_index().loc[0].y_true['mean'],
                            '是否通過標準' : test_auc > 0.7
                                }, index=[0])

    if write_db_Y_N :
        table_name = 'retrain_log'
        col_types = {
            '模型名稱' : String(30),
            'retrain日期' : String(30),
            '訓練集區間' : String(50),
            '測試集區間': String(50),
            '驗證集區間': String(30),
            '測試集auc': Float(),
            '驗證集auc': Float(),
            '母體人數': Integer(),
            'y=1人數': Integer(),
            '是否通過標準': String(30)
        }

        # Output to DB
        write_data_to_SQL(table_name, retrain_log, account, pwd,  chunk_size=10000, col_types=col_types)
    else:
        print('[retrain_log]don"t require for writing down to db ')
    return retrain_log


# In[30]:


def retrain_log_v230314(write_db_Y_N, project_name, mother, df_combined, X_train, y_train, test_level_df, vali_level_df, test_auc, backtest_auc):
    #retrain_log
    retrain_log = pd.DataFrame({
                            '模型名稱' :project_name + '模型',
                            '母體' : mother,
                            'retrain日期' :datetime.today().strftime('%Y%m%d %H:%M:%S'),
                            '訓練集區間' :str(list(df_combined['yyyymm'].sort_values().unique()[:-1])),
                            '測試集區間' :str(list(df_combined['yyyymm'].sort_values().unique()[:-1])),
                            '驗證集區間' :str(df_combined['yyyymm'].sort_values().unique()[-1]),
                            '測試集auc' :test_auc,
                            '驗證集auc' :backtest_auc,
                            '模型母體人數' :int(X_train.shape[0]),
                            '模型y=1人數' :int(sum(y_train)),
                            '測試集人數' :str(list(test_level_df.reset_index().y_true['count'])),
                            '測試集購買率' :str(list(test_level_df.reset_index().y_true['mean'])),
                            '驗證集人數' :str(list(vali_level_df.reset_index().y_true['count'])),
                            '驗證集購買率' :str(list(vali_level_df.reset_index().y_true['mean'])),
                            '是否通過標準' : test_auc > 0.7
                                }, index=[0])

    if write_db_Y_N :
        table_name = 'retrain_log_version0314'
        col_types = {
            '模型名稱' : String(30),
            '母體' : String(30),
            'retrain日期' : String(30),
            '訓練集區間' : String(50),
            '測試集區間': String(50),
            '驗證集區間': String(30),
            '測試集auc': Float(),
            '驗證集auc': Float(),
            '模型母體人數': Integer(),
            '模型y=1人數': Integer(),
            '測試集人數' :String(200),
            '測試集購買率' :String(200),
            '驗證集人數' :String(200),
            '驗證集購買率' :String(200),
            '是否通過標準': String(30)
        }

        # Output to DB
        write_data_to_SQL(table_name, retrain_log, account, pwd,  chunk_size=10000, col_types=col_types)
    else:
        print('[retrain_log]don"t require for writing down to db ')
    return retrain_log


# In[31]:


def retrain_log_v230410(write_db_Y_N, project_name, mother, df_combined, X_train, y_train, test_level_df, vali_level_df,
                        test_auc, backtest_auc, new_edition, table_name):


    #retrain_log
    retrain_log = pd.DataFrame({
                            '模型名稱' :project_name + '模型',
                            '母體' : mother,
                            'retrain日期' :datetime.today().strftime('%Y%m%d %H:%M'),
                            '訓練集區間' :str(list(df_combined['yyyymm'].sort_values().unique()[:-1])),
                            '測試集區間' :str(list(df_combined['yyyymm'].sort_values().unique()[:-1])),
                            '驗證集區間' :str(df_combined['yyyymm'].sort_values().unique()[-1]),
                            '測試集auc' :test_auc,
                            '驗證集auc' :backtest_auc,
                            '訓練母體人數' :int(X_train.shape[0]),
                            '訓練母體y=1人數' :int(sum(y_train)),
                            '測試集人數' :str(list(test_level_df.reset_index().y_true['count'])),
                            '測試集購買率' :str(  [round(x,6) for x  in list(test_level_df.reset_index().y_true['mean'])]  ),
                            '驗證集人數' :str(list(vali_level_df.reset_index().y_true['count'])),
                            '驗證集購買率' :str(  [round(x,6) for x  in list(vali_level_df.reset_index().y_true['mean'])]  ),
                            '是否通過標準' : test_auc > 0.7,
                            '版本':new_edition
                                }, index=[0])

    if write_db_Y_N :

        col_types = {
            '模型名稱' : String(30),
            '母體' : String(30),
            'retrain日期' : String(30),
            '訓練集區間' : String(250),
            '測試集區間': String(250),
            '驗證集區間': String(30),
            '測試集auc': Float(),
            '驗證集auc': Float(),
            '訓練母體人數': Integer(),
            '訓練母體y=1人數': Integer(),
            '測試集人數' :String(200),
            '測試集購買率' :String(200),
            '驗證集人數' :String(200),
            '驗證集購買率' :String(200),
            '是否通過標準': String(30),
            '版本':String(10)
        }

        # Output to DB
        write_data_to_SQL(table_name, retrain_log, account, pwd, chunk_size=10000, col_types=col_types)
    else:
        print('[retrain_log]don"t require for writing down to db ')
    return retrain_log


# In[32]:


def retrain_log_v230726(write_db_Y_N, project_name, mother, df_combined, X_train, y_train, test_level_df, vali_level_df,
                        train_auc, test_auc, backtest_auc, new_edition, table_name ,
                        vali_level_df_detail=pd.DataFrame(), vali_predict_df_detail=pd.DataFrame(),
                        account=account, pwd=pwd):

    if not vali_level_df_detail.empty:
        TEST_number_detail = str(list(vali_level_df_detail.reset_index().y_true['count']))
        TEST_hit_rate_detail = str(  [round(x,6) for x  in list(vali_level_df_detail.reset_index().y_true['mean'])]  )
    else:
        TEST_number_detail = np.nan
        TEST_hit_rate_detail = np.nan

#     if not vali_predict_df.empty:
    if not vali_predict_df_detail.empty:
        vali_level_df_prob = pd.DataFrame(vali_predict_df_detail.groupby('model_level')['model_pred'].min()).reset_index()
        vali_level_df_prob = vali_level_df_prob.sort_values(by='model_pred',ascending=False)
        vali_level_df_prob.model_pred = vali_level_df_prob.model_pred.apply(lambda x: round(x, 7))
        TEST_bin_prob_detail = str(list(vali_level_df_prob.model_pred))

    else:
        TEST_bin_prob_detail = np.nan

    #retrain_log
    retrain_log = pd.DataFrame({
                            '模型名稱' :project_name + '模型',
                            '母體' : mother,
                            'retrain日期' :datetime.today().strftime('%Y%m%d %H:%M'),
                            'train_period' :str(list(df_combined['yyyymm'].sort_values().unique()[:-1])),
                            'valid_period' :str(list(df_combined['yyyymm'].sort_values().unique()[:-1])),
                            'test_period' :str(df_combined['yyyymm'].sort_values().unique()[-1]),
                            'valid_auc' :test_auc,
                            'test_auc' :backtest_auc,
                            '訓練母體人數' :int(X_train.shape[0]),
                            '訓練母體y=1人數' :int(sum(y_train)),
                            'VALID_人數' :str(list(test_level_df.reset_index().y_true['count'])),
                            'VALID_購買率' :str(  [round(x,6) for x  in list(test_level_df.reset_index().y_true['mean'])]  ),
                            'TEST_人數' :str(list(vali_level_df.reset_index().y_true['count'])),
                            'TEST_購買率' :str(  [round(x,6) for x  in list(vali_level_df.reset_index().y_true['mean'])]  ),
                            '是否通過標準' : test_auc > 0.7,
                            '版本':new_edition ,
                            'train_auc': train_auc,
                            'TEST_人數_DETAIL' : TEST_number_detail,
                            'TEST_購買率_DETAIL' : TEST_hit_rate_detail,
                            'TEST_切分機率_DETAIL' :TEST_bin_prob_detail
                                }, index=[0])

    if write_db_Y_N :

        col_types = {
            '模型名稱' : String(30),
            '母體' : String(30),
            'retrain日期' : String(30),
            'train_period' : String(250),
            'valid_period': String(250),
            'test_period': String(30),
            'valid_auc': Float(),
            'test_auc': Float(),
            '訓練母體人數': Integer(),
            '訓練母體y=1人數': Integer(),
            'VALID_人數' :String(200),
            'VALID_購買率' :String(200),
            'TEST_人數' :String(200),
            'TEST_購買率' :String(200),
            '是否通過標準': String(30),
            '版本':String(10),
            'train_auc': Float(),
            'TEST_人數_DETAIL' :String(400),
            'TEST_購買率_DETAIL' :String(400),
            'TEST_切分機率_DETAIL' :String(600)
        }

        # Output to DB
        write_data_to_SQL(table_name, retrain_log, account, pwd, chunk_size=10000, col_types=col_types)
    else:
        print('[retrain_log]don"t require for writing down to db ')
    return retrain_log


# In[33]:


def retrain_log_v240115(write_db_Y_N, project_name, mother, df_combined, X_train, y_train, test_level_df, vali_level_df,
                        train_auc, test_auc, backtest_auc, new_edition, table_name ,
                        vali_level_df_detail=pd.DataFrame(), vali_predict_df_detail=pd.DataFrame(),
                        CHANCE_TEST_bin=np.nan, CHANCE_TEST_num=np.nan, CHANCE_TEST_hit_rate=np.nan, CHANCE_TEST_prob=np.nan,
                        account=account, pwd=pwd):

    if not vali_level_df_detail.empty:
        TEST_number_detail = str(list(vali_level_df_detail.reset_index().y_true['count']))
        TEST_hit_rate_detail = str(  [round(x,6) for x  in list(vali_level_df_detail.reset_index().y_true['mean'])]  )
    else:
        TEST_number_detail = np.nan
        TEST_hit_rate_detail = np.nan

#     if not vali_predict_df.empty:
    if not vali_predict_df_detail.empty:
        vali_level_df_prob = pd.DataFrame(vali_predict_df_detail.groupby('model_level')['model_pred'].min()).reset_index()
        vali_level_df_prob = vali_level_df_prob.sort_values(by='model_pred',ascending=False)
        vali_level_df_prob.model_pred = vali_level_df_prob.model_pred.apply(lambda x: round(x, 7))
        TEST_bin_prob_detail = str(list(vali_level_df_prob.model_pred))

    else:
        TEST_bin_prob_detail = np.nan

    #retrain_log
    retrain_log = pd.DataFrame({
                            '模型名稱' :project_name + '模型',
                            '母體' : mother,
                            'retrain日期' :datetime.today().strftime('%Y%m%d %H:%M'),
                            'train_period' :str(list(df_combined['yyyymm'].sort_values().unique()[:-1])),
                            'valid_period' :str(list(df_combined['yyyymm'].sort_values().unique()[:-1])),
                            'test_period' :str(df_combined['yyyymm'].sort_values().unique()[-1]),
                            'valid_auc' :test_auc,
                            'test_auc' :backtest_auc,
                            '訓練母體人數' :int(X_train.shape[0]),
                            '訓練母體y=1人數' :int(sum(y_train)),
                            'VALID_人數' :str(list(test_level_df.reset_index().y_true['count'])),
                            'VALID_購買率' :str(  [round(x,6) for x  in list(test_level_df.reset_index().y_true['mean'])]  ),
                            'TEST_人數' :str(list(vali_level_df.reset_index().y_true['count'])),
                            'TEST_購買率' :str(  [round(x,6) for x  in list(vali_level_df.reset_index().y_true['mean'])]  ),
                            '是否通過標準' : test_auc > 0.7,
                            '版本':new_edition ,
                            'train_auc': train_auc,
                            'TEST_人數_DETAIL' : TEST_number_detail,
                            'TEST_購買率_DETAIL' : TEST_hit_rate_detail,
                            'TEST_切分機率_DETAIL' :TEST_bin_prob_detail,
                            'CHANCE_TEST_切分分數' : str(CHANCE_TEST_bin),
                            'CHANCE_TEST_人數' : str(CHANCE_TEST_num),
                            'CHANCE_TEST_購買率' : str(CHANCE_TEST_hit_rate),
                            'CHANCE_TEST_切分機率': str(CHANCE_TEST_prob)
                                }, index=[0])

    if write_db_Y_N :

        col_types = {
            '模型名稱' : String(30),
            '母體' : String(30),
            'retrain日期' : String(30),
            'train_period' : String(250),
            'valid_period': String(250),
            'test_period': String(30),
            'valid_auc': Float(),
            'test_auc': Float(),
            '訓練母體人數': Integer(),
            '訓練母體y=1人數': Integer(),
            'VALID_人數' :String(200),
            'VALID_購買率' :String(200),
            'TEST_人數' :String(200),
            'TEST_購買率' :String(200),
            '是否通過標準': String(30),
            '版本':String(10),
            'train_auc': Float(),
            'TEST_人數_DETAIL' :String(400),
            'TEST_購買率_DETAIL' :String(400),
            'TEST_切分機率_DETAIL' :String(600),
            'CHANCE_TEST_切分分數' : String(400),
            'CHANCE_TEST_人數' : String(400),
            'CHANCE_TEST_購買率' : String(400),
            'CHANCE_TEST_切分機率': String(400)
        }

        # Output to DB
        write_data_to_SQL(table_name, retrain_log, account, pwd, chunk_size=10000, col_types=col_types)
    else:
        print('[retrain_log]don"t require for writing down to db ')
    return retrain_log


# In[34]:


def backtest_log_v231018(write_db_Y_N, project_name, mother, df_combined, vali_level_df,
                         backtest_auc, edition, table_name ,
                         vali_level_df_detail=pd.DataFrame(), vali_predict_df_detail=pd.DataFrame(),
                         CHANCE_TEST_bin=np.nan, CHANCE_TEST_num=np.nan, CHANCE_TEST_hit_rate=np.nan, CHANCE_TEST_prob=np.nan
                         ):

    if not vali_level_df_detail.empty:
        TEST_number_detail = str(list(vali_level_df_detail.reset_index().y_true['count']))
        TEST_hit_rate_detail = str(  [round(x,6) for x  in list(vali_level_df_detail.reset_index().y_true['mean'])]  )
    else:
        TEST_number_detail = np.nan
        TEST_hit_rate_detail = np.nan

    if not vali_predict_df_detail.empty:
        vali_level_df_prob = pd.DataFrame(vali_predict_df_detail.groupby('model_level')['model_pred'].min()).reset_index()
        vali_level_df_prob = vali_level_df_prob.sort_values(by='model_pred',ascending=False)
        vali_level_df_prob.model_pred = vali_level_df_prob.model_pred.apply(lambda x: round(x, 7))
        TEST_bin_prob_detail = str(list(vali_level_df_prob.model_pred))

    else:
        TEST_bin_prob_detail = np.nan

    #backtest_log
    backtest_log = pd.DataFrame({
                            '模型名稱' :project_name + '模型',
                            '母體' : mother,
                            'backtest日期' :datetime.today().strftime('%Y%m%d %H:%M'),
                            'test_period' :str(df_combined['yyyymm'].sort_values().unique()[-1]),
                            'test_auc' :backtest_auc,
                            'TEST_人數' :str(list(vali_level_df.reset_index().y_true['count'])),
                            'TEST_購買率' :str(  [round(x,6) for x  in list(vali_level_df.reset_index().y_true['mean'])]  ),
                            '版本':edition ,
                            'TEST_人數_DETAIL' : TEST_number_detail,
                            'TEST_購買率_DETAIL' : TEST_hit_rate_detail,
                            'TEST_切分機率_DETAIL' :TEST_bin_prob_detail,
                            'CHANCE_TEST_切分分數' : str(CHANCE_TEST_bin),
                            'CHANCE_TEST_人數' : str(CHANCE_TEST_num),
                            'CHANCE_TEST_購買率' : str(CHANCE_TEST_hit_rate),
                            'CHANCE_TEST_切分機率': str(CHANCE_TEST_prob)
                                }, index=[0])

    if write_db_Y_N :

        col_types = {
            '模型名稱' : String(30),
            '母體' : String(30),
            'backtest日期' : String(30),
            'test_period': String(30),
            'test_auc': Float(),
            'TEST_人數' :String(200),
            'TEST_購買率' :String(200),
            '版本':String(10),
            'TEST_人數_DETAIL' :String(400),
            'TEST_購買率_DETAIL' :String(400),
            'TEST_切分機率_DETAIL' :String(600),
            'CHANCE_TEST_切分分數' : String(400),
            'CHANCE_TEST_人數' : String(400),
            'CHANCE_TEST_購買率' : String(400),
            'CHANCE_TEST_切分機率': String(400)
        }

        # Output to DB
        write_data_to_SQL(table_name, backtest_log, account, pwd, chunk_size=10000, col_types=col_types)
    else:
        print('[backtest_log]don"t require for writing down to db ')
    return backtest_log


# In[35]:


def model_log_v230410(write_db_Y_N, project_name, mother, target, algorithm, bins, new_edition, table_name,
                      account=account, pwd=pwd):
    if type(bins[0]) == str:
        if 'C' in bins[0]:
            model_log = pd.DataFrame({
                                '模型名稱' : project_name + '模型',
                                'retrain日期' : datetime.today().strftime('%Y%m%d %H:%M'),
                                '母體' : mother,
                                'target': target,
                                '等級數': len(bins)-1,
                                '演算法': algorithm,
                                '切分分數': str(bins),
                                '版本':new_edition
                                }, index=[0])
        else:
            model_log = pd.DataFrame({
                                '模型名稱' : project_name + '模型',
                                'retrain日期' : datetime.today().strftime('%Y%m%d %H:%M'),
                                '母體' : mother,
                                'target': target,
                                '等級數': len(bins),
                                '演算法': algorithm,
                                '切分分數': str(bins),
                                '版本':new_edition
                                }, index=[0])
    else:
        model_log = pd.DataFrame({
                            '模型名稱' : project_name + '模型',
                            'retrain日期' : datetime.today().strftime('%Y%m%d %H:%M'),
                            '母體' : mother,
                            'target': target,
                            '等級數': len(bins),
                            '演算法': algorithm,
                            '切分分數': str(bins),
                            '版本':new_edition
                            }, index=[0])

    if write_db_Y_N :
        col_types = {
            '模型名稱' : String(30),
            'retrain日期' : String(30),
            '母體' : String(30),
            'target': String(30),
            '等級數': Integer(),
            '演算法': String(20),
            '切分分數': String(60),
            '版本':String(10)
        }

        # Output to DB
        write_data_to_SQL(table_name, model_log, account, pwd, chunk_size=10000, col_types=col_types)
    else:
        print('[model_log]don"t require for writing down to db ')
    return model_log


# In[36]:


def model_log(write_db_Y_N, project_name, mother, target, algorithm, bins):
    model_log = pd.DataFrame({
                        '模型名稱' : project_name + '模型',
                        'yyyymm' : datetime.today().strftime('%Y%m%d %H:%M:%S'),
                        '母體' : mother,
                        'target': target,
                        '等級數': len(bins)-1,
                        '演算法': algorithm,
                        '切分分數': str(bins)
                        }, index=[0])

    if write_db_Y_N :
        table_name = 'model_log'
        col_types = {
            '模型名稱' : String(30),
            'yyyymm' : String(30),
            '母體' : String(30),
            'target': String(30),
            '等級數': Integer(),
            '演算法': String(20),
            '切分分數': String(50)
        }

        # Output to DB
        write_data_to_SQL(table_name, model_log, account, pwd, chunk_size=10000, col_types=col_types)
    else:
        print('[model_log]don"t require for writing down to db ')
    return model_log


# In[37]:


def predict_log(write_db_Y_N, project_name, mother, ym, df_combined):
    predict_log = pd.DataFrame({'模型名稱' : project_name,
                    '母體' : mother,
                    '預測日期' : datetime.today().strftime('%Y%m%d %H:%M:%S'),
                    '資料年月' : str(ym),
                    '母體人數' : df_combined.shape[0],
                    '是否預測' : 1
                   }, index=[0])
    if write_db_Y_N:
        table_name = 'predict_log'
        col_types = {
            '模型名稱' : String(15),
            '母體' : String(15),
            '預測日期' : String(15),
            '資料年月' : String(10),
            '母體人數': String(30),
            '是否預測' : String(15)
        }
        # Output to DB
        write_data_to_SQL(table_name, predict_log, account, pwd, chunk_size=10000, col_types=col_types)
    else:
        print('[predict_log]don"t require for writing down to db ')
    return predict_log


# In[38]:


def save_db_mlops(write_db_Y_N, level, table_name, account=account, pwd=pwd ):

    col_types = {
        'model_name': String(15),
        'population': String(6),
        'pred_date':  String(20),
        'yyyymm' : String(6),
        'customer_id' : String(20),
        'model_pred': Float(),
        'model_level': String(20)
    }

    # Output to DB
    if write_db_Y_N:
        write_data_to_SQL(table_name, level, account, pwd, chunk_size=10000, col_types=col_types)
    else:
        print('[save_db_mlops]don"t require for writing down to db ')


# In[39]:


def confirm_retrain_log(project_name, mother_list, table_name, account=account, pwd=pwd, view = 's_ianleong'):
    confirm_list = []
    model_name = project_name+'模型'
    print(f'[{model_name}]starting check model performance by {mother_list} ...')
    for mother in mother_list:
        #評估是否需預測
        query = f"""
        SELECT 是否通過標準,"retrain日期", test_period
        FROM {view}.{table_name}
        WHERE 模型名稱 = '{model_name}' and 母體 = '{mother}' 
        order by "retrain日期" desc
        fetch first row only
        """
        #確認該母體下是否通過
        retrain_result = get_SQL_raw_data(query, account, pwd)
        print(retrain_result)
        if (retrain_result['是否通過標準'][0] !='1'):
            confirm_list.append(False)
        else:
            confirm_list.append(True)
        time.sleep(2)
    # 判斷每個母體下的效度是否通過
    if sum(confirm_list) != len(mother_list):
        log = pd.DataFrame({'模型名稱' : project_name,
                        '母體' : '',
                        '預測日期' : datetime.today().strftime('%Y%m%d %H:%M:%S'),
                        '資料年月' : '',
                        '母體人數' : '',
                        '是否預測' : 0
                       }, index=[0])
        save_predict_log(log, project_name='predict_log')
        raise Exception('Retrain未達標準')
    print(f'[{model_name}] check completed !!!')


# In[40]:


def get_bin_with_dif_mother(project_name, mother_list, table_name, account=account, pwd=pwd):
    bins_list = []
    bins0_list = []
    model_name = project_name+'模型'
    print(f'[{model_name}]starting get bin by {mother_list} ...')
    for mother in mother_list:
        query = f"""
        with list_ as(
            select *
            from s_ianleong.{table_name}
            where "模型名稱" = '{model_name}' and "母體" = '{mother}'
            order by yyyymm desc
        )
        select "切分分數"
        from list_
        where rownum = 1
        """
        bins0 = get_SQL_raw_data(query, account, pwd)
        bins0_list.append(bins0)
        bins0 = bins0.iloc[0,0]
        bins0 = bins0.strip('"[]"')
        bins = [a for a in bins0.split(', ')]
        if "'" in bins[1]:
            bins = [a.strip("'") for a in bins]
        elif float(bins[1])%1 == 0:
            bins = [int(float(a)) for a in bins]
        elif float(bins[1])%1 != 0:
            bins = [float(a) for a in bins]

        bins_list.append(bins)
        print(f'{mother} bins: {bins}')
        time.sleep(2)
    print(f'[{model_name}]get finished !!!')
    return bins_list


# In[41]:


def get_bin_with_dif_mother_version230417(project_name, mother_list, table_name, account=account, pwd=pwd, view = 's_ianleong'):
    bins_list = []
    bins0_list = []
    model_name = project_name+'模型'
    print(f'[{model_name}]starting get bin by {mother_list} ...')
    for mother in mother_list:
        query = f"""
        with list_ as(
            select *
            from {view}.{table_name}
            where "模型名稱" = '{model_name}' and "母體" = '{mother}'
            order by "retrain日期" desc
        )
        select "切分分數"
        from list_
        where rownum = 1
        """
        bins0 = get_SQL_raw_data(query, account, pwd)
        bins0_list.append(bins0)
        bins0 = bins0.iloc[0,0]
        bins0 = bins0.strip('"[]"')
        bins = [a for a in bins0.split(', ')]
        mappling_table = table_name.replace('model','retrain')
        if 'C' in bins[1]:
            bins = [a.strip("'") for a in bins]
        elif "'" in bins[1]:
            bins = [a.strip("'") for a in bins]
            query = f"""
            with list_ as(
                select *
                from {view}.{mappling_table}
                where "模型名稱" = '{model_name}' and "母體" = '{mother}'
                order by "retrain日期" desc
            )
            select "TEST_人數"
            from list_
            where rownum = 1
            """
            bins0 = get_SQL_raw_data(query, account, pwd)
            bins0_list.append(bins0)
            bins0 = bins0.iloc[0,0]
            bins0 = bins0.strip('"[]"')
            bins_bycover = [a for a in bins0.split(', ')]
            bins_bycover = [int(float(a)) for a in bins_bycover]
            bins = []
            bins.append(0)
            for i,b in enumerate(bins_bycover):
                bins.append(sum(bins_bycover[0:i+1]))
            bins[-1] = 5000000
        elif float(bins[1])%1 == 0:
            bins = [int(float(a)) for a in bins]
        elif float(bins[1])%1 != 0:
            bins = [float(a) for a in bins]

        bins_list.append(bins)
        print(f'{mother} bins: {bins}')
        time.sleep(2)

    print(f'[{model_name}]get finished !!!')
    return bins_list


# In[42]:


def get_bin_with_dif_mother_conversion(project_name, mother_list, edition_list, table_name):
    bins_list = []
    bins_label_list = []
    model_name = project_name+'模型'
    if 'backtest' in table_name:
        sorted_col = 'backtest日期'
    elif 'retrain' in table_name:
        sorted_col = 'retrain日期'
    print(f'[{model_name}]starting get bin by {mother_list} ...')
    for index, mother in enumerate(mother_list):
        edition = edition_list[index]
        query = f"""
        with list_ as(
            select *
            from s_ianleong.{table_name}
            where "模型名稱" = '{model_name}' and "母體" = '{mother}' and "版本" = '{edition}'
            order by "{sorted_col}" desc
        )
        select "CHANCE_TEST_切分機率","CHANCE_TEST_切分分數"
        from list_
        where rownum = 1
        """
        cut_table = get_SQL_raw_data(query=query)
        # 切分機率
        bins0 = cut_table.iloc[0,0]
        bins0 = bins0.strip('"[]"')
        bins = [a for a in bins0.split(', ')]
        check_index = math.floor(len(bins)/2)
        if float(bins[check_index])%1 == 0:
            bins = [int(float(a)) for a in bins]
        elif float(bins[check_index])%1 != 0:
            bins = [float(a) for a in bins]
        # 插入最大機率
        bins.insert(0, 1.0)
        # 取代最小機率
        bins[-1] = 0.0

        bins_list.append(bins)
        # 切分標籤
        bins_label = cut_table.iloc[0,1]
        bins_label = bins_label.strip('"[]"')
        bins_label = [a for a in bins_label.replace("'",'').split(', ')]


        bins_label_list.append(bins_label)

        print(f'{mother} bins: {bins} bins_label:{bins_label}')
        time.sleep(2)

    print(f'[{model_name}]get finished !!!')
    return bins_list, bins_label_list


# In[43]:


def predict_df(df_combined, predict_yyyymm, writing_path, project_name,
               mother, algorithm, bins, papulation_colname, papulation_except_value):
    # 抓取最新的模型路徑
    model_fold = writing_path+ '/' + mother
    file_list = [f for f in listdir(model_fold) if isfile(join(model_fold, f))]
    model_list = [f for f in file_list if 'pickle' in f and algorithm in f and 'pre' not in f and mother in f]
    model_date_list = [int(str(a).split('_')[2]) for a in model_list]
    new_model_date = max(model_date_list)
    new_model_filename = algorithm +'_' + project_name +'_' + str(new_model_date) + '_' + mother +'.pickle'
    model_path = model_fold + '/'+ new_model_filename
    # 讀取模型
    print(f'[predict_df]Loading model with {model_path} ...')
    model = pickle.load(open(model_path, 'rb'))
    print('[predict_df]Starting predicting ...')
    # 抓取母體
    df_combined_papulation = df_combined[~df_combined[papulation_colname].isin(papulation_except_value)]
    # 預測機率
    current_preds = model_predict(df_combined_papulation, model, predict_yyyymm)
    # 獲取模型重要特徵by排序
    feat_imp = get_model_feature_and_imp(model)
    cols = feat_imp['feature'].tolist()
    # 抓需要預測的年月資料
    df_combined_pt = df_combined_papulation[df_combined_papulation['yyyymm'].isin(predict_yyyymm)]
    pred_date = datetime.today().strftime('%Y%m%d %H:%M:%S')
    # 產出預測 dataframe
    df_current_preds = pd.DataFrame(data={
        'model_name': project_name,
        'population': mother,
        'pred_date':  pred_date,
        'yyyymm': df_combined_pt['yyyymm'],
        'customer_id': df_combined_pt['customer_id'],
        'model_pred': current_preds
    })

    # 各等級下人數 dataframe
    df_level_with_bin = get_bins_table(df_current_preds.drop('customer_id', axis=1), bins=bins, level_type = 'with_bin')
    df_level_with_bin_num =  df_level_with_bin.groupby(['yyyymm', 'model_level','model_name','population','pred_date']).count().reset_index().sort_values(by='model_level',ascending = False)
    df_level_with_bin_num = df_level_with_bin_num.rename({'model_pred': 'num'}, axis=1)


    # 預測結果 dataframe
    df_predict = get_bins_table(df_current_preds, bins=bins ,level_type = 'level')
    df_predict['model_level'] = df_predict['model_level'].astype('str')
    # 除了預測母體以外的 dataframe
    except_name =''
    if papulation_except_value == [1]:
        except_name = papulation_colname
    else:
        except_name = '不是'+ papulation_colname
    df_papulation_except = pd.DataFrame(data={
        'model_name': project_name,
        'population': mother,
        'pred_date':  pred_date,
        'yyyymm': df_combined['yyyymm'][df_combined[papulation_colname].isin(papulation_except_value)],
        'customer_id':  df_combined['customer_id'][df_combined[papulation_colname].isin(papulation_except_value)],
        'model_pred': 999,
        'model_level': except_name
    })
    # 結合預測結果跟母體以外的表格((寫入DB))
    df_to_db = df_predict.append(df_papulation_except)

    # 預測Dataframe + 特徵TOP
    show_columns = ['customer_id', 'yyyymm', 'y']
    show_columns = show_columns + cols
    df_pred_and_features = df_combined_pt[show_columns]
    df_pred_and_features = df_pred_and_features.rename({'y': 'y_true'}, axis=1)
    df_pred_and_features['model_pred'] = df_predict['model_pred']
    df_pred_and_features['model_level'] = df_predict['model_level']

    print('[predict_df] predict finished and return table !!!')

    return df_level_with_bin_num, df_to_db , df_pred_and_features



# In[44]:


def predict_df_v230314(df_combined, predict_yyyymm, writing_path, project_name, mother, algorithm, bins,
                       papulation_colname, papulation_except_value, target, frequency, edition_detail):
    # 抓取最新的模型路徑
    model_fold = writing_path+ '/' + mother
    file_list = [f for f in listdir(model_fold) if isfile(join(model_fold, f))]
    model_list = [f for f in file_list if 'pickle' in f and algorithm in f and 'pre' not in f and mother in f]
    model_date_list = [int(str(a).split('_')[2]) for a in model_list]
    new_model_date = max(model_date_list)
    new_model_filename = algorithm +'_' + project_name +'_' + str(new_model_date) + '_' + mother +'.pickle'
    model_path = model_fold + '/'+ new_model_filename
    # 讀取模型
    print(f'[predict_df]Loading model with {model_path} ...')
    model = pickle.load(open(model_path, 'rb'))
    print('[predict_df]Starting predicting ...')
    # 抓取母體
    df_combined_papulation = df_combined[~df_combined[papulation_colname].isin(papulation_except_value)]
    # 預測機率
    current_preds = model_predict(df_combined_papulation, model, predict_yyyymm)
    # 獲取模型重要特徵by排序
    feat_imp = get_model_feature_and_imp(model)
    cols = feat_imp['feature'].tolist()
    # 抓需要預測的年月資料
    df_combined_pt = df_combined_papulation[df_combined_papulation['yyyymm'].isin(predict_yyyymm)]

    # 計算該月最後一天
    year=predict_yyyymm[0][0:4]
    month=predict_yyyymm[0][4:6]
    end_day = calendar.monthrange(int(year),int(month))[1]
    date_last_day = year+'/'+month+'/'+str(end_day)
    # 預測時間
    pred_date = datetime.today().strftime('%Y/%m/%d %H:%M:%S')
    # 產出預測 dataframe
    df_current_preds = pd.DataFrame(data={
        'product': project_name,
        'target': target ,
        'population': mother,
        'frequency': frequency,
        'pred_date': pred_date,
        'snap_date':  date_last_day,
        'customer_id': df_combined_pt['customer_id'],
        'tag_name': project_name+target,
        'model_pred': current_preds,

    })

    # 各等級下人數 dataframe
    df_level_with_bin = get_bins_table(df_current_preds.drop('customer_id', axis=1), bins=bins, level_type = 'with_bin')
    df_level_with_bin_num =  df_level_with_bin.groupby(list(df_level_with_bin.drop('model_pred', axis=1).columns)).count().reset_index().sort_values(by='model_level',ascending = False)
    df_level_with_bin_num = df_level_with_bin_num.rename({'model_pred': 'num'}, axis=1)

     ################################### 以下寫入DB#######################################################
    #  主數據庫 TABLE1:參照資訊檔 #
    #############################
    exception_text = papulation_colname
    if '舊戶' in papulation_colname: exception_text = '舊戶均註記為舊戶'
    else: exception_text = '非母體均註記為未達' + papulation_colname + '條件'
    ref_info_to_db = pd.DataFrame(data={
        'product': project_name,
        'target': target ,
        'population': mother,
        'frequency': frequency,
        'edition': '',
        'pred_date': pred_date,
        'last_tag_date':  date_last_day,
        'tag_name': project_name+target,
        'exception': exception_text ,
        'edition_detail': edition_detail,
        'pred_date': pred_date
    },index=[0])
    ###########################
    #  主數據庫 TABLE1:名單檔  #
    ###########################
    # 預測結果 dataframe
    df_predict = get_bins_table(df_current_preds, bins=bins ,level_type = 'level')
    df_predict['model_level'] = df_predict['model_level'].astype('str')
    df_predict = df_predict.rename({'model_level': 'rank','model_pred':'probability'}, axis=1)
    # 除了預測母體以外的 dataframe
    except_name =''
    if papulation_except_value == [1]:
        except_name = papulation_colname
    else:
        except_name = '不是'+ papulation_colname
    df_papulation_except = pd.DataFrame(data={
        'product': project_name,
        'target': target ,
        'population': mother,
        'frequency': frequency,
        'pred_date': pred_date,
        'snap_date':  date_last_day,
        'customer_id': df_combined['customer_id'][df_combined[papulation_colname].isin(papulation_except_value)],
        'tag_name': project_name+target,
        'probability': np.nan,
        'rank': except_name
    })

    # 結合預測結果跟母體以外的表格((寫入DB))
    id_list_to_db = df_predict.append(df_papulation_except)
    ################################### 以上寫入DB#######################################################

    # 預測Dataframe + 特徵TOP
    show_columns = ['customer_id', 'yyyymm', 'y']
    show_columns = show_columns + cols
    df_pred_and_features = df_combined_pt[show_columns]
    df_pred_and_features = df_pred_and_features.rename({'y': 'y_true'}, axis=1)
    df_pred_and_features['model_pred'] = df_predict['probability']
    df_pred_and_features['model_level'] = df_predict['rank']

    print('[predict_df] predict finished and return table !!!')

    return df_level_with_bin_num, df_pred_and_features, id_list_to_db, ref_info_to_db



# In[45]:


def predict_df_v230410(df_combined, predict_yyyymm, writing_path, project_name, mother, algorithm, bins,papulation_colname,
                       papulation_except_value, target, frequency, edition_detail, new_edition):
    # 抓取最新的模型路徑
    model_fold = writing_path+ '/' + mother
    new_model_filename = algorithm +'_' + project_name +'_' + str(new_edition) + '_' + mother +'.pickle'
    model_path = model_fold + '/'+ new_model_filename
    # 讀取模型
    print(f'[predict_df]Loading model with {model_path} ...')
    model = pickle.load(open(model_path, 'rb'))
    print('[predict_df]Starting predicting ...')
    # 抓取母體
    df_combined_papulation = df_combined[~df_combined[papulation_colname].isin(papulation_except_value)]
    # 預測機率
    current_preds = model_predict(df_combined_papulation, model, predict_yyyymm)
    # 獲取模型重要特徵by排序
    feat_imp = get_model_feature_and_imp(model)
    cols = feat_imp['feature'].tolist()
    # 抓需要預測的年月資料
    df_combined_pt = df_combined_papulation[df_combined_papulation['yyyymm'].isin(predict_yyyymm)]

    # 計算該月最後一天
    year=predict_yyyymm[0][0:4]
    month=predict_yyyymm[0][4:6]
    end_day = calendar.monthrange(int(year),int(month))[1]
    date_last_day = year+'/'+month+'/'+str(end_day)
    # 預測時間
    pred_date = datetime.today().strftime('%Y/%m/%d %H:%M')
    # 產出預測 dataframe
    df_current_preds = pd.DataFrame(data={
        'product': project_name,
        'target': target ,
        'population': mother,
        'frequency': frequency,
        'pred_date': pred_date,
        'snap_date':  date_last_day,
        'party_id': df_combined_pt['customer_id'],
        'tag_name': project_name+target,
        'model_pred': current_preds,

    })

    # 各等級下人數 dataframe
    df_level_with_bin = get_bins_table(df_current_preds.drop('party_id', axis=1), bins=bins, level_type = 'with_bin')
    df_level_with_bin_num =  df_level_with_bin.groupby(list(df_level_with_bin.drop('model_pred', axis=1).columns)).count().reset_index().sort_values(by='model_level',ascending = False)
    df_level_with_bin_num = df_level_with_bin_num.rename({'model_pred': 'num'}, axis=1)

     ################################### 以下寫入DB#######################################################
    #  主數據庫 TABLE1:參照資訊檔 #
    #############################
    exception_text = papulation_colname
    if '舊戶' in papulation_colname: exception_text = '舊戶均註記為舊戶'
    else: exception_text = '非母體均註記為未達' + papulation_colname + '條件'
    ref_info_to_db = pd.DataFrame(data={
        'product': project_name,
        'target': target ,
        'population': mother,
        'frequency': frequency,
        'edition': str(new_edition),
        'snap_date': date_last_day,
        'model_valid_falg':'',
        'tag_name': project_name+target,
        'exception': exception_text ,
        'edition_detail': edition_detail,
        'pred_date': pred_date
    },index=[0])
    ###########################
    #  主數據庫 TABLE1:名單檔  #
    ###########################
    # 預測結果 dataframe
    df_predict = get_bins_table(df_current_preds, bins=bins ,level_type = 'level')
    df_predict['model_level'] = df_predict['model_level'].astype('str')
    df_predict = df_predict.rename({'model_level': 'rank','model_pred':'probability'}, axis=1)
    # 除了預測母體以外的 dataframe
    except_name =''
    if papulation_except_value == [1]:
        except_name = papulation_colname
    else:
        except_name = '不是'+ papulation_colname
    df_papulation_except = pd.DataFrame(data={
        'product': project_name,
        'target': target ,
        'population': mother,
        'frequency': frequency,
        'snap_date':  date_last_day,
        'party_id': df_combined['customer_id'][df_combined[papulation_colname].isin(papulation_except_value)],
        'tag_name': project_name+target,
        'probability': np.nan,
        'rank': except_name,
        'pred_date': pred_date,
    })
    df_predict = df_predict[['product','target','population','frequency','snap_date','party_id','tag_name','probability',
        'rank','pred_date']]
    # 結合預測結果跟母體以外的表格((寫入DB))
    id_list_to_db = df_predict.append(df_papulation_except)
    ################################### 以上寫入DB#######################################################

    # 預測Dataframe + 特徵TOP
    show_columns = ['customer_id', 'yyyymm', 'y']
    show_columns = show_columns + cols
    df_pred_and_features = df_combined_pt[show_columns]
    df_pred_and_features = df_pred_and_features.rename({'y': 'y_true'}, axis=1)
    df_pred_and_features['model_pred'] = df_predict['probability']
    df_pred_and_features['model_level'] = df_predict['rank']

    print('[predict_df] predict finished and return table !!!')

    return df_level_with_bin_num, df_pred_and_features, id_list_to_db, ref_info_to_db



# In[46]:


#TAG
def predict_df_v230517(df_combined, predict_yyyymm, writing_path, project_name, mother, algorithm, bins,papulation_colname,
                       papulation_except_value, target, frequency, edition_detail, new_edition):
    # 抓取最新的模型路徑
    model_path = get_model_path(writing_path, project_name, mother, new_edition, target, algorithm)
    # 讀取模型
    print(f'[predict_df]Loading model with {model_path} ...')
    model = pickle.load(open(model_path, 'rb'))
    print('[predict_df]Starting predicting ...')
    # 抓取母體
    df_combined_papulation = df_combined[~df_combined[papulation_colname].isin(papulation_except_value)]
    # 預測機率
    current_preds = model_predict(df_combined_papulation, model, predict_yyyymm)
    # 獲取模型重要特徵by排序
    feat_imp = get_model_feature_and_imp(model)
    cols = feat_imp['feature'].tolist()
    # 抓需要預測的年月資料
    df_combined_pt = df_combined_papulation[df_combined_papulation['yyyymm'].isin(predict_yyyymm)]

    # 計算該月最後一天
    year=predict_yyyymm[0][0:4]
    month=predict_yyyymm[0][4:6]
    end_day = calendar.monthrange(int(year),int(month))[1]
    date_last_day = year+'/'+month+'/'+str(end_day)
    # 預測時間
    pred_date = datetime.today().strftime('%Y/%m/%d %H:%M')
    # 產出預測 dataframe
    df_current_preds = pd.DataFrame(data={
        'product': project_name,
        'target': target ,
        'population': mother,
        'frequency': frequency,
        'pred_date': pred_date,
        'snap_date':  date_last_day,
        'party_id': df_combined_pt['customer_id'],
        'tag_name': project_name+target,
        'model_pred': current_preds,

    })

    # 各等級下人數 dataframe
    df_level_with_bin = get_bins_table(df_current_preds.drop('party_id', axis=1), bins=bins, level_type = 'with_bin')
    df_level_with_bin_num =  df_level_with_bin.groupby(list(df_level_with_bin.drop('model_pred', axis=1).columns)).count().reset_index().sort_values(by='model_level',ascending = False)
    df_level_with_bin_num = df_level_with_bin_num.rename({'model_pred': 'num'}, axis=1)

     ################################### 以下寫入DB#######################################################
    #  主數據庫 TABLE1:參照資訊檔 #
    #############################
    exception_text = papulation_colname
    if '舊戶' in papulation_colname: exception_text = '舊戶均註記為舊戶'
    else: exception_text = '非母體均註記為未達' + papulation_colname + '條件'
    ref_info_to_db = pd.DataFrame(data={
        'product': project_name,
        'target': target ,
        'population': mother,
        'frequency': frequency,
        'edition': str(new_edition),
        'snap_date': date_last_day,
        'model_valid_falg':'',
        'tag_name': project_name+target,
        'exception': exception_text ,
        'edition_detail': edition_detail,
        'pred_date': pred_date
    },index=[0])
    ###########################
    #  主數據庫 TABLE1:名單檔  #
    ###########################
    # 預測結果 dataframe
    df_predict = get_bins_table(df_current_preds, bins=bins ,level_type = 'level')
    df_predict['model_level'] = df_predict['model_level'].astype('str')
    df_predict = df_predict.rename({'model_level': 'rank','model_pred':'probability'}, axis=1)
    # 除了預測母體以外的 dataframe
    except_name =''
    if papulation_except_value == [1]:
        except_name = papulation_colname
    else:
        except_name = '不是'+ papulation_colname
    df_papulation_except = pd.DataFrame(data={
        'product': project_name,
        'target': target ,
        'population': mother,
        'frequency': frequency,
        'snap_date':  date_last_day,
        'party_id': df_combined['customer_id'][df_combined[papulation_colname].isin(papulation_except_value)],
        'tag_name': project_name+target,
        'probability': np.nan,
        'rank': except_name,
        'pred_date': pred_date,
    })
    df_predict = df_predict[['product','target','population','frequency','snap_date','party_id','tag_name','probability',
        'rank','pred_date']]
    # 結合預測結果跟母體以外的表格((寫入DB))
    id_list_to_db = df_predict.append(df_papulation_except)
    ################################### 以上寫入DB#######################################################

    # 預測Dataframe + 特徵TOP
    show_columns = ['customer_id', 'yyyymm', 'y']
    show_columns = show_columns + cols
    df_pred_and_features = df_combined_pt[show_columns]
    df_pred_and_features = df_pred_and_features.rename({'y': 'y_true'}, axis=1)
    df_pred_and_features['model_pred'] = df_predict['probability']
    df_pred_and_features['model_level'] = df_predict['rank']

    print('[predict_df] predict finished and return table !!!')

    return df_level_with_bin_num, df_pred_and_features, id_list_to_db, ref_info_to_db



# In[47]:


#TAG
def predict_df_with_old_cust(df_combined, predict_yyyymm, writing_path, project_name, mother, algorithm, bins,papulation_colname,
                       papulation_except_value, target, frequency, edition_detail, new_edition):
    # 抓取最新的模型路徑
    model_path = get_model_path(writing_path, project_name, mother, new_edition, target, algorithm)
    # 讀取模型
    print(f'[predict_df]Loading model with {model_path} ...')
    model = pickle.load(open(model_path, 'rb'))
    print('[predict_df]Starting predicting ...')
    # 抓取母體
#     df_combined_papulation = df_combined[~df_combined[papulation_colname].isin(papulation_except_value)]
    df_combined_papulation = df_combined
    # 預測機率
    current_preds = model_predict(df_combined_papulation, model, predict_yyyymm)
    # 獲取模型重要特徵by排序
    feat_imp = get_model_feature_and_imp(model)
    cols = feat_imp['feature'].tolist()
    # 抓需要預測的年月資料
    df_combined_pt = df_combined_papulation[df_combined_papulation['yyyymm'].isin(predict_yyyymm)]


    # 計算該月最後一天
    year=predict_yyyymm[0][0:4]
    month=predict_yyyymm[0][4:6]
    end_day = calendar.monthrange(int(year),int(month))[1]
    date_last_day = year+'/'+month+'/'+str(end_day)
    # 預測時間
    pred_date = datetime.today().strftime('%Y/%m/%d %H:%M')
    # 產出預測 dataframe
    df_current_preds = pd.DataFrame(data={
        'product': project_name,
        'target': target ,
        'population': mother,
        'frequency': frequency,
        'pred_date': pred_date,
        'snap_date':  date_last_day,
        'party_id': df_combined_pt['customer_id'],
        'tag_name': project_name+target,
        'model_pred': current_preds,
        'exception_tag': df_combined_pt[papulation_colname]
    })

    # 整包含舊戶名單_各等級下人數 dataframe
    df_level_with_bin = get_bins_table(df_current_preds.drop(['party_id','exception_tag'], axis=1), bins=bins, level_type = 'with_bin')
    df_level_with_bin_num =  df_level_with_bin.groupby(list(df_level_with_bin.drop('model_pred', axis=1).columns)).count().reset_index().sort_values(by='model_level',ascending = False)
    df_level_with_bin_num = df_level_with_bin_num.rename({'model_pred': 'num'}, axis=1)

    # 舊戶名單_各等級下人數 dataframe
    df_current_preds_wold = df_current_preds[df_current_preds['exception_tag'].isin(papulation_except_value)]
    df_level_with_bin_wold = get_bins_table(df_current_preds_wold.drop(['party_id','exception_tag'], axis=1), bins=bins, level_type = 'with_bin')
    df_level_with_bin_num_wold =  df_level_with_bin_wold.groupby(list(df_level_with_bin_wold.drop('model_pred', axis=1).columns)).count().reset_index().sort_values(by='model_level',ascending = False)
    df_level_with_bin_num_wold = df_level_with_bin_num_wold.rename({'model_pred': 'num'}, axis=1)

    # 新戶靜止名單_各等級下人數 dataframe
    df_current_preds_new = df_current_preds[~df_current_preds['exception_tag'].isin(papulation_except_value)]
    df_level_with_bin_new = get_bins_table(df_current_preds_new.drop(['party_id','exception_tag'], axis=1), bins=bins, level_type = 'with_bin')
    df_level_with_bin_num_new =  df_level_with_bin_new.groupby(list(df_level_with_bin_new.drop('model_pred', axis=1).columns)).count().reset_index().sort_values(by='model_level',ascending = False)
    df_level_with_bin_num_new = df_level_with_bin_num_new.rename({'model_pred': 'num'}, axis=1)

     ################################### 以下寫入DB#######################################################
    #  主數據庫 TABLE1:參照資訊檔 #
    #############################
    exception_text = papulation_colname
    if '舊戶' in papulation_colname: exception_text = '舊戶仍產出機率與標籤'
    else: exception_text = '非母體均註記為未達' + papulation_colname + '條件'
    ref_info_to_db = pd.DataFrame(data={
        'product': project_name,
        'target': target ,
        'population': mother,
        'frequency': frequency,
        'edition': str(new_edition),
        'snap_date': date_last_day,
        'model_valid_falg':'',
        'tag_name': project_name+target,
        'exception': exception_text ,
        'edition_detail': edition_detail,
        'pred_date': pred_date
    },index=[0])

    ###########################
    #  主數據庫 TABLE1:名單檔  #
    ###########################
    # 預測結果 dataframe
    df_predict = get_bins_table(df_current_preds, bins=bins ,level_type = 'level')
    df_predict['model_level'] = df_predict['model_level'].astype('str')
    df_predict = df_predict.rename({'model_level': 'rank','model_pred':'probability'}, axis=1)
    df_predict = df_predict[['product','target','population','frequency','snap_date','party_id','tag_name',
                             'probability','rank','exception_tag','pred_date']]

    # 結合預測結果跟母體以外的表格((寫入DB))
    id_list_to_db = df_predict

    ################################### 以上寫入DB#######################################################

    # 預測Dataframe + 特徵TOP
    show_columns = ['customer_id', 'yyyymm', 'y', papulation_colname]
    show_columns = show_columns + cols
    df_pred_and_features = df_combined_pt[show_columns]
    df_pred_and_features = df_pred_and_features.rename({'y': 'y_true'}, axis=1)
    df_pred_and_features['model_pred'] = df_predict['probability']
    df_pred_and_features['model_level'] = df_predict['rank']

    print('[predict_df] predict finished and return table !!!')

    return df_level_with_bin_num, df_level_with_bin_num_wold, df_level_with_bin_num_new,  df_pred_and_features, id_list_to_db, ref_info_to_db



# In[48]:


#TAG
def ntb_predict_df(df_combined, predict_yyyymm, writing_path, project_name, mother, algorithm, bins,papulation_colname,
                       papulation_except_value, target, frequency, edition_detail, new_edition):
    # 抓取最新的模型路徑
    model_path = get_model_path(writing_path, project_name, mother, new_edition, target, algorithm)
    # 讀取模型
    print(f'[predict_df]Loading model with {model_path} ...')
    model = pickle.load(open(model_path, 'rb'))
    print('[predict_df]Starting predicting ...')
    # 抓取母體
    df_combined_papulation = df_combined[~df_combined[papulation_colname].isin(papulation_except_value)]

    # 獲取模型重要特徵by排序
    feat_imp = get_model_feature_and_imp(model)
    cols = feat_imp['feature'].tolist()

    # 對齊模型的特徵
    for i in cols:
        if (i in list(df_combined_papulation.columns)):
            pass
        else:
            df_combined_papulation[i] = 0


    # 預測機率
    current_preds = model_predict(df_combined_papulation, model, predict_yyyymm)
    # 抓需要預測的年月資料
    df_combined_pt = df_combined_papulation[df_combined_papulation['yyyymm'].isin(predict_yyyymm)]

    # 計算該月最後一天
    year=predict_yyyymm[0][0:4]
    month=predict_yyyymm[0][4:6]
    day=predict_yyyymm[0][6:8]
    date_last_day = year+'/'+month+'/'+day
    # 預測時間
    pred_date = datetime.today().strftime('%Y/%m/%d %H:%M')
    # 產出預測 dataframe
    df_current_preds = pd.DataFrame(data={
        'product': project_name,
        'target': target ,
        'population': mother,
        'frequency': frequency,
        'pred_date': pred_date,
        'snap_date':  date_last_day,
        'party_id': df_combined_pt['customer_id'],
        'tag_name': project_name+target,
        'model_pred': current_preds,

    })

    # 各等級下人數 dataframe
    df_level_with_bin = get_bins_table(df_current_preds.drop('party_id', axis=1), bins=bins, level_type = 'with_bin')
    df_level_with_bin_num =  df_level_with_bin.groupby(list(df_level_with_bin.drop('model_pred', axis=1).columns)).count().reset_index().sort_values(by='model_level',ascending = False)
    df_level_with_bin_num = df_level_with_bin_num.rename({'model_pred': 'num'}, axis=1)

     ################################### 以下寫入DB#######################################################
    #  主數據庫 TABLE1:參照資訊檔 #
    #############################
    exception_text = papulation_colname
    if '舊戶' in papulation_colname: exception_text = '舊戶均註記為舊戶'
    else: exception_text = '非母體均註記為未達' + papulation_colname + '條件'
    ref_info_to_db = pd.DataFrame(data={
        'product': project_name,
        'target': target ,
        'population': mother,
        'frequency': frequency,
        'edition': str(new_edition),
        'snap_date': date_last_day,
        'model_valid_falg':'',
        'tag_name': project_name+target,
        'exception': exception_text ,
        'edition_detail': edition_detail,
        'pred_date': pred_date
    },index=[0])
    ###########################
    #  主數據庫 TABLE1:名單檔  #
    ###########################
    # 預測結果 dataframe
    df_predict = get_bins_table(df_current_preds, bins=bins ,level_type = 'level')
    df_predict['model_level'] = df_predict['model_level'].astype('str')
    df_predict = df_predict.rename({'model_level': 'rank','model_pred':'probability'}, axis=1)
    # 除了預測母體以外的 dataframe
    except_name =''
    if papulation_except_value == [1]:
        except_name = papulation_colname
    else:
        except_name = '不是'+ papulation_colname
    df_papulation_except = pd.DataFrame(data={
        'product': project_name,
        'target': target ,
        'population': mother,
        'frequency': frequency,
        'snap_date':  date_last_day,
        'party_id': df_combined['customer_id'][df_combined[papulation_colname].isin(papulation_except_value)],
        'tag_name': project_name+target,
        'probability': np.nan,
        'rank': except_name,
        'pred_date': pred_date,
    })
    df_predict = df_predict[['product','target','population','frequency','snap_date','party_id','tag_name','probability',
        'rank','pred_date']]
    # 結合預測結果跟母體以外的表格((寫入DB))
    id_list_to_db = df_predict.append(df_papulation_except)
    ################################### 以上寫入DB#######################################################

    # 預測Dataframe + 特徵TOP
    show_columns = ['customer_id', 'yyyymm', 'y']
    show_columns = show_columns + cols
    df_pred_and_features = df_combined_pt[show_columns]
    df_pred_and_features = df_pred_and_features.rename({'y': 'y_true'}, axis=1)
    df_pred_and_features['model_pred'] = df_predict['probability']
    df_pred_and_features['model_level'] = df_predict['rank']

    print('[predict_df] predict finished and return table !!!')

    return df_level_with_bin_num, df_pred_and_features, id_list_to_db, ref_info_to_db



# In[49]:


def save_db_ref_info_mlops(write_db_Y_N, ref_info_to_db, table_name, account=account, pwd=pwd ):

    col_types = {
        'product': String(50),
        'target': String(20),
        'population': String(20),
        'frequency': String(20),
        'edition': String(10),
        'snap_date': String(15),
        'model_valid_falg': String(10),
        'tag_name':  String(50),
        'exception': String(50),
        'edition_detail': String(50),
        'pred_date': String(20)
    }

    # Output to DB
    if write_db_Y_N:
        write_data_to_SQL(table_name, ref_info_to_db, account, pwd, chunk_size=10000, col_types=col_types)
    else:
        print('[save_db_ref_info_mlops]don"t require for writing down to db ')


# In[50]:


def save_db_id_list_mlops(write_db_Y_N, id_list_to_db, table_name, account=account, pwd=pwd ):

    col_types = {
        'product': String(20),
        'target': String(20),
        'population': String(20),
        'frequency': String(20),
        'snap_date':  String(20),
        'party_id' : String(20),
        'tag_name': String(20),
        'probability': Float(),
        'rank': String(20),
        'pred_date': String(20)
    }

    # Output to DB
    if write_db_Y_N:
        write_data_to_SQL(table_name, id_list_to_db, account, pwd, chunk_size=10000, col_types=col_types)
    else:
        print('[save_db_id_list_mlops]don"t require for writing down to db ')


# In[51]:


def save_db_id_list_mlops_with_wold(write_db_Y_N, id_list_to_db, table_name, account=account, pwd=pwd ):

    col_types = {
        'product': String(20),
        'target': String(20),
        'population': String(20),
        'frequency': String(20),
        'snap_date':  String(20),
        'party_id' : String(20),
        'tag_name': String(20),
        'probability': Float(),
        'rank': String(20),
        'exception_tag': String(40),
        'pred_date': String(20)
    }

    # Output to DB
    if write_db_Y_N:
        write_data_to_SQL(table_name, id_list_to_db, account, pwd, chunk_size=10000, col_types=col_types)
    else:
        print('[save_db_id_list_mlops]don"t require for writing down to db ')


# In[52]:


def get_conversion_rank(vali_predict_df_dt, hit_rate):
    output_df = pd.DataFrame()
    print(f'========產出{hit_rate}購買力機率與名單數===========')
    st_time = time.time()
    temp_df = vali_predict_df_dt.sort_values(by='model_pred', ascending=False)
    temp_df['avg_y_true'] = temp_df['y_true'].expanding().mean()
    temp_df['num'] = temp_df['y_true'].expanding().count()
    temp_df['hit_num'] = temp_df['y_true'].expanding().sum()

    temp_df = temp_df.sort_values(by='model_pred', ascending=False)
    elapsed_time = time.time() - st_time

    for ht in [0]+hit_rate:
        try:

            h1 = temp_df['avg_y_true']>=ht
            temp_output_df = temp_df[h1].iloc[-1]
            if  ht == 0:
                temp_output_df['購買力'] = f"C{int(round(ht * 100)):02d}"
            elif  ht < 0.001:
                temp_output_df['購買力'] = f"C{int(round(ht * 10000)):04d}"
            elif  ht < 0.01:
                temp_output_df['購買力'] = f"C{int(round(ht * 1000)):03d}"
            else:
                temp_output_df['購買力'] = f"C{int(round(ht * 100)):02d}"
            output_df = output_df.append(temp_output_df)

        except:
            print(f'{ht}沒有此購買力')


    output_df = output_df.sort_values(by=['model_pred','購買力'], ascending=False)
    output_df = output_df.groupby('model_pred').first().reset_index()
    output_df = output_df.sort_values(by=['model_pred','購買力'], ascending=False)
    CHANCE_TEST_bin = list(output_df['購買力'])
    CHANCE_TEST_num = list(output_df['num'])
    CHANCE_TEST_hit_rate = list(output_df['avg_y_true'])
    CHANCE_TEST_prob = list(output_df['model_pred'])
    print(f'產出{hit_rate}購買力機率與名單數: this run time : {time.time()-st_time} sec')
    return CHANCE_TEST_bin, CHANCE_TEST_num, CHANCE_TEST_hit_rate, CHANCE_TEST_prob

    print(f'函數執行時間: {elapsed_time}秒')
    return temp_df


# In[53]:


def get_conversion_rank_lowefficient(vali_predict_df_dt, hit_rate):
    output_df = pd.DataFrame()
    print(f'========產出{hit_rate}購買力機率與名單數===========')
    st_time = time.time()

    temp_df = vali_predict_df_dt.sort_values(by='model_pred', ascending=False)

    # 假设你已经有了上述的DataFrame，将其存储在一个名为 'df' 的变量中
    # 首先，按 'model_pred' 列进行升序排序
    temp_df = temp_df.sort_values(by='model_pred', ascending=False)
    # 创建一个新列 'avg_y_true' 来存储每个行的平均 'y_true'
    temp_df['avg_y_true'] = 0.0  # 初始化 'avg_y_true' 列
    temp_df['num'] = np.nan  # 初始化 'avg_y_true' 列
    temp_df['hit_num'] = np.nan  # 初始化 'avg_y_true' 列

    # 遍历每一行，计算平均 'y_true'
    cumulative_sum = 0
    num_rows = 0
    for index, row in temp_df.iterrows():
        cumulative_sum += row['y_true']
        num_rows += 1
        temp_df.at[index, 'num'] = num_rows
        temp_df.at[index, 'hit_num'] = cumulative_sum
        temp_df.at[index, 'avg_y_true'] = cumulative_sum / num_rows
    # 现在，DataFrame 'df' 包含了 'avg_y_true' 列，表示每行的平均 'y_true'
    # 如果需要，可以将 'df' 再次按 'model_pred' 列排序，以便按 'model_pred' 列的升序进行查看
    temp_df = temp_df.sort_values(by='model_pred', ascending=False)

    for ht in hit_rate+[0]:
        try:
            h1 = temp_df['avg_y_true']>=ht
            temp_output_df = temp_df[h1].iloc[-1]
            if  ht == 0:
                temp_output_df['購買力'] = f"C{int(round(ht * 100)):02d}"
            elif  ht < 0.001:
                temp_output_df['購買力'] = f"C{int(round(ht * 10000)):04d}"
            elif  ht < 0.01:
                temp_output_df['購買力'] = f"C{int(round(ht * 1000)):03d}"
            else:
                temp_output_df['購買力'] = f"C{int(round(ht * 100)):02d}"
            output_df = output_df.append(temp_output_df)
        except:
            print(f'{ht}沒有此購買力')

    output_df = output_df.sort_values(by=['model_pred','購買力'], ascending=False)
    CHANCE_TEST_bin = list(output_df['購買力'])
    CHANCE_TEST_num = list(output_df['num'])
    CHANCE_TEST_hit_rate = list(output_df['avg_y_true'])
    CHANCE_TEST_prob = list(output_df['model_pred'])
    print(f'產出{hit_rate}購買力機率與名單數: this run time : {time.time()-st_time} sec')
    return CHANCE_TEST_bin, CHANCE_TEST_num, CHANCE_TEST_hit_rate, CHANCE_TEST_prob


# In[54]:


def Convert_log_v240111(write_db_Y_N, project_name, mother, df_combined,
                        backtest_auc, edition, table_name ,
                        CHANCE_TEST_bin=np.nan, CHANCE_TEST_num=np.nan, CHANCE_TEST_hit_rate=np.nan, CHANCE_TEST_prob=np.nan,
                        account=account, pwd=pwd):

    def get_bound_list(CHANCE_TEST_bin):
        upper_bound = []
        lower_bound = []
        for i,bin in enumerate(CHANCE_TEST_bin):
            if i == 0:
                upper_bound.append(1.0)
            else:
                upper_bound.append(CHANCE_TEST_prob[i-1])
            if i == len(CHANCE_TEST_bin)-1:
                lower_bound.append(0.0)
            else:
                lower_bound.append(CHANCE_TEST_prob[i])
        return lower_bound, upper_bound
    lower_bound, upper_bound  = get_bound_list(CHANCE_TEST_bin)
    #backtest_log
    Convert_log = pd.DataFrame({
                            '模型名稱' :project_name + '模型',
                            '母體' : mother,
                            'backtest日期' :datetime.today().strftime('%Y%m%d %H:%M'),
                            'test_period' :str(df_combined['yyyymm'].sort_values().unique()[-1]),
                            'test_auc' :backtest_auc,
                            '版本':edition ,

                            'Cv_TEST_Label' : CHANCE_TEST_bin,
                            'Cv_TEST_人數' : CHANCE_TEST_num,
                            'Cv_TEST_真實轉換率' : CHANCE_TEST_hit_rate,
                            'Cv_TEST_prob_upbound': upper_bound,
                            'Cv_TEST_prob_lowbound': lower_bound
                                }
    )

    if write_db_Y_N :

        col_types = {
            '模型名稱' : String(30),
            '母體' : String(30),
            'backtest日期' : String(30),
            'test_period': String(30),
            'test_auc': Float(),
            '版本':String(10),

            'Cv_TEST_Label' : String(30),
            'Cv_TEST_人數' : Float(),
            'Cv_TEST_真實轉換率' : Float(),
            'Cv_TEST_prob_upbound': Float(),
            'Cv_TEST_prob_lowbound': Float()
        }

        # Output to DB
        write_data_to_SQL(table_name, Convert_log, account, pwd, chunk_size=10000, col_types=col_types)
    else:
        print('[Convert_log]don"t require for writing down to db ')
    return Convert_log


# In[55]:


# df 需要有欄位 y跟yyyymmm
def get_backtest_df_pvalue_IV(project_name, target, mother, df, col=[]):
    sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
    import config
    from Sql_module import get_SQL_raw_data

    def del_dupuli_end_word(r):
        if r['feature'][-2:] == '_x' or r['feature'][-2:] == '_y'or r['feature'][-2:] == '.1':
            return r['feature'][:-2]
        else:
            return r['feature']

    from scipy import stats
    #IV值
    def bin_woe(df, tar_name, var_name, n=5, cat='category'):
        tar = df[tar_name]
        var_not_0 = df[df[var_name]!=0][var_name]
        var = df[var_name]

        total_bad = tar.sum()
        total_good = tar.count()-total_bad
        totalRate = total_good/total_bad

        if cat == 'number':
            bins_0 = pd.qcut(var_not_0, n, duplicates='drop')
            bins = [min(var.fillna(0))-0.001]+list(bins_0.cat.categories.right)
            msheet = pd.DataFrame({tar.name:tar,var.name:var,'var_bins':pd.cut(var, bins, duplicates='drop')})
            grouped = msheet.groupby(['var_bins'])
        elif cat == 'category':
            msheet = pd.DataFrame({tar.name:tar, var.name:var})
            grouped = msheet.groupby([var.name])

        groupBad = grouped.sum()[tar.name]
        groupTotal = grouped.count()[tar.name]
        groupGood = groupTotal - groupBad
        groupRate = groupGood/groupBad
        groupBadRate = groupBad/groupTotal
        groupGoodRate = groupGood/groupTotal

        woe = np.log(groupRate/totalRate)
        iv = np.sum((groupGood/total_good-groupBad/total_bad)*woe)

        return iv
    # 特徵翻譯
    feature_trans = get_SQL_raw_data(''' select distinct lower(特徵) as 特徵,business_glossory as 翻譯 from s_ianleong.new_feature_spec_2025
                                    union
                                    select distinct 特徵 ,翻譯 from s_ianleong.new_fin_feature_spec_2025 ''',
                                     account=config.account, pwd=config.pwd)
    feature_trans = feature_trans.rename(columns={'特徵':'feature', '翻譯':'feature_chinese'})

    backtest_ym = max(df['yyyymm'])
    r1 = df['yyyymm'] == backtest_ym
    backtest_ym_df = df[r1]

    print(f'test_period dataframe length is {len(backtest_ym_df)}')

    if col == []:
        col = list(backtest_ym_df.columns)

    for i in ['yyyymm','y','customer_id']:
        col = [a for a in col if a not in i]

    print(f'計算col的IV & 卡方/T檢定 ， col = {col}')
    print(backtest_ym_df[col].dtypes)
    numeric_features = backtest_ym_df[col].select_dtypes(include=[np.number, 'float64']).columns
    category_features = backtest_ym_df[col].select_dtypes(include=['category']).columns

    feature_importance = pd.DataFrame(col,columns=['feature'])
    feature_importance['prod'] = project_name

    #判斷重要特徵dtype
    conditions = [feature_importance['feature'].isin(category_features),feature_importance['feature'].isin(numeric_features)]
    choices = ['category', 'number']
    feature_importance['dtype'] = np.select(conditions, choices, default = np.nan)

    #IV值 & T檢定 & 卡方檢定
    feat_result = pd.DataFrame()
    t0 = time.time()

    prod = project_name
    for i, j in feature_importance[feature_importance['prod']==prod][['feature','dtype']].itertuples(index = False):
        completed = 'N'
        IV = np.nan
        p = np.nan
        try:
            tar_name = 'y'
            var_name = i
            IV = bin_woe(backtest_ym_df, tar_name, var_name, n=3, cat=j)
            if j == 'number':
                A = backtest_ym_df[backtest_ym_df[tar_name]==True][var_name].fillna(0)
                B = backtest_ym_df[backtest_ym_df[tar_name]==False][var_name].fillna(0)
                t, p = stats.ttest_ind(A, B, equal_var = False)
                feat_result_0 = pd.DataFrame({'prod':[prod], 'target':[target], 'population':[mother],
                                              'test_period':[backtest_ym],'feature':[i], 'dtype':[j], 'p_value':[p], 'IV':[IV],
                                              'error_msg':np.nan})
                feat_result = feat_result.append(feat_result_0)

            elif j == 'category':
                contingency_table = pd.crosstab(backtest_ym_df[var_name], backtest_ym_df[tar_name])
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                feat_result_0 = pd.DataFrame({'prod':[prod], 'target':[target], 'population':[mother],
                                              'test_period':[backtest_ym],'feature':[i], 'dtype':[j], 'p_value':[p], 'IV':[IV],
                                              'error_msg':np.nan})
                feat_result = feat_result.append(feat_result_0)

            completed = 'y'
        except Exception as e:
            feat_result_0 = pd.DataFrame({'prod':[prod], 'target':[target], 'population':[mother],
                                          'test_period':[backtest_ym],'feature':[i], 'dtype':[j], 'p_value':[p], 'IV':[IV],
                                          'error_msg':str(e)})
            feat_result = feat_result.append(feat_result_0)
            print(i, e)

        t1 = time.time()
        print('執行狀態:', completed, prod,':',i,'/dtype：',j, '\n','Total Running Time: %.0f sec'%(t1-t0), end = '\r')

    # 調整重複特徵名字
    feat_result['feature'] = feat_result.apply(del_dupuli_end_word, axis=1)
    # 串上中文特徵
    feat_result = pd.merge(feat_result, feature_trans, how='left', on=['feature'])
    feat_result = feat_result[['prod', 'target', 'population', 'test_period', 'feature', 'feature_chinese',
                               'dtype', 'p_value', 'IV', 'error_msg']]
    feat_result = feat_result.drop_duplicates()

    return feat_result.replace(float('inf'), 99)


# In[56]:


get_ipython().system('jupyter nbconvert --to script Model.ipynb')


# In[ ]:




