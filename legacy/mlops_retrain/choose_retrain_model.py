#!/usr/bin/env python

# In[ ]:


def main_code():
    import importlib
    import os
    import random
    import sys
    import warnings
    from datetime import datetime, timedelta
    warnings.filterwarnings("ignore")
    sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
    import retrain_mlops
    importlib.reload(retrain_mlops)
    import config
    from retrain_mlops import (
        retrain_mlops_np_and_p_double_opt_detailym_fillna_20250326,
        retrain_mlops_np_and_p_double_opt_fillna_20250326,
    )
    from Sql_module import get_SQL_raw_data


    do_detail_ym_prod_list = [
     '債券型基金',
     '不限用途',
     '保險商品',
     '台股信用交易',
     '台股定期定額',
     '海外股票定期定額',
     '儲蓄型保險商品',
     '基金',
     '基金定期定額',
     '境內結構型',
     '海外債',
     '平衡型基金',
     '海外股票',
     '結構型商品',
     '股票型基金',
     '境外結構型',
     '投資型保險商品',
     '期貨',
     '雙向借券',
     '財管商品',
    '基金定期定額']
    new_prod = []
    not_market_prod_12m = ['流失預警']
    not_market_prod_3m = ['客群上送', '潛在高價值客戶']

    completed_df = get_SQL_raw_data('select * from s_ianleong.mlops_retrain_log_double',
                                    account=config.account,pwd=config.pwd )


    def get_completed_list(completed_df, prod_colname):
        completed_list = []
        try:
            for prod in set(completed_df[prod_colname]):
                if len(completed_df[completed_df[prod_colname]==prod])==2 or '境內結構型' in prod and  sum(completed_df[prod_colname]==prod)==1 or '境外結構型' in prod and  sum(completed_df[prod_colname]==prod)==1 or '流失' in prod and  sum(completed_df[prod_colname]==prod)==1 or '客群上送' in prod and  sum(completed_df[prod_colname]==prod)==1 or '潛在高價值客戶' in prod and  sum(completed_df[prod_colname]==prod)==1:
                    completed_list.append(prod.replace('模型',''))
        except:
            pass
        return completed_list
    def last_day_of_previous_months(date_string):
        date_format='%Y%m'
        input_date = datetime.strptime(date_string, date_format)
        first_day_of_current_month = input_date.replace(day=1)
        last_day_of_previous_month = first_day_of_current_month - timedelta(days=1)
        return last_day_of_previous_month.strftime('%Y/%m/%d')
    def make_query_check_last_ref_info(ym):
        return f"""
        SELECT distinct product, target, population
        FROM s_ianleong.mlops_ref_info_double
        where snap_date = '{last_day_of_previous_months(ym)}'
        order by product, target, population

    """
    check_last_ref_info = get_SQL_raw_data(make_query_check_last_ref_info(config.ym),
                                           account=config.account,pwd=config.pwd )

    last_ref_info_completed_list =get_completed_list(check_last_ref_info, prod_colname="product")



    # 非潛客，需修改LEFT JOIN商品
    def make_query_mlops_retrain_status(ym):
        return f"""
    SELECT DISTINCT A."模型名稱",A."母體"
    FROM s_ianleong.mlops_model_log_double A
    left join s_ianleong.mlops_retrain_log_double B
    on A."模型名稱" = B."模型名稱" and A."母體" = B."母體" and ABS(TO_DATE(A."retrain日期",'YYYYMMDD HH24:MI') - TO_DATE(B."retrain日期",'YYYYMMDD HH24:MI'))<= 0.0007
    where B."retrain日期" >= '{str(int(ym))}' 
    order by  A."模型名稱",A."母體" 
    """
    def del_trash_in_list(L):
        if '.ipynb_checkpoints' in L : L.remove('.ipynb_checkpoints')
        if 'mlops_predict' in L : L.remove('mlops_predict')
        if 'mlops_retrain' in L : L.remove('mlops_retrain')
        return L
    # query = make_query_mlops_retrain_status(config.ym+'24')
    query = make_query_mlops_retrain_status(config.mlops_retrain_day)
    completed_df = get_SQL_raw_data(query)



    retrain_completed_list =get_completed_list(completed_df, prod_colname="模型名稱")

    all_prod_list = os.listdir('/home/cdsw/Tony/Mlops_new/審核通過模型_雙證')
    not_yet_run_list = set(all_prod_list)-set(retrain_completed_list)
    not_yet_run_list = del_trash_in_list(not_yet_run_list)
    can_run_prod_list = sorted(not_yet_run_list)


    retrain_completed_list


    not_yet_run_list


    can_run_prod_list


    C20 = [0.05, 0.1, 0.2]
    C10 = [0.03, 0.05, 0.10]
    C05 = [0.01, 0.03, 0.05]
    C03 = [0.005, 0.01, 0.03]
    C01 = [0.005, 0.01]
    C005 = [0.005]


    n = random.randint(0,len(can_run_prod_list)-1)
    # this_prod = '台股定期定額'
    this_prod = can_run_prod_list[n]

    print(f'=========這個是跑{this_prod}模型')
    this_file_path = '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/' + this_prod
    ##一般版
    def load_population1(this_prod,mother,ym,papulation_colname):
        return f"""
    select customer_id
        ,yyyymm
        ,NVL("{this_prod}{papulation_colname}",0) as {papulation_colname}
        ,NVL("{this_prod}Y",0) AS Y 
    from s_ianleong.mlops_population a
    where segment = '{mother}' and yyyymm = '{ym}'
    """
    ##例外(結構型)
    def load_population2(this_prod,mother,ym,papulation_colname):
        return f"""
    select customer_id
        ,yyyymm
        ,NVL("{this_prod}{papulation_colname}",0) as {papulation_colname}
        ,NVL("{this_prod}Y",0) AS Y 
    from s_ianleong.mlops_population a
    where segment = '{mother}' and yyyymm = '{ym}' AND "98戶" is null
    """
    ##例外(客群上送，潛在高價值客戶)
    def load_population3(this_prod,mother,ym,papulation_colname):
        return f"""
    select customer_id
        ,yyyymm
        ,NVL("{this_prod}{papulation_colname}R",0) as {papulation_colname}
        ,NVL("{this_prod}Y",0) AS Y 
    from s_ianleong.mlops_population a
    where yyyymm = '{ym}'
    """
    ##例外(流失預警)
    def load_population4(this_prod,mother,ym,papulation_colname):
        return f"""
    select customer_id
        ,yyyymm
        ,NVL("{this_prod}{papulation_colname}",0) as {papulation_colname}
        ,NVL("{this_prod}Y",1) AS Y 
    from s_ianleong.mlops_population a
    where yyyymm = '{ym}'
    """

    # 模型間需修改
    if this_prod == '不限用途':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_train_value = [0]
        drop_key_word = ['LOAN']
        mother_list = ['非潛客', '潛客' ]
        bins_list = [ [1.0, 0.62, 0.40, 0.31, 0.22, 0], [1.0, 0.62, 0.40, 0.31, 0.22, 0] ]
        hit_rate_list = [C01, C01]
        query_function = [load_population1]

    elif this_prod == '保險商品':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_train_value = [0]
        drop_key_word = ['INSURANCE']
        mother_list = ['非潛客', '潛客' ]
        bins_list = [ [1.0, 0.62, 0.40, 0.31, 0.22, 0], [1.0, 0.62, 0.40, 0.31, 0.22, 0] ]
        hit_rate_list = [C03, C03]
        query_function = [load_population1]

    elif this_prod == '債券型基金':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_train_value = [0]
        drop_key_word = ['FD']
        mother_list = ['非潛客', '潛客' ]
        bins_list = [ [1.0, 0.62, 0.40, 0.31, 0.22, 0], [1.0, 0.62, 0.40, 0.31, 0.22, 0] ]
        hit_rate_list = [C05, C05]
        query_function = [load_population1]

    elif this_prod == '儲蓄型保險商品':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_train_value = [0]
        drop_key_word = ['INSURANCE']
        mother_list = ['非潛客', '潛客' ]
        bins_list = [ [1.0, 0.62, 0.40, 0.31, 0.22, 0], [1.0, 0.62, 0.40, 0.31, 0.22, 0] ] #非潛客級距
        hit_rate_list = [C03, C03]
        query_function = [load_population1]

    elif this_prod == '台股信用交易' or this_prod == '台股定期定額' or this_prod == '海外股票定期定額':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_train_value = [0]
        drop_key_word = []
        mother_list = ['非潛客', '潛客' ]
        bins_list = [ [1.0, 0.62, 0.40, 0.31, 0.22, 0], [1.0, 0.62, 0.40, 0.31, 0.22, 0] ] #非潛客級距
        hit_rate_list = [C10, C10]
        query_function = [load_population1]

    elif this_prod == '基金':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_train_value = [0]
        drop_key_word = ['FD']
        mother_list = ['非潛客', '潛客' ]
        bins_list = [ [1.0, 0.62, 0.40, 0.31, 0.22, 0], [1.0, 0.62, 0.40, 0.31, 0.22, 0] ]
        hit_rate_list = [C05, C05]
        query_function = [load_population1]

    elif this_prod == '基金定期定額':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_train_value = [0]
        drop_key_word = []
        mother_list = ['非潛客', '潛客' ]
        bins_list = [ [1.0, 0.62, 0.40, 0.31, 0.22, 0], [1.0, 0.62, 0.40, 0.31, 0.22, 0] ] #非潛客級距
        hit_rate_list = [C05, C05]
        query_function = [load_population1]

    elif this_prod == '境內結構型' or this_prod == '境外結構型':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近三年舊戶'
        papulation_train_value = [0]
        drop_key_word = ['SN','PRO_INVEST_FLSG']
        mother_list = ['非潛客']
        bins_list = [ [1.0, 0.065, 0.01, 0.004, 0.002, 0] ]
        hit_rate_list = [C05, C05]
        query_function = [load_population2]

    elif this_prod == '平衡型基金':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_train_value = [0]
        drop_key_word = ['FD']
        mother_list = ['非潛客', '潛客' ]
        bins_list = [ [1.0, 0.62, 0.40, 0.31, 0.22, 0], [1.0, 0.62, 0.40, 0.31, 0.22, 0] ]
        hit_rate_list = [C05, C05]
        query_function = [load_population1]

    elif this_prod == '投資型保險商品':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_train_value = [0]
        drop_key_word = ['INSURANCE']
        mother_list = ['非潛客', '潛客' ]
        bins_list = [ [1.0, 0.62, 0.40, 0.31, 0.22, 0], [1.0, 0.62, 0.40, 0.31, 0.22, 0] ] #非潛客級距
        hit_rate_list = [C03, C03]
        query_function = [load_population1]

    elif this_prod == '期貨':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_train_value = [0]
        drop_key_word = ['FU_']
        mother_list = ['非潛客', '潛客' ]
        bins_list = [ [1.0, 0.62, 0.40, 0.31, 0.22, 0], [1.0, 0.62, 0.40, 0.31, 0.22, 0] ] #非潛客級距
        hit_rate_list = [C10, C10]
        query_function = [load_population1]

    elif this_prod == '海外債':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_train_value = [0]
        drop_key_word = ['FD']
        mother_list = ['非潛客', '潛客' ]
        bins_list = [ [1.0, 0.62, 0.40, 0.31, 0.22, 0], [1.0, 0.62, 0.40, 0.31, 0.22, 0] ] #非潛客級距
        hit_rate_list = [C05, C05]
        query_function = [load_population1]

    elif this_prod == '海外股票':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近半年舊戶'
        papulation_train_value = [0]
        drop_key_word = []
        mother_list = ['非潛客', '潛客' ]
        bins_list = [ [1.0, 0.62, 0.40, 0.31, 0.22, 0], [1.0, 0.62, 0.40, 0.31, 0.22, 0] ] #非潛客級距
        hit_rate_list = [C10, C10]
        query_function = [load_population1]

    elif this_prod == '結構型商品':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近三年舊戶'
        papulation_train_value = [0]
        drop_key_word = ['SN','PRO_INVEST_FLSG']
        mother_list = ['非潛客', '潛客']
        bins_list = [ [1.0, 0.065, 0.01, 0.004, 0.002, 0], [1.0, 0.065, 0.01, 0.004, 0.002, 0]  ]
        hit_rate_list = [C05, C05]
        query_function = [load_population2]

    elif this_prod == '股票型基金':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_train_value = [0]
        drop_key_word = ['FD']
        mother_list = ['非潛客', '潛客' ]
        bins_list = [ [1.0, 0.62, 0.40, 0.31, 0.22, 0], [1.0, 0.62, 0.40, 0.31, 0.22, 0] ]
        hit_rate_list = [C05, C05]
        query_function = [load_population1]

    elif this_prod == '雙向借券':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_train_value = [0]
        drop_key_word = ['BSBL']
        mother_list = ['非潛客', '潛客' ]
        bins_list = [ [1.0, 0.62, 0.40, 0.31, 0.22, 0], [1.0, 0.62, 0.40, 0.31, 0.22, 0] ]
        hit_rate_list = [C10, C10]
        query_function = [load_population1]

    elif this_prod == '財管商品':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_train_value = [0]
        drop_key_word = []
        mother_list = ['非潛客', '潛客' ]
        bins_list = [ [1.0, 0.62, 0.40, 0.31, 0.22, 0], [1.0, 0.62, 0.40, 0.31, 0.22, 0] ]
        hit_rate_list = [C20, C20]
        query_function = [load_population1]

    elif this_prod == '客群上送':
        target = '客群上送預測'
        papulation_colname = '前季高交易量客戶'
        papulation_train_value = [0]
        drop_key_word = []
        mother_list = ['不分潛客' ]
        bins_list = [ [0,1000,2000,3000,10000,100000,10000000] ]
        query_function = [load_population3]

    elif this_prod == '潛在高價值客戶':
        target = '潛在高價值客戶'
        papulation_colname = '前季高交易量客戶'
        papulation_train_value = [0]
        drop_key_word = ['AMT', 'NUM', 'PROD', 'BNF', 'AUM', 'SAFETY', 'TURNOVER', 'DAY', 'NET_TXN', 'SEGMENT']
        mother_list = ['不分潛客' ]
        bins_list = [ [0,500,1000,2000,3000,10000,10000000]]
        query_function = [load_population3]

    elif this_prod == '流失預警':
        target = '瞌睡客戶關懷'
        papulation_colname = '近一年實動'
        papulation_train_value = [1]
        drop_key_word = []
        mother_list = ['不分潛客']
        bins_list = [ ['1.0','0.9','0.75','0.6','0.4','0'] ]
        hit_rate_list = [C10, C10]
        query_function = [load_population4]


    frequency = config.frequency
    edition_detail = config.edition_detail
    write_db_Y_N = config.write_db_Y_N

    if this_prod not in do_detail_ym_prod_list:
        print('[採用FILLNA 0] Retrain 使用 retrain_mlops_np_and_p_double_opt_fillna 函數')
        retrain_sop = retrain_mlops_np_and_p_double_opt_fillna_20250326
        edition_detail = 'FILLNA補0'


        if this_prod in not_market_prod_3m:
            do_ym_list = config.do_ym_list_pd_3m
            limit_size_select = int(config.limit_size_select)
            limit_size_build =  int(config.limit_size_build)
            market_flag_Y_N = False
        elif this_prod in not_market_prod_12m:
            do_ym_list = config.do_ym_list_pd_12m
            limit_size_select = int((config.limit_size_select)*2)
            limit_size_build =  int(config.limit_size_build)
            market_flag_Y_N = False

    else:
        print('[採用增加中間年用 + FILLNA 0] Retrain 使用 retrain_mlops_np_and_p_double_opt_detailym_fillna_20231025 函數')
        retrain_sop = retrain_mlops_np_and_p_double_opt_detailym_fillna_20250326
        edition_detail = '中間年月_FILLNA補0'
        limit_size_select = int(config.limit_size_select/2.5)
        limit_size_build =  int(config.limit_size_build/2.5)

        do_ym_list = config.do_ym_list_pd_3m_detail
        market_flag_Y_N = True


    frequency = config.frequency
    # edition_detail = config.edition_detail

    write_feature_Y_N = config.write_feature_Y_N

    print(f'this_prod = {this_prod}')
    print(f'this_file_path = {this_file_path}')
    print(f'target = {target}')
    print(f'papulation_colname = {papulation_colname}')
    print(f'papulation_train_value = {papulation_train_value}')
    print(f'drop_key_word = {drop_key_word}')
    print(f'mother_list = {mother_list}')
    print(f'bins_list = {bins_list}')
    print(f'edition_detail = {edition_detail}')
    print(f'do_ym_list = {do_ym_list}')
    print(f'query_function = {query_function}')
    print(f'write_db_Y_N = {write_db_Y_N}')
    print(f'limit_size_select = {limit_size_select}')
    print(f'limit_size_build = {limit_size_build}')
    print(f'frequency = {frequency}')
    print(f'edition_detail = {edition_detail}')
    print(f'write_feature_Y_N = {write_feature_Y_N}')
    print(f'market_flag_Y_N = {market_flag_Y_N}')




    if this_prod in (not_market_prod_3m + not_market_prod_12m):
        retrain_sop(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
                mother_list, bins_list, frequency, edition_detail, do_ym_list,query_function , write_db_Y_N,
                limit_size_select, limit_size_build, write_feature_Y_N)
    else:
        retrain_sop(this_file_path, target, papulation_colname, papulation_train_value, drop_key_word,
                    mother_list, bins_list, hit_rate_list , frequency, edition_detail, do_ym_list,query_function , write_db_Y_N,
                    limit_size_select, limit_size_build, write_feature_Y_N, market_flag_Y_N)






