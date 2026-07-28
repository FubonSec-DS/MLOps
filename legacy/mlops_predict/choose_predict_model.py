#!/usr/bin/env python

# In[1]:


def main_code():
    import random
    import sys
    import warnings

    import imp
    warnings.filterwarnings("ignore")
    sys.path.append('/home/cdsw/Tony/Mlops_new/Module')
    import predict_mlops
    imp.reload(predict_mlops)
    import calendar

    import config
    from predict_mlops import predict_mlops_202503
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
     '財管商品']


    # 非潛客，需修改LEFT JOIN商品
    def make_query_mlops_retrain_status(ym):
        return f"""
    SELECT DISTINCT A."模型名稱",A."母體"
    FROM s_ianleong.mlops_model_log_double A
    left join s_ianleong.mlops_retrain_log_double B
    on A."模型名稱" = B."模型名稱" and A."母體" = B."母體"  and ABS(TO_DATE(A."retrain日期",'YYYYMMDD HH24:MI') - TO_DATE(B."retrain日期",'YYYYMMDD HH24:MI'))<= 0.0007
    where B."retrain日期" >= '{str(int(ym))}'
    order by  A."模型名稱",A."母體" 
    """
    query = make_query_mlops_retrain_status(config.mlops_retrain_day)

    completed_retrain_df = get_SQL_raw_data(query)

    def make_query_mlops_predict_status(snap_date):
        return f"""
    SELECT DISTINCT product,target,population
    FROM s_ianleong.mlops_ref_info_double 
    where snap_date = '{str(snap_date)}'
    order by  product, population
    """
    year=config.ym[0:4]
    month=config.ym[4:6]

    end_day = calendar.monthrange(int(year),int(month))[1]
    snap_date = year+'/'+month+'/'+str(end_day)
    print(f'snap_date = {snap_date}')

    query = make_query_mlops_predict_status(snap_date)

    completed_predict_df = get_SQL_raw_data(query)



    completed_predict_df




    completed_retrain_df




    def get_completed_list(completed_df, prod_colname):
        completed_list = []
        try:
            for prod in set(completed_df[prod_colname]):
                if len(completed_df[completed_df[prod_colname]==prod])==2 or '流失' in prod and  sum(completed_df[prod_colname]==prod)==1 or '境內結構型' in prod and  sum(completed_df[prod_colname]==prod)==1 or '境外結構型' in prod and  sum(completed_df[prod_colname]==prod)==1 or '客群上送' in prod and  sum(completed_df[prod_colname]==prod)==1 or '潛在高價值客戶' in prod and  sum(completed_df[prod_colname]==prod)==1:
                    completed_list.append(prod.replace('模型',''))
        except:
            pass
        return completed_list

    def get_completed_list_one_mother(completed_df, prod_colname):
        completed_list = []
        try:
            for prod in set(completed_df[prod_colname]):
                if len(completed_df[completed_df[prod_colname]==prod])==1 or '結構' in prod and  sum(completed_df[prod_colname]==prod)==1 or '流失' in prod and  sum(completed_df[prod_colname]==prod)==1:
                    completed_list.append(prod.replace('模型',''))
        except:
            pass
        return completed_list




    completed_retrain_list = get_completed_list(completed_retrain_df, prod_colname="模型名稱")
    completed_pred_list = get_completed_list(completed_predict_df, prod_colname="product")



    completed_retrain_list




    completed_pred_list




    undo_prod_list = sorted(list(set(completed_retrain_list)-set(completed_pred_list)))



    undo_prod_list
    print("undo_prod_list",undo_prod_list)



    n = random.randint(0,len(undo_prod_list)-1)
    this_prod = undo_prod_list[n]

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
        ,NVL("{this_prod}{papulation_colname}P",0) as {papulation_colname}
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
    if this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/不限用途':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_except_value = [1]
        drop_key_word = ['loan']
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population1]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/保險商品':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_except_value = [1]
        drop_key_word = ['insurance']
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population1]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/債券型基金':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_except_value = [1]
        drop_key_word = ['fd']
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population1]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/儲蓄型保險商品':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_except_value = [1]
        drop_key_word = ['insurance']
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population1]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/台股信用交易' or this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/台股定期定額' or this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/海外股票定期定額':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_except_value = [1]
        drop_key_word = []
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population1]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/基金':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_except_value = [1]
        drop_key_word = ['fd']
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population1]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/基金定期定額':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_except_value = [1]
        drop_key_word = []
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population1]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/境內結構型' or this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/境外結構型':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近三年舊戶'
        papulation_except_value = [1]
        drop_key_word = ['sn']
        mother_list = ['非潛客']
        query_list = [load_population2]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/平衡型基金':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_except_value = [1]
        drop_key_word = ['fd']
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population1]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/投資型保險商品':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_except_value = [1]
        drop_key_word = ['insurance']
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population1]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/期貨':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_except_value = [1]
        drop_key_word = ['fu_']
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population1]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/海外債':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_except_value = [1]
        drop_key_word = ['fb']
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population1]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/海外股票':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近半年舊戶'
        papulation_except_value = [1]
        drop_key_word = []
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population1]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/結構型商品':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近三年舊戶'
        papulation_except_value = [1]
        drop_key_word = ['sn']
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population2]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/股票型基金':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_except_value = [1]
        drop_key_word = ['fd']
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population1]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/雙向借券':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_except_value = [1]
        drop_key_word = ['bsbl']
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population1]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/財管商品':
        target = '新戶開發與靜止戶活化'
        papulation_colname = '近一年舊戶'
        papulation_except_value = [1]
        drop_key_word = []
        mother_list = ['非潛客', '潛客' ]
        query_list = [load_population1]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/客群上送':
        target = '客群上送預測'
        papulation_colname = '前季高交易量客戶'
        papulation_except_value = [1]
        drop_key_word = []
        mother_list = ['不分潛客']
        bins_list = [ [0,1000,2000,3000,10000,100000,10000000]  ]
        query_list = [load_population3 ]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/潛在高價值客戶':
        target = '潛在高價值客戶'
        papulation_colname = '前季高交易量客戶'
        papulation_except_value = [1]
        drop_key_word = ['amt','num','prod','bnf','AUM','safety','turnover','day','net_txn','segment']
        mother_list = ['不分潛客']
        bins_list = [ [0,500,1000,2000,3000,10000,10000000] ]
        query_list = [load_population3 ]

    elif this_file_path == '/home/cdsw/Tony/Mlops_new/審核通過模型_雙證/流失預警':
        target = '瞌睡客戶關懷'
        papulation_colname = '近一年實動'
        papulation_except_value = [0]
        drop_key_word = []
        mother_list = ['不分潛客']
        bins_list = [ ['1.0','0.9','0.75','0.6','0.4','0'] ]
        query_list = [load_population4 ]


    ym = config.ym

    algorithm = config.algorithm
    write_db_Y_N = True
    drop_key_word = []
    frequency = config.frequency
    if this_prod not in do_detail_ym_prod_list:
        print('[採用FILLNA 0] Retrain 使用 retrain_mlops_np_and_p_double_opt_fillna 的流程')

        edition_detail = 'FILLNA補0'
        market_flag_Y_N = False
    else:
        print('[採用增加中間年用 + FILLNA 0] Retrain 使用 retrain_mlops_np_and_p_double_opt_detailym_fillna 的流程')
        edition_detail = '中間年月_FILLNA補0'
        market_flag_Y_N = True


    print(f'this_file_path = {this_file_path}')
    print(f'target = {target}')
    print(f'papulation_colname = {papulation_colname}')
    print(f'papulation_except_value = {papulation_except_value}')
    print(f'drop_key_word = {drop_key_word}')
    print(f'mother_list = {mother_list}')
    print(f'query_list = {query_list}')
    print(f'edition_detail = {edition_detail}')
    print(f'ym = {ym}')
    print(f'write_db_Y_N = {write_db_Y_N}')
    print(f'frequency = {frequency}')
    print(f'edition_detail = {edition_detail}')
    print(f'market_flag_Y_N = {market_flag_Y_N}')



    predict_mlops_202503(this_file_path,target,papulation_colname,papulation_except_value,mother_list,query_list,ym,
                           algorithm,write_db_Y_N,drop_key_word,frequency,edition_detail, market_flag_Y_N)



# In[10]:


get_ipython().system('jupyter nbconvert --to script choose_predict_model.ipynb')


# In[ ]:




