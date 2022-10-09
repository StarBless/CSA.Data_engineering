# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
#数据预处理
#1.**********************************处理off_tarin
def Pretreatment_train_data (d):
    dataset = d.copy()
    dataset ['is_Manjian'] = dataset ['Discount_rate'].map(
    lambda x: 1 if ':' in str(x) else 0)
    dataset ['rate'] = dataset ['Discount_rate'].map(
    lambda x:float(x) if':'not in str(x) else
    (float(str(x).split(':')[0])-float(str(x).split(':')[1]))/
    (float(str(x).split(':')[0])))
    dataset ['min in Manjian'] = dataset ['Discount_rate'].map(
    lambda x : -1 
    if ':' not in str(x) else
    int(str(x).split(':')[0]))
    dataset ['Distance'].fillna(
    -1 ,inplace = True)
    dataset ['User_id'].fillna(
    -1 , inplace =  True)
    dataset ['Coupon_id'].fillna(
    -1 , inplace =  True)
    dataset ['null_distance'] = dataset ['Distance'].map(
    lambda x : 1 if x == -1 else 0)
    dataset['min_price'] = dataset['Discount_rate'].map(
    lambda x: -1 if ':' not in str(x) else 
    int(str(x).split(':')[0])) 
    print ('Complete1!')
    return dataset
#2.***********************************处理off_test
def Pretreatment_test_data (d):
    dataset = d.copy()
    dataset ['is_Manjian'] = dataset ['Discount_rate'].map(
    lambda x: 1 if ':' in str(x) else 0)
    dataset ['rate'] = dataset ['Discount_rate'].map(
    lambda x: float(x) if':'not in str(x) else
    (float(str(x).split(':')[0])-float(str(x).split(':')[1]))/
    (float(str(x).split(':')[0])))
    dataset ['min in Manjian'] = dataset ['Discount_rate'].map(
    lambda x : -1 if ':' not in str(x) else
    int(str(x).split(':')[0]))
    dataset ['Distance'].fillna(-1 , inplace = True)
    dataset ['User_id'].fillna(-1 , inplace =  True)
    dataset ['Coupon_id'].fillna(-1 , inplace =  True)
    dataset ['null_distance'] = dataset ['Distance'].map(lambda x : 1 if x == -1 else 0)
    dataset['min_price'] = dataset['Discount_rate'].map(lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0])) 
    print ('Complete2!')
    return dataset
#3.**********************************时间处理
def Pretreatment_time (d):
    dataset = d.copy()
    dataset ['date_received'] = pd.to_datetime(dataset ['Date_received'], 
            format = '%Y%m%d')
    if 'Date' in dataset.columns: 
            dataset ['date'] = pd.to_datetime(dataset ['Date'], 
            format = '%Y%m%d')
    dataset ['weekday_Received'] = dataset ['date_received'].apply(lambda x : x.isoweekday())
    dataset ['isweekend'] = dataset ['weekday_Received'].apply(lambda x : 1 if x==5 or x==6 else 0)
    return dataset

#打标 提取特征
def get_label(dataset):
    data = dataset.copy()
    data['label'] = list(map(lambda x,y:1 if (x-y).total_seconds() / (60*1440) <= 15 else 0,
                             data['date'],
                             data['date_received']))
    return data
def get_label_feature(label_field):
    lf = label_field.copy()
    #便于检查此部分代码错误，所以在这期间进行一次单独的简单数据处理
    lf ['Coupon_id'] = lf ['Coupon_id'].map(
        lambda x : int(x) if x==x else 0)
    lf ['Date_received'] = lf ['Date_received'].map(
        lambda x : int(x) if x==x else 0)
    lf ['Distance'] = lf['Distance'].map(
        lambda x:0 if x==-1 else int(x))
    lf ['User_id'] = lf ['User_id'].map(
        lambda x : int(x) if x==x else 0)
    lf ['cnt'] = 1
    c = lf [(lf ['Coupon_id'] != 'null')]
    a = lf [lf ['is_Manjian'] == 1]
    keys = ['User_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'
    f = pd.pivot_table(lf, 
                       index=['User_id'], 
                       values='cnt', aggfunc=len)
    f = pd.DataFrame(f).rename(columns={'cnt': 
                                        prefixs + 'received_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf, f,
                  on=keys, 
                  how='left')
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    t = lf [ ['User_id']+['Date_received'] ].sort_values( 
        ['Date_received'] , ascending = True)
    f = t.drop_duplicates(subset=keys, keep='first')
    f ['label_field_first_1'] = 1
    lf = pd.merge(lf, f, 
                  how = 'left',
                  on = ['User_id' ] + ['Date_received'])
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    t = lf [ ['User_id']+['Date_received'] ].sort_values( 
        ['Date_received'] , ascending = False)
    f = t.drop_duplicates(keys, keep='first')
    f ['label_field_last_2'] = 1
    lf = pd.merge(lf, f,
                  how = 'left',
                  on = ['User_id' ] + ['Date_received'])
    lf.fillna(0, 
    downcast='infer', 
    inplace=True)
    f = pd.pivot_table(lf, 
    index=keys, values='Merchant_id', 
    aggfunc=lambda x:len(
    set(x)))
    f = pd.DataFrame(f).rename(
    columns={'Merchant_id':
    prefixs + 'received_merchant_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id'], 
                  how='left')
    lf.fillna(0,
              downcast='infer', 
              inplace=True)
    f = pd.pivot_table(lf, 
                       index=['User_id'], values='Coupon_id',
                       aggfunc=lambda x:len(set(x)))
    f = pd.DataFrame(f).rename(columns={'Coupon_id':
                                                prefixs + 'received_coupon_cnt'}).reset_index()
    lf = pd.merge(lf, f,
                  on=['User_id'],
                  how='left')
    lf.fillna(0, 
              downcast='infer',
              inplace=True)
    f = pd.pivot_table(lf, 
                       index=['User_id'], values='Discount_rate', 
                       aggfunc=lambda x:len(set(x)))
    f = pd.DataFrame(f).rename(columns={'Discount_rate':
                                        prefixs + 'received_discount_rate_cnt'}).reset_index()
    lf = pd.merge(lf, f, 
                      on=['User_id'], 
                      how='left')
    lf.fillna(0,
              downcast='infer', 
              inplace=True)
    #便于检查此部分代码错误，所以在这期间进行一次单独的简单数据处理
    lf ['Coupon_id'] = lf ['Coupon_id'].map(
        lambda x : int(x) if x==x else 0)
    lf ['Date_received'] = lf ['Date_received'].map(
        lambda x : int(x) if x==x else 0)
    lf ['Distance'] = lf['Distance'].map(
        lambda x:0 if x==-1 else int(x))
    lf ['User_id'] = lf ['User_id'].map(
        lambda x : int(x) if x==x else 0)
    lf ['cnt'] = 1
    c = lf [(lf ['Coupon_id'] != 'null')]
    a = lf [lf ['is_Manjian'] == 1]
    keys = ['User_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'
    lf[prefixs + 'distance_true_rank'] = lf.groupby(keys)['Distance'].rank(ascending = True)
    lf[prefixs + 'distance_false_rank'] = lf.groupby(keys)['Distance'].rank(ascending = False)
    lf[prefixs + 'date_received_true_rank'] = lf.groupby(keys)['Date_received'].rank(ascending = True)
    lf[prefixs + 'date_received_false_rank'] = lf.groupby(keys)['Date_received'].rank(ascending = False)
    lf[prefixs + 'discount_rate_true_rank'] = lf.groupby(keys)['rate'].rank(ascending = True)
    lf[prefixs + 'discount_rate_false_rank'] = lf.groupby(keys)['rate'].rank(ascending = False)
    lf[prefixs + 'min_cost_of_manjian_true_rank'] = lf.groupby(keys)['min in Manjian'].rank(ascending = True)
    lf[prefixs + 'min_cost_of_manjian_false_rank'] = lf.groupby(keys)['min in Manjian'].rank(ascending = False)
    print ("2")
    c = lf [(lf ['Coupon_id'] != 'null')]
    a = lf [lf ['is_Manjian'] == 1]
    
    keys = ['User_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'
    #与距离相关的用户特征------------------------------------------------------------
    dis_a = c [c ['Distance'] != -1]
    f = pd.pivot_table(dis_a,
                       index=['User_id'], values='Distance',
                       aggfunc=lambda x:np.max([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_mean_distance'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id'],
                  how='left')
    lf.fillna(-1, downcast='infer',
              inplace=True)  
    dis_a = c [c ['Distance'] != -1]
    f = pd.pivot_table(dis_a, 
                       index=['User_id'], values='Distance',
                       aggfunc=lambda x:np.max([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_max_distance'}).reset_index()
    lf = pd.merge(lf, f,
                  on=['User_id'],
                  how='left')
    lf.fillna(-1,
              downcast='infer',
              inplace=True)
    dis_a = c [c ['Distance'] != -1]
    f = pd.pivot_table(dis_a, 
                       index=['User_id'], values='Distance',
                       aggfunc=lambda x:np.min([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_min_distance'}).reset_index()
    lf = pd.merge(lf, f,
                  on=['User_id'],
                  how='left')
    lf.fillna(-1,
              downcast='infer',
              inplace=True)
    dis_a = c [c ['Distance'] != -1]
    f = pd.pivot_table(dis_a, 
                       index=['User_id'], values='Distance',
                       aggfunc=lambda x:np.var([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_var_distance'}).reset_index()
    lf = pd.merge(lf, f,
                  on=['User_id'],
                  how='left')
    lf.fillna(-1, 
              downcast='infer',
              inplace=True)
    dis_a = c [c ['Distance'] != -1]
    f = pd.pivot_table(dis_a, 
                       index=['User_id'], values='Distance',
                       aggfunc=lambda x:np.median([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_median_distance'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id'],
                  how='left')
    lf.fillna(-1, 
              downcast='infer',
              inplace=True)
    c = lf [(lf ['Coupon_id'] != 'null')]
    a = lf [lf ['is_Manjian'] == 1]
    
    keys = ['User_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'
    f = pd.pivot_table(a,
                       index=['User_id'], 
                       values='cnt', aggfunc=len)
    f = pd.DataFrame(f).rename(columns={'cnt':
                                        prefixs + 'received_manjian_cnt'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id'],
                  how='left')
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    print ("4")
    f = pd.pivot_table(lf, 
                       index=['User_id'], 
                       values='cnt', aggfunc=len)
    f = pd.DataFrame(f).rename(columns={'cnt': 
                                        prefixs + 'received_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf, f,
                  on=keys, 
                  how='left')
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    c = lf [(lf ['Coupon_id'] != 'null')]
    a = lf [lf ['is_Manjian'] == 1]
    keys = ['Merchant_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'
    lf[prefixs + 'distance_true_rank'] = lf.groupby(keys)['Distance'].rank(
    ascending = True)
    lf[prefixs + 'distance_false_rank'] = lf.groupby(keys)['Distance'].rank(
    ascending = False)
    lf[prefixs + 'date_received_true_rank'] = lf.groupby(keys)['Date_received'].rank(ascending = True)
    lf[prefixs + 'date_received_false_rank'] = lf.groupby(keys)['Date_received'].rank(
    ascending = False)
    lf[prefixs + 'discount_rate_true_rank'] = lf.groupby(keys)['rate'].rank(
    ascending = True)
    lf[prefixs + 'discount_rate_false_rank'] = lf.groupby(keys)['rate'].rank(
    ascending = False)
    lf[prefixs + 'min_cost_of_manjian_true_rank'] = lf.groupby(keys)['min in Manjian'].rank(
    ascending = True)
    lf[prefixs + 'min_cost_of_manjian_false_rank'] = lf.groupby(keys)['min in Manjian'].rank(
    ascending = False)
    f = pd.pivot_table(lf, 
    index=keys, values='User_id', 
    aggfunc=lambda x:len(set(x)))
    f = pd.DataFrame(f).rename(columns={'User_id':
    prefixs + 'received_User_cnt'}).reset_index()
    lf = pd.merge(lf, f, 
    on=keys, 
    how='left')
    lf.fillna(0,
    downcast='infer', 
    inplace=True)
    dis_g = c [c ['Distance'] != -1]
    f = pd.pivot_table(dis_g,
    index=['Merchant_id'], 
    values='Distance', 
    aggfunc=lambda x:np.max(
    [np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
    prefixs + 'received_max_distance'}).reset_index()
    lf = pd.merge(lf, f,
                  on=keys,
                  how='left')
    lf.fillna(-1, 
              downcast='infer', 
              inplace=True)
    dis_g = c [c ['Distance'] != -1]
    f = pd.pivot_table(dis_g, 
                       index=keys, values='Distance', 
                       aggfunc=lambda x:np.min([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_min_distance'}).reset_index()
    lf = pd.merge(lf, f,
                  on=keys,
                  how='left')
    lf.fillna(-1, 
              downcast='infer', 
              inplace=True)
    dis_g = c [c ['Distance'] != -1]
    f = pd.pivot_table(dis_g, 
                       index=keys, values='Distance', 
                      aggfunc=lambda x:np.median([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_mean_distance'}).reset_index()
    lf = pd.merge(lf, f,
                  on=keys,
                  how='left')
    lf.fillna(-1, 
              downcast='infer', 
              inplace=True)
    dis_g = c [c ['Distance'] != -1]
    f = pd.pivot_table(dis_g,
                       index=keys, values='Distance', 
                       aggfunc=lambda x:np.var([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_var_distance'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=keys,
                  how='left')
    lf.fillna(-1, 
              downcast='infer', 
              inplace=True)
    print ("10")
    keys = ['Coupon_id']
    prefixs = 'label_field_' + '_'.join(keys) + '_'
    lf[prefixs + 'distance_true_rank'] = lf.groupby(keys)['Distance'].rank(
        ascending = True)
    lf[prefixs + 'distance_false_rank'] = lf.groupby(keys)['Distance'].rank(
        ascending = False)

    lf[prefixs + 'date_received_true_rank'] = lf.groupby(keys)['Date_received'].rank(
        ascending = True)
    lf[prefixs + 'date_received_false_rank'] = lf.groupby(keys)['Date_received'].rank(
        ascending = False)
    f = pd.pivot_table(lf, 
                       index=['Coupon_id'], values='User_id',
                       aggfunc=lambda x:len(set(x)))
    f = pd.DataFrame(f).rename(columns={'User_id':
                                        prefixs + 'received_user_cnt'}).reset_index()
    lf = pd.merge(lf, f,
                  on=['Coupon_id'], 
                  how='left')
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    dis_g = c [c ['Distance'] != -1]
    f = pd.pivot_table(dis_g, 
                       index=keys, values='Distance', 
                       aggfunc=lambda x:np.mean([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_mean_distance'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=keys, 
                  how='left')
    lf.fillna(-1,
              downcast='infer', 
              inplace=True)
    dis_g = c [c ['Distance'] != -1]
    f = pd.pivot_table(dis_g, 
                       index=keys, values='Distance', 
                       aggfunc=lambda x:np.max([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_max_distance'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=keys, 
                  how='left')
    lf.fillna(-1, 
              downcast='infer', 
              inplace=True)
    dis_g = c [c ['Distance'] != -1]
    f = pd.pivot_table(dis_g, 
                       index=keys, values='Distance', 
                      aggfunc=lambda x:np.min([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_min_distance'}).reset_index()
    lf = pd.merge(lf, f,
                  on=keys, 
                  how='left')
    lf.fillna(-1, 
              downcast='infer', 
              inplace=True)
    dis_g = c [c ['Distance'] != -1]
    f = pd.pivot_table(dis_g,
                       index=keys, values='Distance', 
                       aggfunc=lambda x:np.var([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_var_distance'}).reset_index()
    lf = pd.merge(lf, f,
                  on=keys, 
                  how='left')
    lf.fillna(-1, 
              downcast='infer', 
              inplace=True)
    c = lf [(lf ['Coupon_id'] != 'null')]
    a = lf [lf ['is_Manjian'] == 1]
    keys = ['User_id', 
            'Merchant_id']
    prefixs = 'label_field_' + '_'.join(['User_id', 
                                         'Merchant_id']) + '_'
    f = pd.pivot_table(lf, 
                       index=['User_id', 'Merchant_id'], values='cnt', aggfunc=len)
    f = pd.DataFrame(f).rename(columns={'cnt':
                                        prefixs + 'received_cnt'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id', 'Merchant_id'], 
                  how='left')
    lf.fillna(0, 
              downcast='infer',
              inplace=True)
    t = lf[['User_id', 'Merchant_id'] + ['Date_received']].sort_values(['Date_received'], ascending=True)
    f = t.drop_duplicates(['User_id', 'Merchant_id'], keep='first')
    f[prefixs + 'is_first_receive'] = 1
    lf = pd.merge(lf, f , 
                  on=['User_id', 'Merchant_id'] + ['Date_received'],
                  how='left')
    lf.fillna(0, 
              downcast='infer',
              inplace=True)
    t = lf[['User_id', 'Merchant_id'] + ['Date_received']].sort_values(['Date_received'], ascending=False)
    f = t.drop_duplicates(['User_id', 'Merchant_id'], keep='last')
    f[prefixs + 'is_last_receive'] = 1
    lf = pd.merge(lf, f , 
                  on=['User_id', 'Merchant_id'] + ['Date_received'],
                  how='left')
    lf.fillna(0, 
              downcast='infer',
              inplace=True)
    f = pd.pivot_table(lf,
                       index=['User_id', 'Merchant_id'], 
                       values='Coupon_id',
                       aggfunc=lambda x:len(set(x)))
    f = pd.DataFrame(f).rename(columns={'Coupon_id':
                                        prefixs + 'received_coupon_cnt'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id', 'Merchant_id'], 
                  how='left')
    lf.fillna(0,
              downcast='infer',
              inplace=True)
    t = lf [['User_id', 'Coupon_id'] + ['Date_received']].sort_values(
        ['Date_received'], ascending=True)
    f = t.drop_duplicates(['User_id', 'Coupon_id'], 
                          keep='first')
    f [prefixs + 'is_first_receive'] = 1
    lf = pd.merge(lf, f, 
                  on=['User_id', 'Coupon_id'] + ['Date_received'], 
                  how='left')
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    t = lf [['User_id', 'Coupon_id'] + ['Date_received']].sort_values(
        ['Date_received'], ascending=False)
    f = t.drop_duplicates(['User_id', 'Coupon_id'], 
                          keep='last')
    f [prefixs + 'is_last_receive'] = 1
    lf = pd.merge(lf, f, 
                  on=['User_id', 'Coupon_id'] + ['Date_received'], 
                  how='left')
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    c = lf [(lf ['Coupon_id'] != 'null')]
    a = lf [lf ['is_Manjian'] == 1]
    keys = ['User_id', 'Merchant_id', 'Date_received']
    prefixs = 'label_field_' + '_'.join(['User_id', 'Merchant_id', 'Date_received']) + '_' 
    t = lambda x: 1 if len(x) > 1 else 0
    f = pd.pivot_table(lf, 
                       index=['User_id', 'Merchant_id', 'Date_received'], 
                       values='cnt',aggfunc=t)
    print ("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{")
    f = pd.DataFrame(f).rename(columns={'cnt': 
                                        prefixs + 'repeat_receive'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id', 'Merchant_id', 'Date_received'], 
                  how='left')
    lf.fillna(0, 
              downcast='infer',
              inplace=True)
    f = pd.pivot_table(lf, 
                       index=['User_id', 'Merchant_id', 'Date_received'],
                       values='cnt', aggfunc=np.mean)
    f = pd.DataFrame(f).rename(columns={'cnt': 
                                        prefixs + 'received_mean_cnt'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id', 'Merchant_id', 'Date_received'],
                  how='left')
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    f = pd.pivot_table(lf, 
                       index=['User_id', 'Merchant_id', 'Date_received'], 
                       values='cnt', aggfunc=np.max)
    f = pd.DataFrame(f).rename(columns={'cnt': 
                                        prefixs + 'received_max_cnt'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id', 'Merchant_id', 'Date_received'],
                  how='left')
    lf.fillna(0, 
              downcast='infer',
              inplace=True)
    f = pd.pivot_table(lf,
                       index=['User_id', 'Merchant_id', 'Date_received'], 
                       values='cnt', aggfunc=np.min)
    f = pd.DataFrame(f).rename(columns={'cnt': 
                                        prefixs + 'received_min_cnt'}).reset_index()
    lf = pd.merge(lf, f,
                  on=['User_id', 'Merchant_id', 'Date_received'],
                  how='left')
    lf.fillna(0, 
              downcast='infer',
              inplace=True)
    f = pd.pivot_table(lf, 
                       index=['User_id', 'Merchant_id', 'Date_received'], 
                       values='cnt', aggfunc=np.mean)
    f = pd.DataFrame(f).rename(columns={'cnt': 
                                        prefixs + 'received_mean_cnt'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id', 'Merchant_id', 'Date_received'],
                  how='left')
    lf.fillna(0,
              downcast='infer',
              inplace=True)
    f = pd.pivot_table(lf, index=['User_id', 'Merchant_id', 'Date_received'], 
                       values='cnt', aggfunc=np.median)
    f = pd.DataFrame(f).rename(columns={'cnt': 
                                        prefixs + 'received_median_cnt'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id', 'Merchant_id', 'Date_received'],
                  how='left')
    lf.fillna(0, 
              downcast='infer',
              inplace=True)
    f = pd.pivot_table(lf,
                       index=['User_id', 'Merchant_id', 'Date_received'], 
                       values='cnt', aggfunc=np.var)
    f = pd.DataFrame(f).rename(columns={'cnt': 
                                        prefixs + 'received_var_cnt'}).reset_index()
    lf = pd.merge(lf, f,
                  on=['User_id', 'Merchant_id', 'Date_received'],
                  how='left')
    lf.fillna(0, 
              downcast='infer',
              inplace=True)

    lf.fillna(0,
              downcast='infer', 
              inplace=True)
    lf.drop(['cnt'], axis=1, 
            inplace=True)
    return lf
def get_history_User_feature(history_field, label_field):
    data = history_field.copy()
    data['Date_received'] = data['Date_received'].map(
        lambda x : int(x) if x==x else 0)
    data['Coupon_id'] = data['Coupon_id'].map(
        lambda x : int(x) if x==x else 0)
    data['cnt'] = 1
    a = data[data['Date'].map(lambda x:str(x) != 'nan')]
    b = data[data['Date'].map(lambda x:str(x) == 'nan')]
    c = data[data['label'] == 1]
    d = data[data['label'] != 1]
    keys = ['User_id']
    prefixs = 'history_field_' + '_'.join(['User_id']) + '_'
    lf = label_field[keys].drop_duplicates(keep = 'first')
    

    f = pd.pivot_table(data, 
                       index=['User_id'],
                       values='cnt', aggfunc=len)
    f = pd.DataFrame(f).rename(columns={'cnt':
                                            prefixs + 'received_cnt'}).reset_index()
    lf = pd.merge(lf, f,
                  on=['User_id'],
                  how='left')
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    
    f = pd.pivot_table(a, 
                       index=keys, values='cnt', 
                       aggfunc=len)
    f = pd.DataFrame(f).rename(columns={'cnt':
                                        prefixs + 'received_and_cost_cnt'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id'], 
                  how='left')
    lf.fillna(0,
    downcast='infer',
    inplace=True)
    f = pd.pivot_table(
    b,index=keys, 
    values='cnt', 
    aggfunc=len)
    f = pd.DataFrame(f).rename(columns={'cnt':
    prefixs + 'received_not_cost_cnt'}).reset_index()
    lf = pd.merge(lf, f,
    on=keys, 
    how='left')
    lf.fillna(0, 
    downcast='infer',
    inplace=True)
    lf[prefixs + 'received_and_cost_rate'] = list(map(
    lambda x,y:x/y if y != 0 else 0, 
    lf[prefixs + 'received_and_cost_cnt'], 
    lf[prefixs + 'received_cnt']))
    f = pd.pivot_table(b, 
    index=['User_id'],
    values='cnt', aggfunc=len)
    f = pd.DataFrame(f).rename(columns={'cnt':
    prefixs + 'received_and_cost_15d_cnt'}).reset_index()
    lf = pd.merge(lf, f, 
    on=keys,
    how='left')
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    f = pd.pivot_table(d,
                       index=['User_id'],
                       values='cnt', aggfunc=len)
    f = pd.DataFrame(f).rename(columns={'cnt':
                                        prefixs + 'received_not_cost_15d_cnt'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id'], 
                  how='left')
    lf.fillna(0, 
              downcast='infer',
              inplace=True)
    lf[prefixs + 'received_and_cost_15d_rate'] = list(map(
        lambda x,y:x/y if y != 0 else 0, 
          lf[prefixs + 'received_and_cost_15d_cnt'], 
          lf[prefixs + 'received_cnt']))
    f = pd.pivot_table(data, 
                       index=['User_id'], 
                       values='Merchant_id', aggfunc=lambda x:len(set(x)))
    f = pd.DataFrame(f).rename(columns={'Merchant_id':
                                        prefixs + 'received_differ_merchant'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id'], 
                  how='left')
    lf.fillna(0,
              downcast='infer', 
              inplace=True)
    
    f = pd.pivot_table(c,index=['User_id'],
    values='Merchant_id', aggfunc=lambda x:len(set(x)))
    f = pd.DataFrame(f).rename(columns={'Merchant_id':
    prefixs + 'received_and_cost_differ_merchant_15d'})
    f = f.reset_index()
    lf = pd.merge(lf, f,
    on=['User_id'],
    how='left')
    lf.fillna(0, 
    downcast='infer', 
    inplace=True)
    lf[prefixs + 'received_and_cost_differ_merchant_15d_rate'] = list(map(
    lambda x,y:x/y if y != 0 else 0, 
    lf[prefixs + 'received_and_cost_differ_merchant_15d'],
    lf[prefixs + 'received_differ_merchant']))
    lf[prefixs + 'received_and_cost_div_15d_differ_merchant'] = list(map(
    lambda x,y:x/y if y != 0 else 0, 
    lf[prefixs + 'received_and_cost_cnt'], 
    lf[prefixs + 'received_and_cost_differ_merchant_15d']))
    f = pd.pivot_table(c, 
                       index=['User_id'], 
                       values='Distance', aggfunc=lambda x:np.mean(
                           [np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_and_cost_15d_mean_distance'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id'],
                  how='left')
    lf.fillna(-1, 
              downcast='infer',
              inplace=True)
    f = pd.pivot_table(c, index=keys, 
    values='Distance', aggfunc=lambda x:
    np.max([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_and_cost_15d_max_distance'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id'], 
                  how='left')
    lf.fillna(-1, 
              downcast='infer', 
              inplace=True)
    f = pd.pivot_table(c, 
                       index=['User_id'], 
                       values='Distance', aggfunc=lambda x:np.min(
                           [np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_and_cost_15d_min_distance'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id'], 
                  how='left')
    lf.fillna(-1, 
              downcast='infer', 
              inplace=True)
    f = pd.pivot_table(c, 
                       index=['User_id'],
                       values='Distance', aggfunc=lambda x:np.median(
                           [np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_and_cost_15d_median_distance'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=keys,
                  how='left')
    lf.fillna(-1,
              downcast='infer', 
              inplace=True)
    f = pd.pivot_table(c, index=keys, 
    values='Distance', aggfunc=lambda x:
    np.var([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'Distance':
                                        prefixs + 'received_and_cost_15d_var_distance'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id'], 
                  how='left')
    lf.fillna(-1, 
              downcast='infer', 
              inplace=True)
    f = pd.pivot_table(data, 
                       index=['User_id'],
                       values='Coupon_id', aggfunc=lambda x:len(set(x)))
    f = pd.DataFrame(f).rename(columns={'Coupon_id':
                                        prefixs + 'received_differ_coupon'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id'],
                  how='left')
    lf.fillna(0,
              downcast='infer', 
              inplace=True)
    f = pd.pivot_table(c, 
                       index=['User_id'], 
                       values='Coupon_id', aggfunc=lambda x:len(set(x)))
    f = pd.DataFrame(f).rename(columns={'Coupon_id':
                                        prefixs + 'received_and_cost_differ_coupon_15d'}).reset_index()
    lf = pd.merge(lf, f,
                  on=['User_id'], 
                  how='left')
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    lf[prefixs + 'received_and_cost_differ_coupon_15d_rate'] = list(map(
    lambda x,y:x/y if y != 0 else 0, 
    lf[prefixs + 'received_and_cost_differ_coupon_15d'], 
    lf[prefixs + 'received_differ_coupon']))
    t = data[data['label'] == 1]
    t['gap'] = (t['date'] - t['date_received']).map(
    lambda x:x.total_seconds()/
    (60*1440))
    f = pd.pivot_table(t, index=['User_id'], 
                       values='gap', aggfunc=np.mean)
    f = pd.DataFrame(f).rename(columns={'gap':
                                        prefixs + 'cost_mean_time_gap_15d'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id'], 
                  how='left')
    lf.fillna(-1, 
              downcast='infer', 
              inplace=True)
    f = pd.pivot_table(t, 
                       index=['User_id'],
                       values='gap', aggfunc=np.min)
    f = pd.DataFrame(f).rename(columns={'gap':
                                        prefixs + 'cost_min_time_gap_15d'}).reset_index()
    lf = pd.merge(lf, f,
                  on=['User_id'],
                  how='left')
    lf.fillna(-1, 
              downcast='infer',
              inplace=True)
    f = pd.pivot_table(c, 
                       index=['User_id'], 
                       values='rate', aggfunc=np.mean)
    f = pd.DataFrame(f).rename(columns={'rate':
    prefixs + 'received_and_cost_15d_mean_discount_rate'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id'],
                  how='left')
    lf.fillna(0, 
              downcast='infer',
              inplace=True)
    f = pd.pivot_table(c, 
                       index=['User_id'],
                       values='rate', aggfunc=np.max)
    f = pd.DataFrame(f).rename(columns={'rate':
    prefixs + 'received_and_cost_15d_max_discount_rate'}).reset_index()
    lf = pd.merge(lf, f,
                  on=['User_id'],
                  how='left')
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    f = pd.pivot_table(c,
                       index=['User_id'],
                       values='rate', aggfunc=np.min)
    f = pd.DataFrame(f).rename(columns={'rate':
    prefixs + 'received_and_cost_15d_min_discount_rate'}).reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id'],
                  how='left')
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    
    #用户15天内核销的折扣率中位数
    f = pd.pivot_table(c, 
                       index=['User_id'],
                       values='rate', aggfunc=np.median)
    f = pd.DataFrame(f).rename(columns={'rate':
    prefixs + 'received_and_cost_15d_median_discount_rate'}).reset_index()
    lf = pd.merge(lf, f,
                  on=keys, 
                  how='left')
    lf.fillna(0,
              downcast='infer', 
              inplace=True)
    
    #用户15天内核销的折扣率方差
    f = pd.pivot_table(c,
                       index=['User_id'], 
                       values='rate', aggfunc=np.var)
    f = pd.DataFrame(f).rename(columns={'rate':
    prefixs + 'received_and_cost_15d_var_discount_rate'}).reset_index()
    lf = pd.merge(lf, f, 
    on=keys,
    how='left')
    lf.fillna(0, 
    downcast='infer',
    inplace=True)
    #填充空值
    lf.fillna(0, 
    downcast='infer', 
    inplace=True)
    data.drop(['cnt'],
    axis=1, 
    inplace=True)
    return lf
def get_history_Merchant_feature(history_field, label_field):
    #进行深度复制
    hf = history_field.copy()
    #便于检查此部分代码错误，所以在这期间进行一次单独的简单数据处理
    hf ['Coupon_id'] = hf ['Coupon_id'].map(
        lambda x : int(x) if x==x else 0)
    hf ['Date_received'] = hf ['Date_received'].map(
        lambda x : int(x) if x==x else 0)
    hf ['cnt'] = 1
#与消费相关的用户特征------------------------------------------------------------
#只选出15天内核销的数据
    g = hf [hf ['label'] == 1]
    e = hf [(hf ['date'] != 'nan')]
    h = hf [(hf ['date'] == 'nan')]
    keys = ['Merchant_id']
    prefixs = 'history_merchant_feature' + '_'.join(keys) +'_'
    lf = label_field [keys].drop_duplicates (keep = 'first')
    f = pd.pivot_table(hf , 
                       index = ['Merchant_id'] ,
                       values = 'cnt' , aggfunc = len)
    f = pd.DataFrame(f).rename(columns={'cnt':
                                        prefixs + 'received_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf,f,
                  how = 'left'
                  ,on = ['Merchant_id']) 
    lf.fillna(0 , 
              downcast = 'infer' ,
              inplace = True)
    f = pd.pivot_table(e ,
                       index = ['Merchant_id'] ,
                       values = 'cnt' , aggfunc = len)
    f = pd.DataFrame(f).rename(columns={'cnt':
                                        prefixs + 'received_and_cost_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf,f,
                  how = 'left',
                  on = ['Merchant_id']) 
    lf.fillna(0 ,
              downcast = 'infer' , 
              inplace = True)
    f = pd.pivot_table(h , 
                       index = ['Merchant_id'] ,
                       values = 'cnt' , aggfunc = len)
    f = pd.DataFrame(f).rename(columns={'cnt':
                                        prefixs + 'received_and_not_cost_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf,f,
                  how = 'left',
                  on = ['Merchant_id']) 
    lf.fillna(0 , 
              downcast = 'infer' ,
              inplace = True)
    lf ['M_consume_rate'] = lf [prefixs + 'received_and_cost_cnt'] / (
        lf [prefixs + 'received_cnt'] )#+ lf [prefixs + 'received_and_not_cost_cnt'] )
    f = pd.pivot_table(hf , 
                       index = ['Merchant_id'] ,
                       values = 'User_id' , aggfunc = lambda x:len(set(x)))
    f = pd.DataFrame(f).rename(columns={'User_id':
                                        prefixs + 'received_different_user_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf,f,
                  how = 'left',
                  on = ['Merchant_id']) 
    lf.fillna(0 , 
              downcast = 'infer' ,
              inplace = True)
    f = g.groupby( ['Merchant_id'] ).User_id.agg(
        lambda x:len(set(x))).reset_index(
        name = 'M_differernt_user_receive_consume')
    lf = pd.merge(lf, f, 
                  how = 'left', 
                  on = ['Merchant_id'])
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    f = pd.pivot_table(g , 
                       index = ['Merchant_id'] ,
                       values = 'cnt' , aggfunc = len)
    f = pd.DataFrame(f).rename(columns={'cnt':
                                        prefixs + '15_received_and_cost_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf,f,
                  how = 'left',
                  on = ['Merchant_id']) 
    lf.fillna(0 , 
              downcast = 'infer' ,
              inplace = True)
    f = pd.pivot_table(hf, 
                       index=keys, 
                       values='Coupon_id', aggfunc=lambda x:len(set(x)))
    f = pd.DataFrame(f).rename(columns={'Coupon_id':
                                        prefixs + 'differ_coupon_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf, f ,
                  on = 'Merchant_id' , 
                  how = 'left')
    lf.fillna(0, 
              downcast = 'infer' , 
              inplace=True)
    lf [prefixs + 'per_u_cost_cnt'] = lf [prefixs + 'received_and_cost_cnt'] / lf[prefixs + 'received_cnt']
    lf [prefixs + 'per_c_cost_cnt'] = lf [prefixs + 'received_and_cost_cnt'] / lf[prefixs + 'differ_coupon_cnt']
    f = pd.pivot_table(g , 
                       index = ['Merchant_id'] ,
                       values = 'rate', 
                       aggfunc=lambda x:np.max([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'rate':
                                        prefixs + '15_received_and_consume_rate_max'})
    f = f.reset_index()
    lf = pd.merge(lf,f,
                  how = 'left',
                  on = ['Merchant_id']) 
    lf.fillna(0,
              downcast='infer', 
              inplace=True)
    f = pd.pivot_table(g , 
                       index = ['Merchant_id'] ,
                       values = 'rate',
                       aggfunc=lambda x:np.min([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'rate':
                                        prefixs + '15_received_and_consume_rate_min'})
    f = f.reset_index()
    lf = pd.merge(lf,f,
                  how = 'left',
                  on = ['Merchant_id']) 
    lf.fillna(0,
    downcast='infer', 
    inplace=True)
    f = pd.pivot_table(g , 
    index = ['Merchant_id'] ,
    values = 'rate', 
    aggfunc=lambda x:np.mean([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'rate':
    prefixs + '15_received_and_consume_rate_mean'})
    f = f.reset_index()
    lf = pd.merge(lf,f,
    how = 'left',
    on = ['Merchant_id']) 
    lf.fillna(0, 
    downcast='infer', 
    inplace=True)
    f = pd.pivot_table(g , index = keys ,values = 'rate', 
    aggfunc=lambda x:np.var([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'rate':
    prefixs + '15_received_and_consume_rate_var'})
    f = f.reset_index()
    lf = pd.merge(lf,f,how = 'left',
    on = ['Merchant_id']) 
    lf.fillna(0,
    downcast='infer',
    inplace=True)
    f = pd.pivot_table(g , index = keys ,values = 'rate', 
    aggfunc=lambda x:np.median([np.nan if i == -1 else i for i in x]))
    f = pd.DataFrame(f).rename(columns={'rate':
    prefixs + '15_received_and_consume_rate_median'})
    f = f.reset_index()
    lf = pd.merge(lf,f,
    how = 'left',
    on = ['Merchant_id']) 
    lf.fillna(0, 
    downcast='infer',
    inplace=True)
#有关于商家15天内的时间特征------------------------------------------------------
    g ['gap'] = ( g ['date'] - g ['date_received'] ).map(
        lambda x: x.total_seconds() / (60 * 1440))
    f = pd.pivot_table(g , 
                       index = ['Merchant_id'] , values ='gap' ,
                      aggfunc = np.max)
    f = pd.DataFrame(f).rename(columns = {'gap':
                                          prefixs + 'max_consume_gap_merchant'})
    f = f.reset_index()
    lf = pd.merge( lf , f,
                  on='Merchant_id',
                  how='left' )
    g ['gap'] = ( g ['date'] - g ['date_received'] ).map(
        lambda x: x.total_seconds() / (60 * 1440))
    f = pd.pivot_table(g , 
                       index = ['Merchant_id'], 
                       values ='gap' ,
                      aggfunc = np.min)
    f = pd.DataFrame(f).rename(columns = {'gap':
                                          prefixs + 'min_consume_gap_merchant'})
    f = f.reset_index()
    lf = pd.merge( lf , f,
                  on='Merchant_id',
                  how='left' ) 
    g ['gap'] = ( g ['date'] - g ['date_received'] ).map(
        lambda x: x.total_seconds() / (60 * 1440))
    f = pd.pivot_table(g , 
                       index = ['Merchant_id'] ,
                       values ='gap' ,
                      aggfunc = np.mean)
    f = pd.DataFrame(f).rename(columns = {'gap':
                                          prefixs + 'mean_consume_gap_merchant'})
    f = f.reset_index()
    lf = pd.merge( lf , f,
                  on='Merchant_id',
                  how='left' )
#有关与商家十五天内消费距离的特征-------------------------------------------------
    dis_g = g [g ['Distance'] != -1]
    f ['Distance'] = dis_g ['Distance']
    f = f [['Merchant_id',
    'Distance']]
    f ['Distance'] = dis_g ['Distance']
    f = f.groupby ('Merchant_id').agg(
    'max').reset_index()
    f = f.rename(columns={'Distance':
    prefixs + '15_max_receive_consume_distance'})
    lf = pd.merge(lf, f, how = 'left', 
    on = ['Merchant_id'])
    dis_g = g [g ['Distance'] != -1]
    f ['Distance'] = dis_g ['Distance']
    f = f[['Merchant_id',
    'Distance']]
    f ['Distance'] = dis_g ['Distance']
    f = f.groupby ('Merchant_id').agg(
    'min').reset_index()
    f = f.rename(columns={'Distance':
    prefixs + '15_min_receive_consume_distance'})
    lf = pd.merge(lf, f, how = 'left', 
    on = ['Merchant_id'])
    dis_g = g [g ['Distance'] != -1]
    f ['Distance'] = dis_g ['Distance']
    f = f[['Merchant_id',
    'Distance']]
    f ['Distance'] = dis_g ['Distance']
    f = f.groupby ('Merchant_id').agg(
    'mean').reset_index()
    f = f.rename(columns={'Distance':
    prefixs + '15_mean_receive_consume_distance'})
    lf = pd.merge(lf, f, how = 'left', 
    on = ['Merchant_id'])
    dis_g = g [g ['Distance'] != -1]
    f ['Distance'] = dis_g ['Distance']
    f = f[['Merchant_id',
    'Distance']]
    f ['Distance'] = dis_g ['Distance']
    f = f.groupby ('Merchant_id').agg(
    'var').reset_index()
    f = f.rename(columns={'Distance':
    prefixs + '15_var_receive_consume_distance'})
    lf = pd.merge(lf, f, how = 'left',
    on = ['Merchant_id'])
    dis_g = g [g ['Distance'] != -1]
    f ['Distance'] = dis_g ['Distance']
    f = f[['Merchant_id',
    'Distance']]
    f ['Distance'] = dis_g ['Distance']
    f = f.groupby ('Merchant_id').agg(
    'median').reset_index()
    f = f.rename(columns={'Distance':
    prefixs + '15_median_receive_consume_distance'})
    lf = pd.merge(lf, f, how = 'left', 
    on = ['Merchant_id'])
    print (lf)
    print ("55555555555555555555555555555555555555")
    return lf
def get_history_Coupon_feature(history_field, label_field):
    hf = history_field.copy()
    #便于检查此部分代码错误，所以在这期间进行一次单独的简单数据处理
    hf ['Coupon_id'] = hf ['Coupon_id'].map(
        lambda x : int(x) if x==x else 0)
    hf ['Date_received'] = hf ['Date_received'].map(
        lambda x : int(x) if x==x else 0)
    hf ['cnt'] = 1
    a = hf [hf ['is_Manjian'] == 1]
#与消费相关的用户特征------------------------------------------------------------
    g = hf [hf ['label'] == 1]
    keys = ['Coupon_id']
    prefixs = 'history_coupon_feature' + '_'.join(keys) +'_'
    lf = label_field [keys].drop_duplicates (keep = 'first')
    f = pd.pivot_table(hf , 
                       index=['Coupon_id'],
                       values='cnt', aggfunc=len)
    f = pd.DataFrame(f).rename(columns={'cnt':
                                                prefixs + 'received_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf, f, 
                  on=['Coupon_id'], 
                  how='left')
    lf.fillna(-1, 
              downcast='infer',
              inplace=True)
    f = pd.pivot_table(g ,
                       index=['Coupon_id'],
                       values='cnt', aggfunc=len)
    f = pd.DataFrame(f).rename(columns={'cnt':
                                                prefixs + '15_received_and_consume_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf, f, 
                  on=['Coupon_id'], 
                  how='left')
    lf.fillna(-1,
              downcast='infer',
              inplace=True)
    lf [prefixs + '15_r_&_c_rate'] = (
        lf [prefixs + '15_received_and_consume_cnt'] / lf[prefixs + 'received_cnt'])
    g ['gap'] = (g ['date'] - g ['date_received']).map(
        lambda x:x.total_seconds()/(60*1440))
    f = pd.pivot_table(g, 
                       index=['Coupon_id'], 
                       values='gap', aggfunc=np.mean)
    f = pd.DataFrame(f).rename(columns={'gap':
                                        prefixs + 'cost_mean_time_gap_15d'})
    f = f.reset_index()
    lf = pd.merge(lf, f, 
                  on=['Coupon_id'], 
                  how='left')
    lf.fillna(-1,
              downcast='infer',
              inplace=True)
    f = pd.pivot_table(a,
                       index= ['Coupon_id'], 
                       values='min in Manjian', aggfunc=np.mean)
    f = pd.DataFrame(f).rename(columns={'min in Manjian':
                                        prefixs + 'min_cost_of_manjian_mean'})
    f = f.reset_index()
    lf = pd.merge(lf, f,
                  on=keys,
                  how='left')
    lf.fillna(-1,
              downcast='infer', 
              inplace=True)
    print (lf)
    print ("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^6")
    return lf
def get_history_User_Merchant_feature(history_field, label_field):
    hf = history_field.copy()
    hf ['Coupon_id'] = hf ['Coupon_id'].map(
        lambda x : int(x) if x==x else 0)
    hf ['Date_received'] = hf ['Date_received'].map(
        lambda x : int(x) if x==x else 0)
    hf ['cnt'] = 1
    keys = ['User_id', 'Merchant_id']
    prefixs = 'history_u_m_field_' + '_'.join(keys) + '_'
    lf = label_field[keys].drop_duplicates(keep = 'first')
    f = pd.pivot_table(hf, 
                       index=['User_id', 'Merchant_id'], 
                       values='cnt', aggfunc=len)
    f = pd.DataFrame(f).rename(columns={'cnt':
                                        prefixs + 'received_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id', 'Merchant_id'], 
                  how='left')
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    f = pd.pivot_table(hf [hf['Date'].map(
        lambda x:str(x) != 'nan')], 
                       index=['User_id', 'Merchant_id'],
                       values='cnt', aggfunc=len)
    f = pd.DataFrame(f).rename(columns={'cnt':
                                        prefixs + 'received_and_cost_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf, f, 
                  on=['User_id', 'Merchant_id'],
                  how='left')
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    f = pd.pivot_table(hf [hf ['Date'].map(
        lambda x:str(x) == 'nan')], 
                       index=['User_id', 'Merchant_id'], 
                       values='cnt', aggfunc=len)
    f = pd.DataFrame(f).rename(columns={'cnt':
                                        prefixs + 'received_not_cost_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf, f,
                  on=['User_id', 'Merchant_id'], 
                  how='left')
    lf.fillna(0, 
              downcast='infer', 
              inplace=True)
    return lf
def get_history_User_Coupon_received_feature(history_field, label_field):
    hf = history_field.copy()
    hf ['Date_received'] = hf ['Date_received'].map(
        lambda x : int(x) if x==x else 0)
    hf ['Coupon_id'] = hf ['Coupon_id'].map(
        lambda x : int(x) if x==x else 0)
    hf ['cnt'] = 1
    keys = ['User_id', 'Coupon_id']
    prefixs = 'history_u_c_field_' + '_'.join(keys) + '_'
    lf = label_field[keys].drop_duplicates(
        keep = 'first')
    f = pd.pivot_table(hf,
                       index=keys, 
                       values='cnt',
                       aggfunc=len)
    f = pd.DataFrame(f).rename(columns={
        'cnt':
        prefixs + 'received_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf, f, 
        on=['User_id', 
        'Coupon_id'],
        how='left')
    lf.fillna(-1,
        downcast='infer', 
        inplace=True)
    f = pd.pivot_table(hf[hf['Date'].map(
        lambda x:str(x) != 'nan')], 
        index=keys, 
        values='cnt', aggfunc=len)
    f = pd.DataFrame(f).rename(columns={'cnt':
        prefixs + 'received_and_cost_cnt'})
    f = f.reset_index()
    lf = pd.merge(lf, f, 
        on=['User_id', 'Coupon_id'], 
        how='left')
    lf.fillna(-1, 
        downcast='infer',
        inplace=True)
    lf ['received_consume_devide_received_cnt'] = (
        lf [prefixs + 'received_and_cost_cnt'] / lf [
        prefixs + 'received_cnt'])
    hf.drop(['cnt'], 
        axis=1, 
        inplace=True)
    return lf
def get_history_User_Date_received_feature(history_field, label_field):
    hf = history_field.copy()
    hf ['Date_received'] = hf ['Date_received'].map(
        lambda x : int(x) if x==x else 0)
    hf ['Coupon_id'] = hf ['Coupon_id'].map(
        lambda x : int(x) if x==x else 0)
    hf ['cnt'] = 1
    keys = ['User_id', 'Date_received']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    lf = label_field[keys].drop_duplicates(
        keep = 'first')
    f = pd.pivot_table(hf, 
        index=['User_id', 'Date_received'],
        values='cnt',
        aggfunc = len)
    f = pd.DataFrame(f).rename(columns={'cnt':
        prefixs + 'received_cnt'}).reset_index()
    lf = pd.merge(lf, f, 
        on=['User_id', 'Date_received'],
        how='left')
    lf.fillna(0,
        downcast='infer',
        inplace=True)
    hf.drop(['cnt'], 
        axis=1, 
        inplace=True)
    return lf
def get_all_history_feature(history, label):   
    u_feature = get_history_User_feature(
        history, 
        label)
    m_feature = get_history_Merchant_feature(
        history, 
        label)
    c_feature = get_history_Coupon_feature(
        history,
        label)
    um_feature = get_history_User_Merchant_feature(
        history, 
        label)
    ud_feature = get_history_User_Date_received_feature(
        history,
        label)
    uc_feature = get_history_User_Coupon_received_feature(
        history,
        label)
    lf = label.copy()   
    lf = pd.merge(lf, u_feature,
        on=['User_id'], 
        how='left')
    lf = pd.merge(lf, m_feature, 
        on=['Merchant_id'], 
        how='left')
    lf = pd.merge(lf, c_feature, 
        on=['Coupon_id'],
        how='left')
    lf = pd.merge(lf, uc_feature,
        on=['User_id', 'Coupon_id'],
        how='left')
    lf = pd.merge(lf, um_feature, 
        on=['User_id', 'Merchant_id'],
        how='left')
    lf = pd.merge(lf, ud_feature,
        on=['User_id', 'Date_received'],
        how='left') 
    return lf
def get_week_feature(label_field):
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(
        lambda x : int(x) if x==x else 0)
    data['Date_received'] = data['Date_received'].map(
        lambda x : int(x) if x==x else 0)
    # 返回的特征数据集
    w_feat = data.copy()
    w_feat['week'] = w_feat['date_received'].map(
        lambda x: 
        x.weekday())  
    w_feat['is_weekend'] = w_feat['week'].map(
        lambda x: 
        1 if x == 5 or 
        x == 6 else
        0)  
    w_feat = pd.concat([w_feat, 
        pd.get_dummies(w_feat['week'],
        prefix='week')], 
        axis=1) 
    w_feat.index = range(
        len(w_feat))  
    return w_feat
def get_dataset(history_field, middle_field, label_field):
    label_feat = get_label_feature(label_field)
    history_feat = get_all_history_feature(history_field, label_field)
    week_feat = get_week_feature(label_field)
    share_characters = list(set(
        label_feat.columns.tolist()) & 
        set(history_feat.columns.tolist()) & 
        set(week_feat.columns.tolist()))  
    dataset = pd.concat([week_feat, 
        label_feat.drop(
        share_characters, axis=1)], 
        axis=1)
    dataset = pd.concat([dataset, history_feat.drop
        (share_characters, axis=1)],
        axis=1)
    if 'Date' in dataset.columns.tolist(): 
        dataset.drop(['Merchant_id', 'Discount_rate',
                      'Date', 'date_received', 'date'],
        axis=1, 
        inplace=True)
        label = dataset['label'].tolist()
        dataset.drop(['label'],
        axis=1,
        inplace=True)
        dataset['label'] = label
    else:
        dataset.drop(['Merchant_id', 'Discount_rate',
                      'date_received'],
        axis=1,
        inplace=True)
    dataset['User_id'] = dataset['User_id'].map(lambda x:int(x) if x==x else 0)
    dataset['Coupon_id'] = dataset['Coupon_id'].map(lambda x:int(x) if x==x else 0)
    dataset['Date_received'] = dataset['Date_received'].map(lambda x:int(x) if x==x else 0)
    dataset['Distance'] = dataset['Distance'].map(lambda x:int(x) if x==x else 0)
    if 'label' in dataset.columns.tolist():
        dataset['label'] = dataset['label'].map(lambda x:int(x) if x==x else 0)
    dataset.drop_duplicates(
        keep='first', inplace=True)
    dataset.index = range(
    len(dataset))
    return dataset
#模型
def model_xgb(train, test):
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'silent': 1,
              'eta': 0.01,
              'max_depth': 5,   
              'min_child_weight': 1,
              'gamma': 0,
              'lambda': 1,
              'colsample_bylevel': 0.7,   
              'colsample_bytree': 0.7,   
              'subsample': 0.9,
              'scale_pos_weight': 1
            }
    #dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    watchlist = [(dtest, 'train')]
    model = xgb.train(params, dtest, num_boost_round=1000, evals=watchlist)   
    predict = model.predict(dtest)
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    feat_importance = pd.DataFrame(columns=['feature_name', 'importance'])
    feat_importance['feature_name'] = model.get_score().keys()
    feat_importance['importance'] = model.get_score().values()
    feat_importance.sort_values(['importance'], ascending=False, inplace=True)
    return result, feat_importance
def divide (off_train ,  off_test ):
    train_history_field = off_train[off_train['date_received'].isin(
        pd.date_range('2016/3/1', periods=2*30))]  
    train_middle_field = off_train[off_train['date'].isin(
        pd.date_range('2016/5/1', periods=3*5))]  
    train_label_field = off_train[off_train['date_received'].isin(
        pd.date_range('2016/5/16', periods=31))]  
    validate_history_field = off_train[off_train['date_received'].isin(
        pd.date_range('2016/1/16', periods=2*30))]  
    validate_middle_field = off_train[off_train['date'].isin(
        pd.date_range('2016/3/16', periods=3*5))] 
    validate_label_field = off_train[off_train['date_received'].isin(
        pd.date_range('2016/3/31', periods=31))]  
    test_history_field = off_train[off_train['date_received'].isin(
        pd.date_range('2016/4/17', periods=2*30))]  
    test_middle_field = off_train[off_train['date'].isin(
        pd.date_range('2016/6/16', periods=3*5))]  
    test_label_field = off_test.copy()    
    train = get_dataset(train_history_field, train_middle_field, train_label_field)
    validate = get_dataset(validate_history_field, validate_middle_field, validate_label_field)
    test = get_dataset(test_history_field, test_middle_field, test_label_field)
    return train,validate,test
off_train = pd.read_csv('ccf_offline_stage1_train.csv')
off_test = pd.read_csv('ccf_offline_stage1_test_revised.csv')
off_train = Pretreatment_train_data(off_train)
off_test = Pretreatment_test_data(off_test)
off_train = Pretreatment_time(off_train)
off_test = Pretreatment_time(off_test)
off_train = get_label(off_train)
train,validate,test = divide(off_train ,  off_test )
big_train = pd.concat([train, validate], axis=0)
result, feat_importance = model_xgb(big_train, test)
result.to_csv(r'resultgf.csv', index=False, header=None)