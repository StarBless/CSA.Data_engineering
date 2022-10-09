import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import warnings

warnings.filterwarnings('ignore')

#  数据提取
off_tr = pd.read_csv(r'ccf_offline_stage1_train.csv',
                     header=0,
                     keep_default_na=False)
off_tr.columns = ['user_id',
                  'merchant_id',
                  'coupon_id',
                  'discount_rate',
                  'distance',
                  'date_received',
                  'date']
off_te = pd.read_csv(r'ccf_offline_stage1_test_revised.csv',
                     header=0,
                     keep_default_na=False)
off_te.columns = ['user_id',
                  'merchant_id',
                  'coupon_id',
                  'discount_rate',
                  'distance',
                  'date_received']

# on_tr=pd.read_csv(r'ccf_online_stage1_train.csv',
#                      header=0,
#                   keep_default_na=False)
# on_tr.columns=['user_id',
#                   'merchant_id',
#                   'action',
#                   'coupon_id','discount_rate',
#                   'date_received',
#                   'date']

# 数据预处理


#  数据集划分

dataset1 = off_tr[(off_tr['date_received'] >= '20160414') &
                  (off_tr['date_received'] <= '20160514')]
data1 = dataset1.copy()
featset1 = off_tr[((off_tr['date'] >= '20160101') &
                   (off_tr['date'] <= '20160413')) |
                  ((off_tr.date == 'null') &
                   (off_tr['date_received'] >= '20160101') &
                   (off_tr['date_received'] <= '20160413'))]
feat1 = featset1.copy()

dataset2 = off_tr[(off_tr['date_received'] >= '20160515') &
                  (off_tr['date_received'] <= '20160615')]
data2 = dataset2.copy()
featset2 = off_tr[((off_tr['date'] >= '20160201') &
                   (off_tr['date'] <= '20160514')) |
                  ((off_tr['date'] == 'null') &
                   (off_tr['date_received'] >= '20160201') &
                   (off_tr['date_received'] <= '20160514'))]
feat2 = featset2.copy()

dataset3 = off_te
data3 = dataset3.copy()
featset3 = off_tr[((off_tr['date'] >= '20160315') &
                   (off_tr['date'] <= '20160630')) |
                  ((off_tr['date'] == 'null') &
                   (off_tr['date_received'] >= '20160315') &
                   (off_tr['date_received'] <= '20160630'))]
feat3 = featset3.copy()

# 去除重复(经过验证发现对结果有提升)
# for j in [data3,
#           feat3,
#           data2,
#           feat2,
#           data1,
#           feat1]:
#     j.drop_duplicates(
#         inplace=True)
#     j.reset_index(drop=True,
#                   inplace=True)


# 提取leakage特征（包含数据处理）
def get_feat_leakage(data):
    # 用户领取的所有优惠券数目
    t0 = data[['user_id']]
    t0.loc['this_month_user_receive_all_coupon_count'] = 1
    t0 = t0.groupby('user_id'). \
        agg('sum'). \
        reset_index()

    # 用户领取的特定优惠券数目
    t1 = data[['user_id',
               'coupon_id']]
    t1.loc['this_month_user_receive_same_coupon_count'] = 1
    t1 = t1.groupby(['user_id', 'coupon_id']). \
        agg('sum'). \
        reset_index()

    # 如果用户领取特定优惠券2次以上，那么提取出第一次和最后一次领取的时间
    t2 = data[['user_id',
               'coupon_id',
               'date_received']]
    t2.date_received = t2.date_received. \
        astype('str')
    t2 = t2.groupby(['user_id',
                     'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t2['receive_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
    t2 = t2[t2.receive_number > 1]
    t2['max_date_received'] = t2.date_received.apply(lambda s: max([int(d) for d in s.split(':')]))
    t2['min_date_received'] = t2.date_received.apply(lambda s: min([int(d) for d in s.split(':')]))
    t2 = t2[['user_id',
             'coupon_id',
             'max_date_received',
             'min_date_received']]

    # 用户领取特定优惠券的时间，是不是最后一次&第一次
    t3 = data[['user_id',
               'coupon_id',
               'date_received']]
    t3 = pd.merge(t3,
                  t2,
                  on=['user_id', 'coupon_id'],
                  how='left')
    t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received.astype('int')
    t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received.astype('int') - t3.min_date_received

    def is_first_or_last_one(x):
        if x == 0:
            return 1
        elif x > 0:
            return 0
        else:
            return -1

    t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(
        is_first_or_last_one)
    t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(
        is_first_or_last_one)
    t3 = t3[['user_id', 'coupon_id', 'date_received', 'this_month_user_receive_same_coupon_lastone',
             'this_month_user_receive_same_coupon_firstone']]

    # 用户在领取优惠券的当天，共领取了多少张优惠券
    t4 = data[['user_id', 'date_received']]
    t4['this_day_user_receive_all_coupon_count'] = 1
    t4 = t4.groupby(['user_id', 'date_received']). \
        agg('sum'). \
        reset_index()

    # 用户在领取特定优惠券的当天，共领取了多少张特定的优惠券
    t5 = data[['user_id',
               'coupon_id',
               'date_received']]
    t5['this_day_user_receive_same_coupon_count'] = 1
    t5 = t5.groupby(['user_id',
                     'coupon_id',
                     'date_received']). \
        agg('sum'). \
        reset_index()

    # 对用户领取特定优惠券的日期进行组合
    t6 = data[['user_id',
               'coupon_id',
               'date_received']]
    t6.date_received = t6.date_received.astype('str')
    t6 = t6.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t6.rename(columns={'date_received': 'dates'},
              inplace=True)

    def get_gap_before(temp):
        date_received, dates = temp.split('-')
        dates = dates.split(':')
        gaps = []
        for d in dates:
            this_gap = (date(int(date_received[0:4]),
                             int(date_received[4:6]),
                             int(date_received[6:8])) -
                        date(int(d[0:4]),
                             int(d[4:6]),
                             int(d[6:8]))).days
            if this_gap > 0:
                gaps.append(this_gap)
        if len(gaps) == 0:
            return -1
        else:
            return min(gaps)

    def get_gap_after(temp):
        date_received, dates = temp.split('-')
        dates = dates.split(':')
        gaps = []
        for d in dates:
            this_gap = (date(int(d[0:4]), int(d[4:6]), int(d[6:8])) -
                        date(int(date_received[0:4]), int(date_received[4:6]), int(date_received[6:8]))).days
            if this_gap > 0:
                gaps.append(this_gap)
        if len(gaps) == 0:
            return -1
        else:
            return min(gaps)

    # 用户领取特定优惠券的当天，与上一次/下一次领取此优惠券的相隔天数
    t7 = data[['user_id',
               'coupon_id',
               'date_received']]
    t7 = pd.merge(t7,
                  t6,
                  on=['user_id', 'coupon_id'],
                  how='left')
    t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
    t7['day_gap_before'] = t7.date_received_date.apply(get_gap_before)
    t7['day_gap_after'] = t7.date_received_date.apply(get_gap_after)
    t7 = t7[['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after']]

    # 上述提取的特征进行合并
    feat_leakage = pd.merge(t1,
                            t0,
                            on='user_id')
    feat_leakage = pd.merge(feat_leakage,
                            t3,
                            on=['user_id', 'coupon_id'])
    feat_leakage = pd.merge(feat_leakage,
                            t4,
                            on=['user_id', 'date_received'])
    feat_leakage = pd.merge(feat_leakage,
                            t5,
                            on=['user_id', 'coupon_id', 'date_received'])
    feat_leakage = pd.merge(feat_leakage,
                            t7,
                            on=['user_id', 'coupon_id', 'date_received'])

    # 去重；重置索引
    feat_leakage.drop_duplicates(inplace=True)
    feat_leakage.reset_index(drop=True,
                             inplace=True)
    return feat_leakage


# 对数据集进行leakage特征的提取
feat_leakage1 = get_feat_leakage(data1)
feat_leakage1.to_csv('feat_leakage1.csv',index=None)
feat_leakage2 = get_feat_leakage(data2)
feat_leakage2.to_csv('feat_leakage2.csv',index=None)
feat_leakage3 = get_feat_leakage(data3)
feat_leakage3.to_csv('feat_leakage3.csv',index=None)


# 提取优惠券相关特征
def get_coupon_feature(data):
    # 计算折扣率函数
    def calc_discount_rate(s):
        s = str(s)
        s = s.split(':')
        if len(s) == 1:
            return float(s[0])
        else:
            return 1.0 - float(s[1]) / float(s[0])

    # 提取满减优惠券中，‘满‘对应的金额
    def get_discount_man(s):
        s = str(s)
        s = s.split(':')
        if len(s) == 1:
            return 'null'
        else:
            return int(s[0])

    # 提取满减优惠券中，‘减‘对应的金额
    def get_discount_jian(s):
        s = str(s)
        s = s.split(':')
        if len(s) == 1:
            return 'null'
        else:
            return int(s[1])

    # 是否满减
    def is_man_jian(s):
        s = str(s)
        s = s.split(':')
        if len(s) == 1:
            return 0
        else:
            return 1.0

    # 周几领取的优惠券
    data['day_of_week'] = data.date_received.astype('str').apply(
        lambda x: date(int(x[0:4]),
                       int(x[4:6]),
                       int(x[6:8])).weekday() + 1)

    # 每月的第几天领取的优惠券
    data['day_of_month'] = data.date_received. \
        astype('str'). \
        apply(lambda x: int(x[6:8]))

    # 领取优惠券的时间与当月初距离多少天
    data['days_distance'] = data.date_received.astype('str').apply(
        lambda x: (date(int(x[0:4]), int(x[4:6]), int(x[6:8])) - date(2016, 6, 30)).days)

    # 满减优惠券中，满对应的金额
    data['discount_man'] = data.discount_rate. \
        apply(get_discount_man)

    # 满减优惠券中，减对应的金额
    data['discount_jian'] = data.discount_rate. \
        apply(get_discount_jian)

    # 优惠券是不是满减卷
    data['is_man_jian'] = data.discount_rate. \
        apply(is_man_jian)

    # 优惠券的折扣率（满减卷进行折扣率转换）
    data['discount_rate'] = data.discount_rate. \
        apply(calc_discount_rate)

    # 特定优惠券的总数量
    d = data[['coupon_id']]
    d['coupon_count'] = 1
    d = d.groupby('coupon_id').agg('sum').reset_index()
    data = pd.merge(data, d, on='coupon_id', how='left')

    return data


# 对数据集进行coupon_feature的提取
coupon_feature1 = get_coupon_feature(data1)
coupon_feature1.to_csv('coupon_feature1.csv',index=None)
coupon_feature2 = get_coupon_feature(data2)
coupon_feature2.to_csv('coupon_feature2.csv',index=None)
coupon_feature3 = get_coupon_feature(data3)
coupon_feature3.to_csv('coupon_feature3.csv',index=None)

# 提取商家相关特征
def get_merchant_feature(feat):
    merchant = feat[['merchant_id',
                     'coupon_id',
                     'distance',
                     'date_received',
                     'date']]

    # 提取不重复的商户集合
    t = merchant[['merchant_id']]
    t.drop_duplicates(inplace=True)

    # 商户的总销售次数
    t1 = merchant[merchant.date != 'null'][['merchant_id']]
    t1['total_sales'] = 1
    t1 = t1.groupby('merchant_id'). \
        agg('sum'). \
        reset_index()

    # 商户被核销优惠券的销售次数
    t2 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null')][['merchant_id']]
    t2['sales_use_coupon'] = 1
    t2 = t2.groupby('merchant_id'). \
        agg('sum'). \
        reset_index()

    # 商户发行优惠券的总数
    t3 = merchant[merchant.coupon_id != 'null'][['merchant_id']]
    t3['total_coupon'] = 1
    t3 = t3.groupby('merchant_id'). \
        agg('sum'). \
        reset_index()

    # 商户被核销优惠券的用户-商户距离，转化为int数值类型
    t4 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null')][['merchant_id', 'distance']]
    t4.replace('null',
               -1,
               inplace=True)
    t4.distance = t4.distance.astype('int')
    t4.replace(-1,
               np.nan,
               inplace=True)

    # 商户被核销优惠券的最小用户-商户距离
    t5 = t4.groupby('merchant_id'). \
        agg('min'). \
        reset_index()
    t5.rename(columns={'distance': 'merchant_min_distance'},
              inplace=True)

    # 商户被核销优惠券的最大用户-商户距离
    t6 = t4.groupby('merchant_id'). \
        agg('max'). \
        reset_index()
    t6.rename(columns={'distance': 'merchant_max_distance'},
              inplace=True)

    # 商户被核销优惠券的平均用户-商户距离
    t7 = t4.groupby('merchant_id'). \
        agg('mean'). \
        reset_index()
    t7.rename(columns={'distance': 'merchant_mean_distance'},
              inplace=True)

    # 商户被核销优惠券的用户-商户距离的中位数
    t8 = t4.groupby('merchant_id').agg('median').reset_index()
    t8.rename(columns={'distance': 'merchant_median_distance'},
              inplace=True)

    # 合并上述特征
    merchant_feature = pd.merge(t,
                                t1,
                                on='merchant_id',
                                how='left')
    merchant_feature = pd.merge(merchant_feature,
                                t2,
                                on='merchant_id',
                                how='left')
    merchant_feature = pd.merge(merchant_feature,
                                t3,
                                on='merchant_id',
                                how='left')
    merchant_feature = pd.merge(merchant_feature,
                                t5,
                                on='merchant_id',
                                how='left')
    merchant_feature = pd.merge(merchant_feature,
                                t6,
                                on='merchant_id',
                                how='left')
    merchant_feature = pd.merge(merchant_feature,
                                t7,
                                on='merchant_id',
                                how='left')
    merchant_feature = pd.merge(merchant_feature,
                                t8,
                                on='merchant_id',
                                how='left')

    # 商户被核销优惠券的销售次数，如果为空，填充为0
    merchant_feature.sales_use_coupon = merchant_feature.sales_use_coupon.replace(np.nan, 0)
    merchant_feature.sales_use_coupon = merchant_feature.sales_use_coupon.replace('null', 0)

    # 商户发行优惠券的转化率
    merchant_feature['merchant_coupon_transfer_rate'] = merchant_feature.sales_use_coupon.astype(
        'float') / merchant_feature.total_coupon

    # 商户被核销优惠券的销售次数占比
    merchant_feature['coupon_rate'] = merchant_feature.sales_use_coupon.astype('float') / merchant_feature.total_sales
    merchant_feature.total_coupon = merchant_feature.total_coupon.replace(np.nan, 0)
    merchant_feature.total_coupon = merchant_feature.total_coupon.replace('null', 0)

    return merchant_feature


# 对特征数据集进行merchant_feature的提取
merchant_feature1 = get_merchant_feature(feat1)
coupon_feature3.to_csv('coupon_feature1.csv',index=None)
merchant_feature2 = get_merchant_feature(feat2)
coupon_feature3.to_csv('coupon_feature2.csv',index=None)
merchant_feature3 = get_merchant_feature(feat3)
coupon_feature3.to_csv('coupon_feature3.csv',index=None)


# 用户相关特征
def get_user_feature(feat):
    # 用户核销优惠券与领取优惠券日期间隔
    def get_user_date_gap(s):
        s = s.split(':')
        return (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) -
                date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days

    user = feat[['user_id',
                 'merchant_id',
                 'coupon_id',
                 'discount_rate',
                 'distance',
                 'date_received',
                 'date']]

    # 提取不重复的所有用户集合
    t = user[['user_id']]
    t.drop_duplicates(
        inplace=True)

    # 用户在特定商户的消费次数
    t1 = user[user.date != 'null'][['user_id', 'merchant_id']]
    t1.drop_duplicates(
        inplace=True)
    t1.merchant_id = 1
    t1 = t1.groupby('user_id'). \
        agg('sum'). \
        reset_index()
    t1.rename(columns={'merchant_id': 'count_merchant'},
              inplace=True)

    # 提取用户核销优惠券的用户-商户距离
    t2 = user[(user.date != 'null') & (user.coupon_id != 'null')][['user_id', 'distance']]
    t2.replace('null',
               -1,
               inplace=True)
    t2.distance = t2.distance.astype('int')
    t2.replace(-1,
               np.nan,
               inplace=True)

    # 用户核销优惠券中的最小用户-商户距离
    t3 = t2.groupby('user_id'). \
        agg('min'). \
        reset_index()
    t3.rename(columns={'distance': 'user_min_distance'},
              inplace=True)

    # 用户核销优惠券中的最大用户-商户距离
    t4 = t2.groupby('user_id'). \
        agg('max'). \
        reset_index()
    t4.rename(columns={'distance': 'user_max_distance'},
              inplace=True)

    # 用户核销优惠券中的平均用户-商户距离
    t5 = t2.groupby('user_id'). \
        agg('mean'). \
        reset_index()
    t5.rename(columns={'distance': 'user_mean_distance'},
              inplace=True)

    # 用户核销优惠券的用户-商户距离的中位数
    t6 = t2.groupby('user_id'). \
        agg('median'). \
        reset_index()
    t6.rename(columns={'distance': 'user_median_distance'},
              inplace=True)

    # 用户核销优惠券的总次数
    t7 = user[(user.date != 'null') & (user.coupon_id != 'null')][['user_id']]
    t7['buy_use_coupon'] = 1
    t7 = t7.groupby('user_id'). \
        agg('sum'). \
        reset_index()

    # 用户购买的总次数
    t8 = user[user.date != 'null'][['user_id']]
    t8['buy_total'] = 1
    t8 = t8.groupby('user_id'). \
        agg('sum'). \
        reset_index()

    # 用户领取优惠券的总次数
    t9 = user[user.coupon_id != 'null'][['user_id']]
    t9['coupon_received'] = 1
    t9 = t9.groupby('user_id'). \
        agg('sum'). \
        reset_index()

    # 用户核销优惠券与领取优惠券的日期间隔
    t10 = user[(user.date_received != 'null') & (user.date != 'null')][['user_id', 'date_received', 'date']]
    t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
    t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_gap)
    t10 = t10[['user_id', 'user_date_datereceived_gap']]

    # 用户核销优惠券与领取优惠券的日期间隔的平均值
    t11 = t10.groupby('user_id'). \
        agg('mean'). \
        reset_index()
    t11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'},
               inplace=True)

    # 用户核销优惠券与领取优惠券的日期间隔的最小值
    t12 = t10.groupby('user_id'). \
        agg('min'). \
        reset_index()
    t12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'},
               inplace=True)

    # 用户核销优惠券与领取优惠券的日期间隔的最大值
    t13 = t10.groupby('user_id'). \
        agg('max'). \
        reset_index()
    t13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'},
               inplace=True)

    # 合并上述特征
    user_feature = pd.merge(t,
                            t1,
                            on='user_id',
                            how='left')
    user_feature = pd.merge(user_feature,
                            t3,
                            on='user_id',
                            how='left')
    user_feature = pd.merge(user_feature,
                            t4,
                            on='user_id',
                            how='left')
    user_feature = pd.merge(user_feature,
                            t5,
                            on='user_id',
                            how='left')
    user_feature = pd.merge(user_feature,
                            t6,
                            on='user_id',
                            how='left')
    user_feature = pd.merge(user_feature,
                            t7,
                            on='user_id',
                            how='left')
    user_feature = pd.merge(user_feature,
                            t8,
                            on='user_id',
                            how='left')
    user_feature = pd.merge(user_feature,
                            t9,
                            on='user_id',
                            how='left')
    user_feature = pd.merge(user_feature,
                            t10,
                            on='user_id',
                            how='left')
    user_feature = pd.merge(user_feature,
                            t11,
                            on='user_id',
                            how='left')
    user_feature = pd.merge(user_feature,
                            t12,
                            on='user_id',
                            how='left')
    user_feature = pd.merge(user_feature,
                            t13,
                            on='user_id',
                            how='left')

    # 特征缺失值填充
    user_feature.count_merchant = user_feature.count_merchant.replace(np.nan, 0)
    user_feature.count_merchant = user_feature.count_merchant.replace('null', 0)
    user_feature.buy_use_coupon = user_feature.buy_use_coupon.replace(np.nan, 0)
    user_feature.buy_use_coupon = user_feature.buy_use_coupon.replace('null', 0)

    # 用户核销优惠券消费次数占用户总消费次数的比例
    user_feature['buy_use_coupon_rate'] = user_feature.buy_use_coupon.astype('float') / user_feature.buy_total.astype(
        'float')

    # 用户核销优惠券消费次数占用户领取优惠券次数的比例
    user_feature['user_coupon_transfer_rate'] = user_feature.buy_use_coupon.astype(
        'float') / user_feature.coupon_received.astype('float')

    # 特征缺失值填充
    user_feature.buy_total = user_feature.buy_total.replace(np.nan, 0)
    user_feature.buy_total = user_feature.buy_total.replace('null', 0)
    user_feature.coupon_received = user_feature.coupon_received.replace(np.nan, 0)
    user_feature.coupon_received = user_feature.coupon_received.replace('null', 0)

    return user_feature


# 对特征数据集进行user_related_feature的提取
user_feature1 = get_user_feature(feat1)
user_feature1.to_csv('user_feature1.csv',index=None)
user_feature2 = get_user_feature(feat2)
user_feature2.to_csv('user_feature2.csv',index=None)
user_feature3 = get_user_feature(feat3)
user_feature3.to_csv('user_feature3.csv',index=None)


# 用户-商户交叉特征
def get_user_merchant_feature(feat):
    # 提取用户-商户交叉集合
    all_user_merchant = feat[['user_id',
                              'merchant_id']]
    all_user_merchant.drop_duplicates(
        inplace=True)

    # 用户在特定商户下的消费次数
    t = feat[['user_id',
              'merchant_id',
              'date']]
    t = t[t.date != 'null'][['user_id', 'merchant_id']]
    t['user_merchant_buy_total'] = 1
    t = t.groupby(['user_id', 'merchant_id']). \
        agg('sum'). \
        reset_index()
    t.drop_duplicates(
        inplace=True)

    # 用户在特定商户处领取优惠券次数
    t1 = feat[['user_id',
               'merchant_id',
               'coupon_id']]
    t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
    t1['user_merchant_received'] = 1
    t1 = t1.groupby(['user_id', 'merchant_id']). \
        agg('sum'). \
        reset_index()
    t1.drop_duplicates(
        inplace=True)

    # 用户在特定商户处核销优惠券的次数
    t2 = feat[['user_id',
               'merchant_id',
               'date',
               'date_received']]
    t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][['user_id', 'merchant_id']]
    t2['user_merchant_buy_use_coupon'] = 1
    t2 = t2.groupby(['user_id', 'merchant_id']). \
        agg('sum'). \
        reset_index()
    t2.drop_duplicates(
        inplace=True)

    # 用户在特定商户处发生行为的总次数
    t3 = feat[['user_id',
               'merchant_id']]
    t3['user_merchant_any'] = 1
    t3 = t3.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    t3.drop_duplicates(
        inplace=True)

    # 用户在特定商户处未领取优惠券产生的消费次数
    t4 = feat[['user_id',
               'merchant_id',
               'date',
               'coupon_id']]
    t4 = t4[(t4.date != 'null') & (t4.coupon_id == 'null')][['user_id', 'merchant_id']]
    t4['user_merchant_buy_common'] = 1
    t4 = t4.groupby(['user_id', 'merchant_id']). \
        agg('sum'). \
        reset_index()
    t4.drop_duplicates(
        inplace=True)

    # 合并上述特征
    user_merchant = pd.merge(all_user_merchant,
                             t,
                             on=['user_id', 'merchant_id'],
                             how='left')
    user_merchant = pd.merge(user_merchant,
                             t1,
                             on=['user_id', 'merchant_id'],
                             how='left')
    user_merchant = pd.merge(user_merchant,
                             t2,
                             on=['user_id', 'merchant_id'],
                             how='left')
    user_merchant = pd.merge(user_merchant,
                             t3,
                             on=['user_id', 'merchant_id'],
                             how='left')
    user_merchant = pd.merge(user_merchant,
                             t4,
                             on=['user_id', 'merchant_id'],
                             how='left')

    # 相关特征缺失值填充
    user_merchant.user_merchant_buy_use_coupon = user_merchant.user_merchant_buy_use_coupon.replace(np.nan, 0)
    user_merchant.user_merchant_buy_use_coupon = user_merchant.user_merchant_buy_use_coupon.replace('null', 0)
    user_merchant.user_merchant_buy_common = user_merchant.user_merchant_buy_common.replace(np.nan, 0)
    user_merchant.user_merchant_buy_common = user_merchant.user_merchant_buy_common.replace('null', 0)

    # 用户在特定商户处核销优惠券占领取优惠券数量的比例
    user_merchant['user_merchant_coupon_transfer_rate'] = user_merchant.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant.user_merchant_received.astype('float')

    # 用户在特定商户处核销优惠券占购买次数的比例
    user_merchant['user_merchant_coupon_buy_rate'] = user_merchant.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant.user_merchant_buy_total.astype('float')

    # 用户在特定商户处购买次数占发生行为次数的比例
    user_merchant['user_merchant_rate'] = user_merchant.user_merchant_buy_total.astype(
        'float') / user_merchant.user_merchant_any.astype('float')

    # 用户在特定商户下未用优惠券购买占购买次数的占比
    user_merchant['user_merchant_common_buy_rate'] = user_merchant.user_merchant_buy_common.astype(
        'float') / user_merchant.user_merchant_buy_total.astype('float')

    return user_merchant


# 对特征数据集进行user_merchant_related_feature的提取
user_merchant1 = get_user_merchant_feature(feat1)
user_merchant1.to_csv('user_merchant1.csv',index=None)
user_merchant2 = get_user_merchant_feature(feat2)
user_merchant1.to_csv('user_merchant1.csv',index=None)
user_merchant3 = get_user_merchant_feature(feat3)
user_merchant1.to_csv('user_merchant1.csv',index=None)


# 训练数据及测试数据集的构建

# 提取题目要求的标签：15天内核销
def get_label(s):
    s = s.split(':')
    if s[0] == 'null':
        return 0
    elif (date(int(s[0][0:4]),
               int(s[0][4:6]),
               int(s[0][6:8])) -
          date(int(s[1][0:4]),
               int(s[1][4:6]),
               int(s[1][6:8]))).days <= 15:
        return 1
    else:
        return -1


# 合并特征
data3 = pd.merge(coupon_feature3,
                 merchant_feature3,
                 on='merchant_id',
                 how='left')
data3 = pd.merge(data3,
                 user_feature3,
                 on='user_id',
                 how='left')
data3 = pd.merge(data3,
                 user_merchant3,
                 on=['user_id', 'merchant_id'],
                 how='left')
data3 = pd.merge(data3,
                 feat_leakage3,
                 on=['user_id', 'coupon_id', 'date_received'],
                 how='left')
data3.drop_duplicates(
    inplace=True)

# 特征缺失值填充
data3.user_merchant_buy_total = data3.user_merchant_buy_total.replace(np.nan, 0)
data3.user_merchant_buy_total = data3.user_merchant_buy_total.replace('null', 0)
data3.user_merchant_any = data3.user_merchant_any.replace(np.nan, 0)
data3.user_merchant_any = data3.user_merchant_any.replace('null', 0)
data3.user_merchant_received = data3.user_merchant_received.replace(np.nan, 0)
data3.user_merchant_received = data3.user_merchant_received.replace('null', 0)

# 用户领取优惠券日期是否在周末
data3['is_weekend'] = data3.day_of_week.apply(
    lambda x: 1 if x in (6, 7) else 0)

# 对优惠券领取日期进行ont-hot编码
weekday_dummies = pd.get_dummies(data3.day_of_week)
weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
data3 = pd.concat([data3, weekday_dummies],
                  axis=1)

data3.drop(['merchant_id',
            'day_of_week',
            'coupon_count'],
           axis=1,
           inplace=True)

data3 = data3.replace('null',
                      np.nan)

data2 = pd.merge(coupon_feature2,
                 merchant_feature2,
                 on='merchant_id',
                 how='left')
data2 = pd.merge(data2,
                 user_feature2,
                 on='user_id',
                 how='left')
data2 = pd.merge(data2,
                 user_merchant2,
                 on=['user_id', 'merchant_id'],
                 how='left')
data2 = pd.merge(data2,
                 feat_leakage2,
                 on=['user_id', 'coupon_id', 'date_received'],
                 how='left')
data2.drop_duplicates(
    inplace=True)

# 处理基本与上述data3一致，这里特殊有一部添加需要的label
data2.user_merchant_buy_total = data2.user_merchant_buy_total.replace(np.nan,
                                                                      0)
data2.user_merchant_any = data2.user_merchant_any.replace(np.nan,
                                                          0)
data2.user_merchant_received = data2.user_merchant_received.replace(np.nan,
                                                                    0)

data2['is_weekend'] = data2.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
weekday_dummies = pd.get_dummies(data2.day_of_week)
weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
data2 = pd.concat([data2, weekday_dummies],
                  axis=1)

data2['label'] = data2.date.astype('str') + ':' + data2.date_received.astype('str')
data2.label = data2.label.apply(get_label)

data2.drop(['merchant_id',
            'day_of_week',
            'date',
            'date_received',
            'coupon_id',
            'coupon_count'],
           axis=1,
           inplace=True)

data2 = data2.replace('null',
                      np.nan)
data2 = data2.replace('nan',
                      np.nan)

data1 = pd.merge(coupon_feature1,
                 merchant_feature1,
                 on='merchant_id',
                 how='left')
data1 = pd.merge(data1,
                 user_feature1,
                 on='user_id',
                 how='left')
data1 = pd.merge(data1,
                 user_merchant1,
                 on=['user_id', 'merchant_id'],
                 how='left')
data1 = pd.merge(data1,
                 feat_leakage1,
                 on=['user_id', 'coupon_id', 'date_received'],
                 how='left')
data1.drop_duplicates(
    inplace=True)

data1.user_merchant_buy_total = data1.user_merchant_buy_total.replace(np.nan,
                                                                      0)
data1.user_merchant_any = data1.user_merchant_any.replace(np.nan,
                                                          0)
data1.user_merchant_received = data1.user_merchant_received.replace(np.nan,
                                                                    0)

data1['is_weekend'] = data1. \
    day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
weekday_dummies = pd.get_dummies(data1.day_of_week)
weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
data1 = pd.concat([data1,
                   weekday_dummies],
                  axis=1)

data1['label'] = data1.date.astype('str') + ':' + data1.date_received.astype('str')
data1.label = data1.label. \
    apply(get_label)

data1.drop(['merchant_id',
            'day_of_week',
            'date',
            'date_received',
            'coupon_id',
            'coupon_count'],
           axis=1,
           inplace=True)

data1 = data1.replace('null',
                      np.nan)
data1 = data1.replace('nan',
                      np.nan)

data1 = data1.replace('null', 0)
data1 = data1.replace(np.nan, 0)
data2 = data2.replace('null', 0)
data2 = data2.replace(np.nan, 0)
data3 = data3.replace('null', 0)
data3 = data3.replace(np.nan, 0)

data1 = data1.astype(float)
data2 = data2.astype(float)
data3 = data3.astype(float)

#去重

data1.drop_duplicates(
    inplace=True)
data2.drop_duplicates(
    inplace=True)
data3.drop_duplicates(
    inplace=True)

data1.to_csv('data1.csv',index=None)
data2.to_csv('data2.csv',index=None)
data3.to_csv('data3.csv',index=None)


# 将训练集一和训练集二合并，作为调参后的总训练数据集
data12 = pd.concat([data1, data2],
                   axis=0)

# xgb模型

data1_y = data1.label
data1_x = data1.drop(['user_id',
                      'label',
                      'day_gap_before',
                      'day_gap_after'],
                     axis=1)
data2_y = data2.label
data2_x = data2.drop(['user_id',
                      'label',
                      'day_gap_before',
                      'day_gap_after'],
                     axis=1)
data12_y = data12.label
data12_x = data12.drop(['user_id',
                        'label',
                        'day_gap_before',
                        'day_gap_after'],
                       axis=1)
data3_preds = data3[['user_id',
                     'coupon_id',
                     'date_received']]
data3_preds = data3_preds.astype(int)

data3_x = data3.drop(['user_id',
                      'coupon_id',
                      'date_received',
                      'day_gap_before',
                      'day_gap_after'],
                     axis=1)

# 转换为xgb需要的数据类型
data1 = xgb.DMatrix(data1_x, label=data1_y)
data2 = xgb.DMatrix(data2_x, label=data2_y)
data12 = xgb.DMatrix(data12_x, label=data12_y)
data3 = xgb.DMatrix(data3_x)

# xgb参数
params = {'booster': 'gbtree',
          'objective': 'rank:pairwise',
          'eval_metric': 'auc',
          'gamma': 0.1,
          'min_child_weight': 1.1,
          'max_depth': 5,
          'lambda': 10,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'eta': 0.01,
          'tree_method': 'exact',
          'seed': 0,
          'nthread': 12}

# 训练模型
watchlist = [(data12,
              'train')]
model = xgb.train(params,
                  data12,
                  num_boost_round=5000,
                  evals=watchlist)

# 预测
data3_preds['label'] = model.predict(data3)
data3_preds.label = MinMaxScaler().fit_transform(np.array(data3_preds.label).reshape(-1, 1))
data3_preds.sort_values(by=['coupon_id',
                            'label'],
                        inplace=True)

data3_preds.describe()

data3_preds.to_csv(r'xgb_preds.csv',
                   index=None,
                   header=None)

# save feature score
# 这一步可以输出各特征的重要性，可以作为特征筛选的一种方式
feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(),
                       key=lambda x: x[1],
                       reverse=True)

fs = []
for (key, value) in feature_score:
    fs.append('{0},{1}\n'.format(key, value))

with open(r'xgb_feature_score.csv', 'w') as f:
    f.writelines('feature,score\n')
    f.writelines(fs)
