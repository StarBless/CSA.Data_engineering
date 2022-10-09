# -*- coding: utf-8 -*-
import pandas as pd

# 读取数据集
df = pd.read_csv('../ccf_offline_stage1_test_revised.csv')
dft = df.copy()

# 提取满减类型的折扣
dft['is_full_reduction'] = dft['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
# 将满减转化为折扣率   (折扣率=(满多少-减多少)/满多少)
dft['discount_rate'] = dft['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
(float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
# 满减最低消费
dft['min_cost_full_reduction'] = dft['Discount_rate'].map(lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0]))
# 时间转换
dft['date_received'] = pd.to_datetime(dft['Date_received'], format='%Y%m%d')
# 统计缺失值
user_id_null = dft['User_id'].isnull().sum(axis=0)
merchant_id_null = dft['Merchant_id'].isnull().sum(axis=0)
coupon_id_null = dft['Coupon_id'].isnull().sum(axis=0)
discount_rate_null = dft['Discount_rate'].isnull().sum(axis=0)
distance_null = dft['Distance'].isnull().sum(axis=0)
date_received_null = dft['Date_received'].isnull().sum(axis=0)
print(f'User_id 缺失值为        {user_id_null}')
print(f'Merchant_id 缺失值为    {merchant_id_null}')
print(f'Coupon_id 缺失值为      {coupon_id_null}')
print(f'Discount_rate 缺失值为  {discount_rate_null}')
print(f'Distance 缺失值为       {distance_null}')
print(f'Date_received 缺失值为  {date_received_null}')
# 将Distance中缺失的值填充为-1
dft['Distance'].fillna(-1, inplace=True)

# 将日期转换周几
dft['week'] = dft['date_received'].map(lambda x: x.isoweekday())
print(dft)
