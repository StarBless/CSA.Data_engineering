import pandas as pd
#读取数据集
off_train = pd.read_csv('../ccf_offline_stage1_test_revised.csv')

#统计缺失值
User_id_null = off_train['User_id'].isnull().sum(axis=0)
Merchant_id_null = off_train['Merchant_id'].isnull().sum(axis=0)
Coupon_id_null = off_train['Coupon_id'].isnull().sum(axis=0)
Discount_rate_null = off_train['Discount_rate'].isnull().sum(axis=0)
Distance_null = off_train['Distance'].isnull().sum(axis=0)
Date_received_null = off_train['Date_received'].isnull().sum(axis=0)

print(f'User_id 缺失值为        {User_id_null}')
print(f'Merchant_id 缺失值为    {Merchant_id_null}')
print(f'Coupon_id 缺失值为      {Coupon_id_null}')
print(f'Discount_rate 缺失值为  {Discount_rate_null}')
print(f'Distance 缺失值为       {Distance_null}')
print(f'Date_received 缺失值为  {Date_received_null}')