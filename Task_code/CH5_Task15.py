# 导入pandas包并重命名为pd
import pandas as pd

# 读取数据集
off_train = pd.read_csv(r'ccf_offline_stage1_train.csv')

# 共有多少条记录
record_count = off_train.shape[0]
# 共有多少条优惠券的领取记录
received_count = off_train['Date_received'].count()
# 共有多少种不同的优惠券
coupon_count = len(off_train['Coupon_id'].value_counts())
# 共有多少个用户
user_count = len(off_train['User_id'].value_counts())
# 共有多少个商家
merchant_count = len(off_train['Merchant_id'].value_counts())
# 最早领券时间
min_received = str(int(off_train['Date_received'].min()))
# 最晚领券时间
max_received = str(int(off_train['Date_received'].max()))
# 最早消费时间
min_date = str(int(off_train['Date'].min()))
# 最晚消费时间
max_date = str(int(off_train['Date'].max()))

print('record_count', record_count)
print('received_count', received_count)
print('coupon_count', coupon_count)
print('user_count', user_count)
print('merchant_count', merchant_count)
print('min_received', min_received)
print('max_received', max_received)
print('min_date', min_date)
print('max_date',max_date)