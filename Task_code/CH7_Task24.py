import os
import pandas as pd

# 源数据
off_train = pd.read_csv(r'../input_files/ccf_offline_stage1_train.csv')
off_train['date_received'] = pd.to_datetime(off_train['Date_received'], format='%Y%m%d')
label_field = off_train[off_train['date_received'].isin(pd.date_range('2016/5/16', periods=31))]
data = label_field.copy()
# 将Coupon_id列中float类型的元素转换为int类型,列中存在np.nan会让整列元素变为float
data['Coupon_id'] = data['Coupon_id'].map(int)
# 将Date_received列中float类型的元素转换为int类型,理由同上
data['Date_received'] = data['Date_received'].map(int)
# 方便特征提取
data['cnt'] = 1
# 返回的特征数据集
feature = data.copy()

path = './tmp/CH7_Task24_output'
if not os.path.exists(path):
            os.makedirs(path)
feature.to_csv(path+'/feature.csv', index=False)