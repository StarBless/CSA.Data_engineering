import os
import pandas as pd

# 源数据
off_train = pd.read_csv(r'../input_files/ccf_offline_stage1_train.csv')
off_test = pd.read_csv(r'../input_files/ccf_offline_stage1_test_revised.csv')


# 距离处理
off_train['Distance'].fillna(-1, inplace=True)  # 空距离填充为-1
off_train['null_distance'] = off_train['Distance'].map(lambda x: 1 if x == -1 else 0)
# 时间处理
off_train['date_received'] = pd.to_datetime(off_train['Date_received'], format='%Y%m%d')
off_train['date'] = pd.to_datetime(off_train['Date'], format='%Y%m%d')

# 距离处理
off_test['Distance'].fillna(-1, inplace=True)  # 空距离填充为-1
off_test['null_distance'] = off_test['Distance'].map(lambda x: 1 if x == -1 else 0)
# 时间处理
off_test['date_received'] = pd.to_datetime(off_test['Date_received'], format='%Y%m%d')


# 保存
path = './tmp/CH8_Task28_output'
if not os.path.exists(path):
        os.makedirs(path)
off_train.to_csv(path+'/off_train.csv', index=False)
off_test.to_csv(path+'/off_test.csv', index=False)

