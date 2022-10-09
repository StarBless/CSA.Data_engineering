import os
import pandas as pd

# 源数据
off_train = pd.read_csv(r'../input_files/ccf_offline_stage1_train.csv')
off_test = pd.read_csv(r'../input_files/ccf_offline_stage1_test_revised.csv')

# 折扣率处理
off_train['is_manjian'] = off_train['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)  # Discount_rate是否为满减
off_train['discount_rate'] = off_train['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
(float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))  # 满减全部转换为折扣率
off_train['min_cost_of_manjian'] = off_train['Discount_rate'].map(
    lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0]))  # 满减的最低消费

# 折扣率处理
off_test['is_manjian'] = off_test['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)  # Discount_rate是否为满减
off_test['discount_rate'] = off_test['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
(float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))  # 满减全部转换为折扣率
off_test['min_cost_of_manjian'] = off_test['Discount_rate'].map(
    lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0]))  # 满减的最低消费

# 保存
path = './tmp/CH8_Task27_output'
if not os.path.exists(path):
        os.makedirs(path)
off_train.to_csv(path+'/off_train.csv', index=False)
off_test.to_csv(path+'/off_test.csv', index=False)

