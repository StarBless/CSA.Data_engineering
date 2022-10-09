import os
# 导入pandas包并重命名为pd
import pandas as pd
# 读入数据集
offline = pd.read_csv(r'../input_files/ccf_offline_stage1_train.csv')
# Discount_rate是否为满减
offline['is_manjian'] = offline['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
# 满减全部转为折扣率
offline['discount_rate'] = offline['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else 
(float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
# 满减最低消费
offline['min_cost_of_manjian'] = offline['Discount_rate'].map(lambda x: 1 if ':' not in str(x) else int(str(x).split(':')[0]))

# 时间转换
offline['date_received'] = pd.to_datetime(offline['Date_received'], format='%Y%m%d')
offline['date'] = pd.to_datetime(offline['Date'], format='%Y%m%d')


path = './tmp/CH5_Task20_output'
if not os.path.exists(path):
        os.makedirs(path)
offline.to_csv(path+'/CH5_Task20_output.csv', index=False)
