import os
# 导入pandas包并重命名为pd
import pandas as pd
# 读入数据集
offline = pd.read_csv(r'../input_files/ccf_offline_stage1_train.csv')
# 查看缺失值
print(offline.isnull().any())
# 缺失值比例查看
print(offline.isnull().sum()/len(offline))

# 距离空值填充为-1
offline['Distance'].fillna(-1, inplace=True)
# 判断是否是空距离
offline['null_distance'] = offline['Distance'].map(lambda x: 1 if x == -1 else 0)

path = './tmp/CH5_Task21_output'
if not os.path.exists(path):
        os.makedirs(path)
offline.to_csv(path+'/CH5_Task21_output.csv', index=False)
