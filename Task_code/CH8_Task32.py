import os
import pandas as pd

# 源数据
data = pd.read_csv(r'../input_files/ccf_offline_stage1_train.csv')
data['Distance'].fillna(-1, downcast='infer', inplace=True)

res = pd.get_dummies(data['Distance'])

path = './tmp/CH8_Task32_output'
if not os.path.exists(path):
        os.makedirs(path)
res.to_csv(path+'/lisan.csv', index=False)