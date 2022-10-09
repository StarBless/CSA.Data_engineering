import os
import pandas as pd

# 源数据
data = pd.read_csv(r'ccf_offline_stage1_train.csv')

# 复制data数据集为其做一下预处理方便绘图
offline = data.copy()

# 将Distance的空值填充为-1
offline['Distance'].fillna(-1, inplace=True)
# 将领券时间转为时间类型
offline['date_received'] = pd.to_datetime(offline['Date_received'], format='%Y%m%d')
#将消费时间转为事件类型s
offline['date'] = pd.to_datetime(offline['Date'], format='%Y%m%d')
# 将折扣券转为折扣率
offline['discount_rate'] = offline['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else 
(float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
# 为数据打标
offline['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0,
        offline['date'],
        offline['date_received']))
# 添加优惠券是否为满减类型
offline['is_manjian'] = offline['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
# 领券时间为周几
offline['weekday_received'] = offline['date_received'].apply(lambda x: x.isoweekday())

# 为offline数据添加一个received_month即领券月份
offline['received_month'] = offline['date_received'].apply(lambda x: x.month)

# 为offline数据添加一个date_month即消费月份
offline['date_month'] = offline['date'].apply(lambda x: x.month)


path = './tmp/CH5_Task16_output'
if not os.path.exists(path):
        os.makedirs(path)
offline.to_csv(path+'/CH5_Task16_output.csv', index=False)