import os
import pandas as pd
import pyecharts
from pyecharts.charts import Pie
from pyecharts import options as opts

######################### 数据准备
offline = pd.read_csv(r'../input_files/ccf_offline_stage1_train.csv')
# 将领券时间转为时间类型
offline['date_received'] = pd.to_datetime(offline['Date_received'], format='%Y%m%d')
#将消费时间转为事件类型
offline['date'] = pd.to_datetime(offline['Date'], format='%Y%m%d')
# 为数据打标
offline['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0,
        offline['date'],
        offline['date_received']))
# 添加优惠券是否为满减类型
offline['is_manjian'] = offline['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)

# 各类优惠券数量占比饼图
v1 = ['折扣', '满减']
v2 = list(offline[offline['Date_received'].notna()]['is_manjian'].value_counts(True))
print(v2)
pie_1 = (
    Pie()
    .add('', [list(v) for v in zip(v1, v2)])
    .set_global_opts(title_opts={'text': '各类优惠券数量占比饼图'})
    .set_series_opts(label_opts=opts.LabelOpts(formatter='{b}: {c}')) #格式化标签输出内容
)



# 核销优惠券数量占比饼图
v3 = list(offline[offline['label']==1].is_manjian.value_counts(True))
pie_2 = (
    Pie()
    .add('', [list(v) for v in zip(v1, v3)])
    .set_global_opts(title_opts={'text': '核销优惠券数量占比饼图'})
    .set_series_opts(label_opts=opts.LabelOpts(formatter='{b}: {c}')) # 格式化标签输出内容
)



# 正负例饼图
v4 = ['正例', '负例']
v5 = list(offline['label'].value_counts(True))

pie_3 = (
    Pie()
    .add('', [list(v) for v in zip(v4, v5)])
    .set_global_opts(title_opts={'text': '正负例饼图'})
    .set_series_opts(label_opts=opts.LabelOpts(formatter='{b}: {c}')) # 格式化标签输出内容
)


path = './tmp/CH5_Task19_output'
if not os.path.exists(path):
        os.makedirs(path)
pie_1.render(path+'/CH5_Task19_pie_1.html')
pie_2.render(path+'/CH5_Task19_pie_2.html')
pie_3.render(path+'/CH5_Task19_pie_3.html')
