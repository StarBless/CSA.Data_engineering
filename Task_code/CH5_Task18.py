import os
import pandas as pd
import pyecharts
from pyecharts.charts import Line
from pyecharts import options as opts

################# 数据准备
offline = pd.read_csv(r'../input_files/ccf_offline_stage1_train.csv')
offline['Distance'].fillna(-1, downcast='infer', inplace=True)
# 将领券时间转为时间类型
offline['date_received'] = pd.to_datetime(offline['Date_received'], format='%Y%m%d')
#将消费时间转为事件类型
offline['date'] = pd.to_datetime(offline['Date'], format='%Y%m%d')
# 打标
offline['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0,
        offline['date'],
        offline['date_received']))
# 为offline数据添加一个received_month即领券月份
offline['received_month'] = offline['date_received'].apply(lambda x: x.month)
# 为offline数据添加一个date_month即消费月份
offline['date_month'] = offline['date'].apply(lambda x: x.month)
# 领券时间为周几
offline['weekday_received'] = offline['date_received'].apply(lambda x: x.isoweekday())

# 每月各类消费折线图
# 每月核销的数量
consume_coupon = offline[offline['label']==1]['received_month'].value_counts(sort=False)
# 每月收到的数量
received = offline['received_month'].value_counts(sort=False)
# 每月消费数量
consume = offline['date_month'].value_counts(sort=False)

consume_coupon.sort_index(inplace=True)
received.sort_index(inplace=True)
consume.sort_index(inplace=True)

line_1 = (
    Line()
    .add_xaxis([str(x) for x in range(1, 7)])
    .add_yaxis('核销', list(consume_coupon))
    .add_yaxis('领取', list(received))
    .add_yaxis('消费', list(consume))
    .set_global_opts(title_opts={'text': '每月各类消费折线图'})
    .set_series_opts(
        opts.LabelOpts(is_show=False) # 显示值大小
    )
)




# 每周领券数与核销数折线图
# 统计每周核销的优惠券数量
week_coupon = offline[offline['label']==1]['weekday_received'].value_counts()
# 统计每周领券的优惠券数量
week_received = offline[offline['weekday_received'].notna()]['weekday_received'].value_counts()

week_coupon.sort_index(inplace=True)
week_received.sort_index(inplace=True)

line_2 = (
    Line()
    .add_xaxis([str(x) for x in range(1, 8)])
    .add_yaxis('领取', list(week_received))
    .add_yaxis('核销', list(week_coupon))
    .set_global_opts(title_opts={'text': '每周领券数与核销数折线图'})
)



path = './tmp/CH5_Task18_output'
if not os.path.exists(path):
        os.makedirs(path)
line_1.render(path+'/CH5_Task18_line_1.html')
line_2.render(path+'/CH5_Task18_line_2.html')
