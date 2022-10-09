import os
import pandas as pd
import pyecharts
from pyecharts.charts import Bar
from pyecharts import options as opts

# 查看pyecharts版本，本节代码只适合1.x版本
print(pyecharts.__version__)

################# 数据准备
offline = pd.read_csv(r'../input_files/ccf_offline_stage1_train.csv')
offline['Distance'].fillna(-1, downcast='infer', inplace=True)
# 将领券时间转为时间类型
offline['date_received'] = pd.to_datetime(offline['Date_received'], format='%Y%m%d')
#将消费时间转为事件类型
offline['date'] = pd.to_datetime(offline['Date'], format='%Y%m%d')
# 将折扣券转为折扣率
offline['discount_rate'] = offline['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else 
(float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
# 打标
offline['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0,
        offline['date'],
        offline['date_received']))

##########################
# 选取领券日期不为空的数据
df_1 = offline[offline['Date_received'].notna()]
# 以Date_received为分组目标并统计优惠券的数量
tmp = df_1.groupby('Date_received', as_index=False)['Coupon_id'].count()

# 建立柱状图
bar_1 = (
    Bar(
        init_opts = opts.InitOpts(width='1500px', height='600px')
    )
    .add_xaxis(list(tmp['Date_received']))
    .add_yaxis('', list(tmp['Coupon_id']))
    .set_global_opts(
        title_opts = opts.TitleOpts(title='每天被领券的数量'), # title
        legend_opts = opts.LegendOpts(is_show=True), # 显示ToolBox
        xaxis_opts = opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=60), interval=1), # 旋转60度
    )
    .set_series_opts(
        opts.LabelOpts(is_show=False), # 显示值大小
        markline_opts = opts.MarkLineOpts(
            data = [
                opts.MarkLineItem(type_='max', name='最大值')
            ]
        )
    )
)




######################
# 消费距离柱状图
# 统计各类距离的消费次数
import collections
dis = offline[offline['Distance']!=-1]['Distance'].values
dis = dict(collections.Counter(dis))

x = list(dis.keys())
y = list(dis.values())

# 建立柱状图
bar_2 = (
    Bar()
    .add_xaxis(x)
    .add_yaxis('', y)
    .set_global_opts(
        title_opts=opts.TitleOpts(title='用户消费距离统计'), # title
    )
)



##########
# 消费距离与核销率
rate = [offline[offline['Distance']==i]['label'].value_counts()[1]/
offline[offline['Distance']==i]['label'].value_counts().sum() for i in range(11)]

bar_3 = (
    Bar()
    .add_xaxis(list(range(11)))
    .add_yaxis('核销率', list(rate))
    .set_global_opts(title_opts=opts.TitleOpts(title='消费距离与核销率柱状图'))
    .set_series_opts(
        opts.LabelOpts(is_show=False) # 显示值大小
    )
)



##################
# 各种折扣率的领取和核销数量
# 统计领券日期不为空的数据中各种折扣率的优惠券领取数量
received = offline[['discount_rate']]
received['cnt'] = 1
received = received.groupby('discount_rate').agg('sum').reset_index()

# 统计领券日期不为空的数据中各种折扣率的优惠券核销数量
consume_coupon = offline[offline['label']==1][['discount_rate']]
consume_coupon['cnt_2'] = 1
consume_coupon = consume_coupon.groupby('discount_rate').agg('sum').reset_index()
data = received.merge(consume_coupon, on='discount_rate', how='left').fillna(0)  # 0.97500填充为0，用merge直接填充

bar_4 = (
    Bar()
    .add_xaxis([float('%.4f'%x) for x in list(data.discount_rate)])
    .add_yaxis('领取', list(data.cnt))
    .add_yaxis('核销', list(data.cnt_2))
    .set_global_opts(title_opts={'text': '领取与核销'})
    .set_series_opts(
        opts.LabelOpts(is_show=False) # 显示值大小
    )
)


path = './tmp/CH5_Task17_output'
if not os.path.exists(path):
        os.makedirs(path)
# render会生成本地HTML文件，默认在当前目录生成render.html文件
bar_1.render(path+'/bar_1.html')
bar_2.render(path+'/bar_2.html')
bar_3.render(path+'/bar_3.html')
bar_4.render(path+'/bar_4.html')
