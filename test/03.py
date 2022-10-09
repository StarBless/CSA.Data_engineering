import pandas as pd
import pyecharts
from pyecharts.charts import Bar,Line,Pie
from pyecharts import options as opts
import collections

#数据观察
f = pd.read_csv('../ccf_offline_stage1_test_revised.csv')
record = f.shape[0]
print("有{}条记录".format(record))
received = f['Date_received'].count()
print("有{}条优惠券领取记录".format(received))
coupon = len(f['Coupon_id'].value_counts())
print("有{}种不同的优惠券".format(coupon))
user = len(f['User_id'].value_counts())
print("有{}个用户".format(user))
merchant = len(f['Merchant_id'].value_counts())
print("有{}个商家".format(merchant))
min_received = str(int(f['Date_received'].min()))
print("最早领券时间{}".format(min_received))
max_received = str(int(f['Date_received'].max()))
print("最晚领券时间{}".format(max_received))
print(f.isnull().sum())
print(pyecharts.__version__)

# =============================================================================
#对数据进行处理（缺失项等）

f1 = f.copy()
f1['Distance'].fillna(-1,inplace = True)
f1['date_received']=pd.to_datetime(f1['Date_received'],format='%Y%m%d')
f1['discount_rate']=f1['Discount_rate'].map(lambda x:float(x) if ':' not in str(x) else
                                            (float(str(x).split(':')[0]) - float(str(x).split(':')[1]))
                                            /(float(str(x).split(':')[0])))

f1['ismanjian'] = f1['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
f1['week_receive'] = f1['date_received'].apply(lambda x:x.isoweekday())

# =============================================================================
# 建立柱状图
df_1 = f1[f1['Date_received'].notna()]
tmp=df_1.groupby('Date_received',as_index=False)['Coupon_id'].count()
bar_1 = (
    Bar(
        init_opts=opts.InitOpts(width="1500px",height="600px")
    )
    .add_xaxis(list(tmp['Date_received']))
    .add_yaxis('',list(tmp['Coupon_id']))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="每天被领券的数量"),
        legend_opts=opts.LegendOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=60),interval=1),
        )
    .set_series_opts(
        opts.LabelOpts(is_show=False),
        markline_opts=opts.MarkLineOpts(
                data=[
                    opts.MarkLineItem(type_="max",name="最大值"),
                ]
    )
    )
)
bar_1.render('bar_1.html')

# =============================================================================
#周一到周末领券数量折线图

week_coupon = f1['week_receive'].value_counts()
week_coupon.sort_index(inplace = True)
line_1 = (
          Line()
          .add_xaxis([('周'+x) for x in '一二三四五六天'])
          .add_yaxis("领取",list(week_coupon))
          .set_global_opts(title_opts = {"text":"星期几与领券数关系折线图"})
          )
line_1.render('line_1.html')

# =============================================================================
# 不同消费距离的优惠券数量

dis = f1[f1['Distance'] != -1]['Distance'].values
dis = dict(collections.Counter(dis))
x = list(dis.keys())
y = list(dis.values())
#建立柱状图
bar_2 = (
    Bar()
    .add_xaxis(x)
    .add_yaxis('',y)
    .set_global_opts(
        title_opts=opts.TitleOpts(title="用户消费距离统计"),
    )
)
bar_2.render('bar_2.html')

# =============================================================================
#各类优惠券数量占比饼图

v1 = ['折扣','满减']
v2 = list(f1[f1['Date_received'].notna()]['ismanjian'].value_counts(normalize=True))#value_counts(normalize=True)统计占比
print(v2)
pie_1 = (
    Pie()
    .add("",[list(v) for v in zip(v1,v2)])
    .set_global_opts(title_opts = {"text":"各类优惠券数量占比饼图"})
    .set_series_opts(label_opts=opts.LabelOpts(formatter = "{b}:{c}"))
)
pie_1.render('pie_1.html')

# =============================================================================
#各种折扣率的优惠券使用数量

discount_received = f1[['discount_rate']]
discount_received['cnt'] =1
discount_received= discount_received.groupby('discount_rate').agg('sum').reset_index()
bar_3= (
        Bar(
            init_opts = opts.InitOpts(width="1600px", height = "600px")
            )
        .add_xaxis(list(discount_received['discount_rate']))
        .add_yaxis('领取数量',list(discount_received['cnt']))
        .set_global_opts(
            title_opts = opts.TitleOpts(title="不同折扣率的优惠券领取数量")
            )
        .set_series_opts(
            opts.LabelOpts(is_show = True)
            )
        )
bar_3.render("bar_3.html")
