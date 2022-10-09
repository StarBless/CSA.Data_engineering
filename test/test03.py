import pandas as pd
from pyecharts.charts import Bar, Line, Pie
from pyecharts import options as opts
import collections

# 读取数据集
df = pd.read_csv('../ccf_offline_stage1_test_revised.csv')
dft = df.copy()


# 显示数据记录和数据缺失情况
def show_data():
    # 共有多少条记录
    record_count = dft.shape[0]
    # 共有多少条优惠券的领取记录
    received_count = dft['User_id'].count()
    # 共有多少个用户
    user_count = len(dft['User_id'].value_counts())
    # 共有多少个商家
    merchant_count = len(dft['Merchant_id'].value_counts())
    # 共有多少种不同的优惠券
    coupon_count = len(dft['Coupon_id'].value_counts())
    # 最早领券时间
    min_received = str(int(dft['Date_received'].min()))
    # 最晚领券时间
    max_received = str(int(dft['Date_received'].max()))

    print(f'共有 {record_count} 条记录')
    print(f'共有 {received_count} 条优惠券的领取记录')
    print(f'共有 {user_count} 个用户')
    print(f'共有 {merchant_count} 个商家')
    print(f'共有 {coupon_count} 种不同的优惠券')
    print(f'最早领券时间 {min_received}')
    print(f'最晚领券时间 {max_received}')

    # 统计缺失值
    user_id_null = dft['User_id'].isnull().sum(axis=0)
    merchant_id_null = dft['Merchant_id'].isnull().sum(axis=0)
    coupon_id_null = dft['Coupon_id'].isnull().sum(axis=0)
    discount_rate_null = dft['Discount_rate'].isnull().sum(axis=0)
    distance_null = dft['Distance'].isnull().sum(axis=0)
    date_received_null = dft['Date_received'].isnull().sum(axis=0)

    print(f'User_id 缺失值为        {user_id_null}')
    print(f'Merchant_id 缺失值为    {merchant_id_null}')
    print(f'Coupon_id 缺失值为      {coupon_id_null}')
    print(f'Discount_rate 缺失值为  {discount_rate_null}')
    print(f'Distance 缺失值为       {distance_null}')
    print(f'Date_received 缺失值为  {date_received_null}')


# 建立每日领劵情况的柱状图
def show_bar_1():
    # 提取数据(无缺失数据)
    df_1 = dft[dft['Date_received'].notna()]
    # 分别统计每日领劵情况
    tmp = df_1.groupby('Date_received', as_index=False)['Coupon_id'].count()
    # 建立柱状图
    bar_1 = (
        Bar(
            init_opts=opts.InitOpts(width="1000px", height="500px")
        )
        .add_xaxis(list(tmp['Date_received']))
        .add_yaxis('', list(tmp['Coupon_id']))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="每日领劵情况"),
            legend_opts=opts.LegendOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=60), interval=1),
        )
        .set_series_opts(
            opts.LabelOpts(is_show=False),
            markline_opts=opts.MarkLineOpts(
                data=[
                    opts.MarkLineItem(type_="max", name="最大值"),
                ]
            )
        )
    )
    bar_1.render('bar_1.html')


# 绘制领劵数量与周几关系折线图
def show_line():
    # 将领劵日期转化为周几格式
    dft['date_received'] = pd.to_datetime(dft['Date_received'], format='%Y%m%d')
    dft['week_receive'] = dft['date_received'].apply(lambda x: x.isoweekday())

    week_coupon = dft['week_receive'].value_counts()
    week_coupon.sort_index(inplace=True)
    line_1 = (
        Line()
        .add_xaxis([('周' + x) for x in '一二三四五六日'])
        .add_yaxis("领劵数量", list(week_coupon))
        .set_global_opts(title_opts={"text": "领劵数量与周几关系折线图"})
    )
    line_1.render('line_1.html')


# 建立用户不同消费距离下的领劵数量统计柱状图
def show_bar_2():
    # 对数据缺失项进行补足
    dft['Distance'].fillna(-1, inplace=True)
    # 将数据按照消费距离分组统计
    dic = dft[dft['Distance'] != -1]['Distance'].values
    # 排序
    dic.sort()
    dic = dict(collections.Counter(dic))
    x = list(dic.keys())
    y = list(dic.values())
    # 建立柱状图
    bar_2 = (
        Bar()
        .add_xaxis(x)
        .add_yaxis('', y)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="用户不同消费距离下的领劵数量统计"),
        )
    )
    bar_2.render('bar_2.html')


# 绘制不同折扣率的优惠券领取数量折线图
def show_bar_3():
    # 将满减格式的折扣数据进行转换
    dft['discount_rate'] = dft['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1]))
    / (float(str(x).split(':')[0])))
    # 将折扣率数据转换为各种折扣率下的优惠券数量
    discount_data = dft[['discount_rate']].copy()
    discount_data['cnt'] = 1
    discount_data = discount_data.groupby('discount_rate').agg('sum').reset_index()
    # 绘制柱状图
    bar_3 = (
        Bar(
            init_opts=opts.InitOpts(width="1000px", height="500px")
        )
        .add_xaxis(list(discount_data['discount_rate']))
        .add_yaxis('领取数量', list(discount_data['cnt']))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="不同折扣率的优惠券领取数量折线图")
        )
        .set_series_opts(
            opts.LabelOpts(is_show=True)
        )
    )
    bar_3.render("bar_3.html")


# 两种优惠券数量占比饼图
def show_pie_1():
    # 获得满减类型的优惠券数据
    dft['is_full_reduction'] = dft['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
    # 优惠券分为满减和直接折扣两类
    v1 = ['折扣', '满减']
    v2 = list(dft[dft['Date_received'].notna()]['is_full_reduction'].value_counts(normalize=True))
    pie_1 = (
        Pie()
        .add("", [list(v) for v in zip(v1, v2)])
        .set_global_opts(title_opts={"text": "两种优惠券数量占比饼图"})
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}:{c}"))
    )
    pie_1.render('pie_1.html')

    # 提取距离不为空的数据
    df_1 = dft[dft['Distance'] != -1]
    df_1 = df_1.loc[:, ['Distance', 'discount_rate']]
    # 分组统计各个距离下的各种优惠卷领取情况
    df_1 = df_1.groupby([df_1['Distance'], df_1['discount_rate']]).agg('size')
    # 提取对应距离对应的折扣率的统计个数
    cnt = [dict(df_1[i]) for i in range(11)]
    idx = list(df_1[0].keys())
    # 将折扣率保存为3位小数
    idx = [float('%.3f' % x) for x in idx]
    # 选取距离为[0，4，8, 10]
    dis = [0, 4, 8, 10]
    num = []
    # 将对应统计个数按相同长度存到一个List里
    for i in dis:
        tmp = []
        for x in idx:
            if x in cnt[i]:
                tmp.append(int(cnt[i][x]))
            else:
                tmp.append(int(0))
        num.append(tmp)
    for x in range(4):
        pie = (
            Pie()
            .add(' ', [list(v) for v in zip(idx, num[x])])
            .set_global_opts(title_opts=opts.TitleOpts(title='消费距离为%d时的不同折扣率优惠券占比' % dis[x], pos_top='90%'))
        )
        pie.render('消费距离为%d时的不同折扣率优惠券占比.html' % dis[x])


show_data()
show_bar_1()
show_line()
show_bar_2()
show_bar_3()
show_pie_1()
