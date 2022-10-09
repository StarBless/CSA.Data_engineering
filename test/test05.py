import pandas as pd
import xgboost as xgb


# 预处理

def prepare(f):
    # Discount_rate是否为满减
    f['is_full_reduction'] = f['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
    # 满减全部转化为折扣率(折扣率=(满多少-减多少)/满多少)
    f['discount_rate'] = f['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
    # 满减最低消费(不满减则填为-1)
    f['min_cost_full_reduction'] = f['Discount_rate'].map(
        lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0]))

    # 距离和时间预处理
    # 将Distance中缺失的值填充为-1
    f['Distance'].fillna(-1, inplace=True)
    # 增加一列判断是否为空Distance
    f['null_distance'] = f['Distance'].map(lambda x: 1 if x == -1 else 0)

    # 时间转换
    f['date_received'] = pd.to_datetime(f['Date_received'], format='%Y%m%d')
    if 'Date' in f.columns.tolist():
        f['date'] = pd.to_datetime(f['Date'], format='%Y%m%d')
    return f


# =============================================================================
# 打标
def get_label(dataset):
    # 源数据
    data = dataset.copy()
    # 打标：领券后15天内消费为1，否则为0
    data['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0, data['date'],
                             data['date_received']))
    return data


# =============================================================================
def get_simple_feature(label_field):
    # 源数据
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)  # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(int)
    # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['cnt'] = 1  # 方便特征提取
    # 返回的特征数据集
    feature = data.copy()

    # 用户领券数
    keys = ['User_id']  # 主键
    prefixs = 'simple_' + '_'.join(keys) + '_'  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')  # 将id列与特征列左连

    # 用户领取特定优惠券数
    keys = ['User_id', 'Coupon_id']  # 主键
    prefixs = 'simple_' + '_'.join(keys) + '_'  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')  # 将id列与特征列左连

    # 用户当天领券数
    keys = ['User_id', 'Date_received']  # 主键
    prefixs = 'simple_' + '_'.join(keys) + '_'  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')  # 将id列与特征列左连

    # 用户当天领取特定优惠券数
    keys = ['User_id', 'Coupon_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户是否在同一天重复领取了特定优惠券
    keys = ['User_id', 'Coupon_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt',
                           aggfunc=lambda x: 1 if len(x) > 1 else 0)  # 以keys为键,'cnt'为值,判断领取次数是否大于1
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'repeat_receive'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 删除辅助提特征的'cnt'
    feature.drop(['cnt'], axis=1, inplace=True)

    # 返回
    return feature


# =============================================================================

def get_week_feature(label_field):
    # 源数据
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)  # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(
        int)  # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    # 返回的特征数据集
    feature = data.copy()
    feature['week'] = feature['date_received'].map(lambda x: x.weekday())  # 星期几
    feature['is_weekend'] = feature['week'].map(lambda x: 1 if x == 5 or x == 6 else 0)  # 判断领券日是否为休息日
    feature = pd.concat([feature, pd.get_dummies(feature['week'], prefix='week')], axis=1)  # one-hot离散星期几
    feature.index = range(len(feature))  # 重置index
    # 返回
    return feature


# =============================================================================
# 提取用户特征值
def extract(label_field, history_field):
    history = history_field.copy()
    label = label_field.copy()
    # 将Coupon_id和Date_received中的float类型转换为int类型
    history['Coupon_id'] = history['Coupon_id'].map(int)
    history['Date_received'] = history['Date_received'].map(int)
    # 方便特征值的提取
    history['cnt'] = 1
    # 主键
    keys = ['User_id']
    # 特征名前缀
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    # 返回特征数据集,设置主表
    u_feat = label_field[keys].drop_duplicates(keep='first')

    ########1  用户领券数
    # 以keys为键，'cnt'为值，使用len统计出现的次数
    pivot = pd.pivot_table(history, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')  # concat
    # 缺失值填充为0，最好加上downcast= 'infer'，不然可能会改变DataFrame中某些列的类型
    u_feat.fillna(0, downcast='infer', inplace=True)

    ########2  用户领券未消费数
    # 筛选出Date为nan的样本，以keys为键，'cnt'为值，使用len统计出现次数
    pivot = pd.pivot_table(history[history['Date'].map(lambda x: str(x) == 'nan')], index=keys, values='cnt',
                           aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_not_consume_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    ########3  用户领券消费数
    # 筛选出Date不为nan的样本，以keys为键，'cnt'为值，使用len统计出现次数
    pivot = pd.pivot_table(history[history['Date'].map(lambda x: str(x) == 'nan')], index=keys, values='cnt',
                           aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={'cnt': prefixs + 'receive_and_consume_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    ########4  用户核销率
    # 核销率=领券并消费数/领券数
    u_feat[prefixs + 'receive_and_consume_rate'] = list(map(
        lambda x, y: x / y if y != 0 else 0,
        u_feat[prefixs + 'receive_and_consume_cnt'],
        u_feat[prefixs + 'receive_cnt']))

    ########5  领取并消费优惠券的平均折扣率
    pivot = pd.pivot_table(history[history['Date'].map(lambda x: str(x) != 'nan')],
                           index=keys, values='discount_rate', aggfunc='mean')
    pivot = pd.DataFrame(pivot).rename(columns={
        'discount_rate': prefixs + 'receive_and_consume_mean_discount_rate'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    ########6  领取并消费优惠券的平均距离
    tmp = history[history['Date'].map(lambda x: str(x) != 'nan')]
    pivot = pd.pivot_table(tmp[tmp['Distance'].map(lambda x: int(x) != -1)],
                           index=keys, values='Distance', aggfunc='mean')
    pivot = pd.DataFrame(pivot).rename(columns={
        'Distance': prefixs + 'receive_and_consume_mean_distance'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    ########7  在多少不同商家领取并消费优惠券
    pivot = pd.pivot_table(history[history['Date'].map(lambda x: str(x) != 'nan')],
                           index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={
        'Merchant_id': prefixs + 'receive_differ_Merchant_and_consume_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    ########8  在多少不同商家领取优惠券
    pivot = pd.pivot_table(history,
                           index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={
        'Merchant_id': prefixs + 'receive_differ_Merchant_cnt'}).reset_index()
    u_feat = pd.merge(u_feat, pivot, on=keys, how='left')
    u_feat.fillna(0, downcast='infer', inplace=True)

    ########9  在多少不同商家领取并消费优惠券 / 在多少不同商家领取优惠券
    u_feat[prefixs + 'receive_differ_Merchant_consume_rate'] = list(map(
        lambda x, y: x / y if y != 0 else 0,
        u_feat[prefixs + 'receive_differ_Merchant_and_consume_cnt'],
        u_feat[prefixs + 'receive_differ_Merchant_cnt']))

    #整合数据表并返回
    label = pd.merge(label, u_feat, on=['User_id'], how='left')
    return label


# =============================================================================
# 划分函数
def get_dataset(history_field, middle_field, label_field):
    # 特征工程
    simple_feat = get_simple_feature(label_field)  # 历史区间特征
    week_feat = get_week_feature(label_field)  # 中间区间特征
    user_feat = extract(label_field, history_field)  # 标签区间特征

    # 构造数据集
    share_chara = list(set(simple_feat.columns.tolist()) & set(week_feat.columns.tolist()) &
                       set(user_feat.columns.tolist()))  # 共有属性，包括id和一些基础特征，为每个特征快的交集,tolist矩阵变列表
    dataset = pd.concat([week_feat, simple_feat.drop(share_chara, axis=1)], axis=1)
    # 删除无用属性并将label置于最后一列
    if 'Date' in dataset.columns.tolist():
        dataset.drop(['Merchant_id', 'Discount_rate', 'Date', 'date_received', 'date'], axis=1, inplace=True)
        label = dataset['label'].tolist()
        dataset.drop(['label'], axis=1, inplace=True)
        dataset['label'] = label
    else:  # 表示测试集
        dataset.drop(['Merchant_id', 'Discount_rate', 'date_received'], axis=1, inplace=True)
    # 修正数据类型
    dataset['User_id'] = dataset['User_id'].map(int)
    dataset['Coupon_id'] = dataset['Coupon_id'].map(int)
    dataset['Date_received'] = dataset['Date_received'].map(int)
    dataset['Distance'] = dataset['Distance'].map(int)
    if 'label' in dataset.columns.tolist():
        dataset['label'] = dataset['label'].map(int)

    #去重并设置索引
    dataset.drop_duplicates(keep='first', inplace=True)
    dataset.index = range(len(dataset))
    return dataset


def model_xgb(train, test):
    # xgb参数
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'silent': 1,
              'eta': 0.01,
              'max_depth': 5,
              'min_child_weight': 1,
              'gamma': 0,
              'lambda': 1,
              'colsample_bylevel': 0.7,
              'colsample_bytree': 0.7,
              'subsample': 0.9,
              'scale_pos_weight': 1}
    # 数据集
    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    # 训练
    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=5167, evals=watchlist)
    # 预测
    predict = model.predict(dtest)
    predict = pd.DataFrame(predict)
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    # 特征重要性
    feat_importance = pd.DataFrame(columns=['feature_name', 'importance'])
    feat_importance['feature_name'] = model.get_score().keys()
    feat_importance['importance'] = model.get_score().values()
    feat_importance.sort_values(['importance'], ascending=False, inplace=True)
    # 返回
    return result, feat_importance


if __name__ == '__main__':
    off_train = pd.read_csv('ccf_offline_stage1_train.csv')
    # on_train=pd.read_csv('ccf_online_stage1_train.csv')
    off_test = pd.read_csv('ccf_offline_stage1_test_revised.csv')
    # 预处理
    off_train = prepare(off_train)
    off_test = prepare(off_test)
    # on_train = prepare(on_train)
    # 打标
    off_train = get_label(off_train)
    # off_test = get_label(off_test)
    # on_train = get_label(on_train)
    # 划分区间
    # 训练集历史区间、中间区间、标签区间
    train_history_field = off_train[off_train['date_received'].isin(pd.date_range('2016/3/2', periods=60))]
    train_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/5/1', periods=15))]
    train_label_field = off_train[off_train['date_received'].isin(pd.date_range('2016/5/16', periods=31))]
    # 验证集历史区间、中间区间、标签区间
    validate_history_field = off_train[off_train['date_received'].isin(pd.date_range('2016/1/16', periods=60))]
    validate_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/3/16', periods=15))]
    validate_label_field = off_train[off_train['date_received'].isin(pd.date_range('2016/3/31', periods=31))]
    # 测试集历史区间、中间区间、标签区间
    test_history_field = off_train[off_train['date_received'].isin(pd.date_range('2016/4/17', periods=60))]
    test_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/6/16', periods=15))]
    test_label_field = off_test.copy()

    # 构造训练集、验证集、测试集
    print('构造训练集')
    train = get_dataset(train_history_field, train_middle_field, train_label_field)
    print('构造验证集')
    validate = get_dataset(validate_history_field, validate_middle_field, validate_label_field)
    print('构造测试集')
    test = get_dataset(test_history_field, test_middle_field, test_label_field)

    big_train = pd.concat([train, validate], axis=0)
    result, feat_importance = model_xgb(big_train, test)
    result.to_csv('result.csv', index=False, header=None)
