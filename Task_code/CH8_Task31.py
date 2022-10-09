import os
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')  # 不显示警告


def prepare(dataset):
    """数据预处理

    1.时间处理(方便计算时间差):
        将Date_received列中int或float类型的元素转换成datetime类型,新增一列date_received存储;
        将Date列中int类型的元素转换为datetime类型,新增一列date存储;

    2.折扣处理:
        判断折扣率是“满减”(如10:1)还是“折扣率”(0.9);
        将“满减”折扣转换为“折扣率”形式(如10:1转换为0.9);
        得到“满减”折扣的最低消费(如折扣10:1的最低消费为10);
    3.距离处理:
        将空距离填充为-1(区别于距离0,1,2,3,4,5,6,7,8,9,10);
        判断是否为空距离;

    Args:
        dataset: DataFrame类型的数据集off_train和off_test,包含属性'User_id','Merchant_id','Coupon_id','Discount_rate',
            'Distance','Date_received','Date'(off_test没有'Date'属性);

    Returns:
        预处理后的DataFrame类型的数据集.
    """
    # 源数据
    data = dataset.copy()
    # 折扣率处理
    data['is_manjian'] = data['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)  # Discount_rate是否为满减
    data['discount_rate'] = data['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))  # 满减全部转换为折扣率
    data['min_cost_of_manjian'] = data['Discount_rate'].map(
        lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0]))  # 满减的最低消费
    # 距离处理
    data['Distance'].fillna(-1, inplace=True)  # 空距离填充为-1
    data['null_distance'] = data['Distance'].map(lambda x: 1 if x == -1 else 0)
    # 时间处理
    data['date_received'] = pd.to_datetime(data['Date_received'], format='%Y%m%d')
    if 'Date' in data.columns.tolist():  # off_train
        data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    # 返回
    return data


def get_label(dataset):
    """打标

    领取优惠券后15天内使用的样本标签为1,否则为0;

    Args:
        dataset: DataFrame类型的数据集off_train,包含属性'User_id','Merchant_id','Coupon_id','Discount_rate',
            'Distance','Date_received','Date'

    Returns:
        打标后的DataFrame类型的数据集.
    """
    # 源数据
    data = dataset.copy()
    # 打标:领券后15天内消费为1,否则为0
    data['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0, data['date'],
                             data['date_received']))
    # 返回
    return data

def get_history_field_coupon_feature(label_field, history_field):
    # 源数据
    data = history_field.copy()
    # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Coupon_id'] = data['Coupon_id'].map(int)
    # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1  # 方便特征提取

    # 主键
    keys = ['Coupon_id']
    # 特征名前缀,由history_field和主键组成
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    # 返回的特征数据集
    c_feat = label_field[keys].drop_duplicates(keep='first')

    # 优惠券被领取的次数
    # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    pivot = pd.DataFrame(pivot).rename(
        columns={'cnt': prefixs + 'received_cnt'}).reset_index()
    # 将id列与特征列左连
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    # 缺失值填充为0,最好加上参数downcast='infer',不然可能会改变DataFrame某些列中元素的类型
    c_feat.fillna(0, downcast='infer', inplace=True)

    # 优惠券15天内被核销的次数
    # 先筛选出label为1即领券15天内核销的样本,以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.pivot_table(data[data['label'] == 1],
                           index=keys, values='cnt', aggfunc=len)
    # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    pivot = pd.DataFrame(pivot).rename(
        columns={'cnt': prefixs + 'received_and_consumed_cnt_15'}).reset_index()
    # 将id列与特征列左连
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    # 缺失值填充为0,最好加上参数downcast='infer',不然可能会改变DataFrame某些列中元素的类型
    c_feat.fillna(0, downcast='infer', inplace=True)

    # 优惠券15天内的被核销率
    c_feat[prefixs + 'received_and_consumed_rate_15'] = list(map(
        lambda x, y: x / y if y != 0 else 0,
        c_feat[prefixs + 'received_and_consumed_cnt_15'],
        c_feat[prefixs + 'received_cnt']))

    # 优惠券15天内被核销的平均时间间隔
    # 筛选出label为1即领券15天内核销的样本
    tmp = data[data['label'] == 1]
    # 核销与领券的时间间隔,以天为单位
    tmp['gap'] = (tmp['date'] - tmp['date_received']
                  ).map(lambda x: x.total_seconds() / (60 * 60 * 24))
    # 以keys为键,'gap'为值,使用np.mean统计平均时间间隔
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=np.mean)
    # pivot_table后keys会成为index,统计出的特征列会以values即'gap'命名,将其改名为特征名前缀+特征意义,并将index还原
    pivot = pd.DataFrame(pivot).rename(
        columns={'gap': prefixs + 'consumed_mean_time_gap_15'}).reset_index()
    # 将id列与特征列左连
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    # 缺失值填充为-1,最好加上参数downcast='infer',不然可能会改变DataFrame某些列中元素的类型
    c_feat.fillna(-1, downcast='infer', inplace=True)

    # 满减优惠券最低消费的中位数
    # 先筛选出is_manjian为1即满减券的样本,以keys为键,'min_cost_of_manjian'为值,使用np.median统计中位数
    pivot = pd.pivot_table(data[data['is_manjian'] == 1], index=keys, values='min_cost_of_manjian',
                           aggfunc=np.median)
    # pivot_table后keys会成为index,统计出的特征列会以values即'min_cost_of_manjian'命名,将其改名为特征名前缀+特征意义,并将index还原
    pivot = pd.DataFrame(pivot).rename(
        columns={'min_cost_of_manjian': prefixs + 'median_of_min_cost_of_manjian'}).reset_index()
    # 将id列与特征列左连
    c_feat = pd.merge(c_feat, pivot, on=keys, how='left')
    # 缺失值填充为0,最好加上参数downcast='infer',不然可能会改变DataFrame某些列中元素的类型
    c_feat.fillna(0, downcast='infer', inplace=True)

    # 缺失值填充为0,最好加上参数downcast='infer',不然可能会改变DataFrame某些列中元素的类型
    c_feat.fillna(0, downcast='infer', inplace=True)

    # 返回
    return c_feat




def get_history_field_feature(label_field, history_field):
    # 用户特征
    c_feat = get_history_field_coupon_feature(label_field, history_field)
    # 添加特征
    history_feat = label_field.copy()
    # 添加用户特征
    history_feat = pd.merge(history_feat, c_feat, on=['Coupon_id'], how='left')
    # 返回
    return history_feat



def get_dataset(history_field, middle_field, label_field):
    # 特征工程
    # 历史区间特征
    history_feat = get_history_field_feature(label_field, history_field)
    # 构造数据集
    # 共有属性,包括id和一些基础特征,为每个特征块的交集
    share_characters = list(set(label_field.columns.tolist()) &
                            set(history_feat.columns.tolist()))
    label_field.index = range(len(label_field))
    dataset = pd.concat(
        [label_field, history_feat.drop(share_characters, axis=1)], axis=1)
    
    # 删除无用属性并将label置于最后一列
    if 'Date' in dataset.columns.tolist():  # 表示训练集和验证集
        # 删除无用属性
        dataset.drop(['Merchant_id', 'Discount_rate', 'Date',
                      'date_received', 'date'], axis=1, inplace=True)
        label = dataset['label'].tolist()
        dataset.drop(['label'], axis=1, inplace=True)
        dataset['label'] = label
    else:  # 表示测试集
        dataset.drop(['Merchant_id', 'Discount_rate',
                      'date_received'], axis=1, inplace=True)
    # 修正数据类型
    dataset['User_id'] = dataset['User_id'].map(int)
    dataset['Coupon_id'] = dataset['Coupon_id'].map(int)
    dataset['Date_received'] = dataset['Date_received'].map(int)
    dataset['Distance'] = dataset['Distance'].map(int)
    if 'label' in dataset.columns.tolist():
        dataset['label'] = dataset['label'].map(int)
    # 去重
    dataset.drop_duplicates(keep='first', inplace=True)
    # 这里一定要重置index,若不重置index会导致pd.concat出现问题
    dataset.index = range(len(dataset))
    # 返回
    return dataset





if __name__ == '__main__':
    # 源数据
    off_train = pd.read_csv(r'../input_files/ccf_offline_stage1_train.csv')
    off_test = pd.read_csv(r'../input_files/ccf_offline_stage1_test_revised.csv')
    # 预处理
    off_train = prepare(off_train)
    off_test = prepare(off_test)
    # 打标
    off_train = get_label(off_train)

    # 划分区间
    # 训练集历史区间、中间区间、标签区间
    train_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/3/2', periods=60))]  # [20160302,20160501)
    train_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/5/1', periods=15))]  # [20160501,20160516)
    train_label_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/5/16', periods=31))]  # [20160516,20160616)
    # 验证集历史区间、中间区间、标签区间
    validate_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/1/16', periods=60))]  # [20160116,20160316)
    validate_middle_field = off_train[
        off_train['date'].isin(pd.date_range('2016/3/16', periods=15))]  # [20160316,20160331)
    validate_label_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/3/31', periods=31))]  # [20160331,20160501)
    # 测试集历史区间、中间区间、标签区间
    test_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/4/17', periods=60))]  # [20160417,20160616)
    test_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/6/16', periods=15))]  # [20160616,20160701)
    test_label_field = off_test.copy()  # [20160701,20160801)

    # 构造训练集、验证集、测试集
    print('构造训练集')
    train = get_dataset(train_history_field, train_middle_field, train_label_field)
    print('构造验证集')
    validate = get_dataset(validate_history_field, validate_middle_field, validate_label_field)
    print('构造测试集')
    test = get_dataset(test_history_field, test_middle_field, test_label_field)

    path = './tmp/CH8_Task31_output'
    if not os.path.exists(path):
            os.makedirs(path)
    train.to_csv(path+'/train.csv', index=False)
    validate.to_csv(path+'/validate.csv', index=False)
    test.to_csv(path+'/test.csv', index=False)







