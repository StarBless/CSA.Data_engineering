import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')  # 不显示警告

def prepare(dataset):
    """数据预处理

    1.折扣处理:
        判断折扣是“满减”(如10:1)还是“折扣率”(0.9), 新增一列'is_manjian'表示该信息;
        将“满减”折扣转换为“折扣率”形式(如10:1转换为0.9), 新增一列'discount_rate'表示该信息;
        得到“满减”折扣的最低消费(如折扣10:1的最低消费为10), 新增一列'min_cost_of_manjian'表示该信息;
    2.距离处理:
        将空距离填充为-1(区别于距离0,1,2,3,4,5,6,7,8,9,10);
        判断是否为空距离, 新增一列'null_distance'表示该信息;
    3.时间处理(方便计算时间差):
        将'Date_received'列中int或float类型的元素转换成datetime类型, 新增一列'date_received'表示该信息;
        将'Date'列中int类型的元素转换为datetime类型, 新增一列'date'表示该信息;

    Args:
        dataset: off_train和off_test, DataFrame类型的数据集包含属性'User_id', 'Merchant_id', 'Coupon_id',
                'Discount_rate', 'Distance', 'Date_received', 'Date'(off_test没有'Date'属性);

    Returns:
        data: 预处理后的DataFrame类型的数据集.
    """
    # 源数据
    data = dataset.copy()

    # 折扣处理
    # Discount_rate是否为满减
    data['is_manjian'] = data['Discount_rate'].map(
        lambda x: 1 if ':' in str(x) else 0)
    # 满减全部转换为折扣率
    data['discount_rate'] = data['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
                                                      (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
    # 满减的最低消费
    data['min_cost_of_manjian'] = data['Discount_rate'].map(lambda x: -1 if ':' not in str(x) else
                                                            int(str(x).split(':')[0]))

    # 距离处理
    # 空距离填充为-1
    data['Distance'].fillna(-1, inplace=True)
    # 判断是否是空距离
    data['null_distance'] = data['Distance'].map(lambda x: 1 if x == -1 else 0)

    # 时间处理
    #  float类型转换为datetime类型
    data['date_received'] = pd.to_datetime(
        data['Date_received'], format='%Y%m%d')
    if 'Date' in data.columns.tolist():  # off_train
        #  float类型转换为datetime类型
        data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    # 返回
    return data


def get_label(dataset):
    """打标

    领取优惠券后15天内使用的样本标签为1, 否则标签为0, 新增一列'label'表示该打标信息;

    Args:
        dataset: off_train, DataFrame类型的数据集, 包含属性'User_id', 'Merchant_id', 
                 'Coupon_id', 'Discount_rate', 'Distance', 'Date_received','Date';

    Returns:
        data: 打标后的DataFrame类型的数据集.
    """
    # 源数据
    data = dataset.copy()
    # 打标:领券后15天内消费为1,否则为0
    data['label'] = list(map(
        lambda x, y: 1 if (x - y).total_seconds() /
        (60 * 60 * 24) <= 15 else 0,
        data['date'],
        data['date_received']))
    # 返回
    return data

def get_label_field_feature(label_field):
    """
    写入提取标签特征的代码，提取的特征按需要的主键连接到label_field，返回label_field(这里作为示例没有补全)
    """
    return label_field

def get_middle_field_feature(label_field, middle_field):
    """
    写入提取中间特征的代码，提取的特征按需要的主键连接到label_field，返回label_field(这里作为示例没有补全)
    """
    return label_field

def get_history_field_feature(label_field, history_field):
    """
    写入提取历史特征的代码，提取的特征按需要的主键连接到label_field，返回label_field(这里作为示例没有补全)
    """
    return label_field




def get_dataset(history_field, middle_field, label_field):
    """构造数据集

    从历史区间, 中间区间, 标签区间提取特征并合成最终的数据集;

    Args:
        history_field: 历史区间, DataFrame类型的数据集;
        middle_field: 中间区间, DataFrame类型的数据集;
        label_field: 标签区间, DataFrame类型的数据集;

    Returns:
        dataset: 构造完成的DataFrame类型的数据集.
    """
    # 特征工程
    # 标签区间特征
    label_feat = get_label_field_feature(label_field)
    # 中间区间特征
    middle_feat = get_middle_field_feature(label_field, middle_field)
    # 历史区间特征
    history_feat = get_history_field_feature(label_field, history_field)
    # 构造数据集
    # 共有属性,包括id和一些基础特征,为每个特征块的交集
    share_characters = list(set(history_feat.columns.tolist()) &
                            set(middle_feat.columns.tolist()) &
                            set(label_feat.columns.tolist()))
    # 这里使用concat连接而不用merge,因为几个特征块的样本顺序一致,index一致,但需要注意在连接两个特征块时要删去其中一个特征块的共有属性
    dataset = pd.concat(
        [history_feat, middle_feat.drop(share_characters, axis=1)], axis=1)
    dataset = pd.concat(
        [dataset, label_feat.drop(share_characters, axis=1)], axis=1)
    
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
    off_test = pd.read_csv(
        r'../input_files/ccf_offline_stage1_test_revised.csv')
    # 预处理
    off_train = prepare(off_train)
    off_test = prepare(off_test)
    # 打标
    off_train = get_label(off_train)

    # 划分区间
    # 训练集历史区间[20160302,20160501)、中间区间[20160501,20160516)、标签区间[20160516,20160616)
    train_history_field = off_train[off_train['date_received'].isin(
        pd.date_range('2016/3/2', periods=60))]
    train_middle_field = off_train[off_train['date'].isin(
        pd.date_range('2016/5/1', periods=15))]
    train_label_field = off_train[off_train['date_received'].isin(
        pd.date_range('2016/5/16', periods=31))]
    # 验证集历史区间[20160116,20160316)、中间区间[20160316,20160331)、标签区间[20160331,20160501)
    validate_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/1/16', periods=60))]
    validate_middle_field = off_train[
        off_train['date'].isin(pd.date_range('2016/3/16', periods=15))]
    validate_label_field = off_train[off_train['date_received'].isin(
        pd.date_range('2016/3/31', periods=31))]
    # 测试集历史区间[20160417,20160616)、中间区间[20160616,20160701)、标签区间[20160701,20160801)
    test_history_field = off_train[off_train['date_received'].isin(
        pd.date_range('2016/4/17', periods=60))]
    test_middle_field = off_train[off_train['date'].isin(
        pd.date_range('2016/6/16', periods=15))]
    test_label_field = off_test.copy()

    # 构造训练集、验证集、测试集
    print('构造训练集')
    train = get_dataset(train_history_field,
                        train_middle_field, train_label_field)
    print('构造验证集')
    validate = get_dataset(validate_history_field,
                           validate_middle_field, validate_label_field)
    print('构造测试集')
    test = get_dataset(test_history_field, test_middle_field, test_label_field)

path = './tmp/CH6_Task22_output'
if not os.path.exists(path):
        os.makedirs(path)
train.to_csv(path+'/train.csv', index=False)
validate.to_csv(path+'/validate.csv', index=False)
test.to_csv(path+'/test.csv', index=False)

    