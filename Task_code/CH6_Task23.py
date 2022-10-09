import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')  # 不显示警告

def prepare(dataset):
    """数据预处理

    """
    # 源数据
    data = dataset.copy()

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

if __name__ == "__main__":
    # 源数据
    off_train = pd.read_csv(r'../input_files/ccf_offline_stage1_train.csv')
    # 预处理
    off_train = prepare(off_train)
    # 打标
    off_train = get_label(off_train)

    path = './tmp/CH6_Task23_output'
    if not os.path.exists(path):
            os.makedirs(path)
    off_train.to_csv(path+'/off_train.csv', index=False)