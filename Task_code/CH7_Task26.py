import os
import pandas as pd
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')  # 不显示警告

def model_xgb(train, test):
    """xgb模型

    Args:

    Returns:

    """
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
    model = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist)
    # 预测
    predict = model.predict(dtest)
    # 处理结果
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    # 特征重要性
    feat_importance = pd.DataFrame(columns=['feature_name', 'importance'])
    feat_importance['feature_name'] = model.get_score().keys()
    feat_importance['importance'] = model.get_score().values()
    feat_importance.sort_values(['importance'], ascending=False, inplace=True)
    # 返回
    return result, feat_importance

train = pd.read_csv('tmp/CH7_Task25_output/train.csv')
validate = pd.read_csv('tmp/CH7_Task25_output/validate.csv')
test = pd.read_csv('tmp/CH7_Task25_output/test.csv')

# 线下验证
result_off, feat_importance_off = model_xgb(train, validate.drop(['label'], axis=1))
# 线上训练
big_train = pd.concat([train, validate], axis=0)
result, feat_importance = model_xgb(big_train, test)

# 保存
path = './tmp/CH7_Task26_output'
if not os.path.exists(path):
        os.makedirs(path)
result_off.to_csv(path+'/result_off.csv', index=False)
result.to_csv(path+'/result.csv', index=False)
