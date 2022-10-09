import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition


# 初始化(初始化聚类中心并构建分类结果数组)
def initialization(train_data, k):
    row, column = train_data.shape
    cent = np.zeros((k, column))
    # 两列分别储存分类结果以及样本至簇中心点的误差
    result = np.array(np.zeros((row, 2)))

    for i in range(k):
        x = int(np.random.uniform(0, row))
        cent[i, :] = train_data[x, :]
    return cent, result


# 计算欧式距离
def euclid_distance(x, cent):
    return np.sqrt(np.sum(np.power(x - cent, 2)))


# 获得离样本最近的质心
def min_dist(cent, train_data, k, j):
    # 初始化距质心最小距离以及簇
    min_dis = 1e8
    min_index = -1
    # 遍历所有质心
    for i in range(k):
        # 计算样本到质心的欧式距离
        distance = euclid_distance(cent[i, :], train_data[j, :])
        if distance < min_dis:
            min_dis = distance
            min_index = i
    return min_dis, min_index


# k均值聚类
def k_means(train_data, k, count):
    # 设置迭代结束标记
    flag = True
    # 第1步-初始化
    cent, result = initialization(train_data, k)
    row, _ = train_data.shape
    # 开始迭代
    for x in range(count):
        flag = False
        # 遍历所有样本
        for i in range(row):
            # 第2步-找出最近的质心
            min_dis, min_index = min_dist(cent, train_data, k, i)
            # 第3步-更新每一行样本所属的簇
            if result[i, 0] != min_index:
                flag = True
                result[i, :] = min_index, min_dis ** 2
        # 第4步-更新质心
        for i in range(k):
            # 获取簇类所有的点
            point = train_data[np.nonzero(result[:, 0] == i)[0]]
            # 对簇中所有的点求均值更新为质心
            cent[i, :] = np.mean(point, axis=0)
        # 聚类稳定时停止迭代
        if not flag:
            break
    cent = pd.DataFrame(cent)
    result = pd.DataFrame(result)
    return cent, result


# 可视化(绘制散点图)
def show(data):
    ax = data[data['label'] == 0].plot(kind='scatter', x='x', y='y', color='red', label='0')
    data[data['label'] == 1].plot(kind='scatter', x='x', y='y', color='blue', label='1', ax=ax)
    data[data['label'] == 2].plot(kind='scatter', x='x', y='y', color='green', label='2', ax=ax)
    data[data['label'] == 3].plot(kind='scatter', x='x', y='y', color='orange', label='3', ax=ax)
    data[data['label'] == 4].plot(kind='scatter', x='x', y='y', color='purple', label='4', ax=ax)
    data[data['label'] == 5].plot(kind='scatter', x='x', y='y', color='black', label='5', ax=ax)
    data[data['label'] == 6].plot(kind='scatter', x='x', y='y', color='yellow', label='6', ax=ax)
    plt.show()


# pca降维并建立散点图
def pca_2d(train_data, result):
    train_data = pd.DataFrame(train_data)
    pca = decomposition.PCA(n_components=2)
    new_data = pd.DataFrame()
    data = pca.fit_transform(train_data.iloc[:, :-1].values)
    new_data['x'] = data[:, 0]
    new_data['y'] = data[:, 1]
    new_data['label'] = result[0]
    return new_data


# 手肘法确定k值
def SSE(train_data):
    sse = []
    # 枚举k值
    for i in range(1, 11):
        cent, result = k_means(train_data, i, 100)
        sum = 0
        for j in range(i):
            for k in range(train_data.shape[0]):
                if result.loc[k].loc[0] == j:
                    sum += (result.loc[k].loc[1]) ** 2
        sse.append(sum)
    plt.plot(range(1, 11), sse)
    plt.show()


# 鸢尾花数据集
def iris():
    data = pd.read_csv('iris.data', names=['a', 'b', 'c', 'd', 'e'])
    data = data.drop(columns='e')
    train_data = data.values
    SSE(train_data)
    cent, result = k_means(train_data, 3, 10000000)
    new_data = pca_2d(train_data, result)
    show(new_data)


# 葡萄酒数据集
def wine():
    data = pd.read_csv('wine.data', header=None)
    data = data.drop(columns=13)
    train_data = data.values
    SSE(train_data)
    cent, result = k_means(train_data, 4, 10000000)
    new_data = pca_2d(train_data, result)
    show(new_data)


# 钞票认证数据集
def banknote():
    data = pd.read_csv('data_banknote_authentication.txt', header=None)
    data = data.drop(columns=4)
    train_data = data.values
    SSE(train_data)
    cent, result = k_means(train_data, 4, 10000000)
    new_data = pca_2d(train_data, result)
    show(new_data)

# balance_scale数据集
def balance_scale():
    data = pd.read_csv('balance-scale.data', header=None)
    data = data.drop(columns=0)
    train_data = data.values
    SSE(train_data)
    cent, result = k_means(train_data, 4, 10000000)
    new_data = pca_2d(train_data, result)
    show(new_data)

# 红酒质量数据集
def cervical_carcinoma():
    data = pd.read_csv('winequality-red.csv', sep = ';')
    data = data.drop('quality', axis= 1)
    train_data = data.values
    SSE(train_data)
    cent, result = k_means(train_data, 6, 10000000)
    new_data = pca_2d(train_data, result)
    show(new_data)


iris()
wine()
banknote()
balance_scale()
cervical_carcinoma()