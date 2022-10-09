import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 用循环和list生成行列名 用numpy自带的random函数生成10行5列的数据
df1 = pd.DataFrame(np.random.rand(10, 5), columns=list('abcde'),
                   index=[i for i in range(10)])
#散点图
df1.plot.scatter(x='a', y='b')
#柱状图
df1.plot.bar()
#面积图
df1.plot.area()
#密度图
df1.plot.kde()
plt.show()

