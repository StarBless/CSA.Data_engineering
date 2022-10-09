'plot'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'饼图绘制'
# 对生成的一列随机数的series数据类型绘制饼图
df1 = pd.Series(3*np.random.rand(4), index=['a', 'b', 'c', 'd'], name='series')
df1.plot.pie(figsize=(6, 6))
'柱状图绘制'
# 对生成的四列随机数的dataframe数据类型绘制柱状图
df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
df2.plot.bar()
'箱线图绘制'
# 对生成的五列随机数的dataframe数据类型绘制箱线图
df3 = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
df3.plot.box()
'散点图绘制'
# 对生成的四列随机数的dataframe数据类型绘制散点图
df4 = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
df4.plot.scatter(x='a', y='b')
plt.show()