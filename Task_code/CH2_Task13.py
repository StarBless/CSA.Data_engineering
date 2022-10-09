### pandas库的使用教程 ###
import numpy as np
import pandas as pd
'Series使用'
# 用于返回行索引为abcde的随机数的series类型数据
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
print(s)
# 用于返回将字典类型数据强制转换为series类型的数据
d = {'a': 0., 'b': 1., 'c': 2.}
print(pd.Series(d))

'DataFrame使用'
# 用于返回将字典转换为dataframe类型的数据，行索引为abcd，列索引为one,two
d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
print(df)
# 用于返回dataframe数据类型的行索引
print(df.index)
# 用于返回dataframe数据类型的列索引
print(df.columns)