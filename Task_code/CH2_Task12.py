### 代码块2-3 ###
### numpy库的使用教程 ###
import numpy as np
'random函数使用'
# 用于返回指定size=1的随机数
print(np.random.random(size=1))
# 用于返回指定size=2的随机数
print(np.random.random(size=2))
# 用于生成[-5, 0]的3*2的随机数组
print(5*np.random.random_sample((3, 2))-5)

'array函数使用'
# 用于返回1行3列的整型数组
print(np.array([1, 2, 3]))
# 用于返回1行3列的浮点型数组
print(np.array([1, 2, 3.0]))
# 用于从numpy的矩阵子类返回2行2列的数组
print(np.array(np.mat('1 2; 3 4')))

'arange函数使用'
# 用于返回0到3的整型数组(不包含3)
print(np.arange(3))
# 用于返回3到7的浮点数数组(不包含7.0)
print(np.arange(3, 7.0))
# 用于返回3到7的间隔为2的数组(不包含7)
print(np.arange(3, 7, 2))