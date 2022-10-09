# 新建集合
s = set([1, 2, 3])
print(s)
# 集合会自动去重
s = set([1, 1, 2, 2, 3, 3])
print(s)
# 添加元素
s.add(4)
print(s)
# 删除元素
s.remove(4)
print(s)
# 集合的交集和并集
s1 = set([1, 2, 3])
s2 = set([2, 3, 4])
print(s1&s2)
print(s1|s2)