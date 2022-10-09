# 生成一个元素从1到10的列表
print(list(range(1, 11)))
# 生成一个列表，其元素为1到10各个数的平方
print([x*x for x in range(1, 11)])
# 生成一个列表，其元素为1到10中的偶数的平方
print([x*x for x in range(1, 11) if x%2==0])
# 同时遍历连个字符串的字符，结合起来作为列表的元素
print([m+n for m in 'ABC' for n in 'XYZ'])