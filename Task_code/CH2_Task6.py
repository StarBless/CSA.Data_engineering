# 新建字典
d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
print(d['Michael'])
# 添加元素
d['Adam'] = 67
print(d['Adam'])
d['Jack'] = 90
print(d['Jack'])
# 修改某个key对应的value
d['Jack'] = 88
print(d['Jack'])
# 3种方式判断'Thomas'是否为d的一个key
print('Thomas' in d)
print(d.get('Thomas')) # 如果字典不包含该key则返回None
print(d.get('Thomas', 1)) # 如果字典不包含该key则返回-1
# 删除元素
d.pop('Bob')
print(d)