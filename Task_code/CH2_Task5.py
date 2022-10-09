# 新建list并输出其长度
classmates = ['Michael', 'Bob', 'Tracy']
print(len(classmates))
# 查看列表中第一个和第二个元素
print(classmates[0])
print(classmates[1])
# 添加元素
classmates.append('Adam')
print(classmates)
# 指定位置插入元素
classmates.insert(1, 'Jack')
print(classmates)
# 删除末尾元素
classmates.pop()
print(classmates)
# 删除指定位置元素
classmates.pop(1)
print(classmates)
# 替换
classmates[1] = 'Sarah'
print(classmates)
# list中存放不同元素类型
L = ['Apple', 123, True]
print(L)
# 二维数组
s = ['python', 'java', ['asp', 'php'], 'scheme']
print(s)
print(len(s))
# 二维数组，内层数组可以用数组变量代替
p = ['asp', 'php']
s = ['python', 'java', p, 'scheme']
print(s)
# 空列表的长度为0
L = []
print(len(L))