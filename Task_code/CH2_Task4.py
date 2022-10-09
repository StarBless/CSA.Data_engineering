# 使用位置参数定义和调用函数
# 该函数返回参数x的绝对值
def my_abs(x):
    if x >= 0: # 如果x大于等于0，则返回x
        return x
    else:
        return x
# 调用my_abs
result = my_abs(-2)
print(result)


# 使用关键字参数定义和调用函数
# 该函数输出传入的所有参数
def person(name, age, **kw):
    print('name: ', name, 'age: ', age, 'other: ', kw)
# 调用person
person('Michael', 30)
person('Adam', 45, gender='M', job='Engineer')
