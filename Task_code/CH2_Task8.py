# 循环打印list中的每个元素
names = ['Michael', 'Bob', 'Tracy']
for name in names:
    print(name)

# 如果age大于等于18则打印输出‘adult’
age = 20
if age >= 18:
    print('your age is', age)
    print('adult')

# 如果age大于等于18则打印输出‘adult’否则输出‘teenager’
age = 3
if age >= 18:
    print('your age is', age)
    print('adult')
else:
    print('your age is', age)
    print('teenager')

# 如果age大于等于18则打印输出‘adult’
# 如果age大于等于6则打印输出‘teenager’，否则输出‘kid’
age = 3
if age >= 18:
    print('adult')
elif age >= 6:
    print('teenager')
else:
    print('kid')