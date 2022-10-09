f1 = open('test.txt', 'rb')

temp = f1.readlines()

f1.close()

f2 = open('test01.txt', 'wb')

t2 = temp[0]
t2 = t2.decode('gbk')
print(t2)
t2 = t2.encode('gbk')
# t2 = t2.decode('gbk')
print(t2)

# f2.write(t2)

f2.close()