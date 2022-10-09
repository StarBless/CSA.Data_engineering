# 英文字符串，计算位数
print(len('ABC'))
# 中文字符串，计算位数
print(len('中文'))
# 查看bytes类型的数据‘ABC’的字节数
print(len(b'ABC'))
# 查看一个bytes的字节数
print(len(b'\xe4\xe8\xad\xe6\x96\x87'))
# 查看中文被‘utf-8’编码后的位数
print(len('中文'.encode('utf-8')))

# 字符串格式化输出
print('Hello, %s' % 'world')
print('Hi, %s, you have $%d.' % ('Michael', 1000000))
