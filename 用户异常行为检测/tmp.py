arg = input("请输入用户编号和操作序列编号 (Usage: UserNumber BashListNumber)：");
arg=list(arg.split())
print ("你输入的内容是: ", arg)
for i in range(0,25):

    print('\033[1;34;45m1 \033[0m',end='')

for i in range(0,25):
    print('\033[1;34;45m异\033[0m',end='')