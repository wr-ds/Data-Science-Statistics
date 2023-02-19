# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 15:38:25 2022

@author: asus
"""

## 重新学Python，将Runoob中所教授方法从开始编程之后的通解型代码敲入
##Python字典案例题
confusion={}
confusion[1]=1
confusion["1"]=2
confusion[1]+=1
sum=0
for k in confusion:
    sum+=confusion[k]
    print(sum)
##编程第一步，将对应难点程序直接敲过来，
## while的用法
    #Fibonacci Series
a,b=0,1
while b<10:
    print(b)
    a,b=b,a+b

##end等于可以使print的输出控制在一行内   
#Fibonacci Series
a,b=0,1
while b<100:
    print(b,end=",")
    a,b=b,a+b
##while 循环加if条件控制
a=1
while a<7:
    if (a%2==0):
        print(a,"is even")
    else:
        print(a,"is odd")
    a=a+1
##布尔变量中0是false，所以需要注意
##做一个猜数字的游戏
Number=10
Guess=3
print("开始数字猜谜游戏")
while Guess!=Number :
    Guess=int(input("输入你猜的数字"))
    if Guess<Number:
        print("数字猜小了")
    elif Guess>Number:
        print("数字猜大了")
    else: 
        print("恭喜你猜对了")
##if条件控制也可以嵌套
## Python3 循环语句
##Python 中的循环有 for 和 while
##输出小于100的所有奇数
a=1
while a<100:
    print(a,end=",")
    a+=2
## print数值可以用%d来写
##1到100的求和计算
sum=0
counter=1
n=100
while counter<=n:
    sum=sum+counter
    counter+=1
print("1到%d的和为%d"%(n,sum))
##面对无限循环时使用Ctrl+C可以退出
##while循环也可以加else，当else之后不会再进入当前while循环
b=1
while b<5:
    print(b,"小于5")
    b=b+1
else:
    print(b,"大于等于5")        
## Python简单语句组
while 1:
    print("新手程序员")
##请按ctrl+c
##for循环
##for 循环一般格式
##for x in   seq
#    statement1
#    else:
#        statement2
languages=["C","C++","Python"]
for x in languages:
    print(x)    
sites=["Baidu","Taobao","Runoob"]
##利用break自己做程序
for site in sites:
    if site=="Runoob":
        print("Runoob菜鸟教程")
    else:
        print(site,"循环数据")
else:
    print("没有循环数据")
print("循环结束")
##下面这个是else和for的联动，上面的else和if的联动，else和if的很好理解，else和for在后面体会更深
##for循环本质上也利用了布尔，对于在序列中的执行，else为不在序列中的执行
##利用len 和 range 做遍历的索引
a=["语文","数学","英语"]
for i in range(len(a)):
    print(i,a[i])
##在for和while循环中，break强制执行进入循环末尾，continue强制执行回到循环开端，优先级大于循环内其他语句
##补一个continue案例，直接跳过一次print
n=5
while n>0:
    n-=1
    if n==2:
        continue
    print(n)
print("我把2跳过了")
##循环字符串，碰到o跳过输出
for i in "Runoob":
    if i=="o":
        continue
    print(i)
print("我把o跳过了")
##找出100以内的质数（开始比较难，挑战一下）
##这里突出一个for循环中else的理解，for循环的else是实际循环时候for后面指定的数不在序列中执行
for q in range(2,100):
    for x in range(2,q):
        if q%x==0:
            print(q,"等于",x,"乘",q/x)
            break
    else:
        print(q,"是质数")
##上面遍历质数的程序中的else对应的是小的for循环所以的数都不能整除，因此执行是质数的命令
##进行变体，不光找出质数，还遍历所有乘法可能的组合
for q in range(2,10):
    for x in range(2,q):
        if q%x==0:
            print(q,"等于",x,"乘",q/x)
    else:
        print(q,"是质数")
##循环理解非常深刻了，可以进行下一步学习
##if之后弱想要不符合这个条件进行操作，后面还是需要跟进一个else
for letter in 'Runoob': 
    if letter == 'o':
      pass
      print ('执行 pass 块')
    print ('当前字母 :', letter)
print ("Good bye!")
##Python需要非常注意语句前面的空格，同样的语句，位置不对会导致逻辑的变化

##Python迭代器与生成器
##iter(),对于迭代器需要逐步，迭代器只能往前不能后退
##迭代器中的循环
import sys
diedai=[1,2,3,4]
it=iter(diedai)
while True:
    try:
        print(next(it))
    except StopIteration:
        sys.exit
##注意这里的True需要大写
##创建迭代器需要集合面向对象编程，这部分在学面向对象时再学
##是用来yield的函数被成为生成器，生成器本质上也是一种迭代器
##斐波那契数列，两组数为一组
import sys
def febonnaci(n):
    a,b,counter=0,1,1
    while n>=counter:
        yield(a)
        counter+=1
        a,b=b,a+b
f=febonnaci(10)
while True:
    try:
        print(next(f),end=",")
    except StopIteration:
        sys.exit
##不用迭代器
import sys
def febonnaci(n):
    a,b,counter=0,1,1
    while n>=counter:
        print(a,end=",")
        counter+=1
        a,b=b,a+b
f=febonnaci(10)
##本来生成一列数据的，可以通过迭代器（生成器），配合next的循环只生成一个，节省内存
##在leetcode刷题的过程中也很有可能会遇到优化解法的题，对应继续体会迭代器的用法
##Python 函数
##一般用def进行定义，定义过后的函数后续调用会很方便
##计算面积函数
def area(x,y):
    return x*y
width=12
length=8
print("width=",width,"length=",length,"area=",area(width,length))
##在def函数时需要注意要携带return，return的结果才是函数对应的结果
##关键字参数不需要注意顺序，只要指示明确就可以
##在定义函数时可以使用默认函数，即如果不在对应位置富裕对应的输入值，全部按照默认来取
#不定长参数则是在def函数的时候没有命名，*tuple，在后续输入的时候，
##所有未定义的变量自动构成一个元组，然后再继续输出
##由于处理成了元组，可以配合循环来进行遍历输出
##双星号情况，处理成字典，会带着键和值
##定义函数时可以单独使用星号，星号后的定义的参数必须用关键字传入
##lamda可以构建匿名函数，相比内联函数可以增加运行效率
su=lambda a,b: a+b
print(su(15,56))
##匿名函数也可以进行封装，这样可以用同样的代码构建不同的封包
def myfunc(n):
    return lambda a:a*n
dou=myfunc(2)
tri=myfunc(3)
##这里的dou和tri也都是函数，定义一个函数也可以做到返回一个函数，之前的例子返回的都是字符串或者表达式
##这里返回的是匿名函数
dou(3)
tri(3)

# 可写函数说明
def sum( arg1, arg2 ):
   # 返回2个参数的和."
   total = arg1 + arg2
   print ("函数内 : ", total)
   return total
 
# 调用sum函数
tot = sum( 10, 20 )
print ("函数外 : ", tot)
##tot和total可以看出来，total其实是在函数内部自己设定的，在函数中有print这一样所以输出函数内
##tot是原来函数内计算的total由于return的关系被赋值，与函数里的total其实不是一个概念
##可以看出只要调用了定义的函数，就会跟着进行函数内部的操作，但是最后获得的东西其实就是return后面跟着的
#3.8多了一个定义，/之前必须是指定位置参数，不能带=，*之后是关键字形参，必须带=
##数据结构
##列表可以修改，字符串和元组不能修改
##以下只是示范操作用法，并没有赋值,使用时提前定义好列表，列表名称替换了list即可
list.append(x)
list.extend(L)
list.insert(i,x)
list.remove()
list.pop()
list.index()
list.clear()
list.count()
list.sort()
list.reverse()
list.copy()
#注意remove只能移除第一个为remove里的元素的元素，后面相同的元素移除不了
##sort是从小到大排序，pop需要指定索引，如果没有索引就删除最后一个元素了
##列表方法使得列表可以很方便的作为一个堆栈来使用，堆栈作为特定的数据结构，最先进入的元素最后一个被释放（后进先出）
##也可以把列表当做队列用，只是在队列里第一加入的元素，第一个取出来
##这时可以用popleft，从最左边开始移除元素
##列表推导式也可以应用在二维推导式中for in 连续两个即可
##注意矩阵的表示形式
matrix=[
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        ]
##矩阵转置,开始训练对矩阵的理解
[[row[i] for row in matrix ]for i in range(4)]
transposed=[]
for i in range(4):
    transposed.append([row[i]for row in matrix])
transposed
#犯了错误，需要注意append是序列的情况下，要加[],也要注意row[]，本质上是list而不是函数，需要注意
##目测跟上面方法一致，当做练习
##自己写一遍果然发现了区别，区别是一个column做好了清空原始后才进行下一个
transposd1=[]
for i in range(4):
    transposd2=[]
    for row in matrix:
        transposd2.append(row[i])
        ##这里需要缩进四个格子，因为要等到2中的序列完全填好了才加，后面语句进入第二个for循环会报错
        ##这也是细节，此处的append不需要[]，是因为此处的row[i]就对应了一个数值，不是list所以不需要[]
    transposd1.append(transposd2)
transposd1
##del语句
#del删除时只能根据index删除，不能根据具体的值，与pop之类的有区别
##元组在输入时可以不带(),但是输出一定要有，正常情况还是需要带的
##集合，要用{}，但是如果利用set(),则不用{}，因为set是函数，函数同意对应使用()
##集合与数学上的定义一直，无序且不重复
##集合运算
a-b#在a不在b
a|b#在a或者在b，等于a并b
a&b#在a且在b，等于a交b
a^b#在a或者在b，但是不能同时存在a和b之中，等于a并b减去a交b
##数据结构
##字典也是一样，只是把index换成键而已
# dict() 直接从键值对元组列表中构建字典。如果有固定的模式，列表推导式指定特定的键值对：
##遍历技巧
##字典的遍历，用items
ab={"w":"ang","r":"ui"}
for a,b in ab.items():
    print(a,b)    
##序列的遍历,如果需要二维则第一个参数对应的是index，用enumerate
for a,b in enumerate(["av","sv","xv"]):
    print(a,b)
##同时遍历两个序列，可以用zip,用format来代替里面内容时，需要注意里面空白地方要用{数字从0开始对应位置}
questions=["name","age"]
answers=["rw","22"]
for a,b in zip(questions,answers):
    print("What is your {0}? It is {1}".format(a,b))
##遍历的时候也可以利用reverse倒序和利用sort排序
basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
for f in sorted(basket):
    print(f)
##此时basket是序列，如果需要去重，则再用一下set把序列改成集合即可
basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
for f in sorted(set(basket)):
    print(f)  
##Python 模块
##import调用某一模块，之后可以使用模块.函数对模块里的函数进行调用
import sys
for i in sys.argv:
    print("i")
print("Python路径为",sys.path)
##程序没出错，因为sys.argv在此处得到的是空
##import语句
##from import可以不导入整个模块，只导入几个函数，import*是导入模块里面的所有函数
##一般不推荐使用import*,有可能会把以前有点变量重新复制产生错误
##_name_=_main_时才是程序自身在运行，否则是其他模块在运行
##dir()函数，可以将模块内所有定义的名称列出来，如果（）内没有参数，则会列出当前定义的所有名称
##这里dirt还有问题
a = [1, 2, 3, 4, 5]
import fibo,sys
fib = fibo.fib
dir() # 得到一个当前模块中定义的属性列表
a = 5 # 建立一个新的变量 'a'
dir()
del a # 删除变量名a
dir()
#Python中有一些标准模块，是前人做好的，直接调用即可，省事
##不同包的导入方式不一样，一般用from import import是模块比较好解决，直接.函数就使用了
##如果要导入一个包里的多个子模块，需要吧_all_进行赋值，在import*，只会导入_all_中的所有模块
##使用 from Package import specific_submodule 这种方法永远不会有错
##输入和输出
##str显示是表现出读者可读的形式，repr是显示出解释器可读的形式，但都是把变量转换成字符串
##在运行的时候二者是一回事，都转换成了字符串
h="runoob\nrunoob"
print(h)
print(repr(h))
##repr在处理转义运算符不按照转义进行计算
#rjust为让字符串向右排列，左边加空格,rjust(空格的个数)
##print函数不是简单输入就完事了，里面其实是有参数的print("{位置:宽度d}".format(d))
##输出平方与立方表
for x in range(1,11):
    print(repr(x).rjust(2),repr(x*x).rjust(3),repr(x*x*x).rjust(4))
##空格数也是根据计算结果写的位数，都是有根据的
##第二种方法，{}中第一位数字是对应后面format里面的赋值
for x in range(1,11):
    print("{0:2d} {1:3d} {2:4d}".format(x,x*x,x*x*x))
##zfill是在数字左边补0
"2.01".zfill(5)
##str.format()可以将括号里的内容赋值到左边{},{}里面还可以有位置信息和宽度
print("{}网站：{}".format("菜鸟","www.runoob.com"))
##格式化字符串还可以利用[]里面的键，来表示键值
##读和写文件可以用open
open(filename,mode)
##open读文件时注意mode，不同mode用法有区别，记得去查图，默认为只读
##r为只读，w为只写不读，不看file直接从头改，a为只写不读，不看file接着原始的往后改
##又写又读的为r+，w+，a+；w+是从头写，r+为指针放在开头，a+为续写，w+与r+的区别是w+若不存在文件会建立新文件，强烈的w愿望，而r+不会重新建立
##处理文本文件时，使用with很好用，不会再出现关闭报错
##with open("文件","w") as f:
   #操作即可
   #f.closed
##pickle 模块
##通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存到文件中去，永久存储。
#通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象
##pickle保存数据不需要把数字转换成字符串，直接存起来；读写文件时要用rb或者wb，bytes类型
##pickle是将数据序列化，人为不可读，但是可以永久保存，protocal是bytes的形式，用dump将其序列化并保存
## pickle.load(file)，从文件中读取pickle类型数据并转化成python可读类型
##open也是open一个file
# Pickle dictionary using protocol 0.
#pickle.dump(data1, output)
# Pickle the list using the highest protocol available.
#pickle.dump(selfref_list, output, -1)
##都序列化到output中，转化类型不一样，转化后的为新类型，需要再用pickle重构成python类型
##pickle.load从指针位置开始往后读取file的一个并重构成python类型，每load一次，指针往后走一个
##file
##使用 open() 方法一定要保证关闭文件对象，即调用 close() 方法，这点与R是一致的，也可以先读取完毕载打开
##open的mode默认为“r”，在读写file时，要注意写好路径，还有一个方式是t,t是文本模式
##file对象
##file.close
##file.flush  刷新文件内部缓冲   
##file.fileno  返回一个整型的文件描述符   
##file.isatty 如果是文件连接到一个终端设备则返回true
##file.next 返回文件下一行
#file.read([size])读取指定字节数
#file.readline([size])读取指定字节数的行数
#file.readline([sizeint]) 读取所有行并返回列表，返回总和大约为sizeint的行
#file.seek(offset[,whence])移动文件读取指针到指定位置
#file.tell 返回文件当前位置
#file.truncate([size]) 从文件的首行首字符开始阶段，阶段文件为size个字符 
##file.write(str) 返回的是字符串的长度
##file.writelines 向文件写入一个序列字符串列表
##OS模块，非常有用，设置路径就靠它，需要先import
##os.access(path,mode) 检查权限模式
##os.chdir 设定路径，读取数据非常有用
##os.chflags(path,flags) 设置路径的标记为数字标记
##os.chmod(path,mod) 改变权限
##os.chown(path,uid,gid) 改变所有者
##os.chroot(path)  改变根目录
##os.close(fd) 关闭文件描述符fd
##os,closerange(fd_low,fd_high) 关闭从fd_low到fd_high的所有文件描述符fd
##os.dup(fd) 复制文件描述符
##os.dup(fd,fd2) 复制文件描述符fd到fd2
##os.fchdir(fd) 通过文件描述符改变当前工作目录
##os.fchmod(fd, mode) 改变一个文件的访问权限，该文件由参数fd指定，参数mode是Unix下的文件访问权限。
##os.fchown(fd, uid, gid) 修改一个文件的所有权，这个函数修改一个文件的用户ID和用户组ID，该文件由文件描述符fd指定。
##os.fdatasync(fd)  强制将文件写入磁盘，该文件由文件描述符fd指定，但是不强制更新文件的状态信息。   
##os.fdopen(fd[, mode[, bufsize]])  通过文件描述符 fd 创建一个文件对象，并返回这个文件对象
##os.fpathconf(fd, name) 返回一个打开的文件的系统配置信息。name为检索的系统配置的值，它也许是一个定义系统值的字符串，这些名字在很多标准中指定（POSIX.1, Unix 95, Unix 98, 和其它）。
##os.fstat(fd)  返回文件描述符fd的状态，像stat()。
##os.fstatvfs(fd)  返回包含文件描述符fd的文件的文件系统的信息，Python 3.3 相等于 statvfs()。
##os.fsync(fd)  强制将文件描述符为fd的文件写入硬盘。
##os.ftruncate(fd, length)  裁剪文件描述符fd对应的文件, 所以它最大不能超过文件大小。
##os.getcwd()  返回当前工作目录
##os.getcwdb()  返回一个当前工作目录的Unicode对象
##os.isatty(fd)  如果文件描述符fd是打开的，同时与tty(-like)设备相连，则返回true, 否则False。
##os.lchflags(path, flags)  设置路径的标记为数字标记，类似 chflags()，但是没有软链接
##os.lchmod(path, mode)  修改连接文件权限
##os.lchown(path, uid, gid)  更改文件所有者，类似 chown，但是不追踪链接。
##os.link(src, dst)  创建硬链接，名为参数 dst，指向参数 src
##os.listdir(path)  返回path指定的文件夹包含的文件或文件夹的名字的列表。
##os.lseek(fd, pos, how)  设置文件描述符 fd当前位置为pos, how方式修改: SEEK_SET 或者 0 设置从文件开始的计算的pos; SEEK_CUR或者 1 则从当前位置计算; os.SEEK_END或者2则从文件尾部开始. 在unix，Windows中有效
##os.lstat(path)  像stat(),但是没有软链接
##os.major(device)  从原始的设备号中提取设备major号码 (使用stat中的st_dev或者st_rdev field)。
##os.makedev(major, minor)  以major和minor设备号组成一个原始设备号
##os.makedirs(path[, mode])  递归文件夹创建函数。像mkdir(), 但创建的所有intermediate-level文件夹需要包含子文件夹。
##os.minor(device)  从原始的设备号中提取设备minor号码 (使用stat中的st_dev或者st_rdev field )。
##os.mkdir(path[, mode])  以数字mode的mode创建一个名为path的文件夹.默认的 mode 是 0777 (八进制)。
##os.mkfifo(path[, mode])  创建命名管道，mode 为数字，默认为 0666 (八进制)
##os.mknod(filename[, mode=0600, device])  创建一个名为filename文件系统节点（文件，设备特别文件或者命名pipe）。
##os.open(file, flags[, mode])  打开一个文件，并且设置需要的打开选项，mode参数是可选的
##os.openpty()  打开一个新的伪终端对。返回 pty 和 tty的文件描述符。
##os.pathconf(path, name)  返回相关文件的系统配置信息。
##os.pipe()  创建一个管道. 返回一对文件描述符(r, w) 分别为读和写
##os.popen(command[, mode[, bufsize]])  从一个 command 打开一个管道
##os.read(fd, n)  从文件描述符 fd 中读取最多 n 个字节，返回包含读取字节的字符串，文件描述符 fd对应文件已达到结尾, 返回一个空字符串。
##os.readlink(path)  返回软链接所指向的文件
##os.remove(path)  删除路径为path的文件。如果path 是一个文件夹，将抛出OSError; 查看下面的rmdir()删除一个 directory。
##os.removedirs(path)  递归删除目录。
##os.rename(src, dst)  重命名文件或目录，从 src 到 dst
##os.renames(old, new)  递归地对目录进行更名，也可以对文件进行更名。
##os.rmdir(path)  删除path指定的空目录，如果目录非空，则抛出一个OSError异常。
##os.stat(path)  获取path指定的路径的信息，功能等同于C API中的stat()系统调用。
##os.stat_float_times([newvalue])  决定stat_result是否以float对象显示时间戳
##os.statvfs(path)  获取指定路径的文件系统统计信息
##os.symlink(src, dst)  创建一个软链接
##os.tcgetpgrp(fd) 返回与终端fd（一个由os.open()返回的打开的文件描述符）关联的进程组
##os.tcsetpgrp(fd, pg)  设置与终端fd（一个由os.open()返回的打开的文件描述符）关联的进程组为pg。
##os.ttyname(fd)  返回一个字符串，它表示与文件描述符fd 关联的终端设备。如果fd 没有与终端设备关联，则引发一个异常。
##os.unlink(path)  删除文件路径
##os.utime(path, times)  返回指定的path文件的访问和修改的时间。
##os.walk(top[, topdown=True[, onerror=None[, followlinks=False]]])  输出在文件夹中的文件名通过在树中游走，向上或者向下。
##os.write(fd, str)  写入字符串到文件描述符 fd中. 返回实际写入的字符串长度
##os.path 模块  获取文件的属性信息。
##os.pardir()  获取当前目录的父目录，以字符串形式显示目录名。
## Python3 错误和异常
## 发生错误坐左边的词其实已经表现出了错误原因
##try和except的构建可以试错
##while True:
##    try:
##        {}
##    except(什么error):(根据error类型执行except之后对应的语句，except可以有多个)
##try 尝试执行代码；except 为对应报错执行；如果不报错就接else了，执行else里的语句；finally是不管报不报错都一定会往后执行
##raise 可以出发异常
x=10
if x>5:
    raise Exception("x的值大于5，x的值是{}".format(x))    
##练习
a=1
while a<50:
    if a%7==0:
        print(a,"*")
    else:
        print(a)
    a=a+1
##类可以封装对象，同一个类可以对应不同的对象进行封装
##自己定义异常类
class MyError(Exception):
    def __init__(self,value):
        self.value=value
    def __str__(self):
        return repr(self.value)
try:
    raise MyError(25)
except MyError as E:
    print("My Exception ocuured,value:",E.value)
    
##下面为网站的代码，可以运行，上面为第一次自己写，出现报错，定义类没写好
class MyError(Exception):
    def __init__(self, value):
            self.value = value
    def __str__(self):
            return repr(self.value)
   
try:
    raise MyError(2*2)
except MyError as ee:
    print('My exception occurred, value:', ee.value)
##注意__init__的用法规范，只有前后都是两个_才是正确的使用方法
##Python环境比较完善，内置变量都是双_
##定义清理行为
##try 后面一定执行，except和else二选一，根据try是否有异常，异常是什么来判断执行哪个对应语句，Finally无论如何都会执行，如果try里有报错，先执行完finally里面的语句之后再报错
##用 with是好习惯，因为用with表示使用完之后正确执行对应的清理方法，因此即使在后续的处理过程中出问题了，文件因为及时关闭了而不会受到影响
## Python3 面向对象
##定义的类需要先实例化，找到具体对象，之后才能进一步调动
class jiandan:
    i=123
    def f(self):
        return "hello word"
x=jiandan()
print(x.i)
print(x.f())

##类有一个名为 __init__() 的特殊方法（构造方法），该方法在类实例化时会自动调用：
class lianxiinit:
    def __init__(self,a,b):
        self.shu=a
        self.zhi=b
x=lianxiinit(3,5)
print(x.shu,x.zhi)

##类的方法
class people:
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def speak(self):
        print("%s今年%d岁"%(self.name,self.age))
p=people("王睿",23)
p.speak()
##注意print的位置用法
##类的继承
##可以新建一个类，直接调用已经做好的类
class jicheng(people):
    def __init__(self,name,age,grade):
        people.__init__(self,name,age)
        self.grade=grade
    def speak(self):
        print("%s今年%d岁,在上大学%d年级。"%(self.name,self.age,self.grade))
f=jicheng("wr",23,4)
f.speak()
##bug点：注意调用时要加（），引用时候要带原始类加.
##需要注意圆括号中父类的顺序，若是父类中有相同的方法名，而在子类使用时未指定，python从左至右搜索 即方法在子类中未找到时，从左到右查找父类中是否包含方法。
##多重继承时，看父类的位置，从继承的位置看，在左边的先继承，之后从左往右依次进行，直到最后若为出现则报错
##super函数可以调用已经被掩盖的父类函数
class Parent:        # 定义父类
   def myMethod(self):
      print ('调用父类方法')
 
class Child(Parent): # 定义子类
   def myMethod(self):
      print ('调用子类方法')
 
c = Child()          # 子类实例
c.myMethod()         # 子类调用重写方法
super(Child,c).myMethod() #用子类对象调用父类已被覆盖的方法
##__private_method：两个下划线开头，声明该方法为私有方法，只能在类的内部调用 ，不能在类的外部调用。
##同样，外部不能调用内部私有方法
##类的专有运算，最常用就是__init__
#__init__ : 构造函数，在生成对象时调用
#__del__ : 析构函数，释放对象时使用
#__repr__ : 打印，转换
#__setitem__ : 按照索引赋值
#__getitem__: 按照索引获取值
#__len__: 获得长度
#__cmp__: 比较运算
#__call__: 函数调用
#__add__: 加运算
#__sub__: 减运算
#__mul__: 乘运算
#__truediv__: 除运算
#__mod__: 求余运算
#__pow__: 乘方
##Python运算符重载
##直接用print加不行，必须在类的内部提前定于好，用类专属的函数来解决
class Vector:
   def __init__(self, a, b):
      self.a = a
      self.b = b
 
   def __str__(self):
      return 'Vector (%d, %d)' % (self.a, self.b)
   
   def __add__(self,other):
      return Vector(self.a + other.a, self.b + other.b)
 
v1 = Vector(2,10)
v2 = Vector(5,-2)
print (v1 + v2)
##不是很常用和实用，等需要用了再回来看
##Python3 命名空间和作用域
##正常编写的是全局名称，def里面的是局部名称，Python自带的函数是内嵌名称，内嵌最广
##三个命名空间是相互独立的，即使名字一致也互不影响，引用时从小往上查，都没有返回error
## Python3作用域
##L（Local）：最内层，包含局部变量，比如一个函数/方法内部。
##E（Enclosing）：包含了非局部(non-local)也非全局(non-global)的变量。比如两个嵌套函数，一个函数（或类） A 里面又包含了一个函数 B ，那么对于 B 中的名称来说 A 中的作用域就为 nonlocal。
##G（Global）：当前脚本的最外层，比如当前模块的全局变量。
##B（Built-in）： 包含了内建的变量/关键字等，最后被搜索。
##规则顺序： L –> E –> G –> B。
##在局部找不到，便会去局部外的局部找（例如闭包），再找不到就会去全局找，再者去内置中找
##如果将 msg 定义在函数中，则它就是局部变量，外部不能访问：
##def里面的和外面的可以重名，不会相互影响，因为一个是局部的，一个是全局的
##当内部作用域想要修改外部作用域时，需要提前使用global或者nonlocal进行限制，即可修改
a = 10
def test(b):
    b = b + 1
    print(b)
test(a)
## Python3标准库概述
import os
os.getcwd()
##改变工作路径的时候从Editor上看就可以
os.chdir('d:\\实习数据')
dir(os)
##针对日常的文件和目录管理任务，:mod:shutil 模块提供了一个易于使用的高级接口:
import shutil
##glob模块提供了一个函数用于从目录通配符搜索中生成文件列表:
import glob
glob.glob("*.py")
##字符串修改
"tea for too".replace("too","two")
##数学
import math
math.cos(math.pi)
#随机数
import random
random.sample(range(50),3)
##访问互联网，处理邮件时可以使用，个人觉得不实用，如果以后需要再来补>>> 
from urllib.request import urlopen 
for line in urlopen('http://tycho.usno.navy.mil/cgi-bin/timer.pl'):
    line = line.decode('utf-8')  # Decoding the binary data to text.
    if 'EST' in line or 'EDT' in line:  # look for Eastern Time
print(line)
import smtplib
server = smtplib.SMTP('localhost')
server.sendmail('soothsayer@example.org', 'jcaesar@example.org',
"""To: jcaesar@example.org
From: soothsayer@example.org
 Beware the Ides of March.
 """)
server.quit()
##日期和时间,可以转化为时序变量，更加方便
from datetime import date
now=date.today()
birthday=date(1999,7,12)
days=now-birthday
##可以对数据包进行压缩，zlib.compress为压缩，zlib.decompress为解压缩。
##性能度量，比较重要，leetcode刷题也要非常注重性能，刷题时保证自己的代码用时短
from timeit import Timer
Timer("t5=0;t5=a;c=t5","a=1").timeit()
Timer("c=a","a=1").timeit()
##注意代码换行不要用，用的是；
##测试模块
def average(values):
    """
    >>> print(average([20, 30, 70]))
    40.0
    """
    return sum(values) / len(values)

import doctest
doctest.testmod()   # 自动验证嵌入测试
##测试是在定义函数内部标注一个完整的输出正确的结果，doctest再跟着去测试
##unittest可以测试类
import unittest

class TestStatisticalFunctions(unittest.TestCase):
    def test_average(self):
        self.assertEqual(average([20, 30, 70]), 40.0)
        self.assertEqual(round(average([1, 5, 7]), 1), 4.3)        
unittest.main() # Calling from the command line invokes all tests
##一般用doctest即可，也比较简单


######Python3实例，终于开始应用， 做好每一个例子
##Python Hello word实例
print("Hello word!")
##Python 数字之和实例
num1=input("输出第一个数字：")
num2=input("输出第二个数字：")
sum=float(num1)+float(num2)
print("{0}和{1}的和是{2}".format(num1,num2,sum))

##简化程序,%相比传统format可以锁定公式
print("两数之和为%.2f"%(float(input("输入一个数字："))+float(input("输入另一个数字："))))

## Python 平方根实例
num=float(input("输入一个数字："))
sqrtnum=num**0.5
print("%.3f的平方根是%.3f"%(num,sqrtnum))

##另一种解法，适用于全体实数，不局限于整数
import cmath
num=float(input("输入一个数字："))
num_sqrt=cmath.sqrt(num)
print("{0}的平方根是{1:.3f}+{2:.3f}j".format(num,num_sqrt.real,num_sqrt.imag))

##Python 二次方程
import cmath
a=float(input("输入一个数字："))
b=float(input("输入另一个数字："))
c=float(input("再输入一个数字："))
d=cmath.sqrt(b**2-4*a*c)
x1=(-b+d)/(2*a)
x2=(-b-d)/(2*a)
print("结果为：{0:.2f}和{1:.2f}".format(x1,x2))

##计算三角形的面积
##利用了海伦公式
##海伦公式为（p=(a+b+c)/2）  S=sqrt[p(p-a)(p-b)(p-c)]
import cmath
a=float(input("输入一个数字："))
b=float(input("输入另一个数字："))
c=float(input("再输入一个数字："))
p=(a+b+c)/2
s=cmath.sqrt(p*(p-a)*(p-b)*(p-c))
print("三角形的面积是{0:.0f}".format(s.real))

##计算圆的面积
def circularea(r):
    Pi=3.14
    return Pi*r*r
r=float(input("输入圆的半径："))
print("圆的面积是{0:.2f}".format(circularea(r)))

##随机数生成
import random
random.randint(0,99)

##摄氏温度转华氏温度
cel=float(input("输入一个摄氏温度："))
fa=cel*1.8+32
print("转换后的华氏温度是{0:.1f}".format(fa))

## Python3交换变量
x=input("输入一个x值：")
y=input("输入一个y值：")
x,y=y,x
print("转换后x的值为{0}".format(x))
print("转换后y的值为{0}".format(y))

## if 语句
x=float(input("输入一个值："))
if x>0:
    print("x是正数")
elif x==0:
    print("x是0")
else:
    print("x是负数")
## if里的循环嵌套
x=float(input("输入一个值："))
if x>=0:
    if x>0:
        print("x是正数")
    else:
        print("x是0")
else:
    print("x是负数")

##判断字符串是不是数字
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (ValueError,TypeError):
        pass
    return False
print(is_number('foo'))   # False
print(is_number('1'))     # True
print(is_number('1.3'))   # True
print(is_number('-1.37')) # True
print(is_number('1e3'))   #True
# 测试 Unicode
# 阿拉伯语 5
print(is_number('٥'))  # True
# 泰语 2
print(is_number('๒'))  # True
# 中文数字
print(is_number('四')) # True
# 版权号
print(is_number('©'))  # False
##注意标识时单词首字母要大写
##unicodedata的包，ASCii编码

##判断字符串是不是数字,尝试有无float的区别
def is_number(s):
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (ValueError,TypeError):
        pass
    return False
print(is_number('foo'))   # False
print(is_number('1'))     # True
print(is_number('1.3'))   # True
print(is_number('-1.37')) # True
print(is_number('1e3'))   #True
# 测试 Unicode
# 阿拉伯语 5
print(is_number('٥'))  # True
# 泰语 2
print(is_number('๒'))  # True
# 中文数字
print(is_number('四')) # True
# 版权号
print(is_number('©'))  # False
##发现区别，只用unicodedata的numeric转换不了float，都是是针对int型的，所以需要先测试float

##判断奇数偶数
pd=int(input("输入一个数字："))
if pd%2==0:
    print("{0}是偶数".format(pd))
if pd%2==1:
    print("{0}是奇数".format(pd))

##闰年的计算方法
##注意对问题的解析，闰年不一定是被4整除就一定是闰年，如果被100整除了，那么必须要能整除400才是闰年
nian=int(input("输入一个年份："))

def runpd(nian):
    if nian%4==0:
        if nian%100==0:
            if nian%400==0:
                print("{0}是闰年".format(nian))
            else:
                print("{0}不是闰年".format(nian))
        else:
            print("{0}是闰年".format(nian))
    else:
        print("{0}不是闰年".format(nian))

runpd(nian)
## Python 获取最大值函数
print(max(1,2))
print(max("a","b"))
print(max([1,2]))
print(max((1,2)))

##Python质数判断
def is_zs(num):
    for i in range(2,num):
        if num%i==0:
            print("{0}不是质数".format(num))
            break
    else:
        print("{0}是质数".format(num))
num=int(input("输入一个数字："))
is_zs(num)

##Python输出制定范围内的素数
upper=int(input("输入一个较大的数："))
lower=int(input("输入一个较小的数："))

for i in range(lower,upper+1):
    for j in range(2,i):
        if i%j==0:
            break
    else:
        print("{0}是素数".format(i))

##Python 阶乘实例
num=int(input("输入一个数去计算阶乘："))
def jiecheng(num):
    s=1
    if num<0:
        print("负数没有阶乘。")
    elif num==0:
        print("0的阶乘是1。")
    else:
        for i in range(1,num+1):
            s=s*i
        return s
jiecheng(num)

##九九乘法表
##\t表示空4个字符
for i in range(1,10):
    for j in range(1,i+1):
        print("{0}x{1}={2}\t".format(i,j,i*j),end="")
    print(end="\n")
##\n表示换行，print（）里面默认的参数就是\n，因此可以直接在每个j循环结束用print（），表示后面下一轮可以换行了
##斐波那契数列
term=int(input("你需要几项:"))
def fibonaci(n):
    a=0
    b=1
    if n<1:
        print("请输入一个整数")
    elif n==1:
        print("斐波那契数列：")
        print(a)
    else:
        print("斐波那契数列：")
        print(a,end=",")
        for i in range(1,n):
            print(b,end=",")
            a,b=b,a+b
fibonaci(term) 
##判断是否为阿姆斯特朗数
##如果一个n为正整数数的等于其各位数字的n次方之和，则该数为阿姆斯特朗数
num=int(input("输入你需要判断的阿姆斯特朗数："))
def is_amstl(num):
    n=len(str(num))
    amstl=0
    for i in (1,n+1):
        amstl=amstl+(num%(10*(n+1-i))*i
    if num==amstl:
        print("输入的数为阿姆斯特朗数。")
    else:
        print("输入的数不是阿姆斯特朗数。")
is_amstl(num)
##尝试是否if的判断是否必须是具体数值
num=int(input("输入你需要判断的阿姆斯特朗数："))
n=len(str(num))
amstl=0
pd=num
while num>0:
    digit=num%10
    amstl=amstl+digit**n
    num=num//10
def is_amstl(num):
    if pd==amstl:
        print("输入的数为阿姆斯特朗数。")
    else:
        print("输入的数不是阿姆斯特朗数。")
is_amstl(pd)
##只是之前的代码逻辑错误，重新排查还是ok的，可以打包。
num=int(input("输入你需要判断的阿姆斯特朗数："))
def is_amstl(num):
    n=len(str(num))
    amstl=0
    pd=num
    while pd>0:
        digit=pd%10
        amstl=amstl+digit**n
        pd=pd//10
    if num==amstl:
        print("输入的数为阿姆斯特朗数。")
    else:
        print("输入的数不是阿姆斯特朗数。")
is_amstl(num)

##把数字转换成二进制，八进制，十六进制
dec = int(input("输入数字："))
print("十进制数为：", dec)
print("转换为二进制为：", bin(dec))
print("转换为八进制为：", oct(dec))
print("转换为十六进制为：", hex(dec))

##ASCII码与字符相互互换
##ASCII码与实际的字符是有一一对应关系的，ASCII码对应数字
c = input("请输入一个字符: ")
 
# 用户输入ASCII码，并将输入的数字转为整型
a = int(input("请输入一个ASCII码: "))
 
 
print( c + " 的ASCII 码为", ord(c))
print( a , " 对应的字符为", chr(a))

##最大公约数算法
a=int(input("输入第一个数："))
b=int(input("输入第二个数："))
def zdgys(a,b):
    m=min(a,b)
    while m>0:
        if a%m==0:
            if b%m==0:
                print("最大公约数是{0}".format(m))
                break
        m=m-1
zdgys(a,b)

##最小公倍数算法
a=int(input("输入第一个数："))
b=int(input("输入第二个数："))
def zxgbs(a,b):
    for i in range(max(a,b),a*b+1):
        if (i%a==0) and (i%b==0):
            print("{0}和{1}的最小公倍数是{2}".format(a,b,i))
            break
zxgbs(a,b)

##简单计算器实现
def add(x,y):
    return x+y
def substract(x,y):
    return x-y
def multiply(x,y):
    return x*y
def divide(x,y):
    return x/y
choice=input("输入加/减/乘/除：")
num1=int(input("输入第一个数字："))
num2=int(input("输入第二个数字："))
if choice=="加":
    print(add(num1,num2))
elif choice=="减": 
    print(substract(num1,num2))
elif choice=="乘": 
    print(multiply(num1,num2))
else: 
    print(divide(num1,num2))

##Python生成日历
##引入日历模块
import calendar
yy=int(input("输入一个年份："))
mm=int(input("输入一个月份："))
print(calendar.month(yy,mm))

##使用递归生成斐波那契数列
def fibonaci(n):
    if n <=1:
        return n
    else:
        return(fibonaci(n-1)+fibonaci(n-2))
n=int(input("输入你需要多少项："))
if n<=0:
    print("请输入一个正数")
else:
    num=0
    while num<n:
        print(fibonaci(num),end=",")
        num=num+1
##文件IO
##需要查看下如何读取文件
with open("text.txt","wt") as out_file:
    out_file.write("写文本！")
with open("text.txt","r") as in_file:
    text=in_file.read()
print(text)
##以上操作是如何读写文本，注意read也要先open，open之后赋值read即可

##测试判断字符串
print("测试实例一")
str = "runoob.com"
print(str.isalnum()) # 判断所有字符都是数字或者字母
print(str.isalpha()) # 判断所有字符都是字母
print(str.isdigit()) # 判断所有字符都是数字
print(str.islower()) # 判断所有字符都是小写
print(str.isupper()) # 判断所有字符都是大写
print(str.istitle()) # 判断所有单词都是首字母大写，像标题
print(str.isspace()) # 判断所有字符都是空白字符、\t、\n、\r

print("------------------------")

# 测试实例二
print("测试实例二")
str = "runoob"
print(str.isalnum())
print(str.isalpha())
print(str.isdigit())
print(str.islower())
print(str.isupper())
print(str.istitle())
print(str.isspace())
str[1]
str[1].upper()

##没有意义且会报错，因为字符串会重新生成新的，直接赋值不了
def strdx(str):
    i=0
    new_str=""
    while i<len(str):
        if str[i].isupper()==True:
            new_str[i]=str[i]
        if str[i].islower()==True:
            new_str[i]=str[i].upper()
        i=i+1
    return new_str
strdx(str)
##本质上是重写了upper函数，而且没写对

##字符串大小写转换
str="RuNooB"
str.upper()
str.capitalize()
str.lower()
str.title()

##计算每个月天数
##直接用calendar包即可
import calendar
monthrange=calendar.monthrange(2022,3)
print(monthrange)

##查看昨天日期
##引入datetime模块,即可查看日期
import datetime
def getyesterday():
    yesterday=datetime.date.today()+datetime.timedelta(-1)
    return yesterday
getyesterday()
##Python list 常规操作
## list 定义
li=["a","b","c","d","e"]
li[2]
##list 索引
li[0:3]
li[-2]
li[1:-1]
## list增加元素
li.append("f")
li
li.insert(2,"B")
li.extend(["E","F"])
##list 搜索
"c" in li
"C" in li
li.index("B")
##list 删除元素
li.remove("B")
li.pop()
li
##list 运算符
li=li+["E","F"]
li+=["g"]
li=li*2
li
##join可以把list转换为字符串
";".join(li)
para={"a":"A","b":"B"}
lil=["{0}={1}".format(i,j) for i,j in para.items()]
lil
##list中的for循环
##同时可以利用split把字符串分割为list
s=";".join(li)
s
a=s.split(";")
a
a2=s.split(";",3)
a2
##list的映射解析
##字典才用item，序列直接排就好了
li2=[i*2 for i in lil]
li2
##字典中的解析
pa=[v for k,v in para.items()]
pa
pap=[k for k,v in para.items()]
pap
lil=["{0}={1}".format(i,j) for i,j in para.items()]
lil
##list过滤
a=[i for i in li if i.isupper()==True]
a
##让li中只出现一次的，重复的丢掉
b=[i for i in li if li.count(i)==1]
b
##由于现在的li是之前的list*2，所以产生空集才是正常的

##约瑟夫生者死者小游戏
##30 个人在一条船上，超载，需要 15 人下船。

##于是人们排成一队，排队的位置即为他们的编号。

##报数，从 1 开始，数到 9 的人下船。

##如此循环，直到船上仅剩 15 人为止，问都有哪些编号的人下船了呢？
##设计位置，考虑用字典写程序，可以标明位置
##考虑到用三维解决，报数占一个维度，排队占一个维度，再用一个维度判断是否下船
##注意分析需求，如果维度在较小情况下难以实现考虑升维
people={}
for i in range(1,31):
    people[i]=1
print(people)
i=1
j=0
check=1
while i<=31:
    if i==31:
        i=1
    if j==15:
        break
    if people[i]==0:
        i=i+1
        continue
    if check%9==0:
        j=j+1
        print("{0}号下船了".format(i))
        people[i]=0
        i=i+1
        check=check+1
    else:
        i=i+1
        check=check+1
##成功做对，注意对问题的解析

##Python五人分鱼
##A、B、C、D、E 五人在某天夜里合伙去捕鱼，到第二天凌晨时都疲惫不堪，于是各自找地方睡觉。
##日上三杆，A 第一个醒来，他将鱼分为五份，把多余的一条鱼扔掉，拿走自己的一份。

##B 第二个醒来，也将鱼分为五份，把多余的一条鱼扔掉拿走自己的一份。 

##C、D、E依次醒来，也按同样的方法拿鱼。

##问他们至少捕了多少条鱼?
num=0
while True:
    num=num+1
    if num%5==1:
        bcde=4*(num-1)/5
        if bcde%5==1:
            cde=4*(bcde-1)/5
            if cde%5==1:
                de=4*(cde-1)/5
                if de%5==1:
                    e=4*(de-1)/5
                    if e%5==1:
                        print("至少有{0}条鱼".format(num))
                        break
##虽然做对了，但是看看答案怎么做的
## break是跳出循环，因此对应为自己那一拨的循环进行跳出
##答案的循环手法直接是整个重复取变量的操作编入循环，会更快
fish=0
while True:
    fish=fish+1
    total=fish
    enough=True
    for _ in range(5):
        if (total-1)%5==0:
            total=4*(total-1)//5
        else:
            enough=False
            break
    if enough:
        print("至少有{0}条鱼".format(fish))
        break
##ctrl+c终止运行，无限循环时报错就会停不下来
##无限报错时如果每次报的错误不一样，那就是逻辑设计存在问题，再仔细过一遍流程。

##Python实现秒表功能
##感觉基本不在业务范围内，主要学思路
import time
print("按下回车开始计时，按下ctrl+c停止计时")
while True:
    input("")
    starttime=time.time()
    print("开始")
    try:
        while True:
            print("计时：",round(time.time()-starttime,0),"s",end="\r")
            time.sleep(2)
    except KeyboardInterrupt:
        print("结束")
        endtime=time.time()
        print("总共用时",round(endtime-starttime,0),"s")
        break
##time.sleep()表示暂停多长时间，因此在循环中需要暂停几秒再循环，可以用来控制循环次数
##计算公式 13 + 23 + 33 + 43 + …….+ n3

#实现要求：

#输入 : n = 5

#输出 : 225

#公式 : 13 + 23 + 33 + 43 + 53 = 225


#输入 : n = 7

#输入 : 784

#公式 : 13 + 23 + 33 + 43 + 53 + 63 + 73 = 784
n=int(input("输入你需要计算多少项的立方和："))
def lfh(n):
    sum=0
    for i in range(1,n+1):
        sum=sum+i**3
    return sum
lfh(n)

##定义一个整型数组，并计算元素之和。

#实现要求：

#输入 : arr[] = {1, 2, 3}

#输出 : 6

#计算: 1 + 2 + 3 = 6
def _sum(arr):
    return(sum(arr))

arr={1,2,3,4}
_sum(arr)    

##(ar[], d, n) 将长度为 n 的 数组 arr 的前面 d 个元素翻转到数组尾部。

def fz(ar,d):
    n=len(ar)
    i=0
    while i<d:
        new_ar[i+n-d]=ar[i]
        i=i+1
    while i<n:
        new_ar[i-d]=ar[i]
        i=i+1
    return(new_ar)
ar=[1,2,3,4,5,6,7]
fz(ar,2)      
##这种循环中没办法直接出数组，原例子出的是print，没有直接做出数组
##最优办法是直接用列表翻转
def fz(ar,d):
    ar=ar[d:]+ar[:d]
    return ar

##再做一遍最初始解法,数组就是数组，也可以用【】，题目里会说明,注意列表就要用列表的方法，不像数可以随意循环
def fz(ar,d):
    n=len(ar)
    new_ar=[]
    for i in range(d,n):
        new_ar.append(ar[i])
    for j in range(0,d):
        new_ar.append(ar[j])
    return(new_ar)

##将列表中的头尾两个元素对调
def dd(arr):
    n=len(arr)
    new_arr=[]
    new_arr.append(arr[n-1])
    for i in range(1,n-1):
        new_arr.append(arr[i])
    new_arr.append(arr[0])
    return new_arr
arr=[1,2,3,4,5]
dd(arr)
##新的做法，如果既想保存原始的，又想用新的可以简单生成全1数组，只要个数规定对，替换对应位置的数即可
##遇到不像数可以随意循环的数组，可以多尝试设定来进行解决
def swdd(newarr):
    newarr[-1],newarr[0]=newarr[0],newarr[-1]
    return newarr
arr=[1,2,3,4,5,6]
swdd(arr)

##将列表中的指定位置的两个元素对调
##定义一个列表，并将列表中的指定位置的两个元素对调。

##例如，对调第一个和第三个元素：
def zddd(arr,post1,post2):
    arr[post1-1],arr[post2-1]=arr[post2-1],arr[post1-1]
    return arr
list=[1,2,3,4,5,6]
post1=3
post2=6
zddd(list,post1,post2)

##定义一个列表，并将它翻转。
def fzlb(arr):
    n=len(arr)
    list=[]
    for i in range(1,n+1):
        list.append(arr[-i])
    return list
arr=[1,2,3,4,5,6]

##简单写法
arr.reverse()
##这个reverse是直接把原序列翻转了，运行一次之后arr就已经直接变了
arr

##Python判断元素是否在列表中存在
def test_ele(arr,ele):
    pd=True
    for i in arr:
        if i==ele:
            print("该元素存在")
            pd=False
    if pd:
        print("该元素不存在")
arr=["wang","rui","23"]
ele="wang"
test_ele(arr,ele)
set(arr)
##set为创建无序不重复数集

##清空列表
##用clear实现
list2=[1,2,3,4]
print("清空前：",list)
list2.clear()
print("清空后：",list)

##移除列表中重复的元素
list1=[1,2,3,4,5,2,3,4]
##set为创建无序不重复数集
print(list(set(list1)))
##注意定义变量名称时尽量不要把变量的名称设计成和外环境内置函数名称一样，会引起误会

##移除两个列表中同时存在的元素
##可以利用^  ^表示两个集合的对称差，即互相独有部分的合
list1=[1,2,3,4,3,4]
list2=[3,4,5,6,4,6]
print(list(set(list1)^set(list2)))

##复制列表
##多种方法，但其实没啥太大用
##list,append等
##extend是直接加一个新列表，append是就加元素
def clone_list(li):
    new_list=li[0:]
    return new_list
lil=[1,2,3]
clone_list(lil)

def clone2(li):
    new_list=[]
    new_list.append(lil)
    return new_list
clone2(lil)

def clone3(li):
    new_list=list(li)
    return new_list
clone3(lil)


##计算元素在列表中出现的次数
def lico(list,i):
    return list.count(i)
list=[1,2,3,4,5,6,1,2,1,2]
lico(list,2)
lico(list,4)

##计算列表元素之和
def lisum(list):
    return sum(list)
list=[1,2,3,4]
lisum(list)

def lissum(list):
    sum=0
    for i in list:
        sum=sum+i
    return sum
lissum(list)

##计算列表元素之积
def sup(list):
    supl=1
    for i in list:
        supl=supl*i
    return supl
arr=[1,2,3,5]
sup(arr)

##查找列表中最小元素
def mi(list):
    return min(list)
arr=[2,4,5,6,7]
mi(arr)
##自己定义
def atmi(list):
    pd=list[0]
    for i in list:
        if i<=pd:
            pd=i
    return pd
atmi(arr)    
ar=[5,3,6,8,1]
atmi(ar)
##也可以用sort排序，之后选取第一个位置的元素
##查找列表最大元素
max(arr)
##sort排序查找
ar.sort()
n=len(ar)
ar[n-1]
##更简单办法
ar[-1]
##移除字符串中指定位置的字符
old_arr="runoob"
##replace的第三个参数是用来限定替换的数量的
new_arr=old_arr.replace(old_arr[3],"",1)
new_arr

##第二种方法
new_ar=old_arr[:2]+old_arr[3:]
new_ar
##剔除对应位置的字符定义函数
def rem(array,n):
    new_array=array[:n-1]+array[n:]
    return new_array
rem(old_arr,3)

#判断字符串是否存在子字符串
string="www.runoob.com"
sub_string="runoob"
##可以使用find,但是直接直接判断更简单
if sub_string in string:
    print("存在")
else:
    print("不存在")
##用find来完成,find到为-1则表示不存在
def ch(string,sub_string):
    if string.find(sub_string)==-1:
        print("不存在")
    else:
        print("存在")
ch(string,sub_string)
ch(string,"runrun")
##此实例拓展了逻辑思维，不光是一般的布尔和判断是否相等，直接用in也是逻辑严谨的包不包含

#判断字符串长度
len(string)        
##使用循环来解
def lenth(string):
    count=0
    for i in string:
        count=count+1
    return count
lenth(string)

##Python使用正则表达式提取字符串中URL
##基本算法中没啥用，后期如果需要再回来补
##这个网页的设计格式，知道有这个操作就行，函数里面查找格式现用现查
import re 
  
def Find(string): 
    # findall() 查找匹配正则表达式的字符串
    url = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', string)
    return url 
##核心思想其实是查找，re包中的findall，但是utrls的写法与普通的查找存在区别，所以需要查找对应写法      
 
string = 'Runoob 的网页地址为：https://www.runoob.com，Google 的网页地址为：https://www.google.com'
print("Urls: ", Find(string))

##将字符串作为代码执行
##使用exec()函数进行
##有时候测验中需要字符串可能会有用
strdm="print(3*5)"
exec(strdm)
##其实没啥大用

##字符串翻转
##reduce用来迭代，记住思想
from functools import reduce
string="123456"
print(reduce(lambda x, y: y+ x ,string))
##lambda记得前后都有空格，空出来否则会出现 not defined
##函数定义的简单写法
print("".join(reversed(string)))
##直接reversed出不来，还需要再建一个空字符串之后join才可以读出，不然只会告诉你reverse之后的位置
string[::-1]
##相比于一个冒号的，两个冒号会出现第三个参数，即步长，步长决定了怎么读的
##对字符串切片及翻转
##给定一个字符串，从头部或尾部截取指定数量的字符串，然后将其翻转拼接。
##理解错题意，这个翻转拼接是指大位置翻转，内容保持不变
def rotate_(string,d):
    n=len(string)
    le=string[d:]+string[:d]
    ri=string[n-d:]+string[:n-d]
    print("头部翻转：",le)
    print("尾部翻转：",ri)

rotate_(string,4)

##如果里面位置也需要翻转则利用上衣实例的翻转即可，高层次有迭代配合隐式，速成有reversed和【：：-1】，但是直接套需要先截取
##题目已经做对，回去之后再看一眼笔记即可 
import random
random.random()
list1=[1,2,3]

##模拟计算余响
##定义余响伤害计算函数，以一次e的15次伤害为基准
def yx():
    import random
    import numpy as np
    damage_list=[]
    for i in range(1,16):
        p_initial=0.36
        lunci=0
        for j in range(1,6):
            lunci=lunci+1
            if random.random()<=p_initial:
                break                                
            else:
                p_initial=p_initial+0.2
        damage_list.append(float((lunci+0.7)/lunci)) 
    return np.mean(damage_list)
#检查函数是否准确
yx()
##模拟10词e，进行展示
for i in range(1,11):
    print(round(yx(),2),end=",")
    
##Python按键或者值给字典排序
##按照键排序，正常sorted就是按照键来的
# 声明字典
key_value ={}     
 
# 初始化
key_value[2] = 56       
key_value[1] = 2 
key_value[5] = 12 
key_value[4] = 24
key_value[6] = 18      
key_value[3] = 323 
key_value
##sorted 是对所有可迭代的对象进行操作，由于字典中的值不可迭代因此只对键进行了排序    
sorted(key_value)    
for i in sorted(key_value):
    print((i,key_value[i]),end="")
##按照值排序
key_value.items()
sorted(key_value.items())
##直接sorted还是会把分类按照key分
##该实例本质上对sorted的高级运用，第二个参数为key，结合lambda自定义可迭代数据的顺序对象基本为可迭代数据中的每一个元素，第三个参数为是否倒置输出
##lambda 后面再定义基本都是冒号
sorted(key_value.items(),key=lambda kv:(kv[1],kv[0]))
##这里面的key并不是直接操作，而是先按照这个隐式进行变化，排序之后再进行复原，注意隐式的写法即可，这里面的的隐式不是上面那种迭代
##这里并不涉及之后的连续操作，lambda后面只跟了一个对象，说明这个定义是在一个对象内部的，当设计多个对象时，迭代的概率比较到，需要用到reduce
##正常sorted里面的key只是对应一个排序，因此sorted里的情况大部分是lambda后面只跟一个对象，在后面是对象的操作手法，按照变换之后的操作手法排序之后会自动复原

##字典列表排序
lis = [{ "name" : "Taobao", "age" : 100},  
{ "name" : "Runoob", "age" : 7 }, 
{ "name" : "Google", "age" : 100 }, 
{ "name" : "Wiki" , "age" : 200 }] 
# 通过 age 升序排序
##锁定按照键值锁定，因为key是可以迭代的，value不可以
sorted(lis, key=lambda l:l["age"])
# 先按 age 排序，再按 name 排序
##遇到多维可以像坐标一样，括号进行处理
sorted(lis, key=lambda l:(l["age"],l["name"]))
# 按 age 降序排序
sorted(lis, key=lambda l:l["age"], reverse=True)

##计算字典值之和
key_value
sum=0
for i in sorted(key_value):
    sum=sum+key_value[i]
print(sum)
##代码量最少
print(sum(key_value.values()))

##利用隐函数
from functools import reduce
print(reduce(lambda x,y: x+y,key_value.values()))

##移除字典键值对
##直接利用键删除即可
key_value
del key_value[2]
key_value
##字典的处理方式基本上都是通过键来完成，主要利用的键可以迭代但是值不可以，并且键与值是一一对应的
##使用delete删除没有的key会报错

##另一种方法
##使用pop移除
a=key_value.pop(1)
a#a为pop删除的值
key_value
#pop不光可以删除指定的键值组，还可以将对应删除的键值组提取出来
##使用pop删除没有的key，不会报错，且会提醒

##自己直接重新设置一个即可，不利用函数，循环手动解决
new_keyvalue={k:v for k,v in key_value.items() if k!=5}
new_keyvalue

##合并字典
##使用update参数
up={2:3,3:4}
up.update(new_keyvalue)
up
#两个字典合在一起了，如果两个字典中有的key是一样的则会更新到合成之后的字典中

##第二种方法
##利用**将参数以字典的形式导入
upi={1:2}
upp={**up,**upi}
upp
uppi={up,upi}##直接运行会报错，因为字典是不可哈希的，列表和字典由于可变所以不可哈希，无法直接写到字典中，因此需要利用**

##给定一个字符串的时间，将其转换为时间戳。
##不涉及在算法中，属于现查现会的知识，长时间不用也很有可能会忘
##时间包
##对于字符串，首先要利用time包的strptime将时间转化为结构化时间，这样机器才能进行处理
import time
a1 = "2019-5-10 23:40:00"
# 先转换为时间数组
timeArray = time.strptime(a1, "%Y-%m-%d %H:%M:%S")
 
# 转换为时间戳
timeStamp = int(time.mktime(timeArray))
print(timeStamp)
##时间戳不是传统意义上的时间，属于区块链技术，相比传统时间更加可以证明什么时间发生什么事，无人可以篡改，因此更具有公信力
##广泛应用于区块链技术，如果需要设计区块链的知识再回来补充 

##strf是将传字符串转变为可识别类型，strp则是将可识别类型的结构化时间数据再换成正常字符串
##这种来回的变化主要是为了改变时间的表达形式 
# 格式转换 - 转为 /
a2 = "2019/5/10 23:40:00"
# 先转换为时间数组,然后转换为其他格式，最后其实只是换了表示形式，还是string
timeArray = time.strptime(a2, "%Y/%m/%d %H:%M:%S")
otherStyleTime = time.strftime("%Y/%m/%d %H:%M:%S", timeArray)
print(otherStyleTime)
a2==otherStyleTime
#可以看到，机器自动生成的时间不会保证每个位置都有两位（前面不需要补0）

##获取前几天的时间
import datetime
import time
def qjt(n):
    da=datetime.datetime.now()-datetime.timedelta(n)
    return da

qjt(3)    
timeStamp=int(time.mktime(qjt(3).timetuple()))
##转化为时间戳需要先变成tuple
print(qjt(3).strftime("%Y/%m/%d %H:%M:%S"))


##将时间戳转化为指定日期格式
import time
import datetime
now=datetime.datetime.now()
timeStamp=int(time.mktime(now.timetuple()))
dateArray = datetime.datetime.utcfromtimestamp(timeStamp)
otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
print(otherStyleTime)
c="d"
c.lower()
##打印自己的字体
##没什么特别，为了省事只定义几个
##换行符不是/，是\
def zt(string1):
    n=len(string1)
    for i in range(0,n):
        c=0
        c=string1[i].lower()
        if c=="r":
            print("""RRR---\nRR----""")
        elif c=="o":
            print("""--oo---
o----""",end="\n")
        elif c=="u":
            print("""--UUUU---
--UUU--""",end="\n")
        elif c=="n":
             print("""--NNN---
--N--""",end="\n")
        elif c=="b":
            print("""--BBB---
--BBBB--""",end="\n")
        else:
            print("还没有对这个字母进行定义")
string1="RUNOOB"
zt(string1)    
zt("rbu")
zt("seu")
##如果遇到额外的空行，则说明程序不够智能自己识别不出来，要用转义语句进行插入达到目的

##Python 二分查找
###二分搜索是一种在有序数组中查找某一特定元素的搜索算法。搜索过程从数组的中间元素开始，如果中间元素正好是要查找的元素，则搜索过程结束；如果某一特定元素大于或者小于中间元素，则在数组大于或小于中间元素的那一半中查找，而且跟开始一样从中间元素开始比较。如果在某一步骤数组为空，则代表找不到。这种搜索算法每一次比较都使搜索范围缩小一半。
##直接找显然也可以找到，但是属于暴力破解法，考虑时间复杂度和空间复杂度的情况下显然不是最优解
int(4.5)
int(4.8)
##int属于向下取整
### 返回 x 在 arr 中的索引，如果不存在返回 -1
##二分查找的前提：有序序列
##不能无穷写下去，在思路保持一致的情况下，考虑运用递归
##目前情况下能够做到比较好的解决方法就是判断，循环和递归
##递归在定义函数时记得把参数的位置留出来，好进行下一次递归
arr=[1,2,4,5,6,7,8,13,15,17]
def binarysearch(arr,l,r,x):
    if (r-l)>=0:
        mid=int((l+r)/2)
        if x==arr[mid]:
            return mid
        elif x>arr[mid]:
            return binarysearch(arr,mid+1,r,x)
        else:
            return binarysearch(arr,l,mid-1,x)
    else:
        print("找不到{0}的位置".format(x))
binarysearch(arr,0,len(arr)-1,15) 
binarysearch(arr,0,len(arr)-1,9)            
binarysearch(arr,0,len(arr)-1,1)       
##优化了题目给的代码

##线性查找
##线性查找指按一定的顺序检查数组中每一个元素，直到找到所要寻找的特定值为止。
##非常普通，就是正常查找的逻辑，一般不采用，因为效率比较低
##在序列中找出元素对应的位置(索引)，若不存在则返回不存在
list1=[1,3,4,6,14,7,9,13,40]
def xxse(li,x):
    n=len(li)
    select=True
    for i in range(0,n):
        if x==li[i]:
            select=False
            return i
            break
    if select:
        print("找不到这个元素。")
xxse(list1,5)
xxse(list1,6)
xxse(list1,13)           

##插入排序（英语：Insertion Sort）是一种简单直观的排序算法。它的工作原理是通过构建有序序列，
##对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。
list1=[2,5,9,3,6,8,1,14,16]
def insertsort(li):
    n=len(li)
    for i in range(1,n):
        key=li[i]##先锁死每个元素，元素卡主了往前排
        j=i-1
        while j>=0 and key<li[j]:
            li[j+1]=li[j]
            j=j-1
        li[j+1]=key
    return li
insertsort(list1)
##逐个比较，两个循环要解析出来，首先是每个元素都要排，之后是每个对应的元素锁死之后要和之前的元素开始比较，对应循环条件
##对应个数在进行循环，注意题目条件对应的解析，操作对应了代码过程
##列表需要连排，至少有元素对应的一个全列表
##总结归纳的方法，看了答案提示自己也可以做出来

##快速排序
##快速排序使用分治法（Divide and conquer）策略来把一个序列（list）分为较小和较大的2个子序列，然后递归地排序两个子序列。

#步骤为：

#挑选基准值：从数列中挑出一个元素，称为"基准"（pivot）;
#分割：重新排序数列，所有比基准值小的元素摆放在基准前面，所有比基准值大的元素摆在基准后面（与基准值相等的数可以到任何一边）。在这个分割结束之后，对基准值的排序就已经完成;
#递归排序子序列：递归地将小于基准值元素的子序列和大于基准值元素的子序列排序。
#递归到最底部的判断条件是数列的大小是零或一，此时该数列显然已经有序。

##lc的中等难度题了，不是不需要算法的普通题，需要先学习算法与数据结构的知识
##代码变刷题变学即可，最终刷题多少其实也是提升个人实力
##额外的数据结构的题，不是普通的list，变成了链表，需要补数据结构和算法的知识才行
##链表不是普通的列表，是需要递归进行跳转才可以的，时间复杂度也很大，以下代码是按照列表算的，所以提交会出问题
##和数据量有关，不是常数操作
##class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        n1=len(l1)
        n2=len(l2)
        sum1=0
        sum2=0
        list1=[]
        for i in range(0,n1):
            sum1=sum1+l1[i]*10**i
        for j in range(0,n2):
            sum2=sum2+l2[j]*10**j
        su=sum1+sum2
        sumd=repr(su)
        return [sumd[len(su)-k] for k in range(0,len(su))]

#选取基准值有数种具体方法，此选取方法对排序的时间性能有决定性影响。
##涉及到复杂度了，复杂度先知道是选取中间位置的元素的复杂度最低，之后在学数据结构与算法之后再注意
list1=[2,5,9,3,6,8,1,14,16]

def dac(arr):
    n=len(arr)
    pi=arr[int(n-1/2)]
    for i in range(0,n):
        if arr[i]<pi:

            
            
        