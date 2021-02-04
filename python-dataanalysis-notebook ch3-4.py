#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


# tab键自动补全功能 对象内省（？） %run命令运行脚本
def f(x,y,z):
    return (x+y)/z
a=5
b=6
c=7.5
result = f(a,b,c)


# In[3]:


#pwd工作目录 %matplotlib inline设置集成(似乎不影响)
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.plot(np.random.randn(50))         #random.randn()从标准正态分布中返回样本值
plt.plot(np.random.randn(50).cumsum()) 


# In[4]:


np.random.randn(4,2)


# In[5]:


np.random.randn(10)


# In[6]:


a = 4.5; b=2
print("a is {0}, b is {1}".format(type(a),type(b)))


# In[7]:


a = 5.0
print(isinstance(a,float))           #isinstance查看对象是否为特定类型
print(type(a))


# In[8]:


a ={"anhui":10,"hebei":12,"shanghai":30}
a.values()


# In[9]:


getattr(a,"keys")


# In[10]:


c = 8/2
print(type(c))


# In[11]:


a="leozhou"
list(a)


# In[12]:


template = "{0:.2f} {1:s} are worth US${2:d}"
template.format(4.5560,"Argentine Pesos",2)   #字符串格式化


# # 第3章 内建数据结构、函数及文件

# tuple元祖存储对象

# In[13]:


tup = tuple(["foo",[1,2],True])


# In[14]:


tup


# In[15]:


tup[1][1] = 3
#tup[1] = [1,3]          #无法运行


# In[16]:


tup


# 元祖中存储的对象本身有可能是可变的，但是元组各位置上的对象是无法修改的

# In[17]:


#使用加号连接元祖生成更长的元祖
(4,None,'foo') +( 6,0)+(9,)


# 2021.1.25-1.31

# enumerate函数，返回（i,value）元祖的序列

# In[18]:


some_list = ['foo','bar','baz']
mapping = {}
for i,v in enumerate(some_list):
    mapping[v] = i
mapping
#使用enumerate构造一个字典


# In[19]:


for i,v in enumerate(some_list):
    print(f'i:{i}')
    print('v:{}'.format(v))


# In[20]:


i=5
print('i={}'.format(i))


# zip函数将列表、元祖或其他序列元素配对，新建一个元祖构成的列表

# In[21]:


seq1 = ['foo','bar','baz']
seq2 = ['one','two','three']
zipped = zip(seq1,seq2)
print(list(zipped))


# In[22]:


for i,(a,b) in enumerate(zip(seq1,seq2)):
    print('{}:{},{}'.format(i,a,b))
    print('{1}:{0},{2}'.format(i,a,b))
    print(f'{i}:{a},{b}')
#.format 0,1,2索引顺序


# In[23]:


'''zip拆分序列'''
capital_list = [('Beijing','China'),('Washionton','America'),('Berlin','Germany')]
capital,country = zip(*capital_list)
print(f'capital:{capital}')
print(f'country:{country}')


# ### 3.1.4字典常用名字是哈希表或者关联数组，是拥有灵活尺寸的键值对集合 其中键和值都是python对象

# In[24]:


empty_dict = {}
d1 = {'a':'some value','b':[1,2,3,4]}
d1[8] = 'an integer'
d1['9'] = 'a string'
d1['happy'] = 'joy'
d1


# In[25]:


8 in d1
#检查字典是否含有一个键


# In[26]:


del d1[8]
d1


# In[27]:


d1.pop('9')
d1


# In[28]:


list(d1.keys())


# In[29]:


list(d1.values())


# ##### 使用update方法将两个字典合并

# In[30]:


d1.update({'b':'foo','c':12})
d1


# ##### 字典可以接受一个2-元祖的列表作为参数

# In[31]:


mapping = dict(zip(range(5),reversed(range(5))))
mapping


# ### 3.1.5集合是一种 <u>无序且元素唯一</u> 的容器，通过set函数或者字面值集与大括号的语法定义 

# In[32]:


set([1,2,3,6,9,4,3,9])


# ##### 可以认为集合像字典，但是只有键没有值 键必须是不可变对象 

# In[33]:


{1,6,(4,7,8,7),(2,3,4)}


# In[34]:


{1,6,tuple([2,3])}


# ##### 通过hash函数可以判断一个对象是否可以哈希化 （不可变对象，可以作为字典键） 

# In[35]:


hash((1,))


# In[36]:


hash((3,5,(2,4)))
#不可变对象可以哈希化


# In[37]:


#hash([1,2])         
#报错 可变对象不能哈希化


# In[38]:


#{1,4，[1,3]}
#报错 集合的键必须为不可变对象


# In[39]:


get_ipython().run_line_magic('pinfo', 'map')


# ##### 列表、集合和字典的推导式 

# In[40]:


strings = ['a','as','bat','car','dove','python']
[x.upper() for x in strings if len(x)>2]


# In[41]:


{val : index for index,val in enumerate(strings)}


# In[42]:


{index:val for index,val in enumerate(strings)}


# ## 3.2函数 函数是python中最重要的代码组织和代码复用方式

# - 函数声明时用def关键字 返回时用return关键字 
# - 每个函数可以有位置参数和关键字参数，关键字参数常用于指定默认值或可选参数
# - 关键字参数必须跟在位置参数之后(定义时)

# In[43]:


def my_function(x,y,z=2):
    if z>1:
        return (x+y)/z
    else:
        return (x+y)*z
my_function(3,5,2)


# In[44]:


my_function(3,5,0.2)


# In[45]:


my_function(z=1,x=3,y=5)


# In[46]:


states = ['   Alabama ','Geogia!','georgia','FlOrIda',
         'south carolina##','West virginia?']
#定义函数进行数据处理
#方法一 使用内建的字符串方法，结合标准库中的正则表达式re
import re
def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()                        #删除空格
        value = re.sub('[!#?]','',value)             #移除符号
        value = value.title()                        #首字母大写
        result.append(value)
    return result
clean_strings(states)


# In[47]:


#方法二 将特定的列表操作应用到某个字符串的集合上 
'''
import re
def remove_punctuation(value):
    return re.sub('['#!?']','',value)
'''
'''
clean_ops = [str.strip,remove_punctuation,str.title]                #似乎有问题？？？
def clean_strings(strings,ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result
'''


# In[48]:


'''
def remove_punctuation(value):
    return re.sub('['#!?']','',value)
for x in map(remove_punctuation,states):
    print(x)
'''


# ##### 3.2.4 匿名函数 （Lambda函数）通过单个语句生成函数，其结果是返回值

# In[49]:


def short_function(x):
    return x*2
equi_anon = lambda x: x*2


# In[50]:


#使用匿名函数根据字符串中不同字母的数量对一个字符串集合进行排序
strings = ['foo','card','animals','aaaaa','festival']
strings.sort(key=lambda x: len(set(list(x))))
strings


# ##### 3.2.6生成器 

# - 生成器是构造新的可遍历对象的一种非常简洁的方式。如需创建生成器，只需将函数中的return关键字替换为yield。
# - 当实际调用生成器时，代码不会立即执行

# In[51]:


def squares(n=10):
    print(f'Generating squares from 1 to {n**2}')
    for i in range(1,n+1):
        yield i**2


# In[52]:


squares()


# In[53]:


for x in squares():
    print(x,end=' ')


# 用生成器表达式创建生成器更简单，创建时只需要将列表推导式中的中括号替换为小括号即可

# In[54]:


gen = (x**2 for x in range(10))
for x in gen:
    print(x,end=' ')


# 生成器表达式可以作为函数参数用于替代列表推导式

# In[55]:


sum(x**2 for x in range(100))           #使用生成器作为函数参数计算前100个自然数的平方和


# **itertools模块** 
# 
# - itertools模块是一个生成器集合
# - groupby函数可以根据任意的序列和一个函数，通过函数返回值对序列中连续元素进行分组

# In[56]:


import itertools
first_letter = lambda x:x[0]
names = ['Alan','Adam','Amy','Steven','Albert','Wes','William','Will','Rachel']
for letter,names in itertools.groupby(names,first_letter):                 #接受序列和一个函数
    print(letter,list(names))


# ##### 3.2.7错误和异常处理 

# - 想要运行失败后返回原参数，可以使用try/except代码段
# - 可以使用sys模块检查文件的默认编码
# - 当使用open来创建文件对象时，在结束操作时显示地关闭文件是非常重要的 f=open(path),f.close() 或者使用with语句 with open（path）as f
# 

# In[57]:


def attempt_float(x):
    try:
        return float(x)
    except:
        return x


# In[58]:


attempt_float(1.2345)


# In[59]:


attempt_float('Happy')


# 可以通过将多个异常类型写成元祖的方式同时捕获多个异常

# In[60]:


def attemt_float(x):
    try:
        return float(x)
    except (TypeError,VakueError):
        return x


# In[61]:


attempt_float((1,))


# In[62]:


import sys
sys.getdefaultencoding()


# # 第4章 Numpy基础：数组与向量化计算（210201-210207）

# - Numpy是python数值计算的基石，numpy在内部将数据存储在连续的内存上，numpy数组使用的内存量小于python内建序列
# - Numpy提供了一个非常易用的C语言API，这个特征可以使得python对存量C/C++/Fortran代码库进行封装，并为这些代码提供动态、易用的借口
# - Numpy的算法库使用C语言写的，在操作数据内存时，不需要任何类型检查或者其他管理操作
# - 利用数组表达式来替代显示循环的方法，称为向量化

# In[63]:


# 处理高维数组，numpy的方法比python方法快10到100倍，并且使用内存也更少
import numpy as np
my_array = np.arange(1000000)
my_list = list(range(1000000))
get_ipython().run_line_magic('time', 'for _ in range(50):my_arr2 = my_array *2')
get_ipython().run_line_magic('time', 'for _ in range(50):my_list2 = [x*2 for x in my_list]')


# ## 4.1Numpy ndarray:多维数组对象 

# numpy的核心特征之一是N-维数组对象ndarray，ndarray是Python中一个快速、灵活的大型数据集容器,也是一个通用的多维**同类**数据容器

# In[64]:


import numpy as np
data = np.random.randn(2,3)


# In[65]:


data


# In[66]:


data *10


# 每一个数组都有一个shape属性，用来表征数组每一维度的数量，都有一个dtype属性，用来描述数组的数据类型

# In[67]:


data.shape


# In[68]:


data.dtype


# ### 4.1.1 生成ndarray
# - array函数接受任意的序列型对象
# - 其他函数，例如给定形状后，zero、ones、empty、eye等函数可以创造各类型的数组
# - arange是python内建函数range的数组版

# In[69]:


data1 = [6,7.5,8,0,1]
arr1 = np.array(data1)
arr1


# In[70]:


data2 = [[1,2,3,4],[5,6,7,8]]
arr2 = np.array(data2)
data2


# In[71]:


arr2.shape


# In[72]:


arr2.ndim             #ndarray.ndim 数组的维度


# In[73]:


np.zeros(5)


# In[74]:


np.eye(4)


# In[75]:


np.ones(5)


# In[76]:


np.full(5,6)


# In[77]:


np.arange(15)


# ##### ndarray数据类型

# In[78]:


arr1 = np.array([1,2,3],dtype=np.float64)
arr1


# In[79]:


arr2 = np.array([1,2,3],dtype=np.int32)
arr2


# 可以使用astype方法显式地转换数组的数据类型

# In[80]:


arr = np.array([1,2,3,4,5])
arr.dtype


# In[81]:


float_arr = arr.astype(np.float64)
float_arr.dtype


# In[82]:


np.empty(8)


# In[83]:


np.empty(8,dtype='u4')                  #使用类型代码传入数据类型


# ### 4.1.4 基础索引与切片 
# - 数组的切片是原数组的视图，任何对于视图的修改都将反映到原数组上
# - 对于一个二维数组，每个索引值对应的元素不再是一个值，而是一个一维数组

# In[84]:


arr = np.arange(10)
arr


# In[85]:


arr[5:8] = 12
arr


# In[86]:


arr_slice = arr[5:8]
arr_slice


# In[87]:


arr_slice[1] = 132


# In[88]:


arr_slice


# In[89]:


arr


# In[90]:


arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d


# In[91]:


arr2d[2]


# In[92]:


arr2d[1:,:]


# In[93]:


arr2d[:,1:]


# In[94]:


arr3d= np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
arr3d


# In[95]:


arr3d.shape


# In[96]:


arr3d[0]


# In[97]:


arr3d[0].shape


# In[98]:


arr3d[0][1]


# In[99]:


arr3d[0][1].shape


# ##### 布尔索引 

# In[100]:


names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data = np.random.randn(7,4)


# In[101]:


data


# In[102]:


names == 'Bob'


# In[103]:


data[names == 'Bob']    #在索引数组时可以传入布尔值数组


# In[104]:


data[~(names == 'Bob')]     #~对一个通用条件取反时使用


# In[105]:


#可以使用多个布尔值条件进行联合，需要使用数学操作符，如&和|
mask = (names == 'Bob') | (names == 'Will')
mask


# In[106]:


data[mask]


# ### 4.1.6神奇索引 
# 神奇索引是numpy术语，用于描述使用整数数组进行数据索引

# In[107]:


arr = np.empty((8,4))
for i in range(8):
    arr[i] = i
arr


# In[108]:


arr[[4,3,0,5]]                    #通过传递一个包含指明所需顺序的列表或数组 选出符合特定顺序的子集


# In[109]:


arr[np.array((0,3,5,7))]


# In[110]:


arr[[-3,-4,-1,-8]]                  #使用负索引，从尾部进行选择


# In[111]:


arr = np.arange(32).reshape((8,4))
arr


# In[112]:


arr[[1,5,7,2],[0,3,1,2]]         #传递多个索引数组 选出一个一维数组（1,0），（5,3），（7,1），（2,2）被选中


# In[113]:


arr[[1,5,7,2]][:,[0,3,1,2]]       #选择矩阵中行列子集形成的矩形区域


# ### 4.1.7 数组转置和换轴 

# In[114]:


arr = np.arange(15).reshape((3,5))
arr


# In[115]:


arr.T


# In[116]:


arr = np.random.randn(6,3)
arr


# In[117]:


# 数组中矩阵相乘，用dot 对应元素相乘用*、multiply
np.dot(arr.T,arr)


# In[118]:


arr.T.dot(arr)


# In[119]:


arr * arr


# In[120]:


np.multiply(arr,arr)


# In[121]:


arr = np.arange(16).reshape((2,2,4))
arr


# In[122]:


arr.transpose((1,0,2))             #轴重新排序


# In[123]:


# swapaxes方法对轴进行调整用于重组数据 接受一对轴编号作为参数
arr


# In[124]:


arr.swapaxes(1,2)


# ## 4.2通用函数：快速的逐元数组函数 

# 通用函数，也称为ufunc，是一种在ndarray中进行逐元素操作的函数。

# In[125]:


arr = np.arange(10)
arr


# In[126]:


np.sqrt(arr)


# In[127]:


np.exp(arr)


# In[128]:


# sqrt、exp等是一元通用函数，还有add、maximum等二元通用函数
x = np.random.randn(8)
x


# In[129]:


y = np.random.randn(8)
y


# In[130]:


np.maximum(x,y)


# In[131]:


np.add(x,y)


# In[132]:


# modf函数是Python内建函数divmoid的向量化版本 它返回一个浮点值数组的小数部分和整数部分
arr = np.random.randn(7)*5
arr


# In[133]:


remainder,whole_part = np.modf(arr)
remainder


# In[134]:


whole_part


# ## 4.3利用数组进行面向数组编程 

# - numpy中的meshgrid函数可以这么理解，用两个坐标轴上的点在平面上画网格
# - X,Y = np.meshgrid(x,y)
# - X的行向量是向量x的简单复制，Y的列向量是对y的简单复制

# In[142]:


# 计算函数sqrt(x^2+y^2)
points = np.arange(-5,5,0.01)
xs,ys = np.meshgrid(points,points)              #np.meshgrid接受两个一维数组，并根据两个数组的所有（x，y）生成一个二维矩阵
xs.shape


# In[143]:


ys.shape


# In[144]:


xs


# In[145]:


ys


# In[146]:


xs == ys


# In[147]:


z = np.sqrt(xs ** 2 + ys **2)
z


# In[152]:


plt.imshow(z,cmap=plt.cm.gray);plt.colorbar()
plt.title("Image plot of $\sqrt{x^2+y^2}$ for a grid of values")


# np.where用法

# In[153]:


xarr = np.array([1.1,1.2,1.3,1.4,1.5])
yarr = np.array([2.1,2.2,2.3,2.4,2.5])
cond = np.array([True,False,True,True,False])
result = np.where(cond,xarr,yarr)                  #cond元素为True时，取xarr，否则取yarr
result


# In[156]:


arr = np.random.randn(4,4)
arr


# In[157]:


# 将所有正数替换为1，负数替换为负一
np.where(arr>0,1,-1)


# In[158]:


# 可以使用np.where将标量和数组联合
np.where(arr>0,1,arr)


# ### 4.3.2 数学和统计方法 

# In[159]:


arr = np.random.randn(5,4)


# In[160]:


arr


# In[161]:


arr.mean()


# In[163]:


arr.sum()


# In[164]:


# mean和sum可以接受一个可选参数axis 0表示列 1表示行
arr.mean(axis=0)


# In[165]:


arr.sum(axis=1)


# In[166]:


arr = np.arange(8)
arr


# In[167]:


arr.cumsum()


# In[168]:


arr.cumprod()


# In[169]:


# cumsum和cumprod这样的累积函数返回相同长度的数组，但是可以在指定轴向上进行部分聚合 
# 累加和累乘
arr0 = np.arange(9)
arr1 = arr0.reshape(3,3)
arr1


# In[170]:


arr1.cumsum(axis=1)


# In[172]:


arr1.cumprod(axis=0)


# ### 4.3.3布尔值数组的方法 

# In[175]:


arr = np.random.randn(100)
arr


# In[176]:


(arr > 0).sum()          #正值的个数


# 对于bool值数组，有两个非常有用的方法，any和all,any检查数组中是否至少有一个为True，all检查是否每个值都为True

# In[195]:


arr1 = np.arange(15)
arr2 = np.arange(15)*2


# In[196]:


arr2


# In[197]:


arr1


# In[198]:


bools = (arr1 == arr2)
bools


# In[199]:


bools.any()


# In[200]:


bools.all()


# ### 4.3.4 排序 

# In[203]:


arr = np.random.randn(6)
arr


# In[205]:


arr.sort()
arr                           #从小到大排序


# In[206]:


# 也可以在多维数组中根据传递的axis，沿着轴向对每一个一位数据段进行排序
arr = np.random.randn(5,3)
arr


# In[208]:


arr.sort(0)                 #0表示每一列从小到大排序
arr


# In[219]:


# 选出分位数对应的值
large_arr = np.random.randn(1000000)
large_arr.sort()


# In[220]:


large_arr[int(0.05*len(large_arr))]             #5%分位数


# In[221]:


large_arr[int(0.95*len(large_arr))]             #95%分位数


# ### 4.3.3 唯一值与其他集合逻辑 

# In[224]:


# np.unique,返回数组中唯一值排序后形成的数组
names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
np.unique(names)


# In[225]:


# np.unique与python实现对比
set(names)


# In[226]:


sorted(set(names))


# In[227]:


# np.in1d,可以检查1一个数组中的值是否在另一个数组中
values = np.array([6,0,0,3,2,5,6])
np.in1d(values,[2,3,6])


# ## 4.4使用数组进行文件输入和输出 

# In[230]:


# 数组在默认情况下是以未压缩的格式进行存储的，后缀名是.npy
arr = np.arange(10)
np.save('some_array',arr)          #文件存放路径没有写.npy时,后缀名会自动加上


# In[231]:


# 硬盘上的数组可以使用np.load载入
np.load('some_array.npy')


# In[232]:


# 可以使用np.savez并将数组作为参数传递给该函数，用于在未压缩文件中保存多个数组
np.savez('array_archive.npz',a=arr,b=arr*2,c=arr/2)


# In[234]:


# 当载入一个.npy文件时，你会获得一个字典型对象，并通过该对象很方便地载入单个数组
arch = np.load('array_archive.npz')


# In[235]:


arch['a']


# In[236]:


arch['c']


# ## 4.5 线性代数

# In[237]:


x = np.array([[1.,2.,3.],[4.,5.,6.]])
y = np.array([[6.,23.,],[-1.,7.],[8.,9.]])


# In[238]:


x.shape


# In[239]:


x


# In[240]:


y.shape


# In[241]:


y


# In[242]:


# x.dot(y)等价于np.dot(x,y)
np.dot(x,y)


# In[243]:


x.dot(np.ones(3))


# In[244]:


# 也可以使用@，用于矩阵乘积
x @ np.ones(3)


# In[254]:


X = np.random.randn(5,5)
X


# In[255]:


mat = X.T.dot(X)
mat


# In[250]:


from numpy.linalg import inv,qr


# In[256]:


inv(mat)


# In[257]:


mat @ inv(mat)


# In[258]:


mat.dot(inv(mat))


# In[259]:


q,r = qr(mat)                           #计算QR分解 分解成一个正交矩阵和一个上三角矩阵


# In[260]:


q


# In[262]:


q.T @ q


# In[261]:


r


# ## 4.6 伪随机数的产生

# In[264]:


samples = np.random.normal(size=(4,4))


# In[265]:


samples


# In[269]:


np.random.normal(1,3,size=(5,6))


# In[276]:


# 可以通过np.random..seed更改Numpy中的随机数种子
np.random.seed(1234)  
np.random.randn(10)


# numpy.random中的数据生成函数公用了一个全局的随机数种子，为了避免全局状态，可以使用numpy.random.RandomState生成一个随机数生成器

# In[272]:


rng = np.random.RandomState(1235)
rng.randn(10)


# In[273]:


rng = np.random.RandomState(1236)
rng.randn(10)


# In[275]:


np.random.randn(10)


# ## 4.7 示例：随机漫步 

# In[278]:


# 实现一个1000步的随机漫步
import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0,1) else -1
    position += step
    walk.append(position)
plt.plot(walk[:500])                 #对随机漫步的前500步可视化


# In[280]:


# walk只是对随机步进的累积，可以通过一个数组表达式来实现
nsteps = 1000
draws = np.random.randint(0,2,size=nsteps)
steps = np.where(draws>0,1,-1)
walk = steps.cumsum()
plt.plot(walk)


# In[281]:


walk.min()


# In[282]:


walk.max()


# In[283]:


# argmax函数可以返回布尔值数组中最大值的第一个位置
(np.abs(walk)>=10).argmax()


# In[287]:


(abs(walk)>= 6).argmax()


# In[288]:


(np.abs(walk)>=45).argmax()


# In[290]:


# 模拟多次随机漫步，比如说5000次
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0,2,size=(nwalks,nsteps))
steps = np.where(draws>0,1,-1)
walksf = steps.cumsum(1)                        #每次走1000步，模拟5000次
walksf


# In[292]:


walksf.shape


# In[293]:


walkss =steps.cumsum(0)
walkss


# In[294]:


walkss.shape                                 #每次走5000步 模拟1000次


# In[295]:


walksf.min()


# In[297]:


walkss.min()


# In[298]:


walksf.max()


# In[299]:


walkss.max()


# In[302]:


hits30 = (np.abs(walksf)>=30).any(1)                #检查最大步长绝对值大于30的行数
hits30


# In[303]:


hits30.shape


# In[304]:


hits30.sum()


# In[307]:


crossing_times = (np.abs(walkss[hits30]>=30)).argmax(1)            #使用argmax从轴向1上获取穿越时间
crossing_times


# In[308]:


crossing_times.mean()


# In[309]:


plt.plot(crossing_times)


# In[310]:


crossing_times.shape


# In[311]:


# 也可以使用其他的分布是实验随机漫步
nwalks =50
nsteps = 100
draws = np.random.normal(loc=1,scale=2,size=(nwalks,nsteps))
steps = np.where(draws>0,1,-1)
walks = steps.cumsum(1)
walks


# In[312]:


walks.min()


# In[313]:


walks.max()


# In[318]:


hits30 = (np.abs(walks)>=30).any(1)  


# In[319]:


crossing_times = (np.abs(walks[hits30]>=30)).argmax(1) 
crossing_times


# In[320]:


crossing_times.shape

