# Python内置对象“列表list”综述

本文从三个部分介绍Python的内置对象列表: 列表简介、对列表的操作以及列表和数组的异同。

## 列表简介

1. 列表是什么

在Python中，使用方括号([])来表示列表，并用逗号分隔其中的元素，下图是一个简单的列表实例：

![](https://imgkr2.cn-bj.ufileos.com/920cc520-bd42-4737-bb75-fbedeeabd62f.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=lMmpe7SZadj3AxJa4DXiMAFjuis%253D&Expires=1604307706)



2. 列表索引

在Python中，第一个列表的索引为0，而不是1。要访问列表的任何元素，都可将其位置减1，并将结果作为索引。(正向索引为0到n-1)

![](https://imgkr2.cn-bj.ufileos.com/26a452e2-381f-4c03-a127-e19042196410.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=sOz3lBhuINAtCfetYB64vULymk0%253D&Expires=1604307789)


Python可将索引指定为-1，访问最后一个列表元素。要访问列表中的其他元素，看其在列表中是倒数第i位，索引为-i。（负向索引为-n到-1）

![](https://imgkr2.cn-bj.ufileos.com/379f8d51-6f54-4897-b3d6-d26bea937eab.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=51%252BcCGuUTE90etJ3ySUJoTlWbgs%253D&Expires=1604307822)



## 对列表的操作

1. 修改、添加元素

修改元素：

首先定义一个列表，通过列表索引可以改变列表中的一个或多个值，列表中其他元素的值不变。

![](https://imgkr2.cn-bj.ufileos.com/a0923b2f-ff32-43e1-b017-42fa54307362.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=UVJwmxUniCGWHX82LDJQi2k8Vvw%253D&Expires=1604307848)


添加元素：

可以使用append（）将元素添加到列表末尾，也可以使用insert（）方法在列表任何位置添加新元素。

![](https://imgkr2.cn-bj.ufileos.com/8e01e053-54b6-4ee3-9898-34bd7aa6bb2d.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=AL%252FyEPH5PuPeq0m%252Bboby3GXAUAk%253D&Expires=1604307959)

![](https://imgkr2.cn-bj.ufileos.com/470aa395-cb10-4f31-8afb-377a2f119340.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=R0zBolq4z4RZP5gcplbtjMy64rc%253D&Expires=1604307974)

insert表示在索引处添加新元素，并将新元素值存储到这个地方。

2. 删除元素

- 使用del语句删除列表中的元素，需要知道该元素的索引，之后，该元素将不再存在于列表之中，无法对其进行访问。

![](https://imgkr2.cn-bj.ufileos.com/cc7861a5-cdd4-482e-99e9-6ee722527a86.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=0Af%252F6nnX7%252B1bO%252FnLO%252BIU9VjjWLc%253D&Expires=1604311151)

- 使用pop()可以删除列表中的任何位置元素，在括号内指定要删除的元素的索引即可，若不指定，默认删除列表中的最后一个元素。pop()方法删除元素之后，元素仍然可以被访问。

![](https://imgkr2.cn-bj.ufileos.com/e25f1ead-d37e-4f23-846e-086b937c77fc.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=%252B%252F926pHRB40Hqp7yB7zTR%252FnKYV8%253D&Expires=1604311582)

- 根据元素值删除列表，使用remove（）方法

![](https://imgkr2.cn-bj.ufileos.com/9ec98a49-86e5-4901-8fc3-9ab3aa49a46c.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=%252Fwx9DefK5EOlIu1f4G1pBwC%252F6nk%253D&Expires=1604311786)

3. 对列表进行排序

- 使用sort()对列表进行永久性排序

![](https://imgkr2.cn-bj.ufileos.com/bd29b996-f5e1-45a5-846c-f78bf81829c6.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=fF0IMMvkIlHm1dMYNH%252FEH6MoEoA%253D&Expires=1604312311)



- 使用sorted()对列表进行临时排序

![](https://imgkr2.cn-bj.ufileos.com/46b3ee6d-4567-47be-af71-b9739e7e70ff.png?UCloudPublicKey=TOKEN_8d8b72be-579a-4e83-bfd0-5f6ce1546f13&Signature=NphLB3oYDpY0UDjo%252BZfiAfBKA9k%253D&Expires=1604312337)



## 列表和数组的异同

- 相同之处

Numpy使用array对象来处理多维数组，该对象是一个快速而灵活的大数据容器。使用Python列表可以存储一维数组，通过列表的嵌套可以实现多维数组。

- 不同之处

  - Numpy是专门针对数组的操作和运算进行了设计，所以数组的存储效率和输入输出性能远优于Python中的嵌套列表，数组越大，Numpy的优势就越明显。 

  - Numpy数组中的所有元素的类型都是相同的，而Python列表中的元素类型是任意的，所以在通用性能方面Numpy数组不及Python列表，但在科学计算中，可以省掉很多循环语句，代码使用方面比Python列表简单的多。



