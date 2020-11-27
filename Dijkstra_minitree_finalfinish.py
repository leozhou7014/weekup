# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:02:07 2020

topic: 使用Dijkstra方法求一个无向图G的最小树

"""


Inf = float("inf")                                                             #使用Inf表示正无穷，简化表示方法                      
graph_weight = [[Inf,50,30,40,25],
                [50,Inf,15,20,Inf],
                [30,15,Inf,10,20],
                [40,20,10,Inf,10],
                [25,Inf,20,10,Inf]]                                            #使用一个二维列表表示图G的权矩阵

underline_weight = []                                                          #划线部分元素组成的列表  

"""
最终要输出的两个列表：最小树各边权、最小树各顶点   
"""         
      
mini_weight = []                                                               #利用列表存储最小树各边的权重
mini_index = []                                                                #利用列表存储最小树的各顶点

n = 0
for i in range(5):
    if graph_weight[0][i] == Inf:
        n += 1
if n == 5:
    print("图G中不含有最小树")                                                  #若权矩阵第一行最小值为Inf，则图G不存在最小树
else:
    mini_index.append(1)                                                       #若权矩阵第一行元素不全为Inf，最小树顶点列表添加元素1                                                     
                                                                               
def func(n):
    if n %4 == 0:
        return(4)
    else:
        return(n%4)                                                            #func函数的功能是实现underline_weight索引到图G权矩阵（列表graph_weight）的列索引的转化
                                                                               #最后的索引为0,1,2,3,4   

p = 4                                                                          #外层循环一共四次
k = 0                                                                          #循环中graph_weight第k列赋值Inf，并将其第k行元素添加到underline_list                                                                    
while(p):
    for j in range(5):
        graph_weight[j][k]= Inf                                                #第k列元素赋值为Inf
        underline_weight.append(graph_weight[k][j])                            #将graph_weight第k行元素添加到underline_list  
    if min(underline_weight) == Inf:
            print('图G中不含有最小树')                                          #若划线部分元素最小值为Inf,则图G中不存在最小树   
   
    mini_weight.append(min(underline_weight))                                  #将当前underline_weight的最小值添加到mini_weight列表中
    mini_index.append(func(underline_weight.index(min(underline_weight)))+1)   #将undrline_weight最小值的索引对应到图矩阵的列索引，列索引加一即为顶点编号
    k = func(underline_weight.index(min(underline_weight)))                    #k等于当前under_weight最小值在图矩阵中的列索引
    underline_weight.remove(min(underline_weight))                             #去除underline_weight中当前最小值
    p = p-1

print("最小树中各边的权为:")
print(mini_weight)
print("最小树中各顶点为:")
print(mini_index)                                                              #输出最小树的各边的权和顶点

print("划线部分元素为")
print(underline_weight) 
print("最终的图矩阵")
print(graph_weight)                                                            #验证：划线部分元素列表和最后的图矩阵




        
        
        
        
    
    

    
                           




