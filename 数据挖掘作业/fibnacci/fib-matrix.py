# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:12:41 2020

@author: admin
"""

import numpy as np
Matrix = np.matrix('1 1;1 0')
# 矩阵Matrix的 n-1 次方的第一位就是斐波那契数列的解   

def Fibonacci_Matrix_tool(n):
    Matrix = np.matrix('1 1;1 0')
    return pow(Matrix, n)                  # 求矩阵的n次方 
def Fibonacci_Matrix(n):
    result_list = []
    for i in range(0, n): 
        result_list.append(np.array(Fibonacci_Matrix_tool(i))[0][0])
    return result_list                     # 生成斐波那契数列

p = input("请输入1000以内的一个正整数k,程序将会输出斐波那契数列的前k项")
while not p.isdigit():
    print("请输入一个整数!")
    break

k=int(p)
if k>1000:
    print("输入的值过大")
elif k<1:
    print("输入的值过小")
elif 1<=k<=1000:
    print(Fibonacci_Matrix(k))