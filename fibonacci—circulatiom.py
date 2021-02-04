# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 21:03:30 2020

@author: admin
"""

#利用python中的循环输出斐波那契数列#

def fibonacci(n):
    a,b = 0,1
    for i in range(n):
        a,b = b,a+b
        print(a)

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
    print(f"前{k}项斐波那契数列为：")
    fibonacci(k)