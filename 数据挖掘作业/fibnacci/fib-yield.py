# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:47:27 2020

@author: admin
"""

#利用python中的生成器输出斐波那契数列#

def fib(max):
    n,a,b =0,0,1
    while n < max:
        yield b            #生成器函数，利用关键字yield一次性返回一个结果#
        a,b =b,a+b
        n = n+1
    return 'done'          #给循环设置一个条件来退出循环，不然就会产生一个无限数列出来#

p = input("请输入1000以内的一个正整数k,程序将会输出斐波那契数列的前k项")
while not p.isdigit():
    print("请输入一个整数!")
    break                  #检查字符串是否只由数字组成#

k=int(p)
if k>1000:
    print("输入的值过大")
elif k<1:
    print("输入的值过小")
elif 1<=k<=1000:
    print(f"前{k}项斐波那契数列为：")

for i in fib(k):
    print(i)