# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:26:36 2020

@author: admin

topic: Arabic numerals converted to Roman numerals


"""

class Datatransfer():                                                                  #定义一个类，实现数据转换功能
    def arabTransferRome(self,arab):                                                   #在类中定义一个方法arabTransferRome
        num_list = [1000,900,500,400,100,90,50,40,10,9,5,4,1]                          #阿拉伯数字集
        str_list = ["M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"]         #罗马字符
        # 阿拉伯数字与罗马数字的对应 M = 1000 D=500 C=100 L=50 X=10 V=5 I = 1 
        res = ""
        for i in range(len(num_list)):
            while arab >= num_list[i]:
                arab -= num_list[i]
                res += str_list[i]
        return res                                                                     #实现阿拉伯数字到罗马字符的转换
    
while 1:                                                                               #实现多次运行
    a = input("请输入1-3999内的一个整数")
    while not a.isdigit():
        a = input("请输入一个整数:")                                                    #更新a值
    if int(a )< 1 or int(a)> 3999:
        print("输入的值不在指定的范围内")                                                #判断用户输入的是1-3999的一个整数
    if 1<=int(a)<=3999:                                                              
        date_example = Datatransfer()                                                  #创建一个实例data_example
        outputdata = date_example.arabTransferRome(int(a))                             #调用arabTransferRome方法
        print(f"{a}对应的罗马数字为{outputdata}")                                       #输出结果
        
        bl = input('是否想要查找其它数 yes = 1，no = 0：')                              #判断用户是否想要继续转换
        while bl != "0" and bl != "1":                   
            bl = input("请重新输入 yes = 1，no = 0 ")
        bl = int(bl)
        if not bl:
            break                                                                      #用户不想查找，结束循环

                                                                                       
                                                                                   
