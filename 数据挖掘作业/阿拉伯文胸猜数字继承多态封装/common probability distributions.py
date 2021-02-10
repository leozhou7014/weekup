# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:01:53 2020

topic: Find the expectation, variance and extreme value of common probability distributions

"""
import numpy as np
from scipy import stats
from numpy.random import default_rng

rng = default_rng()

'''二项分布'''
BNL = rng.binomial(10, 0.4, 100)
print(f'b(10, 0.4)分布的均值为：{stats.describe(BNL).mean}')
print(f'b(10, 0.4)分布的方差为：{stats.describe(BNL).variance}')

'''均匀分布'''
uni = rng.uniform(2, 4, 100)
print(f'U(2,4)分布的均值为：{stats.describe(uni).mean}')
print(f'U(2,4)分布的方差为：{stats.describe(uni).variance}')


'''标准正态分布'''
Std_norm = rng.standard_normal(100)
print(f'标准正态分布的均值为：{stats.describe(Std_norm).mean}')
print(f'标准正态分布的方差为：{stats.describe(Std_norm).variance}')

'''正态分布'''
norm = rng.normal(9, 20, 100)
print(f'N(9, 20)分布的均值为：{stats.describe(norm).mean}')
print(f'N(9, 20)分布的方差为：{stats.describe(norm).variance}')

'''伽马分布'''
gam = rng.gamma(10, 0.4, 100)
print(f'Ga(10, 0.4)分布的均值为：{stats.describe(gam).mean}')
print(f'Ga(10, 0.4)分布的方差为：{stats.describe(gam).variance}')


'''指数分布'''
E = rng.exponential(0.4, 100)
print(f'Exp(0.4)分布的均值为：{stats.describe(E).mean}')
print(f'Exp(0.4)分布的方差为：{stats.describe(E).variance}')

