#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
from scipy import stats
from numpy.random import default_rng

RNG = default_rng()

#二项分布 均值、方差、极差

bnl = RNG.binomial(10,0.2,1000)
print(stats.describe(bnl))
print(f"服从二项分布B（10,0.2）的1000个样本的均值为{stats.describe(bnl).mean}")
print(f"服从二项分布B（10,0.2）的1000个样本的方差为{stats.describe(bnl).variance}")
print(f"服从二项分布B（10,0.2）的1000个样本的极值为{stats.describe(bnl).minmax}")
print("\n")

#正太分布 均值、方差、极差
normal = RNG.normal(10,0.04,1000)
print(stats.describe(normal))
print(f"服从正太分布N（10,0.04）的1000个样本的均值为{stats.describe(normal).mean}")
print(f"服从正太分布N（10,0.04）的1000个样本的方差为{stats.describe(normal).variance}")
print(f"服从正太分布N（10,0.04）的1000个样本的极值为{stats.describe(normal).minmax}")
print("\n")

#指数分布 均值、方差、极差
exp = RNG.exponential(0.2,1000)
print(stats.describe(exp))
print(f"服从指数分布Exp（0.2）的1000个样本的均值为{stats.describe(exp).mean}")
print(f"服从指数分布Exp（0.2）的1000个样本的方差为{stats.describe(exp).variance}")
print(f"服从指数分布Exp（0.2）的1000个样本的极值为{stats.describe(exp).minmax}")
print("\n")

#几何分布 均值、方差、极差
geo = RNG.geometric(0.2,1000)
print(stats.describe(exp))
print(f"服从几何分布Ge（0.2）的1000个样本的均值为{stats.describe(geo).mean}")
print(f"服从几何分布Ge（0.2）的1000个样本的方差为{stats.describe(geo).variance}")
print(f"服从几何分布Ge（0.2）的1000个样本的极值为{stats.describe(geo).minmax}")
print("\n")

#泊松分布 均值、方差、极差
poisson  = RNG.poisson(0.2,1000)
print(stats.describe(exp))
print(f"服从泊松分布P（0.2）的1000个样本的均值为{stats.describe(poisson).mean}")
print(f"服从泊松分布P（0.2）的1000个样本的方差为{stats.describe(poisson).variance}")
print(f"服从泊松分布P（0.2）的1000个样本的极值为{stats.describe(poisson).minmax}")
print("\n")

#伽马分布 均值、方差、极差
gamma = RNG.gamma(1,0.2,1000)
print(stats.describe(gamma))
print(f"服从伽马分布Ga(1,0.2)的1000个样本的均值为{stats.describe(gamma).mean}")
print(f"服从伽马分布Ga(1,0.2)的1000个样本的方差为{stats.describe(gamma).variance}")
print(f"服从伽马分布Ga(1,0.2)的1000个样本的极值为{stats.describe(gamma).minmax}")
print("\n")

#贝塔分布 均值、方差、极差
beta = RNG.beta(6,8,1000)
print(stats.describe(beta))
print(f"服从贝塔分布Be(6,8)的1000个样本的均值为{stats.describe(beta).mean}")
print(f"服从贝塔分布Be(6,8)的1000个样本的方差为{stats.describe(beta).variance}")
print(f"服从贝塔分布Be(6,8)的1000个样本的极值为{stats.describe(beta).minmax}")
print("\n")

#F分布 均值、方差、极差
F = RNG.f(6,8,1000)
print(stats.describe(F))
print(f"服从F分布F(6,8)的1000个样本的均值为{stats.describe(F).mean}")
print(f"服从F分布F(6,8)的1000个样本的方差为{stats.describe(F).variance}")
print(f"服从F分布F(6,8)的1000个样本的极值为{stats.describe(F).minmax}")
print("\n")

#t分布 均值、方差、极差
t = RNG.standard_t(2,1000)
print(stats.describe(t))
print(f"服从t分布t(2)的1000个样本的均值为{stats.describe(t).mean}")
print(f"服从t分布t(2)的1000个样本的方差为{stats.describe(t).variance}")
print(f"服从t分布t(2)的1000个样本的极值为{stats.describe(t).minmax}")
print("\n")


#卡方分布 均值、方差、极差
chisquare = RNG.chisquare(2,1000)
print(stats.describe(t)) 
print(f"服从自由度为2的卡方分布的1000个样本的均值为{stats.describe(chisquare).mean}")
print(f"服从自由度为2的卡方分布的1000个样本的方差为{stats.describe(chisquare).variance}")
print(f"服从自由度为2的卡方分布的1000个样本的极值为{stats.describe(chisquare).minmax}")
print("\n")












# In[ ]:





# In[ ]:





# In[ ]:




