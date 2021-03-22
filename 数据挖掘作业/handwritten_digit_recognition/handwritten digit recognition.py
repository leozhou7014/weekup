#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB       #导入三种常用的朴素贝叶斯算法
from sklearn.metrics import accuracy_score                                 #分类准确率分数是指所有分类正确的百分比。
from sklearn.model_selection import train_test_split                        #返回切分的数据集train/test
import matplotlib.pyplot as plt


# In[2]:


digits = load_digits()
x = digits.data
y = digits.target


# In[3]:


print(digits.images.shape)                                             #查看数据集的维度


# In[4]:


print(digits.images[0])
print(digits.images[0].shape)                                           #查看第一张图的属性


# ## imshow()函数格式
# 
# matplotlib.pyplot.imshow(X, cmap=None)
# 
# X: 要绘制的图像或数组。
# 
# cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间。

# In[5]:


plt.imshow(digits.images[0],plt.cm.jet)                              #将第一张图绘制出来
plt.show()


# ## train_test_split()函数划分训练、测试数据
# train_X,test_X,train_y,test_y = train_test_split(train_data,train_target,test_size=0.3,random_state=5)
# 
# 参数解释：
# 
# train_data：待划分样本数据
# 
# train_target：待划分样本数据的结果（标签）
# 
# test_size：测试数据占样本数据的比例，若整数则样本数量
# 
# random_state：设置随机数种子，保证每次都是同一个随机数。若为0或不填，则每次得到数据都不一样

# In[6]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=7)


# # 伯努利朴素贝叶斯模型

# In[7]:


model_b = BernoulliNB()
model_b = model_b.fit(x_train,y_train)
pred_b = model_b.predict(x_test)
accuracy_score_b = accuracy_score(y_test,pred_b)
print("Bernoulli朴素贝叶斯模型的准确率为： %.3lf"% accuracy_score_b)     


# In[8]:


print(model_b.class_log_prior_)                   #各类标记的平滑先验概率对数值(其取值会受fit_prior和class_prior参数的影响)
print("\n")
print(model_b.feature_log_prob_)                  #指定类的各特征概率(条件概率)对数值，返回形状为(n_classes, n_features)数组


# In[9]:


print(model_b.class_count_)                  #class_count_属性：获取各类标记对应的训练样本数
print("\n")
print(model_b.feature_count_)                #获取各个类标记在各个特征上的均值


# # 高斯朴素贝叶斯model

# In[10]:


model_g = GaussianNB()
model_g.fit(x_train,y_train)
pred_g =model_g.predict(x_test)
accuracy_score_g = accuracy_score(y_test,pred_g)
print('GaussianNB朴素贝叶斯模型的准确率为: %.4lf' % accuracy_score_g)


# In[11]:


print(model_g.class_prior_)                  #获取各个类标记对应的先验概率
print(model_g.class_count_)                  #获取各类标记对应的训练样本数


# In[12]:


print(model_g.theta_)                        #获取各个类标记在各个特征上的均值
print("\n")
print(model_g.sigma_)                        #获取各个类标记在各个特征上的方差  


# # 多项式朴素贝叶斯模型

# In[13]:


model_m = MultinomialNB(fit_prior=True)
model_m = model_m.fit(x_train,y_train)
pred_m = model_m.predict(x_test)
accuracy_score_m = accuracy_score(y_test,pred_m)
print("MultinomialNB朴素贝叶斯模型的准确率为： %.4lf" % accuracy_score_m)


# In[14]:


print(model_m.class_log_prior_)             #各类标记的平滑先验概率对数值
print("\n")
print(model_m.feature_log_prob_)            #指定类的各特征概率(条件概率)对数值


# In[15]:


print(model_m.class_count_)            #获取各类标记对应的样本数
print("\n")
print(model_m.feature_count_)          #获取各类标记在各特征上的均值


# In[16]:


print(model_m.coef_)                                  #将多项式朴素贝叶斯解释feature_log_prob_映射成线性模型，其值和feature_log_prob相同
print("\n")
print(model_m.intercept_)                             #将多项式朴素贝叶斯解释的class_log_prior_映射为线性模型，其值和class_log_prior相同


# 本例中模型预测准确率 多项式朴素贝叶斯（0.8852）>伯努利朴素贝叶斯（0.852）>高斯朴素贝叶斯（0.8204）

# ## 总结：
# scikit-learn根据不同场景提供了三种常用的朴素贝叶斯算法：
# 
# （1）如果样本特征的分布大部分是连续值，使用GaussianNB会比较好；
# 
# （2）如果样本特征的分大部分是多元离散值，使用MultinomialNB比较合适。例如文本分类单词统计，以出现的次数作为特征值；
# 
# （3）如果样本特征是二元离散值或者很稀疏的多元离散值，应该使用BernoulliNB。
