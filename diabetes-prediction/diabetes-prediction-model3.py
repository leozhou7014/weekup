#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier                            #导入决策树模型
from sklearn.naive_bayes import MultinomialNB,BernoulliNB                  #导入朴素贝叶斯算法
from sklearn.metrics import accuracy_score                                 #分类准确率分数是指所有分类正确的百分比。
from sklearn.model_selection import train_test_split                        #返回切分的数据集train/test
from sklearn.metrics import classification_report                          


# In[2]:


import os
print(os.path.abspath('.'))


# In[3]:


diabetes_data = pd.read_csv("C:/Users/admin/Desktop/homework/dm/diabetes-prediction/diabetes.csv")
diabetes_data.info()


# In[4]:


listname = ["Pregnancies","Glucose","BP","Skin","Insulin","BMI","Pedigree","Age","Label"]
diabetes_data.columns = listname                                                              #修改列名
diabetes_data.info()


# ## 异常值处理

# In[5]:


colume = ['Glucose', 'BP', 'Skin', 'Insulin', 'BMI']
diabetes_data[colume] = diabetes_data[colume].replace(0,np.nan)               #将0值替换为nan
import missingno as msno                                         #missingno提供了一个灵活且易于使用的缺失数据可视化和实用程序的小工具集
p=msno.bar(diabetes_data)


# In[6]:


print(f"数据处理之前缺失值个数：\n{diabetes_data.isnull().sum()}" )
medians = diabetes_data.median() 
diabetes_data = diabetes_data.fillna(medians)
print(f"数据处理之后缺失值个数：\n{diabetes_data.isnull().sum()}")


# # 描述性统计

# In[7]:


diabetes_data.describe()


# ## corr()和cov()方法会以DataFrame格式返回相关性和协方差矩阵

# In[8]:


diabetes_data.corr()


# In[9]:


diabetes_data.cov()


# In[10]:


dia_df = diabetes_data                   
label_rate = dia_df.Label.value_counts() / len(dia_df)                   #（不）患糖尿病的比率
label_rate 


# In[11]:


label_Summary = dia_df.groupby('Label')
label_Summary.mean()                                                     #分组的数据比较


# ### 使用seaborn中的pairplot()绘制两两特征图

# In[12]:


import timeit                                     #计算程序运行时间
start=timeit.default_timer()
import seaborn as sns
p=sns.pairplot(diabetes_data, hue = 'Label')
end=timeit.default_timer()
timeGNB=end-start
print(f'程序运行时间: {timeGNB} Seconds')


# # 使用多种方法对糖尿病进行预测

# ## 1.决策树

# In[13]:


# 选择预测所需的特征
feature_cols = ['Pregnancies','Glucose' ,'Insulin', 'BP','BMI', 'Age','Skin','Pedigree']
X = dia_df[feature_cols] # 特征
y = dia_df.Label # 类别标签


# In[14]:


# 将数据分为训练和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3) # 70% training and 30% test


# In[15]:


# 创建决策树分类器
clf_dt = DecisionTreeClassifier(criterion='entropy' )                              #criterion：gini或者entropy,前者是基尼系数，后者是信息熵

# 训练模型
clf_dt = clf_dt.fit(X_train,y_train)

# 使用训练好的模型做预测
y_pred = clf_dt.predict(X_test)

auc_dt = accuracy_score(y_test, y_pred)
print ("---决策树---")
print ("决策树 AUC = %2.2f" % auc_dt)
print(classification_report(y_test, y_pred))

# 模型的准确性
print("使用决策树预测的准确率:",auc_dt)


# ## 2.朴素贝叶斯法

# In[16]:


from sklearn.naive_bayes import GaussianNB                #导入朴素贝叶斯算法
model_g =  GaussianNB()
model_g = model_g.fit(X_train,y_train)
pred_g = model_g.predict(X_test)

# 训练模型
nbm = model_g.fit(X_train,y_train)

# 使用训练好的模型做预测
y_pred = model_g.predict(X_test)

auc_nbg = accuracy_score(y_test, y_pred)
print ("---高斯朴素贝叶斯---")
print ("高斯朴素贝叶斯 AUC = %2.2f" % auc_nbg)
print(classification_report(y_test, y_pred))

# 模型的准确性
print("高斯朴素贝叶斯预测的准确率:",auc_nbg)


# ## 3.SVM方法

# In[17]:


start=timeit.default_timer()
from sklearn import svm
model_svm = svm.SVC(C=1.0, kernel='linear', degree=3, gamma=1.0, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=6)
model_svm.fit(X_train,y_train)

# 训练模型
svm = model_svm.fit(X_train,y_train)

# 使用训练好的模型做预测
y_pred = model_svm.predict(X_test)

auc_svm = accuracy_score(y_test, y_pred)
print ("---SVM---")
print ("SVM AUC = %2.2f" % auc_svm)
print(classification_report(y_test, y_pred))

# 模型的准确性
print(f"SVM预测的准确率:{auc_svm}")

end=timeit.default_timer()
timeGNB=end-start
print(f'程序运行时间: {timeGNB} Seconds')


# ### 调参

# ## 1.选择不同kernel
# - (1)kernel='rbf'
# - (2)kernel='poly'
# “linear”(线性）、“rbf”（径向基)、“poly”（多项式）(默认值是“rbf”)

# In[18]:


start=timeit.default_timer()

from sklearn import svm
model_svm = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=1.0, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=6)
model_svm.fit(X_train,y_train)

# 训练模型
svm = model_svm.fit(X_train,y_train)

# 使用训练好的模型做预测
y_pred = model_svm.predict(X_test)

auc_svm = accuracy_score(y_test, y_pred)
print ("---SVM---")
print ("SVM AUC = %2.2f" % auc_svm)
print(classification_report(y_test, y_pred))

# 模型的准确性
print(f"SVM预测的准确率:{auc_svm}")

end=timeit.default_timer()
timeGNB=end-start
print(f'程序运行时间: {timeGNB} Seconds')


# **kenel选择"linear"比选择"rbf"模型的预测效果更好，选择"poly"时，程序运行时间很长，故不做比较。**

# ## 2.选择不同C
# - (1)C=10
# - (2)C=100
# 误差项的惩罚参数C,它还控制平滑决策边界和正确分类训练点之间的权衡。

# In[19]:


start=timeit.default_timer()
from sklearn import svm
model_svm = svm.SVC(C=10, kernel='linear', degree=3, gamma=1.0, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=6)
model_svm.fit(X_train,y_train)

# 训练模型
svm = model_svm.fit(X_train,y_train)

# 使用训练好的模型做预测
y_pred = model_svm.predict(X_test)

auc_svm = accuracy_score(y_test, y_pred)
print ("---SVM---")
print ("SVM AUC = %2.2f" % auc_svm)
print(classification_report(y_test, y_pred))

# 模型的准确性
print(f"SVM预测的准确率:{auc_svm}")

end=timeit.default_timer()
timeGNB=end-start
print(f'程序运行时间: {timeGNB} Seconds')


# In[20]:


start=timeit.default_timer()
from sklearn import svm
model_svm = svm.SVC(C=100, kernel='linear', degree=3, gamma=1.0, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=6)
model_svm.fit(X_train,y_train)

# 训练模型
svm = model_svm.fit(X_train,y_train)

# 使用训练好的模型做预测
y_pred = model_svm.predict(X_test)

auc_svm = accuracy_score(y_test, y_pred)
print ("---SVM---")
print ("SVM AUC = %2.2f" % auc_svm)
print(classification_report(y_test, y_pred))

# 模型的准确性
print(f"SVM预测的准确率:{auc_svm}")

end=timeit.default_timer()
timeGNB=end-start
print(f'程序运行时间: {timeGNB} Seconds')


# **惩罚参数C增大时，模型预测的准确率和召回率有所提高，但是程序运行时间较长，需要做好权衡。**

# ## 3.选择不同gamma
# - (1)gamma=10
# - (2)gamma=100
# 
# gamma值高，将尝试精确匹配每一个训练数据集，但可能会导致泛化误差和引起过度拟合问题。

# In[21]:


start=timeit.default_timer()
from sklearn import svm
model_svm = svm.SVC(C=30, kernel='linear', degree=3, gamma=10, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=6)
model_svm.fit(X_train,y_train)

# 训练模型
svm = model_svm.fit(X_train,y_train)

# 使用训练好的模型做预测
y_pred = model_svm.predict(X_test)

auc_svm = accuracy_score(y_test, y_pred)
print ("---SVM---")
print ("SVM AUC = %2.2f" % auc_svm)
print(classification_report(y_test, y_pred))

# 模型的准确性
print(f"SVM预测的准确率:{auc_svm}")

end=timeit.default_timer()
timeGNB=end-start
print(f'程序运行时间: {timeGNB} Seconds')


# In[22]:


start=timeit.default_timer()
from sklearn import svm
model_svm = svm.SVC(C=30, kernel='linear', degree=3, gamma=100, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=6)
model_svm.fit(X_train,y_train)

# 训练模型
svm = model_svm.fit(X_train,y_train)

# 使用训练好的模型做预测
y_pred = model_svm.predict(X_test)

auc_svm = accuracy_score(y_test, y_pred)
print ("---SVM---")
print ("SVM AUC = %2.2f" % auc_svm)
print(classification_report(y_test, y_pred))

# 模型的准确性
print(f"SVM预测的准确率:{auc_svm}")

end=timeit.default_timer()
timeGNB=end-start
print(f'程序运行时间: {timeGNB} Seconds')


# **gamma从1增加到10，模型预测准确率和召回率略有提升，从10增加到100，模型准确率和召回率基本不变**

# - 选择最终模型 kernel="linear",gamma=10,C=30

# In[24]:


start=timeit.default_timer()
from sklearn import svm
model_svm = svm.SVC(C=30, kernel='linear', degree=3, gamma=10, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=6)
model_svm.fit(X_train,y_train)

# 训练模型
svm = model_svm.fit(X_train,y_train)

# 使用训练好的模型做预测
y_pred = model_svm.predict(X_test)

auc_svm = accuracy_score(y_test, y_pred)
print ("---SVM---")
print ("SVM AUC = %2.2f" % auc_svm)
print(classification_report(y_test, y_pred))

# 模型的准确性
print(f"SVM预测的准确率:{auc_svm}")

end=timeit.default_timer()
timeGNB=end-start
print(f'程序运行时间: {timeGNB} Seconds')

