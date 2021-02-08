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


diabetes_data = pd.read_csv("C:/Users/admin/Desktop/data/数据挖掘/diabetes.csv")
diabetes_data.info()


# In[3]:


listname = ["Pregnancies","Glucose","BP","skin","insulin","BMI","pedigree","age","label"]
diabetes_data.columns = listname                                                              #修改列名
diabetes_data.info()


# ## 描述性统计

# ### describe() 计算series或DataFrame各列的汇总统计集合
# - count 非NA值的个数
# - mean 均值
# - std 样本标准差

# In[4]:


diabetes_data.describe()


# ## corr()和cov()方法会以DataFrame格式返回相关性和协方差矩阵

# In[5]:


diabetes_data.corr()


# 从label一列中可以看出，特征血糖、BMI、年龄、妊娠次数与糖尿病相关性较高（0.22-0.47）

# In[6]:


diabetes_data.cov()


# In[7]:


dia_df = diabetes_data                   
label_rate = dia_df.label.value_counts() / len(dia_df)                   #（不）患糖尿病的比率
label_rate 


# In[8]:


label_Summary = dia_df.groupby('label')
label_Summary.mean()                                                     #分组的数据比较


# 从各组数据的平均值可以看出，患糖尿病的人群比不患糖尿病的人群血糖、胰岛素、BMI明显要高，除BP之外，其它指标也存在比较明显的差异

# ## 制作数据集

# In[9]:


# 选择预测所需的特征
feature_cols = ['Pregnancies','Glucose' ,'insulin', 'BMI', 'age','pedigree']
X = dia_df[feature_cols] # 特征
y = dia_df.label # 类别标签

# 将数据分为训练和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3) # 70% training and 30% test


# # 使用多种方法对糖尿病进行预测

# ## 1.决策树

# In[10]:


# 创建决策树分类器
clf_dt0 = DecisionTreeClassifier(criterion='entropy' )                              #criterion：gini或者entropy,前者是基尼系数，后者是信息熵

# 训练模型
clf_dt0 = clf_dt0.fit(X_train,y_train)

# 使用训练好的模型做预测
y_pred = clf_dt0.predict(X_test)

dt_roc_auc0 = accuracy_score(y_test, y_pred)
print ("---决策树0---")
print ("决策树0 AUC = %2.2f" % dt_roc_auc0)
print(classification_report(y_test, y_pred))

# 模型的准确性
print("使用决策树0预测的准确率:",dt_roc_auc0)


# In[ ]:





# ### NEW
# 1. sklearn中的classification_report函数用于显示主要分类指标的文本报告．在报告中显示每个类的精确度，召回率，F1值等信息。
# 
# classification_report(y_true, y_pred, target_names=target_names)
# 
# 主要参数:
# - y_true：1维数组，或标签指示器数组/稀疏矩阵，目标值。
# - y_pred：1维数组，或标签指示器数组/稀疏矩阵，分类器返回的估计值。
# 
# - labels：array，shape = [n_labels]，报表中包含的标签索引的可选列表。
# - target_names：字符串列表，与标签匹配的可选显示名称（相同顺序）。
# 
# - sample_weight：类似于shape = [n_samples]的数组，可选项，样本权重。
# - digits：int，输出浮点值的位数．

# In[ ]:




