#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn import tree,metrics,svm,ensemble,neighbors
import pandas as pd
from sklearn.model_selection import train_test_split,KFold
from sklearn import linear_model
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy import optimize


# In[3]:


mixture = ['氮化钒铁FeV55N11-A','低铝硅铁','钒氮合金(进口)','钒铁(FeV50-A)','钒铁(FeV50-B)','硅铝钙','硅铝合金FeAl30Si25','硅铝锰合金球',
           '硅锰面（硅锰渣）','硅铁(合格块)','硅铁FeSi75-B','石油焦增碳剂','锰硅合金FeMn64Si27(合格块)','锰硅合金FeMn68Si18(合格块)',
           '碳化硅(55%)','硅钙碳脱氧剂']
columns = ['转炉终点温度', '转炉终点C', '转炉终点Mn', '转炉终点S', '转炉终点P',
       '转炉终点Si', '钢水净重', '连铸正样C', '连铸正样Mn', '连铸正样S', '连铸正样P', '连铸正样Si', '氮化钒铁FeV55N11-A', '低铝硅铁',
       '钒氮合金(进口)', '钒铁(FeV50-A)', '钒铁(FeV50-B)', '钒铁(FeV50-B).1', '硅铝钙',
       '硅铝合金FeAl30Si25', '硅铝锰合金球', '硅锰面（硅锰渣）', '硅铁(合格块)', '硅铁FeSi75-B',
       '石油焦增碳剂', '锰硅合金FeMn64Si27(合格块)', '锰硅合金FeMn68Si18(合格块)', '碳化硅(55%)',
       '硅钙碳脱氧剂']
x_col = [ '转炉终点C', '转炉终点Mn', '转炉终点S', '转炉终点P',
       '转炉终点Si','转炉终点温度', '氮化钒铁FeV55N11-A', '低铝硅铁',
       '钒氮合金(进口)', '钒铁(FeV50-A)', '钒铁(FeV50-B)', '硅铝钙',
       '硅铝合金FeAl30Si25', '硅铝锰合金球', '硅锰面（硅锰渣）', '硅铁(合格块)', '硅铁FeSi75-B',
       '石油焦增碳剂', '锰硅合金FeMn64Si27(合格块)', '锰硅合金FeMn68Si18(合格块)', '碳化硅(55%)',
       '硅钙碳脱氧剂']
price = [350000,6500,350000,205000,205000,11800,1000,8500,7600,6000,6000,4600,8150,8150,6100,4000]
len(price)


# ### 计算单位钢水重量加入配料的质量，对最终合金元素浓度的预测模型

# In[4]:


data = pd.read_excel('./data1.xlsx')
data.drop(data[data['连铸正样C'].isnull()].index,inplace=True)

# data=data[data['钢号']=='HRB400B    '][columns]
# 离群值处理
outs=['转炉终点温度', '转炉终点C', '转炉终点Mn', '转炉终点S', '转炉终点P',
       '转炉终点Si', '钢水净重', '连铸正样C', '连铸正样Mn', '连铸正样S', '连铸正样P', '连铸正样Si']
for out in outs:
    data.drop(data[(data[out]>(data[out].mean()+3*data[out].std()))|(data[out]<(data[out].mean()-3*data[out].std()))].index,inplace=True)
#单位钢水重量的浓度

for m in mixture:
    data[m]/=data['钢水净重']
data = data[columns].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)+1e-15))

data.drop('钢水净重',axis=1,inplace=True)
data.fillna(data.mean(),inplace=True)
data2 = pd.read_excel('data2.xlsx')
data2.head(2)
data['钒铁(FeV50-B)'] = data['钒铁(FeV50-B)']+data['钒铁(FeV50-B).1']
data.drop('钒铁(FeV50-B).1',axis=1,inplace=True)
# X = data[x_col].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)+1e-10)).values
X = data[x_col].values

len(data)#.isna().sum()


# ### 以下分别计算五种元素的预测模型，变量X为1kg钢水质量加入的每种合金配料质量的向量、温度、初始合金元素浓度

# In[6]:


Y = data['连铸正样Si'].values
clf = linear_model.LinearRegression()
ki = 8
kf = KFold(n_splits=ki)
r2,mae,mse=0,0,0
for train_index,test_index in kf.split(X):
    clf_Si = clf.fit(X[train_index],Y[train_index])
    pre_y = clf_Si.predict(X[test_index])
    r2+=metrics.r2_score(Y[test_index],pre_y)
    mae+=metrics.mean_absolute_error(Y[test_index],pre_y)
    mse+=metrics.mean_squared_error(Y[test_index],pre_y)
r2/ki,mae/ki,mse/ki,np.mean(np.abs((Y[test_index]-pre_y)/Y[test_index])) # r2系数，平均绝对误差


# In[7]:


Y = data['连铸正样C'].values
clf = linear_model.LinearRegression()
ki = 8
kf = KFold(n_splits=ki)
r2,mae,mse=0,0,0
for train_index,test_index in kf.split(X):
    clf_C = clf.fit(X[train_index],Y[train_index])
    pre_y = clf_C.predict(X[test_index])
    r2+=metrics.r2_score(Y[test_index],pre_y)
    mae+=metrics.mean_absolute_error(Y[test_index],pre_y)
    mse+=metrics.mean_squared_error(Y[test_index],pre_y)
r2/ki,mae/ki,mse/ki,np.mean(np.abs((Y[test_index]-pre_y)/Y[test_index]))


# In[8]:


Y = data['连铸正样S'].values
clf = linear_model.LinearRegression()
ki = 8
kf = KFold(n_splits=ki)
r2,mae,mse=0,0,0
for train_index,test_index in kf.split(X):
    clf_S = clf.fit(X[train_index],Y[train_index])
    pre_y = clf_S.predict(X[test_index])
    r2+=metrics.r2_score(Y[test_index],pre_y)
    mae+=metrics.mean_absolute_error(Y[test_index],pre_y)
    mse+=metrics.mean_squared_error(Y[test_index],pre_y)
r2/ki,mae/ki,mse/ki,np.mean(np.abs((Y[test_index]-pre_y)/Y[test_index]))


# In[9]:


## S,P缺失值较多，去掉缺失值行后计算模型


# In[10]:


data.drop(data[data['转炉终点Mn'].isnull()].index,inplace=True)
data.drop(data[data['转炉终点P'].isnull()].index,inplace=True)


# In[11]:


Y = data['连铸正样P'].values
clf = linear_model.LinearRegression()
ki = 8
kf = KFold(n_splits=ki)
r2,mae,mse=0,0,0
for train_index,test_index in kf.split(X):
    clf_P = clf.fit(X[train_index],Y[train_index])
    pre_y = clf_P.predict(X[test_index])
    r2+=metrics.r2_score(Y[test_index],pre_y)
    mae+=metrics.mean_absolute_error(Y[test_index],pre_y)
    mse+=metrics.mean_squared_error(Y[test_index],pre_y)
r2/ki,mae/ki,mse/ki,np.mean(np.abs((Y[test_index]-pre_y)/Y[test_index]))


# In[12]:


Y = data['连铸正样Mn'].values
clf = linear_model.BayesianRidge()
ki = 8
kf = KFold(n_splits=ki)
r2,mae,mse=0,0,0
for train_index,test_index in kf.split(X):
    clf_Mn = clf.fit(X[train_index],Y[train_index])
    pre_y = clf_Mn.predict(X[test_index])
    r2+=metrics.r2_score(Y[test_index],pre_y)
    mae+=metrics.mean_absolute_error(Y[test_index],pre_y)
    mse+=metrics.mean_squared_error(Y[test_index],pre_y)
## r2决定系数，绝对误差，绝对均方误差，绝对误差百分比
r2/ki,mae/ki,mse/ki,np.mean(np.abs((Y[test_index]-pre_y)/Y[test_index]))


# ## 使用常量替代不是配料的变量，计算一吨合金在下列条件下的最低配料费用

# In[13]:


init_ = [data['转炉终点C'].mean(),data['转炉终点Mn'].mean(),data['转炉终点S'].mean(),data['转炉终点P'].mean(),data['转炉终点Si'].mean(),data['转炉终点温度'].mean()]


# ## 建立线性约束不等式以及目标函数,并优化求解

# In[15]:



c = data2['价格(元/吨)'].values/1000 #配料价格作为目标函数系数/kg
a = np.array([
    list(clf_C.coef_[6:]),list(-clf_C.coef_[6:]),list(clf_Mn.coef_[6:]),list(-clf_Mn.coef_[6:]),
    list(clf_S.coef_[6:]),list(clf_P.coef_[6:]),
    list(clf_Si.coef_[6:]),list(-clf_Si.coef_[6:])
])
b = np.array([
    0.0025-sum(clf_C.coef_[:6]*init_),-0.0019+sum(clf_C.coef_[:6]*init_),
    0.016-sum(clf_Mn.coef_[:6]*init_),-0.013+sum(clf_Mn.coef_[:6]*init_),
    0.00045-sum(clf_S.coef_[:6]*init_),0.00045-sum(clf_P.coef_[:6]*init_),
    0.0065-sum(clf_Si.coef_[:6]*init_),-0.005+sum(clf_Si.coef_[:6]*init_)
])

#设置配料变量取值范围为0到正无穷
bds = []
for pl in data2['合金配料']:
    bds.append((0,None))
optimize.linprog(c,A_ub=a,b_ub=b,bounds=bds,method='interior-point')


# In[26]:


v = [1.48472616e-14, 2.07354363e-13, 4.21345325e-02, 1.68022881e-01,
       2.62549062e-14, 2.93584684e-14, 1.21898570e-14, 1.43150865e-13,
       4.71352500e+00, 8.95557247e-15, 5.43594869e-15, 1.05918868e+00,
       4.94799808e-14, 1.19180051e-14, 4.39332087e-14, 1.71373088e+00]
for i in range(8):
    print(data2['合金配料'][i],'&',1000*v[i],'&',data2['合金配料'][i+8],'&',1000*v[i+8],'\\\\')

