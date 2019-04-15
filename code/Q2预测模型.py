#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
from sklearn import tree,metrics,svm,ensemble,neighbors
import pandas as pd
from sklearn.model_selection import train_test_split,KFold
from sklearn import linear_model
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


# ### 去掉缺失值较多的列

# In[20]:


nume = ['转炉终点温度', '转炉终点C', '转炉终点S',
        '钢水净重', '氮化钒铁FeV55N11-A', '低铝硅铁',
       '钒氮合金(进口)', '钒铁(FeV50-A)', '钒铁(FeV50-B)', '钒铁(FeV50-B).1', '硅铝钙',
       '硅铝合金FeAl30Si25', '硅铝锰合金球', '硅锰面（硅锰渣）', '硅铁(合格块)', '硅铁FeSi75-B',
       '石油焦增碳剂', '锰硅合金FeMn64Si27(合格块)', '锰硅合金FeMn68Si18(合格块)', '碳化硅(55%)',
       '硅钙碳脱氧剂']


# ### 缺失值处理、归一化

# In[31]:


CTrain = pd.read_excel('q1_1_收得率.xls')[nume+['C收得率']]
# CTrain = CTrain.dropna()
CTrain = CTrain.drop(CTrain[CTrain['C收得率'].isnull()].index)
# 离群值处理
outs=['转炉终点温度', '转炉终点C','钢水净重']
for out in outs:
    CTrain.drop(CTrain[(CTrain[out]>(CTrain[out].mean()+3*CTrain[out].std()))|(CTrain[out]<(CTrain[out].mean()-3*CTrain[out].std()))].index,inplace=True)
    
CTrain = CTrain.fillna(CTrain.mean())
X = CTrain[nume] #[['钢水净重','连铸正样C', '连铸正样Ceq_val', '低铝硅铁','石油焦增碳剂']]
X = X[nume].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)+1e-10)).values
Y = CTrain[['C收得率']].values
len(CTrain.values)


# In[32]:


# cget = (CTrain['C收得率']-CTrain['C收得率'].min())/(CTrain['C收得率'].max()-CTrain['C收得率'].min())
plt.plot(X['钢水净重'][::50])
plt.plot(CTrain['C收得率'][::50])


# In[33]:


# #缺失值数量统计
# CTrain.isnull().sum()


# In[34]:


CTrain.head(1)


# In[35]:


# train_data, test_data, train_target, test_target = train_test_split(X.values,Y.values,test_size=0.2)


# ### 决策树回归

# In[36]:


clf = tree.DecisionTreeRegressor() 
ki = 10
kf = KFold(n_splits=ki)
r2,mae,mse=0,0,0
for train_index,test_index in kf.split(X):
    clf = clf.fit(X[train_index],Y[train_index])
    pre_y = clf.predict(X[test_index])
    r2+=metrics.r2_score(pre_y,Y[test_index])
    mae+=metrics.mean_absolute_error(pre_y,Y[test_index])
    mse+=metrics.mean_squared_error(pre_y,Y[test_index])
r2/ki,mae/ki,mse/ki


# 以上三个值分别为R2决定系数、平均绝对误差、平均平方误差

# ### 线性回归

# In[49]:


clf = linear_model.LinearRegression()
ki = 4
kf = KFold(n_splits=ki)
r2,mae,mse=0,0,0
for train_index,test_index in kf.split(X):
    clf = clf.fit(X[train_index],Y[train_index])
    pre_y = clf.predict(X[test_index])
    r2+=metrics.r2_score(pre_y,Y[test_index])
    mae+=metrics.mean_absolute_error(pre_y,Y[test_index])
    mse+=metrics.mean_squared_error(pre_y,Y[test_index])
r2/ki,mae/ki,mse/ki


# In[50]:


mapk={}
for i in range(len(nume)):
    mapk[nume[i]]=np.round(clf.coef_[0][i],4)
    print(nume[i],' & ',np.round(clf.coef_[0][i],4))
# sorted(mapk.items(),key=lambda item:abs(item[1]))


# ### SVM

# In[115]:


clf = svm.SVR(gamma='auto')
ki = 10
kf = KFold(n_splits=ki)
r2,mae,mse=0,0,0
for train_index,test_index in kf.split(X):
    clf = clf.fit(X[train_index],Y[train_index].flatten().astype(np.float32))
    pre_y = clf.predict(X[test_index])
    r2+=metrics.r2_score(pre_y,Y[test_index])
    mae+=metrics.mean_absolute_error(pre_y,Y[test_index])
    mse+=metrics.mean_squared_error(pre_y,Y[test_index])
r2/ki,mae/ki,mse/ki


# ### 贝叶斯

# In[116]:


clf = linear_model.BayesianRidge()
ki = 10
kf = KFold(n_splits=ki)
r2,mae,mse=0,0,0
for train_index,test_index in kf.split(X):
    clf = clf.fit(X[train_index],Y[train_index].flatten().astype(np.float32))
    pre_y = clf.predict(X[test_index])
    r2+=metrics.r2_score(pre_y,Y[test_index])
    mae+=metrics.mean_absolute_error(pre_y,Y[test_index])
    mse+=metrics.mean_squared_error(pre_y,Y[test_index])
r2/ki,mae/ki,mse/ki


# ### 集成

# In[117]:


clf = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls')
ki = 10
kf = KFold(n_splits=ki)
r2,mae,mse=0,0,0
for train_index,test_index in kf.split(X):
    clf = clf.fit(X[train_index],Y[train_index].flatten().astype(np.float32))
    pre_y = clf.predict(X[test_index])
    r2+=metrics.r2_score(pre_y,Y[test_index])
    mae+=metrics.mean_absolute_error(pre_y,Y[test_index])
    mse+=metrics.mean_squared_error(pre_y,Y[test_index])
r2/ki,mae/ki,mse/ki


# ### 多项式回归（emm）

# In[142]:


clf = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', linear_model.LinearRegression(fit_intercept=False))])
ki = 10
kf = KFold(n_splits=ki)
r2,mae,mse=0,0,0
for train_index,test_index in kf.split(X):
    clf = clf.fit(X[train_index],Y[train_index].flatten().astype(np.float32))
    pre_y = clf.predict(X[test_index])
    r2+=metrics.r2_score(pre_y,Y[test_index])
    mae+=metrics.mean_absolute_error(pre_y,Y[test_index])
    mse+=metrics.mean_squared_error(pre_y,Y[test_index])
r2/ki,mae/ki,mse/ki


# # Mn元素预测

# ### 数据预处理

# In[123]:


nume = ['转炉终点温度', '转炉终点C', '转炉终点Mn', '转炉终点S',
       '转炉终点Si', '钢水净重', '连铸正样Mn', 
#         '连铸正样C','连铸正样S', '连铸正样P', '连铸正样Si',
#        '连铸正样Ceq_val', '连铸正样Cr', '连铸正样Ni_val', '连铸正样Cu_val',
#        '连铸正样V_val', '连铸正样Alt_val', '连铸正样Als_val', '连铸正样Mo_val', '连铸正样Ti_val',
        '氮化钒铁FeV55N11-A', '低铝硅铁',
       '钒氮合金(进口)', '钒铁(FeV50-A)', '钒铁(FeV50-B)', '钒铁(FeV50-B).1', '硅铝钙',
       '硅铝合金FeAl30Si25', '硅铝锰合金球', '硅锰面（硅锰渣）', '硅铁(合格块)', '硅铁FeSi75-B',
       '石油焦增碳剂', '锰硅合金FeMn64Si27(合格块)', '锰硅合金FeMn68Si18(合格块)', '碳化硅(55%)',
       '硅钙碳脱氧剂']


# In[131]:



MnTrain = pd.read_excel('data1.xlsx')[nume]
# Mn初始含量较低，对反应收得率影响小，使用均值替换缺失值
MnTrain[['转炉终点Mn']] = MnTrain[['转炉终点Mn']].fillna(MnTrain[['转炉终点Mn']].mean())
# 去掉为对合金化后合金钢采样的数据
MnTrain = MnTrain.drop(MnTrain[MnTrain['连铸正样Mn'].isnull()].index)
# 使用均值替代少量的缺失值
MnTrain[nume] = MnTrain[nume].fillna(MnTrain[nume].mean())
# 重新计算Mn收得率
mn_t=['硅铝锰合金球','硅锰面（硅锰渣）','锰硅合金FeMn64Si27(合格块)','锰硅合金FeMn68Si18(合格块)']
mn_p = [0.3,0.664,0.664,0.664]
mn_total=(MnTrain[mn_t]*mn_p).sum(axis=1)
mn_comsu=(MnTrain['连铸正样Mn']-MnTrain['转炉终点Mn'])*MnTrain['钢水净重']
MnTrain['Mn收得率']=mn_comsu/(mn_total)
# 删掉异常数据
MnTrain = MnTrain.drop(MnTrain[MnTrain['Mn收得率']>1.5].index)

# 构造训练数据并对自变量值进行归一化
X = MnTrain[nume].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)+1e-9))
X.drop('连铸正样Mn',axis=1,inplace=True)
X=X.values
Y = MnTrain[['Mn收得率']].values
# train_data, test_data, train_target, test_target = train_test_split(X.values,Y.values,test_size=0.2)
len(Y)


# In[132]:


# MnTrain #.isnull().sum()
# Y.sort_values('Mn收得率')


# ### 多项式拟合

# In[133]:


clf = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', linear_model.LinearRegression(fit_intercept=False))])
ki = 10
kf = KFold(n_splits=ki)
r2,mae,mse=0,0,0
for train_index,test_index in kf.split(X):
    clf = clf.fit(X[train_index],Y[train_index].flatten().astype(np.float32))
    pre_y = clf.predict(X[test_index])
    r2+=metrics.r2_score(pre_y,Y[test_index])
    mae+=metrics.mean_absolute_error(pre_y,Y[test_index])
    mse+=metrics.mean_squared_error(pre_y,Y[test_index])
r2/ki,mae/ki,mse/ki


# In[134]:


clf = linear_model.BayesianRidge()
clf = clf.fit(train_data,train_target.flatten())
pre_y = clf.predict(test_data)
metrics.r2_score(pre_y,test_target),metrics.mean_absolute_error(pre_y,test_target),metrics.mean_squared_error(pre_y,test_target)


# In[135]:


clf = linear_model.LinearRegression()
clf = clf.fit(train_data,train_target)
pre_y = clf.predict(test_data)
metrics.r2_score(pre_y,test_target),metrics.mean_absolute_error(pre_y,test_target),metrics.mean_squared_error(pre_y,test_target)


# In[ ]:




