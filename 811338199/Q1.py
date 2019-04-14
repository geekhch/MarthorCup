import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab
pylab.rcParams['figure.figsize'] = (20.0, 15.0)
data = pd.read_excel('data1.xlsx')
data.drop(data[data['连铸正样C'].isnull()].index,inplace=True)
c_t=['钒铁(FeV50-A)','钒铁(FeV50-B)','硅铝合金FeAl30Si25','硅锰面（硅锰渣）','硅铁(合格块)','硅铁FeSi75-B','石油焦增碳剂','锰硅合金FeMn64Si27(合格块)','锰硅合金FeMn68Si18(合格块)','碳化硅(55%)','硅钙碳脱氧剂']
c_p=[0.0031,0.0031,0.00374,0.017,0.0006,0.0006,0.96,0.017,0.017,0.3,0.225692308]
mn_t=['硅铝锰合金球','硅锰面（硅锰渣）','锰硅合金FeMn64Si27(合格块)','锰硅合金FeMn68Si18(合格块)']
mn_p = [0.3,0.664,0.664,0.664]
data['加入C含量']=(data[c_t]*c_p).sum(axis=1)
data['吸收C质量']=(data['连铸正样C']-data['转炉终点C'])*data['钢水净重']
data['C收得率'] = data['吸收C质量']/data['加入C含量']
data['加入Mn含量']=(data[mn_t]*mn_p).sum(axis=1)
data['吸收Mn质量']=(data['连铸正样Mn']-data['转炉终点Mn'])*data['钢水净重']
data['Mn收得率']=data['吸收Mn质量']/data['加入Mn含量']
h = data['转炉终点C'].hist(bins=100,)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('转炉终点C',fontproperties="SimHei",fontsize=35)
h.grid(False)
nume = ['转炉终点温度', '转炉终点C', '转炉终点S',
       '转炉终点Si', '钢水净重', 
        '氮化钒铁FeV55N11-A', '低铝硅铁',
       '钒氮合金(进口)', '钒铁(FeV50-A)', '钒铁(FeV50-B)', '钒铁(FeV50-B).1', '硅铝钙',
       '硅铝合金FeAl30Si25', '硅铝锰合金球', '硅锰面（硅锰渣）', '硅铁(合格块)', '硅铁FeSi75-B',
       '石油焦增碳剂', '锰硅合金FeMn64Si27(合格块)', '锰硅合金FeMn68Si18(合格块)', '碳化硅(55%)',
       '硅钙碳脱氧剂','C收得率','Mn收得率']
nm_df = data[nume].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
data[list(used.index)+['C收得率']].to_csv('C_train_data.csv',index=False)

c_corr = data[nume].corr()[['C收得率']]
used = c_corr[(abs(c_corr['C收得率'])>0.15)]
used
c_corr = data[nume].corr()[['Mn收得率']]
used = c_corr[(abs(c_corr['Mn收得率'])>0.15)]
used.sort_values('Mn收得率')
data.head()

