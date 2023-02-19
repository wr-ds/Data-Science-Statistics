# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:24:07 2019

@author: asus
"""
#python期末作业
#1.Suicide rates overview
import os
import pandas as pd
os.chdir('d:\\workpath')
suicide=pd.read_csv('suicide.csv')
#描述性统计
#从一到四阶矩来描述自杀率的分布
suicide['suicides/100k pop'].mean()#计算均值
suicide['suicides/100k pop'].var()#计算方差
suicide['suicides/100k pop'].skew()#计算偏度
suicide['suicides/100k pop'].kurt()#计算峰度
import matplotlib.pyplot as plt
import seaborn as sbn#数据可视化
plt.hist(suicide['suicides/100k pop'],bins=50)#(密率分布)；bins参数（组数）
#分组统计
import numpy as np
##单因素方差分析 不同年龄层
y11=suicide[['suicides/100k pop']][(suicide['age']==str(np.unique(suicide['age'])[0]))]
y12=suicide[['suicides/100k pop']][(suicide['age']==str(np.unique(suicide['age'])[1]))]
y13=suicide[['suicides/100k pop']][(suicide['age']==str(np.unique(suicide['age'])[2]))]
y14=suicide[['suicides/100k pop']][(suicide['age']==str(np.unique(suicide['age'])[3]))]
y15=suicide[['suicides/100k pop']][(suicide['age']==str(np.unique(suicide['age'])[4]))]
y16=suicide[['suicides/100k pop']][(suicide['age']==str(np.unique(suicide['age'])[5]))]
import scipy.stats as stats
stats.f_oneway(y11,y12,y13,y14,y15,y16)######不同年龄层间均值不同
##单因素方差分析  不同性别
y21=suicide[['suicides/100k pop']][(suicide['sex']==str(np.unique(suicide['sex'])[0]))]
y22=suicide[['suicides/100k pop']][(suicide['sex']==str(np.unique(suicide['sex'])[1]))]
stats.f_oneway(y21,y22)##性别对自杀率有影响
##标准化(0,1)  
from sklearn import preprocessing
minmax_scaler=preprocessing.MinMaxScaler()
temp1=suicide.loc[:,['gdp_per_capita ($)','suicides/100k pop','HDI for year']]
suicide.loc[:,['gdp_per_capita ($)','suicides/100k pop','HDI for year']]=minmax_scaler.fit_transform(temp1)
X=suicide.loc[:,['gdp_per_capita ($)','suicides/100k pop']]
X.cov()
X.corr()
X.plot(kind='scatter',x='gdp_per_capita ($)',y='suicides/100k pop')
x1,x2=suicide['gdp_per_capita ($)'],suicide['suicides/100k pop']
stats.kendalltau(x1,x2)
stats.spearmanr(x1,x2)##人均GDP与自杀率相关
Y=suicide.loc[:,['HDI for year','suicides/100k pop']]
Y.cov()
Y.corr()
Y.plot(kind='scatter',x='HDI for year',y='suicides/100k pop')
Y1,Y2=suicide[['HDI for year']],suicide[['suicides/100k pop']]
df=Y1.merge(Y2,on=Y1.index)
df=df.dropna(axis=0)
Y1,Y2=df[['HDI for year']],df[['suicides/100k pop']]
stats.kendalltau(Y1,Y2)
stats.spearmanr(Y1,Y2)##人均GDP与HDI相关
##聚类分析
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import datetime as dt
from sklearn import cluster
#删去空值
df=pd.read_csv('suicide.csv')
df.set_index('country-year',inplace=True)
df.drop(df.columns[0],axis=1,inplace=True)
df.drop(df.columns[0],axis=1,inplace=True)
df=df.dropna(axis=0)
df.info()
df=pd.get_dummies(df,columns=['sex','age','generation'])
kmeans=cluster.KMeans(n_clusters=2).fit(df)
mdl_hc=cluster.AgglomerativeClustering(n_clusters=2).fit(df)##层次聚类
from sklearn import metrics
x1,x2=kmeans.labels_,mdl_hc.labels_
metrics.davies_bouldin_score(df,x1)
metrics.davies_bouldin_score(df,x2) ##DBI值最小为0，越小越好
metrics.silhouette_score(df,x1)
metrics.silhouette_score(df,x2) ##si值最大为1，越大越好


#第二题代码
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
os.chdir('d:\\workpath')
income=pd.read_csv('adult-income.csv')##在excel中将1843个含有？的数据替换为na，此处为处理过后的数据
income.drop(income.columns[2],axis=1,inplace=True)##去除原数据序号列（该序号列已混乱）
income.isnull().sum()##查看缺失值，因其均为分类变量，因此难以插补，故直接删除
income=income.dropna(axis=0)
##将分类变量进行处理得到其对应的哑变量
income=pd.get_dummies(income,columns=['workclass','education','martital','occupation','relationship','race','sex','native_country'])
train=income.sample(frac=0.7)##抽取训练集
test=income[~income.index.isin(train.index)]##剩余当成测试集
train.to_csv('income_train')##生成训练集csv文件
test.to_csv('income_test')##生成测试集csv文件
xtrain,ytrain=train.drop('Y',axis=1),train['Y']
xtest,ytest=test.drop('Y',axis=1),test['Y']
from sklearn.neighbors import KNeighborsClassifier
mdl_knn=KNeighborsClassifier(n_neighbors=17,weights='distance',p=1)
mdl_knn.fit(xtrain,ytrain)
yhat=mdl_knn.predict(xtrain)
ypred=mdl_knn.predict(xtest)
import numpy as np
from sklearn.model_selection import cross_val_score
cross_val_score(mdl_knn,xtrain,ytrain,cv=5)##交叉折数为5的准确率
np.mean(cross_val_score(mdl_knn,xtrain,ytrain,cv=5))##5折平均准确率
##格子搜索，选取最优参数为n_neighbors=17，p=1
from sklearn.model_selection import GridSearchCV 
parameters={'n_neighbors':[1,3,5,7,9,11,13,15,17,19],'p':[1,2]}
knn=KNeighborsClassifier()
knn_grid=GridSearchCV(knn,parameters,cv=5)
knn_grid.fit(xtrain,ytrain)
knn_grid.best_params_
knn_grid.best_score_
##分类性能评价
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print(accuracy_score(ytrain,yhat))##训练集准确率
print(accuracy_score(ytest,ypred))##测试集准确率
print(np.sum(ypred==1))
print(np.sum(ypred==0))
print(np.sum(ytest==1))
print(np.sum(ytest==0))
print(confusion_matrix(ytest,ypred))##测试集混淆矩阵
print(classification_report(ytest,ypred))##测试集分类报告
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt#加载绘图模块
ypred_prob=mdl_knn.predict_proba(xtest)#计算预测概率
fpr,tpr,_=roc_curve(ytest,ypred_prob[:,1])
plt.plot(fpr,tpr,color='darkorange',lw=4)
auc=roc_auc_score(ytest,ypred_prob[:,1])
##logistics分类
from sklearn.linear_model import LogisticRegression
logit=LogisticRegression(penalty='none',solver='lbfgs',class_weight='balanced')##处理过样本不平衡
logit=LogisticRegression(penalty='none',solver='lbfgs')##未处理过样本不平衡
logit.fit(xtrain,ytrain)
yhat=logit.predict(xtrain)
ypred=logit.predict(xtest)
print(accuracy_score(ytest,ypred))##测试集准确率
print(confusion_matrix(ytest,ypred))##测试集混淆矩阵
print(classification_report(ytest,ypred))##测试集分类报告
##朴素贝叶斯分类
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
gnb,mnb,bnb=GaussianNB(),MultinomialNB(),BernoulliNB()
gnb.fit(xtrain,ytrain)
mnb.fit(xtrain,ytrain)
bnb.fit(xtrain,ytrain)
yhat,yhat_,yhat__=gnb.predict(xtrain),mnb.predict(xtrain),bnb.predict(xtrain)
ypred,ypred_,ypred__=gnb.predict(xtest),mnb.predict(xtest),bnb.predict(xtest)
print(accuracy_score(ytest,ypred))##测试集准确率
print(accuracy_score(ytest,ypred_))##测试集准确率
print(accuracy_score(ytest,ypred__))##测试集准确率
print(accuracy_score(ytest,ypred))##测试集准确率
print(confusion_matrix(ytest,ypred))##测试集混淆矩阵
print(classification_report(ytest,ypred))##测试集分类报告
##决策树分类
from sklearn import tree
from sklearn.model_selection import GridSearchCV 
mdl_tree=tree.DecisionTreeClassifier(max_depth=8)
parameters={'max_depth':[3,4,5,6,7,8,9,10,11,12]}
tree_grid=GridSearchCV(mdl_tree,parameters,cv=5)
tree_grid.fit(xtrain,ytrain)
tree_grid.best_params_
tree_grid.best_score_
mdl_tree.fit(xtrain,ytrain)
yhat=mdl_tree.predict(xtrain)
ypred=mdl_tree.predict(xtest)
print(accuracy_score(ytest,ypred))##测试集准确率
print(accuracy_score(ytest,ypred))##测试集准确率
print(confusion_matrix(ytest,ypred))##测试集混淆矩阵
print(classification_report(ytest,ypred))##测试集分类报告


#第三题
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
os.chdir('d:\\workpath')
BJ_PM=pd.read_csv('BeijingPM20100101_20151231.csv')
CD_PM=pd.read_csv('ChengduPM20100101_20151231.csv')
GZ_PM=pd.read_csv('GuangzhouPM20100101_20151231.csv')
SH_PM=pd.read_csv('ShanghaiPM20100101_20151231.csv')
SY_PM=pd.read_csv('ShenyangPM20100101_20151231.csv')
out_columns = ['year','month','day','hour','season','PM_US Post','HUMI','TEMP','DEWP','PRES','cbwd','Iws','precipitation','Iprec']
BJ_PM=BJ_PM[out_columns]
CD_PM=CD_PM[out_columns]
GZ_PM=GZ_PM[out_columns]
SH_PM=SH_PM[out_columns]
SY_PM=SY_PM[out_columns]
################BJ
temp_list =BJ_PM[BJ_PM.HUMI < 0].index.tolist()
BJ_PM.drop(axis=0,labels=temp_list,inplace= True)
BJ_PM['HUMI'].fillna(BJ_PM['HUMI'].mean(),inplace=True)
BJ_PM['TEMP'].fillna(BJ_PM['TEMP'].mean(),inplace=True)
BJ_PM['DEWP'].fillna(BJ_PM['DEWP'].mean(),inplace=True)
BJ_PM['PRES'].fillna(BJ_PM['PRES'].mean(),inplace=True)
BJ_PM['Iws'].fillna(BJ_PM['Iws'].mean(),inplace=True)
BJ_PM['season'] = BJ_PM['season'].map({1:'Spring',2:'Summer',3:'Autumn',4:'Winter'})
BJ_PM.dropna(axis=0,how='any',subset=['PM_US Post','precipitation','Iprec','cbwd'],inplace= True)
BJ_PM.isnull().sum()
BJ_PM.info()
regress=pd.get_dummies(BJ_PM,columns=['cbwd','season'])
Y1=BJ_PM[['PM_US Post']][(BJ_PM['season']==str(np.unique(BJ_PM['season'])[0]))]
Y2=BJ_PM[['PM_US Post']][(BJ_PM['season']==str(np.unique(BJ_PM['season'])[1]))]
Y3=BJ_PM[['PM_US Post']][(BJ_PM['season']==str(np.unique(BJ_PM['season'])[2]))]
Y4=BJ_PM[['PM_US Post']][(BJ_PM['season']==str(np.unique(BJ_PM['season'])[3]))]
stats.f_oneway(Y1,Y2,Y3,Y4)
BJ_PM['cbwd'].value_counts()
Y1=BJ_PM[['PM_US Post']][(BJ_PM['cbwd']==str(np.unique(BJ_PM['cbwd'])[0]))]
Y2=BJ_PM[['PM_US Post']][(BJ_PM['cbwd']==str(np.unique(BJ_PM['cbwd'])[1]))]
Y3=BJ_PM[['PM_US Post']][(BJ_PM['cbwd']==str(np.unique(BJ_PM['cbwd'])[2]))]
Y4=BJ_PM[['PM_US Post']][(BJ_PM['cbwd']==str(np.unique(BJ_PM['cbwd'])[3]))]
stats.f_oneway(Y1,Y2,Y3,Y4)
Y=BJ_PM[out_columns]
Y.corr()[['PM_US Post']]
regress.drop(['month','hour','PRES','precipitation','Iprec'],axis=1,inplace=True)
train=regress.sample(frac=0.7)
test=regress[~regress.index.isin(train.index)]
xtrain,ytrain=train.drop('PM_US Post',axis=1),train["PM_US Post"]
xtest,ytest=test.drop('PM_US Post',axis=1),test["PM_US Post"]
import statsmodels.api as sm
from sklearn.metrics import*
xtrain=sm.add_constant(xtrain)
xtest=sm.add_constant(xtest)
results=sm.OLS(ytrain,xtrain).fit()
print(results.summary())
ypred=results.predict(xtest)
yhat=results.predict(xtrain)
rmse_=mean_squared_error(ytrain,yhat)**0.5
frmse=mean_squared_error(ytest,ypred)**0.5
#########################CD
temp_list =CD_PM[CD_PM.HUMI < 0].index.tolist()
CD_PM.drop(axis=0,labels=temp_list,inplace= True)
CD_PM['HUMI'].fillna(CD_PM['HUMI'].mean(),inplace=True)
CD_PM['TEMP'].fillna(CD_PM['TEMP'].mean(),inplace=True)
CD_PM['DEWP'].fillna(CD_PM['DEWP'].mean(),inplace=True)
CD_PM['PRES'].fillna(CD_PM['PRES'].mean(),inplace=True)
CD_PM['Iws'].fillna(CD_PM['Iws'].mean(),inplace=True)
CD_PM['season'] = CD_PM['season'].map({1:'Spring',2:'Summer',3:'Autumn',4:'Winter'})
CD_PM.dropna(axis=0,how='any',subset=['PM_US Post','precipitation','Iprec','cbwd'],inplace= True)
CD_PM.isnull().sum()
CD_PM.info()
regress=pd.get_dummies(CD_PM,columns=['cbwd','season'])
Y1=CD_PM[['PM_US Post']][(CD_PM['season']==str(np.unique(CD_PM['season'])[0]))]
Y2=CD_PM[['PM_US Post']][(CD_PM['season']==str(np.unique(CD_PM['season'])[1]))]
Y3=CD_PM[['PM_US Post']][(CD_PM['season']==str(np.unique(CD_PM['season'])[2]))]
Y4=CD_PM[['PM_US Post']][(CD_PM['season']==str(np.unique(CD_PM['season'])[3]))]
stats.f_oneway(Y1,Y2,Y3,Y4)
CD_PM['cbwd'].value_counts()
Y1=CD_PM[['PM_US Post']][(CD_PM['cbwd']==str(np.unique(CD_PM['cbwd'])[0]))]
Y2=CD_PM[['PM_US Post']][(CD_PM['cbwd']==str(np.unique(CD_PM['cbwd'])[1]))]
Y3=CD_PM[['PM_US Post']][(CD_PM['cbwd']==str(np.unique(CD_PM['cbwd'])[2]))]
Y4=CD_PM[['PM_US Post']][(CD_PM['cbwd']==str(np.unique(CD_PM['cbwd'])[3]))]
Y5=CD_PM[['PM_US Post']][(CD_PM['cbwd']==str(np.unique(CD_PM['cbwd'])[4]))]
stats.f_oneway(Y1,Y2,Y3,Y4,Y5)
Y=CD_PM[out_columns]
Y.corr()[['PM_US Post']]
regress.drop(['month','hour','PRES','precipitation','Iprec'],axis=1,inplace=True)
train=regress.sample(frac=0.7)
test=regress[~regress.index.isin(train.index)]
xtrain,ytrain=train.drop('PM_US Post',axis=1),train["PM_US Post"]
xtest,ytest=test.drop('PM_US Post',axis=1),test["PM_US Post"]
import statsmodels.api as sm
from sklearn.metrics import*
xtrain=sm.add_constant(xtrain)
xtest=sm.add_constant(xtest)
results=sm.OLS(ytrain,xtrain).fit()
print(results.summary())
ypred=results.predict(xtest)
yhat=results.predict(xtrain)
rmse_=mean_squared_error(ytrain,yhat)**0.5
frmse=mean_squared_error(ytest,ypred)**0.5
####################GZ
temp_list =GZ_PM[GZ_PM.HUMI < 0].index.tolist()
GZ_PM.drop(axis=0,labels=temp_list,inplace= True)
GZ_PM['HUMI'].fillna(GZ_PM['HUMI'].mean(),inplace=True)
GZ_PM['TEMP'].fillna(GZ_PM['TEMP'].mean(),inplace=True)
GZ_PM['DEWP'].fillna(GZ_PM['DEWP'].mean(),inplace=True)
GZ_PM['PRES'].fillna(GZ_PM['PRES'].mean(),inplace=True)
GZ_PM['Iws'].fillna(GZ_PM['Iws'].mean(),inplace=True)
GZ_PM['season'] = GZ_PM['season'].map({1:'Spring',2:'Summer',3:'Autumn',4:'Winter'})
GZ_PM.dropna(axis=0,how='any',subset=['PM_US Post','precipitation','Iprec','cbwd'],inplace= True)
GZ_PM.isnull().sum()
GZ_PM.info()
regress=pd.get_dummies(GZ_PM,columns=['cbwd','season'])
Y1=GZ_PM[['PM_US Post']][(GZ_PM['season']==str(np.unique(GZ_PM['season'])[0]))]
Y2=GZ_PM[['PM_US Post']][(GZ_PM['season']==str(np.unique(GZ_PM['season'])[1]))]
Y3=GZ_PM[['PM_US Post']][(GZ_PM['season']==str(np.unique(GZ_PM['season'])[2]))]
Y4=GZ_PM[['PM_US Post']][(GZ_PM['season']==str(np.unique(GZ_PM['season'])[3]))]
stats.f_oneway(Y1,Y2,Y3,Y4)
GZ_PM['cbwd'].value_counts()
Y1=GZ_PM[['PM_US Post']][(GZ_PM['cbwd']==str(np.unique(GZ_PM['cbwd'])[0]))]
Y2=GZ_PM[['PM_US Post']][(GZ_PM['cbwd']==str(np.unique(GZ_PM['cbwd'])[1]))]
Y3=GZ_PM[['PM_US Post']][(GZ_PM['cbwd']==str(np.unique(GZ_PM['cbwd'])[2]))]
Y4=GZ_PM[['PM_US Post']][(GZ_PM['cbwd']==str(np.unique(GZ_PM['cbwd'])[3]))]
Y5=GZ_PM[['PM_US Post']][(GZ_PM['cbwd']==str(np.unique(GZ_PM['cbwd'])[4]))]
stats.f_oneway(Y1,Y2,Y3,Y4,Y5)
Y=GZ_PM[out_columns]
Y.corr()[['PM_US Post']]
regress.drop(['month','hour','PRES','precipitation','Iprec'],axis=1,inplace=True)
train=regress.sample(frac=0.7)
test=regress[~regress.index.isin(train.index)]
xtrain,ytrain=train.drop('PM_US Post',axis=1),train["PM_US Post"]
xtest,ytest=test.drop('PM_US Post',axis=1),test["PM_US Post"]
import statsmodels.api as sm
from sklearn.metrics import*
xtrain=sm.add_constant(xtrain)
xtest=sm.add_constant(xtest)
results=sm.OLS(ytrain,xtrain).fit()
print(results.summary())
ypred=results.predict(xtest)
yhat=results.predict(xtrain)
rmse_=mean_squared_error(ytrain,yhat)**0.5
frmse=mean_squared_error(ytest,ypred)**0.5
##############SH
temp_list =SH_PM[SH_PM.HUMI < 0].index.tolist()
SH_PM.drop(axis=0,labels=temp_list,inplace= True)
SH_PM['HUMI'].fillna(SH_PM['HUMI'].mean(),inplace=True)
SH_PM['TEMP'].fillna(SH_PM['TEMP'].mean(),inplace=True)
SH_PM['DEWP'].fillna(SH_PM['DEWP'].mean(),inplace=True)
SH_PM['PRES'].fillna(SH_PM['PRES'].mean(),inplace=True)
SH_PM['Iws'].fillna(SH_PM['Iws'].mean(),inplace=True)
SH_PM['season'] = SH_PM['season'].map({1:'Spring',2:'Summer',3:'Autumn',4:'Winter'})
SH_PM.dropna(axis=0,how='any',subset=['PM_US Post','precipitation','Iprec','cbwd'],inplace= True)
SH_PM.isnull().sum()
SH_PM.info()
regress=pd.get_dummies(SH_PM,columns=['cbwd','season'])
Y1=SH_PM[['PM_US Post']][(SH_PM['season']==str(np.unique(SH_PM['season'])[0]))]
Y2=SH_PM[['PM_US Post']][(SH_PM['season']==str(np.unique(SH_PM['season'])[1]))]
Y3=SH_PM[['PM_US Post']][(SH_PM['season']==str(np.unique(SH_PM['season'])[2]))]
Y4=SH_PM[['PM_US Post']][(SH_PM['season']==str(np.unique(SH_PM['season'])[3]))]
stats.f_oneway(Y1,Y2,Y3,Y4)
SH_PM['cbwd'].value_counts()
Y1=SH_PM[['PM_US Post']][(SH_PM['cbwd']==str(np.unique(SH_PM['cbwd'])[0]))]
Y2=SH_PM[['PM_US Post']][(SH_PM['cbwd']==str(np.unique(SH_PM['cbwd'])[1]))]
Y3=SH_PM[['PM_US Post']][(SH_PM['cbwd']==str(np.unique(SH_PM['cbwd'])[2]))]
Y4=SH_PM[['PM_US Post']][(SH_PM['cbwd']==str(np.unique(SH_PM['cbwd'])[3]))]
Y5=SH_PM[['PM_US Post']][(SH_PM['cbwd']==str(np.unique(SH_PM['cbwd'])[4]))]
stats.f_oneway(Y1,Y2,Y3,Y4,Y5)
Y=SH_PM[out_columns]
Y.corr()[['PM_US Post']]
regress.drop(['month','hour','PRES','precipitation','Iprec'],axis=1,inplace=True)
train=regress.sample(frac=0.7)
test=regress[~regress.index.isin(train.index)]
xtrain,ytrain=train.drop('PM_US Post',axis=1),train["PM_US Post"]
xtest,ytest=test.drop('PM_US Post',axis=1),test["PM_US Post"]
import statsmodels.api as sm
from sklearn.metrics import*
xtrain=sm.add_constant(xtrain)
xtest=sm.add_constant(xtest)
results=sm.OLS(ytrain,xtrain).fit()
print(results.summary())
ypred=results.predict(xtest)
yhat=results.predict(xtrain)
rmse_=mean_squared_error(ytrain,yhat)**0.5
frmse=mean_squared_error(ytest,ypred)**0.5
##################SY
temp_list =SY_PM[SY_PM.HUMI < 0].index.tolist()
SY_PM.drop(axis=0,labels=temp_list,inplace= True)
SY_PM['HUMI'].fillna(SY_PM['HUMI'].mean(),inplace=True)
SY_PM['TEMP'].fillna(SY_PM['TEMP'].mean(),inplace=True)
SY_PM['DEWP'].fillna(SY_PM['DEWP'].mean(),inplace=True)
SY_PM['PRES'].fillna(SY_PM['PRES'].mean(),inplace=True)
SY_PM['Iws'].fillna(SY_PM['Iws'].mean(),inplace=True)
SY_PM['season'] = SY_PM['season'].map({1:'Spring',2:'Summer',3:'Autumn',4:'Winter'})
SY_PM.dropna(axis=0,how='any',subset=['PM_US Post','precipitation','Iprec','cbwd'],inplace= True)
SY_PM.isnull().sum()
SY_PM.info()
regress=pd.get_dummies(SY_PM,columns=['cbwd','season'])
Y1=SY_PM[['PM_US Post']][(SY_PM['season']==str(np.unique(SY_PM['season'])[0]))]
Y2=SY_PM[['PM_US Post']][(SY_PM['season']==str(np.unique(SY_PM['season'])[1]))]
Y3=SY_PM[['PM_US Post']][(SY_PM['season']==str(np.unique(SY_PM['season'])[2]))]
Y4=SY_PM[['PM_US Post']][(SY_PM['season']==str(np.unique(SY_PM['season'])[3]))]
stats.f_oneway(Y1,Y2,Y3,Y4)
SY_PM['cbwd'].value_counts()
Y1=SY_PM[['PM_US Post']][(SY_PM['cbwd']==str(np.unique(SY_PM['cbwd'])[0]))]
Y2=SY_PM[['PM_US Post']][(SY_PM['cbwd']==str(np.unique(SY_PM['cbwd'])[1]))]
Y3=SY_PM[['PM_US Post']][(SY_PM['cbwd']==str(np.unique(SY_PM['cbwd'])[2]))]
Y4=SY_PM[['PM_US Post']][(SY_PM['cbwd']==str(np.unique(SY_PM['cbwd'])[3]))]
Y5=SY_PM[['PM_US Post']][(SY_PM['cbwd']==str(np.unique(SY_PM['cbwd'])[4]))]
stats.f_oneway(Y1,Y2,Y3,Y4,Y5)
Y=SY_PM[out_columns]
Y.corr()[['PM_US Post']]
regress.drop(['month','hour','PRES','precipitation','Iprec'],axis=1,inplace=True)
train=regress.sample(frac=0.7)
test=regress[~regress.index.isin(train.index)]
xtrain,ytrain=train.drop('PM_US Post',axis=1),train["PM_US Post"]
xtest,ytest=test.drop('PM_US Post',axis=1),test["PM_US Post"]
import statsmodels.api as sm
from sklearn.metrics import*
xtrain=sm.add_constant(xtrain)
xtest=sm.add_constant(xtest)
results=sm.OLS(ytrain,xtrain).fit()
print(results.summary())
ypred=results.predict(xtest)
yhat=results.predict(xtrain)
rmse_=mean_squared_error(ytrain,yhat)**0.5
frmse=mean_squared_error(ytest,ypred)**0.5

#第四题
import os
import pandas as pd
import numpy as np
os.chdir('d:\\workpath')
AB=pd.read_csv('AB_NYC_2019处理.csv')
AB.isnull().sum()
AB.drop(['Unnamed: 9','Unnamed: 10','Unnamed: 11'],axis=1,inplace=True)
AB.info()
AB['reviews_per_month'].fillna(0,inplace=True)
AB['neighbourhood_group'].value_counts()
AB['room_type'].value_counts()
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
##地区分布、房间类型分布
fig, axes = plt.subplots(1,2, figsize=(12,4))
sns.catplot(x='neighbourhood_group', kind='count', data=AB,ax=axes[0])
sns.catplot(x='room_type', kind='count', data=AB,ax=axes[1])
ctable=pd.pivot_table(AB,index=['room_type'],values=['price'],columns=['neighbourhood_group'],aggfunc=[np.count_nonzero])
ctable.plot()
#列联表卡方检验(拒绝原假设)有影响
stats.chi2_contingency([ctable.iloc[0,1:],ctable.iloc[0,1:]])
##价格
fig, axes = plt.subplots(1,3, figsize=(15,4))
sns.catplot(x='neighbourhood_group', y='price', kind='violin',data=AB,ax=axes[0])
sns.catplot(x='room_type', y='price', kind='violin',data=AB,ax=axes[1])
sns.catplot(x='neighbourhood_group', y='price',hue='room_type', data=AB,ax=axes[2])
##拥有房间数
fig, axes = plt.subplots(1,3, figsize=(15,4))
sns.catplot(x='neighbourhood_group', y='calculated_host_listings_count', kind='violin',data=AB,ax=axes[0])
sns.catplot(x='room_type', y='calculated_host_listings_count', kind='violin',data=AB,ax=axes[1])
sns.catplot(x='neighbourhood_group', y='calculated_host_listings_count',hue='room_type', data=AB,ax=axes[2])
##每月评论数
fig, axes = plt.subplots(1,3, figsize=(15,4))
sns.catplot(x='neighbourhood_group', y='reviews_per_month', kind='violin',data=AB,ax=axes[0])
sns.catplot(x='room_type', y='reviews_per_month', kind='violin',data=AB,ax=axes[1])
##相关
corrmatrix = AB.corr()
f, ax = plt.subplots(figsize=(12,6))
sns.heatmap(corrmatrix, vmax=1, square=True,annot=True)

