# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:09:08 2019

@author: asus
"""

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
