# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
os.chdir('d:\\workpath')
stockprice=pd.read_table('stockprice.txt',header=None)
stockprice.columns=['date','open','high','low','close','amt','vol']
stockprice.set_index('date',inplace=True)
stockprice.index=pd.to_datetime(stockprice.index)
creditcard=pd.read_csv('creditcard.csv')
creditcard_subset1=creditcard[(creditcard["SEX"]==2)]
creditcard_subset2=creditcard[['AGE','MARRIAGE','LIMIT_BAL']][(creditcard["SEX"]==2)&(creditcard["LIMIT_BAL"]>50000)]
stockprice["2017-1-1":"2017-1-31"]["close"]
stockprice["2017-1-1":]["close"]
stockprice[:"2016-12-31"]["close"]
df1=stockprice.sample(frac=0.1)
df2=stockprice.sample(frac=0.1)
df=pd.concat([df1,df2])
df1=stockprice[['open']]
df2=stockprice[['close']]
df3=df1.merge(df2,on=df1.index)
import numpy as np
creditcard.mean()
creditcard.median()
creditcard.mode()
creditcard.quantile(0.2)
creditcard['LIMIT_BAL'].mean()
creditcard[['LIMIT_BAL','BILL_AMT1']].mean()
creditcard.var()
creditcard.std()
creditcard.max()-creditcard.min()
creditcard.quantile(0.75)-creditcard.quantile(0.25)
creditcard.std()/creditcard.mean()
import matplotlib.pyplot as plt
plt.boxplot(creditcard['AGE'])
creditcard.describe()
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sbn##数据可视化的包
creditcard.skew()##计算数据的偏度
creditcard.kurt()##计算数据集的峰度
plt.hist(creditcard['LIMIT_BAL'],bins=50)#(密率分布)；bins参数（组数）
sbn.kdeplot(creditcard['LIMIT_BAL']) 
import statsmodels.api as sm
import scipy.stats as stats##python科学计算的包,stats统计模块
sm.qqplot(creditcard['LIMIT_BAL'])
sm.qqplot(creditcard['LIMIT_BAL'],stats.chi2,distargs=(4,))##distargs(4,1)卡方四分布
#######分组统计
##pandas 的groupby
gp1=creditcard.groupby('EDUCATION')
gp2=creditcard.groupby(['EDUCATION','SEX'])
gp3=creditcard.groupby('EDUCATION')['LIMIT_BAL']
gp2.size();gp3.mean();gpstats=gp3.agg(['mean','std','median','max','min'])
def jbstats(df):
    N,S,K=len(df),df.skew(),df.kurtosis()
    JB=(S**2+(K-3)**2/4)*N/6
    return JB
groupstats=gp3.agg([jbstats])

mypivot=pd.pivot_table(creditcard,index=['SEX','MARRIAGE'],values=['LIMIT_BAL'],columns=['EDUCATION'],aggfunc=(np.mean))


wine=pd.read_csv('wine.csv')
X=wine.drop(['Y'],axis=1)##斯皮尔曼相关系数和，待查
X.cov()
X.corr()
X.corr(method='spearman')
X.plot(kind='scatter',x='X1',y='X2')

############
#########
#####
##假设检验
#单总体均值假设检验
import scipy.stats as stats
x1=wine['X1']
x1.mean()
stats.ttest_1samp(x1,6.86)
stats.ttest_1samp(x1,6.9)
u=wine['X1'][wine['Y']==7]
v=wine['X1'][wine['Y']==9]
stats.ttest_ind(u,v)
y1=creditcard[['LIMIT_BAL']][(creditcard["MARRIAGE"]==1)]
y2=creditcard[['LIMIT_BAL']][(creditcard["MARRIAGE"]==2)]
y3=creditcard[['LIMIT_BAL']][(creditcard["MARRIAGE"]==3)]
stats.f_oneway(y1,y2,y3)
a1=creditcard[['LIMIT_BAL']][(creditcard["SEX"]==1)]
a2=creditcard[['LIMIT_BAL']][(creditcard["SEX"]==2)]
stats.f_oneway(a1,a2)
b1=creditcard[['LIMIT_BAL']][(creditcard["EDUCATION"]==1)]
b2=creditcard[['LIMIT_BAL']][(creditcard["EDUCATION"]==2)]
b3=creditcard[['LIMIT_BAL']][(creditcard["EDUCATION"]==3)]
stats.f_oneway(b1,b2,b3)
u=wine['X1'][wine['Y']==7]
v=wine['X1'][wine['Y']==6]
stats.kstest(v,'norm')
stats.kstest(v,'t',(10,))
stats.ks_2samp(u,v)
male=creditcard['LIMIT_BAL'][creditcard['SEX']==1]
female=creditcard['LIMIT_BAL'][creditcard['SEX']==2]
stats.mannwhitneyu(male,female)
ctable=pd.pivot_table(creditcard,index=['y'],values=['LIMIT_BAL'],columns=['EDUCATION'],aggfunc=[np.count_nonzero])
stats.chi2_contingency([ctable.iloc[0,1:],ctable.iloc[1,1:]])
x1,x2=wine['X1'],wine['X2']
stats.kendalltau(x1,x2)
stats.spearmanr(x1,x2)
#for循环记得编写
#主成分分析
from sklearn.decomposition import PCA
wine=pd.read_csv('wine.csv')
wine_=wine.iloc[:,0:11]
wine_pac=PCA（n_components=1).fit_transform(wine_)
PCA(n_components=1).fit(wine_).explained_variance_ratio_.sum()
wine_pac=PCA（n_components=2).fit_transform(wine_)
PCA(n_components=2).fit(wine_).explained_variance_ratio_.sum()
#因子分析
from sklearn.decomposition import FactorAnalysis
fa=FactorAnalysis(n_components=2).fit_transform(wine_)
score=FactorAnalysis(n_components=2).fit(wine_).components_
#典型相关分析
X=wine.drop(['Y'],axis=1)
X_=X[['X1','X2','X3','X4']]
Y_=X[['X5','X6','X7','X8','X9','X10','X11']]
from sklearn.cross_decomposition import CCA
cca=CCA(n_components=2)
X_c,Y_c=cca.fit_transform(X_,Y_)#两个典型相关分析
#第一典型相关成分的相关系数矩阵
np.corrcoef(X_c[:,0],Y_c[:,0],rowvar=False)
#第二典型相关成分的相关系数矩阵
np.corrcoef(X_c[:,1],Y_c[:,1],rowvar=False)

##机器学习
##定义训练集，验证集，测试集
train=creditcard.sample(frac=0.5)
rest=creditcard[~creditcard.index.isin(train.index)]
validation=rest.sample(frac=0.5)
test=rest[~creditcard.index.isin(validatoin.index)]
##文字向量
##1.分词（解霸等）2.词，向量化（200维），3.
##数据审计
##数据转换
###缺失值处理
################
##############
#########
#####
#分类变量的处理(转变为虚拟变量)
creditcard2=pd.get_dummies(creditcard,columns=['SEX','EDUCATION'])
###数据归一化（0-1标准化或z-score标准化）
from sklearn import preprocessing
minmax_scaler=preprocessing.MinMaxScaler()
zscore_scaler=preprocessing.StandardScaler()
##对amt和vol进行标准化处理
temp=stockprice.loc[:,['amt','vol']]
stockprice.loc[:,['amt','vol']]=minmax_scaler.fit_transform(temp)
stockprice.loc[:,['amt','vol']]=zscore_scaler.fit_transform(temp)
#########回归分析
import os
import pandas as pd
os.chdir('d:\\workpath')
BLOG=pd.read_csv("blogData_train.csv",header=None)
ColName=['X'+str(k+1) for k in np.arange(280)]
ColName.append('Y')
BLOG.columns=ColName
ytrain=BLOG['Y']
Xtrain=BLOG.drop('Y',axis=1)
from sklearn import linear_model
mdl_ols=linear_model.LinearRegression()
mdl_ols.fit(Xtrain,ytrain)
coef_ols=mdl_ols.coef_
yhat=mdl_ols.predict(Xtrain)
####用测试集进行预测
BLOG_Test=pd.read_csv("blogData_test.csv",header=None)
BLOG_Test.columns=ColName
Xtest=BLOG_Test.drop('Y',axis=1)
ytest=BLOG_Test['Y']
ypred=mdl_ols.predict(Xtest)
##回归模型的评价
from sklearn.metrics import*
rmse=mean_squared_error(ytrain,yhat)**0.5
frmse=mean_squared_error(ytest,ypred)**0.5
r2_score(ytrain,yhat)
r2_score(ytest,ypred)
###回归示例
house=pd.read_csv("housing.csv")
house['age']=2014-house["yr_built"]
house.drop("yr_built",axis=1,inplace=True)
house=pd.get_dummies(house,columns=['waterfront','view','grade','condition'])
train=house.sample(frac=0.7)
test=house[~house.index.isin(train.index)]
xtrain,ytrain=train.drop('price',axis=1),train["price"]
xtest,ytest=test.drop('price',axis=1),test["price"]
import statsmodels.api as sm
xtrain=sm.add_constant(xtrain)
xtest=sm.add_constant(xtest)
results=sm.OLS(ytrain,xtrain).fit()
print(results.summary())
ypred=results.predict(xtest)

##过度拟合
xtrain_=xtrain.drop(['view_2','grade_1','condition_1','condition_2'],axis=1)
xtest_=xtest.drop(['view_2','grade_1','condition_1','condition_2'],axis=1)
import statsmodels.api as sm
xtrain=sm.add_constant(xtrain)
xtest=sm.add_constant(xtest)
xtrain_=sm.add_constant(xtrain_)
xtest_=sm.add_constant(xtest_)
results=sm.OLS(ytrain,xtrain).fit()
results_=sm.OLS(ytrain,xtrain_).fit()
print(results.summary())
yhat=results.predict(xtrain)
ypred=results.predict(xtest)
yhat_=results_.predict(xtrain_)
ypred_=results_.predict(xtest_)
rmse=mean_squared_error(ytrain,yhat)**0.5
frmse=mean_squared_error(ytest,ypred)**0.5
rmse=mean_squared_error(ytrain,yhat_)**0.5
frmse=mean_squared_error(ytest,ypred_)**0.5
##统计分类
import os
import pandas as pd
os.chdir('d:\\workpath')
df=pd.read_csv("HR_comma_sep.csv")
df=pd.get_dummies(df,columns=['position','salary'])
train=df.sample(frac=0.7)
train=df.sample(frac=0.7)
test=df[~df.index.isin(train.index)]
xtrain,ytrain=train.drop('left',axis=1),train['left']
xtest,ytest=test.drop('left',axis=1),test['left']
from sklearn.neighbors import KNeighborsClassifier
mdl_knn=KNeighborsClassifier(n_neighbors=5)
mdl_knn.fit(xtrain,ytrain)
yhat=mdl_knn.predict(xtrain)
ypred=mdl_knn.predict(xtest)

##提高分类效果
mdl_wknn=KNeighborsClassifier(n_neighbors=5,weights='distance')#加权knn
mdl_wknn.fit(xtrain,ytrain)
yhat=mdl_knn.predict(xtrain)
ypred=mdl_knn.predict(xtest)

mdl_wknn=KNeighborsClassifier(n_neighbors=5,weights='distance',p=2)#加权knn,曼哈顿距离
mdl_wknn.fit(xtrain,ytrain)
yhat=mdl_wknn.predict(xtrain)
ypred=mdl_wknn.predict(xtest)

mdl_wknn=KNeighborsClassifier(n_neighbors=5,weights='distance',p=1)#加权knn,曼哈顿距离
mdl_wknn.fit(xtrain,ytrain)
yhat=mdl_wknn.predict(xtrain)
ypred=mdl_wknn.predict(xtest)

#分类算法的性能评价
##(knn,加权knn，p=1的情况，p=2的情况（k的选择）)
##演示代码，四种算法的指标孰高孰低
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print(accuracy_score(ytrain,yhat))##训练集准确率
print(confusion_matrix(ytest,ypred))##测试集混淆矩阵
print(classification_report(ytest,ypred))##测试集分类报告
#############新的选做题，搜集各种数据研究猪肉的生产需求，国家如何干预，何时期稳，选座作业
###AUC预测优度测定
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt#加载绘图模块
ypred_prob=mdl_knn.predict_proba(xtest)#计算预测概率
fpr,tpr,_=roc_curve(ytest,ypred_prob[:,1])
plt.plot(fpr,tpr,color='darkorange',lw=2)
auc=roc_auc_score(ytest,ypred_prob[:,1])
import numpy as np
from sklearn.model_selection import cross_val_score
cross_val_score(mdl_knn,xtrain,ytrain,cv=5)
np.mean(cross_val_score(mdl_knn,xtrain,ytrain,cv=5))
from sklearn.model_selection import GridSearchCV 

parameters={'n_neighbors':[1,3,5,7,9,11],
            'p':[1,2]}
knn=KNeighborsClassifier()
knn_grid=GridSearchCV(knn,parameters,cv=5)
knn_grid.fit(xtrain,ytrain)
knn_grid.best_params_
knn_grid.best_score_
#逻辑回归
from sklearn.linear_model import LogisticRegression
mdl_logit=LogisticRegression(penalty='none',solver='lbfgs')
mdl_logit.fit(xtrain,ytrain)
yhat=mdl_logit.predict(xtrain)
ypred=mdl_logit.predict(xtest)
print(accuracy_score(ytrain,yhat))
print(accuracy_score(ytest,ypred))
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt#加载绘图模块
ypred_prob=mdl_logit.predict_proba(xtest)#计算预测概率
fpr,tpr,_=roc_curve(ytest,ypred_prob[:,1])
plt.plot(fpr,tpr,color='darkorange',lw=2)#横轴和纵轴为假的准确率与的真的准确率
auc=roc_auc_score(ytest,ypred_prob[:,1])
#样本不平衡问题,多了weighted
logit=LogisticRegression(penalty='none',solver='lbfgs',class_weight='balanced')
logit.fit(xtrain,ytrain)
print(accuracy_score(ytrain,yhat))
print(accuracy_score(ytest,ypred))
print(classification_report(ytest,ypred))
#编一个程序，输入是一个数据集，输出还是一个数据集，确保0，1的数目完全一样

#解决样本的不平衡问题可以提高召回率，但不一定能提高准确率
#LogisticRegression还可以基于softmax损失函数直接估计多类别Logistic模型

#OVR和Multinormal为两种方法
df=pd.read_csv("wine.csv")
df=df[(df['Y']<8)&(df['Y']>4)]#只选择其中3类
train=df.sample(frac=0.7)
test=df[~df.index.isin(train.index)]
xtrain,ytrain=train.drop('Y',axis=1),train['Y']
xtest,ytest=test.drop('Y',axis=1),test['Y']
mlogit=LogisticRegression(penalty='none',solver='lbfgs',max_iter=500,multi_class='ovr')
mlogit_=LogisticRegression(penalty='none',solver='lbfgs',max_iter=500,multi_class='multinomial')
mlogit.fit(xtrain,ytrain)
mlogit_.fit(xtrain,ytrain)
yhat=mlogit.predict(xtrain)
ypred=mlogit.predict(xtest)
print(accuracy_score(ytrain,yhat))
print(accuracy_score(ytest,ypred))
print(classification_report(ytest,ypred))

#程序计时（手机拍照）


#朴素贝叶斯判别方法,离散的多就用bernoulliNB,计算非常快
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
gnb,mnb,bnb=GaussianNB(),MultinomialNB(),BernoulliNB()
gnb.fit(xtrain,ytrain)
mnb.fit(xtrain,ytrain)
bnb.fit(xtrain,ytrain)
yhat,yhat_,yhat__=gnb.predict(xtrain),mnb.predict(xtrain),bnb.predict(xtrain)
ypred,ypred_,ypred__=gnb.predict(xtest),mnb.predict(xtest),bnb.predict(xtest)
print(accuracy_score(ytrain,yhat))#训练集准确率
print(accuracy_score(ytrain,yhat_))#训练集准确率
print(accuracy_score(ytrain,yhat__))#训练集准确率
print(accuracy_score(ytest,ypred))#测试集准确率
print(accuracy_score(ytest,ypred_))#测试集准确率
print(accuracy_score(ytest,ypred__))#测试集准确率
#额外作业（提取K线特征，找到相似k线特征），选做，3组数据哪组都可以
#决策树，分类算基尼系数（离散型），越小越好
#连续型数值处理引进分裂点思想，假设样本集中某个属性共n个连续值，有n-1个分裂点，每个分裂点为相邻两个连续值的均值，当数过多时，用直方图，缩小划分区间
#每一次分类操作均相同
#好的模型必须处理过度拟合问题
#对于决策树来说，要进行剪枝
from sklearn import tree
mdl_tree=tree.DecisionTreeClassifier(max_depth=3)
mdl_tree.fit(xtrain,ytrain) 
yhat=mdl_tree.predict(xtrain)
ypred=mdl_tree.predict(xtest)
print(accuracy_score(ytrain,yhat))
print(accuracy_score(ytest,ypred))
print(classification_report(ytest,ypred))
#找出最优的max_depth参数
parameters={'max_depth':[3,4,5,6,7,8,9,10,11,12]}
tree_grid=GridSearchCV(mdl_tree,parameters,cv=5)
tree_grid.fit(xtrain,ytrain)
tree_grid.cv_results_
#报告做决策树可视化，报告加分

#聚类分析
from sklearn import cluster
df=pd.read_csv('rnadata.csv')
labels=pd.read_csv('rnalabels.csv')
df.drop(df.columns[0],axis=1,inplace=True)
truelabel=labels['Class']
kmeans=cluster.KMeans(n_clusters=5).fit(df)
kmeans.labels_#观察聚类分析后的类标签
kmeans_minibatch=cluster.MiniBatchKMeans(n_clusters=5,batch_size=50).fit(df)
#层次分析,更新sklearn，davies在高版本sklearn才有
mdl_hc=cluster.AgglomerativeClustering(n_clusters=5).fit(df)
metrics.davies_bouldin_score(df,mdl_hc.labels_)
metrics.adjusted_rand_score(truelabel,mdl_hc.labels_)
#聚类指标评价
from sklearn import metrics
labels,labels_=kmeans.labels_,kmeans_minibatch.labels_
metrics.davies_bouldin_score(df,labels)
metrics.davies_bouldin_score(df,labels_)
metrics.silhouette_score(df,labels)
metrics.silhouette_score(df,labels_)
metrics.adjusted_rand_score(truelabel,labels)
metrics.adjusted_rand_score(truelabel,labels_)
metrics.adjusted_mutual_info_score(truelabel,labels)
metrics.adjusted_mutual_info_score(truelabel,labels_)


#python期末作业
#1.Suicide rates overview
import os
import pandas as pd
os.chdir('d:\\workpath')
suicide=pd.read_csv('suicide.csv',header=None)
import statsmodels.api as sm
import scipy.stats as stats##python科学计算的包,stats统计模块
sm.qqplot(creditcard['LIMIT_BAL'])
sm.qqplot(creditcard['LIMIT_BAL'],stats.chi2,distargs=(4,))##distargs(4,1)卡方四分布
#######分组统计
##pandas 的groupby
gp1=creditcard.groupby('EDUCATION')
gp2=creditcard.groupby(['EDUCATION','SEX'])
gp3=creditcard.groupby('EDUCATION')['LIMIT_BAL']
gp2.size();gp3.mean();gpstats=gp3.agg(['mean','std','median','max','min'])
def jbstats(df):
    N,S,K=len(df),df.skew(),df.kurtosis()
    JB=(S**2+(K-3)**2/4)*N/6
    return JB
groupstats=gp3.agg([jbstats])

mypivot=pd.pivot_table(creditcard,index=['SEX','MARRIAGE'],values=['LIMIT_BAL'],columns=['EDUCATION'],aggfunc=(np.mean))



##实习
#GBDT算法（是迭代，但是 GBDT 要求弱学习器必须是 CART 模型，而且 GBDT 在模型训练的时候，
#是要求模型预测的样本损失尽可能的小。）
#每一轮预测和实际值有残差，下一轮根据残差再进行预测，最后将所有预测相加，就是结果。
creditcard.avebill=creditcard['BILL_AMT1']/6+creditcard['BILL_AMT2']/6+creditcard['BILL_AMT3']/6+creditcard['BILL_AMT4']/6+creditcard['BILL_AMT5']/6+creditcard['BILL_AMT6']/6
creditcard['BILL_AVE']=creditcard.avebill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from __future__ import division, print_function
pip install progressbar2
import progressbar
from utils import train_test_split, standardize, to_categorical
from utils import mean_squared_error, accuracy_score, Plot
from utils.loss_functions import SquareLoss
from utils.misc import bar_widgets
from gradient_boosting_decision_tree.gbdt_model import GBDTRegressor
train=creditcard.sample(frac=0.7)
test=creditcard[~creditcard.index.isin(train.index)]
##广义高斯分布与SHAP
##广义高斯模型

import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from scipy.optimize import curve_fit
#用三个不同的高斯分布生成三个聚类作为GMM算法的数据
num1, mu1, covar1 = 400, [0.5, 0.5], np.array([[1,0.5],[0.5,3]])
X1 = np.random.multivariate_normal(mu1, covar1, num1)
# 第二簇的数据
num2, mu2, covar2 = 600, [5.5, 2.5], np.array([[2,1],[1,2]])
X2 = np.random.multivariate_normal(mu2, covar2, num2)
# 第三簇的数据
num3, mu3, covar3 = 1000, [1, 7], np.array([[6,2],[2,1]])
X3 = np.random.multivariate_normal(mu3, covar3, num3)
# 合并在一起
X = np.vstack((X1, X2, X3))
plt.figure(figsize=(10, 8))
plt.axis([-10, 15, -5, 15])
plt.scatter(X1[:, 0], X1[:, 1], s=5)
plt.scatter(X2[:, 0], X2[:, 1], s=5)
plt.scatter(X3[:, 0], X3[:, 1], s=5)
#定义高斯混合模型
##进行拟合
#SHAP找出交互作用（基于GaussianNB,MultinomialNB,BernoulliNB进行判别）
##pip install shap
##？？下载不了，甚至跑不了别的代码
import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
from sklearn.model_selection import train_test_split

##时间序列建模
##降维考虑因子或者聚类
from sklearn.decomposition import PCA
wine=pd.read_csv('wine.csv')
wine_=wine.iloc[:,0:11]
wine_pac=PCA（n_components=1).fit_transform(wine_)
PCA(n_components=1).fit(wine_).explained_variance_ratio_.sum()
wine_pac=PCA（n_components=2).fit_transform(wine_)
PCA(n_components=2).fit(wine_).explained_variance_ratio_.sum()
#因子分析
from sklearn.decomposition import FactorAnalysis
fa=FactorAnalysis(n_components=2).fit_transform(wine_)
score=FactorAnalysis(n_components=2).fit(wine_).components_
##简单聚类，之后要结合时间序列数据，运用多种聚类方法进行比较，并且要提前对时间序列数据进行处理
from sklearn import cluster
df=pd.read_csv('rnadata.csv')
labels=pd.read_csv('rnalabels.csv')
df.drop(df.columns[0],axis=1,inplace=True)
truelabel=labels['Class']
kmeans=cluster.KMeans(n_clusters=5).fit(df)
kmeans.labels_#观察聚类分析后的类标签
kmeans_minibatch=cluster.MiniBatchKMeans(n_clusters=5,batch_size=50).fit(df)
mdl_hc=cluster.AgglomerativeClustering(n_clusters=5).fit(df)
metrics.davies_bouldin_score(df,mdl_hc.labels_)
metrics.adjusted_rand_score(truelabel,mdl_hc.labels_)
#聚类指标评价（此处用两种方法：k-means和MiniBatchk-means)
from sklearn import metrics
labels,labels_=kmeans.labels_,kmeans_minibatch.labels_
metrics.davies_bouldin_score(df,labels)
metrics.davies_bouldin_score(df,labels_)
metrics.silhouette_score(df,labels)
metrics.silhouette_score(df,labels_)
metrics.adjusted_rand_score(truelabel,labels)
metrics.adjusted_rand_score(truelabel,labels_)
metrics.adjusted_mutual_info_score(truelabel,labels)
metrics.adjusted_mutual_info_score(truelabel,labels_)

input("\n\n按下enter键后退出")
#!/usr/bin/python3
 
import sys; x = 'runoob'; sys.stdout.write(x + '\n')
x="a"
y="b"
# 换行输出
print( x )
print( y )
 
print('---------')
# 不换行输出
print( x, end=" " )
print( y, end=" " )
print()

##小插曲，计算余响的期望
e=1.7*0.36+1.35*0.64*0.56+3.7/3*0.64*0.44*0.76+4.7/4*0.64*0.44*0.24*0.96+5.7/5*0.64*0.44*0.24*0.04
p_total=0.36+0.64*0.56+0.64*0.44*0.76+0.64*0.44*0.24*0.96+0.64*0.44*0.24*0.04
print(e)

names = "{}, {}, {}, {}".format('John', 'Bill', 'Sean') 

print(names)
x = 74

x /= 2

print(x)
print(3 == 3.0 and bool('0') or -2**2 != 4)

sd=1
c1 = "# You can't see this #1" # You can't see this #2

print(c1)

names = "{}, {}, {}, {}".format('John', 'Bill', 'Sean') 

print(names)

x = input('input a int')
x *= 2

print(x)
x = 74

x /= 2

print(x)
print(3 == 3.0 and bool('0') or -2**2 != 4)
_jkl
x = 5

y = 10

x, y = y, x

print(x, y)

if x==10:

    print ("Hello World!")

elif x!=10:

    print ("Hello World!")

if x==10:

    print ("Hello World!")

_sd=1
print(_sd)
2ad=1
$tyu
@we=1
if=1

sd="\tsdf"
print(sd)
sd1="sde"
print(sd1)
sd=sd.strip()
print(sd)
print(sd[2])
x=input("s")
x=x.strip()
print(x)
x=200
import math
y=math.sqrt(x)
print(y)

lst=["cat","dog","tiger","bird"]
lst.append("dog")
pos=lst.index("dog")
lst.insert(pos,"another dog")
print(lst)

lst1=["one","two","three"]
lst1[0]="zero"
a=lst1.pop(0)
print(a)
print(lst1)
list[1][1]="x"
print(lst1)

list1=[[1,2,3]]
list2=list1
list3=list1.copy()
list1[0][0]=2
print(list1)
print(list2,end="")
print(list3)

def mod_lis(lst,x):
    x-=2
    lst[1]=x

x=5
lst=[3,4,5]
mod_lis(lst,x)
mod_lis(lst,x)
print(lst)
lst=['a','b','z','c','d']
lst[2] = [] 
lst[2:3] = []
lst.remove('z') 
del lst[2] 
lst[2:2] = [] 
print[lst]
import numpy as np
print(np.random.random())
x=np.random.random()
list_sample=[]
for k in range(0,51):
    list_sample.append([])
for i in range(1,1001):
    for j in range(1,51):
        x=0
        n=0
        judge=0
        while(n<j):
            judge=np.random.random()
            n+=1
            if judge<0.5:
                x+=1
            else:
                x-=1
        list_sample[j].append(x)
liz=[1,2,3,4]
x=max(liz)
print(x)

import os
import pandas as pd
os.chdir('d:\\workpath')
data_exponential=pd.read_csv('data_HW2.csv'，head=)
import numpy as np
sample_mean_built_in=data_exponential.mean()
sample_variance_built_in=data_exponential.var()
print(sample_variance_built_in)
import pandas as pd
data_exponential=pd.read_csv("d:/workpath/data_HW2.csv",names=["exponential_data"])
calculate_list=data_exponential["exponential_data"]
# calculate sample mean
sum=0
for i in range(0,len(calculate_list)):
    sum+=calculate_list[i]
sample_mean_calculate=sum/len(calculate_list)
print(sample_mean_calculate)
# calculate sample variance
sum_variance=0
for i in range(0,len(calculate_list)):
    sum_variance+=(calculate_list[i]-sample_mean_calculate)**2
sample_variance_calculate=sum_variance/(len(calculate_list)-1)
print(sample_variance_calculate)





import pandas as pd
data_exponential=pd.read_csv("d:/workpath/data_HW2.csv",names=["exponential_data"])
calculate_list=data_exponential["exponential_data"]
len(calculate_list)
# calculate sample mean
sum=0
for i in range(0,len(calculate_list)):
    sum+=calculate_list[i]
sample_mean_calculate=sum/len(calculate_list)
print(sample_mean_calculate)
# calculate sample variance
sum_variance=0
for i in range(0,len(calculate_list)):
    sum_variance+=(calculate_list[i]-sample_mean_calculate)**2
sample_variance_calculate=sum_variance/(len(calculate_list)-1)
print(sample_variance_calculate)

x1=open("celebrity_deaths_2016.xlsx")
import numpy as np
mu=np.arange(-10,10,0.01)
