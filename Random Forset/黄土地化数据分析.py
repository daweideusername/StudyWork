import numpy as np
import scipy
import pandas as pd #读取数据
from sklearn.ensemble import RandomForestClassifier #随机森林分类器
from sklearn.tree import DecisionTreeClassifier #决策树分类器 (我暂时不需要)
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV #数据分割
from sklearn.metrics import roc_curve,auc,roc_auc_score #评估标准
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV #交叉验证
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt #生成图

# 读取 xlsx 文件
data = pd.read_excel('AllData.xlsx',sheet_name=0)#第一页源区数据 - 0

category = data['Category']
category = category.drop(0,axis=0) #删去第一行空白行 - 0,  0 - x轴

#对某列数据进行分析
counts = data['Category'].value_counts()
# print(counts)
#数据映射
data['Category'] = data['Category'].astype(str).map({'Junggar Basin':1,
                                                     'Tarim Basin':2,
                                                     'North Alxa Plateau':3,
                                                     'Qaidam Basin':4,
                                                     'northeastern sandy deserts':5,
                                                     'Hetao Graben':6,
                                                     'South Alxa Plateau':7,
                                                     'eastern Tibetan Plateau':8})

#数据整理:去掉样品源区,去掉经纬度,去掉样品序号
x = data.drop('Category',axis=1) #1 - y轴
x = x.drop('Longitude',axis=1)
x = x.drop('Latitude',axis=1)
x = x.drop('No.',axis=1) #(列)
x = x.drop(0,axis=0) #去掉单位那一行 (当前的第一行)

# print(x)
# print(data)

#训练集和测试集分割!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#x是清理好后的数据,y是结果数据
y = category
seed = 5
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state=seed)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#这个比例可以调,然后增大正确率
#训练集0.8,  测试集0.2
"""
seed = 5 #结果有幅度变化
0.3 --- 0.7387387387387387
0.4 --- 0.7491525423728813
0.45 --- 0.7289156626506024
0.5 --- 0.7073170731707317
"""

#===================================== 随机森林分类器的使用 ========================================
rfc = RandomForestClassifier() #实例化
rfc = rfc.fit(xtrain,ytrain) #用训练集数据训练模型 (接口fit)

result = rfc.score(xtest,ytest) #导入测试集,rfc的接口score计算的是模型的准确率accuracy

print(result) #得出结果
print('所有的树:%s' % rfc.estimators_) #显示森林的所有树
print(rfc.classes_) # 显示类别(物源地)
print(rfc.n_classes_) # 分类数量

# print('判定结果: %s' % rfc.predict(xtest))
# print('判定结果: %s' % rfc.predict_proba(xtest)[:,:])
# print('判定结果: %s' % rfc.predict_proba(xtest)[:,0]) #0 1 2 3 4 ...分别是之前定义的物源的可能性
# print('各特征的重要性 : %s' %rfc.feature_importances_) #重要性
# importances = rfc.feature_importances_ #对重要性排序
# print(np.argsort(importances))



importances = rfc.feature_importances_ # 显示各元素的特征重要性值的大小

std = np.std([tree.feature_importances_ for tree in rfc.estimators_],axis=0)

indices = np.argsort(importances)[::-1] #对重要性从大到小排序
print('Feature ranking: ')
for f in range(min(20,xtrain.shape[1])): #对前20个元素的重要性进行排序显示
    print("%2d) %-*s %f" %(f + 1, 30, xtrain.columns[indices[f]],importances[indices[f]])) #

#生成元素重要性图片
plt.figure()
plt.title("Feature importances")
plt.bar(range(xtrain.shape[1]),importances[indices], color="r",yerr=std[indices],align="center")
plt.xticks(range(xtrain.shape[1]),indices)
plt.xlim([-1,xtrain.shape[1]])
plt.show()


#交叉验证:
#决策树 -- 之后删去
# clf = DecisionTreeClassifier(max_depth=None,min_samples_split=2,random_state=0)
# scores = cross_val_score(clf,xtrain,ytrain)
# print(scores.mean())
#随机森林
#计算得到结果 n_estimators = 350 准确率最高  0.77
clf = RandomForestClassifier(n_estimators=350,max_depth=None,min_samples_split=2,random_state=0)
scores = cross_val_score(clf,xtrain,ytrain)
print(scores.mean())





#超参数调优:(29:00)


#利用网格搜索,交叉验证
# param_test1 = {'n_estimators': range(25,500,25)}#对分类器数量进行调优
# gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,#将要分割的样本数不能小于100
#                                                            min_samples_leaf=20,#分割出来的不能小于20
#                                                            max_depth=8,
#                                                            random_state=10), #分类器
#                         param_grid=param_test1,
#                         scoring = 'accuracy',#{'Accuracy': make_scorer(accuracy_score), 'AUC': make_scorer(roc_auc_score, multi_class='ovo')},#scoring='roc_auc',# scoring='roc_auc',#性能评估用roc分数 --- 适合二元分类问题
#                         cv=5)#交叉验证的折数 5折
#
# gsearch1.fit(xtrain, ytrain)
# print(gsearch1.best_params_, gsearch1.best_score_)


param_test2 = {'min_samples_split':range(60,200,20),'min_samples_leaf':range(10,110,10)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=350,
                                                           max_depth=8,
                                                           random_state=10),
                        param_grid=param_test2,
                        scoring='accuracy',# scoring='roc_auc',#性能评估用roc分数 --- 适合二元分类问题
                        cv=5)#交叉验证的折数 5折

gsearch2.fit(xtrain, ytrain)
print(gsearch2.best_params_, gsearch2.best_score_)


#对黄土数据的读取
data_LGL = pd.read_excel('AllData.xlsx',sheet_name=1)#第二页源区数据 - 1

data1 = data_LGL.drop('No.',axis=1)#删列
data1 = data1.drop('Longitude',axis=1)
data1 = data1.drop('Latitude',axis=1)
data1 = data1.drop(0,axis=0) #删行
# print(data1)
#预测

print('判定结果: %s' % rfc.predict(data1))
# print('判定结果: %s' % rfc.predict_proba(xtest)[:,:])
# print('判定结果: %s' % rfc.predict_proba(xtest)[:,0]) #0 1 2 3 4 ...分别是之前定义的物源的可能性
print("运行完毕")