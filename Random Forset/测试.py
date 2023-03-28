import numpy as np
import scipy
import pandas as pd #读取数据
from sklearn.ensemble import RandomForestClassifier #随机森林分类器
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV #数据分割
from sklearn.model_selection import GridSearchCV #交叉验证
import matplotlib.pyplot as plt #生成图
import sklearn.preprocessing as sp #数据预处理,数据缩放
from imblearn.over_sampling import SMOTE #重采样--解决样本不平衡问题

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

# 数据预处理 (0.801-0.803) 零均值标准化 Zero_Mean_Normalization
# scaler1 = sp.StandardScaler()
# temp = x.columns
# x = scaler1.fit_transform(x)
# x = pd.DataFrame(x) # numpy.ndarray数据类型和 pandas.core.frame.DataFrame 的互相转化
# x.columns = temp #把标签栏恢复成元素

y = category
##重采样(使不同类别样品数均衡) Resampling --- 0.958
x,y = SMOTE().fit_resample(x,y)

#训练集和测试集分割
#x是清理好后的数据,y是结果数据
seed = 5
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state=seed)
#这个比例可以调,然后增大正确率
#训练集0.8,  测试集0.2

#==================================随机森林分类器的使用=======================================
rfc0 = RandomForestClassifier() #实例化
rfc1 = RandomForestClassifier(n_estimators=350,
                             max_depth=None,
                             min_samples_split=2,#2?#60?
                             random_state=0)
rfc = RandomForestClassifier(random_state=3,
                                  criterion='gini',
                                  class_weight=None,
                                  bootstrap=True,
                                  oob_score=True,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  n_jobs=-1,
                                  verbose=0,
                                  max_leaf_nodes=None,
                                  max_samples=None,
                                  min_impurity_decrease=0.0,
                                  max_features='sqrt',# max_features='auto', #已删除
                                  # min_impurity_split = None,
                                  min_weight_fraction_leaf=0.0,
                                  warm_start=False,
                                  max_depth=11,
                                  n_estimators=351)

rfc = rfc.fit(xtrain,ytrain) #用训练集数据训练模型 (接口fit)

scores = cross_val_score(rfc,xtrain,ytrain,cv=10,scoring='f1_weighted',n_jobs=-1) #评估性能 F1分数
"""当scoring='f1_weighted’时，表示对每个标签（label）计算F1分数，然后根据每个标签的支持度（support），
即真实样本的数量，进行加权平均2。这样可以考虑到标签的不平衡性，避免某些标签的评分过高或过低3"""
print('测试集的平均准确率:',scores.mean())

# result = rfc.score(xtest,ytest) #导入测试集,rfc的接口score计算的是模型的平均准确率accuracy
# print('测试集的平均准确率:',result) #得出结果

importances = rfc.feature_importances_ # 显示各元素的特征重要性值的大小
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],axis=0) #计算数组中的标准差
indices = np.argsort(importances)[::-1] #对重要性从大到小排序
print('Feature ranking: ')
for f in range(min(20,xtrain.shape[1])): #对前20个元素的重要性进行排序显示
    print("%2d) %-*s %f" %(f + 1, 30, xtrain.columns[indices[f]],importances[indices[f]])) #

# #生成元素重要性图片
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(xtrain.shape[1]),importances[indices], color="r",yerr=std[indices],align="center")
# plt.xticks(range(xtrain.shape[1]),indices)
# plt.xlim([-1,xtrain.shape[1]])
# plt.show()


# # 利用网格搜索,交叉验证
# param_test1 = {'n_estimators': range(25,500,25)}#对分类器数量进行调优
# gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,#将要分割的样本数不能小于100
#                                                            min_samples_leaf=20,#分割出来的不能小于20
#                                                            max_depth=8,
#                                                            random_state=10), #分类器
#                         param_grid=param_test1,
#                         scoring = 'accuracy',#{'Accuracy': make_scorer(accuracy_score), 'AUC': make_scorer(roc_auc_score, multi_class='ovo')},#scoring='roc_auc',# scoring='roc_auc',#性能评估用roc分数 --- 适合二元分类问题
#                         n_jobs=-1,#加速
#                         cv=10
#                         )#交叉验证的折数 10折
#
# gsearch1.fit(xtrain, ytrain)
# print(gsearch1.best_params_, gsearch1.best_score_)
# #{'n_estimators': 200} 0.5959582790091265





#对黄土数据的读取
data_LGL = pd.read_excel('AllData.xlsx',sheet_name=1)#第二页源区数据 - 1
data2 = data_LGL.drop('No.', axis=1)
data2 = data2.drop('Longitude', axis=1)
data2 = data2.drop('Latitude', axis=1)
data2 = data2.drop(0, axis=0)
#预测
print('判定结果: %s' % rfc.predict(data2))
# print('判定结果: %s' % rfc.predict_proba(xtest)[:,:])
# print('判定结果: %s' % rfc.predict_proba(xtest)[:,0]) #0 1 2 3 4 ...分别是之前定义的物源的可能性
print("运行完毕")