import pandas as pd

# 读取 xlsx 文件
data = pd.read_excel('AllData.xlsx')

category = data['Category']
category = category.drop(0,axis=0) #删去第一行空白行
print(category)

# 打印数据
# print(data)

# a = data.head()
# print(a)

#对某列数据进行分析
counts = data['Category'].value_counts()
# print(counts)

data['Category'] = data['Category'].astype(str).map({'Junggar Basin':1,
                                                     'Tarim Basin':2,
                                                     'North Alxa Plateau':3,
                                                     'Qaidam Basin':4,
                                                     'northeastern sandy deserts':5,
                                                     'Hetao Graben':6,
                                                     'South Alxa Plateau':7,
                                                     'eastern Tibetan Plateau':8})
# b = data
# print(b)

# c = data['Category']
# print(c)

#数据整理:去掉样品源区,去掉经纬度,去掉样品序号
x = data.drop('Category',axis=1)
x = x.drop('Longitude',axis=1)
x = x.drop('Latitude',axis=1)
x = x.drop('No.',axis=1)
x = x.drop(0,axis=0) #去掉单位那一行(第二行:1?)(不对,除了表头,单位是第一行:0

print(x)
# print(data)



#=============================================================================


# #训练集和测试集分割
# from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
#
# seed = 5
# xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3,random_state=seed)