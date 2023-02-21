
import pandas as pd

#对黄土数据的读取
data_LGL = pd.read_excel('AllData.xlsx',sheet_name=1)#第二页源区数据 - 1

data1 = data_LGL.drop('No.',axis=1)#删列
data1 = data1.drop('Longitude',axis=1)
data1 = data1.drop('Latitude',axis=1)
data1 = data1.drop(0,axis=0) #删行
# data1 = data1.drop(0,axis=0) #去掉单位那一行(第二行:1?)(不对,除了表头,单位是第一行:0

print(data1)