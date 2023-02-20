from sklearn.ensemble import RandomForestRegressor  # 随机森林回归
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error  # 回归准确率判断-mse
from sklearn.model_selection import train_test_split  # 拆分数据
import pandas as pd
import numpy as np

X, Y = load_boston(return_X_y=True)
x_train, x_validation, y_train, y_validation = \
    train_test_split(X, Y, test_size=0.2)
Test_set = (x_train[1] + x_train[2]) / 2
# 可以增加数据处理
rfr = RandomForestRegressor(
    n_estimators=200,
    random_state=0,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features=None,
    min_impurity_decrease=0.1,
    oob_score=True
)

rfr.fit(x_train, y_train)  # 训练模型
result_y = rfr.predict(x_validation)  # 测试结果
print(mean_squared_error(result_y, y_validation))  # 准确率
print(rfr.oob_score_)  # 袋外数据验证分数
pre = rfr.predict(np.array(Test_set).reshape(1, -1))  # 预测


# X = load_boston() # 波士顿房价的数据  data target feature_name
# print(X)

