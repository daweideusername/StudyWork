"""
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
来自网址的例子
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification #sklearn.datasets是scikit-learn中用于加载和管理常用数据集的模块。它提供了许多常用数据集，如Iris数据集、Boston房价数据集、手写数字识别数据集等，可以方便用户在机器学习和数据分析中使用。
X, y = make_classification(n_samples=1000,
                           n_features=4,
                           n_informative=2,
                           n_redundant=0,
                           random_state=0,
                           shuffle=False)

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X, y)

# print(clf.predict([[0, 2, 0, 1]]))
print(y)