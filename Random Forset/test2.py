from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

param_grid = {'max_depth': [2, 4, 6, 8]}

clf = DecisionTreeClassifier()

grid_search = GridSearchCV(clf, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X, y)

print(grid_search.best_params_)
print(grid_search.best_score_)




# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
#
# # 定义参数范围
# param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 20]}
#
# # 定义模型
# model = RandomForestClassifier()
#
# # 定义评分指标
# scoring = {'Accuracy': make_scorer(accuracy_score), 'AUC': make_scorer(roc_auc_score, multi_class='ovo')}
#
# # 使用 GridSearchCV 进行网格搜索
# grid_search = GridSearchCV(model, param_grid, scoring=scoring, refit='Accuracy', cv=5)
#
# # 训练模型并进行参数调优
# grid_search.fit(X_train, y_train)
#
# # 打印最佳参数和评分
# print('Best parameters: ', grid_search.best_params_)
# print('Best accuracy score: ', grid_search.best_score_)
# print('Best AUC score: ', grid_search.cv_results_['mean_test_AUC'][grid_search.best_index_])
