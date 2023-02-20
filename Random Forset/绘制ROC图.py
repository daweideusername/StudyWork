"""ROC 曲线是一种用于评估二分类模型的指标，通过绘制真正率（True Positive Rate）和
假正率（False Positive Rate）的变化情况来评估模型的效果。

真正率表示正样本被正确预测的概率，假正率表示负样本被错误预测为正样本的概率。

ROC 曲线的下方的面积，也就是 AUC (Area Under the Curve) 指标，越大，说明模型的预测效果越好。"""
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
#roc_curve 接受两个参数，分别是真实值和预测值的分数，返回三个值：false positive rate, true positive rate, 以及thresholds.
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#roc_auc_score是用来评估ROC曲线下面积的。ROC曲线下面积（AUC）越大说明模型的预测性能越好，
# AUC值越接近1则说明分类模型的效果更好，AUC值越接近0.5则说明分类模型的效果越差。
# roc_auc_score函数可以计算出模型的AUC值，以评估模型的效果。
from sklearn.metrics import roc_auc_score
a = roc_auc_score(y_true, y_scores)
print(a)
