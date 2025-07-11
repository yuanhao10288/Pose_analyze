# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn import svm
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import f1_score, classification_report, confusion_matrix
# import joblib
#
# # 加载数据
# data = pd.read_csv('data.csv')
#
# # 检查并处理缺失值
# print("Missing values:\n", data.isnull().sum())
# data.fillna(data.median(), inplace=True)
#
# # 特征与目标变量
# features = ['langle', 'rangle', 'lsangle', 'rsangle', 'lhangle', 'rhangle', 'lkangle', 'rkangle']
# x = data[features].values
# y = data['class'].values
#
# # 划分数据集（分层抽样）
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=20, stratify=y
# )
#
# # 特征标准化
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
#
# # 超参数调优
# param_grid = {'C': [0.1, 1, 1.2, 10], 'gamma': ['scale', 'auto', 0.1, 1]}
# grid = GridSearchCV(
#     svm.SVC(kernel='rbf', probability=True, decision_function_shape='ovr'),
#     param_grid, cv=5, scoring='f1_micro'
# )
# grid.fit(x_train, y_train)
# best_model = grid.best_estimator_
#
# # 交叉验证评估
# cv_scores = cross_val_score(best_model, x_train, y_train, cv=5, scoring='f1_micro')
# print("Cross-Validation Scores:", cv_scores)
#
# # 训练最终模型
# best_model.fit(x_train, y_train)
#
# # 预测与评估
# result_test = best_model.predict(x_test)
# print("Test F1 (micro): {0:.2f}".format(f1_score(y_test, result_test, average='micro')))
# print("\nClassification Report:\n", classification_report(y_test, result_test))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, result_test))
#
# # 保存模型和标准化器
# joblib.dump(best_model, 'model/optimized_tennis_pose_svm.pkl')
# joblib.dump(scaler, 'model/scaler.pkl')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import joblib
import os

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建保存图像的目录
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# 加载数据
data = pd.read_csv('data.csv')

# 检查并处理缺失值
print("Missing values:\n", data.isnull().sum())
data.fillna(data.median(), inplace=True)

# 特征与目标变量
features = ['langle', 'rangle', 'lsangle', 'rsangle', 'lhangle', 'rhangle', 'lkangle', 'rkangle']
x = data[features].values
y = data['class'].values

# 划分数据集（分层抽样）
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=20, stratify=y
)

# 特征标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 超参数调优
param_grid = {'C': [0.1, 1, 1.2, 10], 'gamma': ['scale', 'auto', 0.1, 1]}
grid = GridSearchCV(
    svm.SVC(kernel='rbf', probability=True, decision_function_shape='ovr'),
    param_grid, cv=5, scoring='f1_micro'
)
grid.fit(x_train, y_train)
best_model = grid.best_estimator_

# 交叉验证评估
cv_scores = cross_val_score(best_model, x_train, y_train, cv=5, scoring='f1_micro')
print("Cross-Validation Scores:", cv_scores)

# 训练最终模型
best_model.fit(x_train, y_train)

# 预测与评估
result_test = best_model.predict(x_test)
print("Test F1 (micro): {0:.2f}".format(f1_score(y_test, result_test, average='micro')))
print("\nClassification Report:\n", classification_report(y_test, result_test))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, result_test))


# 可视化函数

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='混淆矩阵', cmap=plt.cm.Blues):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(f'visualizations/{title.replace(" ", "_")}.png')
    plt.close()


def plot_learning_curve(estimator, X, y, title='学习曲线', cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """绘制学习曲线"""
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.xlabel("训练样本数")
    plt.ylabel("得分")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='f1_micro')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="训练得分")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="交叉验证得分")
    plt.legend(loc="best")
    plt.savefig(f'visualizations/{title.replace(" ", "_")}.png')
    plt.close()


def plot_feature_importance(model, X, y, feature_names, title='特征重要性'):
    """使用排列重要性绘制特征重要性"""
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    importances = result.importances_mean
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig(f'visualizations/{title.replace(" ", "_")}.png')
    plt.close()

    # 打印特征重要性排序
    print("\n特征重要性排序:")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")


def plot_decision_boundary(X, y, model, feature_names, class_names, title='决策边界'):
    """使用PCA降维后绘制二维决策边界"""
    # 使用PCA将特征降维到2D以便可视化
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # 训练模型（使用降维后的数据）
    model_2d = svm.SVC(kernel='rbf', C=best_model.C, gamma=best_model.gamma,
                       probability=True, decision_function_shape='ovr')
    model_2d.fit(X_2d, y)

    # 创建网格以绘制决策边界
    h = 0.02  # 网格步长
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 预测网格点的类别
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, alpha=0.8, edgecolors='k')

    # 获取PCA解释方差
    explained_variance = pca.explained_variance_ratio_
    plt.xlabel(f'主成分1 (解释方差: {explained_variance[0]:.2%})')
    plt.ylabel(f'主成分2 (解释方差: {explained_variance[1]:.2%})')
    plt.title(title)
    plt.legend(class_names)
    plt.tight_layout()
    plt.savefig(f'visualizations/{title.replace(" ", "_")}.png')
    plt.close()


def plot_hyperparameter_tuning(grid_search, param_grid, title='超参数调优结果'):
    """绘制超参数调优结果热图"""
    results = pd.DataFrame(grid_search.cv_results_)

    # 提取参数和得分
    param1 = list(param_grid.keys())[0]
    param2 = list(param_grid.keys())[1]

    # 创建得分矩阵
    scores = np.array(results['mean_test_score']).reshape(len(param_grid[param1]), len(param_grid[param2]))

    # 绘制热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(scores, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=param_grid[param2], yticklabels=param_grid[param1])
    plt.xlabel(param2)
    plt.ylabel(param1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'visualizations/{title.replace(" ", "_")}.png')
    plt.close()


# 可视化混淆矩阵
classes = np.unique(y)
plot_confusion_matrix(y_test, result_test, classes, title='混淆矩阵')
plot_confusion_matrix(y_test, result_test, classes, normalize=True, title='归一化混淆矩阵')

# 可视化学习曲线
plot_learning_curve(best_model, x_train, y_train, title='SVM学习曲线')

# 可视化特征重要性
plot_feature_importance(best_model, x_test, y_test, features, title='SVM特征重要性')

# 可视化决策边界
plot_decision_boundary(x_test, y_test, best_model, features, classes, title='SVM决策边界')

# 可视化超参数调优结果
plot_hyperparameter_tuning(grid, param_grid, title='SVM超参数调优结果')

# 保存模型和标准化器
if not os.path.exists('model'):
    os.makedirs('model')
joblib.dump(best_model, 'model/optimized_tennis_pose_svm.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("\n所有可视化图表已保存至 'visualizations' 目录")