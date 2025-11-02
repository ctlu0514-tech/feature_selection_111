import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris

def evaluate_classifiers(X, y, dataset_name, n_splits=5):
    num_samples = X.shape[0]
    num_features = X.shape[1]
    print(f"Dataset: {dataset_name}, Features: {num_features}, Samples: {num_samples}")

    classifiers = {
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "SVM": SVC(kernel='rbf', random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    results = {}
    print(f"\n=== {dataset_name} 的分类器性能评估 ({n_splits}折交叉验证) ===")
    for name, clf in classifiers.items():
        acc_list, pre_list, rec_list, f1_list = [], [], [], []
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc_list.append(accuracy_score(y_test, y_pred))
            pre_list.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            rec_list.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_list.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        results[name] = {
            "Accuracy": np.mean(acc_list),
            "Precision": np.mean(pre_list),
            "Recall": np.mean(rec_list),
            "F1": np.mean(f1_list)
        }

        print(f"{name}: Acc={np.mean(acc_list):.4f} (±{np.std(acc_list):.4f}), ",
              f"Pre={np.mean(pre_list):.4f}, Rec={np.mean(rec_list):.4f}, F1={np.mean(f1_list):.4f}")

    return results

def evaluate_with_repeats(X, y, selected_features, dataset_name, n_splits=5):
    classifiers = {
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "SVM": SVC(kernel='rbf', random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42) 
    }
    results = {}
    print(f"\n=== {dataset_name} 的分类器性能评估 ({n_splits}折交及验证) ===")
    for name, clf in classifiers.items():
        acc_list, pre_list, rec_list, f1_list = [], [], [], []
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index][:, selected_features], X[test_index][:, selected_features]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc_list.append(accuracy_score(y_test, y_pred))
            pre_list.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            rec_list.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_list.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        results[name] = {
            "Accuracy": np.mean(acc_list),
            "Precision": np.mean(pre_list),
            "Recall": np.mean(rec_list),
            "F1": np.mean(f1_list)
        }

        print(f"{name}: Acc={np.mean(acc_list):.4f} (±{np.std(acc_list):.4f}), ",
              f"Pre={np.mean(pre_list):.4f}, Rec={np.mean(rec_list):.4f}, F1={np.mean(f1_list):.4f}")

    return results


# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
# from sklearn.model_selection import StratifiedKFold
# from sklearn.datasets import load_iris

# def evaluate_classifiers(X, y, dataset_name, n_splits=5):
#     """
#     评估不同分类器在给定数据集上的性能（使用5折交叉验证）
#     Args:
#         X (array-like): 特征矩阵，形状为(n_samples, n_features)
#         y (array-like): 标签向量，形状为(n_samples,)
#         dataset_name (str): 数据集标识名称（用于结果打印）
#         n_splits (int, optional): 交叉验证折数，默认5折

#     Returns:
#         dict: 包含各分类器平均准确率的字典，格式为：
#             {
#                 "KNN": 0.85,
#                 "SVM": 0.89,
#                 "AdaBoost": 0.82
#                 "RandomForest": 0.90
#             }
#     """
#     num_samples = X.shape[0]
#     num_features = X.shape[1]
#     print(f"Dataset: {dataset_name}, Features: {num_features}, Samples: {num_samples}")

#     classifiers = {
#         "KNN": KNeighborsClassifier(n_neighbors=3),
#         "SVM": SVC(kernel='rbf', random_state=42),
#         "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
#         "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
#     }
#     results = {}
#     print(f"\n=== {dataset_name} 的分类器性能评估 ({n_splits}折交叉验证) ===")
#     for name, clf in classifiers.items():
#         acc_list = []
#         kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#         for train_index, test_index in kf.split(X, y):
#             X_train, X_test = X[train_index], X[test_index]
#             y_train, y_test = y[train_index], y[test_index]
#             clf.fit(X_train, y_train)
#             y_pred = clf.predict(X_test)
#             acc_list.append(accuracy_score(y_test, y_pred))
#         avg_acc = np.mean(acc_list)
#         std_acc = np.std(acc_list)
#         results[name] = avg_acc
#         print(f"{name} 平均准确率: {avg_acc:.4f}, 标准差: {std_acc:.4f}")
#     return results

# def evaluate_with_repeats(X, y, selected_features, dataset_name, n_splits=5):
#     """
#     评估分类器性能（使用5折交叉验证）
#     Args:
#         X (array-like): 原始特征矩阵，形状为(n_samples, n_features)
#         y (array-like): 标签向量，形状为(n_samples,)
#         selected_features (list): 选定的特征索引列表，例如[0, 2, 5]
#         dataset_name (str): 数据集标识名称（用于结果打印）
#         n_splits (int, optional): 交叉验证折数，默认5折

#     Returns:
#         dict: 包含各分类器平均准确率的字典，格式同evaluate_classifiers
#     """
#     classifiers = {
#         "KNN": KNeighborsClassifier(n_neighbors=3),
#         "SVM": SVC(kernel='rbf', random_state=42),
#         "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
#         "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42) 
#     }
#     results = {}
#     print(f"\n=== {dataset_name} 的分类器性能评估 ({n_splits}折交叉验证) ===")
#     for name, clf in classifiers.items():
#         acc_list = []
#         kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#         for train_index, test_index in kf.split(X, y):
#             X_train, X_test = X[train_index], X[test_index]
#             X_train_selected = X_train[:, selected_features]
#             X_test_selected = X_test[:, selected_features]
#             y_train, y_test = y[train_index], y[test_index]
#             clf.fit(X_train_selected, y_train)
#             y_pred = clf.predict(X_test_selected)
#             acc_list.append(accuracy_score(y_test, y_pred))
#         avg_acc = np.mean(acc_list)
#         std_acc = np.std(acc_list)
#         results[name] = avg_acc
#         print(f"{name} 平均准确率: {avg_acc:.4f}, 标准差: {std_acc:.4f}")
#     return results

    # 验证代码的功能正确性
if __name__ == "__main__":
    # 使用 Iris 数据集进行测试
    iris = load_iris()
    X = iris.data
    y = iris.target
    evaluate_classifiers(X, y, "Iris 数据集测试")
    selected_features = [0, 1, 2, 3]  # 全部特征
    evaluate_with_repeats(X, y, selected_features, "Iris")

    # data = pd.read_csv('/data/qh_20T_share_file/lct/CDGAFS/data_arrhythmia/arrhythmia.csv')
    # data = data.apply(pd.to_numeric, errors="coerce")  # 将所有列转换为数值型，强制非数值变为 NaN
    # data.fillna(data.mean(), inplace=True)
    # X = data.drop(data.columns[-1], axis=1)  # 删除最后一列，剩余列作为数据
    # y = data[data.columns[-1]]  
    # evaluate_classifiers(X, y, "data_arcene")

    # 使用指定特征并重复评估
    # selected_features = [0, 1, 2, 3]  # 全部特征
    # evaluate_with_repeats(X, y, selected_features, "Iris", repeats=5)