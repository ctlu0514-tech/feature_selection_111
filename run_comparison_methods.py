# 文件名: run_comparison_methods.py

import numpy as np
import pandas as pd
import time  # 用于记录时间

# --- 从您的项目中导入必要的模块 ---
try:
    from data_loader import load_and_preprocess_data
    from fisher_score import compute_fisher_score
    from evaluate import evaluate_with_repeats
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保此脚本与 data_loader.py, fisher_score.py, evaluate.py 在同一PYTHONPATH下，"
          "或者这些模块已正确安装。")
    exit()

# --- 导入对比方法所需的库 ---
try:
    import pymrmr
except ImportError:
    print("错误: 未找到 'pymrmr' 库。请运行 'pip install pymrmr'")
    exit()

try:
    from skrebate import ReliefF
except ImportError:
    print("错误: 未找到 'skrebate' 库。请运行 'pip install scikit-rebate'")
    exit()

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression # RFE需要一个评估器
from sklearn.preprocessing import StandardScaler   # LASSO/RFE可能需要标准化

# ========================> 参数设置 <========================
# --- 数据文件和标签列 ---
# !! 修改为您要运行的数据集 !!
FILE_NAME = 'Brain_GSE50161_byGene.csv' 
LABEL_COL_NAME = 'type' 
DATASET_FRIENDLY_NAME = "brain Cancer" # 用于输出和绘图

# --- 特征选择参数 ---
# !! 设置您希望所有对比方法选择的特征数量 (K) !!
#    可以参考您自己方法通常选出的数量，或者设置一个固定值进行比较
NUM_FEATURES_TO_SELECT = 10

# --- RFE 参数 ---
RFE_ESTIMATOR = LogisticRegression(solver='liblinear', random_state=42, max_iter=500) # 为RFE选择一个评估器
RFE_STEP = 0.1 # 每次迭代移除10%的特征，加快速度；设为1则每次移除1个

# --- ReliefF 参数 ---
RELIEF_NEIGHBORS = 10 # ReliefF考虑的近邻数

# --- 随机种子 ---
SEED = 42
np.random.seed(SEED)
# ============================================================

def run_fisher_baseline(X, y, k, gene_list):
    """基线方法：仅使用 Fisher Score 排序选择 top K"""
    print("\n--- 运行基线方法: Fisher Score Top K ---")
    start_time = time.time()
    if X is None or y is None:
        print("数据未加载，无法运行 Fisher Score 基线。")
        return None
        
    try:
        fisher_scores = compute_fisher_score(X, y)
        # 获取得分最高的 K 个特征的索引
        top_indices = np.argsort(fisher_scores)[-k:]
        end_time = time.time()
        print(f"Fisher Score Top {k} 选择完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"选定特征数: {len(top_indices)}")
        return list(top_indices)
    except Exception as e:
        print(f"运行 Fisher Score 基线时出错: {e}")
        return None

def run_mrmr(X, y, k, gene_list):
    """运行 mRMR 特征选择"""
    print("\n--- 运行对比方法: mRMR ---")
    start_time = time.time()
    if X is None or y is None or gene_list is None:
        print("数据未加载，无法运行 mRMR。")
        return None
        
    # mRMR 需要 Pandas DataFrame
    df = pd.DataFrame(X, columns=gene_list)
    # 确保标签是 Pandas Series
    if isinstance(y, np.ndarray):
        y_series = pd.Series(y, name='label')
    else: # 假设已经是 Series 或类似结构
        y_series = y.rename('label')
    df['label'] = y_series 

    try:
        # 使用互信息 ('MIQ') 进行选择
        selected_features_names = pymrmr.mRMR(df, 'MIQ', k)
        
        # 将选定的基因名转换回原始索引
        # 使用字典查找提高效率
        gene_to_index = {name: i for i, name in enumerate(gene_list)}
        selected_indices = [gene_to_index.get(name) for name in selected_features_names if name in gene_to_index]
        
        end_time = time.time()
        print(f"mRMR 选择完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"选定特征数: {len(selected_indices)}")
        return selected_indices
    except Exception as e:
        print(f"运行 mRMR 时出错: {e}")
        return None

def run_reliefF(X, y, k):
    """运行 ReliefF 特征选择"""
    print("\n--- 运行对比方法: ReliefF ---")
    start_time = time.time()
    if X is None or y is None:
        print("数据未加载，无法运行 ReliefF。")
        return None

    try:
        # 初始化 ReliefF
        relief = ReliefF(n_neighbors=RELIEF_NEIGHBORS, n_features_to_select=k) 
        
        # ReliefF 对数据尺度不敏感，通常不需要标准化，但如果特征差异巨大，标准化可能仍有帮助
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)
        # relief.fit(X_scaled, y)
        
        # 直接在原始（或预处理过的）数据上运行
        relief.fit(X, y) # skrebate 的 ReliefF 可以处理多分类

        # 获取选定的特征索引 (skrebate 直接返回索引)
        # 注意：relief.top_features_ 返回的是 *排序后* 的索引，我们需要的是它们在原始X中的位置
        # 我们让 relief 计算所有特征的分数，然后自己选 top K
        relief_all_scores = ReliefF(n_neighbors=RELIEF_NEIGHBORS, n_features_to_select=X.shape[1])
        relief_all_scores.fit(X,y)
        scores = relief_all_scores.feature_importances_
        top_indices = np.argsort(scores)[-k:]

        end_time = time.time()
        print(f"ReliefF 选择完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"选定特征数: {len(top_indices)}")
        return list(top_indices)
    except Exception as e:
        print(f"运行 ReliefF 时出错: {e}")
        return None

def run_rfe(X, y, k):
    """运行 RFE (方法一：固定K) 特征选择"""
    print("\n--- 运行对比方法: RFE ---")
    start_time = time.time()
    if X is None or y is None:
        print("数据未加载，无法运行 RFE。")
        return None

    try:
        # RFE 对特征尺度敏感，建议标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 初始化 RFE 选择器
        selector_rfe = RFE(estimator=RFE_ESTIMATOR, 
                           n_features_to_select=k, 
                           step=RFE_STEP) # step=1 最精细但最慢

        # 运行 RFE
        print("RFE 正在运行，这可能需要较长时间...")
        selector_rfe = selector_rfe.fit(X_scaled, y) # RFE 会在内部多次训练评估器

        # 获取选中的特征索引
        selected_indices = selector_rfe.get_support(indices=True)
        
        end_time = time.time()
        print(f"RFE 选择完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"选定特征数: {len(selected_indices)}")
        return list(selected_indices)
    except Exception as e:
        print(f"运行 RFE 时出错: {e}")
        return None

# ========================> 主程序 <========================
if __name__ == "__main__":
    
    # 1. 加载数据 (使用您项目中的函数)
    X_full, y_full, gene_list_full = load_and_preprocess_data(
        file_name=FILE_NAME, 
        label_line_name=LABEL_COL_NAME
    )

    if X_full is None:
        print("数据加载失败，程序终止。")
        exit()
        
    print(f"\n数据集 '{DATASET_FRIENDLY_NAME}' 加载完成。")
    print(f"完整特征矩阵形状: {X_full.shape}")
    print(f"标签数组形状: {y_full.shape}")
    print(f"基因列表长度: {len(gene_list_full)}")

    # 存储结果的字典
    all_selected_indices = {}
    all_evaluation_results = {}

    # --- 运行各种特征选择方法 ---
    
    # a) Fisher Score Baseline
    indices_fisher = run_fisher_baseline(X_full, y_full, NUM_FEATURES_TO_SELECT, gene_list_full)
    if indices_fisher:
        all_selected_indices['Fisher_TopK'] = indices_fisher

    # b) mRMR
    indices_mrmr = run_mrmr(X_full, y_full, NUM_FEATURES_TO_SELECT, gene_list_full)
    if indices_mrmr:
        all_selected_indices['mRMR'] = indices_mrmr

    # c) ReliefF
    indices_relief = run_reliefF(X_full, y_full, NUM_FEATURES_TO_SELECT)
    if indices_relief:
        all_selected_indices['ReliefF'] = indices_relief

    # d) RFE
    indices_rfe = run_rfe(X_full, y_full, NUM_FEATURES_TO_SELECT)
    if indices_rfe:
        all_selected_indices['RFE'] = indices_rfe
        
    # 您自己的方法的结果也可以加入这里，例如：
    # print("\n--- 运行您自己的方法 (示例) ---")
    # indices_cdgafs = cdgafs_feature_selection(...) 
    # if indices_cdgafs:
    #     all_selected_indices['CDGAFS'] = indices_cdgafs
    # indices_centrality = select_features_by_centrality(...)
    # if indices_centrality:
    #     all_selected_indices['Centrality'] = indices_centrality


    # --- 对每种方法选出的特征集进行评估 ---
    print("\n" + "="*50)
    print("开始评估所有方法的分类性能...")
    print("="*50)

    for method_name, selected_indices in all_selected_indices.items():
        if selected_indices:
            print(f"\n--- 评估方法: {method_name} (选择了 {len(selected_indices)} 个特征) ---")
            # 使用您项目中的评估函数
            eval_results = evaluate_with_repeats(
                X=X_full, 
                y=y_full, 
                selected_features=selected_indices, 
                dataset_name=f"{DATASET_FRIENDLY_NAME} - {method_name}" # 传递方法名用于打印
            )
            all_evaluation_results[method_name] = eval_results
        else:
            print(f"\n--- 跳过评估方法: {method_name} (因为特征选择失败或未选择特征) ---")

    # --- 打印最终结果总结 ---
    print("\n" + "#"*60)
    print(f"       最终性能对比总结 ({DATASET_FRIENDLY_NAME}, 目标特征数 K={NUM_FEATURES_TO_SELECT})")
    print("#"*60)
    print(f"{'Method':<15} | {'Selected K':<10} | {'KNN Acc':<10} | {'SVM Acc':<10} | {'AdaBoost Acc':<12} | {'RF Acc':<10}")
    print("-" * 75)

    # 确定所有出现过的分类器名称，以防某些方法评估失败
    classifier_names = ["KNN", "SVM", "AdaBoost", "RandomForest"] 

    # 排序方法名以便比较
    sorted_methods = sorted(all_evaluation_results.keys())

    for method in sorted_methods:
        results = all_evaluation_results.get(method, {})
        selected_k = len(all_selected_indices.get(method, [])) # 获取实际选出的 K
        
        knn_acc = f"{results.get('KNN', {}).get('Accuracy', 'N/A'):.4f}" if isinstance(results.get('KNN', {}).get('Accuracy'), (int, float)) else 'N/A'
        svm_acc = f"{results.get('SVM', {}).get('Accuracy', 'N/A'):.4f}" if isinstance(results.get('SVM', {}).get('Accuracy'), (int, float)) else 'N/A'
        ada_acc = f"{results.get('AdaBoost', {}).get('Accuracy', 'N/A'):.4f}" if isinstance(results.get('AdaBoost', {}).get('Accuracy'), (int, float)) else 'N/A'
        rf_acc = f"{results.get('RandomForest', {}).get('Accuracy', 'N/A'):.4f}" if isinstance(results.get('RandomForest', {}).get('Accuracy'), (int, float)) else 'N/A'
        
        print(f"{method:<15} | {selected_k:<10} | {knn_acc:<10} | {svm_acc:<10} | {ada_acc:<12} | {rf_acc:<10}")

    print("#"*60)
    print("\n对比实验运行完毕。")