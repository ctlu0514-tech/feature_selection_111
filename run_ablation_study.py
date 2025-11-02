# 描述: 专门用于运行和对比两种图构建方法下的最终分类性能。
import pandas as pd
import numpy as np
from CDGAFS import cdgafs_feature_selection
from evaluate import evaluate_classifiers, evaluate_with_repeats

# --- 参数设置 ---
FILE_NAME = 'Brain_GSE50161_byGene.csv'
LABEL_COL_NAME = 'type'
PRE_FILTER_TOP_N = 4000
THETA = 0.8
OMEGA = 0.5
POPULATION_SIZE = 100

# --- 数据加载与预处理 (与 main.py 逻辑一致) ---
print("--- 1. 加载和预处理数据 ---")
total_data = pd.read_csv('/data/qh_20T_share_file/lct/CuMiDa/' + FILE_NAME)
col_name = total_data.columns
patient_id_col = col_name[0]
feature_columns = [col for col in col_name if col not in [LABEL_COL_NAME, patient_id_col]]
variances = total_data[feature_columns].var()
non_zero_var_features = variances[variances > 0].index.tolist()
total_data_clean = total_data.dropna(subset=non_zero_var_features)
labels = total_data_clean[LABEL_COL_NAME].values
feature_min = total_data_clean[non_zero_var_features].min()
feature_max = total_data_clean[non_zero_var_features].max()
normalized_features = (total_data_clean[non_zero_var_features] - feature_min) / (feature_max - feature_min + 1e-10)
gene_list_full = normalized_features.columns.tolist()
feature_data_full = normalized_features.values
label_data = np.array(labels)

# --- 运行实验 1: 融合知识图 (你的原始方法) ---
print("\n" + "="*50)
print("EXPERIMENT 1: Fused Knowledge Graph")
print("="*50)
selected_features_fused = cdgafs_feature_selection(
    X=feature_data_full, 
    y=label_data,
    gene_list=gene_list_full,
    theta=THETA, 
    omega=OMEGA, 
    population_size=POPULATION_SIZE,
    pre_filter_top_n=PRE_FILTER_TOP_N,
    graph_type='fused'  # 明确指定使用融合图
)
print("\n--- 正在评估融合知识图的性能 ---")
results_fused = evaluate_with_repeats(
    feature_data_full, 
    label_data, 
    selected_features_fused, 
    "CDGAFS (Fused)"
)

# --- 运行实验 2: 仅皮尔逊图 (对照组) ---
print("\n" + "="*50)
print("EXPERIMENT 2: Pearson Only Graph (Ablation)")
print("="*50)
selected_features_pearson = cdgafs_feature_selection(
    X=feature_data_full, 
    y=label_data,
    gene_list=gene_list_full,
    theta=THETA, 
    omega=OMEGA, 
    population_size=POPULATION_SIZE,
    pre_filter_top_n=PRE_FILTER_TOP_N,
    graph_type='pearson_only'  # 明确指定使用仅皮尔逊图
)
print("\n--- 正在评估仅皮尔逊图的性能 ---")
results_pearson = evaluate_with_repeats(
    feature_data_full, 
    label_data, 
    selected_features_pearson, 
    "CDGAFS (Pearson Only)"
)

# --- 最终对比总结 ---
print("\n" + "#"*60)
print(" " * 10 + "消融实验总结: 分类器平均准确率")
print("#"*60)
print(f"{'Classifier':<15} | {'融合图 (Fused) Acc':<25} | {'基础图 (Pearson) Acc':<25}")
print("-" * 65)

# 确保两个结果字典都包含相同的分类器键
all_classifiers = sorted(list(results_fused.keys()))

for clf in all_classifiers:
    acc_fused = results_fused.get(clf, {}).get('Accuracy', 'N/A')
    acc_pearson = results_pearson.get(clf, {}).get('Accuracy', 'N/A')
    
    # 格式化输出
    fused_str = f"{acc_fused:.4f}" if isinstance(acc_fused, float) else acc_fused
    pearson_str = f"{acc_pearson:.4f}" if isinstance(acc_pearson, float) else acc_pearson
    
    print(f"{clf:<15} | {fused_str:<25} | {pearson_str:<25}")

print("#"*60)
