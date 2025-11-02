import numpy as np
import random
import networkx as nx
from collections import defaultdict
import os

# 从我们创建的模块中导入函数
from data_loader import load_and_preprocess_data
from CDGAFS import cdgafs_feature_selection
from centrality_based_selection import select_features_by_centrality

# 导入您项目中的其他核心模块
from evaluate import evaluate_with_repeats
from fisher_score import compute_fisher_score
from construct_feature_graph import construct_feature_graph
from community_detection import iscd_algorithm_auto_k

# 导入必要的库 (用于LASSO)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

# ========================> 保存基因列表的函数 <========================
def save_gene_list(indices, gene_list, directory, filename):
    """将筛选出的基因名保存到指定文件。"""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    
    gene_names = [gene_list[i] for i in indices]
    
    with open(filepath, 'w') as f:
        for name in gene_names:
            f.write(f"{name}\n")
    print(f"基因列表已成功保存至: {filepath}")

# ========================> LASSO精选函数 <========================
def refine_features_with_lasso(X_candidate, y, candidate_indices):
    """
    使用L1正则化的逻辑回归 (LASSO) 从候选特征集中精选特征。
    """
    print("\n--- 步骤 2.2: 使用LASSO进行特征精选 ---")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_candidate)

    l1_logreg = LogisticRegressionCV(
        Cs=10, 
        cv=5, 
        penalty='l1', 
        solver='saga',
        random_state=42,
        max_iter=2000
    )
    
    l1_logreg.fit(X_scaled, y)
    print(f"LASSO找到的最佳正则化强度C值为: {l1_logreg.C_[0]:.4f}")

    non_zero_coeffs_mask = np.abs(l1_logreg.coef_[0]) > 1e-5
    selected_indices_in_candidate = np.where(non_zero_coeffs_mask)[0]
    
    if len(selected_indices_in_candidate) == 0:
        print("警告: LASSO筛选后没有剩下任何特征。")
        return []

    print(f"LASSO从 {X_candidate.shape[1]} 个候选特征中精选出 {len(selected_indices_in_candidate)} 个特征。")

    final_indices = [candidate_indices[i] for i in selected_indices_in_candidate]
    return final_indices

# ========================> 主程序 <========================
if __name__ == "__main__":
    # --- 全局随机种子 ---
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

    # --- 实验参数 ---
    PRE_FILTER_TOP_N = 4000
    THETA = 0.8
    RESULTS_DIR = "enrichment_inputs_Lung"
    
    # 1. 加载数据
    X, y, gene_list = load_and_preprocess_data(file_name='Prostate_GSE11682_byGene.csv', label_line_name='type')

    if X is None:
        print("数据加载失败，程序终止。")
    else:
        # --- 公共步骤 (预筛选、建图、社团检测) ---
        print("\n--- 执行公共步骤：预筛选、建图、社团检测 ---")
        fisher_scores = compute_fisher_score(X, y)
        top_indices = np.argsort(fisher_scores)[-PRE_FILTER_TOP_N:]
        X_subset = X[:, top_indices]
        
        feature_graph = construct_feature_graph(X_subset, y, [gene_list[i] for i in top_indices], THETA)
        
        partition = iscd_algorithm_auto_k(feature_graph)
        clusters_dict = defaultdict(list)
        for node, community in partition.items():
            clusters_dict[community].append(node)
        clusters = [cluster for cluster in clusters_dict.values()]
        print(f"公共步骤完成，检测到 {len(clusters)} 个社团。")
        
        # --- 方案一：原CDGAFS (遗传算法) ---
        print("\n" + "="*50)
        print("方案一: 原CDGAFS (遗传算法)")
        print("="*50)
        selected_indices_ga = cdgafs_feature_selection(
            X, y, gene_list, THETA, omega=0.0025, population_size=100, pre_filter_top_n=PRE_FILTER_TOP_N
        )
        save_gene_list(selected_indices_ga, gene_list, RESULTS_DIR, "ga_selected_genes.txt")
        
        # # --- 方案二：新方法 (中心性 + LASSO) ---
        print("\n" + "="*50)
        print("方案二: 新方法 (中心性排序 + LASSO精选)")
        print("="*50)
        
        # --- 步骤 2.1: 中心性排序 ---
        print("--- 步骤 2.1: 中心性排序初筛 ---")
        selected_indices_centrality = select_features_by_centrality(
            feature_graph, clusters, top_indices, selection_ratio=0.3
        )
        num_centrality_features = len(selected_indices_centrality)
        print(f"中心性排序从 {PRE_FILTER_TOP_N} 个特征中初筛出 {num_centrality_features} 个候选特征。")
        save_gene_list(selected_indices_centrality, gene_list, RESULTS_DIR, "centrality_selected_genes.txt")


        # **** 新增的评估步骤 ****
        print("\n--- 评估【中心性排序后】的特征集性能 ---")
        results_centrality = evaluate_with_repeats(X, y, selected_indices_centrality, "Centrality-only")


        # --- 步骤 2.2: LASSO精选 ---
        X_candidate = X[:, selected_indices_centrality]
        final_selected_indices_lasso = refine_features_with_lasso(X_candidate, y, selected_indices_centrality)
        num_lasso_features = len(final_selected_indices_lasso)
        save_gene_list(final_selected_indices_lasso, gene_list, RESULTS_DIR, "centrality_lasso_selected_genes.txt")


        # # **** 对LASSO之后的结果进行评估 ****
        print("\n--- 评估【中心性+LASSO后】的最终特征集性能 ---")
        results_lasso = evaluate_with_repeats(X, y, final_selected_indices_lasso, "Centrality+LASSO")
        

        # --- 最终总结与对比 ---
        print("\n" + "#"*70)
        print(" " * 25 + "最终实验总结与对比")
        print("#"*70)

        print("\n【方案一: 原CDGAFS (遗传算法)】")
        print(f"  - 最终特征数量: {len(selected_indices_ga)}")
        print("  - 分类性能:")
        evaluate_with_repeats(X, y, selected_indices_ga, "CDGAFS (GA)")

        print("\n【方案二: 新方法 (分步评估)】")
        print(f"  - 步骤 2.1 (中心性排序后):")
        print(f"    - 特征数量: {num_centrality_features}")
        print(f"    - 分类性能:")
        # 再次打印一次结果，方便集中对比
        for clf, metrics in results_centrality.items():
            print(f"      - {clf}: Acc={metrics['Accuracy']:.4f}")

        print(f"\n  - 步骤 2.2 (LASSO精选后):")
        print(f"    - 特征数量: {num_lasso_features}")
        print(f"    - 分类性能:")
        # 再次打印一次结果，方便集中对比
        for clf, metrics in results_lasso.items():
            print(f"      - {clf}: Acc={metrics['Accuracy']:.4f}")

        print("#"*70)