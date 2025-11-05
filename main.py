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
def refine_features_with_lasso(X_candidate, y, candidate_indices, top_n):
    """
    使用L1正则化的逻辑回归 (LASSO) 对特征进行重要性排序，并选择最重要的前 N 个。
    
    参数:
    X_candidate (np.array): 候选特征的数据矩阵。
    y (np.array): 标签。
    candidate_indices (list): 候选特征在原始数据集中的索引。
    top_n (int): 希望最终选出的特征数量。
    """
    print(f"\n--- 步骤 2.2: 使用LASSO进行特征精选 (选取Top {top_n}) ---")

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

    # 获取所有特征的系数绝对值，作为它们的重要性得分
    feature_importances = np.abs(l1_logreg.coef_[0])

    # 如果所有系数都为0，则无法选择
    if np.sum(feature_importances) == 0:
        print("警告: LASSO筛选后所有特征系数均为0，无法进行选择。")
        return []

    # 1. 根据重要性得分对特征的索引进行降序排序
    # np.argsort返回的是从小到大的索引，所以用[::-1]来反转它
    sorted_indices = np.argsort(feature_importances)[::-1]

    # 2. 从排好序的索引中，选出前 top_n 个
    top_n_indices_in_candidate = sorted_indices[:top_n]

    print(f"LASSO从 {X_candidate.shape[1]} 个候选特征中精选出最重要的 {len(top_n_indices_in_candidate)} 个特征。")

    # 3. 将这些在候选集中的索引，映射回原始数据集的索引
    final_indices = [candidate_indices[i] for i in top_n_indices_in_candidate]
    return final_indices

# ========================> 主程序 <========================
if __name__ == "__main__":
    # --- 全局随机种子 ---
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

    # --- 实验参数 ---
    PRE_FILTER_TOP_N = 4000
    THETA = 0.4
    W_BIO_BOOST = 0.5 
    RESULTS_DIR = "enrichment_inputs_Colorectal"
    GA_POPULATION_SIZE = 100

    GA_OMEGA = 0.0125
    SELECTION_RATIO= 0.5
    lasso_top_n= 50

    # 1. 加载数据
    X, y, gene_list = load_and_preprocess_data(file_name='Colorectal_GSE21510_byGene.csv', label_line_name='type')

    if X is None:
        print("数据加载失败，程序终止。")
    else:
        # --- 方案一：GA (基于 Pearson 图) ---
        # (调用 CDGAFS 来执行预筛选、构建 Pearson 图、运行 GA，并返回物料)
        print("\n" + "="*50)
        print("方案一: 运行 GA (基于 Pearson 图) 并准备物料")
        print("="*50)
        
        (selected_indices_ga, 
         X_subset, 
         top_indices, 
         gene_list_subset, 
         _) = cdgafs_feature_selection(
            X, y, gene_list, 
            theta=THETA, 
            omega=GA_OMEGA, 
            population_size=GA_POPULATION_SIZE,
            w_bio_boost=W_BIO_BOOST,
            pre_filter_top_n=PRE_FILTER_TOP_N,
            graph_type='pearson_only'  # <-- 明确要求方案一使用 Pearson 图
        )
        
        print(f"GA (Pearson图) 最终选出 {len(selected_indices_ga)} 个特征。")
        save_gene_list(selected_indices_ga, gene_list, RESULTS_DIR, "ga_pearson_selected_genes.txt")
        
        # # --- 方案二：新方法 (中心性 + LASSO) ---
        print("\n" + "="*50)
        print("方案二: 新方法 (中心性排序 + LASSO精选，基于 Fused 图)")
        print("="*50)
        
        print("方案二: 步骤 2.1 (Centrality) - 构建 Fused Graph (乘法融合)...")
        # (使用返回的物料)
        graph_centrality = construct_feature_graph(
            X_subset, y, gene_list_subset, THETA, w_bio_boost=W_BIO_BOOST
        )
        
        print("方案二: 步骤 2.2 (Centrality) - 在 Fused 图上进行社团检测...")
        # 根据融合图检测社区
        partition_centrality = iscd_algorithm_auto_k(graph_centrality)
        clusters_centrality_dict = defaultdict(list)
        for node, community in partition_centrality.items():
            clusters_centrality_dict[community].append(node)
        clusters_centrality = [cluster for cluster in clusters_centrality_dict.values()]
        print(f"Centrality 方案检测到 {len(clusters_centrality)} 个社团。")

        print("方案二: 步骤 2.3 (Centrality) - 中心性排序初筛...")
        # (使用返回的 top_indices 进行映射)
        selected_indices_centrality = select_features_by_centrality(
            graph_centrality, clusters_centrality, top_indices, selection_ratio=SELECTION_RATIO
        )

        num_centrality_features = len(selected_indices_centrality)
        print(f"中心性排序从 {PRE_FILTER_TOP_N} 个特征中初筛出 {num_centrality_features} 个候选特征。")
        save_gene_list(selected_indices_centrality, gene_list, RESULTS_DIR, "centrality_selected_genes.txt")


        # **** 新增的评估步骤 ****
        print("\n--- 评估【中心性排序后】的特征集性能 ---")
        results_centrality = evaluate_with_repeats(X, y, selected_indices_centrality, "Centrality-only")


        # --- 步骤 2.2: LASSO精选 ---
        X_candidate = X[:, selected_indices_centrality]
        final_selected_indices_lasso = refine_features_with_lasso(X_candidate, y, selected_indices_centrality,lasso_top_n)
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

        # print("\n【方案二: 新方法 (分步评估)】")
        # print(f"  - 步骤 2.1 (中心性排序后):")
        # print(f"    - 特征数量: {num_centrality_features}")
        # print(f"    - 分类性能:")
        # # 再次打印一次结果，方便集中对比
        # for clf, metrics in results_centrality.items():
        #     print(f"      - {clf}: Acc={metrics['Accuracy']:.4f}")

        print("\n【方案二: 中心性+LASSO后】")
        print(f"    - 特征数量: {num_lasso_features}")
        print(f"    - 分类性能:")
        # 再次打印一次结果，方便集中对比
        for clf, metrics in results_lasso.items():
            print(f"      - {clf}: Acc={metrics['Accuracy']:.4f}")

        print("#"*70)
