# =================================================================
# 文件: CDGAFS.py (修改后)
# 主要改动：
# 1. 函数签名增加了 pre_filter_top_n=None 参数。
# 2. 在函数内部，增加了预筛选逻辑：如果 pre_filter_top_n 被设置，
#    则只选择 Fisher Score 最高的N个特征进行后续处理。
# 3. 在函数末尾，将选出的特征子集的索引映射回它们在原始2万多个特征中的索引。
# =================================================================
import pandas as pd
import numpy as np
from collections import defaultdict
import random
from fisher_score import compute_fisher_score
from normalize_scores import normalize_scores
from construct_feature_graph import construct_feature_graph, construct_pearson_only_graph
from compare_graphs import construct_pearson_only_graph, analyze_and_visualize
from compute_similarity_matrix import compute_similarity_matrix
from community_detection import iscd_algorithm_auto_k
from calculate_fitness import calculate_fitness 
from genetic_algorithm_utils import initialize_population, genetic_algorithm
from final_subset_selection import final_subset_selection

def cdgafs_feature_selection(X, y, gene_list, theta, omega, population_size, pre_filter_top_n=None, graph_type='fused'):
    """
    输入:
        X (np.ndarray): 原始完整特征矩阵
        y (np.ndarray): 标签
        gene_list (list): 原始完整基因列表
        theta (float): 特征图构造阈值
        omega (float): 每个社区选取的特征数比例
        population_size (int): 遗传算法种群大小
        pre_filter_top_n (int, optional): 预筛选的特征数量。如果提供，则只对前N个特征进行处理。
        graph_type (str): 'fused' 或 'pearson_only'，用于选择图构建模式。
    返回:
        selected_original_indices (list): 选定的特征在 *原始* 矩阵中的索引列表
    """

    # Step 1: 计算 Fisher Scores
    print("Step 1: 计算 Fisher Scores...")
    fisher_scores = compute_fisher_score(X, y)
    
    # ========================> 核心修改：预筛选 <========================
    original_indices = np.arange(X.shape[1])
    
    if pre_filter_top_n is not None and pre_filter_top_n < X.shape[1]:
        print(f"执行预筛选，选取 Fisher Score 最高的 {pre_filter_top_n} 个特征...")
        # 获取得分最高的 top N 特征的索引
        top_indices = np.argsort(fisher_scores)[-pre_filter_top_n:]
        
        # 根据这些索引筛选数据
        X_subset = X[:, top_indices]
        fisher_scores_subset = fisher_scores[top_indices]
        gene_list_subset = [gene_list[i] for i in top_indices]
        
        # 保存原始索引的映射
        original_indices = top_indices
        
        print(f"预筛选完成，特征数量从 {X.shape[1]} 减少到 {X_subset.shape[1]}")
    else:
        print("未执行预筛选，使用全部特征。")
        X_subset = X
        fisher_scores_subset = fisher_scores
        gene_list_subset = gene_list
    # =================================================================

    normalized_fisher_scores = normalize_scores(fisher_scores_subset)
    print("归一化后的 Fisher Scores (子集):", normalized_fisher_scores[:5]) # 打印前5个看看

    # Step 2: 构造特征图 (在数据子集上进行)
    print(f"\nStep 2: 构造特征图 (模式: {graph_type})...")
    
    if graph_type == 'fused':
        feature_graph = construct_feature_graph(X_subset, y, gene_list_subset, theta)
    elif graph_type == 'pearson_only':
        feature_graph = construct_pearson_only_graph(X_subset, theta)
    else:
        raise ValueError("graph_type 必须是 'fused' 或 'pearson_only'")

    similarity_matrix = compute_similarity_matrix(X_subset)

    # ========================> 核心修改 <========================
    # 原代码，下面 clusters = defaultdict(list) 一同注释
    # # Step 3: 社区检测
    # print("\nStep 3: 进行社区检测...")
    # partition = iscd_algorithm_auto_k(feature_graph)

    
    # Step 3: 将所有节点定义为一个社区 (跳过ISCD)
    print("\nStep 3: 跳过社群检测，将所有节点定义为一个大社区...")
    # 获取特征子集中的节点总数（即特征数量）
    num_features_subset = X_subset.shape[1]
    # 创建一个包含所有节点索引的列表
    all_nodes_as_one_community = list(range(num_features_subset))
    # `clusters` 的数据结构是一个列表的列表，所以我们将上面的列表作为唯一的元素
    clusters = [all_nodes_as_one_community]
    partition = {node: 0 for node in range(num_features_subset)}
    print(f"已手动定义 {len(clusters)} 个社群。")
    # ============================================================

    # 新增步骤: 使用上一步的结果进行图结构分析
    # print("\n--- 开始进行图结构分析 ---")
    # if graph_type == 'fused':
    #     analyze_and_visualize(feature_graph, partition, "Fused_Knowledge_Graph", gene_list_subset)
    # elif graph_type == 'pearson_only':
    #     analyze_and_visualize(feature_graph, partition, "Pearson_Only_Graph", gene_list_subset)

    # 使用同一个 partition 结果继续后续流程
    # clusters = defaultdict(list)
    # for node, community in partition.items():
    #     clusters[community].append(node)
    # clusters = [cluster for cluster in clusters.values()]
    # print(f"检测到 {len(clusters)} 个社区。")

    # Step 4: 初始化种群并执行遗传算法
    print("\nStep 4: 执行遗传算法...")
    num_features_subset = X_subset.shape[1]
    population = initialize_population(num_features_subset, clusters, omega, population_size)
    print(f"初始化种群大小: {len(population)}")
    
    fitness_values = calculate_fitness(population, X_subset, y, similarity_matrix, n_jobs=10)
    print(f"初始适应度值示例: {fitness_values[:5]}")

    population, fitness_values = genetic_algorithm(
        population, fitness_values, X_subset, y, clusters, omega, 
        similarity_matrix, population_size, num_features_subset, normalized_fisher_scores
    )
    
    # Step 5: 选择最终特征子集
    print("\nStep 5: 选择最终特征子集...")
    best_chromosome, selected_subset_indices = final_subset_selection(population, fitness_values)
    print(f"从子集中选出了 {len(selected_subset_indices)} 个特征。")
    
    # 将子集中的索引映射回原始数据集中的索引
    selected_original_indices = [original_indices[i] for i in selected_subset_indices]
    
    print(f"\n最终最佳染色体 (在子集上): {best_chromosome}")
    print(f"映射回原始索引: {selected_original_indices[:10]} ...") # 打印前10个看看
    
    return selected_original_indices


