import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from normalize_scores import normalize_scores  # 归一化函数
from draw_graph import draw_graph
from mi_label import build_mi_feature, build_mi_label
from pathway_cooccurrence_network import build_pathway_cooccurrence_network_offline_fast
from PPIN import build_string_ppi_adjacency

def construct_feature_graph(X, y, gene_list, theta, alpha=1, beta=0, mi_bins=10, mi_discrete_threshold=20):
    """
    融合Pearson和MI的特征图构建
    Args:
        X (np.ndarray): 特征矩阵 (n_samples, n_features)
        y (np.ndarray): 标签数组，形状为 (样本数, ) 或 (样本数, 1)。
        theta (float): 边权重过滤阈值 (0-1)
        alpha (float): Pearson权重 (默认0.4)
        beta (float): MI权重 (默认0.6)
        mi_bins (int): MI计算的分箱数
        mi_discrete_threshold (int): 离散特征判定阈值
    Returns:
        networkx.Graph
    """
    # 计算 Pearson 相关系数矩阵
    pearson_matrix = np.corrcoef(X, rowvar=False)  # 计算相关系数矩阵，rowvar=False 表示按列计算
    # 取绝对值
    abs_corr = np.abs(pearson_matrix)
    #软最大缩放归一化
    pearson_norm = normalize_scores(abs_corr)

    # 2. 互信息矩阵
    # mi_feature = build_mi_feature(X, bins=mi_bins, discrete_threshold=mi_discrete_threshold)
    # mi_feature_norm = normalize_scores(mi_feature)

    # mi_label = build_mi_label(X, y)
    # mi_label_norm = normalize_scores(mi_label)

    # 3. 通路计算需要基因名称：读入 gene list（只要行索引）
    gene_pathways_matrix = build_pathway_cooccurrence_network_offline_fast(gene_list, thresh=0.05)

    # # 4. 蛋白质互作用网络
    ppi_matrix = build_string_ppi_adjacency(gene_list, score_threshold=None)
    
    # # --- 矩阵融合与过滤 ---    
    mix_matrix = 0 * gene_pathways_matrix + 0.5 * ppi_matrix
    combined_matrix =  0.5 * pearson_norm  +  mix_matrix

    # 将前面融合得到 combined_matrix（可能是 DataFrame）转成 numpy 数组
    combined_matrix = np.asarray(combined_matrix, dtype=float)
    # 记录删除前的边数
    num_edges_before = np.count_nonzero(combined_matrix) // 2  # 因为矩阵是对称的，计算非零元素的数量并除以2

    # 根据阈值过滤边
    combined_matrix[combined_matrix < theta] = 0
    
    # 记录删除后的边数
    num_edges_after = np.count_nonzero(combined_matrix) // 2  # 同样计算非零元素的数量并除以2

    # 打印删除的边数
    print(f"根据阈值过滤的边数: {num_edges_before - num_edges_after}")

    # 设置对角线为 0，避免自环
    np.fill_diagonal(combined_matrix, 0)
    
    # 用零替代无效值
    combined_matrix = np.nan_to_num(combined_matrix, nan=0)

    # 将邻接矩阵转换为 NetworkX 图
    G = nx.from_numpy_array(combined_matrix)
    # draw_graph(G)

    print("图:")
    print(G)
    # # 为边添加权重属性（增强可解释性）
    # for u, v, data in G.edges(data=True):
    #     data['weight'] = filtered_corr[u, v]

    return G

def construct_pearson_only_graph(X, theta):
    """
    仅使用皮尔逊相关系数构建特征图。
    这是用于消融实验的对照组函数。
    """
    print("\n--- 构建仅基于皮尔逊的图 ---")
    # 计算 Pearson 相关系数矩阵
    pearson_matrix = np.corrcoef(X, rowvar=False)
    abs_corr = np.abs(pearson_matrix)
    
    # 使用你项目中的归一化方法
    pearson_norm = normalize_scores(abs_corr)
    
    # 复制一份用于过滤
    graph_matrix = pearson_norm.copy()
    
    # 根据阈值过滤边
    num_edges_before = np.count_nonzero(graph_matrix) // 2
    graph_matrix[graph_matrix < theta] = 0
    num_edges_after = np.count_nonzero(graph_matrix) // 2
    print(f"根据阈值 {theta} 过滤的边数: {num_edges_before - num_edges_after}")

    np.fill_diagonal(graph_matrix, 0)
    graph_matrix = np.nan_to_num(graph_matrix, nan=0)
    
    G = nx.from_numpy_array(graph_matrix)
    print("皮尔逊图构建完成:")
    print(G)
    return G