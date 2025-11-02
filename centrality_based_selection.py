import networkx as nx
import math

def select_features_by_centrality(G, clusters, original_indices, selection_ratio=0.05):
    """
    该方法在每个社区内部，根据节点的度中心性（连接度）进行排序，并选择Top N%的基因。

    参数:
        G (networkx.Graph): 预筛选后构建的特征图。
        clusters (list): 社区划分结果，一个包含多个列表的列表，每个子列表是一个社区。
        original_indices (np.array): 预筛选后，特征在原始2w+数据中的索引。
        selection_ratio (float): 在每个社区中要选择的特征比例，例如0.05代表Top 5%。

    返回:
        final_selected_original_indices (list): 最终选出的特征在 *原始* 矩阵中的索引列表。
    """
    print("\n" + "="*50)
    print("开始执行... 基于中心性的新选择方法")
    print("="*50)

    # 1. 一次性计算整个图所有节点的度中心性，效率最高
    # 度中心性衡量了一个节点有多少连接，是“连接度”最直接的体现
    print("步骤一: 计算所有节点的度中心性...")
    centrality = nx.degree_centrality(G)
    
    final_selected_subset_indices = set()

    print(f"步骤二: 从 {len(clusters)} 个社区中，按 {selection_ratio*100}% 的比例挑选核心基因...")
    # 2. 遍历每一个社区
    for i, community in enumerate(clusters):
        if not community:
            continue

        # 提取当前社区所有节点的中心性分数
        community_centrality = {node_idx: centrality.get(node_idx, 0) for node_idx in community}

        # 根据中心性分数对社区内的节点进行降序排序
        sorted_community = sorted(community_centrality.items(), key=lambda item: item[1], reverse=True)
        
        # 计算需要从这个社区中选择多少个基因
        # 至少选择一个，以保证每个社区都有代表
        num_to_select = math.ceil(len(community) * selection_ratio)
        if num_to_select == 0 and len(community) > 0:
            num_to_select = 1

        # 选出Top N%的基因
        top_nodes = sorted_community[:num_to_select]
        
        # 将选出的节点索引（相对于子集）添加到最终集合中
        for node_idx, _ in top_nodes:
            final_selected_subset_indices.add(node_idx)
    
    # 3. 将子集索引映射回原始的2w+基因的索引
    final_selected_original_indices = [original_indices[i] for i in final_selected_subset_indices]
    
    print(f"新方法选择完成，共选出 {len(final_selected_original_indices)} 个特征。")
    print("="*50)

    return final_selected_original_indices
