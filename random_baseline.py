# random_baseline.py

import random
import numpy as np

# 确保 fisher_score.py 在同一个目录下，或者在Python的搜索路径中
from fisher_score import compute_fisher_score

def random_selection_baseline(X, y, pre_filter_top_n, num_features_to_select):
    """
    执行一个基线特征选择方法：先用Fisher Score预筛选，然后随机选择。
    
    参数:
        X (np.ndarray): 完整的特征矩阵
        y (np.ndarray): 标签数组
        pre_filter_top_n (int): 预筛选要保留的特征数量。
        num_features_to_select (int): 最终要随机选择的特征数量。
    
    返回:
        selected_original_indices (list): 最终随机选定的特征在 *原始* 矩阵中的索引列表。
    """
    print("\n" + "="*50)
    print("开始执行... 随机选择基线方法")
    print("="*50)

    # 步骤一：Fisher Score 预筛选
    print("步骤一: (随机基线) - 计算 Fisher Scores 并预筛选...")
    fisher_scores = compute_fisher_score(X, y)
    
    # 获取得分最高的 top N 特征的原始索引
    top_indices = np.argsort(fisher_scores)[-pre_filter_top_n:]
    print(f"预筛选完成，保留 Fisher Score 最高的 {len(top_indices)} 个特征。")

    # 步骤二：从预筛选池中进行随机选择
    print(f"\n步骤二: (随机基线) - 从预筛选的 {len(top_indices)} 个特征中，随机选择 {num_features_to_select} 个...")
    
    if num_features_to_select > len(top_indices):
        print(f"警告：要选择的数量({num_features_to_select})大于预筛选池({len(top_indices)})，将使用所有预筛选特征。")
        selected_original_indices = list(top_indices)
    else:
        # 从预筛选出的 top_indices 中，随机抽取指定数量的特征
        selected_original_indices = random.sample(list(top_indices), num_features_to_select)
        
    print(f"已从预筛选池中随机选择 {len(selected_original_indices)} 个特征。")
    print("="*50)
    
    return selected_original_indices

def pure_random_selection(X, num_features_to_select):
    """
    执行一个最基础的基线方法：直接从所有特征中随机选择。
    完全不使用任何预筛选步骤。
    
    参数:
        X (np.ndarray): 完整的特征矩阵，用于确定特征总数。
        num_features_to_select (int): 最终要随机选择的特征数量。
    
    返回:
        selected_original_indices (list): 最终随机选定的特征索引列表。
    """
    print("\n" + "="*50)
    print("开始执行... 纯随机选择基线方法 (无预筛选)")
    print("="*50)

    # 获取原始特征的总数
    total_features = X.shape[1]
    print(f"特征总池大小: {total_features}")

    # 创建一个包含所有可能索引的列表
    all_indices = list(range(total_features))

    # 步骤一：直接从所有索引中随机抽样
    print(f"\n步骤一: 从 {total_features} 个特征中，随机选择 {num_features_to_select} 个...")
    
    if num_features_to_select > total_features:
        print(f"警告：要选择的数量({num_features_to_select})大于特征总数({total_features})，将返回所有特征。")
        selected_original_indices = all_indices
    else:
        selected_original_indices = random.sample(all_indices, num_features_to_select)
        
    print(f"已从总池中随机选择 {len(selected_original_indices)} 个特征。")
    print("="*50)
    
    return selected_original_indices