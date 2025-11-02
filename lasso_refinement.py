# 导入必要的库
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import numpy as np

def refine_features_with_lasso(X_candidate, y, candidate_indices):
    """
    使用L1正则化的逻辑回归 (LASSO) 从候选特征集中精选特征。

    参数:
        X_candidate (np.ndarray): 经过中心性等方法初步筛选后的特征矩阵 (样本数, 候选特征数)。
        y (np.ndarray): 标签数组。
        candidate_indices (np.ndarray): 候选特征在原始2万多个特征中的索引。

    返回:
        final_indices (list): 经过LASSO最终筛选出的特征在原始数据集中的索引列表。
    """
    print("\n--- 步骤三: 使用LASSO进行特征精选 ---")
    
    # 1. 数据标准化 (重要！LASSO对数据尺度敏感)
    # 尽管您的数据已经归一化，但在送入LASSO前再次进行标准化是一个好习惯。
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_candidate)

    # 2. 使用LogisticRegressionCV进行LASSO筛选
    # penalty='l1'表示使用LASSO。
    # solver='liblinear'或'saga'支持L1惩罚。
    # cv=5表示5折交叉验证来寻找最佳正则化强度C。
    # Cs=10可以理解为尝试10个不同的正则化强度。
    l1_logreg = LogisticRegressionCV(
        Cs=10, 
        cv=5, 
        penalty='l1', 
        solver='liblinear', 
        random_state=42,
        max_iter=1000  # 增加最大迭代次数以确保收敛
    )
    
    l1_logreg.fit(X_scaled, y)
    print(f"LASSO通过交叉验证找到的最佳正则化强度C值为: {l1_logreg.C_[0]:.4f}")

    # 3. 提取系数不为零的特征
    # l1_logreg.coef_[0] 包含了每个候选特征的系数
    non_zero_coeffs_mask = np.abs(l1_logreg.coef_[0]) > 1e-5  # 使用一个很小的阈值来判断非零
    
    # 获取这些非零系数在 a_candidate 矩阵中的索引
    selected_indices_in_candidate = np.where(non_zero_coeffs_mask)[0]
    
    if len(selected_indices_in_candidate) == 0:
        print("警告: LASSO筛选后没有剩下任何特征。可能是正则化强度过高或候选特征集与标签的关联性弱。")
        return []

    print(f"LASSO从 {X_candidate.shape[1]} 个候选特征中精选出 {len(selected_indices_in_candidate)} 个特征。")

    # 4. 映射回原始索引
    # 这是关键一步，将 candidate 空间中的索引映射回2万多个基因的原始空间
    final_indices = [candidate_indices[i] for i in selected_indices_in_candidate]

    return final_indices