import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from gprofiler import GProfiler
import requests
from sklearn.metrics import pairwise_distances

def read_gmt(gmt_path):
    """
    解析 .gmt 文件，返回两列 DataFrame: ['Gene','Pathway']
    """
    gene2path = []
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            pathway = parts[0]
            genes = parts[2:]       # 第一列是通路名，第二列常为描述，后面才是基因列表
            for g in genes:
                gene2path.append((g, pathway))
    return pd.DataFrame(gene2path, columns=['Gene','Pathway'])

# 2. 构建网络的函数
def build_pathway_cooccurrence_network_offline_fast(
    gene_list, thresh=0.0, normalize=True 
):
    df_kegg     = read_gmt('/data/qh_20T_share_file/lct/CDGAFS/gmt/c2.cp.kegg.v7.5.1.symbols.gmt.txt')
    df_reactome = read_gmt('/data/qh_20T_share_file/lct/CDGAFS/gmt/ReactomePathways.gmt')
    gene_pathway_df = pd.concat([df_kegg, df_reactome], ignore_index=True)

    # 1. 标准化并筛选
    genes = [g.strip().upper() for g in gene_list]
    df = gene_pathway_df.copy()
    df['Gene'] = df['Gene'].str.upper().str.strip()
    df_sub = df[df['Gene'].isin(genes)]

    # 2. 构建 M 矩阵 (genes × pathways)
    pathways = df_sub['Pathway'].unique().tolist()
    M_df = pd.crosstab(df_sub['Gene'], df_sub['Pathway'])
    M_df = M_df.reindex(index=genes, columns=pathways, fill_value=0)
    M = M_df.values.astype(int)

    # 3. 一次性计算 Jaccard 相似度
    adj_matrix = 1 - pairwise_distances(M, metric="jaccard")
    np.fill_diagonal(adj_matrix, 0)  # 对角线置零
    
    # 4. 阈值 & 归一化
    if thresh > 0:
        adj_matrix[adj_matrix <= thresh] = 0.0
    if normalize and adj_matrix.max() > 0:
        adj_matrix /= adj_matrix.max()

    return adj_matrix

# 示例调用：
# expr = pd.read_csv('/data/qh_20T_share_file/lct/论文代码复现/乳腺癌基因表达数据.csv', index_col=0)
# gene_list = sorted(expr.index)
# A_fast = build_pathway_cooccurrence_network_offline_fast(gene_list, thresh=0.05)
