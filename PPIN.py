import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

def build_string_ppi_adjacency(
    gene_list,
    prot_info_path='/data/qh_20T_share_file/lct/CDGAFS/network_of_data/9606.protein.info.v12.0.txt.gz',
    ppi_path='/data/qh_20T_share_file/lct/CDGAFS/network_of_data/9606.protein.links.v12.0.txt.gz',
    score_threshold=None
):
    """
    构建基因列表对应的 STRING PPI 加权邻接矩阵。

    参数:
        gene_list: list of str
            目标基因列表（与 STRING preferred_name 对应）。
        prot_info_path: str
            STRING protein info 文件路径（.txt.gz）。
        ppi_path: str
            STRING PPI 边表路径（.txt.gz）。
        score_threshold: int or None
            如果不为 None，只保留 combined_score > score_threshold 的边。

    返回:
        A: pd.DataFrame, shape=(len(gene_list), len(gene_list))
            对称加权邻接矩阵，行列均为 gene_list。
    """
    # 1. 读取 protein→gene 映射
    prot_info = pd.read_csv(
        prot_info_path,
        sep='\t', compression='gzip',
        usecols=['#string_protein_id','preferred_name']
    )
    prot_info['ensp'] = prot_info['#string_protein_id'].str.split('.').str[1]
    ensp2gene = dict(zip(prot_info['ensp'], prot_info['preferred_name']))

    # 2. 读取 PPI 边表
    ppi = pd.read_csv(
        ppi_path,
        sep=' ', compression='gzip',
        usecols=['protein1','protein2','combined_score']
    )
    # 3. （可选）过滤低置信度
    if score_threshold is not None:
        ppi = ppi[ppi['combined_score'] > score_threshold]

    # 4. 提取 ENSP 并映射到 gene
    ppi['ensp1'] = ppi['protein1'].str.split('.').str[1]
    ppi['ensp2'] = ppi['protein2'].str.split('.').str[1]
    ppi['gene1'] = ppi['ensp1'].map(ensp2gene)
    ppi['gene2'] = ppi['ensp2'].map(ensp2gene)
    ppi = ppi.dropna(subset=['gene1','gene2'])

    # 5. 仅保留都在 gene_list 中的交互
    ppi_sub = ppi[
        ppi['gene1'].isin(gene_list) &
        ppi['gene2'].isin(gene_list)
    ].copy()

    # 6. 归一化 combined_score 到 [0,1]
    scaler = MinMaxScaler(feature_range=(0,1))
    ppi_sub['weight'] = scaler.fit_transform(ppi_sub[['combined_score']])

    # 7. 构建 NetworkX 图
    G = nx.Graph()
    G.add_nodes_from(gene_list)
    edges = ppi_sub[['gene1','gene2','weight']].values.tolist()
    G.add_weighted_edges_from(edges)  # 默认使用 'weight'

    # 8. 导出邻接矩阵
    A = nx.to_pandas_adjacency(
        G,
        nodelist=gene_list,
        dtype=float,
        weight='weight'
    )
    return A