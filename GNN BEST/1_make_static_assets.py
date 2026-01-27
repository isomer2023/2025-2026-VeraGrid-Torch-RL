import os
import torch
import numpy as np
import simbench as sb
import networkx as nx
import pandapower.topology as top
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee, laplacian
from scipy.sparse.linalg import eigsh

# ⚙️ 配置保持一致
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(CURRENT_DIR, "dataset_output_1mv_urban")
SB_CODE = "1-MV-urban--0-sw"
PE_DIM = 8

os.makedirs(SAVE_DIR, exist_ok=True)


def get_topology_info_v2(net):
    # 1. 建立 NetworkX 图 (最稳健的方法)
    G_nx = top.create_nxgraph(net, respect_switches=False)

    # 原始索引映射
    original_bus_indices = net.bus.index.tolist()
    n_bus = len(original_bus_indices)
    id_to_idx = {bid: i for i, bid in enumerate(original_bus_indices)}

    # 2. RCM 重排计算
    print("🔄 计算 RCM 重排...")
    adj = nx.adjacency_matrix(G_nx, nodelist=original_bus_indices)
    perm = reverse_cuthill_mckee(adj)

    # 建立映射: 旧SimBench索引 -> RCM新索引
    # 先把 perm 转成 "old -> new" 的查找表
    # perm[k] 是新位置 k 对应的旧索引
    # 所以 old_to_new[perm[k]] = k
    old_to_new = np.zeros(n_bus, dtype=int)
    old_to_new[perm] = np.arange(n_bus)

    # 3. 提取 Edge Index 并 **重映射到新索引** (关键修复)
    print("🔗 提取并重映射边索引...")
    src_list, dst_list = [], []
    for u, v in G_nx.edges():
        if u in id_to_idx and v in id_to_idx:
            # 原始 SimBench ID -> 0..N 索引 -> RCM 新索引
            new_u = old_to_new[id_to_idx[u]]
            new_v = old_to_new[id_to_idx[v]]
            src_list.extend([new_u, new_v])
            dst_list.extend([new_v, new_u])  # 双向

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # 4. 计算最短路径矩阵 (给 Transformer 用)
    print("📏 计算全图最短路径 (Floyd-Warshall)...")
    # 计算原始距离
    dist_mat = nx.floyd_warshall_numpy(G_nx, nodelist=original_bus_indices)
    # 处理不连通 (inf)
    if not np.isfinite(dist_mat).all():
        dist_mat[~np.isfinite(dist_mat)] = 100.0
        # **重排距离矩阵** (行列都要变)
    dist_mat_perm = dist_mat[perm][:, perm]
    dist_tensor = torch.from_numpy(dist_mat_perm).float()

    # 5. 计算 PE (备用)
    print("🧮 计算位置编码 PE...")
    try:
        adj_reordered = adj[perm][:, perm]
        lap = laplacian(adj_reordered, normed=True)
        vals, vecs = eigsh(lap, k=PE_DIM + 1, which='SM')
        pe = vecs[:, 1:]
        if pe.shape[1] < PE_DIM:
            pad = np.zeros((pe.shape[0], PE_DIM - pe.shape[1]))
            pe = np.hstack([pe, pad])
    except:
        pe = np.random.randn(n_bus, PE_DIM)

    return perm, torch.from_numpy(pe).float(), edge_index, dist_tensor


def main():
    print(f"🚀 生成静态资产 -> {SAVE_DIR}")
    net = sb.get_simbench_net(SB_CODE)

    perm, pe, edge_index, dist_matrix = get_topology_info_v2(net)

    assets = {
        'perm': perm,  # 以后 dataset 用这个重排 X
        'edge_index': edge_index,  # ✅ 修复：GNN 用这个 (已对齐)
        'dist_matrix': dist_matrix,  # ✅ 新增：Transformer 用这个 bias
        'pe': pe,
        'num_nodes': len(net.bus)
    }

    torch.save(assets, os.path.join(SAVE_DIR, "static_assets.pt"))
    print("✅ static_assets.pt 更新完毕！")
    print(f"   Nodes: {assets['num_nodes']}")
    print(f"   Edges: {edge_index.shape[1]}")


if __name__ == "__main__":
    main()