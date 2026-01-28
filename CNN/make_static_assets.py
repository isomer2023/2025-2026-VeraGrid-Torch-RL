import os
import torch
import numpy as np
import simbench as sb
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian, reverse_cuthill_mckee
from scipy.sparse.linalg import eigsh

# ================= ⚙️ 绝对路径配置 (防迷路版) =================
# 1. 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 强制指定数据保存目录在当前脚本旁边
SAVE_DIR = os.path.join(CURRENT_DIR, "dataset_output_1MVLV-urban")

# 3. 参数设置
SB_CODE = "1-MVLV-urban-5.303-0-no_sw"#"1-MV-urban--0-sw"
PE_DIM = 8  # 🔥 已按你的要求改为 8 层
# ========================================================

os.makedirs(SAVE_DIR, exist_ok=True)


def get_topology_info(net):
    n_bus = len(net.bus)
    bus_idx_map = {b: i for i, b in enumerate(net.bus.index)}

    from_bus = []
    to_bus = []

    for _, line in net.line.iterrows():
        if line.in_service:
            f = bus_idx_map.get(line.from_bus)
            t = bus_idx_map.get(line.to_bus)
            if f is not None and t is not None:
                from_bus.extend([f, t])
                to_bus.extend([t, f])

    for _, trafo in net.trafo.iterrows():
        if trafo.in_service:
            f = bus_idx_map.get(trafo.hv_bus)
            t = bus_idx_map.get(trafo.lv_bus)
            if f is not None and t is not None:
                from_bus.extend([f, t])
                to_bus.extend([t, f])

    edges_src = np.array(from_bus)
    edges_dst = np.array(to_bus)
    data = np.ones(len(edges_src))

    adj = csr_matrix((data, (edges_src, edges_dst)), shape=(n_bus, n_bus))

    print("🔄 计算 RCM 重排...")
    perm = reverse_cuthill_mckee(adj)

    print(f"🧮 计算 Laplacian PE (前 {PE_DIM} 维)...")
    adj_reordered = adj[perm][:, perm]
    lap = laplacian(adj_reordered, normed=True)
    vals, vecs = eigsh(lap, k=PE_DIM + 1, which='SM')
    pe = vecs[:, 1:]

    return perm, torch.from_numpy(pe).float()


def main():
    print(f"📍 脚本位置: {CURRENT_DIR}")
    print(f"📂 数据将生成在: {SAVE_DIR}")
    print("-" * 40)

    net = sb.get_simbench_net(SB_CODE)
    perm, pe = get_topology_info(net)

    assets = {
        'perm': perm,
        'pe': pe,
        'num_nodes': len(net.bus)
    }

    save_path = os.path.join(SAVE_DIR, "static_assets.pt")
    torch.save(assets, save_path)
    print(f"✅ 静态资产已保存: {save_path}")
    print(f"   PE Shape: {pe.shape}")


if __name__ == "__main__":
    main()