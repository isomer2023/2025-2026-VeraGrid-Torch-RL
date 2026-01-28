# 5_runtime_compare.py
# ============================================================
# Runtime comparison of Hybrid / GCN / MLP / RF / VeraGrid OPF
# ============================================================

import os
import time
import glob
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import simbench as sb
import GC_PandaPowerImporter

from VeraGridEngine import api as gce

warnings.filterwarnings("ignore")

# ================= ⚙️ 配置 =================
SB_CODE = "1-MV-urban--0-sw"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "dataset_output_1mv_urban20260127_152301")

STATS_PATH = os.path.join(DATA_DIR, "stats.pt")
ASSETS_PATH = os.path.join(DATA_DIR, "static_assets.pt")

HYBRID_CKPT = "best_hybrid_transformer20260127_152301.pth"

NUM_TEST_SCENES = 10
BATCH_SIZE = 1
IN_DIM = 6
D_MODEL = 128
RF_ESTIMATORS = 50
# ============================================================


# ================= Dataset =================
class GridDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    x = torch.stack([b["x"] for b in batch])
    y = torch.stack([b["y"] for b in batch])
    mask = torch.stack([b["mask"] for b in batch])
    return x, y, mask


# ================= Feature Engineering =================
def process_batch_features(x_raw, x_mean, x_std):
    x_norm = (x_raw[:, :3, :] - x_mean[:, :3, :]) / (x_std[:, :3, :] + 1e-6)
    v_raw = x_raw[:, 3, :]
    v_phys = (v_raw - 1.0) / 0.05
    ang = x_raw[:, 4, :]

    feat = torch.cat([
        x_norm,
        v_phys.unsqueeze(1),
        torch.sin(ang).unsqueeze(1),
        torch.cos(ang).unsqueeze(1)
    ], dim=1)

    return feat.transpose(1, 2).contiguous()   # (B, N, 6)


# ================= Models =================
class BaselineMLP(nn.Module):
    def __init__(self, in_dim=6, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x, adj=None):
        return torch.sigmoid(self.net(x)).squeeze(-1)


class NativeGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.GELU()

    def forward(self, x, adj):
        out = self.linear(x)
        out = torch.einsum("nm,bmf->bnf", adj, out)
        return self.act(out)


class BaselineGNN(nn.Module):
    def __init__(self, in_dim=6, d_model=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.gcn1 = NativeGCNLayer(d_model, d_model)
        self.gcn2 = NativeGCNLayer(d_model, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, adj):
        h = self.proj(x)
        h = h + self.gcn1(h, adj)
        h = h + self.gcn2(h, adj)
        return torch.sigmoid(self.head(h)).squeeze(-1)


# ===== Hybrid Transformer =====
from models import HybridGridTransformer


# ================= Timing Helper =================
def timed_forward(fn):
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


# ================= OPF =================
def run_opf(grid):
    opts = gce.OptimalPowerFlowOptions()
    opts.solver = gce.SolverType.NONLINEAR_OPF
    drv = gce.OptimalPowerFlowDriver(grid, opts)

    t0 = time.perf_counter()
    drv.run()
    return time.perf_counter() - t0


# ================= Main =================
def main():
    print("📂 Loading data & assets...")

    # =========================================================
    # 1. 加载数据（ML / RF 用）
    # =========================================================
    all_data = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "chunk_*.pt"))):
        all_data.extend(torch.load(f, weights_only=False))

    stats = torch.load(STATS_PATH, map_location=DEVICE, weights_only=False)
    x_mean = stats["x_mean"].view(1, -1, 1).to(DEVICE)
    x_std  = stats["x_std"].view(1, -1, 1).to(DEVICE)

    assets = torch.load(ASSETS_PATH, map_location=DEVICE, weights_only=False)
    edge_index = assets["edge_index"]
    dist_matrix = assets["dist_matrix"].to(DEVICE)
    num_nodes = int(assets["num_nodes"])

    # =========================================================
    # 2. 构建 GCN 邻接矩阵
    # =========================================================
    adj = torch.zeros((num_nodes, num_nodes), device=DEVICE)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj += torch.eye(num_nodes, device=DEVICE)
    d = adj.sum(1).pow(-0.5)
    d[torch.isinf(d)] = 0.0
    adj = torch.diag(d) @ adj @ torch.diag(d)

    # =========================================================
    # 3. 随机选 10 个测试场景（ML / RF）
    # =========================================================
    idx = np.random.choice(len(all_data), NUM_TEST_SCENES, replace=False)
    test_data = [all_data[i] for i in idx]
    dl = DataLoader(
        GridDataset(test_data),
        batch_size=1,
        collate_fn=collate_fn
    )

    # =========================================================
    # 4. 加载模型
    # =========================================================
    mlp = BaselineMLP(IN_DIM, D_MODEL).to(DEVICE).eval()
    gnn = BaselineGNN(IN_DIM, D_MODEL).to(DEVICE).eval()

    hybrid = HybridGridTransformer(
        in_dim=IN_DIM,
        d_model=128,
        n_heads=8,
        n_layers=6,
        d_ff=256
    ).to(DEVICE)
    hybrid.load_state_dict(torch.load(HYBRID_CKPT, map_location=DEVICE))
    hybrid.eval()

    # =========================================================
    # 5. 训练 RF（只用一次）
    # =========================================================
    rf = RandomForestRegressor(n_estimators=RF_ESTIMATORS, n_jobs=-1)

    X_rf, y_rf = [], []
    for s in all_data[:500]:
        x_raw = s["x"].unsqueeze(0).to(DEVICE)
        x_feat = process_batch_features(x_raw, x_mean, x_std)
        X_rf.append(x_feat.reshape(-1, IN_DIM).cpu().numpy())
        y_rf.append(s["y"].reshape(-1).numpy())

    rf.fit(np.vstack(X_rf), np.concatenate(y_rf))

    # =========================================================
    # 6. ML / RF 推理 + 计时
    # =========================================================
    results = {}

    for name in ["MLP", "GCN", "Hybrid", "RF"]:
        times, ys, ps = [], [], []

        for x, y, mask in dl:
            x = x.to(DEVICE)
            y = y.to(DEVICE).squeeze(1)
            mask = mask.to(DEVICE)

            x_feat = process_batch_features(x, x_mean, x_std)

            with torch.no_grad():
                if name == "MLP":
                    pred, t = timed_forward(lambda: mlp(x_feat))
                elif name == "GCN":
                    pred, t = timed_forward(lambda: gnn(x_feat, adj))
                elif name == "Hybrid":
                    bias = dist_matrix.unsqueeze(0).unsqueeze(0)
                    pred, t = timed_forward(lambda: hybrid(x_feat, adj, bias))
                else:  # RF
                    X = x_feat.reshape(-1, IN_DIM).cpu().numpy()
                    _, t = timed_forward(lambda: rf.predict(X))
                    pred = torch.tensor(
                        rf.predict(X).reshape(1, -1),
                        device=DEVICE
                    )

            times.append(t)
            ys.extend(y[mask].cpu().numpy())
            ps.extend(pred[mask].cpu().numpy())

        results[name] = {
            "time": np.mean(times),
            "mae": mean_absolute_error(ys, ps),
            "r2": r2_score(ys, ps)
        }

    # =========================================================
    # 7. OPF Runtime Benchmark（独立于 ML 数据）
    #    完全复用 2_data_generation.py 的建网方式
    # =========================================================

    opf_times = []

    for _ in range(NUM_TEST_SCENES):
        # --- SimBench → pandapower ---
        net_pp = sb.get_simbench_net(SB_CODE)

        # 负荷扰动（轻量即可）
        scale = np.random.uniform(0.9, 1.1)
        net_pp.load.p_mw *= scale
        net_pp.load.q_mvar *= scale

        # --- pandapower → VeraGrid ---
        grid = GC_PandaPowerImporter.PP2GC(net_pp)

        # --- OPF ---
        opf_times.append(run_opf(grid))

    results["OPF"] = {
        "time": np.mean(opf_times),
        "mae": 0.0,
        "r2": 1.0
    }

    # =========================================================
    # 8. 输出结果
    # =========================================================
    print("\n================ Runtime Comparison ================")
    for k, v in results.items():
        print(
            f"{k:10s} | "
            f"Time: {v['time']:.4f}s | "
            f"MAE: {v['mae']:.4f} | "
            f"R2: {v['r2']:.4f}"
        )


if __name__ == "__main__":
    main()
