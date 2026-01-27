import os
import glob
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

# ================= ⚙️ 配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径配置 (请确保与生成脚本一致)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "dataset_output_1mv_urban")
STATS_PATH = os.path.join(DATA_DIR, "stats.pt")
ASSETS_PATH = os.path.join(DATA_DIR, "static_assets.pt")

SAVE_PATH = "best_hybrid_transformer.pth"
SCATTER_SAVE_PATH = "result_scatter_hybrid.png"

# 训练超参
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 50
NUM_WORKERS = 2
PIN_MEMORY = True

# 模型架构参数
IN_DIM = 6  # P, Q, PV, V, sin, cos
D_MODEL = 128
N_HEADS = 8  # 4个物理头 + 4个自由头
N_LAYERS = 6
D_FF = 256
DROPOUT = 0.1
BIAS_TAU = 2.0  # 物理距离衰减常数


# ==========================================


# ========== 1. Dataset & DataLoader ==========
class GridDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    x = torch.stack([item["x"] for item in batch])  # (B, 5, N)
    y = torch.stack([item["y"] for item in batch])  # (B, 1, N)
    mask = torch.stack([item["mask"] for item in batch])  # (B, N)
    return x, y, mask


# ========== 2. Data Loading & Graph Prep ==========
def load_data_and_assets():
    print("📂 Loading data chunks...")
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"❌ 找不到数据目录: {DATA_DIR}")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "chunk_*.pt")))
    if not files:
        raise FileNotFoundError("❌ 没有找到数据 chunk_*.pt，请先运行 dataset_generation.py")

    all_data = []
    for f in files:
        all_data.extend(torch.load(f, weights_only=False))
    print(f"✅ Loaded {len(all_data)} samples.")

    # 加载统计量
    stats = torch.load(STATS_PATH, weights_only=False, map_location=DEVICE)
    x_mean = stats["x_mean"].view(1, -1, 1)
    x_std = stats["x_std"].view(1, -1, 1)

    # 加载静态资产 (必须包含 dist_matrix 和 edge_index)
    print("🧠 Loading static topology assets...")
    assets = torch.load(ASSETS_PATH, weights_only=False, map_location=DEVICE)

    # 1. 准备 Transformer 的 Attn Bias
    # dist_matrix: (N, N) -> bias: (1, 1, N, N)
    dist_mat = assets["dist_matrix"].float()
    bias = -dist_mat / BIAS_TAU
    bias.fill_diagonal_(0.0)  # 自注意力由模型自己决定，Bias设为0
    attn_bias = bias.unsqueeze(0).unsqueeze(0)

    # 2. 准备 GCN 的 归一化邻接矩阵
    # edge_index: (2, E) -> Dense Adj (N, N)
    edge_index = assets["edge_index"]
    num_nodes = int(assets["num_nodes"])

    adj = torch.zeros((num_nodes, num_nodes), device=DEVICE)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = adj + torch.eye(num_nodes, device=DEVICE)  # 添加自环 A = A + I

    # 归一化: D^-0.5 * A * D^-0.5
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat = torch.diag(d_inv_sqrt)
    norm_adj = d_mat @ adj @ d_mat  # (N, N)

    return all_data, x_mean, x_std, norm_adj, attn_bias


def process_batch_features(x_raw, x_mean, x_std):
    """
    输入: (B, 5, N) -> P, Q, PV, V_mag, V_ang
    输出: (B, N, 6) -> P, Q, PV, V_phys, sin, cos
    """
    # 前3个通道 (P, Q, PV) 使用统计标准化
    x_norm_part = (x_raw[:, :3, :] - x_mean[:, :3, :]) / (x_std[:, :3, :] + 1e-6)

    # V_mag (通道3) 使用物理归一化 (V-1.0)/0.05
    v_raw = x_raw[:, 3, :]
    v_phys = (v_raw - 1.0) / 0.05

    # V_ang (通道4) 使用 sin/cos 嵌入
    ang = x_raw[:, 4, :]
    s = torch.sin(ang)
    c = torch.cos(ang)

    # 拼接
    x_feat = torch.cat([
        x_norm_part,  # (B, 3, N)
        v_phys.unsqueeze(1),  # (B, 1, N)
        s.unsqueeze(1),  # (B, 1, N)
        c.unsqueeze(1)  # (B, 1, N)
    ], dim=1)  # -> (B, 6, N)

    return x_feat.transpose(1, 2).contiguous()  # -> (B, N, 6)


# ========== 3. Model Components ==========

class NativeGCNLayer(nn.Module):
    """ 原生实现的 GCN 层: X' = A * X * W """

    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: (B, N, Fin), adj: (N, N)
        out = self.linear(x)  # (B, N, Fout)
        # 矩阵乘法: (N,N) x (B,N,F) -> (B,N,F)
        # 既然 B 在前，可以用 einsum 或者把 B 移到后面
        out = torch.einsum('nm, bmf -> bnf', adj, out)
        return self.drop(self.act(out))


class HybridMultiHeadAttention(nn.Module):
    """ 混合注意力: 一半头看物理距离，一半头自由学习 """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_phys = n_heads // 2  # 物理头数量

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        # 物理头的权重系数
        self.beta = nn.Parameter(torch.ones(1, self.n_phys, 1, 1))

    def forward(self, x, attn_bias):
        B, N, D = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # (B, H, N, d)
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.d_head)
        logits = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)

        # --- 混合处理 ---
        phys_logits = logits[:, :self.n_phys, :, :]
        free_logits = logits[:, self.n_phys:, :, :]

        # 物理头加上 Bias
        phys_logits = phys_logits + (self.beta * attn_bias)

        # 合并
        logits = torch.cat([phys_logits, free_logits], dim=1)

        attn = torch.softmax(logits, dim=-1)
        attn = self.drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.drop(self.proj(out))


class HybridGridTransformer(nn.Module):
    def __init__(self, in_dim=6, d_model=128, n_heads=8, n_layers=6, d_ff=256, dropout=0.1):
        super().__init__()

        self.embedding = nn.Linear(in_dim, d_model)

        # GCN 分支 (Local)
        self.gcn1 = NativeGCNLayer(d_model, d_model, dropout)
        self.gcn2 = NativeGCNLayer(d_model, d_model, dropout)
        self.gcn_norm = nn.LayerNorm(d_model)

        # Transformer 分支 (Global)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn': HybridMultiHeadAttention(d_model, n_heads, dropout),
                'norm2': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model), nn.Dropout(dropout)
                )
            }) for _ in range(n_layers)
        ])

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, adj, attn_bias):
        # x: (B, N, 6)
        h = self.embedding(x)

        # GCN 提取局部特征并残差连接
        h_local = self.gcn1(h, adj)
        h_local = self.gcn2(h_local, adj)
        h = self.gcn_norm(h + h_local)

        # Transformer 提取全局特征
        for layer in self.layers:
            h_norm = layer['norm1'](h)
            h = h + layer['attn'](h_norm, attn_bias)

            h_norm = layer['norm2'](h)
            h = h + layer['ff'](h_norm)

        out = self.head(h).squeeze(-1)  # (B, N)
        return torch.sigmoid(out)


# ========== 4. Main Training Loop ==========

def main():
    # 1. 准备数据
    all_data, x_mean, x_std, adj_matrix, attn_bias = load_data_and_assets()

    split = int(len(all_data) * 0.8)
    train_dl = DataLoader(GridDataset(all_data[:split]), batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_dl = DataLoader(GridDataset(all_data[split:]), batch_size=BATCH_SIZE,
                         shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"🚀 Initializing Hybrid Grid Transformer on {DEVICE}...")
    model = HybridGridTransformer(
        in_dim=IN_DIM, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    # 🔥 使用 L1 Loss 对齐 MAE 指标
    criterion = nn.L1Loss()

    best_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Ep {epoch}/{EPOCHS}")

        for x, y, mask in pbar:
            x, y, mask = x.to(DEVICE), y.to(DEVICE).squeeze(1), mask.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(DEVICE.type == "cuda")):
                x_feat = process_batch_features(x, x_mean, x_std)
                pred = model(x_feat, adj_matrix, attn_bias)

                if mask.sum() > 0:
                    loss = criterion(pred[mask], y[mask])
                else:
                    loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.5f}"})

        avg_train_loss = train_loss / len(train_dl)

        # --- Validation (Fixed Logic) ---
        model.eval()
        total_abs_error = 0.0
        total_valid_nodes = 0

        with torch.no_grad():
            for x, y, mask in test_dl:
                x, y, mask = x.to(DEVICE), y.to(DEVICE).squeeze(1), mask.to(DEVICE)
                x_feat = process_batch_features(x, x_mean, x_std)
                pred = model(x_feat, adj_matrix, attn_bias)

                if mask.sum() > 0:
                    # 🔥 关键修正：累加总误差，最后再除
                    total_abs_error += torch.abs(pred[mask] - y[mask]).sum().item()
                    total_valid_nodes += mask.sum().item()

        avg_val_mae = total_abs_error / (total_valid_nodes + 1e-6)
        curr_lr = optimizer.param_groups[0]['lr']

        print(f"📊 Epoch {epoch} | Loss: {avg_train_loss:.5f} | Val MAE: {avg_val_mae:.5f} | LR: {curr_lr:.2e}")

        scheduler.step(avg_val_mae)

        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            torch.save(model.state_dict(), SAVE_PATH)
            print("💾 Best Model Saved!")

    # --- Plotting ---
    print("\n🎨 Generating Results Plot...")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    trues, preds = [], []
    with torch.no_grad():
        for x, y, mask in tqdm(test_dl, desc="Predicting"):
            x, y, mask = x.to(DEVICE), y.to(DEVICE).squeeze(1), mask.to(DEVICE)
            x_feat = process_batch_features(x, x_mean, x_std)
            pred = model(x_feat, adj_matrix, attn_bias)

            if mask.sum() > 0:
                trues.extend(y[mask].cpu().numpy())
                preds.extend(pred[mask].cpu().numpy())

    if trues:
        y_t, y_p = np.array(trues), np.array(preds)
        mae = mean_absolute_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)

        plt.figure(figsize=(9, 8))
        hb = plt.hexbin(y_t, y_p, gridsize=60, mincnt=1, cmap='inferno', bins='log')
        plt.colorbar(hb, label='Log Count')
        plt.plot([0, 1], [0, 1], "w--", linewidth=1.5)
        plt.xlabel("Ground Truth (Alpha)")
        plt.ylabel("Prediction")
        plt.title(f"Hybrid Transformer Results\nMAE: {mae:.4f} | R2: {r2:.4f}")

        plt.tight_layout()
        plt.savefig(SCATTER_SAVE_PATH, dpi=300)
        print(f"✅ Saved plot to {SCATTER_SAVE_PATH}")


if __name__ == "__main__":
    main()