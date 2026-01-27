import os
import glob
import math
import warnings
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.cuda.amp import autocast, GradScaler

# ================= ⚙️ 配置 (与主模型保持一致) =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "dataset_output_1mv_urban")
STATS_PATH = os.path.join(DATA_DIR, "stats.pt")
ASSETS_PATH = os.path.join(DATA_DIR, "static_assets.pt")

# 结果保存目录
RESULT_DIR = os.path.join(CURRENT_DIR, "baseline_results")
os.makedirs(RESULT_DIR, exist_ok=True)

BATCH_SIZE = 32
NUM_WORKERS = 2
PIN_MEMORY = True

# 特征维度
IN_DIM = 6  # P, Q, PV, V, sin, cos

# RF 配置
RF_N_ESTIMATORS = 50  # 树的数量
RF_N_JOBS = -1  # 并行CPU核心数

# 深度学习模型通用配置
DL_LR = 3e-4
DL_EPOCHS = 30
D_MODEL = 128
D_FF = 256
DROPOUT = 0.1
# ============================================================

warnings.filterwarnings("ignore")


# ========== 1. 数据加载与处理 (复用原逻辑) ==========
class GridDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    x = torch.stack([item["x"] for item in batch])
    y = torch.stack([item["y"] for item in batch])
    mask = torch.stack([item["mask"] for item in batch])
    return x, y, mask


def load_data_and_assets():
    print("📂 Loading data...")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "chunk_*.pt")))
    if not files: raise FileNotFoundError("❌ No data found.")
    all_data = []
    for f in files: all_data.extend(torch.load(f, weights_only=False))

    stats = torch.load(STATS_PATH, weights_only=False, map_location=DEVICE)
    x_mean = stats["x_mean"].view(1, -1, 1)
    x_std = stats["x_std"].view(1, -1, 1)

    assets = torch.load(ASSETS_PATH, weights_only=False, map_location=DEVICE)
    edge_index = assets["edge_index"]
    num_nodes = int(assets["num_nodes"])

    # 构造 GCN 邻接矩阵
    adj = torch.zeros((num_nodes, num_nodes), device=DEVICE)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = adj + torch.eye(num_nodes, device=DEVICE)
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat = torch.diag(d_inv_sqrt)
    norm_adj = d_mat @ adj @ d_mat

    return all_data, x_mean, x_std, norm_adj


def process_batch_features(x_raw, x_mean, x_std):
    # 保持完全一致的特征工程
    x_norm_part = (x_raw[:, :3, :] - x_mean[:, :3, :]) / (x_std[:, :3, :] + 1e-6)
    v_raw = x_raw[:, 3, :]
    v_phys = (v_raw - 1.0) / 0.05
    ang = x_raw[:, 4, :]
    x_feat = torch.cat([
        x_norm_part, v_phys.unsqueeze(1),
        torch.sin(ang).unsqueeze(1), torch.cos(ang).unsqueeze(1)
    ], dim=1)
    return x_feat.transpose(1, 2).contiguous()  # (B, N, 6)


# ========== 2. 模型定义 ==========

# --- Baseline A: MLP ---
class BaselineMLP(nn.Module):
    def __init__(self, in_dim=6, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )

    def forward(self, x, adj=None):
        # adj is ignored for MLP
        out = self.net(x)
        return torch.sigmoid(out).squeeze(-1)


# --- Baseline B: Pure GNN ---
class NativeGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.GELU()

    def forward(self, x, adj):
        out = self.linear(x)
        out = torch.einsum('nm, bmf -> bnf', adj, out)
        return self.act(out)


class BaselineGNN(nn.Module):
    def __init__(self, in_dim=6, d_model=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.gcn1 = NativeGCNLayer(d_model, d_model)
        self.gcn2 = NativeGCNLayer(d_model, d_model)
        self.gcn3 = NativeGCNLayer(d_model, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, adj):
        h = self.proj(x)
        h = h + self.gcn1(h, adj)  # Residuals for better training
        h = h + self.gcn2(h, adj)
        h = h + self.gcn3(h, adj)
        return torch.sigmoid(self.head(h)).squeeze(-1)


# ========== 3. 训练与评估逻辑 ==========

def evaluate_model(model, dl, x_mean, x_std, adj, model_type="torch"):
    """ 通用评估函数 """
    trues, preds = [], []

    if model_type == "torch":
        model.eval()
        with torch.no_grad():
            for x, y, mask in dl:
                x, y, mask = x.to(DEVICE), y.to(DEVICE).squeeze(1), mask.to(DEVICE)
                x_feat = process_batch_features(x, x_mean, x_std)

                # MLP 忽略 adj, GNN 使用 adj
                pred = model(x_feat, adj)

                if mask.sum() > 0:
                    trues.extend(y[mask].cpu().numpy())
                    preds.extend(pred[mask].cpu().numpy())

    elif model_type == "sklearn":
        # 对于 RF，我们需要先把数据搬回 CPU 并展平
        for x, y, mask in dl:
            # 放到 GPU 做特征工程（为了代码复用），再搬回 CPU
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            y = y.squeeze(1)  # CPU y

            with torch.no_grad():
                x_feat = process_batch_features(x, x_mean, x_std)  # (B, N, 6)

            # 转 numpy
            x_np = x_feat.cpu().numpy().reshape(-1, IN_DIM)
            y_np = y.numpy().reshape(-1)
            mask_np = mask.cpu().numpy().reshape(-1)

            # 过滤 Mask
            if mask_np.sum() > 0:
                valid_x = x_np[mask_np]
                valid_y = y_np[mask_np]

                # 预测
                p = model.predict(valid_x)

                trues.extend(valid_y)
                preds.extend(p)

    return np.array(trues), np.array(preds)


def save_results(name, trues, preds):
    """ 保存散点图和指标文本 """
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    mse = mean_squared_error(trues, preds)

    print(f"   📊 {name} Results: MAE={mae:.5f}, R2={r2:.5f}")

    # 1. 保存指标文本
    txt_path = os.path.join(RESULT_DIR, f"{name}_metrics.txt")
    with open(txt_path, "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"R2 Score: {r2:.6f}\n")

    # 2. 画图
    plt.figure(figsize=(8, 7))
    plt.hexbin(trues, preds, gridsize=60, mincnt=1, cmap='inferno', bins='log')
    plt.colorbar(label='Log Count')
    plt.plot([0, 1], [0, 1], "w--", linewidth=1.5)
    plt.xlabel("Ground Truth (Alpha)")
    plt.ylabel("Prediction")
    plt.title(f"{name}\nMAE={mae:.4f} | R2={r2:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{name}_scatter.png"), dpi=300)
    plt.close()


# ========== 4. 训练流程 ==========

def train_sklearn_rf(train_dl, test_dl, x_mean, x_std):
    print("\n🌲 Training Random Forest (sklearn)...")
    print("   (Data Prep: Flattening batches and applying mask...)")

    # 准备训练数据
    X_train_list, y_train_list = [], []
    for x, y, mask in tqdm(train_dl, desc="Prep RF Data"):
        x, mask = x.to(DEVICE), mask.to(DEVICE)
        y = y.squeeze(1)  # CPU

        with torch.no_grad():
            x_feat = process_batch_features(x, x_mean, x_std)

        x_np = x_feat.cpu().numpy().reshape(-1, IN_DIM)
        y_np = y.numpy().reshape(-1)
        mask_np = mask.cpu().numpy().reshape(-1)

        if mask_np.sum() > 0:
            X_train_list.append(x_np[mask_np])
            y_train_list.append(y_np[mask_np])

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    print(f"   RF Training Data Shape: {X_train.shape}")
    rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, n_jobs=RF_N_JOBS, verbose=1)
    rf.fit(X_train, y_train)

    # 评估
    print("   Evaluting RF...")
    t, p = evaluate_model(rf, test_dl, x_mean, x_std, None, model_type="sklearn")
    save_results("RandomForest", t, p)


def train_torch_model(name, model, train_dl, test_dl, x_mean, x_std, adj):
    print(f"\n🔥 Training {name} (PyTorch)...")
    optimizer = optim.AdamW(model.parameters(), lr=DL_LR)
    criterion = nn.L1Loss()  # 与主模型保持一致

    for epoch in range(1, DL_EPOCHS + 1):
        model.train()
        loss_sum = 0
        for x, y, mask in train_dl:
            x, y, mask = x.to(DEVICE), y.to(DEVICE).squeeze(1), mask.to(DEVICE)

            optimizer.zero_grad()
            with autocast(enabled=(DEVICE.type == "cuda")):
                x_feat = process_batch_features(x, x_mean, x_std)
                pred = model(x_feat, adj)

                if mask.sum() > 0:
                    loss = criterion(pred[mask], y[mask])
                    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))  # 简单实例化
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    loss_sum += loss.item()

        if epoch % 5 == 0:
            print(f"   Ep {epoch}: Loss = {loss_sum / len(train_dl):.4f}")

    # 评估
    print(f"   Evaluating {name}...")
    t, p = evaluate_model(model, test_dl, x_mean, x_std, adj, model_type="torch")
    save_results(name, t, p)


def main():
    # 1. 准备数据
    all_data, x_mean, x_std, adj = load_data_and_assets()
    split = int(len(all_data) * 0.8)

    train_ds = GridDataset(all_data[:split])
    test_ds = GridDataset(all_data[split:])

    # DataLoader (对于 RF 只需要 batch 获取数据即可)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    # 2. 运行 Random Forest
    train_sklearn_rf(train_dl, test_dl, x_mean, x_std)

    # 3. 运行 MLP
    mlp = BaselineMLP(in_dim=IN_DIM, d_model=D_MODEL).to(DEVICE)
    train_torch_model("MLP_Baseline", mlp, train_dl, test_dl, x_mean, x_std, adj=None)

    # 4. 运行 Pure GNN
    gnn = BaselineGNN(in_dim=IN_DIM, d_model=D_MODEL).to(DEVICE)
    train_torch_model("GNN_Baseline", gnn, train_dl, test_dl, x_mean, x_std, adj=adj)

    print(f"\n✅ 所有基线运行完毕！结果保存在: {RESULT_DIR}")


if __name__ == "__main__":
    main()