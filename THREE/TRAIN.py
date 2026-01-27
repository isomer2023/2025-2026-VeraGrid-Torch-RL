import os
import glob
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

from torch.cuda.amp import GradScaler

# 引入你的模型（已改成 MPNN 输出 4 通道）
from model_architecture import HybridGridTransformer


# ================= ⚙️ 配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "dataset_output_1mv_urban_dynamic_topo_FULL_STATE"
STATS_PATH = os.path.join(DATA_DIR, "stats.pt")

# 统一输出目录
RUN_DIR = "runs_ablation_full"
os.makedirs(RUN_DIR, exist_ok=True)

# 🚀 训练配置
BATCH_SIZE = 128
LR = 5e-4
EPOCHS = 60
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0

# 模型参数（按你原来的节奏）
IN_DIM = 4          # ✅ 现在是 4: [P,Q,Pav,vn_kv]
EDGE_DIM = 2        # ✅ edge_attr=[r_pu,x_pu]
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 128
DROPOUT = 0.0

# Loss 权重
W_P = 1.0
W_VM = 100.0
W_VA = 1.0

# vn_kv 通道位置（默认最后一维）
VN_IDX = 3

# 固定随机性（让消融可比）
SEED = 42


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================= 数据逻辑 =================
class GridDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """
    sample fields expected:
      x: (C,N)    C=4
      y: (4,N)    [alpha, Vm, sinVa, cosVa]
      mask: (N,)
      adj: (N,N)
      attn_bias: (N,N)
      edge_index: (2,E)  (perm space)
      edge_attr: (E,2)
    """
    x = torch.stack([item["x"] for item in batch])                # (B,C,N)
    y = torch.stack([item["y"] for item in batch])                # (B,4,N)
    mask = torch.stack([item["mask"] for item in batch])          # (B,N)
    adj = torch.stack([item["adj"] for item in batch])            # (B,N,N)
    attn_bias = torch.stack([item["attn_bias"] for item in batch]).unsqueeze(1)  # (B,1,N,N)

    edge_index = torch.stack([item["edge_index"] for item in batch])  # (B,2,E)
    edge_attr = torch.stack([item["edge_attr"] for item in batch])    # (B,E,2)
    return x, y, mask, adj, attn_bias, edge_index, edge_attr


def load_data_system():
    print("📂 Loading chunks...")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "chunk_*.pt")))
    if not files:
        raise FileNotFoundError(f"No chunks found in {DATA_DIR}")

    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError("Stats file missing.")

    stats = torch.load(STATS_PATH, weights_only=False, map_location="cpu")
    x_mean = stats["x_mean"].view(1, -1, 1)  # (1,C,1)
    x_std = stats["x_std"].view(1, -1, 1)    # (1,C,1)

    print("📥 Reading into RAM...")
    all_data = []
    for f in tqdm(files):
        all_data.extend(torch.load(f, weights_only=False, map_location="cpu"))

    return all_data, x_mean, x_std


def process_features(x_raw, x_mean, x_std):
    """
    x_raw: (B,C,N)
    -> normalized and transpose to (B,N,C)
    """
    x_norm = (x_raw - x_mean) / (x_std + 1e-6)
    return x_norm.transpose(1, 2).contiguous()  # (B,N,C)


# ================= Metrics (Angle) =================
def angle_from_sincos(sin_t, cos_t):
    return torch.atan2(sin_t, cos_t)  # rad


def wrapped_angle_error(pred_rad, true_rad):
    # wrap to [-pi, pi]
    diff = pred_rad - true_rad
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def calc_r2_mae(trues, preds):
    if len(trues) < 2:
        return 0.0, 0.0
    return float(r2_score(trues, preds)), float(mean_absolute_error(trues, preds))


# ================= Plotting =================
def save_scatter_plots(out_png, alpha_true, alpha_pred, vm_true, vm_pred, ang_true_deg, ang_pred_deg, metrics_txt):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Alpha
    axs[0].hexbin(alpha_true, alpha_pred, gridsize=50, mincnt=1, bins='log')
    axs[0].plot([0, 1], [0, 1], 'r--')
    axs[0].set_title(f"Alpha\n{metrics_txt['alpha']}")
    axs[0].set_xlabel("True")
    axs[0].set_ylabel("Pred")

    # Vm
    vmin, vmax = float(np.min(vm_true)), float(np.max(vm_true))
    axs[1].hexbin(vm_true, vm_pred, gridsize=50, mincnt=1, bins='log')
    axs[1].plot([vmin, vmax], [vmin, vmax], 'r--')
    axs[1].set_title(f"Vm (p.u.)\n{metrics_txt['vm']}")
    axs[1].set_xlabel("True")
    axs[1].set_ylabel("Pred")

    # Angle
    amin, amax = float(np.min(ang_true_deg)), float(np.max(ang_true_deg))
    axs[2].hexbin(ang_true_deg, ang_pred_deg, gridsize=50, mincnt=1, bins='log')
    axs[2].plot([amin, amax], [amin, amax], 'r--')
    axs[2].set_title(f"Angle (deg)\n{metrics_txt['angle']}")
    axs[2].set_xlabel("True")
    axs[2].set_ylabel("Pred")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


# ================= Experiment Runner =================
def train_one_experiment(exp_name, train_dl, test_dl, x_mean, x_std, cfg):
    """
    cfg switches:
      - use_attn_bias: bool
      - zero_edge_attr: bool
      - zero_vn_after_norm: bool
    """
    exp_dir = os.path.join(RUN_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    ckpt_path = os.path.join(exp_dir, "best_model.pth")
    metrics_path = os.path.join(exp_dir, "metrics.json")
    scatter_path = os.path.join(exp_dir, "scatter_test.png")
    npz_path = os.path.join(exp_dir, "test_predictions.npz")

    # init model
    model = HybridGridTransformer(
        in_dim=IN_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        edge_dim=EDGE_DIM,
        mpnn_steps=2,
        use_mpnn=True,
        use_transformer=True,
        n_phys_heads=None,
        branch_mlp_depth=2,
        normalize_angle=True,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_dl))
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    criterion = nn.L1Loss()
    best_val = float("inf")

    x_mean_dev = x_mean.to(DEVICE)
    x_std_dev = x_std.to(DEVICE)

    # ---------------- Train ----------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"[{exp_name}] Ep {epoch}/{EPOCHS}")

        for x, y, mask, adj, attn_bias, edge_index, edge_attr in pbar:
            # y: (B,4,N) -> (B,N,4)
            y = y.transpose(1, 2).contiguous()

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            mask = mask.to(DEVICE)
            adj = adj.to(DEVICE)
            attn_bias = attn_bias.to(DEVICE)
            edge_index = edge_index.to(DEVICE)
            edge_attr = edge_attr.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            # ---- x normalize ----
            x_feat = process_features(x, x_mean_dev, x_std_dev)  # (B,N,C)

            # vn_kv ablation: normalize 后把该通道置 0（真正“去信息”）
            if cfg.get("zero_vn_after_norm", False):
                x_feat[..., VN_IDX] = 0.0

            # attn_bias ablation
            attn_in = attn_bias if cfg.get("use_attn_bias", True) else None

            # edge_attr ablation
            edge_attr_in = edge_attr
            if cfg.get("zero_edge_attr", False):
                edge_attr_in = torch.zeros_like(edge_attr)

            # AMP
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
                autocast_ctx = torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu')
            else:
                from torch.cuda.amp import autocast
                autocast_ctx = autocast(enabled=(DEVICE.type == "cuda"))

            with autocast_ctx:
                pred = model(x_feat, adj, attn_in, edge_index=edge_index, edge_attr=edge_attr_in)  # (B,N,4)

                # Targets
                y_alpha = y[..., 0]
                y_vm = y[..., 1]
                y_sin = y[..., 2]
                y_cos = y[..., 3]

                # Pred
                p_alpha = pred[..., 0]
                p_vm_resid = pred[..., 1]
                p_sin = pred[..., 2]
                p_cos = pred[..., 3]

                # alpha loss only on mask nodes
                if mask.sum() > 0:
                    l_p = criterion(p_alpha[mask], y_alpha[mask])
                else:
                    l_p = torch.tensor(0.0, device=DEVICE)

                # vm residual loss
                l_vm = criterion(p_vm_resid, (y_vm - 1.0))

                # angle loss in sin/cos space
                l_va = 0.5 * (criterion(p_sin, y_sin) + criterion(p_cos, y_cos))

                loss = W_P * l_p + W_VM * l_vm + W_VA * l_va

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # ---------------- Validation (weighted loss) ----------------
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, y, mask, adj, attn_bias, edge_index, edge_attr in test_dl:
                y = y.transpose(1, 2).contiguous()

                x = x.to(DEVICE)
                y = y.to(DEVICE)
                mask = mask.to(DEVICE)
                adj = adj.to(DEVICE)
                attn_bias = attn_bias.to(DEVICE)
                edge_index = edge_index.to(DEVICE)
                edge_attr = edge_attr.to(DEVICE)

                x_feat = process_features(x, x_mean_dev, x_std_dev)
                if cfg.get("zero_vn_after_norm", False):
                    x_feat[..., VN_IDX] = 0.0

                attn_in = attn_bias if cfg.get("use_attn_bias", True) else None
                edge_attr_in = torch.zeros_like(edge_attr) if cfg.get("zero_edge_attr", False) else edge_attr

                pred = model(x_feat, adj, attn_in, edge_index=edge_index, edge_attr=edge_attr_in)

                y_alpha = y[..., 0]
                y_vm = y[..., 1]
                y_sin = y[..., 2]
                y_cos = y[..., 3]

                p_alpha = pred[..., 0]
                p_vm_resid = pred[..., 1]
                p_sin = pred[..., 2]
                p_cos = pred[..., 3]

                if mask.sum() > 0:
                    l_p = criterion(p_alpha[mask], y_alpha[mask])
                else:
                    l_p = torch.tensor(0.0, device=DEVICE)

                l_vm = criterion(p_vm_resid, (y_vm - 1.0))
                l_va = 0.5 * (criterion(p_sin, y_sin) + criterion(p_cos, y_cos))

                vloss = (W_P * l_p + W_VM * l_vm + W_VA * l_va).item()
                val_losses.append(vloss)

        avg_val = float(np.mean(val_losses)) if val_losses else float("inf")
        print(f"\n[{exp_name}] ✅ Val Loss: {avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), ckpt_path)
            print(f"[{exp_name}] 💾 Saved Best Model -> {ckpt_path}")

    # ---------------- Final Test Eval + Save Scatter + Save NPZ ----------------
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    alpha_true, alpha_pred = [], []
    vm_true, vm_pred = [], []
    sin_true, sin_pred = [], []
    cos_true, cos_pred = [], []
    ang_true_deg, ang_pred_deg = [], []

    with torch.no_grad():
        for x, y, mask, adj, attn_bias, edge_index, edge_attr in test_dl:
            y = y.transpose(1, 2).contiguous()

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            mask = mask.to(DEVICE)
            adj = adj.to(DEVICE)
            attn_bias = attn_bias.to(DEVICE)
            edge_index = edge_index.to(DEVICE)
            edge_attr = edge_attr.to(DEVICE)

            x_feat = process_features(x, x_mean_dev, x_std_dev)
            if cfg.get("zero_vn_after_norm", False):
                x_feat[..., VN_IDX] = 0.0

            attn_in = attn_bias if cfg.get("use_attn_bias", True) else None
            edge_attr_in = torch.zeros_like(edge_attr) if cfg.get("zero_edge_attr", False) else edge_attr

            pred = model(x_feat, adj, attn_in, edge_index=edge_index, edge_attr=edge_attr_in)

            # true
            y_a = y[..., 0]
            y_v = y[..., 1]
            y_s = y[..., 2]
            y_c = y[..., 3]

            # pred
            p_a = pred[..., 0]
            p_v = pred[..., 1] + 1.0
            p_s = pred[..., 2]
            p_c = pred[..., 3]

            # alpha (mask)
            if mask.sum() > 0:
                alpha_true.extend(y_a[mask].detach().cpu().numpy().tolist())
                alpha_pred.extend(p_a[mask].detach().cpu().numpy().tolist())

            # vm (all nodes)
            vm_true.extend(y_v.detach().cpu().numpy().reshape(-1).tolist())
            vm_pred.extend(p_v.detach().cpu().numpy().reshape(-1).tolist())

            # sin/cos (all nodes)
            sin_true.extend(y_s.detach().cpu().numpy().reshape(-1).tolist())
            cos_true.extend(y_c.detach().cpu().numpy().reshape(-1).tolist())
            sin_pred.extend(p_s.detach().cpu().numpy().reshape(-1).tolist())
            cos_pred.extend(p_c.detach().cpu().numpy().reshape(-1).tolist())

            # angle deg for plotting
            y_ang = angle_from_sincos(y_s, y_c)  # rad
            p_ang = angle_from_sincos(p_s, p_c)  # rad

            ang_true_deg.extend((y_ang.detach().cpu().numpy().reshape(-1) * 180.0 / math.pi).tolist())
            ang_pred_deg.extend((p_ang.detach().cpu().numpy().reshape(-1) * 180.0 / math.pi).tolist())

    # ---- Metrics ----
    r2_a, mae_a = calc_r2_mae(alpha_true, alpha_pred)
    r2_v, mae_v = calc_r2_mae(vm_true, vm_pred)

    # angle metrics: MAE on wrapped error (deg), R2 on sincos space
    # MAE(deg)
    y_s_t = torch.tensor(sin_true)
    y_c_t = torch.tensor(cos_true)
    p_s_t = torch.tensor(sin_pred)
    p_c_t = torch.tensor(cos_pred)

    y_ang = torch.atan2(y_s_t, y_c_t)
    p_ang = torch.atan2(p_s_t, p_c_t)
    ang_err = wrapped_angle_error(p_ang, y_ang)
    mae_ang_deg = float(torch.mean(torch.abs(ang_err)).item() * 180.0 / math.pi)

    # R2 on concatenated sin/cos
    sc_true = np.concatenate([np.array(sin_true), np.array(cos_true)], axis=0)
    sc_pred = np.concatenate([np.array(sin_pred), np.array(cos_pred)], axis=0)
    r2_sc, mae_sc = calc_r2_mae(sc_true.tolist(), sc_pred.tolist())

    metrics = {
        "exp_name": exp_name,
        "best_val_loss": best_val,
        "alpha": {"r2": r2_a, "mae": mae_a},
        "vm": {"r2": r2_v, "mae": mae_v},
        "angle": {"mae_deg_wrapped": mae_ang_deg, "r2_sincos": r2_sc, "mae_sincos": mae_sc},
        "cfg": cfg,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # save detailed arrays
    np.savez(
        npz_path,
        alpha_true=np.array(alpha_true),
        alpha_pred=np.array(alpha_pred),
        vm_true=np.array(vm_true),
        vm_pred=np.array(vm_pred),
        ang_true_deg=np.array(ang_true_deg),
        ang_pred_deg=np.array(ang_pred_deg),
        sin_true=np.array(sin_true),
        sin_pred=np.array(sin_pred),
        cos_true=np.array(cos_true),
        cos_pred=np.array(cos_pred),
    )

    # scatter plot
    metrics_txt = {
        "alpha": f"R2: {r2_a:.4f} | MAE: {mae_a:.4f}",
        "vm": f"R2: {r2_v:.4f} | MAE: {mae_v:.5f}",
        "angle": f"MAE(deg): {mae_ang_deg:.4f} | R2(sincos): {r2_sc:.4f}",
    }
    save_scatter_plots(scatter_path,
                       np.array(alpha_true), np.array(alpha_pred),
                       np.array(vm_true), np.array(vm_pred),
                       np.array(ang_true_deg), np.array(ang_pred_deg),
                       metrics_txt)

    print(f"\n[{exp_name}] ✅ Test Metrics Saved: {metrics_path}")
    print(f"[{exp_name}] ✅ Scatter Saved: {scatter_path}")
    print(f"[{exp_name}] ✅ NPZ Saved: {npz_path}")

    return metrics


# ================= Ridge Baseline =================
def run_ridge_baseline(exp_name, train_data, test_data, x_mean, x_std):
    exp_dir = os.path.join(RUN_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    metrics_path = os.path.join(exp_dir, "metrics.json")
    scatter_path = os.path.join(exp_dir, "scatter_test.png")
    npz_path = os.path.join(exp_dir, "test_predictions.npz")

    # Build node-level dataset (no topology)
    def extract_xy(data_list):
        X_all = []
        y_alpha_all = []
        y_vm_all = []
        y_sin_all = []
        y_cos_all = []
        mask_all = []

        for item in data_list:
            x = item["x"].numpy()   # (C,N)
            y = item["y"].numpy()   # (4,N)
            m = item["mask"].numpy().astype(bool)  # (N,)

            # normalize x with stats, and transpose to (N,C)
            x_norm = (x - x_mean.numpy().reshape(-1, 1)) / (x_std.numpy().reshape(-1, 1) + 1e-6)
            x_norm = x_norm.T  # (N,C)

            X_all.append(x_norm)
            y_alpha_all.append(y[0].reshape(-1, 1))
            y_vm_all.append(y[1].reshape(-1, 1))
            y_sin_all.append(y[2].reshape(-1, 1))
            y_cos_all.append(y[3].reshape(-1, 1))
            mask_all.append(m.reshape(-1, 1))

        X_all = np.vstack(X_all)
        y_alpha_all = np.vstack(y_alpha_all).reshape(-1)
        y_vm_all = np.vstack(y_vm_all).reshape(-1)
        y_sin_all = np.vstack(y_sin_all).reshape(-1)
        y_cos_all = np.vstack(y_cos_all).reshape(-1)
        mask_all = np.vstack(mask_all).reshape(-1).astype(bool)

        return X_all, y_alpha_all, y_vm_all, y_sin_all, y_cos_all, mask_all

    X_tr, ya_tr, yv_tr, ys_tr, yc_tr, m_tr = extract_xy(train_data)
    X_te, ya_te, yv_te, ys_te, yc_te, m_te = extract_xy(test_data)

    # Alpha: train only on mask nodes
    ridge_alpha = Ridge(alpha=1.0, random_state=SEED)
    ridge_alpha.fit(X_tr[m_tr], ya_tr[m_tr])
    ya_pred = ridge_alpha.predict(X_te[m_te])

    # Physics: train on all nodes (Vm, sin, cos)
    ridge_vm = Ridge(alpha=1.0, random_state=SEED)
    ridge_sin = Ridge(alpha=1.0, random_state=SEED)
    ridge_cos = Ridge(alpha=1.0, random_state=SEED)

    ridge_vm.fit(X_tr, yv_tr)
    ridge_sin.fit(X_tr, ys_tr)
    ridge_cos.fit(X_tr, yc_tr)

    yv_pred = ridge_vm.predict(X_te)
    ys_pred = ridge_sin.predict(X_te)
    yc_pred = ridge_cos.predict(X_te)

    # Metrics
    r2_a, mae_a = calc_r2_mae(ya_te[m_te].tolist(), ya_pred.tolist())
    r2_v, mae_v = calc_r2_mae(yv_te.tolist(), yv_pred.tolist())

    # angle mae (deg) and sincos r2
    y_ang = np.arctan2(ys_te, yc_te)
    p_ang = np.arctan2(ys_pred, yc_pred)
    diff = np.arctan2(np.sin(p_ang - y_ang), np.cos(p_ang - y_ang))
    mae_ang_deg = float(np.mean(np.abs(diff)) * 180.0 / math.pi)

    sc_true = np.concatenate([ys_te, yc_te], axis=0)
    sc_pred = np.concatenate([ys_pred, yc_pred], axis=0)
    r2_sc, mae_sc = calc_r2_mae(sc_true.tolist(), sc_pred.tolist())

    metrics = {
        "exp_name": exp_name,
        "alpha": {"r2": r2_a, "mae": mae_a},
        "vm": {"r2": r2_v, "mae": mae_v},
        "angle": {"mae_deg_wrapped": mae_ang_deg, "r2_sincos": r2_sc, "mae_sincos": mae_sc},
        "cfg": {"baseline": "Ridge(alpha=1.0)"},
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # For scatter, need full arrays:
    # Alpha scatter uses mask nodes only
    alpha_true = ya_te[m_te]
    alpha_pred = ya_pred

    # Vm scatter uses all nodes
    vm_true = yv_te
    vm_pred = yv_pred

    # Angle scatter uses all nodes (deg)
    ang_true_deg = y_ang * 180.0 / math.pi
    ang_pred_deg = p_ang * 180.0 / math.pi

    np.savez(
        npz_path,
        alpha_true=np.array(alpha_true),
        alpha_pred=np.array(alpha_pred),
        vm_true=np.array(vm_true),
        vm_pred=np.array(vm_pred),
        ang_true_deg=np.array(ang_true_deg),
        ang_pred_deg=np.array(ang_pred_deg),
        sin_true=np.array(ys_te),
        sin_pred=np.array(ys_pred),
        cos_true=np.array(yc_te),
        cos_pred=np.array(yc_pred),
    )

    metrics_txt = {
        "alpha": f"R2: {r2_a:.4f} | MAE: {mae_a:.4f}",
        "vm": f"R2: {r2_v:.4f} | MAE: {mae_v:.5f}",
        "angle": f"MAE(deg): {mae_ang_deg:.4f} | R2(sincos): {r2_sc:.4f}",
    }

    save_scatter_plots(scatter_path,
                       np.array(alpha_true), np.array(alpha_pred),
                       np.array(vm_true), np.array(vm_pred),
                       np.array(ang_true_deg), np.array(ang_pred_deg),
                       metrics_txt)

    print(f"\n[{exp_name}] ✅ Ridge Metrics Saved: {metrics_path}")
    print(f"[{exp_name}] ✅ Ridge Scatter Saved: {scatter_path}")
    print(f"[{exp_name}] ✅ Ridge NPZ Saved: {npz_path}")

    return metrics


# ================= Main =================
def main():
    set_seed(SEED)

    # 1) 数据加载
    all_data, x_mean, x_std = load_data_system()

    # 固定划分（消融一致）
    split = int(len(all_data) * 0.90)
    train_data = all_data[:split]
    test_data = all_data[split:]

    train_dl = DataLoader(
        GridDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS
    )
    test_dl = DataLoader(
        GridDataset(test_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS
    )

    # 2) 定义实验列表（主模型 + 三组消融）
    experiments = [
        ("MPNN_FULL", {"use_attn_bias": True,  "zero_edge_attr": False, "zero_vn_after_norm": False}),
        ("ABL_NO_BIAS", {"use_attn_bias": False, "zero_edge_attr": False, "zero_vn_after_norm": False}),
        ("ABL_ZERO_EDGE", {"use_attn_bias": True,  "zero_edge_attr": True,  "zero_vn_after_norm": False}),
        ("ABL_ZERO_VN", {"use_attn_bias": True,  "zero_edge_attr": False, "zero_vn_after_norm": True}),
    ]

    all_metrics = []

    # 3) 逐个训练 + 测试 + 保存图/npz
    for name, cfg in experiments:
        metrics = train_one_experiment(name, train_dl, test_dl, x_mean, x_std, cfg)
        all_metrics.append(metrics)

    # 4) Ridge baseline
    ridge_metrics = run_ridge_baseline("BASELINE_RIDGE", train_data, test_data, x_mean, x_std)
    all_metrics.append(ridge_metrics)

    # 5) 汇总保存
    summary_path = os.path.join(RUN_DIR, "summary_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
