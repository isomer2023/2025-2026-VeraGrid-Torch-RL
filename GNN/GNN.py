import os
import glob
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 导入你的模型
try:
    from gnn_model import GridGNN
except ImportError:
    print("❌ NOT FOUND gnn_model.py")
    exit()

# ================= 配置 =================
DATA_DIR = "./dataset_output_1mv_urban_4000"
LR = 3e-4 # adjust automatically
EPOCHS = 40
BATCH_SIZE = 32
HIDDEN_DIM = 128
HEADS = 4
NORMALIZE = 'zscore' # minmax or zscore
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "best_gnn_model_auto.pth"
SEED = 4000
# =======================================

random.seed(SEED)
torch.manual_seed(SEED)

# --- 🔥 自动加载统计量 ---
stats_path = os.path.join(DATA_DIR, "stats.pt")
if not os.path.exists(stats_path):
    print(f"❌ CANNOT FIND STATS FILE AT:  {stats_path}, PLEASE RUN DATA.py FIRST!")
    exit()

print(f"📥 LOAD STATS SUCCESSFULLY: {stats_path}")
stats = torch.load(stats_path, map_location=DEVICE)
X_MEAN = stats['x_mean'].to(DEVICE)
X_STD = stats['x_std'].to(DEVICE)
E_MEAN = stats['e_mean'].to(DEVICE)
E_STD = stats['e_std'].to(DEVICE)

X_MIN = stats['x_min'].to(DEVICE)
X_MAX = stats['x_max'].to(DEVICE)
E_MIN = stats['e_min'].to(DEVICE)
E_MAX = stats['e_max'].to(DEVICE)

# 防止除零（对 Min-Max 也需要）
range_x = X_MAX - X_MIN
range_e = E_MAX - E_MIN
range_x[range_x < 1e-6] = 1.0
range_e[range_e < 1e-6] = 1.0

# 防止除以 0 (如果某特征全是0，std就是0)
X_STD[X_STD < 1e-6] = 1.0
E_STD[E_STD < 1e-6] = 1.0


# ------------------------

def normalize_batch(batch, method):
    # 注意：这里会原地修改 batch.x，所以 DataLoader 如果 num_workers > 0 可能会有副作用
    # 但在简单的训练循环中通常没问题。如果想更安全，可以 clone 一下。
    """归一化批次数据

        Args:
            batch: graph data patches
            method: 'minmax' or 'zscore'
        """
    if method == 'minmax':
        # Min-Max normalize to [0, 1]
        batch.x = (batch.x - X_MIN) / (X_MAX - X_MIN)
        batch.edge_attr = (batch.edge_attr - E_MIN) / (E_MAX - E_MIN)

    elif method == 'zscore':
        # Z-score normalize
        batch.x = (batch.x - X_MEAN) / X_STD
        batch.edge_attr = (batch.edge_attr - E_MEAN) / E_STD

    else:
        raise ValueError(f"Unknown normalize method: {method}")

    return batch



def load_all_chunks(data_dir):
    pattern = os.path.join(data_dir, "chunk_*.pt")
    files = sorted(glob.glob(pattern))
    all_data = []
    for f in files:
        try:
            chunk = torch.load(f, weights_only=False)
            all_data.extend(chunk)
        except:
            pass
    print(f"✅ LOADED {len(all_data)} EXAMPLES")
    return all_data


def evaluate_model(model, loader, device, return_arrays=False):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch = normalize_batch(batch, method=NORMALIZE)

            pred = model(batch)
            if pred.dim() == 1: pred = pred.unsqueeze(-1)
            target = batch.y

            # 🔥 这里的 mask 已经在生成时处理好了
            # Bus 86 的 mask 必然是 False，所以这里绝对不会选到它
            if batch.mask.sum() == 0: continue

            all_pred.append(pred[batch.mask].view(-1).cpu().numpy())
            all_true.append(target[batch.mask].view(-1).cpu().numpy())

    if not all_true: return (None,) * 5 if return_arrays else (None,) * 3
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return (mae, rmse, r2, y_true, y_pred) if return_arrays else (mae, rmse, r2)

def main():
    print("🚀 START AUTOMATIC TRAINING")
    all_data = load_all_chunks(DATA_DIR)

    if len(all_data) == 0:
        print("❌ EMPTY DATASET, PLEASE CHECK PATH")
        return

    # 划分
    random.shuffle(all_data)
    cut = int(len(all_data) * 0.8)
    train_loader = DataLoader(all_data[:cut], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(all_data[cut:], batch_size=BATCH_SIZE, shuffle=False)

    # 获取一个样本批次用于调试
    # sample_batch = next(iter(train_loader))
    # sample_batch = sample_batch.to(DEVICE)
    # print("\n=== 原始数据统计 ===")
    # debug_batch_info(sample_batch, stage="RAW", method="none")

    # 测试 minmax 归一化
    # batch_minmax = sample_batch.clone()
    # batch_minmax = normalize_batch(batch_minmax, method=NORMALIZE)
    print(f"\n...USING {NORMALIZE} NORMALIZATION...")
    # debug_batch_info(batch_minmax, stage="AFTER", method=NORMALIZE)

    # 打印统计量信息
    print("\n=== STATS INFO ===")
    print(f"X_MEAN: {X_MEAN.cpu().numpy()}")
    print(f"X_STD:  {X_STD.cpu().numpy()}")
    print(f"X_MIN:  {X_MIN.cpu().numpy()}")
    print(f"X_MAX:  {X_MAX.cpu().numpy()}")

    model = GridGNN(6, 4, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = 3e-4, weight_decay = 5e-4)
    # optimizer = optim.Adam(model.parameters(), lr=LR)

    # 添加调度器定义（放在优化器定义后）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20
    )

    best_mae = float('inf')

    print(f"\n{'Epoch':<6} | {'Loss':<10} | {'ValMAE':<10}")
    print("-" * 35)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            batch = normalize_batch(batch, method=NORMALIZE)
            pred = model(batch)
            if pred.dim() == 1: pred = pred.unsqueeze(-1)

            # 🔥 Loss 计算自动忽略 Bus 86 (因为它 mask 是 False)
            if batch.mask.sum() == 0: continue

            loss = F.smooth_l1_loss(pred[batch.mask], batch.y[batch.mask])
            loss.backward()

            # 可选：梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses) if losses else 0
        mae, rmse, r2 = evaluate_model(model, test_loader, DEVICE)

        if mae is not None and mae < best_mae:
            scheduler.step(mae)
            best_mae = mae
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            # 如果mae为None，使用损失作为替代
            scheduler.step(avg_loss)

        val_str = f"{mae:.6f}" if mae is not None else "nan"
        print(f"{epoch:<6} | {avg_loss:<10.6f} | {val_str:<10}")

    print("\n🎉 TRAINING FINISH")
    print(f"💾 SAVE BEST MODEL AT {SAVE_PATH}")

    # ================= 📊 最终评估 & 画图 =================
    # 重新加载最佳模型
    try:
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        model.eval()
        print("📥 LOAD BEST MODEL TO EVALUATE...")
    except Exception as e:
        print(f"⚠️ CANNOT LOAD MODEL: {e}")
        return

    mae, rmse, r2, y_true, y_pred = evaluate_model(
        model, test_loader, DEVICE, return_arrays=True
    )

    if mae is not None:
        print("\n EVALUATION:")
        print(f"   MAE  : {mae:.6f}")
        print(f"   RMSE : {rmse:.6f}")
        print(f"   R2   : {r2:.6f}")

        # --- 画散点图 ---
        plt.figure(figsize=(7, 6))

        # Hexbin 图：适合点很多的情况，颜色越亮代表点越密集
        hb = plt.hexbin(
            y_true,
            y_pred,
            gridsize=60,
            mincnt=1,
            bins='log',
            cmap='inferno'
        )
        cb = plt.colorbar(hb)
        cb.set_label("log10(count)")

        # 画对角线（完美预测线）
        min_val = 0.0
        max_val = 1.0
        plt.plot([min_val, max_val], [min_val, max_val], "w--", linewidth=1.5, label="Perfect Prediction")

        plt.xlabel("Ideal Action (OPF Target)")
        plt.ylabel("Predicted Action (GNN Output)")
        plt.title(f"Test Set Prediction (R2={r2:.4f})")
        plt.legend()
        plt.tight_layout()

        save_img_path = f"result_scatter_{NORMALIZE}_{SEED}.png"
        plt.savefig(save_img_path, dpi=300)
        print(f"📈 SCATTER PLOT SAVED: {save_img_path}")
    else:
        print("⚠️ NO VALID SAMPLE AT TEST DATASET, CANNOT PLOT")


if __name__ == "__main__":
    main()