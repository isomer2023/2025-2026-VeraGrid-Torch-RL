import os
import glob
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from cnn_model import GridCNN
except ImportError:
    print("❌ 缺少 cnn_model.py")
    exit()

# ================= ⚙️ 配置 =================
SCRIPT_PATH = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(SCRIPT_PATH)
DATA_DIR = os.path.join(CURRENT_DIR, "dataset_output_1MVLV-urban")
SAVE_PATH = os.path.join(CURRENT_DIR, "best_cnn_model_1MVLV-urban.pth")
#get datetime and transfer to yymmdd-hhmmss
datetime = dt.datetime.now().strftime("%y%m%d-%H%M%S")
IMG_SAVE_PATH = os.path.join(CURRENT_DIR, f"result_scatter_{datetime}.png")

LR = 1e-3
EPOCHS = 200  # 多跑一点，给 Plateau 调度器机会
BATCH_SIZE = 64
HIDDEN_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
PE_DIM_KEEP = 4  # 🔥【Idea 1】只保留前 4 维 PE
VOLT_SCALE = 5.0  # 🔥【Idea 3】电压通道放大倍数
# ==========================================

random.seed(SEED);
torch.manual_seed(SEED)

print(f"📂 工作目录: {CURRENT_DIR}")

# 1. 加载资产
assets_path = os.path.join(DATA_DIR, "static_assets.pt")
if not os.path.exists(assets_path):
    print("❌ 找不到静态资产")
    exit()
assets = torch.load(assets_path, map_location=DEVICE, weights_only=False)
full_pe = assets['pe'].t().to(DEVICE)  # 原始 [8, N]

# 🔥【Idea 1 实现】PE 降维：只取前 4 行
STATIC_PE = full_pe[:PE_DIM_KEEP, :]
print(f"✂️  PE 已削减: 原始 8 维 -> 保留前 {PE_DIM_KEEP} 维")

stats_path = os.path.join(DATA_DIR, "stats.pt")
if not os.path.exists(stats_path):
    print("❌ 找不到统计量")
    exit()
stats = torch.load(stats_path, map_location=DEVICE, weights_only=False)
X_MEAN = stats['x_mean'].to(DEVICE)
X_STD = stats['x_std'].to(DEVICE)
X_STD[X_STD < 1e-6] = 1.0


# 2. 数据集
class GridDataset(Dataset):
    def __init__(self, data_list): self.data = data_list

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return self.data[idx]


def collate_fn(batch):
    x = torch.stack([item['x'] for item in batch])
    y = torch.stack([item['y'] for item in batch])
    mask = torch.stack([item['mask'] for item in batch])
    return x, y, mask


def load_all_chunks(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.pt")))
    all_data = []
    print("📂 读取数据中...")
    for f in files:
        try:
            all_data.extend(torch.load(f, weights_only=False))
        except:
            pass
    print(f"✅ 加载 {len(all_data)} 样本")
    return all_data


# 3. 评估
def evaluate_model(model, loader, device, return_arrays=False):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            # 归一化
            x_norm = (x - X_MEAN.view(1, -1, 1)) / X_STD.view(1, -1, 1)

            # 🔥【Idea 3 实现】强调电压通道
            # 假设通道 3 是 V_mag。我们手动放大它的数值。
            # 此时 x_norm 已经是 Mean=0, Std=1 的分布。
            # 放大后，电压异常点的数值会变得很大（比如从 2.0 变成 10.0），强迫 Loss 关注它。
            x_norm[:, 3, :] = x_norm[:, 3, :] * VOLT_SCALE

            # 拼接 PE (只有 4 维)
            pe_batch = STATIC_PE.unsqueeze(0).expand(x.shape[0], -1, -1)
            x_input = torch.cat([x_norm, pe_batch], dim=1)

            pred = model(x_input)

            if mask.sum() == 0: continue
            valid_pred = pred[mask.unsqueeze(1)]
            valid_true = y[mask.unsqueeze(1)]
            all_pred.append(valid_pred.cpu().numpy())
            all_true.append(valid_true.cpu().numpy())

    if not all_true: return (None,) * 5 if return_arrays else (None,) * 3
    y_t, y_p = np.concatenate(all_true), np.concatenate(all_pred)
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)
    return (mae, rmse, r2, y_t, y_p) if return_arrays else (mae, rmse, r2)


def main():
    all_data = load_all_chunks(DATA_DIR)
    if not all_data: return
    random.shuffle(all_data)
    cut = int(len(all_data) * 0.8)
    train_dl = DataLoader(GridDataset(all_data[:cut]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_dl = DataLoader(GridDataset(all_data[cut:]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 计算输入通道数
    sample_dim = all_data[0]['x'].shape[0]  # 5
    # 输入 = 物理(5) + PE(4)
    # (Global 的 +3 是在 model 内部处理的，这里不需要算)
    in_channels = sample_dim + PE_DIM_KEEP

    print(f"🚀 输入维度: {sample_dim} (Phys) + {PE_DIM_KEEP} (PE) = {in_channels}")
    print(f"⚡ 电压通道放大倍数: {VOLT_SCALE}x")

    model = GridCNN(in_channels=in_channels, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 换回稳健的调度器，应对高噪声数据
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    print(f"ℹ️  配置: Global Feats + Voltage Boost + Reduced PE")
    best_mae = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, y, mask in train_dl:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()

            # 1. 归一化
            x_norm = (x - X_MEAN.view(1, -1, 1)) / X_STD.view(1, -1, 1)

            # 2. 🔥【Idea 3】手动放大电压特征
            # 通道 3 是 V_mag，放大它！
            x_norm[:, 3, :] = x_norm[:, 3, :] * VOLT_SCALE

            # 3. 拼接削减后的 PE
            pe_batch = STATIC_PE.unsqueeze(0).expand(x.shape[0], -1, -1)
            x_input = torch.cat([x_norm, pe_batch], dim=1)

            # 4. 进模型 (内部会自动计算 Global Features 并拼接)
            pred = model(x_input)

            if mask.sum() == 0: continue
            mask_exp = mask.unsqueeze(1)
            loss = F.smooth_l1_loss(pred[mask_exp], y[mask_exp])

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses) if losses else 0
        mae, rmse, r2 = evaluate_model(model, test_dl, DEVICE)
        curr_lr = optimizer.param_groups[0]['lr']
        val_str = f"{mae:.6f}" if mae is not None else "nan"

        print(f"{epoch:<6} | Loss:{avg_loss:<8.6f} | MAE:{val_str:<9} | R2:{r2:.4f} | LR:{curr_lr:.1e}")

        if mae is not None:
            scheduler.step(mae)

        if mae is not None and mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), SAVE_PATH)

    print("\n🎉 训练结束！")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))
    mae, rmse, r2, y_t, y_p = evaluate_model(model, test_dl, DEVICE, True)

    if mae is not None:
        plt.figure(figsize=(8, 8))
        hb = plt.hexbin(y_t, y_p, gridsize=50, mincnt=1, cmap='inferno', bins='log')
        plt.colorbar(hb)
        plt.plot([0, 1], [0, 1], "w--")
        plt.title(f"Voltx{VOLT_SCALE} + Global + PE{PE_DIM_KEEP}\nMAE={mae:.4f}, R2={r2:.4f}")
        plt.savefig(IMG_SAVE_PATH, dpi=300)
        try:
            os.startfile(CURRENT_DIR)
        except:
            pass


if __name__ == "__main__":
    main()