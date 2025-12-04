import os
import glob
import random
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch_geometric.loader import DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt  # ★ 新增：用于画散点图

warnings.filterwarnings('ignore')

# ===== 导入 GNN 模型 =====
try:
    from gnn_model import GridGNN
except ImportError as e:
    print(f"❌ 缺少 gnn_model.py 或 GridGNN 定义: {e}")
    exit()

# ================= 配置参数 =================
DATA_DIR   = "./dataset_output_1mv_urban"   # ★ 这里要和生成数据脚本的一致
LR         = 5e-4
EPOCHS     = 400
BATCH_SIZE = 32
HIDDEN_DIM = 128
HEADS      = 4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH  = "best_gnn_model_offline.pth"
SEED       = 0
# ===========================================

# 固定随机种子，保证可复现
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============ 数据加载部分 ============

def load_all_chunks(data_dir):
    """
    从指定目录读取所有 chunk_*.pt 文件，合并为一个 Data 列表。
    每个 chunk 文件是一个 [Data, Data, ...] 的列表。
    """
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        exit()

    pattern = os.path.join(data_dir, "chunk_*.pt")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        print(f"❌ 在目录 {data_dir} 下没有找到 chunk_*.pt 文件")
        exit()

    all_data = []
    for f in files:
        try:
            chunk = torch.load(f, weights_only=False)

            # chunk 应该是一个 Data 列表
            if isinstance(chunk, list):
                all_data.extend(chunk)
            else:
                all_data.append(chunk)
            print(f"📦 已加载 {f}, 当前样本总数: {len(all_data)}")
        except Exception as e:
            print(f"⚠️ 加载 {f} 失败: {e}")

    print(f"\n✅ 数据加载完成，总样本数: {len(all_data)}")
    return all_data


def train_test_split(data_list, train_ratio=0.8):
    """
    简单随机划分训练集和测试集。
    （也可以改成按 data.t_idx / data.stress 做更高级的划分）
    """
    indices = list(range(len(data_list)))
    random.shuffle(indices)

    train_size = int(len(indices) * train_ratio)
    train_idx = indices[:train_size]
    test_idx  = indices[train_size:]

    train_data = [data_list[i] for i in train_idx]
    test_data  = [data_list[i] for i in test_idx]

    print(f"📊 训练集: {len(train_data)} 样本, 测试集: {len(test_data)} 样本")
    return train_data, test_data


# ============ 评估函数 ============

def evaluate_model(model, loader, device, return_arrays=False):
    """
    在给定 DataLoader 上评估：
    - 只对 mask=True 的 sgen 节点计算 MAE/RMSE/R2。
    - 如果 return_arrays=True，则额外返回 y_true, y_pred（numpy 数组）
    """
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # 前向
            pred = model(batch)   # 形状 [total_nodes_in_batch, 1] 或 [total_nodes]
            if pred.dim() == 1:
                pred = pred.unsqueeze(-1)

            target = batch.y      # [total_nodes, 1]
            mask   = batch.mask   # [total_nodes]

            if mask.sum() == 0:
                continue

            pred_sgen   = pred[mask].view(-1).cpu().numpy()
            target_sgen = target[mask].view(-1).cpu().numpy()

            all_pred.append(pred_sgen)
            all_true.append(target_sgen)

    if len(all_true) == 0:
        print("⚠️ 测试集中没有有效的 sgen 节点。")
        if return_arrays:
            return None, None, None, None, None
        else:
            return None, None, None

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    if return_arrays:
        return mae, rmse, r2, y_true, y_pred
    else:
        return mae, rmse, r2


# ============ 主训练流程 ============

def main():
    print(f"🚀 启动离线训练 GNN")
    print(f"📂 数据目录: {DATA_DIR}")
    print(f"🧠 设备: {DEVICE}")

    # 1. 加载数据
    all_data = load_all_chunks(DATA_DIR)

    # （可选）过滤一下 Data，确保都有 y 和 mask
    filtered = []
    for d in all_data:
        if hasattr(d, "y") and hasattr(d, "mask"):
            filtered.append(d)
    if len(filtered) < len(all_data):
        print(f"⚠️ 有 {len(all_data) - len(filtered)} 个样本缺少 y/mask，被丢弃")
    all_data = filtered

    # 2. 划分训练集 / 测试集
    train_data, test_data = train_test_split(all_data, train_ratio=0.8)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    # 3. 初始化模型和优化器
    #   注意：num_node_features=6, num_edge_features=4，需要和生成数据脚本完全一致
    model = GridGNN(
        num_node_features=6,
        num_edge_features=4,
        hidden_dim=HIDDEN_DIM,
        heads=HEADS
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_mae = float("inf")

    print(f"\n{'Epoch':<6} | {'TrainLoss':<10} | {'ValMAE':<10} | {'BestMAE':<10}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            pred = model(batch)   # [total_nodes, 1] or [total_nodes]
            if pred.dim() == 1:
                pred = pred.unsqueeze(-1)

            target = batch.y      # [total_nodes, 1]
            mask   = batch.mask   # [total_nodes]

            if mask.sum() == 0:
                continue

            loss = F.smooth_l1_loss(
                pred[mask],
                target[mask],
                beta=0.1
            )
            loss.backward()

            # 梯度裁剪（和你在线版一致）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            epoch_losses.append(loss.item())

        # 一个 epoch 完成后的平均 loss
        if len(epoch_losses) > 0:
            train_loss = float(np.mean(epoch_losses))
        else:
            train_loss = float("nan")

        # 每 N 个 epoch 做一次验证（这里每个 epoch 都做也行）
        mae, rmse, r2 = evaluate_model(model, test_loader, DEVICE)
        if mae is not None and mae < best_val_mae:
            best_val_mae = mae
            torch.save(model.state_dict(), SAVE_PATH)

        if mae is not None:
            print(f"{epoch:<6} | {train_loss:<10.6f} | {mae:<10.6f} | {best_val_mae:<10.6f}")
        else:
            print(f"{epoch:<6} | {train_loss:<10.6f} | {'nan':<10} | {best_val_mae:<10.6f}")

    print("\n🎉 训练结束！")
    print(f"💾 最佳模型已保存到: {SAVE_PATH}")

    # 最终再用最佳模型评估一次，打印最终指标 + 画散点图
    try:
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"⚠️ 无法重新加载最佳模型权重: {e}")

    final_mae, final_rmse, final_r2, y_true, y_pred = evaluate_model(
        model, test_loader, DEVICE, return_arrays=True
    )
    if final_mae is not None:
        print("\n🏆 最终测试集指标:")
        print(f"   MAE  : {final_mae:.6f}")
        print(f"   RMSE : {final_rmse:.6f}")
        print(f"   R2   : {final_r2:.6f}")

        # ========= 实际动作 vs 理想动作 散点图 =========
        # 默认：y_true = 理想动作（标签），y_pred = GNN 预测动作
        # 如点太多可以在这里加采样
        # idx = np.random.choice(len(y_true), size=min(10000, len(y_true)), replace=False)
        # y_true_plot = y_true[idx]
        # y_pred_plot = y_pred[idx]
        # 现在先用全量
        y_true_plot = y_true
        y_pred_plot = y_pred

        plt.figure(figsize=(6, 6))

        # 使用 hexbin 按频次着色，bins='log' 让颜色按对数缩放，看起来更平滑
        hb = plt.hexbin(
            y_true_plot,
            y_pred_plot,
            gridsize=60,  # 格子数量，可以调大/调小
            mincnt=1,  # 只有至少有 1 个点的格子才画
            bins='log'  # 颜色按 log(count) 显示
        )
        cb = plt.colorbar(hb)
        cb.set_label("log10(count)")  # 颜色条标签：点的对数数量

        # 画一条 y = x 参考线（完美预测）
        min_val = min(np.min(y_true_plot), np.min(y_pred_plot))
        max_val = max(np.max(y_true_plot), np.max(y_pred_plot))
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

        plt.xlabel("Ideal action (OPF target, $\\alpha_{true}$)")
        plt.ylabel("Predicted action (GNN output, $\\alpha_{pred}$)")
        plt.title("Action density plot: predicted vs. ideal (all 132 generators)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("action_density_hexbin.png", dpi=300)
        print("📈 密度散点图已保存为: action_density_hexbin.png")

if __name__ == "__main__":
    main()
