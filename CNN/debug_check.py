import os
import glob
import torch
import numpy as np

# ================= 配置 =================
DATA_DIR = "./dataset_output_1mv_urban"

import torch
import numpy as np

def safe_torch_load(path: str):

    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([np.core.multiarray._reconstruct])  # 关键：放行这个符号
        return torch.load(path, weights_only=True, map_location="cpu")
    except Exception as e1:
        print(f"⚠️ weights_only=True 仍失败: {e1}")

    return torch.load(path, weights_only=False, map_location="cpu")

# =======================================

def check_static_assets():
    print("🔍 [1/3] 检查静态资产 (static_assets.pt)...")
    # 1. 这里定义了变量名是 'path'
    path = os.path.join(DATA_DIR, "static_assets.pt")

    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return None, None

    try:
        # 2. 修正这里：把 static_assets_path 改成 path
        assets = safe_torch_load(path)

        perm = assets['perm']
        pe = assets['pe']
        n_nodes = assets['num_nodes']

        print(f"   ✅ 文件加载成功")
        print(f"   - 记录节点数 (N): {n_nodes}")
        print(f"   - RCM 索引长度: {len(perm)}")
        print(f"   - PE 形状: {pe.shape} (应为 [N, 16])")

        # 验证 RCM 索引是否有效
        if len(perm) != n_nodes:
            print(f"❌ 错误: Perm 长度 ({len(perm)}) 与 N ({n_nodes}) 不一致!")

        # 验证 PE 是否有 NaN
        if torch.isnan(pe).any():
            print("❌ 错误: PE 中包含 NaN!")

        return n_nodes, pe.shape[1]  # 返回 N 和 PE_dim

    except Exception as e:
        print(f"❌ 读取出错: {e}")
        # 打印详细错误栈，方便排查
        import traceback
        traceback.print_exc()
        return None, None


def check_stats():
    print("\n🔍 [2/3] 检查统计量 (stats.pt)...")
    path = os.path.join(DATA_DIR, "stats.pt")

    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return None, None

    try:
        stats = torch.load(path)
        mean = stats['x_mean']
        std = stats['x_std']

        print(f"   ✅ 文件加载成功")
        print(f"   - Mean 形状: {mean.shape} (应为 [6])")
        print(f"   - Std  形状: {std.shape} (应为 [6])")
        print(f"   - Mean 数值: {mean.numpy()}")
        print(f"   - Std  数值: {std.numpy()}")

        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("❌ 错误: 统计量包含 NaN!")

        if (std == 0).any():
            print("⚠️ 警告: 某些特征的 Std 为 0，这可能导致归一化除零错误 (训练脚本里需要处理)。")

        return mean, std

    except Exception as e:
        print(f"❌ 读取出错: {e}")
        return None, None


def check_data_chunks(expected_n_nodes):
    print("\n🔍 [3/3] 检查数据块样本 (chunk_*.pt)...")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "chunk_*.pt")))

    if len(files) == 0:
        print("❌ 没有找到数据块文件 (chunk_*.pt)")
        return

    print(f"   ✅ 找到 {len(files)} 个数据块文件")

    # 只检查第一个文件，避免刷屏
    first_file = files[0]
    print(f"   👉 正在深入检查第一个文件: {os.path.basename(first_file)}")

    try:
        samples = torch.load(first_file)
        print(f"   - 样本数量: {len(samples)}")

        if len(samples) == 0:
            print("⚠️ 警告: 数据块是空的!")
            return

        # 抽取第0个样本进行详细解剖
        sample = samples[0]
        x = sample['x']
        y = sample['y']
        mask = sample['mask']

        print(f"   --- 样本 #0 详情 ---")
        print(f"   - x shape: {x.shape} (应为 [6, {expected_n_nodes}])")
        print(f"   - y shape: {y.shape} (应为 [1, {expected_n_nodes}])")
        print(f"   - mask shape: {mask.shape} (应为 [{expected_n_nodes}])")

        # 1. 维度检查
        if x.shape[1] != expected_n_nodes:
            print(f"❌ 致命错误: 样本节点数 ({x.shape[1]}) 与静态资产 ({expected_n_nodes}) 不一致！训练必挂！")

        # 2. 数值检查
        if torch.isnan(x).any():
            print("❌ 错误: 输入特征 x 包含 NaN")
        if torch.isnan(y).any():
            print("❌ 错误: 标签 y 包含 NaN")

        # 3. Mask 逻辑检查 (Bus 86)
        active_gens = mask.sum().item()
        print(f"   - 有效发电机数 (Mask=True): {active_gens}")

        if active_gens == 0:
            print("⚠️ 警告: 该样本没有有效的发电机 (全是 False)！可能是过滤逻辑太严，或者所有发电机都被关停了。")
        else:
            # 检查 Mask 为 True 的地方，y 是否在 [0, 1] 之间
            # mask 需要扩展维度才能索引 y [1, N]
            y_valid = y[0][mask]
            print(f"   - 有效 y 值示例: {y_valid[:5].numpy()}")
            if (y_valid < 0).any() or (y_valid > 1.0).any():
                print("⚠️ 警告: 某些目标值 y 超出了 [0, 1] 范围！")
            else:
                print("   ✅ 目标值范围正常 [0, 1]")

        # 4. 特征范围检查 (Sanity Check)
        v_mag = x[3, :]  # 第4行是电压幅值
        print(f"   - 电压幅值 (x[3]) 范围: Min={v_mag.min():.4f}, Max={v_mag.max():.4f}")
        if v_mag.max() > 1.2 or v_mag.min() < 0.8:
            print("⚠️ 警告: 电压幅值看起来有点异常 (偏离 1.0 太多)，请确认单位是否正确。")

    except Exception as e:
        print(f"❌ 检查数据块时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("🚑 启动数据体检程序...\n")

    # 1. 检查静态资产
    n_nodes, pe_dim = check_static_assets()

    if n_nodes is None:
        print("\n🚫 体检终止：静态资产缺失，无法继续检查。")
        return

    # 2. 检查统计量
    check_stats()

    # 3. 检查数据样本
    check_data_chunks(n_nodes)

    print("\n✅ 体检结束。如果没有红色❌，你可以放心地开始训练了！")


if __name__ == "__main__":
    main()