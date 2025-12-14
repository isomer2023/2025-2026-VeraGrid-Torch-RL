import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import simbench as sb
from torch_geometric.data import Data
import warnings
from copy import deepcopy
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
# 忽略 Pandas 警告
warnings.filterwarnings('ignore')

# 导入环境和引擎
try:
    import GC_PandaPowerImporter
    from VeraGridEngine import api as gce
    from gnn_model import GridGNN
except ImportError as e:
    print(f"❌ 缺少依赖文件: {e}")
    exit()

# ================= 配置参数 =================
SB_CODE = "1-MV-urban--0-sw"
LR = 0.0005
EPOCHS = 4000
# ✅ [关键修改] 增大 Batch Size，让 BatchNorm 正常工作
BATCH_SIZE = 32
HIDDEN_DIM = 128
HEADS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = "best_gnn_model.pth"


# ===========================================

# --- 辅助函数 ---
def get_gc_id(obj):
    if hasattr(obj, 'id') and obj.id is not None: return obj.id
    if hasattr(obj, 'idtag') and obj.idtag is not None: return obj.idtag
    if hasattr(obj, 'uuid') and obj.uuid is not None: return obj.uuid
    if hasattr(obj, 'name') and obj.name is not None: return obj.name
    return str(obj)


def _set_val(obj, attr_list, val):
    for attr in attr_list:
        try:
            setattr(obj, attr, val);
            return
        except:
            continue


def _get_val(obj, attr_list, default=0.0):
    for attr in attr_list:
        if hasattr(obj, attr):
            try:
                return float(getattr(obj, attr))
            except:
                continue
    return default


def _is_slack_safe(g):
    val = getattr(g, 'is_slack', None)
    if val is not None: return bool(val)
    val = getattr(g, 'slack', None)
    if val is not None: return bool(val)
    name = str(getattr(g, 'name', ''))
    if "Ext_Grid" in name: return True
    return False


def get_safe_gen_results(driver, grid):
    if driver and hasattr(driver, 'results'):
        res = driver.results
        if hasattr(res, 'gen_p') and res.gen_p is not None: return res.gen_p
        if hasattr(res, 'P') and res.P is not None: return res.P
    p_list = []
    for g in grid.generators:
        p_val = _get_val(g, ['P', 'p_mw', 'p'], 0.0)
        p_list.append(p_val)
    return p_list


# --- 核心模块 ---

def setup_and_run_opf_teacher(grid, current_profile_sgen):
    """【老师模块】"""
    for g in grid.generators:
        g_name = str(getattr(g, 'name', ''))

        if _is_slack_safe(g) or "Ext_Grid" in g_name:
            _set_val(g, ['Pmax', 'P_max'], 99999.0)
            _set_val(g, ['Pmin', 'P_min'], -99999.0)
            _set_val(g, ['cost_a', 'Cost1'], 1.0)
            _set_val(g, ['cost_b', 'Cost2'], 100.0)
            _set_val(g, ['is_controlled', 'controlled'], True)

        elif "sgen" in g_name:
            try:
                sgen_idx = int(g_name.split('_')[1])
                p_avail = current_profile_sgen.get(sgen_idx, 0.0)
            except:
                p_avail = 0.0

            _set_val(g, ['Pmax', 'P_max'], p_avail)
            _set_val(g, ['Pmin', 'P_min'], 0.0)
            _set_val(g, ['cost_a', 'Cost1'], 0.01)
            _set_val(g, ['cost_b', 'Cost2'], 0.1)
            _set_val(g, ['is_controlled', 'controlled'], True)
            _set_val(g, ['Qmax', 'Q_max'], 0.0)
            _set_val(g, ['Qmin', 'Q_min'], 0.0)

    opf_opts = gce.OptimalPowerFlowOptions()
    if hasattr(gce, 'SolverType'):
        _set_val(opf_opts, ['solver', 'solver_type'], gce.SolverType.NONLINEAR_OPF)

    opf_opts.objective = 0
    _set_val(opf_opts, ['activate_voltage_limits', 'voltage_limits'], True)
    _set_val(opf_opts, ['vmin', 'Vmin'], 0.98)
    _set_val(opf_opts, ['vmax', 'Vmax'], 1.02)
    _set_val(opf_opts, ['activate_thermal_limits', 'thermal_limits'], True)
    _set_val(opf_opts, ['dispatch_P', 'control_active_power'], True)
    _set_val(opf_opts, ['dispatch_Q', 'control_reactive_power'], False)
    _set_val(opf_opts, ['allow_soft_limits', 'soft_limits'], True)
    _set_val(opf_opts, ['initialize_with_dc', 'init_dc'], False)

    opf_driver = gce.OptimalPowerFlowDriver(grid, opf_opts)
    try:
        opf_driver.run()
    except:
        pass
    return opf_driver


def get_graph_data(grid, pf_results, bus_idx_map):
    """
    GridCal -> PyG Data
    修正版：
    1. Node Features: 6维 (含电压相角)
    2. Edge Features: 4维 (Rate除以1000, Loading取绝对值, R/X放大)
    """
    num_nodes = len(grid.buses)
    x = np.zeros((num_nodes, 6), dtype=np.float32)

    # 1. 节点负荷 (Load P, Q) -> 真实值 * 3
    for l in grid.loads:
        bus_ref = getattr(l, 'bus', getattr(l, 'node', None))
        if bus_ref:
            idx = bus_idx_map.get(get_gc_id(bus_ref))
            if idx is not None:
                p_val = _get_val(l, ['P', 'p_mw', 'p'])
                q_val = _get_val(l, ['Q', 'q_mvar', 'q'])
                x[idx, 0] += p_val * 3.0
                x[idx, 1] += q_val * 3.0

    # 2. 节点电源 (Gen P) -> 真实值 / 10
    sgen_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for g in grid.generators:
        bus_ref = getattr(g, 'bus', getattr(g, 'node', None))
        if bus_ref:
            idx = bus_idx_map.get(get_gc_id(bus_ref))
            if idx is not None:
                x[idx, 4] = 1.0
                if "sgen" in getattr(g, 'name', ''):
                    p_max = _get_val(g, ['Pmax', 'P_max'], 0.0)
                    x[idx, 2] += p_max / 10.0
                    sgen_mask[idx] = True

    # 3. 节点电压 (幅值 + 相角)
    v_mag_scaled = np.zeros(num_nodes)
    v_ang = np.zeros(num_nodes)
    if pf_results and hasattr(pf_results, 'voltage'):
        v_complex = np.array(pf_results.voltage, dtype=np.complex128)
        if len(v_complex) == num_nodes:
            v_abs = np.abs(v_complex)
            v_mag_scaled = (v_abs - 1.0) * 10.0  # (|V|-1)*10
            v_ang = np.angle(v_complex)  # 相角 (弧度)

    x[:, 3] = v_mag_scaled
    x[:, 5] = v_ang

    # 4. 边特征处理
    src, dst, attr = [], [], []
    branches = []
    if hasattr(grid, 'branches'):
        branches = grid.branches
    elif hasattr(grid, 'get_branches'):
        branches = grid.get_branches()
    else:
        branches = list(getattr(grid, 'lines', [])) + list(getattr(grid, 'transformers', []))

    branch_loadings = []
    if pf_results and hasattr(pf_results, 'loading'):
        branch_loadings = pf_results.loading

    for i, br in enumerate(branches):
        try:
            if _get_val(br, ['active', 'status'], 1.0) < 0.5: continue
            f_ref = getattr(br, 'bus_from', getattr(br, 'from_node', getattr(br, 'busFrom', None)))
            t_ref = getattr(br, 'bus_to', getattr(br, 'to_node', getattr(br, 'busTo', None)))

            if f_ref and t_ref:
                u = bus_idx_map.get(get_gc_id(f_ref))
                v = bus_idx_map.get(get_gc_id(t_ref))
                if u is not None and v is not None:
                    # 原始物理参数
                    r = float(_get_val(br, ['r', 'R']))
                    x_val = float(_get_val(br, ['x', 'X']))
                    rate = float(_get_val(br, ['rate', 'Rate'], 100.0))

                    loading_val = 0.0
                    if branch_loadings is not None and i < len(branch_loadings):
                        loading_val = float(branch_loadings[i])

                    # =========== 关键修正 ===========
                    feat_rate = rate / 1000.0  # Rate 归一化
                    feat_load = abs(loading_val)  # Loading 取绝对值
                    feat_r = r * 100.0  # 放大 R
                    feat_x = x_val * 100.0  # 放大 X

                    edge_feat = [feat_r, feat_x, feat_rate, feat_load]
                    # ===============================

                    src.extend([u, v])
                    dst.extend([v, u])
                    attr.extend([edge_feat, edge_feat])
        except:
            continue

    x_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    edge_index = torch.tensor([src, dst], dtype=torch.long).to(DEVICE)
    edge_attr = torch.tensor(attr, dtype=torch.float32).to(DEVICE)
    return Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr), sgen_mask.to(DEVICE)

# ================= 修正后的评估函数 =================
def evaluate_model(model, grid, bus_idx_map, test_idx, df_load_p, df_load_q, df_sgen_p, device):
    print("\n" + "=" * 40)
    print("🧪 启动测试集评估 (Evaluation Phase)")
    print("=" * 40)

    try:
        model.load_state_dict(torch.load("best_gnn_model.pth"))
        print("✅ 已加载最佳模型权重: best_gnn_model.pth")
    except Exception as e:
        print(f"⚠️ 无法加载模型权重，将使用当前权重: {e}")

    model.eval()

    # 随机采样 1000 个
    num_samples = 1000
    if len(test_idx) > num_samples:
        eval_indices = np.random.choice(test_idx, num_samples, replace=False)
    else:
        eval_indices = test_idx

    print(f"📊 采样数量: {len(eval_indices)} (来自测试集)")
    results_list = []

    with torch.no_grad():
        for i, t in enumerate(eval_indices):
            if (i + 1) % 100 == 0:
                print(f"   进度: {i + 1}/{len(eval_indices)}...")

            # --- A. 环境重构 ---
            stress_factor = np.random.uniform(6.0, 8.0)

            # 设置负荷
            current_load_p = df_load_p.iloc[t]
            current_load_q = df_load_q.iloc[t]
            for l in grid.loads:
                try:
                    idx = int(l.name.split('_')[1])
                    _set_val(l, ['P', 'p'], current_load_p.get(idx, 0.0))
                    _set_val(l, ['Q', 'q'], current_load_q.get(idx, 0.0))
                except:
                    pass

            # 设置发电机
            sgen_p_dict = (df_sgen_p.iloc[t] * stress_factor).to_dict()
            grid_pf = deepcopy(grid)
            for g in grid_pf.generators:
                if "sgen" in getattr(g, 'name', ''):
                    try:
                        idx = int(g.name.split('_')[1])
                        p_avail = sgen_p_dict.get(idx, 0.0)
                        _set_val(g, ['Pmax', 'P_max'], p_avail)
                        _set_val(g, ['P', 'p'], p_avail)
                    except:
                        pass

            # --- B. 计算 Pre-PF ---
            pf_driver = None
            current_pf_results = None
            try:
                pf_opts = gce.PowerFlowOptions(gce.SolverType.NR, verbose=False)
                pf_driver = gce.PowerFlowDriver(grid_pf, pf_opts)
                pf_driver.run()
                current_pf_results = pf_driver.results
            except:
                pass

            # --- C. 计算 OPF Teacher ---
            teacher_driver = setup_and_run_opf_teacher(grid, sgen_p_dict)
            if not (teacher_driver and hasattr(teacher_driver.results,
                                               'converged') and teacher_driver.results.converged):
                continue

            gen_p_vec = get_safe_gen_results(teacher_driver, grid)

            # --- D. GNN 预测 ---
            data, mask = get_graph_data(grid, current_pf_results, bus_idx_map)
            # data.x 的形状是 [num_nodes, features]
            # pred 的形状通常是 [num_nodes, 1] 或者 [num_nodes]
            pred = model(data)

            for g_idx, g in enumerate(grid.generators):
                if "sgen" in getattr(g, 'name', ''):
                    bus = getattr(g, 'bus', getattr(g, 'node', None))
                    if bus:
                        node_idx = bus_idx_map.get(get_gc_id(bus))
                        if node_idx is not None:
                            # 1. 获取真值 (Teacher)
                            p_opt = float(gen_p_vec[g_idx])
                            p_avail = _get_val(g, ['Pmax', 'P_max'])

                            # 2. 获取预测值 (Student) - 直接通过节点索引拿
                            pred_val = float(pred[node_idx].item())

                            # 3. 过滤并保存
                            # 我们只关心那些有能力发电的机组
                            if p_avail > 0.001:
                                true_alpha = np.clip(p_opt / p_avail, 0.0, 1.0)
                                # 预测值通常也需要截断到 [0,1] 区间以便分析，虽然模型输出可能略微越界
                                pred_alpha_clamped = np.clip(pred_val, 0.0, 1.0)

                                results_list.append({
                                    "Time_Step": t,
                                    "Gen_ID": g.name,
                                    "True_Alpha": true_alpha,
                                    "Pred_Alpha": pred_alpha_clamped,
                                    "Error": pred_alpha_clamped - true_alpha,
                                    "Abs_Error": abs(pred_alpha_clamped - true_alpha)
                                })

    df = pd.DataFrame(results_list)
    if df.empty:
        print("❌ 没有收集到有效的测试数据。")
        return

    df.to_csv("eval_results_detailed.csv", index=False)
    print(f"\n✅ 详细数据已保存: eval_results_detailed.csv ({len(df)} 条记录)")

    mae = mean_absolute_error(df["True_Alpha"], df["Pred_Alpha"])
    rmse = np.sqrt(mean_squared_error(df["True_Alpha"], df["Pred_Alpha"]))
    r2 = r2_score(df["True_Alpha"], df["Pred_Alpha"])

    print(f"\n🏆 评估指标:")
    print(f"   MAE  : {mae:.6f}")
    print(f"   RMSE : {rmse:.6f}")
    print(f"   R2   : {r2:.6f}")

    with open("eval_metrics.txt", "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}\nSamples: {len(df)}")

    sns.set_theme(style="whitegrid")

    # 1. Scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(df["True_Alpha"], df["Pred_Alpha"], alpha=0.15, s=10, color="#1f77b4")
    plt.plot([0, 1], [0, 1], "r--", linewidth=2, label="Ideal")
    plt.xlabel("Teacher (OPF) Alpha")
    plt.ylabel("Student (GNN) Alpha")
    plt.title(f"Prediction vs Ground Truth (N={len(df)})\nMAE={mae:.4f}, R2={r2:.4f}")
    plt.legend()
    plt.savefig("eval_1_scatter.png", dpi=300)
    plt.close()

    # 2. Hist
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Error"], bins=100, kde=True, color="purple", stat="density")
    plt.axvline(0, color='r', linestyle='--')
    plt.xlabel("Error (Pred - True)")
    plt.title("Error Distribution Histogram")
    plt.savefig("eval_2_error_hist.png", dpi=300)
    plt.close()

    # 3. Boxplot
    plt.figure(figsize=(14, 6))
    gen_errors = df.groupby("Gen_ID")["Abs_Error"].mean().sort_values(ascending=False)
    top_gens = gen_errors.head(30).index
    df_filtered = df[df["Gen_ID"].isin(top_gens)]
    sns.boxplot(x="Gen_ID", y="Abs_Error", data=df_filtered, palette="Reds_r", order=top_gens)
    plt.xticks(rotation=90)
    plt.title("Absolute Error per Generator (Top 30 Worst Controlled)")
    plt.ylabel("Absolute Error")
    plt.tight_layout()
    plt.savefig("eval_3_gen_boxplot.png", dpi=300)
    plt.close()

    print("🖼️  所有图像已生成: eval_1_scatter.png, eval_2_error_hist.png, eval_3_gen_boxplot.png")


# 必须引入 Batch 工具
from torch_geometric.data import Batch


def main():
    print(f"🚀 启动训练: {SB_CODE} (Batch={BATCH_SIZE})")

    # 1. 准备数据和环境
    net_pp = sb.get_simbench_net(SB_CODE)
    grid = GC_PandaPowerImporter.PP2GC(net_pp)
    bus_idx_map = {get_gc_id(b): i for i, b in enumerate(grid.buses)}

    print("📦 加载 Profiles...")
    profiles = sb.get_absolute_values(net_pp, profiles_instead_of_study_cases=True)
    df_load_p = profiles[('load', 'p_mw')]
    df_load_q = profiles[('load', 'q_mvar')]
    df_sgen_p = profiles[('sgen', 'p_mw')]

    n_time_steps = len(df_load_p)
    all_idx = np.arange(n_time_steps)
    split1 = int(0.8 * n_time_steps)
    train_idx = all_idx[:split1]
    test_idx = all_idx[split1:]

    # 2. 模型初始化
    model = GridGNN(num_node_features=6, num_edge_features=4,
                    hidden_dim=HIDDEN_DIM, heads=HEADS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n{'Epoch':<6} | {'Loss':<10} | {'Info'}")
    print("-" * 50)

    log_f = open("loss_log.csv", mode="w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["epoch", "time_index", "loss"])

    best_loss = float('inf')

    # 【修改点 1】 创建一个列表来暂存 Batch 数据
    batch_data_list = []

    optimizer.zero_grad()  # 移到循环外初始化

    for epoch in range(EPOCHS):
        t = int(np.random.choice(train_idx))

        # --- 环境生成 (保持不变) ---
        stress_factor = np.random.uniform(6.0, 8.0)
        current_load_p = df_load_p.iloc[t]
        current_load_q = df_load_q.iloc[t]
        for l in grid.loads:
            try:
                idx = int(l.name.split('_')[1])
                _set_val(l, ['P', 'p'], current_load_p.get(idx, 0.0))
                _set_val(l, ['Q', 'q'], current_load_q.get(idx, 0.0))
            except:
                pass

        sgen_p_dict = (df_sgen_p.iloc[t] * stress_factor).to_dict()
        grid_pf = deepcopy(grid)
        for g in grid_pf.generators:
            if "sgen" in getattr(g, 'name', ''):
                try:
                    idx = int(g.name.split('_')[1])
                    p_avail = sgen_p_dict.get(idx, 0.0)
                    _set_val(g, ['Pmax', 'P_max'], p_avail)
                    _set_val(g, ['P', 'p'], p_avail)
                except:
                    pass

        # --- Pre-PF (保持不变) ---
        pf_driver = None
        current_pf_results = None
        try:
            pf_opts = gce.PowerFlowOptions(gce.SolverType.NR, verbose=False)
            pf_driver = gce.PowerFlowDriver(grid_pf, pf_opts)
            pf_driver.run()
            current_pf_results = pf_driver.results
        except:
            pass

        # --- OPF Teacher (保持不变) ---
        teacher_driver = setup_and_run_opf_teacher(grid, sgen_p_dict)
        if not (teacher_driver and hasattr(teacher_driver.results, 'converged') and teacher_driver.results.converged):
            continue

        gen_p_vec = get_safe_gen_results(teacher_driver, grid)

        # 【修改点 2】 Target 生成：需要存储到 CPU，不需要立刻转 GPU
        # 我们需要一个全零的 Target 向量，长度等于节点数
        full_target = torch.zeros(len(grid.buses), 1)
        valid_sample = False

        for i, g in enumerate(grid.generators):
            if "sgen" in getattr(g, 'name', ''):
                bus = getattr(g, 'bus', getattr(g, 'node', None))
                if bus:
                    idx = bus_idx_map.get(get_gc_id(bus))
                    if idx is not None:
                        p_opt = float(gen_p_vec[i])
                        p_avail = _get_val(g, ['Pmax', 'P_max'])
                        if p_avail > 0.001:
                            alpha = np.clip(p_opt / p_avail, 0.0, 1.0)
                            full_target[idx] = alpha
                            valid_sample = True

        if not valid_sample: continue

        # --- 获取图数据 ---
        # 注意：这里我们只要 CPU 数据，最后 Batch 了一起转 GPU
        data, mask = get_graph_data(grid, current_pf_results, bus_idx_map)

        # 【修改点 3】 将 target 和 mask 挂载到 data 对象上
        # 因为 Batch() 函数会自动拼接 data 对象里的属性，只要维度对得上
        data.y_target = full_target  # [Num_Nodes, 1]
        data.mask = mask.cpu()  # [Num_Nodes] (转回 CPU 方便 Batch)
        data.to('cpu')  # 确保都在 CPU 上

        batch_data_list.append(data)

        # 【修改点 4】 真正的 Batch 训练逻辑
        if len(batch_data_list) >= BATCH_SIZE:
            model.train()
            optimizer.zero_grad()

            # A. 物理拼接：把 32 个小图拼成 1 个大图
            # 这时候 BatchNorm 看到的是 (Num_Nodes * 32) 个点，统计数据非常稳定
            big_batch = Batch.from_data_list(batch_data_list).to(DEVICE)

            # B. 前向传播
            pred = model(big_batch)

            # C. 取出拼接后的 Target 和 Mask
            target_batch = big_batch.y_target.to(DEVICE)
            mask_batch = big_batch.mask.to(DEVICE)

            # D. 计算 Loss
            if mask_batch.sum() > 0:
                loss = F.smooth_l1_loss(pred[mask_batch], target_batch[mask_batch], beta=0.1)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # E. 记录和保存
                current_loss = loss.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    torch.save(model.state_dict(), SAVE_PATH)

                print(f"{epoch:<6} | {current_loss:.5f}       | Best: {best_loss:.5f}")
                log_writer.writerow([epoch, t, current_loss])

            # F. 清空列表
            batch_data_list = []

    log_f.close()
    print(f"\n🎉 训练完成！启动评估...")
    evaluate_model(model, grid, bus_idx_map, test_idx, df_load_p, df_load_q, df_sgen_p, DEVICE)

if __name__ == "__main__":
    main()