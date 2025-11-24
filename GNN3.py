import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import simbench as sb
from torch_geometric.data import Data
import warnings
from copy import deepcopy
import csv

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
EPOCHS = 40000
BATCH_SIZE = 4
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
    2. Edge Features: 4维 (Rate除以1000, Loading取绝对值)
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

    # 4. 边特征处理 (修复数值过大问题)
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
                    # 1. Rate 除以 1000，使其落在 0~10 范围内
                    feat_rate = rate / 1000.0

                    # 2. Loading 取绝对值，处理负数方向问题
                    #    GridCal loading 看起来是 p.u. (0.7 = 70%)
                    feat_load = abs(loading_val)

                    # 3. 这里的 R 和 X 虽然小 (0.0003)，但因为是物理属性，
                    #    且现在 Rate 已经被压下来了，GNN 应该能学到了。
                    #    (如果后续 Loss 还不降，可以考虑把 R, X 乘 100)
                    edge_feat = [r, x_val, feat_rate, feat_load]
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

def check_data_stats(data):
    """
    打印 Node 和 Edge 特征的统计分布，检查数值范围是否合理。
    """
    x = data.x.cpu().numpy()
    edge = data.edge_attr.cpu().numpy()

    print("\n🔎 ----- 输入数据数值范围检查 (Input Data Stats) -----")

    # --- 节点特征 ---
    node_cols = [
        "0: Load P (x3)",
        "1: Load Q (x3)",
        "2: Gen P (/10)",
        "3: V_mag_sc",
        "4: Mask",
        "5: V_angle"
    ]
    print("📌 [Node Features]")
    for i, name in enumerate(node_cols):
        col = x[:, i]
        print(
            f"  {name:<15} | Min: {col.min():.4f} | Max: {col.max():.4f} | Mean: {np.mean(col):.4f} | Std: {np.std(col):.4f}")

    # --- 边特征 ---
    edge_cols = [
        "0: R (Resistance)",
        "1: X (Reactance)",
        "2: Rate (MVA)",
        "3: Loading (%)"
    ]
    print("🔗 [Edge Features]")
    for i, name in enumerate(edge_cols):
        col = edge[:, i]
        print(
            f"  {name:<15} | Min: {col.min():.4f} | Max: {col.max():.4f} | Mean: {np.mean(col):.4f} | Std: {np.std(col):.4f}")

    print("----------------------------------------------------\n")

def main():
    print(f"🚀 启动训练: {SB_CODE} (Gen x6-8, Features Scaled)")

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

    # 这里的特征维度已经是正确的 6 和 4
    model = GridGNN(num_node_features=6, num_edge_features=4,
                    hidden_dim=HIDDEN_DIM, heads=HEADS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_loss = float('inf')
    accumulated_loss = 0.0

    print(f"\n{'Epoch':<6} | {'Time':<6} | {'OPF':<8} | {'Loss (Avg)':<12} | {'Teacher':<8} | {'Student':<8}")
    print("-" * 70)

    optimizer.zero_grad()
    log_f = open("loss_log.csv", mode="w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["epoch", "time_index", "avg_loss", "avg_teacher", "avg_student"])

    for epoch in range(EPOCHS):
        t = int(np.random.choice(train_idx))

        # Modification: 1. Generation Multiplier 6-8x
        stress_factor = np.random.uniform(6.0, 8.0)

        current_load_p = df_load_p.iloc[t]
        current_load_q = df_load_q.iloc[t]
        for l in grid.loads:
            try:
                idx = int(l.name.split('_')[1])
                _set_val(l, ['P', 'p_mw', 'p'], current_load_p.get(idx, 0.0))
                _set_val(l, ['Q', 'q_mvar', 'q'], current_load_q.get(idx, 0.0))
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

        # 3. Pre-PF Run
        pf_ok = False
        current_pf_results = None
        pf_driver = None
        try:
            pf_opts = gce.PowerFlowOptions(gce.SolverType.NR, verbose=False)
            pf_driver = gce.PowerFlowDriver(grid_pf, pf_opts)
            pf_driver.run()
            if hasattr(pf_driver, 'results'):
                current_pf_results = pf_driver.results
                if hasattr(pf_driver.results, 'converged'):
                    pf_ok = pf_driver.results.converged
        except:
            pass

        if pf_driver and hasattr(pf_driver, 'results'):
            current_pf_results = pf_driver.results

        # 4. OPF Teacher
        teacher_driver = setup_and_run_opf_teacher(grid, sgen_p_dict)

        opf_ok = False
        if teacher_driver and hasattr(teacher_driver.results, 'converged'):
            opf_ok = teacher_driver.results.converged

        if not opf_ok:
            if epoch % 50 == 0: print(f"{epoch:<6} | {t:<6} | ❌       | Skipped      | -        | -")
            continue

        gen_p_vec = get_safe_gen_results(teacher_driver, grid)
        target_alphas = []
        full_target = torch.zeros(len(grid.buses), 1).to(DEVICE)

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
                            target_alphas.append(alpha)

        if not target_alphas: continue

        # 5. GNN Training
        data, mask = get_graph_data(grid, current_pf_results, bus_idx_map)

        # ==========================================
        # 👇【修改点】在这里插入检查代码 👇
        # ==========================================
        if epoch == 0:
            print(f"\n[DEBUG] 正在检查第 {epoch} 轮的数据特征分布...")
            check_data_stats(data)
            # 如果你希望看清楚数据再继续，可以取消下面这行的注释
            # input("🔴 请检查上方数据统计，按回车键继续训练...")
        # ==========================================

        pred = model(data)

        if mask.sum() > 0:
            loss = F.smooth_l1_loss(pred[mask], full_target[mask], beta=0.1)

            loss.backward()

            accumulated_loss += loss.item() * BATCH_SIZE

            if (epoch + 1) % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()

                avg_loss = accumulated_loss / BATCH_SIZE
                accumulated_loss = 0.0

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), SAVE_PATH)

                # === 修改这里 ===
                # 计算 Teacher 的统计
                t_min = np.min(target_alphas)
                t_max = np.max(target_alphas)
                t_mean = np.mean(target_alphas)

                # 计算 Student (Pred) 的统计
                s_vals = pred[mask].detach().cpu().numpy()
                s_min = np.min(s_vals)
                s_max = np.max(s_vals)
                s_mean = np.mean(s_vals)

                print(f"Epoch {epoch:<5} | Loss: {avg_loss:.5f}")
                print(f"  👨‍🏫 Teacher: Mean={t_mean:.3f} | Min={t_min:.3f} | Max={t_max:.3f}")
                print(f"  🧑‍🎓 Student: Mean={s_mean:.3f} | Min={s_min:.3f} | Max={s_max:.3f}")
                print("-" * 50)

                log_writer.writerow([epoch + 1, t, avg_loss, t_mean, s_mean])

    print(f"\n🎉 训练完成！最佳 Loss: {best_loss:.6f}")
    log_f.close()


if __name__ == "__main__":
    main()