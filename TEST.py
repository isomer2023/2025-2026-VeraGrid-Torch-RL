import torch
import numpy as np
import simbench as sb
from torch_geometric.data import Data
import warnings
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 忽略 Pandas 警告
warnings.filterwarnings('ignore')

# 引入你的环境依赖
try:
    import GC_PandaPowerImporter
    from VeraGridEngine import api as gce
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    exit()

# ================= 配置 =================
SB_CODE = "1-MV-urban--0-sw"
CHECK_SAMPLES = 5  # 检查多少个样本


# =======================================

# --- 复制你代码中的辅助函数 (必须保持完全一致) ---
def get_gc_id(obj):
    if hasattr(obj, 'id') and obj.id is not None: return obj.id
    if hasattr(obj, 'idtag') and obj.idtag is not None: return obj.idtag
    if hasattr(obj, 'uuid') and obj.uuid is not None: return obj.uuid
    if hasattr(obj, 'name') and obj.name is not None: return obj.name
    return str(obj)


def _set_val(obj, attr_list, val):
    for attr in attr_list:
        try:
            setattr(obj, attr, val); return
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


# --- 核心逻辑 (必须与训练代码一致) ---
def setup_and_run_opf_teacher(grid, current_profile_sgen):
    # (复制你训练代码中的 setup_and_run_opf_teacher 函数逻辑)
    # 为了节省篇幅，这里简写，请确保逻辑与你训练代码完全一致
    for g in grid.generators:
        g_name = str(getattr(g, 'name', ''))
        if _is_slack_safe(g) or "Ext_Grid" in g_name:
            _set_val(g, ['Pmax', 'P_max'], 99999.0);
            _set_val(g, ['Pmin', 'P_min'], -99999.0)
            _set_val(g, ['cost_a', 'Cost1'], 1.0);
            _set_val(g, ['cost_b', 'Cost2'], 100.0)
            _set_val(g, ['is_controlled', 'controlled'], True)
        elif "sgen" in g_name:
            try:
                sgen_idx = int(g_name.split('_')[1]); p_avail = current_profile_sgen.get(sgen_idx, 0.0)
            except:
                p_avail = 0.0
            _set_val(g, ['Pmax', 'P_max'], p_avail);
            _set_val(g, ['Pmin', 'P_min'], 0.0)
            _set_val(g, ['cost_a', 'Cost1'], 0.01);
            _set_val(g, ['cost_b', 'Cost2'], 0.1)
            _set_val(g, ['is_controlled', 'controlled'], True)
            _set_val(g, ['Qmax', 'Q_max'], 0.0);
            _set_val(g, ['Qmin', 'Q_min'], 0.0)

    opf_opts = gce.OptimalPowerFlowOptions()
    if hasattr(gce, 'SolverType'): _set_val(opf_opts, ['solver', 'solver_type'], gce.SolverType.NONLINEAR_OPF)
    opf_opts.objective = 0
    _set_val(opf_opts, ['activate_voltage_limits', 'voltage_limits'], True)
    _set_val(opf_opts, ['vmin', 'Vmin'], 0.98);
    _set_val(opf_opts, ['vmax', 'Vmax'], 1.02)
    _set_val(opf_opts, ['activate_thermal_limits', 'thermal_limits'], True)
    _set_val(opf_opts, ['dispatch_P', 'control_active_power'], True)

    opf_driver = gce.OptimalPowerFlowDriver(grid, opf_opts)
    try:
        opf_driver.run()
    except:
        pass
    return opf_driver


def get_graph_data(grid, pf_results, bus_idx_map):
    # (复制你训练代码中的 get_graph_data 函数逻辑)
    # 必须保证完全一致，才能检测出问题
    num_nodes = len(grid.buses)
    x = np.zeros((num_nodes, 6), dtype=np.float32)

    for l in grid.loads:
        bus_ref = getattr(l, 'bus', getattr(l, 'node', None))
        if bus_ref:
            idx = bus_idx_map.get(get_gc_id(bus_ref))
            if idx is not None:
                p_val = _get_val(l, ['P', 'p_mw', 'p']);
                q_val = _get_val(l, ['Q', 'q_mvar', 'q'])
                x[idx, 0] += p_val * 3.0;
                x[idx, 1] += q_val * 3.0

    sgen_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for g in grid.generators:
        bus_ref = getattr(g, 'bus', getattr(g, 'node', None))
        if bus_ref:
            idx = bus_idx_map.get(get_gc_id(bus_ref))
            if idx is not None:
                x[idx, 4] = 1.0
                if "sgen" in getattr(g, 'name', ''):
                    p_max = _get_val(g, ['Pmax', 'P_max'], 0.0)
                    x[idx, 2] += p_max / 10.0;
                    sgen_mask[idx] = True

    v_mag_scaled = np.zeros(num_nodes);
    v_ang = np.zeros(num_nodes)
    if pf_results and hasattr(pf_results, 'voltage'):
        v_complex = np.array(pf_results.voltage, dtype=np.complex128)
        if len(v_complex) == num_nodes:
            v_abs = np.abs(v_complex);
            v_mag_scaled = (v_abs - 1.0) * 10.0;
            v_ang = np.angle(v_complex)
    x[:, 3] = v_mag_scaled;
    x[:, 5] = v_ang

    # Edge Features (简化处理，假设逻辑一致)
    src, dst, attr = [], [], []
    branches = list(getattr(grid, 'lines', [])) + list(getattr(grid, 'transformers', []))
    branch_loadings = pf_results.loading if (pf_results and hasattr(pf_results, 'loading')) else []

    for i, br in enumerate(branches):
        try:
            if _get_val(br, ['active', 'status'], 1.0) < 0.5: continue
            f_ref = getattr(br, 'bus_from', None);
            t_ref = getattr(br, 'bus_to', None)
            if f_ref and t_ref:
                u = bus_idx_map.get(get_gc_id(f_ref));
                v = bus_idx_map.get(get_gc_id(t_ref))
                if u is not None and v is not None:
                    r = float(_get_val(br, ['r', 'R']));
                    x_val = float(_get_val(br, ['x', 'X']))
                    rate = float(_get_val(br, ['rate', 'Rate'], 100.0))
                    loading_val = float(branch_loadings[i]) if i < len(branch_loadings) else 0.0
                    edge_feat = [r * 100.0, x_val * 100.0, rate / 1000.0, abs(loading_val)]
                    src.extend([u, v]);
                    dst.extend([v, u]);
                    attr.extend([edge_feat, edge_feat])
        except:
            continue

    x_tensor = torch.tensor(x, dtype=torch.float32)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attr, dtype=torch.float32)
    return Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr), sgen_mask


# ================= 检查脚本主程序 =================
def inspect_data():
    print(f"🔍 开始数据检查: {SB_CODE}")
    net_pp = sb.get_simbench_net(SB_CODE)
    grid = GC_PandaPowerImporter.PP2GC(net_pp)
    bus_idx_map = {get_gc_id(b): i for i, b in enumerate(grid.buses)}

    profiles = sb.get_absolute_values(net_pp, profiles_instead_of_study_cases=True)
    df_load_p = profiles[('load', 'p_mw')];
    df_load_q = profiles[('load', 'q_mvar')]
    df_sgen_p = profiles[('sgen', 'p_mw')]

    node_features_log = []
    targets_log = []

    print("-" * 60)
    print(f"Sampling {CHECK_SAMPLES} random scenarios...")

    for i in range(CHECK_SAMPLES):
        t = np.random.randint(0, len(df_load_p))
        stress_factor = np.random.uniform(6.0, 8.0)

        # 1. 设置环境
        current_load_p = df_load_p.iloc[t]
        current_load_q = df_load_q.iloc[t]
        for l in grid.loads:
            idx = int(l.name.split('_')[1])
            _set_val(l, ['P', 'p'], current_load_p.get(idx, 0.0))
            _set_val(l, ['Q', 'q'], current_load_q.get(idx, 0.0))

        sgen_p_dict = (df_sgen_p.iloc[t] * stress_factor).to_dict()
        grid_pf = deepcopy(grid)
        for g in grid_pf.generators:
            if "sgen" in getattr(g, 'name', ''):
                idx = int(g.name.split('_')[1])
                p_avail = sgen_p_dict.get(idx, 0.0)
                _set_val(g, ['Pmax', 'P_max'], p_avail);
                _set_val(g, ['P', 'p'], p_avail)

        # 2. 运行 Pre-PF (观测输入)
        pf_opts = gce.PowerFlowOptions(gce.SolverType.NR, verbose=False)
        pf_driver = gce.PowerFlowDriver(grid_pf, pf_opts)
        try:
            pf_driver.run(); current_pf_results = pf_driver.results
        except Exception as e:
            print(f"⚠️ Sample {i}: Pre-PF Failed! ({e})");
            current_pf_results = None

        # 3. 运行 OPF Teacher (动作/Target)
        teacher_driver = setup_and_run_opf_teacher(grid, sgen_p_dict)
        if not (teacher_driver and hasattr(teacher_driver.results, 'converged') and teacher_driver.results.converged):
            print(f"⚠️ Sample {i}: OPF Not Converged - Skipping")
            continue

        gen_p_vec = get_safe_gen_results(teacher_driver, grid)

        # 4. 获取数据
        data, mask = get_graph_data(grid, current_pf_results, bus_idx_map)

        # 记录特征统计
        feat = data.x.numpy()  # [Nodes, 6]
        # Features: 0:LoadP, 1:LoadQ, 2:GenP_max, 3:V_mag, 4:IsGen, 5:V_ang
        node_features_log.append(pd.DataFrame(feat, columns=["LoadP", "LoadQ", "GenP_Max", "V_Mag", "IsGen", "V_Ang"]))

        # 记录 Target
        for g_idx, g in enumerate(grid.generators):
            if "sgen" in getattr(g, 'name', ''):
                p_opt = float(gen_p_vec[g_idx])
                p_avail = _get_val(g, ['Pmax', 'P_max'])
                if p_avail > 0.001:
                    targets_log.append(np.clip(p_opt / p_avail, 0.0, 1.0))

    # ================= 结果分析与绘图 =================
    if not node_features_log:
        print("❌ 没有有效数据！检查你的引擎是否都在报错。")
        return

    all_feats = pd.concat(node_features_log)
    all_targets = np.array(targets_log)

    print("\n📊 --- 数据统计报告 ---")
    print(f"Valid Samples Collected: {len(node_features_log)}")
    print(f"Total Nodes Processed: {len(all_feats)}")
    print(f"Feature Statistics:\n{all_feats.describe().T[['min', 'max', 'mean', 'std']]}")

    # 🚨 关键检查点
    print("\n🚨 关键质量检查 (Red Flags Check):")

    # Check 1: 电压是否变化？
    v_std = all_feats['V_Mag'].std()
    if v_std < 1e-6:
        print(f"❌ [严重] 电压特征 (V_Mag) 方差极小 ({v_std})！Pre-PF 可能根本没跑或者没更新结果。GNN 无法感知电压越限。")
    else:
        print(f"✅ 电压特征正常变化 (std={v_std:.4f})。")

    # Check 2: 负载是否有输入？
    if all_feats['LoadP'].max() == 0:
        print("❌ [严重] 负载特征 (LoadP) 全为 0！检查 Profiles 加载或单位转换。")
    else:
        print(f"✅ 负载特征存在 (Max={all_feats['LoadP'].max():.2f})。")

    # Check 3: Target 分布
    zero_cnt = np.sum(all_targets < 0.01)
    one_cnt = np.sum(all_targets > 0.99)
    mid_cnt = len(all_targets) - zero_cnt - one_cnt
    print(f"\n🎯 Target 分布 (Total {len(all_targets)}):")
    print(f"   - 0.0 (Off/Min): {zero_cnt} ({zero_cnt / len(all_targets):.1%})")
    print(f"   - 1.0 (Max/Full): {one_cnt} ({one_cnt / len(all_targets):.1%})")
    print(f"   - Middle (Active Control): {mid_cnt} ({mid_cnt / len(all_targets):.1%})")

    if mid_cnt < 10:
        print(
            "⚠️ [警告] 绝大多数 Target 都是 0 或 1。这意味着场景要么太简单(满发)，要么太难(全关)。模型可能学不到精细控制。")

    # === 绘图 ===
    plt.figure(figsize=(15, 5))

    # 图1：特征箱线图 (检查数值范围)
    plt.subplot(1, 3, 1)
    sns.boxplot(data=all_feats)
    plt.title("Input Features Distribution")
    plt.yscale('symlog')  # 使用对数轴，因为不同特征尺度差异大
    plt.grid(True, alpha=0.3)

    # 图2：Target 直方图
    plt.subplot(1, 3, 2)
    sns.histplot(all_targets, bins=20, kde=False)
    plt.title("Target (Alpha) Distribution")
    plt.xlabel("Alpha [0, 1]")

    # 图3：电压与Target的相关性 (简单看一眼)
    # 我们只取那些是发电机的点的 V_Mag 和 对应的 Target
    # 这是一个粗略的对齐
    plt.subplot(1, 3, 3)
    plt.text(0.1, 0.5, "Check Console Log\nfor Detailed Stats", fontsize=12)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("data_inspection_report.png")
    print("\n🖼️  已保存检查图表: data_inspection_report.png")


if __name__ == "__main__":
    inspect_data()