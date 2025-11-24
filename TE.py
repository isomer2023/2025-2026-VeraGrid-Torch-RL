import torch
import numpy as np
import simbench as sb
import pandas as pd
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 导入依赖
try:
    import GC_PandaPowerImporter
    from VeraGridEngine import api as gce
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    exit()

# ================= 配置 =================
SB_CODE = "1-MV-urban--0-sw"
CHECK_SAMPLES = 5  # 抽查 5 个样本


# ========================================

# --- 搬运辅助函数 ---
def get_gc_id(obj):
    if hasattr(obj, 'id') and obj.id is not None: return obj.id
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


def get_graph_data_debug(grid, pf_results, bus_idx_map):
    """
    这是 GNN.py 里 get_graph_data 的调试版
    只为了提取数据，逻辑完全一致
    """
    num_nodes = len(grid.buses)
    # [P_load, Q_load, P_gen_max, V, Is_Gen]
    x = np.zeros((num_nodes, 5), dtype=np.float32)

    # Load
    for l in grid.loads:
        bus = getattr(l, 'bus', getattr(l, 'node', None))
        if bus:
            idx = bus_idx_map.get(get_gc_id(bus))
            if idx is not None:
                x[idx, 0] += _get_val(l, ['P', 'p_mw', 'p'])
                x[idx, 1] += _get_val(l, ['Q', 'q_mvar', 'q'])

    # Generator (Sgen Only for Pmax visualization)
    for g in grid.generators:
        bus = getattr(g, 'bus', getattr(g, 'node', None))
        if bus:
            idx = bus_idx_map.get(get_gc_id(bus))
            if idx is not None:
                x[idx, 4] = 1.0
                if "sgen" in getattr(g, 'name', ''):
                    x[idx, 2] += _get_val(g, ['Pmax', 'P_max'], 0.0)

    # Voltage (原始值，未归一化)
    if pf_results:
        v_vec = None
        if hasattr(pf_results, 'voltage_module'):
            v_vec = pf_results.voltage_module
        elif hasattr(pf_results, 'Vm'):
            v_vec = pf_results.Vm

        if v_vec is not None and len(v_vec) == num_nodes:
            x[:, 3] = np.array(v_vec)
        else:
            x[:, 3] = 1.0  # Fallback
            print("⚠️ 警告: 无法读取电压向量，使用了默认值 1.0")
    else:
        x[:, 3] = 1.0  # Fallback

    return x  # 返回 numpy 数组方便查看


def main():
    print(f"🔬 GNN 输入特征深度体检: {SB_CODE}")
    print("-" * 60)

    # 1. 初始化
    net_pp = sb.get_simbench_net(SB_CODE)
    grid = GC_PandaPowerImporter.PP2GC(net_pp)
    bus_idx_map = {get_gc_id(b): i for i, b in enumerate(grid.buses)}

    # 获取基准电压 (用于判断电压是否为标幺值)
    base_kv = grid.buses[0].vn_kv if hasattr(grid.buses[0], 'vn_kv') else 0.0
    print(f"ℹ️ 电网基准电压 (Base kV): {base_kv} kV")

    print("📦 加载 Profiles...")
    profiles = sb.get_absolute_values(net_pp, profiles_instead_of_study_cases=True)
    df_load_p = profiles[('load', 'p_mw')]
    df_load_q = profiles[('load', 'q_mvar')]
    df_sgen_p = profiles[('sgen', 'p_mw')]
    n_steps = len(df_load_p)

    print("\n🚀 开始抽查...")

    for i in range(CHECK_SAMPLES):
        t = np.random.randint(0, n_steps)
        stress = np.random.uniform(4.0, 8.0)

        print(f"\n[{i + 1}/{CHECK_SAMPLES}] Time={t}, Stress={stress:.2f}x")

        # 注入数据
        current_load_p = df_load_p.iloc[t]
        current_load_q = df_load_q.iloc[t]
        for l in grid.loads:
            try:
                idx = int(l.name.split('_')[1])
                _set_val(l, ['P', 'p_mw', 'p'], current_load_p.get(idx, 0.0))
                _set_val(l, ['Q', 'q_mvar', 'q'], current_load_q.get(idx, 0.0))
            except:
                pass

        current_sgen_p = df_sgen_p.iloc[t] * stress
        sgen_p_dict = current_sgen_p.to_dict()
        for g in grid.generators:
            if "sgen" in getattr(g, 'name', ''):
                try:
                    idx = int(g.name.split('_')[1])
                    val = sgen_p_dict.get(idx, 0.0)
                    _set_val(g, ['Pmax', 'P_max'], val)
                    _set_val(g, ['P', 'p'], val)  # 满发用于测电压
                except:
                    pass

        # 运行 Pre-PF 获取电压
        pf_converged = False
        try:
            pf_opts = gce.PowerFlowOptions(gce.SolverType.NR, verbose=False)
            pf_driver = gce.PowerFlowDriver(grid, pf_opts)
            pf_driver.run()
            if hasattr(pf_driver.results, 'converged'):
                pf_converged = pf_driver.results.converged
        except:
            pass

        print(f"   Pre-PF 收敛状态: {'✅' if pf_converged else '❌'}")

        # 获取原始特征 (Raw Features)
        raw_x = get_graph_data_debug(grid, pf_driver.results, bus_idx_map)

        # 模拟 GNN.py 里的归一化 (Normalized Features)
        # 你的代码逻辑: 功率 / 100.0, 电压不动
        norm_x = raw_x.copy()
        norm_x[:, 0:3] = norm_x[:, 0:3] / 5

        # --- 打印诊断报告 ---

        # 1. 功率特征 (Load & Gen)
        p_load = raw_x[:, 0]
        p_gen = raw_x[:, 2]

        print("   📊 功率特征 (P):")
        print(f"      Raw Load (MW):  Max={np.max(p_load):.4f}, Mean={np.mean(p_load):.4f}")
        print(f"      Raw Sgen (MW):  Max={np.max(p_gen):.4f},  Mean={np.mean(p_gen):.4f}")
        print(f"      Tensor Input:   Max={np.max(norm_x[:, 0:3]):.4f} (理想范围: 0.01 ~ 1.0)")

        if np.max(norm_x[:, 0:3]) < 0.001:
            print("      ⚠️ [警告] 功率输入太小！GNN 可能学不到东西。建议减少除数 (例如 /10.0)。")

        # 2. 电压特征 (Voltage)
        v_vals = raw_x[:, 3]
        print("   ⚡ 电压特征 (V):")
        print(f"      Raw Value:      Max={np.max(v_vals):.4f}, Min={np.min(v_vals):.4f}")

        # 致命检查：是 p.u. 还是 kV？
        if np.max(v_vals) > 1.5:
            print(f"      ❌ [致命错误] 电压是 kV 值 ({np.max(v_vals):.1f})，不是标幺值！")
            print("         GNN 会被这个大数值搞晕。")
            print(f"         建议: 在 get_graph_data 里除以基准电压 ({base_kv} kV)。")
        else:
            print("      ✅ 电压看起来是标幺值 (p.u.)，范围正常。")

        # 3. 供需比 (Supply/Demand)
        total_load = np.sum(p_load)
        total_gen = np.sum(p_gen)
        if total_load > 0:
            ratio = total_gen / total_load
            print(f"   ⚖️ 供需比 (S/L): {ratio:.2f} (如果 > 1.0 说明有过剩/倒送)")

        print("-" * 40)


if __name__ == "__main__":
    main()