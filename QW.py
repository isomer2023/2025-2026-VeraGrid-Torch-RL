import numpy as np
import simbench as sb
import pandas as pd
import warnings
from copy import deepcopy

# 忽略警告
warnings.filterwarnings('ignore')

try:
    import GC_PandaPowerImporter
    from VeraGridEngine import api as gce
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    exit()

# ================= 配置 =================
SB_CODE = "1-MV-urban--0-sw"
CHECK_SAMPLES = 5  # 检查 5 次
STRESS_FACTOR = 8.0  # 8倍光伏 (极限施压)


# ========================================

def _set_val(obj, attr_list, val):
    for attr in attr_list:
        try:
            setattr(obj, attr, val); return
        except:
            continue


def _get_val(obj, attr_list):
    for attr in attr_list:
        if hasattr(obj, attr): return float(getattr(obj, attr))
    return 0.0


def main():
    print(f"\n{'=' * 80}")
    print(f"📏 归一化逻辑专项检查: {SB_CODE}")
    print(f"{'=' * 80}")

    # 1. 初始化
    net_pp = sb.get_simbench_net(SB_CODE)
    grid = GC_PandaPowerImporter.PP2GC(net_pp)

    print("📦 加载 SimBench Profiles...")
    profiles = sb.get_absolute_values(net_pp, profiles_instead_of_study_cases=True)
    df_load_p = profiles[('load', 'p_mw')]
    df_sgen_p = profiles[('sgen', 'p_mw')]

    # 强制选中午 (光伏最大时刻)
    t_noon = int(df_sgen_p.sum(axis=1).idxmax())

    print(f"\n🚀 锁定时刻: t={t_noon}, 施加压力: {STRESS_FACTOR}x (光伏翻倍)")
    print(f"{'-' * 90}")
    print(f"{'Type':<10} | {'Raw Max (物理值)':<20} | {'Formula (公式)':<20} | {'Norm Max (GNN输入)':<20} | {'评价'}")
    print(f"{'-' * 90}")

    # --- 1. 注入数据 ---
    current_load_p = df_load_p.iloc[t_noon]

    # 注入负荷
    max_load_raw = 0.0
    for l in grid.loads:
        try:
            idx = int(l.name.split('_')[1])
            val = current_load_p.get(idx, 0.0)
            _set_val(l, ['P', 'p_mw', 'p'], val)
            if val > max_load_raw: max_load_raw = val
        except:
            pass

    # 注入光伏
    current_sgen_p = df_sgen_p.iloc[t_noon] * STRESS_FACTOR
    sgen_dict = current_sgen_p.to_dict()

    max_sgen_raw = 0.0
    for g in grid.generators:
        if "sgen" in getattr(g, 'name', ''):
            try:
                idx = int(g.name.split('_')[1])
                val = sgen_dict.get(idx, 0.0)
                _set_val(g, ['Pmax', 'P_max'], val)
                _set_val(g, ['P', 'p'], val)
                if val > max_sgen_raw: max_sgen_raw = val
            except:
                pass

    # --- 2. 运行 Pre-PF 获取电压 ---
    max_v_raw = 1.0
    try:
        pf_opts = gce.PowerFlowOptions(gce.SolverType.NR, verbose=False)
        pf_driver = gce.PowerFlowDriver(grid, pf_opts)
        pf_driver.run()
        if hasattr(pf_driver.results, 'voltage'):
            v_vec = np.abs(pf_driver.results.voltage)
            max_v_raw = np.max(v_vec)
    except:
        pass

    # =================================================
    # 3. 验证归一化 (这里模拟 GNN.py 里的逻辑)
    # =================================================

    # --- A. 负荷 (P_Load) ---
    # 公式: x / 10.0
    norm_load = max_load_raw / 10.0

    status_load = "✅ 完美"
    if norm_load < 0.01: status_load = "⚠️ 太小 (建议 /1.0)"
    if norm_load > 2.0:  status_load = "⚠️ 太大 (建议 /100.0)"

    print(f"{'Load P':<10} | {max_load_raw:<20.4f} | {'/ 10.0':<20} | {norm_load:<20.4f} | {status_load}")

    # --- B. 光伏 (P_Gen) ---
    # 公式: x / 10.0
    norm_sgen = max_sgen_raw / 10.0

    status_sgen = "✅ 完美"
    if norm_sgen > 5.0: status_sgen = "⚠️ 有点大 (考虑 /20.0)"

    print(f"{'Sgen P':<10} | {max_sgen_raw:<20.4f} | {'/ 10.0':<20} | {norm_sgen:<20.4f} | {status_sgen}")

    # --- C. 电压 (Voltage) ---
    # 之前的逻辑是没处理，现在建议用: (V - 1.0) * 10.0
    # 这样 1.05 -> 0.5, 0.95 -> -0.5
    norm_v = (max_v_raw - 1.0) * 10.0

    status_v = "✅ 完美"
    if abs(norm_v) > 5.0: status_v = "❌ 炸了 (这是kV值?)"
    if abs(norm_v) == 0.0: status_v = "⚠️ 无波动 (没压力?)"

    print(f"{'Voltage':<10} | {max_v_raw:<20.4f} | {'(V - 1.0) * 10':<20} | {norm_v:<20.4f} | {status_v}")
    print("-" * 90)

    # --- 4. 最终建议 ---
    print("\n💡 修改建议 (基于当前数据):")
    if max_load_raw < 1.0:
        print(f"   👉 负荷很小 (<1MW)，建议把归一化改成: data.x[:, 0] = data.x[:, 0] / 1.0 (不除)")

    if max_v_raw > 1.5:
        print(f"   👉 电压是 kV 值！必须除以基准电压 (例如 20kV) 再减 1.0！")
    else:
        print(f"   👉 电压是 p.u. 值。建议使用公式: (V - 1.0) * 10.0 来放大偏差特征。")


if __name__ == "__main__":
    main()