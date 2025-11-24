import numpy as np
import simbench as sb
import pandas as pd
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

try:
    import GC_PandaPowerImporter
    from VeraGridEngine import api as gce
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    exit()

SB_CODE = "1-MV-urban--0-sw"


def _set_val(obj, attr_list, val):
    for attr in attr_list:
        try:
            setattr(obj, attr, val); return
        except:
            continue


def main():
    print(f"🔥 高压环境电压体检: {SB_CODE}")
    print("-" * 60)

    # 1. 初始化
    net_pp = sb.get_simbench_net(SB_CODE)
    grid = GC_PandaPowerImporter.PP2GC(net_pp)

    # 2. 加载数据
    print("📦 加载 Profiles...")
    profiles = sb.get_absolute_values(net_pp, profiles_instead_of_study_cases=True)
    df_load_p = profiles[('load', 'p_mw')]
    df_sgen_p = profiles[('sgen', 'p_mw')]
    n_steps = len(df_load_p)

    # =====================================================
    # 3. 制造高压场景 (Stress Injection)
    # =====================================================
    # 随机抽一个有阳光的中午时刻 (10:00 - 14:00)
    # 简单起见，我们找光伏出力最大的那一刻，看极限情况
    print("🔍 寻找光伏最强时刻...")
    sgen_sum = df_sgen_p.sum(axis=1)
    t = int(sgen_sum.idxmax())

    # 【关键】生成随机压力系数 (4.0 ~ 8.0)
    stress_factor = np.random.uniform(14.0, 18.0)

    print(f"⚡ 模拟场景: Time={t}, Stress={stress_factor:.2f}x (光伏翻倍)")

    # 注入负荷
    current_load_p = df_load_p.iloc[t]
    for l in grid.loads:
        try:
            idx = int(l.name.split('_')[1])
            # 负荷稍微轻一点 (0.8倍)，让电压更容易飘高
            _set_val(l, ['P', 'p_mw', 'p'], current_load_p.get(idx, 0.0) * 0.8)
            _set_val(l, ['Q', 'q_mvar', 'q'], 0.0)
        except:
            pass

    # 注入光伏 (打鸡血!)
    current_sgen_p = df_sgen_p.iloc[t] * stress_factor
    sgen_p_dict = current_sgen_p.to_dict()

    total_gen = 0.0
    for g in grid.generators:
        if "sgen" in getattr(g, 'name', ''):
            try:
                idx = int(g.name.split('_')[1])
                val = sgen_p_dict.get(idx, 0.0)
                # 设为满发，模拟不控制的状态
                _set_val(g, ['P', 'p'], val)
                _set_val(g, ['Pmax', 'P_max'], val)
                total_gen += val
            except:
                pass

    print(f"   📈 注入总光伏: {total_gen:.2f} MW")

    # 4. 运行潮流 (Pre-PF)
    print("🌊 运行潮流计算 (查看电压)...")
    pf_opts = gce.PowerFlowOptions(gce.SolverType.NR, verbose=False)
    pf_driver = gce.PowerFlowDriver(grid, pf_opts)

    try:
        pf_driver.run()
    except Exception as e:
        print(f"❌ 潮流计算失败: {e}")
        return

    if not pf_driver.results.converged:
        print("❌ 潮流未收敛！说明电压可能已经崩了 (太高导致不收敛)。")
        # 即使不收敛，我们也可以尝试打印最后一步的电压看看
    else:
        print("✅ 潮流收敛成功。")

    # 5. 提取电压
    v_complex = pf_driver.results.voltage
    v_mag = np.abs(v_complex)

    # 打印前 40 个节点
    print("\n📊 节点电压详情 (前40个 + 最值):")
    print(f"{'ID':<5} | {'Bus Name':<20} | {'Base kV':<8} | {'Voltage (p.u.)':<15} | {'Status'}")
    print("-" * 80)

    count = 0
    max_v = 0.0
    max_v_idx = -1

    for i, bus in enumerate(grid.buses):
        name = getattr(bus, 'name', f"Bus_{i}")

        # 获取基准电压
        base_kv = 0.0
        if hasattr(bus, 'Vnom'):
            base_kv = float(bus.Vnom)
        elif hasattr(bus, 'vn_kv'):
            base_kv = float(bus.vn_kv)
        elif hasattr(bus, 'nominal_voltage'):
            base_kv = float(bus.nominal_voltage)

        v_pu = v_mag[i]

        # 记录最大值
        if v_pu > max_v:
            max_v = v_pu
            max_v_idx = i

        # 状态标记
        status = "OK"
        if v_pu > 1.05: status = "⚠️ High"
        if v_pu > 1.10: status = "❌ Critical"

        # 只打印前40个
        if count < 150:
            print(f"{i:<5} | {name:<20} | {base_kv:<8.1f} | {v_pu:<15.4f} | {status}")
            count += 1

    print("-" * 80)
    print(f"🔥 全网最高电压: {max_v:.4f} p.u. (在节点 {max_v_idx})")

    if max_v > 1.05:
        print("✅ 验证成功：高压场景已复现！Teacher 肯定会削峰。")
    else:
        print("🤔 奇怪：即使加了 8 倍光伏，电压依然没超标？说明网架太强了。")


if __name__ == "__main__":
    main()