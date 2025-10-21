# ===== Imports =====
import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw

# VeraGrid / GridCal（Vera 分支）
import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en

# 把 pandapower 的 net → VeraGrid 的 MultiCircuit
import GC_PandaPowerImporter as GC_PandaPowerImporter


# ===== 简易潮流执行函数 =====
def ExecutePF(grid, show=False, solver=en.SolverType.NR):
    pf_options = gce.PowerFlowOptions(solver_type=solver, verbose=False)
    pf = gce.PowerFlowDriver(grid=grid, options=pf_options)
    pf.run()
    if show:
        S_loss = pf.results.losses.sum()
        print(f"Converged: {pf.results.converged} | error: {pf.results.error:.3e}")
        print(f"Losses: P={float(S_loss.real):.3f} MW, Q={float(S_loss.imag):.3f} MVAr")
    return pf


# ===== 工具：读取 case14 → 先在 PP 侧跑潮流 → 转换到 VeraGrid → 清洗/统一设置 =====
def load_case14_as_veragrid(run_pp_before=True, sanitize=True, show_summary=True, set_line_rate_100=True):
    net_pp = nw.case14()

    # 先把 PP 侧电压上限放宽一点，减少 vm_pu > max_vm_pu 警告
    net_pp.bus["max_vm_pu"] = np.maximum(net_pp.bus.get("max_vm_pu", 1.05), 1.06)

    if run_pp_before:
        try:
            pp.runpp(net_pp, algorithm="nr")
        except Exception as e:
            print("pandapower runpp 失败：", e)

    # 转 VeraGrid
    grid_gc = GC_PandaPowerImporter.PP2GC(net_pp)

    # 统一清洗
    if sanitize:
        # 统一母线电压上下限
        for b in grid_gc.buses:
            b.Vmin = 0.95
            b.Vmax = 1.05

        # 合理化机组 Vset（避免超过 Vmax）
        for g in grid_gc.generators:
            if g.Vset > 1.03:
                g.Vset = 1.03
            if g.Vset < 0.95:
                g.Vset = 0.95

    # 线路额定统一为 100 MVA（用于计算 loading%）
    if set_line_rate_100:
        for ln in grid_gc.lines:
            ln.rate = 100.0

    if show_summary:
        print(f"[pandapower case14] buses={len(net_pp.bus)}, lines={len(net_pp.line)}, "
              f"trafos={len(net_pp.trafo)}, gens={len(net_pp.gen)}+sgen={len(net_pp.sgen)}, loads={len(net_pp.load)}")
        print(f"[VeraGrid after convert] buses={len(grid_gc.buses)}, lines={len(grid_gc.lines)}, "
              f"trafos={len(grid_gc.transformers2w)}, gens={len(grid_gc.generators)}, loads={len(grid_gc.loads)}")

        print("\n=== After sanitize ===")
        print("Buses (Vmin/Vmax) sample:")
        for b in grid_gc.buses[:10]:
            print(f"  Bus {int(b.name):3d}: Vmin={b.Vmin:.2f}, Vmax={b.Vmax:.2f}, slack={b.is_slack}")

        print("\nGenerators (Vset / Qlim / P lim / dispatch):")
        for i, g in enumerate(grid_gc.generators):
            print(f"  Gen#{i:02d} @Bus {int(g.bus.name):3d} | Vset={g.Vset:.02f} | "
                  f"Q[{g.Qmin:.1f},{g.Qmax:.1f}] MVAr | P[{g.Pmin:.1e},{g.Pmax:.10f}] MW | "
                  f"enabled_dispatch={g.enabled_dispatch}")

    return net_pp, grid_gc


# ===== 小工具：按母线名找对象 / 找母线上的机组 =====
def get_bus_by_name(grid_gc, bus_name: str | int):
    key = str(bus_name)
    for b in grid_gc.buses:
        if b.name == key:
            return b
    raise ValueError(f"Bus {bus_name} not found")

def get_generators_at_bus(grid_gc, bus_name: str | int):
    b = get_bus_by_name(grid_gc, bus_name)
    return [g for g in grid_gc.generators if g.bus is b]


# ===== 推荐配置 + 随机化“风/光”可用容量 + AC-OPF + PF 校验 =====
def apply_recommended_mix_and_run_ac_opf(grid_gc, show=True,
                                         pv_range=(0.30, 0.90),  # 光伏可用率范围
                                         wt_range=(0.20, 0.95),  # 风电可用率范围
                                         seed=None):
    """
    - 火电：Bus 1、2、3（主力），设置较高成本与合适的上下限
    - 光伏：Bus 6（可弃光，Pmin=0），在给定范围内随机化 Pmax（可用容量）
    - 风电：Bus 8（可弃风，Pmin=0），在给定范围内随机化 Pmax（可用容量）
    - 线路额定统一 100 MVA（在 load_case14_as_veragrid 已设）
    """
    if seed is not None:
        np.random.seed(seed)

    QMIN, QMAX = -80.0, 80.0

    def set_gen(g, Pmin, Pmax, vset, cost1, cost2=0.0, cost0=0.0, qmin=QMIN, qmax=QMAX, dispatch=True):
        g.Pmin = float(Pmin)
        g.Pmax = float(Pmax) + 1e-10  # 防止数值等号
        g.Vset = float(vset)
        g.Cost  = float(cost1)  # 线性项
        g.Cost2 = float(cost2)  # 二次项
        g.Cost0 = float(cost0)  # 常数项
        g.Qmin = float(qmin)
        g.Qmax = float(qmax)
        g.enabled_dispatch = bool(dispatch)

    gens_bus1 = get_generators_at_bus(grid_gc, 1)  # 火
    gens_bus2 = get_generators_at_bus(grid_gc, 2)  # 火
    gens_bus3 = get_generators_at_bus(grid_gc, 3)  # 火
    gens_bus6 = get_generators_at_bus(grid_gc, 6)  # 光
    gens_bus8 = get_generators_at_bus(grid_gc, 8)  # 风

    # 火电（固定成本与上下限）
    if gens_bus1: set_gen(gens_bus1[0], Pmin=10.0, Pmax=200.0, vset=1.03, cost1=30.0, cost2=0.02)
    if gens_bus2: set_gen(gens_bus2[0], Pmin= 5.0, Pmax=120.0, vset=1.03, cost1=35.0, cost2=0.03)
    if gens_bus3: set_gen(gens_bus3[0], Pmin= 5.0, Pmax= 80.0, vset=1.01, cost1=40.0, cost2=0.04)

    # 光伏：随机可用容量
    if gens_bus6:
        af_pv = np.random.uniform(*pv_range)
        set_gen(gens_bus6[0], Pmin=0.0, Pmax=40.0 * af_pv, vset=1.02, cost1=2.0, cost2=0.0)

    # 风电：随机可用容量
    if gens_bus8:
        af_wt = np.random.uniform(*wt_range)
        set_gen(gens_bus8[0], Pmin=0.0, Pmax=50.0 * af_wt, vset=1.02, cost1=1.0, cost2=0.0)

    # —— AC-OPF —— #
    ac_opt = gce.OptimalPowerFlowOptions(
        solver=en.SolverType.NONLINEAR_OPF,
        mip_solver=en.MIPSolvers.HIGHS,  # 非 MIP 也可设置，不影响
        power_flow_options=gce.PowerFlowOptions(solver_type=en.SolverType.NR, verbose=False)
    )
    ac_opf = gce.OptimalPowerFlowDriver(grid=grid_gc, options=ac_opt)
    ac_opf.run()

    # —— 用 OPF 结果再跑 NR 潮流校验 —— #
    pf = gce.PowerFlowDriver(
        grid=grid_gc,
        options=gce.PowerFlowOptions(solver_type=en.SolverType.NR, verbose=False),
        opf_results=ac_opf.results
    )
    pf.run()

    if show:
        print("\n=== AC OPF → PF Summary (with randomized PV/Wind Pmax) ===")
        print(f"PF convergence: {pf.results.converged} | error: {pf.results.error:.3e}")
        S_loss = pf.results.losses.sum()
        print(f"Losses: P={float(S_loss.real):.3f} MW, Q={float(S_loss.imag):.3f} MVAr")

        # 打印 OPF 的机组出力
        if hasattr(ac_opf.results, "generator_power"):
            gP = ac_opf.results.generator_power
            print("\n--- Generator dispatch (from AC-OPF) ---")
            for i, g in enumerate(grid_gc.generators):
                print(f"Gen#{i:02d} @Bus {g.bus.name:>2} | P* = {float(gP[i]):7.2f} MW | "
                      f"[Pmin={g.Pmin:5.1f}, Pmax={g.Pmax:6.1f}] | Cost1={g.Cost:5.2f}, Cost2={g.Cost2:5.3f}")

        # 电压与线路（含负载率%）
        bus_df    = pf.results.get_bus_df()
        branch_df = pf.results.get_branch_df().copy()

        # 计算负载率%
        def row_loading_pct(row):
            s = np.hypot(row['Pf'], row['Qf'])          # |S| = sqrt(P^2 + Q^2)
            rate = row.get('rate_MVA', np.nan)
            if pd.isna(rate) or rate == 0:
                rate = 100.0
            return 100.0 * s / rate

        # 如果结果表里没有 rate_MVA 列，补一列（按我们统一设置的 100.0）
        if 'rate_MVA' not in branch_df.columns:
            branch_df['rate_MVA'] = 100.0

        branch_df['loading_pct'] = branch_df.apply(row_loading_pct, axis=1)

        print("\n=== Buses (Vm, Va, P, Q) (head) ===")
        print(bus_df.head(12))

        print("\n=== Branches with Loading% (head) ===")
        cols = [c for c in ['Pf','Qf','Pt','Qt','rate_MVA','loading_pct','Ploss','Qloss'] if c in branch_df.columns]
        print(branch_df[cols].head(12))

        print(f"\nVm range: [{bus_df.Vm.min():.3f}, {bus_df.Vm.max():.3f}] pu")

    return ac_opf, pf


# ===== 主流程 =====
if __name__ == "__main__":
    # 1) 读取→转换→清洗（含线路额定=100 MVA）
    net_pp, grid_gc = load_case14_as_veragrid(run_pp_before=True, sanitize=True,
                                              show_summary=True, set_line_rate_100=True)



    ac_opf, pf = apply_recommended_mix_and_run_ac_opf(
        grid_gc,
        show=True,
        pv_range=(0.30, 0.90),   # 光伏可用率范围
        wt_range=(0.20, 0.95),   # 风电可用率范围
        seed=None                # 设为整数即可复现结果，如 42
    )
