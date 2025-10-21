# ===== Imports（尽量与前面一致）=====
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

# PP → VeraGrid 转换器
import GC_PandaPowerImporter as GC_PandaPowerImporter


# ===== 简易潮流执行（用于 OPF 解后校验）=====
def ExecutePF(grid, show=False, solver=en.SolverType.NR, opf_results=None):
    pf_options = gce.PowerFlowOptions(solver_type=solver, verbose=False)
    pf = gce.PowerFlowDriver(grid=grid, options=pf_options, opf_results=opf_results)
    pf.run()
    if show:
        S_loss = pf.results.losses.sum()
        print(f"Converged: {pf.results.converged} | error: {pf.results.error:.3e}")
        print(f"Losses: P={float(S_loss.real):.3f} MW, Q={float(S_loss.imag):.3f} MVAr")
    return pf


# ===== 读取 case14 → 在 PP 侧消除电压告警 → 转到 Vera =====
def load_case14_as_veragrid(run_pp_before=True, sanitize=True, show_summary=True):
    net_pp = nw.case14()

    # —— 在 pandapower 侧避免 “gen vm_pu > bus max_vm_pu” 的告警 —— #
    net_pp.bus["max_vm_pu"] = 1.06
    if "vm_pu" in net_pp.gen.columns:
        net_pp.gen["vm_pu"] = np.minimum(net_pp.gen["vm_pu"].values, 1.06)

    if run_pp_before:
        try:
            pp.runpp(net_pp, algorithm="nr")
        except Exception as e:
            print("pandapower runpp 失败：", e)

    # 转换到 VeraGrid（MultiCircuit）
    grid_gc = GC_PandaPowerImporter.PP2GC(net_pp)

    # —— （可选）统一做一些“卫生处理” —— #
    if sanitize:
        # 母线电压限值统一到 0.95~1.05
        for b in grid_gc.buses:
            b.Vmin = 0.95
            b.Vmax = 1.05
        # 机组 Vset 限幅到 [0.95, 1.03]
        for g in grid_gc.generators:
            g.Vset = max(0.95, min(1.03, g.Vset))

    # 给线路默认额定（没填时），以便后面计算负载率
    for ln in grid_gc.lines:
        if getattr(ln, "rate", None) is None or ln.rate <= 0:
            ln.rate = 100.0  # MVA

    if show_summary:
        print(f"[pandapower case14] buses={len(net_pp.bus)}, lines={len(net_pp.line)}, "
              f"trafos={len(net_pp.trafo)}, gens={len(net_pp.gen)}+sgen={len(net_pp.sgen)}, "
              f"loads={len(net_pp.load)}")
        print(f"[VeraGrid after convert] buses={len(grid_gc.buses)}, lines={len(grid_gc.lines)}, "
              f"trafos={len(grid_gc.transformers2w)}, gens={len(grid_gc.generators)}, "
              f"loads={len(grid_gc.loads)}")
        print("\n=== After sanitize ===")
        print("Buses (Vmin/Vmax) sample:")
        for b in grid_gc.buses[:10]:
            print(f"  Bus {int(b.name):3d}: Vmin={b.Vmin:.2f}, Vmax={b.Vmax:.2f}, slack={b.is_slack}")

        print("\nGenerators (Vset / Qlim / P lim / dispatch):")
        for i, g in enumerate(grid_gc.generators):
            print(f"  Gen#{i:02d} @Bus {int(g.bus.name):3d} | Vset={g.Vset:.2f} | "
                  f"Q[{g.Qmin:.1f},{g.Qmax:.1f}] MVAr | "
                  f"P[{g.Pmin:.1e},{g.Pmax:.10f}] MW | enabled_dispatch={g.enabled_dispatch}")

    return net_pp, grid_gc


def get_generators_at_bus(grid_gc, bus_name: int | str):
    key = str(bus_name)
    return [g for g in grid_gc.generators if g.bus.name == key]


# ===== 设置“风+光+火”并只跑 AC（非线性）OPF，然后用 PF 校验 =====
def apply_mix_and_run_ac_opf(grid_gc, show=True):
    # 宽一些的无功限值
    QMIN, QMAX = -80.0, 80.0

    def set_gen(g, Pmin, Pmax, vset, cost1, cost2=0.0, cost0=0.0, qmin=QMIN, qmax=QMAX, dispatch=True):
        g.Pmin = float(Pmin)
        g.Pmax = float(Pmax) + 1e-10
        g.Vset = float(vset)
        g.Cost  = float(cost1)   # 线性项
        g.Cost2 = float(cost2)   # 二次项
        g.Cost0 = float(cost0)   # 常数项
        g.Qmin = float(qmin)
        g.Qmax = float(qmax)
        g.enabled_dispatch = bool(dispatch)

    # 假设 case14 转换后机组在 1,2,3,6,8
    gens_bus1 = get_generators_at_bus(grid_gc, 1)   # 火电（主力）
    gens_bus2 = get_generators_at_bus(grid_gc, 2)   # 火电
    gens_bus3 = get_generators_at_bus(grid_gc, 3)   # 火电
    gens_bus6 = get_generators_at_bus(grid_gc, 6)   # 光伏，可弃
    gens_bus8 = get_generators_at_bus(grid_gc, 8)   # 风电，可弃

    # 线路额定值兜底
    for ln in grid_gc.lines:
        if getattr(ln, "rate", None) is None or ln.rate <= 0:
            ln.rate = 100.0

    # —— 配置成本与出力边界（示例）——
    if gens_bus1: set_gen(gens_bus1[0], Pmin=10.0, Pmax=200.0, vset=1.03, cost1=30.0, cost2=0.02)
    if gens_bus2: set_gen(gens_bus2[0], Pmin= 5.0, Pmax=120.0, vset=1.03, cost1=35.0, cost2=0.03)
    if gens_bus3: set_gen(gens_bus3[0], Pmin= 5.0, Pmax= 80.0, vset=1.01, cost1=40.0, cost2=0.04)
    if gens_bus6: set_gen(gens_bus6[0], Pmin= 0.0, Pmax= 40.0, vset=1.02, cost1= 2.0, cost2=0.00)  # PV
    if gens_bus8: set_gen(gens_bus8[0], Pmin= 0.0, Pmax= 50.0, vset=1.02, cost1= 1.0, cost2=0.00)  # Wind

    # —— AC（非线性）OPF —— #
    ac_opt = gce.OptimalPowerFlowOptions(
        solver=en.SolverType.NONLINEAR_OPF,
        power_flow_options=gce.PowerFlowOptions(
            solver_type=en.SolverType.NR,
            verbose=False
        )
    )
    ac_opf = gce.OptimalPowerFlowDriver(grid=grid_gc, options=ac_opt)
    ac_opf.run()

    # —— 用 OPF 结果做一次 NR 潮流验证 —— #
    pf = ExecutePF(grid_gc, show=False, solver=en.SolverType.NR, opf_results=ac_opf.results)

    if show:
        print("\n=== AC OPF → PF Summary ===")
        print(f"PF convergence: {pf.results.converged} | error: {pf.results.error:.3e}")
        S_loss = pf.results.losses.sum()
        print(f"Losses: P={float(S_loss.real):.3f} MW, Q={float(S_loss.imag):.3f} MVAr")

        # 机组出力（OPF 解）
        if hasattr(ac_opf.results, "generator_power"):
            gP = ac_opf.results.generator_power  # numpy array
            print("\n--- Generator dispatch (from AC-OPF) ---")
            for i, g in enumerate(grid_gc.generators):
                print(f"Gen#{i:02d} @Bus {g.bus.name:>2} | P* = {float(gP[i]):7.2f} MW | "
                      f"[Pmin={g.Pmin:5.1f}, Pmax={g.Pmax:6.1f}] | Cost1={g.Cost:5.2f}, Cost2={g.Cost2:5.3f}")

        # Bus / Branch 一览
        bus_df    = pf.results.get_bus_df()
        branch_df = pf.results.get_branch_df()

        # —— 计算并打印“负载率(%)”：使用 Line.rate —— #
        # 建立 name -> rate 的映射（branch_df 的 index 与 line.name 对齐）
        rate_map = {ln.name: float(getattr(ln, "rate", 100.0)) for ln in grid_gc.lines}
        # 以“from”侧视在功率计算负载率
        S_from = np.sqrt(branch_df.Pf.values**2 + branch_df.Qf.values**2)
        rates  = np.array([rate_map.get(idx, 100.0) for idx in branch_df.index])
        loading_pct = 100.0 * S_from / np.maximum(rates, 1e-6)
        branch_df = branch_df.copy()
        branch_df["rate_MVA"]    = rates
        branch_df["loading_pct"] = loading_pct

        print("\n=== Buses (Vm, Va, P, Q) (head) ===")
        print(bus_df.head(12))
        print("\n=== Branches with Loading% (head) ===")
        cols = ["Pf","Qf","Pt","Qt","rate_MVA","loading_pct","Ploss","Qloss"]
        print(branch_df[cols].head(12))
        print(f"\nVm range: [{bus_df.Vm.min():.3f}, {bus_df.Vm.max():.3f}] pu")

    return ac_opf, pf


# ===== 主流程 =====
if __name__ == "__main__":
    # 读取→转换→“消毒”
    net_pp, grid_gc = load_case14_as_veragrid(run_pp_before=True, sanitize=True, show_summary=True)
    # ★ 统一把所有线路额定值设为 100 MVA（OPF 就会按这个限制约束）
    for ln in grid_gc.lines:
        ln.rate = 120.0
        ln.monitor_loading = True  # 让 OPF/PF都按loading监视

    ac_opf, pf = apply_mix_and_run_ac_opf(grid_gc, show=True)
