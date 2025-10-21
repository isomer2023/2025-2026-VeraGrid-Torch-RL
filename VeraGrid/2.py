# ===== Imports（沿用你的风格）=====
import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pandas as pd
import pandapower as pp
import pandapower.networks as nw

# VeraGrid / GridCal（Vera 分支）
import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en

# 这个是转换器：把 pandapower 的 net → VeraGrid 的 MultiCircuit
import GC_PandaPowerImporter as GC_PandaPowerImporter


# ===== 一个简易潮流执行函数，方便迅速验证（沿用你的接口）=====
def ExecutePF(grid, show=False, solver=en.SolverType.NR):
    pf_options = gce.PowerFlowOptions(solver_type=solver, verbose=False)
    pf = gce.PowerFlowDriver(grid=grid, options=pf_options)
    pf.run()
    S_loss = pf.results.losses.sum()
    if show:
        print(f"Converged: {pf.results.converged} | error: {pf.results.error:.3e}")
        print(f"Losses: P={float(S_loss.real):.3f} MW, Q={float(S_loss.imag):.3f} MVAr")
    return pf


# ===== 工具：把 PP 的 case14 读入并先跑一次 PP 潮流，随后转换到 Vera =====
def load_case14_as_veragrid(run_pp_before=True, sanitize=True, show_summary=True):
    # 1) 读取 pandapower 的示例网络 case14
    net_pp = nw.case14()

    # 2)（可选）在 pandapower 侧先跑一次潮流，检查可行性
    if run_pp_before:
        # 为避免 vm_pu > bus.max_vm_pu 告警，先放宽母线上限
        net_pp.bus["max_vm_pu"] = 1.06
        try:
            pp.runpp(net_pp, algorithm="nr")  # NR 潮流
        except Exception as e:
            print("pandapower runpp 失败：", e)

    # 3) 转换为 VeraGrid（GridCal Vera）对象
    grid_gc = GC_PandaPowerImporter.PP2GC(net_pp)

    # 4)（可选）统一做一些“卫生处理”（电压上下限、机组 Vset 限幅等）
    if sanitize:
        # 母线统一 0.95~1.05
        for b in grid_gc.buses:
            b.Vmin = 0.95
            b.Vmax = 1.05
        # 机组 Vset 限幅
        for g in grid_gc.generators:
            if g.Vset > 1.03:
                g.Vset = 1.03
            if g.Vset < 0.95:
                g.Vset = 0.95

    for ln in grid_gc.lines:
        ln.rate = 110.0
    for ln in grid_gc.lines:
        # 简单规则：含有 9~14 的支路当末端
        if any(tag in ln.name for tag in ['_9_', '_10_', '_11_', '_12_', '_13_', '_14_']):
            ln.rate = 60.0

    if show_summary:
        buses_pp   = len(net_pp.bus)
        lines_pp   = len(net_pp.line)
        trafos_pp  = len(net_pp.trafo)
        gens_pp    = len(net_pp.gen)
        sgen_pp    = len(net_pp.sgen)
        loads_pp   = len(net_pp.load)
        print(f"[pandapower case14] buses={buses_pp}, lines={lines_pp}, trafos={trafos_pp}, gens={gens_pp}+sgen={sgen_pp}, loads={loads_pp}")

        print(f"[VeraGrid after convert] buses={len(grid_gc.buses)}, lines={len(grid_gc.lines)}, trafos={len(grid_gc.transformers2w)}, "
              f"gens={len(grid_gc.generators)}, loads={len(grid_gc.loads)}")

        # 简要展示 sanitization
        print("\n=== After sanitize ===")
        print("Buses (Vmin/Vmax) sample:")
        for i, b in enumerate(grid_gc.buses[:10]):
            print(f"  Bus {int(b.name):3d}: Vmin={b.Vmin:.2f}, Vmax={b.Vmax:.2f}, slack={b.is_slack}")

        print("\nGenerators (Vset / Qlim / P lim / dispatch):")
        for i, g in enumerate(grid_gc.generators):
            print(f"  Gen#{i:02d} @Bus {int(g.bus.name):3d} | Vset={g.Vset:.2f} | Q[{g.Qmin:.1f},{g.Qmax:.1f}] MVAr | "
                  f"P[{g.Pmin:.1e},{g.Pmax:.10f}] MW | enabled_dispatch={g.enabled_dispatch}")

    return net_pp, grid_gc


def get_bus_by_name(grid_gc, bus_name: str | int):
    key = str(bus_name)
    for b in grid_gc.buses:
        if b.name == key:
            return b
    raise ValueError(f"Bus {bus_name} not found")

def get_generators_at_bus(grid_gc, bus_name: str | int):
    b = get_bus_by_name(grid_gc, bus_name)
    return [g for g in grid_gc.generators if g.bus is b]


# ===== 按你的要求：把 14 节点改成“风+光+火”的推荐配置，并跑线性 OPF =====
def apply_recommended_mix_and_run_lin_opf(grid_gc, show=True):
    """
    推荐配置：
      - 火电：Bus 1、2、3（主力、成本较高，可设 Pmin>0）
      - 光伏：Bus 6（Pmin=0，可弃光，成本很低，Vset 可 1.02~1.03）
      - 风电：Bus 8（Pmin=0，可弃风，成本很低）
      - 线路额定值：统一给个合理值（比如 100 MVA）以便 loading 有参考意义
    """
    # —— 线路额定值：默认 PP→Vera 转进来很多是占位的，这里统一设置一下，让 loading 有意义 ——
    for ln in grid_gc.lines:
        ln.rate = 100.0  # MVA

    # —— 宽一些的无功限值（避免 Q 过紧）：±80 MVAr（示例） ——
    QMIN, QMAX = -80.0, 80.0

    # —— 定义成本（线性Cost=Cost1，二次Cost2，常数Cost0） ——
    #   风/光尽量优先出力，所以给低成本；火电成本更高
    def set_gen(g, Pmin, Pmax, vset, cost1, cost2=0.0, cost0=0.0, qmin=QMIN, qmax=QMAX, dispatch=True):
        g.Pmin = float(Pmin)
        g.Pmax = float(Pmax) + 1e-10  # 防止数值等号
        g.Vset = float(vset)
        g.Cost  = float(cost1)  # 线性项（Cost1）
        g.Cost2 = float(cost2)  # 二次项
        g.Cost0 = float(cost0)  # 常数项
        g.Qmin = float(qmin)
        g.Qmax = float(qmax)
        g.enabled_dispatch = bool(dispatch)

    # —— 找到各母线处的机组（case14 转换后通常有 5 台，分别在 1,2,3,6,8） ——
    gens_bus1 = get_generators_at_bus(grid_gc, 1)
    gens_bus2 = get_generators_at_bus(grid_gc, 2)
    gens_bus3 = get_generators_at_bus(grid_gc, 3)
    gens_bus6 = get_generators_at_bus(grid_gc, 6)
    gens_bus8 = get_generators_at_bus(grid_gc, 8)

    # 安全起见：若某处没有机组，则跳过
    # —— 火电：Bus1 主力，Bus2、Bus3 次之 ——
    if gens_bus1:
        set_gen(gens_bus1[0], Pmin=10.0, Pmax=200.0, vset=1.03, cost1=30.0, cost2=0.02)
    if gens_bus2:
        set_gen(gens_bus2[0], Pmin= 5.0, Pmax=120.0, vset=1.03, cost1=35.0, cost2=0.03)
    if gens_bus3:
        set_gen(gens_bus3[0], Pmin= 5.0, Pmax= 80.0, vset=1.01, cost1=40.0, cost2=0.04)

    # —— 光伏：Bus6，可弃光 ——
    if gens_bus6:
        set_gen(gens_bus6[0], Pmin=0.0, Pmax=40.0, vset=1.02, cost1=2.0, cost2=0.0)

    # —— 风电：Bus8，可弃风 ——
    if gens_bus8:
        set_gen(gens_bus8[0], Pmin=0.0, Pmax=50.0, vset=1.02, cost1=1.0, cost2=0.0)

    # —— 跑线性 OPF ——
    lin_opt = gce.OptimalPowerFlowOptions(
        solver=en.SolverType.LINEAR_OPF,                     # 线性 OPF
        power_flow_options=gce.PowerFlowOptions(             # PF 选项（用于可行性检查/嵌套）
            solver_type=en.SolverType.NR,
            verbose=False
        )
    )

    lin_opf = gce.OptimalPowerFlowDriver(grid=grid_gc, options=lin_opt)
    lin_opf.run()

    # —— 用 OPF 结果做一次潮流验证（把 OPF 的机组出力作为注入） ——
    pf = gce.PowerFlowDriver(
        grid=grid_gc,
        options=gce.PowerFlowOptions(solver_type=en.SolverType.NR, verbose=False),
        opf_results=lin_opf.results
    )
    pf.run()

    if show:
        print("\n=== Linear OPF → PF Summary ===")
        print(f"OPF solved. PF convergence: {pf.results.converged} | error: {pf.results.error:.3e}")
        S_loss = pf.results.losses.sum()
        print(f"Losses: P={float(S_loss.real):.3f} MW, Q={float(S_loss.imag):.3f} MVAr")

        # 机组出力（OPF 解）
        if hasattr(lin_opf.results, "generator_power"):
            gP = lin_opf.results.generator_power  # numpy array
            # 显示（机组 -> 母线 -> P）
            print("\n--- Generator dispatch (from OPF) ---")
            for i, g in enumerate(grid_gc.generators):
                print(f"Gen#{i:02d} @Bus {g.bus.name:>2} | P* = {float(gP[i]):7.2f} MW | "
                      f"[Pmin={g.Pmin:5.1f}, Pmax={g.Pmax:6.1f}] | Cost1={g.Cost:5.2f}, Cost2={g.Cost2:5.3f}")

        # 电压 / 线路一览
        bus_df    = pf.results.get_bus_df()
        branch_df = pf.results.get_branch_df()
        print("\n=== Buses (Vm, Va, P, Q) (head) ===")
        print(bus_df.head(12))
        print("\n=== Branches (Pf, Qf, Pt, Qt, loading, Ploss, Qloss) (head) ===")
        print(branch_df.head(12))
        print(f"\nVm range: [{bus_df.Vm.min():.3f}, {bus_df.Vm.max():.3f}] pu")

    return lin_opf, pf


# ====== 主流程 ======
if __name__ == "__main__":
    # 读取→转换→简单“消毒”
    net_pp, grid_gc = load_case14_as_veragrid(run_pp_before=True, sanitize=True, show_summary=True)

    # 套用“风+光+火”推荐配置，执行线性 OPF 并 PF 验证
    lin_opf, pf = apply_recommended_mix_and_run_lin_opf(grid_gc, show=True)
