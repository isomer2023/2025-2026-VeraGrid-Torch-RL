# ===== Imports（和前面保持一致）=====
import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pandapower as pp
import pandapower.networks as nw

# VeraGrid / GridCal（Vera 分支）
import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en

# 这个是转换器：把 pandapower 的 net → VeraGrid 的 MultiCircuit
import GC_PandaPowerImporter as GC_PandaPowerImporter

# （可选）一个简易潮流执行函数，方便迅速验证
def ExecutePF(grid, show=False, solver=en.SolverType.NR):
    pf_options = gce.PowerFlowOptions(solver_type=solver, verbose=False)
    pf = gce.PowerFlowDriver(grid=grid, options=pf_options)
    pf.run()
    S_loss = pf.results.losses.sum()
    if show:
        print(f"Converged: {pf.results.converged} | error: {pf.results.error:.3e}")
        print(f"Losses: P={float(S_loss.real):.3f} MW, Q={float(S_loss.imag):.3f} MVAr")
    return pf

# ===== 接口函数：读取 case30 并转换为 VeraGrid =====
def load_case30_as_veragrid(run_pp_before=True, show_summary=True):
    """
    读取 pandapower 自带网络 case30，并转换为 VeraGrid 的 MultiCircuit。
    参数:
      - run_pp_before: 转换前是否先在 pandapower 侧跑一遍潮流（推荐 True）
      - show_summary : 是否打印两边网络的简要信息
    返回:
      - net_pp:  pandapower 的 net（case30）
      - grid_gc: VeraGrid 的 MultiCircuit（已转换）
    """
    # 1) 读取 pandapower 的示例网络 case30
    net_pp = nw.case14()

    # 2)（可选）在 pandapower 侧先跑一次潮流，检查可行性
    if run_pp_before:
        try:
            pp.runpp(net_pp, algorithm="nr")  # NR 潮流
        except Exception as e:
            print("pandapower runpp 失败：", e)

    # 3) 转换为 VeraGrid（GridCal Vera）对象
    grid_gc = GC_PandaPowerImporter.PP2GC(net_pp)

    # 4) 打印两边网络的简要信息（便于核对）
    if show_summary:
        buses_pp   = len(net_pp.bus)
        lines_pp   = len(net_pp.line)
        trafos_pp  = len(net_pp.trafo)
        gens_pp    = len(net_pp.gen)
        sgen_pp    = len(net_pp.sgen)
        loads_pp   = len(net_pp.load)
        print(f"[pandapower case30] buses={buses_pp}, lines={lines_pp}, trafos={trafos_pp}, gens={gens_pp}+sgen={sgen_pp}, loads={loads_pp}")

        print(f"[VeraGrid after convert] buses={len(grid_gc.buses)}, lines={len(grid_gc.lines)}, trafos={len(grid_gc.transformers2w)}, "
              f"gens={len(grid_gc.generators)}, loads={len(grid_gc.loads)}")

    return net_pp, grid_gc

# ===== 示例用法：读取→转换→在 VeraGrid 侧跑潮流 =====
if __name__ == "__main__":
    net_pp, grid_gc = load_case30_as_veragrid(run_pp_before=True, show_summary=True)

    # 在 VeraGrid 侧验证一次潮流
    pf = ExecutePF(grid_gc, show=True)

    # 需要 DataFrame 就拿：
    bus_df    = pf.results.get_bus_df()
    branch_df = pf.results.get_branch_df()
    print(bus_df.head())
    print(branch_df.head())
