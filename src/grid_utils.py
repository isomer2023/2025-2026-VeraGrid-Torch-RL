import numpy as np
import copy

import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en

def get_bus_by_name(grid_gc, bus_name: str | int):
    key = str(bus_name)
    for b in grid_gc.buses:
        if b.name == key:
            return b
    raise ValueError(f"Bus {bus_name} not found")


def get_generators_at_bus(grid_gc, bus_name: str | int):
    b = get_bus_by_name(grid_gc, bus_name)
    return [g for g in grid_gc.generators if g.bus is b]


def _pin_gen_power(generator, p_val: float):
    """
    把发电机硬钉在某个有功功率点：Pmin=Pmax=p_val
    （注意：Slack 不会用这个钉死，它自由补平。）
    """
    generator.Pmin = float(p_val)
    generator.Pmax = float(p_val) + 1e-10


def _run_high_RE_scenario_once(
    base_grid,
    th2_set: float,
    th3_set: float,
    pv_set: float,
    wt_set: float,
    load_stress_res: float,
):
    """
    构造单个“高可再生场景”并跑潮流：
      - 负荷按 load_stress_res 缩放 (例如 1.1 → 偏高负荷)
      - 火电机组 (Bus2, Bus3) 拉到高值 (th2_set, th3_set)
      - 风光机组 (Bus6=PV, Bus8=WT) 拉到高值 (pv_set, wt_set)
      - Slack (Bus1) 不钉，自动平衡（可以负功率吸收多余绿电）

    返回：
      Sf_RE: 每条线路的视在功率 |S| (MVA)，按潮流结果顺序
    """
    gtmp = copy.deepcopy(base_grid)

    # 1) 负荷缩放
    for ld in gtmp.loads:
        ld.P *= load_stress_res
        ld.Q *= load_stress_res

    # 2) 将主要发电单元钉到目标出力
    def _get_first_gen(ggrid, bus_no):
        gens = [g for g in ggrid.generators if str(g.bus.name) == str(bus_no)]
        return gens[0] if len(gens) > 0 else None

    g2 = _get_first_gen(gtmp, 2)   # 火电 @ Bus2
    g3 = _get_first_gen(gtmp, 3)   # 火电 @ Bus3
    g6 = _get_first_gen(gtmp, 6)   # PV    @ Bus6
    g8 = _get_first_gen(gtmp, 8)   # WT    @ Bus8

    if g2 is not None:
        _pin_gen_power(g2, th2_set)
    if g3 is not None:
        _pin_gen_power(g3, th3_set)
    if g6 is not None:
        _pin_gen_power(g6, pv_set)
    if g8 is not None:
        _pin_gen_power(g8, wt_set)

    # Slack 不 pin → 让它自然作为平衡机组

    pf_tmp = gce.PowerFlowDriver(
        grid=gtmp,
        options=gce.PowerFlowOptions(solver_type=en.SolverType.NR, verbose=False),
        opf_results=None
    )
    pf_tmp.run()

    bdf = pf_tmp.results.get_branch_df().copy()
    Sf_RE = np.hypot(
        bdf["Pf"].astype(float).to_numpy(),
        bdf["Qf"].astype(float).to_numpy()
    )  # 每条线的 |S| (MVA)

    return Sf_RE


def compute_static_branch_limits_high_RE(
    base_grid,
    th2_max=120.0,
    th3_max=80.0,
    pv_max_possible=36.0,     # ~ 40 * 0.90 (pv_range max)
    wt_max_possible=47.5,     # ~ 50 * 0.95 (wt_range max)
    load_stress_res=1.1,      # 稍高一点的负荷水平来定容量
    safety_factor=1.2,
    floor_MVA=5.0,
):
    """
    高可再生极端工况 → 固定线路容量。

    步骤：
      1. 按给定工况跑潮流：
         - 火电在高出力 (th2_max, th3_max)
         - 风光在高出力 (pv_max_possible, wt_max_possible)
         - 负荷在较高水平 (load_stress_res)
         - Slack 吸收/外送多余功率
      2. 对每条线取 |S| (MVA)
      3. 线路容量 = max(floor_MVA, safety_factor * |S|)

    返回：
      np.array([...])，长度=线路数，对应每条线的固定容量(MVA)。
    """
    Sf_RE = _run_high_RE_scenario_once(
        base_grid=base_grid,
        th2_set=th2_max,
        th3_set=th3_max,
        pv_set=pv_max_possible,
        wt_set=wt_max_possible,
        load_stress_res=load_stress_res,
    )

    rate_list = []
    for s_now in Sf_RE:
        limit_val = max(floor_MVA, safety_factor * float(s_now))
        rate_list.append(float(limit_val))

    return np.array(rate_list, dtype=float)