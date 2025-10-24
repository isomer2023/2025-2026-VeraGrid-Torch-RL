# ===== Grid-ACOPF DRL Env (case14, PV/Wind/Load randomized) =====
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandapower as pp
import pandapower.networks as nw

# VeraGrid / GridCal（Vera 分支）
import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en

# PP → VeraGrid
import GC_PandaPowerImporter as GC_PandaPowerImporter


# ---------- 基础工具 ----------
def _load_case14_as_veragrid(run_pp_before=True, sanitize=True, set_line_rate_100=True):
    """读取 pp.case14 → (可选)先在 PP 侧跑一遍 → 转 VeraGrid → 统一清洗"""
    net_pp = nw.case14()
    # 放宽 PP 侧电压上限，避免无关告警
    net_pp.bus["max_vm_pu"] = np.maximum(net_pp.bus.get("max_vm_pu", 1.05), 1.06)

    if run_pp_before:
        try:
            pp.runpp(net_pp, algorithm="nr")
        except Exception:
            pass  # 不抛异常，后续在 Vera 侧再检查

    grid_gc = GC_PandaPowerImporter.PP2GC(net_pp)

    if sanitize:
        for b in grid_gc.buses:
            b.Vmin, b.Vmax = 0.95, 1.05
        for g in grid_gc.generators:
            g.Vset = min(max(g.Vset, 0.95), 1.03)

    if set_line_rate_100:
        for ln in grid_gc.lines:
            ln.rate = 100.0

    return net_pp, grid_gc


def _get_bus(grid_gc, name: int | str):
    key = str(name)
    for b in grid_gc.buses:
        if b.name == key:
            return b
    raise ValueError(f"Bus {name} not found")


def _get_gens_at_bus(grid_gc, name: int | str):
    b = _get_bus(grid_gc, name)
    return [g for g in grid_gc.generators if g.bus is b]


def _run_ac_opf(grid_gc) -> gce.OptimalPowerFlowDriver:
    """运行 AC-OPF，返回 OPF 驱动对象（含结果）"""
    opt = gce.OptimalPowerFlowOptions(
        solver=en.SolverType.NONLINEAR_OPF,
        power_flow_options=gce.PowerFlowOptions(solver_type=en.SolverType.NR, verbose=False),
        ips_tolerance=1e-3,          # 适当放宽加快
        ips_iterations=60,           # 适度迭代数
        ips_init_with_pf=False       # 如要进一步提速，可改 True 并传入 acopf_v0/acopf_S0
    )
    opf = gce.OptimalPowerFlowDriver(grid=grid_gc, options=opt)
    opf.run()
    return opf


def _run_pf_with_current_grid(grid_gc):
    """按当前 grid 设置（含各机组 P）跑 NR 潮流，返回 PF 驱动对象"""
    pf = gce.PowerFlowDriver(grid=grid_gc, options=gce.PowerFlowOptions(solver_type=en.SolverType.NR, verbose=False))
    pf.run()
    return pf


# ---------- DRL 环境 ----------
class GridOPFEnv:
    """
    环境说明（Case14，5台机组）：
      - 动作向量 a = [P_t1, P_t2, P_t3, P_pv, P_wind] （单位 MW）
      - 状态向量 s = concat([all_loads_MW(by bus order), pv_avail_MW, wind_avail_MW])
      - 奖励 r = -lambda_dist * sum(|a - a*|) - lambda_pf_fail * 1[PF不收敛]
      - 每步都会随机：所有负荷、PV/Wind 可用容量（范围可配置）
    """

    def __init__(
        self,
        seed: int | None = None,
        # 负荷随机范围（按比例乘到原始 PP 负荷）：“温和系数”略放大
        load_factor_range=(0.85, 1.15),
        # PV/Wind 可用率范围（作用于它们的额定上限：PV=40MW, Wind=50MW）
        pv_avail_frac_range=(0.30, 0.90),
        wind_avail_frac_range=(0.20, 0.95),
        # 奖励系数（你可在外部构造时自定）
        lambda_dist: float = 1.0,
        lambda_pf_fail: float = 1000.0,
    ):
        self.rng = np.random.default_rng(seed)
        self.load_factor_range = load_factor_range
        self.pv_avail_frac_range = pv_avail_frac_range
        self.wind_avail_frac_range = wind_avail_frac_range
        self.lambda_dist = float(lambda_dist)
        self.lambda_pf_fail = float(lambda_pf_fail)

        # 初始建网 & 标注机组
        self.net_pp, self.grid = _load_case14_as_veragrid(run_pp_before=True, sanitize=True, set_line_rate_100=True)
        # 确认机组位置（case14→一般是 1,2,3,6,8）
        self.g_t1 = _get_gens_at_bus(self.grid, 1)[0]  # Thermal 1
        self.g_t2 = _get_gens_at_bus(self.grid, 2)[0]  # Thermal 2
        self.g_t3 = _get_gens_at_bus(self.grid, 3)[0]  # Thermal 3
        self.g_pv = _get_gens_at_bus(self.grid, 6)[0]  # PV
        self.g_wt = _get_gens_at_bus(self.grid, 8)[0]  # Wind

        # 固定火电额定上限（可改）
        self.t1_Pmax_nom, self.t2_Pmax_nom, self.t3_Pmax_nom = 200.0, 120.0, 80.0
        # PV/Wind 额定上限（随机可用率会乘到这里）
        self.pv_Pmax_nom, self.wt_Pmax_nom = 40.0, 50.0

        # 统一设置机组基础参数（成本不影响奖励，只用于 AC-OPF 解的“老师动作”）
        self._apply_base_gen_params()

        # 缓存：本步随机得到的可用容量（MW）
        self.pv_avail = None
        self.wt_avail = None

        # 记录：负荷的原始 PP 数据（per-bus MW），用于每步随机化
        self.base_loads_by_bus = self._extract_pp_base_loads_by_bus()

        # 初始化一帧状态
        self.state = None

    # ---- 内部：固定机组基础参数 ----
    def _apply_base_gen_params(self):
        QMIN, QMAX = -80.0, 80.0
        def set_gen(g, Pmin, Pmax, vset, cost1, cost2=0.0, cost0=0.0, qmin=QMIN, qmax=QMAX):
            g.Pmin = float(Pmin); g.Pmax = float(Pmax) + 1e-10
            g.Vset = float(vset); g.Cost = float(cost1); g.Cost2 = float(cost2); g.Cost0 = float(cost0)
            g.Qmin = float(qmin); g.Qmax = float(qmax); g.enabled_dispatch = True

        set_gen(self.g_t1, 10.0, self.t1_Pmax_nom, 1.03, 30.0, 0.02)
        set_gen(self.g_t2,  5.0, self.t2_Pmax_nom, 1.03, 35.0, 0.03)
        set_gen(self.g_t3,  5.0, self.t3_Pmax_nom, 1.01, 40.0, 0.04)
        set_gen(self.g_pv,  0.0, self.pv_Pmax_nom, 1.02,  2.0, 0.00)
        set_gen(self.g_wt,  0.0, self.wt_Pmax_nom, 1.02,  1.0, 0.00)

    # ---- 内部：抽取 PP 原始负荷（按 bus 索引 1..14 的顺序合成）----
    def _extract_pp_base_loads_by_bus(self):
        # pandapower 的 case14 中，load 表给出了各 bus 的 P、Q；我们取 P 做基准
        loads = np.zeros(len(self.net_pp.bus), dtype=float)  # 索引按 bus idx
        if len(self.net_pp.load) > 0:
            for _, row in self.net_pp.load.iterrows():
                b = int(row['bus'])
                loads[b] += float(row['p_mw'])
        # PP 的 bus index 与 Vera 的 bus.name 对应（1..N），这里保存成 1..N 顺序数组
        return loads  # shape (14,)

    # ---- 内部：把当前“动作 P”写入 VeraGrid 机组，然后跑 PF 可行性检查 ----
    def _apply_action_and_pf(self, action_vec):
        # 设置机组 P（MW）
        self.g_t1.P = float(action_vec[0])
        self.g_t2.P = float(action_vec[1])
        self.g_t3.P = float(action_vec[2])
        self.g_pv.P = float(action_vec[3])
        self.g_wt.P = float(action_vec[4])
        # 跑 NR 潮流
        return _run_pf_with_current_grid(self.grid)

    # ---- 内部：剪裁动作到 Pmin/Pmax（含 PV/Wind 的随机可用上限）----
    def _clip_action(self, a):
        lo = np.array([self.g_t1.Pmin, self.g_t2.Pmin, self.g_t3.Pmin, 0.0, 0.0], dtype=float)
        hi = np.array([
            self.g_t1.Pmax, self.g_t2.Pmax, self.g_t3.Pmax,
            self.pv_avail if self.pv_avail is not None else self.pv_Pmax_nom,
            self.wt_avail if self.wt_avail is not None else self.wt_Pmax_nom
        ], dtype=float)
        return np.clip(np.asarray(a, dtype=float), lo, hi)

    # ---- 内部：重置一步的随机扰动（负荷 & PV/Wind 可用容量）----
    def _randomize_loads_and_renewables(self):
        # 1) 负荷：按比例缩放（逐 bus）
        f_low, f_high = self.load_factor_range
        factors = self.rng.uniform(f_low, f_high, size=self.base_loads_by_bus.shape)
        scaled_loads = self.base_loads_by_bus * factors

        # 写回 Vera：按 bus.name 设置负荷（合并写有功即可；无功保持原模型即可）
        # 简易做法：按比例变动总负荷，使用 bus 的 loads 集合逐个改动 P
        # （Vera 的 load 对象在 grid_gc.loads 中，每个 load 关联到某个 bus）
        pp_bus_P = {i: float(p) for i, p in enumerate(scaled_loads)}  # i: PP bus idx
        # 清零再累加：先全部设为 0，再按 PP 的 bus idx → Vera bus.name 写
        for ld in self.grid.loads:
            ld.P = 0.0
        # 如果转换器将 PP 的多个负荷分配到 Vera 多个 load，我们按 bus 累加写回
        for ld in self.grid.loads:
            bus_idx = int(ld.bus.name)  # Vera bus name 即数字字符串
            # PP 的 bus 索引是从 0 开始，Vera 是 '1'..'14'，做个映射（常见 case14 正好 idx+1 = name）
            pp_idx_guess = bus_idx - 1
            if pp_idx_guess in pp_bus_P:
                # 简单地平均分摊到该 bus 的多个 load 上：这里直接累加（每个 load 都写同样值会放大）
                # 更稳：先统计该 bus 有几个 load，再均分
                pass
        # 均分写法：
        # 先统计每个 bus 的 load 数
        bus_load_count = {}
        for ld in self.grid.loads:
            b = int(ld.bus.name)
            bus_load_count[b] = bus_load_count.get(b, 0) + 1
        # 再分发
        for ld in self.grid.loads:
            b = int(ld.bus.name)
            pp_idx = b - 1
            totalP = pp_bus_P.get(pp_idx, 0.0)
            n = bus_load_count.get(b, 1)
            ld.P = float(totalP) / float(n)

        # 2) PV/Wind 可用容量（MW）
        pv_frac = self.rng.uniform(*self.pv_avail_frac_range)
        wt_frac = self.rng.uniform(*self.wind_avail_frac_range)
        self.pv_avail = pv_frac * self.pv_Pmax_nom
        self.wt_avail = wt_frac * self.wt_Pmax_nom

        # 同时把机组 Pmax 写成“当步可用上限”（便于 OPF 与动作裁剪一致）
        self.g_pv.Pmax = float(self.pv_avail) + 1e-10
        self.g_wt.Pmax = float(self.wt_avail) + 1e-10

    # ---- 内部：组装状态 ----
    def _make_state(self):
        # 取当前 VeraLoad 的 P 合成按 bus 顺序的 MW 序列
        nbus = len(self.grid.buses)
        loads_by_bus = np.zeros(nbus, dtype=float)
        count_by_bus = np.zeros(nbus, dtype=float)
        for ld in self.grid.loads:
            b = int(ld.bus.name) - 1
            loads_by_bus[b] += float(ld.P)
            count_by_bus[b] += 1.0
        # 状态： [loads_by_bus..., pv_avail, wt_avail]
        pvA = 0.0 if self.pv_avail is None else float(self.pv_avail)
        wtA = 0.0 if self.wt_avail is None else float(self.wt_avail)
        return np.concatenate([loads_by_bus, np.array([pvA, wtA], dtype=float)], axis=0)

    # ---- 外部接口：reset ----
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # 回到基础参数
        self._apply_base_gen_params()
        # 随机化负荷与可再生可用容量
        self._randomize_loads_and_renewables()
        # 生成状态
        self.state = self._make_state()
        return self.state.copy()

    # ---- 外部接口：获取“老师动作”（AC-OPF最优出力）----
    def get_optimal_action(self):
        opf = _run_ac_opf(self.grid)
        gP = np.array(opf.results.generator_power, dtype=float)  # 顺序与 self.grid.generators 一致
        # 映射到 [t1,t2,t3,pv,wt]
        idx_map = {
            self.g_t1: None, self.g_t2: None, self.g_t3: None, self.g_pv: None, self.g_wt: None
        }
        for i, g in enumerate(self.grid.generators):
            if g is self.g_t1: idx_map[self.g_t1] = i
            elif g is self.g_t2: idx_map[self.g_t2] = i
            elif g is self.g_t3: idx_map[self.g_t3] = i
            elif g is self.g_pv: idx_map[self.g_pv] = i
            elif g is self.g_wt: idx_map[self.g_wt] = i
        a_star = np.array([
            gP[idx_map[self.g_t1]],
            gP[idx_map[self.g_t2]],
            gP[idx_map[self.g_t3]],
            gP[idx_map[self.g_pv]],
            gP[idx_map[self.g_wt]],
        ], dtype=float)
        return a_star

    # ---- 外部接口：step ----
    def step(self, action):
        """
        输入：action = [t1, t2, t3, pv, wt] (MW)
        流程：
          1) 裁剪动作到 Pmin/Pmax（PV/Wind 用本步可用上限）
          2) 计算 AC-OPF 得到 a*（同一随机场景）
          3) 用动作写入机组 P，跑 PF：不收敛则给 PF 惩罚
          4) 计算 reward 并给出下一状态（这里是“单步”环境，下一步会再随机，因此可返回同一 state 或者调用 randomize）
        """
        # 1) 裁剪
        a = self._clip_action(action)

        # 2) AC-OPF → 老师动作
        a_star = self.get_optimal_action()

        # 3) PF 可行性检查（用你的动作）
        pf = self._apply_action_and_pf(a)
        pf_ok = bool(pf.results.converged)

        # 4) 奖励
        dist = np.abs(a - a_star).sum()
        reward = - self.lambda_dist * dist
        if not pf_ok:
            reward -= self.lambda_pf_fail

        # 5) info（可选：返回损耗等）
        info = {
            "a_star": a_star,
            "action_clipped": a,
            "pf_converged": pf_ok,
            "pf_error": float(getattr(pf.results, "error", 0.0)),
        }

        # 6) 单步环境：这里 done=True（也可换成多步，把下一步再随机）
        done = True
        next_state = self.state.copy()
        return next_state, float(reward), done, info


# ---------- 如果需要独立快速测试 ----------
if __name__ == "__main__":
    env = GridOPFEnv(
        seed=42,
        load_factor_range=(0.85, 1.15),
        pv_avail_frac_range=(0.30, 0.90),
        wind_avail_frac_range=(0.20, 0.95),
        lambda_dist=1.0,
        lambda_pf_fail=1000.0,
    )
    s = env.reset()
    # 随便来个动作（会被裁剪）
    a = np.array([100.0, 40.0, 10.0, 20.0, 20.0], dtype=float)
    s2, r, done, info = env.step(a)
    # 这里不打印，外部算法可自行处理 s2, r, done, info
