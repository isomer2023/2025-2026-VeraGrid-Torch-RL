# ============================
# network_env.py  —— PF 奖励版（Bus1 为 Slack，动作维度=4）
# 线路容量 = 高可再生极端场景下的潮流 * 安全系数（固定不变）
# 观测 = 随机化后各母线负荷 + 当期PV/WT上限（不依赖潮流结果）
# ============================
from __future__ import annotations
import numpy as np
import copy

# VeraGrid / GridCal（Vera 分支）
import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en

# Pandapower → VeraGrid 转换
import GC_PandaPowerImporter as GC_PandaPowerImporter


# ---------------------------
# 工具函数
# ---------------------------

def load_case14_as_veragrid(run_pp_before=True, sanitize=True):
    """
    加载 IEEE-14（pandapower），做基础清洗，然后转换为 VeraGrid MultiCircuit。
    不在这里设置线路限额；线路额定容量稍后由高可再生产景计算并固定。
    """
    import pandapower as pp
    import pandapower.networks as nw
    import numpy as np
    import GC_PandaPowerImporter as GC_PandaPowerImporter

    # 1. 载入 pandapower 自带的 case14
    net_pp = nw.case14()

    # 2. 放宽电压上/下限 & 限制发电机电压指令，避免初始状态太紧
    net_pp.bus["max_vm_pu"] = np.maximum(net_pp.bus.get("max_vm_pu", 1.05), 1.07)
    if "vm_pu" in net_pp.gen.columns:
        net_pp.gen["vm_pu"] = np.minimum(net_pp.gen["vm_pu"].fillna(1.03), 1.03)

    # 3. 可选：先在 pandapower 里跑一次潮流，保证有可行初始点
    if run_pp_before:
        try:
            pp.runpp(net_pp, algorithm="nr", init="flat")
        except Exception as e:
            print("pandapower runpp 失败：", e)

    # 4. 转成 VeraGrid MultiCircuit
    grid_gc = GC_PandaPowerImporter.PP2GC(net_pp)

    # 5. 清理：统一电压上下限、发电机电压设定值
    if sanitize:
        for b in grid_gc.buses:
            b.Vmin, b.Vmax = 0.95, max(getattr(b, "Vmax", 1.05), 1.05)
        for g in grid_gc.generators:
            g.Vset = min(max(getattr(g, "Vset", 1.01), 0.95), 1.03)

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


# ---------------------------
# 环境类
# ---------------------------

class GridOPFEnv:

    RECOMMENDED_ACT_LIMIT = 200.0

    def __init__(self,
                 pv_range=(0.30, 0.90),
                 wt_range=(0.20, 0.95),
                 load_jitter=0.15,
                 lambda_loss=1.0,
                 lambda_overload=5000.0,
                 lambda_diverge=1e4,
                 seed=None):
        # 载入基础网
        self.base_net, self.base_grid = load_case14_as_veragrid(
            run_pp_before=True,
            sanitize=True
        )
        self.rng = np.random.default_rng(seed)

        # 哪些母线对应哪些机组
        self.slack_bus = 1
        self.thermal_buses = [2, 3]  # 火电机组所在母线
        self.pv_bus = 6              # PV母线
        self.wt_bus = 8              # WT母线

        # 火电机组出力范围 (MW)
        self.th_limits = [
            (5.0, 120.0),   # Bus2
            (5.0, 80.0),    # Bus3
        ]

        # slack 发电机的估计范围 (仅用于成本核算，不是硬钉)
        self.slack_limits = (10.0, 200.0)

        # PV/WT 随机可用范围
        self.pv_range = pv_range
        self.wt_range = wt_range

        # 负荷扰动强度：N(1, load_jitter) 后截断到 [0.8,1.1]
        self.load_jitter = float(load_jitter)

        # reward 系数
        self.lambda_loss = float(lambda_loss)
        self.lambda_overload = float(lambda_overload)
        self.lambda_diverge = float(lambda_diverge)

        # 观测、动作维度
        # state = [bus1_load,...,bus14_load, pv_cap, wt_cap]
        self.action_dim = 4
        self.state_dim = 14 + 2

        # 运行时变量
        self.grid = None
        self.pv_pmax_rand = None
        self.wt_pmax_rand = None
        self.load_scale = None
        self.last_pf = None  # 只在 step() 后reward用

        # === 固定线路热限 (单一高可再生产景推出来) ===
        pv_max_possible = 40.0 * max(self.pv_range)   # e.g. 40 * 0.90 = 36 MW
        wt_max_possible = 50.0 * max(self.wt_range)   # e.g. 50 * 0.95 = 47.5 MW

        self.static_line_limits = compute_static_branch_limits_high_RE(
            base_grid=self.base_grid,
            th2_max=self.th_limits[0][1],        # 120.0
            th3_max=self.th_limits[1][1],        # 80.0
            pv_max_possible=pv_max_possible,     # ~36 MW
            wt_max_possible=wt_max_possible,     # ~47.5 MW
            load_stress_res=1.1,                 # 用稍高负荷去定额定容量
            safety_factor=1.2,                   # 留20%裕度
            floor_MVA=5.0,
        )

    def _set_generator_costs(self):
        """
        为当前 self.grid 写入机组的成本系数、电压设定等。
        Slack 给范围仅用于成本计算。
        """
        # Slack（Bus1）
        g1 = get_generators_at_bus(self.grid, self.slack_bus)[0]
        g1.Cost, g1.Cost2, g1.Vset = 60.0, 0.0, 1.03
        g1.Pmin, g1.Pmax = self.slack_limits[0], self.slack_limits[1] + 1e-6

        # 火电机组：Bus2 / Bus3
        g2 = get_generators_at_bus(self.grid, self.thermal_buses[0])[0]
        g3 = get_generators_at_bus(self.grid, self.thermal_buses[1])[0]
        g2.Cost, g2.Cost2, g2.Vset = 35.0, 0.03, 1.03
        g3.Cost, g3.Cost2, g3.Vset = 40.0, 0.04, 1.01

        # 所有母线的电压上限再保险一点
        for b in self.grid.buses:
            b.Vmax = max(getattr(b, "Vmax", 1.05), 1.05)

    def _randomize_pv_wt(self):
        """
        随机决定本 episode 的 PV/WT 最大可发功率上限 (Pmax)。
        """
        self.pv_pmax_rand = 40.0 * self.rng.uniform(*self.pv_range)
        self.wt_pmax_rand = 50.0 * self.rng.uniform(*self.wt_range)

        g6 = get_generators_at_bus(self.grid, self.pv_bus)[0]
        g8 = get_generators_at_bus(self.grid, self.wt_bus)[0]

        g6.Pmin, g6.Pmax = 0.0, self.pv_pmax_rand + 1e-10
        g8.Pmin, g8.Pmax = 0.0, self.wt_pmax_rand + 1e-10
        g6.Cost, g6.Cost2 = 2.0, 0.0
        g8.Cost, g8.Cost2 = 1.0, 0.0
        g6.Vset, g8.Vset = 1.02, 1.02

    def _rewrite_pv_wt_limits_only(self):
        """
        每个 step 后恢复 PV/WT 的 Pmin/Pmax，避免潮流求解器给它们乱改。
        """
        g6 = get_generators_at_bus(self.grid, self.pv_bus)[0]
        g8 = get_generators_at_bus(self.grid, self.wt_bus)[0]
        g6.Pmin, g6.Pmax = 0.0, float(self.pv_pmax_rand) + 1e-10
        g8.Pmin, g8.Pmax = 0.0, float(self.wt_pmax_rand) + 1e-10

    def _randomize_loads(self):
        """
        给所有负荷乘一个随机系数 (clip到[0.8,1.1])，模拟系统高/低负荷时段。
        """
        self.load_scale = float(self.rng.normal(1.0, self.load_jitter))
        # clip：你说的“抽0.8-1.1就行”
        self.load_scale = max(0.8, min(1.1, self.load_scale))

        for ld in self.grid.loads:
            ld.P *= self.load_scale
            ld.Q *= self.load_scale

    def _apply_action_to_generators(self, action_vec):
        """
        Agent 动作: [Bus2火电出力, Bus3火电出力, PV出力, WT出力]
        我们把这四台机组硬钉到该出力：Pmin=Pmax=该值。
        Slack 自动补平系统功率差。
        """
        th2, th3, pv, wt = map(float, action_vec)

        def clamp(x, lo, hi):
            return float(max(lo, min(hi, x)))

        g2 = get_generators_at_bus(self.grid, self.thermal_buses[0])[0]  # Bus2
        g3 = get_generators_at_bus(self.grid, self.thermal_buses[1])[0]  # Bus3
        g6 = get_generators_at_bus(self.grid, self.pv_bus)[0]            # PV
        g8 = get_generators_at_bus(self.grid, self.wt_bus)[0]            # WT

        (lo2, hi2), (lo3, hi3) = self.th_limits
        th2 = clamp(th2, lo2, hi2)
        th3 = clamp(th3, lo3, hi3)
        pv  = clamp(pv,  0.0, self.pv_pmax_rand)
        wt  = clamp(wt,  0.0, self.wt_pmax_rand)

        _pin_gen_power(g2, th2)
        _pin_gen_power(g3, th3)
        _pin_gen_power(g6, pv)
        _pin_gen_power(g8, wt)

        return np.array([th2, th3, pv, wt], dtype=float)

    # ---------- 潮流 ----------
    def _run_pf(self):
        pf = gce.PowerFlowDriver(
            grid=self.grid,
            options=gce.PowerFlowOptions(solver_type=en.SolverType.NR, verbose=False),
            opf_results=None
        )
        pf.run()
        self.last_pf = pf
        return pf

    def _build_obs(self):
        """
        观测向量 (state):
          - 随机化后的母线有功负荷 (MW, 正数)，按 bus1..bus14 顺序；
            没负荷的母线给 0。
          - 当前 episode 的 [pv_pmax_rand, wt_pmax_rand]

        注意：这里不跑潮流，不看 pf.results。观测是“需求场景”，
        action 决定发电，潮流只在 step() 里用于打分。
        """

        # 1. 按母线聚合当前 self.grid 里的负荷 (这些负荷已经被 _randomize_loads() 缩放过了)
        demand_map = {}
        for ld in self.grid.loads:
            bus_name = str(ld.bus.name)
            P_load = float(getattr(ld, "P", 0.0))
            demand_map[bus_name] = demand_map.get(bus_name, 0.0) + P_load

        # 2. 生成长度14的向量 [bus1,...,bus14]，无负荷母线=0
        demand_vec = []
        for bus_idx in range(1, 15):  # 1..14
            val = demand_map.get(str(bus_idx), 0.0)
            demand_vec.append(val)

        demand_vec = np.array(demand_vec, dtype=float)

        # 3. 拼上当期风光最大可发
        obs = np.concatenate([
            demand_vec,
            [self.pv_pmax_rand, self.wt_pmax_rand],
        ]).astype(np.float32)

        return obs

    def reset(self, seed=None):
        """
        新 episode：
          1. 拷贝 base_grid
          2. 设置成本/电压上限
          3. 随机 PV/WT 上限
          4. 随机负荷 (全网统一scale)
          5. 把固定线路上限写入 ln.rate
          6. 不需要先跑潮流来生成观测
          7. 返回观测
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 1. 深拷贝基础网
        self.grid = copy.deepcopy(self.base_grid)

        # 2. 写成本等
        self._set_generator_costs()

        # 3. 本episode的PV/WT装机上限
        self._randomize_pv_wt()

        # 4. 随机负荷
        self._randomize_loads()

        # 5. 写死线路容量（高RE极端工况预先算好的 self.static_line_limits）
        for ln, cap in zip(self.grid.lines, self.static_line_limits):
            ln.rate = float(cap)

        # 6. 这里可以不跑潮流来生成obs，
        #    但我们可以提前算一次潮流放在 self.last_pf 里，方便 debug/step()
        self._run_pf()

        # debug 打印机组信息（可注释掉）
        self.debug_print_generators()

        # 7. 返回观察（基于随机负荷+风光cap，不依赖潮流结果）
        return self._build_obs()

    def step(self, action: np.ndarray):
        """
        单步任务：
          - 用 action 钉住4台机组的出力
          - Slack 自动补
          - 跑潮流
          - 根据成本 + 网损 + 线路过载 + 潮流可行性 给 reward
          - done=True
        """
        action = np.asarray(action, dtype=float).copy()
        assert action.shape[0] == 4, "action 维度应为 4（Bus2/3 + PV + WT）"

        # 1. 应用动作到机组（除了 slack）
        act_used = self._apply_action_to_generators(action)

        # 2. 跑潮流
        pf = self._run_pf()
        converged = bool(pf.results.converged)

        info = {"action": act_used.tolist()}

        if not converged:
            # 潮流没收敛，直接巨大罚分
            reward = - self.lambda_diverge
            obs = self._build_obs()
            info.update({
                "converged": False,
                "C_gen": None, "C_loss": None, "C_ov": None,
                "total_cost": None,
                "diverge_penalty": float(self.lambda_diverge),
                "P_loss": None,
                "slack_P": None, "slack_cost": None,
                "line_monitor": None,
            })
            return obs, float(reward), True, info

        # =====================
        # 成本部分
        # =====================

        # 我们直接控制的四个点：Bus2, Bus3, PV(6), WT(8)
        C_gen = 0.0
        per_gen = []

        ctrl_buses = [self.thermal_buses[0],
                      self.thermal_buses[1],
                      self.pv_bus,
                      self.wt_bus]  # 2,3,6,8

        for i, bus in enumerate(ctrl_buses):
            g = get_generators_at_bus(self.grid, bus)[0]
            P = float(act_used[i])
            c1 = float(getattr(g, "Cost", 0.0))
            c2 = float(getattr(g, "Cost2", 0.0))
            cost_i = c1 * P + c2 * (P ** 2)
            C_gen += cost_i
            per_gen.append({
                "name": getattr(g, "name", f"gen_bus{bus}"),
                "P": P,
                "c1": c1,
                "c2": c2,
                "cost": cost_i
            })

        # slack 补余功率：负荷 + 网损 - 我们指定的四台机组出力
        bus_df = pf.results.get_bus_df()
        P_bus = np.asarray(bus_df["P"], dtype=float)

        # 系统总负荷(正数)
        P_load = float(-np.sum(P_bus[P_bus < 0.0]))

        # 网损 (MW)：pf.results.losses 是 per-branch 复功率(?)，之前我们用 sum(real(losses))
        S_tot = np.asarray(pf.results.losses).sum()
        P_loss = float(np.real(S_tot))

        P_known = float(np.sum(act_used))
        P_slack = P_load + P_loss - P_known

        g_slack = get_generators_at_bus(self.grid, self.slack_bus)[0]  # Bus1
        c1_s = float(getattr(g_slack, "Cost", 60.0))
        c2_s = float(getattr(g_slack, "Cost2", 0.0))
        slack_cost = c1_s * P_slack + c2_s * (P_slack ** 2)

        C_gen += slack_cost
        per_gen.append({
            "name": getattr(g_slack, "name", "slack_bus1"),
            "P": float(P_slack),
            "c1": c1_s,
            "c2": c2_s,
            "cost": float(slack_cost)
        })

        # 线损惩罚
        C_loss = self.lambda_loss * P_loss

        # ========== 线路载荷与过载惩罚 ==========
        branches = self.grid.get_branches(add_vsc=False, add_hvdc=False, add_switch=True)
        loading_arr = np.abs(pf.results.loading)  # shape == len(branches)

        line_monitor_list = []
        max_loading_pct = 0.0
        overload_penalty_sum = 0.0

        for i, br in enumerate(branches):
            # from/to 名称
            fb = getattr(getattr(br, "bus_from", None), "name", f"?{i}")
            tb = getattr(getattr(br, "bus_to", None), "name", f"?{i}")

            # 热限 (MVA)
            rate_val = float(getattr(br, "rate", np.nan))

            # loading p.u. -> 百分比
            ld_pu = float(loading_arr[i])  # 1.20 表示 120%
            ld_pct = ld_pu * 100.0

            # 估算当前分支流量 "实际MVA" = loading_pu * 额定容量
            if rate_val is not None and not np.isnan(rate_val):
                flow_est = ld_pu * rate_val
            else:
                flow_est = np.nan

            # 记录 max loading
            if ld_pct > max_loading_pct:
                max_loading_pct = ld_pct

            # 过载惩罚：只对 ld_pu>1 的部分加权
            over = max(0.0, ld_pu - 1.0)
            overload_penalty_sum += over ** 2

            line_info = {
                "idx": int(i),
                "from": fb,
                "to": tb,
                "rate_MVA": rate_val,
                "flow_MVA_est": flow_est,
                "loading_pct": ld_pct,
                "type": type(br).__name__,
                "name": getattr(br, "name", f"branch_{i}"),
            }
            line_monitor_list.append(line_info)

        # 调试输出
        print("=== LINE MONITOR (this step, aligned with VeraGrid results.loading) ===")
        for li in line_monitor_list:
            print(
                f"[{li['idx']:02d}] {li['from']} -> {li['to']}  "
                f"{li['flow_MVA_est']:.2f} MVA / {li['rate_MVA']:.2f} MVA  "
                f"({li['loading_pct']:.1f}%)  "
                f"{li['type']}  {li['name']}"
            )
        print("=== END LINE MONITOR ===")

        # 线路过载惩罚项
        C_ov = self.lambda_overload * float(overload_penalty_sum)

        # =====================
        # 总成本 & 回报
        # =====================
        total_cost = C_gen / 1000.0 + C_loss + C_ov / 500.0
        reward = - total_cost

        # =====================
        # 观测 + info
        # =====================

        # 恢复 PV/WT 上限，别让潮流器把限制改掉
        self._rewrite_pv_wt_limits_only()

        obs = self._build_obs()

        Vm_min = float(bus_df["Vm"].min())
        Vm_max = float(bus_df["Vm"].max())

        info.update({
            "converged": True,
            "C_gen": float(C_gen),
            "C_loss": float(C_loss),
            "C_ov": float(C_ov),
            "total_cost": float(total_cost),
            "slack_P": float(P_slack),
            "slack_cost": float(slack_cost),
            "P_loss": float(P_loss),

            "branch_loading_pct_max": float(max_loading_pct),
            "Vm_min": Vm_min,
            "Vm_max": Vm_max,

            "pv_pmax_rand": float(self.pv_pmax_rand),
            "wt_pmax_rand": float(self.wt_pmax_rand),
            "load_scale": float(self.load_scale),
            "per_generator": per_gen,

            # 每条线的当前loading信息（索引与 pf.results.loading 对齐）
            "line_monitor": line_monitor_list,
        })

        done = True  # 单步任务
        return obs, float(reward), done, info

    def debug_print_generators(self):
        print("=== DEBUG GEN LIST ===")
        for g in self.grid.generators:
            bus_name = getattr(g.bus, "name", "<no_name>")
            print({
                "gen_name": getattr(g, "name", None),
                "bus": bus_name,
                "Pmin": getattr(g, "Pmin", None),
                "Pmax": getattr(g, "Pmax", None),
                "Vset": getattr(g, "Vset", None),
                "Cost": getattr(g, "Cost", None),
                "Cost2": getattr(g, "Cost2", None),
                "maybe_slack_flag": (
                    getattr(g, "is_slack", None)
                    or getattr(g, "is_ref", None)
                    or getattr(g, "slack", None)
                )
            })


def make_env(seed: int | None = None) -> GridOPFEnv:
    env = GridOPFEnv(seed=seed)
    env.reset(seed=seed)
    return env


def get_env_spec(seed: int | None = None) -> dict:
    env = make_env(seed=seed)
    spec = {
        "state_dim": int(env.state_dim),
        "action_dim": int(env.action_dim),
        "act_limit": float(GridOPFEnv.RECOMMENDED_ACT_LIMIT),
    }
    return spec


__all__ = [
    "GridOPFEnv",
    "make_env",
    "get_env_spec",
    "load_case14_as_veragrid",
    "get_bus_by_name",
    "get_generators_at_bus",
]
