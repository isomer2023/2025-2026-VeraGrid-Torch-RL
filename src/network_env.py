# network_env_ver1_opfdif.py
from __future__ import annotations

import numpy as np
import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en
import src.GC_PandaPowerImporter as GC_PandaPowerImporter
import src.network_loader as net_loader
import src.grid_utils as g_utils
import torch
import torch_geometric as pyg
import copy

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
        self.base_net, self.base_grid = net_loader.load_network(
            run_pp_before=True,
            sanitize=True
        )
        self.rng = np.random.default_rng(seed)

        # 哪些母线对应哪些机组
        self.slack_bus = 1
        self.thermal_buses = [2, 3]  # 火电机组所在母线
        self.pv_bus = 6  # PV母线
        self.wt_bus = 8  # WT母线

        # 火电机组出力范围 (MW)
        self.th_limits = [
            (5.0, 120.0),  # Bus2
            (5.0, 80.0),  # Bus3
        ]

        # slack 发电机的估计范围 (仅用于成本核算，不是硬钉)
        self.slack_limits = (10.0, 200.0)

        # PV/WT 随机可用范围
        self.pv_range = pv_range
        self.wt_range = wt_range

        # 负荷扰动强度：N(1, load_jitter) 后截断到 [0.8,1.2]
        self.load_jitter = float(load_jitter)

        # reward 系数
        self.lambda_loss = float(lambda_loss)
        self.lambda_overload = float(lambda_overload)
        self.lambda_diverge = float(lambda_diverge)

        # 观测、动作维度
        self.action_dim = 4
        self.state_dim = 14 + 2  # 14个bus需求 + [pv_cap, wt_cap]

        # 运行时变量
        self.grid = None
        self.pv_pmax_rand = None
        self.wt_pmax_rand = None
        self.load_scale = None
        self.last_pf = None

        # === 固定线路热限 (单一高可再生产景推出来) ===
        pv_max_possible = 40.0 * max(self.pv_range)  # e.g. 40 * 0.90 = 36 MW
        wt_max_possible = 50.0 * max(self.wt_range)  # e.g. 50 * 0.95 = 47.5 MW

        self.static_line_limits = g_utils.compute_static_branch_limits_high_RE(
            base_grid=self.base_grid,
            th2_max=self.th_limits[0][1],  # 120.0
            th3_max=self.th_limits[1][1],  # 80.0
            pv_max_possible=pv_max_possible,  # ~36 MW
            wt_max_possible=wt_max_possible,  # ~47.5 MW
            load_stress_res=1.0,  # 假设白天负荷大约正常水平
            safety_factor=1.2,  # 给线路留20%裕度
            floor_MVA=5.0,
        )

    def _set_generator_costs(self):
        """
        为当前 self.grid 写入机组的成本系数、电压设定等。
        Slack 给范围仅用于成本计算。
        """
        # Slack（Bus1）
        g1 = g_utils.get_generators_at_bus(self.grid, self.slack_bus)[0]
        g1.Cost, g1.Cost2, g1.Vset = 60.0, 0.0, 1.03
        g1.Pmin, g1.Pmax = self.slack_limits[0], self.slack_limits[1] + 1e-6

        # 火电机组：Bus2 / Bus3
        g2 = g_utils.get_generators_at_bus(self.grid, self.thermal_buses[0])[0]
        g3 = g_utils.get_generators_at_bus(self.grid, self.thermal_buses[1])[0]
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

        g6 = g_utils.get_generators_at_bus(self.grid, self.pv_bus)[0]
        g8 = g_utils.get_generators_at_bus(self.grid, self.wt_bus)[0]

        g6.Pmin, g6.Pmax = 0.0, self.pv_pmax_rand + 1e-10
        g8.Pmin, g8.Pmax = 0.0, self.wt_pmax_rand + 1e-10
        g6.Cost, g6.Cost2 = 2.0, 0.0
        g8.Cost, g8.Cost2 = 1.0, 0.0
        g6.Vset, g8.Vset = 1.02, 1.02

    def _rewrite_pv_wt_limits_only(self):
        """
        每个 step 后恢复 PV/WT 的 Pmin/Pmax，避免潮流求解器给它们乱改。
        """
        g6 = g_utils.get_generators_at_bus(self.grid, self.pv_bus)[0]
        g8 = g_utils.get_generators_at_bus(self.grid, self.wt_bus)[0]
        g6.Pmin, g6.Pmax = 0.0, float(self.pv_pmax_rand) + 1e-10
        g8.Pmin, g8.Pmax = 0.0, float(self.wt_pmax_rand) + 1e-10

    def _randomize_loads(self):
        """
        给所有负荷乘一个随机系数 (clip到[0.8,1.2])，模拟一天的负荷波动。
        """
        self.load_scale = float(self.rng.normal(1.0, self.load_jitter))
        self.load_scale = max(0.8, min(1.2, self.load_scale))

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

        g2 = g_utils.get_generators_at_bus(self.grid, self.thermal_buses[0])[0]  # Bus2
        g3 = g_utils.get_generators_at_bus(self.grid, self.thermal_buses[1])[0]  # Bus3
        g6 = g_utils.get_generators_at_bus(self.grid, self.pv_bus)[0]  # PV
        g8 = g_utils.get_generators_at_bus(self.grid, self.wt_bus)[0]  # WT

        (lo2, hi2), (lo3, hi3) = self.th_limits
        th2 = clamp(th2, lo2, hi2)
        th3 = clamp(th3, lo3, hi3)
        pv = clamp(pv, 0.0, self.pv_pmax_rand)
        wt = clamp(wt, 0.0, self.wt_pmax_rand)

        g_utils._pin_gen_power(g2, th2)
        g_utils._pin_gen_power(g3, th3)
        g_utils._pin_gen_power(g6, pv)
        g_utils._pin_gen_power(g8, wt)

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
        #self.debug_print_generators()

        # 7. 返回观察（基于随机负荷+风光cap，不依赖潮流结果）
        return self._build_obs()


    def step(self, action: np.ndarray):
        """
        单步任务：
          - 应用 agent 动作到四台可控机组（Bus2/3/PV/WT）
          - 执行潮流计算
          - 潮流不收敛 → 巨罚
          - 潮流收敛 → 计算发电成本、网损、线路过载惩罚
          - reward = - total_cost
          - done = True（单步任务）

        打印：
          - 对齐 VeraGrid 的 pf.results.loading，输出每条线路的流量、容量、loading%
          - info["line_monitor"] 记录详细线路信息
        """

        action = np.asarray(action, dtype=float).copy()
        assert action.shape[0] == 4, "action 维度应为 4（Bus2/3 + PV + WT）"

        # 1️⃣ 应用动作到机组（不包括 slack）
        act_used = self._apply_action_to_generators(action)

        # 2️⃣ 运行潮流计算
        pf = self._run_pf()
        info = {"action": act_used.tolist()}
        if not pf.results.converged:
            return self._handle_pf_failure(info)

        # 3️⃣ 成本计算部分（机组 + Slack + 网损）
        gen_cost, slack_cost, loss_cost, P_slack, P_loss, per_gen = \
            self._compute_costs(pf, act_used)

        # 4️⃣ 线路监控与过载惩罚
        line_monitor, max_loading_pct, overload_penalty = \
            self._compute_line_monitor(pf)

        # 5️⃣ 总成本与奖励
        total_cost = gen_cost / 1000.0 + loss_cost + overload_penalty / 500.0
        reward = -float(total_cost)

        # 6️⃣ 恢复 PV/WT 上限
        self._rewrite_pv_wt_limits_only()

        # 7️⃣ 构建观测与信息
        obs = self._build_obs()
        bus_df = pf.results.get_bus_df()
        Vm_min, Vm_max = float(bus_df["Vm"].min()), float(bus_df["Vm"].max())

        info.update({
            "converged": True,
            "C_gen": float(gen_cost),
            "C_loss": float(loss_cost),
            "C_ov": float(overload_penalty),
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
            "line_monitor": line_monitor,
        })

        return obs, reward, True, info

    # ----------------------------------------------------------
    # 辅助函数
    # ----------------------------------------------------------

    def _handle_pf_failure(self, info):
        """处理潮流不收敛的情况"""
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

    def _compute_costs(self, pf, act_used):
        """计算机组成本、Slack 成本、网损"""
        C_gen, per_gen = 0.0, []
        ctrl_buses = [self.thermal_buses[0], self.thermal_buses[1],
                      self.pv_bus, self.wt_bus]  # 2,3,6,8

        # 控制机组成本
        for i, bus in enumerate(ctrl_buses):
            g = g_utils.get_generators_at_bus(self.grid, bus)[0]
            P = float(act_used[i])
            c1, c2 = float(getattr(g, "Cost", 0.0)), float(getattr(g, "Cost2", 0.0))
            cost_i = c1 * P + c2 * (P ** 2)
            C_gen += cost_i
            per_gen.append({
                "name": getattr(g, "name", f"gen_bus{bus}"),
                "P": P, "c1": c1, "c2": c2, "cost": cost_i
            })

        # 潮流结果
        bus_df = pf.results.get_bus_df()
        P_bus = np.asarray(bus_df["P"], dtype=float)
        P_load = float(-np.sum(P_bus[P_bus < 0.0]))  # 正数
        S_tot = np.asarray(pf.results.losses).sum()
        P_loss = float(np.real(S_tot))
        P_known = float(np.sum(act_used))
        P_slack = P_load + P_loss - P_known

        # Slack 成本
        g_slack = g_utils.get_generators_at_bus(self.grid, self.slack_bus)[0]
        c1_s, c2_s = float(getattr(g_slack, "Cost", 60.0)), float(getattr(g_slack, "Cost2", 0.0))
        slack_cost = c1_s * P_slack + c2_s * (P_slack ** 2)
        C_gen += slack_cost
        per_gen.append({
            "name": getattr(g_slack, "name", "slack_bus1"),
            "P": float(P_slack), "c1": c1_s, "c2": c2_s, "cost": float(slack_cost)
        })

        # 网损惩罚
        C_loss = self.lambda_loss * P_loss
        return C_gen, slack_cost, C_loss, P_slack, P_loss, per_gen

    def _compute_line_monitor(self, pf):
        """根据 VeraGrid 的 results.loading 输出线路监控与过载惩罚"""
        branches = self.grid.get_branches(add_vsc=False, add_hvdc=False, add_switch=True)
        loading_arr = np.abs(pf.results.loading)
        line_monitor, overload_penalty_sum, max_loading_pct = [], 0.0, 0.0

        if getattr(self, "verbose", False):
            print("=== LINE MONITOR (aligned with VeraGrid results.loading) ===")
            for i, br in enumerate(branches):
                fb = getattr(getattr(br, "bus_from", None), "name", f"?{i}")
                tb = getattr(getattr(br, "bus_to", None), "name", f"?{i}")
                rate = float(getattr(br, "rate", np.nan))
                ld_pu = float(loading_arr[i])
                ld_pct = ld_pu * 100.0
                flow_est = ld_pu * rate if not np.isnan(rate) else np.nan
                max_loading_pct = max(max_loading_pct, ld_pct)
                over = max(0.0, ld_pu - 1.0)
                overload_penalty_sum += over ** 2

                line_info = {
                    "idx": int(i),
                    "from": fb, "to": tb,
                    "rate_MVA": rate,
                    "flow_MVA_est": flow_est,
                    "loading_pct": ld_pct,
                    "type": type(br).__name__,
                    "name": getattr(br, "name", f"branch_{i}"),
                }
                line_monitor.append(line_info)


                print(f"[{i:02d}] {fb} -> {tb}  "
                          f"{flow_est:.2f} MVA / {rate:.2f} MVA  "
                          f"({ld_pct:.1f}%)  {line_info['type']}  {line_info['name']}")
            print("=== END LINE MONITOR ===")

        C_ov = self.lambda_overload * float(overload_penalty_sum)
        return line_monitor, max_loading_pct, C_ov

# --------------------------------
# 供上层自动读取的几个便捷函数
# --------------------------------
def make_env(seed: int | None = None) -> GridOPFEnv:
    """上层直接调用创建环境。"""
    env = GridOPFEnv(seed=seed)
    # 先 reset 一次，确保内部 OPF/PF 缓存初始化好（SAC 可能立刻要 obs_dim）
    env.reset(seed=seed)
    return env


def get_env_spec(seed: int | None = None) -> dict:
    """
    让 SAC 自动读取环境规格（state_dim / action_dim / act_limit）。
    - act_limit: 给策略 tanh 输出做线性缩放的推荐幅度（与原脚本保持一致 200.0）
    """
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
]
