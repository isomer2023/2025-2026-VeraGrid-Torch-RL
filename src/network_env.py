# network_env.py
from __future__ import annotations

import numpy as np
import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en
import src.GC_PandaPowerImporter as GC_PandaPowerImporter
import src.network_loader as net_loader
import src.grid_utils as g_utils
import torch
import torch_geometric as pyg

class GridOPFEnv:

    RECOMMENDED_ACT_LIMIT = 200.0

    def __init__(self,
                 pv_range=(0.30, 0.90),
                 wt_range=(0.20, 0.95),
                 load_jitter=0.15,         # random load jitter(+random)
                 lambda_dist=1.0,          # distance loss
                 lambda_diverge=50.0,      # not conv
                 seed=None):

        self.base_net, self.base_grid = net_loader.load_network(
            run_pp_before=True, sanitize=True, set_line_rate_100=True
        )
        self.rng = np.random.default_rng(seed)

        # 可控机组位置：火电(1,2,3)，光伏(6)，风电(8)
        self.thermal_buses = [1, 2, 3]
        self.pv_bus = 6
        self.wt_bus = 8

        # 火电上下限（固定）
        self.th_limits = [
            (10.0, 200.0),  # Bus 1
            (5.0, 120.0),   # Bus 2
            (5.0, 80.0),    # Bus 3
        ]

        # 风/光可用上限每回合随机
        self.pv_range = pv_range
        self.wt_range = wt_range

        # 负荷温和随机
        self.load_jitter = float(load_jitter)

        # 奖励系数
        self.lambda_dist = float(lambda_dist)
        self.lambda_diverge = float(lambda_diverge)

        # 维度（SAC 将自动读取）
        self.action_dim = 5              # th1, th2, th3, pv, wt
        self.state_dim = 14 + 2          # 14个母线近似净负荷 + [pv_max, wt_max]

        # 内部状态
        self.grid = None
        self.pv_pmax_rand = None
        self.wt_pmax_rand = None
        self.load_scale = None

        # 缓存 OPF/PF
        self.last_ac_opf = None
        self.last_pf = None

    # ---------- 核心动作映射 ----------
    def _apply_action_to_generators(self, action_vec):
        """
        根据动作（MW）直接设置 5 台机组的 Pmin/Pmax = action（锁定指令），
        让 PF 用这个 dispatch（注意：这是“约束校验”的动作；OPF 仍用于得到 a*）
        """
        th1, th2, th3, pv, wt = action_vec

        def clamp(x, lo, hi):
            return float(max(lo, min(hi, x)))

        # 获取机组对象
        g1 = g_utils.get_generators_at_bus(self.grid, self.thermal_buses[0])[0]
        g2 = g_utils.get_generators_at_bus(self.grid, self.thermal_buses[1])[0]
        g3 = g_utils.get_generators_at_bus(self.grid, self.thermal_buses[2])[0]
        g6 = g_utils.get_generators_at_bus(self.grid, self.pv_bus)[0]
        g8 = g_utils.get_generators_at_bus(self.grid, self.wt_bus)[0]

        # 火电锁指令
        (lo1, hi1), (lo2, hi2), (lo3, hi3) = self.th_limits
        th1 = clamp(th1, lo1, hi1)
        th2 = clamp(th2, lo2, hi2)
        th3 = clamp(th3, lo3, hi3)

        # 光/风锁指令（不超过“本回合随机上限”）
        pv = clamp(pv, 0.0, self.pv_pmax_rand)
        wt = clamp(wt, 0.0, self.wt_pmax_rand)

        # 设置 Pmin=Pmax=action
        def pin(g, p):
            g.Pmin = float(p)
            g.Pmax = float(p) + 1e-10

        pin(g1, th1); pin(g2, th2); pin(g3, th3)
        pin(g6, pv);  pin(g8, wt)

        return np.array([th1, th2, th3, pv, wt], dtype=float)

    # ---------- 每回合随机化 ----------
    def _randomize_pv_wt(self):
        self.pv_pmax_rand = 40.0 * self.rng.uniform(*self.pv_range)
        self.wt_pmax_rand = 50.0 * self.rng.uniform(*self.wt_range)

        # 写到机组上（用于 OPF 上限）
        g6 = g_utils.get_generators_at_bus(self.grid, self.pv_bus)[0]
        g8 = g_utils.get_generators_at_bus(self.grid, self.wt_bus)[0]
        g6.Pmin, g6.Pmax = 0.0, self.pv_pmax_rand + 1e-10
        g8.Pmin, g8.Pmax = 0.0, self.wt_pmax_rand + 1e-10
        g6.Cost, g6.Cost2 = 2.0, 0.0
        g8.Cost, g8.Cost2 = 1.0, 0.0
        g6.Vset, g8.Vset = 1.02, 1.02

    def _randomize_loads(self):
        """
        温和随机化每个 load 的 P/Q：乘以 N(1, load_jitter)
        """
        self.load_scale = float(self.rng.normal(1.0, self.load_jitter))
        self.load_scale = max(0.8, min(1.2, self.load_scale))  # 再夹一下

        for ld in self.grid.loads:
            ld.P *= self.load_scale
            ld.Q *= self.load_scale

    # ---------- OPF（求最优 a*） ----------
    def _run_ac_opf(self):
        # 设置火电成本与限额
        g1 = g_utils.get_generators_at_bus(self.grid, self.thermal_buses[0])[0]
        g2 = g_utils.get_generators_at_bus(self.grid, self.thermal_buses[1])[0]
        g3 = g_utils.get_generators_at_bus(self.grid, self.thermal_buses[2])[0]
        g1.Pmin, g1.Pmax = self.th_limits[0][0], self.th_limits[0][1] + 1e-10
        g2.Pmin, g2.Pmax = self.th_limits[1][0], self.th_limits[1][1] + 1e-10
        g3.Pmin, g3.Pmax = self.th_limits[2][0], self.th_limits[2][1] + 1e-10
        g1.Cost, g1.Cost2, g1.Vset = 30.0, 0.02, 1.03
        g2.Cost, g2.Cost2, g2.Vset = 35.0, 0.03, 1.03
        g3.Cost, g3.Cost2, g3.Vset = 40.0, 0.04, 1.01

        ac_opt = gce.OptimalPowerFlowOptions(
            solver=en.SolverType.NONLINEAR_OPF,
            mip_solver=en.MIPSolvers.HIGHS,
            power_flow_options=gce.PowerFlowOptions(solver_type=en.SolverType.NR, verbose=False),
            ips_init_with_pf=False
        )
        ac_opf = gce.OptimalPowerFlowDriver(grid=self.grid, options=ac_opt)
        ac_opf.run()
        self.last_ac_opf = ac_opf
        return ac_opf

    # ---------- PF（用 OPF 结果做校验 / 用动作锁出力做校验） ----------
    def _run_pf(self, opf_results=None):
        pf = gce.PowerFlowDriver(
            grid=self.grid,
            options=gce.PowerFlowOptions(solver_type=en.SolverType.NR, verbose=False),
            opf_results=opf_results
        )
        pf.run()
        self.last_pf = pf
        return pf

    # ---------- 观测 ----------
    def _build_obs(self):
        """
        观测：14 维母线近似“净有功注入的负向和”（即需求）的粗略向量 + [pv_max, wt_max]
        这里用 PF 结果的 bus_df 来近似
        """
        # 用 OPF 结果跑一次 PF，拿 bus_df（稳定）
        pf = self._run_pf(opf_results=self.last_ac_opf.results)
        bus_df = pf.results.get_bus_df()

        # 粗略取每个母线的 P，如果是发电为正、负荷为负，这里取负向代表“需求”
        p_col = np.array(bus_df["P"].values, dtype=float)
        demand_like = np.where(p_col < 0.0, -p_col, 0.0)

        if demand_like.shape[0] < 14:
            demand_like = np.pad(demand_like, (0, 14 - demand_like.shape[0]))
        elif demand_like.shape[0] > 14:
            demand_like = demand_like[:14]

        obs = np.concatenate([demand_like, [self.pv_pmax_rand, self.wt_pmax_rand]]).astype(np.float32)
        return obs

    # output
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 复制基网
        _, self.grid = net_loader.load_network(run_pp_before=True, sanitize=True, set_line_rate_100=True)

        # 每回合：随机风/光、随机负荷
        self._randomize_pv_wt()
        self._randomize_loads()

        # 先跑一次 OPF（得到 a*）
        self._run_ac_opf()

        # 构建观测
        obs = self._build_obs()
        return obs

    # GNN
    def get_graph_data(self):

        node_feats = [] # node feature
        for b in self.grid.buses:
            node_feats.append([b.Vset, getattr(b, "Pload", 0.0), getattr(b, "Qload", 0.0)])
        x = torch.tensor(node_feats, dtype=torch.float)

        edge_index = [] # edge connection
        for ln in self.grid.lines:
            i = self.grid.buses.index(ln.bus_from)
            j = self.grid.buses.index(ln.bus_to)
            edge_index.append([i, j])
            edge_index.append([j, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return pyg.data.Data(x=x, edge_index=edge_index)

    def step(self, action: np.ndarray):
        """
        单步任务：
          - 应用 agent 动作（钉死四台机组出力）
          - 跑潮流
          - 如果潮流不收敛 → 巨罚直接终止
          - 如果收敛 → 计算成本、损耗、线路过载惩罚
          - reward = - total_cost
          - done=True（单步）

        新增：
          - 使用 VeraGrid 的 pf.results.loading 和 get_branches()，
            稳定打印每条支路的:
              from_bus -> to_bus,
              实际流量 MVA (估算),
              额定容量 rate MVA,
              loading %
          - 把这些信息塞进 info["line_monitor"]
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
            g = g_utils.get_generators_at_bus(self.grid, bus)[0]
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
        # VeraGrid 里 pf.results.losses 是数组 (MVA per branch)；我们继续用之前定义的逻辑：
        S_tot = np.asarray(pf.results.losses).sum()
        P_loss = float(np.real(S_tot))

        P_known = float(np.sum(act_used))
        P_slack = P_load + P_loss - P_known

        g_slack = g_utils.get_generators_at_bus(self.grid, self.slack_bus)[0]  # Bus1
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

        # =====================
        # 线路过载惩罚 & 打印线路利用率
        # =====================

        # VeraGrid 自己在 PowerFlowDriver.add_report() 里这样写：
        #   loading = np.abs(self.results.loading)
        #   branches = self.grid.get_branches(add_vsc=False, add_hvdc=False, add_switch=True)
        #   for i, branch in enumerate(branches):
        #       if loading[i] > 1.0: ...
        #
        # 这说明:
        #   - pf.results.loading[i] 和 branches[i] 是一一对应的
        #   - loading 是 per-branch 的 MVA/MW 占额定容量的比例(p.u.)
        #
        # 所以我们不再猜 branch_df，也不再假设顺序。
        # 我们直接用这对齐关系来构建 line_monitor。

        branches = self.grid.get_branches(add_vsc=False, add_hvdc=False, add_switch=True)
        loading_arr = np.abs(pf.results.loading)  # shape == len(branches)

        line_monitor_list = []
        max_loading_pct = 0.0
        overload_penalty_sum = 0.0

        for i, br in enumerate(branches):
            # from/to 名称
            fb = getattr(getattr(br, "bus_from", None), "name", f"?{i}")
            tb = getattr(getattr(br, "bus_to", None), "name", f"?{i}")

            # 热限 (MVA 或 MW, 取决于 AC/DC, 但对loading是统一的基准)
            rate = float(getattr(br, "rate", np.nan))

            # loading p.u. -> 百分比
            ld_pu = float(loading_arr[i])  # 1.20 表示 120%
            ld_pct = ld_pu * 100.0

            # 估算当前分支流量 "实际MVA" = loading_pu * 额定容量
            # 这是合理的：loading = |S_now| / rate
            if rate is not None and not np.isnan(rate):
                flow_est = ld_pu * rate
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
                "rate_MVA": rate,
                "flow_MVA_est": flow_est,
                "loading_pct": ld_pct,
                "type": type(br).__name__,
                "name": getattr(br, "name", f"branch_{i}"),
            }
            line_monitor_list.append(line_info)

        # 控制台打印
        print("=== LINE MONITOR (this step, aligned with VeraGrid results.loading) ===")
        for li in line_monitor_list:
            print(
                f"[{li['idx']:02d}] {li['from']} -> {li['to']}  "
                f"{li['flow_MVA_est']:.2f} MVA / {li['rate_MVA']:.2f} MVA  "
                f"({li['loading_pct']:.1f}%)  "
                f"{li['type']}  {li['name']}"
            )
        print("=== END LINE MONITOR ===")

        # 线路过载惩罚项 C_ov：我们沿用之前的系数，但现在用 VeraGrid 的 loading 结果
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
            # GNN
            #"graph_data" : float(self.get_graph_data),

            # 我们现在放的是对齐后的线路状态（idx 是和 results.loading 对齐的）
            "line_monitor": line_monitor_list,
        })

        done = True  # 单步任务
        return obs, float(reward), done, info
'''
    def step(self, action: np.ndarray):
        """
        action：长度 5 的 MW 指令（外部 SAC 传入）
        """
        action = np.asarray(action, dtype=float).copy()
        assert action.shape[0] == 5, "action 维度应为 5"

        # 1) AC-OPF 得到 a*
        ac_opf = self.last_ac_opf  # 已在 reset 中做过
        opf_genP = np.array(ac_opf.results.generator_power, dtype=float)  # 所有机组

        # 只抽取 (th1, th2, th3, pv, wt) 顺序的 a*
        g1 = g_utils.get_generators_at_bus(self.grid, self.thermal_buses[0])[0]
        g2 = g_utils.get_generators_at_bus(self.grid, self.thermal_buses[1])[0]
        g3 = g_utils.get_generators_at_bus(self.grid, self.thermal_buses[2])[0]
        g6 = g_utils.get_generators_at_bus(self.grid, self.pv_bus)[0]
        g8 = g_utils.get_generators_at_bus(self.grid, self.wt_bus)[0]
        idx_map = [self.grid.generators.index(g1),
                   self.grid.generators.index(g2),
                   self.grid.generators.index(g3),
                   self.grid.generators.index(g6),
                   self.grid.generators.index(g8)]
        a_star = opf_genP[idx_map]

        # 2) 距离项
        dist = np.mean(np.abs(action - a_star))

        # 3) 用动作锁定出力，做一次 PF 校验
        self._apply_action_to_generators(action)
        pf_chk = self._run_pf(opf_results=None)  # 按锁定出力直接潮流
        converged = bool(pf_chk.results.converged)

        # 4) 奖励
        r_dist = - self.lambda_dist * dist
        r_penalty = - self.lambda_diverge if not converged else 0.0
        reward = r_dist + r_penalty

        # 5) 观测（仍基于 OPF 的 PF）
        # 恢复 OPF 的机组上下限（以免下个状态不一致）
        self._randomize_pv_wt()  # 只重写 PV/WT 的 Pmax，不改随机值
        obs = self._build_obs()

        # 6) info（用于日志/评测）
        info = {}
        info["action"] = action.tolist()
        info["a_star"] = a_star.tolist()
        info["r_dist"] = float(r_dist)
        info["r_penalty"] = float(r_penalty)

        # PF（基于 OPF 结果）关键指标
        pf = self.last_pf
        info["pf_converged"] = bool(pf.results.converged)
        S_loss = pf.results.losses.sum()
        info["pf_losses_P"] = float(S_loss.real)
        info["pf_losses_Q"] = float(S_loss.imag)

        bus_df = pf.results.get_bus_df()
        info["Vm_min"] = float(bus_df["Vm"].min())
        info["Vm_max"] = float(bus_df["Vm"].max())

        branch_df = pf.results.get_branch_df().copy()
        if "rate_MVA" not in branch_df.columns:
            branch_df["rate_MVA"] = 100.0
        loading_pct = 100.0 * np.hypot(branch_df["Pf"], branch_df["Qf"]) / branch_df["rate_MVA"]
        info["branch_loading_pct_max"] = float(loading_pct.max())

        info["pv_pmax_rand"] = float(self.pv_pmax_rand)
        info["wt_pmax_rand"] = float(self.wt_pmax_rand)
        info["load_scale"] = float(self.load_scale)


        done = True  # 单步任务（模仿最优），每回合一步
        return obs, float(reward), done, info
'''

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
    #"load_case14_as_veragrid",
    #"get_bus_by_name",
    #"get_generators_at_bus",
]
