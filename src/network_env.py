from __future__ import annotations

import numpy as np
import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en
import src.GC_PandaPowerImporter as GC_PandaPowerImporter
import src.network_loader as net_loader
import src.grid_utils as g_utils

# -------------------------------------------
# 环境：GridOPFEnv（模仿最优 + 约束校验）
# -------------------------------------------
class GridOPFEnv:

    RECOMMENDED_ACT_LIMIT = 200.0

    def __init__(self,
                 pv_range=(0.30, 0.90),
                 wt_range=(0.20, 0.95),
                 load_jitter=0.15,         # 负荷随机温和系数（越大越随机）
                 lambda_dist=1.0,          # 距离损失系数
                 lambda_diverge=50.0,      # 不收敛惩罚
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

    # ---------- 对外接口 ----------
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
