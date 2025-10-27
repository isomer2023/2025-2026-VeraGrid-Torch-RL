# ============================
# network_env.py  —— PF 奖励版（Bus1 为 Slack，动作维度=4）
# ============================
from __future__ import annotations
import numpy as np

# VeraGrid / GridCal（Vera 分支）
import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en

# Pandapower → VeraGrid 转换
import GC_PandaPowerImporter as GC_PandaPowerImporter


# ---------------------------
# 工具函数
# ---------------------------
def load_case14_as_veragrid(run_pp_before=True, sanitize=True, set_line_rate_100=True):
    """
    加载 IEEE-14（pandapower），做基础清洗，然后转换为 VeraGrid MultiCircuit。
    """
    import pandapower as pp
    import pandapower.networks as nw
    import numpy as np
    import GC_PandaPowerImporter as GC_PandaPowerImporter

    net_pp = nw.case14()

    net_pp.bus["max_vm_pu"] = np.maximum(net_pp.bus.get("max_vm_pu", 1.05), 1.07)
    if "vm_pu" in net_pp.gen.columns:
        net_pp.gen["vm_pu"] = np.minimum(net_pp.gen["vm_pu"].fillna(1.03), 1.03)

    if run_pp_before:
        try:
            pp.runpp(net_pp, algorithm="nr", init="flat")
        except Exception as e:
            print("pandapower runpp 失败：", e)

    grid_gc = GC_PandaPowerImporter.PP2GC(net_pp)

    if sanitize:
        for b in grid_gc.buses:
            b.Vmin, b.Vmax = 0.95, max(getattr(b, "Vmax", 1.05), 1.05)
        for g in grid_gc.generators:
            g.Vset = min(max(getattr(g, "Vset", 1.01), 0.95), 1.03)

    if set_line_rate_100:
        for ln in grid_gc.lines:
            ln.rate = 100.0

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
        self.base_net, self.base_grid = load_case14_as_veragrid(
            run_pp_before=True, sanitize=True, set_line_rate_100=True
        )
        self.rng = np.random.default_rng(seed)

        self.slack_bus = 1
        self.thermal_buses = [2, 3]
        self.pv_bus = 6
        self.wt_bus = 8

        self.th_limits = [
            (5.0, 120.0),   # Bus2
            (5.0, 80.0),    # Bus3
        ]
        self.slack_limits = (10.0, 200.0)

        self.pv_range = pv_range
        self.wt_range = wt_range
        self.load_jitter = float(load_jitter)

        self.lambda_loss = float(lambda_loss)
        self.lambda_overload = float(lambda_overload)
        self.lambda_diverge = float(lambda_diverge)

        self.action_dim = 4
        self.state_dim = 14 + 2

        self.grid = None
        self.pv_pmax_rand = None
        self.wt_pmax_rand = None
        self.load_scale = None

        self.last_pf = None

    def _set_generator_costs(self):

        # Slack（Bus1）
        g1 = get_generators_at_bus(self.grid, self.slack_bus)[0]
        g1.Cost, g1.Cost2, g1.Vset = 60.0, 0.0, 1.03
        g1.Pmin, g1.Pmax = self.slack_limits[0], self.slack_limits[1] + 1e-6

        g2 = get_generators_at_bus(self.grid, self.thermal_buses[0])[0]
        g3 = get_generators_at_bus(self.grid, self.thermal_buses[1])[0]
        g2.Cost, g2.Cost2, g2.Vset = 35.0, 0.03, 1.03
        g3.Cost, g3.Cost2, g3.Vset = 40.0, 0.04, 1.01

        for b in self.grid.buses:
            b.Vmax = max(getattr(b, "Vmax", 1.05), 1.05)

    def _randomize_pv_wt(self):
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
        g6 = get_generators_at_bus(self.grid, self.pv_bus)[0]
        g8 = get_generators_at_bus(self.grid, self.wt_bus)[0]
        g6.Pmin, g6.Pmax = 0.0, float(self.pv_pmax_rand) + 1e-10
        g8.Pmin, g8.Pmax = 0.0, float(self.wt_pmax_rand) + 1e-10

    def _randomize_loads(self):
        self.load_scale = float(self.rng.normal(1.0, self.load_jitter))
        self.load_scale = max(0.8, min(1.2, self.load_scale))
        for ld in self.grid.loads:
            ld.P *= self.load_scale
            ld.Q *= self.load_scale

    def _apply_action_to_generators(self, action_vec):

        th2, th3, pv, wt = map(float, action_vec)

        def clamp(x, lo, hi):
            return float(max(lo, min(hi, x)))

        g2 = get_generators_at_bus(self.grid, self.thermal_buses[0])[0]  # Bus2
        g3 = get_generators_at_bus(self.grid, self.thermal_buses[1])[0]  # Bus3
        g6 = get_generators_at_bus(self.grid, self.pv_bus)[0]
        g8 = get_generators_at_bus(self.grid, self.wt_bus)[0]

        (lo2, hi2), (lo3, hi3) = self.th_limits
        th2 = clamp(th2, lo2, hi2)
        th3 = clamp(th3, lo3, hi3)
        pv  = clamp(pv,  0.0, self.pv_pmax_rand)
        wt  = clamp(wt,  0.0, self.wt_pmax_rand)

        def pin(g, p):
            g.Pmin = float(p)
            g.Pmax = float(p) + 1e-10

        pin(g2, th2)
        pin(g3, th3)
        pin(g6, pv)
        pin(g8, wt)

        return np.array([th2, th3, pv, wt], dtype=float)

    # ---------- PF ----------
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
        pf = self.last_pf or self._run_pf()
        bus_df = pf.results.get_bus_df()
        p_col = np.array(bus_df["P"].values, dtype=float)
        demand_like = np.where(p_col < 0.0, -p_col, 0.0)
        if demand_like.shape[0] < 14:
            demand_like = np.pad(demand_like, (0, 14 - demand_like.shape[0]))
        elif demand_like.shape[0] > 14:
            demand_like = demand_like[:14]
        obs = np.concatenate([demand_like, [self.pv_pmax_rand, self.wt_pmax_rand]]).astype(np.float32)
        return obs

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        _, self.grid = load_case14_as_veragrid(run_pp_before=True, sanitize=True, set_line_rate_100=True)

        self._set_generator_costs()

        self._randomize_pv_wt()

        self._randomize_loads()

        self._run_pf()
        self.debug_print_generators()
        return self._build_obs()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float).copy()
        assert action.shape[0] == 4, "action 维度应为 4（Bus2/3 + PV + WT）"

        act_used = self._apply_action_to_generators(action)

        pf = self._run_pf()
        converged = bool(pf.results.converged)

        info = {"action": act_used.tolist()}

        if not converged:
            reward = - self.lambda_diverge
            obs = self._build_obs()
            info.update({
                "converged": False,
                "C_gen": None, "C_loss": None, "C_ov": None,
                "total_cost": None,
                "diverge_penalty": float(self.lambda_diverge),
                "P_loss": None,
                "slack_P": None, "slack_cost": None
            })
            return obs, float(reward), True, info

        C_gen = 0.0
        per_gen = []

        ctrl_buses = [self.thermal_buses[0], self.thermal_buses[1], self.pv_bus, self.wt_bus]  # 2,3,6,8
        for i, bus in enumerate(ctrl_buses):
            g = get_generators_at_bus(self.grid, bus)[0]
            P = float(act_used[i])  # 我们自己锁定的出力
            c1 = float(getattr(g, "Cost", 0.0))
            c2 = float(getattr(g, "Cost2", 0.0))
            cost_i = c1 * P + c2 * (P ** 2)
            C_gen += cost_i
            per_gen.append({"name": getattr(g, "name", f"gen_bus{bus}"), "P": P, "c1": c1, "c2": c2, "cost": cost_i})

        bus_df = pf.results.get_bus_df()
        P_bus  = np.asarray(bus_df["P"], dtype=float)
        P_load = float(-np.sum(P_bus[P_bus < 0.0]))

        S_tot  = np.asarray(pf.results.losses).sum()
        P_loss = float(np.real(S_tot))

        P_known = float(np.sum(act_used))
        P_slack = P_load + P_loss - P_known
        g_slack = get_generators_at_bus(self.grid, self.slack_bus)[0]  # Bus1
        c1_s = float(getattr(g_slack, "Cost", 60.0))
        c2_s = float(getattr(g_slack, "Cost2", 0.0))
        slack_cost = c1_s * P_slack + c2_s * (P_slack ** 2)
        C_gen += slack_cost
        per_gen.append({"name": getattr(g_slack, "name", "slack_bus1"),
                        "P": float(P_slack), "c1": c1_s, "c2": c2_s, "cost": float(slack_cost)})

        C_loss = self.lambda_loss * P_loss

        bdf = pf.results.get_branch_df().copy()
        if "rate_MVA" not in bdf.columns:
            bdf["rate_MVA"] = 100.0  # 与初始化一致
        Sf = np.hypot(bdf["Pf"].astype(float).to_numpy(),
                      bdf["Qf"].astype(float).to_numpy())
        rate = bdf["rate_MVA"].astype(float).to_numpy()
        over = np.maximum(0.0, Sf / np.maximum(rate, 1e-6) - 1.0)
        C_ov = self.lambda_overload * float(np.sum(over ** 2))

        total_cost = C_gen/1000 + C_loss + C_ov/500
        reward = - total_cost

        self._rewrite_pv_wt_limits_only()
        obs = self._build_obs()

        Vm_min = float(bus_df["Vm"].min())
        Vm_max = float(bus_df["Vm"].max())
        branch_loading_pct_max = float(100.0 * np.max(Sf / np.maximum(rate, 1e-6)))

        info.update({
            "converged": True,
            "C_gen": float(C_gen),
            "C_loss": float(C_loss),
            "C_ov": float(C_ov),
            "total_cost": float(total_cost),
            "slack_P": float(P_slack),
            "slack_cost": float(slack_cost),
            "P_loss": float(P_loss),
            "branch_loading_pct_max": branch_loading_pct_max,
            "Vm_min": Vm_min,
            "Vm_max": Vm_max,
            "pv_pmax_rand": float(self.pv_pmax_rand),
            "wt_pmax_rand": float(self.wt_pmax_rand),
            "load_scale": float(self.load_scale),
            "per_generator": per_gen,
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
