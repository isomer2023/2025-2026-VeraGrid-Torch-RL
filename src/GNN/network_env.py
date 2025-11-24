# network_env.py —— 显式切网 + sgen→generator 映射
# GNN观测仅含“当步负荷/出力”，动作直接生效；支持每步重采样与多步回合
# network_env.py -- Explicit network switching + sgen→generator mapping
# GNN observations include only "current step load/output", action takes effect directly.
# Support per-step resampling and multistep rounds
from __future__ import annotations
import numpy as np
import copy
import pandas as pd

import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en

from src.GNN.network_loader import GraphObs
from src.GNN.network_loader import load_simbench_as_veragrid


pd.set_option('future.no_silent_downcasting', True)
# Settings
GLOBAL_SB_CODE = "1-HVMV-urban-2.203-0-no_sw"
USE_ALL_SGEN = True                      # sgen = Generator
RESAMPLE_LOAD_EACH_STEP = True           # Random load adjusting
LOAD_SIGMA = 0.03                        # load sigma
EP_HORIZON = 24                          # ep per round

# 电网控制环境大类
class GridEnv:
    RECOMMENDED_ACT_LIMIT = 1.0  # 策略输出 ∈ [-1,1]

    def __init__(self,
                 lambda_loss=1.0,
                 lambda_overload=5000.0,
                 lambda_diverge=1e4,
                 seed=None,
                 enable_opf_eval=False,
                 opf_solver="NONLINEAR_OPF",
                 sb_code: str | None = None,
                 obs_mode: str = "graph",
                 reward_w_gen: float = 1.0,
                 reward_w_loss: float = 0.01,
                 reward_w_ov: float = 1000.0
                 ):
        self.sb_code = sb_code or GLOBAL_SB_CODE
        self.obs_mode = obs_mode
        self.base_net, self.base_grid = load_simbench_as_veragrid(self.sb_code)
        self.rng = np.random.default_rng(seed)
        self.reward_w_gen = float(reward_w_gen)
        self.reward_w_loss = float(reward_w_loss)
        self.reward_w_ov = float(reward_w_ov)

        # id slack bus
        self.slack_bus_name = None
        try:
            if len(self.base_net.ext_grid):
                eg_bus_idx = int(self.base_net.ext_grid.iloc[0]["bus"])
                try:
                    self.slack_bus_name = str(self.base_net.bus.at[eg_bus_idx, "name"])
                except Exception:
                    self.slack_bus_name = str(eg_bus_idx)
        except Exception:
            pass
        if self.slack_bus_name is None:
            for g in self.base_grid.generators:
                if getattr(g, "is_slack", False) or getattr(g, "slack", False):
                    self.slack_bus_name = str(getattr(getattr(g, "bus", None), "name", "1"))
                    break
        if self.slack_bus_name is None:
            self.slack_bus_name = "1"

        # list sgen
        self.ctrl_gens = []
        for g in self.base_grid.generators:
            gname = str(getattr(g, "name", ""))
            if gname.startswith("sgen_"):
                if float(getattr(g, "Pmax", 0.0)) > float(getattr(g, "Pmin", 0.0)) + 1e-9:
                    self.ctrl_gens.append(g)

        self.action_dim = len(self.ctrl_gens)
        assert self.action_dim > 0, "NO sgen CAN BE CONTROL，CHECK NETWORK OR USE_ALL_SGEN。"

        self.state_dim = self._infer_state_dim()
        self.lambda_loss     = float(lambda_loss)
        self.lambda_overload = float(lambda_overload)
        self.lambda_diverge  = float(lambda_diverge)

        self.static_line_limits = np.array(
            [float(getattr(ln, "rate", 10.0)) if getattr(ln, "rate", None) is not None else 10.0
             for ln in self.base_grid.lines], dtype=float
        )

        self.enable_opf_eval = bool(enable_opf_eval)
        self.opf_solver = opf_solver

        self.grid = None
        self.last_pf = None
        self.t = 0

    def _infer_state_dim(self):
        n_bus = len(self.base_grid.buses)
        return int(n_bus + 3)

    # 负荷重采样（可选：每步扰动）
    def _resample_loads(self, grid):
        if not RESAMPLE_LOAD_EACH_STEP:
            return
        sigma = max(0.0, float(LOAD_SIGMA))
        for ld in grid.loads:
            try:
                baseP = float(getattr(ld, "P", 0.0))
                if baseP <= 0.0:
                    continue
                scale = float(self.rng.normal(1.0, sigma))
                scale = float(np.clip(scale, 0.85, 1.15))
                newP = baseP * scale
                setattr(ld, "P", newP)
            except Exception:
                pass

    # 构造观测空间（graph）
    def _build_obs(self, grid=None):
        g = self.grid if grid is None else grid
        if getattr(self, "obs_mode", "graph") == "graph":
            node_names = [str(b.name) for b in g.buses]
            n = len(node_names)
            name_to_idx = {name: i for i, name in enumerate(node_names)}

            P_load = np.zeros(n, dtype=np.float32)
            for ld in g.loads:
                bi = name_to_idx.get(str(ld.bus.name), None)
                if bi is not None:
                    P_load[bi] += max(float(getattr(ld, "P", 0.0)), 0.0)

            P_cap = np.zeros(n, dtype=np.float32)
            for gg in g.generators:
                gname = str(getattr(gg, "name", ""))
                if gname.startswith("sgen_"):
                    bi = name_to_idx.get(str(getattr(getattr(gg, "bus", None), "name", "")), None)
                    if bi is not None:
                        Pmax = float(getattr(gg, "Pmax", 0.0))
                        P_cap[bi] += max(Pmax, 0.0)

            node_feat = np.stack([P_load, P_cap], axis=1).astype(np.float32)

            branches = g.get_branches(add_vsc=False, add_hvdc=False, add_switch=False)
            ei_src, ei_dst, efeats = [], [], []
            for br in branches:
                f = name_to_idx.get(str(getattr(getattr(br, "bus_from", None), "name", "")), None)
                t = name_to_idx.get(str(getattr(getattr(br, "bus_to",   None), "name", "")), None)
                if f is None or t is None:
                    continue
                rate_val = float(getattr(br, "rate", 0.0) or 0.0)
                ei_src.append(f); ei_dst.append(t); efeats.append([rate_val])
                ei_src.append(t); ei_dst.append(f); efeats.append([rate_val])

            if len(ei_src) == 0:
                edge_index = np.zeros((2, 0), dtype=np.int64)
                edge_feat  = np.zeros((0, 1), dtype=np.float32)
            else:
                edge_index = np.vstack([
                    np.asarray(ei_src, dtype=np.int64),
                    np.asarray(ei_dst, dtype=np.int64)
                ])
                edge_feat  = np.asarray(efeats, dtype=np.float32)

            sgen_map_list, sgen_pmax_list = [], []
            act_min_list, act_max_list = [], []
            for gk in self.ctrl_gens:
                bname = str(getattr(getattr(gk, "bus", None), "name", ""))
                bi = name_to_idx.get(bname, None)
                sgen_map_list.append(-1 if bi is None else bi)
                pmin = float(getattr(gk, "Pmin", 0.0))
                pmax = float(getattr(gk, "Pmax", 0.0))
                sgen_pmax_list.append(pmax)
                act_min_list.append(pmin)
                act_max_list.append(pmax)

            sgen_map  = np.asarray(sgen_map_list,  dtype=np.int64)
            sgen_pmax = np.asarray(sgen_pmax_list, dtype=np.float32)
            act_min   = np.asarray(act_min_list,   dtype=np.float32)
            act_max   = np.asarray(act_max_list,   dtype=np.float32)

            return GraphObs(node_feat=node_feat,
                            edge_index=edge_index,
                            edge_feat=edge_feat,
                            node_names=node_names,
                            sgen_map=sgen_map,
                            sgen_pmax=sgen_pmax,
                            act_min=act_min,
                            act_max=act_max)

        # 向量模式（备用）
        idx_by_name = {str(b.name): i for i, b in enumerate(g.buses)}
        demand_vec = np.zeros(len(g.buses), dtype=float)
        for ld in g.loads:
            bi = idx_by_name.get(str(ld.bus.name))
            if bi is not None:
                demand_vec[bi] += max(float(getattr(ld, "P", 0.0)), 0.0)
        total_load = float(np.sum(demand_vec))
        total_gens = float(len([gg for gg in g.generators if str(getattr(gg, "name","")).startswith("sgen_")]))
        total_sgen = total_gens
        obs = np.concatenate([demand_vec, [total_load, total_gens, total_sgen]]).astype(np.float32)
        return obs

    # reset
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.grid = copy.deepcopy(self.base_grid)

        for g in self.grid.generators:
            bus_name = str(getattr(getattr(g, "bus", None), "name", ""))
            is_slack = (bus_name == self.slack_bus_name) \
                       or bool(getattr(g, "is_slack", False)) \
                       or bool(getattr(g, "slack", False))
            if is_slack:
                g.Cost, g.Cost2, g.Vset = 60.0, 0.0, 1.02
            elif str(getattr(g, "name", "")).startswith("sgen_"):
                g.Cost, g.Cost2, g.Vset = 5.0, 0.0, 1.02
            else:
                g.Cost, g.Cost2, g.Vset = 5.0, 0.0, 1.02

            bn = getattr(g, "bus", None)
            if bn:
                bn.Vmax = max(getattr(bn, "Vmax", 1.05), 1.05)
                bn.Vmin = min(getattr(bn, "Vmin", 0.95), 0.95)

        for ln, cap in zip(self.grid.lines, self.static_line_limits):
            ln.rate = float(cap)

        # 初始 可选 扰动一次负荷
        if RESAMPLE_LOAD_EACH_STEP:
            self._resample_loads(self.grid)

        self._run_pf()
        self.t = 0
        return self._build_obs()

    # 动作映射：[-1,1] 各机组功率
    def _apply_action_to_ctrl_gens(self, action_vec):
        act_used = []
        for u, g in zip(map(float, action_vec), self.ctrl_gens):
            Plo = float(getattr(g, "Pmin", 0.0))
            Phi = float(getattr(g, "Pmax", max(Plo, 0.0)))
            P   = Plo + (np.clip(u, -1, 1) + 1.0) * 0.5 * (Phi - Plo)

            # 关键：把设定出力写回去（多字段兜底）
            for attr in ("P", "Pset", "Pg", "P_sched", "Ptarget"):
                if hasattr(g, attr):
                    try:
                        setattr(g, attr, float(P))
                    except Exception:
                        pass

            # 收紧边界，避免被别的流程改走
            # g.Pmin = float(P)
            # g.Pmax = float(P) + 1e-9

            # 明确不是slack
            if hasattr(g, "is_slack"): g.is_slack = False
            if hasattr(g, "slack"):    g.slack = False

            act_used.append(P)
        return np.asarray(act_used, dtype=float)

    # 潮流计算
    def _run_pf(self):
        pf = gce.PowerFlowDriver(
            grid=self.grid,
            options=gce.PowerFlowOptions(solver_type=en.SolverType.NR, verbose=False),
            opf_results=None
        )
        pf.run()
        self.last_pf = pf
        return pf

    # step
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float)
        assert action.shape[0] == self.action_dim, f"action dim {self.action_dim}"

        # 每步重采样负荷（可选）
        self._resample_loads(self.grid)

        # 1) 映射动作
        act_used = self._apply_action_to_ctrl_gens(action)

        # 2) 跑潮流（失败返回高惩罚值）
        try:
            pf = self._run_pf()
        except Exception as e:
            obs = self._build_obs()
            info = {
                "action": act_used.tolist(),
                "converged": False,
                "error": f"PF exception: {type(e).__name__}: {e}",
                "C_gen": None, "C_loss": None, "C_ov": None,
                "total_cost": None,
                "diverge_penalty": float(self.lambda_diverge),
                "line_monitor": [],
                "reward_breakdown": f"diverged => reward = -{self.lambda_diverge:.6f}"
            }
            return obs, -float(self.lambda_diverge), True, info

        converged = bool(getattr(pf.results, "converged", False))
        info = {"action": act_used.tolist(), "converged": converged}

        if not converged:
            obs = self._build_obs()
            info.update({
                "C_gen": None, "C_loss": None, "C_ov": None,
                "total_cost": None,
                "diverge_penalty": float(self.lambda_diverge),
                "line_monitor": [],
                "reward_breakdown": f"not converged => reward = -{self.lambda_diverge:.6f}"
            })
            return obs, -float(self.lambda_diverge), True, info

        # 3) 成本项
        C_gen = 0.0; C_loss = 0.0; C_ov = 0.0

        # 发电成本使用实际出力计算
        try:
            if hasattr(pf.results, "gen_p"):
                P_arr_actual = np.asarray(pf.results.gen_p, dtype=float)
            elif hasattr(pf.results, "p_mw"):
                P_arr_actual = np.asarray(pf.results.p_mw, dtype=float)
            elif hasattr(pf.results, "gen_P"):
                P_arr_actual = np.asarray(pf.results.gen_P, dtype=float)
            else:
                raise AttributeError("pf.results cannot find gen_p / p_mw / gen_P")

            if len(P_arr_actual) != len(self.grid.generators):
                raise ValueError(f"generation result {len(P_arr_actual)} does not equal generator numbers {len(self.grid.generators)}")

            P_mw = P_arr_actual / 1e6 if np.nanmax(np.abs(P_arr_actual)) > 1e4 else P_arr_actual
            C_tmp = 0.0
            for i, g in enumerate(self.grid.generators):
                P = float(P_mw[i])
                c1 = float(getattr(g, "Cost", 0.0))
                c2 = float(getattr(g, "Cost2", 0.0))
                C_tmp += c1 * P + c2 * (P ** 2)
            C_gen = float(C_tmp)
        except Exception as e:
            info["warn_gen_cost"] = f"gen_cost_fallback: {type(e).__name__}: {e}"
            C_gen = 0.0

        # 线路损耗
        try:
            S_tot = np.asarray(getattr(pf.results, "losses", []), dtype=complex).sum()
            P_loss = float(np.real(S_tot))
            P_loss_mw = P_loss / 1e6 if P_loss > 1e4 else P_loss
            C_loss = float(self.lambda_loss) * P_loss_mw
            info["P_loss"] = P_loss_mw
        except Exception as e:
            info["warn_loss"] = f"loss calc failed: {type(e).__name__}: {e}"
            C_loss = float(self.lambda_loss) * 0.0

        # 过载惩罚
        line_monitor_list = []
        max_loading_pct = 0.0
        overload_penalty_sum = 0.0
        missing_rate_cnt = 0
        try:
            branches = self.grid.get_branches(add_vsc=False, add_hvdc=False, add_switch=False)
            loading_arr = np.asarray(getattr(pf.results, "loading", []))
            loading_abs = np.abs(loading_arr).astype(float)
            for i, br in enumerate(branches):
                ld_pu = float(loading_abs[i]) if i < len(loading_abs) else np.nan
                rate_val = getattr(br, "rate", None)
                rate_val = float(rate_val) if (rate_val is not None and not np.isnan(rate_val)) else np.nan
                if np.isnan(rate_val):
                    missing_rate_cnt += 1

                ld_pct = float(ld_pu * 100.0) if ld_pu == ld_pu else np.nan
                max_loading_pct = max(max_loading_pct, (ld_pct if ld_pct == ld_pct else 0.0))
                over = max(0.0, (ld_pu if ld_pu == ld_pu else 0.0) - 1.0)
                overload_penalty_sum += over ** 2

                fb = getattr(getattr(br, "bus_from", None), "name", f"?{i}")
                tb = getattr(getattr(br, "bus_to", None), "name", f"?{i}")
                flow_est = (ld_pu * rate_val) if (ld_pu == ld_pu and rate_val == rate_val) else None

                line_monitor_list.append({
                    "idx": int(i),
                    "from": fb, "to": tb,
                    "rate_MVA": (None if np.isnan(rate_val) else float(rate_val)),
                    "flow_MVA_est": (None if flow_est is None else float(flow_est)),
                    "loading_pct": (0.0 if ld_pct != ld_pct else float(ld_pct)),
                    "type": type(br).__name__,
                    "name": getattr(br, "name", f"branch_{i}"),
                })
        except Exception as e:
            info["warn_overload"] = f"overload calc failed: {type(e).__name__}: {e}"

        C_ov = float(self.lambda_overload) * float(overload_penalty_sum)
        if missing_rate_cnt > 0:
            info["soft_error_line_rate_missing"] = int(missing_rate_cnt)

        # 汇总奖励
        # total_cost = (float(C_gen) * 0.1) + (float(C_loss) * 1.0) + (float(C_ov) * 0.001) # 可按需调权重
        # 汇总奖励（可配置，靠近传统 OPF：以发电成本为主）
        # C_gen: total generation cost (same units as cost parameters)
        # C_loss: loss cost already computed as lambda_loss * P_loss_mw
        # C_ov: overload penalty computed earlier (already multiplied by lambda_overload)
        total_cost = (float(C_gen) * self.reward_w_gen) \
                     + (float(C_loss) * self.reward_w_loss) \
                     + (float(C_ov) * self.reward_w_ov)
        reward = - total_cost

        # 观测与附加信息
        obs = self._build_obs()
        try:
            bus_df = pf.results.get_bus_df()
            Vm_min = float(bus_df["Vm"].min()); Vm_max = float(bus_df["Vm"].max())
        except Exception:
            Vm_min, Vm_max = np.nan, np.nan

        info.update({
            "C_gen": float(C_gen),
            "C_loss": float(C_loss),
            "C_ov": float(C_ov),
            "total_cost": float(total_cost),
            "Vm_min": Vm_min, "Vm_max": Vm_max,
            "branch_loading_pct_max": float(max_loading_pct),
            "line_monitor": line_monitor_list,
            "reward_breakdown": (
                f"reward = - total_cost = -({C_gen:.6f}*0.1 + {C_loss:.6f}*1.0 + {C_ov:.6f}*0.001) "
                f"= {-reward:.6f}"
            ),
        })

        # 多步回合
        self.t += 1
        done = (self.t >= EP_HORIZON)
        return obs, float(reward), done, info

    def _current_gen_P_array(self):
        P = []
        for g in self.grid.generators:
            P.append(float(getattr(g, "Pmin", 0.0)))
        return np.asarray(P, dtype=float)

# 主外部接口
def make_env(seed: int | None = None,
             sb_code: str | None = None,
             obs_mode: str = "graph") -> GridEnv:
    env = GridEnv(seed=seed,
                     enable_opf_eval=False,
                     opf_solver="NONLINEAR_OPF",
                     sb_code=sb_code,
                     obs_mode=obs_mode)
    env.reset(seed=seed)
    if obs_mode == "graph":
        o = env._build_obs()
        print(f"[Info] Using SimBench: {env.sb_code} | USE_ALL_SGEN={USE_ALL_SGEN} | action_dim={env.action_dim} "
              f"| graph: N={o.node_feat.shape[0]}, F_n={o.node_feat.shape[1]}, E={o.edge_index.shape[1]}, F_e={o.edge_feat.shape[1]}")
    else:
        print(f"[Info] Using SimBench: {env.sb_code} | USE_ALL_SGEN={USE_ALL_SGEN} | action_dim={env.action_dim} | state_dim={env.state_dim} | obs_mode={obs_mode}")
    return env

def get_env_spec(seed: int | None = None,
                 sb_code: str | None = None,
                 obs_mode: str = "graph") -> dict:
    env = make_env(seed=seed, sb_code=sb_code, obs_mode=obs_mode)
    spec = {
        "action_dim": int(env.action_dim),
        "act_limit": float(GridEnv.RECOMMENDED_ACT_LIMIT),
        "sb_code": env.sb_code,
        "use_all_sgen": USE_ALL_SGEN,
        "obs_mode": obs_mode,
    }
    if obs_mode == "graph":
        o = env._build_obs()
        spec.update({
            "N_nodes": int(o.node_feat.shape[0]),
            "node_feat_dim": int(o.node_feat.shape[1]),
            "E_edges": int(o.edge_index.shape[1]),
            "edge_feat_dim": int(o.edge_feat.shape[1]),
        })
    else:
        spec.update({"state_dim": int(env.state_dim)})
    return spec
