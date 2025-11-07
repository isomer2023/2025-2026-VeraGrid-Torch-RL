# ============================
# network_env.py  —— SimBench → VeraGrid 环境（GNN就绪版）
# 关键特性：
# 1) 只在观测中包含“不依赖本步动作/不依赖PF”的量（网表+外生+上一步动作），
#    支持 obs_mode="graph"/"flat"（默认graph，便于后续GNN接入）。
# 2) 动作控制：**全部 sgen**（按 Pmax 降序固定索引），每步把 u∈[-1,1] 映射到各自 [Pmin,Pmax]。
# 3) 奖励：施加动作后再跑 PF（VeraGrid NR），用损耗+越限罚+失稳罚（可选OPF基线只做评测不入奖励）。
# 4) 线路容量：优先用 SimBench 的 max_i_ka 与母线电压 vn_kv 换算 rate_MVA；缺失时兜底。
# 5) 默认使用 SimBench: "1-HV-urban--0-sw"；可改 sb_code。
# ============================

from __future__ import annotations
import copy
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import simbench as sb
import pandapower as pp
import pandas as pd

import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en
import GC_PandaPowerImporter as GC_PandaPowerImporter


# ---------------------------
# 工具函数（清洗/加载）
# ---------------------------

def _clean_nan_fields(net: pp.pandapowerNet) -> None:
    # bus
    if "vn_kv" in net.bus.columns:
        net.bus["vn_kv"] = net.bus["vn_kv"].fillna(110)
    if "in_service" in net.bus.columns:
        net.bus["in_service"] = net.bus["in_service"].fillna(True)
    net.bus = net.bus.fillna(0)

    # line
    if len(net.line):
        for col in ["r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "max_i_ka", "length_km"]:
            if col in net.line.columns:
                net.line[col] = net.line[col].fillna(0)
        if "in_service" in net.line.columns:
            net.line["in_service"] = net.line["in_service"].fillna(True)
        net.line = net.line.fillna(0)

    # trafo
    if len(net.trafo):
        net.trafo = net.trafo.fillna(0)
        if "in_service" in net.trafo.columns:
            net.trafo["in_service"] = net.trafo["in_service"].fillna(True)

    # generic tables
    for elm in ["load", "sgen", "gen", "ext_grid", "storage", "switch"]:
        df = getattr(net, elm, None)
        if df is not None and len(df):
            df = df.fillna(0)
            if "in_service" in df.columns:
                df["in_service"] = df["in_service"].fillna(True)
            setattr(net, elm, df)


def _pandapower_to_veragrid(net_pp: pp.pandapowerNet):
    """先在 pandapower 上跑一次潮流，确保数据自洽，再转 VeraGrid。"""
    # 有些网可能因无机组而 NR 不收敛，这里只做一次尝试，不强制
    try:
        pp.runpp(net_pp, algorithm="nr", numba=False, enforce_q_lims=False)
    except Exception:
        pass
    grid_gc = GC_PandaPowerImporter.PP2GC(net_pp)
    return grid_gc


def _assign_line_rates_from_simbench(net_pp: pp.pandapowerNet, grid_gc) -> np.ndarray:
    """
    用 SimBench 的 max_i_ka + 母线电压 vn_kv 来换算每条线的额定容量：
        rate_MVA = sqrt(3) * V_kV * I_kA
    若数据缺失，稍后用兜底。
    返回与 grid_gc.lines 顺序一致的 rate 数组。
    """
    rates = []
    for ln_pp, ln_gc in zip(net_pp.line.itertuples(), grid_gc.lines):
        try:
            fb = net_pp.bus.at[ln_pp.from_bus, "vn_kv"]
            tb = net_pp.bus.at[ln_pp.to_bus, "vn_kv"]
            vn = float(fb if not np.isnan(fb) else tb)
            ika = float(getattr(ln_pp, "max_i_ka", np.nan))
            if not np.isnan(vn) and not np.isnan(ika) and ika > 0:
                rate = float(np.sqrt(3.0) * vn * ika)
            else:
                rate = np.nan
        except Exception:
            rate = np.nan
        rates.append(rate)
    return np.array(rates, dtype=float)


def load_simbench_as_veragrid(sb_code: str):
    """读取 SimBench 网并转 VeraGrid；仅带当前时刻的静态快照。"""
    net_pp = sb.get_simbench_net(sb_code)
    _clean_nan_fields(net_pp)

    # 给 ext_grid 命名（可选）
    if len(net_pp.ext_grid):
        net_pp.ext_grid.loc[net_pp.ext_grid.index[0], "name"] = "grid_ext"

    grid_gc = _pandapower_to_veragrid(net_pp)
    line_rates = _assign_line_rates_from_simbench(net_pp, grid_gc)

    # 将可用的 rate 写入 VeraGrid；空缺后面兜底
    for ln, cap in zip(grid_gc.lines, line_rates):
        if cap == cap:  # 非 NaN
            ln.rate = float(cap)

    return net_pp, grid_gc, line_rates


# ---------------------------
# Graph 观测格式（GNN用）
# ---------------------------

@dataclass
class GraphObs:
    """
    node_feat: (N_bus, F_node)  不依赖PF的静态/外生/上一动作聚合特征
    edge_index: (2, E)         int64，按 (u_idx, v_idx) 无向/双向各一条
    edge_feat: (E, F_edge)     线路/变压器的静态特征（电气参数/容量等）
    node_types: (N_bus,)       0: PQ/普通; 1: PV/gen_bus; 2: slack/ext_grid
    sgen_map:  (Nsgen,)        第 i 个 sgen 所在母线的 node 索引（按固定降序索引）
    act_min, act_max: (Nsgen,) 每个 sgen 的 [Pmin, Pmax]（MW）
    last_action: (Nsgen,)      上一步已设定的有功（MW），用于马尔可夫性（初始0）
    """
    node_feat: np.ndarray
    edge_index: np.ndarray
    edge_feat: np.ndarray
    node_types: np.ndarray
    sgen_map: np.ndarray
    act_min: np.ndarray
    act_max: np.ndarray
    last_action: np.ndarray


# ---------------------------
# 环境类
# ---------------------------

class GridOPFEnv:
    RECOMMENDED_ACT_LIMIT = 1.0

    def __init__(self,
                 sb_code: str = "1-HV-urban--0-sw",
                 obs_mode: str = "graph",  # "graph" | "flat"
                 load_jitter: float = 0.15,
                 lambda_loss: float = 1.0,
                 lambda_overload: float = 5000.0,
                 lambda_diverge: float = 1e4,
                 seed: Optional[int] = None,
                 enable_opf_eval: bool = True,
                 opf_solver: str = "NONLINEAR_OPF",
                 line_rate_floor: float = 5.0,
                 line_rate_safety: float = 1.10):
        """
        obs_mode='graph'：返回 GraphObs（便于后续GNN接入）；
        obs_mode='flat' ：返回扁平向量（兼容旧SAC，暂不含PF结果）。
        """
        self.sb_code = sb_code
        self.obs_mode = obs_mode
        self.rng = np.random.default_rng(seed)

        # 读取网
        self.base_net, self.base_grid, self.sb_line_rates = load_simbench_as_veragrid(sb_code)

        # 线路热限兜底：若缺失，用 floor 或安全系数扩充（这里轻兜底）
        self.static_line_limits = []
        for ln, cap in zip(self.base_grid.lines, self.sb_line_rates):
            if not (cap == cap):  # NaN
                cap = line_rate_floor
            cap = float(max(line_rate_floor, cap * line_rate_safety))
            self.static_line_limits.append(cap)
        self.static_line_limits = np.array(self.static_line_limits, dtype=float)

        # 控制对象：全部 sgen（按 Pmax 降序固定索引）
        self.sgen_list_pp = self._build_sgen_table(self.base_net)  # pandas DataFrame 附加 min/max
        self.sgen_index = np.argsort(-self.sgen_list_pp["max_p_mw"].to_numpy())  # 降序
        self.sgen_list_pp = self.sgen_list_pp.iloc[self.sgen_index].reset_index(drop=True)
        self.Nsgen = int(len(self.sgen_list_pp))

        # 扁平观测维度（仅当 obs_mode='flat' 用到；不含 PF）
        # 这里给一个稳定、信息充分的拼接：
        #   loads(P/Q聚合到bus, 2*Nbus) + sgen(min,max归一到系统规模, 2*Nsgen) + onehot(bus类别, Nbus)
        # 实际返回时动态生成，不预设死值。
        self.flat_dim_cache = None

        # 奖励项系数
        self.lambda_loss = float(lambda_loss)
        self.lambda_overload = float(lambda_overload)
        self.lambda_diverge = float(lambda_diverge)

        # 其他
        self.enable_opf_eval = bool(enable_opf_eval)
        self.opf_solver = opf_solver

        # 当前场景 + 上一步动作（MW）
        self.grid = None
        self.last_pf = None
        self.last_action_mw = np.zeros(self.Nsgen, dtype=float)

        # 载荷扰动参数
        self.load_jitter = float(load_jitter)

        # Gym 需要的两个规格
        self.action_dim = self.Nsgen
        self.state_dim = None  # graph 模式不使用；flat 模式运行时动态计算

    # ---------- 构造 sgen 表：补 min/max、定位 bus ----------
    def _build_sgen_table(self, net_pp: pp.pandapowerNet) -> pd.DataFrame:
        sgen = net_pp.sgen.copy()
        if "min_p_mw" not in sgen.columns:
            sgen["min_p_mw"] = 0.0
        if "max_p_mw" not in sgen.columns:
            # 没有装机列时，用 |p_mw| 作为装机上限的近似
            sgen["max_p_mw"] = sgen["p_mw"].abs().clip(lower=0.0)
        # 记录 bus index
        sgen["bus_idx"] = sgen["bus"].astype(int).values
        return sgen[["bus_idx", "min_p_mw", "max_p_mw"]].reset_index(drop=True)

    # ---------- 映射 VeraGrid 元素 ----------
    @staticmethod
    def _get_bus_by_name(grid_gc, name: str | int):
        key = str(name)
        for b in grid_gc.buses:
            if b.name == key:
                return b
        raise ValueError(f"Bus {name} not found")

    # ---------- 载荷扰动：独立采样静态工况（不跑PF即可生成观测） ----------
    def _randomize_loads(self, grid=None):
        g = self.grid if grid is None else grid
        scale = float(self.rng.normal(1.0, self.load_jitter))
        scale = max(0.8, min(1.2, scale))
        for ld in g.loads:
            ld.P *= scale
            ld.Q *= scale
        return scale

    # ---------- 动作映射：[-1,1] → [Pmin,Pmax]（全部 sgen） ----------
    def _apply_action_all_sgen(self, action_vec: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(action_vec, dtype=float), -1.0, 1.0)
        applied = np.zeros_like(u, dtype=float)

        # 注意：VeraGrid 中 sgen 也被建为 generator；这里按“装机从大到小”找到匹配的 Nsgen 台
        gens_sorted = sorted(self.grid.generators, key=lambda g: getattr(g, "Pmax", 0.0), reverse=True)
        if len(gens_sorted) < self.Nsgen:
            raise RuntimeError("VeraGrid generators fewer than sgen count; please check PP2GC mapping.")

        for i in range(self.Nsgen):
            g = gens_sorted[i]
            Pmin = float(getattr(g, "Pmin", 0.0))
            Pmax = float(getattr(g, "Pmax", max(Pmin, 0.0) + 1.0))
            Pi = Pmin + (u[i] + 1.0) * 0.5 * (Pmax - Pmin)
            # 钉死有功
            g.Pmin = float(Pi)
            g.Pmax = float(Pi) + 1e-10
            applied[i] = Pi

        self.last_action_mw = applied.copy()
        return applied

    # ---------- PF / OPF ----------
    def _run_pf(self):
        pf = gce.PowerFlowDriver(
            grid=self.grid,
            options=gce.PowerFlowOptions(solver_type=en.SolverType.NR, verbose=False),
            opf_results=None
        )
        pf.run()
        self.last_pf = pf
        return pf

    def _run_opf_eval(self) -> Tuple[bool, Dict]:
        try:
            grid_opf = copy.deepcopy(self.grid)
            for ln, cap in zip(grid_opf.lines, self.static_line_limits):
                ln.rate = float(cap)
            # 用当前 cost/Vset（本环境未复杂设置成本，只作评测用）
            solver = getattr(gce.SolverType, self.opf_solver)
            opf_opts = gce.OptimalPowerFlowOptions(
                solver=solver,
                power_flow_options=gce.PowerFlowOptions(solver_type=en.SolverType.NR, verbose=False)
            )
            opf = gce.OptimalPowerFlowDriver(grid=grid_opf, options=opf_opts)
            opf.run()

            pf_drv = gce.PowerFlowDriver(grid=grid_opf,
                                         options=gce.PowerFlowOptions(solver_type=en.SolverType.NR, verbose=False),
                                         opf_results=opf.results)
            pf_drv.run()

            P_arr = np.asarray(opf.results.generator_power, dtype=float)
            C_gen = 0.0
            for idx, g in enumerate(grid_opf.generators):
                P = float(P_arr[idx])
                c1 = float(getattr(g, "Cost", 0.0))
                c2 = float(getattr(g, "Cost2", 0.0))
                C_gen += c1 * P + c2 * (P ** 2)

            S_tot = np.asarray(pf_drv.results.losses).sum()
            P_loss = float(np.real(S_tot))
            total_cost = C_gen / 100.0 + self.lambda_loss * P_loss

            return True, {
                "opf_converged": bool(pf_drv.results.converged),
                "opf_C_gen": float(C_gen),
                "opf_P_loss": float(P_loss),
                "opf_total_cost": float(total_cost),
            }
        except Exception as e:
            return False, {"opf_error": str(e)}

    # ---------- Graph 观测构建（不依赖PF） ----------
    def _build_graph_obs(self) -> GraphObs:
        # 节点：母线；节点类型：0 PQ/普通, 1 PV/gen_bus, 2 slack/ext_grid
        bus_df = self.base_net.bus
        Nbus = int(len(bus_df))

        # node_types（简化判定：ext_grid 所在 bus → 2；有 gen（或被PP2GC映射的gen）→1；其余0）
        node_types = np.zeros(Nbus, dtype=np.int64)
        if len(self.base_net.ext_grid):
            eg_bus = int(self.base_net.ext_grid.iloc[0]["bus"])
            node_types[eg_bus] = 2
        if len(self.base_net.gen):
            for _, row in self.base_net.gen.iterrows():
                node_types[int(row["bus"])] = np.maximum(node_types[int(row["bus"])], 1)

        # 聚合到母线的负荷 (P/Q)
        load_P = np.zeros(Nbus, dtype=float)
        load_Q = np.zeros(Nbus, dtype=float)
        if len(self.base_net.load):
            for _, row in self.base_net.load.iterrows():
                b = int(row["bus"])
                load_P[b] += float(row.get("p_mw", 0.0))
                load_Q[b] += float(row.get("q_mvar", 0.0))

        # 聚合到母线的 sgen 上限（静态装机）
        sgen_max_at_bus = np.zeros(Nbus, dtype=float)
        for _, row in self.sgen_list_pp.iterrows():
            b = int(row["bus_idx"])
            sgen_max_at_bus[b] += float(row["max_p_mw"])

        # 上一步 sgen 动作聚合（MW）
        last_act_at_bus = np.zeros(Nbus, dtype=float)
        for i, row in self.sgen_list_pp.iterrows():
            b = int(row["bus_idx"])
            last_act_at_bus[b] += float(self.last_action_mw[i])

        # 归一尺度
        P_sys = max(load_P.sum(), 1e-6)
        node_feat = np.stack([
            load_P / P_sys,
            load_Q / (P_sys + 1e-6),
            sgen_max_at_bus / (P_sys + 1e-6),
            last_act_at_bus / (P_sys + 1e-6),
            bus_df["vn_kv"].to_numpy(dtype=float) / 110.0  # 粗略归一，HV网多为 ~110kV
        ], axis=1)  # (Nbus, 5)

        # 边：线路 + 变压器
        edges_u, edges_v, edge_feat_list = [], [], []
        # line
        for ln in self.base_net.line.itertuples():
            u = int(ln.from_bus); v = int(ln.to_bus)
            r = float(getattr(ln, "r_ohm_per_km", 0.0))
            x = float(getattr(ln, "x_ohm_per_km", 0.0))
            c = float(getattr(ln, "c_nf_per_km", 0.0))
            L = float(getattr(ln, "length_km", 0.0))
            # 对应 VeraGrid 写入的 rate
            # 因为转换后 line 顺序一致，这里可用 sb_line_rates；统一再归一：
            idx = ln.Index
            cap = self.sb_line_rates[idx] if idx < len(self.sb_line_rates) else np.nan
            if not (cap == cap):
                cap = 5.0
            cap_n = float(max(5.0, cap)) / 100.0  # 100MVA为粗归一

            # 无向→双向
            edges_u += [u, v]; edges_v += [v, u]
            edge_feat_list += [
                [r, x, c, L, cap_n],
                [r, x, c, L, cap_n]
            ]

        # trafo（当作边；参数简单放标幺/容量近似）
        if len(self.base_net.trafo):
            for tr in self.base_net.trafo.itertuples():
                u = int(tr.hv_bus); v = int(tr.lv_bus)
                sn_mva = float(getattr(tr, "sn_mva", 40.0))
                vk = float(getattr(tr, "vk_percent", 10.0)) / 100.0
                vkr = float(getattr(tr, "vkr_percent", 0.5)) / 100.0
                cap_n = sn_mva / 100.0
                edges_u += [u, v]; edges_v += [v, u]
                edge_feat_list += [
                    [0.0, vk, vkr, 0.0, cap_n],
                    [0.0, vk, vkr, 0.0, cap_n]
                ]

        edge_index = np.stack([np.asarray(edges_u, dtype=np.int64),
                               np.asarray(edges_v, dtype=np.int64)], axis=0)
        edge_feat = np.asarray(edge_feat_list, dtype=np.float32)

        # sgen_map & 动作区间
        sgen_map = self.sgen_list_pp["bus_idx"].to_numpy(dtype=np.int64)
        act_min = self.sgen_list_pp["min_p_mw"].to_numpy(dtype=np.float32)
        act_max = self.sgen_list_pp["max_p_mw"].to_numpy(dtype=np.float32)

        return GraphObs(
            node_feat=node_feat.astype(np.float32),
            edge_index=edge_index,
            edge_feat=edge_feat,
            node_types=node_types,
            sgen_map=sgen_map,
            act_min=act_min,
            act_max=act_max,
            last_action=self.last_action_mw.astype(np.float32)
        )

    # ---------- Flat 观测（不依赖PF；用于兼容旧SAC，之后可弃） ----------
    def _build_flat_obs(self) -> np.ndarray:
        bus_df = self.base_net.bus
        Nbus = int(len(bus_df))
        # load 聚合
        load_P = np.zeros(Nbus, dtype=float)
        load_Q = np.zeros(Nbus, dtype=float)
        if len(self.base_net.load):
            for _, row in self.base_net.load.iterrows():
                b = int(row["bus"])
                load_P[b] += float(row.get("p_mw", 0.0))
                load_Q[b] += float(row.get("q_mvar", 0.0))
        P_sys = max(load_P.sum(), 1e-6)

        # node one-hot（ext_grid 简化为一列标记）
        node_types = np.zeros(Nbus, dtype=np.float32)
        if len(self.base_net.ext_grid):
            eg_bus = int(self.base_net.ext_grid.iloc[0]["bus"])
            node_types[eg_bus] = 1.0

        # sgen min/max（按固定降序索引）
        sgen_min = self.sgen_list_pp["min_p_mw"].to_numpy(dtype=np.float32) / (P_sys + 1e-6)
        sgen_max = self.sgen_list_pp["max_p_mw"].to_numpy(dtype=np.float32) / (P_sys + 1e-6)

        flat = np.concatenate([
            load_P / P_sys,
            load_Q / (P_sys + 1e-6),
            node_types,                      # (Nbus,)
            sgen_min, sgen_max,              # (2*Nsgen,)
            self.last_action_mw / (P_sys + 1e-6)
        ], axis=0).astype(np.float32)

        self.flat_dim_cache = int(flat.shape[0])
        return flat

    # ---------- 观测入口 ----------
    def _build_obs(self):
        if self.obs_mode == "graph":
            return self._build_graph_obs()
        else:
            return self._build_flat_obs()

    # ---------- reset：独立采样静态工况（不跑PF即可生成观测） ----------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 构场景
        self.grid = copy.deepcopy(self.base_grid)

        # 线路容量写入
        for ln, cap in zip(self.grid.lines, self.static_line_limits):
            ln.rate = float(cap)

        # 载荷扰动（对 VeraGrid 施加）
        self._randomize_loads(self.grid)

        # 上一步动作清零（MW）
        self.last_action_mw = np.zeros(self.Nsgen, dtype=float)

        # 不跑 PF 就能给出观测
        obs = self._build_obs()

        # 为 step 奖励准备：需要在 step 里施加动作后才跑 PF
        self.last_pf = None
        return obs

    # ---------- step：施加动作→跑PF→算奖励→（单步任务）done=True ----------
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float).reshape(-1)
        assert action.shape[0] == self.Nsgen, f"action 维度应为 Nsgen={self.Nsgen}"

        # 施加动作
        act_used = self._apply_action_all_sgen(action)

        # 跑 PF
        pf = self._run_pf()
        converged = bool(pf.results.converged)

        info = {"action_mw": act_used.tolist(), "converged": converged}

        if not converged:
            reward = - self.lambda_diverge
            obs_next = self._build_obs()  # 单步任务无所谓
            info.update({
                "C_loss": None, "C_ov": None, "total_cost": None,
                "diverge_penalty": float(self.lambda_diverge),
                "line_monitor": []
            })
            if self.enable_opf_eval:
                ok, opf_info = self._run_opf_eval()
                info.update(opf_info)
            return obs_next, float(reward), True, info

        # 计算损耗
        S_tot = np.asarray(pf.results.losses).sum()
        P_loss = float(np.real(S_tot))
        C_loss = self.lambda_loss * P_loss

        # 过载罚（用 loading > 1.0 的平方和）
        branches = self.grid.get_branches(add_vsc=False, add_hvdc=False, add_switch=False)
        loading_arr = np.abs(pf.results.loading)
        line_monitor = []
        overload_penalty = 0.0
        max_loading_pct = 0.0

        for i, br in enumerate(branches):
            ld_pu = float(loading_arr[i])
            ld_pct = ld_pu * 100.0
            if ld_pct > max_loading_pct:
                max_loading_pct = ld_pct
            over = max(0.0, ld_pu - 1.0)
            overload_penalty += over ** 2

            rate_val = float(getattr(br, "rate", np.nan))
            flow_est = float(ld_pu * rate_val) if rate_val == rate_val else np.nan  # NaN检查

            fb = getattr(getattr(br, "bus_from", None), "name", f"?{i}")
            tb = getattr(getattr(br, "bus_to", None), "name", f"?{i}")
            line_monitor.append({
                "idx": int(i),
                "from": fb, "to": tb,
                "rate_MVA": rate_val,
                "flow_MVA_est": flow_est,
                "loading_pct": ld_pct,
                "type": type(br).__name__,
                "name": getattr(br, "name", f"branch_{i}")
            })

        C_ov = self.lambda_overload * float(overload_penalty)

        # 本例未计发电成本（多数 sgen 设成本~0），仅以损耗+越限为总成本
        total_cost = C_loss + C_ov / 500.0
        reward = - total_cost

        # obs（单步任务无所谓）+ info
        obs_next = self._build_obs()
        info.update({
            "C_loss": float(C_loss),
            "C_ov": float(C_ov),
            "total_cost": float(total_cost),
            "P_loss": float(P_loss),
            "branch_loading_pct_max": float(max_loading_pct),
            "line_monitor": line_monitor
        })

        if self.enable_opf_eval:
            ok, opf_info = self._run_opf_eval()
            info.update(opf_info)
            if ok and ("opf_total_cost" in opf_info) and (info["total_cost"] is not None):
                info["optimality_gap"] = (info["total_cost"] - opf_info["opf_total_cost"]) / max(1e-6, opf_info["opf_total_cost"])

        done = True  # 单步任务
        return obs_next, float(reward), done, info

    # ---------- 兼容工具 ----------
    def debug_print_generators(self):
        print("=== DEBUG GEN LIST (Top by Pmax) ===")
        gens_sorted = sorted(self.grid.generators, key=lambda g: getattr(g, "Pmax", 0.0), reverse=True)
        for i, g in enumerate(gens_sorted[:min(10, len(gens_sorted))]):
            print(i, {
                "name": getattr(g, "name", None),
                "bus": getattr(getattr(g, "bus", None), "name", None),
                "Pmin": getattr(g, "Pmin", None),
                "Pmax": getattr(g, "Pmax", None),
                "Vset": getattr(g, "Vset", None),
                "Cost": getattr(g, "Cost", None),
                "Cost2": getattr(g, "Cost2", None)
            })


# ---------------------------
# 工厂函数
# ---------------------------

def make_env(seed: Optional[int] = None,
             sb_code: str = "1-HV-urban--0-sw",
             obs_mode: str = "graph") -> GridOPFEnv:
    env = GridOPFEnv(seed=seed, sb_code=sb_code, obs_mode=obs_mode,
                     enable_opf_eval=True, opf_solver="NONLINEAR_OPF")
    env.reset(seed=seed)
    return env


def get_env_spec(seed: Optional[int] = None,
                 sb_code: str = "1-HV-urban--0-sw",
                 obs_mode: str = "graph") -> dict:
    env = make_env(seed=seed, sb_code=sb_code, obs_mode=obs_mode)
    spec = {
        "obs_mode": obs_mode,
        "action_dim": int(env.action_dim),
        "act_limit": float(GridOPFEnv.RECOMMENDED_ACT_LIMIT),
    }
    if obs_mode == "flat":
        # 动态计算维度
        obs = env._build_flat_obs()
        spec["state_dim"] = int(obs.shape[0])
    else:
        # graph 模式不返回单一 state_dim；交由 GNN 处理
        spec["state_dim"] = None
    return spec


__all__ = [
    "GridOPFEnv",
    "make_env",
    "get_env_spec",
]
