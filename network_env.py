# ============================
# network_env.py —— 显式切网 + sgen→generator 一对一映射（可开关）
# ============================
from __future__ import annotations
import numpy as np
import copy

import simbench as sb
import pandapower as pp

import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en
import GC_PandaPowerImporter as GC_PandaPowerImporter

# ========= 显式切换 SimBench 网络（只改这里）=========
GLOBAL_SB_CODE = "1-MV-semiurb--1-sw"
# 例如：
# GLOBAL_SB_CODE = "1-HV-mixed--1-sw"
# GLOBAL_SB_CODE = "1-MV-semiurb--1-sw"
# GLOBAL_SB_CODE = "1-MV-rural--0-sw"
# ===================================================

# ========= 控制策略开关 =========
# True : 一对一控制 PP.sgen（通过把每个 sgen 转成 VeraGrid 的 Generator）
# False: 只控制 VeraGrid 自带的机组（排除 slack），sgen 作为固定注入
USE_ALL_SGEN = True
# =================================


# ---------------------------
# Pandapower 清洗
# ---------------------------
def _clean_nan_fields(net):
    # bus
    if 'vn_kv' in net.bus.columns:
        s = net.bus['vn_kv'].copy()
        s = s.fillna(110)
        net.bus['vn_kv'] = s
    if 'in_service' in net.bus.columns:
        s = net.bus['in_service'].copy()
        s = s.fillna(True)
        net.bus['in_service'] = s
    net.bus = net.bus.fillna(0)

    # line
    if len(net.line):
        for col in ['r_ohm_per_km','x_ohm_per_km','c_nf_per_km','max_i_ka','length_km']:
            if col in net.line.columns:
                s = net.line[col].copy()
                s = s.fillna(0)
                net.line[col] = s
        if 'in_service' in net.line.columns:
            s = net.line['in_service'].copy()
            s = s.fillna(True)
            net.line['in_service'] = s
        net.line = net.line.fillna(0)

    # trafo
    if len(net.trafo):
        net.trafo = net.trafo.fillna(0)
        if 'in_service' in net.trafo.columns:
            s = net.trafo['in_service'].copy()
            s = s.fillna(True)
            net.trafo['in_service'] = s

    # 其他元素
    for elm in ['load','sgen','gen','ext_grid','storage','switch']:
        df = getattr(net, elm, None)
        if df is not None and len(df):
            df = df.fillna(0)
            if 'in_service' in df.columns:
                s = df['in_service'].copy()
                s = s.fillna(True)
                df['in_service'] = s
            setattr(net, elm, df)


# ---------------------------
# PP → VeraGrid 转换 + 线路额定值 + sgen 映射
# ---------------------------
def load_simbench_as_veragrid(sb_code: str):
    """读取 SimBench 网并转 VeraGrid；做：
       1) 清洗 + 先在 pandapower 上跑一次潮流（自洽性检查）
       2) PP→VeraGrid（使用你已有的 PP2GC）
       3) 设置线路热限（有 max_i_ka 就按 √3·V(kV)·I(kA)）
       4) 把 PP.sgen 一对一映射成 VeraGrid.Generator（可控），用于控制所有分布式电源
    """
    net_pp = sb.get_simbench_net(sb_code)

    _clean_nan_fields(net_pp)

    # 可选：给第一个外部电源一个更清晰的名字
    if len(net_pp.ext_grid):
        net_pp.ext_grid.at[net_pp.ext_grid.index[0], 'name'] = "grid_ext"

    # 先用 pandapower 跑一次潮流，尽早发现问题
    pp.runpp(net_pp, algorithm='nr', numba=False, enforce_q_lims=False)

    # PP → VeraGrid
    grid_gc = GC_PandaPowerImporter.PP2GC(net_pp)

    # 线路热限（若 PP 有 max_i_ka 和母线电压）
    try:
        for ln_pp, ln_gc in zip(net_pp.line.itertuples(), grid_gc.lines):
            if hasattr(ln_pp, 'max_i_ka') and hasattr(ln_pp, 'length_km'):
                fb = net_pp.bus.at[ln_pp.from_bus, 'vn_kv']
                tb = net_pp.bus.at[ln_pp.to_bus, 'vn_kv']
                vn = float(fb if not np.isnan(fb) else tb)
                if not np.isnan(getattr(ln_pp, 'max_i_ka', np.nan)) and not np.isnan(vn):
                    ln_gc.rate = float(np.sqrt(3.0) * vn * ln_pp.max_i_ka)  # MVA
    except Exception:
        pass  # 没有额定电流就不改

    # ===== sgen → generator 映射（可控）=====
    # 我们建立一个 PP bus 索引/名称 → VG bus 的查找表
    name_to_bus = {}
    for b in grid_gc.buses:
        key = str(getattr(b, "name", ""))
        name_to_bus[key] = b
        # 兼容：有些转换把 bus.name 设为纯数字字符串
        try:
            name_to_bus[str(int(key))] = b
        except:
            pass

    created = 0
    try:
        sgen_df = net_pp.sgen.copy()
        if len(sgen_df):
            # 个别网没有 bus 名称列；我们用索引做回退
            has_bus_name = 'name' in net_pp.bus.columns
            for sid, row in sgen_df.iterrows():
                bus_idx = int(row["bus"])
                pp_bus_name = str(net_pp.bus.at[bus_idx, "name"]) if has_bus_name else str(bus_idx)
                b_gc = name_to_bus.get(pp_bus_name) or name_to_bus.get(str(bus_idx))
                if b_gc is None:
                    continue  # 找不到就跳过该 sgen

                p_mw = float(row.get("p_mw", 0.0))
                pmax = max(p_mw, 0.0)

                # 构造一个 VeraGrid 的 Generator，并挂到该母线
                g = gce.Generator()
                g.name = f"sgen_{sid}"
                g.bus = b_gc
                g.Pmin, g.Pmax = 0.0, pmax + 1e-9   # 可从 0 到该 sgen 名义出力
                g.Cost, g.Cost2 = 0.0, 0.0          # 可再按需设成本
                g.Vset = getattr(b_gc, "Vset", 1.0)
                grid_gc.generators.append(g)
                created += 1
        print(f"[Info] Created {created} controllable generators from PP.sgen.")
    except Exception as e:
        print(f"[Warn] sgen→generator mapping skipped: {e}")

    return net_pp, grid_gc


# ---------------------------
# 工具函数
# ---------------------------
def get_bus_by_name(grid_gc, bus_name: str | int):
    key = str(bus_name)
    for b in grid_gc.buses:
        if b.name == key:
            return b
    raise ValueError(f"Bus {bus_name} not found")


# ---------------------------
# 环境类
# ---------------------------
class GridOPFEnv:
    RECOMMENDED_ACT_LIMIT = 1.0  # 策略输出 ∈ [-1,1]

    def __init__(self,
                 lambda_loss=1.0,
                 lambda_overload=5000.0,
                 lambda_diverge=1e4,
                 seed=None,
                 enable_opf_eval=False,
                 opf_solver="NONLINEAR_OPF",
                 ):
        # 1) 显式网络选择
        self.sb_code = GLOBAL_SB_CODE
        self.base_net, self.base_grid = load_simbench_as_veragrid(self.sb_code)
        self.rng = np.random.default_rng(seed)

        # 2) 识别 slack（简单按 bus name == "1"；也可查 is_slack 属性）
        self.slack_bus_name = "1"

        # 3) 构造“可控机组列表”
        #    - 若 USE_ALL_SGEN=True：只挑 sgen_* 这些我们刚创建的机组（可控分布式）
        #    - 否则：挑非 slack 的原生 generator
        self.ctrl_gens = []
        for g in self.base_grid.generators:
            gname = str(getattr(g, "name", ""))
            busname = str(getattr(getattr(g, "bus", None), "name", ""))
            is_slack = (busname == self.slack_bus_name) \
                       or getattr(g, "is_slack", False) \
                       or getattr(g, "slack", False)
            if USE_ALL_SGEN:
                if gname.startswith("sgen_"):  # 只控我们映射出来的 sgen 机组
                    self.ctrl_gens.append(g)
            else:
                if (not is_slack) and (not gname.startswith("sgen_")):
                    self.ctrl_gens.append(g)

        self.action_dim = len(self.ctrl_gens)
        assert self.action_dim > 0, \
            f"没有可控机组（当前 USE_ALL_SGEN={USE_ALL_SGEN}），检查网络 {self.sb_code} 或改开关。"

        # 4) 状态定义（不含潮流依赖项，避免“动作→潮流→再观测”的闭环）
        #    使用：各母线有功负荷向量 + 统计量（总负荷、机组数、sgen数）
        self.state_dim = self._infer_state_dim()

        # 5) 惩罚权重
        self.lambda_loss     = float(lambda_loss)      # 线路损耗权重
        self.lambda_overload = float(lambda_overload)  # 过载惩罚权重
        self.lambda_diverge  = float(lambda_diverge)   # 潮流不收敛惩罚

        # 6) 线路静态热限（若前面已从 PP 赋值，这里只做“兜底”）
        self.static_line_limits = np.array(
            [float(getattr(ln, "rate", 10.0)) if getattr(ln, "rate", None) is not None else 10.0
             for ln in self.base_grid.lines], dtype=float
        )

        # 7) OPF 评测（可选）
        self.enable_opf_eval = bool(enable_opf_eval)
        self.opf_solver = opf_solver

        # 8) 运行期变量
        self.grid = None
        self.last_pf = None

    # ---------- 内部：状态维度 ----------
    def _infer_state_dim(self):
        # 按 bus 序号构造负荷向量长度（假设母线命名为 "1"..."N"）
        bus_ids = []
        for b in self.base_grid.buses:
            try:
                bus_ids.append(int(str(b.name)))
            except:
                pass
        max_bus = max(bus_ids) if bus_ids else len(self.base_grid.buses)

        # 负荷向量（长度 = 最大母线号），再加3个统计量
        return int(max_bus + 3)

    # ---------- 构造观测 ----------
    def _build_obs(self, grid=None):
        g = self.grid if grid is None else grid

        # 1) 母线负荷向量（有功 P>0 记到该母线；按 name 当作索引）
        #    （VeraGrid 的 bus P>0/负号方向需以 results 为准；这里用 loads 的 P 聚合）
        demand_map = {}
        for ld in g.loads:
            bus_name = str(ld.bus.name)
            P_load = float(getattr(ld, "P", 0.0))
            demand_map[bus_name] = demand_map.get(bus_name, 0.0) + max(P_load, 0.0)

        # 推断最大母线号
        bus_ids = []
        for b in g.buses:
            try:
                bus_ids.append(int(str(b.name)))
            except:
                pass
        max_bus = max(bus_ids) if bus_ids else len(g.buses)

        demand_vec = []
        for i in range(1, max_bus + 1):
            demand_vec.append(demand_map.get(str(i), 0.0))
        demand_vec = np.array(demand_vec, dtype=float)

        # 2) 总负荷 + 机组数 + sgen（映射后）数
        total_load = float(np.sum(demand_vec))
        total_gens = float(len([g for g in self.base_grid.generators if not getattr(g, "slack", False)]))
        total_sgen = float(len([g for g in self.base_grid.generators if str(getattr(g, "name","")).startswith("sgen_")]))

        obs = np.concatenate([demand_vec, [total_load, total_gens, total_sgen]]).astype(np.float32)
        return obs

    # ---------- 建场景 ----------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 每个 episode 用同一 base_grid 的深拷贝（静态样本）
        self.grid = copy.deepcopy(self.base_grid)

        # 给所有机组设置成本/电压上限（可按需微调）
        for g in self.grid.generators:
            if str(getattr(g.bus, "name", "")) == self.slack_bus_name:
                g.Cost, g.Cost2, g.Vset = 60.0, 0.0, 1.02
            else:
                # sgen_* 给低成本；传统机组稍高
                if str(getattr(g, "name","")).startswith("sgen_"):
                    g.Cost, g.Cost2, g.Vset = 1.0, 0.0, 1.02
                else:
                    g.Cost, g.Cost2, g.Vset = 35.0, 0.02, 1.02

            # 电压上限略放宽
            bn = getattr(g, "bus", None)
            if bn:
                bn.Vmax = max(getattr(bn, "Vmax", 1.05), 1.05)

        # 线路热限（兜底）
        for ln, cap in zip(self.grid.lines, self.static_line_limits):
            ln.rate = float(cap)

        # 初次 PF（不参与奖励，只为检查）
        self._run_pf()

        return self._build_obs()

    # ---------- 动作映射：[-1,1] → 各机组 Pmin~Pmax ----------
    def _apply_action_to_ctrl_gens(self, action_vec):
        act_used = []
        for u, g in zip(map(float, action_vec), self.ctrl_gens):
            Plo = float(getattr(g, "Pmin", 0.0))
            Phi = float(getattr(g, "Pmax", max(Plo, 0.0)))
            P   = Plo + (np.clip(u, -1, 1) + 1.0) * 0.5 * (Phi - Plo)
            # 固定该机组出力
            g.Pmin = float(P)
            g.Pmax = float(P) + 1e-10
            act_used.append(P)
        return np.asarray(act_used, dtype=float)

    # ---------- 跑潮流 ----------
    def _run_pf(self):
        pf = gce.PowerFlowDriver(
            grid=self.grid,
            options=gce.PowerFlowOptions(solver_type=en.SolverType.NR, verbose=False),
            opf_results=None
        )
        pf.run()
        self.last_pf = pf
        return pf

    # ---------- 一步交互 ----------
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float)
        assert action.shape[0] == self.action_dim, f"action 维度应为 {self.action_dim}"

        act_used = self._apply_action_to_ctrl_gens(action)
        pf = self._run_pf()
        converged = bool(pf.results.converged)

        info = {"action": act_used.tolist(), "converged": converged}

        if not converged:
            reward = - self.lambda_diverge
            obs = self._build_obs()
            info.update({
                "C_gen": None, "C_loss": None, "C_ov": None,
                "total_cost": None,
                "diverge_penalty": float(self.lambda_diverge),
                "line_monitor": [],
            })
            return obs, float(reward), True, info

        # ===== 成本：发电成本（线性+二次）+ 线路损耗 + 过载惩罚 =====
        # 发电成本
        P_arr = self._current_gen_P_array()  # 与 self.grid.generators 对齐的出力数组
        C_gen = 0.0
        for i, g in enumerate(self.grid.generators):
            P = float(P_arr[i])
            c1 = float(getattr(g, "Cost", 0.0))
            c2 = float(getattr(g, "Cost2", 0.0))
            C_gen += c1 * P + c2 * (P ** 2)

        # 线路损耗
        S_tot = np.asarray(pf.results.losses).sum()
        P_loss = float(np.real(S_tot))
        C_loss = self.lambda_loss * P_loss

        # 过载惩罚
        branches = self.grid.get_branches(add_vsc=False, add_hvdc=False, add_switch=True)
        loading_arr = np.abs(pf.results.loading)
        line_monitor_list = []
        overload_penalty_sum = 0.0
        max_loading_pct = 0.0

        for i, br in enumerate(branches):
            rate_val = float(getattr(br, "rate", np.nan))
            ld_pu = float(loading_arr[i])
            ld_pct = ld_pu * 100.0
            flow_est = ld_pu * rate_val if (rate_val is not None and not np.isnan(rate_val)) else np.nan

            max_loading_pct = max(max_loading_pct, ld_pct)
            over = max(0.0, ld_pu - 1.0)
            overload_penalty_sum += over ** 2

            fb = getattr(getattr(br, "bus_from", None), "name", f"?{i}")
            tb = getattr(getattr(br, "bus_to", None), "name", f"?{i}")
            line_monitor_list.append({
                "idx": int(i),
                "from": fb,
                "to": tb,
                "rate_MVA": rate_val,
                "flow_MVA_est": float(flow_est) if flow_est == flow_est else None,
                "loading_pct": ld_pct,
                "type": type(br).__name__,
                "name": getattr(br, "name", f"branch_{i}"),
            })

        C_ov = self.lambda_overload * float(overload_penalty_sum)

        total_cost = C_gen / 100.0 + C_loss + C_ov / 500.0
        reward = - total_cost

        obs = self._build_obs()
        bus_df = pf.results.get_bus_df()
        Vm_min = float(bus_df["Vm"].min())
        Vm_max = float(bus_df["Vm"].max())

        info.update({
            "C_gen": float(C_gen),
            "C_loss": float(C_loss),
            "C_ov": float(C_ov),
            "total_cost": float(total_cost),
            "P_loss": float(P_loss),
            "Vm_min": Vm_min, "Vm_max": Vm_max,
            "branch_loading_pct_max": float(max_loading_pct),
            "line_monitor": line_monitor_list,
        })

        done = True  # 单步样本
        return obs, float(reward), done, info

    def _current_gen_P_array(self):
        """返回与 self.grid.generators 顺序对齐的当前机组出力（按 Pmin 近似，因为我们已把 Pmin≈Pmax 固定）"""
        P = []
        for g in self.grid.generators:
            P.append(float(getattr(g, "Pmin", 0.0)))
        return np.asarray(P, dtype=float)


# ---------------------------
# 工厂/规格
# ---------------------------
def make_env(seed: int | None = None) -> GridOPFEnv:
    env = GridOPFEnv(seed=seed, enable_opf_eval=False, opf_solver="NONLINEAR_OPF")
    env.reset(seed=seed)
    print(f"[Info] Using SimBench: {env.sb_code} | USE_ALL_SGEN={USE_ALL_SGEN} | action_dim={env.action_dim} | state_dim={env.state_dim}")
    return env


def get_env_spec(seed: int | None = None) -> dict:
    env = make_env(seed=seed)
    return {
        "state_dim": int(env.state_dim),
        "action_dim": int(env.action_dim),
        "act_limit": float(GridOPFEnv.RECOMMENDED_ACT_LIMIT),
        "sb_code": env.sb_code,
        "use_all_sgen": USE_ALL_SGEN,
    }


__all__ = [
    "GridOPFEnv",
    "make_env",
    "get_env_spec",
]
