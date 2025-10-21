# DRL 训练 + AC-OPF 模仿最优环境
# =========================================
import os
import csv
import time
import math
import random
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import pandapower as pp
import pandapower.networks as nw

# VeraGrid / GridCal（Vera 分支）
import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en

# 把 pandapower 的 net → VeraGrid 的 MultiCircuit
import src.GC_PandaPowerImporter as GC_PandaPowerImporter


# ===========================
# 工具函数：读取 + 转换 + 清洗
# ===========================
def load_case14_as_veragrid(run_pp_before=True, sanitize=True, set_line_rate_100=True):
    import pandapower as pp
    import pandapower.networks as nw
    import numpy as np
    import src.GC_PandaPowerImporter as GC_PandaPowerImporter

    net_pp = nw.case14()

    # 放宽 PP 侧电压上限，避免 vm_pu > max_vm_pu 告警
    net_pp.bus["max_vm_pu"] = np.maximum(net_pp.bus.get("max_vm_pu", 1.05), 1.06)

    # ✅ 关键：先在 PP 侧跑一次潮流，用 flat 初始化，生成 res_* 表
    if run_pp_before:
        try:
            pp.runpp(net_pp, algorithm="nr", init="flat")
        except Exception as e:
            print("pandapower runpp 失败：", e)

    # 转 VeraGrid
    grid_gc = GC_PandaPowerImporter.PP2GC(net_pp)

    # 清洗
    if sanitize:
        for b in grid_gc.buses:
            b.Vmin, b.Vmax = 0.95, 1.05
        for g in grid_gc.generators:
            g.Vset = min(max(g.Vset, 0.95), 1.03)

    # 统一线路额定 100 MVA（loading% 计算用）
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


# ==================================
# 环境：GridOPFEnv（模仿最优 + 约束校验）
# ==================================
class GridOPFEnv:
    """
    动作 a = [P_th1, P_th2, P_th3, P_pv, P_wt] 的“指令”（MW）
    奖励： -lambda_dist * mean(|a - a*|) - lambda_div * 1{PF不收敛}
    其中 a* 为 AC-OPF 的最优解（同维度的可控机组出力）
    观测：负荷分布（14维母线有功净负荷估计） + [pv_pmax_rand, wt_pmax_rand]
    """

    def __init__(self,
                 pv_range=(0.30, 0.90),
                 wt_range=(0.20, 0.95),
                 load_jitter=0.15,         # 负荷随机温和系数（越大越随机）
                 lambda_dist=1.0,          # 距离损失系数
                 lambda_diverge=50.0,      # 不收敛惩罚
                 seed=None):

        self.base_net, self.base_grid = load_case14_as_veragrid(
            run_pp_before=True, sanitize=True, set_line_rate_100=True
        )
        self.rng = np.random.default_rng(seed)

        # 可控机组位置：火电(1,2,3)，光伏(6)，风电(8)
        self.thermal_buses = [1, 2, 3]
        self.pv_bus = 6
        self.wt_bus = 8

        # 火电上下限（固定）
        # 你可以按需修改
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

        # 维度
        self.action_dim = 5  # th1, th2, th3, pv, wt
        self.state_dim = 14 + 2  # 14个母线近似净负荷 + [pv_max, wt_max]

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
        # NOTE：我们不把 action 直接喂给 OPF（OPF求最优 a*），
        # 这里只为了 PF 收敛校验：把 action 固定成发电出力（即 Pmin=Pmax=action）
        # —— 只在 PF 验证时使用 —— #
        th1, th2, th3, pv, wt = action_vec

        def clamp(x, lo, hi):
            return float(max(lo, min(hi, x)))

        # 获取机组对象
        g1 = get_generators_at_bus(self.grid, self.thermal_buses[0])[0]
        g2 = get_generators_at_bus(self.grid, self.thermal_buses[1])[0]
        g3 = get_generators_at_bus(self.grid, self.thermal_buses[2])[0]
        g6 = get_generators_at_bus(self.grid, self.pv_bus)[0]
        g8 = get_generators_at_bus(self.grid, self.wt_bus)[0]

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
        g6 = get_generators_at_bus(self.grid, self.pv_bus)[0]
        g8 = get_generators_at_bus(self.grid, self.wt_bus)[0]
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
        g1 = get_generators_at_bus(self.grid, self.thermal_buses[0])[0]
        g2 = get_generators_at_bus(self.grid, self.thermal_buses[1])[0]
        g3 = get_generators_at_bus(self.grid, self.thermal_buses[2])[0]
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
        # 维度固定为 14，按 bus_df 的顺序截或补
        p_col = np.array(bus_df["P"].values, dtype=float)
        # 取“需求”尺度：对负值取其正值（|负|），对正值置 0
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
        _, self.grid = load_case14_as_veragrid(run_pp_before=True, sanitize=True, set_line_rate_100=True)

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
        g1 = get_generators_at_bus(self.grid, self.thermal_buses[0])[0]
        g2 = get_generators_at_bus(self.grid, self.thermal_buses[1])[0]
        g3 = get_generators_at_bus(self.grid, self.thermal_buses[2])[0]
        g6 = get_generators_at_bus(self.grid, self.pv_bus)[0]
        g8 = get_generators_at_bus(self.grid, self.wt_bus)[0]
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


# ===================
# SAC 轻量实现
# ===================
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs=self.obs_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    act=self.act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs])


def mlp(sizes, act_fn=nn.ReLU, out_act=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = act_fn if j < len(sizes) - 2 else out_act
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


LOG_STD_MIN, LOG_STD_MAX = -20, 2


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), nn.ReLU, nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit  # 我们输出的是“原始 MW 指令”的缩放前[-1,1]，外部再映射到 MW

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = torch.distributions.Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)  # [-1, 1]
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.forward(obs, deterministic, False)
            return a.cpu().numpy()


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1])

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # 形状 [batch]


class SACAgent:
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256),
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, gamma=0.99, polyak=0.995,
                 target_entropy=None, device="cpu", act_limit=1.0):

        self.device = torch.device(device)
        self.act_limit = act_limit

        self.actor = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, act_limit).to(self.device)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q1_targ = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q2_targ = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        self.pi_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)

        # 温度参数 α
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().item()

        self.gamma = gamma
        self.polyak = polyak
        self.target_entropy = -act_dim if target_entropy is None else target_entropy

    def select_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor.act(obs_t, deterministic=deterministic)[0]
        return a  # [-act_limit, act_limit]

    def update(self, data):
        o = torch.as_tensor(data["obs"], dtype=torch.float32, device=self.device)
        o2 = torch.as_tensor(data["obs2"], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(data["act"], dtype=torch.float32, device=self.device)
        r = torch.as_tensor(data["rew"], dtype=torch.float32, device=self.device)
        d = torch.as_tensor(data["done"], dtype=torch.float32, device=self.device)

        # 目标 Q
        with torch.no_grad():
            a2, logp_a2 = self.actor.forward(o2)
            q1_pi_targ = self.q1_targ(o2, a2)
            q2_pi_targ = self.q2_targ(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ) - self.log_alpha.exp() * logp_a2
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # Q1 & Q2 损失
        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()

        self.q1_optimizer.zero_grad()
        loss_q1.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        loss_q2.backward()
        self.q2_optimizer.step()

        # 策略损失
        for p in self.q1.parameters():
            p.requires_grad = False
        for p in self.q2.parameters():
            p.requires_grad = False

        pi, logp_pi = self.actor.forward(o)
        q1_pi = self.q1(o, pi)
        q2_pi = self.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (self.log_alpha.exp() * logp_pi - q_pi).mean()

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.q1.parameters():
            p.requires_grad = True
        for p in self.q2.parameters():
            p.requires_grad = True

        # 温度 α
        alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # 软更新
        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(), self.q1_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.q2.parameters(), self.q2_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


# ===========================
# 训练与评测（带 CSV 记录）
# ===========================
def evaluate_5_scenarios(env, agent, out_csv):
    seeds = [11, 22, 33, 44, 55]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario_id",
            "action",
            "a_star",
            "pf_converged",
            "pf_losses_P",
            "pf_losses_Q",
            "Vm_min",
            "Vm_max",
            "branch_loading_pct_max",
            "pv_pmax_rand",
            "wt_pmax_rand",
            "load_scale",
            "episode_reward"
        ])

        for i, sd in enumerate(seeds):
            s = env.reset(seed=sd)
            done = False
            ep_ret = 0.0
            while not done:
                a = agent.select_action(s, deterministic=True)  # 测试用均值动作
                s_, r, done, info = env.step(a)
                ep_ret += r
                s = s_

            w.writerow([
                i,
                np.asarray(info.get("action", [])).tolist(),
                np.asarray(info.get("a_star", [])).tolist(),
                info.get("pf_converged", None),
                info.get("pf_losses_P", None),
                info.get("pf_losses_Q", None),
                info.get("Vm_min", None),
                info.get("Vm_max", None),
                info.get("branch_loading_pct_max", None),
                info.get("pv_pmax_rand", None),
                info.get("wt_pmax_rand", None),
                info.get("load_scale", None),
                float(ep_ret)
            ])

    print(f"[Eval] 5 场景评测结果已保存到: {out_csv}")


def train(max_episodes=500,
          start_steps=1000,
          update_after=1000,
          update_every=50,
          batch_size=128,
          replay_size=100000,
          log_dir="logs",
          device="cpu"):
    os.makedirs(log_dir, exist_ok=True)
    rewards_csv = os.path.join(log_dir, "episode_rewards.csv")
    with open(rewards_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "total_reward"])

    # 环境 & Agent
    env = GridOPFEnv(
        pv_range=(0.30, 0.90),
        wt_range=(0.20, 0.95),
        load_jitter=0.25,      # —— 你要求“温和系数放大一点”
        lambda_dist=1.0,
        lambda_diverge=50.0,
        seed=42
    )
    obs_dim = env.state_dim
    act_dim = env.action_dim

    # 动作限制（SAC actor 输出范围 [-1,1]，我们外部把它线性映射成“MW指令”）
    # 这里直接让 SAC 输出“MW 级别”的动作区间，也就是用 act_limit 做个大致的尺度；
    # 实际合法区间由 step 内 clamp 到 [th_min,max] / [0,pmax_rand]。
    act_limit = 200.0  # 足够覆盖火电最大阶（动作再被环境 clamp）

    agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit, device=device)

    buf = ReplayBuffer(obs_dim, act_dim, size=replay_size)

    # 训练
    total_steps = max_episodes  # 每回合一步
    o = env.reset()
    for ep in range(max_episodes):
        if ep < start_steps:
            a = np.random.uniform(-act_limit, act_limit, size=act_dim)
        else:
            a = agent.select_action(o, deterministic=False)

        o2, r, d, info = env.step(a)
        buf.store(o, a, r, o2, d)

        ep_ret = r
        o = env.reset()  # 单步任务，直接开下一个回合

        # 更新
        if ep >= update_after and (ep - update_after) % update_every == 0:
            for _ in range(update_every):
                batch = buf.sample_batch(batch_size)
                agent.update(batch)

        # 记录 reward
        with open(rewards_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ep, float(ep_ret)])

        if (ep + 1) % 10 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Episode {ep+1}/{max_episodes} | Reward={ep_ret:.3f} | "
                  f"r_dist={info.get('r_dist', 0):.3f}, r_penalty={info.get('r_penalty', 0):.3f}")

    # 训练结束评测
    evaluate_5_scenarios(env, agent, os.path.join(log_dir, "eval5.csv"))


# ==============
# 入口
# ==============
if __name__ == "__main__":
    # CPU / CUDA 按需选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        max_episodes=400,     # 每回合一步，等价于总步数
        start_steps=50,       # 前 50 回合用随机动作探索
        update_after=50,
        update_every=10,
        batch_size=128,
        replay_size=50000,
        log_dir="logs",
        device=device
    )