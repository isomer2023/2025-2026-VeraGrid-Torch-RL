# =========================================
# SAC.py —— 训练脚本（自动读取 network_env 规格）
# 每个 episode（单步）打印：
#  - 每条线路：cap(额定MVA), flow(当前估算MVA), loading%
#  - 当前步 reward
#  - 若开启 OPF 基线：opf_total_cost / optimality_gap / a_star
# =========================================
from __future__ import annotations
import os, csv, argparse, json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import network_env as net

# ------------------
# 简单经验回放
# ------------------
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf  = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf  = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf  = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr]  = obs
        self.act_buf[self.ptr]  = act
        self.rew_buf[self.ptr]  = rew
        self.obs2_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs  = self.obs_buf[idxs],
            obs2 = self.obs2_buf[idxs],
            act  = self.act_buf[idxs],
            rew  = self.rew_buf[idxs],
            done = self.done_buf[idxs],
        )

# ------------------
# 网络
# ------------------
def mlp(sizes, act_fn=nn.ReLU, out_act=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        layers += [
            nn.Linear(sizes[j], sizes[j + 1]),
            act_fn() if j < len(sizes) - 2 else out_act()
        ]
    return nn.Sequential(*layers)

LOG_STD_MIN, LOG_STD_MAX = -20, 2

class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256,256)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), nn.ReLU, nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = float(act_limit)

    def forward(self, obs, deterministic=False, with_logprob=True):
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        u = mu if deterministic else dist.rsample()
        a_tanh = torch.tanh(u)
        pi_action = a_tanh * self.act_limit

        logp_pi = None
        if with_logprob:
            logp_u = dist.log_prob(u).sum(dim=-1)
            eps = 1e-6
            log_act_limit = torch.log(torch.tensor(self.act_limit, device=obs.device))
            log_jac = (log_act_limit + torch.log(1 - a_tanh.pow(2) + eps)).sum(dim=-1)
            logp_pi = logp_u - log_jac
        return pi_action, logp_pi

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        a, _ = self.forward(obs, deterministic, with_logprob=False)
        return a.cpu().numpy()

class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256)):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1])
    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1)).squeeze(-1)

# ------------------
# SAC Agent
# ------------------
class SACAgent:
    def __init__(self, obs_dim, act_dim, act_limit,
                 hidden_sizes=(256, 256),
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, polyak=0.995, target_entropy=None,
                 device="cpu"):

        self.device = torch.device(device)
        self.act_limit = float(act_limit)

        self.actor   = SquashedGaussianMLPActor(obs_dim, act_dim, self.act_limit, hidden_sizes).to(self.device)
        self.q1      = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q2      = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q1_targ = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q2_targ = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        self.pi_opt  = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_opt  = optim.Adam(self.q1.parameters(),    lr=critic_lr)
        self.q2_opt  = optim.Adam(self.q2.parameters(),    lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().item()

        self.gamma = gamma
        self.polyak = polyak
        self.target_entropy = -act_dim if target_entropy is None else target_entropy

    def select_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor.act(obs_t, deterministic=deterministic)[0]

    def update(self, batch):
        o  = torch.as_tensor(batch["obs"],  dtype=torch.float32, device=self.device)
        o2 = torch.as_tensor(batch["obs2"], dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(batch["act"],  dtype=torch.float32, device=self.device)
        r  = torch.as_tensor(batch["rew"],  dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            a2, logp_a2 = self.actor.forward(o2)
            q1_t = self.q1_targ(o2, a2)
            q2_t = self.q2_targ(o2, a2)
            q_t  = torch.min(q1_t, q2_t) - self.log_alpha.exp() * logp_a2
            backup = r + self.gamma * (1 - d) * q_t

        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        self.q1_opt.zero_grad(); loss_q1.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); loss_q2.backward(); self.q2_opt.step()

        for p in self.q1.parameters(): p.requires_grad = False
        for p in self.q2.parameters(): p.requires_grad = False

        pi, logp_pi = self.actor.forward(o)
        q1_pi = self.q1(o, pi)
        q2_pi = self.q2(o, pi)
        q_pi  = torch.min(q1_pi, q2_pi)
        loss_pi = (self.log_alpha.exp() * logp_pi - q_pi).mean()
        self.pi_opt.zero_grad(); loss_pi.backward(); self.pi_opt.step()

        for p in self.q1.parameters(): p.requires_grad = True
        for p in self.q2.parameters(): p.requires_grad = True

        alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        with torch.no_grad():
            for p, p_t in zip(self.q1.parameters(), self.q1_targ.parameters()):
                p_t.data.mul_(self.polyak); p_t.data.add_((1 - self.polyak) * p.data)
            for p, p_t in zip(self.q2.parameters(), self.q2_targ.parameters()):
                p_t.data.mul_(self.polyak); p_t.data.add_((1 - self.polyak) * p.data)

# ------------------
# 评估（保留）
# ------------------
def evaluate_5(env, agent, out_csv):
    import csv as _csv
    seeds = [11, 22, 33, 44, 55]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["scenario_id", "episode_reward", "converged",
                    "Vm_min", "Vm_max", "branch_loading_pct_max",
                    "pv_pmax_rand", "wt_pmax_rand", "load_scale",
                    "action", "a_star", "opf_total_cost", "optimality_gap"])
        for i, sd in enumerate(seeds):
            s = env.reset(seed=sd)
            done, ep_ret = False, 0.0
            while not done:
                a = agent.select_action(s, deterministic=True).astype(np.float32)
                s2, r, done, info = env.step(a)
                ep_ret += r
                s = s2
            w.writerow([
                i, float(ep_ret),
                info.get("converged"),
                info.get("Vm_min"), info.get("Vm_max"),
                info.get("branch_loading_pct_max"),
                info.get("pv_pmax_rand"), info.get("wt_pmax_rand"),
                info.get("load_scale"),
                info.get("action"), info.get("a_star"),
                info.get("opf_total_cost"), info.get("optimality_gap")
            ])
    print(f"[Eval] 5-scenario results saved to: {out_csv}")

# ------------------
# 训练
# ------------------
def train(args):
    os.makedirs(args.log_dir, exist_ok=True)

    spec = net.get_env_spec(seed=args.seed)
    obs_dim   = spec["state_dim"]
    act_dim   = spec["action_dim"]
    act_limit = spec["act_limit"]  # = 1.0

    env = net.make_env(seed=args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(obs_dim, act_dim, act_limit,
                     hidden_sizes=(args.h1, args.h2),
                     actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr,
                     gamma=args.gamma, polyak=args.polyak,
                     target_entropy=None, device=device)

    if args.load_ckpt and os.path.isfile(args.load_ckpt):
        agent.load(args.load_ckpt, map_location=device)
        print(f"[Info] Loaded checkpoint: {args.load_ckpt}")

    buf = ReplayBuffer(obs_dim, act_dim, size=args.replay_size)

    rewards_csv = os.path.join(args.log_dir, "episode_rewards.csv")
    with open(rewards_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["episode", "total_reward"])

    o = env.reset(seed=args.seed)
    for ep in range(args.max_episodes):
        if ep < args.start_random_eps:
            a = np.random.uniform(-act_limit, act_limit, size=act_dim).astype(np.float32)
        else:
            a = agent.select_action(o, deterministic=False).astype(np.float32)

        o2, r, d, info = env.step(a)
        buf.store(o, a, r, o2, d)
        ep_ret = r

        # === 每步打印：线路容量 / 当前线路功率 / reward ===
        print(f"\n[Step {ep+1}] reward = {float(ep_ret):.6f}")
        lm = info.get("line_monitor", [])
        if lm:
            for li in lm:
                print(f"  [Line {li['idx']:02d}] {li['from']}->{li['to']}  "
                      f"flow {li['flow_MVA_est']:.3f} / cap {li['rate_MVA']:.3f} MVA  "
                      f"({li['loading_pct']:.1f}%)")
        # 若开启 OPF 基线，打印对照
        if "opf_total_cost" in info:
            print(f"  [OPF] total_cost={info['opf_total_cost']:.6f}  "
                  f"gap={info.get('optimality_gap', None)}  "
                  f"a_star={info.get('a_star', None)}")

        with open(rewards_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, float(ep_ret)])

        o = env.reset()  # 单步任务；下一步启动新场景

        if ep >= args.update_after and (ep - args.update_after) % args.update_every == 0:
            for _ in range(args.update_every):
                batch = buf.sample_batch(args.batch_size)
                agent.update(batch)

        if (ep + 1) % args.print_every == 0:
            C_gen = info.get("C_gen"); C_loss = info.get("C_loss"); C_ov = info.get("C_ov")
            total_cost = info.get("total_cost")
            diverge_penalty = info.get("diverge_penalty", 0.0) if not info.get("converged", True) else 0.0
            slack_P = info.get("slack_P"); slack_cost = info.get("slack_cost")
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Ep {ep + 1}/{args.max_episodes} | "
                f"Reward={float(ep_ret):.6f} | "
                f"A(C_gen)={None if C_gen is None else f'{C_gen:.3f}'} | "
                f"B(C_loss)={None if C_loss is None else f'{C_loss:.3f}'} | "
                f"C(C_ov)={None if C_ov is None else f'{C_ov:.3f}'} | "
                f"diverge_penalty={diverge_penalty:.3f} | "
                f"slack_P={None if slack_P is None else f'{slack_P:.3f}'} | "
                f"slack_cost={None if slack_cost is None else f'{slack_cost:.3f}'} | "
                f"Vm[min,max]=({info.get('Vm_min', None):.3f},{info.get('Vm_max', None):.3f}) | "
                f"branch_loading_max={info.get('branch_loading_pct_max', None):.1f}%"
            )

        if args.save_every and (ep + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.log_dir, f"sac_ep{ep+1}.pt")
            agent.save(ckpt_path)
            with open(os.path.join(args.log_dir, "spec.json"), "w", encoding="utf-8") as jf:
                json.dump(spec, jf, indent=2)
            print(f"[Save] Checkpoint: {ckpt_path}")

    if args.final_ckpt:
        agent.save(args.final_ckpt)
        print(f"[Save] Final checkpoint: {args.final_ckpt}")

    evaluate_5(env, agent, os.path.join(args.log_dir, "eval5.csv"))

# ------------------
# CLI
# ------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, default="logs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--max_episodes", type=int, default=4000)
    ap.add_argument("--start_random_eps", type=int, default=50)
    ap.add_argument("--update_after", type=int, default=50)
    ap.add_argument("--update_every", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--replay_size", type=int, default=50000)

    ap.add_argument("--actor_lr", type=float, default=3e-4)
    ap.add_argument("--critic_lr", type=float, default=3e-4)
    ap.add_argument("--alpha_lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--polyak", type=float, default=0.995)

    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=256)

    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--final_ckpt", type=str, default="logs/sac_final.pt")
    ap.add_argument("--load_ckpt", type=str, default="")

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
