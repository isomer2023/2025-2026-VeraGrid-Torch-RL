# =========================================
# SAC.py —— 训练脚本（自动读取 network_env 规格）
# =========================================
from __future__ import annotations
import os, csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import network_env as net  # 读取环境与规格


# ------------------
# 简单经验回放
# ------------------
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
        return dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )


# ------------------
# 神经网络组件
# ------------------
def mlp(sizes, act_fn=nn.ReLU, out_act=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[j], sizes[j + 1]),
                   act_fn() if j < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


LOG_STD_MIN, LOG_STD_MAX = -20, 2


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), nn.ReLU, nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = float(act_limit)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = torch.clamp(self.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_dist = torch.distributions.Normal(mu, std)
        pi_action = mu if deterministic else pi_dist.rsample()

        logp_pi = None
        if with_logprob:
            logp_pi = pi_dist.log_prob(pi_action).sum(axis=-1)

        pi_action = torch.tanh(pi_action) * self.act_limit  # [-act_limit, act_limit]
        return pi_action, logp_pi

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        a, _ = self.forward(obs, deterministic, with_logprob=False)
        return a.cpu().numpy()


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
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

        self.actor = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, self.act_limit).to(self.device)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q1_targ = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q2_targ = MLPQFunction(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        self.pi_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=critic_lr)

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
        o = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        o2 = torch.as_tensor(batch["obs2"], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)
        r = torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device)
        d = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            a2, logp_a2 = self.actor.forward(o2)
            q1_t = self.q1_targ(o2, a2)
            q2_t = self.q2_targ(o2, a2)
            q_t = torch.min(q1_t, q2_t) - self.log_alpha.exp() * logp_a2
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
        q_pi = torch.min(q1_pi, q2_pi)
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
# 评测（5个场景）
# ------------------
def evaluate_5(env, agent, out_csv):
    seeds = [11, 22, 33, 44, 55]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario_id", "episode_reward", "pf_converged",
                    "Vm_min", "Vm_max", "branch_loading_pct_max",
                    "pv_pmax_rand", "wt_pmax_rand", "load_scale",
                    "action", "a_star"])
        for i, sd in enumerate(seeds):
            s = env.reset(seed=sd)
            done, ep_ret = False, 0.0
            while not done:
                a = agent.select_action(s, deterministic=True)
                s2, r, done, info = env.step(a)
                ep_ret += r
                s = s2
            w.writerow([
                i, float(ep_ret),
                info.get("pf_converged"),
                info.get("Vm_min"), info.get("Vm_max"),
                info.get("branch_loading_pct_max"),
                info.get("pv_pmax_rand"), info.get("wt_pmax_rand"),
                info.get("load_scale"),
                info.get("action"), info.get("a_star")
            ])


# ------------------
# 训练主流程
# ------------------
def train(max_episodes=400,
          start_random_eps=50,
          update_after=50,
          update_every=10,
          batch_size=128,
          replay_size=50000,
          log_dir="logs",
          device=None):

    os.makedirs(log_dir, exist_ok=True)
    rewards_csv = os.path.join(log_dir, "episode_rewards.csv")
    with open(rewards_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["episode", "total_reward"])

    # 读环境规格 & 创建环境
    spec = net.get_env_spec(seed=42)
    obs_dim = spec["state_dim"]
    act_dim = spec["action_dim"]
    act_limit = spec["act_limit"]

    env = net.make_env(seed=42)

    # Agent & Buffer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SACAgent(obs_dim, act_dim, act_limit, device=device)
    buf = ReplayBuffer(obs_dim, act_dim, size=replay_size)

    # 训练（单步任务：每回合一步）
    o = env.reset()
    for ep in range(max_episodes):
        if ep < start_random_eps:
            a = np.random.uniform(-act_limit, act_limit, size=act_dim)
        else:
            a = agent.select_action(o, deterministic=False)

        o2, r, d, info = env.step(a)
        buf.store(o, a, r, o2, d)
        ep_ret = r

        o = env.reset()  # 开下一个回合（单步任务）

        if ep >= update_after and (ep - update_after) % update_every == 0:
            for _ in range(update_every):
                batch = buf.sample_batch(batch_size)
                agent.update(batch)

        with open(rewards_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, float(ep_ret)])

        if (ep + 1) % 10 == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Ep {ep+1}/{max_episodes} | "
                  f"Reward={ep_ret:.3f} | r_dist={info.get('r_dist', 0):.3f} | "
                  f"r_penalty={info.get('r_penalty', 0):.3f}")

    evaluate_5(env, agent, os.path.join(log_dir, "eval5.csv"))


if __name__ == "__main__":
    train(
        max_episodes=400,
        start_random_eps=50,
        update_after=50,
        update_every=10,
        batch_size=128,
        replay_size=50000,
        log_dir="logs",
    )
