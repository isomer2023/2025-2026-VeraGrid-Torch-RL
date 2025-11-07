# =========================================
# GNNSAC.py —— 图观测 + SAC（PyTorch Geometric）
# 依赖：torch, torch_geometric
# 环境：使用我们提供的 network_env.py（默认 obs_mode="graph"）
# 与 network_env 的 GraphObs 字段严格对齐：
#   node_feat(N,2: [P_load, P_gen]), edge_index(2,E), edge_feat(E,1:[rate_MVA]),
#   sgen_map(Nsgen,), sgen_pmax(Nsgen,), act_min(Nsgen,), act_max(Nsgen,)
# =========================================
from __future__ import annotations
import os, csv, argparse, json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import network_env as net

# --- PyG ---
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
except Exception as e:
    raise RuntimeError(
        "需要安装 PyTorch Geometric。示例：\n"
        "pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-$(python -c \"import torch;print(torch.__version__)\".whl)"
    ) from e


# -------------------------
# 将 GraphObs 转成 PyG Data
# -------------------------
def obs_to_data(obs) -> Data:
    x = torch.as_tensor(obs.node_feat, dtype=torch.float32)          # (N,F_n)
    edge_index = torch.as_tensor(obs.edge_index, dtype=torch.long)   # (2,E)
    edge_attr  = torch.as_tensor(obs.edge_feat, dtype=torch.float32) # (E,F_e)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # 附加与动作相关的映射与边界（环境内部做 [−1,1]→[min,max] 映射，但留存这些以便将来用）
    data.sgen_map   = torch.as_tensor(obs.sgen_map,   dtype=torch.long)    # (Nsgen,)
    data.sgen_pmax  = torch.as_tensor(obs.sgen_pmax,  dtype=torch.float32) # (Nsgen,)
    data.act_min    = torch.as_tensor(obs.act_min,    dtype=torch.float32) # (Nsgen,)
    data.act_max    = torch.as_tensor(obs.act_max,    dtype=torch.float32) # (Nsgen,)
    return data


# -------------------------
# GNN Backbone
# -------------------------
class GNNBackbone(nn.Module):
    def __init__(self, in_nf: int, in_ef: int, hid: int = 128, layers: int = 2):
        super().__init__()
        self.node_in = nn.Linear(in_nf, hid)
        # 这里先不显式用 edge_attr（GCNConv 不接收 edge_attr）；若要用可换 GIN/GraphConv 或自定义消息
        self.convs = nn.ModuleList([GCNConv(hid, hid) for _ in range(layers)])
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr=None):
        h = self.node_in(x)
        for conv in self.convs:
            h = self.act(conv(h, edge_index))
        return h  # (N,hid)


# -------------------------
# Actor：节点嵌入 → 取 sgen_map → 输出 Nsgen 连续动作（[-1,1]）
# -------------------------
LOG_STD_MIN, LOG_STD_MAX = -20, 2

class GraphActor(nn.Module):
    def __init__(self, in_nf: int, in_ef: int, hid: int, out_dim: int):
        super().__init__()
        self.backbone = GNNBackbone(in_nf, in_ef, hid=hid, layers=2)
        self.head_mu = nn.Linear(hid, 1)
        self.head_logstd = nn.Linear(hid, 1)
        self.tanh = nn.Tanh()
        self.out_dim = out_dim

    def forward(self, data: Data, deterministic=False, with_logprob=True):
        H = self.backbone(data.x, data.edge_index, data.edge_attr)   # (N,hid)
        Z = H[data.sgen_map]                                         # (Nsgen,hid)

        mu = self.head_mu(Z).squeeze(-1)                             # (Nsgen,)
        log_std = torch.clamp(self.head_logstd(Z).squeeze(-1), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mu, std)
        u = mu if deterministic else dist.rsample()
        a_tanh = torch.tanh(u)                                       # [-1,1]

        logp_pi = None
        if with_logprob:
            # standard tanh-squash修正
            eps = 1e-6
            logp_u = dist.log_prob(u).sum(dim=-1)
            log_jac = torch.log(1 - a_tanh.pow(2) + eps).sum(dim=-1)
            logp_pi = logp_u - log_jac

        return a_tanh, logp_pi

    @torch.no_grad()
    def act(self, data: Data, deterministic=False):
        a, _ = self.forward(data, deterministic, with_logprob=False)
        return a.cpu().numpy()


# -------------------------
# Critic：图 + 动作
# -------------------------
def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]),
                   act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)

class GraphQ(nn.Module):
    def __init__(self, in_nf: int, in_ef: int, hid: int):
        super().__init__()
        self.backbone = GNNBackbone(in_nf, in_ef, hid=hid, layers=2)
        self.sgen_mlp = mlp([hid + 1, hid, hid], nn.ReLU, nn.ReLU)
        self.head = nn.Linear(hid, 1)

    def forward(self, data: Data, action_norm: torch.Tensor):
        """
        action_norm: (Nsgen,) 归一到 [-1,1] 的动作
        """
        H = self.backbone(data.x, data.edge_index, data.edge_attr)   # (N,hid)
        Z = H[data.sgen_map]                                         # (Nsgen,hid)
        a = action_norm.unsqueeze(-1)                                # (Nsgen,1)
        ZA = torch.cat([Z, a], dim=-1)                               # (Nsgen,hid+1)
        z_agg = self.sgen_mlp(ZA).mean(dim=0, keepdim=True)          # (1,hid)
        q = self.head(z_agg).squeeze(-1)                             # (1,)
        return q


# -------------------------
# SAC（Graph 版）
# -------------------------
class GNNSAC:
    def __init__(self, in_nf, in_ef, act_dim, hid=128,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, polyak=0.995, target_entropy=None, device="cpu"):
        self.device = torch.device(device)
        self.gamma = gamma
        self.polyak = polyak

        self.actor   = GraphActor(in_nf, in_ef, hid, out_dim=act_dim).to(self.device)
        self.q1      = GraphQ(in_nf, in_ef, hid).to(self.device)
        self.q2      = GraphQ(in_nf, in_ef, hid).to(self.device)
        self.q1_targ = GraphQ(in_nf, in_ef, hid).to(self.device)
        self.q2_targ = GraphQ(in_nf, in_ef, hid).to(self.device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        self.pi_opt  = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_opt  = optim.Adam(self.q1.parameters(),    lr=critic_lr)
        self.q2_opt  = optim.Adam(self.q2.parameters(),    lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)

        # 连续动作 Nsgen：常用目标熵为 -Nsgen
        self.target_entropy = -act_dim if target_entropy is None else target_entropy

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, data: Data, deterministic=False):
        data = data.to(self.device)
        return self.actor.act(data, deterministic=deterministic)

    def update(self, batch):
        """
        batch: list of tuples (data, action_norm, reward, next_data, done)
        简化：逐条样本更新；若需加速可实现自定义 collate 做并行 batch。
        """
        for (data, a_np, r, data2, d) in batch:
            data  = data.to(self.device)
            data2 = data2.to(self.device)
            a  = torch.as_tensor(a_np, dtype=torch.float32, device=self.device)  # (Nsgen,)
            r  = torch.as_tensor(r, dtype=torch.float32, device=self.device).view(1)
            d  = torch.as_tensor(d, dtype=torch.float32, device=self.device).view(1)

            # --- target ---
            with torch.no_grad():
                a2, logp_a2 = self.actor.forward(data2, deterministic=False, with_logprob=True)
                q1_t = self.q1_targ(data2, a2)
                q2_t = self.q2_targ(data2, a2)
                q_t  = torch.min(q1_t, q2_t) - self.alpha * logp_a2
                backup = r + (1 - d) * self.gamma * q_t

            # --- critic ---
            q1 = self.q1(data, a)
            q2 = self.q2(data, a)
            loss_q1 = ((q1 - backup) ** 2).mean()
            loss_q2 = ((q2 - backup) ** 2).mean()
            self.q1_opt.zero_grad(); loss_q1.backward(); self.q1_opt.step()
            self.q2_opt.zero_grad(); loss_q2.backward(); self.q2_opt.step()

            # --- actor ---
            for p in self.q1.parameters(): p.requires_grad = False
            for p in self.q2.parameters(): p.requires_grad = False

            pi, logp_pi = self.actor.forward(data, deterministic=False, with_logprob=True)
            q1_pi = self.q1(data, pi)
            q2_pi = self.q2(data, pi)
            q_pi  = torch.min(q1_pi, q2_pi)
            loss_pi = (self.alpha * logp_pi - q_pi).mean()

            self.pi_opt.zero_grad(); loss_pi.backward(); self.pi_opt.step()

            for p in self.q1.parameters(): p.requires_grad = True
            for p in self.q2.parameters(): p.requires_grad = True

            # --- temperature ---
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

            # --- polyak ---
            with torch.no_grad():
                for p, p_t in zip(self.q1.parameters(), self.q1_targ.parameters()):
                    p_t.data.mul_(self.polyak); p_t.data.add_((1 - self.polyak) * p.data)
                for p, p_t in zip(self.q2.parameters(), self.q2_targ.parameters()):
                    p_t.data.mul_(self.polyak); p_t.data.add_((1 - self.polyak) * p.data)


# -------------------------
# 简易“图版”回放（列表存 Data）
# -------------------------
class GraphReplay:
    def __init__(self, max_size=50000):
        self.buf: List[Tuple[Data, np.ndarray, float, Data, bool]] = []
        self.max = max_size
    def store(self, data: Data, act_norm: np.ndarray, rew: float, next_data: Data, done: bool):
        if len(self.buf) >= self.max:
            self.buf.pop(0)
        self.buf.append((data, act_norm, rew, next_data, done))
    def sample_batch(self, batch_size=32):
        if not self.buf:
            return []
        idxs = np.random.randint(0, len(self.buf), size=min(batch_size, len(self.buf)))
        return [self.buf[i] for i in idxs]
    def __len__(self):
        return len(self.buf)


# -------------------------
# 训练主程序
# -------------------------
def train(args):
    os.makedirs(args.log_dir, exist_ok=True)

    # 显式切网（与 network_env 保持一致）
    env = net.make_env(seed=args.seed, sb_code=args.sb_code, obs_mode="graph")

    # 拿一帧观测以确定维度
    o = env.reset(seed=args.seed)
    in_nf   = int(o.node_feat.shape[1])
    in_ef   = int(o.edge_feat.shape[1])
    act_dim = int(env.action_dim)   # Nsgen

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    agent = GNNSAC(in_nf, in_ef, act_dim, hid=args.hid,
                   actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr,
                   gamma=args.gamma, polyak=args.polyak, device=device)

    buf = GraphReplay(max_size=args.replay_size)

    # 记录
    rewards_csv = os.path.join(args.log_dir, "episode_rewards.csv")
    with open(rewards_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["episode", "total_reward"])

    for ep in range(args.max_episodes):
        obs = env.reset()
        data = obs_to_data(obs)

        # 动作
        if ep < args.start_random_eps:
            a = np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            a = agent.select_action(data, deterministic=False).astype(np.float32)

        # 交互
        obs2, r, d, info = env.step(a)
        data2 = obs_to_data(obs2)
        buf.store(data, a, float(r), data2, bool(d))

        # 在线更新
        if ep >= args.update_after and (ep - args.update_after) % args.update_every == 0 and len(buf) > 0:
            batch = buf.sample_batch(args.batch_size)
            if batch:
                agent.update(batch)

        # 打印 & 记录
        print(f"\n[Ep {ep+1}] reward={float(r):.6f} | Nsgen={act_dim} | in_nf={in_nf} in_ef={in_ef}")
        lm = info.get("line_monitor", [])
        if lm:
            for li in lm[:min(10, len(lm))]:
                flow_val = li['flow_MVA_est']
                flow_str = "None" if (flow_val is None) else f"{flow_val:.3f}"
                print(f"  [Line {li['idx']:02d}] {li['from']}->{li['to']}  "
                      f"flow {flow_str} / cap {li['rate_MVA']:.3f} MVA  "
                      f"({li['loading_pct']:.1f}%)")

        with open(rewards_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, float(r)])

        # 存档
        if args.save_every and (ep + 1) % args.save_every == 0:
            ckpt = os.path.join(args.log_dir, f"gnnsac_ep{ep+1}.pt")
            torch.save({
                "actor": agent.actor.state_dict(),
                "q1": agent.q1.state_dict(),
                "q2": agent.q2.state_dict(),
                "alpha": float(agent.alpha.item()),
                "in_nf": in_nf, "in_ef": in_ef, "act_dim": act_dim
            }, ckpt)
            with open(os.path.join(args.log_dir, "spec.json"), "w", encoding="utf-8") as jf:
                json.dump({"sb_code": args.sb_code, "in_nf": in_nf, "in_ef": in_ef, "act_dim": act_dim}, jf, indent=2)
            print(f"[Save] {ckpt}")


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, default="logs_gnn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save_every", type=int, default=0)

    # 显式切换 SimBench 网
    ap.add_argument("--sb_code", type=str, default="1-HV-urban--0-sw")

    # SAC
    ap.add_argument("--max_episodes", type=int, default=3000)
    ap.add_argument("--start_random_eps", type=int, default=50)
    ap.add_argument("--update_after", type=int, default=50)
    ap.add_argument("--update_every", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--replay_size", type=int, default=50000)
    ap.add_argument("--actor_lr", type=float, default=3e-4)
    ap.add_argument("--critic_lr", type=float, default=3e-4)
    ap.add_argument("--alpha_lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--polyak", type=float, default=0.995)

    # GNN
    ap.add_argument("--hid", type=int, default=128)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
