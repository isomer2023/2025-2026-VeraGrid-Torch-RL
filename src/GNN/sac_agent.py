from __future__ import annotations
from typing import List, Tuple

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
except Exception as e:
    raise RuntimeError(
        "PLS INSTALL PyTorch Geometric"
    ) from e

# Turn Graph obs to PyG format
# 将读入的观测信息转化为 PyG 格式图结构，输入来自 GraphObs 的观测信息
def obs_to_data(obs) -> Data:
    x = torch.as_tensor(obs.node_feat, dtype=torch.float32)  # (N,F_n)
    edge_index = torch.as_tensor(obs.edge_index, dtype=torch.long)  # (2,E)
    edge_attr = torch.as_tensor(obs.edge_feat, dtype=torch.float32)  # (E,F_e)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # 附加与动作相关的映射与边界（环境内部做 [−1,1]→[min,max] 映射，但留存这些以便将来用）
    data.sgen_map = torch.as_tensor(obs.sgen_map, dtype=torch.long)  # (Nsgen,)
    data.sgen_pmax = torch.as_tensor(obs.sgen_pmax, dtype=torch.float32)  # (Nsgen,)
    data.act_min = torch.as_tensor(obs.act_min, dtype=torch.float32)  # (Nsgen,)
    data.act_max = torch.as_tensor(obs.act_max, dtype=torch.float32)  # (Nsgen,)
    return data


# 特征提取骨干模块，使用 GCNConv，输出每个节点的 embedding向量
class GNNBackbone(nn.Module):
    def __init__(self, in_nf: int, in_ef: int, hid: int = 128, layers: int = 2):
        # in_nf 节点特征维度，in_ef 边的特征维度
        super().__init__()
        self.node_in = nn.Linear(in_nf, hid)
        # 这里先不显式用 edge_attr（GCNConv 不接收 edge_attr）；若要用可换 GIN/GraphConv 或自定义消息
        self.convs = nn.ModuleList([GCNConv(hid, hid) for _ in range(layers)])
        self.act = nn.ReLU()

    # 电网节点（Bus）之间的连接结构由 edge_index 决定
    def forward(self, x, edge_index, edge_attr=None):
        h = self.node_in(x)
        for conv in self.convs:
            h = self.act(conv(h, edge_index))
        return h  # (N,hid)


# Actor节点嵌入
# 由 sgen_map 选出可控发电机节点
# 为这些节点分别输出均值（μ）和标准差（σ）
# 输出 Nsgen 可控发电机连续动作（范围[-1,1]）
# 计算 log 概率（用于 SAC 的熵正则项）
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
        H = self.backbone(data.x, data.edge_index, data.edge_attr)  # (N,hid)
        Z = H[data.sgen_map]  # (Nsgen,hid)

        mu = self.head_mu(Z).squeeze(-1)  # (Nsgen,)
        log_std = torch.clamp(self.head_logstd(Z).squeeze(-1), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mu, std)
        u = mu if deterministic else dist.rsample()
        a_tanh = torch.tanh(u)  # [-1,1]

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

# 基于图的价值网络（Critic），输入图和动作，输出单一的标量 Q，用于评估动作价值
# GNNBackbone 得到节点嵌入
# 从发电机节点取出对应嵌入
# 拼接动作（Z,a）
# 经过 MLP，最后聚合（求平均）得到整体 Q 值
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
        ZA = torch.cat([Z, a], dim=-1)                        # (Nsgen,hid+1)
        z_agg = self.sgen_mlp(ZA).mean(dim=0, keepdim=True)          # (1,hid)
        q = self.head(z_agg).squeeze(-1)                             # (1,)
        return q


# 主算法类（Soft Actor-Critic）
# 使用 双 Q 网络 减少高估偏差；
# 使用 tanh-squash 保证动作范围；
# 使用 Polyak 平滑更新 目标网络；
# 使用 可学习 α 参数 平衡探索与利用。
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

        # 用于记录整个训练过程的 Q-value 对比数据
        self.q_record_pred = []
        self.q_record_targ = []

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, data: Data, deterministic=False):
        data = data.to(self.device)
        return self.actor.act(data, deterministic=deterministic)

    def update(self, batch, log_dir=None, update_step=0, global_ep=None):
        # batch: list of tuples (data, action_norm, reward, next_data, done)
        # 简化：逐条样本更新；若需加速可实现自定义 collate 做并行 batch。
        for (data, a_np, r, data2, d) in batch:
            data  = data.to(self.device)
            data2 = data2.to(self.device)
            reward_scale = 1.0 # 尺度调整
            a  = torch.as_tensor(a_np, dtype=torch.float32, device=self.device)  # (Nsgen,)
            r  = torch.as_tensor(r * reward_scale, dtype=torch.float32, device=self.device).view(1)
            d  = torch.as_tensor(d, dtype=torch.float32, device=self.device).view(1)

            # target
            with torch.no_grad():
                a2, logp_a2 = self.actor.forward(data2, deterministic=False, with_logprob=True)
                q1_t = self.q1_targ(data2, a2)
                q2_t = self.q2_targ(data2, a2)
                q_t  = torch.min(q1_t, q2_t) - self.alpha * logp_a2
                backup = r + (1 - d) * self.gamma * q_t
                backup = torch.clamp(backup, -1e3, 1e3) # 防止极端负值

            # critic
            q1 = self.q1(data, a)
            q2 = self.q2(data, a)
            loss_q1 = ((q1 - backup) ** 2).mean()
            loss_q2 = ((q2 - backup) ** 2).mean()

            self.q1_opt.zero_grad()
            loss_q1.backward()
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=1.0) # 梯度裁剪（避免梯度爆炸）
            self.q1_opt.step()

            self.q2_opt.zero_grad()
            loss_q2.backward()
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=1.0) # 梯度裁剪（避免梯度爆炸）
            self.q2_opt.step()

            # actor
            for p in self.q1.parameters(): p.requires_grad = False
            for p in self.q2.parameters(): p.requires_grad = False

            pi, logp_pi = self.actor.forward(data, deterministic=False, with_logprob=True)
            q1_pi = self.q1(data, pi)
            q2_pi = self.q2(data, pi)
            q_pi  = torch.min(q1_pi, q2_pi)
            loss_pi = (self.alpha * logp_pi - q_pi).mean()

            self.pi_opt.zero_grad()
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) # 梯度裁剪（避免梯度爆炸）
            self.pi_opt.step()

            for p in self.q1.parameters(): p.requires_grad = True
            for p in self.q2.parameters(): p.requires_grad = True

            # temperature
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

            # polyak
            with torch.no_grad():
                for p, p_t in zip(self.q1.parameters(), self.q1_targ.parameters()):
                    p_t.data.mul_(self.polyak); p_t.data.add_((1 - self.polyak) * p.data)
                for p, p_t in zip(self.q2.parameters(), self.q2_targ.parameters()):
                    p_t.data.mul_(self.polyak); p_t.data.add_((1 - self.polyak) * p.data)

            # --- 记录 Q 数据用于最终散点图 ---
            with torch.no_grad():
                self.q_record_pred.append(q1.item())
                self.q_record_targ.append(backup.item())

    def plot_q_scatter_final(self, save_path=None):
        import matplotlib.pyplot as plt
        if len(self.q_record_pred) == 0:
            print("[WARN] No Q data recorded, nothing to plot.")
            return

        pred = np.array(self.q_record_pred)
        targ = np.array(self.q_record_targ)

        plt.figure(figsize=(6, 6))
        plt.scatter(targ, pred, alpha=0.4, s=10)
        lims = [min(targ.min(), pred.min()), max(targ.max(), pred.max())]
        plt.plot(lims, lims, 'r--', linewidth=1.5, label='y = x')
        plt.xlabel("Target Q (ground truth)")
        plt.ylabel("Predicted Q (critic output)")
        plt.title("Final Q-value Scatter (all updates)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            print(f"[Info] Saved Q scatter plot to {save_path}")
        else:
            plt.show()


# Replay
# 简易回放（列表存 Data）
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