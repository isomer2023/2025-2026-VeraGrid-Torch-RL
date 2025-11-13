from __future__ import annotations
import os, csv, json

import numpy as np
import torch

import src.GNN.network_env as net
from src.GNN.sac_agent import GNNSAC
from src.GNN.sac_agent import GraphReplay
from src.GNN.sac_agent import obs_to_data

# 库报错检查
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
except Exception as e:
    raise RuntimeError(
        "PLS INSTALL PyTorch Geometric"
    ) from e

# 主训练程序
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

        # 更新
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

        # 保存每个 ep 的预测机组出力
        save_predicted_outputs(env, args.log_dir, ep, float(r))

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

        agent.plot_q_scatter_final(save_path="./logs/q_scatter_final.png")

# 将当前环境中的所有可控发电机出力写入 csv
# 每个 ep 追加一行
def save_predicted_outputs(env, log_dir, epoch, reward_sum):
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "predicted_outputs.csv")
    # 获取机组名称与当前出力
    gen_names = [g.name for g in env.ctrl_gens]
    # generator.P 是实际输出功率(MW)
    gen_outputs = [float(getattr(g, "P", 0.0)) for g in env.ctrl_gens]  # 实际出力

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "reward_sum"] + gen_names)
        writer.writerow([epoch, reward_sum] + gen_outputs)