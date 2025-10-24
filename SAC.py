# =========================================
# SAC.py —— 训练脚本（自动读取 network_env 规格）
# =========================================
from __future__ import annotations
import os, csv, argparse, json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import src.network_env as net  # 读取环境与规格
import src.sac_agent as SACAgent
import src.sac_train as SACTrain
import src.sac_buffer as ReplayBuffer
print(type(SACAgent))
print(type(SACTrain))
print(type(ReplayBuffer))

# ------------------
# 训练主流程
# ------------------
def train(args):
    os.makedirs(args.log_dir, exist_ok=True)

    # 读环境规格 & 创建环境
    spec = net.get_env_spec(seed=args.seed)
    obs_dim   = spec["state_dim"]
    act_dim   = spec["action_dim"]
    act_limit = spec["act_limit"]

    env = net.make_env(seed=args.seed)

    # Agent & Buffer
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(spec["state_dim"], spec["action_dim"], spec["act_limit"], device=args.device)

    if args.load_ckpt and os.path.isfile(args.load_ckpt):
        agent.load(args.load_ckpt, map_location=device)
        print(f"[Info] Loaded checkpoint: {args.load_ckpt}")

    buf = ReplayBuffer(spec["state_dim"], spec["action_dim"], args.replay_size)

    # 日志
    rewards_csv = os.path.join(args.log_dir, "episode_rewards.csv")
    with open(rewards_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["episode", "total_reward"])

    # 训练（单步任务：每回合一步）
    o = env.reset(seed=args.seed)
    for ep in range(args.max_episodes):
        if ep < args.start_random_eps:
            a = np.random.uniform(-act_limit, act_limit, size=act_dim)
        else:
            a = agent.select_action(o, deterministic=False)

        o2, r, d, info = env.step(a)
        buf.store(o, a, r, o2, d)
        ep_ret = r

        o = env.reset()  # 开下一个回合（单步任务）

        if ep >= args.update_after and (ep - args.update_after) % args.update_every == 0:
            for _ in range(args.update_every):
                batch = buf.sample_batch(args.batch_size)
                agent.update(batch)

        with open(rewards_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, float(ep_ret)])

        if (ep + 1) % args.print_every == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Ep {ep+1}/{args.max_episodes} | "
                  f"Reward={ep_ret:.3f} | r_dist={info.get('r_dist', 0):.3f} | "
                  f"r_penalty={info.get('r_penalty', 0):.3f}")

        if args.save_every and (ep + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.log_dir, f"sac_ep{ep+1}.pt")
            agent.save(ckpt_path)
            with open(os.path.join(args.log_dir, "spec.json"), "w", encoding="utf-8") as jf:
                json.dump(spec, jf, indent=2)
            print(f"[Save] Checkpoint: {ckpt_path}")

    # 最后存一次
    if args.final_ckpt:
        agent.save(args.final_ckpt)
        print(f"[Save] Final checkpoint: {args.final_ckpt}")

    SACTrain.evaluate_5(env, agent, os.path.join(args.log_dir, "eval5.csv"))


# ------------------
# CLI
# ------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, default="logs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--max_episodes", type=int, default=400)
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
    ap.add_argument("--save_every", type=int, default=0, help=">0 to save every N episodes")
    ap.add_argument("--final_ckpt", type=str, default="logs/sac_final.pt")
    ap.add_argument("--load_ckpt", type=str, default="")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
