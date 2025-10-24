# =========================================
# src/sac_train.py
# 说明：SAC 训练与评测模块（独立于主 CLI）
# =========================================
from __future__ import annotations
import os
import csv
import json
from datetime import datetime
import numpy as np
import torch  # <----- 修正：导入 torch

from src.sac_buffer import ReplayBuffer
from src.sac_agent import SACAgent


# ------------------------------
# 评测（5个随机场景）
# ------------------------------
def evaluate_5(env, agent, out_csv: str):
    """固定5个随机种子场景评估智能体性能"""
    seeds = [11, 22, 33, 44, 55]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario_id",
            "episode_reward",
            "pf_converged",
            "Vm_min", "Vm_max",
            "branch_loading_pct_max",
            "pv_pmax_rand", "wt_pmax_rand",
            "load_scale",
            "action", "a_star"
        ])

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
    print(f"[Eval] 5-scenario results saved to: {out_csv}")


# ------------------------------
# 训练主流程
# ------------------------------
def train(args, env_module):
    """
    args: argparse.Namespace（或等价的字典）
    env_module: 提供 make_env() 与 get_env_spec() 的环境模块（如 src.network_env）
    """
    os.makedirs(args.log_dir, exist_ok=True)

    # ---- 1. 环境 & 规格 ----
    spec = env_module.get_env_spec(seed=args.seed)
    obs_dim   = spec["state_dim"]
    act_dim   = spec["action_dim"]
    act_limit = spec["act_limit"]
    env = env_module.make_env(seed=args.seed)

    # ---- 2. Agent & Buffer ----
    # 处理 device 参数：优先 args.device，否则自动选择
    device = args.device if getattr(args, "device", None) else ("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(
        obs_dim, act_dim, act_limit,
        hidden_sizes=(args.h1, args.h2),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        gamma=args.gamma,
        polyak=args.polyak,
        device=device,
    )

    if getattr(args, "load_ckpt", "") and os.path.isfile(args.load_ckpt):
        agent.load(args.load_ckpt, map_location=device)
        print(f"[Info] Loaded checkpoint: {args.load_ckpt}")

    buf = ReplayBuffer(obs_dim, act_dim, size=args.replay_size)

    # ---- 3. 日志文件 ----
    rewards_csv = os.path.join(args.log_dir, "episode_rewards.csv")
    with open(rewards_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["episode", "total_reward"])

    # ---- 4. 训练循环（单步任务） ----
    o = env.reset(seed=args.seed)
    for ep in range(args.max_episodes):
        # 随机探索或策略动作
        if ep < args.start_random_eps:
            a = np.random.uniform(-act_limit, act_limit, size=act_dim)
        else:
            a = agent.select_action(o, deterministic=False)

        # 与环境交互（单步）
        o2, r, d, info = env.step(a)
        buf.store(o, a, r, o2, d)
        ep_ret = r
        o = env.reset()  # 每回合一步，直接 reset

        # 更新策略
        if ep >= args.update_after and (ep - args.update_after) % args.update_every == 0:
            for _ in range(args.update_every):
                # 注意：在 buffer 未满之前 sample 可能会失败或重复采样，确保 replay_size 足够或在 sample 前检查 buf.size
                batch = buf.sample_batch(args.batch_size)
                agent.update(batch)

        # 写入日志
        with open(rewards_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, float(ep_ret)])

        # 打印进度
        if (ep + 1) % args.print_every == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Ep {ep+1}/{args.max_episodes} | "
                  f"Reward={ep_ret:.3f} | "
                  f"r_dist={info.get('r_dist', 0):.3f} | "
                  f"r_penalty={info.get('r_penalty', 0):.3f}")

        # 中途保存模型
        if args.save_every and (ep + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.log_dir, f"sac_ep{ep+1}.pt")
            agent.save(ckpt_path)
            # ensure dir exists
            os.makedirs(os.path.dirname(os.path.join(args.log_dir, "spec.json")), exist_ok=True)
            with open(os.path.join(args.log_dir, "spec.json"), "w", encoding="utf-8") as jf:
                json.dump(spec, jf, indent=2)
            print(f"[Save] Checkpoint: {ckpt_path}")

    # ---- 5. 最终保存 ----
    if getattr(args, "final_ckpt", None):
        agent.save(args.final_ckpt)
        print(f"[Save] Final checkpoint: {args.final_ckpt}")

    # ---- 6. 评估 ----
    eval_csv = os.path.join(args.log_dir, "eval5.csv")
    evaluate_5(env, agent, eval_csv)
    print(f"[Done] Training finished. Evaluation saved to {eval_csv}")
