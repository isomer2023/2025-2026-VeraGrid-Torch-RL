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
import torch

from src.sac_buffer import ReplayBuffer
from src.sac_agent import SACAgent


# ------------------------------
# sac_train.py
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
def train(args,env_module):
    os.makedirs(args.log_dir, exist_ok=True)

    spec = env_module.get_env_spec(seed=args.seed)
    obs_dim   = spec["state_dim"]
    act_dim   = spec["action_dim"]
    act_limit = spec["act_limit"]

    env = env_module.make_env(seed=args.seed)

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
            a = np.random.uniform(-act_limit, act_limit, size=act_dim)
        else:
            a = agent.select_action(o, deterministic=False)

        o2, r, d, info = env.step(a)
        a_used = np.array(info["action"], dtype=np.float32)
        buf.store(o, a_used, r, o2, d)
        ep_ret = r
        o = env.reset()

        if ep >= args.update_after and (ep - args.update_after) % args.update_every == 0:
            for _ in range(args.update_every):
                batch = buf.sample_batch(args.batch_size)
                agent.update(batch)

        with open(rewards_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, float(ep_ret)])

        if (ep + 1) % args.print_every == 0:
            C_gen = info.get("C_gen")
            C_loss = info.get("C_loss")
            C_ov = info.get("C_ov")
            total_cost = info.get("total_cost")
            diverge_penalty = info.get("diverge_penalty", 0.0) if not info.get("converged", True) else 0.0
            slack_P = info.get("slack_P")
            slack_cost = info.get("slack_cost")

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Ep {ep + 1}/{args.max_episodes} | "
                f"Reward={float(ep_ret):.3f} | "
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


    # 最后存一次
    if args.final_ckpt:
        agent.save(args.final_ckpt)
        print(f"[Save] Final checkpoint: {args.final_ckpt}")

    evaluate_5(env, agent, os.path.join(args.log_dir, "eval5.csv"))
