# =========================================
# train_sac.py
# Main training code
# src.sac_train.train() --> SAC
# =========================================

from __future__ import annotations
import argparse

# import env module & training logic
from src import network_env as net
from src.sac_train import train


def parse_args():
    ap = argparse.ArgumentParser(description="SAC Training Script for GridOPFEnv")

    # regular
    ap.add_argument("--log_dir", type=str, default="logs", help="日志输出目录")
    ap.add_argument("--device", type=str, default=None, help="'cpu' 或 'cuda'；为空则自动检测")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")

    # parameters
    ap.add_argument("--max_episodes", type=int, default=100, help="总训练回合数（每回合一步）")
    ap.add_argument("--start_random_eps", type=int, default=50, help="前多少回合使用随机动作探索")
    ap.add_argument("--update_after", type=int, default=50, help="从第多少回合后开始更新网络")
    ap.add_argument("--update_every", type=int, default=10, help="每多少回合执行一次多次更新")
    ap.add_argument("--batch_size", type=int, default=128, help="每次更新采样的 batch 大小")
    ap.add_argument("--replay_size", type=int, default=50000, help="经验回放容量")

    # learning rate, hyperpara
    ap.add_argument("--actor_lr", type=float, default=3e-4, help="actor 学习率")
    ap.add_argument("--critic_lr", type=float, default=3e-4, help="critic 学习率")
    ap.add_argument("--alpha_lr", type=float, default=3e-4, help="温度参数 alpha 的学习率")
    ap.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    ap.add_argument("--polyak", type=float, default=0.995, help="目标网络软更新系数")

    # network stru
    ap.add_argument("--h1", type=int, default=256, help="隐藏层 1 单元数")
    ap.add_argument("--h2", type=int, default=256, help="隐藏层 2 单元数")

    # output and save
    ap.add_argument("--print_every", type=int, default=10, help="每 N 回合打印一次训练信息")
    ap.add_argument("--save_every", type=int, default=0, help=">0 时，每 N 回合保存一次模型")
    ap.add_argument("--final_ckpt", type=str, default="logs/sac_final.pt", help="训练结束后保存路径")
    ap.add_argument("--load_ckpt", type=str, default="", help="可选，加载已有模型继续训练")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=====================================")
    print("Start SAC Training")
    print(f"log dir: {args.log_dir}")
    print(f"device: {args.device or 'Auto'}")
    print(f"seed: {args.seed}")
    print("=====================================")

    train(args, net)