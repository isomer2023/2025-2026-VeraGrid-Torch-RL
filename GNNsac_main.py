from __future__ import annotations
import argparse
from src.GNN.sac_train import train
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)
np.seterr(invalid='ignore', divide='ignore')

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, default="./logs_gnn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save_every", type=int, default=0)

    # 显式切换 SimBench 网
    ap.add_argument("--sb_code", type=str, default="1-HV-urban--0-sw")

    # SAC
    ap.add_argument("--max_episodes", type=int, default=500)
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

# 注释：
# 奖励函数调整每项权重： network_env