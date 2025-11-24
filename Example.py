#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_agent_csv(path):
    """
    Load the agent's predicted sgen outputs.
    CSV format:
      epoch, reward_sum, sgen_0, sgen_1, ...
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Agent CSV file not found: {path}")
    return pd.read_csv(path)


def load_opf_results(log_dir):
    """
    Load all OPF result JSON files like: opf_res_ep12.json
    Extract res_sgen_p_mw for each epoch.
    Returns a DataFrame: epoch × sgen_<sid>
    """
    opf_data = {}
    for fname in os.listdir(log_dir):
        if fname.startswith("opf_res_ep") and fname.endswith(".json"):
            ep_str = fname[len("opf_res_ep"):-5]  # strip prefix and ".json"
            try:
                ep = int(ep_str)
            except:
                continue

            fpath = os.path.join(log_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                js = json.load(f)

            if not js.get("success", False):
                # skip failed OPF
                continue

            sgen_dict = js.get("res_sgen_p_mw", {})
            opf_data[ep] = sgen_dict

    if not opf_data:
        raise RuntimeError("No valid OPF result JSON files found.")

    # Convert dict-of-dict to DataFrame
    df = pd.DataFrame.from_dict(opf_data, orient="index")
    df.index.name = "epoch"
    return df.sort_index()


def plot_opf_vs_agent(agent_df, opf_df, out_path=None):
    """
    Plot OPF vs Agent results for each sgen_i.
    agent_df: agent csv
    opf_df: DataFrame from load_opf_results
    """

    # find common epochs
    common_epochs = sorted(set(agent_df["epoch"]).intersection(opf_df.index))
    if not common_epochs:
        raise RuntimeError("No overlapping epochs between OPF and agent outputs.")

    # Align
    agent_df = agent_df.set_index("epoch").loc[common_epochs]
    opf_df = opf_df.loc[common_epochs]

    # Identify sgen columns in agent CSV
    sgen_cols = [c for c in agent_df.columns if c.startswith("sgen_")]
    if not sgen_cols:
        raise RuntimeError("No sgen_* columns found in predicted_outputs CSV.")

    plt.figure(figsize=(12, 6))
    for sgen_name in sgen_cols:
        if sgen_name in opf_df.columns:
            plt.plot(common_epochs,
                     agent_df[sgen_name],
                     label=f"Agent {sgen_name}", linestyle="-")
            plt.plot(common_epochs,
                     opf_df[sgen_name],
                     label=f"OPF {sgen_name}", linestyle="--")
        else:
            print(f"[WARN] OPF does not contain {sgen_name}, skip plotting OPF for it.")
            plt.plot(common_epochs,
                     agent_df[sgen_name],
                     label=f"Agent {sgen_name}", linestyle="-")

    plt.xlabel("Epoch")
    plt.ylabel("Active Power (MW)")
    plt.title("Comparison of Agent vs OPF sgen outputs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"[Saved] {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str,
                        help="Directory containing predicted_outputs_HVMV.csv and OPF JSON files", default="logs_gnn")
    parser.add_argument("--agent_csv", type=str, default="logs_gnn/predicted_outputs_1.csv")
    parser.add_argument("--out", type=str, default="opf_vs_agent.png")
    args = parser.parse_args()

    agent_csv_path = os.path.join(args.log_dir, args.agent_csv)
    opf_df = load_opf_results(args.log_dir)
    agent_df = load_agent_csv(agent_csv_path)

    plot_path = os.path.join(args.log_dir, args.out)
    plot_opf_vs_agent(agent_df, opf_df, out_path=plot_path)


if __name__ == "__main__":
    main()

