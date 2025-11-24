# src/GNN/network_opf.py  （或放到 network_env.py）
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np

import pandapower as pp

from src.GNN.network_loader import load_simbench_as_veragrid


def run_pandapower_opf_for_env(env, sb_code: Optional[str] = None, use_sgen_values_from_env: bool = True) -> Dict[
    str, Any]:
    """
    运行 pandapower OPF，返回结构化的结果数据
    """
    sb_code = sb_code or getattr(env, "sb_code", None)
    if sb_code is None:
        return {"success": False, "error": "sb_code not provided and not found in env."}

    # 重新加载 pandapower net_pp（fresh copy）
    try:
        net_pp, grid_gc = load_simbench_as_veragrid(sb_code)
    except Exception as e:
        return {"success": False, "error": f"load_simbench_as_veragrid error: {type(e).__name__}: {e}"}

    # 如果需要，用 env 中的 sgen 值覆盖 net_pp.sgen.p_mw
    if use_sgen_values_from_env:
        if len(net_pp.sgen):
            for g in env.ctrl_gens:
                gname = str(getattr(g, "name", "")).strip()
                if not gname.startswith("sgen_"):
                    continue
                try:
                    sid_str = gname.split("_", 1)[1]
                    sid = int(sid_str)
                except Exception:
                    continue
                val = float(getattr(g, "Pmax", 0.0))
                try:
                    if sid in net_pp.sgen.index:
                        net_pp.sgen.at[sid, "p_mw"] = val
                except Exception:
                    pass

    # 尝试运行 pandapower OPF
    try:
        pp.runopp(net_pp)
    except Exception as e:
        return {"success": False, "error": f"pandapower runopp failed: {type(e).__name__}: {e}"}

    # 收集结果
    res = {"success": True}
    try:
        # sgen 结果
        if hasattr(net_pp, "res_sgen") and len(net_pp.res_sgen):
            res["res_sgen_p_mw"] = net_pp.res_sgen["p_mw"].to_dict()

        # gen 结果
        if hasattr(net_pp, "res_gen") and len(net_pp.res_gen):
            res["res_gen_p_mw"] = net_pp.res_gen["p_mw"].to_dict()

        # 线路损耗
        if hasattr(net_pp, "res_line") and len(net_pp.res_line):
            res["total_line_pl_mw"] = float(
                np.sum(np.abs(net_pp.res_line["pl_mw"].values))) if "pl_mw" in net_pp.res_line else None

        # 母线电压
        if hasattr(net_pp, "res_bus") and len(net_pp.res_bus):
            res["bus_vm_pu"] = net_pp.res_bus["vm_pu"].to_dict()

    except Exception as e:
        res["warn"] = f"result parsing issue: {type(e).__name__}: {e}"

    return res


import csv
import os
from typing import Dict, List


def init_opf_csv(csv_path: str, opf_res: Dict[str, Any]) -> None:
    """初始化CSV文件并写入表头"""
    fieldnames = ["episode", "success", "total_line_pl_mw"]

    # 添加sgen字段
    if "res_sgen_p_mw" in opf_res:
        for sgen_id in sorted(opf_res["res_sgen_p_mw"].keys()):
            fieldnames.append(f"sgen_{sgen_id}_p_mw")

    # 添加gen字段
    if "res_gen_p_mw" in opf_res:
        for gen_id in sorted(opf_res["res_gen_p_mw"].keys()):
            fieldnames.append(f"gen_{gen_id}_p_mw")

    # 添加错误信息字段
    fieldnames.extend(["error", "warn"])

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def write_opf_to_csv(csv_path: str, episode: int, opf_res: Dict[str, Any]) -> None:
    """将OPF结果写入CSV文件"""
    row = {
        "episode": episode,
        "success": opf_res.get("success", False),
        "total_line_pl_mw": opf_res.get("total_line_pl_mw", ""),
        "error": opf_res.get("error", ""),
        "warn": opf_res.get("warn", "")
    }

    # 添加sgen数据
    if "res_sgen_p_mw" in opf_res:
        for sgen_id, value in opf_res["res_sgen_p_mw"].items():
            row[f"sgen_{sgen_id}_p_mw"] = value

    # 添加gen数据
    if "res_gen_p_mw" in opf_res:
        for gen_id, value in opf_res["res_gen_p_mw"].items():
            row[f"gen_{gen_id}_p_mw"] = value

    # 写入文件
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 如果是新文件，写入表头
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)