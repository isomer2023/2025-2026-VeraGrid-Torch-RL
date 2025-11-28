# src/GNN/network_opf.py
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import traceback
import inspect

import VeraGridEngine.api as gce
from src.GNN.network_loader import load_simbench_as_veragrid


# --------------------------------------------------------------
# 工具：解析 sgen_{id}
# --------------------------------------------------------------
def _parse_sgen_id_from_name(name: str) -> Optional[int]:
    if not isinstance(name, str):
        return None
    if name.startswith("sgen_"):
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return None
    return None


# --------------------------------------------------------------
# 最终版：安全构造 OPF
# 不传 options，不传 verbose，不传 solver
# 只要 __init__ 支持就传入 grid_gc
# --------------------------------------------------------------
def safe_construct_opf(driver_cls, grid_gc):
    """
    自动检测 driver_cls 的 __init__，仅在其接受多个参数时传入 grid_gc。
    若仅支持 driver_cls(grid) 则只传 grid。
    """
    try:
        sig = inspect.signature(driver_cls.__init__)
        params = sig.parameters
    except Exception:
        # 无法检查，直接用 grid-only
        return driver_cls(grid_gc)

    # 如果只有 self 和一个参数，那么就是 grid-only 模式
    if len(params) <= 2:
        return driver_cls(grid_gc)

    # 如果支持多个参数，我们尝试只传 grid 参数，不传任何 keyword argument
    try:
        return driver_cls(grid_gc)
    except Exception:
        # 最后 fallback
        return driver_cls(grid_gc)


# --------------------------------------------------------------
# 运行 VeraGrid OPF
# --------------------------------------------------------------
def run_veragrid_opf_for_env(env, sb_code: Optional[str] = None,
                             use_sgen_values_from_env: bool = True) -> Dict[str, Any]:

    sb_code = sb_code or getattr(env, "sb_code", None)
    if sb_code is None:
        return {"success": False, "error": "sb_code not provided and not found in env."}

    # 加载网表
    try:
        net_pp, grid_gc = load_simbench_as_veragrid(sb_code)
    except Exception as e:
        return {
            "success": False,
            "error": f"load_simbench_as_veragrid error: {type(e).__name__}: {e}",
            "trace": traceback.format_exc()
        }

    # 按名字建立 generator 映射
    gridgc_gen_by_name = {str(getattr(g, "name", "")): g for g in grid_gc.generators}

    # ----------------------------------------------------------
    # 强制 sgen 可控
    # ----------------------------------------------------------
    try:
        for g_env in env.ctrl_gens:
            gname = str(getattr(g_env, "name", "")).strip()
            if not gname:
                continue

            g_gc = gridgc_gen_by_name.get(gname)

            if g_gc is None:
                sid = _parse_sgen_id_from_name(gname)
                if sid is not None:
                    g_gc = gridgc_gen_by_name.get(f"sgen_{sid}")

            if g_gc is None:
                continue

            # 写 P/Pmin/Pmax/Cost
            P = float(getattr(g_env, "P", getattr(g_env, "Pmax", 0.0)))
            Pmin = float(getattr(g_env, "Pmin", 0.0))
            Pmax = float(getattr(g_env, "Pmax", max(Pmin, P)))
            if Pmax < Pmin:
                Pmax = Pmin + 1e-6

            # 写入输出
            for attr in ("P", "Pset", "Pg", "Ptarget"):
                try:
                    setattr(g_gc, attr, P)
                except:
                    pass

            # 边界
            try:
                g_gc.Pmin = Pmin
                g_gc.Pmax = Pmax
            except:
                pass

            # 成本
            try:
                g_gc.Cost = float(getattr(g_env, "Cost", 0.0))
                g_gc.Cost2 = float(getattr(g_env, "Cost2", 0.0))
            except:
                pass

            # 非 slack
            try:
                if hasattr(g_gc, "is_slack"): g_gc.is_slack = False
                if hasattr(g_gc, "slack"):    g_gc.slack = False
            except:
                pass

    except Exception as e:
        return {
            "success": False,
            "error": f"apply ctrl_gens error: {type(e).__name__}: {e}",
            "trace": traceback.format_exc()
        }

    # ----------------------------------------------------------
    # 获取 OPF Driver（兼容多个命名）
    # ----------------------------------------------------------
    OpfDriverCls = (
        getattr(gce, "OptimalPowerFlowDriver", None) or
        getattr(gce, "OptimalPowerFlow", None) or
        getattr(gce, "OPFDriver", None)
    )

    if OpfDriverCls is None:
        return {
            "success": False,
            "error": "No OPF driver found in VeraGridEngine.api"
        }

    # ----------------------------------------------------------
    # 构造 & 运行 OPF
    # ----------------------------------------------------------
    try:
        opf = safe_construct_opf(OpfDriverCls, grid_gc)
        opf.run()
    except Exception as e:
        return {
            "success": False,
            "error": f"VeraGrid OPF failed: {type(e).__name__}: {e}",
            "trace": traceback.format_exc()
        }

    # ----------------------------------------------------------
    # 提取结果
    # ----------------------------------------------------------
    res: Dict[str, Any] = {"success": True}

    try:
        res_sgen = {}
        res_gen = {}

        for g in grid_gc.generators:
            name = str(getattr(g, "name", "")).strip()

            try:
                P = float(getattr(g, "P", getattr(g, "Pg", 0.0)))
            except:
                P = 0.0

            sid = _parse_sgen_id_from_name(name)
            if sid is not None:
                res_sgen[sid] = P
            else:
                res_gen[name if name else f"gen_{id(g)}"] = P

        if res_sgen:
            res["res_sgen_p_mw"] = res_sgen
        if res_gen:
            res["res_gen_p_mw"] = res_gen

        # 总损耗（尝试从 results.losses 中读取）
        total_loss = None
        try:
            if hasattr(opf, "results") and opf.results is not None:
                if hasattr(opf.results, "losses"):
                    L = np.array(opf.results.losses, dtype=complex)
                    if L.size > 0:
                        total_loss = float(np.real(L.sum()))
        except:
            total_loss = None

        res["total_line_pl_mw"] = total_loss

    except Exception as e:
        res["warn"] = f"parsing warning: {type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc()

    return res


# --------------------------------------------------------------
# CSV 输出（保持你的结构）
# --------------------------------------------------------------
import csv, os

def init_opf_csv(csv_path: str, opf_res: Dict[str, Any]) -> None:
    fieldnames = ["episode", "success", "total_line_pl_mw"]

    if "res_sgen_p_mw" in opf_res:
        for sid in sorted(opf_res["res_sgen_p_mw"].keys()):
            fieldnames.append(f"sgen_{sid}_p_mw")

    if "res_gen_p_mw" in opf_res:
        for gid in sorted(opf_res["res_gen_p_mw"].keys()):
            fieldnames.append(f"gen_{gid}_p_mw")

    fieldnames += ["error", "warn"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def write_opf_to_csv(csv_path: str, ep: int, opf_res: Dict[str, Any]) -> None:
    row = {
        "episode": ep,
        "success": opf_res.get("success", False),
        "total_line_pl_mw": opf_res.get("total_line_pl_mw", ""),
        "error": opf_res.get("error", ""),
        "warn": opf_res.get("warn", "")
    }

    if "res_sgen_p_mw" in opf_res:
        for sid, v in opf_res["res_sgen_p_mw"].items():
            row[f"sgen_{sid}_p_mw"] = v

    if "res_gen_p_mw" in opf_res:
        for gid, v in opf_res["res_gen_p_mw"].items():
            row[f"gen_{gid}_p_mw"] = v

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
