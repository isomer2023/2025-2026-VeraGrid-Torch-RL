#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import copy
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

import pandapower as pp
import simbench as sb


# =========================
# 配置
# =========================
NET_CODE = "1-MV-urban--0-sw"
SAMPLE_IDS_TO_TEST = [81, 14, 3, 94, 35]

TOL_P_SANITY = 1e-5

# 默认 OPF 约束（你可以按需改）
DEFAULT_VM_MIN = 0.95
DEFAULT_VM_MAX = 1.05
DEFAULT_LINE_MAX_LOADING = 100.0
DEFAULT_TRAFO_MAX_LOADING = 100.0

# 默认成本（线性成本足够稳定）
COST_EXT_GRID_CP1 = 1.0      # €/MW
COST_SGEN_CP1 = 2.0          # €/MW（略高一点也无所谓）


# =========================
# 数据结构
# =========================
@dataclass
class Scenario:
    sample_id: int
    load_p_mw: np.ndarray
    load_q_mvar: np.ndarray
    sgen_pav_mw: np.ndarray


@dataclass
class RunResult:
    ok: bool
    elapsed_ms: float
    net: "pp.pandapowerNet"
    used_cfg: str
    err: Optional[Exception] = None


# =========================
# 关键：布尔列清洗（避免 ~float）
# =========================
def sanitize_boolean_columns(net: "pp.pandapowerNet") -> None:
    elem_tables = [
        "bus", "line", "trafo", "trafo3w", "load", "sgen", "gen", "ext_grid",
        "shunt", "switch", "impedance", "storage", "ward", "xward", "dcline",
        "motor", "asymmetric_load", "asymmetric_sgen",
    ]

    for t in elem_tables:
        if hasattr(net, t):
            df = getattr(net, t)
            if isinstance(df, pd.DataFrame) and len(df):
                if "in_service" in df.columns:
                    df["in_service"] = df["in_service"].fillna(True).astype(bool)
                if "controllable" in df.columns:
                    df["controllable"] = df["controllable"].fillna(False).astype(bool)

    if hasattr(net, "switch") and isinstance(net.switch, pd.DataFrame) and len(net.switch):
        if "closed" in net.switch.columns:
            net.switch["closed"] = net.switch["closed"].fillna(True).astype(bool)


# =========================
# ✅ 补齐 OPF 常见缺失：电压上下限、loading 上限、Q 限值
# =========================
def ensure_opf_limits(net: "pp.pandapowerNet") -> None:
    # bus 电压上下限
    if "min_vm_pu" not in net.bus.columns:
        net.bus["min_vm_pu"] = DEFAULT_VM_MIN
    else:
        net.bus["min_vm_pu"] = net.bus["min_vm_pu"].fillna(DEFAULT_VM_MIN)

    if "max_vm_pu" not in net.bus.columns:
        net.bus["max_vm_pu"] = DEFAULT_VM_MAX
    else:
        net.bus["max_vm_pu"] = net.bus["max_vm_pu"].fillna(DEFAULT_VM_MAX)

    # line loading 上限
    if hasattr(net, "line") and len(net.line):
        if "max_loading_percent" not in net.line.columns:
            net.line["max_loading_percent"] = DEFAULT_LINE_MAX_LOADING
        else:
            net.line["max_loading_percent"] = net.line["max_loading_percent"].fillna(DEFAULT_LINE_MAX_LOADING)

    # trafo loading 上限
    if hasattr(net, "trafo") and len(net.trafo):
        if "max_loading_percent" not in net.trafo.columns:
            net.trafo["max_loading_percent"] = DEFAULT_TRAFO_MAX_LOADING
        else:
            net.trafo["max_loading_percent"] = net.trafo["max_loading_percent"].fillna(DEFAULT_TRAFO_MAX_LOADING)

    # 给 controllable sgen / gen / ext_grid 一个“宽松但有限”的 Q 限值（帮助收敛）
    # pandapower：sgen/gen 有 min_q_mvar/max_q_mvar 列时才会进 OPF 约束
    big_q = 1e3

    if hasattr(net, "sgen") and len(net.sgen):
        if "min_q_mvar" not in net.sgen.columns:
            net.sgen["min_q_mvar"] = -big_q
        else:
            net.sgen["min_q_mvar"] = net.sgen["min_q_mvar"].fillna(-big_q)

        if "max_q_mvar" not in net.sgen.columns:
            net.sgen["max_q_mvar"] = big_q
        else:
            net.sgen["max_q_mvar"] = net.sgen["max_q_mvar"].fillna(big_q)

    if hasattr(net, "gen") and len(net.gen):
        if "min_q_mvar" not in net.gen.columns:
            net.gen["min_q_mvar"] = -big_q
        else:
            net.gen["min_q_mvar"] = net.gen["min_q_mvar"].fillna(-big_q)

        if "max_q_mvar" not in net.gen.columns:
            net.gen["max_q_mvar"] = big_q
        else:
            net.gen["max_q_mvar"] = net.gen["max_q_mvar"].fillna(big_q)

    if hasattr(net, "ext_grid") and len(net.ext_grid):
        # ext_grid 也支持 q 限值列（可选）
        if "min_q_mvar" not in net.ext_grid.columns:
            net.ext_grid["min_q_mvar"] = -big_q
        else:
            net.ext_grid["min_q_mvar"] = net.ext_grid["min_q_mvar"].fillna(-big_q)

        if "max_q_mvar" not in net.ext_grid.columns:
            net.ext_grid["max_q_mvar"] = big_q
        else:
            net.ext_grid["max_q_mvar"] = net.ext_grid["max_q_mvar"].fillna(big_q)


# =========================
# ✅ 如果没有成本，自动加 poly_cost，避免 “默认最小总发电”退化
# =========================
def ensure_opf_costs(net: "pp.pandapowerNet") -> None:
    has_poly = hasattr(net, "poly_cost") and isinstance(net.poly_cost, pd.DataFrame) and len(net.poly_cost)
    has_pwl = hasattr(net, "pwl_cost") and isinstance(net.pwl_cost, pd.DataFrame) and len(net.pwl_cost)
    if has_poly or has_pwl:
        return

    # ext_grid 成本
    if hasattr(net, "ext_grid") and len(net.ext_grid):
        for idx in net.ext_grid.index.tolist():
            pp.create_poly_cost(
                net, element=idx, et="ext_grid",
                cp1_eur_per_mw=COST_EXT_GRID_CP1,
                cp0_eur=0.0, cp2_eur_per_mw2=0.0
            )

    # controllable sgen 成本（只给 controllable 的）
    if hasattr(net, "sgen") and len(net.sgen):
        if "controllable" in net.sgen.columns:
            mask = net.sgen["controllable"].fillna(False).astype(bool).values
            sgen_ids = net.sgen.index.values[mask]
        else:
            sgen_ids = net.sgen.index.values

        for idx in sgen_ids.tolist():
            pp.create_poly_cost(
                net, element=int(idx), et="sgen",
                cp1_eur_per_mw=COST_SGEN_CP1,
                cp0_eur=0.0, cp2_eur_per_mw2=0.0
            )


# =========================
# 场景：先用“模板网当前值”造一个能跑的场景
# =========================
def get_scenario_from_existing_net(net_template: "pp.pandapowerNet", sample_id: int) -> Scenario:
    load_p = net_template.load["p_mw"].values.astype(float) if len(net_template.load) else np.array([])
    load_q = net_template.load["q_mvar"].values.astype(float) if len(net_template.load) else np.array([])

    if len(net_template.sgen):
        pav = np.abs(net_template.sgen["p_mw"].values.astype(float))
        # 如果模板里 sgen 本来全是 0，就给一个小上限（否则 OPF 会变得奇怪）
        if np.all(pav <= 0.0):
            pav = np.ones_like(pav) * 0.01
    else:
        pav = np.array([])

    return Scenario(sample_id=sample_id, load_p_mw=load_p, load_q_mvar=load_q, sgen_pav_mw=pav)


# =========================
# 把 Pav 写进 OPF 约束：sgen 可控，P in [0, Pav]
# =========================
def apply_scenario_to_net(net: "pp.pandapowerNet", sc: Scenario) -> None:
    if len(net.load) != len(sc.load_p_mw) or len(net.load) != len(sc.load_q_mvar):
        raise ValueError(f"load dim mismatch: net.load={len(net.load)} scen={len(sc.load_p_mw)}")

    net.load["p_mw"] = sc.load_p_mw
    net.load["q_mvar"] = sc.load_q_mvar

    if len(net.sgen) != len(sc.sgen_pav_mw):
        raise ValueError(f"sgen dim mismatch: net.sgen={len(net.sgen)} scen={len(sc.sgen_pav_mw)}")

    pav = np.clip(sc.sgen_pav_mw, 0.0, None)

    net.sgen["controllable"] = True
    net.sgen["controllable"] = net.sgen["controllable"].fillna(False).astype(bool)

    net.sgen["min_p_mw"] = 0.0
    net.sgen["max_p_mw"] = pav

    # 初值：clip 到 [0, Pav]，不固定
    net.sgen["p_mw"] = np.clip(net.sgen["p_mw"].values.astype(float), 0.0, pav)


# =========================
# sanity check：Pav≈0 但 Pg>0
# =========================
def sanity_check_pav_vs_pg(net: "pp.pandapowerNet", tol: float = TOL_P_SANITY) -> Tuple[int, List[str]]:
    bad_lines: List[str] = []
    if not hasattr(net, "res_sgen") or len(net.res_sgen) != len(net.sgen):
        return 0, bad_lines

    pav = net.sgen["max_p_mw"].values.astype(float)
    pg = net.res_sgen["p_mw"].values.astype(float)

    bad_idx = np.where((pav <= tol) & (pg > tol))[0]
    for i in bad_idx[:50]:
        bus = int(net.sgen.at[i, "bus"])
        bus_name = str(net.bus.at[bus, "name"]) if "name" in net.bus.columns else f"bus{bus}"
        name = str(net.sgen.at[i, "name"]) if "name" in net.sgen.columns else f"sgen_{i}"
        bad_lines.append(f"{name} bus={bus_name} Pav={pav[i]:.6g} Pg={pg[i]:.6g}")

    return int(len(bad_idx)), bad_lines


def voltage_mae(net1: "pp.pandapowerNet", net2: "pp.pandapowerNet") -> float:
    v1 = net1.res_bus["vm_pu"].values.astype(float)
    v2 = net2.res_bus["vm_pu"].values.astype(float)
    return float(np.mean(np.abs(v1 - v2)))


# =========================
# OPF：多配置自动重试（保证“能跑”）
# =========================
def _try_runopp(net_local: "pp.pandapowerNet", cfg: Dict[str, Any]) -> None:
    # cfg 里会包含 runopp 参数
    pp.runopp(
        net_local,
        **cfg
    )


def run_opf_with_retries(
    net: "pp.pandapowerNet",
    mode: str,
    init_with_pf: bool,
    warm_from: Optional["pp.pandapowerNet"] = None,
) -> RunResult:
    t0 = time.perf_counter()
    net_local = copy.deepcopy(net)

    try:
        sanitize_boolean_columns(net_local)
        ensure_opf_limits(net_local)
        ensure_opf_costs(net_local)

        # warm start：注入 results 初值
        if mode == "warm":
            if warm_from is None:
                raise ValueError("warm_from is required in warm mode")
            sanitize_boolean_columns(warm_from)

            if hasattr(warm_from, "res_bus") and len(warm_from.res_bus) == len(net_local.bus):
                net_local.bus["vm_pu"] = warm_from.res_bus["vm_pu"].values
                net_local.bus["va_degree"] = warm_from.res_bus["va_degree"].values

            if hasattr(warm_from, "res_sgen") and len(warm_from.res_sgen) == len(net_local.sgen) and len(net_local.sgen):
                net_local.sgen["p_mw"] = warm_from.res_sgen["p_mw"].values

            if hasattr(warm_from, "res_gen") and len(warm_from.res_gen) == len(net_local.gen) and len(net_local.gen):
                net_local.gen["p_mw"] = warm_from.res_gen["p_mw"].values

            init_mode = "results"
        else:
            init_mode = "pf" if init_with_pf else "flat"

        # 多套配置：从“最严格”到“更宽松”
        cfgs: List[Tuple[str, Dict[str, Any]]] = [
            ("AC_Qlims", dict(ac=True, enforce_q_lims=True, init=init_mode,
                             calculate_voltage_angles=False, numba=False, max_iteration=200)),
            ("AC_noQ",   dict(ac=True, enforce_q_lims=False, init=init_mode,
                             calculate_voltage_angles=False, numba=False, max_iteration=200)),
            ("AC_flat",  dict(ac=True, enforce_q_lims=False, init="flat",
                             calculate_voltage_angles=False, numba=False, max_iteration=300)),
            ("DC_fallback", dict(ac=False, init="flat",
                             calculate_voltage_angles=False, numba=False, max_iteration=200)),
        ]

        last_err: Optional[Exception] = None
        for name, cfg in cfgs:
            try:
                _try_runopp(net_local, cfg)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return RunResult(ok=True, elapsed_ms=elapsed_ms, net=net_local, used_cfg=name, err=None)
            except Exception as e:
                last_err = e
                continue

        # 全失败
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RunResult(ok=False, elapsed_ms=elapsed_ms, net=net_local, used_cfg="ALL_FAILED", err=last_err)

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RunResult(ok=False, elapsed_ms=elapsed_ms, net=net_local, used_cfg="PRECHECK_FAILED", err=e)


# =========================
# main
# =========================
def main():
    print(f"[INFO] loading simbench net: {NET_CODE}")
    net_template = sb.get_simbench_net(NET_CODE)

    sanitize_boolean_columns(net_template)

    # 补齐名字，方便看日志
    if "name" not in net_template.bus.columns:
        net_template.bus["name"] = [f"bus_{i}" for i in range(len(net_template.bus))]
    if len(net_template.sgen) and "name" not in net_template.sgen.columns:
        net_template.sgen["name"] = [f"sgen_{i}" for i in range(len(net_template.sgen))]

    rows = []

    for k, sid in enumerate(SAMPLE_IDS_TO_TEST, 1):
        print(f"\n================ SCENE {k}/{len(SAMPLE_IDS_TO_TEST)} (sample_id={sid}) ================")

        sc = get_scenario_from_existing_net(net_template, sid)
        base = copy.deepcopy(net_template)
        apply_scenario_to_net(base, sc)

        sanitize_boolean_columns(base)
        ensure_opf_limits(base)
        ensure_opf_costs(base)

        print("[DBG] mode=cold init=pf")
        r1 = run_opf_with_retries(base, mode="cold", init_with_pf=True, warm_from=None)
        if not r1.ok:
            print(f"[ERR] cold failed ({r1.used_cfg}): {r1.err}")
            rows.append((sid, r1.elapsed_ms, np.nan, np.nan, False, False, r1.used_cfg, ""))
            continue
        else:
            print(f"[DBG] cold ok using={r1.used_cfg}  time={r1.elapsed_ms:.2f}ms")

        print("[DBG] mode=warm init=results")
        r2 = run_opf_with_retries(base, mode="warm", init_with_pf=False, warm_from=r1.net)
        if not r2.ok:
            print(f"[ERR] warm failed ({r2.used_cfg}): {r2.err}")
            rows.append((sid, r1.elapsed_ms, r2.elapsed_ms, np.nan, True, False, r1.used_cfg, r2.used_cfg))
            continue
        else:
            print(f"[DBG] warm ok using={r2.used_cfg}  time={r2.elapsed_ms:.2f}ms")

        spd = r1.elapsed_ms / max(r2.elapsed_ms, 1e-9)
        vmae = voltage_mae(r1.net, r2.net)

        print(f"[{k:02d}/{len(SAMPLE_IDS_TO_TEST)}] cold={r1.elapsed_ms:8.2f}ms({r1.used_cfg}) | "
              f"warm={r2.elapsed_ms:8.2f}ms({r2.used_cfg}) | speedup={spd:6.2f}x | V_MAE={vmae:.6f}")

        bad1, bad1_lines = sanity_check_pav_vs_pg(r1.net)
        bad2, bad2_lines = sanity_check_pav_vs_pg(r2.net)
        if bad1 or bad2:
            print("⚠️ SANITY WARNING: found (Pmax≈0 but Pg>0) non-slack generators.")
            print(f"   OPF#1 bad={bad1} | OPF#2 bad={bad2}")
            for line in (bad1_lines[:10] + bad2_lines[:10])[:10]:
                print(f"   bad_gen: {line}")

        rows.append((sid, r1.elapsed_ms, r2.elapsed_ms, spd, True, True, r1.used_cfg, r2.used_cfg))

    df = pd.DataFrame(
        rows,
        columns=["sample_id", "cold_ms", "warm_ms", "speedup", "ok1", "ok2", "cold_cfg", "warm_cfg"]
    )
    print("\n==================== SUMMARY ====================")
    print(df.to_string(index=False))
    print("=================================================\n")


if __name__ == "__main__":
    main()
