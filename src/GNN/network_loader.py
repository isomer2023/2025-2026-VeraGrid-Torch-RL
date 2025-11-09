from __future__ import annotations
import numpy as np
import copy
import pandas as pd

import simbench as sb
import pandapower as pp

import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en
import src.GC_PandaPowerImporter as GC_PandaPowerImporter

# 轻量图观测（供 GNN 使用）
class GraphObs:
    # node_feat : (N, F_n) float32  列：[P_load, P_cap]
    # edge_index: (2, E) int64      0-based，双向边
    # edge_feat : (E, F_e) float32  列：[rate_MVA]
    __slots__ = ("node_feat","edge_index","edge_feat",
                 "node_names","sgen_map","sgen_pmax","act_min","act_max")
    def __init__(self, node_feat, edge_index, edge_feat,
                 node_names, sgen_map, sgen_pmax, act_min, act_max):
        self.node_feat = node_feat
        self.edge_index = edge_index
        self.edge_feat = edge_feat
        self.node_names = node_names
        self.sgen_map = sgen_map
        self.sgen_pmax = sgen_pmax
        self.act_min = act_min
        self.act_max = act_max

# Pandapower 清洗
def _clean_nan_fields(net):
    if 'vn_kv' in net.bus.columns:
        net.bus['vn_kv'] = net.bus['vn_kv'].copy().fillna(110)
    if 'in_service' in net.bus.columns:
        net.bus['in_service'] = net.bus['in_service'].copy().fillna(True)
    net.bus = net.bus.fillna(0)

    if len(net.line):
        for col in ['r_ohm_per_km','x_ohm_per_km','c_nf_per_km','max_i_ka','length_km']:
            if col in net.line.columns:
                net.line[col] = net.line[col].copy().fillna(0)
        if 'in_service' in net.line.columns:
            net.line['in_service'] = net.line['in_service'].copy().fillna(True)
        net.line = net.line.fillna(0)

    if len(net.trafo):
        net.trafo = net.trafo.fillna(0)
        if 'in_service' in net.trafo.columns:
            net.trafo['in_service'] = net.trafo['in_service'].copy().fillna(True)

    for elm in ['load','sgen','gen','ext_grid','storage','switch']:
        df = getattr(net, elm, None)
        if df is not None and len(df):
            df = df.fillna(0)
            if 'in_service' in df.columns:
                df['in_service'] = df['in_service'].copy().fillna(True)
            setattr(net, elm, df)

# Pandapower to VeraGrid，线路额定值与 sgen映射
def load_simbench_as_veragrid(sb_code: str):
    net_pp = sb.get_simbench_net(sb_code)
    _clean_nan_fields(net_pp)

    if len(net_pp.ext_grid):
        net_pp.ext_grid.at[net_pp.ext_grid.index[0], 'name'] = "grid_ext"

    pp.runpp(net_pp, algorithm='nr', numba=False, enforce_q_lims=True,
             init='auto', tolerance_mva=1e-6, calculate_voltage_angles=True)

    grid_gc = GC_PandaPowerImporter.PP2GC(net_pp)

    # 线路热低限
    try:
        assigned = 0
        missing_idxs = []
        if len(net_pp.line) == len(grid_gc.lines):
            for idx, (ln_pp, ln_gc) in enumerate(zip(net_pp.line.itertuples(), grid_gc.lines)):
                i_ka = getattr(ln_pp, 'max_i_ka', np.nan)
                fb = net_pp.bus.at[ln_pp.from_bus, 'vn_kv'] if ln_pp.from_bus in net_pp.bus.index else np.nan
                tb = net_pp.bus.at[ln_pp.to_bus, 'vn_kv'] if ln_pp.to_bus in net_pp.bus.index else np.nan
                vn = float(fb if not np.isnan(fb) else tb)
                if (not np.isnan(i_ka)) and (not np.isnan(vn)) and (i_ka > 0.0) and (vn > 0.0):
                    ln_gc.rate = float(np.sqrt(3.0) * vn * i_ka)  # MVA
                    assigned += 1
                else:
                    missing_idxs.append(idx)

        if len(net_pp.line) and (assigned < len(net_pp.line)):
            missing = len(net_pp.line) - assigned
            print(f"[Error-Soft] LINE RHO LOWER LIMIT MISSING{missing}/{len(net_pp.line)}, USING DEFAULT")
            DEFAULT_RATE = 10.0
            for idx in missing_idxs:
                try:
                    if getattr(grid_gc.lines[idx], "rate", None) in [None, 0.0] or np.isnan(
                            getattr(grid_gc.lines[idx], "rate", np.nan)):
                        grid_gc.lines[idx].rate = float(DEFAULT_RATE)
                except Exception:
                    pass
    except Exception as e:
        print(f"[Error-Soft] LINE RHO LOWER LIMIT ERROR{e}, USING DEFAULT")
        DEFAULT_RATE = 10.0
        try:
            for ln_gc in grid_gc.lines:
                if getattr(ln_gc, "rate", None) in [None, 0.0] or np.isnan(getattr(ln_gc, "rate", np.nan)):
                    ln_gc.rate = float(DEFAULT_RATE)
        except Exception:
            pass

    # sgen映射所有可控发电机
    name_to_bus = {}
    for b in grid_gc.buses:
        key = str(getattr(b, "name", ""))
        name_to_bus[key] = b
        try:
            name_to_bus[str(int(key))] = b
        except Exception:
            pass

    created = 0
    try:
        sgen_df = net_pp.sgen.copy()
        if len(sgen_df):
            has_bus_name = 'name' in net_pp.bus.columns
            for sid, row in sgen_df.iterrows():
                p_mw = float(row.get("p_mw", 0.0))
                if p_mw <= 1e-9:
                    continue
                bus_idx = int(row["bus"])
                pp_bus_name = str(net_pp.bus.at[bus_idx, "name"]) if has_bus_name else str(bus_idx)
                b_gc = name_to_bus.get(pp_bus_name) or name_to_bus.get(str(bus_idx))
                if b_gc is None:
                    continue
                Pmax = p_mw; Pmin = 0.0
                g = gce.Generator()
                g.name = f"sgen_{sid}"
                g.bus = b_gc
                g.Pmin, g.Pmax = float(Pmin), float(Pmax)
                g.Qmin, g.Qmax = -0.5 * Pmax, 0.5 * Pmax
                g.Cost, g.Cost2 = 0.0, 0.0
                g.Vset = getattr(b_gc, "Vset", 1.0)
                grid_gc.generators.append(g)
                created += 1
        print(f"[Info] Created {created} controllable generators from PP.sgen (Pmax := sgen.p_mw).")
    except Exception as e:
        print(f"[Warn] sgen→generator mapping skipped: {e}")

    return net_pp, grid_gc
