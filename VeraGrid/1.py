# -*- coding: utf-8 -*-
"""
IEEE-14 (pandapower) -> VeraGrid (MultiCircuit)
并打印：母线/发电机/负荷/线路/变压器 的静态建模数据表
"""

import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pandas as pd
import pandapower as pp
import pandapower.networks as nw

# VeraGrid / GridCal（Vera 分支）
import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en

# 你现有的转换器：pandapower net -> VeraGrid MultiCircuit
import GC_PandaPowerImporter as GC_PandaPowerImporter


# =========================
# 一些数据查看的小工具函数
# =========================
def buses_df(grid):
    rows = []
    for i, bus in enumerate(grid.buses):
        rows.append({
            "idx": i,
            "idtag": bus.idtag,
            "name": bus.name,
            "Vnom_kV": bus.Vnom,
            "is_slack": getattr(bus, "is_slack", False),
            "Vmin_pu": getattr(bus, "Vmin", None),
            "Vmax_pu": getattr(bus, "Vmax", None),
        })
    return pd.DataFrame(rows)

def generators_df(grid):
    rows = []
    for i, g in enumerate(grid.generators):
        rows.append({
            "idx": i,
            "idtag": g.idtag,
            "name": g.name,
            "bus_name": g.bus.name if g.bus else None,
            "P_MW": g.P,
            "Pmin_MW": getattr(g, "Pmin", None),
            "Pmax_MW": getattr(g, "Pmax", None),
            "Qmin_MVAr": getattr(g, "Qmin", None),
            "Qmax_MVAr": getattr(g, "Qmax", None),
            "Vset_pu": getattr(g, "Vset", None),
            "Cost0": getattr(g, "Cost0", None),
            "Cost1": getattr(g, "Cost",  None),
            "Cost2": getattr(g, "Cost2", None),
            "active": getattr(g, "active", True),
        })
    return pd.DataFrame(rows)

def loads_df(grid):
    rows = []
    for i, ld in enumerate(grid.loads):
        rows.append({
            "idx": i,
            "idtag": ld.idtag,
            "name": ld.name,
            "bus_name": ld.bus.name if ld.bus else None,
            "P_MW": ld.P,
            "Q_MVAr": ld.Q,
            "active": getattr(ld, "active", True),
        })
    return pd.DataFrame(rows)

def lines_df(grid):
    rows = []
    for i, ln in enumerate(grid.lines):
        rows.append({
            "idx": i,
            "idtag": ln.idtag,
            "name": ln.name,
            "from_bus": ln.bus_from.name if ln.bus_from else None,
            "to_bus": ln.bus_to.name if ln.bus_to else None,
            "R_pu": ln.R,
            "X_pu": ln.X,
            "B_pu": getattr(ln, "B", None),
            "length_km": getattr(ln, "length", None),
            "rate_MVA": getattr(ln, "rate", None),
            "active": ln.active,
        })
    return pd.DataFrame(rows)

def trafos_df(grid):
    rows = []
    for i, tr in enumerate(grid.transformers2w):
        rows.append({
            "idx": i,
            "idtag": tr.idtag,
            "name": tr.name,
            "hv_bus": tr.bus_from.name if tr.bus_from else None,
            "lv_bus": tr.bus_to.name if tr.bus_to else None,
            "R_pu": tr.R,
            "X_pu": tr.X,
            "rate_MVA": getattr(tr, "rate", None),
            "tap_pos": getattr(tr, "tap_position", None),
            "active": tr.active,
        })
    return pd.DataFrame(rows)

def inspect_bus(grid, bus_name_or_idx):
    """按母线查看挂接设备（打印版）"""
    if isinstance(bus_name_or_idx, int):
        bus = grid.buses[bus_name_or_idx]
    else:
        bus = next(b for b in grid.buses if b.name == str(bus_name_or_idx))

    print(f"\n[BUS] {bus.name}  Vnom={bus.Vnom} kV  slack={getattr(bus,'is_slack',False)}")
    lds = [ld for ld in grid.loads if ld.bus is bus]
    gens = [g for g in grid.generators if g.bus is bus]
    con_lines = [ln for ln in grid.lines if (ln.bus_from is bus or ln.bus_to is bus)]
    con_tr = [tr for tr in grid.transformers2w if (tr.bus_from is bus or tr.bus_to is bus)]

    print(f"  Loads ({len(lds)}): {[ld.name for ld in lds]}")
    print(f"  Gens  ({len(gens)}): {[g.name for g in gens]}")
    print(f"  Lines ({len(con_lines)}): {[ln.name for ln in con_lines]}")
    print(f"  Trafos({len(con_tr)}): {[tr.name for tr in con_tr]}")


# =========================
# 主流程：读取->转换->查看
# =========================
def main():
    # 1) 读取 IEEE-14
    net_pp = nw.case14()
    title = getattr(net_pp, "name", "pandapower net")
    print(f"[{title}] buses={len(net_pp.bus)}, lines={len(net_pp.line)}, "
          f"trafos={len(net_pp.trafo)}, gens={len(net_pp.gen)}+sgen={len(net_pp.sgen)}, loads={len(net_pp.load)}")

    # 2) 可选：在 pandapower 侧先跑一次潮流，检查网是否可行
    try:
        pp.runpp(net_pp, algorithm="nr")
    except Exception as e:
        print("pandapower runpp 失败：", e)

    # 3) 转换为 VeraGrid
    grid_gc = GC_PandaPowerImporter.PP2GC(net_pp)
    print(f"[VeraGrid after convert] buses={len(grid_gc.buses)}, lines={len(grid_gc.lines)}, "
          f"trafos={len(grid_gc.transformers2w)}, gens={len(grid_gc.generators)}, loads={len(grid_gc.loads)}")

    # 4) 打印静态建模数据表（不是仿真结果）
    bdf = buses_df(grid_gc)
    gdf = generators_df(grid_gc)
    lddf = loads_df(grid_gc)
    lndf = lines_df(grid_gc)
    tdf = trafos_df(grid_gc)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 120)

    print("\n=== Buses (静态建模数据) ===")
    print(bdf)

    print("\n=== Generators (静态建模数据) ===")
    print(gdf)

    print("\n=== Loads (静态建模数据) ===")
    print(lddf)

    print("\n=== Lines (静态建模数据) ===")
    print(lndf)

    print("\n=== 2W Transformers (静态建模数据) ===")
    print(tdf)

    # 5) 示例：按母线查看挂接情况（你可以改成你想看的母线）
    inspect_bus(grid_gc, 0)       # 索引 0 的母线
    # inspect_bus(grid_gc, "Bus 5")  # 若你的母线名称是这种风格

    # 6) 如需另存为 CSV，取消下面注释
    # bdf.to_csv("veragrid_buses.csv", index=False, encoding="utf-8-sig")
    # gdf.to_csv("veragrid_generators.csv", index=False, encoding="utf-8-sig")
    # lddf.to_csv("veragrid_loads.csv", index=False, encoding="utf-8-sig")
    # lndf.to_csv("veragrid_lines.csv", index=False, encoding="utf-8-sig")
    # tdf.to_csv("veragrid_trafos.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
