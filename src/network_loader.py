import numpy as np
import pandapower as pp
import pandapower.networks as nw
import src.GC_PandaPowerImporter as GC_PandaPowerImporter

def load_network(name="case14", run_pp_before=True, sanitize=True, set_line_rate_100=True):
    if name.lower() == "case14":
        net_pp = nw.case14()
    elif name.lower() == "case39":
        net_pp = nw.case39()
    elif name.lower() == "case118":
        net_pp = nw.case118()
    else:
        raise ValueError(f"Unsupported network: {name}")

    net_pp.bus["max_vm_pu"] = np.maximum(net_pp.bus.get("max_vm_pu", 1.05), 1.06)
    if run_pp_before:
        try:
            pp.runpp(net_pp, algorithm="nr", init="flat")
        except Exception as e:
            print(f"pandapower runpp failed: {e}")

    grid_gc = GC_PandaPowerImporter.PP2GC(net_pp)

    if sanitize:
        for b in grid_gc.buses:
            b.Vmin, b.Vmax = 0.95, 1.05
        for g in grid_gc.generators:
            g.Vset = min(max(g.Vset, 0.95), 1.03)

    if set_line_rate_100:
        for ln in grid_gc.lines:
            ln.rate = 100.0

    return net_pp, grid_gc