# network_loader.py
import numpy as np
import pandapower as pp
import pandapower.networks as nw
import VeraGridEngine.api as gce
import VeraGridEngine.enumerations as en
import src.GC_PandaPowerImporter as GC_PandaPowerImporter

def set_line_limits_auto(grid_gc, k_scale=0.8, margin=1.25):
    """Line rate automatically set, based on simple impedance / PF, adapt to any network."""
    # imp
    for ln in grid_gc.lines:
        V = getattr(ln.bus_from, "Vnom", getattr(ln.bus_to, "Vnom", 110.0))# prevent no bus_from
        R, X = ln.R, ln.X
        Z = max((R**2 + X**2)**0.5, 1e-6)
        ln.rate = k_scale * (V**2 / Z)
        ln.rate = min(ln.rate, 1000.0)  # prevent unlimited rate

    # PF
    try:
        pf = gce.PowerFlowDriver(
            grid=grid_gc,
            options=gce.PowerFlowOptions(solver_type=en.SolverType.NR, verbose=False)
        )
        pf.run()
        if pf.results.converged:
            bdf = pf.results.get_branch_df()
            Sf = np.hypot(bdf["Pf"].astype(float), bdf["Qf"].astype(float))
            for ln, sf in zip(grid_gc.lines, Sf):
                ln.rate = max(ln.rate, margin * float(sf))
    except Exception as e:
        print("[WARNING] Auto line limit PF failed:", e)

def load_network(name="case14", run_pp_before=True, sanitize=True, set_line_rate_100=True):
    if name.lower() == "case14":
        net_pp = nw.case14()
    elif name.lower() == "case39":
        net_pp = nw.case39()
    elif name.lower() == "case118":
        net_pp = nw.case118()
    else:
        raise ValueError(f"Unsupported network: {name}")

    net_pp.bus["max_vm_pu"] = np.maximum(net_pp.bus.get("max_vm_pu", 1.08), 1.09)# 1.05,1.06
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

    set_line_limits_auto(grid_gc)

    #if set_line_rate_100:
    #    for ln in grid_gc.lines:
    #        ln.rate = 100.0

    return net_pp, grid_gc