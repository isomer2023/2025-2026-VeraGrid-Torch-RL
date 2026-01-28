
import GC_PandaPowerImporter
from VeraGridEngine import api as gce

def _idx_from_name(name):
    try:
        return int(str(name).split("_")[1])
    except Exception:
        return None

def _set_val(obj, attr_list, val):
    for a in attr_list:
        try:
            setattr(obj, a, val)
            return True
        except Exception:
            pass
    return False

def _get_val(obj, attr_list, default=None):
    for a in attr_list:
        if hasattr(obj, a):
            try:
                v = getattr(obj, a)
                if v is None:
                    continue
                return v
            except Exception:
                pass
    return default

def _get_float(obj, attr_list, default=0.0):
    v = _get_val(obj, attr_list, None)
    try:
        return float(v)
    except Exception:
        return float(default)

def _is_slack(g):
    return "Ext_Grid" in str(getattr(g, "name", ""))

def _is_sgen(g):
    return "sgen" in str(getattr(g, "name", ""))

def lock_Q_as_PQ(gen, Q_fixed=0.0):
    _set_val(gen, ["Q", "q"], Q_fixed)
    _set_val(gen, ["Qmin", "qmin_set"], Q_fixed)
    _set_val(gen, ["Qmax", "qmax_set"], Q_fixed)

def get_opf_gen_p(opf_driver, grid):
    res = getattr(opf_driver, "results", None)
    if res is not None and hasattr(res, "generator_power"):
        try:
            return [float(v) for v in res.generator_power]
        except Exception:
            pass
    return [_get_float(g, ["P", "p"], 0.0) for g in grid.generators]

def apply_scene_PQ(grid_scene, load_p_row, load_q_row, pav_dict):
    # loads
    for l in getattr(grid_scene, "loads", []):
        lid = _idx_from_name(getattr(l, "name", "load_0"))
        if lid is None:
            continue
        _set_val(l, ["P", "p"], float(load_p_row.get(lid, 0.0)))
        _set_val(l, ["Q", "q"], float(load_q_row.get(lid, 0.0)))

    # generators
    for g in getattr(grid_scene, "generators", []):
        if _is_slack(g):
            _set_val(g, ["active", "in_service"], True)
            _set_val(g, ["is_controlled"], True)
            _set_val(g, ["Pmax", "P_max"], 99999.0)
            _set_val(g, ["Pmin", "P_min"], -99999.0)
            continue

        if _is_sgen(g):
            gid = _idx_from_name(getattr(g, "name", "sgen_0"))
            pav = float(pav_dict.get(gid, 0.0))

            # 关键：P 和 Pmax 分开设置（避免只设到Pmax）
            _set_val(g, ["P", "p"], pav)
            _set_val(g, ["Pmax", "P_max"], pav)
            _set_val(g, ["Pmin", "P_min"], 0.0)

            # PQ 化
            _set_val(g, ["is_controlled"], False)
            _set_val(g, ["enabled_dispatch"], True)

            # Q 锁死
            lock_Q_as_PQ(g, 0.0)

def run_opf_teacher(grid_opf, thermal_limits=True):
    # Costs: PV cheap, slack expensive
    for g in getattr(grid_opf, "generators", []):
        _set_val(g, ["active", "in_service"], True)
        if _is_slack(g):
            _set_val(g, ["cost_a"], 1.0)
            _set_val(g, ["is_controlled"], True)  # slack 保持可控（平衡节点）
            _set_val(g, ["Pmax", "P_max"], 99999.0)
            _set_val(g, ["Pmin", "P_min"], -99999.0)
        elif _is_sgen(g):
            _set_val(g, ["cost_a"], 0.01)
            # sgen 保持 PQ + Q锁死，只让 OPF 调P
            _set_val(g, ["is_controlled"], False)
            _set_val(g, ["enabled_dispatch"], True)

            # 确保Q仍是锁死（防止有人改了scene）
            lock_Q_as_PQ(g, 0.0)

            # 确保Pmin存在
            _set_val(g, ["Pmin", "P_min"], 0.0)

    opf_opts = gce.OptimalPowerFlowOptions()
    if hasattr(gce, "SolverType"):
        _set_val(opf_opts, ["solver", "solver_type"], gce.SolverType.NONLINEAR_OPF)

    _set_val(opf_opts, ["activate_voltage_limits"], True)
    _set_val(opf_opts, ["activate_thermal_limits"], bool(thermal_limits))
    _set_val(opf_opts, ["dispatch_P"], True)
    _set_val(opf_opts, ["objective"], 0)

    drv = gce.OptimalPowerFlowDriver(grid_opf, opf_opts)
    try:
        drv.run()
    except Exception:
        pass
    return drv