import os
import glob
import time
import copy
import random
import warnings
import numpy as np
import torch
import simbench as sb

import VeraGridEngine.api as gce
import GC_PandaPowerImporter

warnings.filterwarnings("ignore")

# ===================== ⚙️ CONFIG =====================
SB_CODE = "1-MV-urban--0-sw"
DATA_DIR = "dataset_output_1mv_urban_dynamic_topo_FULL_STATE"
ASSETS_PATH = os.path.join(DATA_DIR, "static_assets.pt")
CHUNK_PATTERN = os.path.join(DATA_DIR, "chunk_*.pt")

NUM_SAMPLES = 10            # 抽样数
SEED = 123
ENFORCE_LIMITS = True       # True: 开电压/热限；False: 只测算法
INJECT_P_INIT = True        # warm 是否注入 sgen 的 P 初值 (pav*alpha)；建议 True
USE_ACOPF_V0 = True         # 如果 VeraGrid 支持 opts.acopf_v0，则用复数电压向量做 warm start（更硬）
PRINT_SOLVER_DBG = True     # 打印 solver/初始化配置

# ===================== HELPERS =====================
def _set_val(obj, attr_list, val):
    ok = False
    for a in attr_list:
        try:
            setattr(obj, a, val)
            ok = True
        except Exception:
            pass
    return ok

def _get_float(obj, attr_list, default=0.0):
    for a in attr_list:
        if hasattr(obj, a):
            try:
                return float(getattr(obj, a))
            except Exception:
                pass
    return float(default)

def _bus_key(b):
    # 稳：idtag 优先，fallback name
    return str(getattr(b, "idtag", getattr(b, "name", "")))

def build_bus_idx_map_by_template(grid_template):
    # 建立 key->index（包含 idtag 和 name 两套）
    m = {}
    for i, b in enumerate(grid_template.buses):
        k = _bus_key(b)
        if k:
            m[k] = i
        if getattr(b, "name", None):
            m[str(b.name)] = i
    return m

def invert_perm(perm_idx, n):
    inv = np.zeros(n, dtype=int)
    inv[np.array(perm_idx, dtype=int)] = np.arange(n)
    return inv

def load_one_random_sample():
    if not os.path.exists(ASSETS_PATH):
        raise FileNotFoundError(f"Missing {ASSETS_PATH}")

    assets = torch.load(ASSETS_PATH, weights_only=False, map_location="cpu")
    perm = assets["perm"]
    perm_idx = perm.numpy() if hasattr(perm, "numpy") else np.array(perm, dtype=int)

    chunk_files = sorted(glob.glob(CHUNK_PATTERN))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files: {CHUNK_PATTERN}")

    # 随机选 chunk，再随机选样本
    cf = random.choice(chunk_files)
    chunk = torch.load(cf, weights_only=False, map_location="cpu")
    sample = random.choice(chunk)
    return sample, perm_idx

def extract_x_y_in_grid_order(sample, perm_idx, num_nodes):
    """
    sample['x'] shape (3, N) in perm order
    sample['y'] shape (3, N) in perm order
    Return x_phys, y_phys in grid bus order
    """
    invp = invert_perm(perm_idx, num_nodes)
    x = sample["x"].numpy() if hasattr(sample["x"], "numpy") else np.array(sample["x"])
    y = sample["y"].numpy() if hasattr(sample["y"], "numpy") else np.array(sample["y"])
    x_phys = x[:, invp]
    y_phys = y[:, invp]
    return x_phys, y_phys

def find_slack_bus_indices(grid):
    """
    优先使用 bus.is_slack；如果全 False，则回退用 Ext_Grid generator 查 bus
    """
    slack = set()
    for i, b in enumerate(grid.buses):
        if bool(getattr(b, "is_slack", False)):
            slack.add(i)

    if len(slack) > 0:
        return slack

    # fallback: generator name contains Ext_Grid
    for g in getattr(grid, "generators", []):
        if "Ext_Grid" in str(getattr(g, "name", "")):
            b = getattr(g, "bus", None)
            if b is None:
                continue
            try:
                slack.add(grid.buses.index(b))
            except Exception:
                pass
    return slack

# ===================== SCENE INJECTION =====================
def inject_scene_only(grid, x_phys, bus_idx_map):
    """
    只注入“场景边界条件”：Load P/Q；SGEN Pav->Pmax/Pmin；对齐 teacher flags
    不注入任何初值 (P/Vm0/Va0) — cold/warm 都要一致
    """
    n = len(grid.buses)

    bus_loads = {i: [] for i in range(n)}
    for l in getattr(grid, "loads", []):
        bus_ref = getattr(l, "bus", None)
        if bus_ref is None:
            continue
        idx = bus_idx_map.get(_bus_key(bus_ref), bus_idx_map.get(str(getattr(bus_ref, "name", "")), -1))
        if idx != -1:
            bus_loads[idx].append(l)

    bus_gens = {i: [] for i in range(n)}
    for g in getattr(grid, "generators", []):
        bus_ref = getattr(g, "bus", None)
        if bus_ref is None:
            continue
        idx = bus_idx_map.get(_bus_key(bus_ref), bus_idx_map.get(str(getattr(bus_ref, "name", "")), -1))
        if idx != -1:
            bus_gens[idx].append(g)

    for i in range(n):
        # Loads (均分写回；如果你想按原比例分配，可再加权)
        if bus_loads[i]:
            p, q = float(x_phys[0, i]), float(x_phys[1, i])
            for l in bus_loads[i]:
                _set_val(l, ["P", "p"], p / len(bus_loads[i]))
                _set_val(l, ["Q", "q"], q / len(bus_loads[i]))

        # Generators bounds & flags
        pav = float(x_phys[2, i])
        for g in bus_gens[i]:
            name = str(getattr(g, "name", ""))

            # slack/ext grid
            if "Ext_Grid" in name:
                _set_val(g, ["active", "in_service"], True)
                _set_val(g, ["is_controlled"], True)
                _set_val(g, ["enabled_dispatch"], True)
                _set_val(g, ["Pmax", "P_max"], 99999.0)
                _set_val(g, ["Pmin", "P_min"], -99999.0)
                continue

            # sgen
            if "sgen" in name:
                _set_val(g, ["active", "in_service"], True)
                _set_val(g, ["enabled_dispatch"], True)
                _set_val(g, ["is_controlled"], False)   # 对齐你 teacher 脚本
                _set_val(g, ["Pmax", "P_max"], pav)
                _set_val(g, ["Pmin", "P_min"], 0.0)

                # Q 固定为 0（对齐 teacher）
                _set_val(g, ["Q", "q"], 0.0)
                _set_val(g, ["Qmin", "qmin_set"], 0.0)
                _set_val(g, ["Qmax", "qmax_set"], 0.0)

# ===================== INITIALS INJECTION =====================
def set_flat_start(grid, slack_idx=None):
    if slack_idx is None:
        slack_idx = set()
    for i, b in enumerate(grid.buses):
        if i in slack_idx:
            continue
        _set_val(b, ["Vm0"], 1.0)
        _set_val(b, ["Va0"], 0.0)

def inject_warm_initials_from_saved_opf(grid, x_phys, y_phys_deg, slack_idx):
    """
    y_phys_deg: [alpha, Vm, Va(deg)] in grid order
    """
    n = len(grid.buses)

    # A) generator P init (optional)
    if INJECT_P_INIT:
        for i in range(n):
            pav = float(x_phys[2, i])
            alpha = float(y_phys_deg[0, i])
            p_init = pav * alpha
            for g in getattr(grid, "generators", []):
                if getattr(g, "bus", None) == grid.buses[i] and ("sgen" in str(getattr(g, "name", ""))):
                    _set_val(g, ["P", "p"], p_init)

    # B) Vm/Va init (skip slack)
    for i, b in enumerate(grid.buses):
        if i in slack_idx:
            continue
        vm = float(y_phys_deg[1, i])
        va_deg = float(y_phys_deg[2, i])
        va_rad = np.deg2rad(va_deg)
        # wrap to [-pi, pi]
        va_rad = (va_rad + np.pi) % (2 * np.pi) - np.pi

        _set_val(b, ["Vm0"], vm)
        _set_val(b, ["Va0"], va_rad)

def build_v0_from_saved(grid, y_phys_deg, slack_idx):
    """
    用 Vm + Va(deg) 构造复数电压 v0（grid order），slack 可保持 1∠0
    """
    n = len(grid.buses)
    vm = np.array(y_phys_deg[1, :], dtype=float)
    va = np.deg2rad(np.array(y_phys_deg[2, :], dtype=float))
    va = (va + np.pi) % (2 * np.pi) - np.pi

    # slack 强制 1∠0
    for i in slack_idx:
        vm[i] = 1.0
        va[i] = 0.0

    v0 = vm * np.exp(1j * va)
    return v0

# ===================== SOLVER CONFIG =====================
def configure_opf_driver(grid, mode="cold", v0_complex=None):
    """
    mode: cold/warm
    cold: ips_init_with_pf=True
    warm: ips_init_with_pf=False (不覆盖初值)
    """
    opts = gce.OptimalPowerFlowOptions()

    # 尽量把 solver 切到 NONLINEAR_OPF（你之前也是这么写的）
    if hasattr(gce, "SolverType"):
        # 兼容不同属性名/写法：你之前用过 ["solver","solver_type"]
        _set_val(opts, ["solver", "solver_type"], gce.SolverType.NONLINEAR_OPF)

    _set_val(opts, ["activate_voltage_limits"], bool(ENFORCE_LIMITS))
    _set_val(opts, ["activate_thermal_limits"], bool(ENFORCE_LIMITS))
    _set_val(opts, ["dispatch_P"], True)
    _set_val(opts, ["objective"], 0)

    if mode == "cold":
        _set_val(opts, ["ips_init_with_pf"], True)
        # cold 不传 v0
        if hasattr(opts, "acopf_v0"):
            try:
                opts.acopf_v0 = None
            except Exception:
                pass
    else:
        _set_val(opts, ["ips_init_with_pf"], False)
        # warm: 尝试写入 acopf_v0（更硬）
        if USE_ACOPF_V0 and v0_complex is not None and hasattr(opts, "acopf_v0"):
            try:
                opts.acopf_v0 = v0_complex
            except Exception:
                pass

    if PRINT_SOLVER_DBG:
        try:
            print(f"[DBG] mode={mode} solver={getattr(opts,'solver',None)} ips_init_with_pf={getattr(opts,'ips_init_with_pf',None)} acopf_v0={'set' if (hasattr(opts,'acopf_v0') and getattr(opts,'acopf_v0',None) is not None) else 'None'}")
        except Exception:
            pass

    drv = gce.OptimalPowerFlowDriver(grid, opts)
    return drv, opts

def run_driver_timed(drv):
    t0 = time.perf_counter()
    try:
        drv.run()
        ok = bool(getattr(drv.results, "converged", False))
    except Exception:
        ok = False
    t1 = time.perf_counter()
    return ok, (t1 - t0) * 1000.0

def voltage_mae(v_final, v_init):
    if v_final is None or v_init is None:
        return None
    try:
        vf = np.array(v_final)
        vi = np.array(v_init)
        if len(vf) != len(vi):
            return None
        return float(np.mean(np.abs(vf - vi)))
    except Exception:
        return None

# ===================== MAIN =====================
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("🔬 Warm-start 回环验证：用数据集中保存的 OPF 解写回初值，对比 cold/warm 时间")
    print(f"   DATA_DIR={DATA_DIR}")
    print(f"   NUM_SAMPLES={NUM_SAMPLES} ENFORCE_LIMITS={ENFORCE_LIMITS} INJECT_P_INIT={INJECT_P_INIT} USE_ACOPF_V0={USE_ACOPF_V0}")

    # Build template
    net_pp = sb.get_simbench_net(SB_CODE)
    grid_template = GC_PandaPowerImporter.PP2GC(net_pp)
    n = len(grid_template.buses)
    bus_idx_map = build_bus_idx_map_by_template(grid_template)

    # Load assets perm once
    assets = torch.load(ASSETS_PATH, weights_only=False, map_location="cpu")
    perm = assets["perm"]
    perm_idx = perm.numpy() if hasattr(perm, "numpy") else np.array(perm, dtype=int)

    # Collect results
    rows = []
    for k in range(NUM_SAMPLES):
        # random sample
        chunk_files = sorted(glob.glob(CHUNK_PATTERN))
        cf = random.choice(chunk_files)
        chunk = torch.load(cf, weights_only=False, map_location="cpu")
        sample_idx = random.randrange(len(chunk))
        sample = chunk[sample_idx]

        x_phys, y_phys = extract_x_y_in_grid_order(sample, perm_idx, n)

        # Prepare slack list (from template clone)
        grid_tmp_for_slack = copy.deepcopy(grid_template)
        slack_idx = find_slack_bus_indices(grid_tmp_for_slack)

        # -------- Cold --------
        grid_cold = copy.deepcopy(grid_template)
        inject_scene_only(grid_cold, x_phys, bus_idx_map)
        set_flat_start(grid_cold, slack_idx=slack_idx)

        drv_c, opts_c = configure_opf_driver(grid_cold, mode="cold", v0_complex=None)
        ok_c, tc = run_driver_timed(drv_c)

        # -------- Warm --------
        grid_warm = copy.deepcopy(grid_template)
        inject_scene_only(grid_warm, x_phys, bus_idx_map)

        # 写回 warm 初值（Vm0/Va0 + 可选 P init）
        inject_warm_initials_from_saved_opf(grid_warm, x_phys, y_phys, slack_idx=slack_idx)

        # 构造 v0（复数）给 acopf_v0（如果支持）
        v0 = build_v0_from_saved(grid_warm, y_phys, slack_idx=slack_idx)
        drv_w, opts_w = configure_opf_driver(grid_warm, mode="warm", v0_complex=v0)
        ok_w, tw = run_driver_timed(drv_w)

        # 验证写回是否生效：对比最终电压与 v0
        v_mae = None
        try:
            v_final = getattr(drv_w.results, "voltage", None)
            v_mae = voltage_mae(v_final, v0)
        except Exception:
            v_mae = None

        speedup = tc / max(tw, 1e-3)
        status = "PASS" if (ok_c and ok_w) else "FAIL"
        rows.append((os.path.basename(cf), sample_idx, tc, tw, speedup, ok_c, ok_w, v_mae, status))

        print(f"[{k+1:02d}/{NUM_SAMPLES}] cold={tc:8.2f}ms ok={ok_c} | warm={tw:8.2f}ms ok={ok_w} | speedup={speedup:6.2f}x | V_MAE={v_mae} | {status}")

    # Summary
    ok_rows = [r for r in rows if r[5] and r[6]]
    if ok_rows:
        cold_mean = float(np.mean([r[2] for r in ok_rows]))
        warm_mean = float(np.mean([r[3] for r in ok_rows]))
        speed_mean = float(np.mean([r[4] for r in ok_rows]))
        vmae_vals = [r[7] for r in ok_rows if r[7] is not None]
        vmae_mean = float(np.mean(vmae_vals)) if vmae_vals else None

        print("\n==================== SUMMARY (converged only) ====================")
        print(f"Samples converged: {len(ok_rows)}/{len(rows)}")
        print(f"Cold mean (ms): {cold_mean:.2f}")
        print(f"Warm mean (ms): {warm_mean:.2f}")
        print(f"Mean speedup :  {speed_mean:.2f} x")
        print(f"V_init vs V_final MAE: {vmae_mean}")
        print("==================================================================\n")

    # Table
    print(f"{'chunk':<22} {'id':>6} | {'cold(ms)':>10} {'warm(ms)':>10} {'spd':>7} | {'okC':>3} {'okW':>3} | {'V_MAE':>10} | {'st':>4}")
    print("-" * 92)
    for (ck, sid, tc, tw, spd, okc, okw, vmae, st) in rows:
        vmae_s = f"{vmae:.3e}" if isinstance(vmae, (float, np.floating)) else str(vmae)
        print(f"{ck:<22} {sid:>6} | {tc:10.2f} {tw:10.2f} {spd:7.2f} | {str(okc)[0]:>3} {str(okw)[0]:>3} | {vmae_s:>10} | {st:>4}")

if __name__ == "__main__":
    main()
