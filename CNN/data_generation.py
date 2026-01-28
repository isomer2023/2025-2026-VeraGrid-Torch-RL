import os
import warnings
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
import simbench as sb

warnings.filterwarnings("ignore")

try:
    import GC_PandaPowerImporter
    from VeraGridEngine import api as gce
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    raise

# ================= ⚙️ 配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(CURRENT_DIR, "dataset_output_1MVLV-urban")
SB_CODE = "1-MVLV-urban-5.303-0-no_sw"#"1-MV-urban--0-sw"

NUM_SAMPLES = 2000
CHUNK_SIZE = 100

SGEN_POWER_THRESHOLD = 1e-3          # 过滤“整天没光伏”的时刻
SGEN_NODE_THRESHOLD = 1e-4           # 单台sgen低于这个，就不当作可控PV节点（mask=False）
RATE_TIGHTEN_FACTOR = 1.0

# k 采样（默认足够覆盖你测出来的电压范围）
K_GLOBAL_MIN, K_GLOBAL_MAX = 0.4, 3.6
K_LOCAL_MIN, K_LOCAL_MAX = 0.85, 1.15

# 分桶比例（默认 40/40/20）
USE_BUCKET_SAMPLING = True
BUCKET_TARGET = {"safe": 0.40, "mid": 0.40, "high": 0.20}
V_SAFE = 1.03
V_LIMIT = 1.05

MAX_TRIES_PER_SAMPLE = 200  # 防止死循环

# ==========================================

# 加载资产
ASSETS_PATH = os.path.join(SAVE_DIR, "static_assets.pt")
if not os.path.exists(ASSETS_PATH):
    raise FileNotFoundError("❌ 找不到 static_assets.pt")

ASSETS = torch.load(ASSETS_PATH, weights_only=False)
PERM_IDX = ASSETS["perm"]
NUM_NODES = ASSETS["num_nodes"]

# ----------- robust helpers -----------
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

def _idx_from_name(name):
    try:
        return int(str(name).split("_")[1])
    except Exception:
        return None

def lock_Q_as_PQ(gen, Q_fixed=0.0):
    _set_val(gen, ["Q", "q"], Q_fixed)
    _set_val(gen, ["Qmin", "qmin_set"], Q_fixed)
    _set_val(gen, ["Qmax", "qmax_set"], Q_fixed)

def tighten_thermal_limits(grid, factor=1.0):
    branches = list(getattr(grid, "lines", [])) + list(getattr(grid, "transformers", []))
    for br in branches:
        old_rate = _get_float(br, ["rate", "Rate"], 100.0)
        _set_val(br, ["rate", "Rate"], old_rate * factor)

def find_slack_bus_idx(grid, bus_idx_map):
    for g in getattr(grid, "generators", []):
        if _is_slack(g):
            bus_ref = getattr(g, "bus", getattr(g, "node", None))
            if bus_ref is None:
                return None
            bid = str(getattr(bus_ref, "idtag", getattr(bus_ref, "id", getattr(bus_ref, "name", None))))
            if bid in bus_idx_map:
                return bus_idx_map[bid]
            # fallback by name
            if hasattr(bus_ref, "name") and str(bus_ref.name) in bus_idx_map:
                return bus_idx_map[str(bus_ref.name)]
    return None

# ----------- PF / OPF drivers -----------
def run_pf(grid_pf):
    pf_opts = gce.PowerFlowOptions(gce.SolverType.NR, verbose=False)
    _set_val(pf_opts, ["control_taps_modules"], False)
    _set_val(pf_opts, ["control_taps_phase"], False)
    _set_val(pf_opts, ["control_remote_voltage"], False)

    drv = gce.PowerFlowDriver(grid_pf, pf_opts)
    drv.run()
    return drv

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

def get_opf_gen_p(opf_driver, grid):
    res = getattr(opf_driver, "results", None)
    if res is not None and hasattr(res, "generator_power"):
        try:
            return [float(v) for v in res.generator_power]
        except Exception:
            pass
    return [_get_float(g, ["P", "p"], 0.0) for g in grid.generators]

# ----------- scene application -----------
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

def set_bus_voltage_limits(grid, vmin=0.95, vmax=1.05):
    for b in getattr(grid, "buses", []):
        _set_val(b, ["Vmin"], vmin)
        _set_val(b, ["Vmax"], vmax)

# ----------- build tensor sample -----------
def build_tensor_sample(grid_opf, pf_v_abs, pf_v_angle, opf_gen_p, bus_idx_map, pav_dict):
    num_nodes = len(grid_opf.buses)
    if num_nodes != NUM_NODES:
        return None

    # 5通道: Load P, Load Q, PV_avail(Pmax), V_mag, V_angle
    x = np.zeros((num_nodes, 5), dtype=np.float32)

    # loads
    for l in grid_opf.loads:
        bus_ref = getattr(l, "bus", getattr(l, "node", None))
        if bus_ref is None:
            continue
        bid = str(getattr(bus_ref, "idtag", getattr(bus_ref, "id", getattr(bus_ref, "name", None))))
        idx = bus_idx_map.get(bid, None)
        if idx is None and hasattr(bus_ref, "name"):
            idx = bus_idx_map.get(str(bus_ref.name), None)
        if idx is None:
            continue
        x[idx, 0] += _get_float(l, ["P", "p"], 0.0)
        x[idx, 1] += _get_float(l, ["Q", "q"], 0.0)

    # voltage
    x[:, 3] = pf_v_abs
    x[:, 4] = pf_v_angle

    # generators -> PV avail & labels
    sgen_mask = np.zeros(num_nodes, dtype=bool)
    y_target = np.zeros((num_nodes, 1), dtype=np.float32)

    for i, g in enumerate(grid_opf.generators):
        bus_ref = getattr(g, "bus", getattr(g, "node", None))
        if bus_ref is None:
            continue
        bid = str(getattr(bus_ref, "idtag", getattr(bus_ref, "id", getattr(bus_ref, "name", None))))
        idx = bus_idx_map.get(bid, None)
        if idx is None and hasattr(bus_ref, "name"):
            idx = bus_idx_map.get(str(bus_ref.name), None)
        if idx is None:
            continue

        if _is_sgen(g):
            gid = _idx_from_name(getattr(g, "name", "sgen_0"))
            pav = float(pav_dict.get(gid, 0.0))

            # 输入通道：PV_avail
            x[idx, 2] += pav

            # mask：只有 pav 足够大才当可控PV
            if pav > SGEN_NODE_THRESHOLD:
                sgen_mask[idx] = True
                popt = float(opf_gen_p[i])
                alpha = np.clip(popt / max(pav, 1e-12), 0.0, 1.0)
                y_target[idx, 0] = alpha

        elif _is_slack(g):
            # slack 也可以把 Pmax 填进通道2（可选），但不进mask、不写y
            x[idx, 2] += 0.0

    # RCM 重排
    x_re = x[PERM_IDX]
    y_re = y_target[PERM_IDX]
    m_re = sgen_mask[PERM_IDX]

    return {
        "x": torch.tensor(x_re.T, dtype=torch.float32),      # (5, N)
        "y": torch.tensor(y_re.T, dtype=torch.float32),      # (1, N)
        "mask": torch.tensor(m_re, dtype=torch.bool)         # (N,)
    }

# ----------- bucket helper -----------
def bucket_of_vmax(vmax):
    if vmax <= V_SAFE:
        return "safe"
    elif vmax <= V_LIMIT:
        return "mid"
    else:
        return "high"

def should_accept_bucket(bucket_counts, total_kept, bucket_name):
    if not USE_BUCKET_SAMPLING:
        return True
    # 目标数（动态）
    targets = {k: int(BUCKET_TARGET[k] * NUM_SAMPLES) for k in BUCKET_TARGET}
    return bucket_counts[bucket_name] < targets[bucket_name]

# ================= main =================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"🚀 生成数据中... 目标: {NUM_SAMPLES}")
    net_pp = sb.get_simbench_net(SB_CODE)
    grid_template = GC_PandaPowerImporter.PP2GC(net_pp)

    # bus idx map（用 idtag / id / name 做key）
    bus_idx_map = {}
    for i, b in enumerate(grid_template.buses):
        key = str(getattr(b, "idtag", getattr(b, "id", getattr(b, "name", None))))
        if key is not None:
            bus_idx_map[str(key)] = i
        if hasattr(b, "name") and b.name is not None:
            bus_idx_map[str(b.name)] = i

    profiles = sb.get_absolute_values(net_pp, profiles_instead_of_study_cases=True)
    df_load_p = profiles[("load", "p_mw")]
    df_load_q = profiles[("load", "q_mvar")]
    df_sgen_p = profiles[("sgen", "p_mw")]

    # 只挑“当天总PV>阈值”的时刻
    valid_ts = [t for t in range(len(df_load_p)) if df_sgen_p.iloc[t].sum() > SGEN_POWER_THRESHOLD]
    if not valid_ts:
        raise RuntimeError("❌ 没有找到满足 PV 总出力阈值的时刻 valid_ts")

    collected = 0
    chunk_idx = 0
    chunk_buffer = []

    # stats
    sum_x = torch.zeros(5, dtype=torch.float64)
    sq_sum_x = torch.zeros(5, dtype=torch.float64)
    total_pixels = 0

    bucket_counts = {"safe": 0, "mid": 0, "high": 0}

    pbar = tqdm(total=NUM_SAMPLES)

    while collected < NUM_SAMPLES:
        accepted = False
        for _try in range(MAX_TRIES_PER_SAMPLE):
            t = int(np.random.choice(valid_ts))

            # 采样缩放
            k_global = np.random.uniform(K_GLOBAL_MIN, K_GLOBAL_MAX)
            base_sgen_row = df_sgen_p.iloc[t]

            pav_dict = {}
            for gid, p_base in base_sgen_row.items():
                k_local = np.random.uniform(K_LOCAL_MIN, K_LOCAL_MAX)
                pav_dict[int(gid)] = float(p_base) * k_global * k_local

            # 场景网
            grid_scene = deepcopy(grid_template)
            apply_scene_PQ(grid_scene, df_load_p.iloc[t], df_load_q.iloc[t], pav_dict)
            set_bus_voltage_limits(grid_scene, vmin=0.95, vmax=1.05)
            tighten_thermal_limits(grid_scene, RATE_TIGHTEN_FACTOR)

            # PF
            grid_pf = deepcopy(grid_scene)
            pf_drv = run_pf(grid_pf)
            if not pf_drv.results.converged:
                continue
            V = np.abs(pf_drv.results.voltage)

            # 排除 slack 后的 Vmax（避免被 slack 固定电压影响）
            slack_idx = find_slack_bus_idx(grid_pf, bus_idx_map)
            V2 = V.copy()
            if slack_idx is not None and 0 <= slack_idx < len(V2):
                V2[slack_idx] = -1.0
            vmax_wo_slack = float(np.max(V2))

            bname = bucket_of_vmax(vmax_wo_slack)
            if not should_accept_bucket(bucket_counts, collected, bname):
                continue

            pf_v_abs = V.astype(np.float32)
            pf_v_angle = np.angle(pf_drv.results.voltage).astype(np.float32)

            # OPF teacher
            grid_opf = deepcopy(grid_scene)
            opf_drv = run_opf_teacher(grid_opf, thermal_limits=True)
            if not opf_drv.results.converged:
                continue

            opf_gen_p = get_opf_gen_p(opf_drv, grid_opf)

            sample = build_tensor_sample(grid_opf, pf_v_abs, pf_v_angle, opf_gen_p, bus_idx_map, pav_dict)
            if sample is None:
                continue

            # 至少要有一个可控PV节点
            if sample["mask"].sum().item() == 0:
                continue

            # accept
            bucket_counts[bname] += 1
            accepted = True

            # stats
            sum_x += sample["x"].double().sum(dim=1)
            sq_sum_x += (sample["x"].double() ** 2).sum(dim=1)
            total_pixels += sample["x"].shape[1]

            chunk_buffer.append(sample)
            collected += 1
            pbar.update(1)

            if len(chunk_buffer) >= CHUNK_SIZE:
                torch.save(chunk_buffer, os.path.join(SAVE_DIR, f"chunk_{chunk_idx:05d}.pt"))
                chunk_buffer = []
                chunk_idx += 1

            break

        if not accepted:
            # 太难凑满某个桶时，放宽策略：临时关闭分桶
            if USE_BUCKET_SAMPLING:
                print("⚠️ 某些桶过难凑样本，临时放宽分桶限制以避免卡死。")
                globals()["USE_BUCKET_SAMPLING"] = False

    if chunk_buffer:
        torch.save(chunk_buffer, os.path.join(SAVE_DIR, f"chunk_{chunk_idx:05d}.pt"))

    mean_x = (sum_x / total_pixels).float()
    std_x = torch.sqrt(sq_sum_x / total_pixels - mean_x.double() ** 2).float()
    torch.save({"x_mean": mean_x, "x_std": std_x}, os.path.join(SAVE_DIR, "stats.pt"))

    print("✅ 数据生成完毕！")
    print("Bucket counts:", bucket_counts)

if __name__ == "__main__":
    main()
