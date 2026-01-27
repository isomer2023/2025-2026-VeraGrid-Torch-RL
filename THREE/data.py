import os
import warnings
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
import simbench as sb
import scipy.sparse as sp
import networkx as nx
import pandapower.topology as top

warnings.filterwarnings("ignore")

try:
    import GC_PandaPowerImporter
    from VeraGridEngine import api as gce
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    raise

# ================= ⚙️ 配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 🌟 改名以区分旧数据
SAVE_DIR = os.path.join(CURRENT_DIR, "dataset_output_1mv_urban_dynamic_topo_FULL_STATE")
SB_CODE = "1-MV-urban--0-sw"

NUM_SAMPLES = 1000
CHUNK_SIZE = 100

SGEN_POWER_THRESHOLD = 1e-3
SGEN_NODE_THRESHOLD = 1e-6
RATE_TIGHTEN_FACTOR = 1.0

K_MEAN = 1.2        # 以前是 3.0
K_STD = 0.20        # 以前是 0.40 (降低波动，让数据更集中在合理范围)
K_MIN = 1         # 覆盖一点点欠发的情况
K_MAX = 1.3

MAX_TRIES_PER_SAMPLE = 200

def get_edge_data(grid_net, perm_idx):
    """
    Build directed edges in PERM space.
    edge_attr = [r_pu, x_pu]  (float32)
    edge_index: (2, E) long
    """
    buses = list(getattr(grid_net, "buses", []))
    N = len(buses)
    perm = np.array(perm_idx, dtype=np.int64)
    inv_perm = np.empty(N, dtype=np.int64)
    inv_perm[perm] = np.arange(N, dtype=np.int64)  # old bus idx -> perm position

    def _bus_key(b):
        return str(getattr(b, "idtag", getattr(b, "id", getattr(b, "name", ""))))

    key2idx = {}
    for i, b in enumerate(buses):
        k = _bus_key(b)
        if k: key2idx[k] = i
        if hasattr(b, "name") and b.name:
            key2idx[str(b.name)] = i

    def _get_bus_idx(bus_ref):
        if bus_ref is None: return None
        k = _bus_key(bus_ref)
        if k in key2idx: return key2idx[k]
        nm = str(getattr(bus_ref, "name", ""))
        return key2idx.get(nm, None)

    def _get_endpoints(br):
        b1 = getattr(br, "bus_from", getattr(br, "from_bus", getattr(br, "from_node", getattr(br, "from", None))))
        b2 = getattr(br, "bus_to", getattr(br, "to_bus", getattr(br, "to_node", getattr(br, "to", None))))
        i = _get_bus_idx(b1)
        j = _get_bus_idx(b2)
        return i, j

    def _get_rx_pu(br):
        # try direct pu
        r = _get_float(br, ["r_pu", "Rpu", "r", "R"], 0.0)
        x = _get_float(br, ["x_pu", "Xpu", "x", "X"], 0.0)

        # if transformer only provides vk/vkr in percent, derive r_pu/x_pu
        if (abs(r) < 1e-12 and abs(x) < 1e-12):
            vk = _get_float(br, ["vk_percent", "vk", "Vk_percent", "Vk"], 0.0)
            vkr = _get_float(br, ["vkr_percent", "vkr", "Vkr_percent", "Vkr"], 0.0)
            if vk > 0.0:
                z_pu = vk / 100.0
                r_pu = vkr / 100.0 if vkr > 0.0 else 0.0
                x_sq = max(z_pu * z_pu - r_pu * r_pu, 0.0)
                x_pu = float(x_sq ** 0.5)
                r, x = float(r_pu), float(x_pu)

        return float(r), float(x)

    src_list = []
    dst_list = []
    ea_list = []

    branches = list(getattr(grid_net, "lines", [])) + list(getattr(grid_net, "transformers", []))
    for br in branches:
        i, j = _get_endpoints(br)
        if i is None or j is None or i == j:
            continue

        # map to perm space
        pi = int(inv_perm[i])
        pj = int(inv_perm[j])

        r_pu, x_pu = _get_rx_pu(br)

        # add BOTH directions (directed edges)
        src_list += [pi, pj]
        dst_list += [pj, pi]
        ea_list += [[r_pu, x_pu], [r_pu, x_pu]]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(ea_list, dtype=torch.float32)
    return edge_index, edge_attr


def get_topology_data(grid_net, perm_idx, tau=2.0, use_x_only=True):
    g_nx = nx.Graph()
    perm = np.array(perm_idx, dtype=np.int64)

    buses = list(getattr(grid_net, "buses", []))
    N = len(buses)
    g_nx.add_nodes_from(range(N))

    # --- bus key -> index (VeraGrid bus order) ---
    def _bus_key(b):
        return str(getattr(b, "idtag", getattr(b, "id", getattr(b, "name", ""))))

    key2idx = {}
    for i, b in enumerate(buses):
        k = _bus_key(b)
        if k:
            key2idx[k] = i
        if hasattr(b, "name") and b.name:
            key2idx[str(b.name)] = i

    def _get_bus_idx(bus_ref):
        if bus_ref is None:
            return None
        k = _bus_key(bus_ref)
        if k in key2idx:
            return key2idx[k]
        nm = str(getattr(bus_ref, "name", ""))
        return key2idx.get(nm, None)

    # --- endpoints + impedance helpers ---
    def _get_endpoints(br):
        b1 = getattr(br, "bus_from", getattr(br, "from_bus", getattr(br, "from_node", getattr(br, "from", None))))
        b2 = getattr(br, "bus_to", getattr(br, "to_bus", getattr(br, "to_node", getattr(br, "to", None))))
        i = _get_bus_idx(b1)
        j = _get_bus_idx(b2)
        return i, j

    def _get_r_x(br):
        r = _get_float(br, ["r_pu", "Rpu", "r", "R", "r_ohm", "R_ohm"], 0.0)
        x = _get_float(br, ["x_pu", "Xpu", "x", "X", "x_ohm", "X_ohm"], 0.0)
        return r, x

    branches = list(getattr(grid_net, "lines", [])) + list(getattr(grid_net, "transformers", []))
    for br in branches:
        i, j = _get_endpoints(br)
        if i is None or j is None or i == j:
            continue

        r, x = _get_r_x(br)

        # electrical length: use |X| (better for Va) or |Z|
        if use_x_only:
            wlen = abs(x)
        else:
            wlen = float((r * r + x * x) ** 0.5)

        if not np.isfinite(wlen) or wlen <= 0.0:
            wlen = 1e-6

        # keep the smaller electrical length if parallel edges exist
        if g_nx.has_edge(i, j):
            prev = g_nx[i][j].get("weight", wlen)
            g_nx[i][j]["weight"] = min(prev, wlen)
        else:
            g_nx.add_edge(i, j, weight=wlen)

    # --- electrical shortest-path distance -> attn_bias ---
    dist = nx.floyd_warshall_numpy(g_nx, weight="weight")
    dist = np.asarray(dist, dtype=float)
    dist[~np.isfinite(dist)] = 100.0
    dist_perm = dist[np.ix_(perm, perm)]

    attn_bias = -dist_perm / float(tau)
    np.fill_diagonal(attn_bias, 0.0)

    # --- adjacency: keep TOPOLOGY ONLY (0/1) + self loop + symmetric norm ---
    adj = nx.adjacency_matrix(g_nx, nodelist=perm)  # unweighted structure
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

    adj_dense = torch.tensor(norm_adj.todense(), dtype=torch.float32)
    bias_dense = torch.tensor(attn_bias, dtype=torch.float32)
    return adj_dense, bias_dense



# ==========================================
ASSETS_PATH = os.path.join(SAVE_DIR, "static_assets.pt")


def _set_val(obj, attr_list, val):
    for a in attr_list:
        try:
            setattr(obj, a, val); return True
        except:
            pass
    return False


def _get_float(obj, attr_list, default=0.0):
    for a in attr_list:
        if hasattr(obj, a):
            try:
                return float(getattr(obj, a))
            except:
                pass
    return float(default)


def _is_slack(g): return "Ext_Grid" in str(getattr(g, "name", ""))


def _is_sgen(g): return "sgen" in str(getattr(g, "name", ""))


def _idx_from_name(name):
    try:
        return int(str(name).split("_")[1])
    except:
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


# ----------- OPF Driver -----------
def run_opf_teacher(grid_opf, thermal_limits=True):
    for g in getattr(grid_opf, "generators", []):
        _set_val(g, ["active", "in_service"], True)
        if _is_slack(g):
            _set_val(g, ["cost_a"], 1.0)
            _set_val(g, ["is_controlled"], True)
            _set_val(g, ["Pmax", "P_max"], 99999.0)
            _set_val(g, ["Pmin", "P_min"], -99999.0)
        elif _is_sgen(g):
            _set_val(g, ["cost_a"], 0.01)
            _set_val(g, ["is_controlled"], False)
            _set_val(g, ["enabled_dispatch"], True)
            lock_Q_as_PQ(g, 0.0)
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
    except:
        pass
    return drv


def get_opf_gen_p(opf_driver, grid):
    res = getattr(opf_driver, "results", None)
    if res is not None and hasattr(res, "generator_power"):
        try:
            return [float(v) for v in res.generator_power]
        except:
            pass
    return [_get_float(g, ["P", "p"], 0.0) for g in grid.generators]


def apply_scene_PQ(grid_scene, load_p_row, load_q_row, pav_dict):
    for l in getattr(grid_scene, "loads", []):
        lid = _idx_from_name(getattr(l, "name", "load_0"))
        if lid is not None:
            _set_val(l, ["P", "p"], float(load_p_row.get(lid, 0.0)))
            _set_val(l, ["Q", "q"], float(load_q_row.get(lid, 0.0)))
    for g in getattr(grid_scene, "generators", []):
        if _is_slack(g):
            _set_val(g, ["active", "in_service"], True)
            _set_val(g, ["is_controlled"], True)
            continue
        if _is_sgen(g):
            gid = _idx_from_name(getattr(g, "name", "sgen_0"))
            pav = float(pav_dict.get(gid, 0.0))
            _set_val(g, ["P", "p"], pav)
            _set_val(g, ["Pmax", "P_max"], pav)
            _set_val(g, ["Pmin", "P_min"], 0.0)
            _set_val(g, ["is_controlled"], False)
            _set_val(g, ["enabled_dispatch"], True)
            lock_Q_as_PQ(g, 0.0)


def set_bus_voltage_limits(grid, vmin=0.95, vmax=1.05):
    for b in getattr(grid, "buses", []):
        _set_val(b, ["Vmin"], vmin)
        _set_val(b, ["Vmax"], vmax)


# ================= 🌟 核心修改：构建样本 (接收 voltage 数组) =================
def build_tensor_sample(grid_opf, opf_gen_p, opf_results_voltage, bus_idx_map, pav_dict,
                        topo_adj, topo_bias, edge_index, edge_attr):
    num_nodes = len(grid_opf.buses)

    # 安全检查：如果电压数组长度和节点数不一致，直接跳过
    if len(opf_results_voltage) != num_nodes:
        return None

    # 1. 预计算电压幅值和相角
    # opf_results_voltage 是一个复数数组
    vm_all = np.abs(opf_results_voltage)  # 幅值 (p.u.)
    va_all = np.angle(opf_results_voltage, deg=True)  # 相角 (degrees)

    # Input: Load P, Load Q, PV Avail, vn_kv
    x = np.zeros((num_nodes, 4), dtype=np.float32)
    # Target: P (ch0), Vm (ch1), sin(Va) (ch2), cos(Va) (ch3)
    y_target = np.zeros((num_nodes, 4), dtype=np.float32)

    # --- 填充 Input Load ---
    for l in grid_opf.loads:
        bus_ref = getattr(l, "bus", getattr(l, "node", None))
        if bus_ref is None: continue
        bid = str(getattr(bus_ref, "idtag", getattr(bus_ref, "id", getattr(bus_ref, "name", None))))
        idx = bus_idx_map.get(bid, bus_idx_map.get(str(getattr(bus_ref, "name", "")), None))
        if idx is not None:
            x[idx, 0] += _get_float(l, ["P", "p"], 0.0)
            x[idx, 1] += _get_float(l, ["Q", "q"], 0.0)

    # --- 填充 Input Sgen & Target P ---
    sgen_mask = np.zeros(num_nodes, dtype=bool)
    for i, g in enumerate(grid_opf.generators):
        bus_ref = getattr(g, "bus", getattr(g, "node", None))
        if bus_ref is None: continue
        bid = str(getattr(bus_ref, "idtag", getattr(bus_ref, "id", getattr(bus_ref, "name", None))))
        idx = bus_idx_map.get(bid, bus_idx_map.get(str(getattr(bus_ref, "name", "")), None))
        if idx is None: continue

        if _is_sgen(g):
            gid = _idx_from_name(getattr(g, "name", "sgen_0"))
            pav = float(pav_dict.get(gid, 0.0))
            x[idx, 2] += pav
            if pav > SGEN_NODE_THRESHOLD:
                sgen_mask[idx] = True
                popt = float(opf_gen_p[i])
                y_target[idx, 0] = np.clip(popt / max(pav, 1e-12), 0.0, 1.0)
        elif _is_slack(g):
            x[idx, 2] += 0.0

    va_rad_all = np.angle(opf_results_voltage, deg=False)  # radians

    for i, b in enumerate(grid_opf.buses):
        bid = str(getattr(b, "idtag", getattr(b, "id", getattr(b, "name", None))))
        idx = bus_idx_map.get(bid, bus_idx_map.get(str(getattr(b, "name", "")), None))

        if idx is not None:
            # ---- add vn_kv to x (only this extra feature) ----
            x[idx, 3] = _get_float(b, ["vn_kv", "Vn", "base_kv", "kv"], 0.0)

            # ---- targets ----
            y_target[idx, 1] = float(vm_all[i])
            y_target[idx, 2] = float(np.sin(va_rad_all[i]))
            y_target[idx, 3] = float(np.cos(va_rad_all[i]))

    # 重排
    x_re = x[PERM_IDX]
    y_re = y_target[PERM_IDX]
    m_re = sgen_mask[PERM_IDX]

    return {
        "x": torch.tensor(x_re.T, dtype=torch.float32),
        "y": torch.tensor(y_re.T, dtype=torch.float32),  # (4, N) 你已改成 sin/cos 后是4
        "mask": torch.tensor(m_re, dtype=torch.bool),
        "adj": topo_adj.clone(),
        "attn_bias": topo_bias.clone(),
        "edge_index": edge_index.clone(),
        "edge_attr": edge_attr.clone(),
    }


# ================= Main =================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"🚀 生成全状态数据 (P, V, Angle)... 目标: {NUM_SAMPLES}")
    print(f"📂 输出目录: {SAVE_DIR}")

    net_pp = sb.get_simbench_net(SB_CODE)
    grid_template = GC_PandaPowerImporter.PP2GC(net_pp)

    # Bus Map
    bus_idx_map = {}
    for i, b in enumerate(grid_template.buses):
        key = str(getattr(b, "idtag", getattr(b, "id", getattr(b, "name", None))))
        if key: bus_idx_map[key] = i
        if hasattr(b, "name") and b.name: bus_idx_map[str(b.name)] = i

    global PERM_IDX, NUM_NODES
    if not os.path.exists(ASSETS_PATH):
        NUM_NODES = len(grid_template.buses)
        PERM_IDX = np.arange(NUM_NODES)
        torch.save({"perm": PERM_IDX, "num_nodes": NUM_NODES}, ASSETS_PATH)
    else:
        ASSETS = torch.load(ASSETS_PATH, weights_only=False)
        PERM_IDX = ASSETS["perm"]
        NUM_NODES = ASSETS["num_nodes"]

    profiles = sb.get_absolute_values(net_pp, profiles_instead_of_study_cases=True)
    df_load_p = profiles[("load", "p_mw")]
    df_load_q = profiles[("load", "q_mvar")]
    df_sgen_p = profiles[("sgen", "p_mw")]
    valid_ts = [t for t in range(len(df_load_p)) if df_sgen_p.iloc[t].sum() > SGEN_POWER_THRESHOLD]

    current_adj, current_bias = get_topology_data(grid_template, PERM_IDX)
    current_edge_index, current_edge_attr = get_edge_data(grid_template, PERM_IDX)
    collected = 0
    chunk_idx = 0
    chunk_buffer = []

    sum_x = torch.zeros(4, dtype=torch.float64)
    sq_sum_x = torch.zeros(4, dtype=torch.float64)
    total_pixels = 0

    pbar = tqdm(total=NUM_SAMPLES)

    while collected < NUM_SAMPLES:
        for _try in range(MAX_TRIES_PER_SAMPLE):
            t = int(np.random.choice(valid_ts))
            k_factor = float(np.clip(np.random.normal(K_MEAN, K_STD), K_MIN, K_MAX))

            base_sgen_row = df_sgen_p.iloc[t]
            pav_dict = {int(gid): float(p) * k_factor for gid, p in base_sgen_row.items()}

            grid_scene = deepcopy(grid_template)
            apply_scene_PQ(grid_scene, df_load_p.iloc[t], df_load_q.iloc[t], pav_dict)
            set_bus_voltage_limits(grid_scene)
            tighten_thermal_limits(grid_scene, RATE_TIGHTEN_FACTOR)

            grid_opf = deepcopy(grid_scene)
            opf_drv = run_opf_teacher(grid_opf)

            if not opf_drv.results.converged: continue

            # 🌟 核心：提取 Complex Voltage 并传入
            # 如果没找到 voltage 属性，这里会报错，但你刚才的测试证明它一定在
            voltage_complex = opf_drv.results.voltage

            sample = build_tensor_sample(
                grid_opf,
                get_opf_gen_p(opf_drv, grid_opf),
                voltage_complex,
                bus_idx_map,
                pav_dict,
                current_adj,
                current_bias,
                current_edge_index,
                current_edge_attr
            )

            if sample is None or sample["mask"].sum() == 0: continue

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

    if chunk_buffer:
        torch.save(chunk_buffer, os.path.join(SAVE_DIR, f"chunk_{chunk_idx:05d}.pt"))

    mean_x = (sum_x / total_pixels).float()
    std_x = torch.sqrt(sq_sum_x / total_pixels - mean_x.double() ** 2).float()
    torch.save({"x_mean": mean_x, "x_std": std_x}, os.path.join(SAVE_DIR, "stats.pt"))
    print("✅ 全状态数据生成完成！请进行下一步训练。")


if __name__ == "__main__":
    main()