import os
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import simbench as sb

warnings.filterwarnings('ignore')

# 导入环境
try:
    import GC_PandaPowerImporter
    from VeraGridEngine import api as gce
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    exit()

# ================= ⚙️ 配置参数 =================
SB_CODE = "1-MV-urban--0-sw"
NUM_SAMPLES = 4000
CHUNK_SIZE = 100
STRESS_MIN = 1.0
STRESS_MAX = 10.0
LOAD_SCALE = 1.0
SGEN_POWER_THRESHOLD = 1e-3
RATE_TIGHTEN_FACTOR = 1.0
SAVE_DIR = "./dataset_output_1mv_urban_4000"
# =================================================

os.makedirs(SAVE_DIR, exist_ok=True)


# --- 辅助函数 (保持不变) ---
def get_gc_id(obj):
    if hasattr(obj, 'id') and obj.id is not None: return obj.id
    if hasattr(obj, 'idtag') and getattr(obj, 'idtag', None) is not None: return obj.idtag
    if hasattr(obj, 'uuid') and getattr(obj, 'uuid', None) is not None: return obj.uuid
    if hasattr(obj, 'name') and obj.name is not None: return obj.name
    return str(obj)


def _set_val(obj, attr_list, val):
    for attr in attr_list:
        try:
            setattr(obj, attr, val)
            return
        except:
            continue

def _get_val(obj, attr_list, default=0.0):
    for attr in attr_list:
        if hasattr(obj, attr):
            try:
                return float(getattr(obj, attr))
            except:
                continue
    return default

def _is_slack_safe(g):
    val = getattr(g, 'is_slack', getattr(g, 'slack', None))
    if val is not None: return bool(val)
    if "Ext_Grid" in str(getattr(g, 'name', '')): return True
    return False

def get_robust_opf_results(driver, grid):
    # (保持原样，省略以节省空间)
    if driver is None or not hasattr(driver, 'results'): return [0.0] * len(grid.generators)
    res = driver.results
    if hasattr(res, 'generator_power') and hasattr(res, 'generator_names'):
        r_names = res.generator_names
        r_vals = res.generator_power
        res_map = {str(n): float(v) for n, v in zip(r_names, r_vals)}
        final_p = []
        for i, g in enumerate(grid.generators):
            val = None
            if hasattr(g, 'name'): val = res_map.get(str(g.name))
            if val is None: val = res_map.get(str(get_gc_id(g)))
            if val is None and i < len(r_vals): val = float(r_vals[i])
            final_p.append(val if val is not None else 0.0)
        return final_p
    return [_get_val(g, ['P', 'p']) for g in grid.generators]

def tighten_thermal_limits(grid, factor=0.3):
    if hasattr(grid, 'branches'):
        branches = list(grid.branches)
    else:
        branches = list(getattr(grid, 'lines', [])) + list(getattr(grid, 'transformers', []))
    for br in branches:
        old_rate = _get_val(br, ['rate', 'Rate'], 100.0)
        _set_val(br, ['rate', 'Rate'], old_rate * factor)


def setup_and_run_opf_teacher(grid, current_profile_sgen):
    # (保持原样，OPF 配置逻辑)
    for g in grid.generators:
        g_name = str(getattr(g, 'name', ''))
        _set_val(g, ['active', 'in_service', 'status'], True)
        if _is_slack_safe(g) or "Ext_Grid" in g_name:
            _set_val(g, ['Pmax', 'P_max'], 99999.0)
            _set_val(g, ['Pmin', 'P_min'], -99999.0)
            _set_val(g, ['cost_a', 'Cost1'], 1.0)
            _set_val(g, ['cost_b', 'Cost2'], 100.0)
            _set_val(g, ['is_controlled', 'controlled'], True)
        elif "sgen" in g_name:
            # 注意：这里我们信任传入 grid 对象里的 Pmax 已经被修改过了
            p_avail = _get_val(g, ['Pmax', 'P_max'], 0.0)
            _set_val(g, ['Pmin', 'P_min'], 0.0)
            _set_val(g, ['cost_a', 'Cost1'], 0.01)
            _set_val(g, ['cost_b', 'Cost2'], 0.1)
            _set_val(g, ['is_controlled', 'controlled'], True)
            _set_val(g, ['Qmax', 'Q_max'], 0.0)
            _set_val(g, ['Qmin', 'Q_min'], 0.0)

    opf_opts = gce.OptimalPowerFlowOptions()
    if hasattr(gce, 'SolverType'): _set_val(opf_opts, ['solver', 'solver_type'], gce.SolverType.NONLINEAR_OPF)
    opf_opts.objective = 0
    _set_val(opf_opts, ['activate_voltage_limits', 'voltage_limits'], True)
    _set_val(opf_opts, ['vmin', 'Vmin'], 0.98)
    _set_val(opf_opts, ['vmax', 'Vmax'], 1.02)
    _set_val(opf_opts, ['activate_thermal_limits', 'thermal_limits'], True)
    _set_val(opf_opts, ['dispatch_P', 'control_active_power'], True)
    _set_val(opf_opts, ['allow_soft_limits', 'soft_limits'], True)

    opf_driver = gce.OptimalPowerFlowDriver(grid, opf_opts)
    try:
        opf_driver.run()
    except:
        pass
    return opf_driver


# --- 构造图样本 (核心修改：Bus 86 彻底变平民) ---
def build_graph_sample(grid, pf_v_abs, pf_v_angle, pf_loading, opf_gen_p, bus_idx_map):
    num_nodes = len(grid.buses)
    x = np.zeros((num_nodes, 6), dtype=np.float32)

    # 1. 负荷
    for l in grid.loads:
        bus_ref = getattr(l, 'bus', getattr(l, 'node', None))
        if bus_ref is None: continue
        idx = bus_idx_map.get(get_gc_id(bus_ref))
        if idx is None: continue
        x[idx, 0] += _get_val(l, ['P', 'p'])
        x[idx, 1] += _get_val(l, ['Q', 'q'])

    # 2. 电压 (PF 结果)
    x[:, 3] = pf_v_abs
    x[:, 5] = pf_v_angle

    # 3. 发电机 (处理 Bus 86)
    sgen_mask_np = np.zeros(num_nodes, dtype=bool)
    y_target = np.zeros((num_nodes, 1), dtype=np.float32)

    for i, g in enumerate(grid.generators):
        bus_ref = getattr(g, 'bus', getattr(g, 'node', None))
        if bus_ref is None: continue
        idx = bus_idx_map.get(get_gc_id(bus_ref))
        if idx is None: continue

        # 🛑 检查是否是被禁用的 generator (Bus 86)
        # 如果 Pmax 已经被设为 0 (我们在 main 里做了)，那么它就不算 generator
        p_max = _get_val(g, ['Pmax', 'P_max'], 0.0)

        # 只有当 Pmax > 0 时，才认为这是个有效的 sgen 节点
        # 否则，它就保留为普通的 Load 节点 (x[idx, 4] = 0)
        if p_max > 1e-4 and "sgen" in str(getattr(g, 'name', '')):
            x[idx, 4] = 1.0  # 标记：我是发电机
            x[idx, 2] += p_max
            sgen_mask_np[idx] = True

            try:
                p_opt = float(opf_gen_p[i])
            except:
                p_opt = 0.0
            alpha = np.clip(p_opt / p_max, 0.0, 1.0)
            y_target[idx] = alpha

        # Ext_Grid 等其他发电机 (x[4]=1, 但 mask=False)
        elif _is_slack_safe(g):
            x[idx, 4] = 1.0
            # Slack 也不参与 Alpha 预测

    # 4. 边特征
    src, dst, attr = [], [], []
    if hasattr(grid, 'branches'):
        branches = list(grid.branches)
    else:
        branches = list(getattr(grid, 'lines', [])) + list(getattr(grid, 'transformers', []))

    for i, br in enumerate(branches):
        if _get_val(br, ['active', 'status'], 1.0) < 0.5: continue
        f_ref = getattr(br, 'bus_from', getattr(br, 'from_node', getattr(br, 'busFrom', None)))
        t_ref = getattr(br, 'bus_to', getattr(br, 'to_node', getattr(br, 'busTo', None)))
        if f_ref is None or t_ref is None: continue
        u = bus_idx_map.get(get_gc_id(f_ref))
        v = bus_idx_map.get(get_gc_id(t_ref))
        if u is None or v is None: continue

        r = float(_get_val(br, ['R', 'r']))
        x_val = float(_get_val(br, ['X', 'x']))
        rate = float(_get_val(br, ['rate', 'Rate'], 100.0))
        load_val = 0.0
        if pf_loading is not None and i < len(pf_loading):
            load_val = abs(float(pf_loading[i]))

        edge_feat = [r, x_val, rate, load_val]
        src.extend([u, v])
        dst.extend([v, u])
        attr.extend([edge_feat, edge_feat])

    if len(src) == 0: return None

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(attr, dtype=torch.float32),
        y=torch.tensor(y_target, dtype=torch.float32),
        mask=torch.tensor(sgen_mask_np, dtype=torch.bool)
    )
    return data


# ================= 主程序 =================
def main():
    print(f"🚀 启动数据生成 (目标 {NUM_SAMPLES} 个样本)")

    # 1. 准备网络
    net_pp = sb.get_simbench_net(SB_CODE)
    grid_template = GC_PandaPowerImporter.PP2GC(net_pp)
    bus_idx_map = {get_gc_id(b): i for i, b in enumerate(grid_template.buses)}

    profiles = sb.get_absolute_values(net_pp, profiles_instead_of_study_cases=True)
    df_load_p = profiles[('load', 'p_mw')]
    df_load_q = profiles[('load', 'q_mvar')]
    df_sgen_p = profiles[('sgen', 'p_mw')]
    n_timesteps = len(df_load_p)
    valid_ts = [t for t in range(n_timesteps) if df_sgen_p.iloc[t].sum() > SGEN_POWER_THRESHOLD]

    collected = 0
    chunk_idx = 0
    chunk_buffer = []

    # --- 🔥 在线统计量初始化 ---
    # Running mean/std calculation variables
    # X shape: [N, 6], Edge shape: [E, 4]
    sum_x = torch.zeros(6, dtype=torch.float64)
    sq_sum_x = torch.zeros(6, dtype=torch.float64)
    count_x = 0

    sum_e = torch.zeros(4, dtype=torch.float64)
    sq_sum_e = torch.zeros(4, dtype=torch.float64)
    count_e = 0

    # 🔥 新增 Min-Max 统计
    min_x = torch.full((6,), float('inf'), dtype=torch.float64)
    max_x = torch.full((6,), float('-inf'), dtype=torch.float64)

    min_e = torch.full((4,), float('inf'), dtype=torch.float64)
    max_e = torch.full((4,), float('-inf'), dtype=torch.float64)
    # -------------------------

    pbar = tqdm(total=NUM_SAMPLES)

    while collected < NUM_SAMPLES:
        t = int(np.random.choice(valid_ts))

        # 1. Scale
        prob = np.random.rand()
        if prob < 0.25:
            sgen_scale = np.random.uniform(0.7, 0.85)
        elif prob < 0.75:
            sgen_scale = np.random.uniform(0.85, 1.15)
        else:
            sgen_scale = np.random.uniform(1.15, 1.3)

        sgen_vals = (df_sgen_p.iloc[t] * sgen_scale).to_dict()
        grid_scene = deepcopy(grid_template)

        # 🛑 【在此处把 Bus 86 彻底抹除】
        for g in grid_scene.generators:
            bus_ref = getattr(g, 'bus', getattr(g, 'node', None))
            bus_name = str(getattr(bus_ref, 'name', ''))
            gen_name = str(getattr(g, 'name', ''))

            # 只要涉及 Bus 86，直接关停
            if "Bus 86" in bus_name or "Bus 86" in gen_name:
                _set_val(g, ['active', 'in_service', 'status'], False)
                _set_val(g, ['Pmax', 'P_max'], 0.0)
                _set_val(g, ['P', 'p'], 0.0)

        # 2. 设置负荷
        l_p_now = df_load_p.iloc[t]
        l_q_now = df_load_q.iloc[t]
        for l in grid_scene.loads:
            try:
                lid = int(l.name.split('_')[1])
                _set_val(l, ['P', 'p'], LOAD_SCALE * l_p_now.get(lid, 0.0))
                _set_val(l, ['Q', 'q'], LOAD_SCALE * l_q_now.get(lid, 0.0))
            except:
                continue

        # 3. 设置 Sgen
        for g in grid_scene.generators:
            if _get_val(g, ['active', 'in_service'], True) is False: continue  # 跳过已关停的
            if "sgen" in str(getattr(g, 'name', '')):
                try:
                    gid = int(g.name.split('_')[1])
                    pav = sgen_vals.get(gid, 0.0)
                    _set_val(g, ['Pmax', 'P_max'], pav)
                    _set_val(g, ['P', 'p'], pav)
                except:
                    continue

        # 4. 过滤简单样本
        active_cnt = 0
        for g in grid_scene.generators:
            if _get_val(g, ['active', 'in_service'], True):
                if _get_val(g, ['Pmax', 'P_max'], 0.0) > 1e-4: active_cnt += 1
        if active_cnt <= 1: continue

        tighten_thermal_limits(grid_scene, RATE_TIGHTEN_FACTOR)

        # 5. Run PF
        try:
            grid_pf = deepcopy(grid_scene)
            pf_opts = gce.PowerFlowOptions(gce.SolverType.NR, verbose=False)
            pf_driver = gce.PowerFlowDriver(grid_pf, pf_opts)
            pf_driver.run()
            if not bool(pf_driver.results.converged): continue

            res_v = pf_driver.results.voltage
            pf_v_abs = np.abs(np.array(res_v, dtype=np.complex128))
            pf_v_angle = np.angle(np.array(res_v, dtype=np.complex128))
            pf_loading = np.array(pf_driver.results.loading, dtype=np.float32)
        except:
            continue

        # 6. Run OPF
        grid_opf = deepcopy(grid_scene)
        opf_driver = setup_and_run_opf_teacher(grid_opf, sgen_vals)

        is_conv = False
        if hasattr(opf_driver, 'results'):
            rc = opf_driver.results.converged
            if isinstance(rc, (bool, np.bool_)):
                is_conv = bool(rc)
            elif hasattr(rc, '__len__'):
                is_conv = bool(rc[0])
        if not is_conv: continue

        opf_gen_p = get_robust_opf_results(opf_driver, grid_opf)

        # 7. Build Graph
        data = build_graph_sample(grid_opf, pf_v_abs, pf_v_angle, pf_loading, opf_gen_p, bus_idx_map)
        if data is None: continue

        # --- 🔥 更新统计量 ---
        # 累加 x
        sum_x += data.x.double().sum(dim=0)
        sq_sum_x += (data.x.double() ** 2).sum(dim=0)
        count_x += data.x.shape[0]

        # 累加 edge
        sum_e += data.edge_attr.double().sum(dim=0)
        sq_sum_e += (data.edge_attr.double() ** 2).sum(dim=0)
        count_e += data.edge_attr.shape[0]

        # 🔥 新增 Min-Max 更新
        min_x = torch.min(min_x, data.x.double().min(dim=0).values)
        max_x = torch.max(max_x, data.x.double().max(dim=0).values)

        min_e = torch.min(min_e, data.edge_attr.double().min(dim=0).values)
        max_e = torch.max(max_e, data.edge_attr.double().max(dim=0).values)
        # -------------------

        chunk_buffer.append(data)
        collected += 1
        pbar.update(1)

        if len(chunk_buffer) >= CHUNK_SIZE:
            torch.save(chunk_buffer, os.path.join(SAVE_DIR, f"chunk_{chunk_idx:05d}.pt"))
            chunk_buffer = []
            chunk_idx += 1

    if len(chunk_buffer) > 0:
        torch.save(chunk_buffer, os.path.join(SAVE_DIR, f"chunk_{chunk_idx:05d}.pt"))

    pbar.close()

    # --- 🔥 计算并保存统计量 ---
    print("🧮 正在计算并保存全局统计量...")
    mean_x = (sum_x / count_x).float()
    std_x = torch.sqrt(sq_sum_x / count_x - mean_x.double() ** 2).float()

    mean_e = (sum_e / count_e).float()
    std_e = torch.sqrt(sq_sum_e / count_e - mean_e.double() ** 2).float()

    # 🔥 新增 Min-Max 统计量
    max_x = max_x.float()
    min_e = min_e.float()
    max_e = max_e.float()

    min_x = min_x.float()
    stats = {
        'x_mean': mean_x, 'x_std': std_x,
        'e_mean': mean_e, 'e_std': std_e,
        'x_min': min_x, 'x_max': max_x,  # 新增
        'e_min': min_e, 'e_max': max_e  # 新增
    }
    torch.save(stats, os.path.join(SAVE_DIR, "stats.pt"))

    print(f"✅ 生成完成！统计量已保存至 {os.path.join(SAVE_DIR, 'stats.pt')}")
    print(f"📊 Node Mean: {mean_x.numpy()}")
    print(f"📊 Node Std : {std_x.numpy()}")
    print(f"📊 Edge Mean: {mean_e.numpy()}")
    print(f"📊 Edge Std : {std_e.numpy()}")
    print(f"{min_x.numpy(), max_x.numpy(), min_e.numpy(), max_e.numpy()}")


if __name__ == "__main__":
    main()