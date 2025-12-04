import os
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import simbench as sb

# 忽略警告
warnings.filterwarnings('ignore')

# 导入环境
try:
    import GC_PandaPowerImporter
    from VeraGridEngine import api as gce
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    exit()

# ================= ⚙️ 配置参数 =================
SB_CODE = "1-MV-urban--0-sw"   # 可以之后换别的 SimBench 网

NUM_SAMPLES = 20000            # ★ 目标样本数（你可以根据需要调大）
CHUNK_SIZE = 1000               # 每 CHUNK_SIZE 个样本存一个文件

STRESS_MIN = 1.0              # 压力系数下界
STRESS_MAX = 10.0             # 压力系数上界
LOAD_SCALE = 1.0               # 负荷缩放系数（1.0 表示原始负荷）

SGEN_POWER_THRESHOLD = 1e-3    # 过滤掉总 sgen 功率太小的时间步
RATE_TIGHTEN_FACTOR = 1.0      # 热限收紧系数 (1.0 表示不收紧)

SAVE_DIR = "./dataset_output_1mv_urban"  # 数据集输出目录
# =================================================

os.makedirs(SAVE_DIR, exist_ok=True)

# --- 辅助函数 ---

def get_gc_id(obj):
    """尽可能稳定地获取对象的唯一标识"""
    if hasattr(obj, 'id') and obj.id is not None:
        return obj.id
    if hasattr(obj, 'idtag') and getattr(obj, 'idtag', None) is not None:
        return obj.idtag
    if hasattr(obj, 'uuid') and getattr(obj, 'uuid', None) is not None:
        return obj.uuid
    if hasattr(obj, 'name') and obj.name is not None:
        return obj.name
    return str(obj)


def _set_val(obj, attr_list, val):
    """尝试给对象设置多个可能名称的属性，成功一个就返回"""
    for attr in attr_list:
        try:
            setattr(obj, attr, val)
            return
        except Exception:
            continue


def _get_val(obj, attr_list, default=0.0):
    """尝试从对象读取多个可能名称的属性，成功就返回 float"""
    for attr in attr_list:
        if hasattr(obj, attr):
            try:
                return float(getattr(obj, attr))
            except Exception:
                continue
    return default


def _is_slack_safe(g):
    """判断一个发电机是不是平衡节点/外部电网"""
    val = getattr(g, 'is_slack', None)
    if val is not None:
        return bool(val)
    val = getattr(g, 'slack', None)
    if val is not None:
        return bool(val)
    name = str(getattr(g, 'name', ''))
    if "Ext_Grid" in name:
        return True
    return False


def get_robust_opf_results(driver, grid):
    """
    终极版结果读取：
    使用 results.generator_names / generator_power 按名字匹配，
    匹配失败再按索引兜底。
    """
    if driver is None or not hasattr(driver, 'results'):
        print("⚠️ OPF Driver 无结果")
        return [0.0] * len(grid.generators)

    res = driver.results

    if hasattr(res, 'generator_power') and hasattr(res, 'generator_names'):
        r_names = res.generator_names
        r_vals = res.generator_power

        # 建立映射字典 {name: val}
        res_map = {str(n): float(v) for n, v in zip(r_names, r_vals)}

        final_p = []

        for i, g in enumerate(grid.generators):
            val = None

            # A: 用 g.name 查
            if hasattr(g, 'name'):
                val = res_map.get(str(g.name))

            # B: 用 get_gc_id 查
            if val is None:
                gid = get_gc_id(g)
                val = res_map.get(str(gid))

            # C: 按索引兜底
            if val is None and i < len(r_vals):
                val = float(r_vals[i])

            final_p.append(val if val is not None else 0.0)

        return final_p

    # 兜底：如果连 generator_power 都没有，回退到读输入值
    print("⚠️ 警告: 结果中没有 generator_power，回退到读取输入值")
    return [_get_val(g, ['P', 'p']) for g in grid.generators]


def tighten_thermal_limits(grid, factor=0.3):
    """
    将所有线路/变压器的热限 Rate 按给定系数缩小，用于制造“紧张场景”。
    """
    if hasattr(grid, 'branches'):
        branches = list(grid.branches)
    elif hasattr(grid, 'get_branches'):
        branches = list(grid.get_branches())
    else:
        branches = list(getattr(grid, 'lines', [])) + list(getattr(grid, 'transformers', []))

    for br in branches:
        old_rate = _get_val(br, ['rate', 'Rate'], 100.0)
        new_rate = old_rate * factor
        _set_val(br, ['rate', 'Rate'], new_rate)


# --- OPF 老师模块 ---

def setup_and_run_opf_teacher(grid, current_profile_sgen):
    """
    在给定场景 (负荷 + sgen 可用功率) 下配置并运行 OPF。
    - 激活电压限值和热限
    - 控制有功出力 (dispatch_P)
    - 设置成本系数，实现“运行成本最小化”
    """
    for g in grid.generators:
        g_name = str(getattr(g, 'name', ''))
        _set_val(g, ['active', 'in_service', 'status'], True)

        # 平衡机组 / 外部电网：高成本但可正负出力
        if _is_slack_safe(g) or "Ext_Grid" in g_name:
            _set_val(g, ['Pmax', 'P_max'], 99999.0)
            _set_val(g, ['Pmin', 'P_min'], -99999.0)
            _set_val(g, ['cost_a', 'Cost1'], 1.0)      # 昂贵电源
            _set_val(g, ['cost_b', 'Cost2'], 100.0)
            _set_val(g, ['is_controlled', 'controlled'], True)

        # sgen：有限制的廉价电源
        elif "sgen" in g_name:
            try:
                sgen_idx = int(g_name.split('_')[1])
                p_avail = current_profile_sgen.get(sgen_idx, 0.0)
            except Exception:
                p_avail = 0.0

            _set_val(g, ['Pmax', 'P_max'], p_avail)
            _set_val(g, ['Pmin', 'P_min'], 0.0)
            _set_val(g, ['cost_a', 'Cost1'], 0.01)     # 便宜电源
            _set_val(g, ['cost_b', 'Cost2'], 0.1)
            _set_val(g, ['is_controlled', 'controlled'], True)
            _set_val(g, ['Qmax', 'Q_max'], 0.0)
            _set_val(g, ['Qmin', 'Q_min'], 0.0)

    opf_opts = gce.OptimalPowerFlowOptions()
    if hasattr(gce, 'SolverType'):
        _set_val(opf_opts, ['solver', 'solver_type'], gce.SolverType.NONLINEAR_OPF)

    # 成本最小化
    opf_opts.objective = 0

    # 电压约束
    _set_val(opf_opts, ['activate_voltage_limits', 'voltage_limits'], True)
    _set_val(opf_opts, ['vmin', 'Vmin'], 0.98)
    _set_val(opf_opts, ['vmax', 'Vmax'], 1.02)

    # 热限约束
    _set_val(opf_opts, ['activate_thermal_limits', 'thermal_limits'], True)

    # 控制有功出力
    _set_val(opf_opts, ['dispatch_P', 'control_active_power'], True)

    # 允许软约束
    _set_val(opf_opts, ['allow_soft_limits', 'soft_limits'], True)

    opf_driver = gce.OptimalPowerFlowDriver(grid, opf_opts)
    try:
        opf_driver.run()
    except Exception:
        pass

    return opf_driver


# --- 构造图样本 ---

def build_graph_sample(grid, pf_v_abs, pf_v_angle, pf_loading, opf_gen_p, bus_idx_map):
    """
    使用给定的 Grid + PF 结果 + OPF 结果构造一个 PyG Data：
    - x: [N, 6]
    - edge_index: [2, E]
    - edge_attr: [E, 4]
    - y: [N, 1]  (alpha)
    - mask: [N]  (sgen 位置)
    """
    num_nodes = len(grid.buses)
    x = np.zeros((num_nodes, 6), dtype=np.float32)

    # 0/1 列：负荷 P/Q (放大)
    for l in grid.loads:
        bus_ref = getattr(l, 'bus', getattr(l, 'node', None))
        if bus_ref is None:
            continue
        idx = bus_idx_map.get(get_gc_id(bus_ref))
        if idx is None:
            continue
        p_val = _get_val(l, ['P', 'p'])
        q_val = _get_val(l, ['Q', 'q'])
        x[idx, 0] += p_val * 3.0
        x[idx, 1] += q_val * 3.0

    # PF 电压特征检查
    if pf_v_abs is None or len(pf_v_abs) != num_nodes:
        return None
    if np.isnan(pf_v_abs).any() or np.isinf(pf_v_abs).any():
        return None
    if pf_v_abs.min() < 0.1 or pf_v_abs.max() > 2.0:
        return None
    if pf_v_angle is None or len(pf_v_angle) != num_nodes:
        return None

    # 3,5 列：电压幅值偏移 & 相角
    x[:, 3] = (pf_v_abs - 1.0) * 10.0
    x[:, 5] = pf_v_angle

    # Label 相关：sgen mask + y
    if len(opf_gen_p) != len(grid.generators):
        return None

    sgen_mask_np = np.zeros(num_nodes, dtype=bool)
    y_target = np.zeros((num_nodes, 1), dtype=np.float32)

    for i, g in enumerate(grid.generators):
        bus_ref = getattr(g, 'bus', getattr(g, 'node', None))
        if bus_ref is None:
            continue
        idx = bus_idx_map.get(get_gc_id(bus_ref))
        if idx is None:
            continue

        # 第 4 列：该节点存在 generator
        x[idx, 4] = 1.0

        if "sgen" in str(getattr(g, 'name', '')):
            p_max = _get_val(g, ['Pmax', 'P_max'], 0.0)
            # 第 2 列：可用功率 Pmax / 10
            x[idx, 2] += p_max / 10.0
            sgen_mask_np[idx] = True

            if p_max > 1e-4:
                try:
                    p_opt = float(opf_gen_p[i])
                except Exception:
                    p_opt = 0.0
                alpha = np.clip(p_opt / p_max, 0.0, 1.0)
                y_target[idx] = alpha

    # 边特征
    src = []
    dst = []
    attr = []

    if hasattr(grid, 'branches'):
        branches = list(grid.branches)
    elif hasattr(grid, 'get_branches'):
        branches = list(grid.get_branches())
    else:
        branches = list(getattr(grid, 'lines', [])) + list(getattr(grid, 'transformers', []))

    for i, br in enumerate(branches):
        if _get_val(br, ['active', 'status'], 1.0) < 0.5:
            continue

        f_ref = getattr(br, 'bus_from', getattr(br, 'from_node', getattr(br, 'busFrom', None)))
        t_ref = getattr(br, 'bus_to', getattr(br, 'to_node', getattr(br, 'busTo', None)))

        if f_ref is None or t_ref is None:
            continue

        u = bus_idx_map.get(get_gc_id(f_ref))
        v = bus_idx_map.get(get_gc_id(t_ref))
        if u is None or v is None:
            continue

        r = float(_get_val(br, ['R', 'r'])) * 100.0
        x_val = float(_get_val(br, ['X', 'x'])) * 100.0
        rate = float(_get_val(br, ['rate', 'Rate'], 100.0)) / 1000.0

        load_val = 0.0
        if pf_loading is not None and i < len(pf_loading):
            try:
                load_val = abs(float(pf_loading[i]))
            except Exception:
                load_val = 0.0

        edge_feat = [r, x_val, rate, load_val]
        src.extend([u, v])
        dst.extend([v, u])
        attr.extend([edge_feat, edge_feat])

    if len(src) == 0:
        return None

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(attr, dtype=torch.float32),
    )
    data.y = torch.tensor(y_target, dtype=torch.float32)       # [N, 1]
    data.mask = torch.tensor(sgen_mask_np, dtype=torch.bool)   # [N]

    # 如果你以后想跟在线训练脚本对齐，也可以顺带加一个别名：
    data.y_target = data.y.clone()

    return data


def check_data_validity(data):
    """
    检查 Data 是否有效：
    1. x / edge_index / edge_attr / y / mask 是否存在
    2. 是否有 NaN / Inf
    3. mask 是否有至少一个 True
    """
    if data is None:
        return False

    required_attrs = ['x', 'edge_index', 'edge_attr', 'y', 'mask']
    for attr in required_attrs:
        if not hasattr(data, attr):
            return False

    def has_nan_or_inf(tensor):
        return torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item()

    if has_nan_or_inf(data.x):
        return False
    if has_nan_or_inf(data.edge_attr):
        return False
    if has_nan_or_inf(data.y):
        return False
    if data.mask.sum().item() == 0:
        return False

    return True


# ================= 主程序 =================

def main():
    print(f"🚀 启动数据生成 (目标 {NUM_SAMPLES} 个样本)")
    print(f"📦 SimBench 网: {SB_CODE}, STRESS ∈ [{STRESS_MIN}, {STRESS_MAX}]")
    print(f"⚙️  热限收紧系数: {RATE_TIGHTEN_FACTOR}, 负荷缩放: {LOAD_SCALE}")

    # 1. 加载网络并转换
    net_pp = sb.get_simbench_net(SB_CODE)
    grid_template = GC_PandaPowerImporter.PP2GC(net_pp)

    bus_idx_map = {get_gc_id(b): i for i, b in enumerate(grid_template.buses)}
    print(f"✅ 网络转换完成: {len(grid_template.buses)} 个节点, {len(grid_template.generators)} 个发电机")

    # 2. 加载时间序列 Profile
    profiles = sb.get_absolute_values(net_pp, profiles_instead_of_study_cases=True)
    df_load_p = profiles[('load', 'p_mw')]
    df_load_q = profiles[('load', 'q_mvar')]
    df_sgen_p = profiles[('sgen', 'p_mw')]
    n_timesteps = len(df_load_p)
    print(f"📈 时间步数: {n_timesteps}")

    # (A) 过滤掉总 sgen 功率太小的时间步
    valid_ts = [t for t in range(n_timesteps) if df_sgen_p.iloc[t].sum() > SGEN_POWER_THRESHOLD]
    if len(valid_ts) == 0:
        print("⚠️ 警告: 所有时间步的 sgen 总功率都太小，将回退为使用全部时间步。")
        valid_ts = list(range(n_timesteps))
    else:
        print(f"✅ 有效时间步数量 (sgen 总功率 > {SGEN_POWER_THRESHOLD}): {len(valid_ts)}")

    collected = 0
    chunk_idx = 0
    chunk_buffer = []

    pbar = tqdm(total=NUM_SAMPLES)

    while collected < NUM_SAMPLES:
        # --- A. 构造场景 ---
        t = int(np.random.choice(valid_ts))
        prob_selector = np.random.rand()  # 生成 0~1 的随机数

        if prob_selector < 0.25:
            # 前 25%: [0.7, 0.85]
            sgen_scale = np.random.uniform(0.7, 0.85)
        elif prob_selector < 0.75:
            # 中间 50% (0.25 ~ 0.75): [0.85, 1.15]
            sgen_scale = np.random.uniform(0.85, 1.15)
        else:
            # 后 25% (0.75 ~ 1.0): [1.15, 1.3]
            sgen_scale = np.random.uniform(1.15, 1.3)

        # 应用这个系数来计算 sgen 的值
        # 注意：这里把原来的 stress_factor 换成了 sgen_scale
        sgen_vals = (df_sgen_p.iloc[t] * sgen_scale).to_dict()
        grid_scene = deepcopy(grid_template)

        # 负荷
        l_p_now = df_load_p.iloc[t]
        l_q_now = df_load_q.iloc[t]
        for l in grid_scene.loads:
            try:
                lid = int(l.name.split('_')[1])
                _set_val(l, ['P', 'p'], LOAD_SCALE * l_p_now.get(lid, 0.0))
                _set_val(l, ['Q', 'q'], LOAD_SCALE * l_q_now.get(lid, 0.0))
            except Exception:
                continue

        # sgen Pmax / 初值
        sgen_vals = (df_sgen_p.iloc[t] * stress_factor).to_dict()
        for g in grid_scene.generators:
            if "sgen" in str(getattr(g, 'name', '')):
                try:
                    gid = int(g.name.split('_')[1])
                    pav = sgen_vals.get(gid, 0.0)
                    _set_val(g, ['Pmax', 'P_max'], pav)
                    _set_val(g, ['P', 'p'], pav)
                except Exception:
                    continue

        # ✅ 过滤：只有一个有功 sgen 的场景不要
        active_sgen_count = 0
        for g in grid_scene.generators:
            if "sgen" in str(getattr(g, 'name', '')):
                pmax = _get_val(g, ['Pmax', 'P_max'], 0.0)
                if pmax > 1e-4:
                    active_sgen_count += 1

        if active_sgen_count <= 1:
            continue

        # 热限收紧
        tighten_thermal_limits(grid_scene, RATE_TIGHTEN_FACTOR)

        # --- B. PF (bad state snapshot) ---
        pf_snapshot_v_abs = None
        pf_snapshot_v_angle = None
        pf_snapshot_loading = None

        try:
            grid_pf = deepcopy(grid_scene)
            pf_opts = gce.PowerFlowOptions(gce.SolverType.NR, verbose=False)
            pf_driver = gce.PowerFlowDriver(grid_pf, pf_opts)
            pf_driver.run()

            res_v = pf_driver.results.voltage
            if res_v is None or len(res_v) == 0:
                continue

            if not bool(pf_driver.results.converged):
                continue

            v_c = np.array(res_v, dtype=np.complex128)
            pf_snapshot_v_abs = np.abs(v_c).copy()
            pf_snapshot_v_angle = np.angle(v_c).copy()

            if pf_driver.results.loading is not None:
                pf_snapshot_loading = np.array(pf_driver.results.loading, dtype=np.float32).copy()
        except Exception:
            continue

        if pf_snapshot_v_abs is None:
            continue

        # --- C. OPF Teacher ---
        grid_opf = deepcopy(grid_scene)
        opf_driver = setup_and_run_opf_teacher(grid_opf, sgen_vals)

        is_opf_converged = False
        if hasattr(opf_driver, 'results'):
            rc = opf_driver.results.converged
            if isinstance(rc, (bool, np.bool_)):
                is_opf_converged = bool(rc)
            elif hasattr(rc, '__len__') and len(rc) > 0:
                is_opf_converged = bool(rc[0])

        if not is_opf_converged:
            continue

        opf_gen_p = get_robust_opf_results(opf_driver, grid_opf)

        # --- D. 构造样本并检查 ---
        data = build_graph_sample(
            grid_opf,
            pf_snapshot_v_abs,
            pf_snapshot_v_angle,
            pf_snapshot_loading,
            opf_gen_p,
            bus_idx_map
        )

        if not check_data_validity(data):
            continue

        # --- E. 写入缓冲 & 存盘 ---
        chunk_buffer.append(data)
        collected += 1
        pbar.update(1)

        if len(chunk_buffer) >= CHUNK_SIZE:
            save_path = os.path.join(SAVE_DIR, f"chunk_{chunk_idx:05d}.pt")
            torch.save(chunk_buffer, save_path)
            chunk_buffer = []
            chunk_idx += 1

    # 把剩余不满一个 chunk 的也存一下
    if len(chunk_buffer) > 0:
        save_path = os.path.join(SAVE_DIR, f"chunk_{chunk_idx:05d}.pt")
        torch.save(chunk_buffer, save_path)

    pbar.close()
    print(f"\n✅ 数据生成完成！总样本数: {collected}，输出目录: {SAVE_DIR}")


if __name__ == "__main__":
    main()
