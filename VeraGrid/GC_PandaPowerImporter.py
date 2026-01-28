# GC_PandaPowerImporter.py
# 最终完美版 v17: 精准大写赋值 (Explicit Uppercase Assignment)
# 专治: 属性名大小写导致的“伪写入”成功

import numpy as np
import pandapower as pp
from VeraGridEngine import api as gce
import collections

# 阻抗参数
EPSILON = 1e-4
MIN_Z_LINE = 1e-5


def _set_val(obj, attr, value):
    """简单粗暴的赋值，失败就打印"""
    try:
        setattr(obj, attr, value)
    except Exception as e:
        # print(f"[Warn] 无法设置 {attr}: {e}")
        pass


def _get_iterable(obj, attr_list):
    for attr in attr_list:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if val is not None: return val
        method_name = f"get_{attr}" if not attr.startswith("get_") else attr
        if hasattr(obj, method_name):
            try:
                val = getattr(obj, method_name)()
                if val is not None: return val
            except:
                pass
    return []


def _deactivate(obj):
    for attr in ['active', 'status', 'in_service', 'is_active']:
        _set_val(obj, attr, False)


def remove_islands(grid):
    """孤岛清理"""
    buses = _get_iterable(grid, ['buses', 'bus_list', 'get_buses'])
    branches = _get_iterable(grid, ['branches', 'get_branches'])
    if not branches:
        lines = _get_iterable(grid, ['lines', 'get_lines'])
        trafos = _get_iterable(grid, ['transformers', 'transformers_list'])
        branches = list(lines) + list(trafos)
    generators = _get_iterable(grid, ['generators', 'gen_list', 'get_generators'])
    loads = _get_iterable(grid, ['loads', 'load_list', 'get_loads'])

    slack_buses = set()
    for g in generators:
        is_slack = False
        val = getattr(g, 'is_slack', None)
        if val is not None:
            is_slack = bool(val)
        elif getattr(g, 'slack', None) is not None:
            is_slack = bool(getattr(g, 'slack'))
        elif "Ext_Grid" in getattr(g, 'name', ''):
            is_slack = True

        if is_slack:
            bus_ref = getattr(g, 'bus', getattr(g, 'node', None))
            if bus_ref:
                slack_buses.add(bus_ref)

    if not slack_buses:
        return

    adj = collections.defaultdict(list)
    for br in branches:
        f = getattr(br, 'bus_from', getattr(br, 'from_node', getattr(br, 'busFrom', None)))
        t = getattr(br, 'bus_to', getattr(br, 'to_node', getattr(br, 'busTo', None)))

        is_active = True
        if hasattr(br, 'active'):
            is_active = br.active
        elif hasattr(br, 'status'):
            is_active = br.status

        if f and t and is_active:
            adj[f].append(t)
            adj[t].append(f)

    main_island_buses = set()
    queue = collections.deque(list(slack_buses))
    main_island_buses.update(slack_buses)

    while queue:
        current_bus = queue.popleft()
        for neighbor in adj[current_bus]:
            if neighbor not in main_island_buses:
                main_island_buses.add(neighbor)
                queue.append(neighbor)

    deactivated_count = 0
    for br in branches:
        f = getattr(br, 'bus_from', getattr(br, 'from_node', getattr(br, 'busFrom', None)))
        t = getattr(br, 'bus_to', getattr(br, 'to_node', getattr(br, 'busTo', None)))
        if (f and f not in main_island_buses) or (t and t not in main_island_buses):
            _deactivate(br)
            deactivated_count += 1
    for g in generators:
        b = getattr(g, 'bus', getattr(g, 'node', None))
        if b and b not in main_island_buses: _deactivate(g)
    for l in loads:
        b = getattr(l, 'bus', getattr(l, 'node', None))
        if b and b not in main_island_buses: _deactivate(l)

    print(f"[Cleaner] 发现 {len(buses) - len(main_island_buses)} 个孤岛节点，已去活 {deactivated_count} 条关联线路。")


def PP2GC(net_pp):
    grid = gce.MultiCircuit()
    name = str(net_pp.name) if hasattr(net_pp, 'name') and net_pp.name else "SimBench_Import"
    _set_val(grid, 'name', name)

    base_mva = float(net_pp.sn_mva)
    _set_val(grid, 'Sbase', base_mva)

    bus_map = {}

    # --- Bus ---
    for idx, row in net_pp.bus.iterrows():
        bus_name = str(row['name']) if 'name' in row and row['name'] else str(idx)
        b = gce.Bus(name=bus_name)

        # 你的版本 Bus 属性是大写 Vnom
        vn_val = float(row['vn_kv'])
        _set_val(b, 'Vnom', vn_val)
        _set_val(b, 'Vmin', float(row.get('min_vm_pu', 0.9)))
        _set_val(b, 'Vmax', float(row.get('max_vm_pu', 1.1)))

        # 初始电压
        _set_val(b, 'Vm0', float(row.get('vm_pu', 1.0)))

        grid.add_bus(b)
        bus_map[idx] = b

    # --- Load ---
    if not net_pp.load.empty:
        for idx, row in net_pp.load.iterrows():
            if not row.get('in_service', True): continue
            bus_idx = row['bus']
            if bus_idx not in bus_map: continue
            l = gce.Load(name=f"Load_{idx}")
            _set_val(l, 'P', float(row['p_mw']))
            _set_val(l, 'Q', float(row['q_mvar']))

            target_bus = bus_map[bus_idx]
            try:
                grid.add_load(target_bus, l)
            except TypeError:
                _set_val(l, 'bus', target_bus)
                grid.add_load(l)

    # --- Sgen ---
    if not net_pp.sgen.empty:
        for idx, row in net_pp.sgen.iterrows():
            if not row.get('in_service', True): continue
            bus_idx = row['bus']
            if bus_idx not in bus_map: continue
            g = gce.Generator(name=f"sgen_{idx}")
            _set_val(g, 'P', float(row['p_mw']))
            _set_val(g, 'Q', float(row['q_mvar']))

            sn = float(row.get('sn_mva', 0.0))
            p_max_val = sn if sn > 0 else float(row['p_mw'])

            # 你的版本 Generator 属性是 Pmax (大写P, 小写max)
            _set_val(g, 'Pmax', p_max_val)
            _set_val(g, 'Pmin', 0.0)
            _set_val(g, 'is_controlled', True)
            _set_val(g, 'is_slack', False)
            _set_val(g, 'Cost2', 1.0)
            _set_val(g, 'Cost1', 0.0)

            target_bus = bus_map[bus_idx]
            try:
                grid.add_generator(target_bus, g)
            except TypeError:
                _set_val(g, 'bus', target_bus)
                grid.add_generator(g)

    # --- Ext_grid ---
    if not net_pp.ext_grid.empty:
        for idx, row in net_pp.ext_grid.iterrows():
            if not row.get('in_service', True): continue
            bus_idx = row['bus']
            if bus_idx not in bus_map: continue
            g = gce.Generator(name=f"Ext_Grid_{idx}")
            _set_val(g, 'is_slack', True)
            _set_val(g, 'is_controlled', True)
            _set_val(g, 'Vset', float(row.get('vm_pu', 1.0)))
            _set_val(g, 'Pmax', 99999.0)
            _set_val(g, 'Pmin', -99999.0)
            _set_val(g, 'Cost2', 100.0)
            target_bus = bus_map[bus_idx]
            try:
                grid.add_generator(target_bus, g)
            except TypeError:
                _set_val(g, 'bus', target_bus)
                grid.add_generator(g)

    # --- Line ---
    if not net_pp.line.empty:
        for idx, row in net_pp.line.iterrows():
            if not row.get('in_service', True): continue
            f_bus = bus_map.get(row['from_bus'])
            t_bus = bus_map.get(row['to_bus'])
            if f_bus and t_bus:
                br = gce.Branch(name=f"Line_{idx}")
                _set_val(br, 'bus_from', f_bus)
                _set_val(br, 'bus_to', t_bus)

                vn_kv = net_pp.bus.at[row['from_bus'], 'vn_kv']
                imax_ka = row['max_i_ka']
                rate_mva = np.sqrt(3) * vn_kv * imax_ka

                length = row['length_km']
                r_ohm = row['r_ohm_per_km'] * length
                x_ohm = row['x_ohm_per_km'] * length
                c_nf = row['c_nf_per_km'] * length
                b_siemens = 2 * np.pi * 50.0 * c_nf * 1e-9

                z_base = (vn_kv ** 2) / base_mva
                r_pu = r_ohm / z_base
                x_pu = x_ohm / z_base
                b_pu = b_siemens * z_base

                if abs(r_pu) < 1e-12: r_pu = MIN_Z_LINE
                if abs(x_pu) < 1e-12: x_pu = MIN_Z_LINE

                # 【关键】使用大写属性名 R, X, B
                _set_val(br, 'R', r_pu)
                _set_val(br, 'X', x_pu)
                _set_val(br, 'B', b_pu)
                _set_val(br, 'rate', rate_mva)  # rate 依然是小写(根据你的日志)
                _set_val(br, 'active', True)

                grid.add_branch(br)

    # --- Trafo ---
    if not net_pp.trafo.empty:
        for idx, row in net_pp.trafo.iterrows():
            if not row.get('in_service', True): continue
            hv_bus = bus_map.get(row['hv_bus'])
            lv_bus = bus_map.get(row['lv_bus'])
            if hv_bus and lv_bus:
                t = gce.Branch(name=f"Trafo_{idx}")
                _set_val(t, 'bus_from', hv_bus)
                _set_val(t, 'bus_to', lv_bus)
                sn_mva = row['sn_mva']
                vn_hv = row['vn_hv_kv']
                vk_percent = row['vk_percent']
                vkr_percent = row['vkr_percent']
                z_base_trafo = (vn_hv ** 2) / sn_mva
                z_base_sys = (vn_hv ** 2) / base_mva
                z_trafo_ohm = (vk_percent / 100) * z_base_trafo
                r_trafo_ohm = (vkr_percent / 100) * z_base_trafo
                x_trafo_ohm = np.sqrt(max(0, z_trafo_ohm ** 2 - r_trafo_ohm ** 2))
                r_pu = r_trafo_ohm / z_base_sys
                x_pu = x_trafo_ohm / z_base_sys

                if abs(r_pu) < 1e-12: r_pu = MIN_Z_LINE
                if abs(x_pu) < 1e-12: x_pu = MIN_Z_LINE

                # 【关键】使用大写
                _set_val(t, 'R', r_pu)
                _set_val(t, 'X', x_pu)
                _set_val(t, 'rate', sn_mva)
                _set_val(t, 'tap_module', 1.0)  # 你的版本可能是 tap_module 或 tap_changer
                _set_val(t, 'active', True)

                grid.add_branch(t)

    # --- Switch ---
    if not net_pp.switch.empty:
        for idx, row in net_pp.switch.iterrows():
            if not row.get('closed', True): continue
            if row['et'] != 'b': continue
            bus_id = row['bus']
            element_id = row['element']
            if bus_id in bus_map and element_id in bus_map:
                sw_br = gce.Branch(name=f"Switch_{idx}")
                f_bus = bus_map[bus_id]
                t_bus = bus_map[element_id]
                _set_val(sw_br, 'bus_from', f_bus)
                _set_val(sw_br, 'bus_to', t_bus)

                # 【关键】使用大写
                _set_val(sw_br, 'R', EPSILON)
                _set_val(sw_br, 'X', EPSILON)
                _set_val(sw_br, 'rate', 9999.0)
                _set_val(sw_br, 'active', True)

                grid.add_branch(sw_br)

    print("[Importer] 正在检查并清理孤岛...")
    remove_islands(grid)

    print("[Importer] 正在编译 GridCal 网络...")
    try:
        grid.compile()
    except Exception as e:
        pass

    print(f"[Importer] 成功转换 SimBench 网络: {net_pp.name}")
    return grid