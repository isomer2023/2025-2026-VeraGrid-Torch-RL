import pandapower as pp
import matplotlib.pyplot as plt
import pandas as pd


def simple_grid_plot():
    """
    使用pandapower内置功能绘制电网拓扑
    """
    try:
        # 尝试读取SimBench网络
        import simbench as sb
        # 选择一个较小的网络用于测试
        net = sb.get_simbench_net('1-MV-urban--0-sw')
        print(f"读取SimBench网络: {net.name}")
    except Exception as e:
        print(f"读取SimBench网络失败: {e}")
        # 使用pandapower测试网络
        net = pp.networks.case9()
        net.name = "IEEE 9-bus System"
        print(f"使用测试网络: {net.name}")

    # 显示基本信息
    print(f"\n网络基本信息:")
    print(f"节点数: {len(net.bus)}")
    print(f"线路数: {len(net.line)}")

    # 使用pandapower的简单绘图
    from pandapower.plotting import simple_plot

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制拓扑图
    simple_plot(net, ax=ax,
                plot_loads=True,
                plot_gens=True,
                plot_sgens=True,
                load_size=1.2,
                gen_size=1.2,
                sgen_size=1.0,
                bus_size=1.0,
                bus_color='blue',
                line_color='gray',
                trafo_color='orange',
                ext_grid_color='purple',
                plot_line_switches=False,
                show_plot=False)

    # 设置标题
    ax.set_title(f'电网拓扑图 - {net.name}', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

    # 输出发电机信息
    print("\n" + "=" * 60)
    print("发电机信息汇总:")
    print("=" * 60)

    # 创建信息表格
    gen_data = []

    # 处理常规发电机
    if 'gen' in net and not net.gen.empty:
        for idx, gen in net.gen.iterrows():
            bus_idx = gen.bus
            bus_name = net.bus.loc[bus_idx, 'name'] if 'name' in net.bus.columns else f"Bus_{bus_idx}"

            gen_info = {
                '类型': 'Generator',
                '节点编号': bus_idx,
                '节点名称': bus_name,
                '有功功率(MW)': gen.get('p_mw', 'N/A'),
                '电压(pu)': gen.get('vm_pu', 'N/A'),
                '额定功率(MVA)': gen.get('sn_mva', 'N/A')
            }
            gen_data.append(gen_info)

    # 处理静态发电机
    if 'sgen' in net and not net.sgen.empty:
        for idx, sgen in net.sgen.iterrows():
            bus_idx = sgen.bus
            bus_name = net.bus.loc[bus_idx, 'name'] if 'name' in net.bus.columns else f"Bus_{bus_idx}"

            sgen_info = {
                '类型': 'Static Generator',
                '节点编号': bus_idx,
                '节点名称': bus_name,
                '有功功率(MW)': sgen.get('p_mw', 'N/A'),
                '无功功率(MVar)': sgen.get('q_mvar', 'N/A'),
                '额定功率(MVA)': sgen.get('sn_mva', 'N/A')
            }
            gen_data.append(sgen_info)

    # 显示表格
    if gen_data:
        df = pd.DataFrame(gen_data)
        print(df.to_string(index=False))

        # 统计信息
        print(f"\n总计: {len(gen_data)} 台发电设备")

        # 计算总有功功率
        total_p = 0
        for gen in gen_data:
            p = gen.get('有功功率(MW)', 0)
            if isinstance(p, (int, float)):
                total_p += p
        print(f"总有功功率: {total_p:.2f} MW")
    else:
        print("网络中未找到发电机设备")

    print("=" * 60)


# 运行程序
simple_grid_plot()