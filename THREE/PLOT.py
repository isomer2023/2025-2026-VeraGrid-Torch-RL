# pip install simbench pandapower networkx matplotlib
# 推荐额外安装（强烈建议）：pip install adjustText
# 可选更好布局（强烈建议）：conda install -c conda-forge pygraphviz
# 或者：pip install pygraphviz  (Windows 可能需要额外依赖)

import simbench as sb
import pandapower.topology as top
import networkx as nx
import matplotlib.pyplot as plt

def plot_simbench_topology(
    case="1-MV-urban--0-sw",
    figsize=(34, 22),
    dpi=220,
    node_size=85,
    font_size=7,
    edge_width=0.9,
    seed=2,
):
    # 1) load
    net = sb.get_simbench_net(case)

    # 2) topology graph (respect switches + include trafos)
    G = top.create_nxgraph(net, respect_switches=True, include_trafos=True)

    # 3) labels = bus name
    bus_name = net.bus["name"].astype(str).to_dict()
    labels = {b: bus_name.get(b, str(b)) for b in G.nodes()}

    # 4) layout: Graphviz (best) -> spring fallback
    pos = None
    layout_used = "spring_layout"
    try:
        # needs pygraphviz
        from networkx.drawing.nx_agraph import graphviz_layout
        # sfdp 对大图很稳，neato 有时更“规整”
        pos = graphviz_layout(G, prog="sfdp")
        layout_used = "graphviz:sfdp"
    except Exception:
        # fallback
        pos = nx.spring_layout(G, seed=seed, k=0.25, iterations=400)
        layout_used = "spring_layout"

    # 5) draw big + clear
    plt.figure(figsize=figsize, dpi=dpi)

    # edges first
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.55)

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=0.95)

    # 6) labels with overlap-reduction (adjustText)
    texts = []
    for n, (x, y) in pos.items():
        t = plt.text(
            x, y, labels[n],
            fontsize=font_size,
            ha="center", va="center",
            # 白底框让线穿过也能看清
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.70)
        )
        texts.append(t)

    # label repulsion
    try:
        from adjustText import adjust_text
        adjust_text(
            texts,
            x=[pos[n][0] for n in G.nodes()],
            y=[pos[n][1] for n in G.nodes()],
            # 这几个参数会让“挪动更激进”，重叠明显减少
            expand_text=(1.10, 1.25),
            expand_points=(1.10, 1.15),
            force_text=(0.20, 0.35),
            force_points=(0.05, 0.10),
            lim=3000,
            arrowprops=dict(arrowstyle="-", lw=0.3, alpha=0.35),
        )
        repulsion_used = "adjustText"
    except Exception:
        repulsion_used = "none (install adjustText to reduce overlaps)"

    plt.title(f"SimBench topology: {case} | layout={layout_used} | labels={repulsion_used}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # quick connectivity info
    comps = list(nx.connected_components(G))
    print(f"[{case}] buses={G.number_of_nodes()} edges={G.number_of_edges()} components={len(comps)}")

# ---- run
plot_simbench_topology("1-MV-urban--0-sw")
