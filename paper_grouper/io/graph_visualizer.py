from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from paper_grouper.core.data import ClusteringResult


def render_graph_png(G: nx.Graph,
                     clustering: ClusteringResult,
                     output_root: Path,
                     filename: str = "graph_overview.png") -> Path:

    color_map = []
    sizes = []
    for node in G.nodes():
        cid = clustering.article_to_cluster.get(node, -1)
        color_map.append(cid)
        sizes.append(80 + 1200 * clustering.centrality.get(node, 0.0))

    pos = nx.spring_layout(G, weight="weight", seed=42)

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5)
    plt.axis("off")

    out_path = output_root / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path
