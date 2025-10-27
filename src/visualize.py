# src/visualize.py
import networkx as nx
import matplotlib.pyplot as plt

def draw_path(G: nx.Graph, path, title="Best Path"):
    # Use stored positions if present; otherwise compute a layout
    if all("pos" in G.nodes[n] for n in G.nodes):
        pos = {n: G.nodes[n]["pos"] for n in G.nodes}
    else:
        pos = nx.spring_layout(G, seed=42)

    plt.figure()
    nx.draw(G, pos, node_size=50, alpha=0.5)  # base graph
    if path:
        edgelist = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=120)
        nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=3)
        nx.draw_networkx_labels(G, pos, font_size=6)
        plt.title(title)
    else:
        plt.title(title + " (no solution)")
    plt.tight_layout()
    plt.show()
