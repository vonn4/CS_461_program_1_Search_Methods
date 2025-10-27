# src/benchmark.py
import statistics as stats
import pandas as pd
import networkx as nx
from typing import Callable, List, Tuple

from algorithms import bfs, dfs, iddfs, greedy_best_first, astar
from heuristics import euclidean_h

def compare_algorithms(G: nx.Graph, start: str, goal: str, repeats: int = 5) -> pd.DataFrame:
    h = euclidean_h(G, goal)
    algos: List[Tuple[str, Callable]] = [
        ("BFS",     lambda: bfs(G, start, goal)),
        ("DFS",     lambda: dfs(G, start, goal)),
        ("IDDFS",   lambda: iddfs(G, start, goal)),
        ("Greedy",  lambda: greedy_best_first(G, start, goal, h)),
        ("A*",      lambda: astar(G, start, goal, h)),
    ]

    rows = []
    for name, fn in algos:
        runtimes, peaks, nodes, costs, depths = [], [], [], [], []
        for _ in range(repeats):
            path, meta = fn()
            runtimes.append(meta.get("runtime_sec", float("nan")))
            peaks.append(meta.get("peak_tracemalloc_bytes", float("nan")))
            nodes.append(meta.get("nodes_expanded", float("nan")))
            costs.append(meta.get("path_cost", float("nan")))
            depths.append(meta.get("solution_depth", float("nan")))
        rows.append({
            "algorithm": name,
            "runtime_mean_s": stats.mean(runtimes),
            "runtime_std_s": stats.pstdev(runtimes),
            "peak_mem_mean_bytes": stats.mean(peaks),
            "nodes_expanded_mean": stats.mean(nodes),
            "path_cost_mean": stats.mean(costs),
            "solution_depth_mean": stats.mean(depths),
        })

    return pd.DataFrame(rows).sort_values("runtime_mean_s")

def plot_bars(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    # runtime chart
    plt.figure()
    plt.bar(df["algorithm"], df["runtime_mean_s"])
    plt.title("Runtime (mean)")
    plt.ylabel("seconds")
    plt.tight_layout()
    plt.show()

    # memory chart
    plt.figure()
    plt.bar(df["algorithm"], df["peak_mem_mean_bytes"])
    plt.title("Peak Memory (mean)")
    plt.ylabel("bytes")
    plt.tight_layout()
    plt.show()
