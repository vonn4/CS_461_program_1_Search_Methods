# src/main.py
import argparse
from algorithms import bfs, dfs, iddfs, greedy_best_first, astar
from graph_loader import load_set1_graph
from heuristics import euclidean_h
from visualize import draw_path
from benchmark import compare_algorithms, plot_bars

def parse_args():
    ap = argparse.ArgumentParser(description="CS 461 Program 1 — AI Search: Route Finding & Benchmarking")
    ap.add_argument("--adj", default="data/Adjacencies.txt", help="Adjacency pairs file (A B per line)")
    ap.add_argument("--coords", default="data/coordinates.csv", help="Coordinates CSV with name/lat/lon")
    ap.add_argument("--start", required=True, help="Start city name (use underscores if in file)")
    ap.add_argument("--goal", required=True, help="Goal city name (use underscores if in file)")
    ap.add_argument("--algo", choices=["bfs","dfs","iddfs","greedy","astar"], default="astar")
    ap.add_argument("--visualize", action="store_true", help="Draw the path on the graph")
    ap.add_argument("--benchmark", action="store_true", help="Compare algorithms (table & optional charts)")
    ap.add_argument("--repeats", type=int, default=5, help="Benchmark repeats")
    ap.add_argument("--max_depth", type=int, default=50, help="IDDFS max depth")
    return ap.parse_args()

def main():
    args = parse_args()
    G = load_set1_graph(args.adj, args.coords)

    if args.benchmark:
        df = compare_algorithms(G, args.start, args.goal, repeats=args.repeats)
        print(df.to_string(index=False))
        try:
            plot_bars(df)
        except Exception:
            pass  # okay in headless environments
        return

    h = euclidean_h(G, args.goal)
    if args.algo == "bfs":
        path, meta = bfs(G, args.start, args.goal)
    elif args.algo == "dfs":
        path, meta = dfs(G, args.start, args.goal)
    elif args.algo == "iddfs":
        path, meta = iddfs(G, args.start, args.goal, max_depth=args.max_depth)
    elif args.algo == "greedy":
        path, meta = greedy_best_first(G, args.start, args.goal, h)
    else:
        path, meta = astar(G, args.start, args.goal, h)

    print(f"Algorithm: {meta.get('algorithm')}")
    print(f"Path: {' -> '.join(path) if path else 'NO SOLUTION'}")
    print(f"Path cost: {meta.get('path_cost'):.4f}")
    print(f"Nodes expanded: {meta.get('nodes_expanded')}")
    print(f"Solution depth: {meta.get('solution_depth')}")
    print(f"Runtime (s): {meta.get('runtime_sec'):.6f}")
    print(f"Peak memory (bytes): {meta.get('peak_tracemalloc_bytes')}")

    if args.visualize:
        draw_path(G, path, title=f"{meta.get('algorithm')} path: {args.start} -> {args.goal}")

if __name__ == "__main__":
    main()
