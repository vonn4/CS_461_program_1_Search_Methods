# src/graph_loader.py
from pathlib import Path
import math
import networkx as nx
import pandas as pd

def load_set1_graph(adj_path: str, coord_path: str) -> nx.Graph:
    """
    Build an undirected weighted graph from Adjacencies.txt and coordinates.csv.
    Edge weights = Euclidean distance (lat/lon treated as a 2D plane).
    """
    G = nx.Graph()
    coords_df = pd.read_csv(coord_path)

    # Auto-detect likely column names
    name_col = [c for c in coords_df.columns if c.lower() in ("name", "city", "town")]
    lat_col = [c for c in coords_df.columns if "lat" in c.lower()][0]
    lon_col = [c for c in coords_df.columns if "lon" in c.lower() or "lng" in c.lower()][0]
    name_col = name_col[0] if name_col else coords_df.columns[0]

    # Add nodes with positions (lon, lat)
    for _, row in coords_df.iterrows():
        G.add_node(str(row[name_col]), pos=(float(row[lon_col]), float(row[lat_col])))

    # Read adjacency pairs (A B per line) and add undirected weighted edges
    with open(adj_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 2:
                parts = [p for p in parts if p]
            if len(parts) != 2:
                continue
            a, b = parts
            if a not in G:
                G.add_node(a)
            if b not in G:
                G.add_node(b)

            pos_a = G.nodes[a].get("pos")
            pos_b = G.nodes[b].get("pos")
            if pos_a and pos_b:
                dx = pos_a[0] - pos_b[0]
                dy = pos_a[1] - pos_b[1]
                w = math.hypot(dx, dy)
            else:
                w = 1.0  # fallback if either is missing coords

            if not G.has_edge(a, b):
                G.add_edge(a, b, weight=w)

    return G
