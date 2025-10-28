# src/graph_loader.py
from pathlib import Path
import math
import re
import networkx as nx
import pandas as pd

def _find_col(columns, keywords):
    """
    Return the first column whose lowercase name contains any of the keywords.
    """
    cl = [c.strip() for c in columns]
    for c in cl:
        name = c.lower().strip()
        for kw in keywords:
            if kw in name:
                return c
    return None

def _infer_name_lat_lon(df: pd.DataFrame):
    """
    Try hard to infer the 'name', 'lat', and 'lon' columns from the CSV.
    Handles many variants; if still ambiguous, falls back to positional guesses.
    """
    cols = list(df.columns)

    # name-like columns
    name_col = _find_col(cols, ["name", "city", "town", "place", "label"])
    # latitude-like columns
    lat_col  = _find_col(cols, ["lat", "latitude", "y"])
    # longitude-like columns
    lon_col  = _find_col(cols, ["lon", "long", "longitude", "lng", "x"])

    # If lat/lon still missing, try heuristics:
    if lat_col is None or lon_col is None:
        # Look for numeric candidates
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        # If we have at least 2 numeric columns, assume the first two are coordinates
        if len(numeric_cols) >= 2:
            # Keep name_col if we already have it; otherwise, assume the first non-numeric is the name
            if name_col is None:
                non_numeric = [c for c in cols if c not in numeric_cols]
                name_col = non_numeric[0] if non_numeric else cols[0]
            # Pick the first two numeric columns for lat/lon, but try guessing order by value ranges
            c1, c2 = numeric_cols[0], numeric_cols[1]
            # Heuristic: latitude usually ranges ~[-90, 90]; longitude ~[-180, 180]
            v1 = df[c1].dropna().astype(float)
            v2 = df[c2].dropna().astype(float)
            if v1.abs().max() <= 90 and v2.abs().max() <= 180:
                lat_col, lon_col = c1, c2
            elif v2.abs().max() <= 90 and v1.abs().max() <= 180:
                lat_col, lon_col = c2, c1
            else:
                # fallback order
                lat_col, lon_col = c1, c2

    # Final fallbacks if name is still None
    if name_col is None:
        name_col = cols[0]

    # If still missing either coordinate column, raise an actionable error
    if lat_col is None or lon_col is None:
        raise ValueError(
            "Could not infer latitude/longitude columns from coordinates.csv. "
            "Please ensure there are columns for lat/latitude and lon/long/longitude, "
            f"or provide two numeric columns after the name column. Found columns: {cols}"
        )

    return name_col, lat_col, lon_col

def load_set1_graph(adj_path: str, coord_path: str) -> nx.Graph:
    """
    Build an undirected weighted graph from Adjacencies.txt and coordinates.csv.
    Edge weights = Euclidean distance (lon,lat) treated as a 2D plane.
    """
    G = nx.Graph()

    # Load coordinates
    coords_df = pd.read_csv(coord_path)
    name_col, lat_col, lon_col = _infer_name_lat_lon(coords_df)

    # Add nodes with positions (lon, lat)
    for _, row in coords_df.iterrows():
        name = str(row[name_col]).strip()
        # Skip empty names
        if not name or name.lower() == "nan":
            continue
        # Some files may have spaces; assignment notes often use underscores
        # We'll keep whatever the CSV provides, and you will use that exact spelling.
        try:
            lon = float(row[lon_col])
            lat = float(row[lat_col])
            G.add_node(name, pos=(lon, lat))
        except Exception:
            # If coords missing, still add the node; edges will fall back to weight=1
            G.add_node(name)

    # Load adjacencies; accept space/comma/tab separated pairs
    with open(adj_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = re.split(r"[,\s]+", s)
            if len(parts) < 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
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
                w = 1.0

            if not G.has_edge(a, b):
                G.add_edge(a, b, weight=w)

    return G
