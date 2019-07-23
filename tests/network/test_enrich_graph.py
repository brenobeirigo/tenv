import os
import sys
import matplotlib.pyplot as plt
from collections import Counter
import osmnx as ox
import networkx as nx
from pprint import pprint
import numpy as np

# Adding project folder
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import tenv.config as config
import tenv.network as nw
import bisect
from shapely.geometry import Point, LineString
from copy import deepcopy


if __name__ == "__main__":

    # Get network graph and save
    H = nw.get_network_from(
        config.tripdata["region"],
        config.root_map,
        config.graph_name,
        config.graph_file_name,
    )

    G = nw.enrich_graph(H, max_travel_time_edge=30, n_coords=100)

    print(f"#Nodes - Original: {len(H.nodes())} - Enriched: {len(G.nodes())}")

    # Comparison precision between distances
    precision = 1

    ox.plot_graph(G)

    print("\n### Comparing distances (original X enriched)")
    try:
        all_dists_gen_g = np.load("enriched_network.npy").item()
    except:
        all_dists_gen_g = dict(
            nx.all_pairs_dijkstra_path_length(G, weight="length")
        )
        np.save("enriched_network.npy", all_dists_gen_g)

    try:
        all_dists_gen_h = np.load("original_network.npy").item()
    except:
        all_dists_gen_h = dict(
            nx.all_pairs_dijkstra_path_length(H, weight="length")
        )
        np.save("original_network.npy", all_dists_gen_h)

    # Loop all possible pairs from original graph and check whether
    # distances remain the same in
    for o, d_dist_original in all_dists_gen_h.items():
        for d, dist in d_dist_original.items():
            dist_g = all_dists_gen_g[o][d]
            if round(dist_g, precision) == round(dist, precision):
                print(f"Different {round(dist_g, 1)} == {round(dist, 1)}")

    ox.plot_graph(G)
