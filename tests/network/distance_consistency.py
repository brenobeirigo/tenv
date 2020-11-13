import functools
import json
import sys
import os
from pprint import pprint
from bisect import bisect_right, bisect_left
import copy
from collections import defaultdict
import pandas as pd
import time

# Adding project folder
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import tenv.config as config
import tenv.network as nw
import tenv.demand as tp
import numpy as np
import tenv.visuals as vi
import math
import logging

logging.getLogger().setLevel(logging.INFO)

print(config.info())

# Time log dictionary - Save time to execute each step
time_dict = dict()

# Network time
t_start = time.time()

print("Loading network...")
G = nw.load_network(config.graph_file_name, folder=config.root_map)

G_duration = nw.add_duration_to_graph(G, config.speed_km_h)

dist_matrix_duration = pd.read_csv(
    config.path_dist_matrix_duration, header=None
)

for o in G_duration.nodes():
    for d in G_duration.nodes():
        if o == d:
            continue

        sp_od = nw.get_sp(G_duration, o, d, weight="duration")
        sp_od2 = nw.get_sp(G_duration, o, d)

        edges_od = list(zip(sp_od[:-1], sp_od[1:]))

        for a, b in edges_od:

            # All edges have 60s length
            dist_graph = G_duration[a][b][0]["duration"]
            assert dist_graph <= 60, f"Distance is higher!{dist}"

            # Saved data is consistent with graph
            dist_matrix = dist_matrix_duration.iloc[a, b]
            assert (
                dist_graph == dist_matrix
            ), f"Distance matrix is wrong! {a}->{b} = {dist_matrix} <> {dist_graph}"

            # Distance function is working
            dist_legacy = nw.get_distance(G_duration, a, b, weight="duration")
            assert (
                dist_graph == dist_legacy
            ), f"Distances differ!{dist_graph} <> {dist_legacy}"

        total_dist_graph = nw.get_distance(G_duration, o, d, weight="duration")
        total_dist_matrix_sum = sum(
            [dist_matrix_duration.iloc[a, b] for a, b in edges_od]
        )
        total_dist_matrix = dist_matrix_duration.iloc[o, d]
        # # dist = distance_matrix[o][d]
        # sp_dists = [dist_matrix_duration[a][b] for a, b in edges_od]
        # sp_dist_sum = sum(sp_dists)

        # dist_sec = dist_matrix_duration[o][d]
        print(total_dist_matrix_sum, total_dist_matrix)
        assert (
            total_dist_graph == total_dist_matrix_sum
            and total_dist_matrix_sum == total_dist_matrix
        ), (
            f"{o} - {d} -> total_dist_graph: {total_dist_graph}"
            f" total_dist_matrix_sum: {total_dist_matrix_sum}"
            f" total_dist_matrix: {total_dist_matrix}"
        )
