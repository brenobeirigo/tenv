import functools
import json
import sys
import os
from pprint import pprint
import copy
from collections import defaultdict

import matplotlib.pyplot as plt


# Adding project folder
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import tenv.config as config
import tenv.network as nw
import tenv.demand as tp
import numpy as np
import math

# Network
G = nw.load_network(config.graph_file_name, folder=config.root_map)

print(config.info())

# Creating distance dictionary [o][d] -> distance
distance_dic = nw.get_distance_dic(config.path_dist_dic, G)

# Reachability dictionary
reachability_dict, steps = nw.get_reachability_dic(
    config.path_reachability_dic,
    distance_dic,
    step=config.step,
    total_range=config.total_range,
    speed_km_h=config.speed_km_h,
    step_list=None,
)

if config.region_slice == config.REGION_REGULAR:
    # All region centers
    region_centers = nw.get_region_centers(
        steps,
        config.path_region_centers,
        reachability_dict,
        list(G.nodes()),
        root_path=config.root_reachability,
        round_trip=config.round_trip,
    )

    # What is the closest region center of every node (given a time limit)?
    region_id_dict = nw.get_region_ids(
        G,
        reachability_dict,
        region_centers,
        path_region_ids=config.path_region_center_ids,
    )

    sorted_neighbors = nw.get_sorted_neighbors(
        distance_dic,
        region_centers,
        path_sorted_neighbors=config.path_sorted_neighbors,
    )

elif config.region_slice == config.REGION_CONCENTRIC:

    region_id_dict, region_centers = nw.concentric_regions(
        G,
        steps,
        reachability_dict,
        list(G.nodes()),
        center=-1,
        root_reachability=config.root_reachability_concentric,
    )

    sorted_neighbors = nw.get_sorted_neighbors(
        distance_dic,
        region_centers,
        path_sorted_neighbors=config.path_sorted_neighbors_concentric,
    )

node_region_ids = nw.get_node_region_ids(G, region_id_dict)
center_nodes = nw.get_center_nodes(region_id_dict)

count_step_center = defaultdict(lambda: defaultdict(int))
for o, step_c in region_id_dict.items():
    for step, c in step_c.items():
        count_step_center[step][c] += 1

for step, c_count in count_step_center.items():
    sorted_c_count = sorted(list(c_count.items()), key=(lambda item: item[1]))
    pprint(sorted_c_count)
    x, y = zip(*sorted_c_count)
    x = [str(c) for c in x]
    plt.title(f"Node count per center (step={step}, nodes={len(x)})")
    plt.xlabel(f"Region center ids")
    plt.ylabel("Node count")
    plt.yscale("log")
    plt.bar(x, y)
    plt.show()
