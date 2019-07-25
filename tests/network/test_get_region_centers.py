import os
import sys
from pprint import pprint
from collections import defaultdict

# Adding project folder to import config and network_gen
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import tenv.config as config
import tenv.network as nw
import numpy as np

# Network
G = nw.load_network(config.graph_file_name, folder=config.root_map)
print(
    "# NETWORK -  NODES: {} ({} -> {}) -- #EDGES: {}".format(
        len(G.nodes()), min(G.nodes()), max(G.nodes()), len(G.edges())
    )
)

# Creating distance dictionary [o][d] -> distance
distance_dic = nw.get_distance_dic(config.path_dist_dic, G)

# Creating reachability dictionary
# 30, 60, ..., 570, 600 (30s to 10min)
steps_sec = 30
total_sec = 600
speed_km_h = 30

reachability_dic = nw.get_reachability_dic(
    config.path_reachability_dic,
    distance_dic,
    step=steps_sec,
    total_range=total_sec,
    speed_km_h=speed_km_h,
    step_list=None,
)

region_centers = nw.get_region_centers(
    config.path_region_centers,
    reachability_dic,
    root_path=config.root_reachability,
    time_limit=config.ilp_time_limit,
)

print("# Regions centers/Max. distance(s)")
pprint(region_centers)


max_distance = 30
print(
    f"\n# Reachable nodes from regions centers (max distance = {max_distance})"
)
for o in region_centers[max_distance]:
    print(f"{o:>5} - {reachability_dic[o][max_distance]}")

node_center = defaultdict(lambda: defaultdict(int))

for d in range(30, 150, 30):
    for o in region_centers[d]:
        for n in reachability_dic[o][d]:
            node_center[n][d] = o

pprint(node_center)
