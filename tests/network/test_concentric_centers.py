import os
import sys
import matplotlib.pyplot as plt
from collections import Counter
import osmnx as ox
import networkx as nx
from pprint import pprint


# Adding project folder
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import tenv.config as config
import tenv.network as nw
import tenv.visuals as vi

if __name__ == "__main__":

    # Max trip time to test reachability dictionary
    MAX_TRIP_TIME = 30

    # Get network graph and save
    G = nw.get_network_from(
        config.tripdata["region"],
        config.root_map,
        config.graph_name,
        config.graph_file_name,
    )

    # Creating distance dictionary [o][d] -> distance
    distance_dic = nw.get_distance_dic(config.path_dist_dic, G)

    # Try to load from path, if does't exist generate
    reachability_dict, steps_in_range_list = nw.get_reachability_dic(
        config.path_reachability_dic, distance_dic, step=60, total_range=600
    )

    steps = sorted(steps_in_range_list)

    concentric, centers = nw.concentric_regions(
        G,
        steps,
        reachability_dict,
        list(G.nodes()),
        center=-1,
        root_reachability=config.root_reachability_concentric,
    )

    pprint(concentric)

    n_regions = nw.get_node_region_ids(G, concentric)

    # Plot region centers (blue) and associated nodes
    print("Plotting regions...")
    vi.plot_regions(
        G,
        centers,
        concentric,
        path=config.root_img_regions_concentric,
        show=False,
        file_format="png",
        replace=False,
    )

    pprint(n_regions)
    pprint(centers)
