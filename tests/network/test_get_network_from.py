
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
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    print("#### Test get network from")

    # Get network graph and save
    H = nw.get_network_from(
        config.tripdata["region"],
        config.root_map,
        config.graph_name,
        config.graph_file_name,
        max_travel_time_edge=30,
        speed_km_h=20
    )

    # Original graph
    G = nw.download_network(config.tripdata["region"], "drive")

    print("# ORIGINAL GRAPH STATS:")
    pprint(ox.basic_stats(G))

    print("\n# ENRICHED GRAPH STATS:")
    pprint(ox.basic_stats(H))