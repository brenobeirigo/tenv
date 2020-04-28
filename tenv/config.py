import os
import json
import sys
from datetime import datetime
from pprint import pprint
import tenv.network as nw


def get_excerpt_name(start, stop, label="excerpt"):
    return (
        "tripdata_{}_{}_{}".format(label, start, stop)
        .replace(":", "")
        .replace(" ", "_")
    )


# TODO create dictionary of paths indexed by the region type
REGION_CONCENTRIC = "con"
REGION_REGULAR = "reg"

# How regions are sliced?
# region_slice = REGION_CONCENTRIC
region_slice = REGION_REGULAR

label_exp = ""
short_path = False
# root = os.getcwd().replace("\\", "/")
root = "C:/Users/LocalAdmin/OneDrive/leap_forward/street_network_server/tenv"
# root = "C:/Users/breno/Documents/phd/tenv"
# root = "C:/Users/LocalAdmin/Documents/GitHub/tenv"

########################################################################
# Dataset structure ####################################################
########################################################################

# Input data
tripdata = None
with open(f"{root}/data/in/config_scenario/rotterdam.json") as js:
    tripdata = json.load(js)

region = tripdata["region"]

# Create and store graph name
graph_name = (
    tripdata["label"]
    if "label" in tripdata
    else tripdata["region"].lower().replace(" ", "-").replace(",", "")
)

if short_path:
    graph_name = graph_name[:4]
    label_exp = label_exp[:1]

# Where dataset is saved
root_path = root + "/data/out/{}{}".format(label_exp, graph_name)

# -------------------------------------------------------------------- #
# Map ##################################################################
# -------------------------------------------------------------------- #

# Transportation network (.graphml and .svg)
root_map = root_path + "/map"

# Get remove superflous data
root_lean = root_path + "/lean_data/"

# Tests
root_test_network = root + "/tests/network"


graph_file_name = "{}.graphml".format(graph_name)

# -------------------------------------------------------------------- #
# Demand data ##########################################################
# -------------------------------------------------------------------- #

root_tripdata = root_path + "/tripdata"
root_tripdata_cleaned = root_tripdata + "/cleaned"
root_tripdata_raw = root_tripdata + "/raw"
root_tripdata_ids = root_tripdata + "/ids"

path_tripdata_ids = None
tripdata_filename = None
path_tripdata_source = None
path_tripdata = None
path_tripdata_clone = None

# Path of trip data with ids
if "url_tripdata" in tripdata:
    local = tripdata["url_tripdata"]

    # Presumably, the last part of the url is the file name
    tripdata_filename = f'{local.split("/")[-1]}'
    path_tripdata_source = "{}/{}".format(root_tripdata_raw, tripdata_filename)

    # Excerpt name shows time interval
    excerpt_name = get_excerpt_name(tripdata["start"], tripdata["stop"])

    path_tripdata_ids = "{}/{}_ids.csv".format(root_tripdata_ids, excerpt_name)
    path_tripdata = "{}/{}.csv".format(root_tripdata, excerpt_name)


# -------------------------------------------------------------------- #
# Distance #############################################################
# -------------------------------------------------------------------- #

root_dist = root_path + "/distance"
# Distance matrix
path_dist_matrix = "{}/dist_matrix_m.csv".format(root_dist)
# Distance dictionary (meters)
path_dist_dic = "{}/dist_dict_m.npy".format(root_dist)

path_dist_matrix_npy = "{}/dist_matrix_m.npy".format(root_dist)

########################################################################
# Reachability #########################################################
########################################################################

# Reachability layers
# (e.g., reachable in 30, 60, ..., total_range steps)
step = tripdata["reachability"]["step"]
total_range = tripdata["reachability"]["total_range"]
# If defined, step and total_range are assumed to be seconds
speed_km_h = tripdata["reachability"]["speed_km_h"]
round_trip = tripdata["reachability"].get("round_trip", False)

# Maximum number of node neighbors queried by application
max_neighbors = 6
# step_list = [0, 60, 300, 600]
# step_list = [0, 150, 300, 600]
step_list = tripdata["reachability"].get("step_list", [])
step_list_concentric = [60, 300, 600]

# Max travel time (seconds) to traverse an edge, i.e., if = 30, every
# edge can be traveled in 30 seconds
max_travel_time_edge = tripdata.get("max_travel_time_edge", 30)

# Max. time to execute ilp (min)
ilp_time_limit = 60

round_trip_label = "_roundtrip" if round_trip else ""

root_reachability = root_map + "/reachability_{}{}_{}_{}{}".format(
    region_slice,
    round_trip_label,
    step,
    total_range,
    ("_kmh{}".format(speed_km_h) if speed_km_h else ""),
)

root_img_regions = root_reachability + "/img_region_centers"

root_img_neighbors = root_reachability + "/img_region_center_neighbors"

# Reachability dictionary {o:{max_dist:[d1, d2, d3]}
path_reachability_dic = "{}/reach_dict.npy".format(root_reachability)

# Reachability dictionary round trip {o:{max_dist:[d1, d2, d3]}
path_reachability_r_dic = "{}/reach_dict_round.npy".format(root_reachability)

# Region centers dictionary {max_dist:[c1, c2, c3, c4, c5]}
path_region_centers = "{}/region_centers.npy".format(root_reachability)

path_region_center_ids = "{}/region_center_ids.npy".format(root_reachability)

path_sorted_neighbors = "{}/sorted_neighbors_region_centers.npy".format(
    root_reachability
)

path_sorted_neighbors_lean = "{}/sorted_neighbors_region_centers_{}.npy".format(
    root_reachability, "_".join([str(t) for t in step_list])
)

path_recheable_neighbors = "{}/sorted_reach_neighbors.npy".format(
    root_reachability
)

# Network
G = nw.load_network(graph_file_name, folder=root_map)


def info():
    return (
        "\n###############################################################"
        f"\n#      Region = {region}"
        f"\n# Aggregation = {region_slice}"
        f"\n#       Speed = {speed_km_h}"
        f"\n#  Step/Total = {step}/{total_range}"
        f"\n#       Steps = {step_list}"
        f"\n#       Round = {round_trip}"
        f"\n#      Source = {root_path}"
        f"\n#     Network = # nodes = {(len(G.nodes()) if G else '')}"
        # (f"({min(G.nodes())} -> {max(G.nodes())}) - #edges = {len(G.edges())}" if G else "")
        "\n###############################################################"
    )


def make_folders():
    # Create all folders where data will be saved
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    if not os.path.exists(root_dist):
        os.makedirs(root_dist)

    if not os.path.exists(root_map):
        os.makedirs(root_map)

    if not os.path.exists(root_reachability):
        os.makedirs(root_reachability)

    # Trip data ###################################################### #

    if not os.path.exists(root_tripdata):
        os.makedirs(root_tripdata)

    if not os.path.exists(root_tripdata_raw):
        os.makedirs(root_tripdata_raw)

    if not os.path.exists(root_tripdata_cleaned):
        os.makedirs(root_tripdata_cleaned)

    if not os.path.exists(root_tripdata_ids):
        os.makedirs(root_tripdata_ids)

    # Plots ########################################################## #
    if not os.path.exists(root_img_regions):
        os.makedirs(root_img_regions)

    if not os.path.exists(root_img_neighbors):
        os.makedirs(root_img_neighbors)
