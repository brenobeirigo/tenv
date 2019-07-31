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


REGION_CONCENTRIC = "CONCENTRIC"
REGION_REGULAR = "REGULAR"

# How regions are sliced?
# region_slice = REGION_CONCENTRIC
region_slice = REGION_REGULAR
# label_exp = "CON"
label_exp = "15"

# root = os.getcwd().replace("\\", "/")
root = "C:/Users/LocalAdmin/OneDrive/leap_forward/street_network_server/tenv"
# root = "C:/Users/breno/Documents/phd/tenv"
# root = "C:/Users/LocalAdmin/Documents/GitHub/tenv"

########################################################################
# Dataset structure ####################################################
########################################################################

# Input data
tripdata = None
with open(f"{root}/data/in/config_scenario/bulk_ny_2011.json") as js:
    tripdata = json.load(js)

region = tripdata["region"]

# Create and store graph name
graph_name = tripdata["region"].lower().replace(" ", "-").replace(",", "")

# Where dataset is saved
root_path = root + "/data/out/{}{}".format(label_exp, graph_name)

# -------------------------------------------------------------------- #
# Map ##################################################################
# -------------------------------------------------------------------- #

# Transportation network (.graphml and .svg)
root_map = root_path + "/map"

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
path_dist_matrix = "{}/distance_matrix_m_{}.csv".format(root_dist, graph_name)
# Distance dictionary (meters)
path_dist_dic = "{}/distance_dic_m_{}.npy".format(root_dist, graph_name)


########################################################################
# Reachability #########################################################
########################################################################

# Reachability layers
# (e.g., reachable in 30, 60, ..., total_range steps)
step = 15
total_range = 600
# If defined, step and total_range are assumed to be seconds
speed_km_h = 20
round_trip = False
# step_list = [60, 120, 300]
step_list = []

# Max travel time (seconds) to traverse an edge, i.e., if = 30, every
# edge can be traveled in 30 seconds
max_travel_time_edge = 30

# Max. time to execute ilp (min)
ilp_time_limit = 60

round_trip_label = "_roundtrip" if round_trip else ""

root_reachability = root_map + "/reachability{}_{}_{}{}".format(
    round_trip_label,
    step,
    total_range,
    ("_kmh{}".format(speed_km_h) if speed_km_h else ""),
)

root_reachability_concentric = (
    root_map
    + "/reachability_concentric{}_{}_{}{}".format(
        round_trip_label,
        step,
        total_range,
        ("_kmh{}".format(speed_km_h) if speed_km_h else ""),
    )
)

root_img_regions = root_reachability + "/img_region_centers"

root_img_regions_concentric = (
    root_reachability_concentric + "/img_region_centers_concentric"
)

root_img_neighbors = root_reachability + "/img_region_center_neighbors"

root_img_neighbors_concentric = (
    root_reachability_concentric + "/img_region_center_neighbors"
)

# Reachability dictionary {o:{max_dist:[d1, d2, d3]}
path_reachability_dic = "{}/reachability_{}.npy".format(
    root_reachability, graph_name
)

path_reachability_dic_concentric = "{}/reachability_{}.npy".format(
    root_reachability_concentric, graph_name
)

# Region centers dictionary {max_dist:[c1, c2, c3, c4, c5]}
path_region_centers = "{}/region_centers_{}.npy".format(
    root_reachability, graph_name
)

# Region centers dictionary {max_dist:[c1, c2, c3, c4, c5]}
path_region_centers_concentric = "{}/region_centers_{}.npy".format(
    root_reachability_concentric, graph_name
)

path_region_center_ids = "{}/region_center_ids_{}.npy".format(
    root_reachability, graph_name
)

path_region_center_concentric_ids = "{}/region_center_ids_{}.npy".format(
    root_reachability_concentric, graph_name
)

path_sorted_neighbors = "{}/sorted_neighbors_region_centers_{}.npy".format(
    root_reachability, graph_name
)

path_sorted_neighbors_concentric = "{}/sorted_neighbors_region_centers_{}.npy".format(
    root_reachability_concentric, graph_name
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
        f"\n#     Network = # nodes = {len(G.nodes())} "
        f"({min(G.nodes())} -> {max(G.nodes())}) - #edges = {len(G.edges())}"
        "\n###############################################################"
    )

