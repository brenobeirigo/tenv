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
# label_exp = "GIANT"
# label_exp = "N"
short_path = False
# root = os.getcwd().replace("\\", "/")
# root = "C:/Users/LocalAdmin/OneDrive/leap_forward/street_network_server/tenv"
# root = "d:/bb/tenv"
# root = "C:/Users/breno/Documents/phd/tenv"
# root = "C:/Users/LocalAdmin/Documents/GitHub/tenv"

with open("case_study_info.json") as js:
    data_paths = json.load(js)
    root = data_paths["root"]
    case_study = data_paths["case_study"]

scenario_path = f"{root}/data/in/config_scenario"
case_study_path = f"{scenario_path}/{case_study}"


########################################################################
# Dataset structure ####################################################
########################################################################

# Input data
mapdata = None
tripdata = None
data_gen = None
with open(case_study_path) as js:
    data = json.load(js)
    mapdata = data.get("mapdata", {})
    tripdata = data.get("tripdata", {})
    data_gen = data.get("data_gen", {})

region = mapdata["region"]

# If defined, step and total_range are assumed to be seconds
speed_km_h = mapdata["speed_km_h"]

# Create and store graph name
graph_name = (
    mapdata["label"]
    if "label" in mapdata
    else mapdata["region"].lower().replace(" ", "-").replace(",", "")
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

# .csv network information (centers, neighbors, etc)
root_network_info = root_path + "/network_info"

# Node id, lon, lat
path_node_info_csv = "{}/nodeset_gps.csv".format(root_network_info)

path_adjacency_matrix = "{}/adjacency_matrix.csv".format(root_network_info)

path_network_data = "{}/network_data.csv".format(root_network_info)

path_nodeset_gps_json = "{}/nodeset_gps.json".format(root_network_info)

path_node_info_json = "{}/nodeset_info.json".format(root_network_info)

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
root_tripdata_generated_ids = root_tripdata + "/gen"

path_tripdata_ids = None
tripdata_filename = None
path_tripdata_source = None
path_tripdata = None
path_tripdata_clone = None

# Path of trip data with ids
if "url_tripdata" in tripdata:
    local = tripdata["url_tripdata"]

    # Max. distance to match nodes to coordinates
    max_dist_km = tripdata["max_dist_km"]

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

# Distance dictionary (meters)
path_dist_dict_duration = "{}/dist_dict_duration_s.npy".format(root_dist)

# Distance matrix duration (using speed)
path_dist_matrix_duration = "{}/dist_matrix_duration_s.csv".format(root_dist)

# Distance dictionary considering edge disttance is truncated delay
# considering speed
path_dist_matrix_duration_npy = "{}/dist_matrix_duration_s.npy".format(
    root_dist
)


########################################################################
# Reachability #########################################################
########################################################################

info_reachability = ""
root_reachability = None
if "reachability" in mapdata:

    # Round trip reachability
    round_trip = mapdata["reachability"].get("round_trip", False)

    # Maximum number of node neighbors queried by application
    max_neighbors = mapdata["reachability"].get("max_neighbors", None)

    # step_list = [0, 60, 300, 600]
    # step_list = [0, 150, 300, 600]
    step_list = mapdata["reachability"].get("step_list", [])

    step_list_concentric = mapdata["reachability"].get(
        "step_list_concentric", []
    )

    # Max. time to execute ilp (min)
    ilp_time_limit = mapdata["reachability"].get("ilp_time_limit_min", 60)

    round_trip_label = "_roundtrip" if round_trip else ""
    speed_label = "_kmh{}".format(speed_km_h) if speed_km_h else ""

    root_reachability = root_map + "/reachability_{}{}_centers_{}{}".format(
        region_slice, round_trip_label, len(step_list), speed_label,
    )

    # Reachability dictionary {o:{max_dist:[d1, d2, d3]}
    path_reachability_dic = "{}/reach_dict.npy".format(root_reachability)

    # Reachability dictionary round trip {o:{max_dist:[d1, d2, d3]}
    path_reachability_r_dic = "{}/reach_dict_round.npy".format(
        root_reachability
    )

    root_img_regions = root_reachability + "/img_region_centers"

    root_img_neighbors = root_reachability + "/img_region_center_neighbors"

    # Region centers dictionary {max_dist:[c1, c2, c3, c4, c5]}
    path_region_centers = "{}/region_centers.npy".format(root_reachability)

    path_region_center_ids = "{}/region_center_ids.npy".format(
        root_reachability
    )

    path_sorted_neighbors = "{}/sorted_neighbors_region_centers.npy".format(
        root_reachability
    )

    path_recheable_neighbors = "{}/sorted_reach_neighbors.npy".format(
        root_reachability
    )

    info_reachability = (
        f"\n#       Speed = {speed_km_h}"
        f"\n#       Steps = {step_list}"
        f"\n#       Round = {round_trip}"
    )

# Max travel time (seconds) to traverse an edge, i.e., if = 30, every
# edge can be traveled in 30 seconds
max_travel_time_edge = mapdata.get("max_travel_time_edge_sec", 30)

# Network
G = nw.load_network(graph_file_name, folder=root_map)


def info():
    return (
        "\n###############################################################"
        f"\n#      Region = {region}"
        f"\n# Aggregation = {region_slice}"
        f"{info_reachability}"
        f"\n#      Source = {root_path}"
        f"\n#     Network = #nodes = {(len(G.nodes()) if G else '')} - "
        f"#edges = {len(G.edges()) if G else ''}"
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

    if not os.path.exists(root_network_info):
        os.makedirs(root_network_info)

    if "reachability" in mapdata:

        if not os.path.exists(root_reachability):
            os.makedirs(root_reachability)

        # Plots ###################################################### #
        if not os.path.exists(root_img_regions):
            os.makedirs(root_img_regions)

        if not os.path.exists(root_img_neighbors):
            os.makedirs(root_img_neighbors)

    # Trip data ###################################################### #

    if not os.path.exists(root_tripdata):
        os.makedirs(root_tripdata)

    if not os.path.exists(root_tripdata_raw):
        os.makedirs(root_tripdata_raw)

    if not os.path.exists(root_tripdata_cleaned):
        os.makedirs(root_tripdata_cleaned)

    if not os.path.exists(root_tripdata_ids):
        os.makedirs(root_tripdata_ids)

    if not os.path.exists(root_tripdata_generated_ids):
        os.makedirs(root_tripdata_generated_ids)
