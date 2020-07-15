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

if G is None:
    print("Creating data network data.")

    # Create all folders
    config.make_folders()

    print(
        "\n>>>>> Target folders:\n"
        + f"\n - Distance matrix (csv) and dictionary (npy): {config.root_dist}"
        + f"\n -   Data excerpt from NYC taxi dataset (csv): {config.root_tripdata}"
        + (
            f"\n -  Reachability (npy) & region centers (npy): {config.root_reachability}.\n"
            if config.root_reachability is not None
            else ""
        )
    )

    print(
        "\n############################"
        "##### Loading network ######"
        "############################"
    )

    # Get network graph and save
    G = nw.get_network_from(
        config.region,
        config.root_map,
        config.graph_name,
        config.graph_file_name,
        max_travel_time_edge=config.max_travel_time_edge,
        speed_km_h=config.speed_km_h,
    )

    nw.save_graph_pic(G, config.root_map)


time_dict["graph"] = time.time() - t_start

if os.path.exists(config.root_lean):
    print("Loading preprocessed data...")
    region_centers = np.load(
        f"{config.root_lean}region_centers.npy", allow_pickle=True
    ).item()
    region_id_dict = np.load(
        f"{config.root_lean}region_id_dict.npy", allow_pickle=True
    ).item()
    sorted_neighbors = np.load(
        f"{config.root_lean}sorted_neighbors.npy", allow_pickle=True
    ).item()
    node_region_ids = np.load(
        f"{config.root_lean}node_region_ids.npy", allow_pickle=True
    ).item()
    node_delay_center_id = np.load(
        f"{config.root_lean}node_delay_center_id.npy", allow_pickle=True
    ).item()
    center_nodes = np.load(
        f"{config.root_lean}center_nodes.npy", allow_pickle=True
    ).item()
    distance_matrix = np.load(f"{config.root_lean}distance_matrix_km.npy")

    reachability_dict, steps = nw.get_reachability_dic(
        config.path_reachability_dic,
        distance_matrix,
        speed_km_h=config.speed_km_h,
        step_list=config.step_list,
    )

else:
    print("Precalculated data couldn't be loaded!")
    config.make_folders()

    t_start = time.time()

    # Creating distance dictionary [o][d] -> distance
    distance_dic = nw.get_distance_dic(config.path_dist_dic, G)

    # Creating distance matrix from dictionary
    distance_matrix = nw.get_distance_matrix(
        config.path_dist_matrix_npy, G, distance_dic_m=distance_dic
    )

    # Create .csv distance matrix
    distance_matrix_df = nw.get_distance_matrix_df(
        config.path_dist_matrix, distance_matrix
    )

    time_dict["distance_matrix"] = time.time() - t_start

    if config.root_reachability:
        # Inbound reachability dictionary: Which nodes can access node n?
        t_start = time.time()
        reachability_dict, steps = nw.get_reachability_dic(
            config.path_reachability_dic,
            distance_matrix,
            speed_km_h=config.speed_km_h,
            step_list=config.step_list,
        )
        time_dict["reachability"] = time.time() - t_start

        # Outbound reachability dictionary: Which nodes node n can access?
        t_start = time.time()
        reachability_r_dict, steps = nw.get_reachability_dic(
            config.path_reachability_r_dic,
            distance_matrix,
            speed_km_h=config.speed_km_h,
            step_list=config.step_list,
            roundtrip=True,
        )
        time_dict["reachability_r"] = time.time() - t_start

        if config.region_slice == config.REGION_REGULAR:
            # All region centers
            t_start = time.time()

            region_centers = nw.get_region_centers(
                steps,
                config.path_region_centers,
                reachability_dict,
                list(G.nodes()),
                root_path=config.root_reachability,
                round_trip=config.round_trip,
            )
            time_dict["region_centers_REG"] = time.time() - t_start

            # What is the closest region center of every node (given a time limit)?
            t_start = time.time()
            region_id_dict = nw.get_region_ids(
                G,
                reachability_dict,
                region_centers,
                path_region_ids=config.path_region_center_ids,
            )
            time_dict["region_id_REG"] = time.time() - t_start

        elif config.region_slice == config.REGION_CONCENTRIC:

            t_start = time.time()
            region_id_dict, region_centers = nw.concentric_regions(
                G,
                config.step_list_concentric,
                reachability_dict,
                list(G.nodes()),
                center=-1,
                root_reachability=config.root_reachability,
            )
            time_dict["region_id_CON"] = time.time() - t_start

        t_start = time.time()
        sorted_neighbors = nw.get_sorted_neighbors(
            distance_matrix,
            region_centers,
            path_sorted_neighbors=config.path_sorted_neighbors,
        )

        time_dict["sorted_neighbors"] = time.time() - t_start

        t_start = time.time()
        node_region_ids = nw.get_node_region_ids(G, region_id_dict)
        time_dict["node_region_ids"] = time.time() - t_start

        t_start = time.time()
        node_delay_center_id = nw.get_node_delay_center_id(G, region_id_dict)
        time_dict["node_delay_center_id"] = time.time() - t_start

        t_start = time.time()
        center_nodes = nw.get_center_nodes(region_id_dict)
        time_dict["center_nodes"] = time.time() - t_start

        t_start = time.time()
        nodeset_json = nw.save_nodeset_json(
            config.path_node_info_json, G, region_id_dict
        )
        time_dict["nodeset_data_json"] = time.time() - t_start

        if config.step_list:
            # ##### Discarding centers to save memory ################ #
            # A higher number of centers might have been processed
            # before (e.g., every 15 seconds) but end up not being used
            # in the end. Hence, their previously loaded information
            # become superfluous and can be excluded.
            t_start = time.time()
            superfluous = set(region_centers.keys()).difference(
                config.step_list
            )
            print(f"Removing superfluous centers {superfluous}.")
            for c in superfluous:
                del region_centers[c]
                for n in region_id_dict:
                    del region_id_dict[n][c]
                del sorted_neighbors[c]
                del node_region_ids[c]
                del center_nodes[c]
                for n in reachability_dict:
                    try:
                        del reachability_dict[n][c]
                    except:
                        pass

            time_dict["superfluous"] = time.time() - t_start

            t_start = time.time()
            lean_sorted_neighbors = dict()
            for c, n_neighbors in sorted_neighbors.items():
                lean_sorted_neighbors[c] = dict()
                for n, neighbors in n_neighbors.items():
                    lean_sorted_neighbors[c][n] = [
                        i for i, d in neighbors[1 : config.max_neighbors + 1]
                    ]
            time_dict["lean_sorted_neighbors"] = time.time() - t_start

            t_start = time.time()
            sorted_neighbors = lean_sorted_neighbors

            # Adding immediate forward neighbors (level 0)
            sorted_neighbors[0] = dict()
            for center_id in G.nodes:
                node_neighbors = nw.node_access(G, center_id, degree=1)

                # Node is not its own neighbor
                node_neighbors.discard(center_id)

                # Sort neighbors by distance
                node_neighbors = list(node_neighbors)
                node_neighbors.sort(
                    key=lambda x: nw.get_distance(G, center_id, x)
                )
                sorted_neighbors[0][center_id] = node_neighbors[
                    : config.max_neighbors
                ]

            time_dict["sorted_neighbors"] = time.time() - t_start

            # Client application uses distances in kilometers
            t_start = time.time()
            distance_matrix = distance_matrix / 1000
            time_dict["distance_matrix_1000"] = time.time() - t_start

            os.makedirs(config.root_lean)

            np.save(f"{config.root_lean}region_centers.npy", region_centers)
            np.save(f"{config.root_lean}region_id_dict.npy", region_id_dict)
            np.save(
                f"{config.root_lean}sorted_neighbors.npy", sorted_neighbors
            )
            np.save(f"{config.root_lean}node_region_ids.npy", node_region_ids)
            np.save(
                f"{config.root_lean}node_delay_center_id.npy",
                node_delay_center_id,
            )
            np.save(f"{config.root_lean}center_nodes.npy", center_nodes)
            np.save(
                f"{config.root_lean}distance_matrix_km.npy", distance_matrix
            )

            print(center_nodes.keys())

    t_start = time.time()
    nodes_df = nw.save_node_info_csv(G, config.path_node_info_csv)
    nodes_dict = nw.save_nodeset_gps_json(G, config.path_nodeset_gps_json)
    time_dict["network_nodes"] = time.time() - t_start

    t_start = time.time()
    adjacency_matrix = nw.save_adjacency_matrix(
        G, config.path_adjacency_matrix
    )
    time_dict["adjacency_matrix"] = time.time() - t_start

    ## DURATION DATA ###################################################
    # Created to solve approximation errors found when comparing:
    # 1 - Sum of shortest path distances from o to d
    # 2 - Total distance between o and d
    # Convert all distances to integer seconds (considering speed), and
    # run dijkstra shortest paths considering these durations, to
    # guarantee there are no fractional data.
    t_start = time.time()
    G_duration = nw.add_duration_to_graph(G, config.speed_km_h)
    time_dict["add_duration_graph"] = time.time() - t_start

    t_start = time.time()
    df_G = nw.get_network_data(config.path_network_data, G_duration)
    time_dict["get_network_data"] = time.time() - t_start

    t_start = time.time()
    dist_matrix_duration_dict = nw.get_distance_dic(
        config.path_dist_dict_duration, G_duration, weight="duration"
    )
    time_dict["dist_matrix_duration_dict"] = time.time() - t_start

    # Creating distance matrix from dictionary
    t_start = time.time()
    dist_matrix_duration = nw.get_distance_matrix(
        config.path_dist_matrix_duration_npy,
        G_duration,
        distance_dic_m=dist_matrix_duration_dict,
    )
    time_dict["dist_matrix_duration_npy"] = time.time() - t_start

    t_start = time.time()
    dist_matrix_duration_df = nw.get_distance_matrix_df(
        config.path_dist_matrix_duration,
        dist_matrix_duration,
        float_format="%.0f",
    )

    time_dict["dist_matrix_duration_csv"] = time.time() - t_start

    # Trip data is saved in external drive
    if config.tripdata:

        t_start = time.time()

        logging.info(
            "\n############################"
            "## Processing NYC trip data ##"
            "############################"
        )

        print("NYC trip data generation settings:")
        pprint(config.data_gen)

        tp.process_tripdata(config, G, distance_dic)
        time_dict["process_tripdata"] = time.time() - t_start

    if config.data_gen:

        t_start = time.time()

        logging.info(
            "\n############################"
            "## Generating random trip data ##"
            "############################"
        )

        print("Trip data generation settings:")
        pprint(config.data_gen)

        tp.gen_random_data(config, G, distance_dic)

        time_dict["generate_tripdata"] = time.time() - t_start

pprint(time_dict)


@functools.lru_cache(maxsize=None)
def sp(o, d):
    """Shortest path between origin and destination (inclusive)

    Parameters
    ----------
    o : int
        Origin id
    d : int
        Destination id

    Returns
    -------
    str
        List of ids separated by ';'

    Example
    -------
    input = http://localhost:4999/sp/1/900
    output = 1;2;67;800;900
    """
    return nw.get_sp(G, o, d)


@functools.lru_cache(maxsize=None)
def sp_coords(o, d):
    """Shortest path between origin and destination (inclusive)

    Parameters
    ----------
    o : int
        Origin id
    d : int
        Destination id

    Returns
    -------
    str
        List of ids separated by ';'

    Example
    -------
    input = http://localhost:4999/sp_coords/1/3
    output = 1;2;67;800;900
    """
    return nw.get_sp_coords(G, o, d)


def get_distance(o, d):
    return distance_matrix[o][d]


def get_distance_sec(o, d):
    dist_s = int(3600 * distance_matrix[o][d] / config.speed_km_h + 0.5)
    return dist_s


@functools.lru_cache(maxsize=None)
def sp_json(o, d, projection="GPS"):
    """Shortest path between origin and destination (inclusive)

    Parameters
    ----------
    o : int
        Origin id
    d : int
        Destination id

    Returns
    -------
    str
        List of ids separated by ';'

    Example
    -------
    input = http://localhost:4999/sp_coords/1/3
    output = 1;2;67;800;900
    """
    return nw.get_sp_coords(G, o, d, projection=projection)


def get_node_count():
    return len(G.nodes())


def get_info():
    """Return network info"""
    info = {
        "region": config.region,
        "label": config.graph_name,
        "node_count": len(G.nodes()),
        "edge_count": len(G.edges()),
        "centers": {
            dist: len(center_ids) for dist, center_ids in center_nodes.items()
        },
        "region_type": config.region_slice,
    }

    return info


@functools.lru_cache(maxsize=None)
def can_reach(n, t):
    """Return list of nodes that can reach node n in t seconds.

    Parameters
    ----------
    n : int
        Node id
    t : int
        Time in seconds

    Returns
    -------
    str
        List of nodes that can reach n in t seconds (separated by ";")

    Example
    -------
    input = http://localhost:4999/can_reach/1/30
    output = 0;1;3720;3721;4112;3152;3092;1754;1309;928;929;1572;3623;
        3624;169;1897;1901;751;1841;308
    """

    return nw.get_can_reach_set(n, reachability_dict, t)


@functools.lru_cache(maxsize=None)
def reachable_neighbors(n, t):
    """Return list of nodes that node n can reach in t seconds.

    Parameters
    ----------
    n : int
        Node id
    t : int
        Time in seconds

    Returns
    -------
    set
        Nodes that n can reach in t seconds (inclusive)
    """
    return nw.get_can_reach_set(n, reachability_r_dict, t)


@functools.lru_cache(maxsize=None)
def reachable_neighbors_l(n, t, limit):
    """Return list of nodes that can reach node n in t seconds.

    Parameters
    ----------
    n : int
        Node id
    t : int
        Time in seconds

    Returns
    -------
    set
        Nodes that n can reach in t seconds (inclusive)
    """
    return nw.get_can_reach_set(n, reachability_r_dict, t)


@functools.lru_cache(maxsize=None)
def sp_sliced(o, d, waypoint, total_points, step_count, projection="GPS"):
    """Return "total_points" coordinates between origin and destination.

    Break coordinates acoording to "step_duration"

    Parameters
    ----------
    o : int
        Origin id
    d : int
        Destination id
    total_points : int
        Number of coordinates between origin and destination (inclusive)
    step_duration : int
        Time steps in seconds to break the coordinates
    projection : str, optional
        Coordinate projection (MERCATOR or GPS), by default "GPS"

    Returns
    -------
    json file
        {
            sp=[[[p1,p2],[p3,p4]], [[p4,p5],[p6,p7]]],
            step_count = 2,
            len = 7,
            duration = ?,
            distance = ?

        }
    """

    (
        list_coords,
        cum_duration,
        cum_distance,
        dist_m,
    ) = nw.get_intermediate_coords(
        G, o, d, total_points, projection=projection, waypoint=waypoint
    )

    if step_count == 0:
        print(list_coords)
    step = total_points // step_count

    list_sliced = [
        list_coords[i : i + step] for i in np.arange(0, total_points, step)
    ]

    return list_sliced


@functools.lru_cache(maxsize=None)
def sp_segmented(
    o, d, waypoint, total_points, step_duration, projection="GPS"
):
    """Return "total_points" coordinates between origin and destination.

    Break coordinates acoording to "step_duration"

    
    Parameters
    ----------
    o : int
        Origin id
    d : int
        Destination id
    total_points : int
        Number of coordinates between origin and destination (inclusive)
    step_duration : int
        Time steps in seconds to break the coordinates
    projection : str, optional
        Coordinate projection (MERCATOR or GPS), by default "GPS"

    Returns
    -------
    json file
        {
            sp=[[[p1,p2],[p3,p4]], [[p4,p5],[p6,p7]]],
            step_count = 2,
            len = 7,
            duration = ?,
            distance = ?

        }
    """

    (
        list_coords,
        cum_duration,
        cum_distance,
        dist_m,
    ) = nw.get_intermediate_coords(
        G, o, d, total_points, projection=projection, waypoint=waypoint
    )

    step_coords = []
    step_duration_cum = 0
    lo_idx = 0

    # Segmenting list of coords
    while True:

        step_duration_cum += step_duration

        hi_idx = bisect_left(cum_duration, step_duration_cum, lo=lo_idx)

        # Happens when list_coords has a single point
        if hi_idx <= 0:
            step_coords.append(list_coords)
            break

        # Slice coordinate list
        step_coords.append(list_coords[lo_idx:hi_idx])

        # Update lower id to slice coordinate list
        lo_idx = hi_idx - 1

        # List of coords has ended
        if hi_idx == len(list_coords):
            break

    return {
        "len": len(list_coords),
        "duration": nw.get_duration(dist_m),
        "distance": dist_m,
        "step_count": len(step_coords),
        "sp": step_coords,
    }


@functools.lru_cache(maxsize=None)
def nodes(projection):
    """Get all network nodes (id, longitude, latitude)

    Returns
    -------
    json
        Json file with list of nodes [{id, x, y}]

    Example
    -------
    input = http://localhost:4999/nodes
    output = {"nodes":[{"id":1360,"xpath_region_ids7}...]}

    """
    if projection == "GPS":
        nodes = [
            {"id": id, "x": G.nodes[id]["x"], "y": G.nodes[id]["y"]}
            for id in G.nodes()
        ]

    else:
        nodes = [
            {"id": id, "x": x, "y": y}
            for id, x, y in [
                (
                    id,
                    *nw.wgs84_to_web_mercator(
                        G.nodes[id]["x"], G.nodes[id]["y"]
                    ),
                )
                for id in G.nodes()
            ]
        ]

    dic = dict(nodes=nodes)
    return dic


@functools.lru_cache(maxsize=None)
def get_center_elements(max_dist, center):
    """Get all node ids reachable from center node.
    Assume center node id belongs to centers calculated for
    max dist.
    Returns
    -------
    str
        Node ids considering time limit (separated by ;)

    Example
    -------
    input = http://localhost:4999/nodes
    output = {"nodes":[{"id":1360,"xpath_region_ids7}...]}

    """
    try:
        nodes = center_nodes[max_dist][center]
        # print("NODES:", nodes)
        return nodes
    except Exception as e:
        print(
            f"ERROR({e})!\n (center={center}, max_dist={max_dist}) does "
            " not exist!"
        )
        return


@functools.lru_cache(maxsize=None)
def level_nodes(time_limit):
    """Get all network nodes (id, longitude, latitude)

    Returns
    -------
    str
        Node ids considering time limit (separated by ;)

    Example
    -------
    input = http://localhost:4999/nodes
    output = {"nodes":[{"id":13roundtripon_ids7}...]}

    """
    nodes = [region_id_dict[node_id][time_limit] for node_id in G.nodes()]
    return nodes


@functools.lru_cache(maxsize=None)
def get_node_region_ids():
    """Get list of node ids for each region (defined with maximum
    reachable time)

    Returns
    -------
    dict
        Dictionary of maximum reachable time keys and node id lists.
    """

    return dict(node_region_ids)


@functools.lru_cache(maxsize=None)
def get_node_region_count():
    """Get count of single ids for each maximum reachable time

    Returns
    -------
    dict
        Dictionary of maximum reachable time keys and counts.
    """

    return {k: len(set(n)) for k, n in node_region_ids.items()}


@functools.lru_cache(maxsize=None)
def get_node_region_ids_step(step):
    """Get list of node ids for each region (defined with minimum
    reachable time)

    Returns
    -------
    dict
        Dictionary of max. reachable time keys and node id lists.
    """
    cut_node_region_ids = copy.deepcopy(node_region_ids)
    min_reachable_time = list(cut_node_region_ids.keys())

    # When no levels are defined, region corresponds to all nodes
    if step == 0:
        return {0: cut_node_region_ids[0]}

    # Removing distances which are not multiples of "step"
    for k in min_reachable_time:
        if k % step != 0:
            del cut_node_region_ids[k]

    return dict(cut_node_region_ids)


@functools.lru_cache(maxsize=None)
def neighbors(node, degree, direction):
    """Get node neighbors within degree levels

    Parameters
    ----------
    node : int
        Node id
    degree : int
        Hops or levels jumped from node
    direction : str
        forward (reachable from node) or backwards (can reach node)

    Returns
    -------
    str
        List of reachable nodes separated by ';'

    Example
    -------
    input = http://localhost:4999/neighbors/0/1/backward
    output = 0;442;475;420
    """
    node_neighbors = nw.node_access(
        G, node, degree=degree, direction=direction
    )
    node_neighbors.remove(node)

    return node_neighbors


@functools.lru_cache(maxsize=None)
def get_centers(time_limit):
    """Region centers considering time limit

    Parameters
    ----------
    time_limit : int
        All points can be reachable from centers within time limit

    Returns
    -------
    str
        Region center ids

    Examples
    --------
    input = http://localhost:4999/centers/360
    output = 1042;1077;1097;1117;1854
    """

    return region_centers[time_limit]


@functools.lru_cache(maxsize=None)
def get_region_id(time_limit, node_id):
    """Get closest region id that can access node (within time limit)

    Parameters
    ----------
    time_limit : int
        Maximum time limit
    node_id : int
        Node id at lowest level

    Returns
    -------
    int
        Region id of node
    """

    return region_id_dict[node_id][time_limit]


@functools.lru_cache(maxsize=None)
def get_center_neighbors(time_limit, center_id, n_neighbors):
    return sorted_neighbors[time_limit][center_id][:n_neighbors]


@functools.lru_cache(maxsize=None)
def get_center_neighbors2(time_limit, center_id, n_neighbors):
    """Get the closest 'n_neighbors' neighbors from region center.

    Parameters
    ----------
    time_limit : int
        Max. distance driving creation of region centers
    center_id : int
        Region center id
    n_neighbors : int
        Max. number of neighbors

    Returns
    -------
    int
        Region center neighbors

    Example
    -------
        input = http://localhost:4999/center_neighbors/120/74/4
        output = 2061;1125;2034;968
    """

    # When time limit is zero, return immediate neighbors
    if time_limit == 0:
        node_neighbors = nw.node_access(G, center_id, degree=1)

        # Node is not its own neighbor
        node_neighbors.discard(center_id)

        # Sort neighbors by distance
        node_neighbors = list(node_neighbors)
        node_neighbors.sort(key=lambda x: nw.get_distance(G, center_id, x))

    else:
        node_neighbors = [
            neighbor_id
            for neighbor_id, distance in sorted_neighbors[time_limit][
                center_id
            ]
            if distance != 0
        ]

    # Restrict set of neighbors and return
    return node_neighbors[:n_neighbors]


@functools.lru_cache(maxsize=None)
def location(id):
    """Return location (lon, lat) of point with node id

    Parameters
    ----------
    id : int
        Node id

    Returns
    -------
    tuple
        Longitude and latitude

    Example
    -------
    http://localhost:4999/location/1

    """

    return {"location": {"x": G.nodes[id]["x"], "y": G.nodes[id]["y"]}}


@functools.lru_cache(maxsize=None)
def lonlat(id):
    """Return location (lon, lat) of point with node id

    Parameters
    ----------
    id : int
        Node id

    Returns
    -------
    tuple
        Longitude and latitude

    Example
    -------
    http://localhost:4999/location/1

    """

    return (G.nodes[id]["x"], G.nodes[id]["y"])
