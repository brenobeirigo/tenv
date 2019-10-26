import functools
import json
import sys
import os
from pprint import pprint
from bisect import bisect_right, bisect_left
import copy
from collections import defaultdict
import pandas as pd

# Adding project folder
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import tenv.config as config
import tenv.network as nw
import tenv.demand as tp
import numpy as np
import math


print(config.info())

# Network
G = nw.load_network(config.graph_file_name, folder=config.root_map)

distance_matrix = nw.get_distance_matrix(config.path_dist_matrix_npy, G)

# Reachability dictionary
reachability_dict, steps = nw.get_reachability_dic(
    config.path_reachability_dic,
    distance_matrix,
    step=config.step,
    total_range=config.total_range,
    speed_km_h=config.speed_km_h,
    step_list=config.step_list,
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
        distance_matrix,
        region_centers,
        path_sorted_neighbors=config.path_sorted_neighbors,
    )

elif config.region_slice == config.REGION_CONCENTRIC:

    region_id_dict, region_centers = nw.concentric_regions(
        G,
        config.step_list_concentric,
        reachability_dict,
        list(G.nodes()),
        center=-1,
        root_reachability=config.root_reachability_concentric,
    )

    sorted_neighbors = nw.get_sorted_neighbors(
        distance_matrix,
        region_centers,
        path_sorted_neighbors=config.path_sorted_neighbors_concentric,
    )

node_region_ids = nw.get_node_region_ids(G, region_id_dict)
center_nodes = nw.get_center_nodes(region_id_dict)

# Saving number of centers per maximal delay
maximal_dist_center_count = {
    m: len(centers)
    for m, centers in center_nodes.items()
}
maximal_dist_center_count[0] = len(G.nodes())

dists, centers = list(zip(*(maximal_dist_center_count.items())))

df = pd.DataFrame({'Maximal delay (s)': dists, "#Centers": centers})
df.sort_values(by=['Maximal delay (s)'], inplace=True)
df.to_csv(config.root_map + "/center_count.csv", index=False)

if config.step_list:
    # ##### Discarding centers to save memory ######################## #
    # A higher number of centers might have been processed before (e.g.,
    # every 15 seconds) but end up not being used in the end. Hence,
    # their previously loaded information become superfluous and can be
    # excluded.
    superfluous = set(region_centers.keys()).difference(config.step_list)
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

    lean_sorted_neighbors = defaultdict(lambda: defaultdict(list))
    for c, n_neighbors in sorted_neighbors.items():
        for n, neighbors in n_neighbors.items():
            lean_sorted_neighbors[c][n] = [
                i for i, d in neighbors[1 : config.max_neighbors + 1]
            ]

    sorted_neighbors = lean_sorted_neighbors

    # Adding immediate forward neighbors (level 0)
    for center_id in G.nodes:
        node_neighbors = nw.node_access(G, center_id, degree=1)

        # Node is not its own neighbor
        node_neighbors.discard(center_id)

        # Sort neighbors by distance
        node_neighbors = list(node_neighbors)
        node_neighbors.sort(key=lambda x: nw.get_distance(G, center_id, x))
        sorted_neighbors[0][center_id] = node_neighbors[: config.max_neighbors]

    # Client application uses distances in kilometers
    distance_matrix = distance_matrix / 1000


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

    list_coords, cum_duration, cum_distance, dist_m = nw.get_intermediate_coords(
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

    list_coords, cum_duration, cum_distance, dist_m = nw.get_intermediate_coords(
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
            {"id": id, "x": G.node[id]["x"], "y": G.node[id]["y"]}
            for id in G.nodes()
        ]

    else:
        nodes = [
            {"id": id, "x": x, "y": y}
            for id, x, y in [
                (
                    id,
                    *nw.wgs84_to_web_mercator(
                        G.node[id]["x"], G.node[id]["y"]
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
    output = {"nodes":[{"id":1360,"xpath_region_ids7}...]}

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

    return {"location": {"x": G.node[id]["x"], "y": G.node[id]["y"]}}


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

    return (G.node[id]["x"], G.node[id]["y"])
