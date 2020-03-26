import os
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import bisect
from collections import defaultdict
import functools
import random
from gurobipy import Model, GurobiError, GRB, quicksum
import math
import logging
import traceback

from shapely.geometry import Point, LineString
from copy import deepcopy
from pprint import pprint

np.set_printoptions(precision=2)


def remove_edges(G_old, sample_size=3):
    """Shrink graph file size (and complexity) by removing internal
    nodes that integrate segments.

    Parameters
    ----------
    G_old : Networkx
        Old street network (edge geometry featuring several nodes)
    sample_size : int, optional
        Max. number of nodes per edge (apart from first and last),
        by default 3
    
    Returns
    -------
    Networkx
        Smaller graph with fewer edge nodes.
    """

    G = deepcopy(G_old)

    for o, d in G_old.edges():

        # Get all edge attributes and broadcast them to all sub edges
        edge_attr_od = G.edges[o, d, 0]
        sp = edge_attr_od.get("geometry", LineString([]))

        try:
            sp_list = list(sp.coords)
            # First node
            a = sp_list.pop(0)
            # Last node
            b = sp_list.pop()
            # Remaining nodes
            m = list(sp.coords)

            # Select random sample of indexes and sort unique
            sam = random.sample(
                list(np.arange(len(m))), min(len(m), sample_size)
            )
            indexes = sorted(set(sam))

            # Get the points corresponding to the indexes
            values = [m[v] for v in indexes]

            # Set up the new edge (fewer nodes)
            coords = [a] + values + [b]
            G.edges[o, d, 0]["geometry"] = LineString(coords)

        except Exception as e:
            logging.info(f"Error! Can't remove node. Exception:'{e}'.")
            pass
            # print(f"{e}-{sp_list}{a}{b}")
    return G


# #################################################################### #
# Create, load, save network ######################################### #
# #################################################################### #


def node_access(G, node, degree=1, direction="forward"):
    """
    Return the set of nodes which lead to "node" (direction = backaward)
    or the set o nodes which can be accessed from "node"
    (direction = forward)

    Parameters:
        G         - Networkx muldigraph
        node      - Node whose accessibility will be tested
        degree    - Number of hops (backwards or forwards)
        direction - Test forwards or backwards

    Return:
        set of backward/forward nodes
    """

    # Access can be forward or backwards
    func = G.successors if direction == "forward" else G.predecessors

    access_set = set()
    access = [node]
    access_set = access_set.union(access)

    for _ in range(0, degree):

        # Predecessors/successors i degrees away
        access_i = set()

        for j in access:
            access_i = access_i.union(set(func(j)))

        access = access_i
        access_set = access_set.union(access)

    return access_set


def is_reachable(G, node, degree):
    """Check if node can be accessed across a chain
    of "degree" nodes (backwards and frontward).

    This guarantees the node is not isolated since it is reachable and
    can reach others.

    Arguments:
        G {networkx} -- Graph that the node belongs too
        node {int} -- Id of node to test reachability
        degree {int} -- Minimum length of path

    Returns:
        boolean -- True, if node can be reached and reach others
    """
    pre = list(G.predecessors(node))
    suc = list(G.successors(node))
    neighbors = set(pre + suc)

    if node in neighbors:
        # if the node appears in its list of neighbors, it self-loops.
        # this is always an endpoint.
        return False

    if len(node_access(G, node, degree, direction="backward")) < degree:
        return False

    if len(node_access(G, node, 10, direction="forward")) < degree:
        return False

    return True

    # Save the equivalence between nodes


def load_network(filename, folder=None):
    """Load and return graph network.

    Parameters
    ----------
        filename {string} -- Name of network

    Keyword Arguments:
        folder {string} -- Target folder (default: {None})

    Returns:
        networkx or None -- The loaded network or None if not found
    """

    path = "{}/{}".format(folder, filename)
    logging.info(f"Loading {path}")

    # if file does not exist write header
    if not os.path.isfile("{}/{}".format(folder, filename)):
        logging.info("Network is not in '{}'".format(path))
        return None

    # Try to load graph
    return ox.load_graphml(filename=filename, folder=folder)


def download_network(region, network_type):
    """Download network from OSM representing the region.

    Arguments:

        region {string} -- Location.
            E.g., "Manhattan Island, New York City, New York, USA"
        network_type {string} -- Options: drive, drive_service, walk,
            bike, all, all_private

    Returns:
        networkx -- downloaded networkx
    """

    # Download graph
    G = ox.graph_from_place(region, network_type=network_type, simplify=True)

    return G


def clean_network(G):
    """Set of nodes with low connectivity (end points) must be 
    eliminated to avoid stuck vehicles (enter but cannot leave).
    
    Parameters
    ----------
    G : networkx
        Graph to be cleaned (e.g., isolated nodes)
    
    Returns
    -------
    networkx
        Cleaned graph
    """

    G = G.copy()

    G = ox.remove_isolated_nodes(G)

    not_reachable = set()

    for node in G.nodes():
        # Node must be accessible by at least 10 nodes
        # forward and backward
        # e.g.: 1--2--3--4--5 -- node --6--7--8--9--10
        if not is_reachable(G, node, 10):
            not_reachable.add(node)

        for target in G.neighbors(node):
            edge_data = G.get_edge_data(node, target)
            keys = len(edge_data.keys())
            try:
                for i in range(1, keys):
                    del edge_data[i]
            except:
                pass

    for node in not_reachable:
        G.remove_node(node)

    # Only the strongest connected component is kept
    disconnected = G.nodes() - get_largest_connected_component(G)
    G.remove_nodes_from(disconnected)

    return G


def get_graph_info(G):
    return "NODES: {} ({} -> {}) -- #EDGES: {}".format(
        len(G.nodes()), min(G.nodes()), max(G.nodes()), len(G.edges())
    )


def get_network_from(
    region,
    root_path,
    graph_name,
    graph_filename,
    max_travel_time_edge=None,
    speed_km_h=20,
    n_points_edges=3,
):
    """Download network from region. If exists (check graph_filename),
    try loading.

    Parameters
    ----------
    region : str
        Location. E.g., "Manhattan Island, New York City, New York, USA"
    root_path : str
        Path where graph is going to saved
    graph_name : str
        Name to be stored in graph structure
    graph_filename : str
        File name .graphml to be saved in root_path
    max_travel_time_edge : int, optional
        Add extra nodes to the network such that each edge can be
        traveled in at most max_travel_time_edge seconds, by default None
    speed_km_h : int, optional
        Used in tandem with max_travel_time_edge to enrich node set,
        by default 20
    n_points_edges : int, optional
        Max. number of nodes in each edge (excluding o and d), by default 3
    
    Returns
    -------
    networkx
        Processed graph (enriched, relabed nodes, and compressed)
    """
    # Street network
    G = load_network(graph_filename, folder=root_path)

    if G is None:
        # Try to download
        try:
            logging.info(f"# Downloading graph from '{region}'.")
            G = download_network(region, "drive")

            logging.info(
                "NODES: {} ({} -> {}) -- #EDGES: {}".format(
                    len(G.nodes()),
                    min(G.nodes()),
                    max(G.nodes()),
                    len(G.edges()),
                )
            )

            # Create and store graph name
            G.graph["name"] = graph_name

            # Save region name
            G.graph["region"] = region

            logging.info("# Downloaded graph")

            logging.info("# Cleaned graph")
            G = clean_network(G)
            logging.info(
                "#NODES: {} ({} -> {}) -- #EDGES: {}".format(
                    len(G.nodes()),
                    min(G.nodes()),
                    max(G.nodes()),
                    len(G.edges()),
                )
            )

            # Relabel nodes
            mapping = {k: i for i, k in enumerate(sorted(G.nodes()))}
            G = nx.relabel_nodes(G, mapping)

            # Add extra nodes to the network such that each edge can
            # be traveled in at most max_travel_time_edge seconds
            if max_travel_time_edge is not None:
                G = enrich_graph(
                    G,
                    max_travel_time_edge=max_travel_time_edge,
                    speed_km_h=speed_km_h,
                    n_coords=4000,
                )

                logging.info("# Enriched graph")
                logging.info(
                    "NODES: {} ({} -> {}) -- #EDGES: {}".format(
                        len(G.nodes()),
                        min(G.nodes()),
                        max(G.nodes()),
                        len(G.edges()),
                    )
                )

                G = clean_network(G)

                logging.info(
                    "NODES: {} ({} -> {}) -- #EDGES: {}".format(
                        len(G.nodes()),
                        min(G.nodes()),
                        max(G.nodes()),
                        len(G.edges()),
                    )
                )

            # Compress graph by removing nodes within edges
            logging.info(f"Removing internal points (max.:{n_points_edges})")
            G = remove_edges(G, sample_size=n_points_edges)

            # Save
            ox.save_graphml(G, filename=graph_filename, folder=root_path)

        except Exception as e:
            logging.info(f"Error loading graph: {e}")
            traceback.print_exc()

    logging.info(
        "\n# NETWORK -  NODES: {} ({} -> {}) -- #EDGES: {}".format(
            len(G.nodes()), min(G.nodes()), max(G.nodes()), len(G.edges())
        )
    )

    return G


def save_graph_pic(G, path, config=dict(), label=""):
    """Save a picture (svg) of graph G.

    Arguments:
        G {networkx} -- Working graph
    """
    default_attrib = dict(
        fig_height=15,
        node_size=0.5,
        edge_linewidth=0.3,
        save=True,
        show=False,
        file_format="svg",
        filename="{}/{}{}".format(path, label, G.graph["name"]),
    )

    default_attrib.update(config)

    fig, ax = ox.plot_graph(G, **default_attrib)


def get_sorted_neighbors(
    distances, region_centers, minimum_distance=0, path_sorted_neighbors=None
):
    neighbors = None
    if os.path.isfile(path_sorted_neighbors):
        neighbors = np.load(path_sorted_neighbors, allow_pickle=True).item()
        logging.info(
            f"\nReading region ids dictionary..."
            f"\nSource: '{path_sorted_neighbors}'."
        )
    else:
        logging.info(
            f"\nFinding closest node region center ids..."
            f"\nTarget: '{path_sorted_neighbors}'."
        )
        neighbors = dict()
        for t, centers in region_centers.items():
            # Minimum distances
            if t < minimum_distance:
                continue

            logging.info(f"{t:04} - {len(centers)}")
            neighbors[t] = dict()
            for c_o in centers:
                neighbors[t][c_o] = list()
                for c_d in centers:
                    neighbors[t][c_o].append((c_d, distances[c_o][c_d]))

                # Sort according to distance
                neighbors[t][c_o].sort(key=lambda tup: tup[1])

        np.save(path_sorted_neighbors, neighbors)

    return neighbors


def concentric_regions(
    G, steps, reachability, nodes, center=None, root_reachability=None
):
    """[summary]
    
    Parameters
    ----------
    G : [type]
        [description]
    steps : List
        Steps are sorted in descending order
    reachability : [type]
        [description]
    nodes : [type]
        [description]
    path_region_centers : [type], optional
        [description], by default None
    root_reachability : [type], optional
        [description], by default None
    """

    # Creating region centers for all max. travel durations
    # in reachability dictionary

    # All steps were processed
    node_dist_center = defaultdict(lambda: defaultdict(int))
    region_centers = dict()
    if not steps:
        return node_dist_center, region_centers

    # Pop the largest step (copy to guarantee step list is not modified)
    sorted_steps = sorted(steps)
    s = sorted_steps.pop()

    # logging.info(f"\n## Processing distance {s}")

    # Region centers with step s
    region_centers = get_region_centers(
        [s],
        f"{root_reachability}/region_centers_{s:04}_{center:04}.npy",
        reachability,
        nodes,
        root_path=root_reachability,
        round_trip=False,
        parent_center=center,
    )

    # Associate each node within the area determined by the parent
    # region center with childreen region centers. Return dictionary
    # with structure: node -> distance -> center
    region_id_dict = get_region_ids(
        G,
        reachability,
        region_centers,
        nodes=nodes,
        path_region_ids=(
            f"{root_reachability}/" f"region_center_ids_{s:04}_{center:04}.npy"
        ),
    )

    # distance -> center -> nodes
    dist_center_nodes_dict = get_center_nodes(region_id_dict)

    # Get region centers within previously defined regions
    for c, center_nodes in dist_center_nodes_dict[s].items():

        # Update center ids for nodes in distance s
        for n in center_nodes:
            node_dist_center[n][s] = c

        # logging.info(f" -- Center {c:>4} = {center_nodes}")

        # Get region centers considering only c nodes
        node_dist_center_sub, sub_centers = concentric_regions(
            G,
            sorted_steps,
            reachability,
            center_nodes,
            center=c,
            root_reachability=root_reachability,
        )

        # Update center ids for nodes one level below
        for n, dist_center in node_dist_center_sub.items():
            for dist, center in dist_center.items():
                node_dist_center[n][dist] = center

        # Append centers of child
        for d, sub_c in sub_centers.items():
            if d not in region_centers:
                region_centers[d] = sub_c
            else:
                region_centers[d].extend(sub_c)

    return node_dist_center, region_centers


def get_center_nodes(region_id_dict):
    """Relates a distance to the corresponding centers and each center
    to the list of nodes belonging to the center.
    
    Parameters
    ----------
    region_id_dict : dict
        node -> distance -> center id
    
    Returns
    -------
    dict
        distance -> center id -> list of nodes
    """

    dist_region_id_nodes = dict()

    # n -> max_dist -> center_id to max_dist -> center_id -> [nodes]
    for n, max_dist_center_id in region_id_dict.items():
        for max_dist, center_id in max_dist_center_id.items():
            if max_dist not in dist_region_id_nodes:
                dist_region_id_nodes[max_dist] = dict()
            if center_id not in dist_region_id_nodes[max_dist]:
                dist_region_id_nodes[max_dist][center_id] = list()
            dist_region_id_nodes[max_dist][center_id].append(n)
    return dist_region_id_nodes


def get_region_ids(
    G, reachability_dict, region_centers, path_region_ids=None, nodes=[]
):
    """Associate each node to its closest region center within a
     maximum reachable time limit.

    Parameters
    ----------
    G : networkx
        Street network
    reachability_dict : dict
        Associate each node with the reachable nodes within maximum
        distances.
    region_centers : dict
        Region centers for each maximum reachable time limit.

    Returns
    -------
    dict
        Dictionary relating nodes to the ids of its closest region
        centers within each maximum reachable time limit.
    """

    if not nodes:
        nodes = list(range(get_number_of_nodes(G)))

    region_id_dict = None
    if os.path.isfile(path_region_ids):
        region_id_dict = np.load(path_region_ids, allow_pickle=True).item()
        logging.info(
            f"\nReading region ids dictionary..."
            f"\nSource: '{path_region_ids}'."
        )
    else:
        logging.info(
            "\nFinding closest node region center ids..."
            f"\nTarget: '{path_region_ids}'."
        )
        region_id_dict = dict()

        # Loop nodes n to find the closest region centers
        for n in nodes:

            region_id_dict[n] = dict()

            for time_limit, centers in region_centers.items():

                # Set of nodes that can reach n whitin time limit
                can_reach = get_can_reach_set(
                    n, reachability_dict, max_trip_duration=time_limit
                )
                # Set of region centers that can access n
                accessible_regions = list(can_reach.intersection(centers))

                # Find closest region center
                closest = np.argmin(
                    [get_distance(G, c, n) for c in accessible_regions]
                )

                region_id_dict[n][time_limit] = accessible_regions[closest]

        np.save(path_region_ids, region_id_dict)

    return region_id_dict


def get_reachability_dic(
    root_path,
    distances,
    step=30,
    total_range=600,
    speed_km_h=30,
    step_list=None,
    outbound=False,
):
    """Which nodes are reachable from one another in "step" steps?
    E.g.:
    Given the following distance dictionary:

    FROM    TO   DIST(s)
    2       1     35
    3       1     60
    4       1     7
    5       1     20

    If step = 30, the reachability set for 1 is:
    reachable[1][30] = set([4, 5]).

    In other words, node 1 can be reached from nodes 4 and 5 in
    less than 30 steps.

    Hence, for a given OD pair (o, d) and step = s, if o in 
    reachable[d][s], then d can be reached from o in t steps.

    Arguments:
        distances {dict{float}} -- Distance dictionary/matrix
            (dic[o][d] = dist(o,d))
        root_path {str} -- Where to save reachability dictionary

    Keyword Arguments:

        step {int} -- The minimum reachability distance that multiplies
            until it reaches the total range.
        total_range{int} -- Total range used to define concentric
            reachability, step from step. Considered a multiple of step.
        speed_kh_h {int} -- in km/h to convert distance
            (default: {30} km_h). If different of None, 'step' and
            'total_range' are considered in seconds.
        step_list {list} -- Ad-hoc step list.
        outbound {bool} -- If True, get outbound reachability.

    Returns:
        [dict] -- Reachability structure.
                  E.g.: reachable[d][step] = set([o_1, o_2, o_3, o_n])
                  IMPORTANT: for the sake of memory optimization, nodes
                  from step 'x' are NOT included in step 'x+1'.
                  Use 'get_can_reach_set' to derive the overall
                  reachability, across the full range.
    """

    reachability_dict = None

    if step_list:
        steps_in_range_list = step_list
    else:
        # E.g., [30, 60, 90, ..., 600]
        steps_in_range_list = [
            i for i in range(step, total_range + step, step)
        ]

    try:
        reachability_dict = np.load(root_path, allow_pickle=True).item()
        logging.info(
            "\nReading reachability dictionary..." f"\nSource: '{root_path}'."
        )

    except:

        reachability_dict = defaultdict(lambda: defaultdict(set))

        logging.info(
            ("Calculating reachability...\n" + "Steps:{}").format(
                steps_in_range_list
            )
        )

        # Select generator for different structures (dict or matrix)
        if isinstance(distances, dict):
            ods = (
                (o, d) for o in distances.keys() for d in distances[o].keys()
            )
        else:
            ods = (
                (o, d)
                for o in range(len(distances))
                for d in range(len(distances))
            )

        for o, d in ods:
            # Dictionary contains only valid distances
            dist_m = distances[o][d]

            # So far, we are using distance in meters
            dist = dist_m

            # If speed is provided, convert distance to seconds
            # Steps are assumed to be in seconds too
            if speed_km_h:
                dist_s = int(3.6 * dist_m / speed_km_h + 0.5)
                dist = dist_s

            # Find the index of which max_travel_time_edge box dist_s is in
            step_id = bisect.bisect_left(steps_in_range_list, dist)

            if step_id < len(steps_in_range_list):

                if outbound:
                    # Which nodes can o access?
                    reachability_dict[o][steps_in_range_list[step_id]].add(d)
                else:
                    # Which nodes can access d?
                    reachability_dict[d][steps_in_range_list[step_id]].add(o)

        np.save(root_path, dict(reachability_dict))

    return reachability_dict, steps_in_range_list


def get_can_reach_set(n, reach_dic, max_trip_duration=150):
    """Return the set of all nodes whose trip to/from node n takes
    less than "max_trip_duration" seconds.

    Parameters
    ----------
    n : int
        Target node id
    reach_dic : dict[int][dict[int][set]]
        Stores the node ids whose distance to n is whitin max. trip
        duration (e.g., 30, 60, etc.)
    max_trip_duration : int, optional
        Max. trip duration in seconds a node can be distant
        from n, by default 150

    Returns
    -------
    set
        Set of nodes that can reach n in less than 'max_trip_duration'
        seconds.

    Example
    -------
    >>> reach_dic = {1:{0:[1],30:[2,3],60:[4,5,6]}}
    >>> get_can_reach_set(1, reach_dic, max_trip_duration=0)
    {1}
    >>> get_can_reach_set(1, reach_dic, max_trip_duration=30)
    {1,2,3}
    """

    can_reach_target = set()
    for t in reach_dic[n].keys():
        if t <= max_trip_duration:
            can_reach_target.update(reach_dic[n][t])
    return can_reach_target


def get_list_coord(G, o, d, projection="GPS", edge_index=0):
    """Get the list of intermediate coordinates between
    nodes o and d (inclusive).

    Arguments:
        G {networkx} -- Graph
        o {int} -- origin id
        d {int} -- destination id

    Returns:
        list -- E.g.: [(x1, y1), (x2, y2)]
    """

    edge_data = G.get_edge_data(o, d)[edge_index]
    try:
        if projection == "GPS":
            return ox.LineString(edge_data["geometry"]).coords
        else:
            return [
                wgs84_to_web_mercator(x, y)
                for x, y in ox.LineString(edge_data["geometry"]).coords
            ]

    # No intermendiate coordinates return (o,d)
    except Exception as e:
        # print(
        #     f"Cant get edge linestring! o={o}, d={d}, "
        #     f"projection={projection}, "
        #     f"edge_index={edge_index}. Exception: '{e}'."
        # )

        if projection == "GPS":
            return [
                (G.nodes[o]["x"], G.nodes[o]["y"]),
                (G.nodes[d]["x"], G.nodes[d]["y"]),
            ]
        else:
            return [
                wgs84_to_web_mercator(G.nodes[o]["x"], G.nodes[o]["y"]),
                wgs84_to_web_mercator(G.nodes[d]["x"], G.nodes[d]["y"]),
            ]


# #################################################################### #
# Query geojson ###################################################### #
# #################################################################### #


def get_point(G, p, **kwargs):
    """Get geojson point from node id

    Arguments:
        G {networkx} -- Base graph
        p {int} -- Node id

    Returns:
        dict -- Point geojson
    """

    point = {
        "type": "Feature",
        "properties": kwargs,
        "geometry": {
            "type": "Point",
            "coordinates": [G.nodes[p]["x"], G.nodes[p]["y"]],
        },
    }

    return point


def get_linestring(G, o, d, **kwargs):
    """Return geojson linestring corresponding of list of node ids
    in graph G.

    Arguments:
        G {networkx} -- Graph
        list_ids {list} -- List of node ids

    Returns:
        linestring -- Coordinates representing id list
    """

    linestring = []

    list_ids = get_sp(G, o, d)

    for i in range(0, len(list_ids) - 1):
        linestring.extend(get_list_coord(G, list_ids[i], list_ids[i + 1]))
        linestring = linestring[:-1]

    # Add last node (excluded in for loop)
    linestring.append((G.nodes[list_ids[-1]]["x"], G.nodes[list_ids[-1]]["y"]))

    # List of points (x y) connection from_id and to_id
    coords = [[u, v] for u, v in linestring]

    geojson = {
        "type": "Feature",
        "properties": kwargs,
        "geometry": {"type": "LineString", "coordinates": coords},
    }

    return geojson


def get_sp_coords(G, o, d, projection="GPS"):
    """Return coordinates of the shortest path.
    E.g.: [[x, y], [z,w]]

    Arguments:
        G {networkx} -- Graph
        list_ids {list} -- List of node ids

    Returns:
        linestring -- Coordinates representing id list (including od)
    """

    linestring = []

    list_ids = get_sp(G, o, d)

    if projection == "GPS":

        for i in range(0, len(list_ids) - 1):
            linestring.extend(
                get_list_coord(
                    G, list_ids[i], list_ids[i + 1], projection=projection
                )
            )
            linestring = linestring[:-1]

        # Add last node coordinate (excluded in for loop)
        linestring.append(
            (G.nodes[list_ids[-1]]["x"], G.nodes[list_ids[-1]]["y"])
        )

    else:

        for i in range(0, len(list_ids) - 1):
            linestring.extend(
                get_list_coord(
                    G, list_ids[i], list_ids[i + 1], projection=projection
                )
            )
            linestring = linestring[:-1]

        # Add last node coordinate (excluded in for loop)
        linestring.append(
            wgs84_to_web_mercator(
                G.nodes[list_ids[-1]]["x"], G.nodes[list_ids[-1]]["y"]
            )
        )

    # List of points (x y) connection from_id and to_id
    coords = [[u, v] for u, v in linestring]

    return coords


def get_duration(dist_m, speed_km_h=30):
    dist_s = 3.6 * dist_m / speed_km_h
    return dist_s


def get_intermediate_coords(
    G, o, d, n_coords, projection="GPS", waypoint=None, speed_km_h=30
):
    """Get "n_coords" between origin and destination. Populate segments
    in proportion to legs' distance.

    Parameters
    ----------
    G : networkx
        Street network
    o : int
        Origin id
    d : int
        Destination i
    n_coords : int
        Number of desired coordinates between o and d (inclusive)

    Returns
    -------
    list of coordinates
        [description]
    """

    # Shortest path
    if waypoint and waypoint not in [o, d]:
        sp = get_sp_coords(G, o, waypoint)
        sp = sp[:-1] + get_sp_coords(G, waypoint, d)
    else:
        # All coordinates (linestring) between od pair (incluseve)
        sp = get_sp_coords(G, o, d)

    # Number of coordinates is at least the shortest path
    n_coords = max(n_coords, len(sp))

    # print(f"\n\n###### Shortest path (len={len(sp)})")
    # print(sp)

    # If not single point
    if len(sp) > 1:
        # Coordinate pairs, e.g., [p1, p2, p3] = [(p1, p2), (p2, p3)]
        od_pairs = list(zip(sp[:-1], sp[1:]))

        # Distance (euclidian) in meters between each pair
        pair_distances = np.array([distance(*p1, *p2) for p1, p2 in od_pairs])

        # E.g., 1---100m----2, 2----300m----3 => [0.25, 0.75]
        percentage_pair_distances = pair_distances / np.sum(pair_distances)

        # How many points per pair (-1 removes last of the sequence)
        n_points_between_pairs = percentage_pair_distances * (n_coords - 1)
        n_points_between_pairs = np.ceil(n_points_between_pairs).astype(int)

        # Guarantees the right number of points by adding/subtracting
        while np.sum(n_points_between_pairs) + 1 > n_coords:
            # Remove points from the segment with the most points
            max_points_i = np.argmax(n_points_between_pairs)
            n_points_between_pairs[max_points_i] -= 1

        while np.sum(n_points_between_pairs) + 1 < n_coords:
            # Add points to the segment with the fewest points
            min_points_i = np.argmin(n_points_between_pairs)
            n_points_between_pairs[min_points_i] += 1

        # Tuple (od pair, #points between od - including o)
        intermediate_points = list(zip(od_pairs, n_points_between_pairs))
        # assert (min(n_points_between_pairs) >= 1), f"ERROR!{min(n_points_between_pairs)}"

        list_coords = []
        total_distance = 0
        leg_distances = [0]

        # print(n_points_between_pairs, percentage_pair_distances)

        for (p1, p2), n_intermediate in intermediate_points:

            list_coords.append(p1)

            distance_pair = 0

            # Since n_intermediate always include o, it has to be
            # greater or equal 2 to start adding nodes
            if n_intermediate >= 2:

                # Step fraction (if there are intermediate points)
                step_fraction = 1.0 / n_intermediate

                # Start from second point since p1 as already added
                # Finishes before p2 since it will be added in the next round
                all_fraction_steps = np.arange(
                    step_fraction, 0.9999, step_fraction
                )

                # Loop all fractions to derive the intermediate lon, lat
                # coordinates following the line
                for fraction in all_fraction_steps:
                    lon, lat = intermediate_coord(*p1, *p2, fraction)

                    distance_leg = distance(*list_coords[-1], lon, lat)

                    # Update distance between p1 and p2
                    distance_pair += distance_leg

                    # Add intermediate coordinate
                    list_coords.append([lon, lat])

                    # Distance to travel the leg between the current pair
                    leg_distances.append(distance_leg)

            # Add last leg distance
            distance_last_leg = distance(*list_coords[-1], *p2)

            # Update distance between p1 and p2
            distance_pair += distance_last_leg

            # Distance to travel the last leg (to p2)
            leg_distances.append(distance_last_leg)
            total_distance += distance_pair

        # Add last point (last p2)
        list_coords.append(sp[-1])

        # Get cumulative distances
        distance_cum_list = np.cumsum(leg_distances)

        # Get cumulative durations
        leg_durations = [
            get_duration(d, speed_km_h=speed_km_h) for d in leg_distances
        ]
        duration_cum_list = np.cumsum(leg_durations)

    else:
        list_coords = sp
        duration_cum_list = []
        distance_cum_list = []
        total_distance = 0

    if projection == "MERCATOR":
        list_coords = [wgs84_to_web_mercator(*p) for p in list_coords]
    # print(f"list_coords len={len(list_coords)}")
    # pprint(list_coords)
    # print(f"duration_cum_list len={len(duration_cum_list)}")
    # pprint(duration_cum_list)
    # print(f"distance_cum_list len={len(distance_cum_list)}")
    # pprint(distance_cum_list)
    # print("Total:", total_distance)
    return list_coords, duration_cum_list, distance_cum_list, total_distance


def enrich_graph(
    G, max_dist=50, max_travel_time_edge=60, n_coords=100, speed_km_h=20
):
    """Receives cleaned up network (ids starting from 0) and add extra
    points between edges. Points are at least max_travel_time_edge
    apart.

    Parameters
    ----------
    G : networkx
        Graph to enrich
    max_dist : int, optional
        [description], by default 50
    max_travel_time_edge : int, optional
        [description], by default 60
    n_coords_between_od : int, optional
        [description], by default 100
    speed_km_h : int, optional
        [description], by default 20

    Returns
    -------
    networkx
        Enriched graph of max_travel_time_edge edges
    """

    G = G.copy()

    all_edges_to_add = list()
    all_edges_to_remove = list()
    all_nodes_to_add = list()

    # Node ids start from last id
    node_id = len(G.nodes())

    # Loop edges and break them in sub edges
    for o, d in G.edges():

        # Get min_coords_edge coordinates in edge id (od inclusive)
        (
            intermediate_coords,
            cumsum_duration,
            cumsum_dist,
            total_distance,
        ) = get_intermediate_coords(G, o, d, n_coords, speed_km_h=speed_km_h)

        distance_od = cumsum_dist[-1]
        duration_od = cumsum_duration[-1]

        # Number of sub edges
        n_subedges = int(np.ceil(duration_od / max_travel_time_edge))

        # If edge
        if n_subedges > 1:

            # Percentual of each sub edge
            step = 1 / n_subedges

            # print(
            #     "\n###### "
            #     f"{o:04}({G.nodes[o]['x']:11.7f},{G.nodes[o]['y']:11.7f}) -> "
            #     f"{d:04}({G.nodes[d]['x']:11.7f},{G.nodes[d]['y']:11.7f}) - "
            #     f"dist={distance_od:7.2f} - duration={duration_od:7.2f} - "
            #     f"points={n_subedges}"
            # )
            # # pprint(G[o][d])
            # print("COORDS:")
            # pprint(intermediate_coords)

            # od_pairs = list(zip(sp[:-1], sp[1:]))

            # print(
            #     f"cumsum={len(cumsum_distance)} - intermediate={len(intermediate_coords)}"
            # )

            # lon1, lat1 = G.nodes[o]['x'], G.nodes[o]['y']

            # Get all edge attributes and broadcast them to all sub edges
            edge_attr_od = G.edges[o, d, 0]

            sub_o_idx = 0
            lo_node_id = o
            # Guarantee last fraction is always the full distance
            cum_leg_fraction_list = list(np.arange(step, 0.999999, step)) + [1]
            for cum_leg_fraction in cum_leg_fraction_list:

                # Copy od attributes
                edge_attributes = deepcopy(edge_attr_od)

                # Distance and duration where edge will be cut
                partial_dist = cum_leg_fraction * distance_od
                partial_duration = cum_leg_fraction * duration_od

                # Find position of leftmost coordinate whose partial
                # duration is lower than or equal to partial_dist
                sub_d_idx = bisect.bisect_left(
                    cumsum_duration, partial_duration
                )

                # Guarantee edge length does not surpass max travel time
                while (
                    cumsum_duration[sub_d_idx] - cumsum_duration[sub_o_idx]
                    > max_travel_time_edge
                ):
                    sub_d_idx -= 1
                # print(f"Partial duration={partial_duration} n_coords={len(intermediate_coords)}, sub_o_idx={sub_o_idx}, sub_d_idx={sub_d_idx}")

                if sub_o_idx == sub_d_idx:
                    continue

                # Add node if different than destination d
                if cum_leg_fraction < 1:
                    destination_id = node_id
                    x, y = intermediate_coords[sub_d_idx]
                    all_nodes_to_add.append(
                        {"id": destination_id, "attr_dict": {"x": x, "y": y}}
                    )
                    # Increment number of nodes
                    node_id += 1
                else:
                    destination_id = d

                # Construct the geometry of points covered by subedge
                edge_attributes["geometry"] = LineString(
                    [
                        Point(lon, lat)
                        for lon, lat in intermediate_coords[
                            sub_o_idx : sub_d_idx + 1
                        ]
                    ]
                )

                # Length of subedge
                edge_attributes["length"] = (
                    cumsum_dist[sub_d_idx] - cumsum_dist[sub_o_idx]
                )

                all_edges_to_add.append(
                    {
                        "origin": lo_node_id,
                        "destination": destination_id,
                        "attr_dict": edge_attributes,
                    }
                )

                # dur = get_duration(edge_attributes["length"], speed_km_h=speed_km_h)
                # if dur > max_travel_time_edge:
                #     print(f"Duration={dur:.2f}, o={sub_o_idx}, d={sub_d_idx}, fraction={cum_leg_fraction:.2f}, fractions={np.array(cum_leg_fraction_list)}, partial={partial_duration:.2f}, cumsum_sub={cumsum_duration[sub_d_idx] - cumsum_duration[sub_o_idx]:.2f},\n  cumsum_array={cumsum_duration[sub_o_idx:sub_d_idx+1]},\ncumsum_array_0={cumsum_duration[sub_o_idx:sub_d_idx+1]-cumsum_duration[sub_o_idx]}")

                # print(
                #     f"##### leg={cum_leg_fraction} - {sub_o_idx}({lo_node_id}) -- {sub_d_idx}({destination_id}) - length={edge_attributes['length']} ############"
                # )
                # print(cumsum_duration[sub_o_idx : sub_d_idx])
                # print("len.:", len(intermediate_coords[sub_o_idx : sub_d_idx+1]))
                # pprint(intermediate_coords[sub_o_idx : sub_d_idx+1])

                sub_o_idx = sub_d_idx
                lo_node_id = destination_id

            all_edges_to_remove.append((o, d))

    # Adding new nodes
    for node in all_nodes_to_add:
        G.add_node(node["id"], **node["attr_dict"])

    # Adding new edges
    for edge_attr_od in all_edges_to_add:
        G.add_edge(
            edge_attr_od["origin"],
            edge_attr_od["destination"],
            **edge_attr_od["attr_dict"],
        )

    # Removing old edges
    G.remove_edges_from(all_edges_to_remove)

    # Relabel nodes and edges
    for node in G.nodes():
        G.nodes[node]["osmid"] = node

    for i, (o, d) in enumerate(G.edges()):
        G.edges[o, d, 0]["osmid"] = i

    return G


def get_sp_linestring_durations(G, o, d, speed):
    """Return coordinates of the shortest path.
    E.g.: [[x, y], [z,w]]

    Arguments:
        G {networkx} -- Graph
        list_ids {list} -- List of node ids

    Returns:
        linestring -- Coordinates representing id list
    """

    linestring = []

    list_ids = get_sp(G, o, d)

    for i in range(0, len(list_ids) - 1):
        linestring.extend(get_list_coord(G, list_ids[i], list_ids[i + 1]))
        linestring = linestring[:-1]

    # Add last node (excluded in for loop)
    linestring.append((G.nodes[list_ids[-1]]["x"], G.nodes[list_ids[-1]]["y"]))

    # List of points (x y) connection from_id and to_id
    coords = [[u, v] for u, v in linestring]

    return coords


# #################################################################### #
# Query network element ############################################## #
# #################################################################### #


@functools.lru_cache(maxsize=1)
def get_number_of_nodes(G):
    return nx.number_of_nodes(G)


def get_sp(G, o, d):
    """Return shortest path between node ids o and d

    Arguments:
        G {networkx} -- [description]
        o {int} -- Origin node id
        d {int} -- Destination node id

    Returns:
        list -- List of nodes between o and d (included)
    """
    return nx.shortest_path(G, source=o, target=d, weight="length")


def get_random_node(G):
    """Find random node in G

    Arguments:
        G {networkx} -- Transportation network

    Returns:
        [tuple] -- node id, lon, lat
    """
    random_node = random.randint(0, nx.number_of_nodes(G) - 1)
    return random_node, G.nodes[random_node]["x"], G.nodes[random_node]["y"]


def get_coords_node(n, G):
    return G.nodes[n]["x"], G.nodes[n]["y"]


@functools.lru_cache(maxsize=None)
def get_distance(G, o, d):
    return nx.dijkstra_path_length(G, o, d, weight="length")


def get_largest_connected_component(G):
    """Return the largest strongly connected component of graph G.

    Arguments:
        G {networkx} -- Graph

    Returns:
        set -- Set of nodes pertaining to the component
    """

    largest_cc = max(nx.strongly_connected_components(G), key=len)
    s_connected_component = [
        len(c)
        for c in sorted(
            nx.strongly_connected_components(G), key=len, reverse=True
        )
    ]
    logging.info(
        f"Size of strongly connected components: {s_connected_component}"
    )
    return set(largest_cc)


# #################################################################### #
# Distances ########################################################## #
# #################################################################### #


def distance(lon1, lat1, lon2, lat2):
    """Return the great-circle distance (meters) between two points
    using haversine.
    
    Parameters
    ----------
    lon1 : float
        Longitude point 1
    lat1 : float
        Latitude point 1
    lon2 : float
        Longitude point 2
    lat2 : float
        Latitude point 2
    
    Returns
    -------
    Float
        Distance in meters
    """

    return ox.great_circle_vec(lat1, lon1, lat2, lon2)


def intermediate_coord(lon1, lat1, lon2, lat2, fraction):
    """
    Returns the point at given fraction between two coordinates.
    """
    rad_lat_o = math.radians(lat1)
    rad_lon_o = math.radians(lon1)
    rad_lat_d = math.radians(lat2)
    rad_lon_d = math.radians(lon2)

    # distance between points
    rad_lat = rad_lat_d - rad_lat_o
    rad_lon = rad_lon_d - rad_lon_o
    a = math.sin(rad_lat / 2) * math.sin(rad_lat / 2) + (
        math.cos(rad_lat_o)
        * math.cos(rad_lat_d)
        * math.sin(rad_lon / 2)
        * math.sin(rad_lon / 2)
    )

    delta = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    A = math.sin((1 - fraction) * delta) / math.sin(delta)
    B = math.sin(fraction * delta) / math.sin(delta)

    x = A * math.cos(rad_lat_o) * math.cos(rad_lon_o) + B * math.cos(
        rad_lat_d
    ) * math.cos(rad_lon_d)
    y = A * math.cos(rad_lat_o) * math.sin(rad_lon_o) + B * math.cos(
        rad_lat_d
    ) * math.sin(rad_lon_d)
    z = A * math.sin(rad_lat_o) + B * math.sin(rad_lat_d)

    lat_atan2 = math.atan2(z, math.sqrt(x * x + y * y))
    lon_atan2 = math.atan2(y, x)

    lat = math.degrees(lat_atan2)
    lon = math.degrees(lon_atan2)

    return lon, lat


def get_distance_matrix(root_path, G, distance_dic_m=None):
    """Return distance matrix (n x n). Value is 'None' when path does
    not exist

    Arguments:
        G {networkx} -- Graph to loop nodes
        distance_dic_m {dic} -- previously calculated distance dictionary

    Returns:
        [list[list[float]]] -- Distance matrix
    """
    # TODO simplify - test:  nx.shortest_path_length(G, source=o,
    # target=d, weight="length")

    # Creating distance matrix
    dist_matrix = []
    try:
        logging.info(
            "\nTrying to read distance matrix from file:\n'{}'.".format(
                root_path
            )
        )
        dist_matrix = np.load(root_path)

    except Exception as e:
        logging.info(
            f"Reading failed! Exception: {e} \nCalculating shortest paths..."
        )

        try:

            for from_node in range(0, get_number_of_nodes(G)):
                to_distance_list = []
                for to_node in range(0, get_number_of_nodes(G)):

                    try:
                        dist_km = distance_dic_m[from_node][to_node]
                        to_distance_list.append(dist_km)
                    except:
                        to_distance_list.append(None)

                dist_matrix.append(to_distance_list)

            dist_matrix = np.array(dist_matrix)

            np.save(root_path, dist_matrix)

        except Exception as e:
            logging.info(f"Creating distance matrix failed! Exception: {e}.")
            exit(0)

    logging.info(
        f"Distance data loaded successfully. " f" #Nodes: {dist_matrix.shape}"
    )

    return dist_matrix


def get_dt_distance_matrix(path, dist_matrix):
    """Get dataframe from distance matrix

    Arguments:
        path {string} -- File path of distance matrix
        dist_matrix {list[list[float]]} -- Matrix of distances

    Returns:
        pandas dataframe -- Distance matrix
    """

    dt = None

    try:
        # Load tripdata
        # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
        dt = pd.read_csv(path, header=None)

    except Exception as e:
        logging.info(str(e))
        dt = pd.DataFrame(dist_matrix)
        dt.to_csv(
            path, index=False, header=False, float_format="%.6f", na_rep="INF"
        )

    return dt


def get_distance_dic(root_path, G):
    """Get distance dictionary (Dijkstra all to all using path length).
    E.g.: [o][d]->distance

    Arguments:
        root_path {string} -- Try to load path before generating
        G {networkx} -- Street network

    Returns:
        dict -- Distance dictionary (all to all)
    """
    distance_dic_m = None
    try:
        logging.info(
            "\nTrying to read distance data from file:\n'{}'.".format(
                root_path
            )
        )
        distance_dic_m = np.load(root_path, allow_pickle=True).item()

    except Exception as e:

        try:
            logging.info(
                f"Reading failed! Exception: {e} \nCalculating shortest paths..."
            )
            all_dists_gen = nx.all_pairs_dijkstra_path_length(
                G, weight="length"
            )

            # Save with pickle (meters)
            distance_dic_m = dict(all_dists_gen)
            np.save(root_path, distance_dic_m)

        except Exception as e2:
            print(f"Exception {e2}.")

    logging.info(
        f"Distance data loaded successfully. "
        f" #Nodes: {len(distance_dic_m.values())}"
    )

    return distance_dic_m


def wgs84_to_web_mercator(lon, lat):
    k = 6378137
    x = lon * (k * np.pi / 180.0)
    y = np.log(np.tan((90 + lat) * np.pi / 360.0)) * k
    return x, y


# #################################################################### #
# Region centers ##################################################### #
# #################################################################### #


def get_node_region_ids(G, region_id_dict):
    """Get list of node ids for each region (defined with minimum
    reachable time)

    Parameters
    ----------
    G : networkx
        Transportation network
    region_id_dict : dict
        Dictionary associating each id to its region id

    Returns
    -------
    dict
        Dictionary of max. reachable time keys and node id lists.
    """
    node_level_id = defaultdict(list)
    for node_id in G.nodes():
        node_level_id[0].append(node_id)
        for time_limit, region_id in region_id_dict[node_id].items():
            node_level_id[time_limit].append(region_id)
    return node_level_id


def get_node_delay_center_id(G, region_id_dict):
    """Get dictionary associating node ids to center ids according to
    delays.

    Parameters
    ----------
    G : networkx
        Transportation network
    region_id_dict : dict
        Dictionary associating each id to its region id

    Returns
    -------
    dict
        Dictionary of max. reachable time keys and node id lists.
    """

    node_delay_center_id = dict()
    for node_id in G.nodes():
        delay_center_id = dict()
        for time_limit, region_id in region_id_dict[node_id].items():

            delay_center_id[time_limit] = region_id

        node_delay_center_id[node_id] = delay_center_id

    return delay_center_id


def can_reach(origin, target, max_delay, reachability_dic, round_trip=False):
    """Check if 'target' can be reached from 'origin' in less than
    'max_delay' time steps
    
    Parameters
    ----------
    origin : int
        id of departure node
    target : int
        id of node to reach
    max_delay : int
        Maximum trip delay between origin and target
    reachability_dic : dict(int=dict(int=set()))
        Stores the set 's' of nodes that can reach 'target' node in less
        then 't' time steps.
        E.g.: reachability_dic[target][max_delay] = s
    round_trip : bool, optional
        True when 'origin' must reach 'target' and the other way around,
        by default False

    Returns
    -------
    int
        1 if 'target' can be reached from 'origin' in less than
        'max_delay' time steps

    Example
    -------

    ### One way reachability

    >>> reachability_dic[1][30] = {1,2,3,4}
    >>> reachability_dic[1][60] = {7,8,9,10,11,12}
    >>> reachability_dic[1][90] = {13,14,15,17,18,19,20}

    >>> can_reach(2, 1, 30, reachability_dic, round_trip=False)
    1 # node 1 can be reached from node 2 in <= 30s
    >>> can_reach(2, 1, 90, reachability_dic, round_trip=False)
    1 # node 1 can be reached from node 2 in <= 90s

    ### Round reachability
    
    >>> reachability_dic[1][30] = {1,2,3,4}
    >>> reachability_dic[1][60] = {7,8,9,10,11,12}
    >>> reachability_dic[1][90] = {13,14,15,17,18,19,20}

    >>> reachability_dic[2][30] = {2,3}
    >>> reachability_dic[2][60] = {1,3,4}
    >>> reachability_dic[2][90] = {21,24,15}

    >>> can_reach(2, 1, 30, reachability_dic, round_trip=True)
    0 # node 1 can be reached from node 2 in <= 30s but node 2 CANNOT be reached from node 1 in 30s
    >>> can_reach(2, 1, 90, reachability_dic, round_trip=True)
    1 # node 1 can be reached from node 2 in <= 90s and node 2 CAN be reached from node 2 in 30s
    """

    # Target can be reached from origin
    origin_target = False

    # Origin can be reached from target
    target_origin = False

    for step in reachability_dic[target].keys():
        if step <= max_delay:

            # Origin->target has been defined to 1 in previous step
            if not origin_target:
                if origin in reachability_dic[target][step]:
                    origin_target = True

            # If two way, check if target can also reach origin
            if round_trip:
                if not target_origin:
                    if target in reachability_dic[origin][step]:
                        target_origin = True

                # If two way reachability exists
                if origin_target and target_origin:
                    return 1

            # If one way reachability exists (i.e., 'target' can be
            # reached from 'origin' in less than 'max_delay' time steps)
            elif origin_target:
                return 1

    # e.g., reachability[n1][30] = {n1,n2,n3,n4}

    return 0


def ilp_node_reachability(
    reachability_dic,
    node_set_ids,
    max_delay=180,
    log_path=None,
    time_limit=None,
    round_trip=False,
):

    # List of nodes ids

    try:

        # Create a new model
        m = Model("region_centers")

        if log_path:

            # Create log path if not exists
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            round_trip_label = "_round_trip" if round_trip else ""

            m.Params.LogFile = "{}/region_centers{}_{}.log".format(
                log_path, round_trip_label, max_delay
            )

            m.Params.ResultFile = "{}/region_centers{}_{}.lp".format(
                log_path, round_trip_label, max_delay
            )

        # xi = 1, if vertex Vi is used as a region center
        # and 0 otherwise
        x = m.addVars(node_set_ids, vtype=GRB.BINARY, name="x")

        # Ensures that every node in the road network graph is reachable
        # within 'max_delay' travel time by at least one region center
        # selected from the nodes in the graph.
        # To extract the region centers, we select from V all vertices
        # V[i] such that x[i] = 1.

        for origin in node_set_ids:
            m.addConstr(
                (
                    quicksum(
                        x[center]
                        * can_reach(
                            center,
                            origin,
                            max_delay,
                            reachability_dic,
                            round_trip=round_trip,
                        )
                        for center in node_set_ids
                    )
                    >= 1
                ),
                "ORIGIN_{}".format(origin),
            )

        # Set objective
        m.setObjective(quicksum(x), GRB.MINIMIZE)

        if time_limit is not None:
            m.Params.timeLimit = time_limit

        # Solve
        m.optimize()

        region_centers = list()

        # Model statuses
        is_unfeasible = m.status == GRB.Status.INFEASIBLE
        is_umbounded = m.status == GRB.Status.UNBOUNDED
        found_optimal = m.status == GRB.Status.OPTIMAL
        found_time_expired = (
            m.status == GRB.Status.TIME_LIMIT and m.SolCount > 0
        )

        if is_umbounded:
            raise Exception(
                "The model cannot be solved because it is unbounded"
            )

        elif found_optimal or found_time_expired:

            if found_time_expired:
                logging.info("TIME LIMIT ({} s) RECHEADED.".format(time_limit))

            # Sweep x_n = 1 variables to create list of region centers
            var_x = m.getAttr("x", x)
            for n in node_set_ids:
                if var_x[n] > 0.0001:
                    region_centers.append(n)

            return region_centers

        elif is_unfeasible:

            logging.info("Model is infeasible.")
            raise Exception("Model is infeasible.")
            # exit(0)

        elif (
            m.status != GRB.Status.INF_OR_UNBD
            and m.status != GRB.Status.INFEASIBLE
        ):
            logging.info("Optimization was stopped with status %d" % m.status)
            raise Exception("Model is infeasible.")

    except GurobiError as e:
        raise Exception(" Gurobi error code " + str(e.errno))

    except AttributeError as e:
        raise Exception("Encountered an attribute error:" + str(e))


def get_region_centers(
    steps,
    path_region_centers,
    reachability_dic,
    node_set_ids,
    root_path=None,
    time_limit=60,
    round_trip=False,
    parent_center="",
):
    """Find minimum number of region centers, every 'step'

    ILP from:
      Wallar, A., van der Zee, M., Alonso-Mora, J., & Rus, D. (2018).
      Vehicle Rebalancing for Mobility-on-Demand Systems with 
      Ride-Sharing. Iros, 4539–4546.

    Why using regions?
    The region centers are computed a priori and are used to aggregate
    requests together so the rate of requests for each region can be
    computed. These region centers are also used for rebalancing as
    they are the locations that vehicles are proactively sent to.

    Parameters
    ----------
    path_region_centers : str
        Path to save/load dictionary of region centers.

    reachability_dic : dict{int:dict{int:set}
        Stores the set
            's' of nodes that can reach 'target' node in less then 't'
            time steps.  E.g.: reachability_dic[target][max_delay] = s

    root_path : str, optional
        Location where intermediate work (i.e., previous max. durations
        from reachability dictionary), and model logs should be
        saved, by default None

    time_limit : int, optional
        Expiration time (in seconds) of the ILP model
        execution, by default 60

    Returns
    -------
    dict
        Dictionary relating max_delay to region centers
    """

    centers_dic = None
    logging.info(
        "\nReading region center dictionary...\nSource: '{}'.".format(
            path_region_centers
        )
    )
    if os.path.isfile(path_region_centers):
        centers_dic = np.load(path_region_centers, allow_pickle=True).item()

    else:

        logging.info(
            (
                "\nCalculating region center dictionary..."
                "\nMax. durations: {}"
                "\nTarget path: '{}'."
            ).format(steps, path_region_centers)
        )
        # If not None, defines the location of the steps of a solution
        centers_gurobi_log = None
        centers_sub_sols = None

        if root_path is not None:
            # Create folder to save logs
            centers_gurobi_log = "{}/mip_region_centers/gurobi_log".format(
                root_path
            )

            if not os.path.exists(centers_gurobi_log):
                os.makedirs(centers_gurobi_log)

            # Create folder to save intermediate work, that is, previous
            # max_delay steps.
            centers_sub_sols = "{}/mip_region_centers/sub_sols".format(
                root_path
            )

            if not os.path.exists(centers_sub_sols):
                os.makedirs(centers_sub_sols)

        centers_dic = dict()
        for max_delay in sorted(steps, reverse=True):

            if centers_sub_sols is not None:
                # Name of intermediate region centers file for 'max_delay'
                file_name = "{}/{}_{:04}.npy".format(
                    centers_sub_sols, parent_center, max_delay
                )

                # Pre-calculated region center is loaded in case it exists.
                # This helps filling the complete 'centers_dic' without
                # starting the process from the beginning, in case an error
                # has occured.
                if os.path.isfile(file_name):
                    # Load max delay in centers_dic
                    centers_dic[max_delay] = np.load(
                        file_name, allow_pickle=True
                    )
                    logging.info(f"{file_name} already calculated.")
                    continue

            try:
                # Find the list of centers for max_delay
                centers = ilp_node_reachability(
                    reachability_dic,
                    node_set_ids,
                    max_delay=max_delay,
                    log_path=centers_gurobi_log,
                    time_limit=time_limit,
                    round_trip=round_trip,
                )
            except Exception as e:
                logging.info(str(e))
            else:
                centers_dic[max_delay] = centers
                logging.info(
                    "Max. delay: {} = # Nodes: {}".format(
                        max_delay, len(centers)
                    )
                )

                # Save intermediate steps (region centers of 'max_delay')
                if root_path:
                    np.save(file_name, centers)

                np.save(path_region_centers, centers_dic)

    return centers_dic
