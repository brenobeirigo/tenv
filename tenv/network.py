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
    print("Loading ", path)

    # if file does not exist write header
    if not os.path.isfile("{}/{}".format(folder, filename)):
        print("Network is not in '{}'".format(path))
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


def get_network_from(region, root_path, graph_name, graph_filename):
    """Download network from region. If exists (check graph_filename),
    try loading.

    Arguments:
        region {string} -- Location. E.g., "Manhattan Island,
            New York City, New York, USA"
        root_path {string} -- Path where graph is going to saved
        graph_name {string} -- Name to be stored in graph structure
        graph_filename {string} -- File name .graphml to be saved
            in root_path

    Returns:
        [networkx] -- Graph loaeded or downloaded
    """
    # Street network
    G = load_network(graph_filename, folder=root_path)

    if G is None:
        # Try to download
        try:
            G = download_network(region, "drive")

            # Create and store graph name
            G.graph["name"] = graph_name

            # Save region name
            G.graph["region"] = region

            print(
                "#ORIGINAL -  NODES: {} ({} -> {}) -- #EDGES: {}".format(
                    len(G.nodes()),
                    min(G.nodes()),
                    max(G.nodes()),
                    len(G.edges()),
                )
            )

            G = ox.remove_isolated_nodes(G)

            # Set of nodes with low connectivity (end points)
            # Must be eliminated to avoid stuch vehicles
            # (enter but cannot leave)
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

            # Relabel nodes
            mapping = {k: i for i, k in enumerate(sorted(G.nodes()))}
            G = nx.relabel_nodes(G, mapping)

            # Save
            ox.save_graphml(G, filename=graph_filename, folder=root_path)

        except Exception as e:
            print("Error loading graph:", e)

    print(
        "\n# NETWORK -  NODES: {} ({} -> {}) -- #EDGES: {}".format(
            len(G.nodes()), min(G.nodes()), max(G.nodes()), len(G.edges())
        )
    )

    return G


def save_graph_pic(G, path):
    """Save a picture (svg) of graph G.

    Arguments:
        G {networkx} -- Working graph
    """

    fig, ax = ox.plot_graph(
        G,
        fig_height=15,
        node_size=0.5,
        edge_linewidth=0.3,
        save=True,
        show=False,
        file_format="svg",
        filename="{}/{}".format(path, G.graph["name"]),
    )


def get_sorted_neighbors(
    distances, region_centers, minimum_distance=0, path_sorted_neighbors=None
):
    neighbors = None
    if os.path.isfile(path_sorted_neighbors):
        neighbors = np.load(path_sorted_neighbors).item()
        print(
            f"\nReading region ids dictionary..."
            f"\nSource: '{path_sorted_neighbors}'."
        )
    else:
        print(
            f"\nFinding closest node region center ids..."
            f"\nTarget: '{path_sorted_neighbors}'."
        )
        neighbors = dict()
        for t, centers in region_centers.items():
            # Minimum distances
            if t < minimum_distance:
                continue

            print(f"{t:04} - {len(centers)}")
            neighbors[t] = dict()
            for c_o in centers:
                neighbors[t][c_o] = list()
                for c_d in centers:
                    neighbors[t][c_o].append((c_d, distances[c_o][c_d]))

                # Sort according to distance
                neighbors[t][c_o].sort(key=lambda tup: tup[1])

        np.save(path_sorted_neighbors, neighbors)

    return neighbors


def get_center_nodes(region_id_dict):

    a = defaultdict(lambda: defaultdict(list))

    for n, max_dist_center_id in region_id_dict.items():
        for max_dist, center_id in max_dist_center_id.items():
            a[max_dist][center_id].append(n)
    return a


def get_region_ids(G, reachability_dict, region_centers, path_region_ids=None):
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

    region_id_dict = None
    if os.path.isfile(path_region_ids):
        region_id_dict = np.load(path_region_ids).item()
        print(
            f"\nReading region ids dictionary..."
            f"\nSource: '{path_region_ids}'."
        )
    else:
        print(
            "\nFinding closest node region center ids..."
            f"\nTarget: '{path_region_ids}'."
        )
        region_id_dict = dict()

        # Loop nodes n to find the closest region centers
        for n in range(get_number_of_nodes(G)):

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
    root_path, distance_dic, step=30, total_range=600, speed_km_h=30
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
        distance_dic {dict{float}} -- Distance dictionary
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

    Returns:
        [dict] -- Reachability structure.
                  E.g.: reachable[d][step] = set([o_1, o_2, o_3, o_n])
                  IMPORTANT: for the sake of memory optimization, nodes
                  from step 'x' are NOT included in step 'x+1'.
                  Use 'get_can_reach_set' to derive the overall
                  reachability, across the full range.
    """

    reachability_dict = None

    # E.g., [30, 60, 90, ..., 600]
    steps_in_range_list = [i for i in range(step, total_range + step, step)]
    
    try:
        reachability_dict = np.load(root_path).item()
        print(
            "\nReading reachability dictionary..." f"\nSource: '{root_path}'."
        )

    except:

        reachability_dict = defaultdict(lambda: defaultdict(set))


        print(
            ("Calculating reachability...\n" + "Steps:{}").format(
                steps_in_range_list
            )
        )

        for o in distance_dic.keys():
            for d in distance_dic[o].keys():

                # Dictionary contains only valid distances
                dist_m = distance_dic[o][d]

                # So far, we are using distance in meters
                dist = dist_m

                # If speed is provided, convert distance to seconds
                # Steps are assumed to be in seconds too
                if speed_km_h:
                    dist_s = int(3.6 * dist_m / speed_km_h + 0.5)
                    dist = dist_s

                # Find the index of which max_duration box dist_s is in
                step_id = bisect.bisect_left(steps_in_range_list, dist)

                if step_id < len(steps_in_range_list):
                    reachability_dict[d][steps_in_range_list[step_id]].add(o)

        np.save(root_path, dict(reachability_dict))

    return reachability_dict, steps_in_range_list


def get_can_reach_set(n, reach_dic, max_trip_duration=150):
    """Return the set of all nodes whose trip to node n takes
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


def get_list_coord(G, o, d, projection="GPS"):
    """Get the list of intermediate coordinates between
    nodes o and d (inclusive).

    Arguments:
        G {networkx} -- Graph
        o {int} -- origin id
        d {int} -- destination id

    Returns:
        list -- E.g.: [(x1, y1), (x2, y2)]
    """

    edge_data = G.get_edge_data(o, d)[0]
    try:
        if projection == "GPS":
            return ox.LineString(edge_data["geometry"]).coords
        else:
            return [
                wgs84_to_web_mercator(x, y)
                for x,y in ox.LineString(edge_data["geometry"]).coords
            ]
    except:
        if projection == "GPS":
            return [
                (G.node[o]["x"], G.node[o]["y"]),
                (G.node[d]["x"], G.node[d]["y"]),
            ]
        else:
            return [
                wgs84_to_web_mercator(G.node[o]["x"], G.node[o]["y"]),
                wgs84_to_web_mercator(G.node[d]["x"], G.node[d]["y"]),
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
            "coordinates": [G.node[p]["x"], G.node[p]["y"]],
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
    linestring.append((G.node[list_ids[-1]]["x"], G.node[list_ids[-1]]["y"]))

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
        linestring -- Coordinates representing id list
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
            (G.node[list_ids[-1]]["x"], G.node[list_ids[-1]]["y"])
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
                G.node[list_ids[-1]]["x"], G.node[list_ids[-1]]["y"]
            )
        )

    # List of points (x y) connection from_id and to_id
    coords = [[u, v] for u, v in linestring]

    return coords


def get_duration(dist_m, speed_km_h=30):
    dist_s = 3.6 * dist_m / speed_km_h
    return dist_s

def get_intermediate_coords(G, o, d, n_coords, projection="GPS", waypoint=None):
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
        Number of coordinates between o and d (inclusive)

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
        sp = get_sp_coords(G, o, d)
    
    # If not single point
    if len(sp) > 1:
        # Coordinate pairs, e.g., [(p1, p2), (p2, p3)]
        od_pairs = list(zip(sp[:-1], sp[1:]))

        # Distance (euclidian) in meters between each pair
        pair_distances = np.array([distance(*p1, *p2) for p1, p2 in od_pairs])

        # E.g., 1---100m----2, 2----300m----3 => [0.25, 0.75]
        percentage_pair_distances = pair_distances / np.sum(pair_distances)

        # How many points per pair (-1 removes last of the sequence)
        n_points_between_pairs = percentage_pair_distances * (n_coords - 1)
        n_points_between_pairs = np.ceil(n_points_between_pairs)

        # Guarantees the right number of points
        while np.sum(n_points_between_pairs) + 1 > n_coords:
            n_points_between_pairs[np.argmax(n_points_between_pairs)] -= 1

        while np.sum(n_points_between_pairs) + 1 < n_coords:
            n_points_between_pairs[np.argmin(n_points_between_pairs)] += 1

        # Tuple (od pair, #points between od - including o)
        intermediate_points = list(zip(od_pairs, n_points_between_pairs))

        list_coords = []
        total_distance = 0
        duration_s = [0]

        for pair, n_intermediate in intermediate_points:

            p1, p2 = pair

            list_coords.append(p1)

            # Step fraction (if there are intermediate points)
            step_fraction = 1.0 / (n_intermediate if n_intermediate > 0 else 1)

            # Start from second point since p1 as already added
            # Finishes before p2 since it will be added in the next round
            all_fraction_steps = np.arange(
                step_fraction,
                0.9999,
                step_fraction
            )

            distance_pair = 0

            # Loop all fractions to derive the intermediate lon, lat
            # coordinates following the line
            for fraction in all_fraction_steps:
                lon, lat = intermediate_coord(*p1, *p2, fraction)

                distance_leg = distance(*list_coords[-1], lon, lat)

                # Update distance between p1 and p2
                distance_pair += distance_leg

                # Add intermediate coordinate
                list_coords.append([lon, lat])

                # Duration to travel the leg between the current pair
                duration_s.append(get_duration(distance_leg))

            # Add last leg distance
            distance_last_leg = distance(*list_coords[-1], *p2)

            # Update distance between p1 and p2
            distance_pair += distance_last_leg

            # Duration to travel the last leg (to p2)
            duration_s.append(get_duration(distance_last_leg))

            total_distance += distance_pair

        # Add last point (last p2)
        list_coords.append(sp[-1])

        # Cumulative durations, distances
        duration_cum_list = np.cumsum(duration_s)
    
    else:
        list_coords = sp
        duration_cum_list = []
        total_distance=0

    if projection == "MERCATOR":
        list_coords = [wgs84_to_web_mercator(*p) for p in list_coords]

    return list_coords, duration_cum_list, total_distance


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
    linestring.append((G.node[list_ids[-1]]["x"], G.node[list_ids[-1]]["y"]))

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
    return nx.shortest_path(G, source=o, target=d)


def get_random_node(G):
    """Find random node in G

    Arguments:
        G {networkx} -- Transportation network

    Returns:
        [tuple] -- node id, lon, lat
    """
    random_node = random.randint(0, nx.number_of_nodes(G) - 1)
    return random_node, G.node[random_node]["x"], G.node[random_node]["y"]


def get_coords_node(n, G):
    return G.node[n]["x"], G.node[n]["y"]


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
    print("Size of strongly connected components:", s_connected_component)
    return set(largest_cc)


# #################################################################### #
# Distances ########################################################## #
# #################################################################### #


def distance(lon1, lat1, lon2, lat2):

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

    x = (
        A * math.cos(rad_lat_o) * math.cos(rad_lon_o)
        + B * math.cos(rad_lat_d) * math.cos(rad_lon_d)
    )
    y = (
        A * math.cos(rad_lat_o) * math.sin(rad_lon_o)
        + B * math.cos(rad_lat_d) * math.sin(rad_lon_d)
    )
    z = A * math.sin(rad_lat_o) + B * math.sin(rad_lat_d)

    lat_atan2 = math.atan2(z, math.sqrt(x * x + y * y))
    lon_atan2 = math.atan2(y, x)

    lat = math.degrees(lat_atan2)
    lon = math.degrees(lon_atan2)

    return lon, lat

def get_distance_matrix(G, distance_dic_m):
    """Return distance matrix (n x n). Value is 'None' when path does
    not exist

    Arguments:
        G {networkx} -- Graph to loop nodes
        distance_dic_m {dic} -- previosly calculated distance dictionary

    Returns:
        [list[list[float]]] -- Distance matrix
    """
    # TODO simplify - test:  nx.shortest_path_length(G, source=o,
    # target=d, weight="length")

    # Creating distance matrix
    dist_matrix = []
    for from_node in range(0, get_number_of_nodes(G)):
        to_distance_list = []
        for to_node in range(0, get_number_of_nodes(G)):

            try:
                dist_km = distance_dic_m[from_node][to_node]
                to_distance_list.append(dist_km)
            except:
                to_distance_list.append(None)

        dist_matrix.append(to_distance_list)

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
        print(e)
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
        print(
            "\nTrying to read distance data from file:\n'{}'.".format(
                root_path
            )
        )
        distance_dic_m = np.load(root_path).item()

    except:
        print("Reading failed! Calculating shortest paths...")
        all_dists_gen = nx.all_pairs_dijkstra_path_length(G, weight="length")

        # Save with pickle (meters)
        distance_dic_m = dict(all_dists_gen)
        np.save(root_path, distance_dic_m)

    print(
        "Distance data loaded successfully. #Nodes:",
        len(distance_dic_m.values()),
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
    max_delay=180,
    log_path=None,
    time_limit=None,
    round_trip=False
):

    # List of nodes ids
    node_ids = sorted(list(reachability_dic.keys()))

    try:

        # Create a new model
        m = Model("region_centers")

        if log_path:

            # Create log path if not exists
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            
            round_trip_label = ("_round_trip" if round_trip else "")

            m.Params.LogFile = "{}/region_centers{}_{}.log".format(
                log_path, round_trip_label, max_delay
            )

            m.Params.ResultFile = "{}/region_centers{}_{}.lp".format(
                log_path, round_trip_label, max_delay
            )

        # xi = 1, if vertex Vi is used as a region center
        # and 0 otherwise
        x = m.addVars(node_ids, vtype=GRB.BINARY, name="x")

        # Ensures that every node in the road network graph is reachable
        # within 'max_delay' travel time by at least one region center
        # selected from the nodes in the graph.
        # To extract the region centers, we select from V all vertices
        # V[i] such that x[i] = 1.

        for origin in node_ids:
            m.addConstr(
                (
                    quicksum(
                        x[center]
                        * can_reach(
                            center,
                            origin,
                            max_delay,
                            reachability_dic,
                            round_trip=round_trip
                        )
                        for center in node_ids
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
                print("TIME LIMIT ({} s) RECHEADED.".format(time_limit))

            # Sweep x_n = 1 variables to create list of region centers
            var_x = m.getAttr("x", x)
            for n in node_ids:
                if var_x[n] > 0.0001:
                    region_centers.append(n)

            return region_centers

        elif is_unfeasible:

            print("Model is infeasible.")
            raise Exception("Model is infeasible.")
            # exit(0)

        elif (
            m.status != GRB.Status.INF_OR_UNBD
            and m.status != GRB.Status.INFEASIBLE
        ):
            print("Optimization was stopped with status %d" % m.status)
            raise Exception("Model is infeasible.")

    except GurobiError as e:
        raise Exception(" Gurobi error code " + str(e.errno))

    except AttributeError as e:
        raise Exception("Encountered an attribute error:" + str(e))


def get_region_centers(
    steps,
    path_region_centers,
    reachability_dic,
    root_path=None,
    time_limit=60,
    round_trip=False
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
    if os.path.isfile(path_region_centers):
        centers_dic = np.load(path_region_centers).item()
        print(
            "\nReading region center dictionary...\nSource: '{}'.".format(
                path_region_centers
            )
        )

    else:

        print(
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
        for max_delay in sorted(steps):

            if centers_sub_sols is not None:
                # Name of intermediate region centers file for 'max_delay'
                file_name = "{}/{}.npy".format(centers_sub_sols, max_delay)

                # Pre-calculated region center is loaded in case it exists.
                # This helps filling the complete 'centers_dic' without
                # starting the process from the beginning, in case an error
                # has occured.
                if os.path.isfile(file_name):
                    # Load max delay in centers_dic
                    centers_dic[max_delay] = np.load(file_name)
                    print(file_name, "already calculated.")
                    continue

            try:
                # Find the list of centers for max_delay
                centers = ilp_node_reachability(
                    reachability_dic,
                    max_delay=max_delay,
                    log_path=centers_gurobi_log,
                    time_limit=time_limit,
                    round_trip=round_trip
                )
            except Exception as e:
                print(e)
            else:
                centers_dic[max_delay] = centers
                print(
                    "Max. delay: {} = # Nodes: {}".format(
                        max_delay, len(centers)
                    )
                )

                # Save intermediate steps (region centers of 'max_delay')
                if root_path:
                    np.save(file_name, centers)

                np.save(path_region_centers, centers_dic)

    return centers_dic
