import functools
import json
import sys
import os
from flask import Flask, jsonify
from pprint import pprint
import copy

# REST - WSGI + Flask
from waitress import serve


# Adding project folder
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import tenv.config as config
import tenv.network as nw
import tenv.demand as tp

print(
    "\n###############################################################"
    f"\n# GRAPH = {config.graph_file_name}"
    "\n###############################################################"
)

# Network
G = nw.load_network(config.graph_file_name, folder=config.root_map)
print(
    "\n# NETWORK -  NODES: {} ({} -> {}) -- #EDGES: {}".format(
        len(G.nodes()), min(G.nodes()), max(G.nodes()), len(G.edges())
    )
)

# Creating distance dictionary [o][d] -> distance
distance_dic = nw.get_distance_dic(config.path_dist_dic, G)

# Reachability dictionary
reachability_dict = nw.get_reachability_dic(
    config.path_reachability_dic,
    distance_dic,
    step=30,
    total_range=600,
    speed_km_h=30,
)

# All region centers
region_centers = nw.get_region_centers(
    config.path_region_centers,
    reachability_dict,
    root_path=config.root_reachability,
)

# print("#Centers per trip duration:")
# print({dist: len(region_centers[dist]) for dist in region_centers})
# pprint(region_centers)

# What is the closest region center of every each
# node (given a time limit)?
region_id_dict = nw.get_region_ids(
    G,
    reachability_dict,
    region_centers,
    path_region_ids=config.path_region_center_ids,
)

node_region_ids = nw.get_node_region_ids(G, region_id_dict)


sorted_neighbors = nw.get_sorted_neighbors(
    G, region_centers, path_sorted_neighbors=config.path_sorted_neighbors
)

center_nodes = nw.get_center_nodes(region_id_dict)

app = Flask(__name__)


@app.route("/sp/<int:o>/<int:d>")
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
    return ";".join(map(str, nw.get_sp(G, o, d)))


@app.route("/sp_coords/<int:o>/<int:d>")
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
    return ";".join(map(str, nw.get_sp_coords(G, o, d)))


@app.route("/distance_meters/<int:o>/<int:d>")
def get_distance(o, d):
    return str(distance_dic[o][d])


@app.route("/can_reach/<int:n>/<int:t>")
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

    return ";".join(map(str, nw.get_can_reach_set(n, reachability_dict, t)))


@app.route(
    "/linestring_style/<int:o>/<int:d>/"
    "<stroke>/<float:width>/<float:opacity>"
)
def linestring_style(o, d, stroke, width, opacity):
    """Get linestring between origin (o) and destination (d)

    Parameters
    ----------
    o : int
        Origin id
    d : int
        Destination id
    stroke : str
        Color (%23 replaces #)
    width : float
        Width line
    opacity : float
        Opacity line

    Returns
    -------
    geojson
        Geojson linestring with properties stroke, width, and opacity

    How to check?
    -------------
    Add address in browser, example:
    http://localhost:4999/linestring_style/1/900/%23FF0000/2.0/1.0

    Visualize in http://geojson.io
    """

    return jsonify(
        nw.get_linestring(
            G,
            o,
            d,
            **{
                "stroke": stroke,
                "stroke-width": width,
                "stroke-opacity": opacity,
            },
        )
    )


@functools.lru_cache(maxsize=None)
@app.route("/nodes")
def nodes():
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
    nodes = [
        {"id": id, "x": G.node[id]["x"], "y": G.node[id]["y"]}
        for id in G.nodes()
    ]
    dic = dict(nodes=nodes)
    return jsonify(dic)


@functools.lru_cache(maxsize=None)
@app.route("/center_elements/<int:max_dist>/<int:center>")
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
        return ";".join(map(str, nodes))
    except Exception as e:
        print(
            f"ERROR({e})!\n (center={center}, max_dist={max_dist}) does "
            " not exist!"
        )
        return


@functools.lru_cache(maxsize=None)
@app.route("/nodes_level/<int:time_limit>")
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
    nodes = [str(region_id_dict[node_id][time_limit]) for node_id in G.nodes()]
    return ";".join(nodes)


@functools.lru_cache(maxsize=None)
@app.route("/node_region_ids")
def get_node_region_ids():
    """Get list of node ids for each region (defined with maximum
    reachable time)

    Returns
    -------
    dict
        Dictionary of maximum reachable time keys and node id lists.
    """

    return jsonify(dict(node_region_ids))


@functools.lru_cache(maxsize=None)
@app.route("/node_region_ids/<int:step>")
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

    # Removing distances which are not multiples of "step"
    for k in min_reachable_time:
        if k % step != 0:
            del cut_node_region_ids[k]
    return jsonify(dict(cut_node_region_ids))


@functools.lru_cache(maxsize=None)
@app.route("/point_style/<int:p>/<color>/<size>/<symbol>")
def point_style(p, color, size, symbol):
    # E.g.: http://127.0.0.1:4999/point_style/1/%23FF0000/small/circle
    return jsonify(
        nw.get_point(
            G,
            p,
            **{
                "marker-color": color,
                "marker-size": size,
                "marker-symbol": symbol,
            },
        )
    )


@functools.lru_cache(maxsize=None)
@app.route("/neighbors/<int:node>/<int:degree>/<direction>")
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
    return ";".join(map(str, node_neighbors))


@functools.lru_cache(maxsize=None)
@app.route("/centers/<int:time_limit>")
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

    return ";".join(map(str, region_centers[time_limit]))


@functools.lru_cache(maxsize=None)
@app.route("/region_id/<int:time_limit>/<int:node_id>")
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

    return str(region_id_dict[node_id][time_limit])


@functools.lru_cache(maxsize=None)
@app.route(
    "/center_neighbors/<int:time_limit>/<int:center_id>/<int:n_neighbors>"
)
def get_center_neighbors(time_limit, center_id, n_neighbors):
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
    return ";".join(
        [
            str(neighbor_id)
            for neighbor_id, distance in sorted_neighbors[time_limit][
                center_id
            ]
            if distance != 0
        ][:n_neighbors]
    )


@app.route(
    "/point_info/<int:p>/<color>/<size>/<symbol>/<arrival>/<departure>/"
    "<passenger_count>/<vehicle_load>/<customer_class>"
)
def point_info(
    p,
    color,
    size,
    symbol,
    arrival,
    departure,
    passenger_count,
    vehicle_load,
    user_class,
):
    # E.g.: http://127.0.0.1:4999/point_style/1/%23FF0000/small/circle
    return jsonify(
        nw.get_point(
            G,
            p,
            **{
                "marker-color": color,
                "marker-size": size,
                "marker-symbol": symbol,
                "arrival": arrival,
                "departure": departure,
                "passenger-count": passenger_count,
                "vehicle-load": vehicle_load,
                "user-class": user_class,
            },
        )
    )


@functools.lru_cache(maxsize=None)
@app.route("/location/<int:id>")
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

    return jsonify({"location": {"x": G.node[id]["x"], "y": G.node[id]["y"]}})


if __name__ == "__main__":

    serve(app, listen="*:4999")  # With WSGI
    # app.run(port='4999') # No  WSGI (only flask)
