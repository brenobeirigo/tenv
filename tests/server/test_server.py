import requests
from pprint import pprint
from collections import namedtuple

port = 4999
url = f"http://localhost:{port}/"

Point = namedtuple("Point", ["id", "x", "y"])
Coordinate = namedtuple("Coordinate", ["x", "y"])


########################################################################
# List the points (id, x, y) ###########################################
########################################################################

points = list()
r = requests.get(url=f"{url}nodes")
nodes = r.json()["nodes"]
for n in nodes:
    points.append(Point(n["id"], n["x"], n["y"]))

pprint(points)

########################################################################
# List the coordinates (x, y) ##########################################
########################################################################

coords = list()
for p in points:
    r = requests.get(url=f"{url}location/{p.id}")
    coord = r.json()["location"]
    coords.append(Coordinate(coord["x"], coord["y"]))

pprint(coords)


########################################################################
# List the points that can reach each point within distance ############
########################################################################

distance = 30
for p in points:
    r = requests.get(url=f"{url}can_reach/{p.id}/{distance}")

    # Nodes that can reach Point p within distance
    can_reach_set = list(map(int, r.text.split(";")))
    print(p, can_reach_set)

########################################################################
# List the points that can reach each point within distance ############
########################################################################

distance = 30
for p in points:
    r = requests.get(url=f"{url}can_reach/{p.id}/{distance}")

    # Nodes that can reach Point p within distance
    can_reach_set = list(map(int, r.text.split(";")))
    for o in can_reach_set:
        rsp = requests.get(url=f"{url}sp/{o}/{p.id}")
        sp = list(map(int, rsp.text.split(";")))
        print(f"{o:>4}->{p.id:>4} - {sp}")
