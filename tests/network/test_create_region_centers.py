import json
from collections import defaultdict

import pandas as pd

nodeset_info_path = (
    "C:/Users/LocalAdmin/OneDrive/leap_forward/street_network_server/tenv/data/out/manhattan"
    "/network_info/nodeset_info.json"
)

with open(nodeset_info_path) as js:
    nodeset = json.load(js)
    nodeset_dict = dict()
    for node in nodeset["nodes"]:
        nodeset_dict[node["id"]] = node

    step_center_id = "150"
    # print(nodeset_dict)
    from_to_center = defaultdict(list)
    for node, node_data in nodeset_dict.items():
        _, step_center_dict, x, y = node_data.values()
        center_id = step_center_dict[step_center_id]
        center_x = nodeset_dict[center_id]["x"]
        center_y = nodeset_dict[center_id]["y"]
        # print(node, x, y, center_id, center_x, center_y)
        from_to_center["id"].append(node)
        from_to_center["node_lon"].append(x)
        from_to_center["node_lat"].append(y)
        from_to_center["center_id"].append(center_id)
        from_to_center["center_lon"].append(center_x)
        from_to_center["center_lat"].append(center_y)

    df = pd.DataFrame.from_dict(from_to_center)
    df.to_csv(f"from_to_center_{step_center_id}.csv", index=False)
