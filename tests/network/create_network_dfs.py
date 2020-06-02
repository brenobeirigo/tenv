import os
import sys

# Adding project folder
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import tenv.config as config
import tenv.util as util
import pandas as pd
from collections import defaultdict

# Assure all folders are created
config.make_folders()

# #################################################################### #
# ##### Region center count ########################################## #
# #################################################################### #

# level, center_count
dists, centers = list(zip(*(util.region_centers.items())))
df = pd.DataFrame({"level": dists, "centers": centers})
df["center_count"] = df["level"].apply(lambda x: len(util.region_centers[x]))
df.sort_values(by=["level"], inplace=True)
df[["level", "center_count", "centers"]].to_csv(
    config.root_network_info + "/region_centers.csv", index=False
)

# #################################################################### #
# ##### Node level id ################################################ #
# #################################################################### #

# node, level_0, level_1, level_2, ...
dict_node_level_id = defaultdict(list)
for node, level_id in util.region_id_dict.items():
    dict_node_level_id["node"].append(node)
    for level, id_level in sorted(level_id.items(), key=lambda x: (x[0],)):
        dict_node_level_id[level].append(id_level)

df_node_level_id = pd.DataFrame.from_dict(dict(dict_node_level_id))
df_node_level_id.sort_values(by=["node"], inplace=True)
df_node_level_id.to_csv(
    config.root_network_info + "/node_level_id.csv", index=False
)


# #################################################################### #
# ##### Center children list ######################################### #
# #################################################################### #

# level, center, n_childreen, children
dict_level_center_children = defaultdict(list)
for level, center_nodes in util.center_nodes.items():
    for center, children in center_nodes.items():
        dict_level_center_children["level"].append(level)
        dict_level_center_children["center"].append(center)
        dict_level_center_children["n_children"].append(len(children))

        dict_level_center_children["children"].append(list(children))

df_level_center_children = pd.DataFrame.from_dict(
    dict(dict_level_center_children)
)
df_level_center_children.sort_values(
    by=["level", "center", "n_children"], inplace=True,
)
df_level_center_children.to_csv(
    config.root_network_info + "/level_center_children.csv", index=False
)


# #################################################################### #
# ##### Level center neighbors ascending ############################# #
# #################################################################### #

# level, center, n_childreen, children
dict_level_center_sorted_neighbors_list = defaultdict(list)
for level, center_nodes in util.sorted_neighbors.items():
    for center, children in center_nodes.items():
        dict_level_center_sorted_neighbors_list["level"].append(level)
        dict_level_center_sorted_neighbors_list["center"].append(center)
        dict_level_center_sorted_neighbors_list["n_neighbors"].append(
            len(children)
        )

        dict_level_center_sorted_neighbors_list["neighbors_asc"].append(
            list(children)
        )

df_level_center_children = pd.DataFrame.from_dict(
    dict(dict_level_center_sorted_neighbors_list)
)
df_level_center_children.sort_values(
    by=["level", "center", "n_neighbors"], inplace=True,
)
df_level_center_children.to_csv(
    config.root_network_info + "/level_center_neighbors_asc.csv", index=False,
)
