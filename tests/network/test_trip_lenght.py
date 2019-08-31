# Compare edge durations of original and enriched graphs
# Generate network pics and duration histogram

import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Adding project folder
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import tenv.config as config
import tenv.network as nw


def get_edge_durations(G):
    list_durations = list()
    for o, d in G.edges():
        dur = nw.get_duration(G.edges[o, d, 0]["length"], speed_km_h=20)
        list_durations.append(dur)
    return list_durations


# Original map (No enrichment)
path = (
    "C:\\Users\\LocalAdmin\OneDrive\\leap_forward\\"
    + "street_network_server\\tenv\\data\\out\\"
)

graph_file_name = "manhattan-island-new-york-city-new-york-usa.graphml"

original_map = (
    path + "ORIGINAL_manhattan-island-new-york-city-new-york-usa\\map"
)
G1 = nw.load_network(graph_file_name, folder=original_map)
print(nw.get_graph_info(G1))

enriched_edges_map = path + "manhattan-island-new-york-city-new-york-usa\\map"
G2 = nw.load_network(graph_file_name, folder=enriched_edges_map)
print(nw.get_graph_info(G2))

# Saving maps for comparison
config_fig = dict(fig_height=15, node_size=2, file_format="png")
nw.save_graph_pic(G1, path, config=config_fig, label="original_")
nw.save_graph_pic(G2, path, config=config_fig, label="30s_")

print(config.info())

list_durations_g1 = get_edge_durations(G1)
list_durations_g2 = get_edge_durations(G2)


# Plot histogram ##################################################### #

# Set up the matplotlib figure
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


yrange = [1, 10, 100, 1000, 3500]

bins = list(range(0, 355, 5))
sns.distplot(list_durations_g1, bins=bins, kde=False, color="r", ax=ax1)
ax1.set_yscale("log")
ax1.set_ylabel("#Nodes")
ax1.set_yticks(yrange)
ax1.set_yticklabels(yrange)
ax1.set_xlabel("Seconds")

bins2 = list(range(0, 35, 5))
sns.distplot(list_durations_g2, bins=bins2, kde=False, color="b", ax=ax2)
ax2.set_xlabel("Seconds")
ax2.set_xticks(bins2)
ax2.set_xticklabels(bins2)

plt.show()
plt.tight_layout()
