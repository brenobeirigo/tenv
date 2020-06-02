import os
import sys

import json
from pprint import pprint

# Adding project folder
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import tenv.config as config
import tenv.network as gen
import tenv.demand as tp
import tenv.visuals as vi
import tenv.util as util
import pandas as pd
from collections import defaultdict

config.make_folders()

# Plot region centers (blue) and associated nodes
print(f"Plotting regions at '{config.root_img_regions}'...")
vi.plot_regions(
    util.G,
    util.region_centers,
    util.region_id_dict,
    path=config.root_img_regions,
    show=False,
    file_format="png",
    replace=True,
)

print(f"Plotting region neighbors at {config.root_img_neighbors}...")
vi.plot_region_neighbors(
    util.G,
    util.region_centers,
    util.sorted_neighbors,
    path=config.root_img_neighbors,
    show=False,
    file_format="png",
    max_neighbors=4,
    replace=True,
)
