import os
import sys
import pprint
from collections import defaultdict

# Adding project folder
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import tenv.config as config
import tenv.network as nw

if __name__ == "__main__":

    print(config.info())

    # Network
    G = nw.load_network(config.graph_file_name, folder=config.root_map)

    distance_matrix = nw.get_distance_matrix(config.path_dist_matrix_npy, G)

    # Reachability dictionary: Which nodes can access node n?
    reachability_dict, steps = nw.get_reachability_dic(
        config.path_reachability_dic,
        distance_matrix,
        step=config.step,
        total_range=config.total_range,
        speed_km_h=config.speed_km_h,
        step_list=config.step_list,
    )

    if config.region_slice == config.REGION_CONCENTRIC:

        region_id_dict, region_centers = nw.concentric_regions(
            G,
            config.step_list_concentric,
            reachability_dict,
            list(G.nodes()),
            center=-1,
            root_reachability=config.root_reachability,
        )

        center_nodes = nw.get_center_nodes(region_id_dict)

        print("STEP CONCENTRIC", config.step_list_concentric)
        for max_dist, center_id_nodes in center_nodes.items():
            print(f"## {max_dist} ##########################################")
            for center_id, nodes in center_id_nodes.items():
                d = {}
                for step in config.step_list_concentric:
                    d[step] = len(
                        set([region_id_dict[n][step] for n in nodes])
                    )
                count = "-".join([f"{s:>4}={rc_count:>2}" for s, rc_count in d.items()])
                print(f"[{max_dist:>4}] Center id: {center_id:>4} - #nodes={len(nodes):>3} - step:{count}")

# STEP CONCENTRIC [60, 300, 600]
# ## 600 ##########################################
# [ 600] Center id: 6304 - #nodes=670 - step:  60=71- 300= 5- 600= 1
# [ 600] Center id: 6115 - #nodes=432 - step:  60=59- 300= 4- 600= 1
# [ 600] Center id: 6429 - #nodes=221 - step:  60=27- 300= 2- 600= 1
# [ 600] Center id: 6401 - #nodes=622 - step:  60=65- 300= 3- 600= 1
# [ 600] Center id: 6392 - #nodes=346 - step:  60=49- 300= 4- 600= 1
# [ 600] Center id: 6415 - #nodes=479 - step:  60=67- 300= 3- 600= 1
# [ 600] Center id: 6397 - #nodes=358 - step:  60=50- 300= 2- 600= 1
# [ 600] Center id: 6425 - #nodes=152 - step:  60=21- 300= 3- 600= 1
# [ 600] Center id: 6420 - #nodes=480 - step:  60=57- 300= 2- 600= 1
# [ 600] Center id: 6277 - #nodes= 49 - step:  60=12- 300= 2- 600= 1
# [ 600] Center id: 6404 - #nodes=209 - step:  60=39- 300= 3- 600= 1
# [ 600] Center id: 6373 - #nodes=523 - step:  60=74- 300= 3- 600= 1
# [ 600] Center id: 6419 - #nodes=309 - step:  60=38- 300= 3- 600= 1
# [ 600] Center id: 6426 - #nodes=303 - step:  60=36- 300= 5- 600= 1
# [ 600] Center id: 6409 - #nodes=385 - step:  60=43- 300= 3- 600= 1
# [ 600] Center id: 6421 - #nodes=131 - step:  60=25- 300= 2- 600= 1
# [ 600] Center id: 6058 - #nodes=259 - step:  60=43- 300= 5- 600= 1
# [ 600] Center id: 6383 - #nodes=402 - step:  60=45- 300= 4- 600= 1
# [ 600] Center id: 5664 - #nodes= 51 - step:  60=15- 300= 3- 600= 1
# [ 600] Center id: 5807 - #nodes= 22 - step:  60= 8- 300= 2- 600= 1
# [ 600] Center id: 6230 - #nodes= 27 - step:  60= 8- 300= 2- 600= 1
# ## 300 ##########################################
# [ 300] Center id: 6304 - #nodes=235 - step:  60=22- 300= 1- 600= 1
# [ 300] Center id: 5979 - #nodes= 50 - step:  60= 8- 300= 1- 600= 1
# [ 300] Center id:  882 - #nodes=323 - step:  60=29- 300= 1- 600= 1
# [ 300] Center id: 6205 - #nodes= 51 - step:  60= 8- 300= 1- 600= 1
# [ 300] Center id: 4837 - #nodes= 11 - step:  60= 4- 300= 1- 600= 1
# [ 300] Center id: 3153 - #nodes=177 - step:  60=22- 300= 1- 600= 1
# [ 300] Center id:  276 - #nodes=231 - step:  60=29- 300= 1- 600= 1
# [ 300] Center id: 4601 - #nodes= 12 - step:  60= 4- 300= 1- 600= 1
# [ 300] Center id: 5468 - #nodes= 12 - step:  60= 4- 300= 1- 600= 1
# [ 300] Center id: 4506 - #nodes= 21 - step:  60= 5- 300= 1- 600= 1
# [ 300] Center id:  310 - #nodes=200 - step:  60=22- 300= 1- 600= 1
# [ 300] Center id: 1207 - #nodes=298 - step:  60=28- 300= 1- 600= 1
# [ 300] Center id:  626 - #nodes=120 - step:  60=16- 300= 1- 600= 1
# [ 300] Center id:  674 - #nodes=204 - step:  60=21- 300= 1- 600= 1
# [ 300] Center id:  805 - #nodes=131 - step:  60=18- 300= 1- 600= 1
# [ 300] Center id:  409 - #nodes=116 - step:  60=15- 300= 1- 600= 1
# [ 300] Center id: 1171 - #nodes= 87 - step:  60=12- 300= 1- 600= 1
# [ 300] Center id: 3274 - #nodes= 12 - step:  60= 4- 300= 1- 600= 1
# [ 300] Center id: 1488 - #nodes=198 - step:  60=24- 300= 1- 600= 1
# [ 300] Center id: 1382 - #nodes=187 - step:  60=27- 300= 1- 600= 1
# [ 300] Center id: 1724 - #nodes= 94 - step:  60=16- 300= 1- 600= 1
# [ 300] Center id:   61 - #nodes=200 - step:  60=29- 300= 1- 600= 1
# [ 300] Center id: 3615 - #nodes=158 - step:  60=21- 300= 1- 600= 1
# [ 300] Center id: 6425 - #nodes=130 - step:  60=14- 300= 1- 600= 1
# [ 300] Center id: 4953 - #nodes=  9 - step:  60= 3- 300= 1- 600= 1
# [ 300] Center id: 4756 - #nodes= 13 - step:  60= 4- 300= 1- 600= 1
# [ 300] Center id: 4258 - #nodes=247 - step:  60=33- 300= 1- 600= 1
# [ 300] Center id: 1925 - #nodes=233 - step:  60=24- 300= 1- 600= 1
# [ 300] Center id: 6277 - #nodes= 41 - step:  60= 9- 300= 1- 600= 1
# [ 300] Center id: 5169 - #nodes=  8 - step:  60= 3- 300= 1- 600= 1
# [ 300] Center id:   75 - #nodes=116 - step:  60=18- 300= 1- 600= 1
# [ 300] Center id: 4297 - #nodes= 81 - step:  60=17- 300= 1- 600= 1
# [ 300] Center id: 3377 - #nodes= 12 - step:  60= 4- 300= 1- 600= 1
# [ 300] Center id:  763 - #nodes=127 - step:  60=17- 300= 1- 600= 1
# [ 300] Center id: 3749 - #nodes=196 - step:  60=26- 300= 1- 600= 1
# [ 300] Center id: 2370 - #nodes=200 - step:  60=31- 300= 1- 600= 1
# [ 300] Center id: 3265 - #nodes= 37 - step:  60= 7- 300= 1- 600= 1
# [ 300] Center id: 4021 - #nodes=243 - step:  60=23- 300= 1- 600= 1
# [ 300] Center id: 3262 - #nodes= 29 - step:  60= 8- 300= 1- 600= 1
# [ 300] Center id: 6426 - #nodes= 36 - step:  60= 7- 300= 1- 600= 1
# [ 300] Center id: 2547 - #nodes=149 - step:  60=10- 300= 1- 600= 1
# [ 300] Center id: 1018 - #nodes= 95 - step:  60=11- 300= 1- 600= 1
# [ 300] Center id: 4688 - #nodes= 12 - step:  60= 4- 300= 1- 600= 1
# [ 300] Center id: 6266 - #nodes= 11 - step:  60= 4- 300= 1- 600= 1
# [ 300] Center id: 5182 - #nodes= 59 - step:  60=11- 300= 1- 600= 1
# [ 300] Center id: 1989 - #nodes=315 - step:  60=28- 300= 1- 600= 1
# [ 300] Center id: 5059 - #nodes= 11 - step:  60= 4- 300= 1- 600= 1
# [ 300] Center id: 1931 - #nodes= 84 - step:  60=16- 300= 1- 600= 1
# [ 300] Center id:   68 - #nodes= 47 - step:  60= 9- 300= 1- 600= 1
# [ 300] Center id:   96 - #nodes= 57 - step:  60= 9- 300= 1- 600= 1
# [ 300] Center id:  823 - #nodes=126 - step:  60=15- 300= 1- 600= 1
# [ 300] Center id: 3528 - #nodes= 42 - step:  60= 8- 300= 1- 600= 1
# [ 300] Center id: 3271 - #nodes= 18 - step:  60= 5- 300= 1- 600= 1
# [ 300] Center id: 4821 - #nodes= 16 - step:  60= 6- 300= 1- 600= 1
# [ 300] Center id: 2217 - #nodes=253 - step:  60=26- 300= 1- 600= 1
# [ 300] Center id:  610 - #nodes=126 - step:  60=11- 300= 1- 600= 1
# [ 300] Center id: 5104 - #nodes= 12 - step:  60= 4- 300= 1- 600= 1
# [ 300] Center id: 5253 - #nodes= 11 - step:  60= 4- 300= 1- 600= 1
# [ 300] Center id: 5532 - #nodes= 28 - step:  60= 7- 300= 1- 600= 1
# [ 300] Center id: 5664 - #nodes= 12 - step:  60= 4- 300= 1- 600= 1
# [ 300] Center id: 5538 - #nodes= 11 - step:  60= 4- 300= 1- 600= 1
# [ 300] Center id: 6138 - #nodes= 13 - step:  60= 5- 300= 1- 600= 1
# [ 300] Center id: 5807 - #nodes=  9 - step:  60= 3- 300= 1- 600= 1
# [ 300] Center id: 6230 - #nodes= 15 - step:  60= 4- 300= 1- 600= 1
# [ 300] Center id: 5755 - #nodes= 12 - step:  60= 4- 300= 1- 600= 1
# ## 60 ##########################################
# [  60] Center id: 3898 - #nodes=  8 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id: 3550 - #nodes= 23 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id: 4128 - #nodes= 12 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id:  958 - #nodes= 10 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id: 4132 - #nodes= 13 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id: 2672 - #nodes=  8 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id:   46 - #nodes= 11 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id: 3321 - #nodes=  9 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id:  100 - #nodes= 13 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id: 2998 - #nodes=  9 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id: 3542 - #nodes=  3 - step:  60= 1- 300= 1- 600= 1
# ...
# [  60] Center id: 3345 - #nodes=  6 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id: 6230 - #nodes=  3 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id: 3346 - #nodes=  4 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id: 5755 - #nodes=  3 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id: 5758 - #nodes=  2 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id: 6013 - #nodes=  3 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id: 6016 - #nodes=  3 - step:  60= 1- 300= 1- 600= 1
# [  60] Center id: 6091 - #nodes=  3 - step:  60= 1- 300= 1- 600= 1