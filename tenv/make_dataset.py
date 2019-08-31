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

import logging

logging.basicConfig(level=logging.INFO)


def gen_random_data(config, G, distance_dic):
    print(
        "\n############################"
        "## Generating random data ##"
        "############################"
    )

    print("Trip data generation settings:")
    pprint(config.tripdata["data_gen"])

    # Loop all generation function (e.g., clone, cluster)
    for random_func_name in config.tripdata["data_gen"]["funcs"]:

        if random_func_name == "random_clone":

            data_gen_path_tripdata_ids = "{}/{}_{}_ids.csv".format(
                config.root_tripdata,
                random_func_name,
                config.get_excerpt_name(
                    config.tripdata["data_gen"]["start"],
                    config.tripdata["data_gen"]["stop"],
                ),
            )

            tp.gen_requests(
                config.tripdata["data_gen"]["source"],
                config.tripdata["data_gen"]["max_passenger_count"],
                G,
                data_gen_path_tripdata_ids,
                start_timestamp=config.tripdata["data_gen"]["start"],
                end_timestamp=config.tripdata["data_gen"]["stop"],
                distance_dic=distance_dic,
            )


def process_tripdata(config, G, distance_dic):

    # Get excerpt (start, stop)
    print("Cleaning trip data...")

    for file_name, tws in config.tripdata["file_tw"].items():
        for tw in tws:
            earliest, latest = tw

            # Cleaned data setup
            output_cleaned = config.tripdata["output_cleaned_tripdata"]
            file_name_cleaned = (
                config.get_excerpt_name(earliest, latest, label="cleaned")
                + ".csv"
            )

            dt_tripdata = tp.get_trip_data(
                f'{config.tripdata["path_tripdata"]}{file_name}',
                output_cleaned + file_name_cleaned,
                earliest,
                latest,
                index_col=config.tripdata["index_col"],
                filtered_columns=config.tripdata["filtered_columns"],
            )

            # Cleaned data + graph ids setup
            output_ids = config.tripdata["output_ids_tripdata"]
            file_name_ids = (
                config.get_excerpt_name(earliest, latest, label="ids") + ".csv"
            )
            #  street network node ids (from G) to tripdata
            print("Adding ids...")
            tp.add_ids(
                output_cleaned + file_name_cleaned,
                output_ids + file_name_ids,
                G,
                distance_dic,
            )


def download_tripdata(config, G, distance_dic):
    """ Example .json file:
        {
            "region": "Manhattan Island, New York City, New York, USA",
            "url_tripdata": "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2011-02.csv",
            "start": "2011-2-01 00:00:00",
            "stop": "2011-2-01 23:59:59"
        }
    """
    print(
        "\n############################"
        "### Processing trip data ###"
        "############################"
    )
    pprint(config.tripdata["url_tripdata"])

    # Try downloading the raw data if not exists (NY)
    tp.download_file(
        config.tripdata["url_tripdata"],
        config.root_tripdata,
        config.tripdata_filename,
    )

    # Get excerpt (start, stop)
    print("Cleaning trip data...")
    dt_tripdata = tp.get_trip_data(
        config.path_tripdata_source,
        config.path_tripdata,
        start=config.tripdata["start"],
        stop=config.tripdata["stop"],
    )

    #  street network node ids (from G) to tripdata
    print("Adding ids...")
    tp.add_ids(config.path_tripdata, config.path_tripdata_ids, G, distance_dic)


def create_trip_data():
    print(
        "###############################"
        "###### TRIP DATA SANDBOX ######"
        "###############################"
    )

    print("Creating trip data for '{}'.".format(config.region))

    if "url_tripdata" in config.tripdata.keys():
        print(
            "Using request interval from {} to {}.".format(
                config.tripdata["start"], config.tripdata["stop"]
            )
        )

    # Create all folders where data will be saved
    if not os.path.exists(config.root_path):
        os.makedirs(config.root_path)

    if not os.path.exists(config.root_dist):
        os.makedirs(config.root_dist)

    if not os.path.exists(config.root_map):
        os.makedirs(config.root_map)

    if not os.path.exists(config.root_reachability):
        os.makedirs(config.root_reachability)

    if not os.path.exists(config.root_reachability_concentric):
        os.makedirs(config.root_reachability_concentric)

    # Trip data ###################################################### #

    if not os.path.exists(config.root_tripdata):
        os.makedirs(config.root_tripdata)

    if not os.path.exists(config.root_tripdata_raw):
        os.makedirs(config.root_tripdata_raw)

    if not os.path.exists(config.root_tripdata_cleaned):
        os.makedirs(config.root_tripdata_cleaned)

    if not os.path.exists(config.root_tripdata_ids):
        os.makedirs(config.root_tripdata_ids)

    # Plots ########################################################## #
    if not os.path.exists(config.root_img_regions):
        os.makedirs(config.root_img_regions)

    if not os.path.exists(config.root_img_regions_concentric):
        os.makedirs(config.root_img_regions_concentric)

    if not os.path.exists(config.root_img_neighbors):
        os.makedirs(config.root_img_neighbors)

    if not os.path.exists(config.root_img_neighbors_concentric):
        os.makedirs(config.root_img_neighbors_concentric)

    print(
        (
            "\n>>>>> Target folders:\n"
            + "\n - Distance matrix (csv) and dictionary (npy): {}"
            + "\n -   Data excerpt from NYC taxi dataset (csv): {}"
            + "\n -  Reachability (npy) & region centers (npy): {}.\n"
        ).format(
            config.root_dist, config.root_tripdata, config.root_reachability
        )
    )

    print(
        "\n############################"
        "##### Loading network ######"
        "############################"
    )

    # Get network graph and save
    G = gen.get_network_from(
        config.region,
        config.root_map,
        config.graph_name,
        config.graph_file_name,
        max_travel_time_edge=config.max_travel_time_edge,
        speed_km_h=config.speed_km_h,
    )

    gen.save_graph_pic(G, config.root_map)

    print(
        "\n############################"
        "## Creating distance data ##"
        "############################"
    )

    # Creating distance dictionary [o][d] -> distance
    distance_dic = gen.get_distance_dic(config.path_dist_dic, G)

    # Creating distance matrix (n X n) from dictionary
    distance_matrix = gen.get_distance_matrix(
        config.path_dist_matrix_npy, G, distance_dic_m=distance_dic
    )

    # Distance matrix as dataframe
    dt_distance_matrix = gen.get_dt_distance_matrix(
        config.path_dist_matrix, distance_matrix
    )

    # Dataframe info
    # print(dt_distance_matrix.describe())

    print(
        "\n############################"
        "## Reachability & Regions ##"
        "############################"
    )
    # Creating reachability dictionary
    reachability, steps = gen.get_reachability_dic(
        config.path_reachability_dic,
        distance_dic,
        step=config.step,
        total_range=config.total_range,
        step_list=config.step_list,
        speed_km_h=config.speed_km_h,
    )

    # Creating region centers for all max. travel durations
    # in reachability dictionary

    region_centers = gen.get_region_centers(
        steps,
        config.path_region_centers,
        reachability,
        sorted(list(reachability.keys())),
        root_path=config.root_reachability,
        round_trip=config.round_trip,
    )

    # Distance from centers
    sorted_neighbors = gen.get_sorted_neighbors(
        distance_dic,
        region_centers,
        minimum_distance=0,
        path_sorted_neighbors=config.path_sorted_neighbors,
    )

    # pprint(sorted_neighbors)

    print("Plotting region neighbors...")
    vi.plot_region_neighbors(
        G,
        region_centers,
        sorted_neighbors,
        path=config.root_img_neighbors,
        show=False,
        file_format="png",
        max_neighbors=4,
        replace=False,
    )

    # Each node is associated to the closest region center according
    # to list of max. travel durations
    region_ids = gen.get_region_ids(
        G,
        reachability,
        region_centers,
        path_region_ids=config.path_region_center_ids,
    )

    # Plot region centers (blue) and associated nodes
    print("Plotting regions...")
    vi.plot_regions(
        G,
        region_centers,
        region_ids,
        path=config.root_img_regions,
        show=False,
        file_format="png",
        replace=False,
    )
    # ################################################################ #
    # Creating concentric network #################################### #
    # ################################################################ #

    concentric_region_ids, concentric_region_centers = gen.concentric_regions(
        G,
        steps,
        reachability,
        list(G.nodes()),
        center=-1,
        root_reachability=config.root_reachability_concentric,
    )

    # Distance from centers
    sorted_neighbors_concentric = gen.get_sorted_neighbors(
        distance_dic,
        concentric_region_centers,
        minimum_distance=0,
        path_sorted_neighbors=config.path_sorted_neighbors_concentric,
    )

    print("Plotting region neighbors...")
    vi.plot_region_neighbors(
        G,
        concentric_region_centers,
        sorted_neighbors_concentric,
        path=config.root_img_neighbors_concentric,
        show=False,
        file_format="png",
        max_neighbors=4,
        replace=False,
    )

    print("Plotting concentric regions...")
    vi.plot_regions(
        G,
        concentric_region_centers,
        concentric_region_ids,
        path=config.root_img_regions_concentric,
        show=False,
        file_format="png",
        replace=False,
    )

    if "url_tripdata" in config.tripdata:
        download_tripdata(config, G, distance_dic)

    # Trip data is saved in external drive
    if "path_tripdata" in config.tripdata:
        process_tripdata(config, G, distance_dic)

    if "data_gen" in config.tripdata:
        gen_random_data(config, G, distance_dic)


if __name__ == "__main__":

    # execute only if run as a script
    create_trip_data()
