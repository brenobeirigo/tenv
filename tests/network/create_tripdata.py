import os
import sys

# Adding project folder
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import tenv.config as config
import tenv.util as util
import pandas as pd
from collections import defaultdict
from pprint import pprint
import tenv.demand as tp
import tenv.visuals as vi

import logging

logging.basicConfig(level=logging.INFO)


def gen_random_data(config, G, distance_matrix):
    logging.info(
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
                distance_matrix=distance_matrix,
            )


def process_tripdata(config, G, distance_matrix):
    """ Loop trip files, select time windows, and match (lon,lat) to 
    to node ids in G.

    Parameters
    ----------
    config : module
        All configurations derived from .json file.
    G : networkx
        Street graph.
    distance_matrix : float matrix
        Distance matrix (km) of nodes in G
    """

    logging.info("Cleaning trip data...")

    for file_name, tws in config.tripdata["file_tw"].items():
        logging.info(f"File: {file_name}")
        for tw in tws:
            logging.info(f" - TW: {tw}")
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
            logging.info("Adding ids...")
            tp.add_ids(
                output_cleaned + file_name_cleaned,
                output_ids + file_name_ids,
                G,
                distance_matrix,
                filtered_columns=config.tripdata["filtered_columns"],
            )


if __name__ == "__main__":

    # Trip data is saved in external drive
    if "path_tripdata" in config.tripdata:
        process_tripdata(config, util.G, util.distance_matrix)

    if "data_gen" in config.tripdata:
        gen_random_data(config, util.G, util.distance_matrix)
