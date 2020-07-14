import os
import osmnx as ox
from multiprocessing import Pool
import pandas as pd
import requests
from functools import partial
from datetime import datetime, timedelta
import random

import logging
import tenv.network as nw

logging.basicConfig(level=logging.INFO)

# Stores in key = 'name of experiment' the generator used to
# read trip data in file.
request_base = dict()

# #################################################################### #
# Download, clean, and save trip dataset ############################# #
# #################################################################### #


def download_file(url, root_path, file_name):
    """Download online file and save it.

    Arguments:
        url {String} -- Url to download
        output_file {String} -- Target path
    """

    output_file = "{}/{}".format(root_path, file_name)

    logging.info(f"Loading  '{output_file}'")

    if not os.path.exists(output_file):
        # TODO action when url does not exist
        logging.info("Downloading {}".format(url))
        r = requests.get(url, allow_redirects=True)
        open(output_file, "wb").write(r.content)
        logging.info(f"File downloaded successfully to '{output_file}'.")


def gen_random_data(config, G, distance_dic):

    # Loop all generation function (e.g., clone, cluster)
    for random_func_name in config.data_gen["funcs"]:

        if random_func_name == "random_clone":

            data_gen_path_tripdata_ids = "{}/{}_{}_ids.csv".format(
                config.root_tripdata_generated_ids,
                random_func_name,
                config.get_excerpt_name(
                    config.data_gen["start"], config.data_gen["stop"],
                ),
            )

            gen_requests(
                config.data_gen["source"],
                config.data_gen["max_passenger_count"],
                G,
                data_gen_path_tripdata_ids,
                start_timestamp=config.data_gen["start"],
                end_timestamp=config.data_gen["stop"],
                distance_dic=distance_dic,
                min_dist=config.data_gen["min_dist_km"],
                max_dist=config.data_gen["max_dist_km"],
            )


def get_trip_data(
    tripdata_path,
    output_path,
    start=None,
    stop=None,
    index_col="pickup_datetime",
    filtered_columns=[
        "pickup_datetime",
        "passenger_count",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ],
):

    """
    Read raw tripdata csv and filter unnecessary info.

        1 - Check if output path exists
        2 - If output path does not exist
            2.1 - Select columns ("pickup_datetime",
                                "passenger_count",
                                "pickup_longitude",
                                "pickup_latitude",
                                "dropoff_longitude",
                                "dropoff_latitude")
            2.2 - If start and stop are not None, get excerpt
        3 - Save clean tripdata in a csv
        4 - Return dataframe

    Arguments:
        tripdata_path {string} -- Raw trip data csv path
        output_path {string} -- Cleaned trip data csv path
        start {string} -- Datetime where tripdata should start 
            (e.g., 2011-02-01 12:23:00)
        stop {string} -- Datetime where tripdata should end
            (e.g., 2011-02-01 14:00:00)

    Returns:
        Dataframe -- Cleaned tripdata dataframe
    """

    # Trip data dataframe (Valentine's day)
    tripdata_dt_excerpt = None

    try:

        logging.info("Loading file '{}'...".format(output_path))

        # Load tripdata
        tripdata_dt_excerpt = pd.read_csv(
            output_path, parse_dates=True, index_col=index_col
        )

    except:

        logging.info(f"Load failed! Reading trip data '{tripdata_path}'...")

        try:
            # Reading file
            tripdata_dt = pd.read_csv(
                tripdata_path,
                parse_dates=True,
                index_col=index_col,
                usecols=filtered_columns,
                na_values="0",
            )

            tripdata_dt_excerpt = None

            # Get excerpt
            if start and stop:
                tw_filter = (tripdata_dt.index >= start) & (
                    tripdata_dt.index <= stop
                )
                tripdata_dt_excerpt = pd.DataFrame(tripdata_dt.loc[tw_filter])
            else:
                tripdata_dt_excerpt = pd.DataFrame(tripdata_dt)

            # Remove None values - Don't remove None data related to payment
            filter_valid_ods = (
                pd.notnull(tripdata_dt_excerpt["passenger_count"])
                & pd.notnull(tripdata_dt_excerpt["trip_distance"])
                & pd.notnull(tripdata_dt_excerpt["pickup_longitude"])
                & pd.notnull(tripdata_dt_excerpt["pickup_latitude"])
                & pd.notnull(tripdata_dt_excerpt["dropoff_longitude"])
                & pd.notnull(tripdata_dt_excerpt["dropoff_latitude"])
            )

            tripdata_dt_excerpt = tripdata_dt_excerpt[filter_valid_ods]

            # Sort
            tripdata_dt_excerpt.sort_index(inplace=True)

            # Save day data
            logging.info(
                f"Saving {len(tripdata_dt_excerpt)} to '{output_path}'..."
            )
            tripdata_dt_excerpt.to_csv(output_path)

            return tripdata_dt_excerpt

        except Exception as e:
            logging.info(f"Exception: {e}")


def get_ids(
    G, pk_lat, pk_lon, dp_lat, dp_lon, distance_dic_m, max_dist_km=0.05
):

    try:
        # Get pick-up and drop-off coordinates of request
        pk = (pk_lat, pk_lon)
        dp = (dp_lat, dp_lon)

        # TODO add distance to nearest node
        # Get nearest node in graph from coordinates
        pk_node_dist_m = ox.get_nearest_node(G, pk, return_dist=True)
        dp_node_dist_m = ox.get_nearest_node(G, dp, return_dist=True)
        # logging.info(pk, dp, n_pk_km, n_dp_km, max_dist_km)

        # If nearest node is "max_dist" meters far from point, request
        # is discarded
        if (
            pk_node_dist_m[1] / 1000 > max_dist_km
            or dp_node_dist_m[1] / 1000 > max_dist_km
        ):
            return [None, None]

        # pk must be different of dp
        if pk_node_dist_m[0] == dp_node_dist_m[0]:
            return [None, None]

        d = distance_dic_m[pk_node_dist_m[0]][dp_node_dist_m[0]]
        # logging.info("Dist:", d, max_dist_km)

        # Remove short distances
        if d >= max_dist_km:
            return [pk_node_dist_m[0], dp_node_dist_m[0]]
        else:
            return [None, None]

    except Exception as e:
        logging.error(f"Cannot get ids! Exception: {e}.")
        return [None, None]


def add_ids_chunk(G, distance_dic_m, order, max_dist_km, info):
    """Receive a dataframe chunk with tripdata and try adding node ids
    to the pickup and delivery points.


    Arguments:
        G {networkx} -- Street network to performe coordinate/node match
        distance_dic_m {dict{dict(float)}} -- Shortest distances between
            ODs in G (usage: distance_dic_m[o][d] = dist).
        info {dataframe} -- Unmatched tripdata

    Returns:
        dataframe -- Trip data with pickup and delivery node ids from
            graph G.
    """

    # Add pickup and delivery ids
    # If ids can't be found, the trip occurs outside G boundaries
    info[["pk_id", "dp_id"]] = info.apply(
        lambda row: pd.Series(
            get_ids(
                G,
                row["pickup_latitude"],
                row["pickup_longitude"],
                row["dropoff_latitude"],
                row["dropoff_longitude"],
                distance_dic_m,
                max_dist_km=max_dist_km,
            )
        ),
        axis=1,
    )

    original_chunk_size = len(info)

    # Remove trip data outside street network in G
    filter_valid_ods = pd.notnull(info["pk_id"]) & pd.notnull(info["dp_id"])
    info = info[filter_valid_ods]

    logging.info(f"Adding {len(info)} / {original_chunk_size}")

    info = info.astype(
        {"passenger_count": "int32", "pk_id": "int32", "dp_id": "int32"}
    )
    info = info[order]

    return info


def add_ids(
    path_tripdata,
    path_tripdata_ids,
    G,
    distance_dic_km,
    max_dist_km=0.05,
    filtered_columns=[
        "pickup_datetime",
        "passenger_count",
        "pk_id",
        "dp_id",
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ],
    n_mp=2,
):
    """Read large dataframe in chunks of trip data and associate
    node ids from graph G to pickup and delivery coordinates
    (within 50 meters).

    Entries whose coordinates are not matched to any point in G
    are discarded, since their occur outside G boundaries.

    Arguments:
        path_tripdata {str} -- File path of the tripdata.
        path_tripdata_ids {str} -- Target file path of tripdata with
            pickup and delivery ids.
        G {networkx} -- Graph to compare tripdata against.
        distance_dic_m {dict{int:dict{int:float}}} -- Shortest distances
            between ODs in G (usage: distance_dic_m[o][d] = dist).
    """

    # Create local instance and add od ids to list of columns
    filtered_columns = list(filtered_columns)
    filtered_columns.insert(2, "pk_id")
    filtered_columns.insert(3, "dp_id")

    dt = None

    # if file does not exist write header
    if os.path.isfile(path_tripdata_ids):

        # Load tripdata
        dt = pd.read_csv(
            path_tripdata_ids, parse_dates=True, index_col="pickup_datetime"
        )

        logging.info(
            "\nLoading trip data with ids...'{}'.".format(path_tripdata_ids)
        )

    else:

        logging.info(
            f"### NY trip data  "
            f"\n  - {path_tripdata}\n  - {path_tripdata_ids}"
        )
        tripdata = pd.read_csv(path_tripdata)

        tripdata.info()

        # Number of lines to read from huge .csv
        chunksize = 2000

        # Redefine function to add graph and distances
        func = partial(
            add_ids_chunk, G, distance_dic_km, filtered_columns, max_dist_km
        )

        # Total number of chunks to process
        togo = int(len(tripdata) / chunksize)

        # Read chunks of 500 lines
        count = 0
        count_lines = 0

        # Multiprocesses
        p = Pool(n_mp)

        # List of chunks of tripdata to process in parallel
        list_parallel = []

        # Start chunk reading process
        gen_chunks = pd.read_csv(
            path_tripdata, index_col=False, chunksize=chunksize
        )

        # Read first batch from dataframe and save in list
        # for parallel processing
        next_batch = next(gen_chunks)
        list_parallel.append(next_batch)

        # If first batch was read successfully, continue reading
        # following batches
        while next_batch is not None:

            try:
                next_batch = next(gen_chunks)
                list_parallel.append(next_batch)
            except:
                next_batch = None

            # The pool of processes starts if:
            #  - The number of dataframe chunks is equal to the
            #    max number of multiprocesses;
            #  - The end of the dataframe was reached (no next batch).
            if len(list_parallel) == n_mp or next_batch is None:

                # Update progress counters
                count = count + len(list_parallel)
                count_lines = count_lines + sum(map(len, list_parallel))

                # Start multiprocessing
                chunks_with_ids = p.map(func, list_parallel)

                # Writing chunks in target file
                for info_ids in chunks_with_ids:

                    # if file does not exist write header
                    if not os.path.isfile(path_tripdata_ids):

                        info_ids.to_csv(path_tripdata_ids, index=False)

                    # Since file exists, append without writing
                    # the header
                    else:
                        info_ids.to_csv(
                            path_tripdata_ids,
                            mode="a",
                            header=False,
                            index=False,
                        )

                # Clean parallel list to add more chunks
                list_parallel.clear()

                logging.info(
                    (
                        "Chunk progress: {}/{}" " - Dataframe progress: {}/{}"
                    ).format(count, togo, count_lines, len(tripdata))
                )

        # Load tripdata
        dt = pd.read_csv(
            path_tripdata_ids, parse_dates=True, index_col="pickup_datetime"
        )

        logging.info(
            (
                "\nLoading trip data with ids" " (after processing)...'{}'."
            ).format(path_tripdata_ids)
        )

    logging.info(dt.head())
    logging.info(dt.describe())


# #################################################################### #
# Query request data ################################################# #
# #################################################################### #


def get_n_requests_df(
    number_requests,
    tripdata_csv_path=None,
    start_timestamp=None,
    end_timestamp=None,
    class_freq_dict=None,
):

    # Generator of data chunks from csv file
    max_chunk_size = 2000

    gen_chunks = pd.read_csv(
        tripdata_csv_path,
        parse_dates=True,
        index_col="pickup_datetime",
        chunksize=min(number_requests, max_chunk_size),
    )

    trips = pd.DataFrame()
    for next_batch in gen_chunks:
        trips = trips.append(next_batch)
        if len(trips) >= number_requests:
            trips = trips[:number_requests]
            break

    population, frequencies = zip(*class_freq_dict.items())

    service_class_distribution = random.choices(
        population, weights=frequencies, k=len(trips)
    )

    trips["service_class"] = pd.Series(
        service_class_distribution, index=trips.index
    )

    return trips


def gen_requests(
    clone_tripdata_path,
    max_passenger_count,
    G,
    output_path,
    start_timestamp=None,
    end_timestamp=None,
    min_dist=None,
    max_dist=None,
    distance_dic=None,
):

    logging.info("\nGeneration random requests clone mirror...")

    # if file does not exist write header
    if os.path.isfile(output_path):
        logging.info("\nTrip data already exists in '{}'.".format(output_path))
        return
    else:

        logging.info("Saving at '{}'.".format(output_path))

        # Number of lines to read from huge .csv
        chunksize = 2000

        # Start chunk reading process
        gen_chunks = pd.read_csv(
            clone_tripdata_path,
            parse_dates=True,
            index_col="pickup_datetime",
            chunksize=chunksize,
        )

        count = 0
        logging.info("Time window:", start_timestamp, "----", end_timestamp)

        for chunk_dt_clone in gen_chunks:

            chunk_dt_clone = chunk_dt_clone[start_timestamp:end_timestamp]

            if len(chunk_dt_clone) == 0:
                break

            chunk_dt = chunk_dt_clone.apply(
                lambda row: get_random_request_series(
                    row,
                    gen_random_request_od(G, distance_dic=distance_dic),
                    max_passenger_count,
                ),
                axis=1,
            )

            # Convert node ids and passenger count to int
            chunk_dt[["passenger_count", "pk_id", "dp_id"]] = chunk_dt[
                ["passenger_count", "pk_id", "dp_id"]
            ].astype(int)

            # logging.info(chunk_dt.head())

            # if file does not exist write header
            # logging.info("Saving first chunk...")
            if not os.path.isfile(output_path):

                chunk_dt.to_csv(output_path)

            # Since file exists, append without writing the header
            else:
                chunk_dt.to_csv(output_path, mode="a", header=False)

            count = count + 1
            logging.info(
                "Chunk progress: {}----{}".format(
                    chunk_dt.iloc[0].name, chunk_dt.iloc[-1].name
                )
            )

    # dt_start_ts = datetime.strptime(start_timestamp, '%Y-%m-%d %H:%M:%S')
    # dt_end_ts = datetime.strptime(end_timestamp, '%Y-%m-%d %H:%M:%S')
    # td_batch_duration = timedelta(seconds=batch_duration_s)

    # current_dt = dt_start_ts

    # while current_dt < dt_end_ts:
    #     current_dt = current_dt + td_batch_duration
    #     logging.info(dt_start_ts, current_dt, dt_end_ts)

    # TODO Use frequency generation in the future. Example:
    # date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='H')


def get_random_request_series(series, od, max_passenger_count):
    """Create a series for the request data that will be integrated
    in the OD dataframe.

    Arguments:
        series {[type]} -- [description]
        od {[type]} -- [description]
        max_passenger_count {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    data = dict(series)
    data["passenger_count"] = random.randint(1, max_passenger_count)
    data.update(od)
    return pd.Series(data)


def gen_random_vehicle_origin(G):

    n_nodes = nw.get_number_of_nodes(G)

    return random.randint(0, n_nodes - 1)


def gen_random_request_od(
    G, min_dist=100, max_dist=1000000, distance_dic=None
):
    """Generate a single request by choosing random ODs from
    the street network G.

    The distance between ODs is in the range [min_dist, max_dist]

    Arguments:
        min_dist {float} -- Minimum distance between request's OD
            (default = meters)
        max_dist {float} -- Maximum distance between request's OD
            (default = meters)
        G {networkx} -- Graph to get node coordinates.
        distance_dic {dict{int:dict{int:float}}} -- Shortest distances
            between ODs in G (usage: distance_dic_m[o][d] = dist).

    Returns:
        dict -- Random request dictionary. Details:
        'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
        'dropoff_latitude', 'pk_id', 'dp_id'.
    """

    n_nodes = nw.get_number_of_nodes(G)

    while True:
        # Find random origin and destination
        o = random.randint(0, n_nodes - 1)
        d = random.randint(0, n_nodes - 1)
        dist = distance_dic[o][d] if distance_dic else nw.get_distance(G, o, d)
        # Stop searching if adequate destination is found
        if o != d and dist >= min_dist and dist <= max_dist:
            break

    x_from = G.nodes[o]["x"]
    y_from = G.nodes[o]["y"]
    x_to = G.nodes[d]["x"]
    y_to = G.nodes[d]["y"]

    req = {
        "pickup_longitude": x_from,
        "pickup_latitude": y_from,
        "dropoff_longitude": x_to,
        "dropoff_latitude": y_to,
        "pk_id": o,
        "dp_id": d,
    }

    return req


def get_next_batch(
    experiment_name,
    chunk_size=2000,
    batch_size=30,
    tripdata_csv_path=None,
    start_timestamp=None,
    end_timestamp=None,
    classes=None,
    freq=None,
):
    """
    Iteratevely returns trip data within sequential time intervals 
    for each experiment.

    Each experiment is associated to a dictionary entry 
    containing a generator that reads trip data chunks from 
    a csv iteratively.


    generator 
     - CSV reader
     - First tim of size "batch_size"

    Arguments:
        experiment_name {[type]} -- Name experiment calling 
        the batch of requests.

    Keyword Arguments:
        chunk_size {int} -- N. of lines read from dataframe 
                            each time (default: {2000})
        batch_size {int} -- Data interval returned each call
                             (default: {30})
        tripdata_csv_path {str} -- Origin of trip data (default: {None})
        start_timestamp {str} -- Earliest time of tripdata 
                                (e.g., 2011-02-01 00:00:00) (default: {None})
        end_timestamp {str} -- Latest time of tripdata 
                                (e.g., 2011-02-10 00:00:00) (default: {None})

    Returns:
        dataframe -- trip data slice corresponding to the sequence 
        in current time interval
    """

    # Easy access to dictionary reffering to experiment_name
    trips = None

    # Start chunk reading process from the beginning if experiment
    # is being run for the first time
    if experiment_name not in request_base:

        logging.info(
            (
                "\nActivating tripdata generator" " for experiment '{}'..."
            ).format(experiment_name)
        )

        # Generator of data chunks from csv file
        gen_chunks = pd.read_csv(
            tripdata_csv_path,
            parse_dates=True,
            index_col="pickup_datetime",
            chunksize=chunk_size,
        )

        # Save generator and related information in dictionary
        trips = request_base[experiment_name] = dict()

        trips["generator"] = gen_chunks

        # When requests start being read
        trips["start_time"] = datetime.strptime(
            start_timestamp, "%Y-%m-%d %H:%M:%S"
        )

        # When requests stop being read
        trips["final_time"] = datetime.strptime(
            end_timestamp, "%Y-%m-%d %H:%M:%S"
        )

        # Determine where generator last stopped reading
        trips["current_time"] = datetime.strptime(
            start_timestamp, "%Y-%m-%d %H:%M:%S"
        )

        # Interval pulled from request base each time (e.g., 30s)
        trips["batch_size"] = timedelta(seconds=batch_size)

        trips["tripdata_csv_path"] = tripdata_csv_path

        # Store read requests outside current interval (due to chunk size)
        trips["residual_requests"] = None

        trips["service_classes"] = classes
        trips["class_frequencies"] = freq

        # Loop data until batch has info > start_timestamp
        for first_batch in trips["generator"]:

            if not first_batch[start_timestamp:].empty:
                # The first batch goes to the residual request
                trips["residual_requests"] = first_batch
                break
    else:
        # The experiment has already been initiated.
        # Load previous data.
        trips = request_base[experiment_name]

    # If last possible time was reached, stop returning data
    if trips["current_time"] >= trips["final_time"]:
        return None

    batch = trips["residual_requests"]
    left_tw = trips["current_time"]
    right_tw = left_tw + trips["batch_size"]
    # logging.info(left_tw, "--", right_tw)
    right_tw = min(right_tw, trips["final_time"])

    for next_batch in trips["generator"]:

        batch = batch.append(next_batch)

        if not batch[right_tw:].empty:
            break

        # logging.info('Size batch after:', len(batch))

    # Stores requests read (due to the chunk size) but not returned
    trips["residual_requests"] = pd.DataFrame(batch[right_tw:])

    # Cut dataframe until time "right_tw" (not included)
    batch = batch[: right_tw - timedelta(seconds=1)]

    # Update current time
    trips["current_time"] = right_tw

    # Randomly assign user classes to requests if
    # service classes were defined
    if trips["service_classes"]:
        service_class_distribution = random.choices(
            trips["service_classes"],
            weights=trips["class_frequencies"],
            k=len(batch),
        )

        batch["service_class"] = pd.Series(
            service_class_distribution, index=batch.index
        )

    return batch


if __name__ == "__main__":
    pass
    # Get network graph and save
    # G = nw.get_network_from(config.tripdata["region"],
    #                          config.root_path,
    #                          config.graph_name,
    #                          config.graph_file_name)
    # nw.save_graph_pic(G)

    # logging.info( "\nGetting distance data...")
    # # Creating distance dictionary [o][d] -> distance
    # distance_dic = nw.get_distance_dic(config.path_dist_dic, G)
