# Trip data sandbox


## Donwload trip data

1. Access `data\in\tripdata\download_tripdata.ipynb`
2. Set trip link sources. E.g.:

        path_trip_list = "C:/Users/LocalAdmin/Documents/GitHub/tenv/data/in/trip_data/nyc_tripdata.csv"


3. Set the taget folder where files will be saved. E.g.:

        target_path = "C:/Users/LocalAdmin/Documents/GitHub/tenv/data/in/trip_data/"

## Clear and match ids

Create `.csv` data in `in\config_scenario` with download data. Example:

```json
{
    "mapdata": {
        "region": "Manhattan Island, New York City, New York, USA",
        "label": "manhattan_nyc",
        "reachability": {
            "step": 30,
            "total_range": 600,
            "speed_km_h": 30,
            "max_neighbors": 6,
            "step_list": [0, 60, 300, 600], // If null, step_list = [0, 30, 60, ..., 600]
            "round_trip": false
        },
        "max_travel_time_edge": 60, // Edge travel time <= 60
        "info": "Creates data for ITSM paper (old Manhattan graph from 2018)"
    },
    "tripdata": {
        "path_tripdata": "O:/phd/nyc_trips/raw/",
        "output_cleaned_tripdata": "C:/Users/LocalAdmin/manhattan_nyc/tripdata/cleaned/",
        "output_ids_tripdata": "C:/Users/LocalAdmin/manhattan_nyc/tripdata/ids/",
        "index_col": "pickup_datetime",
        "max_dist_km": 0.05,
        "filtered_columns": [
            "pickup_datetime",
            "passenger_count",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "fare_amount",
            "payment_type",
            "total_amount",
            "tip_amount",
            "trip_distance"
        ],
        "file_tw": {
            "yellow_tripdata_2011-02.csv": [
                // Ranges extracted
                [
                "2011-02-01 00:00:00",
                "2011-02-28 23:59:59"
                ]
            ]
        }
    }
}
```

Refer to the file created in file `file_info.json` using:

```json
{
    "root": "C:/Users/LocalAdmin/OneDrive/leap_forward/street_network_server/tenv",
    "case_study": "nyc_business_class.json"
}
```

Execute the file `util.py` to generate the following data will create folder `out\manhattan_nyc\` containing:

    +---distance
    |       dist_dict_m.npy
    |       dist_matrix_m.csv
    |       dist_matrix_m.npy
    |       
    +---lean_data
    |       center_nodes.npy
    |       distance_matrix_km.npy
    |       node_delay_center_id.npy
    |       node_region_ids.npy
    |       region_centers.npy
    |       region_id_dict.npy
    |       sorted_neighbors.npy
    |       
    +---map
    |   |   manhattan_nyc.graphml
    |   |   manhattan_nyc.svg
    |   |   
    |   \---reachability_reg_30_600_kmh30
    |       |   reach_dict.npy
    |       |   reach_dict_round.npy
    |       |   region_centers.npy
    |       |   region_center_ids.npy
    |       |   sorted_neighbors_region_centers.npy
    |       |   
    |       +---img_region_centers
    |       |       max_dist_000_centers_4546.png
    |       |       max_dist_060_centers_284.png
    |       |       max_dist_300_centers_020.png
    |       |       max_dist_600_centers_009.png
    |       |       
    |       +---img_region_center_neighbors
    |       |       max_dist_000_neighbors_004.png
    |       |       max_dist_060_neighbors_004.png
    |       |       max_dist_300_neighbors_004.png
    |       |       max_dist_600_neighbors_004.png
    |       |       
    |       \---mip_region_centers
    |           +---gurobi_log
    |           |       region_centers_0.log
    |           |       region_centers_0.lp
    |           |       region_centers_300.log
    |           |       region_centers_300.lp
    |           |       region_centers_60.log
    |           |       region_centers_60.lp
    |           |       region_centers_600.log
    |           |       region_centers_600.lp
    |           |       
    |           \---sub_sols
    |                   _0000.npy
    |                   _0060.npy
    |                   _0300.npy
    |                   _0600.npy
    |                   
    +---network_info
    |       level_center_children.csv
    |       level_center_neighbors_asc.csv
    |       node_level_id.csv
    |       region_centers.csv
    |       
    \---tripdata
        +---cleaned
        |       tripdata_cleaned_2011-02-01_000000_2011-02-07_235959.csv
        |       
        +---ids
        |       tripdata_ids_2011-02-01_000000_2011-02-07_235959.csv
        |       
        *---raw
                yellow_tripdata_2011-01.csv

Execute `server.py` to create REST server.

## Installing Gurobi on Anaconda

This project implements an ILP model to determine the smallest set of region centers in a network (according to a maximum distance). Follow the steps to run the model:

1. Download and install Gurobi ([link](http://www.gurobi.com/downloads/download-center));
2. Request a free academic license ([link](https://user.gurobi.com/download/licenses/free-academic));
3. Add Gurobi to Anaconda ([link](http://www.gurobi.com/downloads/get-anaconda)).

WARNING: When creating the environment, check python version. Gurobi call only be installed with python 3.7:

    conda create -n yourenvname python=x.x

## Installing OSMNX

Adding `--strict-channel-priority` is essencial to ensure that all the dependencies will come from the conda-forge channel.
    conda config --prepend channels conda-forge
    conda create -n ox --strict-channel-priority osmnx

## Istalling H3 (Uber)

Uber geoindexing with hexagons:

    conda install h3

## Using GIT

Use this project remote:

    https://github.com/brenobeirigo/input_tripdata.git

In the following section, Git tips from the [Git Cheat Sheet](https://www.git-tower.com/blog/) (git-tower).


### Create
Clone an existing repository
    
    git clone <remote>

Create a new local repository
    
    git init

### Tests
To create dataframes related to network structure execute `tests\create_network_dfs.py`. The following `.csv` files will be saved in the `network_info` folder:

|File (.csv) |Headers|
|---|---|
|node_level_id|node, level1, level2, level3, ..., levelN|
|level_center_children|level, center, n_children, children_list|
|level_center_neighbors_asc|level, center, n_neighbors, neighbors_asc|
|region_centers|center, center_count, centers|


### Update & Publish

List all currently  configured remotes
    
    git remote - v

Download changes and directly merge/integrate into HEAD
    
    git pull <remote> <branch>

#### Publish local changes on a remote
    git push <remote> <branch>

## Loading the python environment

Load the python environment in the file `env_slevels.yaml` to install all modules used in this project.

In the following section, tips on manipulating python environments from the [Anaconda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

### Using environments
List all packages and versions installed in active environment

    conda list

Get a list of all my environments, active
environment is shown with *

    conda env list

Create a new environment named py35, install Python 3.5
    
    conda create --name py35 python=3.5 

Create environment from a text file

    conda env create -f environment_name.yaml

Save environment to a text file

    conda env export > environment_name.yaml

Remove environment

    conda env remove --name NAME_ENV

## SERVER

The file `server.py` starts a local server to provide easy access to the trip data.


### Adjusting TCP Settings for Heavy Load on Windows

    SOURCE: https://docs.oracle.com/cd/E23095_01/Search.93/ATGSearchAdmin/html/s1207adjustingtcpsettingsforheavyload01.html

    The underlying Search architecture that directs searches across multiple
    physical partitions uses TCP/IP ports and non-blocking NIO SocketChannels
    to connect to the Search engines.
    
    These connections remain open in the TIME_WAIT state until the operating
    system times them out. Consequently, under heavy load conditions,
    the available ports on the machine running the Routing module can be exhausted.

    On Windows platforms, the default timeout is 120 seconds, and the maximum number
    of ports is approximately 4,000, resulting in a maximum rate of 33
    connections per second.
    
    If your index has four partitions, each search requires four ports, 
    which provides a maximum query rate of 8.3 queries per second.

    (maximum ports/timeout period)/number of partitions = maximum query rate
    If this rate is exceeded, you may see failures as the supply of TCP/IP ports is exhausted.
    Symptoms include drops in throughput and errors indicating failed network connections.
    
    You can diagnose this problem by observing the system while it is under load,
    using the netstat utility provided on most operating systems.

    To avoid port exhaustion and support high connection rates,
    reduce the TIME_WAIT value and increase the port range.

    To set TcpTimedWaitDelay (TIME_WAIT):
    - Use the regedit command to access the registry subkey:
        HKEY_LOCAL_MACHINE\
        SYSTEM\
        CurrentControlSet\
        Services\
        TCPIP\
        Parameters
    - Create a new REG_DWORD value named TcpTimedWaitDelay.
    - Set the value to 60.
    - Stop and restart the system.

    To set MaxUserPort (ephemeral port range):
    - Use the regedit command to access the registry subkey:
        HKEY_LOCAL_MACHINE\
        SYSTEM\
        CurrentControlSet\
        Services\
        TCPIP\
        Parameters
    - Create a new REG_DWORD value named MaxUserPort.
    - Set this value to 32768.
    - Stop and restart the system.
