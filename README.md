# Trip data sandbox

## Installation
### Loading the python environment

Load the python environment in the file `tenv.yaml` to install all modules used in this project.

In the following section, tips on manipulating python environments from the [Anaconda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

#### Using environments
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

### Installing Gurobi on Anaconda

This project implements an ILP model to determine the smallest set of region centers in a network (according to a maximum distance). Follow the steps to run the model:

1. Download and install Gurobi ([link](http://www.gurobi.com/downloads/download-center));
2. Request a free academic license ([link](https://user.gurobi.com/download/licenses/free-academic));
3. Add Gurobi to Anaconda ([link](http://www.gurobi.com/downloads/get-anaconda)).

WARNING: When creating the environment, check python version. Gurobi can only be installed with python 3.7:

    conda create -n yourenvname python=x.x

### Installing OSMNX

Adding `--strict-channel-priority` is essencial to ensure that all the dependencies will come from the conda-forge channel.
    conda config --prepend channels conda-forge
    conda create -n ox --strict-channel-priority osmnx

### Installing H3 (Uber)

Uber geoindexing with hexagons:

    conda install h3

## Setup

Create `.csv` files in `in\config_scenario` with the case study settings. Then, define the target case study and the root directory in file `case_study_info.json`.

### Example: generate Delft network data

Change the `case_study_info.json` file:

```json
{
    "root": "C:/Users/LocalAdmin/street_network_server/tenv",
    "case_study": "delft.json"
}
```

### Example: generate Delft network data and create region centers

Change the `case_study_info.json` file:

```json
{
    "root": "C:/Users/LocalAdmin/street_network_server/tenv",
    "case_study": "delft_reachability.json"
}
```

## Creating network data structures

Execute the file `util.py` to create create folder `out\nyc_manhattan\` (assuming case study `nyc_process_demand.json`) containing:

    +---distance
    |       dist_dict_m.npy
    |       dist_matrix_m.csv
    |       dist_matrix_m.npy
    |       dist_dict_duration_s.npy
    |       dist_matrix_duration_s.csv
    |       dist_matrix_duration_s.npy
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


## Creating trip data

After downloading the NYC trip data, you can create a temporal distribution clone for other regions.

### Downloading
-  Access `data\in\tripdata\download_tripdata.ipynb`
-  Set trip link sources. E.g.:

```python
    path_trip_list = "data/in/trip_data/nyc_tripdata.csv"
```

- Set the taget folder where files will be saved. E.g.:

```python
    target_path = "data/in/trip_data/raw"
```

### Example: Generate Manhattan network data, create region centers, and process downloaded demand data

Change the `case_study_info.json` file:

```json
{
    "root": "C:/Users/LocalAdmin/street_network_server/tenv",
    "case_study": "nyc_process_demand.json"
}
```

Execute file `util.py`.

### Generate network data, create region centers, and clone demand from NYC

Change the `case_study_info.json` file:

```json
{
    "root": "C:/Users/LocalAdmin/street_network_server/tenv",
    "case_study": "delft_clone_demand.json"
}
```

Execute file `util.py`.

## SERVER

The file `server.py` starts a local REST server to provide easy access to the trip data.


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
