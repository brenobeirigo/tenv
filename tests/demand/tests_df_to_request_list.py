import os
import sys

# Adding project folder
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import tenv.demand as de

if __name__ == "__main__":

    service_level = {
        "A": {"pk_delay": 180, "trip_delay": 180, "sharing_preference": 0},
        "B": {"pk_delay": 300, "trip_delay": 600, "sharing_preference": 1},
        "C": {"pk_delay": 600, "trip_delay": 900, "sharing_preference": 1},
    }

    tripdata_csv_path = (
        "C:\\Users\\LocalAdmin\\OneDrive\\leap_forward\\"
        "street_network_server\\input_tripdata\\data\\"
        "manhattan-island-new-york-city-new-york-usa\\tripdata\\"
        "tripdata_excerpt_2011-2-1_2011-2-28_ids.csv"
    )

    NUMBER_REQUESTS = 30000
    bb_segmentation = {"A": 0.16, "B": 0.68, "C": 0.16}

    df = de.get_n_requests_df(
        NUMBER_REQUESTS,
        tripdata_csv_path=tripdata_csv_path,
        class_freq_dict=bb_segmentation,
    )

    print(df)
