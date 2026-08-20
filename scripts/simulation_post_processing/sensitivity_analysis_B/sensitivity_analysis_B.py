# This code is to calculate the sensitivity of the Magnetic field (magnitude and B_z) and other time series parameters
# Based on Talwinder's 2023 simulation results

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from datetime import timedelta, datetime
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from dtaidistance import dtw

########################################
# Settings 
pl.Config.set_tbl_cols(-1)      # polars settings to show all the columns
pl.Config.set_tbl_rows(-1)      # polars settings to show all the rows

# place holder for cme duration - I will figure out how to calculate CME end time later
n_hours = 15
den_jump = 0.1
time_series_parameter = "B_z"

address = "../../../Data/Sensitivity_analysis_B/"
params = ["speed", "latitude", "longitude", "tilt", "half_angle", "aspect_ratio"]

# to store which ensemble numbers belong with which parameter
param_sensitivity = {"speed": [],
                    "latitude": [],
                    "longitude": [],
                    "tilt": [],
                    "half_angle": [],
                    "aspect_ratio": []}
########################################
# Helper functions

def generate_pairs(target_param: str, low_val : pl.Float64, high_val: pl.Float64, param_const: list, data: pl.DataFrame):
    """
    This function returns a list of tuples after calculating the ensemble pairs for a particualt CME

    Params:
        - target_param (str)
        The parameter under scrutiny. This is the parameter we are calculating the 
        sensitivity of
        - low_val, high_val (pl.Float64)
        The low and high values of the target parameter
        - param_const (list)
        This is the list of parameters that need to be constant as one is changing 
        - data (pl.DataFrame)
        This is the data frame that contains the gcs parameters of the ensemble members 

    Returns:
        - gcs_pairs (list)
        A list of tuples that represent the 33 total pairs
    """
    # low value
    low_df = data.filter(
            pl.col(target_param).is_close(low_val)  
        ).select(["ensemble_member", *param_const])
    
    # high value
    high_df = data.filter(
            pl.col(target_param).is_close(high_val)
        ).select(["ensemble_member", *param_const])

    pairs_df = low_df.join(
        high_df,
        on = param_const,
        how = "inner",
        suffix = "_high"
    )

    pairs = list(
        zip(pairs_df["ensemble_member"], pairs_df["ensemble_member_high"])
    )

    return pairs

def is_leap(Year):  
  # Checking if the given year is leap year  
    if((Year % 400 == 0) or  
     (Year % 100 != 0) and  
     (Year % 4 == 0)): 
        return True
    else:
        return False

def convert_partial_year(number):

    year = int(number)
    d = timedelta(days=(number - year)*(365 + is_leap(year)))
    day_one = datetime(year,1,1)
    date = d + day_one
    return date

def find_TOA(df0, df1, df2):
    """
    Finds and returns the arrival time (TOA) of a simulated CME.

    Currently uses a 10% density jump relative to the ambient (df0) density
    as the TOA criterion, applied to both df1 and df2 (e.g. two different
    ensemble members / probe locations). This threshold is subject to change.

    Assumes df0, df1, df2 share a common "time" column aligned by row index.
    """
    density_diff_1 = (df1["density"] - df0["density"]) / df0["density"]
    density_diff_2 = (df2["density"] - df0["density"]) / df0["density"]

    mask_1 = density_diff_1.abs() > 0.1
    mask_2 = density_diff_2.abs() > 0.1

    jump_indices_1 = mask_1.arg_true()
    jump_indices_2 = mask_2.arg_true()

    if len(jump_indices_1) > 0:
        first_idx_1 = jump_indices_1[0]
        time_stamp_1 = df1["date"][first_idx_1]
    else:
        print("no jump found in df1")
        time_stamp_1 = None

    if len(jump_indices_2) > 0:
        first_idx_2 = jump_indices_2[0]
        time_stamp_2 = df2["date"][first_idx_2]
    else:
        print("no jump found in df2")
        time_stamp_2 = None

    return time_stamp_1, time_stamp_2

def plot_difference(param_dict, gcs_param):
    """ It plots the MSE differences and shades each cme differently, in case 
        the sensitivity varies accross CMEs
    """
    y = np.asarray(param_dict[gcs_param])

    cmes = [1, 3, 4, 5, 6]

    n_groups = 5
    group_size = 33
    colors = plt.cm.tab10(np.arange(n_groups))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(n_groups):
        start = i * group_size
        end = start + group_size
        x = np.arange(start, end)

        ax.plot(
            x,
            y[start:end],
            color=colors[i],
            lw=2,
            label=f"CME {cmes[i]}"
        )

        # Optional: lightly shade the x-range belonging to each pair
        ax.axvspan(start - 0.5, end - 0.5, color=colors[i], alpha=0.12)

    ax.set_xlabel("Counts")
    ax.set_ylabel(fr"${time_series_parameter}$ time-series RMSE")
    ax.set_title(f"165 {gcs_param} values: 5 groups of 33")
    ax.legend(title="Group")
    ax.set_xlim(-0.5, len(y) - 0.5)

    plt.tight_layout()
    plt.savefig(f"figures/{time_series_parameter}_{gcs_param}_diff.png", dpi = 300)
    plt.close()

def box_plot(param_dict):
    """
    Takes in the param dictionary and makes a box plot 
    """

    plt.boxplot(param_dict.values(), tick_labels = param_dict.keys(), whis = 1.5)

    # Add details
    plt.title("Box Plot of param sensitivities on Bz")
    plt.xlabel("GCS params")
    plt.ylabel("$B_{z}$ RMSE (nT)")

    plt.tight_layout
    plt.savefig("figures/box_plot_all_params.png", dpi = 300)
    plt.close()

########################################
# The main code

cme_nums = [1, 3, 4, 5, 6]

for cme_num in cme_nums:

    gcs_params = pl.from_pandas(
        pd.read_csv(address + f'GCS_parms{cme_num}.dat', sep=r"\s+", header=None, 
        names = ["ensemble_member", "speed", "latitude", "longitude", "tilt", "half_angle", "aspect_ratio"])
    )

    # get the iunique values of each GCS parameter
    unique_speeds = gcs_params["speed"].unique().to_list()
    unique_lat = gcs_params["latitude"].unique().to_list()
    unique_lon = gcs_params["longitude"].unique().to_list()
    unique_tilt = gcs_params["tilt"].unique().to_list()
    unique_half_angle = gcs_params["half_angle"].unique().to_list()
    unique_aspect_ratio = gcs_params["aspect_ratio"].unique().to_list()

    # calculate the upper and lower bounds in the ensmeble spread
    low_speed, high_speed = unique_speeds[0], unique_speeds[2]
    low_lat, high_lat = unique_lat[0], unique_lat[2]
    low_lon, high_lon = unique_lon[0], unique_lon[2]
    low_tilt, high_tilt = unique_tilt[0], unique_tilt[2]
    low_half_angle, high_half_angle = unique_half_angle[0], unique_half_angle[2]
    low_aspect_ratio, high_aspect_ratio = unique_aspect_ratio[0], unique_aspect_ratio[2]

    # generate the pairs
    speed_pairs = generate_pairs("speed", low_speed, high_speed, ["latitude", "longitude", "tilt", "half_angle", "aspect_ratio"], gcs_params)
    lat_pairs = generate_pairs("latitude", low_lat, high_lat, ["speed", "longitude", "tilt", "half_angle", "aspect_ratio"], gcs_params)
    lon_pairs = generate_pairs("longitude", low_lon, high_lon, ["speed", "latitude", "tilt", "half_angle", "aspect_ratio"], gcs_params)
    tilt_pairs = generate_pairs("tilt", low_tilt, high_tilt, ["speed", "latitude", "longitude", "half_angle", "aspect_ratio"], gcs_params)
    half_angle_pairs = generate_pairs("half_angle", low_half_angle, high_half_angle, ["speed", "latitude", "longitude", "tilt", "aspect_ratio"], gcs_params)
    aspect_ratio_pairs = generate_pairs("aspect_ratio", low_aspect_ratio, high_aspect_ratio, ["speed", "latitude", "longitude", "tilt", "half_angle"], gcs_params)

    # build a map for params
    param_map = {
        "speed" : speed_pairs,
        "latitude" : lat_pairs,
        "longitude" : lon_pairs,
        "tilt" : tilt_pairs,
        "half_angle" : half_angle_pairs,
        "aspect_ratio" : aspect_ratio_pairs
    }

    # Start working on the time series data  
    for param in params:
        # for loop over the pairs
        for pair in param_map[param]:
            # get the twop paired files
            filename0 = f"{address}Probe_files_{cme_num}/Earth_Ensemble_0.dat"
            filename1 = f"{address}Probe_files_{cme_num}/Earth_Ensemble_{pair[0]}.dat"
            filename2 = f"{address}Probe_files_{cme_num}/Earth_Ensemble_{pair[1]}.dat"
            df0 = pl.read_csv(filename0, separator = " ", skip_rows = 3, has_header = False)
            df1 = pl.read_csv(filename1, separator = " ", skip_rows = 3, has_header = False)
            df2 = pl.read_csv(filename2, separator = " ", skip_rows = 3, has_header = False)


            # make the year into datetime columns
            df0 = df0.with_columns(
                pl.col(df0.columns[0]).map_elements(
                    lambda x: convert_partial_year(float(x)), return_dtype=pl.Datetime
                ).alias("date")
            )

            df1 = df1.with_columns(
                pl.col(df1.columns[0]).map_elements(
                    lambda x: convert_partial_year(float(x)), return_dtype=pl.Datetime
                ).alias("date")
            )

            df2 = df2.with_columns(
                pl.col(df2.columns[0]).map_elements(
                    lambda x: convert_partial_year(float(x)), return_dtype=pl.Datetime
                ).alias("date")
            )

            # keep only density (column_5), B_z (12), and date columns
            df0 = df0.select(["date", "column_5", "column_12"])
            df1 = df1.select(["date", "column_5", "column_12"])
            df2 = df2.select(["date", "column_5", "column_12"])
            df0 = df0.rename({"column_5": "density", "column_12": time_series_parameter})
            df1 = df1.rename({"column_5": "density", "column_12": time_series_parameter})
            df2 = df2.rename({"column_5": "density", "column_12": time_series_parameter})

            # interpolate to match df0 time series 
            target_dates = df0.select("date").sort("date")

            df1 = (df1.sort("date").group_by_dynamic("date", every="1h").agg(pl.exclude("date").mean()))
            df1 = (target_dates.join_asof(df1, on="date", strategy="nearest"))
            df1 = df1.with_columns(pl.exclude("date").interpolate())

            df2 = (df2.sort("date").group_by_dynamic("date", every="1h").agg(pl.exclude("date").mean()))
            df2 = (target_dates.join_asof(df2, on="date", strategy="nearest"))
            df2 = df2.with_columns(pl.exclude("date").interpolate())

            # call the find_TOA function
            time_stamp_1, time_stamp_2 = find_TOA(df0, df1, df2)

            # get df1 and df2 Bz for the required times
            sliced_df1 = df1.filter(
                pl.col("date").is_between(time_stamp_1, time_stamp_1 + timedelta(hours = n_hours))
            ).drop("density")

            sliced_df2 = df2.filter(
                pl.col("date").is_between(time_stamp_2, time_stamp_2 + timedelta(hours = n_hours))
            ).drop("density")

            rmse_Bz = root_mean_squared_error(sliced_df1[time_series_parameter], sliced_df2[time_series_parameter])
            
            param_sensitivity[param].append(rmse_Bz)

# make the difference figures for all the parameters accross all the CMEs 
for param in params:
    plot_difference(param_sensitivity, param)

box_plot(param_sensitivity)