# This code is to calculate the sensitivity of the Magnetic field (magnitude and B_z) 
# Based on Talwinder's 2023 simulation results

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

########################################
# Settings 
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

########################################
# The main code

gcs_params_1 = pl.from_pandas(
    pd.read_csv(address + "GCS_parms1.dat", sep=r"\s+", header=None, 
    names = ["ensemble_member", "speed", "latitude", "longitude", "tilt", "half_angle", "aspect_ratio"])
)

# get the iunique values of each GCS parameter
unique_speeds = gcs_params_1["speed"].unique().to_list()
unique_lat = gcs_params_1["latitude"].unique().to_list()
unique_lon = gcs_params_1["longitude"].unique().to_list()
unique_tilt = gcs_params_1["tilt"].unique().to_list()
unique_half_angle = gcs_params_1["half_angle"].unique().to_list()
unique_aspect_ratio = gcs_params_1["aspect_ratio"].unique().to_list()

# calculate the upper and lower bounds in the ensmeble spread
low_speed, high_speed = unique_speeds[0], unique_speeds[2]
low_lat, high_lat = unique_lat[0], unique_lat[2]
low_lon, high_lon = unique_lon[0], unique_lon[2]
low_tilt, high_tilt = unique_tilt[0], unique_tilt[2]
low_half_angle, high_half_angle = unique_half_angle[0], unique_half_angle[2]
low_aspect_ratio, high_aspect_ratio = unique_aspect_ratio[0], unique_aspect_ratio[2]

# generate the pairs
speed_pairs = generate_pairs("speed", low_speed, high_speed, ["latitude", "longitude", "tilt", "half_angle", "aspect_ratio"], gcs_params_1)
lat_pairs = generate_pairs("latitude", low_lat, high_lat, ["speed", "longitude", "tilt", "half_angle", "aspect_ratio"], gcs_params_1)
lon_pairs = generate_pairs("longitude", low_lon, high_lon, ["speed", "latitude", "tilt", "half_angle", "aspect_ratio"], gcs_params_1)
tilt_pairs = generate_pairs("tilt", low_tilt, high_tilt, ["speed", "latitude", "longitude", "half_angle", "aspect_ratio"], gcs_params_1)
half_angle_pairs = generate_pairs("half_angle", low_half_angle, high_half_angle, ["speed", "latitude", "longitude", "tilt", "aspect_ratio"], gcs_params_1)
aspect_ratio_pairs = generate_pairs("aspect_ratio", low_half_angle, high_half_angle, ["speed", "latitude", "longitude", "tilt", "half_angle"], gcs_params_1)

print(lon_pairs)