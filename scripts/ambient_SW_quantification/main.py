# author meta data:
__name__ = "Syed Raza"
__email__ = "sar0033@uah.edu"

# importing my modules
import sw_data_process as sw
import machine_learning_protocols as ml
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Data analysis pipeline:
############################################################################################################
# get the file path for the donki data
file_path = "/Users/syedraza/cmeuncerpy/Data/ambient_SW_quantification/DONKI_new.xlsx"

# Call the clean donki function on this file path
donki = sw.clean_donki(file_path)

# Getting the cleaned OMNI data
sw_data_location = "/Users/syedraza/cmeuncerpy/Data/ambient_SW_quantification/OMNI_COHO1HR_MERGED_MAG_PLASMA_137596.txt"
omni = sw.clean_omni_data(sw_data_location)

# Getting the observed processed wind and the simulated processed wind data frames:
processed_wind_obs = sw.process_observed_parameters(omni, donki)
processed_wind_sim = sw.process_simulated_parameters(donki, "/Users/syedraza/cmeuncerpy/Data/ambient_SW_quantification/ENLIL_data")

# the cme data frame:
cme_data = sw.residual_wind(processed_wind_obs, processed_wind_sim, donki)

# Convert everything to string as-is
cme_data_str = cme_data.astype(str)

# Compute column widths with a bit of padding
col_spacing = 4  # space between columns
widths = {col: max(len(col), cme_data_str[col].map(len).max()) + col_spacing for col in cme_data_str.columns}

# Write to .dat
with open("cme_data.dat", "w") as f:
    # Header
    f.write("".join(f"{col:<{widths[col]}}" for col in cme_data_str.columns) + "\n")
    # Rows
    for _, row in cme_data_str.iterrows():
        f.write("".join(f"{val:<{widths[col]}}" for col, val in row.items()) + "\n")

# making the parameter info for plotting the data analysis plots:
parameter_info = {
    "B_diff": (r"$\Delta B_{SW}$ (G)", r"figures/correlation_B.png"),
    "V_diff": (r"$\Delta v_{SW}$ (km/s)", r"figures/correlation_v.png"),
    "n_diff": (r"$\Delta n_{SW}$ ($cm^{-3}$)", r"figures/correlation_n.png"),
    "T_diff": (r"$\Delta T_{SW}$ (K)", r"figures/correlation_T.png"),
    "P_diff": (r"$\Delta P_{SW}$ (Barye)", r"figures/correlation_P.png"),
    "long": ("Longitude-CONE model (degrees)", r"figures/correlation_long.png"),
    "lat": ("Latitude-CONE model (degrees)", r"figures/correlation_lat.png"),
    "speed": (r"Speed-CONE model (km/s)", r"figures/correlation_speed.png"),
    "half_width": ("half width-CONE model (degrees)", r"figures/correlation_half_width.png"),
}

# making and showing the data analysis plots:
sw.data_analysis_plots(cme_data, parameter_info)

############################################################################################################

# Machine Learning pipeline:
############################################################################################################
# defining the features for the machine learning model:
# normalize the cme_data:
cme_data = ml.normalize_data(cme_data, ["PE", "time_21_5", 'cme_transit_obs', 'cme_transit_sim'])
# save cme_data as a .xlsx file:
cme_data.reset_index(drop=True).to_excel("cme_data.xlsx", index=False)

# all the features:
features = ["B_diff", "P_diff", "speed", "half_width", "n_diff", "lat", "V_diff", "long", "T_diff"]

# uni results:
if Path("ml_PE.xlsx").is_file():
    # read in the file
    ml_data = pd.read_excel("ml_PE.xlsx")
    uni_mae = {"B_diff": {"knn": None, "svm": None, "lr": None}, 
               "P_diff": {"knn": None, "svm": None, "lr": None},
               "V_diff": {"knn": None, "svm": None, "lr": None},
               "T_diff": {"knn": None, "svm": None, "lr": None},
               "n_diff": {"knn": None, "svm": None, "lr": None},
               "speed": {"knn": None, "svm": None, "lr": None},
               "half_width": {"knn": None, "svm": None, "lr": None},
               "lat": {"knn": None, "svm": None, "lr": None},
               "long": {"knn": None, "svm": None, "lr": None},}

    # go through each column and make the dictionary like before
    for i in ml_data.columns:
        feature = i[i.find("_") + 1:i.rfind("_")]
        method = i[i.rfind("_") + 1:]
        
        uni_mae[feature][method] = ml_data[i].abs().mean()

else:
    # calling the univariate model on these features:
    uni_mae = ml.uni_LOOCV(cme_data, features) 
    print(uni_mae)

ml.uni_mae_plots(uni_mae, "figures/uni_mae_plots_new.png", cme_data)

# ranking the uni_features:
uni_ranked_features = ml.ranked_univariate(uni_mae, "figures/ranked_features_new.png")
print(uni_ranked_features)

# multi results:
# multi_mae = ml.multi_LOOCV(cme_data, features)
# print(multi_mae)

# Getting the univariate and multivariate transit data:

# print("These are the univariate ml results: ", uni_mae)
# print("These are the multivariate results: ", multi_mae)

# plot this figure:
# ml.uni_mae_plots(uni_mae, multi_mae, "figures/ml_mae.png", cme_data, features)