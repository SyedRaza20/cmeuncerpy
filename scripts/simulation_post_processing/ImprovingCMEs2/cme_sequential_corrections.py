# These are the ML experiments taht will be reported in the paper
# There are two modes to this:
#               (1) The first one finds the time common between the STEREO A 
#                   and STEREO B and gives a MAE estimate per minute. This 
#                   will tell us how the MAE is changing as the observations 
#                   come in
#               (2) This mode will take only the last 90 minutes and provide the 
#                   mean and std

# THIS FILE REPRESENTS THE FIRST MODE. 


# the imports
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import timedelta, datetime, date
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# SETTINGS
################################################################################
pl.Config.set_tbl_cols(-1)      # polars settings to show all the columns
pl.Config.set_tbl_rows(-1)      # polars settings to show all the rows

address = "../../../Data/ImprovingCMEs2/"
################################################################################
# HELPER FUNCTIONS
################################################################################
# to read times from files
def read_time_from_file(filename):
    with open(filename, 'r') as f:
        last_line = f.readlines()[-1]
    last_line = last_line.split()
    time = last_line[0]
    #make it datetime object
    time = datetime.strptime(time, '%Y/%m/%dT%H:%M')
    return time
################################################################################

cmes = ['01_2010-04-03', '02_2010-05-23', '03_2010-08-01', '04_2011-09-06', 
        '05_2011-09-13', '06_2011-10-22', '07_2012-01-19', '08_2012-03-07', 
        '09_2012-06-14', '10_2012-07-03', '11_2012-07-12' , '12_2012-09-27', 
        '13_2012-10-05']

for cme in cmes:
    # make an outfile for this cme
    outfile = f"sequential_results/{cme}.csv"
    with open(outfile, 'w') as f:
        f.write('CME_num,Actual_TT,Seed_TT,ML_TT_mean,ML_TT_std,Seed_error,ML_error_mean,ML_error_std\n')

    # get the cme eruption time
    erupt_file = address + "erupt_and_arrival_times/" + cme + "/Erupt_time.txt"
    erupt_time = read_time_from_file(erupt_file)

    # get the cme arrival time 
    arrival_file = address + "erupt_and_arrival_times/" + cme + "/Arrival_time.txt"
    arrival_time = read_time_from_file(arrival_file)

    # what is the cme travel time in hours
    cme_obs_travel_time = (arrival_time - erupt_time).total_seconds() / 3600

    # read the data from the train file
    data_A = pl.read_csv(address + 'training_data/Train_diff_' + cme + '_A_v2.txt')
    data_B = pl.read_csv(address + 'training_data/Train_diff_' + cme + '_B_v2.txt')

    # find the seed travel time
    seed_travel_time = data_A.filter(pl.col("ensemble_member") == 11).select(pl.col("Travel_time")).item(0, 0)

    # Combine the two data frames
    data_intersection = data_A.join(data_B, on = ["Time", "ensemble_member"], how = "inner")
    data_intersection = data_intersection.select(["Time", "ensemble_member", "EA_diff_A", "EA_diff_B", "Travel_time"])
    data_intersection = data_intersection.sort("Time")

    unique_times = (data_intersection
                    .select(pl.col("Time").unique())
                    .sort("Time")
                    .to_series()
                    .to_list()
                )

    # do a for loop over time now
    for time in unique_times:
        # grab the training data for this time
        time_data = data_intersection.filter(pl.col("Time") == time)

        X = time_data[["EA_diff_A", "EA_diff_B"]]
        y = time_data[["Travel_time"]]
        y = y.to_numpy().ravel()

        # make the model
        model = linear_model.LassoLarsCV(cv = 2)

        # train the model
        model.fit(X,y)

        # make the inference
        X_inference = pl.DataFrame([[0.0], [0.0]],
                schema = ["EA_diff_A", "EA_diff_B"])
        
        # make the zero point prediction
        prediction = model.predict(X_inference)[0]
        
        # calculate seed and ML errors
        seed_error = cme_obs_travel_time - seed_travel_time
        ML_error = cme_obs_travel_time - prediction

        # 1. Ensure the directory path exists before writing
        output_dir = os.path.dirname(outfile)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 2. Open and safely append/create the file
        with open(outfile, 'a') as f:
            f.write(f"{cme},{cme_obs_travel_time:.2f},{seed_travel_time:.2f},{prediction:.2f},{seed_error:.2f},{ML_error:.2f}\n")