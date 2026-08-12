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

which_craft = "AB" # this can be A, B, or AB

address = "../../../Data/ImprovingCMEs2/"
what_order = "linear"

minutes = 60                    # how many last minutes to use
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
# the main code 

 # the out data frame
df_out = pl.DataFrame(schema = {"CME_num": pl.String, 
                        "Actual_TT": pl.Float64, 
                        "Seed_TT": pl.Float64, 
                        "ML_TT": pl.Float64, 
                        "Seed_error": pl.Float64, 
                        "ML_error": pl.Float64})

cmes = ['01_2010-04-03', '02_2010-05-23', '03_2010-08-01', '04_2011-09-06', 
        '05_2011-09-13', '06_2011-10-22', '07_2012-01-19', '08_2012-03-07', 
        '09_2012-06-14', '10_2012-07-03', '11_2012-07-12' , '12_2012-09-27', 
        '13_2012-10-05']

# read in the observed linear fit
obs_linear_A_fit = pl.read_csv(address + 'training_data/obs_coefficients_' + what_order + '_A_v2.txt')
obs_linear_B_fit = pl.read_csv(address + 'training_data/obs_coefficients_' + what_order + '_B_v2.txt')

# reset/create all output files ONCE per run, before any processing
for i in range(minutes + 1):
    outfile = f"ML_Linear_results_{which_craft}_{i}.txt"
    with open(outfile, 'w') as f:
        f.write('CME_num,Actual_TT,Seed_TT,ML_TT,Seed_error,ML_error\n')

for cme in cmes:
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

    # convert string to datetime object
    data_A = data_A.with_columns(
        pl.col("Time").str.to_datetime(format = "%Y-%m-%d %H:%M:%S")
    )

    data_B = data_B.with_columns(
        pl.col("Time").str.to_datetime(format = "%Y-%m-%d %H:%M:%S")
    )

    # also grab the seed travel time here. look for the travel_time columnm of ensemble member 11
    seed_travel_time = data_A.filter(pl.col("ensemble_member") == 11).select(pl.col("Travel_time")).item(0, 0)
    
    # grab the last minutes of data for both A and B
    data_A = data_A.filter(
        pl.col("Time") >= (pl.col("Time").max() - pl.duration(minutes = minutes))
    ).sort("Time")

    data_B = data_B.filter(
        pl.col("Time") >= (pl.col("Time").max() - pl.duration(minutes = minutes))
    ).sort("Time")

    # unique times for both A and B
    unique_times_A = data_A["Time"].unique(maintain_order = True).to_list()
    unique_times_B = data_B["Time"].unique(maintain_order = True).to_list()

    for i in range(minutes + 1):
        outfile = f"ML_Linear_results_{which_craft}_{i}.txt"

        # pick the data frame A at the smallest
        data_filtered_A = data_A.filter(
            pl.col("Time") == unique_times_A[i]
        )
        data_filtered_B = data_B.filter(
            pl.col("Time") == unique_times_B[i]
        )

        # now do the ML here
        X = pl.concat(
            [data_filtered_A[["EA_diff_A"]], data_filtered_B[["EA_diff_B"]]],
            how = "horizontal"
        )

        y = data_filtered_A[["Travel_time"]]
        y = y.to_numpy().ravel()

        # make the ML model
        # model = linear_model.LassoLarsCV(cv = 2)
        # model = linear_model.RidgeCV(alphas = [0.1, 1.0, 10.0])
        # model = linear_model.ElasticNet(alpha=0.01, l1_ratio=0.5) 
        # model = linear_model.Lasso(alpha=0.01)
        # model = linear_model.BayesianRidge()
        model = linear_model.RANSACRegressor()
        # model = linear_model.TheilSenRegressor()

        # train the model
        model.fit(X, y)

        # now to inference - pass them in as ['EA_diff_A', 'EA_diff_B']
        X_inference = pl.DataFrame([[0.0], [0.0]],
            schema = ['EA_diff_A', 'EA_diff_B'])

        prediction = model.predict(X_inference)[0] # to have it as a float because model.predict() returns an array

        # calculate the errors for analysis later
        seed_error = cme_obs_travel_time - seed_travel_time
        ML_error = cme_obs_travel_time - prediction

        # make the new row and concatenate with df_out
        new_inference = pl.DataFrame([{"CME_num": cme, "Actual_TT": cme_obs_travel_time, "Seed_TT": seed_travel_time, "ML_TT": prediction, "Seed_error": seed_error, "ML_error": ML_error}])

        df_out = pl.concat([df_out, new_inference])

        with open(outfile, 'a') as f:
            f.write(f"{cme},{cme_obs_travel_time:.2f},{seed_travel_time:.2f},"
                    f"{prediction:.2f},{seed_error:.2f},{ML_error:.2f}\n")