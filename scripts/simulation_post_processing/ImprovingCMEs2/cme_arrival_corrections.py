# These are the ML experiments taht will be reported in the paper
# There are two modes to this:
#               (1) The first one finds the time common between the STEREO A 
#                   and STEREO B and gives a MAE estimate per minute. This 
#                   will tell us how the MAE is changing over time
#               (2) This mode will take only the last 90 minutes and provide the 
#                   mean and std

# THIS FILE REPRESENTS THE SECOND MODE. 


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

which_craft = "AB"              # this can be A, B, or AB
cme_table = False               # make a .csv file with CME01 features. Most of the time it should be false

address = "../../../Data/ImprovingCMEs2/"

minutes = 90                    # how many last minutes to use
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
                        "ML_TT_mean": pl.Float64, 
                        "ML_TT_std": pl.Float64,
                        "Seed_error": pl.Float64, 
                        "ML_error_mean": pl.Float64,
                        "ML_error_std": pl.Float64})

cmes = ['01_2010-04-03', '02_2010-05-23', '03_2010-08-01', '04_2011-09-06', 
        '05_2011-09-13', '06_2011-10-22', '07_2012-01-19', '08_2012-03-07', 
        '09_2012-06-14', '10_2012-07-03', '11_2012-07-12' , '12_2012-09-27', 
        '13_2012-10-05']

# making the file for storage
outfile = f"diff_results/ML_diff_results_{which_craft}.csv"
with open(outfile, 'w') as f:
    f.write('CME_num,Actual_TT,Seed_TT,ML_TT_mean,ML_TT_std,Seed_error,ML_error_mean,ML_error_std\n')

for cme in cmes:
    # the two things that need uncertainties 
    ML_uncertainty = {"ML_TT" : [], "ML_error": []}

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

    # make the table for the paper
    if cme_table:
        # only the final times on both data frames 
        data_A = data_A.filter(
            pl.col("Time") == (pl.col("Time").max())
        )
        data_B = data_B.filter(
            pl.col("Time") == (pl.col("Time").max())
        )

        # join the dataframes 
        result = data_A.join(data_B, on = "ensemble_member")

        # delete columns, change column names and order them
        result = result.drop(["Time_since_eruption", "Time_since_eruption_right", "Travel_time_right"])
        result = result.rename({"Time" : "Time_A", "Time_right" : "Time_B"})
        result = result.select(["Time_A", "Time_B", "ensemble_member", "EA_diff_A", "EA_diff_B", "Travel_time"])
        result.write_csv(f"cme_{cme}_table.csv")
        sys.exit("Table is written. Make the variable cme_01_table == False for the code to work!!")

    # convert string to datetime object
    data_A = data_A.with_columns(
        pl.col("Time").str.to_datetime(format = "%Y-%m-%d %H:%M:%S")
    )

    data_B = data_B.with_columns(
        pl.col("Time").str.to_datetime(format = "%Y-%m-%d %H:%M:%S")
    )

    # also grab the seed travel time here. look for the travel_time columnm of ensemble member 11
    seed_travel_time = data_A.filter(pl.col("ensemble_member") == 11).select(pl.col("Travel_time")).item(0, 0)
    
    # grab the last minutes (global variable that you can change) of data for both A and B
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
        # pick the data frame A at the smallest
        data_filtered_A = data_A.filter(
            pl.col("Time") == unique_times_A[i]
        )
        data_filtered_B = data_B.filter(
            pl.col("Time") == unique_times_B[i]
        )

        # now do the ML here
        if which_craft == "AB":
            X = pl.concat(
                [data_filtered_A[["EA_diff_A"]], data_filtered_B[["EA_diff_B"]]],
                how = "horizontal"
            )

        elif which_craft == "A":
            X = data_filtered_A[["EA_diff_A"]]

        elif which_craft == "B":
            X = data_filtered_B[["EA_diff_B"]]

        y = data_filtered_A[["Travel_time"]]
        y = y.to_numpy().ravel()

        # make the ML model (try linear models)
        model = linear_model.LassoLarsCV(cv = 2)                        
        # model = linear_model.RANSACRegressor()
        # model = linear_model.TheilSenRegressor()
        
        # train the model
        model.fit(X, y)

        # now to inference - pass them in as ['EA_diff_A', 'EA_diff_B']
        if which_craft == "AB":
            X_inference = pl.DataFrame([[0.0], [0.0]],
                schema = ['EA_diff_A', 'EA_diff_B'])
            
        elif which_craft == "A":
            X_inference = pl.DataFrame([[0.0]],
                schema = ["EA_diff_A"])
            
        elif which_craft == "B":
            X_inference = pl.DataFrame([[0.0]],
                schema = ["EA_diff_B"])

        prediction = model.predict(X_inference)[0] # to have it as a float because model.predict() returns an array

        # calculate the errors for analysis later
        seed_error = cme_obs_travel_time - seed_travel_time
        ML_error = cme_obs_travel_time - prediction
        ML_uncertainty["ML_TT"].append(prediction)
        ML_uncertainty["ML_error"].append(ML_error)

    # make the new row and concatenate with df_out
    new_inference = pl.DataFrame([{"CME_num": cme, "Actual_TT": cme_obs_travel_time, "Seed_TT": seed_travel_time, "ML_TT_mean": np.mean(ML_uncertainty["ML_TT"]), "ML_TT_std": np.std(ML_uncertainty["ML_TT"]), "Seed_error": seed_error, "ML_error_mean": np.mean(ML_uncertainty["ML_error"]), "ML_error_std": np.std(ML_uncertainty["ML_error"])}])

    df_out = pl.concat([df_out, new_inference])

    with open(outfile, 'a') as f:
        f.write(f"{cme},{cme_obs_travel_time:.2f},{seed_travel_time:.2f},{np.mean(ML_uncertainty["ML_TT"]):.2f},"
            f"{np.std(ML_uncertainty["ML_TT"]):.2f},{seed_error:.2f},{np.mean(ML_uncertainty["ML_error"]):.2f},{np.std(ML_uncertainty["ML_error"]):.2f}\n")
        

# after this is done, you can derive statistics (me, mae, and std) from the written file :)
results = pl.read_csv(outfile)
print("ML MAE = ", results["ML_error_mean"].abs().mean())
print("ML ME = ", results["ML_error_mean"].mean())
print("ML STD = ", results["ML_error_mean"].std())
print("Seed MAE = ", results["Seed_error"].abs().mean())
print("Seed ME = ", results["Seed_error"].mean())
print("Seed STD = ", results["Seed_error"].std())