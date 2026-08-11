import numpy as np
import pandas as pd
import polars as pl
import pickle
from scipy.stats import skew, kurtosis
import warnings
import sys
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# some metadata for data peparation
fourier = False
nan_threshold = 0.10
use_sheath = True
time_series_plot = False
hours_features = 0
sw_data_saved = True

# ------------------ Helper Functions ------------------

def insitu_dt(insitu: np.ndarray, offset_jul: bool) -> pl.DataFrame:
    """
    Takes in a np.ndarray of the insitu measurements and: 
        (1) fixes the time lag problems 
        (2) returns it as a polars dataframe

    Params:
        (1) insitu (np.ndarray)
            The dataframe 
        (2) offset_jul (boolean)
            To see if 719163 off set is needed. I have to do this because the Martin's data needs this,
            but Dinesh's data does not need this.
    """
    # change to polars
    insitu_polars = pl.from_pandas(pd.DataFrame(insitu))

    # fix the time lag and convert to datetime objects
    if offset_jul == True:
        insitu_polars = insitu_polars.with_columns(
            (pl.datetime(1970, 1, 1) + pl.duration(days=pl.col("time") - 719163))
            .alias("time")
        )

    if offset_jul == False:
        insitu_polars = insitu_polars.with_columns(
            (pl.datetime(1970, 1, 1) + pl.duration(days=pl.col("time")))
            .alias("time")
        )

    return insitu_polars

def CME_insitu_extraction(insitu_data: pl.DataFrame, icme_start: pl.Datetime, mo_start: pl.Datetime, mo_end: pl.Datetime, params: list[str], hours: int, which: str) -> list:
    """
    This function extracts the feature or target data for a given CME event based on its magnetic obstacle (MO) 
    start and end times.

    Params:
    - icme_start (pl.Datetime): The start time of the ICME sheath.
    - mo_start (pl.Datetime): The start time of the magnetic obstacle.
    - mo_end (pl.Datetime): The end time of the magnetic obstacle.
    - insitu_data (pl.DataFrame): The in-situ data DataFrame containing time and plasma parameters.
    - params (list[str]): A list of plasma parameter column names to extract as features.
    - hours (int): The number of hours of data to extract for feature extraction
    
    Returns:
    - feature_data (pl.DataFrame): DataFrame containing the extracted features within the specified time range.
    - target_data (pl.DataFrame): DataFrame containing the extracted target data within the MO duration.
    """

    # get the features in the given time range
    if which == "features":
        feature_mask = (insitu_data["time"] > icme_start) & (insitu_data["time"] < mo_start + pl.duration(hours=hours))
        feature_data = insitu_data.filter(feature_mask).select(['time'] + params)
        return feature_data
    
    # get the target data in the MO duration
    elif which == "target":
        target_mask = (insitu_data["time"] > mo_start) & (insitu_data["time"] < mo_end)
        target_data = insitu_data.filter(target_mask).select(['time'] + ['bz', 'bt'])
        return target_data
    
    # check for invalid 'which' parameter
    else:
        raise ValueError("Parameter 'which' must be either 'features' or 'target'.")
    
def make_features(feature_data: pl.DataFrame, cme_id: str, params: list, fourier: bool) -> pl.DataFrame:
    """
    This function computes statistical and Fourier features from the provided feature data.

    Params:
    - feature_data (pl.DataFrame): DataFrame containing time series data for various plasma parameters.
    - cme_id (str): The unique identifier for the CME event.
    - params (list): List of plasma parameter column names to compute features for.
    - fourier (bool): Flag indicating whether to compute Fourier features.

    Returns:
    - features (pl.DataFrame): DataFrame containing the computed features for the CME event.
    """
    features = {}

    features['icmecat_id'] = cme_id

    for par in params:
        arr = np.asarray(feature_data[par], float)

        features[f'{par}_mean'] = np.mean(arr)
        features[f'{par}_std']  = np.std(arr)
        features[f'{par}_min']  = np.min(arr)
        features[f'{par}_max']  = np.max(arr)
        features[f'{par}_minmax'] = features[f'{par}_max'] / features[f'{par}_min']
        features[f'{par}_meanstd'] = features[f'{par}_mean'] / features[f'{par}_std']
        features[f'{par}_skew'] = skew(arr)
        features[f'{par}_kurtosis'] = kurtosis(arr)

    if fourier:
        for par in params:
            arr = np.asarray(feature_data[par], float)

            F = np.fft.rfft(arr)
            mag = np.abs(F)
            pwr = mag**2

            features[f'{par}_fft_mag_mean'] = float(np.mean(mag))
            features[f'{par}_fft_mag_max']  = float(np.max(mag))
            features[f'{par}_fft_mag_std']  = float(np.std(mag))
            features[f'{par}_fft_pow_mean'] = float(np.mean(pwr))
            features[f'{par}_fft_pow_max']  = float(np.max(pwr))
            features[f'{par}_fft_pow_std']  = float(np.std(pwr))

    return pl.DataFrame([features])

def imputation(df : pl.DataFrame) -> pl.DataFrame:
    """
    This function performs simple imputation on the DataFrame by: 
        (1) linear interpolation for NaN values.
        (2) rest of the NaN values are replacede with means.

    Params:
    - df (pl.DataFrame): The input DataFrame with potential NaN values.

    Returns:
    - df_imputed (pl.DataFrame): The DataFrame with NaN values filled.
    """
    
    # Get numerical columns only
    numerical_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]]
    
    # linear interpolation only on the numerical columns 
    for col in numerical_cols:
        df = df.with_columns(
            pl.col(col).interpolate(method="linear").alias(col)
        )

    # fill remaining NaNs with mean
    for col in numerical_cols:
        mean_value = df.select(pl.col(col).mean()).to_numpy()[0][0]
        df = df.with_columns(
            pl.col(col).fill_null(mean_value).alias(col)
        )

    return df

def time_series_CME_plot(icme_start: pl.Datetime, mo_start: pl.Datetime, mo_end: pl.Datetime, insitu_data: pl.DataFrame, cme_id: str):
    """
    This function plots the time series data for a given CME event, highlighting the training and testing periods.
    The training period is icme_sheath to mo_start + 4 hours, and the testing period is mo_start to mo_end, make a 
    legend indicating each period. Highlight them in dashed red and dashed green lines respectively. and only make 
    them for B_tot and B_z arrays for the CME event. They are stored as bt and bz in the time series data. 

    Params:
    - icme_start (pl.Datetime): The start time of the ICME sheath.
    - mo_start (pl.Datetime): The start time of the magnetic obstacle.
    - mo_end (pl.Datetime): The end time of the magnetic obstacle.
    - insitu_data (pl.DataFrame): The in-situ data DataFrame containing time and plasma parameters.
    - cme_id (str): The unique identifier for the CME event.
    """

    # Only plot bt and bz
    params_to_plot = ['bt', 'bz']
    params_labels = {'bt': '$B_{tot}$', 'bz': '$B_z$'}
    
    plt.figure(figsize=(15, 10))
    
    # Convert to datetime if needed and calculate training period end time
    from datetime import timedelta
    
    # Handle both datetime.datetime and polars datetime types
    if hasattr(icme_start, 'to_pydatetime'):
        icme_start_dt = icme_start.to_pydatetime()
        mo_start_dt = mo_start.to_pydatetime()
        mo_end_dt = mo_end.to_pydatetime()
    else:
        icme_start_dt = icme_start
        mo_start_dt = mo_start
        mo_end_dt = mo_end
    
    training_period_end_dt = mo_start_dt + timedelta(hours=4)
    
    # Calculate plot time window - show context before and after CME
    plot_start_dt = icme_start_dt - timedelta(hours=4)
    plot_end_dt = mo_end_dt + timedelta(hours=4)
    
    # Filter data for the plot window
    plot_mask = (insitu_data['time'] >= plot_start_dt) & (insitu_data['time'] <= plot_end_dt)
    plot_data = insitu_data.filter(plot_mask)

    for i, par in enumerate(params_to_plot):
        plt.subplot(len(params_to_plot), 1, i + 1)
        plt.plot(plot_data['time'], plot_data[par], label=params_labels[par])
        
        # Training period boundaries: icme_start to mo_start + 4 hours (green, dashed lines)
        plt.axvline(icme_start_dt, color='green', linestyle='--', linewidth=2, label='Training phase')
        plt.axvline(training_period_end_dt, color='green', linestyle='--', linewidth=2)
        
        # Testing period boundaries: mo_start to mo_end (red, dashed lines)
        plt.axvline(mo_start_dt, color='red', linestyle='--', linewidth=2, label='prediction phase')
        plt.axvline(mo_end_dt, color='red', linestyle='--', linewidth=2)
        
        plt.title(f'CME ID: {cme_id} - {params_labels[par]} Time Series', fontsize=18)
        plt.xlabel('Time', fontsize=16)
        plt.ylabel(f'{params_labels[par]} (nT)', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(fontsize=14)

    plt.tight_layout(h_pad=3.0)
    plt.savefig(f"figures_and_metrics/time_series_CME_{cme_id}.png", dpi=300, bbox_inches='tight')
    plt.show()

def load_SW_from_pickle(sc: str):
    """
    Loads the pickle data from the Data directory and writes out the csv file for the given space craft.

    The way to load the data is taken from my old code and it can be done as follows
        (1) For wind 
            [win, winheader] = pickle.load(open("directory_link", "rb"))
        (2) For sta 
            [sta, atta] = pickle.load(open("directory_link", "rb"))
        (3) for stb
            [stb, attb, stbheader] = pickle.load(open("directory_link", "rb"))

    However, the newer data given to me by Dinesh is structured in a different way. I have the pickle file 
    from 2007-2021 for wind and sta. from 2021-2025 there are 5 different files. In this function, I will
    read them separately and combine them vertically into a polars dataframe before saving in insitu directory
    """

    if sc.lower() == "wind": 

        [win1, win1header] = pickle.load(open("../../Data/pickle_files/wind_2007_2021_heeq_ndarray.p", "rb"))
        [win2, win2header] = pickle.load(open("../../Data/pickle_files/wind_2021_2021_heeq_ndarray.p", "rb"))
        [win3, win3header] = pickle.load(open("../../Data/pickle_files/wind_2022_2022_heeq_ndarray.p", "rb"))
        [win4, win4header] = pickle.load(open("../../Data/pickle_files/wind_2023_2023_heeq_ndarray.p", "rb"))
        [win5, win5header] = pickle.load(open("../../Data/pickle_files/wind_2024_2024_heeq_ndarray.p", "rb"))
        [win6, win6header] = pickle.load(open("../../Data/pickle_files/wind_2025_2025_heeq_ndarray.p", "rb"))

        # 2021 overlapping period where I want to use Dinesh's data

        wind1 = insitu_dt(win1, offset_jul=True)
        wind1 = wind1.filter(pl.col("time") < pl.datetime(2021, 1, 1))

        # also reorder the columns for concatenation
        wind1 = wind1.select(['time', 'bt', 'bx', 'by', 'bz', 'vt', 'np', 
                              'tp', 'x', 'y', 'z', 'r', 'lat', 'lon'])

        wind2 = insitu_dt(win2, offset_jul=False)
        wind3 = insitu_dt(win3, offset_jul=False)
        wind4 = insitu_dt(win4, offset_jul=False)
        wind5 = insitu_dt(win5, offset_jul=False)
        wind6 = insitu_dt(win6, offset_jul=False)

        # concatenate the polars data frames vertically 
        wind = pl.concat([wind1, wind2, wind3, wind4, wind5, wind6], how="vertical")

        return wind

    elif sc.lower() == "sta":

        [sta1, atta1] = pickle.load(open("../../Data/pickle_files/stereoa_2007_2021_sceq_ndarray.p", "rb"))
        [sta2, atta2] = pickle.load(open("../../Data/pickle_files/stereoa_2021_2021_sceq_reference_like_ndarray.p", "rb"))
        [sta3, atta3] = pickle.load(open("../../Data/pickle_files/stereoa_2022_2022_sceq_reference_like_ndarray.p", "rb"))
        [sta4, atta4] = pickle.load(open("../../Data/pickle_files/stereoa_2023_2023_sceq_reference_like_ndarray.p", "rb"))
        [sta5, atta5] = pickle.load(open("../../Data/pickle_files/stereoa_2024_2024_sceq_reference_like_ndarray.p", "rb"))
        [sta6, atta6] = pickle.load(open("../../Data/pickle_files/stereoa_2025_2025_sceq_reference_like_ndarray.p", "rb"))

        # 2021 overlapping period
        sta1 = insitu_dt(sta1, offset_jul=True)
        sta1 = sta1.filter(pl.col("time")  < pl.datetime(2021, 1, 1))

        # reorder columns for concatenation 
        sta1 = sta1.select(['time', 'bt', 'bx', 'by', 'bz', 'vt', 'np', 
                            'tp', 'x', 'y', 'z', 'r', 'lat', 'lon'])

        # reoreder columns for concatenation
        sta2 = insitu_dt(sta2, offset_jul=False)
        sta3 = insitu_dt(sta3, offset_jul=False)
        sta4 = insitu_dt(sta4, offset_jul=False)
        sta5 = insitu_dt(sta5, offset_jul=False)
        sta6 = insitu_dt(sta6, offset_jul=False)

        sta = pl.concat([sta1, sta2, sta3, sta4, sta5, sta6], how="vertical")

        return sta
    
    elif sc.lower() == "stb":
        [stb, attb, stbheader] = pickle.load(open("../../Data/pickle_files/stereob_2007_2014_sceq_ndarray.p", "rb"))
        stb = insitu_dt(stb, offset_jul=True)

        return stb

# ------------------ The Main Function ------------------

if __name__ == "__main__":
    print("Starting Data Preparation...")
    sys.stdout.flush()

    # ------------------ Data Loading ------------------
    # Loading the CME data: version 23 of the Helio4Cast catalog
    CME_data = pl.read_csv("https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v23.csv")

    # Clean column names by stripping leading/trailing spaces
    CME_data = CME_data.rename({col: col.strip() for col in CME_data.columns})

    # Only the data columns are needed 
    columns = ['icmecat_id', 'sc_insitu', 'icme_start_time', 'mo_start_time', 'mo_end_time']

    CME_data = CME_data.select(columns)

    # to grab the indices with win, sta, and stb
    CME_data = CME_data.with_row_index("idx")
    
    # Replace 'T' with ' ' and 'Z' with '' to make the datetime format explicit
    CME_data = CME_data.with_columns(
        pl.col("icme_start_time").str.replace("T", " ").str.replace("Z", "").str.to_datetime("%Y-%m-%d %H:%M"),
        pl.col("mo_start_time").str.replace("T", " ").str.replace("Z", "").str.to_datetime("%Y-%m-%d %H:%M"),
        pl.col("mo_end_time").str.replace("T", " ").str.replace("Z", "").str.to_datetime("%Y-%m-%d %H:%M"),
    )

    # filter out the stuff before 2007
    CME_data = CME_data.filter(pl.col("icme_start_time") > pl.datetime(2006, 12, 31))

    # Loading the SW data now. Testing for wind
    wind_data = load_SW_from_pickle("wind")
    sta_data = load_SW_from_pickle("sta")
    stb_data = load_SW_from_pickle("stb")

    # saving them as .csv files
    if sw_data_saved == False:
        print("saving wind, sta, and stb data as .csv files")

        wind_data.write_csv("../../Data/insitu_csv/wind_insitu.csv")
        sta_data.write_csv("../../Data/insitu_csv/sta_insitu.csv")
        stb_data.write_csv("../../Data/insitu_csv/stb_insitu.csv")

        print("The SW data is saved")

    # a checkpoint
    # sys.exit()

    # ------------------ Data Preprocessing ------------------

    print("Data preprocessing: ")

    if use_sheath == True:
        # get event indices where icme_start_time does not match mo_start_time
        valid_events_mask = CME_data["icme_start_time"] != CME_data["mo_start_time"]
        CME_data = CME_data.filter(valid_events_mask)

    # we only using wind, sta, and stb:
    valid_spacecrafts = ["WIND", "STEREO-A", "STEREO-B"]

    valid_spacecrafts_mask = (
        CME_data["sc_insitu"]
            .str.strip_chars()
            .str.to_uppercase()
            .is_in(valid_spacecrafts)
    )

    CME_data = CME_data.filter(valid_spacecrafts_mask)

    insitu_data_map = {
        "STEREO-A": sta_data, 
        "STEREO-B": stb_data,
        "WIND": wind_data,
    }

    # checkpoint
    # sys.exit()
    # ------------------ Time Series Plot ------------------

    if time_series_plot == True:
        # cme number - pick any number, its just for the picture
        num = 23

        cme_id = CME_data[num, "icmecat_id"]
        sc = CME_data[num, "sc_insitu"].strip().upper()
        icme_start = CME_data[num, "icme_start_time"]
        mo_start = CME_data[num, "mo_start_time"]
        mo_end = CME_data[num, "mo_end_time"]

        first_insitu_data = insitu_data_map.get(sc)
        time_series_CME_plot(icme_start, mo_start, mo_end, first_insitu_data, cme_id)

    # checkpoint
    # sys.exit()

    # ------------------ Feature Generation ------------------

    params = ["bx", "by", "bz", "bt", "vt", "np", "tp"]
    feature_list = []

    N_raw = CME_data.height

    # feature generation loop
    feature_dfs = []
    flag_id = []

    for i in range(N_raw):
        # get the important info for each CME 
        cme_id = CME_data[i, "icmecat_id"]
        sc = CME_data[i, "sc_insitu"].strip().upper()
        icme_start_time = CME_data[i, "icme_start_time"]
        mo_start_time = CME_data[i, "mo_start_time"]
        mo_end_time = CME_data[i, "mo_end_time"]
        insitu_data = insitu_data_map.get(sc)

        # if you are not using sheath -- make sure that icme_start_time is just mo_start_time
        if use_sheath == False:
            icme_start_time = mo_start_time
        
        # get the feature data window for this CME
        feature_data = CME_insitu_extraction(insitu_data, icme_start_time, mo_start_time, mo_end_time, params, hours_features, which="features")

        # if any column in features data has a lot of NaNs/null, skip that event, 
        nan_fraction = feature_data.select([pl.col(col).is_null().sum() / feature_data.height for col in params]).to_dicts()[0]

        if feature_data.height == 0 or any(value > nan_threshold for value in nan_fraction.values()):
            print(f"Skipping event {cme_id} due to excessive NaNs in feature data.")
            # delete this CME from the CME_data as well
            flag_id.append(cme_id)
            continue
        else:
            # interpolate NaNs linearly, and fill remaining NaNs with mean
            feature_data = imputation(feature_data)

        # make features here
        features = make_features(feature_data, cme_id, params, fourier)

        # append it to the list above
        feature_dfs.append(features)

    # remove all the flagged CMEs from CME_data
    if flag_id:
        CME_data = CME_data.filter(~CME_data["icmecat_id"].is_in(flag_id))

    feature_list = pl.concat(feature_dfs, how="vertical")

    # join features to CME data, drop the nans, and reset index
    CME_data = CME_data.join(feature_list, on="icmecat_id", how="left")
    CME_data = CME_data.drop_nans()

    print(CME_data)

    # checkpoint
    # sys.exit()

    # ------------------ Target Generation ------------------

    # in the CME_data, also put the bz/bt targets as the max/min of the entire MO duration
    bz_targets = []
    bt_targets = []
    for i in range(CME_data.height):
        # do the feature target extraction for each CME event
        sc = CME_data[i, "sc_insitu"].strip().upper()
        icme_start_time = CME_data[i, "icme_start_time"] 
        mo_start_time = CME_data[i, "mo_start_time"]
        mo_end_time = CME_data[i, "mo_end_time"]
        insitu_data = insitu_data_map.get(sc)
        cme_id = CME_data[i, "icmecat_id"]

        targets_data = CME_insitu_extraction(insitu_data, icme_start_time, mo_start_time, 
                                            mo_end_time, params, hours_features, which="target")

        bt = np.nanmax(targets_data['bt'].to_numpy())
        bz = np.nanmin(targets_data['bz'].to_numpy())

        bt_targets.append(bt)
        bz_targets.append(bz)

    CME_data = CME_data.with_columns([
        pl.Series("bt_target", bt_targets),
        pl.Series("bz_target", bz_targets)
    ])

    # checkpoint
    # sys.exit()

    # ------------------ Save Feature Space ------------------

    # save the data as a csv file
    if use_sheath == False and hours_features == 4:
        CME_data.write_csv("../../Data/feature_space/CME_features_MO_4.csv")
    elif use_sheath == True and hours_features == 4:
        CME_data.write_csv("../../Data/feature_space/CME_features_sheath_and_MO_4.csv")
    elif use_sheath == True and hours_features == 0:
        CME_data.write_csv("../../Data/feature_space/CME_features_sheath.csv")