# author meta data:
__name__ = "Syed Raza"
__email__ = "sar0033@uah.edu"

# the import statements:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import timedelta
from matplotlib.ticker import ScalarFormatter
import machine_learning_protocols as ml


# the functions
def clean_donki(file_path):
    """
    Clean the DONKI dataset and return a DataFrame with selected columns.
    
    Parameters
    ----------
    file_path : str
        The path to the DONKI.csv file that must contain all the necessary data.
    
    Returns
    -------
    DataFrame
        A DataFrame containing only the specified columns. The 'time_21_5' column is converted to datetime objects.
    
    Raises
    ------
    FileNotFoundError
        If the CSV file is not found at the specified `file_path`.
    ValueError
        If required columns are missing in the CSV file.
    
    Notes
    -----
    The expected format for 'time_21_5' is '%m/%d/%y %H:%M'.
    """

    # Error handling in case the file is not there
    try:
        clean_data = pd.read_excel(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} does not exist.")

    # get only the data listed above:
    clean_data = clean_data[["long", "lat", "speed", "half_width", "time_21_5", "file_name", "cme_transit_obs", "cme_transit_sim", "PE"]]

    # convert the time_21_5 into a datetime object:
    clean_data['time_21_5'] = pd.to_datetime(clean_data['time_21_5'], format='%m/%d/%y %H:%M')

    # return the resulting data set. This is the cleaned donki data set
    return clean_data


def plot_cme_transit(data, title, color, figname):
    """
    Plot a histogram of CME transit times using data from a specified column in a DataFrame.

    Parameters
    ----------
    data : DataFrame
        The column data for making a histogram plot. This column indicates how many hours it took from time 21.5 
        to the CME arrival time.
            Could be 1) observed arrival, 2) ENLIL-simulated arrival, or 3) machine-learning predicted arrival.
    title : str
        The title of the plot.
    color : str
        The color to be used for the histogram bars.
    figname : str
        The file name under which to save the plot. The figure will be saved in the 'figures' directory.

    Returns
    -------
    None
        This function does not return any value. It saves the histogram plot to a file in the current directory.

    Notes
    -----
    The function also calculates the minimum, maximum, mean, and standard deviation values of the specified column and adds a text box with these statistics to the plot. The plot is saved with a resolution of 500 DPI.

    Examples
    --------
    >>> plot_cme_transit(cme_data['cme_transit_obs'], 'Observed CME Arrival Times', 'skyblue', 'observed_cme')
    >>> plot_cme_transit(cme_data['cme_transit_sim'], 'ENLIL CME Arrival Times', 'salmon', 'simulated_cme')
    """

    # making the plot
    plt.figure(figsize=(7, 5))
    ax = plt.gca()  # Get current axis
    sns.histplot(data, kde=True, color=color)
    plt.title(title, fontsize=20)
    plt.xlabel(r'Hours from time at height 21.5 $R_{\odot}$', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    
    # manipulate the size of the ticks:
    ax.tick_params(axis='x', labelsize=16) 
    ax.tick_params(axis='y', labelsize=16) 

    # Calculate statistics
    min_val, max_val, mean_val, std_val = data.min(), data.max(), data.mean(), data.std()

    # Add stats text to the plot
    stats_text = f"Min: {min_val:.2f}\nMax: {max_val:.2f}\nMean: {mean_val:.2f}\nStd: {std_val:.2f}"
    
    # Position the text in the top right corner of the plot
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'),
            fontsize=16)

    # Save the figure
    plt.savefig(f"figures/{figname}", dpi=500, bbox_inches = "tight")

    # return None
    return None


def clean_omni_data(file_path):
    """
    Cleans OMNI solar wind data from a specified file.
    
    Parameters:
    - file_path (str): Path to the data file.
    
    Returns:
    - DataFrame: Cleaned and processed OMNI data, with EPOCH_Time converted to datetime,
      bad values replaced, magnetic field magnitude calculated and converted to Gauss.
    
    Note:
    - Assumes bad values are represented as -1.0000000000000001e+31.
    - EPOCH_Time format should match '%d-%m-%YT%H:%M:%S.%f'.
    """

    # Load the data
    sw_ob = pd.read_csv(file_path, sep = r"\s+")
    
    # join the EPOCH date and time together and restore the index 
    sw_ob["EPOCH_Time"] = sw_ob.index + 'T' + sw_ob["EPOCH_Time"]
    sw_ob['EPOCH_Time'] = pd.to_datetime(sw_ob['EPOCH_Time'], format='%d-%m-%YT%H:%M:%S.%f')
    sw_ob = sw_ob.reset_index(drop=True)

    # Convert EPOCH_Time to datetime, assuming EPOCH_Time is correctly formatted
    # This might need adjustment based on the actual format in your data
    sw_ob['EPOCH_Time'] = pd.to_datetime(sw_ob['EPOCH_Time'], format='%d-%m-%Y %H:%M:%S.%f')

    # Cleaning bad values
    sw_ob.replace(-1.0000000000000001e+31, np.nan, inplace=True)

    # Select only the numeric columns for interpolation
    numeric_cols = sw_ob.select_dtypes(include=[np.number]).columns

    # Interpolate missing values only in numeric columns
    sw_ob[numeric_cols] = sw_ob[numeric_cols].interpolate(method='linear')

    # Calculate magnitude of the magnetic field
    sw_ob["B_mag(nT)"] = np.sqrt(sw_ob["BR_(RTN)(nT)"]**2 + sw_ob["BT_(RTN)(nT)"]**2 + sw_ob["BN_(RTN)(nT)"]**2)
    sw_ob.drop(columns=["BR_(RTN)(nT)", "BT_(RTN)(nT)", "BN_(RTN)(nT)"], inplace=True)

    # Convert magnetic field magnitude from nano Tesla to Gauss
    sw_ob["B_mag(G)"] = sw_ob["B_mag(nT)"]*1e-5
    sw_ob.drop(columns=["B_mag(nT)"], inplace=True)

    # Reset index
    sw_ob.reset_index(drop=True, inplace=True)

    return sw_ob


# Function for processing observed parameters:
def process_observed_parameters(sw_ob, cme_data):
    """
    Processes observed solar wind parameters relative to specific times of interest from donki CME data.
    
    Parameters: 
    sw_ob (pandas.DataFrame): 
        Dataframe containing observed solar wind data with columns: EPOCH_Time, BULK_FLOW_SPEED(km/s), 
        ION_DENSITY(N/cm3), TEMPERATURE(Deg_K), and B_mag(G).
        
    cme_data (pandas.DataFrame): 
        Dataframe containing the DONKI CME catalog data, which includes the "time_21_5" column that 
        specifies the time for which solar wind data should be fetched.
        
    Returns:
    new_df (pandas.DataFrame): 
        A dataframe that contains the solar wind data for the chosen time interval from each time_21_5 value.
    """
    # Boltzmann's constant
    k = 1.380649e-16

    # These are the rows to grab SW data 
    all_rows = []

    # counter 
    counter = 0

    for _, cme_row in cme_data.iterrows():
        # Get the time of interest from the CME data
        time = cme_row["time_21_5"]

        # Determine the dynamic time_range based on the rounded minimum of observed and simulated transit times
        time_range = round(min(cme_row["cme_transit_obs"], cme_row["cme_transit_sim"]))
        
        # Find the closest time in the observed solar wind data
        time_differences = (sw_ob['EPOCH_Time'] - time).abs()
        closest_time_idx = time_differences.idxmin()
        closest_time = sw_ob.loc[closest_time_idx, 'EPOCH_Time']

        # Define the range of times to extract data from
        end_time = closest_time + pd.Timedelta(hours=time_range)

        # Select the rows within the time range
        range_rows = sw_ob[(sw_ob['EPOCH_Time'] >= closest_time) & (sw_ob['EPOCH_Time'] < end_time)].copy()

        # Now, for each row in the range, create a new row with the time and parameters
        for _, row in range_rows.iterrows():
            new_row = {
                "time_21_5": time,  # this time is from cme_data, kept constant for all rows in this range
                "real_time_value": row['EPOCH_Time'],  # add the exact observation time
                "B_ob": row["B_mag(G)"],
                "V_ob": row["BULK_FLOW_SPEED(km/s)"],
                "n_ob": row["ION_DENSITY(N/cm3)"],
                "T_ob": row["TEMPERATURE(Deg_K)"]
            }
            all_rows.append(new_row)

        counter += 1

    # Create a new dataframe from the list of rows
    new_df = pd.DataFrame(all_rows)

    # calcualte and add the (total pressure = n*k*T + B^2/8*pi) exerted by observed solar wind
    new_df["total_pressure_ob"] = 2.0 * new_df["n_ob"]*k*new_df["T_ob"] + (new_df["B_ob"]**2)/(8*np.pi)

    return new_df

def process_simulated_parameters(cme_data, data_directory):
    """
    Processes and retrieves simulated solar wind parameters for given CME events based on the minimum of observed
    and simulated CME transit times, rounded to the nearest whole hour.

    Parameters:
    - cme_data (DataFrame): A DataFrame containing data on CME events, including their times, observed transit times,
      and simulated transit times.
    - data_directory (str): The path to the directory containing the simulation files.

    Returns:
    - DataFrame: A DataFrame containing the simulated solar wind parameters for the given CME events.
    """
    
    # Boltzman constant (in CGS units)
    k = 1.380649e-16
    
    all_rows = []
    
    for _, row in cme_data.iterrows():
        
        # Determine the dynamic time_range based on the rounded minimum of observed and simulated transit times
        time_range = round(min(row['cme_transit_obs'], row['cme_transit_sim']))

        # Retrieve and clean the file name
        file_name = row['file_name'].replace(" ", "")
        folder_name = f"{row['time_21_5'].year}-{row['time_21_5'].month:02d}"
        
        # Construct the full path to the simulation file
        file_name_full = os.path.join(data_directory, folder_name, str(file_name) + "_ENLIL_time_line.dat")

        if os.path.exists(file_name_full):
            # Read in the simulation data
            sim_data = pd.read_csv(file_name_full, sep=r"\s+", header=None)
            sim_data.columns = sim_data.iloc[0]
            sim_data = sim_data.drop([0, 1]).reset_index(drop=True)

            # Ensure numeric columns are read correctly
            for col in ['B_enl', 'V_enl', 'n_enl', 'T_enl']:
                sim_data[col] = pd.to_numeric(sim_data[col], errors='coerce')

            # Create the timestamp for each data point
            sim_data['timestamp'] = pd.to_datetime(sim_data[['year', 'month', 'day', 'hour', 'minute']])
            sim_data = sim_data.drop(["year", "month", "day", "hour", "minute"], axis=1)

            # Find the closest time to 'time_21_5'
            closest_time_idx = (sim_data['timestamp'] - row['time_21_5']).abs().idxmin()
            closest_time = sim_data.loc[closest_time_idx, 'timestamp']

            # Initialize the list to collect data points
            hourly_data_points = []

            for h in range(time_range):  # Use dynamic, rounded time_range
                target_time = closest_time + timedelta(hours=h)
                # Check if this target time is within the simulation data's range
                if target_time <= sim_data['timestamp'].iloc[-1]:
                    # Find the index of the row with the timestamp closest to the target_time
                    closest_hourly_idx = (sim_data['timestamp'] - target_time).abs().idxmin()
                    closest_hourly_time = sim_data.loc[closest_hourly_idx, 'timestamp']

                    new_row = {
                        "time_21_5": row['time_21_5'],
                        "sim_time_value": closest_hourly_time,
                        "B_sim": sim_data.loc[closest_hourly_idx, "B_enl"] * 1e-5,  # Convert from nT to Gauss
                        "V_sim": sim_data.loc[closest_hourly_idx, "V_enl"],
                        "n_sim": sim_data.loc[closest_hourly_idx, "n_enl"],
                        "T_sim": sim_data.loc[closest_hourly_idx, "T_enl"] * 1000  # Convert from kilo Kelvin to Kelvin
                    }
                    hourly_data_points.append(new_row)

            all_rows.extend(hourly_data_points)

    # Create a new DataFrame from the list of rows
    new_df = pd.DataFrame(all_rows)

    # Calculate and add the total pressure exerted by simulated solar wind
    new_df["total_pressure_sim"] = 2.0 * new_df["n_sim"] * k * new_df["T_sim"] + (new_df["B_sim"] ** 2) / (8 * np.pi)

    return new_df

# The function to calcualte residual wind. The difference between simulation and solar wind parameters
def residual_wind(observed_wind, simulated_wind, donki_data):
    """
    Calculate the residual solar wind parameters between observed and simulated data.

    Parameters:
    observed_wind (pandas DataFrame): 
        A DataFrame containing observed solar wind parameters; B_ob, V_ob, n_ob, T_ob, and total_pressure_ob.

    simulated_wind (pandas DataFrame):
        A DataFrame containing simulated solar wind parameters; B_sim, V_sim, n_sim, T_sim, and total_pressure_sim.

    donki (pandas DataFrame):
        A DataFrame containing the prediction error (PE) for each CME event.

    Returns:
    cme_data (pandas DataFrame):
        A DataFrame containing the residual solar wind parameters; B_diff, V_diff, n_diff, T_diff, and total_pressure_diff.
    """
    # Calculate the residuals, and put them in a new DataFrame
    differences = pd.DataFrame({
    "time_21_5": observed_wind["time_21_5"],
    "B_diff": observed_wind["B_ob"] - simulated_wind["B_sim"],
    "V_diff": observed_wind["V_ob"] - simulated_wind["V_sim"],
    "n_diff": observed_wind["n_ob"] - simulated_wind["n_sim"],
    "T_diff": observed_wind["T_ob"] - simulated_wind["T_sim"],
    "P_diff": observed_wind["total_pressure_ob"] - simulated_wind["total_pressure_sim"]
    })
    
    # Now I want to take mean by time_21_5 values and add PE column
    cme_data = differences.groupby("time_21_5", as_index=False).mean()

    # add the columns donki["lat", "long", "speed", "half_width", "cme_transit_obs", "cme_transit_sim"] to the cme_data
    cme_data = cme_data.merge(donki_data[["time_21_5", "lat", "long", "speed", "half_width", "PE", "cme_transit_obs", "cme_transit_sim"]], on="time_21_5")

    # Return the resulting DataFrame
    return cme_data

def data_analysis_plots(cme_data, param_info):
    """
    Plot the correlation between the prediction error and the residual solar wind parameters.

    Parameters:
    cme_data (pandas DataFrame):
        A DataFrame containing the residual solar wind parameters; B_diff, V_diff, n_diff, T_diff, and total_pressure_diff.

    param_info (python dictionary):
        A dictionary containing information about the residual wind parameters; the names, plot title, and figure name.

    Returns:
    None

    But it makes the plots for all the parameters in the list.
    """
    for param, (description, file_name) in param_info.items():
        # Create a new figure and axes
        fig, ax = plt.subplots(figsize=(6, 5))

        # Calculate the Pearson correlation coefficient
        corr_coef = cme_data[param].corr(cme_data["PE"], method='pearson')

        # Create the scatter plot on the axes
        sns.scatterplot(x=cme_data[param], y=cme_data["PE"], ax=ax)
        # Fit and plot a linear regression line on the same axes
        sns.regplot(x=param, y="PE", data=cme_data, scatter=False, color='red', ax=ax, ci=None)

        # Annotate the plot with the correlation coefficient
        ax.annotate(f'Pearson r = {corr_coef:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                    ha='left', va='top', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
        
        ax.tick_params(axis='x', labelsize=18) 
        ax.tick_params(axis='y', labelsize=18) 

        # Set the title and labels
        ax.set_xlabel(description, fontsize=18)
        ax.set_ylabel('Prediction Error (hours)', fontsize=18)

        # Use scientific notation for x-axis
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 4))
        ax.xaxis.set_major_formatter(formatter)
        
        # Draw the canvas to update the formatter
        fig.canvas.draw()
        
        # Extract the offset text
        offset_text = ax.xaxis.get_offset_text().get_text()
        ax.xaxis.get_offset_text().set_visible(False)

        # Add the offset text as an annotation inside the figure
        ax.annotate(offset_text, xy=(1, 0), xycoords='axes fraction', ha='right', va='bottom', fontsize=14)

        # Adjust the layout
        fig.tight_layout()

        # Save the figure
        fig.savefig(file_name, dpi=500)

        # Close the current plot
        plt.close(fig)

    return None

# And with that, I have finished the SW data processing module.
############################################################################################################