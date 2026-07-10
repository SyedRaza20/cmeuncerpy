# This code is for plotting the j-maps of the CME events inspired by the work of Dr. Talwinder Singh

# The purpose of this code is to generate the J-maps by themselves for the CMEs provided.
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import timedelta, datetime, date
import matplotlib.dates as mdates
from matplotlib.dates import num2date
import matplotlib.cm as cm
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from PIL import Image
import io
import os

# for displaying all rows:
pd.set_option('display.max_rows', None)

# Starting out with the functions I need
def is_leap(Year):
    # To see if the given year is leap or not
    if((Year % 400 == 0) or  
     (Year % 100 != 0) and  
     (Year % 4 == 0)): 
        return True
    else:
        return False
    
def convert_partial_year(number):
    # This is the function to convert from decimal year to datetime or something
    year = int(number)
    d = timedelta(days=(number - year)*(365 + is_leap(year)))
    day_one = datetime(year,1,1)
    date = d + day_one
    return date

def find_peak_and_front_location(heatmap_data_run_diff, time_ind,lower_bound, upper_bound):
    # Filter rows with index between lower_bound and upper_bound
    filtered_data = heatmap_data_run_diff[(heatmap_data_run_diff.index >= lower_bound) & (heatmap_data_run_diff.index <= upper_bound)].iloc[:, time_ind]
    # Find the Elong where the first column peaks within the filtered range
    peak_elong = filtered_data.idxmax()
    peak_value = filtered_data.max()
    
    norm_filtered_data = filtered_data/peak_value
    norm_filtered_data = norm_filtered_data.loc[peak_elong:peak_elong+20]
    front_loc = norm_filtered_data[norm_filtered_data < 0.35].first_valid_index()
    if front_loc is not None:
        return front_loc, peak_elong
    else:
        return np.nan, peak_elong
    
def manual_pick_points(ax):
    """
    Interactive manual picking:
      Left-click: add point
      u: undo last point
      r: remove all points
      Enter: finish and return (apply polyfit if enabled)
      Esc: cancel (return empty)
    Returns DataFrame with Time, EA_smooth (fitted or raw), EA_raw (always raw clicks)
    """
    fig = ax.figure
    points = []
    line, = ax.plot([], [], color='tab:cyan', marker='o', ms=4, lw=1, label='Manual picks')

    status_txt = ax.text(0.02, 0.02,
                         "Click to add points | u: undo | r: reset | Enter: finish | Esc: cancel",
                         transform=ax.transAxes, fontsize=8, color='w',
                         bbox=dict(facecolor='k', alpha=0.4, pad=3, edgecolor='none'))

    finished = {'done': False, 'cancel': False}

    def redraw():
        if points:
            xs, ys = zip(*points)
        else:
            xs, ys = [], []
        line.set_data(xs, ys)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            points.append((event.xdata, event.ydata))
            redraw()

    def on_key(event):
        key = event.key
        if key in ('enter', 'return'):
            finished['done'] = True
        elif key == 'u':
            if points:
                points.pop()
                redraw()
        elif key == 'r':
            points.clear()
            redraw()
        elif key == 'escape':
            finished['cancel'] = True
            finished['done'] = True

    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

    print("Manual mode: Left-click to add points; u=undo; r=reset; Enter=finish; Esc=cancel")

    while not finished['done']:
        plt.pause(0.05)

    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)
    status_txt.set_visible(False)
    fig.canvas.draw_idle()

    if finished['cancel'] or not points:
        print("Manual picking canceled or no points selected.")
        return pd.DataFrame(columns=["Time", "EA_smooth", "EA_raw"])

    times = [p[0] for p in points]
    elong = [p[1] for p in points]
    df = pd.DataFrame({"Time": times, "EA_raw": elong})
    df = df.sort_values("Time").reset_index(drop=True)

    # Start EA_smooth as raw
    df["EA_smooth"] = df["EA_raw"]

    # Optional polyfit smoothing
    if manual_poly_order is not None and len(df) >= min_manual_points:
        try:
            coefs = np.polyfit(df.index, df["EA_raw"], manual_poly_order)
            df["EA_smooth"] = np.polyval(coefs, df.index)
        except Exception as e:
            print(f"Polyfit failed ({e}); keeping raw points as EA_smooth.")

    return df

# the address that I need:
address = 'CMEs/'

# some stuff:
make_gif = False
skip_columns = 0
skip_range = [0, 0]
init_low = 15
init_upp = 22

# A flag for manual points:
# Manual picking mode
manual_mode = True          # set False to use automatic method
manual_poly_order = 5       # polynomial fit order for smoothing manual picks
min_manual_points = 3       # minimum points before fitting

cme_date = ["15_2013-06-30"]

for CME in cme_date:
    for member_num in range(22):
        member_num = str(member_num)
        # j-map could be made with the craft 'A' or 'B'
        for craft in ['B']:
            # big try statement:
            try: 
                # read the ensemble member j-map file and the background j-map file (jmap0.dat)
                data0_path = os.path.join(address,CME,'jmapdats',f'Stereo{craft}-Jmap_0.dat')
                data_path = os.path.join(address,CME,'jmapdats',f'Stereo{craft}-Jmap_{member_num}.dat')

                # the background j-map file:
                data0 = pd.read_csv(data0_path,
                    header=None, sep=r'\s+', names=["Time", "Iteration", "Elong", "density"])
                
                # The ensemble member jmap file:
                data = pd.read_csv(data_path,
                    header=None, sep=r'\s+',  names=["Time", "Iteration", "Elong", "density"])

                # proceed from here only if there are no nan values in the data:
                if data.isnull().values.any():
                   print("not preceding due to NaN values in the Jmap.dat files")
                   continue
                else:
                    print("proceeding")
                
                # subtract the background Jmap from the ensemble Jmap:
                merged_df = pd.merge(data, data0, left_on=["Iteration", "Elong"], right_on=["Iteration", "Elong"])
                merged_df['density'] = merged_df['density_x'] - merged_df['density_y']
                merged_df = merged_df.drop(['density_x', 'density_y','Time_y'], axis=1)
                merged_df = merged_df.rename(columns={"Time_x": "Time"})
                heatmap_data = pd.pivot_table(merged_df, values='density', index=['Elong'], columns='Time')

                cols = heatmap_data.columns
                col_count = len(heatmap_data.columns)
                heatmap_data_run_diff = pd.DataFrame()
                heatmap_data_run_diff = heatmap_data.diff(axis=1).drop(cols[0], axis=1)

                print(heatmap_data_run_diff.head)

                # Enter the file of the desired ensemble member:
                with open(address+CME+'/inputs/Inputs_'+str(member_num)+'.inputs', 'r') as file:
                    CMEEnterTime = None

                    # find the line that tells us the CME enter time:
                    for line in file:
                        if line.startswith('mhdam.CMEEnterTime'):
                            # split this line by equal sign and grab the last number as the CME enter time 
                            # (a good point to test if the code is working) --> (Finished testing, its working well)
                            CMEEnterTime = float(line.split('=')[-1].strip())
                            # FOR TESTING: print('The enter time for the CME member ' + str(member_num) + ' is: ', CMEEnterTime)
                            break
                
                # Select a subset of columns: [To add more time]
                subset_columns = [col for col in heatmap_data_run_diff.columns if CMEEnterTime+0.1/365 <= col <= CMEEnterTime+2.5/365]
                heatmap_data_run_diff = heatmap_data_run_diff.loc[:, subset_columns]
                
                # Plot each column and save the plot as a gif:
                if make_gif:
                    images = []
                    for column in heatmap_data_run_diff.columns:
                        fig, ax = plt.subplots()
                        ax.plot(heatmap_data_run_diff[column])
                        ax.set_title(f'Column {column}')
                        plt.tight_layout()

                        # save the plot as an image in memory
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
                        plt.close(fig)
                        buf.seek(0)

                        # Load the image into memory and add it to the list of images
                        img = Image.open(buf)
                        images.append(img)

                        # Save the images as animated gif:
                        gif_filename = 'dataframe_plots.gif'
                        images[0].save(gif_filename, save_all=True, append_images=images[1:], duration=500, loop=0)

                # if the values are less than 0.02, set them to zero
                heatmap_data_run_diff[heatmap_data_run_diff < 0.0] = 0

                y = heatmap_data_run_diff.index.values
                x = heatmap_data_run_diff.columns.values 

                for i in x:
                    heatmap_data_run_diff = heatmap_data_run_diff.rename(columns={i: mdates.date2num(convert_partial_year(i))})

                x = heatmap_data_run_diff.columns.values
                print(f"X-axis limits for ensemble {member_num}: Start={num2date(x.min())}, End={num2date(x.max())}") # uncomment to help out with the time ranges of the manual jmap
                print(f"Y-axis limits for ensemble {member_num}: Start={y.min()}, End={y.max()}")

                # convert the DataFrame to a numpy array:
                data_array = heatmap_data_run_diff.to_numpy()

                # for smoothing, apply a median filter:
                filter_size = 3 # This number should be an odd number (e.g 3, 5, 7, ... etc)
                blurred_array = median_filter(data_array, size = filter_size)

                # convert this numpy array back to a data frame:
                heatmap_data_run_diff = pd.DataFrame(blurred_array, columns=heatmap_data_run_diff.columns, index=heatmap_data_run_diff.index)

                # ===== Manual picking branch (runs before automatic) =====
                if manual_mode:
                    fig, ax = plt.subplots(figsize=(6,4))
                    plt.pcolormesh(x, y, heatmap_data_run_diff, cmap=cm.gray, shading="auto")
                    plt.colorbar()
                    ax.xaxis_date()
                    days = mdates.DayLocator(interval=1)
                    d_fmt = mdates.DateFormatter('%d%b%y %H:%M')
                    ax.xaxis.set_major_locator(days)
                    ax.xaxis.set_major_formatter(d_fmt)
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    plt.xlabel("Date (UT)", size=12)
                    plt.ylabel("Elongation ($^\\circ$)", size=12)
                    plt.title(f"Manual pick CME {CME} Stereo{craft} member {member_num}")
                    plt.tight_layout()
                    plt.ion()
                    plt.show()

                    manual_df = manual_pick_points(ax)  # returns Time (matplotlib date num) and EA_smooth
                    if manual_df.empty:
                        plt.close(fig)
                        print("No manual points selected; skipping member.")
                        continue

                    # Overlay picked curve
                    ax2 = ax.twinx()
                    ax2.set_ylim(ax.get_ylim())
                    ax2.set_xlim(ax.get_xlim())
                    ax2.plot(manual_df["Time"], manual_df["EA_smooth"], color='tab:cyan', marker='o', ms=3, lw=1)
                    ax2.get_yaxis().set_visible(False)

                    # Save figure
                    out_fig = os.path.join(address, CME, "figs", f"Stereo{craft}_{member_num}.png")
                    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
                    print("Saving manual plot:", out_fig)
                    plt.savefig(out_fig, dpi=300)
                    plt.close(fig)

                    # Save CSV
                    save_df = manual_df.copy()
                    save_df["Time"] = save_df["Time"].apply(lambda v: num2date(v).replace(tzinfo=None, microsecond=0))
                    save_df["EA_smooth"] = save_df["EA_smooth"].round(2)
                    out_csv = os.path.join(address, CME, "csv", f"Stereo{craft}_{member_num}.csv")
                    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
                    save_df.to_csv(out_csv, index=False)
                    print("Saved manual track CSV:", out_csv)
                    continue  # Skip automatic processing for this member

                # get stuff for calling the function of find peak and front location:
                lower_bound = init_low
                upper_bound = init_upp
                time_ind = 0

                # loop over all the columns and find the front location:
                dfshape = heatmap_data_run_diff.shape
                df = pd.DataFrame()
                j = 0

                for i in range(skip_columns, len(heatmap_data_run_diff.columns)):
                    # ia manual is true, I need to pick the points
                    front_loc, peak_elong = find_peak_and_front_location(heatmap_data_run_diff, i, lower_bound, upper_bound)
                    # maybe testing statement here. Understand what they mean --> front_loc and peak_elong
                    if i > skip_range[0] and i < skip_range[1]:
                        lower_bound = init_low
                        upper_bound = init_upp
                        df.loc[j, "Time"] = heatmap_data_run_diff.columns[i]
                        df.loc[j, "EA"] = np.nan
                        j+=1
                        continue

                    lower_bound = peak_elong
                    upper_bound = peak_elong + 3
                    df.loc[j,"Time"] = heatmap_data_run_diff.columns[i]
                    df.loc[j,"EA"] = front_loc
                    j+=1

                # There was a print statement here --> check in later:
                df.dropna(inplace=True)

                # if the column "EA" exists then smooth it with a 2nd order polynomial
                if "EA" in df.columns:
                    df["EA_smooth"] = df["EA"]
                    # What is this: df['EA_smooth'] = savgol_filter(df['EA'], 5, 2)

                    x_fit = df.index
                    y_fit = df["EA_smooth"]

                    # Fit a 4nd order polynomial to the data
                    coefficients = np.polyfit(x_fit, y_fit, 4)

                    # Evaluate the polynomial at the original x values
                    df['EA_smooth'] = np.polyval(coefficients, x_fit)

                fig, ax = plt.subplots(figsize=(6,4))
                plt.pcolormesh(x, y, heatmap_data_run_diff, cmap=cm.gray, shading = "auto")
                #show colorbar
                plt.colorbar()

                ax.xaxis_date()
                days = mdates.DayLocator(interval = 1)
                d_fmt = mdates.DateFormatter('%d%b%y %H:%M')
                ax.xaxis.set_major_locator(days)
                ax.xaxis.set_major_formatter(d_fmt)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

                plt.xlabel("Date (UT)", size=12)
                plt.ylabel("Elongation ($^\\circ$)", size=12)
                # plt.title("J-map", size=14)
                plt.tight_layout()
                # plt.gca().invert_yaxis()

                ax2 = ax.twinx()
                ax2.set_ylim(ax.get_ylim())
                ax2.set_xlim(ax.get_xlim())
                # print(ax.get_xlim())
                # ax2.plot(df["Time"], df["EA_smooth"],**{'color': 'lightsteelblue', 'marker': 'o'})
                if 'EA_smooth' in df.columns:
                    ax2.plot(df["Time"], df["EA_smooth"])
                # ax2.plot(df["Time"], df["EA"])
                ax2.get_yaxis().set_visible(False)

    
                print("Saving the plot")
                plt.savefig(address+"/"+CME+"/figs"+'/Stereo'+craft+"_"+str(member_num)+'.png',dpi=300)
                # plt.savefig(address+'Results/'+CME_num+'/Stereo'+Craft+'_'+CME_num+'_'+Member+'.eps', format='eps')
                # plt.show()
                plt.close()
                
                # make the .csv files (these files will be used by the code plot_jmaps.py to plot the time-elongation plots)
                df["Time"] = df["Time"].apply(lambda x: num2date(x).replace(tzinfo=None))
                df = df[['Time', 'EA_smooth']]
                # keep 2 decimal places in EA_smooth column
                df["EA_smooth"] = df['EA_smooth'].apply(lambda x: round(x, 2))
                # round the datetime to the nearest second
                df['Time'] = df['Time'].apply(lambda x: x.replace(microsecond=0))
                df.to_csv(address+'/'+CME+"/csv"+"/Stereo"+craft+'_'+member_num+'.csv', index=False)
            except:
                continue