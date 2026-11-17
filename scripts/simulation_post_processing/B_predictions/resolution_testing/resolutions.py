# THis Python code will be used to see the difference in magnetic field profiles 
# for three different resolution simulations run in HelioCubed. 

# imports
import re
from io import StringIO
from itertools import combinations
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
from pathlib import Path

########## Settings ##########

# pl.Config.set_tbl_cols(-1)      # polars settings to show all the columns
# pl.Config.set_tbl_rows(-1)      # polars settings to show all the rows

# to choose what plots to make
time_series = True
correlation = False

start_time = datetime(2012, 7, 12, 16, 52, 00)
end_time = datetime(2012, 7, 17, 16, 43, 00) + timedelta(hours=1)

########## Helpers ##########
# import many of the helper functions from magnetic_structure.py
helpers_dir = Path(__file__).resolve().parents[1] / "rlatlon_variations"
sys.path.insert(0, str(helpers_dir))

from magnetic_structure import (  # pyright: ignore[reportMissingImports]
    convert_partial_year,
)

def observed_data():
    """ Returns an observed dataframe.
    """
    # reading in the observed data
    path = (
        "/Users/syedraza/cmeuncerpy/Data/ambient_SW_quantification/"
        "OMNI_COHO1HR_MERGED_MAG_PLASMA_137596.txt"
    )

    with open(path) as f:
        columns = next(f).split()
        text = "\n".join(
            re.sub(r"\s{2,}", "\t", line.strip())
            for line in f if line.strip()
        )

    obs_data = pl.read_csv(
        StringIO(text),
        separator="\t",
        has_header=False,
        new_columns=columns,
    ).with_columns(
        pl.col("EPOCH_Time").str.to_datetime("%d-%m-%Y %H:%M:%S%.f"),
        pl.exclude("EPOCH_Time").replace(-1e31, None),
    )

    # make a mask and get the the data between start_time and end_time
    my_mask = pl.col("EPOCH_Time").is_between(start_time, end_time)
    obs_data = obs_data.filter(my_mask)

    # make a the total magnetic field column
    obs_data = obs_data.with_columns([
        (obs_data["BR_(RTN)(nT)"]**2 + obs_data["BT_(RTN)(nT)"]**2 + obs_data["BN_(RTN)(nT)"]**2).sqrt().alias("|B|"),
    ])

    # rename the columns
    obs_data = obs_data.rename({"BR_(RTN)(nT)": "BR", "BT_(RTN)(nT)": "BT", "BN_(RTN)(nT)": "BN", 
                    "BULK_FLOW_SPEED(km/s)": "|V|", "ION_DENSITY(N/cm3)": "number_density"})
    
    return obs_data

def HelioCubed_Earth_data(given_path):
    """ get the simulated data based on the path given
    """
    col_names = ["time", "dist(au)", "lat", "lon", "number_density", "VR", "VT", "VN", "P", "BR", "BT", "BN"]
    
    data = (
        pl.read_lines(given_path).select(
            pl.first()
            .str.strip_chars()
            .str.extract_all(r"\S+")              # Extracts every non-whitespace chunk
            .list.to_struct(fields=col_names)
            .alias("parsed")
        )
        .unnest("parsed")
        .with_columns(pl.all().cast(pl.Float64))  # Use Float64 so 'year' keeps full decimal precision
    )

    # Calculate v and B magnitudes 
    probe_data = get_v_and_B(data)
    # convert from microgauss to nT
    probe_data = probe_data.with_columns(
        pl.col("|B|", "BR", "BT", "BN") * 0.1
    )

    return probe_data

def MS_FLUKSS_Earth_data(given_path):
    col_names = ["time", "dist(au)", "lon", "lat", "number_density", "VR", "VT", "VN", "P", "BR", "BT", "BN", "T"]
    data = (
            pl.read_lines(given_path).slice(3).select(
                pl.first()
                .str.strip_chars()
                .str.extract_all(r"\S+")              # Extracts every non-whitespace chunk
                .list.to_struct(fields=col_names)
                .alias("parsed")
            )
            .unnest("parsed")
            .with_columns(pl.all().cast(pl.Float64))  # Use Float64 so 'year' keeps full decimal precision
        )

    # Calculate v and B magnitudes 
    probe_data = get_v_and_B(data)
    # convert from microgauss to nT
    probe_data = probe_data.with_columns(
        pl.col("|B|", "BR", "BT", "BN") * 0.1
    )

    return probe_data

def get_v_and_B(probe_data):
    """ Calculate the magnitudes speed and magnetic field from their components
    """

    # change time from decimals to datetime and calculate |B| and |v|
    probe_data = probe_data.with_columns([
        pl.col("time").map_elements(convert_partial_year, return_dtype=datetime).alias("time")
    ])
    
    # calculate |B| and |V|
    probe_data = probe_data.with_columns([
        (probe_data["BR"]**2 + probe_data["BT"]**2 + probe_data["BN"]**2).sqrt().alias("|B|"),
        (probe_data["VR"]**2 + probe_data["VT"]**2 + probe_data["VN"]**2).sqrt().alias("|V|"),
    ])

    return probe_data

# the main function:
if __name__ == "__main__":
    # the observed data
    obs_data = observed_data()

    # HelioCubed runs are used for both plots; MS-FLUKSS is added only to the time series.
    Earth_90_30 = "/Users/syedraza/cmeuncerpy/Data/probe_testing/Earth/Earth_90_30_HelioCubed.dat"
    Earth_120_60 = "/Users/syedraza/cmeuncerpy/Data/probe_testing/Earth/Earth_120_60_HelioCubed.dat"
    Earth_180_90 = "/Users/syedraza/cmeuncerpy/Data/probe_testing/Earth/Earth_180_90_HelioCubed.dat"
    Earth_150_256_128 = "/Users/syedraza/cmeuncerpy/Data/probe_testing/Earth/Earth_150_256_128_MSFLUKSS.dat"

    file_paths = [Earth_90_30, Earth_120_60, Earth_180_90]
    resolutions = [r"(90, $3^{\circ}$)", r"(120, $1.5^{\circ}$)", r"(180, $1^{\circ}$)"]
    colors = ["blue", "green", "red"]

    if time_series == True:
        # for loop for each radius that needs to be plotted
        observation_curve = True
        # make the figure for each radial direction
        fig, axs = plt.subplots(6, 1, figsize=(14, 12))
        fig.suptitle("Probe Time Series Data", fontsize=24, fontweight="bold")

        time_series_models = [
            (path, label, color, HelioCubed_Earth_data)
            for path, label, color in zip(file_paths, resolutions, colors)
        ]
        time_series_models.append(
            (Earth_150_256_128, "MS-FLUKSS", "magenta", MS_FLUKSS_Earth_data)
        )

        for path, label, color, reader in time_series_models:
            probe_data = reader(path)
            probe_data = probe_data.filter(pl.col("time").is_between(start_time, end_time))

            # make the plots
            radial_label = f"Earth"
            name = f"Earth_time_series"
            axs[0].plot(probe_data["time"], probe_data["|B|"], label=label, color=color)
            for ax, field in zip(axs[1:], ("BR", "BT", "BN", "|V|", "number_density")):
                ax.plot(probe_data["time"], probe_data[field], label=label, color=color)

            # set the labels for all the plots
            axs[0].set_ylabel("|B| [nT]", fontsize=16)
            axs[1].set_ylabel("BR [nT]", fontsize=16)
            axs[2].set_ylabel("BT [nT]", fontsize=16)
            axs[3].set_ylabel("BN [nT]", fontsize=16)
            axs[4].set_ylabel("|V| [km/s]", fontsize=16)
            axs[5].set_ylabel(r"N [$cm^{-3}$]", fontsize=16)
            axs[5].set_xlabel("Time", fontsize=16)

            fig.text(
                0.97,
                0.97,
                radial_label,
                ha="right",
                va="top",
                fontsize=16,
            )

        # Add observations once per panel, only at the physical 1-AU radius.
        if observation_curve:
            for ax, field in zip(axs, ("|B|", "BR", "BT", "BN", "|V|", "number_density")):
                ax.plot(
                    obs_data["EPOCH_Time"], obs_data[field],
                    label="observation", color="black",
                )

        # One shared legend in the right margin keeps every panel unobstructed.
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.85, 0.85), fontsize=16)
        fig.tight_layout(rect=(0, 0, 0.84, 1))
        fig.savefig(f"figures/Earth_comparison.png", bbox_inches="tight")

    if correlation == True:
        # all the variables for which I have to do this analysis
        correlation_variables = ["|B|", "BR", "BT", "BN", "|V|", "number_density"]

        # Evaluate every simulation at the observation timestamps without shifting events.
        observation_times = obs_data["EPOCH_Time"].to_numpy().astype("datetime64[us]").astype(np.int64)
        simulation_data = []
        for path in file_paths:
            data = HelioCubed_Earth_data(path).sort("time")
            times = data["time"].to_numpy().astype("datetime64[us]").astype(np.int64)
            if np.any(np.diff(times) <= 0):
                raise ValueError(f"Simulation timestamps must be unique and increasing: {path}")
            simulation_data.append({
                var: np.interp(
                    observation_times, times, data[var].to_numpy(),
                    left=np.nan, right=np.nan,
                )
                for var in correlation_variables
            })

        model_pairs = list(combinations(range(len(simulation_data)), 2))
        nrows = max(len(simulation_data), len(model_pairs))

        parameter_labels = {
            "|B|": "|B| [nT]",
            "BR": "BR [nT]",
            "BT": "BT [nT]",
            "BN": "BN [nT]",
            "|V|": "|V| [km/s]",
            "number_density": r"N [$cm^{-3}$]",
        }

        for var in correlation_variables:
            parameter_label = parameter_labels[var]
            fig, axs = plt.subplots(nrows, 2, figsize=(12, 3 * nrows), squeeze=False)
            fig.suptitle(f"Correlation of {parameter_label}", fontsize=24, fontweight="bold")

            observed = obs_data[var].to_numpy()
            for row, (data, resolution, color) in enumerate(zip(simulation_data, resolutions, colors)):
                valid = np.isfinite(data[var]) & np.isfinite(observed)
                axs[row][0].scatter(data[var][valid], observed[valid], s=5, color=color)
                axs[row][0].set_xlabel(f"Simulated {parameter_label} — {resolution}")
                axs[row][0].set_ylabel(f"Observed {parameter_label}")

            for row, (first, second) in enumerate(model_pairs):
                x, y = simulation_data[first][var], simulation_data[second][var]
                valid = np.isfinite(x) & np.isfinite(y)
                axs[row][1].scatter(x[valid], y[valid], s=5)
                axs[row][1].set_xlabel(f"Simulated {parameter_label} — {resolutions[first]}")
                axs[row][1].set_ylabel(f"Simulated {parameter_label} — {resolutions[second]}")

            for row in range(nrows):
                for column, count in enumerate((len(simulation_data), len(model_pairs))):
                    ax = axs[row][column]
                    if row >= count:
                        ax.set_visible(False)
                        continue
                    lower = min(ax.get_xlim()[0], ax.get_ylim()[0])
                    upper = max(ax.get_xlim()[1], ax.get_ylim()[1])
                    ax.plot([lower, upper], [lower, upper], "--", linewidth=1, label="x = y", color="red")
                    ax.set_xlim(lower, upper)
                    ax.set_ylim(lower, upper)
                    ax.legend()
    
            fig.tight_layout()
            fig.savefig(f"figures/correlation_{var}.png")
            plt.close(fig)

