# This python code reworks Erika Palmerio's paper titled, "On the 3D Magnetic Structure of 
# Coronal Mass Ejections through the Solar Corona." Only that my analysis is not through the 
# solar corona, but rather through the interplanetary medium. The code will read the hdf5 file
# made through the MS_FLUKSS simulation and recreate the figures 3,4,5,6 from Erika's paper.

# imports
import numpy as np
import h5py
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import sys
from datetime import timedelta, datetime

########## Settings ##########

# pl.Config.set_tbl_cols(-1)      # polars settings to show all the columns
# pl.Config.set_tbl_rows(-1)      # polars settings to show all the rows

# what variation to analyze
variation = "latitudinal"        # options: "radial", "longitudinal", "latitudinal"
radii_lat_lon_analysis = [0.33, 1.0]
offsets_to_plot = [-20, -10, 0, 10, 20]

# Boltzmann constant in J/K, and mu_0 in H/m
k_B = 1.380649e-23 
mu_0 = 1.256637e-6

########## Helper functions ##########

def get_probe_time_series(
    radial_fraction,
    longitude_offset,
    latitude_offset,
    hdf5_path,
):
    """Return all saved time-series data for the nearest stored probe.

    Parameters
    ----------
    radial_fraction : float
        Requested radial probe coordinate.
    longitude_offset : float
        Requested longitude offset in degrees.
    latitude_offset : float
        Requested latitude offset in degrees.
    hdf5_path : str or pathlib.Path, optional
        Packed probe-output HDF5 file. The default matches the filename used
        in ``plot_probe_hdf5.py`` examples.

    Returns
    -------
    polars.DataFrame
        One row per time record. The columns are ``time``, all fields named
        by ``/field_name``, ``distance_au``, ``longitude_deg``,
        and ``latitude_deg``.

    Notes
    -----
    Each requested coordinate is matched independently to its nearest value
    in ``/axes``. The HDF5 probe layout is flattened in C order as
    ``(radius_fraction, latitude_offset_deg, longitude_offset_deg)``.
    """
    with h5py.File(hdf5_path, "r") as h5:
        required = ("time", "values", "position", "found", "field_name")
        missing = [name for name in required if name not in h5]
        if missing:
            raise ValueError("HDF5 file is missing dataset(s): {}".format(
                ", ".join("/" + name for name in missing)
            ))

        time = np.asarray(h5["time"])
        values = np.asarray(h5["values"])
        positions = np.asarray(h5["position"])
        found = np.asarray(h5["found"], dtype=bool)
        field_names = _decode_hdf5_strings(h5["field_name"])

        if time.ndim != 1 or values.ndim != 3 or positions.ndim != 3:
            raise ValueError("time, values, and position have unexpected dimensions")
        if (
            values.shape[:2] != positions.shape[:2]
            or values.shape[:2] != found.shape
            or values.shape[0] != time.size
            or positions.shape[2] != 3
            or values.shape[2] != len(field_names)
        ):
            raise ValueError("time, position, values, found, and field_name disagree")

        axes = _read_probe_axes(h5, values.shape[1])
        requested_coordinates = (
            radial_fraction,
            longitude_offset,
            latitude_offset,
        )
        axis_indices = tuple(
            _nearest_axis_index(axis, requested)
            for axis, requested in zip(
                axes,
                (radial_fraction, latitude_offset, longitude_offset),
            )
        )
        nearest_coordinates = (
            float(axes[0][axis_indices[0]]),
            float(axes[2][axis_indices[2]]),
            float(axes[1][axis_indices[1]]),
        )
        if requested_coordinates != nearest_coordinates:
            print(
                "Probe was not available at ({}, {}, {}); using nearest "
                "available probe at ({}, {}, {}).".format(
                    *requested_coordinates,
                    *nearest_coordinates,
                )
            )
        probe_index = int(np.ravel_multi_index(
            axis_indices,
            tuple(axis.size for axis in axes),
            order="C",
        ))

        data = {"time": time}
        for field_index, field_name in enumerate(field_names):
            if field_name in data:
                raise ValueError("duplicate field name {!r}".format(field_name))
            data[field_name] = values[:, probe_index, field_index]

        for position_index, name in enumerate(
            ("distance_au", "longitude_deg", "latitude_deg")
        ):
            if name in data:
                raise ValueError("field name conflicts with {!r}".format(name))
            data[name] = positions[:, probe_index, position_index]

    return pl.DataFrame(data)

def _decode_hdf5_strings(dataset):
    values = np.asarray(dataset).reshape(-1)
    return [
        value.decode("utf-8").rstrip("\x00")
        if isinstance(value, (bytes, np.bytes_))
        else str(value)
        for value in values
    ]

def _read_probe_axes(h5, probe_count):
    names = (
        "radius_fraction",
        "latitude_offset_deg",
        "longitude_offset_deg",
    )
    if "axes" not in h5 or any(name not in h5["axes"] for name in names):
        raise ValueError("HDF5 file is missing the /axes probe coordinates")
    axes = tuple(np.asarray(h5["axes"][name]) for name in names)
    if any(axis.ndim != 1 or axis.size == 0 for axis in axes):
        raise ValueError("/axes probe coordinates must be non-empty 1-D arrays")
    if np.prod([axis.size for axis in axes], dtype=np.int64) != probe_count:
        raise ValueError("the copied probe axes do not match /values")
    return axes

def _nearest_axis_index(axis, requested):
    return int(np.nanargmin(np.abs(axis.astype(float) - requested)))

# This function is only for CMEs
def radius_fraction_to_au(radius_fraction):
    """Convert the stored radial coordinate to the physical AU coordinate."""
    simulation_inner_radius_au = 0.33
    simulation_outer_radius_au = 1.25
    stored_inner_radius = 0.33
    stored_outer_radius = 1.0
    return simulation_inner_radius_au + (
        (radius_fraction - stored_inner_radius)
        * (simulation_outer_radius_au - simulation_inner_radius_au)
        / (stored_outer_radius - stored_inner_radius)
    )

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

def plasma_beta(probe_data):
    """
    Calculate the plasma beta from the probe data.
    Parameters
    ----------
    probe_data : polars.DataFrame
        The probe data being read from the hdf5 file.

    Returns
    -------
    probe_data : polars.DataFrame
        The probe data with an additional column called plasma_beta
    """

    # Calculate beta = thermal pressure / magnetic pressure.
    # Density is converted from cm^-3 to m^-3 and |B| from nT to T.
    probe_data = probe_data.with_columns(
        plasma_beta = (
            (pl.col("density") / (0.01**3) * k_B * pl.col("T"))
            / ((pl.col("|B|") * 1e-9) ** 2 / (2 * mu_0))
        )
    )

    # return this
    return probe_data

# the main function
if __name__ == "__main__":
    # the file path
    hdf5_filepath = "/Users/syedraza/cmeuncerpy/Data/probe_cone/cme_probes_172.h5"

    ########## THE RADIAL EVOLUTION PLOTS ##########

    if variation == "radial":

        # Fill the subplots using five radial probe locations.
        # with |B|, B_RTN (one subplot for all three components), |V|, density (N), temperature (T) 
        radii_to_plot = np.linspace(0.33, 1.0, 5)
        for r in radii_to_plot:
            radius_au = radius_fraction_to_au(r)
            fig, axs = plt.subplots(6, 1, figsize=(11, 10))
            fig.suptitle("Probe Time Series Data", fontsize=24, fontweight="bold")

            # get the data at the given radial distance
            probe_data = get_probe_time_series(
                radial_fraction=r,
                longitude_offset=0.0,
                latitude_offset=0.0,
                hdf5_path=hdf5_filepath,
            )

            # Change time from decimals to datetime
            probe_data = probe_data.with_columns([
                pl.col("time").map_elements(convert_partial_year).alias("time")
            ])

            # fix for the magnetic field components
            probe_data = probe_data.with_columns([
                (pl.col("Bx") / 10.0).alias("Bx"),
                (pl.col("By") / 10.0).alias("By"),
                (-pl.col("Bz") / 10.0).alias("Bz"),
            ])

            # calculate |B| and |V|
            probe_data = probe_data.with_columns([
                (probe_data["Bx"]**2 + probe_data["By"]**2 + probe_data["Bz"]**2).sqrt().alias("|B|"),
                (probe_data["Vx"]**2 + probe_data["Vy"]**2 + probe_data["Vz"]**2).sqrt().alias("|V|"),
            ])

            # calculate plasma beta
            probe_data = plasma_beta(probe_data)

            # plot |B|, B_xyz, |V|, N, T
            label = f"r = {radius_au:.2f} AU"
            name = f"r_{radius_au:.2f}_AU"
            axs[0].plot(probe_data["time"], probe_data["|B|"])
            axs[1].plot(probe_data["time"], probe_data["Bx"], label=f"BR")
            axs[1].plot(probe_data["time"], probe_data["By"], label=f"BT")
            axs[1].plot(probe_data["time"], probe_data["Bz"], label=f"BN")
            axs[2].plot(probe_data["time"], probe_data["|V|"])
            axs[3].plot(probe_data["time"], probe_data["density"])
            axs[4].plot(probe_data["time"], probe_data["T"])
            axs[5].plot(probe_data["time"], probe_data["plasma_beta"])
            axs[5].set_yscale('log')

            # set the labels and legends
            axs[0].set_ylabel("|B| [nT]", fontsize=16)
            axs[1].set_ylabel("B_RTN [nT]", fontsize=16)
            axs[2].set_ylabel("|V| [km/s]", fontsize=16)
            axs[3].set_ylabel(r"N [$cm^{-3}$]", fontsize=16)
            axs[4].set_ylabel("T [K]", fontsize=16)
            axs[5].set_ylabel(r"Plasma $\beta$", fontsize=16)
            axs[5].set_xlabel("Time", fontsize=16)
            for ax in axs:
                ax.grid()
            axs[1].legend()

            fig.text(
                0.97,
                0.97,
                label,
                ha="right",
                va="top",
                fontsize=16,
            )

            # save the figure
            fig.tight_layout()
            fig.savefig(f"Radial_evolution/Radial_evolution_{name}.png", dpi=300)
            plt.close(fig)

    ########## THE LONGITUDINAL VARIATIONS PLOTS ##########
    if variation == "longitudinal":
        # the radial loop
        for r in radii_lat_lon_analysis:
            radius_au = radius_fraction_to_au(r)
            name = f"r_{radius_au:.2f}_AU"
            # Each radius needs fresh axes so curves cannot carry over.
            fig, axs = plt.subplots(len(offsets_to_plot) + 1, 1, figsize=(10, 14), sharex=True)
            fig2, axs2 = plt.subplots(len(offsets_to_plot) + 1, 1, figsize=(10, 14), sharex=True)
            fig.suptitle(f"|B| (nT), r = {radius_au:.2f} AU", fontsize=24, fontweight="bold")
            fig2.suptitle(f"B_RTN (nT), r = {radius_au:.2f} AU", fontsize=24, fontweight="bold")

            for i in range(len(offsets_to_plot)):
                # get longitude to plot
                lon = offsets_to_plot[i]
                # get the data at the given radial distance
                probe_data = get_probe_time_series(
                    radial_fraction=r,
                    longitude_offset=lon,
                    latitude_offset=0.0,
                    hdf5_path=hdf5_filepath,
                )

                # Change time from decimals to datetime
                probe_data = probe_data.with_columns([
                    pl.col("time").map_elements(convert_partial_year).alias("time")
                ])

                # fix for the magnetic field components
                probe_data = probe_data.with_columns([
                    (pl.col("Bx") / 10.0).alias("Bx"),
                    (pl.col("By") / 10.0).alias("By"),
                    (-pl.col("Bz") / 10.0).alias("Bz"),
                ])

                # calculate |B|
                probe_data = probe_data.with_columns([
                    (probe_data["Bx"]**2 + probe_data["By"]**2 + probe_data["Bz"]**2).sqrt().alias("|B|")
                ])

                # plot |B|, B_RTN at longitudes
                # different subplots correspond to different longitudes
                # Use one magnetic-field panel per longitude.
                axs[i].plot(probe_data["time"], probe_data["|B|"])
                axs2[i].plot(probe_data["time"], probe_data["Bx"], label=f"BR")
                axs2[i].plot(probe_data["time"], probe_data["By"], label=f"BT")
                axs2[i].plot(probe_data["time"], probe_data["Bz"], label=f"BN")
                axs[i].set_ylabel(rf"lon = ${lon}^\circ$", fontsize=16)
                axs2[i].set_ylabel(rf"lon = ${lon}^\circ$", fontsize=16)

                if i == 0:
                    axs2[i].legend()

                # Compare beta at every longitude in the final panel.
                probe_data = plasma_beta(probe_data)
                for beta_ax in (axs[-1], axs2[-1]):
                    beta_ax.plot(
                        probe_data["time"], probe_data["plasma_beta"],
                        label=rf"${lon}^\circ$",
                    )

            for axes in (axs, axs2):
                for ax in axes:
                    ax.grid()
                axes[-1].set_ylabel(r"Plasma $\beta$", fontsize=16)
                axes[-1].set_xlabel("Time", fontsize=16)
                axes[-1].legend(title="Longitude", ncol=len(offsets_to_plot))
                axes[-1].set_yscale('log')
            fig.tight_layout()
            fig2.tight_layout()

            # save the figure outside the loop
            fig.savefig(f"Longitudinal_variation/Longitudinal_variations_B_{name}.png", dpi=300)
            fig2.savefig(f"Longitudinal_variation/Longitudinal_variations_BRTN_{name}.png", dpi=300)
            plt.close(fig)
            plt.close(fig2)

    ########## THE LATITUDINAL VARIATION PLOTS ##########
    if variation == "latitudinal":
        # the radial loop
        for r in radii_lat_lon_analysis:
            radius_au = radius_fraction_to_au(r)
            name = f"r_{radius_au:.2f}_AU"
            fig, axs = plt.subplots(len(offsets_to_plot) + 1, 1, figsize=(10, 14), sharex=True)
            fig2, axs2 = plt.subplots(len(offsets_to_plot) + 1, 1, figsize=(10, 14), sharex=True)
            fig.suptitle(f"|B| (nT), r = {radius_au:.2f} AU", fontsize=24, fontweight="bold")
            fig2.suptitle(f"B_RTN (nT), r = {radius_au:.2f} AU", fontsize=24, fontweight="bold")

            for i in range(len(offsets_to_plot)):
                # get latitude to plot
                lat = offsets_to_plot[i]
                # get the data at the given radial distance
                probe_data = get_probe_time_series(
                    radial_fraction=r,
                    longitude_offset=0.0,
                    latitude_offset=lat,
                    hdf5_path=hdf5_filepath,
                )

                # Change time from decimals to datetime
                probe_data = probe_data.with_columns([
                    pl.col("time").map_elements(convert_partial_year).alias("time")
                ])

                # fix for the magnetic field components
                probe_data = probe_data.with_columns([
                    (pl.col("Bx") / 10.0).alias("Bx"),
                    (pl.col("By") / 10.0).alias("By"),
                    (-pl.col("Bz") / 10.0).alias("Bz"),
                ])

                # calculate |B|
                probe_data = probe_data.with_columns([
                    (probe_data["Bx"]**2 + probe_data["By"]**2 + probe_data["Bz"]**2).sqrt().alias("|B|")
                ])

                # plot |B|, B_RTN at latitudes
                # different subplots correspond to different latitudes
                # Use one magnetic-field panel per latitude.
                axs[i].plot(probe_data["time"], probe_data["|B|"])
                axs2[i].plot(probe_data["time"], probe_data["Bx"], label=f"BR")
                axs2[i].plot(probe_data["time"], probe_data["By"], label=f"BT")
                axs2[i].plot(probe_data["time"], probe_data["Bz"], label=f"BN")
                axs[i].set_ylabel(rf"lat = ${lat}^\circ$", fontsize=16)
                axs2[i].set_ylabel(rf"lat = ${lat}^\circ$", fontsize=16)

                if i == 0:
                    axs2[i].legend()

                # Compare beta at every latitude in the final panel.
                probe_data = plasma_beta(probe_data)
                for beta_ax in (axs[-1], axs2[-1]):
                    beta_ax.plot(
                        probe_data["time"], probe_data["plasma_beta"],
                        label=rf"${lat}^\circ$",
                    )

            for axes in (axs, axs2):
                for ax in axes:
                    ax.grid()
                axes[-1].set_ylabel(r"Plasma $\beta$", fontsize=16)
                axes[-1].set_xlabel("Time", fontsize=16)
                axes[-1].legend(title="Latitude", ncol=len(offsets_to_plot))
                axes[-1].set_yscale('log')
            fig.tight_layout()
            fig2.tight_layout()

            # save the figure outside the loop
            fig.savefig(f"Latitudinal_variation/Latitudinal_variations_B_{name}.png", dpi=300)
            fig2.savefig(f"Latitudinal_variation/Latitudinal_variations_BRTN_{name}.png", dpi=300)
            plt.close(fig)
            plt.close(fig2)
