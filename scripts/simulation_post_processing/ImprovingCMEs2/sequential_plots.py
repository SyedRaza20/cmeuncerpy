import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

cmes = ['01_2010-04-03', '02_2010-05-23', '03_2010-08-01', '04_2011-09-06',
        '05_2011-09-13', '06_2011-10-22', '07_2012-01-19', '08_2012-03-07',
        '09_2012-06-14', '10_2012-07-03', '11_2012-07-12', '12_2012-09-27',
        '13_2012-10-05']

RESULTS_DIR = Path("sequential_results")
OUT_DIR = Path("sequential_results/figures")
OUT_DIR.mkdir(exist_ok=True)

def load_cme(cme_id: str) -> pl.DataFrame:
    """Load a single CME's sequential results and parse the Time column.
    The time column should represent the difference in hours between the 
    time of training and the arrival time of the given CME. 
        Although, I still want them in the datetime format. 
        just that the datetime number is the amount of time
        before the CME actually arrives.
    """
    df = pl.read_csv(RESULTS_DIR / f"{cme_id}.csv")
    if df["Time"].dtype == pl.Utf8:
        df = df.with_columns(pl.col("Time").str.to_datetime(strict=False))

    # make sure to read in data every 10 minutes
    df = df.group_by_dynamic(
        index_column = "Time",
        every = '5m',
        closed = 'left',
    ).agg(
        pl.col('ML_error').mean()
    )

    # link to the arrival time of the CME
    arrival_link = f"/Users/syedraza/cmeuncerpy/Data/ImprovingCMEs2/erupt_and_arrival_times/{cme_id}/Arrival_time.txt"
    time = pl.read_csv(arrival_link, has_header = False).item()
    time_dt = datetime.strptime(time, "%Y/%m/%dT%H:%M")

    # change the time column as the difference from TOA
    df = df.with_columns(
        Time_duration = time_dt - pl.col("Time")
    )

    return df

# ---------------------------------------------------------------
# 1. Individual, styled plot per CME (saved + optionally shown)
# ---------------------------------------------------------------
def plot_single_cme(cme_id: str, df: pl.DataFrame, save: bool = True, show: bool = False):
    time = df["Time"].to_list()
    err = df["ML_error"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(time, err, color="#1f77b4", lw=1.6, alpha=0.9)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)

    ax.fill_between(time, err, 0, where=(err >= 0), color="#2ca02c", alpha=0.15)
    ax.fill_between(time, err, 0, where=(err < 0), color="#d62728", alpha=0.15)

    ax.set_title(f"ML Error vs. Time — CME {cme_id}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("ML Error (hours)")

    if isinstance(time[0], (np.datetime64,)) or hasattr(time[0], "strftime"):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    if save:
        fig.savefig(OUT_DIR / f"{cme_id}_ml_error.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------
# 2. Grid of small multiples — all 13 CMEs on one figure
# ---------------------------------------------------------------
def plot_all_cmes_grid(cmes, ncols=3):
    nrows = int(np.ceil(len(cmes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), sharey=False)
    axes = axes.flatten()

    n_remainder = len(cmes) % ncols
    last_row_is_single = (n_remainder == 1)

    for i, cme_id in enumerate(cmes):
        # Re-map the lone final-row plot to the middle column.
        if last_row_is_single and i == len(cmes) - 1:
            row = i // ncols
            center_col = ncols // 2
            ax = axes[row * ncols + center_col]
        else:
            ax = axes[i]

        try:
            df = load_cme(cme_id)
        except FileNotFoundError:
            ax.set_visible(False)
            continue
        time = df["Time"].to_list()
        err = df["ML_error"].to_numpy()

        ax.plot(time, err, lw=1.3, color="#1f77b4")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.fill_between(time, err, 0, where=(err >= 0), color="#2ca02c", alpha=0.15)
        ax.fill_between(time, err, 0, where=(err < 0), color="#d62728", alpha=0.15)
        ax.set_title(cme_id, fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=8)

    # Hide unused axes. When the last row has only one plot, hide every
    # other axis in that row (not just the ones past len(cmes)).
    if last_row_is_single:
        last_row_start = (nrows - 1) * ncols
        center_idx = last_row_start + (ncols // 2)
        for j in range(last_row_start, last_row_start + ncols):
            if j != center_idx:
                fig.delaxes(axes[j])
    else:
        for j in range(len(cmes), len(axes)):
            fig.delaxes(axes[j])

    fig.suptitle("ML Error Evolution Across All CME Events", fontsize=30, fontweight="bold", y=1.02)
    fig.text(0.5, -0.01, "Time", ha="center", fontsize=16)
    fig.text(-0.005, 0.5, "ML Error (hours)", va="center", rotation="vertical", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "all_cmes_grid.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------
# 3. Overlay plot — normalized "time since start" so all events
#    line up on one axis (great for spotting systematic trends)
# ---------------------------------------------------------------
def plot_overlay_normalized(cmes, leader_length=0.03, leader_angle_deg=None):
    """
    Overlay all CME ML-error trajectories on a shared lead-time x-axis.

    Parameters
    ----------
    leader_length : float
        Length of the leader line from curve endpoint to label, as a
        fraction of max_lead. Try 0.03-0.10.
    leader_angle_deg : float or sequence of float, optional
        Angle(s) of the leader line, in degrees, measured from the
        horizontal (0 = horizontal, pointing left/away from arrival;
        90 = straight up; -90 = straight down). Positive = label angled
        upward, negative = downward.
        - If a single float, that angle is used for every curve, and its
          sign is still alternated between adjacent curves (sorted by
          endpoint error) to reduce overlap -- same behavior as before.
        - If a sequence (list/array), it must have one entry per curve in
          `cmes` (same order as plotted), giving full manual control over
          each individual label's angle. No automatic sign-alternation is
          applied in this case -- you control the sign directly per entry
          (e.g. pass a negative value to angle a specific label downward).
        - If None (default), uses 20 degrees for every curve, with
          alternation -- matching the previous default behavior.

    Examples
    --------
    # Same angle for all curves (old behavior):
    plot_overlay_normalized(cmes, leader_angle_deg=20)

    # Per-curve manual control, one entry per curve in `cmes` order:
    plot_overlay_normalized(
        cmes,
        leader_angle_deg=[15, -25, 30, 10, -15, 20, 40, -30, 25, -10, 15, -20, 35],
    )
    """
    cb_colors = [
        "#1A6FA0",  # blue
        "#C1963A",  # orange
        "#26AE8A",  # bluish green
        "#932963",  # reddish purple
        "#995725",  # vermillion
        "#166796",  # sky blue
        "#E8B800",  # yellow (darkened for contrast on white background)
        "#000000",  # black
    ]

    n_curves = len(cmes)

    # Normalize leader_angle_deg into a per-curve array aligned with the
    # plotting order (index i in `cmes`), plus a flag for whether we
    # should auto-alternate sign (only when a single scalar/None was given).
    if leader_angle_deg is None:
        angles_by_index = np.full(n_curves, 20.0)
        auto_alternate = True
    elif np.isscalar(leader_angle_deg):
        angles_by_index = np.full(n_curves, float(leader_angle_deg))
        auto_alternate = True
    else:
        angles_by_index = np.asarray(leader_angle_deg, dtype=float)
        if len(angles_by_index) != n_curves:
            raise ValueError(
                f"leader_angle_deg has {len(angles_by_index)} entries, "
                f"but {n_curves} curves were given -- provide one angle per curve."
            )
        auto_alternate = False

    fig, ax = plt.subplots(figsize=(11, 6.5))

    max_lead = 0.0
    end_points = []  # (lead_hours_at_end, err_at_end, number_label, color, angle_deg)

    for i, cme_id in enumerate(cmes):
        try:
            df = load_cme(cme_id)
        except FileNotFoundError:
            continue

        err = df["ML_error"].to_numpy()
        lead_hours = df["Time_duration"].dt.total_seconds().to_numpy() / 3600.0

        max_lead = max(max_lead, lead_hours.max())

        color = cb_colors[i % len(cb_colors)]

        ax.plot(lead_hours, err, color=color, lw=1.6, alpha=0.9, zorder=3)

        # Curves run right -> left toward arrival, so the "end" of each
        # trajectory (closest to arrival) sits at min(lead_hours).
        end_idx = int(np.argmin(lead_hours))
        end_points.append(
            (lead_hours[end_idx], err[end_idx], str(i + 1), color, angles_by_index[i])
        )

    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5, zorder=1)
    ax.axvline(0, color="gray", lw=1, ls=":", alpha=0.6, zorder=1)

    # Arrival (Time_duration = 0) sits at the right edge; runs proceed
    # right -> left as lead time counts down toward arrival.
    ax.set_xlim(max_lead * 1.02, -max_lead * 0.02)

    # --- Numeric endpoint labeling with per-curve leader angle ---
    # Sorting by endpoint error keeps the auto-alternation (when active)
    # spreading labels sensibly through the convergence cluster.
    end_points.sort(key=lambda p: p[1])

    for j, (x, y, label, color, angle_deg) in enumerate(end_points):
        if auto_alternate:
            sign = 1 if j % 2 == 0 else -1
        else:
            sign = 1  # manual angles already encode their own sign
        eff_angle = angle_deg * sign

        dx = -leader_length * max_lead * np.cos(np.radians(eff_angle))
        dy = leader_length * max_lead * np.sin(np.radians(eff_angle))

        label_x = x + dx
        label_y = y + dy

        ax.annotate(
            label,
            xy=(x, y), xycoords="data",
            xytext=(label_x, label_y), textcoords="data",
            fontsize=8, color=color, fontweight="bold",
            va="center", ha="center",
            arrowprops=dict(
                arrowstyle="-", color=color, lw=0.7, alpha=0.7,
                shrinkA=0, shrinkB=2,
            ),
            zorder=4,
        )

    ax.set_xlabel("Lead time to arrival (hours)", fontsize=14)
    ax.set_ylabel("ML Error (hours)", fontsize=14)
    ax.set_title(
        "ML Error Trajectories",
        fontsize=18,
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "overlay_normalized.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------
if __name__ == "__main__":
    for cme_id in cmes:
        try:
            df = load_cme(cme_id)
            plot_single_cme(cme_id, df, save=True, show=False)
        except FileNotFoundError:
            print(f"Missing file for {cme_id}, skipping.")

    plot_all_cmes_grid(cmes)
    plot_overlay_normalized(cmes,
            leader_angle_deg=[5, -70, -3, 10, 20, 60, -30, 40, 25, -10, 15, 30, 10])

    print(f"All figures saved to: {OUT_DIR.resolve()}")