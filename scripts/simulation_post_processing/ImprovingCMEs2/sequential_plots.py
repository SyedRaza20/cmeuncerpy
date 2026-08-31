import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

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
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def load_cme(cme_id: str) -> pl.DataFrame:
    """Load a single CME's sequential results and parse the Time column."""
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

    ax.fill_between(time, err, 0, where=(err >= 0), color="#d62728", alpha=0.15)
    ax.fill_between(time, err, 0, where=(err < 0), color="#2ca02c", alpha=0.15)

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

        ax.plot(time, err, lw=1.3, color="#2ca02c")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.fill_between(time, err, 0, where=(err >= 0), color="#d62728", alpha=0.15)
        ax.fill_between(time, err, 0, where=(err < 0), color="#2ca02c", alpha=0.15)
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

    fig.suptitle("ML Error Evolution Across All CME Events", fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.01, "Time", ha="center", fontsize=11)
    fig.text(-0.005, 0.5, "ML Error (hours)", va="center", rotation="vertical", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "all_cmes_grid.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------
# 3. Overlay plot — normalized "time since start" so all events
#    line up on one axis (great for spotting systematic trends)
# ---------------------------------------------------------------
def plot_overlay_normalized(cmes):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab20")

    for i, cme_id in enumerate(cmes):
        try:
            df = load_cme(cme_id)
        except FileNotFoundError:
            continue
        time = df["Time"].to_list()
        err = df["ML_error"].to_numpy()
        t0 = time[0]
        hours_elapsed = np.array([(t - t0).total_seconds() / 3600 for t in time])
        ax.plot(hours_elapsed, err, label=cme_id, color=cmap(i % 20), lw=1.4, alpha=0.85)

    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Hours since sequential run start")
    ax.set_ylabel("ML Error (hours)")
    ax.set_title("ML Error Trajectories — All CMEs Overlaid", fontsize=13, fontweight="bold")
    ax.legend(ncol=2, fontsize=8, loc="upper right", framealpha=0.9)
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
    plot_overlay_normalized(cmes)

    print(f"All figures saved to: {OUT_DIR.resolve()}")