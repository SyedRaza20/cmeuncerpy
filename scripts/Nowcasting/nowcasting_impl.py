# import statements
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy.stats import norm
import scipy
from sklearn.preprocessing import MinMaxScaler
import warnings
from cmeuncerpy.models.LinearRegression import LinearRegression
from tqdm import tqdm
import time
import argparse
import sys

warnings.filterwarnings('ignore')

# ----------------------------
# Settings
# ----------------------------

statistics_figure = True
parser = argparse.ArgumentParser(description="for the (1) experiment (1, 2, or 3) and (2) target (Bt or Bz)")

# add the arguments that I need 
parser.add_argument("experiment", type=int, help="The experiment number")
parser.add_argument("target", type=str, help="The magnetic field target")
parser.add_argument("file_name", type=str, help="file name to store the results")

# parse the arguments 
args = parser.parse_args()

# access the arguments usign dot notation
print(f"The experiment that will be run is: {args.experiment}")
print(f"The target that will be predicted is: {args.target}")

if args.target.lower().strip() == "bt":
    bt_solution = True 
elif args.target.lower().strip() == "bz":
    bt_solution = False
else:
    raise TypeError("The target value must be bt or bz.")

if args.experiment == 1:
    file = "CME_features_sheath.csv"
elif args.experiment == 2:
    file = "CME_features_sheath_and_MO_4.csv"
elif args.experiment == 3:
    file = "CME_features_MO_4.csv"
else:
    raise TypeError("Experiment must be an int with types 1, 2, 3.")

N_SEEDS = 15

# ----------------------------
# helper functions 
# ----------------------------

def read_cme_csv(file_path: str,
                 float_dtype=pl.Float32,
                 str_cols=("icmecat_id", "sc_insitu"),
                 datetime_cols=("icme_start_time", "mo_start_time", "mo_end_time"),
                 infer_schema_length: int = 10000) -> pl.DataFrame:
    # get column names - read only the header without schema inference to avoid
    # parsing errors (some files contain floats that would otherwise be
    # mis-inferred as integers). Setting ``infer_schema_length=0`` prevents
    # Polars from scanning data when reading the header.
    header = pl.read_csv(file_path, n_rows=0, infer_schema_length=0, ignore_errors=True)
    cols = header.columns

    # build schema: two strings, three datetimes, rest floats
    schema_overrides = {}
    for c in cols:
        if c in str_cols:
            schema_overrides[c] = pl.Utf8
        elif c in datetime_cols:
            schema_overrides[c] = pl.Datetime
        else:
            schema_overrides[c] = float_dtype

    try:
        # prefer enforcing schema during parse
        return pl.read_csv(file_path, schema_overrides=schema_overrides, infer_schema_length=infer_schema_length)
    except Exception:
        # fallback: permissive read then cast explicitly
        df = pl.read_csv(file_path, infer_schema_length=0, ignore_errors=True)
        cast_exprs = []
        for c in df.columns:
            if c in str_cols:
                cast_exprs.append(pl.col(c).cast(pl.Utf8).alias(c))
            elif c in datetime_cols:
                cast_exprs.append(pl.col(c).str.strptime(pl.Datetime, fmt=None, strict=False).alias(c))
            else:
                cast_exprs.append(pl.col(c).cast(float_dtype).alias(c))
        return df.with_columns(cast_exprs)

def plot_cme_statistics(cme_data: pl.DataFrame):
    """
    Plot bt_target, bz_target, sheath duration (hours), and MO duration (hours).
    Legends contain max, min, mean, std (no vertical lines).
    Returns (fig, axs).
    """

    # Convert durations (ns → hours)
    sheath_duration_hours = cme_data.select(
        (
            (pl.col('mo_start_time') - pl.col('icme_start_time'))
            .dt.cast_time_unit('ns')
            .cast(pl.Int64) / 3_600_000_000_000
        ).alias('sheath_hours')
    )['sheath_hours']

    mo_duration_hours = cme_data.select(
        (
            (pl.col('mo_end_time') - pl.col('mo_start_time'))
            .dt.cast_time_unit('ns')
            .cast(pl.Int64) / 3_600_000_000_000
        ).alias('mo_hours')
    )['mo_hours']

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Small helper to format stats text
    def make_stats_legend(ax, data):
        stats_text = (
            f"Mean: {data.mean():.2f}\n"
            f"Std: {data.std():.2f}\n"
            f"Min: {data.min():.2f}\n"
            f"Max: {data.max():.2f}"
        )
        # Empty plot handle just for the legend text
        ax.legend([plt.Line2D([], [], color='none')], [stats_text], frameon=True, fontsize=12)

    # --- (1) bt_target ---
    bt = cme_data['bt_target'].to_numpy()
    axs[0, 0].hist(bt, bins=35, density=True, alpha=0.5, color='goldenrod')
    mu, sigma = norm.fit(bt)
    x = np.linspace(bt.min(), bt.max(), 500)
    axs[0, 0].plot(x, norm.pdf(x, mu, sigma), linewidth=2, color='goldenrod')
    axs[0, 0].set_title('$B_{tot}$', fontsize=18)
    axs[0, 0].set_xlabel('$B_{tot}$ (nT)', fontsize=14)
    axs[0, 0].set_ylabel('Frequency', fontsize=14)
    make_stats_legend(axs[0, 0], bt)

    # --- (2) bz_target ---
    bz = cme_data['bz_target'].to_numpy()
    axs[0, 1].hist(bz, bins=30, density = True, alpha=0.5, color='salmon')
    mu, sigma = norm.fit(bz)
    x = np.linspace(bz.min(), bz.max(), 500)
    axs[0, 1].plot(x, norm.pdf(x, mu, sigma), linewidth=2, color='salmon')
    axs[0, 1].set_title('$B_{z}$', fontsize=18)
    axs[0, 1].set_xlabel('$B_{z}$ (nT)', fontsize=14)
    axs[0, 1].set_ylabel('Frequency', fontsize=14)
    make_stats_legend(axs[0, 1], bz)

    # --- (3) sheath duration ---
    sh = sheath_duration_hours.to_numpy()
    axs[1, 0].hist(sh, bins=15, density = True, alpha = 0.5, color='purple')
    mu, sigma = norm.fit(sh)
    x = np.linspace(sh.min(), sh.max(), 500)
    axs[1, 0].plot(x, norm.pdf(x, mu, sigma), linewidth=2, color='purple')
    axs[1, 0].set_title('Sheath Duration', fontsize=18)
    axs[1, 0].set_xlabel('Sheath Duration (hours)', fontsize=14)
    axs[1, 0].set_ylabel('Frequency', fontsize=14)
    make_stats_legend(axs[1, 0], sh)

    # --- (4) MO duration ---
    mo = mo_duration_hours.to_numpy()
    axs[1, 1].hist(mo, bins=25, density = True, alpha = 0.5, color='green')
    mu, sigma = norm.fit(mo)
    x = np.linspace(mo.min(), mo.max(), 500)
    axs[1, 1].plot(x, norm.pdf(x, mu, sigma), linewidth=2, color='green')
    axs[1, 1].set_title('MO Duration', fontsize=18)
    axs[1, 1].set_xlabel('MO Duration (hours)', fontsize=14)
    axs[1, 1].set_ylabel('Frequency', fontsize=14)
    make_stats_legend(axs[1, 1], mo)

    fig.tight_layout()
    return fig, axs

def calculate_metrics(pred: np.ndarray, y_test: np.ndarray):
    """
    Calculates all the metrics that we need bytaking in the 
    two arrays - prediction and test
    """
    am = np.mean(pred)
    me = np.mean(y_test - pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    pcc = scipy.stats.pearsonr(y_test.flatten(), pred.flatten())[0]
    r_sq = r2_score(y_test, pred)

    return am, me, mae, rmse, pcc, r_sq

# the main function:
if __name__ == "__main__":
    file_path = "../../Data/feature_space/" + file
    cme_data = read_cme_csv(file_path)

    # make the figure if needed 
    if statistics_figure == True:
        fig, axs = plot_cme_statistics(cme_data)
        plt.savefig('figures_and_metrics/histogram.png', dpi = 300, bbox_inches = 'tight')

    # the feature columns
    feature_cols = [
        col for col in cme_data.columns
        if cme_data[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
        and col not in ["bt_target", "bz_target", "idx"]
    ]

    X_np = cme_data.select(feature_cols).to_numpy()
    y_np = cme_data.select(["bt_target", "bz_target"]).to_numpy()

    target_idx = 0 if bt_solution else 1
    y_np = y_np[:, target_idx]


    seeds = range(N_SEEDS)

    metrics = {
        "LR": {"AM": [], "ME": [], "MAE": [], "RMSE": [], "PCC": [], "R_sq": []},
        "SVR": {"AM": [], "ME": [], "MAE": [], "RMSE": [], "PCC": [], "R_sq": []},
        "GBM": {"AM": [], "ME": [], "MAE": [], "RMSE": [], "PCC": [], "R_sq": []},
        "RFR": {"AM": [], "ME": [], "MAE": [], "RMSE": [], "PCC": [], "R_sq": []},
    }

    # ----------------------------
    # loop run accross the seeds
    # ----------------------------

    for seed in tqdm(seeds, bar_format='{l_bar}{bar:20}{r_bar}'):
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=0.3, random_state=int(seed), shuffle=True
        )

        # ----------------------------
        # Gradient Boosting method
        # ----------------------------

        # make the object
        gbm = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3)
        gbm.fit(X_train, y_train)
        y_pred_gbm = gbm.predict(X_test)

        # calculate the metrics
        am_gbm, me_gbm, mae_gbm, rmse_gbm, pcc_gbm, r_sq_gbm = calculate_metrics(y_pred_gbm, y_test)

        # metrics in the dictionary
        metrics["GBM"]["AM"].append(am_gbm)
        metrics["GBM"]["ME"].append(me_gbm)
        metrics["GBM"]["MAE"].append(mae_gbm)
        metrics["GBM"]["RMSE"].append(rmse_gbm)
        metrics["GBM"]["PCC"].append(pcc_gbm)
        metrics["GBM"]["R_sq"].append(r_sq_gbm)

        # ----------------------------
        # Random Forest Regression
        # ----------------------------

        # make the object
        seed_rfr = 42
        rfr = RandomForestRegressor(n_estimators=300, random_state=seed_rfr, max_depth=5)
        rfr.fit(X_train, y_train)
        y_pred_rfr = rfr.predict(X_test)

        # calculate the metrics
        am_rfr, me_rfr, mae_rfr, rmse_rfr, pcc_rfr, r_sq_rfr = calculate_metrics(y_pred_rfr, y_test)
        
        # metrics in the dictionary
        metrics["RFR"]["AM"].append(am_rfr)
        metrics["RFR"]["ME"].append(me_rfr)
        metrics["RFR"]["MAE"].append(mae_rfr)
        metrics["RFR"]["RMSE"].append(rmse_rfr)
        metrics["RFR"]["PCC"].append(pcc_rfr)
        metrics["RFR"]["R_sq"].append(r_sq_rfr)

        # lets scale the data before using SVR and LR
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ----------------------------
        # Support Vector Regression
        # ----------------------------

        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svr.fit(X_train, y_train)
        y_pred_svr = svr.predict(X_test)

        # calculate the metrics
        am_svr, me_svr, mae_svr, rmse_svr, pcc_svr, r_sq_svr = calculate_metrics(y_pred_svr, y_test)

        # metrics in the dictionary
        metrics["SVR"]["AM"].append(am_svr)
        metrics["SVR"]["ME"].append(me_svr)
        metrics["SVR"]["MAE"].append(mae_svr)
        metrics["SVR"]["RMSE"].append(rmse_svr)
        metrics["SVR"]["PCC"].append(pcc_svr)
        metrics["SVR"]["R_sq"].append(r_sq_svr)

        # ----------------------------
        # Linear Regression
        # ----------------------------

        lr = LinearRegression(X_train.shape[1], 0.01, 1000000)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        # calculate the metrics:
        am_lr, me_lr, mae_lr, rmse_lr, pcc_lr, r_sq_lr = calculate_metrics(y_pred_lr, y_test)

        # put them in the dictionary:
        metrics["LR"]["AM"].append(am_lr)
        metrics["LR"]["ME"].append(me_lr)
        metrics["LR"]["MAE"].append(mae_lr)
        metrics["LR"]["RMSE"].append(rmse_lr)
        metrics["LR"]["PCC"].append(pcc_lr)
        metrics["LR"]["R_sq"].append(r_sq_lr)

        # the tqdm stuff
        time.sleep(0.01)
    
    print("Model,Metric,Mean,Std")

    with open(f"figures_and_metrics/{args.file_name}", 'w') as f:
        f.write("Model,Metric,Mean,Std\n")
        for model_name, model_metrics in metrics.items():
            for metric_name, values in model_metrics.items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                line = f"{model_name},{metric_name},{mean_val:.4f},{std_val:.4f}"
                print(line)
                f.write(line + "\n")

    print(f"The metrics calculated for experiment {args.experiment}, and target {args.target} for all the models are saved in the file")
    print(f"{args.file_name}")