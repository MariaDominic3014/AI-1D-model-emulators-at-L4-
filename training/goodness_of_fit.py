from netCDF4 import Dataset
from pathlib import Path
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm, kstest



model_1 = "mse_original"
model_2 = "crps_original"
model_3 = "mse_ln"
model_4 = "crps_ln"

model_1_file = (Path(__file__).parent / model_1 / f"{model_1}_predicted.nc")
model_2_file = (Path(__file__).parent / model_2 / f"{model_2}_predicted.nc")
model_3_file = (Path(__file__).parent / model_3 / f"{model_3}_log_predicted.nc")
model_4_file = (Path(__file__).parent / model_4 / f"{model_4}_log_predicted.nc")

mse_original_ds = Dataset(str(model_1_file))
crps_original_ds = Dataset(str(model_2_file))
mse_ln_ds = Dataset(str(model_3_file))
crps_ln_ds = Dataset(str(model_4_file))


ens_file = Path(__file__).parent.parent / "data" / "processed_ensemble.nc"
ens_ds = Dataset(str(ens_file))

state_variables_names = ["P1_Chl", "P1_Chl_vert", "P2_Chl", "P2_Chl_vert", "Z6_c", "Z6_c_vert", "N1_p", "N3_n", "N4_n", "N5_s", "O2_bot", "O2_o", "P3_Chl", "P3_Chl_vert", "P4_Chl", "P4_Chl_vert", "Z5_c", "Z5_c_vert", "Z4_c", "Z4_c_vert"]

selected_variables_names = ["P1_Chl"]

start_time = 365 + 334   # account for spin up (nearly 2 years)   #699
length_prediction_time = ens_ds.dimensions["time"].size - start_time   # length of the time series for predictions (includes whole of training and validation periods as well as the testing period, and the 30-day lookback/initialisation window)   #8733 - 699 = 8034
lookback=30

# The source dataset starts on 01/02/2002 and ends on 30/12/2025
dataset_origin = pd.Timestamp("2002-02-01")

plot_start = start_time + lookback
plot_length = length_prediction_time - lookback

plot_dates = pd.date_range(
    start=dataset_origin + pd.Timedelta(days=plot_start),
    periods=plot_length,
    freq="D",
)

epsilon = 1e-6



GoodnessOfFit = namedtuple("GoodnessOfFit", ["ks_statistic", "p_value"])

def _pit_goodness_of_fit(pit_values):
    result = kstest(pit_values, "uniform")
    return GoodnessOfFit(float(result.statistic), float(result.pvalue))

def normal_fit_score(values, mean, std):
    values = np.asarray(values, dtype=np.float64)
    if std <= 0:
        raise ValueError("std must be positive")
 
    pit = norm.cdf(values, loc=mean, scale=std)
    return _pit_goodness_of_fit(pit)

def lognormal_fit_score(values, mu, sigma):
    values = np.asarray(values, dtype=np.float64)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if np.any(values <= 0):
        raise ValueError("log-normal fit requires strictly positive values")
 
    pit = lognorm.cdf(values, s=sigma, scale=np.exp(mu))
    return _pit_goodness_of_fit(pit)



for var_name in selected_variables_names:
    var = ens_ds.variables[var_name][start_time + lookback:]

    # linear-space ensemble values
    ens_mean = np.asarray(np.mean(var, axis=1), dtype=np.float64)
    ens_std = np.asarray(np.std(var, axis=1), dtype=np.float64)
    ens_std = np.maximum(ens_std, epsilon)

    # ln-space ensemble values
    log_var = np.log(var + np.exp(-8))
    ens_mu = np.asarray(np.mean(log_var, axis=1), dtype=np.float64)
    ens_sigma = np.asarray(np.std(log_var, axis=1), dtype=np.float64)
    ens_sigma = np.maximum(ens_sigma, epsilon)

    # mse-original
    mse_original_pred_mean = np.asarray(mse_original_ds.variables[f"predicted_mean_{var_name}"][:], dtype=np.float64)
    mse_original_pred_std = np.asarray(mse_original_ds.variables[f"predicted_std_{var_name}"][:], dtype=np.float64)
    mse_original_pred_std = np.where(
        mse_original_pred_std <= 0,
        epsilon,
        mse_original_pred_std
    )

    # crps-original
    crps_original_pred_mean = np.asarray(crps_original_ds.variables[f"predicted_mean_{var_name}"][:], dtype=np.float64)
    crps_original_pred_std = np.asarray(crps_original_ds.variables[f"predicted_std_{var_name}"][:], dtype=np.float64)

    # mse-ln
    mse_ln_pred_mu = np.asarray(mse_ln_ds.variables[f"predicted_mean_{var_name}"][:], dtype=np.float64)
    mse_ln_pred_sigma = np.asarray(mse_ln_ds.variables[f"predicted_std_{var_name}"][:], dtype=np.float64)
    mse_ln_pred_sigma = np.where(
        mse_ln_pred_sigma <= 0,
        epsilon,
        mse_ln_pred_sigma
    )

    # crps-ln
    crps_ln_pred_mu = np.asarray(crps_ln_ds.variables[f"predicted_mean_{var_name}"][:], dtype=np.float64)
    crps_ln_pred_sigma = np.asarray(crps_ln_ds.variables[f"predicted_std_{var_name}"][:], dtype=np.float64)


    ensemble = np.asarray(var, dtype=np.float64)


    ens_original_normal_predicted_ks = []
    ens_ln_lognormal_predicted_ks = []
    mse_original_normal_predicted_ks = []
    crps_original_normal_predicted_ks = []
    mse_ln_lognormal_predicted_ks = []
    crps_ln_lognormal_predicted_ks = []


    for timestep_index in range(length_prediction_time - lookback):
        #choose which ones you want

        # ens_original_normal_gof = normal_fit_score(
        #     ensemble[timestep_index],
        #     ens_mean[timestep_index],
        #     ens_std[timestep_index]
        # )
        # ens_original_normal_predicted_ks.append(ens_original_normal_gof.ks_statistic)

        # ens_ln_lognormal_gof = lognormal_fit_score(
        #     ensemble[timestep_index],
        #     ens_mu[timestep_index],
        #     ens_sigma[timestep_index]
        # )
        # ens_ln_lognormal_predicted_ks.append(ens_ln_lognormal_gof.ks_statistic)

        mse_original_gof = normal_fit_score(
            ensemble[timestep_index],
            mse_original_pred_mean[timestep_index],
            mse_original_pred_std[timestep_index]
        )
        mse_original_normal_predicted_ks.append(mse_original_gof.ks_statistic)

        crps_original_gof = normal_fit_score(
            ensemble[timestep_index],
            crps_original_pred_mean[timestep_index],
            crps_original_pred_std[timestep_index]
        )
        crps_original_normal_predicted_ks.append(crps_original_gof.ks_statistic)

        # mse_ln_gof = lognormal_fit_score(
        #     ensemble[timestep_index],
        #     mse_ln_pred_mu[timestep_index],
        #     mse_ln_pred_sigma[timestep_index]
        # )
        # mse_ln_lognormal_predicted_ks.append(mse_ln_gof.ks_statistic)

        # crps_ln_gof = lognormal_fit_score(
        #     ensemble[timestep_index],
        #     crps_ln_pred_mu[timestep_index],
        #     crps_ln_pred_sigma[timestep_index]
        # )
        # crps_ln_lognormal_predicted_ks.append(crps_ln_gof.ks_statistic)



    # fig, ax = plt.subplots(figsize=(16, 5))

    # ax.plot(plot_dates, ens_original_normal_predicted_ks,
    #         color="tab:blue", label="Original Ensemble GOF")
    # ax.plot(plot_dates, mse_original_normal_predicted_ks,
    #         color="tab:orange", label="mse_original GOF")
    # ax.plot(plot_dates, crps_original_normal_predicted_ks,
    #         color="tab:green", label="crps_original GOF")
    # ax.plot(plot_dates, ens_ln_lognormal_predicted_ks,
    #         color="tab:red", label="Logged Ensemble GOF")
    # ax.plot(plot_dates, mse_ln_lognormal_predicted_ks,
    #         color="tab:purple", label="mse_ln GOF")
    # ax.plot(plot_dates, crps_ln_lognormal_predicted_ks,
    #         color="tab:brown", label="crps_ln GOF")

    # ax.set_xlabel("Date")
    # ax.set_ylabel("GOF KS statistic")
    # ax.set_yscale("log")

    # # Show one labelled tick every 3 years
    # ax.xaxis.set_major_locator(mdates.YearLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # ax.xaxis.set_minor_locator(mdates.MonthLocator())

    # # Dashed lines at the beginning of each season.
    # plot_start_date = plot_dates[0]
    # plot_end_date = plot_dates[-1]
    # # print("First plotted date:", plot_dates[0])
    # # print("Last plotted date:", plot_dates[-1])

    # for year in range(plot_start_date.year, plot_end_date.year + 1):
    #     for month in (3, 6, 9, 12):
    #         line_date = pd.Timestamp(year, month, 1)

    #         if plot_start_date <= line_date <= plot_end_date:
    #             ax.axvline(
    #                 line_date,
    #                 color="grey",
    #                 linestyle="--",
    #                 linewidth=0.6,
    #                 alpha=0.35,
    #                 zorder=0,
    #             )


    # ax.set_title(f"{var_name}: Log-scale Normal vs Lognormal goodness of fit")
    # ax.legend()
    # fig.autofmt_xdate()
    # fig.tight_layout()

    # fig.savefig(
    #     Path(__file__).parent /
    #     f"{var_name}_logscale_normal_vs_lognormal_gof.png"
    # )
    # plt.close(fig)



    plt.figure(figsize=(16,5))
    
    # plt.plot(ens_original_normal_predicted_ks, color="tab:blue", label="Original Ensemble GOF")
    plt.plot(mse_original_normal_predicted_ks, color="tab:orange", label="mse_original GOF")
    plt.plot(crps_original_normal_predicted_ks, color="tab:green", label="crps_original GOF")
    # plt.plot(ens_ln_lognormal_predicted_ks, color="tab:red", label="Logged Ensemble GOF")
    # plt.plot(mse_ln_lognormal_predicted_ks, color="tab:purple", label="mse_ln GOF")
    # plt.plot(crps_ln_lognormal_predicted_ks, color="tab:brown", label="crps_ln GOF")

    plt.xlabel("Timestep")
    plt.ylabel("GOF ks statistic")

    # plt.yscale("log")

    # plt.title(f"{var_name}: Normal goodness of fit")
    # plt.title(f"{var_name}: Lognormal goodness of fit")
    # plt.title(f"{var_name}: Normal vs Lognormal goodness of fit")
    # plt.title(f"{var_name}: Original vs Logged Ensemble goodness of fit")
    plt.title(f"{var_name}: Linear models' goodness of fit")
    # plt.title(f"{var_name}: Ln models' goodness of fit")
    # plt.title(f"{var_name}: Log-scale Normal vs Lognormal goodness of fit")

    plt.legend()
    plt.tight_layout()
    # plt.show()

    # plt.savefig(Path(__file__).parent / f"{var_name}_normal_gof.png")
    # plt.savefig(Path(__file__).parent / f"{var_name}_lognormal_gof.png")
    # plt.savefig(Path(__file__).parent / f"{var_name}_normal_vs_lognormal_gof.png")
    # plt.savefig(Path(__file__).parent / f"{var_name}_original_vs_logged_ensemble_gof.png")
    plt.savefig(Path(__file__).parent / f"{var_name}_linear_models_gof.png")
    # plt.savefig(Path(__file__).parent / f"{var_name}_ln_models_gof.png")
    # plt.savefig(Path(__file__).parent / f"{var_name}_logscale_normal_vs_lognormal_gof.png")

    plt.close()

    

mse_original_ds.close()
crps_original_ds.close()
mse_ln_ds.close()
crps_ln_ds.close()
ens_ds.close()