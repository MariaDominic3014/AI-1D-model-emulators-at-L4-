from netCDF4 import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, lognorm



#PREDICTED VALUES IN LINEAR SPACE
# #OPTION 1a#################################
# #model_name = "mse_ln"
# model_name = "crps_ln"
# prediction_file = (
#     Path(__file__).parent
#     / model_name
#     / f"{model_name}_linear_predicted.nc"
# )
# ##########################################
#OPTION 1b#################################
#model_name = "mse_original"
model_name = "crps_original"
prediction_file = (
    Path(__file__).parent
    / model_name
    / f"{model_name}_predicted.nc"
)
# ###########################################
space = "linear-space"

pred_ds = Dataset(str(prediction_file))

data_file = Path(__file__).parent.parent / "data" / "processed_ensemble.nc"
ens_ds = Dataset(str(data_file))


state_variables_names = ["P1_Chl", "P1_Chl_vert", "P2_Chl", "P2_Chl_vert", "Z6_c", "Z6_c_vert", "N1_p", "N3_n", "N4_n", "N5_s", "O2_bot", "O2_o", "P3_Chl", "P3_Chl_vert", "P4_Chl", "P4_Chl_vert", "Z5_c", "Z5_c_vert", "Z4_c", "Z4_c_vert"]

plotting_variables_names = ["P1_Chl"]

start_time = 365 + 334   # account for spin up (nearly 2 years)   #699
length_training_time = int(12*365.25)   # length of the time series for training data   #4383
length_prediction_time = ens_ds.dimensions["time"].size - start_time   # length of the time series for predictions (includes whole of training and validation periods as well as the testing period)   #8733 - 699 = 8034
lookback=30



#################################
# TIMESERIES
print(model_name)
print(space)

def histogram_mode(values, bins=100):
        values = np.asarray(values, dtype=np.float64)

        counts, edges = np.histogram(values, bins=bins)
        modal_bin = np.argmax(counts)
        return (edges[modal_bin] + edges[modal_bin + 1]) / 2


for var_name in state_variables_names:
    var = ens_ds.variables[var_name][start_time + lookback:]

    # #original/linear-space values
    # ens_mean = np.mean(var, axis=1)
    # ens_std = np.std(var, axis=1)

    empirical_mode = np.array([
        histogram_mode(values, bins=100)
        for values in var
    ])


    target_mean = pred_ds.variables[f"test_mean_{var_name}"][:]
    target_std = pred_ds.variables[f"test_std_{var_name}"][:]
    pred_mean = pred_ds.variables[f"predicted_mean_{var_name}"][:]
    pred_std = pred_ds.variables[f"predicted_std_{var_name}"][:]

    # #MIN/MAX OF ENSEMBLE/TARGET/PREDICTED VALUES
    # print(var_name)
    # print(f"Ensemble mean  ->  min: {ens_mean.min()}  /  max: {ens_mean.max()}")
    # print(f"Target mean  ->  min: {target_mean.min()}  /  max: {target_mean.max()}")
    # print(f"Prediction mean  ->  min: {pred_mean.min()}  /  max: {pred_mean.max()}")

    # print(f"Ensemble std  ->  min: {ens_std.min()}  /  max: {ens_std.max()}")
    # print(f"Target std  ->  min: {target_std.min()}  /  max: {target_std.max()}")
    # print(f"Prediction std  ->  min: {pred_std.min()}  /  max: {pred_std.max()}")
    # print()
    # ##########################

    # #HOW MANY TIMESTEPS HAVE PREDICTED STD (AND MEAN) BELOW ZERO?
    # pred_std_non_positive = np.count_nonzero(pred_std <= 0)
    # pred_mean_non_positive = np.count_nonzero(pred_mean <= 0)
    # print(var_name)
    # print(f"zero or negative predicted STD: {pred_std_non_positive} / {pred_std.size}")
    # print(f"zero or negative predicted MEAN: {pred_mean_non_positive} / {pred_mean.size}")
    # print()
    # print()
    # #######################

    plt.figure(figsize=(12,5))

    # ensemble_colour = "black"
    #mean+std
    # target_colour = "tab:purple"
    # prediction_colour = "tab:green"
    #std
    # target_colour = "tab:blue"
    # prediction_colour = "tab:orange"
    #mean
    target_colour = "tab:red"
    prediction_colour = "tab:cyan"

    #3 modes
    # empirical_colour = "black"
    # target_colour = "tab:blue"
    # prediction_colour = "tab:orange"

    #plt.plot(ens_mean, color=ensemble_colour, label="Ensemble mean")
    #plt.plot(ens_std, color=ensemble_colour, label="Ensemble std")
    # plt.fill_between(
    #     np.arange(len(ens_mean)),
    #     ens_mean - ens_std,
    #     ens_mean + ens_std,
    #     color=ensemble_colour,
    #     alpha=0.4,
    #     label="Ensemble ±1 standard deviation"
    # )

    #plt.plot(empirical_mode, color=empirical_colour, label="Empirical mode")

    plt.plot(target_mean, color=target_colour, label="Target mean/mode")
    #plt.plot(ens_std, color=ensemble_colour, label="Ensemble std")
    # plt.fill_between(
    #     np.arange(len(target_mean)),
    #     target_mean - target_std,
    #     target_mean + target_std,
    #     color=target_colour,
    #     alpha=0.5,
    #     label="Target ±1 standard deviation"
    # )

    plt.plot(pred_mean, color=prediction_colour, label="Predicted mean/mode")
    #plt.plot(pred_std, color=prediction_colour, label="Predicted std")
    # plt.fill_between(
    #     np.arange(len(pred_mean)),
    #     pred_mean - pred_std,
    #     pred_mean + pred_std,
    #     color=prediction_colour,
    #     alpha=0.3,
    #     label="Predicted ±1 standard deviation"
    # )

    plt.xlabel("Time (days)")
    plt.ylabel(var_name)
    #plt.title(f"{model_name} {var_name}: Empirical, Target & Predicted Mode")
    #plt.title(f"{model_name} {var_name}: Target Mean ±1 Std & Predicted Mean ±1 Std, {space}")
    plt.title(f"{model_name} {var_name}: Target Mean & Predicted Mean, {space}")
    #plt.title(f"{model_name} {var_name}: Target Std & Predicted Std, {space}")
    plt.legend()

    plt.tight_layout()
    #plt.show()
    #plt.savefig(Path(__file__).parent / f"{model_name}_{var_name}_EmpTargetPred_Mode_timeseries.png")
    #plt.savefig(Path(__file__).parent / f"{model_name}_{var_name}_targetAndPred_MeanStd_timeseries_{space}")
    plt.savefig(Path(__file__).parent / f"{model_name}_{var_name}_targetAndPred_Mean_timeseries_{space}.png")
    #plt.savefig(Path(__file__).parent / f"{model_name}_{var_name}_targetAndPred_Std_timeseries_{space}.png")

    plt.close()
################################


#################################
# ENSEMBLE HISTOGRAM & PREDICTED GAUSSIAN & TARGET GAUSSIAN
pred_t = (length_prediction_time - lookback) - 1
ens_t = (start_time + length_prediction_time) - 1

for var_name in plotting_variables_names:
    ensemble_values = ens_ds.variables[var_name][ens_t, :]


    #target mean + std at chosen timestep
    target_mean = pred_ds.variables[f"test_mean_{var_name}"][pred_t]
    target_std = pred_ds.variables[f"test_std_{var_name}"][pred_t]
    #predicted mean + std at chosen timestep
    pred_mean = pred_ds.variables[f"predicted_mean_{var_name}"][pred_t]
    pred_std = pred_ds.variables[f"predicted_std_{var_name}"][pred_t]


    # print(var_name)
    # print(space)
    # print("Target mean from prediction file:", target_mean)
    # print("Predicted mean from prediction file:", pred_mean)
    # print("Target std from prediction file:", target_std)
    # print("Predicted std from prediction file:", pred_std)
    # print()


    # GAUSSIAN
    #Gaussian PDF from model prediction values and target values
    x = np.linspace(
        min(target_mean - 4*target_std, pred_mean - 4*pred_std),
        max(target_mean + 4*target_std, pred_mean + 4*pred_std),
        500
    )

    target_pdf = (1 / (target_std * np.sqrt(2 * np.pi))) * np.exp(
    -0.5 * ((x - target_mean) / target_std)**2
    )
    pred_pdf = (1 / (pred_std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - pred_mean) / pred_std)**2
    )


    # Histogram of actual ensemble
    plt.hist(
        ensemble_values,
        bins=100,
        density=True,
        alpha=0.5,
        color="cornflowerblue",
        edgecolor="royalblue",
        linewidth=0.8,
        label="Actual Ensemble"
    )

    plt.plot(
        x,
        target_pdf,
        color="green",
        linewidth=2,
        label="Gaussian from target mean/std"
    )

    plt.plot(
        x,
        pred_pdf,
        color="darkblue",
        linestyle=":",
        linewidth=2,
        label="Gaussian from predicted mean/std"
    )

    plt.xlabel("Value")
    plt.ylabel("Probability density")
    plt.title(f"{model_name} {var_name} distributions at timestep {ens_t}, {space}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    #plt.savefig(f"{model_name}_{var_name}_{space}_distributions_timestep_{ens_t}")
    plt.close()
################################



pred_ds.close()
ens_ds.close()