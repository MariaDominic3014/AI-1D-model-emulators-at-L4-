from netCDF4 import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np



# #OPTION 2: PLOTS IN LOG SPACE
# model_name1 = "mse_ln"
# prediction_file1 = (
#     Path(__file__).parent
#     / model_name
#     / f"{model_name}_log_predicted.nc"
# )

# model_name2 = "crps_ln"
# prediction_file2 = (
#     Path(__file__).parent
#     / model_name
#     / f"{model_name}_log_predicted.nc"
# )

# space = "log-space"
#########################

# #OPTION 1: PLOTS IN LINEAR SPACE
model_name1 = "mse_original"
prediction_file1 = (
    Path(__file__).parent
    / model_name1
    / f"{model_name1}_predicted.nc"
)

model_name2 = "crps_original"
prediction_file2 = (
    Path(__file__).parent
    / model_name2
    / f"{model_name2}_predicted.nc"
)

space = "linear-space"
#########################


pred_ds1 = Dataset(str(prediction_file1))
pred_ds2 = Dataset(str(prediction_file2))

data_file = Path(__file__).parent.parent / "data" / "processed_ensemble.nc"
ens_ds = Dataset(str(data_file))


state_variables_names = ["P1_Chl", "P1_Chl_vert", "P2_Chl", "P2_Chl_vert", "Z6_c", "Z6_c_vert", "N1_p", "N3_n", "N4_n", "N5_s", "O2_bot", "O2_o", "P3_Chl", "P3_Chl_vert", "P4_Chl", "P4_Chl_vert", "Z5_c", "Z5_c_vert", "Z4_c", "Z4_c_vert"]

plotting_variables_names = ["P3_Chl", "P3_Chl_vert"]

start_time = 365 + 334   # account for spin up (nearly 2 years)   #699
length_training_time = int(12*365.25)   # length of the time series for training data   #4383
length_prediction_time = ens_ds.dimensions["time"].size - start_time   # length of the time series for predictions (includes whole of training and validation periods as well as the testing period)   #8733 - 699 = 8034
lookback=30



#################################
# MEAN + STANDARD DEVIATION TIMESERIES
for var_name in state_variables_names:
    var = ens_ds.variables[var_name][start_time + lookback:]

    #OPTION 1: for original/linear-space values
    ens_mean = np.mean(var, axis=1)
    ens_std = np.std(var, axis=1)

    # #OPTION 2: for ln values
    # log_var = np.log(var)
    # ens_mean = np.mean(log_var, axis=1)
    # ens_std = np.std(log_var, axis=1)

    #target mean + std taken from model 1 prediction file (should be the same as model 2 prediction file)
    target_mean = pred_ds1.variables[f"test_mean_{var_name}"][:]
    target_std = pred_ds1.variables[f"test_std_{var_name}"][:]
    #pred mean and std from both models
    pred_mean1 = pred_ds1.variables[f"predicted_mean_{var_name}"][:]
    pred_std1 = pred_ds1.variables[f"predicted_std_{var_name}"][:]
    pred_mean2 = pred_ds2.variables[f"predicted_mean_{var_name}"][:]
    pred_std2 = pred_ds2.variables[f"predicted_std_{var_name}"][:]

    # #MIN/MAX of ensemble/target/predicted values
    # print(var_name)
    # print(space)
    # print(f"Ensemble mean  ->  min: {ens_mean.min()}  /  max: {ens_mean.max()}")
    # #print(f"Target mean  ->  min: {target_mean.min()}  /  max: {target_mean.max()}")
    # print(f"{model_name1} Prediction mean  ->  min: {pred_mean1.min()}  /  max: {pred_mean1.max()}")
    # print(f"{model_name2} Prediction mean  ->  min: {pred_mean2.min()}  /  max: {pred_mean2.max()}")
    # print()
    # print(f"Ensemble std  ->  min: {ens_std.min()}  /  max: {ens_std.max()}")
    # #print(f"Target std  ->  min: {target_std.min()}  /  max: {target_std.max()}")
    # print(f"{model_name1} Prediction std  ->  min: {pred_std1.min()}  /  max: {pred_std1.max()}")
    # print(f"{model_name2} Prediction std  ->  min: {pred_std2.min()}  /  max: {pred_std2.max()}")
    # print("="*20)
    # #########################

    # #HOW MANY TIMESTEPS HAVE PREDICTED STD (AND MEAN) BELOW ZERO?
    # pred_std_non_positive_m1 = np.count_nonzero(pred_std1 <= 0)
    # pred_mean_non_positive_m1 = np.count_nonzero(pred_mean1 <= 0)
    # pred_std_non_positive_m2 = np.count_nonzero(pred_std2 <= 0)
    # pred_mean_non_positive_m2 = np.count_nonzero(pred_mean2 <= 0)

    # print(var_name)
    # print(f"zero or negative predicted STD {model_name1}: {pred_std_non_positive_m1} / {pred_std1.size}")
    # print(f"zero or negative predicted MEAN {model_name1}: {pred_mean_non_positive_m1} / {pred_mean1.size}")
    # print(f"zero or negative predicted STD {model_name2}: {pred_std_non_positive_m2} / {pred_std2.size}")
    # print(f"zero or negative predicted MEAN {model_name2}: {pred_mean_non_positive_m2} / {pred_mean2.size}")
    # print()
    # #######################

    plt.figure(figsize=(12,5))

    #mean+std
    # ensemble_colour = "black"
    # prediction_colour1 = "tab:green"
    # prediction_colour2 = "tab:purple"
    #std
    # ensemble_colour = "black"
    # prediction_colour1 = "tab:orange"
    # prediction_colour2 = "tab:blue"
    #mean
    ensemble_colour = "black"
    prediction_colour1 = "tab:cyan"
    prediction_colour2 = "tab:red"

    plt.plot(ens_mean, color=ensemble_colour, label="Ensemble mean")
    #plt.plot(ens_std, color=ensemble_colour, label="Ensemble std")
    # plt.fill_between(
    #     np.arange(len(ens_mean)),
    #     ens_mean - ens_std,
    #     ens_mean + ens_std,
    #     color=ensemble_colour,
    #     alpha=0.4,
    #     label="±1 standard deviation"
    # )

    plt.plot(pred_mean1, color=prediction_colour1, label=f"{model_name1} Predicted mean")
    #plt.plot(pred_std1, color=prediction_colour, label=f"{model_name1} Predicted std")
    # plt.fill_between(
    #     np.arange(len(pred_mean1)),
    #     pred_mean1 - pred_std1,
    #     pred_mean1 + pred_std1,
    #     color=prediction_colour,
    #     alpha=0.2,
    #     label=f"{model_name1} Predicted ±1 standard deviation"
    # )

    plt.plot(pred_mean2, color=prediction_colour2, label=f"{model_name2} Predicted mean")
    #plt.plot(pred_std2, color=prediction_colour, label=f"{model_name2} Predicted std")
    # plt.fill_between(
    #     np.arange(len(pred_mean2)),
    #     pred_mean2 - pred_std2,
    #     pred_mean2 + pred_std2,
    #     color=prediction_colour,
    #     alpha=0.2,
    #     label=f"{model_name2} Predicted ±1 standard deviation"
    # )

    plt.xlabel("Time (days)")
    plt.ylabel(var_name)
    #plt.title(f"Both Models: {var_name} Ensemble Mean ±1 Std & Predicted Means ±1 Std.s {space}")
    plt.title(f"Both Models: {var_name} Ensemble Mean & Predicted Means {space}")
    #plt.title(f"Both Models: {var_name} Ensemble Std & Predicted Std.s {space}")
    plt.legend()

    plt.tight_layout()
    #plt.show()
    #plt.savefig(f"both_models_{var_name}_predAndActual_MeanStd_timeseries_{space}")
    plt.savefig(f"both_models_{var_name}_predAndActual_Mean_timeseries_{space}.png")
    #plt.savefig(f"both_models_{var_name}_predAndActual_Std_timeseries_{space}.png")

    plt.close()
################################


#################################
# ENSEMBLE HISTOGRAM & PREDICTED GAUSSIAN & TARGET GAUSSIAN (from 2 models)
pred_t = (length_prediction_time - lookback) - 1 - 1000
ens_t = (start_time + length_prediction_time) - 1 - 1000

for var_name in state_variables_names:
    ensemble_values = ens_ds.variables[var_name][ens_t, :]

    #OPTION 1: linear ensemble values at chosen timestep
    linear_values = ensemble_values

    # #OPTION 2: log ensemble values at chosen timestep
    # log_values = np.log(ensemble_values)

    #actual mean + std at chosen timestep
    target_mean = pred_ds1.variables[f"test_mean_{var_name}"][pred_t]
    target_std = pred_ds1.variables[f"test_std_{var_name}"][pred_t]
    #predicted mean + std at chosen timestep
    pred_mean1 = pred_ds1.variables[f"predicted_mean_{var_name}"][pred_t]
    pred_std1 = pred_ds1.variables[f"predicted_std_{var_name}"][pred_t]
    pred_mean2 = pred_ds2.variables[f"predicted_mean_{var_name}"][pred_t]
    pred_std2 = pred_ds2.variables[f"predicted_std_{var_name}"][pred_t]


    print(var_name)
    print(space)
    print(f"Target mean from {model_name1} prediction file:", target_mean)
    print(f"Predicted mean from {model_name1} file:", pred_mean1)
    print(f"Predicted mean from {model_name2} file:", pred_mean2)
    print()
    print(f"Target std from {model_name1} prediction file:", target_std)
    print(f"Predicted std from {model_name1} file:", pred_std1)
    print(f"Predicted std from {model_name2} file:", pred_std2)
    print("="*20)


    # GAUSSIAN
    #Gaussian PDF from model prediction values and target values
    x = np.linspace(
        min(target_mean - 4*target_std, pred_mean1 - 4*pred_std1, pred_mean2 - 4*pred_std2),
        max(target_mean + 4*target_std, pred_mean1 + 4*pred_std1, pred_mean2 + 4*pred_std2),
        500
    )
    target_pdf = (1 / (target_std * np.sqrt(2 * np.pi))) * np.exp(
    -0.5 * ((x - target_mean) / target_std)**2
    )

    pred_pdf1 = (1 / (pred_std1 * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - pred_mean1) / pred_std1)**2
    )
    pred_pdf2 = (1 / (pred_std2 * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - pred_mean2) / pred_std2)**2
    )

    # Histogram of actual ensemble
    plt.hist(
        linear_values, #OPTION 1
        #log_values, #OPTION 2
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
        pred_pdf1,
        color="darkblue",
        linestyle=":",
        linewidth=2,
        label=f"Gaussian from {model_name1} predicted mean/std"
    )
    plt.plot(
        x,
        pred_pdf2,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Gaussian from {model_name2} predicted mean/std"
    )

    plt.xlabel("Value")
    plt.ylabel("Probability density")
    plt.title(f"Both Models: {var_name} distributions at timestep {ens_t}, {space}")
    plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig(f"both_models_{var_name}_{space}_distributions_timestep_{ens_t}")
    plt.close()
#################################

pred_ds1.close()
pred_ds2.close()
ens_ds.close()