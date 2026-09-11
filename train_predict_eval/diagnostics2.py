from netCDF4 import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, lognorm



#PREDICTED VALUES IN LOG SPACE
#model_name = "mse_ln"
model_name = "crps_ln"
prediction_file = (
    Path(__file__).parent
    / model_name
    / f"{model_name}_log_predicted.nc"
)
space = "log-space"


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
print(f"{space} prediction file")

def histogram_mode(values, bins=100):
    values = np.asarray(values, dtype=np.float64)

    counts, edges = np.histogram(values, bins=bins)
    modal_bin = np.argmax(counts)
    return (edges[modal_bin] + edges[modal_bin + 1]) / 2

for var_name in state_variables_names:
    var = ens_ds.variables[var_name][start_time + lookback:]

    #original/linear-space values
    # ens_mean = np.mean(var, axis=1)
    # ens_std = np.std(var, axis=1)

    #ln values
    # log_var = np.log(var + np.exp(-8))
    # ens_mean_ln = np.mean(log_var, axis=1)
    # ens_std_ln = np.std(log_var, axis=1)

    empirical_mode = np.array([
        histogram_mode(values, bins=100)
        for values in var
    ])

    target_mean = pred_ds.variables[f"test_mean_{var_name}"][:]
    target_std = pred_ds.variables[f"test_std_{var_name}"][:]

    pred_mean = pred_ds.variables[f"predicted_mean_{var_name}"][:]
    pred_std = pred_ds.variables[f"predicted_std_{var_name}"][:]

    #Lognormal mode from log mean and std
    #mode_E = np.exp(ens_mean_ln - ens_std_ln**2) - np.exp(-8)
    mode_T = np.exp(target_mean - target_std**2) - np.exp(-8)   #minus the offset
    mode_P = np.exp(pred_mean - pred_std**2) - np.exp(-8)       #minus the offset

    #Lognormal bounds
    # upper_E = np.exp(ens_mean_ln - ens_std_ln**2 + ens_std_ln) - np.exp(-8)
    # lower_E = np.exp(ens_mean_ln - ens_std_ln**2 - ens_std_ln) - np.exp(-8)

    upper_T = np.exp(target_mean - target_std**2 + target_std) - np.exp(-8)
    lower_T = np.exp(target_mean - target_std**2 - target_std) - np.exp(-8)

    upper_P = np.exp(pred_mean - pred_std**2 + pred_std) - np.exp(-8)
    lower_P = np.exp(pred_mean - pred_std**2 - pred_std) - np.exp(-8)

    plt.figure(figsize=(12,5))

    #uncertainty
    target_colour = "tab:purple"
    prediction_colour = "tab:green"

    #all 3 modes
    # empirical_colour = "black"
    # target_colour = "tab:blue"
    # prediction_colour = "tab:orange"

    #mode
    # target_colour = "tab:brown"
    # prediction_colour = "tab:pink"


    #plt.plot(empirical_mode, color=empirical_colour, label="Empirical mode")

    plt.plot(mode_T, color=target_colour, label="Target mode")
    plt.fill_between(
        np.arange(len(mode_T)),
        upper_T,
        lower_T,
        color=target_colour,
        alpha=0.5,
        label="target uncertainty"
    )

    plt.plot(mode_P, color=prediction_colour, label="Predicted mode")
    plt.fill_between(
        np.arange(len(mode_P)),
        upper_P,
        lower_P,
        color=prediction_colour,
        alpha=0.3,
        label="predicted uncertainty"
    )

    plt.xlabel("Time (days)")
    plt.ylabel(var_name)
    #plt.title(f"{model_name} {var_name}: Empirical, Target & Predicted Mode")
    plt.title(f"{model_name} {var_name}: Target & Predicted Mode with uncertainty, lognormal")
    #plt.title(f"{model_name} {var_name}: Target & Predicted Mode, lognormal")
    plt.legend()

    plt.tight_layout()
    #plt.show()
    #plt.savefig(Path(__file__).parent / f"{model_name}_{var_name}_EmpTargetPred_Mode_timeseries.png")
    plt.savefig(Path(__file__).parent / f"{model_name}_{var_name}_targetAndPred_Mode_with_uncertainty.png")
    #plt.savefig(Path(__file__).parent / f"{model_name}_{var_name}_targetAndPred_Mode_lognormal.png")

    plt.close()
################################


#################################
# ENSEMBLE HISTOGRAM & PREDICTED DISTRIBUTION IN LOG-SPACE (NORMAL) AND LINEAR-SPACE (LOGNORMAL)
pred_t = (length_prediction_time - lookback) - 1
ens_t = (start_time + length_prediction_time) - 1

for var_name in state_variables_names:
    ensemble_values = ens_ds.variables[var_name][ens_t, :]

    # linear ensemble values at chosen timestep
    X = ensemble_values
    # log-transformed ensemble values at chosen timestep
    Z = np.log(ensemble_values + np.exp(-8))


    # #target mean + std at chosen timestep from prediction file
    # target_mean = pred_ds.variables[f"test_mean_{var_name}"][pred_t]
    # target_std = pred_ds.variables[f"test_std_{var_name}"][pred_t]

    #calculated ensemble mean and std
    ens_mean = np.mean(Z)
    ens_std = np.std(Z, ddof=1)

    #predicted mean + std at chosen timestep from prediction file
    pred_mean = pred_ds.variables[f"predicted_mean_{var_name}"][pred_t]
    pred_std = pred_ds.variables[f"predicted_std_{var_name}"][pred_t]


    # # Linear-space mean, std, median, mode (of target distribution)
    # mean_T = np.exp(target_mean + target_std**2 / 2) - np.exp(-8)
    # std_T = np.sqrt(
    #     (np.exp(target_std**2) - 1)
    #     * np.exp(2 * target_mean + target_std**2)
    # )
    # median_T = np.exp(target_mean) - np.exp(-8)
    # mode_T = np.exp(target_mean - target_std**2) - np.exp(-8)

    # print(var_name)
    # print("LOG SPACE")
    # print(f"Mean (μ_T): {target_mean:.3f}")
    # print(f"Std  (σ_T): {target_std:.3f}")

    # print("\nLINEAR SPACE")
    # print(f"Mean:   {mean_T:.3f}")
    # print(f"Std:    {std_T:.3f}")
    # print(f"Median: {median_T:.3f}")
    # print(f"Mode:   {mode_T:.3f}")
    # print()


    # Linear-space mean, std, median, mode (of offset-ens distribution)
    mean_X = np.exp(ens_mean + ens_std**2 / 2) - np.exp(-8)
    std_X = np.sqrt(
        (np.exp(ens_std**2) - 1)
        * np.exp(2 * ens_mean + ens_std**2)
    )
    median_X = np.exp(ens_mean) - np.exp(-8)
    mode_X = np.exp(ens_mean - ens_std**2) - np.exp(-8)
    
    print(var_name)
    print("LOG SPACE")
    print(f"Mean (μ_Z): {ens_mean:.3f}")
    print(f"Std  (σ_Z): {ens_std:.3f}")

    print("\nLINEAR SPACE")
    print(f"Mean:   {mean_X:.3f}")
    print(f"Std:    {std_X:.3f}")
    print(f"Median: {median_X:.3f}")
    print(f"Mode:   {mode_X:.3f}")
    print()


    # Linear-space mean, std, median, mode (of pred distribution)
    mean_P = np.exp(pred_mean + pred_std**2 / 2) - np.exp(-8)
    std_P = np.sqrt(
        (np.exp(pred_std**2) - 1)
        * np.exp(2 * pred_mean + pred_std**2)
    )
    median_P = np.exp(pred_mean) - np.exp(-8)
    mode_P = np.exp(pred_mean - pred_std**2) - np.exp(-8)

    print(var_name)
    print("LOG SPACE")
    print(f"Mean (μ_P): {pred_mean:.3f}")
    print(f"Std  (σ_P): {pred_std:.3f}")

    print("\nLINEAR SPACE")
    print(f"Mean:   {mean_P:.3f}")
    print(f"Std:    {std_P:.3f}")
    print(f"Median: {median_P:.3f}")
    print(f"Mode:   {mode_P:.3f}")
    print("="*20)
    print()


    # =======================================================
    # PLOTS
    # =======================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # -------------------------------------------------------
    # LEFT: LOG SPACE
    # -------------------------------------------------------
    z = np.linspace(
        min(Z.min(), ens_mean - 4*ens_std),
        max(Z.max(), ens_mean + 4*ens_std),
        1000
    )

    # pdf_T = norm.pdf(
    #     z,
    #     loc=target_mean,
    #     scale=target_std
    # )

    pdf_Z = norm.pdf(
        z,
        loc=ens_mean,
        scale=ens_std
    )

    pdf_P = norm.pdf(
        z,
        loc=pred_mean,
        scale=pred_std
    )

    axes[0].hist(
        Z,
        bins=50,
        density=True,
        alpha=0.5,
        color="cornflowerblue",
        edgecolor="royalblue",
        linewidth=0.8,
        label="log(X)"
    )

    # axes[0].plot(
    #     z,
    #     pdf_T,
    #     "g-",
    #     linewidth=2,
    #     label="Normal from prediction target"
    # )

    axes[0].plot(
        z,
        pdf_Z,
        "g-",
        linewidth=2,
        label="Normal from offset ensemble"
    )

    axes[0].plot(
        z,
        pdf_P,
        "r-",
        linewidth=2,
        label="Normal from prediction"
    )

    axes[0].axvline(
        pred_mean,
        color="red",
        linestyle="--",
        label=f"pred Mean = {pred_mean:.3f}"
    )

    axes[0].set_xlabel("Z = log(X)")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title(f"{var_name} [timestep {ens_t}] - Log space")
    axes[0].legend()


    # -------------------------------------------------------
    # RIGHT: LINEAR SPACE
    # -------------------------------------------------------
    x = np.linspace(
        X.min(),
        X.max(),
        1000
    )

    pdf_P_lognormal = lognorm.pdf(
        x,
        s=pred_std,
        scale=np.exp(pred_mean)
    )

    axes[1].hist(
        X,
        bins=50,
        density=True,
        alpha=0.5,
        color="cornflowerblue",
        edgecolor="royalblue",
        linewidth=0.8,
        label="X"
    )

    axes[1].plot(
        x,
        pdf_P_lognormal,
        "r-",
        linewidth=2,
        label="pred Log-normal fit"
    )

    axes[1].axvline(
        mean_P,
        color="red",
        linestyle="--",
        label=f"pred Mean = {mean_P:.3f}"
    )

    axes[1].axvline(
        median_P,
        color="black",
        linestyle=":",
        label=f"pred Median = {median_P:.3f}"
    )

    axes[1].set_xlabel("X (linear space)")
    axes[1].set_ylabel("Probability density")
    axes[1].set_title(f"{var_name} [timestep {ens_t}] - Linear space")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
################################



pred_ds.close()
ens_ds.close()