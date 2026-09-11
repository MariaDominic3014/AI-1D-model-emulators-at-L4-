from netCDF4 import Dataset
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt

ds = Dataset("..\\data\\processed_ensemble.nc")



start_time = 365 + 334  # account for spin up (nearly 2 years)
length_time = int(15*365.25)  # length of the time series for training and validation data

# Set to True if you want a plot for every variable
plot_results = False

# State variables
p_variables = ["P1_Chl", "P1_Chl_vert", "P2_Chl", "P2_Chl_vert", "P3_Chl", "P3_Chl_vert", "P4_Chl", "P4_Chl_vert"]
p_test_variables = ["P1_Chl_vert", "P4_Chl_vert"]
n_variables = ["N1_p", "N3_n", "N4_n", "N5_s"]
o_variables = ["O2_bot", "O2_o"]
z_variables = ["Z4_c", "Z4_c_vert", "Z5_c", "Z5_c_vert", "Z6_c", "Z6_c_vert"]


# ==================================================
# Loop through variables in chosen list
# ==================================================

for var_name in z_variables:

    print("\n" + "=" * 50)
    print(var_name)
    print("=" * 50)

    # --------------------------------------------------
    # Get the data for this variable
    # --------------------------------------------------

    state_var = ds.variables[var_name][
        start_time:start_time + length_time
    ]

    # --------------------------------------------------
    # 1. Skewness of original data
    # --------------------------------------------------

    skew_original = skew(
        state_var,
        axis=1,
        nan_policy='omit'
    )

    # --------------------------------------------------
    # 2. Log-transform
    # --------------------------------------------------

    log_state_var = np.log10(state_var)

    # --------------------------------------------------
    # 3. Skewness of log-transformed data
    # --------------------------------------------------

    skew_log = skew(
        log_state_var,
        axis=1,
        nan_policy='omit'
    )

    # --------------------------------------------------
    # 4. Absolute skewness
    # --------------------------------------------------

    abs_skew_original = np.abs(skew_original)
    abs_skew_log = np.abs(skew_log)

    # --------------------------------------------------
    # 5. Mean absolute skewness
    # --------------------------------------------------

    mean_abs_skew_original = np.nanmean(
        abs_skew_original
    )

    mean_abs_skew_log = np.nanmean(
        abs_skew_log
    )

    # --------------------------------------------------
    # 6. Percentage of timesteps where log improves
    #    absolute skewness
    # --------------------------------------------------

    log_improves = (
        abs_skew_log < abs_skew_original
    )

    percentage_improved = (
        np.mean(log_improves) * 100
    )

    # --------------------------------------------------
    # 7. Difference in absolute skewness
    # --------------------------------------------------

    skew_difference = (
        abs_skew_log - abs_skew_original
    )

    mean_change = np.nanmean(skew_difference)
    minimum_change = np.nanmin(skew_difference)
    maximum_change = np.nanmax(skew_difference)

    # --------------------------------------------------
    # 8. Print results
    # --------------------------------------------------

    print(
        f"Mean |skewness| - original: "
        f"{mean_abs_skew_original:.4f}"
    )

    print(
        f"Mean |skewness| - log: "
        f"{mean_abs_skew_log:.4f}"
    )

    print(
        f"Log improves |skewness| at: "
        f"{percentage_improved:.1f}% of timesteps"
    )

    print(
        f"Mean change in |skewness|: "
        f"{mean_change:.4f}"
    )

    print(
        f"Minimum change: "
        f"{minimum_change:.4f}"
    )

    print(
        f"Maximum change: "
        f"{maximum_change:.4f}"
    )

    # --------------------------------------------------
    # 9. Optional plot
    # --------------------------------------------------

    if plot_results:

        plt.figure(figsize=(10, 4))

        plt.plot(
            abs_skew_original,
            label="Original"
        )

        plt.plot(
            abs_skew_log,
            label="Log-transformed"
        )

        plt.xlabel("Timestep")
        plt.ylabel("|Skewness|")
        plt.title(
            f"Absolute skewness: {var_name}"
        )

        plt.legend()
        plt.tight_layout()
        plt.show()