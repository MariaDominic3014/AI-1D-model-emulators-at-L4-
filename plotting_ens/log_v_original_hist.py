from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from datetime import datetime

ds = Dataset("..\\data\\processed_ensemble.nc")



start_time = 365 + 334
length_time = int(15 * 365.25)

# Variables
var_names = [
    # Log-transform seems better
    "P3_Chl",
    "P3_Chl_vert",
    "P4_Chl",
    "P4_Chl_vert",
    "Z5_c",
    "Z5_c_vert",
    "Z4_c",
    "Z4_c_vert",

    # Original seems better
    "P1_Chl",
    "P1_Chl_vert",
    "P2_Chl",
    "P2_Chl_vert",
    "Z6_c",
    "Z6_c_vert",

    # Ambiguous
    "N1_p",
    "N3_n",
    "N4_n",
    "N5_s",
    "O2_bot",
    "O2_o"
]

# ==================================================
# Define the start of the 15-year training period
# ==================================================

training_start = datetime(2004, 1, 1)

# Choose representative dates in the middle of the 15-year training period
season_dates = {
    "Spring": datetime(2011, 3, 1), #March 1, 2011
    "Summer": datetime(2011, 6, 1), #June 1, 2011
    "Autumn": datetime(2011, 9, 1), #September 1, 2011
    "Winter": datetime(2011, 12, 1) #December 1, 2011
}

# Calculate their positions within the 15-year training period
season_positions = {
    season: (date - training_start).days
    for season, date in season_dates.items()
}

# Print the positions so you can check them
print("Selected seasonal positions:")

for season, position in season_positions.items():
    print(
        f"{season}: {season_dates[season].strftime('%d/%m/%Y')} "
        f"-> timestep {position}"
    )


# ==================================================
# Loop through variables
# ==================================================

for var_name in var_names:

    state_var = ds.variables[var_name][
        start_time:start_time + length_time
    ]

    print(f"\nPlotting {var_name}...")

    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(13, 16),
        constrained_layout=True
    )

    # --------------------------------------------------
    # Loop through the four seasons
    # --------------------------------------------------

    for i, (season, position) in enumerate(
        season_positions.items()
    ):

        # Get the 1100 ensemble members at this timestep
        values = state_var[position, :]

        # Log-transform
        log_values = np.log10(values)

        # Calculate skewness
        original_skew = skew(
            values,
            nan_policy="omit"
        )

        log_skew = skew(
            log_values,
            nan_policy="omit"
        )

        # ----------------------------------------------
        # Original histogram
        # ----------------------------------------------

        axes[i, 0].hist(
            values,
            bins=100,
            density=True
        )

        axes[i, 0].set_title(
            f"{season} - Original\n"
            f"{season_dates[season].strftime('%d/%m/%Y')} | "
            f"skewness = {original_skew:.2f}",
            pad=12
        )

        axes[i, 0].set_xlabel(var_name)
        axes[i, 0].set_ylabel("Density")

        # ----------------------------------------------
        # Log-transformed histogram
        # ----------------------------------------------

        axes[i, 1].hist(
            log_values,
            bins=100,
            density=True
        )

        axes[i, 1].set_title(
            f"{season} - Log₁₀\n"
            f"{season_dates[season].strftime('%d/%m/%Y')} | "
            f"skewness = {log_skew:.2f}",
            pad=12
        )

        axes[i, 1].set_xlabel(f"log₁₀({var_name})")
        axes[i, 1].set_ylabel("Density")

    # --------------------------------------------------
    # Overall title
    # --------------------------------------------------

    fig.suptitle(
        f"{var_name}: Original vs Log-transformed",
        fontsize=16
    )

    plt.show()