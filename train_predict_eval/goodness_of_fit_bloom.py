from pathlib import Path
import pandas as pd
import csv


VARIABLES = ["P1_Chl", "P1_Chl_vert", "P2_Chl", "P2_Chl_vert", "Z6_c", "Z6_c_vert", "N1_p", "N3_n", "N4_n", "N5_s", "O2_bot", "O2_o", "P3_Chl", "P3_Chl_vert", "P4_Chl", "P4_Chl_vert", "Z5_c", "Z5_c_vert", "Z4_c", "Z4_c_vert"]

#TRAINING PERIOD
# CSV_FILES = {
#     "mse_original": Path(__file__).parent / "mse_original" / "evaluation" / "mse_original_training_evaluation_per_timestep.csv",
#     "crps_original": Path(__file__).parent / "crps_original" / "evaluation" / "crps_original_training_evaluation_per_timestep.csv",
#     "mse_ln": Path(__file__).parent / "mse_ln" / "evaluation" / "mse_ln_training_evaluation_per_timestep.csv",
#     "crps_ln": Path(__file__).parent / "crps_ln" / "evaluation" / "crps_ln_training_evaluation_per_timestep.csv",
# }

#TEST PERIOD
CSV_FILES = {
    "mse_original": Path(__file__).parent / "mse_original" / "evaluation" / "mse_original_test_evaluation_per_timestep.csv",
    "crps_original": Path(__file__).parent / "crps_original" / "evaluation" / "crps_original_test_evaluation_per_timestep.csv",
    "mse_ln": Path(__file__).parent / "mse_ln" / "evaluation" / "mse_ln_test_evaluation_per_timestep.csv",
    "crps_ln": Path(__file__).parent / "crps_ln" / "evaluation" / "crps_ln_test_evaluation_per_timestep.csv",
}


def calculate_summer_ks_gof(input_file, output_file):
    data = pd.read_csv(input_file)

    required_columns = {
        "variable",
        "timestep",
        "KS_GOF",
    }

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"{input_file} is missing columns: {sorted(missing_columns)}"
        )

    data["timestep"] = pd.to_numeric(data["timestep"], errors="coerce")
    data["KS_GOF"] = pd.to_numeric(data["KS_GOF"], errors="coerce")

    data = data.dropna(subset=["variable", "timestep", "KS_GOF"])

    # Position within each 365-timestep year, starting from timestep 0.
    data["day_of_year"] = data["timestep"] % 365

    # Keep timesteps 60 through 243, inclusive, for every year.
    summer_data = data[
        (data["day_of_year"] >= 60)
        & (data["day_of_year"] <= 243)
        & (data["variable"].isin(VARIABLES))
    ]

    result = (
        summer_data
        .groupby("variable", as_index=False)["KS_GOF"]
        .mean()
        .rename(columns={"KS_GOF": "summer_KS_GOF"})
    )

    result["variable"] = pd.Categorical(
        result["variable"],
        categories=VARIABLES,
        ordered=True,
    )
    result = result.sort_values("variable")
    result["variable"] = result["variable"].astype(str)

    result.to_csv(output_file, index=False)


for model_name, input_file in CSV_FILES.items():
    #output_file = input_file.parent / f"training_period_summer_KS_GOF_{model_name}.csv"
    output_file = input_file.parent / f"test_period_summer_KS_GOF_{model_name}.csv"
    calculate_summer_ks_gof(input_file, output_file)

    print(f"Saved: {output_file}")