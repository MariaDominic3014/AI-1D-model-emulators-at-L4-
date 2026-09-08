from netCDF4 import Dataset
from pathlib import Path
import numpy as np
from collections import namedtuple
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm, kstest



# SECTION 1
#   "mse_original" or "crps_original"
MODEL_NAME = "mse_original"

SCRIPT_DIR = Path(__file__).parent
PREDICTION_FILE = (SCRIPT_DIR / MODEL_NAME / f"{MODEL_NAME}_predicted.nc")
ENSEMBLE_FILE = (SCRIPT_DIR.parent / "data" / "processed_ensemble.nc")

VARIABLES = ["P1_Chl", "P1_Chl_vert", "P2_Chl", "P2_Chl_vert", "Z6_c", "Z6_c_vert", "N1_p", "N3_n", "N4_n", "N5_s", "O2_bot", "O2_o", "P3_Chl", "P3_Chl_vert", "P4_Chl", "P4_Chl_vert", "Z5_c", "Z5_c_vert", "Z4_c", "Z4_c_vert"]
START_TIME = 365 + 334
LOOKBACK = 30
TRAINING_PERIOD = int(12*365.25)

EPSILON = 1e-6


# SECTION 2 - EVALUATION FUNCTIONS
def crps_gaussian_vs_gaussian(pred_mu, pred_sigma, target_mu, target_sigma):
    # CRPS between two Gaussian distributions is the integrated squared difference between their CDFs:
    #    integral [F(x) - G(x)]^2 dx
    # The equivalent closed-form expression is used here.


    # E|X - Y|
    delta = np.sqrt(pred_sigma**2 + target_sigma**2)
    d = (pred_mu - target_mu) / delta

    phi_d = np.exp(-0.5 * d**2) / np.sqrt(2.0 * np.pi)
    Phi_d = ndtr(d)

    expected_abs_difference = (
        delta * (
            2.0 * phi_d
            + d * (2.0 * Phi_d - 1.0)
        )
    )

    # E|X-X'| where X and X' ~ N(pred_mu, pred_sigma^2)
    expected_xx = 2.0 * pred_sigma / np.sqrt(np.pi)

    # E|Y-Y'| where Y and Y' ~ N(target_mu, target_sigma^2)
    expected_yy = 2.0 * target_sigma / np.sqrt(np.pi)

    return (
        expected_abs_difference
        - 0.5 * expected_xx
        - 0.5 * expected_yy
    )


GoodnessOfFit = namedtuple("GoodnessOfFit", ["ks_statistic", "p_value"])

def _pit_goodness_of_fit(pit_values):
    result = kstest(pit_values, "uniform")
    return GoodnessOfFit(float(result.statistic), float(result.pvalue))

def normal_fit_score(values, mean, std):
    """
    Score how well N(mean, std) (linear space) fits `values`.
    """
    values = np.asarray(values, dtype=np.float64)
    if std <= 0:
        raise ValueError("std must be positive")
 
    pit = norm.cdf(values, loc=mean, scale=std)
    return _pit_goodness_of_fit(pit)



def nanmean_or_nan(values):
    # Return the mean of finite values, or NaN if no finite values exist.

    values = np.asarray(values, dtype=np.float64)

    if not np.any(np.isfinite(values)):
        return np.nan

    return np.nanmean(values)



# SECTION 3 — LOAD PREDICTIONS
print("=" * 70)
print("PROBABILISTIC MODEL EVALUATION")
print("=" * 70)
print(f"Model: {MODEL_NAME}")
print(f"Space: LINEAR")
print()

prediction_ds = Dataset(str(PREDICTION_FILE), "r")
ensemble_ds = Dataset(str(ENSEMBLE_FILE), "r")

prediction_file_length = len(prediction_ds.dimensions["time"])
evaluation_length = TRAINING_PERIOD

print(f"Prediction-file timesteps: {prediction_file_length}")
print(f"Evaluation timesteps: {evaluation_length}")
print()



# SECTION 4 — STORAGE FOR RESULTS
results = []
per_timestep_results = []



# SECTION 5 — EVALUATE EACH VARIABLE
for variable_name in VARIABLES:
    print(f"Evaluating {variable_name}...")

    predicted_mean = np.asarray(prediction_ds.variables["predicted_mean_" + variable_name][:evaluation_length], dtype=np.float64)
    predicted_std = np.asarray(prediction_ds.variables["predicted_std_" + variable_name][:evaluation_length], dtype=np.float64)
    #for crps and gof metrics, assume that any negative std are just really small - this is only for the mse model, because the crps model's std already have softplus applied to them
    if MODEL_NAME == "mse_original":
        predicted_std = np.where(
            predicted_std <= 0,
            EPSILON,
            predicted_std
        )

    target_mean = np.asarray(prediction_ds.variables["test_mean_" + variable_name][:evaluation_length], dtype=np.float64)
    target_std = np.asarray(prediction_ds.variables["test_std_" + variable_name][:evaluation_length], dtype=np.float64)
    #for crps and gof metrics, protect against zero target std
    target_std = np.maximum(target_std, EPSILON)

    # Check that every prediction array matches the NetCDF time dimension.
    if not (
        len(predicted_mean)
        == len(predicted_std)
        == len(target_mean)
        == len(target_std)
        == evaluation_length
    ):
        raise ValueError(
            f"Prediction-array length mismatch for {variable_name}: "
            f"predicted_mean={len(predicted_mean)}, "
            f"predicted_std={len(predicted_std)}, "
            f"target_mean={len(target_mean)}, "
            f"target_std={len(target_std)}, "
            f"evaluation_length={evaluation_length}"
        )

    # --------------------------------------------------------
    # Load the actual ensemble corresponding to the prediction period.
    # Prediction starts at START_TIME + LOOKBACK because the first 30 timesteps are used to initialise the LSTM.
    # --------------------------------------------------------
    ensemble_variable = ensemble_ds.variables[variable_name]

    if ensemble_variable.shape[0] < (
        START_TIME + LOOKBACK + evaluation_length
    ):
        raise ValueError(
            f"Ensemble does not contain enough timesteps for {variable_name}: "
            f"available={ensemble_variable.shape[0]}, "
            f"required={START_TIME + LOOKBACK + evaluation_length}"
        )

    ensemble = np.asarray(
        ensemble_variable[
            START_TIME + LOOKBACK:
            START_TIME + LOOKBACK + evaluation_length
        ],
        dtype=np.float64,
    )

    if ensemble.shape[0] != evaluation_length:
        raise ValueError(
            f"Ensemble/prediction length mismatch for {variable_name}: "
            f"{ensemble.shape[0]} vs {evaluation_length}"
        )


    # MEAN ERROR
    squared_errors = np.full(evaluation_length, np.nan, dtype=np.float64)
    bias_errors = np.full(evaluation_length, np.nan, dtype=np.float64)

    residuals = predicted_mean - target_mean

    squared_errors = residuals ** 2
    bias_errors = residuals

    # ========================================================
    # MSE
    mse = nanmean_or_nan(squared_errors)

    # ========================================================
    # MBE
    mbe = nanmean_or_nan(bias_errors)

    # ========================================================
    # CRPS
    crps_target_values = np.full(evaluation_length, np.nan, dtype=np.float64)

    # ========================================================
    # GOF
    normal_predicted_ks = np.full(evaluation_length, np.nan, dtype=np.float64)
   

    for timestep_index in range(evaluation_length):
        # Predicted Gaussian vs Target Gaussian
        crps_target_value = crps_gaussian_vs_gaussian(
            predicted_mean[timestep_index],
            predicted_std[timestep_index],
            target_mean[timestep_index],
            target_std[timestep_index],
        )
        crps_target_values[timestep_index] = crps_target_value


        # Predicted Gaussian GOF
        gof_result_values = normal_fit_score(
            ensemble[timestep_index],
            predicted_mean[timestep_index],
            predicted_std[timestep_index]
        )
        normal_predicted_ks[timestep_index] = gof_result_values.ks_statistic


    crps_target = nanmean_or_nan(crps_target_values)
    gof_result = nanmean_or_nan(normal_predicted_ks)

    # ========================================================
    # SAVE PER-VARIABLE RESULTS
    # ========================================================
    results.append({
        "variable": variable_name,
        "MSE": mse,
        "MBE": mbe,
        "CRPS_vs_target": crps_target,
        "GOF_KS": gof_result,
    })

    # --------------------------------------------------------
    # Save individual timestep metrics
    # --------------------------------------------------------
    for timestep_index in range(evaluation_length):
        per_timestep_results.append({
            "variable": variable_name,
            "timestep": timestep_index,
            "MSE": squared_errors[timestep_index],
            "MBE": bias_errors[timestep_index],
            "CRPS_vs_target": crps_target_values[timestep_index],
            "GOF_KS": normal_predicted_ks[timestep_index],
        })

    print(f"    MSE: {mse:.6g}")
    print(f"    MBE: {mbe:.6g}")
    print(f"    CRPS vs target: {crps_target:.6g}")
    print(f"    GOF_KS: {gof_result:.6g}")
    print()



# SECTION 6 — SAVE RESULTS
results_df = pd.DataFrame(results)
per_timestep_df = pd.DataFrame(per_timestep_results)

results_file = (SCRIPT_DIR / f"{MODEL_NAME}_evaluation_per_variable.csv")
timestep_file = (SCRIPT_DIR / f"{MODEL_NAME}_evaluation_per_timestep.csv")

results_df.to_csv(results_file, index=False)
per_timestep_df.to_csv(timestep_file, index=False)



# SECTION 7 — PRINT SUMMARY
print("=" * 70)
print("PER-VARIABLE RESULTS")
print("=" * 70)
print(results_df.to_string(index=False))
print()



prediction_ds.close()
ensemble_ds.close()