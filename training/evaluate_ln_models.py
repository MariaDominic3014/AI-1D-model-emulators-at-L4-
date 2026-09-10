from netCDF4 import Dataset
from pathlib import Path
import numpy as np
from collections import namedtuple
import pandas as pd
from scipy.special import erf
from scipy.stats import lognorm, kstest



# SECTION 1
#   "mse_ln" or "crps_ln"
MODEL_NAME = "crps_ln"

SCRIPT_DIR = Path(__file__).parent
PREDICTION_FILE = (SCRIPT_DIR / MODEL_NAME / f"{MODEL_NAME}_log_predicted.nc")
ENSEMBLE_FILE = (SCRIPT_DIR.parent / "data" / "processed_ensemble.nc")

VARIABLES = ["P1_Chl", "P1_Chl_vert", "P2_Chl", "P2_Chl_vert", "Z6_c", "Z6_c_vert", "N1_p", "N3_n", "N4_n", "N5_s", "O2_bot", "O2_o", "P3_Chl", "P3_Chl_vert", "P4_Chl", "P4_Chl_vert", "Z5_c", "Z5_c_vert", "Z4_c", "Z4_c_vert"]
START_TIME = 365 + 334
LOOKBACK = 30
TRAINING_PERIOD = 4383   #int(12*365.25)
VALIDATION_PERIOD = 1095   #int(3*365.25)
TEST_PERIOD = 2556   #int(7*365.25)

EPSILON = 1e-6



# SECTION 2
def crps_lognormal_vs_lognormal(pred_mu, pred_sigma, target_mu, target_sigma):
    # Calculate the distributional CRPS / energy distance between
    # predicted and target lognormal distributions.


    # --------------------------------------------------------------
    # Convert lognormal parameters to linear-space means
    #
    # If:
    #     log(X) ~ N(mu, sigma^2)
    #
    # then:
    #     E[X] = exp(mu + sigma^2 / 2)
    # --------------------------------------------------------------

    predicted_mean = np.exp(
        pred_mu + 0.5 * pred_sigma**2
    )

    target_mean = np.exp(
        target_mu + 0.5 * target_sigma**2
    )

    # --------------------------------------------------------------
    # E[|X - Y|]
    #
    # X ~ LogNormal(predicted_mu, predicted_sigma^2)
    # Y ~ LogNormal(target_mu, target_sigma^2)
    # --------------------------------------------------------------

    combined_sigma = np.sqrt(
        pred_sigma**2
        + target_sigma**2
        + 1e-8
    )

    a = (
        target_mu
        - pred_mu
        - pred_sigma**2
    ) / combined_sigma

    b = (
        pred_mu
        - target_mu
        - target_sigma**2
    ) / combined_sigma

    # Standard normal CDF
    Phi_a = 0.5 * (
        1.0 + erf(a / np.sqrt(2.0))
    )

    Phi_b = 0.5 * (
        1.0 + erf(b / np.sqrt(2.0))
    )

    expected_abs_xy = (
        predicted_mean
        + target_mean
        - 2.0 * (
            predicted_mean * Phi_a
            + target_mean * Phi_b
        )
    )

    # --------------------------------------------------------------
    # 0.5 * E[|X - X'|]
    #
    # X and X' are independent draws from the predicted
    # lognormal distribution.
    # --------------------------------------------------------------

    Phi_pred = 0.5 * (
        1.0 + erf(pred_sigma / 2.0)
    )

    expected_abs_xx_half = (
        predicted_mean
        * (
            2.0 * Phi_pred
            - 1.0
        )
    )

    # --------------------------------------------------------------
    # 0.5 * E[|Y - Y'|]
    #
    # Y and Y' are independent draws from the target
    # lognormal distribution.
    # --------------------------------------------------------------

    Phi_target = 0.5 * (
        1.0 + erf(target_sigma / 2.0)
    )

    expected_abs_yy_half = (
        target_mean
        * (
            2.0 * Phi_target
            - 1.0
        )
    )

    # --------------------------------------------------------------
    # Distributional CRPS / energy distance
    #
    # E[|X - Y|]
    # - 0.5 E[|X - X'|]
    # - 0.5 E[|Y - Y'|]
    # --------------------------------------------------------------

    crps = (
        expected_abs_xy
        - expected_abs_xx_half
        - expected_abs_yy_half
    )

    return crps


GoodnessOfFit = namedtuple("GoodnessOfFit", ["ks_statistic", "p_value"])

def _pit_goodness_of_fit(pit_values):
    result = kstest(pit_values, "uniform")
    return GoodnessOfFit(float(result.statistic), float(result.pvalue))

def lognormal_fit_score(values, mu, sigma):
    """
    Score how well the linear-space transform of N(mu, sigma) (i.e.
    X = exp(Z), Z ~ N(mu, sigma)) fits `values`.
    """
    values = np.asarray(values, dtype=np.float64)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if np.any(values <= 0):
        raise ValueError("log-normal fit requires strictly positive values")
 
    pit = lognorm.cdf(values, s=sigma, scale=np.exp(mu))
    return _pit_goodness_of_fit(pit)



def nanmean_or_nan(values):
    # Return the mean of finite values, or NaN if no finite values exist.

    values = np.asarray(values, dtype=np.float64)

    if not np.any(np.isfinite(values)):
        return np.nan

    return np.nanmean(values)



# SECTION 3
print("=" * 70)
print("PROBABILISTIC MODEL EVALUATION")
print("=" * 70)
print(f"Model: {MODEL_NAME}")
print()

prediction_ds = Dataset(str(PREDICTION_FILE), "r")
ensemble_ds = Dataset(str(ENSEMBLE_FILE), "r")

prediction_file_length = len(prediction_ds.dimensions["time"])
TRAINING_EVAL_LENGTH = TRAINING_PERIOD - LOOKBACK   #the first 30 days of the training period are before predictions begin
TEST_EVAL_START = TRAINING_EVAL_LENGTH + VALIDATION_PERIOD
TEST_EVAL_LENGTH = TEST_PERIOD



# SECTION 4
def evaluate_period(period_name, prediction_start, evaluation_length):
    results = []
    per_timestep_results = []

    print(f"Evaluating {period_name} period")
    print(f"Prediction start index: {prediction_start}")
    print(f"Evaluation timesteps: {evaluation_length}")
    print()
    
    for variable_name in VARIABLES:
        print(f"Evaluating {variable_name}...")

        prediction_end = prediction_start + evaluation_length

        predicted_mu = np.asarray(prediction_ds.variables["predicted_mean_" + variable_name][prediction_start:prediction_end], dtype=np.float64)
        predicted_sigma = np.asarray(prediction_ds.variables["predicted_std_" + variable_name][prediction_start:prediction_end], dtype=np.float64)
        #assume that any negative std are just really small - this is only for the mse model, because the crps model's std already have softplus applied to them
        if MODEL_NAME == "mse_ln":
            predicted_sigma = np.where(
                predicted_sigma <= 0,
                EPSILON,
                predicted_sigma
            )
        predicted_lognormal_mode = np.exp(predicted_mu - predicted_sigma**2) - np.exp(-8)   #reverse the offset used before the initial log transform

        target_mu = np.asarray(prediction_ds.variables["test_mean_" + variable_name][prediction_start:prediction_end], dtype=np.float64)
        target_sigma = np.asarray(prediction_ds.variables["test_std_" + variable_name][prediction_start:prediction_end], dtype=np.float64)
        #protect against zero target std
        target_sigma = np.maximum(target_sigma, EPSILON)
        target_lognormal_mode = np.exp(target_mu - target_sigma**2) - np.exp(-8)   #reverse the offset

        # Check that every prediction array matches the NetCDF time dimension.
        if not (
            len(predicted_mu)
            == len(predicted_sigma)
            == len(target_mu)
            == len(target_sigma)
            == evaluation_length
        ):
            raise ValueError(
                f"Prediction-array length mismatch for {variable_name}: "
                f"predicted_mu={len(predicted_mu)}, "
                f"predicted_sigma={len(predicted_sigma)}, "
                f"target_mu={len(target_mu)}, "
                f"target_sigma={len(target_sigma)}, "
                f"evaluation_length={evaluation_length}"
            )

        # --------------------------------------------------------
        # Load the actual ensemble corresponding to the prediction period.
        # Prediction starts at START_TIME + LOOKBACK because the first 30 timesteps are used to initialise the LSTM.
        # --------------------------------------------------------
        ensemble_variable = ensemble_ds.variables[variable_name]

        ensemble_start = START_TIME + LOOKBACK + prediction_start
        ensemble_end = ensemble_start + evaluation_length

        if ensemble_variable.shape[0] < ensemble_end:
            raise ValueError(
                f"Ensemble does not contain enough timesteps for {variable_name}: "
                f"available={ensemble_variable.shape[0]}, "
                f"required={ensemble_end}"
            )

        ensemble = np.asarray(ensemble_variable[ensemble_start:ensemble_end], dtype=np.float64,)



        # METRICS
        squared_errors = (predicted_lognormal_mode - target_lognormal_mode) ** 2
        bias_errors = predicted_lognormal_mode - target_lognormal_mode

        crps_target_values = np.full(evaluation_length, np.nan, dtype=np.float64)
        lognormal_predicted_ks = np.full(evaluation_length, np.nan, dtype=np.float64)
    
        for timestep_index in range(evaluation_length):
            # Predicted Lognormal vs Target Lognormal
            crps_target_values[timestep_index] = crps_lognormal_vs_lognormal(
                predicted_mu[timestep_index],
                predicted_sigma[timestep_index],
                target_mu[timestep_index],
                target_sigma[timestep_index],
            )

            # Predicted Lognormal GOF
            lognormal_predicted_ks[timestep_index] = lognormal_fit_score(
                ensemble[timestep_index],
                predicted_mu[timestep_index],
                predicted_sigma[timestep_index]
            ).ks_statistic


        mse = nanmean_or_nan(squared_errors)
        mbe = nanmean_or_nan(bias_errors)
        crps_target = nanmean_or_nan(crps_target_values)
        gof_result = nanmean_or_nan(lognormal_predicted_ks)


        # SAVE PER-VARIABLE RESULTS
        results.append({
            "variable": variable_name,
            "MSE": mse,
            "MBE": mbe,
            "CRPS_vs_target": crps_target,
            "KS_GOF": gof_result,
        })

        # Save individual timestep metrics
        for timestep_index in range(evaluation_length):
            per_timestep_results.append({
                "variable": variable_name,
                "timestep": timestep_index,
                "MSE": squared_errors[timestep_index],
                "MBE": bias_errors[timestep_index],
                "CRPS_vs_target": crps_target_values[timestep_index],
                "KS_GOF": lognormal_predicted_ks[timestep_index],
            })

        print(f"    MSE: {mse:.6g}")
        print(f"    MBE: {mbe:.6g}")
        print(f"    CRPS vs target: {crps_target:.6g}")
        print(f"    KS_GOF: {gof_result:.6g}")
        print()

    results_df = pd.DataFrame(results)
    per_timestep_df = pd.DataFrame(per_timestep_results)

    results_file = (
        SCRIPT_DIR / f"{MODEL_NAME}_{period_name}_evaluation_per_variable.csv"
    )
    timestep_file = (
        SCRIPT_DIR / f"{MODEL_NAME}_{period_name}_evaluation_per_timestep.csv"
    )

    results_df.to_csv(results_file, index=False)
    per_timestep_df.to_csv(timestep_file, index=False)

    print("=" * 70)
    print(f"{period_name.upper()} PERIOD RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print()

    return results_df, per_timestep_df



# SECTION 5
training_results_df, training_per_timestep_df = evaluate_period(
    period_name="training",
    prediction_start=0,
    evaluation_length=TRAINING_EVAL_LENGTH,
)

test_results_df, test_per_timestep_df = evaluate_period(
    period_name="test",
    prediction_start=TEST_EVAL_START,
    evaluation_length=TEST_EVAL_LENGTH,
)



prediction_ds.close()
ensemble_ds.close()