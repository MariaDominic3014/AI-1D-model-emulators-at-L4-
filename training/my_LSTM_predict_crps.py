from netCDF4 import Dataset

import numpy as np
#from scipy.stats import skew, kurtosis as kurtosis_function

#import tensorflow as tf
import keras

from pathlib import Path

import time



#SECTION 1
def rollout(model, B_init, F, lookback, steps, bio_mean, bio_std, Y_mean, Y_std):
    ###
    # B_init: initial biogeochem history (lookback, n_bio)
    # F: full forcing time series
    # steps: number of steps to predict forward
    ###

    # Keep only the most recent lookback states
    B_hist = B_init.copy()
    preds = []

    start = time.time()

    for i in range(steps):

        if i % 100 == 0:
            elapsed = time.time() - start
            print(f"Rollout Step {i}/{steps}  |  elapsed: {elapsed:.1f}s")

        t = lookback + i

        bio_input = B_hist[None, ...]
        force_input = F[t-lookback:t][None, ...]
        force_future = F[t][None, ...]

        pred = model.predict([bio_input, force_input, force_future], verbose=0)[0]

        # -------------------------------------------------
        # pred is currently in Y-normalised target space - convert to physical first
        # -------------------------------------------------

        # Mean: Y-normalised -> physical
        pred_mean_physical = (
            (pred[0::2] * Y_std) + Y_mean
        )

        # CRPS loss applies softplus to predicted std. Apply the same transformation before reversing the target normalisation.
        # Std: raw network output -> positive normalised std
        pred_std_norm = (
            np.logaddexp(0.0, pred[1::2]) + 1e-6
        )

        # Std: Y-normalised -> physical
        pred_std_physical = pred_std_norm * Y_std

        # -------------------------------------------------
        # Convert physical prediction into BIO input space
        # -------------------------------------------------

        pred_mean_bio = (
            pred_mean_physical - bio_mean
        ) / bio_std

        pred_std_bio = (
            pred_std_physical / bio_std
        )

        # Construct the representation expected by B_hist
        pred_for_input = np.empty_like(pred)

        pred_for_input[0::2] = pred_mean_bio
        pred_for_input[1::2] = pred_std_bio

        # Store the ORIGINAL model prediction
        preds.append(pred)

        # Feed the correctly-transformed prediction back
        B_hist = np.vstack([
            B_hist[1:],
            pred_for_input
        ])

    return np.array(preds)



#SECTION 2
data_file = Path(__file__).parent.parent / "data" / "processed_ensemble.nc"
ds = Dataset(str(data_file))

variables_to_keep_original = ["P1_Chl", "P1_Chl_vert", "P2_Chl", "P2_Chl_vert", "Z6_c", "Z6_c_vert", "N1_p", "N3_n", "N4_n", "N5_s", "O2_bot", "O2_o", "P3_Chl", "P3_Chl_vert", "P4_Chl", "P4_Chl_vert", "Z5_c", "Z5_c_vert", "Z4_c", "Z4_c_vert"]
variables_to_log_transform = []
number_state_variables = len(variables_to_keep_original) + len(variables_to_log_transform)
state_variables_order = variables_to_keep_original + variables_to_log_transform

forcing_variables_names = ["heat", "light_parEIR_surf", "mld_surf", "precip", "salt", "temp", "u10", "v10"]
number_forcing_variables = len(forcing_variables_names)

start_time = 365 + 334   # account for spin up (nearly 2 years)
length_training_time = int(12*365.25)   # length of the time series for training data
length_prediction_time = ds.dimensions["time"].size - start_time   # length of the time series for predictions (includes whole of training and validation periods as well as the testing period)
lookback=30



#SECTION 3
def compute_ensemble_stats(state_var):
    # state_var shape: (time, ensemble_members)
    # returns: mean, std

    state_var = np.asarray(state_var, dtype=np.float64)

    mean = np.mean(state_var, axis=1)
    std = np.std(state_var, axis=1)

    return mean, std

original_stats = {}
for name in variables_to_keep_original:
    state_var = ds.variables[name][start_time:start_time + length_prediction_time]
    mean, std = compute_ensemble_stats(state_var)
    original_stats[name] = {
        "mean": mean,
        "std": std
    }

log_stats = {}
for name in variables_to_log_transform:
    state_var = ds.variables[name][start_time:start_time + length_prediction_time]
    log_state_var = np.log(state_var)
    mean, std = compute_ensemble_stats(log_state_var)
    log_stats[name] = {
        "mean": mean,
        "std": std
    }

combined_stats = {**original_stats, **log_stats}



#SECTION 4
pre_normalised_state_variables_stats = np.zeros((length_prediction_time, number_state_variables*2))  # 2 stats for each variable
col_index = 0
for var_name in combined_stats.keys():
    pre_normalised_state_variables_stats[:, col_index] = combined_stats[var_name]["mean"]
    pre_normalised_state_variables_stats[:, col_index+1] = combined_stats[var_name]["std"]
    col_index += 2

################# correct normalisation statistics (preprocessed stats) ###########
model_name = "crps_original"
preprocessing_path = Path(__file__).parent / model_name / f"{model_name}_preprocessing.npz"
preprocessing = np.load(preprocessing_path)

bio_mean = preprocessing["bio_mean"]
bio_std = preprocessing["bio_std"]
force_mean = preprocessing["force_mean"]
force_std = preprocessing["force_std"]

Y_mean = preprocessing["Y_mean"]
Y_std = preprocessing["Y_std"]
###################################################################################

normalised_state_variables_stats = np.zeros((length_prediction_time, number_state_variables*2))  # 2 stats for each variable
# Ensemble means: z-score using statistics from training ensemble means
normalised_state_variables_stats[:, 0::2] = (
    pre_normalised_state_variables_stats[:, 0::2] - bio_mean
) / bio_std

# Ensemble standard deviations: divide by the std of the corresponding mean
normalised_state_variables_stats[:, 1::2] = (
    pre_normalised_state_variables_stats[:, 1::2] / bio_std
)

forcing_variables = np.zeros((length_prediction_time, number_forcing_variables))
for index, var in enumerate(forcing_variables_names):
    forcing_values = ds.variables[var][
        start_time:start_time + length_prediction_time
    ]

    forcing_variables[:, index] = (
        forcing_values - force_mean[index]
    ) / force_std[index]

ds.close()

print("Bio input:", normalised_state_variables_stats[:lookback].shape)
print("Forcing:", forcing_variables.shape)
print("Prediction steps:", length_prediction_time - lookback)
print()



#SECTION 5
model_file = (
    Path(__file__).parent
    / model_name
    / f"{model_name}_best_model.keras"
)

model = keras.models.load_model(model_file, compile=False)

Y_pred_norm = rollout(model, normalised_state_variables_stats[:lookback], forcing_variables, lookback=30, steps=length_prediction_time-30, bio_mean=bio_mean, bio_std=bio_std, Y_mean=Y_mean, Y_std=Y_std)

print("\nROLLOUT PREDICTION COMPLETE")

####################### correct reverse-normalisation steps #######################
Y_pred = np.empty_like(Y_pred_norm)

# Predicted ensemble means
Y_pred[:, 0::2] = (Y_pred_norm[:, 0::2] * Y_std) + Y_mean

# The CRPS loss applies softplus to predicted std values.
# Apply the same transformation before converting back to physical units
predicted_std_norm = np.logaddexp(0.0, Y_pred_norm[:, 1::2]) + 1e-6

# Predicted ensemble standard deviations
Y_pred[:, 1::2] = predicted_std_norm * Y_std
###################################################################################

o = Dataset(f"{model_name}_predicted.nc", "w", format="NETCDF4_CLASSIC")

o.createDimension("time", length_prediction_time - lookback)

col_index = 0
for var in state_variables_order:

    #predicted vs test mean
    variable = o.createVariable("predicted_mean_"+var, np.float32, ("time",))
    variable[:] = Y_pred[:, col_index]
    variable = o.createVariable("test_mean_"+var, np.float32, ("time",))
    variable[:] = pre_normalised_state_variables_stats[lookback:, col_index]

    #predicted vs test std
    variable = o.createVariable("predicted_std_"+var, np.float32, ("time",))
    variable[:] = Y_pred[:, col_index+1]
    variable = o.createVariable("test_std_"+var, np.float32, ("time",))
    variable[:] = pre_normalised_state_variables_stats[lookback:, col_index+1]

    col_index += 2


o.close()