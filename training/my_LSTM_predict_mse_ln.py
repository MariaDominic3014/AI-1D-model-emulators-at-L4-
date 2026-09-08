from netCDF4 import Dataset

import numpy as np
#from scipy.stats import skew, kurtosis as kurtosis_function

#import tensorflow as tf
import keras

from pathlib import Path

import time


#OPTION 1 - save the model's predicted values and the target values in log-space
#OPTION 2 - convert the model's predicted values back into linear-space before saving


#SECTION 1
MODEL_NAME = "mse_ln"

def rollout(model, B_init, F, lookback, steps, bio_mean, bio_std, Y_mean, Y_std):
    ###
    # B_init: initial biogeochem history (lookback, n_bio)
    # F: full forcing time series
    # steps: number of steps to predict forward
    # *_mean, *_std: preprocessed stats from training script normalisation
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
        # pred is currently in Y-normalised target space - reverse-normalise, then normalise again into bio space
        # -------------------------------------------------

        pred_raw = (
            (pred * Y_std) + Y_mean
        )

        pred_bio = (
            pred_raw - bio_mean
        ) / bio_std

        # Store the ORIGINAL model prediction
        preds.append(pred)

        # Feed the correctly-transformed prediction back
        B_hist = np.vstack([
            B_hist[1:],
            pred_bio
        ])

    return np.array(preds)



#SECTION 2
data_file = Path(__file__).parent.parent / "data" / "processed_ensemble.nc"
ds = Dataset(str(data_file))

variables_to_keep_original = []
variables_to_log_transform = ["P1_Chl", "P1_Chl_vert", "P2_Chl", "P2_Chl_vert", "Z6_c", "Z6_c_vert", "N1_p", "N3_n", "N4_n", "N5_s", "O2_bot", "O2_o", "P3_Chl", "P3_Chl_vert", "P4_Chl", "P4_Chl_vert", "Z5_c", "Z5_c_vert", "Z4_c", "Z4_c_vert"]
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
    log_state_var = np.log(state_var + np.exp(-8))   # add a small offset before logging
    mean, std = compute_ensemble_stats(log_state_var)
    log_stats[name] = {
        "mean": mean,
        "std": std
    }

combined_stats = {**original_stats, **log_stats}

# include for OPTION 2 #################
not_logged_stats = {}
for name in state_variables_order:
    state_var = ds.variables[name][start_time:start_time + length_prediction_time]
    mean, std = compute_ensemble_stats(state_var)
    not_logged_stats[name] = {
        "mean": mean,
        "std": std
    }
#################



#SECTION 4
# include for OPTION 2 #################
not_logged_state_variables_stats = np.zeros((length_prediction_time, number_state_variables*2))
#this is for the physical values
col_index = 0
for var_name in not_logged_stats.keys():
    not_logged_state_variables_stats[:, col_index] = not_logged_stats[var_name]["mean"]
    not_logged_state_variables_stats[:, col_index+1] = not_logged_stats[var_name]["std"]
    col_index += 2
#################

pre_normalised_state_variables_stats = np.zeros((length_prediction_time, number_state_variables*2))  # 2 stats for each variable
# This is for pre-normalised test values
col_index = 0
for var_name in combined_stats.keys():
    pre_normalised_state_variables_stats[:, col_index] = combined_stats[var_name]["mean"]
    pre_normalised_state_variables_stats[:, col_index+1] = combined_stats[var_name]["std"]
    col_index += 2

################# normalisation statistics (preprocessed stats) ###########
preprocessing_path = Path(__file__).parent / MODEL_NAME / f"{MODEL_NAME}_preprocessing.npz"
preprocessing = np.load(preprocessing_path)

bio_mean = preprocessing["bio_mean"]
bio_std = preprocessing["bio_std"]
force_mean = preprocessing["force_mean"]
force_std = preprocessing["force_std"]

Y_mean = preprocessing["Y_mean"]
Y_std = preprocessing["Y_std"]
#################

normalised_state_variables_stats = np.zeros((length_prediction_time, number_state_variables*2))  # 2 stats for each variable
# This is for normalised input values
col_index = 0
for var_name in combined_stats.keys():
    #Mean feature
    normalised_state_variables_stats[:, col_index] = (combined_stats[var_name]["mean"] - bio_mean[col_index]) / bio_std[col_index]

    #Standard deviation feature
    normalised_state_variables_stats[:, col_index+1] = (combined_stats[var_name]["std"] - bio_mean[col_index + 1]) / bio_std[col_index + 1]
    col_index += 2

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
    / MODEL_NAME
    / f"{MODEL_NAME}_best_model.keras"
)

model = keras.models.load_model(model_file, compile=False)

Y_pred_norm = rollout(model, normalised_state_variables_stats[:lookback], forcing_variables, lookback=30, steps=length_prediction_time-30, bio_mean=bio_mean, bio_std=bio_std, Y_mean=Y_mean, Y_std=Y_std)  
print("\nROLLOUT PREDICTION COMPLETE")

#OPTION 1 ################
Y_pred_log = (Y_pred_norm*Y_std) + Y_mean
################

#OPTION 2 ################
Y_pred_physical = Y_pred_log.copy()

for col_index, var in enumerate(state_variables_order):
    mean_col = col_index * 2
    std_col = mean_col + 1

    if var in variables_to_log_transform:
        # predicted mean and std in ln space
        mu_ln = Y_pred_log[:, mean_col]
        sigma_ln = Y_pred_log[:, std_col]

        # Physical-space log-normal mean and standard deviation
        physical_mean = np.exp(mu_ln + 0.5 * sigma_ln**2)
        physical_std = physical_mean * np.sqrt(
            np.exp(sigma_ln**2) - 1.0
        )

        Y_pred_physical[:, mean_col] = physical_mean - np.exp(-8)   # reverse the offset
        Y_pred_physical[:, std_col] = physical_std
#################

#OPTION 1
o_ln = Dataset(Path(__file__).parent / f"{MODEL_NAME}_log_predicted.nc", "w", format="NETCDF4_CLASSIC")
o_ln.createDimension("time", length_prediction_time - lookback)
############
#OPTION 2
o = Dataset(Path(__file__).parent / f"{MODEL_NAME}_linear_predicted.nc", "w", format="NETCDF4_CLASSIC")
o.createDimension("time", length_prediction_time - lookback)
############

#OPTION 1
col_index = 0
for var in state_variables_order:

    #predicted vs test mean in log-space
    variable = o_ln.createVariable("predicted_mean_"+var, np.float32, ("time",))
    variable[:] = Y_pred_log[:, col_index]
    variable = o_ln.createVariable("test_mean_"+var, np.float32, ("time",))
    variable[:] = pre_normalised_state_variables_stats[lookback:, col_index]

    #predicted vs test std in log-space
    variable = o_ln.createVariable("predicted_std_"+var, np.float32, ("time",))
    variable[:] = Y_pred_log[:, col_index+1]
    variable = o_ln.createVariable("test_std_"+var, np.float32, ("time",))
    variable[:] = pre_normalised_state_variables_stats[lookback:, col_index+1]

    col_index += 2
###############
o_ln.close()

#OPTION 2
col_index = 0
for var in state_variables_order:

    #predicted vs test mean in physical-space
    variable = o.createVariable("predicted_mean_"+var, np.float32, ("time",))
    variable[:] = Y_pred_physical[:, col_index]
    variable = o.createVariable("test_mean_"+var, np.float32, ("time",))
    variable[:] = not_logged_state_variables_stats[lookback:, col_index]

    #predicted vs test std in physical-space
    variable = o.createVariable("predicted_std_"+var, np.float32, ("time",))
    variable[:] = Y_pred_physical[:, col_index+1]
    variable = o.createVariable("test_std_"+var, np.float32, ("time",))
    variable[:] = not_logged_state_variables_stats[lookback:, col_index+1]

    col_index += 2
##############
o.close()