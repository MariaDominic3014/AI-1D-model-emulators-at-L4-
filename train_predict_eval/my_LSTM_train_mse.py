from netCDF4 import Dataset

import numpy as np
#from scipy.stats import skew, kurtosis as kurtosis_function

#import tensorflow as tf
import keras
from keras import layers, models

import json
import pandas as pd

from pathlib import Path



#SECTION 1
#mse_original or mse_ln
MODEL_NAME = "mse_ln"

data_file = Path(__file__).parent.parent / "data" / "processed_ensemble.nc"
ds = Dataset(str(data_file))

variables_to_keep_original = []
variables_to_log_transform = ["P1_Chl", "P1_Chl_vert", "P2_Chl", "P2_Chl_vert", "Z6_c", "Z6_c_vert", "N1_p", "N3_n", "N4_n", "N5_s", "O2_bot", "O2_o", "P3_Chl", "P3_Chl_vert", "P4_Chl", "P4_Chl_vert", "Z5_c", "Z5_c_vert", "Z4_c", "Z4_c_vert"]
number_state_variables = len(variables_to_keep_original) + len(variables_to_log_transform)
variable_order = variables_to_keep_original + variables_to_log_transform

forcing_variables_names = ["heat", "light_parEIR_surf", "mld_surf", "precip", "salt", "temp", "u10", "v10"]
number_forcing_variables = len(forcing_variables_names)

start_time = 365 + 334  # account for spin up (nearly 2 years)
length_time = int(15*365.25)  # length of the time series for training and validation data

def compute_ensemble_stats(state_var):
    # state_var shape: (time, ensemble_members)
    # returns: mean, std

    state_var = np.asarray(state_var, dtype=np.float64)

    mean = np.mean(state_var, axis=1)
    std = np.std(state_var, axis=1)

    return mean, std


original_stats = {}
for name in variables_to_keep_original:
    state_var = ds.variables[name][start_time:start_time + length_time]
    mean, std = compute_ensemble_stats(state_var)
    original_stats[name] = {
        "mean": mean,
        "std": std
    }

log_stats = {}
for name in variables_to_log_transform:
    state_var = ds.variables[name][start_time:start_time + length_time]
    log_state_var = np.log(state_var + np.exp(-8))   # add a small offset before logging
    mean, std = compute_ensemble_stats(log_state_var)
    log_stats[name] = {
        "mean": mean,
        "std": std
    }

combined_stats = {**original_stats, **log_stats}



#SECTION 2
state_variables_stats = np.zeros((length_time, number_state_variables*2))  # 2 stats for each variable
# Populate state_variables_stats
col_index = 0
for var_name in combined_stats.keys():
    state_variables_stats[:, col_index] = combined_stats[var_name]["mean"]
    state_variables_stats[:, col_index+1] = combined_stats[var_name]["std"]
    col_index += 2
number_state_variables_stats = state_variables_stats.shape[1]

forcing_variables = np.zeros((length_time, number_forcing_variables))
for index, var in enumerate(forcing_variables_names):
    forcing_variables[:,index] = ds.variables[var][start_time:start_time+length_time]

ds.close()



#SECTION 3
lookback = 30

def create_dataset(B, F, lookback):
    X_bio, X_force, X_force_future, Y = [], [], [], []

    for t in range(lookback, len(B)):
        X_bio.append(B[t-lookback:t])
        X_force.append(F[t-lookback:t])
        X_force_future.append(F[t])  # known forcing at prediction time
        Y.append(B[t])   # Y contains mean + std for each variable at time t

    return (
        np.array(X_bio),
        np.array(X_force),
        np.array(X_force_future),
        np.array(Y)
    )



#SECTION 4
def build_model(lookback, number_state_variables_stats, optimizer='adam', loss='mse', metrics=None, verbose=False):
    if metrics is None:
        metrics = []

    # Inputs
    bio_input = layers.Input(shape=(lookback, number_state_variables_stats))
    force_input = layers.Input(shape=(lookback, number_forcing_variables))
    force_future_input = layers.Input(shape=(number_forcing_variables,))

    # Concatenate past sequences
    x = layers.Concatenate(axis=-1)([bio_input, force_input])

    # LSTM encoder
    x = layers.LSTM(64, return_sequences=False)(x)

    # Concatenate future forcing
    x = layers.Concatenate()([x, force_future_input])

    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(number_state_variables_stats)(x)

    model = models.Model(
        inputs=[bio_input, force_input, force_future_input],
        outputs=output
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    if verbose:
        model.summary()
    return model



#SECTION 5
X_bio, X_force, X_force_future, Y = create_dataset(state_variables_stats, forcing_variables, lookback)

train_frac = 0.8
N = len(X_bio)
train_size = int(N * train_frac)

X_bio_train = X_bio[:train_size]
X_bio_val   = X_bio[train_size:]

X_force_train = X_force[:train_size]
X_force_val   = X_force[train_size:]

X_force_future_train = X_force_future[:train_size]
X_force_future_val   = X_force_future[train_size:]

Y_train = Y[:train_size]
Y_val   = Y[train_size:]

print("\n ATTENTION: X_bio_train", np.shape(X_bio_train))
print("\n ATTENTION: X_force_train", np.shape(X_force_train))
print("\n ATTENTION: X_force_future_train", np.shape(X_force_future_train))
print("\n ATTENTION: Y_train", np.shape(Y_train))
print()



#SECTION 6
def compute_mean_std(X):
    mean = X.mean(axis=(0,1), keepdims=True)
    std  = X.std(axis=(0,1), keepdims=True) + 1e-8
    return mean, std

bio_mean, bio_std = compute_mean_std(X_bio_train)
X_bio_train = (X_bio_train - bio_mean) / bio_std
X_bio_val   = (X_bio_val   - bio_mean) / bio_std

# Forcing (sequence)
force_mean, force_std = compute_mean_std(X_force_train)
X_force_train = (X_force_train - force_mean) / force_std
X_force_val   = (X_force_val   - force_mean) / force_std

# Future forcing (same stats as forcing)
X_force_future_train = (X_force_future_train - force_mean.squeeze(0)) / force_std.squeeze(0)
X_force_future_val   = (X_force_future_val   - force_mean.squeeze(0)) / force_std.squeeze(0)

# Targets (biogeochem)
Y_mean = Y_train.mean(axis=0, keepdims=True)
Y_std  = Y_train.std(axis=0, keepdims=True) + 1e-8
Y_train = (Y_train - Y_mean) / Y_std
Y_val   = (Y_val   - Y_mean) / Y_std


preprocessing_path = Path(__file__).parent / f"{MODEL_NAME}_preprocessing.npz"
np.savez(
    preprocessing_path,
    bio_mean=bio_mean.reshape(-1),
    bio_std=bio_std.reshape(-1),
    force_mean=force_mean.reshape(-1),
    force_std=force_std.reshape(-1),
    Y_mean=Y_mean.reshape(-1),
    Y_std=Y_std.reshape(-1),
)



#SECTION 7
model = build_model(lookback, number_state_variables_stats, loss='mse', metrics=[])
print("\nModel Architecture:")
model.summary()


early_stop_cbk = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

lr_cbk = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

best_model_path = Path(__file__).parent / f"{MODEL_NAME}_best_model.keras"
checkpoint_cbk = keras.callbacks.ModelCheckpoint(
    best_model_path,
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

history = model.fit(
    [X_bio_train, X_force_train, X_force_future_train],
    Y_train,
    validation_data=(
        [X_bio_val, X_force_val, X_force_future_val],
        Y_val
    ),
    callbacks=[early_stop_cbk, lr_cbk, checkpoint_cbk],
    epochs=250,
    batch_size=512,
    shuffle=False  # IMPORTANT
)


with open(Path(__file__).parent / f"{MODEL_NAME}_history.json", "w") as f:
    json.dump(history.history, f)

fig = pd.DataFrame(history.history).plot().get_figure()
fig.savefig(Path(__file__).parent / f"{MODEL_NAME}_training_history.png")

# Save the final model too
final_model_path = Path(__file__).parent / f"{MODEL_NAME}_final_model.keras"
model.save(final_model_path)


print(f"Best model saved to: {best_model_path}")
print(f"Final model saved to: {final_model_path}")