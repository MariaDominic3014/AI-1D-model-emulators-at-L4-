from netCDF4 import Dataset

import numpy as np
#from scipy.stats import skew, kurtosis as kurtosis_function

import tensorflow as tf
import keras
from keras import layers, models

import json
import pandas as pd

from pathlib import Path



#SECTION 1 - Get the ensemble data from the netCDF file, compute the ensemble mean-and-std at every timestep for each variable, and put the stats for each variable into a dictionary
#crps_original or crps_ln
MODEL_NAME = "crps_ln"

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
    log_state_var = np.log(state_var + np.exp(-8))
    mean, std = compute_ensemble_stats(log_state_var)
    log_stats[name] = {
        "mean": mean,
        "std": std
    }

combined_stats = {**original_stats, **log_stats}



#SECTION 2 - create the state_variables_stats array and forcing_variables array and populate them with the correct values needed to create the input and target datasets for the model (at this point, the values haven't been normalised yet)
state_variables_stats = np.zeros((length_time, number_state_variables*2))  # 2 stats for each variable
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



#SECTION 3 - define the function that will create the input and target datasets
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



#SECTION 4 - define the functions that will build the model and create the loss function that the model will use
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

# loss function 1
def gaussian_crps(y_true, y_pred):

    predicted_mean_norm = y_pred[:, 0::2]
    predicted_std_norm  = y_pred[:, 1::2]

    target_mean_norm = y_true[:, 0::2]
    target_std_norm  = y_true[:, 1::2]

    # We can force non-negative std (even in linear space) BUT only on the loss - this might be a sensible safeguard on the actual model instead.
    #
    # This also doesn't mean that the distribution itself can't cross into the negative (which logging prevents)
    #    - it just means we don't generate negative standard deviations (which don't make sense at all!)
    predicted_std_norm = tf.nn.softplus(predicted_std_norm) + 1e-6

    # Combined standard deviation
    s = tf.sqrt(
        predicted_std_norm**2 +
        target_std_norm**2 +
        1e-8
    )

    # Difference between predicted and target means, measured in units of the combined uncertainty
    z = (predicted_mean_norm - target_mean_norm) / s

    # Standard normal PDF
    phi = tf.exp(-0.5 * z**2) / tf.sqrt(2.0 * np.pi)

    # Standard normal CDF
    Phi = 0.5 * (
        1.0 + tf.math.erf(z / tf.sqrt(2.0))
    )

    # Gaussian-Gaussian CRPS
    crps = (
        s * (
            2.0 * phi
            + z * (2.0 * Phi - 1.0)
        )
        - predicted_std_norm / tf.sqrt(np.pi)
    )

    # Average over variables and samples
    return tf.reduce_mean(crps)

#loss function 2
def lognormal_crps(y_true, y_pred):
    """
    Calculate the distributional CRPS / energy distance between
    predicted and target lognormal distributions.

    The parameters are specified in log space:

        log(X) ~ Normal(mu_log, sigma_log^2)

    Input format:
        [mu_1, std_1, mu_2, std_2, ...]

    where:
        - mu is the mean in log space
        - std is the standard deviation in log space

    The predicted standard deviation is passed through softplus
    because the neural network could produce an unconstrained value.

    The loss itself is calculated in LINEAR space.
    """

    # --------------------------------------------------------------
    # Extract parameters
    # --------------------------------------------------------------

    predicted_mu = y_pred[:, 0::2]
    predicted_std_raw = y_pred[:, 1::2]

    target_mu = y_true[:, 0::2]
    target_std = y_true[:, 1::2]

    # --------------------------------------------------------------
    # Ensure predicted log-space standard deviation is positive
    # --------------------------------------------------------------

    predicted_std = tf.nn.softplus(predicted_std_raw) + 1e-6

    # Target std should already be a valid positive standard deviation.
    # Small epsilon protects against zero.
    target_std = tf.maximum(target_std, 1e-6)

    # --------------------------------------------------------------
    # Convert lognormal parameters to linear-space means
    #
    # If:
    #     log(X) ~ N(mu, sigma^2)
    #
    # then:
    #     E[X] = exp(mu + sigma^2 / 2)
    # --------------------------------------------------------------

    predicted_mean = tf.exp(
        predicted_mu + 0.5 * predicted_std**2
    )

    target_mean = tf.exp(
        target_mu + 0.5 * target_std**2
    )

    # --------------------------------------------------------------
    # E[|X - Y|]
    #
    # X ~ LogNormal(predicted_mu, predicted_std^2)
    # Y ~ LogNormal(target_mu, target_std^2)
    # --------------------------------------------------------------

    combined_std = tf.sqrt(
        predicted_std**2
        + target_std**2
        + 1e-8
    )

    a = (
        target_mu
        - predicted_mu
        - predicted_std**2
    ) / combined_std

    b = (
        predicted_mu
        - target_mu
        - target_std**2
    ) / combined_std

    # Standard normal CDF
    Phi_a = 0.5 * (
        1.0 + tf.math.erf(a / tf.sqrt(2.0))
    )

    Phi_b = 0.5 * (
        1.0 + tf.math.erf(b / tf.sqrt(2.0))
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
        1.0
        + tf.math.erf(
            predicted_std / 2.0
        )
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
        1.0
        + tf.math.erf(
            target_std / 2.0
        )
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

    # --------------------------------------------------------------
    # Average over variables and samples
    # --------------------------------------------------------------

    return tf.reduce_mean(crps)



#SECTION 5 - call the function to create the input and target datasets, and then split each of them into separate training and validation sets
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



#SECTION 6 - normalise the values in the training and validation datasets using the training-period mean and std, then save the preprocessing stats for prediction scripts
#NOTE: for the crps model, the standard deviations in the X_bio_ datasets and Y_ datasets have to be normalised differently so that the the model sees both ensemble means and ensemble standard deviations in the same variable-specific normalised units:
# 1) Calculate statistics only from ensemble-mean columns
# 2) For ensemble means: z-score normalisation
# 3) For ensemble standard deviations: divide by the std of the corresponding mean

bio_mean = X_bio_train[:, :, 0::2].mean(
    axis=(0, 1),
    keepdims=True
)
bio_std = X_bio_train[:, :, 0::2].std(
    axis=(0, 1),
    keepdims=True
) + 1e-8

# Create empty copies of both training and validation sets, to be populated with the correctly-normalised values
X_bio_train_norm = np.empty_like(X_bio_train)
X_bio_val_norm = np.empty_like(X_bio_val)

# Normalise the means
X_bio_train_norm[:, :, 0::2] = (
    X_bio_train[:, :, 0::2] - bio_mean
) / bio_std
X_bio_val_norm[:, :, 0::2] = (
    X_bio_val[:, :, 0::2] - bio_mean
) / bio_std

# Scale the standard deviations
X_bio_train_norm[:, :, 1::2] = (
    X_bio_train[:, :, 1::2] / bio_std
)
X_bio_val_norm[:, :, 1::2] = (
    X_bio_val[:, :, 1::2] / bio_std
)

X_bio_train = X_bio_train_norm
X_bio_val = X_bio_val_norm


def compute_mean_std(X):
    mean = X.mean(axis=(0,1), keepdims=True)
    std  = X.std(axis=(0,1), keepdims=True) + 1e-8
    return mean, std

# Forcing (sequence)
force_mean, force_std = compute_mean_std(X_force_train)
X_force_train = (X_force_train - force_mean) / force_std
X_force_val   = (X_force_val   - force_mean) / force_std

# Future forcing (same stats as forcing)
X_force_future_train = (X_force_future_train - force_mean.squeeze(0)) / force_std.squeeze(0)
X_force_future_val   = (X_force_future_val   - force_mean.squeeze(0)) / force_std.squeeze(0)


# Targets (biogeochem)
Y_mean = Y_train[:, 0::2].mean(axis=0, keepdims=True)
Y_std = Y_train[:, 0::2].std(axis=0, keepdims=True) + 1e-8

Y_train_norm = np.empty_like(Y_train)
Y_val_norm = np.empty_like(Y_val)

# Normalise the target means
Y_train_norm[:, 0::2] = (Y_train[:, 0::2] - Y_mean) / Y_std
Y_val_norm[:, 0::2] = (Y_val[:, 0::2] - Y_mean) / Y_std

# Scale the target standard deviations
Y_train_norm[:, 1::2] = Y_train[:, 1::2] / Y_std
Y_val_norm[:, 1::2] = Y_val[:, 1::2] / Y_std

Y_train = Y_train_norm
Y_val = Y_val_norm


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



#SECTION 7 - call the function that builds the model with the crps loss function, define any callbacks, train the model over a maximum of 250 epochs, write the training history to a JSON file and save the best model in a .keras file
model = build_model(lookback, number_state_variables_stats, loss=gaussian_crps, metrics=[])
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