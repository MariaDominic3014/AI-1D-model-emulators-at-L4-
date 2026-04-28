# Code to train a LSTM emulator to predict dynamics of selected surface variables of GOTM-FABM-ERSEM 1D water column physics-biogeochemistry model. The training/validation data are stored in a netCDF file, read in and the CNN model weights are saved in a *.keras file. 
# Initial version written by JS (early April 26) with significant input of code by AI (ChatGPT) 



import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
from netCDF4 import Dataset   
from tensorflow.keras import layers, models 
import json

# -----------------------------------------------  below are the key settings

lookback = 30

# state variables included
  
state_variables_names = ["P1_Chl", "P2_Chl", "P3_Chl", "P4_Chl", "Z4_c", "Z5_c", "Z6_c", "N3_n", "N1_p", "N4_n", "N5_s", "O2_o"]  

# number of state variables

number_state_variables = len(state_variables_names) 

# forcing variables included
    
forcing_variables_names = ["light_parEIR", "u10", "v10", "precip", "heat", "temp", "salt"]

# number of forcing variables

number_forcing_variables = len(forcing_variables_names) 

# parameters defining the dimensions

start_time = 365 + 334  # account for spin up (nearly 2 years)
length_time = int(15*365.25)  # length of the time series for training and validation data



# --------------------------------------------------------------- key functions

i=Dataset("result.nc")

# Arrays with time x depths x variables dimensions

normalized_state_variables = np.zeros((length_time, number_state_variables))   # normalized values
forcing_variables = np.zeros((length_time, number_forcing_variables))   # forcing values


# read in and normalize

for index, var in enumerate(state_variables_names):    
    vinp = i.variables[var][:][start_time:start_time+length_time,-1,0,0]
    normalized_state_variables[:,index] = (vinp - vinp.mean())/vinp.std()
    
for index, var in enumerate(forcing_variables_names):
    if var in ["light_parEIR", "temp", "salt"]:    
        vinp = i.variables[var][:][start_time:start_time+length_time,-1,0,0]
        vinp = (vinp - vinp.mean())/vinp.std()
        forcing_variables[:,index] = vinp        
    else:    
        vinp = i.variables[var][:][start_time:start_time+length_time,0,0]
        vinp = (vinp - vinp.mean())/vinp.std()
        forcing_variables[:,index] = vinp        
    
i.close()


       


# ----------------------------
def create_dataset(B, F, lookback):
    X_bio, X_force, X_force_future, Y = [], [], [], []

    for t in range(lookback, len(B)):
        X_bio.append(B[t-lookback:t])
        X_force.append(F[t-lookback:t])
        X_force_future.append(F[t])  # known forcing at prediction time
        Y.append(B[t])

    return (
        np.array(X_bio),
        np.array(X_force),
        np.array(X_force_future),
        np.array(Y)
    )
    
def compute_stats(X):
    mean = X.mean(axis=(0,1), keepdims=True)
    std  = X.std(axis=(0,1), keepdims=True) + 1e-8
    return mean, std

X_bio, X_force, X_force_future, Y = create_dataset(normalized_state_variables, forcing_variables, lookback)


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

print("ATTENTION", np.shape(X_bio_train))

bio_mean, bio_std = compute_stats(X_bio_train)

X_bio_train = (X_bio_train - bio_mean) / bio_std
X_bio_val   = (X_bio_val   - bio_mean) / bio_std

# Forcing (sequence)
force_mean, force_std = compute_stats(X_force_train)

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

# ----------------------------
# MODEL
# ----------------------------

for ens in range(1,2):

# Inputs
    bio_input = layers.Input(shape=(lookback, number_state_variables))
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
    output = layers.Dense(number_state_variables)(x)

    model = models.Model(
        inputs=[bio_input, force_input, force_future_input],
        outputs=output
    )

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    model.summary()


    history = model.fit(
        [X_bio_train, X_force_train, X_force_future_train],
        Y_train,
        validation_data=(
            [X_bio_val, X_force_val, X_force_future_val],
            Y_val
        ),
        epochs=50,
        batch_size=32,
        shuffle=False  # IMPORTANT
    )

    with open("history_model_LSTM_NPZ_"+str(ens)+".json", "w") as f:
        json.dump(history.history, f)


    model.save("best_model_LSTM_NPZ_"+str(ens)+".keras")  # save the best weights
