# Code to use a trained LSTM emulator for a rollout prediction. The test data are stored in a netCDF file, the outputs are stored in another netCDF file. 
# Initial version written by JS (early April 26) with significant input of code by AI (ChatGPT) 


import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
from netCDF4 import Dataset   
from tensorflow.keras import layers, models 
from matplotlib import pyplot as plt


def rollout(model, B_init, F, lookback, steps):
    """
    B_init: initial biogeochem history (lookback, n_bio)
    F: full forcing time series
    steps: number of steps to predict forward
    """

    B_hist = B_init.copy()
    preds = []

    for i in range(steps):
        t = lookback + i

        bio_input = B_hist[-lookback:][None, ...]
        force_input = F[t-lookback:t][None, ...]
        force_future = F[t][None, ...]

        pred = model.predict([bio_input, force_input, force_future], verbose=0)
        preds.append(pred[0])

        # append prediction for next step
        B_hist = np.vstack([B_hist, pred])

    return np.array(preds)

lookback=30

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
length_training_time = int(15*365.25)  # length of the time series for training and validation data
length_testing_time = 671  # length of the time series for training and validation data



i=Dataset("result.nc")

# Arrays with time x depths x variables dimensions

orig_state_variables = np.zeros((length_training_time+length_testing_time, number_state_variables))  # the original values stored before normalization - store the training period
normalized_state_variables = np.zeros((length_testing_time, number_state_variables))   # normalized values  - from test period (!)
forcing_variables = np.zeros((length_testing_time, number_forcing_variables))   # forcing values  - from test period


# read in and normalize

for index, var in enumerate(state_variables_names):    
    orig_state_variables[:,index] = i.variables[var][:][start_time:,-1,0,0]
    normalized_state_variables[:,index] = (i.variables[var][:][start_time+length_training_time:,-1,0,0] - orig_state_variables[:length_training_time,index].mean())/orig_state_variables[:length_training_time,index].std()
    
for index, var in enumerate(forcing_variables_names):
    if var in ["light_parEIR", "temp", "salt"]:    
        vinp = i.variables[var][:][start_time:start_time+length_training_time,-1,0,0]
        forcing_variables[:,index] = (i.variables[var][:][start_time+length_training_time:,-1,0,0] - vinp.mean())/vinp.std()
    else:    
        vinp = i.variables[var][:][start_time:start_time+length_training_time,0,0]
        forcing_variables[:,index] = (i.variables[var][:][start_time+length_training_time:,0,0] - vinp.mean())/vinp.std()     
    
i.close()



for ens in range(1,2):

    model = tf.keras.models.load_model("best_model_LSTM_NPZ_"+str(ens)+".keras")   

    Y_pred_norm = rollout(model, normalized_state_variables[:lookback], forcing_variables, lookback=30, steps=length_testing_time-30)  

    Y_pred = Y_pred_norm*orig_state_variables[:length_training_time,:].std(axis=0) + orig_state_variables[:length_training_time,:].mean(axis=0)

    o=Dataset("Predicted_LSTM_ens_"+str(ens)+".nc", "w", format="NETCDF4_CLASSIC")

    o.createDimension("time", length_testing_time-lookback)
#o.createDimension("depth", 100)

#print(np.shape(Y_pred), np.shape(Y_test))

    for index, var in enumerate(state_variables_names):

        variable = o.createVariable("predicted_"+var, np.float32, ("time"))
        variable[:] = Y_pred[:,index]
        variable = o.createVariable("test_"+var, np.float32, ("time"))
        variable[:] = orig_state_variables[length_training_time+lookback:,index]    
 
    
    o.close()


