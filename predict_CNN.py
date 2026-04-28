# Code to use a trained 1D CNN emulator for a rollout prediction. The test data are stored in a netCDF file, the outputs are stored in another netCDF file. 
# Initial version written by JS (early April 26) with significant input of code by AI (ChatGPT) 

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt


# -----------------------------------------------  below are the key settings

# state variables included
  
state_variables_names = ["temp", "salt", "P1_Chl", "P2_Chl", "P3_Chl", "P4_Chl", "P1_c", "P2_c", "P3_c", "P4_c", "P1_n", "P2_n", "P3_n", "P4_n", "P1_p", "P2_p", "P3_p", "P4_p", "P1_s", "Z4_c", "Z5_c", "Z6_c", "Z5_n", "Z6_n", "Z5_p", "Z6_p", "R1_c", "R2_c", "R3_c", "R1_n", "R1_p", "R4_c", "R6_c", "R8_c", "R4_n", "R6_n", "R8_n", "R4_p", "R6_p", "R8_p", "R6_s", "R8_s", "B1_c", "B1_n", "B1_p", "N3_n", "N1_p", "N4_n", "N5_s", "O2_o", "O3_c", "O3_pH"]  

# number of state variables

number_state_variables = len(state_variables_names) 

# forcing variables included
    
forcing_variables_names = ["light_parEIR", "u10", "v10", "precip", "heat"]

# number of forcing variables

number_forcing_variables = len(forcing_variables_names) 

# parameters defining the dimensions

start_time = 365 + 334  # account for spin up (nearly 2 years)
length_training_time = int(15*365.25)  # length of the time series for training and validation data
length_testing_time = 671  # length of the time series for training and validation data
n_vertical = 100  # number of vertical layers


i=Dataset("result.nc")

# Arrays with time x depths x variables dimensions

orig_state_variables = np.zeros((length_training_time+length_testing_time, n_vertical, number_state_variables))  # the original values stored before normalization - store the training period
normalized_state_variables = np.zeros((length_testing_time, n_vertical, number_state_variables))   # normalized values  - from test period (!)
forcing_variables = np.zeros((length_testing_time, number_forcing_variables))   # forcing values  - from test period


# read in and normalize

for index, var in enumerate(state_variables_names):    
    orig_state_variables[:,:,index] = i.variables[var][:][start_time:,:,0,0]
    normalized_state_variables[:,:,index] = (i.variables[var][:][start_time+length_training_time:,:,0,0] - orig_state_variables[:length_training_time,:,index].mean())/orig_state_variables[:,:length_training_time,index].std()
    
for index, var in enumerate(forcing_variables_names):
    if var == "light_parEIR":    
        vinp = i.variables[var][:][start_time:start_time+length_training_time,-1,0,0]
        forcing_variables[:,index] = (i.variables[var][:][start_time+length_training_time:,-1,0,0] - vinp.mean())/vinp.std()
    else:    
        vinp = i.variables[var][:][start_time:start_time+length_training_time,0,0]
        forcing_variables[:,index] = (i.variables[var][:][start_time+length_training_time:,0,0] - vinp.mean())/vinp.std()     
    
i.close()


# run prediction 

# loop through ensemble members

for ens in range(1,16):
        
    model = tf.keras.models.load_model("best_model_CNN_"+str(ens)+".keras")    

    predicted_state_variables  = normalized_state_variables[0,:,:]  # initialize the time series

# prediction loop 

    predicted_state_variables = predicted_state_variables.astype(np.float32)

# forcing sequence: f_seq shape = [T, num_forcing_features]
    f_seq = forcing_variables.astype(np.float32)

    trajectory = []  # store predicted states

    predicted_state_variables = tf.convert_to_tensor(predicted_state_variables[None, ...])  # add batch dimension: [1, depth, variables]


    for t in range(length_testing_time):
        f = tf.convert_to_tensor(f_seq[t][None, ...], dtype=tf.float32)  # [1, num_forcing]

    # ensure 2D before adding batch dim
        if len(predicted_state_variables.shape) == 3:
            predicted_state_variables = predicted_state_variables[0]

    # add batch dimension for model
        x_in = predicted_state_variables[None, ...]  # [1, 100, 52]

    # predict next state
        x_out = model([x_in, f], training=False)

    # remove batch dimension
        predicted_state_variables = x_out[0]  # [100, 52]

    # overwrite first 2 features
        new_vals = tf.convert_to_tensor(normalized_state_variables[min(t+1, length_testing_time-1), :, :2], dtype=tf.float32)
        predicted_state_variables = tf.concat([new_vals, predicted_state_variables[:, 2:]], axis=-1)

    # store prediction
        trajectory.append(predicted_state_variables.numpy())

    predicted_state_variables = np.array(trajectory)

    for index, var in enumerate(state_variables_names):
        predicted_state_variables[:,:,index] = predicted_state_variables[:,:,index]*orig_state_variables[:,:length_training_time,index].std() + orig_state_variables[:,:length_training_time,index].mean()

# save outputs
      
    o=Dataset("Predicted_without_depth_ens_"+str(ens)+".nc", "w", format="NETCDF4_CLASSIC")

    o.createDimension("time", length_testing_time)
    o.createDimension("depth", n_vertical)

    for index, var in enumerate(state_variables_names):

        variable = o.createVariable("predicted_"+var, np.float32, ("time", "depth"))
        variable[:] = predicted_state_variables[:,:,index]
        variable = o.createVariable("test_"+var, np.float32, ("time", "depth"))
        variable[:] = orig_state_variables[length_training_time:,:,index]    
 
    
    o.close()
