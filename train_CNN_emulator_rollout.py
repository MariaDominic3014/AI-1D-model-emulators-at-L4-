# Code to train a 1D CNN emulator to predict a time step of GOTM-FABM-ERSEM 1D water column physics-biogeochemistry model. The training/validation data are stored in a netCDF file, read in and the CNN model weights are saved in a *.keras file. 
# Initial version written by JS (early April 26) with significant input of code by AI (ChatGPT) 

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from netCDF4 import Dataset
import json


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
length_time = int(15*365.25)  # length of the time series for training and validation data
n_vertical = 100  # number of vertical layers


# --------------------------------------------------------------- key functions

def residual_block(x, filters, kernel_size=5):
    h = layers.Conv1D(filters, kernel_size, padding='same')(x)
    h = layers.Activation('gelu')(h)
    h = layers.Conv1D(filters, kernel_size, padding='same')(h)
    return layers.Add()([x, h])

def build_model(depth=n_vertical, n_vars=number_state_variables, n_forcing=number_forcing_variables):
    # Inputs
    state_input = layers.Input(shape=(depth, n_vars))
    forcing_input = layers.Input(shape=(n_forcing,))

    # Broadcast forcing to depth
    f = layers.RepeatVector(depth)(forcing_input)

    # Concatenate state + forcing
    x = layers.Concatenate(axis=-1)([state_input, f])

    # Initial projection
    x = layers.Conv1D(128, 5, padding='same')(x)

    # Residual blocks
    for _ in range(6):
        x = residual_block(x, 128)

    # Output: tendency
    dx = layers.Conv1D(n_vars, 1, padding='same')(x)

    # Residual update
    output = layers.Add()([state_input, dx])

    return models.Model(inputs=[state_input, forcing_input], outputs=output)
    
    
def make_sequences(data, forcing, K):
    X0, F_seq, Y_seq = [], [], []

    T = data.shape[0]

    for t in range(T - K):
        X0.append(data[t])                 # initial state
        F_seq.append(forcing[t:t+K])       # forcing sequence
        Y_seq.append(data[t+1:t+K+1])      # true future states

    return np.array(X0), np.array(F_seq), np.array(Y_seq)
    


# ---------------------------------------- read the data in

i=Dataset("result.nc")

# Arrays with time x depths x variables dimensions

normalized_state_variables = np.zeros((length_time, n_vertical, number_state_variables))   # normalized values
forcing_variables = np.zeros((length_time, number_forcing_variables))   # forcing values


# read in and normalize

for index, var in enumerate(state_variables_names):    
    vinp = i.variables[var][:][start_time:start_time+length_time,:,0,0]
    normalized_state_variables[:,:,index] = (vinp - vinp.mean())/vinp.std()
    
for index, var in enumerate(forcing_variables_names):
    if var == "light_parEIR":    
        vinp = i.variables[var][:][start_time:start_time+length_time,-1,0,0]
        vinp = (vinp - vinp.mean())/vinp.std()
        forcing_variables[:,index] = vinp        
    else:    
        vinp = i.variables[var][:][start_time:start_time+length_time,0,0]
        vinp = (vinp - vinp.mean())/vinp.std()
        forcing_variables[:,index] = vinp        
    
i.close()


# extract training inputs and outputs and convert them to tensorflow format

training_state_variables = normalized_state_variables[:-1,:,:] 
training_forcing_variables = forcing_variables[:-1,:]    
training_outputs = normalized_state_variables[1:,:,:]    

training_state_variables = training_state_variables.astype(np.float32)
training_forcing_variables = training_forcing_variables.astype(np.float32)
       

# -------------------------- Loop through ensemble members and individually train and save the best models

for ens in range(1,3):

    history = {
    "train_loss": [],
    "val_loss": []
     }

    @tf.function    
    def train_step(model, optimizer, x0, f_seq, y_seq):
        with tf.GradientTape() as tape:
            x = x0
            total_loss = 0.0

            K = tf.shape(f_seq)[1]

            for k in tf.range(K):
                f = f_seq[:, k]        # [B, F]
                y_true = y_seq[:, k]   # [B, D, V]
           
                x = model([x, f], training=True)

                loss = tf.reduce_mean((x - y_true)**2)
                total_loss += loss

            total_loss /= tf.cast(K, tf.float32)

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return total_loss


    model = build_model()
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

    optimizer = tf.keras.optimizers.Adam(1e-3, clipnorm=1.0)

    batch_size = 32
    epochs = 20

    for K in [10]:

#K = 10  # rollout length

        X0, F_seq, Y_seq = make_sequences(training_state_variables, training_forcing_variables, K)
        N = X0.shape[0]

        split = int(0.8 * N)
        X0_train, F_train, Y_train = X0[:split], F_seq[:split], Y_seq[:split]
        X0_val,   F_val,   Y_val   = X0[split:], F_seq[split:], Y_seq[split:]

        best_val_loss = np.inf
        patience = 5         # stop if no improvement in 5 epochs
        wait = 0

        for epoch in range(epochs):
    # Shuffle training data
            idx = np.random.permutation(X0_train.shape[0])
            X0_train, F_train, Y_train = X0_train[idx], F_train[idx], Y_train[idx]

    # --- training loop ---
                
            train_loss_epoch = 0
            num_batches = 0 
            
            for i in range(0, X0_train.shape[0], batch_size):
                x0_batch = X0_train[i:i+batch_size]
                f_batch  = F_train[i:i+batch_size]
                y_batch  = Y_train[i:i+batch_size]
                loss = train_step(model, optimizer, x0_batch, f_batch, y_batch)
                train_loss_epoch += loss
                num_batches += 1
                train_loss_epoch /= num_batches

        

    # --- compute validation loss after each epoch ---
            val_loss = 0
            for i in range(0, X0_val.shape[0], batch_size):
                x0_batch = X0_val[i:i+batch_size]
                f_batch  = F_val[i:i+batch_size]
                y_batch  = Y_val[i:i+batch_size]

        # forward pass only
                x = x0_batch
                total = 0
                K = f_batch.shape[1]
                for k in range(K):
#            x = model([x, f_batch[:, k]], training=False)
                    xt = tf.convert_to_tensor(x, dtype=tf.float32)
                    f = tf.convert_to_tensor(f_batch[:, k], dtype=tf.float32)
                    x = model([xt, f], training=False)
                    total += tf.reduce_mean((x - y_batch[:, k])**2)
                val_loss += total / K

            val_loss /= (X0_val.shape[0] / batch_size)
            train_loss_epoch = loss.numpy() 
            val_loss_epoch = val_loss.numpy()
            history["train_loss"].append(float(train_loss_epoch))
            history["val_loss"].append(float(val_loss_epoch))
            print(f"Epoch {epoch+1}, Train Loss: {train_loss_epoch:.6f}, Val Loss: {val_loss_epoch:.6f}")

    # --- check for improvement ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                model.save("best_model_CNN_"+str(ens)+".keras")  # save the best weights
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping: no improvement")
                    break

     # save history

        with open(f"history_CNN_ens_{ens}.json", "w") as f:
            json.dump(history, f)


