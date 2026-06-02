import numpy as np
import tensorflow as tf
from netCDF4 import Dataset

lookback = 30

# ==========================================================
# CHOOSE OUTPUT VARIABLE HERE
# ==========================================================
#
# Examples:
# TARGET_OUTPUT = "O2_bot"
# TARGET_OUTPUT = "P1_Chl"
# TARGET_OUTPUT = "P1_Chl_vert"
#
# ==========================================================

#TARGET_OUTPUT = "O2_bot"
TARGET_OUTPUT = "N3_n"

# state variables included

state_variables_names = [
    "P1_Chl", "P2_Chl", "P3_Chl", "P4_Chl",
    "Z4_c", "Z5_c", "Z6_c",
    "N3_n", "N1_p", "N4_n", "N5_s",
    "O2_o",
    "P1_Chl_vert", "P2_Chl_vert",
    "P3_Chl_vert", "P4_Chl_vert",
    "Z4_c_vert", "Z5_c_vert",
    "Z6_c_vert", "O2_bot"
]

forcing_variables_names = [
    "light_parEIR",
    "u10",
    "v10",
    "precip",
    "heat",
    "temp",
    "salt",
    "mld_surf"
]

number_state_variables = len(state_variables_names)
number_forcing_variables = len(forcing_variables_names)

target_index = state_variables_names.index(TARGET_OUTPUT)

start_time = 365 + 334
length_training_time = int(12 * 365.25)
length_testing_time = 8734 - start_time

# ----------------------------------------------------------
# LOAD INPUT DATA
# ----------------------------------------------------------

i1 = Dataset("../data/data_2002_2025_for_LSTM_1_compressed.nc")
i2 = Dataset("../data/data_2002_2025_for_LSTM_2_compressed.nc")

def getvar(varname):
    if varname in i1.variables:
        return i1.variables[varname]
    elif varname in i2.variables:
        return i2.variables[varname]
    else:
        raise KeyError(f"{varname} not found")

h_nc = Dataset("../data/h.nc")
h = h_nc.variables["h"][:][:,:,0,0]

orig_state_variables = np.zeros(
    (length_testing_time, number_state_variables)
)

normalized_state_variables = np.zeros(
    (length_testing_time, number_state_variables)
)

forcing_variables = np.zeros(
    (length_testing_time, number_forcing_variables)
)

# ----------------------------------------------------------
# READ AND NORMALISE
# ----------------------------------------------------------

for index, var in enumerate(state_variables_names):

    if "vert" in var:

        vals = (
            getvar(var[:-5])[:][start_time:,:,0,0]
            * h[start_time:,:]
        ).sum(axis=1) / h[start_time:,:].sum(axis=1)

    elif var == "O2_bot":

        vals = (
            getvar("O2_o")[:][start_time:,0:5,0,0]
            * h[start_time:,0:5]
        ).sum(axis=1) / h[start_time:,0:5].sum(axis=1)

    else:

        vals = getvar(var)[:][start_time:,-1,0,0]

    orig_state_variables[:,index] = vals

    mu = orig_state_variables[:length_training_time,index].mean()
    sigma = orig_state_variables[:length_training_time,index].std()

    normalized_state_variables[:,index] = (vals - mu) / sigma

for index, var in enumerate(forcing_variables_names):

    if var in ["light_parEIR", "temp", "salt"]:

        train_vals = getvar(var)[
            start_time:start_time+length_training_time,
            -1,0,0
        ]

        full_vals = getvar(var)[start_time:,-1,0,0]

    else:

        train_vals = getvar(var)[
            start_time:start_time+length_training_time,
            0,0
        ]

        full_vals = getvar(var)[start_time:,0,0]

    forcing_variables[:,index] = (
        full_vals - train_vals.mean()
    ) / train_vals.std()

i1.close()
i2.close()

# ----------------------------------------------------------
# COMPUTE GRADIENTS FOR EACH ENSEMBLE MEMBER
# ----------------------------------------------------------

for ens in range(1):

    print("Ensemble:", ens)

    model = tf.keras.models.load_model(
        "../models/best_model_LSTM_vert_dm_imp_" +
        str(ens) +
        ".keras"
    )

    nt = length_testing_time - lookback

    grad_bio = np.zeros(
        (nt, lookback, number_state_variables),
        dtype=np.float32
    )

    grad_force_hist = np.zeros(
        (nt, lookback, number_forcing_variables),
        dtype=np.float32
    )

    grad_force_future = np.zeros(
        (nt, number_forcing_variables),
        dtype=np.float32
    )

    for t in range(lookback, length_testing_time):

        print(
            "Ensemble",
            ens,
            "time",
            t,
            "/",
            length_testing_time
        )

        bio_input = tf.Variable(
            normalized_state_variables[
                t-lookback:t
            ][None,...],
            dtype=tf.float32
        )

        force_input = tf.Variable(
            forcing_variables[
                t-lookback:t
            ][None,...],
            dtype=tf.float32
        )

        force_future = tf.Variable(
            forcing_variables[t][None,...],
            dtype=tf.float32
        )

        with tf.GradientTape(persistent=True) as tape:

            pred = model(
                [bio_input,
                 force_input,
                 force_future],
                training=False
            )

            target = pred[0, target_index]

        g_bio = tape.gradient(
            target,
            bio_input
        ).numpy()[0]

        g_force_hist = tape.gradient(
            target,
            force_input
        ).numpy()[0]

        g_force_future = tape.gradient(
            target,
            force_future
        ).numpy()[0]

        grad_bio[t-lookback,:,:] = g_bio
        grad_force_hist[t-lookback,:,:] = g_force_hist
        grad_force_future[t-lookback,:] = g_force_future

        del tape

    # ------------------------------------------------------
    # SAVE NETCDF
    # ------------------------------------------------------

    o = Dataset(
        f"Sensitivities_{TARGET_OUTPUT}_ens_{ens}.nc",
        "w",
        format="NETCDF4"
    )

    o.createDimension("time", nt)
    o.createDimension("lag", lookback)
    o.createDimension(
        "bio_variable",
        number_state_variables
    )
    o.createDimension(
        "forcing_variable",
        number_forcing_variables
    )

    v = o.createVariable(
        "grad_bio",
        np.float32,
        ("time", "lag", "bio_variable")
    )
    v[:] = grad_bio

    v.long_name = (
        f"d({TARGET_OUTPUT})/d(biological_input)"
    )

    v = o.createVariable(
        "grad_forcing_history",
        np.float32,
        ("time", "lag", "forcing_variable")
    )
    v[:] = grad_force_hist

    v.long_name = (
        f"d({TARGET_OUTPUT})/d(forcing_history)"
    )

    v = o.createVariable(
        "grad_forcing_future",
        np.float32,
        ("time", "forcing_variable")
    )
    v[:] = grad_force_future

    v.long_name = (
        f"d({TARGET_OUTPUT})/d(forcing_future)"
    )

    o.target_output = TARGET_OUTPUT
    o.bio_variables = ",".join(state_variables_names)
    o.forcing_variables = ",".join(forcing_variables_names)

    o.close()