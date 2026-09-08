from netCDF4 import Dataset
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

ds = Dataset("..\\data\\processed_ensemble.nc")

variables_to_log_transform = ["P3_Chl", "P3_Chl_vert", "P4_Chl", "P4_Chl_vert", "Z5_c", "Z5_c_vert", "Z4_c", "Z4_c_vert"]
variables_to_keep_original = ["P1_Chl", "P1_Chl_vert", "P2_Chl", "P2_Chl_vert", "Z6_c", "Z6_c_vert", "N1_p", "N3_n", "N4_n", "N5_s", "O2_bot", "O2_o"]

start_time = 365 + 334  # account for spin up (nearly 2 years)
length_time = int(15*365.25)  # length of the time series for training and validation data


### ORIGINAL ###
original_stats = {}
original_central_moments = {}
for name in variables_to_keep_original:

    state_var = ds.variables[name][start_time:start_time+length_time]

    # Compute stats across the ensemble members (axis=1)
    mean = np.mean(state_var, axis=1)
    std = np.std(state_var, axis=1)
    variance = np.var(state_var, axis=1)
    skewness = skew(state_var, axis=1)
    kurt = kurtosis(state_var, axis=1, fisher=True)

    original_stats[name] = {
    "mean": mean,
    "std": std,
    "variance": variance,
    "skewness": skewness,
    "kurtosis": kurt
    }

    # Compute central moments across the ensemble members (axis=1)
    m2 = np.mean((state_var - mean[:, np.newaxis])**2, axis=1)
    m3 = np.mean((state_var - mean[:, np.newaxis])**3, axis=1)
    m4 = np.mean((state_var - mean[:, np.newaxis])**4, axis=1)

    original_central_moments[name] = {
        "mean": mean,
        "m2": m2,
        "m3": m3,
        "m4": m4
    }

### LOG ###
log_stats = {}
log_central_moments = {}
for name in variables_to_log_transform:

    state_var = ds.variables[name][start_time:start_time+length_time]

    # Compute stats across the ensemble members (axis=1)
    # Log-transform first
    log_state_var = np.log10(state_var)

    mean = np.mean(log_state_var, axis=1)
    std = np.std(log_state_var, axis=1)
    variance = np.var(log_state_var, axis=1)
    skewness = skew(log_state_var, axis=1)
    kurt = kurtosis(log_state_var, axis=1, fisher=True)

    log_stats[name] = {
    "mean": mean,
    "std": std,
    "variance": variance,
    "skewness": skewness,
    "kurtosis": kurt
    }

    # Compute central moments across the ensemble members (axis=1)
    m2 = np.mean((log_state_var - mean[:, np.newaxis])**2, axis=1)
    m3 = np.mean((log_state_var - mean[:, np.newaxis])**3, axis=1)
    m4 = np.mean((log_state_var - mean[:, np.newaxis])**4, axis=1)

    log_central_moments[name] = {
        "mean": mean,
        "m2": m2,
        "m3": m3,
        "m4": m4
    }

combined_central_moments = {**original_central_moments, **log_central_moments}



# #####
# # pick one variable
# var_name = "P2_Chl"
# state_var = ds.variables[var_name][start_time:start_time + length_time]
# # compute the ensemble mean for each time step - choose o or l
# mean = np.mean(state_var, axis=1)
# #mean = np.mean(np.log10(state_var), axis=1)

# #####
# #central moments of order 3 and 4 - choose o or l
# m3 = np.mean((state_var - mean[:, np.newaxis])**3, axis=1)
# #m3 = np.mean((np.log10(state_var) - mean[:, np.newaxis])**3, axis=1)
# #skewness = skew(state_var, axis=1)
# #skewness = skew(np.log10(state_var), axis=1)
# m4 = np.mean((state_var - mean[:, np.newaxis])**4, axis=1)
# #m4 = np.mean((np.log10(state_var) - mean[:, np.newaxis])**4, axis=1)
# #kurt = kurtosis(state_var, axis=1, fisher=True)
# #kurt = kurtosis(np.log10(state_var), axis=1, fisher=True)
# #time axis
# #time_axis = np.arange(len(m3))
# time_axis = np.arange(state_var.shape[0])
# #####
# #plot 3rd and 4th moments
# fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
# axes[0].plot(time_axis, m3, color="tab:blue", linewidth=1.5)
# #axes[0].plot(time_axis, skewness, color="tab:blue", linewidth=1.5)
# axes[0].set_title(f"{var_name}: 3rd central moment (m3)")
# #axes[0].set_title(f"{var_name}: skewness")
# axes[0].set_ylabel("m3")
# #axes[0].set_ylabel("skewness")
# axes[0].grid(True, alpha=0.3)
# #####
# axes[1].plot(time_axis, m4, color="tab:orange", linewidth=1.5)
# #axes[1].plot(time_axis, kurt, color="tab:orange", linewidth=1.5)
# axes[1].set_title(f"{var_name}: 4th central moment (m4)")
# #axes[1].set_title(f"{var_name}: kurtosis")
# axes[1].set_ylabel("m4")
# #axes[1].set_ylabel("kurtosis")
# axes[1].set_xlabel("time index")
# axes[1].grid(True, alpha=0.3)
# #####
# plt.tight_layout()
# plt.show()

# #####
# #second central moment: mean of squared deviations from the mean - choose o or l
# m2 = np.mean((state_var - mean[:, np.newaxis])**2, axis=1)
# #m2 = np.mean((np.log10(state_var) - mean[:, np.newaxis])**2, axis=1)
# #std = np.std(state_var, axis=1)
# #std = np.std(np.log10(state_var), axis=1)
# #time axis
# #time_axis = np.arange(len(mean))
# time_axis = np.arange(state_var.shape[0])
# #####
# #plot mean and 2nd moment
# fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
# axes[0].plot(time_axis, mean, color="tab:blue", linewidth=1.5)
# axes[0].set_title(f"{var_name}: ensemble mean")
# axes[0].set_ylabel("mean")
# axes[0].grid(True, alpha=0.3)
# #####
# axes[1].plot(time_axis, m2, color="tab:orange", linewidth=1.5)
# #axes[1].plot(time_axis, std, color="tab:orange", linewidth=1.5)
# axes[1].set_title(f"{var_name}: second central moment (m2)")
# #axes[1].set_title(f"{var_name}: standard deviation")
# axes[1].set_ylabel("m2")
# #axes[1].set_ylabel("std")
# axes[1].set_xlabel("time index")
# axes[1].grid(True, alpha=0.3)
# #####
# plt.tight_layout()
# plt.show()

######