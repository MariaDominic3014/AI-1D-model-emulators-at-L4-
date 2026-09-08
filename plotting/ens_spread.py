from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

ds = Dataset("..\\data\\processed_ensemble.nc")


#-------------------
### MEDIAN + QUARTILES + SPREAD ###
#-------------------
var_names = ["N1_p", "N4_n", "O2_bot", "O2_o", "P1_Chl", "P1_Chl_vert", "Z4_c", "Z4_c_vert"]

start_time = 365 + 334  # account for spin up (nearly 2 years)
length_time = int(15*365.25)  # length of the time series for training and validation data

for name in var_names:
    var = ds.variables[name][start_time:start_time+length_time]

    median = np.median(var, axis=1)
    q1 = np.quantile(var, 0.25, axis=1)
    q3 = np.quantile(var, 0.75, axis=1)
    ensemble_min = np.min(var, axis=1)
    ensemble_max = np.max(var, axis=1)

    x = np.arange(len(median))
    plt.figure(figsize=(14,6))
    plt.plot(x, median, label="Ensemble median")

    # #median and quartiles and full spread
    plt.fill_between(x, q1, q3, alpha=0.3, label="25-75 percentile")
    plt.fill_between(x, ensemble_min, ensemble_max, alpha=0.15, label="Full ensemble spread")

    plt.xlabel("Time (days)")
    plt.ylabel(name)
    plt.title(f"{name} Ensemble Median, Q1, Q3 and full Spread")
    plt.legend()

    plt.savefig(f"{name}_ens_quartiles.png", dpi=200)
    plt.close()



#-------------------
### MEAN + SD ###
#-------------------
#var_name = "N1_p"
#var_name = "N4_n"
#var_name = "O2_bot"
#var_name = "O2_o"
#var_name = "P1_Chl"
#var_name = "P1_Chl_vert"
#var_name = "Z4_c"
#var_name = "Z4_c_vert"

#start_time = 365 + 334  # account for spin up (nearly 2 years)
#length_time = int(15*365.25)  # length of the time series for training and validation data

#var = ds.variables[var_name][start_time:start_time+length_time]

# #Mean and standard deviation
# mean = np.mean(var, axis=1)
# std = np.std(var, axis=1)
# plt.figure(figsize=(12,5))
#plt.plot(mean, label="Ensemble mean")

# plt.fill_between(
#     np.arange(len(mean)),
#     mean - std,
#     mean + std,
#     alpha=0.3,
#     label="±1 standard deviation"
# )
# plt.xlabel("Time (days)")
# plt.ylabel(f"{ds.variables[var_name].name}")
# plt.title(f"{ds.variables[var_name].name} Ensemble Mean and +-1SD")
# plt.legend()
# plt.show()