from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

ds = Dataset("..\\data\\processed_ensemble.nc")


########## ORIGINAL VALUES #########
#var_name = "N1_p"
#var_name = "N4_n"
#var_name = "O2_bot"
#var_name = "O2_o"
var_name = "P1_Chl"
#var_name = "P1_Chl_vert"
#var_name = "Z4_c"
#var_name = "Z4_c_vert"

var = ds.variables[var_name]

# training and validation period begins on day 699, after ~2yrs (365+334) of spin up and lasts until ~ day 6177
# already done 1095, 1195, 1245, 1345
timesteps = [700, 825, 900, 1000, 1400, 1475, 1600, 1675, 1775, 1870, 1970, 2025, 2150, 2225, 2325, 2400]

for t in timesteps:
    values = var[t, :]
    plt.figure(figsize=(6,4))

    # calculating ensemble stats
    # print(f"Mean (original): {np.mean(values)}")
    # print(f"Std (original): {np.std(values)}")
    # print(f"Min (original): {np.min(values)}")
    # print(f"Max (original): {np.max(values)}")

    # #HISTOGRAM
    plt.hist(values, bins=50)   #gives you 22 members per bin
    plt.xlabel(f"{var.name}")
    plt.ylabel("Number of ensemble members")
    plt.title(f"{var.name} distribution at timestep {t}")
    plt.savefig(f"{var.name}_timestep{t}_hist.png", dpi=200)

    plt.close()




######## LOG-TRANSFORMATION ########
#t = 3000   # timestep to look at
#var_names = ["N1_p", "N4_n", "O2_bot", "O2_o", "P1_Chl", "P1_Chl_vert", "Z4_c", "Z4_c_vert"]

##### min/max of all log variables #####
# global_min = np.inf
# global_max = -np.inf

# for name in var_names:
#     values = np.log10(ds.variables[name][t, :])
#     global_min = min(global_min, np.min(values))
#     global_max = max(global_max, np.max(values))

# print(global_min)
# print(global_max)
#########################################

# #Plot histogram of the log-transformed values
# for name in var_names:
#     plt.figure(figsize=(6,4))

#     var = ds.variables[name]
#     values = var[t, :]

#     # #Take the base-10 logarithm
#     log_values = np.log10(values)

#     # #Calculate ensemble statistics on the log-transformed data
#     # print(f"{name} Mean (log10): {np.mean(log_values)}")
#     # print(f"{name} Std (log10): {np.std(log_values)}")
#     # print(f"{name} Min (log10): {np.min(log_values)}")
#     # print(f"{name} Max (log10): {np.max(log_values)}")

#     plt.hist(log_values, bins=30)
#     plt.xlabel(f"xlim log10({name})")
#     plt.ylabel("Number of ensemble members")
#     plt.title(f"xlim log10({name}) distribution at timestep {t}")

#     plt.xlim(-7, 3)     # optional XLIM
#     plt.show()
#     plt.savefig(f"{name}_timestep{t}_histLOG10_lim.png", dpi=200)
#     plt.close()
