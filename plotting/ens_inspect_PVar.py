from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

ds = Dataset("..\\data\\processed_ensemble.nc")

# all_state_variables_names = ["N1_p", "N3_n", "N4_n", "N5_s", "O2_bot", "O2_o", "P1_Chl", "P1_Chl_vert", "P2_Chl", "P2_Chl_vert", "P3_Chl", "P3_Chl_vert", "P4_Chl", "P4_Chl_vert", "Z4_c", "Z4_c_vert", "Z5_c", "Z5_c_vert", "Z6_c", "Z6_c_vert"]
# num_state_variables = len(all_state_variables_names)

start_time = 365 + 334
length_time = int(15*365.25)



vert_P_variables = ["P1_Chl_vert", "P2_Chl_vert", "P3_Chl_vert", "P4_Chl_vert"]
p1_v = ds.variables["P1_Chl_vert"][start_time:]
p2_v = ds.variables["P2_Chl_vert"][start_time:]
p3_v = ds.variables["P3_Chl_vert"][start_time:]
p4_v = ds.variables["P4_Chl_vert"][start_time:]

all_p_v = [p1_v, p2_v, p3_v, p4_v]
p_v_sum = np.sum(all_p_v, axis=0)
mean = np.mean(p_v_sum, axis=0)
# v_min = np.min(mean)
# v_max = np.max(mean)
# print(v_min)
# print(v_max)

# surf_P_variables = ["P1_Chl", "P2_Chl", "P3_Chl", "P4_Chl"]
# p1_s = ds.variables["P1_Chl"][start_time:]
# p2_s = ds.variables["P2_Chl"][start_time:]
# p3_s = ds.variables["P3_Chl"][start_time:]
# p4_s = ds.variables["P4_Chl"][start_time:]

# all_p_s = [p1_s, p2_s, p3_s, p4_s]
# p_s_sum = np.sum(all_p_s, axis=0)
# mean = np.mean(p_s_sum, axis=0)
# s_min = np.min(mean)
# s_max = np.max(mean)
# print(s_min)
# print(s_max)


plt.figure(figsize=(12,5))
plt.hist(
    mean,
    bins=100,
    edgecolor="black",
    linewidth=0.6,
    color="lightblue",
    alpha=0.8
)
#plt.ylim(0, 50)
#plt.xlim(0, 1.3)
plt.xlabel("Time-mean of total P_Chl_vert")
plt.ylabel("Number of ensemble members")
plt.title("Distribution of ensemble time-means")
plt.show()