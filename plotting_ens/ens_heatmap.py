from netCDF4 import Dataset
import matplotlib.pyplot as plt

ds = Dataset("..\\data\\processed_ensemble.nc")


var_names = ["N1_p", "N4_n", "O2_bot", "O2_o", "P1_Chl", "P1_Chl_vert", "Z4_c", "Z4_c_vert"]

start_time = 365 + 334  # account for spin up (nearly 2 years)
length_time = int(15*365.25)  # length of the time series for training and validation data

for name in var_names:
    var = ds.variables[name][start_time:start_time+length_time]

    plt.figure(figsize=(14,6))

    plt.imshow(
        var.T,
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )

    plt.xlabel("Time (days)")
    plt.ylabel("Ensemble member")
    plt.title(f"{name}")
    plt.colorbar(label=f"{name}")
    #plt.show()
    plt.savefig(f"{name}_ens_heatmap.png", dpi=200)
    plt.close()