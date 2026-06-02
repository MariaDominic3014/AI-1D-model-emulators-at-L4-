import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import pandas as pd

plot_var_name = {
    "P1_Chl": "Surface diatoms chlorophyll",
    "P2_Chl": "Surface nanophyto chlorophyll",
    "P3_Chl": "Surface picophyto chlorophyll",
    "P4_Chl": "Surface microphyto chlorophyll",
    "Z4_c": "Surface mesozoo carbon",
    "Z5_c": "Surface microzoo carbon",
    "Z6_c": "Surface hetero flagellates carbon",
    "N3_n": "Surface nitrate",
    "N1_p": "Surface phosphate",
    "N4_n": "Surface ammonium",
    "N5_s": "Surface silicate",
    "O2_o": "Surface oxygen",
    "P1_Chl_vert": "Verically averaged diatoms chlorophyll",
    "P2_Chl_vert": "Verically averaged nanophyto chlorophyll",
    "P3_Chl_vert": "Verically averaged picophyto chlorophyll",
    "P4_Chl_vert": "Verically averaged microphyto chlorophyll",
    "Z4_c_vert": "Verically averaged mesozoo carbon",
    "Z5_c_vert": "Verically averaged microzoo carbon",
    "Z6_c_vert": "Verically averaged hetero flagellates carbon",
    "O2_bot": "O2 Bottom",
    "light_parEIR": "Light",
    "u10": "Zonal Wind",
    "v10": "Meridional Wind",
    "precip": "Precipitation",
    "heat": "Heat",
    "temp": "Temperature",
    "salt": "Salinity",
    "mld_surf": "Mixed layer depth",
}
# =========================================================
# CONFIG
# =========================================================

TARGET = "O2_bot"   # must match filename convention
#N_ENSEMBLES = 15

files = sorted(glob.glob(f"Sensitivities_{TARGET}_ens_*.nc"))

# =========================================================
# LOAD DATA
# =========================================================

def load_ensemble(file):
    nc = Dataset(file)

    grad_bio = nc.variables["grad_bio"][:]   # (time, lag, bio_var)
    grad_forcing = nc.variables["grad_forcing_history"][:]  # (time, lag, forcing_var)
    grad_future_forcing = nc.variables["grad_forcing_future"][:] # (time, forcing_var)

    bio_vars = nc.bio_variables.split(",")
    forcing_vars = nc.forcing_variables.split(",")
    #grad_future_forcing = nc.forcing_variables.split(",")

    nc.close()
    return grad_bio, grad_forcing, grad_future_forcing, bio_vars, forcing_vars

bio_list = []
forcing_list = []
future_forcing_list = []

for f in files:
    gb, gf, gff, bio_vars, forcing_vars = load_ensemble(f)
    bio_list.append(gb)
    forcing_list.append(gf)
    future_forcing_list.append(gff)

bio = np.mean(np.stack(bio_list), axis=0)
forcing = np.mean(np.stack(forcing_list), axis=0)
future_forcing = np.mean(np.stack(future_forcing_list), axis=0)

# absolute sensitivities
bio = np.abs(bio)
forcing = np.abs(forcing)
future_forcing = np.abs(future_forcing)
print(bio.shape, forcing.shape, future_forcing.shape)

nt, lag, nb = bio.shape
_, _, nf = forcing.shape

time = np.arange(nt)
lag_axis = np.arange(lag)

# =========================================================
# 1. HOVMÖLLER (time vs input variable)
# =========================================================

#bio_hov = bio.mean(axis=1)        # (time, bio_var)
#forcing_hov = forcing.mean(axis=1)  # (time, forcing_var)
bio_hov = bio.max(axis=1)        # (time, bio_var)
forcing_hov = forcing.max(axis=1)  # (time, forcing_var)

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

vmax_comb = np.max([np.max(bio_hov), np.max(forcing_hov)])
st,en=4400,8000


lookback=30

start_time = 365 + 334  # account for spin up (nearly 2 years)
length_training_time = int(12*365.25)  # length of the time series for training and validation data
length_testing_time = 8734 - start_time#671 + int(15*365.25)  # length of the time series for training and validation data

# Build a date axis for the full testing period
# Dataset origin: 1 Jan 2002, daily time steps
dataset_origin = pd.Timestamp("2002-01-01")
start_date = dataset_origin + pd.Timedelta(days=start_time)  # ~2 Dec 2003

dates = pd.date_range(start=start_date, periods=length_testing_time, freq="D")
dates_plot = dates[lookback:]   # matches the nt = length_testing_time - lookback axis

# For the sliced region used in the Hovmöller (st:en)
dates_slice = dates_plot[st:en]

# Tick positions and labels: one tick per year, on 1 Jan
year_ticks = []
year_labels = []
for year in range(dates_slice[0].year, dates_slice[-1].year + 1):
    jan1 = pd.Timestamp(f"{year}-01-01")
    if jan1 >= dates_slice[0] and jan1 <= dates_slice[-1]:
        idx = (jan1 - dates_slice[0]).days
        year_ticks.append(idx)
        year_labels.append(str(year))



im0 = axes[0].imshow(
    bio_hov[st:en, :].T,
    aspect="auto",
    origin="lower",
    interpolation='none', cmap="Greens",# vmax=vmax_comb
)
axes[0].set_title(f"Biological input sensitivities for {plot_var_name[TARGET]} (mean over input window)")
#axes[0].set_ylabel("Bio variables")
axes[0].set_yticks(np.arange(nb))
axes[0].set_yticklabels([plot_var_name[v] for v in bio_vars])
axes[0].invert_yaxis()

plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(
    forcing_hov[st:en, :].T,
    aspect="auto",
    origin="lower",
    interpolation='none', cmap="Reds",# vmax=vmax_comb
)
axes[1].set_title(f"Forcing sensitivities for {plot_var_name[TARGET]} (mean over input window)")
#axes[1].set_ylabel("Forcing variables")
axes[1].set_yticks(np.arange(nf))
axes[1].set_yticklabels([plot_var_name[v] for v in forcing_vars])


plt.colorbar(im1, ax=axes[1])


im2 = axes[2].imshow(
    future_forcing[st:en, :].T,
    aspect="auto",
    origin="lower",
    interpolation='none', cmap="Reds",# vmax=vmax_comb
)
axes[2].set_title(f"Future forcing sensitivities for {plot_var_name[TARGET]}")
#axes[2].set_ylabel("Forcing variables")
axes[2].set_yticks(np.arange(nf))
axes[2].set_yticklabels([plot_var_name[v] for v in forcing_vars])


plt.colorbar(im2, ax=axes[2])


for ax in axes:
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels, rotation=45, ha="right")

axes[2].set_xlabel("Year")

plt.tight_layout()
plt.savefig(f"../plots/{TARGET}_to_hovmoller.png")
#plt.show()
plt.close()

# =========================================================
# 2. LAG STRUCTURE PLOTS (one per variable)
# =========================================================

def latex_lag_structure(data, suffix):
    headers = ["Variable name", "50\\%", "90\\%", "95\\%"]

    latex = []
    latex.append(r"\begin{tabular}{lccc}")
    latex.append(r"\hline")
    latex.append(" & ".join(headers) + r" \\")
    latex.append(r"\hline")

    for row in data:
        latex.append(
            f"{row[0]} & {row[1]} & {row[2]} & {row[3]} \\\\"
        )

    latex.append(r"\hline")
    latex.append(r"\end{tabular}")

    with open(f"../plots/lag_table_{suffix}.tex", "w") as f:
        f.write("\n".join(latex))

def consise_lag_structure(data, suffix):

    data = data[::-1]

    legend_handles = [
        Line2D([0], [0],
            marker='o', color='none', markerfacecolor='k', markersize=20, label='50%'
        ),
        Line2D([0], [0],
            marker=r'$\otimes$', color='k', markersize=15, linestyle='None', label='90%'
        ),
        Line2D([0], [0],
            marker='o', color='none', markerfacecolor='w', markeredgecolor='k', markersize=5, label='95%'
        )
    ]


    fig, ax = plt.subplots(figsize=(16, 8))

    for i, (name, p50, p90, p95) in enumerate(data):
        ax.hlines(i, 0, p95, lw=2)
        ax.plot(p50, i, 'o', markersize=20,  c="k", label='50%' if i == 0 else "")

        ax.plot(p90, i, 'o', markersize=15,  c="k", label='90%' if i == 0 else "")
        ax.plot(p90, i, 'o', markersize=13,  c="w")
        ax.plot(p90, i, 'x', markersize=12,  c="k")

        ax.plot(p95, i, 'o', markersize=5 ,  c="k", label='95%' if i == 0 else "")
        ax.plot(p95, i, 'o', markersize=3 ,  c="w")

    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([d[0] for d in data], fontsize=12)
    ax.set_xlabel('Days before forecast', fontsize=12)
    ax.legend(title="Threshold\n(as % of peak sensitivity)", title_fontsize=12, labelspacing=2, handles=legend_handles)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    plt.xlim(0, None)
    plt.xticks(fontsize=12)
    plt.title(f"Sensitivity timescales for {plot_var_name[TARGET]}", fontsize=16)

    plt.tight_layout()
    plt.savefig(f"../plots/{TARGET}_sensitivity_ranges_{suffix}.png")
    plt.close()



def plot_lag_structure(data, var_names, title_prefix):

    mean_curve = data.mean(axis=0)  # (lag, var)

    thresholds = [0.5, 0.1, 0.05]
    threshold_markers = ["D", "s", "*"]
    threshold_days = []

    for v in range(len(var_names)):
        threshold_days.append([plot_var_name[var_names[v]]])

        plt.figure(figsize=(7, 4))

        # grey ensemble/time curves
        for t in range(0, nt, max(1, nt // 200)):
            plt.plot(
                lag_axis,
                data[t, :, v],
                color="grey",
                alpha=0.15
            )

        # mean structure
        plt.plot(
            lag_axis,
            mean_curve[:, v],
            color="black",
            linewidth=2
        )

        # get 95% threshold
        for tidx, thresh in enumerate(thresholds):
            lags = np.arange(len(mean_curve[:, v]))
            threshold = np.max(mean_curve[:, v]) * thresh
            cross_idx = np.where((mean_curve[:, v][:-1] < threshold) & (mean_curve[:, v][1:] >= threshold))[0]
            i = cross_idx[0]  # first crossing
            x_cross = lags[i]
            y_cross = mean_curve[:, v][i]
            plt.scatter(x_cross, y_cross, marker=threshold_markers[tidx], zorder=100, c="b", s=100, label=str((1 - thresh)*100)+"%")
            threshold_days[v].append(int(30-x_cross))
        
        print(threshold_days[v])



        plt.legend(loc="upper left", title="Threshold\n(as % of peak sensitivity)")
        plt.title(f"Sensitivity of {plot_var_name[TARGET]} to {plot_var_name[var_names[v]]}")
        plt.xlabel("Input window offset (days)")
        plt.ylabel("Sensitivity")
        plt.xticks([t for t in range(30)], [t for t in range(-30, 0)])
        plt.tight_layout()
        plt.savefig(f"../plots/{TARGET}_window_input_{var_names[v]}.png")
        #plt.show()
        plt.close()

    latex_lag_structure(threshold_days, title_prefix)
    consise_lag_structure(threshold_days, title_prefix)


plot_lag_structure(bio, bio_vars, "bgc")
plot_lag_structure(forcing, forcing_vars, "forcing")



bio_time_series = bio.mean(axis=1)        # (time, bio_var)
forcing_time_series = forcing.mean(axis=1)  # (time, forcing_var)

# BIO variables only
plt.figure(figsize=(12, 5))

for i in range(nb):
    plt.plot(time[st:en], bio_time_series[st:en, i], label=f"{plot_var_name[bio_vars[i]]}")

plt.title(f"Biological input sensitivities (input window-averaged) to {plot_var_name[TARGET]}")
plt.xlabel("Time index")
plt.ylabel("|sensitivity|")
plt.legend()
plt.tight_layout()
plt.savefig(f"../plots/{TARGET}_timeseries_sensitivity_bio.png")
plt.close()

# FORCING variables only
plt.figure(figsize=(12, 5))

for i in range(nf):
    plt.plot(time[st:en], forcing_time_series[st:en, i], label=f"{plot_var_name[forcing_vars[i]]}")

plt.title(f"Forcing input sensitivities (input window-averaged) to {plot_var_name[TARGET]}")
plt.xlabel("Time index")
plt.ylabel("|sensitivity|")
plt.legend()
plt.tight_layout()
plt.savefig(f"../plots/{TARGET}_timeseries_sensitivity_forcing.png")
plt.close()
