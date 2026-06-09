import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import pandas as pd

plot_var_name = {
    "P1_Chl": "Surface diatoms chl",
    "P2_Chl": "Surface nanophyto chl",
    "P3_Chl": "Surface picophyto chl",
    "P4_Chl": "Surface microphyto chl",
    "Z4_c": "Surface mesozoo carbon",
    "Z5_c": "Surface microzoo carbon",
    "Z6_c": "Surface hetero flagellates carbon",
    "N3_n": "Surface nitrate",
    "N1_p": "Surface phosphate",
    "N4_n": "Surface ammonium",
    "N5_s": "Surface silicate",
    "O2_o": "Surface oxygen",
    "P1_Chl_vert": "Vert. av. diatoms chl",
    "P2_Chl_vert": "Vert. av. nanophyto chl",
    "P3_Chl_vert": "Vert. av. picophyto chl",
    "P4_Chl_vert": "Vert. av. microphyto chl",
    "Z4_c_vert": "Vert. av. mesozoo carbon",
    "Z5_c_vert": "Vert. av. microzoo carbon",
    "Z6_c_vert": "Vert. av. hetero flagellates carbon",
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

TARGET = "P1_Chl"   # must match filename convention
#N_ENSEMBLES = 15

files = sorted(glob.glob(f"Sensitivities_{TARGET}_ens_*.nc"))

# =========================================================
# LOAD DATA
# =========================================================

raw_pred_data = Dataset("Predicted_LSTM_vert_ens_dm_imp_full_0.nc")

for v in raw_pred_data.variables:
    print(v)

diatoms_pred = raw_pred_data["predicted_P1_Chl"]
diatoms_test = raw_pred_data["test_P1_Chl"]

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

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
    gridspec_kw={'height_ratios': [3, 1, 1]})

vmax_comb = np.max([np.max(bio_hov), np.max(forcing_hov), np.max(future_forcing)])
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


# Tick positions and labels: one tick per month
month_starts = pd.date_range(
    dates_slice[0].replace(day=1),
    dates_slice[-1],
    freq="MS"
)

month_ticks = [(d - dates_slice[0]).days for d in month_starts]
month_labels = [d.strftime("%b %Y") for d in month_starts]

im0 = axes[0].imshow(
    bio_hov[st:en, :].T,
    aspect="auto",
    origin="lower",
    interpolation='none', cmap="Greens", vmax=vmax_comb
)
axes[0].set_title(f"Historical biological input sensitivities for {plot_var_name[TARGET]} (max from input window)")
#axes[0].set_ylabel("Bio variables")
axes[0].set_yticks(np.arange(nb))
axes[0].set_yticklabels([plot_var_name[v] for v in bio_vars])
axes[0].invert_yaxis()

plt.colorbar(im0, ax=axes[0], fraction=0.04)

im1 = axes[1].imshow(
    forcing_hov[st:en, :].T,
    aspect="auto",
    origin="lower",
    interpolation='none', cmap="Reds", vmax=vmax_comb
)
axes[1].set_title(f"Historical forcing sensitivities for {plot_var_name[TARGET]} (max from input window)")
#axes[1].set_ylabel("Forcing variables")
axes[1].set_yticks(np.arange(nf))
axes[1].set_yticklabels([plot_var_name[v] for v in forcing_vars])


plt.colorbar(im1, ax=axes[1], fraction=0.04, aspect=7)


im2 = axes[2].imshow(
    future_forcing[st:en, :].T,
    aspect="auto",
    origin="lower",
    interpolation='none', cmap="Reds", vmax=vmax_comb
)
axes[2].set_title(f"Current forcing sensitivities for {plot_var_name[TARGET]}")
#axes[2].set_ylabel("Forcing variables")
axes[2].set_yticks(np.arange(nf))
axes[2].set_yticklabels([plot_var_name[v] for v in forcing_vars])


plt.colorbar(im2, ax=axes[2], fraction=0.04, aspect=7)


for ax in axes:
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels, rotation=45, ha="right")

axes[2].set_xlabel("Year")

plt.tight_layout()
plt.savefig(f"../plots/{TARGET}_to_hovmoller.png")
#plt.show()
plt.close()


# =========================================================
# 1.5 SAME AS ABOVE BUT TUNED FOR A SINGLE YEAR
# =========================================================

fig = plt.figure(figsize=(16, 12))

gs = fig.add_gridspec(
    4, 2,
    width_ratios=[1, 0.04],
    height_ratios=[1, 4, 2, 2],
    wspace=0.05,
    hspace=0.3
)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[3, 0]),
]

vmax_comb = np.max([np.max(bio_hov), np.max(forcing_hov), np.max(future_forcing)])
st,en=7300,7700


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


# Tick positions and labels: one tick per month
month_starts = pd.date_range(
    dates_slice[0].replace(day=1),
    dates_slice[-1],
    freq="MS"
)

month_ticks = [(d - dates_slice[0]).days for d in month_starts]
month_labels = [d.strftime("%b %Y") for d in month_starts]

t = np.arange(en - st)

axes[0].plot(t,diatoms_pred[st:en],label="Predicted",c="k")
axes[0].plot(t,diatoms_test[st:en],"--",label="Target",c="k")

axes[0].set_title("Surface Diatoms")
axes[0].set_ylabel("mg/m³")
axes[0].legend()
axes[0].set_xlim(0, en - st - 1)

im0 = axes[1].imshow(
    bio_hov[st:en, :].T,
    aspect="auto",
    origin="lower",
    interpolation='none', cmap="Greens", vmax=vmax_comb
)
axes[1].set_title(f"Historical biological input sensitivities for {plot_var_name[TARGET]} (max from input window)")
#axes[0].set_ylabel("Bio variables")
axes[1].set_yticks(np.arange(nb))
axes[1].set_yticklabels([plot_var_name[v] for v in bio_vars])
axes[1].invert_yaxis()

cax0 = fig.add_subplot(gs[1, 1])
fig.colorbar(im0, cax=cax0)
#plt.colorbar(im0, ax=axes[1])

im1 = axes[2].imshow(
    forcing_hov[st:en, :].T,
    aspect="auto",
    origin="lower",
    interpolation='none', cmap="Reds", vmax=vmax_comb
)
axes[2].set_title(f"Historical forcing sensitivities for {plot_var_name[TARGET]} (max from input window)")
#axes[1].set_ylabel("Forcing variables")
axes[2].set_yticks(np.arange(nf))
axes[2].set_yticklabels([plot_var_name[v] for v in forcing_vars])

cax1 = fig.add_subplot(gs[2, 1])
fig.colorbar(im1, cax=cax1)
#plt.colorbar(im1, ax=axes[2])


im2 = axes[3].imshow(
    future_forcing[st:en, :].T,
    aspect="auto",
    origin="lower",
    interpolation='none', cmap="Reds", vmax=vmax_comb
)
axes[3].set_title(f"Current forcing sensitivities for {plot_var_name[TARGET]}")
#axes[2].set_ylabel("Forcing variables")
axes[3].set_yticks(np.arange(nf))
axes[3].set_yticklabels([plot_var_name[v] for v in forcing_vars])

cax2 = fig.add_subplot(gs[3, 1])
fig.colorbar(im2, cax=cax2)
#plt.colorbar(im2, ax=axes[3])


for aidx, ax in enumerate(axes):
    ax.set_xticks(month_ticks)
    ax.set_xlim(0, en - st - 1)
    if aidx == 3:
        ax.set_xticklabels(month_labels, rotation=45, ha="right")
    else:
        ax.set_xticklabels(["" for aslokdjnfkasjnbf in month_labels])


#axes[3].set_xlabel("Time")

plt.tight_layout()
fig.subplots_adjust(left=0.18, right=0.92, top=0.95, bottom=0.08)
plt.savefig(f"../plots/{TARGET}_to_hovmoller_singleyear.png")
#plt.show()
plt.close()



# =========================================================
# 1.75 SIMILAR TO ABOVE, BUT AGGREGATE BAR PLOTS BETWEEN YEARS for each season
# =========================================================

fig = plt.figure(figsize=(18, 26))

gs = fig.add_gridspec(
    2, 4,
    height_ratios=[0.5, 3],
    wspace=0.2,
    hspace=0.1
)

axes = [
    fig.add_subplot(gs[0, :]),   # top row, full width
    fig.add_subplot(gs[1, 0]),   # bottom row, col 0
    fig.add_subplot(gs[1, 1]),   # bottom row, col 1
    fig.add_subplot(gs[1, 2]),   # bottom row, col 2
    fig.add_subplot(gs[1, 3]),   # bottom row, col 3
]

vmax_comb = np.max([np.max(bio_hov), np.max(forcing_hov)])

st,en=4750,4750+365

season_offsets = [
    [-60, 30],
    [ 30, 90],
    [90, 210],
    [210, 300]
]

season_cols = ["tab:blue", "tab:orange", "tab:pink", "tab:cyan"]
season_title = ["Light-limited", "Spring bloom", "Nutrient limited", "Late summer bloom"]

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


# Tick positions and labels: one tick per month
month_starts = pd.date_range(
    dates_slice[0].replace(day=1),
    dates_slice[-1],
    freq="MS"
)

month_ticks = [(d - dates_slice[0]).days for d in month_starts]
month_labels = [d.strftime("%b %Y") for d in month_starts]

t = np.arange(en - st)

years_sampled = 8
diatoms_timeseries = []
for i in range(years_sampled):
    diatoms_timeseries.append(diatoms_pred[st+365*i:en+365*i])
diatoms_timeseries = np.array(diatoms_timeseries)
diatoms_timeseries_mean = np.mean(diatoms_timeseries, axis=0)
diatoms_timeseries_std = np.std(diatoms_timeseries, axis=0)



axes[0].errorbar(t,diatoms_timeseries_mean, diatoms_timeseries_std, label="Predicted",c="k")
#axes[0].plot(t,diatoms_test[st:en],"--",label="Target",c="k")
for s, szn in enumerate(season_offsets):
    axes[0].axvspan(
        max(0, szn[0]),
        szn[1],
        color=season_cols[s],
        #alpha=0.2
    )
    

axes[0].axvspan(
    season_offsets[-1][1],
    len(diatoms_timeseries_mean),
    color=season_cols[0],
    #alpha=0.2
)

axes[0].set_title("Surface diatoms behaviour (2017-2025)")
axes[0].set_ylabel("mg/m³")
#axes[0].legend()
axes[0].set_xlim(0, en - st - 1)

for szn_s, szn_e in season_offsets:
    axes[0].plot([szn_e, szn_e], [0, np.max(diatoms_pred[st:en])], c="k", alpha=0.2, linestyle="--")


axes[0].set_xticks(month_ticks)
axes[0].set_xlim(0, en - st - 1)
axes[0].set_xticklabels([i[:-5] for i in month_labels], rotation=45, ha="right")


max_sensitivity_bio          = np.zeros((years_sampled, 4, 20))
max_sensitivity_force        = np.zeros((years_sampled, 4, 8))
max_sensitivity_future_force = np.zeros((years_sampled, 4, 8))
for year_no in range(years_sampled):
    st_y = st+365*year_no
    en_y = en+365*year_no 
    season_num = 0
    for szn_st, szn_en in season_offsets:
        bio_data = bio_hov[st_y+szn_st:st_y+szn_en, :].T
        force_data = forcing_hov[st_y+szn_st:st_y+szn_en, :].T
        future_force_data = future_forcing[st_y+szn_st:st_y+szn_en, :].T
        max_sensitivity_bio[year_no, season_num, :]          = np.max(bio_data         , axis=1)
        max_sensitivity_force[year_no, season_num, :]        = np.max(force_data       , axis=1)
        max_sensitivity_future_force[year_no, season_num, :] = np.max(future_force_data, axis=1)
        season_num += 1


combined_max = np.concatenate(
    [
        max_sensitivity_bio,
        max_sensitivity_force,
        max_sensitivity_future_force
    ],
    axis=2
)

combined_means = np.mean(combined_max, axis=0)
combined_std = np.std(combined_max, axis=0)
print(combined_means.shape)

# combine labels
combined_labels = []

[combined_labels.append(plot_var_name[v] + " historic") for v in bio_vars]
[combined_labels.append(plot_var_name[v] + " historic") for v in forcing_vars]
[combined_labels.append(plot_var_name[v] + " current") for v in forcing_vars]

print(combined_labels)

for i in range(1, 5):
    im0 = axes[i].barh(combined_labels, combined_means[i-1, :], xerr=combined_std[i-1, :], color=season_cols[i-1])
    axes[i].set_yticks(np.arange(len(combined_labels)))
    axes[i].set_title(season_title[i-1])
    axes[i].grid(axis='x')

axes[1].set_yticklabels(combined_labels, ha="right")
axes[2].set_yticklabels(["" for i in combined_labels], rotation=30, ha="right")
axes[3].set_yticklabels(["" for i in combined_labels], rotation=30, ha="right")
axes[4].set_yticklabels(["" for i in combined_labels], rotation=30, ha="right")


for i in range(1, 5):
    axes[i].set_xlim(0, np.max(combined_max))

fig.supxlabel("Maximum |sensitivity| within season", y=0.05)
plt.tight_layout()
fig.subplots_adjust(left=0.18, right=0.92, top=0.95, bottom=0.08)
plt.savefig(f"../plots/{TARGET}_to_barplot.png")
plt.show()
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
            marker=r'$\otimes$', color='k', markersize=15, linestyle='None', label='10%'
        ),
        Line2D([0], [0],
            marker='o', color='none', markerfacecolor='w', markeredgecolor='k', markersize=5, label='5%'
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
    plt.xlim(0, 30)
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
            plt.scatter(x_cross, y_cross, marker=threshold_markers[tidx], zorder=100, c="b", s=100, label=str((thresh)*100)+"%")
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
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

k = 8

# Max value of each variable over the plotted window
max_vals = np.max(bio_time_series[st:en, :], axis=0)

# Indices of top-k variables
topk_idx = np.argsort(max_vals)[-k:][::-1]
topk_set = set(topk_idx)

# Plot all others in grey with a single legend entry
other_label_added = False

for i in range(nb):

    if i in topk_idx:
        continue

    ax1.plot(
        time[st:en],
        bio_time_series[st:en, i],
        color="lightgrey",
        alpha=0.3,
        label="Other" if not other_label_added else None
    )
    other_label_added = True

# Plot top-k with distinct colours and labels
for i in topk_idx:
    ax1.plot(
        time[st:en],
        bio_time_series[st:en, i],
        linewidth=2,
        label=plot_var_name[bio_vars[i]]
    )

ax1.set_ylabel("|sensitivity|")

# Bottom subplot
ax2.plot(
    time[st:en],
    diatoms_pred[st:en],
    label="Predicted diatoms",
    linestyle="-",
    c="k"
)
ax2.plot(
    time[st:en],
    diatoms_test[st:en],
    label="Test diatoms",
    linestyle="--",
    c="k"
)
ax2.set_ylabel("Surface Diatoms")
ax2.set_xlabel("Time index")

# Combine legends from both subplots
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.suptitle(f"Biological input sensitivities (input window-averaged) to {plot_var_name[TARGET]}")
ax2.set_xlim(st, en)


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
