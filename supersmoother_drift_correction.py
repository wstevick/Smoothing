# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import make_interp_spline
from supersmoother import SuperSmoother

from PS5 import load_channel_data, my_peaks, normal_binify


# %%
channel_data = load_channel_data()
values1 = channel_data.loc[1, "values"]
times1 = channel_data.loc[1, "time"]
times1 -= times1.min()

# %%
try:
    with open("peaks1.pickle", "rb") as f:
        peaks1, hist, bins = pickle.load(f)
except FileNotFoundError:
    hist, bins = normal_binify(channel_data.loc[1, "values"])
    peaks1, _ = my_peaks(hist, bins, n=0)
    peaks1 = pd.DataFrame(
        peaks1, columns=["center", "height", "std", "offset", "slope"]
    ).sort_values("height")
    with open("peaks1.pickle", "wb") as f:
        pickle.dump((peaks1, hist, bins), f)

plt.step(bins, hist, where="mid")

# %%
sns.displot(x=times1, y=values1)
ax = plt.gca()
ax.figure.set_size_inches(10, 8)
trim_value = 130000
ax.axvline(trim_value, color="black")

before_falloff_mask = times1 < trim_value
times1 = times1[before_falloff_mask]
values1 = values1[before_falloff_mask]

# %%
hist, bins = normal_binify(channel_data.loc[1, "values"])
try:
    with open("peaks1.pickle", "rb") as f:
        peaks1, hist, bins = pickle.load(f)
except FileNotFoundError:
    peaks1, _ = my_peaks(hist, bins, n=0)
    peaks1 = pd.DataFrame(
        peaks1, columns=["center", "height", "std", "offset", "slope"]
    ).sort_values("height")
    with open("peaks1.pickle", "wb") as f:
        pickle.dump((peaks1, hist, bins), f)

plt.step(bins, hist, where="mid")

# %%
peak = peaks1.iloc[-1]
in_peak_mask = (values1 > (peak["center"] - 3 * peak["std"])) & (
    values1 < peak["center"] + 3 * peak["std"]
)
in_peak_values = values1[in_peak_mask]
in_peak_times = times1[in_peak_mask]


# %%
plt.scatter(in_peak_times, in_peak_values, marker=".", color="black")

plt.gca().figure.set_size_inches(10, 10)

for alpha in [0, 5, 8, 10]:
    model = SuperSmoother(alpha=alpha)
    model.fit(in_peak_times, in_peak_values)

    tfit = np.linspace(in_peak_times.min(), in_peak_times.max(), 1000)
    yfit = model.predict(tfit)

    plt.plot(tfit, yfit, label=f"alpha = {alpha}", lw=5)

plt.legend()


# %%
plt.scatter(in_peak_times, in_peak_values, marker=".", color="black")

plt.gca().figure.set_size_inches(10, 10)

model = SuperSmoother(alpha=0)
model.fit(in_peak_times, in_peak_values)

tfit = np.linspace(in_peak_times.min(), in_peak_times.max(), 1000)
yfit = model.predict(tfit)

plt.plot(tfit, yfit, lw=5)


# %%
y_smoothed = model.predict(in_peak_times)
y_start = y_smoothed[: int(0.05 * len(y_smoothed))].mean()
plt.scatter(
    in_peak_times, y_start + in_peak_values - y_smoothed, color="black", marker="."
)


# %%
corrections = pd.DataFrame(columns=["center", "y_start", "model"])

fig, axes = plt.subplots(nrows=5, ncols=2)
for (_, peak), ax in zip(peaks1.iloc[-10:].iterrows(), axes.flatten()):
    in_peak_mask = (values1 > (peak["center"] - 3 * peak["std"])) & (
        values1 < (peak["center"] + 3 * peak["std"])
    )
    in_peak_values = values1[in_peak_mask]
    in_peak_times = times1[in_peak_mask]
    ax.scatter(in_peak_times, in_peak_values, marker=".")

    model = SuperSmoother()
    model.fit(in_peak_times, in_peak_values)

    y_smoothed = model.predict(in_peak_times)
    y_start = y_smoothed[: int(0.05 * len(y_smoothed))].mean()
    ax.plot(in_peak_times, y_smoothed, color="black")
    corrections.loc[len(corrections)] = [
        peak["center"],
        y_smoothed[: int(0.03 * len(y_smoothed))].mean(),
        model,
    ]

fig.set_size_inches(10, 20)
fig.tight_layout()


# %%
corrections["recommended_shift"] = corrections.apply(
    lambda cor: cor["y_start"] - cor["model"].predict(times1), axis=1
)
corrections.sort_values("center", inplace=True)

recommended_shifts_by_all_peaks = np.vstack(corrections["recommended_shift"]).T
interpolated_shifts = []
for point_shifts, value in zip(recommended_shifts_by_all_peaks, values1):
    spl = make_interp_spline(corrections["center"], point_shifts)
    interpolated_shifts.append(spl(value))

interpolated_shifts = np.array(interpolated_shifts)


# %%
adjusted_values = values1 + interpolated_shifts

# %%
snrs = []
for peakid, peak in peaks1.sort_values("height", ascending=False)[:50].iterrows():
    snr = peak["center"] / peak["std"]
    # print(f"{peakid} SNR:", snr)
    snrs.append(snr)

# %%
try:
    with open("adjpeaks1.pkl", "rb") as f:
        adjpeaks1, adjhist, adjbins = pickle.load(f)
except FileNotFoundError:
    adjhist, adjbins = normal_binify(adjusted_values)
    adjpeaks1, _ = my_peaks(adjhist, adjbins, n=0)
    adjpeaks1 = pd.DataFrame(
        adjpeaks1, columns=["center", "height", "std", "offset", "slope"]
    ).sort_values("height")
    with open("adjpeaks1.pkl", "wb") as f:
        pickle.dump((adjpeaks1, adjhist, adjbins), f)
plt.step(adjbins, adjhist, where="mid")

# %%
adjpeak = adjpeaks1.loc[adjpeaks1["height"].idxmax()]
adjpeaks1
# %%'
adjsnrs = []
for adjpeakid, adjpeak in adjpeaks1.sort_values("height", ascending=False)[
    :50
].iterrows():
    adjsnr = adjpeak["center"] / adjpeak["std"]
    # print(f"{adjpeakid} SNR:",adjsnr)
    adjsnrs.append(adjsnr)


# %%
plt.hist(snrs, color="red", alpha=0.5)
plt.hist(adjsnrs, color="blue")

# density over time

# %%
file_path = "adjpeaks1.pkl"
adjpeaks1.to_pickle(file_path)
print(f"\nDataFrame successfully pickled to '{file_path}'")


plt.scatter(snrs, adjsnrs)
# plt.hist(dtwsnrs, color = 'blue', density = True, label='DTW Corrected SNR')
# plt.title("Histogram of SNR values")
plt.xlabel("Pre Corrected Value")
plt.ylabel("Supsmu Corrected Value")
plt.xscale("log")
plt.yscale("log")
xs = np.linspace(0, max(snrs))
plt.plot(xs, xs, color="black")
# plt.legend()
# %%
import statistics

adjmean = statistics.mean(adjsnrs)
print(round(adjmean, 2))

premean = statistics.mean(snrs)
print(round(premean, 2))

((adjmean - premean) / premean) * 100
