# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 21:47:07 2025

@author: mvigu
"""

import pandas as pd
from supersmoother import SuperSmoother
import pickle
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import numpy as np
from PS5 import load_channel_data, normal_binify, my_peaks,linear_guass
from scipy.optimize import curve_fit
from dtaidistance import dtw, dtw_visualisation as dtwvis
import time
import os

# %%
channel_data = load_channel_data()

with open('peaks1.pickle', 'rb') as f:
   peaks1, hist, bins = pickle.load(f)

values1 = channel_data.loc[1, 'values']
times1 = channel_data.loc[1, 'time']
times1 -= times1.min()

plt.figure(figsize=(10,5))
plt.step(bins, hist, where='mid')
plt.xlabel("Energy")
plt.ylabel("Event Density")
plt.title("Channel 1 Histogram")
plt.xlim(0,20000)
plt.show()
# %%

window_size = 2000
#window_size = 10000  # ~85
chunks = []

current = 0
end_time = 130000

while current < end_time:
    mask = (times1 >= current) & (times1 < current + window_size)
    vals = values1[mask]
    if len(vals) > 0:
        chunks.append(vals)
    current += window_size

newvalues= values1[(times1 >= 0)  & (times1 <= end_time)]
newtimes = times1[(times1 >= 0)  & (times1 <= end_time)]
# %%
import seaborn as sns


ax1 = sns.scatterplot(x=newvalues, y=newtimes / 3600, marker='.', linewidth=0, alpha=0.5)
ax1.set_xlabel("Energy")
ax1.set_ylabel("Hours")
ax1.invert_yaxis()
ax1.figure.set_size_inches(13.32, 6.81)
ax1.set_xlim(0, 20000)

window_size = 2000
current = 0
end_time = 130000

while current < end_time:
    hour = current / 3600
    ax1.axhline(y=hour, color='red', linestyle='-', alpha=0.7)
    current += window_size

ax1.figure.tight_layout()
plt.show()
# %%
global_min = newvalues.min()
global_max = newvalues.max()
nbins = len(bins)
chunk_bins = np.linspace(global_min, global_max, nbins)
hists=[]
for c in chunks:
    h, _ = np.histogram(c, bins=chunk_bins, density=True)
    hists.append(h)
hists
# %%
plt.step(chunk_bins[:-1],hists[1],where = 'mid', color ='blue',alpha = 0.7, label = 'Second Chunk')
plt.step(chunk_bins[:-1],hists[2],where = 'mid', color ='orange', alpha = 0.7,label = 'Third Chunk')
plt.legend()
# %%
def warp_to_ref(ref, h, path):
    warped = np.zeros_like(ref)
    counts = np.zeros_like(ref)

    for i_ref, i_h in path:
        warped[i_ref] += h[i_h]
        counts[i_ref] += 1

    counts[counts == 0] = 1
    return warped / counts

ref = hists[0].copy()
warped_hists = [ref]  

os.makedirs('pickles', exist_ok=True)
for i,h in enumerate(hists[1:]):
    try:
        with open(os.path.join('pickles', f'{i}_path.pickle'), 'rb') as f:
            path = pickle.load(f)
    except FileNotFoundError:
        path = dtw.warping_path(ref/ref.sum(), h/h.sum())
        with open(os.path.join('pickles', f'{i}_path.pickle'), 'wb') as f:
            pickle.dump(path, f)

    h_warped = warp_to_ref(ref, h, path)
    
    ref += h_warped

    warped_hists.append(h_warped)


big_hist = np.sum(warped_hists, axis=0)
big_hist = big_hist / big_hist.sum()
# %%
bin_centers = 0.5 * (bins[:-1] + bins[1:])
plt.figure(figsize=(10, 5))
plt.step(bin_centers, big_hist, where='mid')
plt.xlabel("Energy")
plt.ylabel("Density")
plt.title("DTW Corrected Histogram")
plt.xlim(0,20000)
plt.show()

# %%
plt.step(bins[:-1],warped_hists[0],where = 'mid', color ='k', alpha = .7 ,label='Warped Second Chunk')
plt.step(bins[:-1],warped_hists[1],where = 'mid', color ='orange', label='Warped Third Chunk')
plt.legend()
#%%
plt.step(bins[:-1],hists[2],where = 'mid', color ='orange', alpha = 0.7, label='Original Third Chunk')
# %%
plt.step(bins[:-1],hists[2],where = 'mid', color ='orange', alpha = 0.7, label='Original Third Chunk')
plt.step(bins[:-1],warped_hists[1],where = 'mid', color ='b',alpha = 0.5, label='Warped Third Chunk')
plt.legend()

# %%
plt.step(bins[:-1],hists[1],where = 'mid', color ='k', alpha = .5, label='Original Second Chunk')
plt.step(bins[:-1],warped_hists[0],where = 'mid', color ='orange', alpha = .5, label='Warped Second Chunk')
plt.legend()

# %%
dtwpeaks, _ = my_peaks(big_hist, bin_centers, n=0)
# %%
dtwpeaks = pd.DataFrame(dtwpeaks, columns=['center', 'height', 'std', 'offset', 'slope'])
dtwpeak = dtwpeaks.loc[dtwpeaks['height'].idxmax()]
dtwpeak

# %%
dtwsnrs = []
for dtwpeakid, dtwpeak in dtwpeaks.sort_values('height', ascending=False)[:50].iterrows():
    dtwsnr = dtwpeak['height'] / dtwpeak['std']
    #print(f"{adjpeakid} SNR:",adjsnr)
    dtwsnrs.append(dtwsnr)
dtwsnrs
# %%
snrs = []
for peakid, peak in peaks1.sort_values('height', ascending=False)[:50].iterrows():
    snr = peak['height'] / peak['std']
    #print(f"{peakid} SNR:", snr)
    snrs.append(snr)

# %%
plt.scatter(snrs, dtwsnrs)
#plt.hist(dtwsnrs, color = 'blue', density = True, label='DTW Corrected SNR')
plt.title('Scatter Plot of SNR values')
plt.xlabel('Pre Corrected Value')
plt.ylabel("DTW corrected Value")
plt.xscale('log')
plt.yscale('log')
xs=np.linspace(0,max(snrs))
plt.plot(xs,xs, color ='black')
plt.grid(False)
#plt.legend()

# %%

# %%

import statistics
dtwmean = statistics.mean(dtwsnrs)
print(round(dtwmean,7)) 

premean = statistics.mean(snrs)
print(round(premean,7))

((dtwmean - premean) / premean)* 100

# %%
dtwpeaks.sort_values('height', ascending=False)[:5]
# %%
dtwplotpeaks = dtwpeaks.copy()

dtwplotpeaks = dtwplotpeaks.drop(index = 956)
dtwplotpeaks.sort_values('height', ascending=False)[:5]
dtwplotpeaks = dtwplotpeaks.loc[[958,957,953,955,954]]
 # %%
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25, 8))


all_hists = [hist, big_hist]  
ymin, ymax = 0, max(max(h) for h in all_hists) * 1

for [leftax, rightax], (_, peak), (_, dtwpeak) in zip(
    axes.T, 
    peaks1.sort_values('height', ascending=False)[:5].iterrows(), 
    dtwplotpeaks.iterrows()
):
    leftax.step(chunk_bins[:-1], hist, where='mid')
    rightax.step(bin_centers, big_hist, where='pre')
    
    leftax.set_xlim(peak['center'] - 50, peak['center'] + 50)
    rightax.set_xlim(dtwpeak['center'] - 50, dtwpeak['center'] + 50)
 
    leftax.set_ylim(ymin, ymax)
    rightax.set_ylim(ymin, ymax)


row_titles = ["Original 5 Most Prominent Peaks", "DTW 5 Most Prominent Peaks"]

for row_idx, title in enumerate(row_titles):
   
    y = 0.925 - row_idx * 0.45  
    fig.text(
        0.5,      
        y,
        title,
        ha='center', fontsize=12, 
    )
plt.show()

# %%

# %%
import matplotlib.pyplot as plt
offset = 0  # small offset

plt.figure(figsize=(10, 6))

for i, hist in enumerate(hists):
    plt.step(chunk_bins[:-1], hist + offset, where='mid', label=f'Chunk {i+1}')
    offset += .001

plt.xlabel('Energy')
plt.title('Original Peak Histogram Waterfall Plot')
plt.xlim(7300,7540)
#plt.legend()
plt.yticks([])
plt.show()

# %%

plt.step(chunk_bins[:-1], hists[0], where='mid', label=f'Chunk {i+1}')
plt.step(chunk_bins[:-1], hists[1] + offset, where='mid', label=f'Chunk {i+1}')
# %%
offset = 0  

plt.figure(figsize=(10, 6))

for i, hist in enumerate(warped_hists[1:]):
    plt.step(bin_centers, hist + offset, where='mid')
    offset += .001

plt.title('DTW Peak Histogram Waterfall Plot')
plt.xlim(7300,7540)
plt.xlabel('Energy')
plt.yticks([])
plt.show()

# %%
plt.step(bin_centers, warped_hists[1] + offset, where='mid', label=f'Chunk {i+1}')
plt.xlim(7300,7540)
plt.show()

# %%
plt.figure(figsize=(10,5))
plt.step(bins[:-1], hist, where='mid')
plt.xlabel("Energy")
plt.ylabel("Event Density")
plt.title("Channel 1 Histogram")
plt.ylim(ymin- 0.001,ymax + 0.001)
plt.xlim(0,20000)
plt.show()

# %%
plt.figure(figsize=(10,5))
plt.step(bin_centers, big_hist, where='mid')
plt.xlabel("Energy")
plt.ylabel("Event Density")
plt.title("DTW Channel 1 Histogram")
plt.ylim(ymin- 0.001,ymax + 0.001)
plt.xlim(0,20000)
plt.show()
# %%
peaks1.eval('snr = height / std', inplace=True)
dtwpeaks.eval('snr = height / std', inplace=True)
old_snr_mean = np.mean(peaks1.sort_values('height')[-50:]['snr'])
new_snr_mean = np.mean(dtwpeaks.sort_values('height')[-50:]['snr'])
(new_snr_mean - old_snr_mean) / old_snr_mean * 100
# %%
test_peak_color = sns.color_palette("deep")[1]
ax = peaks1.sort_values('height')[-50:].plot.scatter(x='center', y='snr', alpha=.5, marker='.', label='Original')
dtwpeaks.sort_values('height')[-50:].plot.scatter(x='center', y='snr', ax=ax, color=test_peak_color, alpha=.5, marker='.', label='DTW corrected', xlabel='Energy', ylabel='SNR')
ax.axhline(old_snr_mean)
ax.axhline(new_snr_mean, color=test_peak_color)
ax.figure.set_size_inches(13.32, 6.81)
ax.set_xlim(0,16000)
plt.legend()