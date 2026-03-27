import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 'rb' stands for 'read binary'—essential for pickle files
with open('figure7multi.p', 'rb') as file:
    data_multi = pd.DataFrame(pickle.load(file)).T

with open('figure7multicircle.p', 'rb') as file:
    data_multicircle = pd.DataFrame(pickle.load(file)).T

with open('figure7multioned.p', 'rb') as file:
    data_multioned = pd.DataFrame(pickle.load(file)).T

print(data_multi)
print(data_multicircle)
print(data_multioned)
data_multioned = data_multioned.iloc[1:19]

print(data_multioned)
# Assuming n_replicates is defined elsewhere in your code
n_replicates = 50

def plot_subplot(ax, result, title):
    # Calculations
    Mean = result.mean(axis=1)
    SE = result.std(axis=1) / np.sqrt(n_replicates)
    Top = Mean + 5 * SE
    Bottom = Mean - 5 * SE
    
    # Extracting the index (d_mu_list)
    d_mu_list = result.index 

    # Plotting on the specific axis (ax)
    ax.plot(d_mu_list, Mean, 'k-', linewidth=3)
    ax.fill_between(d_mu_list, Bottom, Top, facecolor='lightgray')
    
    ax.set_xlabel(r'$\Delta \mu / \sigma$', fontsize=18)
    # Formatting
    ax.set_title(title, fontsize=16)

# 1. Create a figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 2. Map your data to the subplots
datasets = [data_multioned, data_multicircle, data_multi]
titles = ['Linear Arrangement (1D)', 'Circular Arrangement (2D)', 'Simplex Arrangement (30D)']

for ax in axes:
    ax.tick_params(labelleft=True)
for i in range(3):
    plot_subplot(axes[i], datasets[i], titles[i])
axes[0].set_ylabel(r'$t^{*}$', fontsize=18)
# 3. Final layout adjustments
plt.tight_layout()
plt.savefig('figure7multi_combined.pdf')
plt.show()