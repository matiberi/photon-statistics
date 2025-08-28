# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:58:43 2024

@author: uceemti
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_photon_counts(filename):
    """
    Reads photon counts from a CSV file and returns them as a NumPy array.
    
    Assumes:
      - The first 5 lines are headers.
      - 'Photons' is the column of interest.
      - Data are separated by whitespace or commas.
    """
    df = pd.read_csv(
        filename,
        skiprows=5,
        engine='python',
        sep=r',\s*',
        skip_blank_lines=True,
        header=0,
    )
    df = df.dropna(axis=1, how='all')
    df.columns = df.columns.str.strip()
    if 'Photons' not in df.columns:
        raise ValueError(f"'Photons' column not found in the file. Columns found: {df.columns.tolist()}")
    return df['Photons'].astype(int).values

#----------------------------------------
# Data Processing (Per File)
#----------------------------------------

# File paths (adjust as needed)
dark_counts_file = 'QRNG2/after-polishing-SPAD2/DCR_5Ve_lowT_1s_50us.csv'
bright_counts_files = [
    'QRNG2/after-polishing-SPAD2/CR_5Ve_lowT_1s_50us-refwg.csv',
    'QRNG2/after-polishing-SPAD2/CR_5Ve_lowT_1s_50us-out1.csv',
    'QRNG2/after-polishing-SPAD2/CR_5Ve_lowT_1s_50us-out2.csv',
    'QRNG2/after-polishing-SPAD2/CR_5Ve_lowT_1s_50us-out3.csv',
    'QRNG2/after-polishing-SPAD2/CR_5Ve_lowT_1s_50us-out4.csv',
    'QRNG2/after-polishing-SPAD2/CR_5Ve_lowT_1s_50us-out5.csv',
    'QRNG2/after-polishing-SPAD2/CR_5Ve_lowT_1s_50us-out6.csv',
    'QRNG2/after-polishing-SPAD2/CR_5Ve_lowT_1s_50us-out7.csv',
    'QRNG2/after-polishing-SPAD2/CR_5Ve_lowT_1s_50us-out8.csv'
]

# Read dark counts once and compute their mean
dark_counts = read_photon_counts(dark_counts_file)
mean_dark_count = np.mean(dark_counts)

# Detector parameters
t_holdoff = 50e-6   # hold-off time (seconds)
pde = 0.225          # photon detection efficiency
time_scale_factor = 1 / 10000  # Converts counts/s to counts/ms

# Process each bright file and store (normalized counts, mean, file label)
combined_data = []
for bright_file in bright_counts_files:
    bright_counts = read_photon_counts(bright_file)
    adjusted = bright_counts - mean_dark_count
    denominator = 1 - adjusted * t_holdoff
    valid_mask = denominator > 0
    impinging = (1.0 / pde) * (adjusted[valid_mask] / denominator[valid_mask])
    impinging = impinging[impinging >= 0]
    # Normalize (photon counts per ms)
    norm_counts = impinging * time_scale_factor
    mean_val = np.mean(norm_counts)
    file_label = bright_file.split("/")[-1]
    combined_data.append((norm_counts, mean_val, file_label))
    
    # (Optional) Plot individual histogram for reference
    plt.figure(figsize=(8, 6))
    plt.hist(norm_counts, bins=100, edgecolor='black', color='purple')
    #plt.title(f'Histogram: {file_label}')
    plt.xlabel('Normalized Impinging Photon Count (1/ms)', fontsize=22)
    plt.ylabel('Frequency', fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.annotate(f'Mean: {mean_val:.2f}', xy=(0.65, 0.75), xycoords='axes fraction',
                 fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8))
    plt.tight_layout()
    plt.show()

#----------------------------------------
# Combined Plot: Histograms Spaced by Delta X, Height Proportional to Mean Photon Count
#----------------------------------------
# For each file, we:
# 1. Compute its histogram (frequency vs. photon count).
# 2. Normalize the x-values (bin centers) to [0,1] to preserve shape.
# 3. Shift each histogram horizontally by a fixed delta x.
# 4. Scale the frequency (bar height) so that the maximum frequency equals the file's mean.
# 5. Label each distribution with its mean photon count.

plt.figure(figsize=(12, 8))
delta_x = 1.0  # horizontal spacing between distributions

for i, (data, mean_val, file_label) in enumerate(combined_data):
    # Compute histogram using 50 bins
    counts, bins = np.histogram(data, bins=50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Normalize bin centers to [0, 1] (preserving the shape)
    norm_bin_centers = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
    # Shift x-values by fixed delta based on file index
    x_values = i * delta_x + norm_bin_centers
    # Scale the histogram counts so that the maximum equals the mean photon count.
    # (If counts.max() is 0, we skip scaling.)
    if counts.max() > 0:
        scaling = mean_val / counts.max()
    else:
        scaling = 1.0
    counts_scaled = counts * scaling
    # Determine a bar width (in the normalized x-scale)
    bar_width = 0.9 * (bins[1] - bins[0]) / (bin_centers.max() - bin_centers.min())
    
    # Plot the histogram as vertical bars
    plt.bar(x_values, counts_scaled, width=bar_width, align='center', alpha=0.6, label=file_label)
    
    # Annotate the distribution with its mean photon count (placed at the center of its x slot)
    x_annotation = i * delta_x + 0.5  # adjust as needed
    y_annotation = np.max(counts_scaled) * 1.01  # a little above the peak
    plt.text(x_annotation, y_annotation, f"Mean: {mean_val:.1f}", ha='center', fontsize=14)

random_number_list=['Ref', '000','001','010','100','011','101','110', '111']
plt.xlabel('Photon count distributions', fontsize=20)
plt.ylabel('Mean photon count (counts/ms)',fontsize=20)
#plt.title('Photon count distributions for each star coupler port')
plt.xticks([i * delta_x + 0.5 for i in range(len(combined_data))],
           [d for d in random_number_list], rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('combined_histograms_proportional_height.png', dpi=350)
plt.show()
