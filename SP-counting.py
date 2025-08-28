# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:58:43 2024

@author: uceemti
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
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

    # Drop empty columns caused by trailing commas
    df = df.dropna(axis=1, how='all')

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Ensure 'Photons' column exists
    if 'Photons' not in df.columns:
        raise ValueError(
            f"'Photons' column not found in the file. Columns found: {df.columns.tolist()}"
        )

    # Convert to integer counts and return as numpy array
    return df['Photons'].astype(int).values

#%%
#----------------------------------------
# Main Analysis
#----------------------------------------

# File paths (adjust as needed)
dark_counts_file = '../QRNG2/after-polishing-SPAD2/DCR_5Ve_lowT_1s_50us.csv'
bright_counts_file = '../QRNG2/after-polishing-SPAD2/CR_5Ve_lowT_1s_50us-out1.csv'

# Read data
dark_counts = read_photon_counts(dark_counts_file)
bright_counts = read_photon_counts(bright_counts_file)

# Calculate mean dark count
mean_dark_count = np.mean(dark_counts)

# Compute adjusted photon counts (bright minus mean dark)
list_bright_count = bright_counts
list_photon_count = list_bright_count - mean_dark_count

# Detector parameters
t_holdoff = 50e-6   # hold-off time (seconds)
pde = 0.225          # photon detection efficiency. Adjust based on Ve of SPAD

# Compute impinging photon counts
denominator = (1 - list_photon_count * t_holdoff)
valid_mask = denominator > 0
impinging_photon_count = (1.0 / pde) * (list_photon_count[valid_mask] / denominator[valid_mask])

# Compute impinging photon counts in gated mode
t_p = 1. # gate period, i.e. duration of one gating cycle, which includes the active time t_on and t_off
t_on = 1. # gate ON time, i.e. duration within each gating cycle during which the SPAD is active. This should be matched with photon arrival window (ns).
gated_impinging_photon_count = impinging_photon_count * (t_p / t_on)

# Filter out any negative impinging photon counts if they occur
impinging_photon_count = impinging_photon_count[impinging_photon_count >= 0]

# Normalize impinging photon counts to photon/ms
time_scale_factor = 1 / 1000  # Converts counts/s to counts/ms
normalized_impinging_photon_count = impinging_photon_count * time_scale_factor

# Compute statistics of impinging photon counts
mean_impinging_photon_count = np.mean(impinging_photon_count)
variance_impinging_photon_count = np.var(impinging_photon_count, ddof=1)

# Compute statistics of dark counts
mean_dark_count = np.mean(dark_counts)
variance_dark_count = np.var(dark_counts, ddof=1)

# Compute statistics of bright counts
mean_bright_count = np.mean(bright_counts)
variance_bright_count = np.var(bright_counts, ddof=1)

#%%
#----------------------------------------
# Print results
#----------------------------------------

print("Impinging Photon Counts:", impinging_photon_count)
print("\nMean Impinging Photon Count:", mean_impinging_photon_count)
print("Variance of Impinging Photon Count:", variance_impinging_photon_count)

print("\nNormalized Impinging Photon Counts (per ms):", normalized_impinging_photon_count)
print("\nDark Counts:", dark_counts)
print("Mean Dark Count:", mean_dark_count)
print("Variance of Dark Count:", variance_dark_count)

print("\nBright Counts:", bright_counts)
print("Mean Bright Count:", mean_bright_count)
print("Variance of Bright Count:", variance_bright_count)

#%%
#----------------------------------------
# Visualization
#----------------------------------------

# Plot dark vs bright counts over time
plt.figure(figsize=(8, 6))
plt.plot(dark_counts, "r.", label='Dark Counts')
plt.plot(bright_counts, "b.", label='Bright Counts')
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('Photon count rate (counts/s)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(0, len(bright_counts))  # Set x-axis limit to the length of the counts
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('figures/dark_bright_counts_over_time.png', dpi=350)
plt.show()

# # Histograms of bright, dark, and adjusted impinging counts
# # ------------------------------------------------------------------------------
# # 1.  Make a single figure containing two horizontal axes that share the y-axis
# # ------------------------------------------------------------------------------

# fig, (ax_left, ax_right) = plt.subplots(
#     1, 2, sharey=True, figsize=(9, 6),
#     gridspec_kw={"width_ratios": [3, 2], "wspace": 0.05}  # tweak to taste
# )

# # ------------------------------------------------------------------------------
# # 2.  Plot the three histograms on *both* axes …
# #     • Left axis: range 0-1250     • Right axis: range 2000-max
# # ------------------------------------------------------------------------------

# # Decide your global x-limit on the fly (or hard-code a number)
# #xmax = max(map(np.max, [bright_counts, dark_counts, impinging_photon_count]))
# xmax = 10000
# common_kwargs = dict(bins=50, edgecolor="black", alpha=0.70)
# ax_left .hist(bright_counts,             color="blue", label="Bright",    **common_kwargs)
# ax_left .hist(dark_counts,               color="red",     label="Dark",      **common_kwargs)
# ax_left .hist(impinging_photon_count,    color="orange",  label="Emitted", **common_kwargs)

# ax_right.hist(bright_counts,             color="blue", **common_kwargs)
# ax_right.hist(dark_counts,               color="red",     **common_kwargs)
# ax_right.hist(impinging_photon_count,    color="orange",  **common_kwargs)

# # ------------------------------------------------------------------------------
# # 3.  Set the independent x-ranges and tidy up labels/ticks
# # ------------------------------------------------------------------------------

# ax_left .set_xlim(0, 1200)
# ax_right.set_xlim(2000, xmax)

# # Hide tick labels between the two panels to reinforce the break
# ax_left .spines["right"].set_visible(False)
# ax_right.spines["left" ].set_visible(False)
# ax_right.yaxis.tick_right()                       # move ticks to the outside
# ax_right.tick_params(labelright=True, labelleft=False)

# for ax in (ax_left, ax_right):
#     ax.tick_params(axis="both",         # x *and* y
#                    which="major",       # major ticks
#                    labelsize=16,        # font size of numbers
#                    length=6, width=1.2) # make the tick marks themselves larger

# ax_left.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
# ax_right.set_xticklabels([2000, 2200, 2400, 2600])

# # ------------------------------------------------------------------------------
# # 4.  Draw diagonal “break-marks” on both axes
# # ------------------------------------------------------------------------------

# d = .015  # size of diagonal lines as a fraction of axis-size
# kwargs = dict(transform=ax_left.transAxes,  color="k", clip_on=False)
# ax_left.plot((1-d, 1+d), (-d, +d), **kwargs)          # bottom-left
# ax_left.plot((1-d, 1+d), (1-d, 1+d), **kwargs)        # top-left

# kwargs.update(transform=ax_right.transAxes)           # reuse but with new axes
# ax_right.plot((-d, +d), (-d, +d), **kwargs)           # bottom-right
# ax_right.plot((-d, +d), (1-d, 1+d), **kwargs)         # top-right

# # ------------------------------------------------------------------------------
# # 5.  Global labelling & save/show
# # ------------------------------------------------------------------------------

# #fig.text(0.5, -0.02, "Photon counts (counts/s)", ha="center", va="center", fontsize=18)
# ax_left.set_ylabel("Frequency of occurrence", fontsize=18)
# ax_left.set_xlabel("Photon counts (counts/ms)", fontsize=18)
# ax_right.set_xlabel("Photon counts (counts/s)", fontsize=18)

# ax_left.legend(fontsize=12, loc="upper left")
# ax_left.grid(True, linestyle="--", alpha=0.7)
# ax_right.grid(True, linestyle="--", alpha=0.7)

# plt.tight_layout()
# plt.savefig("figures/dark_bright_impinging_hist_brokenx.png", dpi=350, bbox_inches="tight")
# plt.show()


#%%
# Histogram for dark counts
plt.figure(figsize=(8, 6))
plt.hist(dark_counts, bins=100, edgecolor="black", color="red", alpha=0.7)
plt.title('Histogram of Dark Counts')
plt.xlabel('Dark Counts')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

# Annotate statistics
plt.annotate(
    f'Mean: {mean_dark_count:.2f}\nVariance: {variance_dark_count:.2f}',
    xy=(0.65, 0.75),
    xycoords='axes fraction',
    fontsize=12,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8),
)

plt.tight_layout()
plt.savefig('figures/dark_count_histogram.png', dpi=350, bbox_inches='tight')
plt.show()

#%%
# Histogram for bright counts
plt.figure(figsize=(8, 6))
plt.hist(bright_counts, bins=100, edgecolor="black", color="blue", alpha=0.7)
plt.title('Histogram of Bright Counts')
plt.xlabel('Bright Counts')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

# Annotate statistics
plt.annotate(
    f'Mean: {mean_bright_count:.2f}\nVariance: {variance_bright_count:.2f}',
    xy=(0.65, 0.75),
    xycoords='axes fraction',
    fontsize=12,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8),
)

plt.tight_layout()
plt.savefig('figures/bright_count_histogram.png', dpi=300, bbox_inches='tight')
plt.show()


#%%
# Histogram of impinging photon counts with annotation
plt.figure(figsize=(8, 6))
plt.hist(
    impinging_photon_count,
    bins=100,
    edgecolor='black',
    color='skyblue',
)
plt.title('Histogram of Impinging Photon Counts')
plt.xlabel('Impinging Photon Count')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

# Annotate statistics
plt.annotate(
    f'Mean: {mean_impinging_photon_count:.2f}\nVariance: {variance_impinging_photon_count:.2f}',
    xy=(0.65, 0.75),
    xycoords='axes fraction',
    fontsize=12,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8),
)

plt.tight_layout()
plt.savefig('figures/impinging_photon_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Histogram of normalized impinging photon counts with annotation
plt.figure(figsize=(8, 6))
plt.hist(
    normalized_impinging_photon_count,
    bins=100,
    edgecolor='black',
    color='purple',
)
plt.title('Histogram of Impinging Photon Counts (Normalized to Photon/ms)')
plt.xlabel('Normalized Impinging Photon Count (per ms)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

# Annotate statistics for normalized data
mean_normalized = np.mean(normalized_impinging_photon_count)
variance_normalized = np.var(normalized_impinging_photon_count, ddof=1)
plt.annotate(
    f'Mean: {mean_normalized:.2f}\nVariance: {variance_normalized:.2f}',
    xy=(0.65, 0.75),
    xycoords='axes fraction',
    fontsize=12,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8),
)

plt.tight_layout()
plt.savefig('figures/normalized_impinging_photon_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
