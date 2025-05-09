#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 02:02:19 2025

@author: saiful
"""

import pandas as pd

# Load the CSV file
file_path = "/data/saiful/ePPI/all_eppi_db_embeddings.csv"
merged_df = pd.read_csv(file_path)

# Print column names
print(merged_df.columns.tolist())
#%%
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "/data/saiful/ePPI/all_eppi_db_embeddings.csv"
df = pd.read_csv(file_path)

# Compute the length of each FASTA sequence
df['seq_length'] = df['FASTA'].apply(len)

# Plotting
plt.figure(figsize=(6, 4))  # ~3.25x2.5 inches for column-width in NAR
# plt.hist(df['seq_length'], bins=50, color='black', edgecolor='white')
# plt.hist(df['seq_length'], bins=50, color='#555555', edgecolor='white')  # dark gray
plt.hist(df['seq_length'], bins=50, color='#4a90e2', edgecolor='white') 
plt.xlabel('Sequence Length (amino acids)', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Distribution of PPI-DB Protein Sequence Lengths', fontsize=10)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save in journal-quality format (vector, suitable for LaTeX inclusion)
plt.savefig('fasta_length_distribution.png', format='png', dpi=400)

# Optional: show in a pop-up (if running interactively)
# plt.show()
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # Load the dataset
# file_path = "/data/saiful/ePPI/all_eppi_db_embeddings.csv"
# df = pd.read_csv(file_path)

# # Compute sequence lengths
df['seq_length'] = df['FASTA'].apply(len)

# Summary statistics
min_len = df['seq_length'].min()
max_len = df['seq_length'].max()
mean_len = df['seq_length'].mean()
median_len = df['seq_length'].median()
percentile_5 = np.percentile(df['seq_length'], 5)
percentile_95 = np.percentile(df['seq_length'], 95)

# Find mode bin (most frequent range from histogram)
counts, bins = np.histogram(df['seq_length'], bins=50)
max_bin_index = np.argmax(counts)
mode_range = (bins[max_bin_index], bins[max_bin_index + 1])

# Print results
print(f"Min: {min_len}")
print(f"Max: {max_len}")
print(f"Mean: {mean_len:.2f}")
print(f"Median: {median_len}")
print(f"5th–95th percentile range: {percentile_5:.0f}–{percentile_95:.0f}")
print(f"Most frequent length range: {mode_range[0]:.0f}–{mode_range[1]:.0f}")
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # Load the CSV file
# file_path = "/data/saiful/ePPI/all_eppi_db_embeddings.csv"
# df = pd.read_csv(file_path)

# Compute sequence lengths
df['seq_length'] = df['FASTA'].apply(len)

# Compute distribution statistics
percentile_5 = np.percentile(df['seq_length'], 5)
percentile_95 = np.percentile(df['seq_length'], 95)
q25 = np.percentile(df['seq_length'], 25)
q75 = np.percentile(df['seq_length'], 75)
counts, bins = np.histogram(df['seq_length'], bins=50)
max_bin_index = np.argmax(counts)
mode_range = (bins[max_bin_index], bins[max_bin_index + 1])

# Plotting
plt.figure(figsize=(6, 4))
plt.hist(df['seq_length'], bins=50, color='#4a90e2', edgecolor='black', alpha=0.8)

# Highlight statistical ranges
plt.axvspan(percentile_5, percentile_95, color='gray', alpha=0.2, label=f'5th–95th percentile ({int(percentile_5)}–{int(percentile_95)})')
plt.axvspan(q25, q75, color='orange', alpha=0.3, label=f'IQR ({int(q25)}–{int(q75)})')
plt.axvspan(mode_range[0], mode_range[1], color='green', alpha=0.3, label=f'Mode range ({int(mode_range[0])}–{int(mode_range[1])})')

# Labels and layout
plt.xlabel('Sequence Length (amino acids)', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Protein Sequence Length Distribution in PPI-DB', fontsize=10)
plt.legend(fontsize=8)
plt.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.7)
plt.tight_layout()

# Save plot
plt.savefig("fasta_length_distribution_ranges.png", format='png', dpi=400)

# Optional: show plot
# plt.show()

