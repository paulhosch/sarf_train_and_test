#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple interactive script to plot cross-site vs same-site performance comparison
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from src.data_handler import load_config
from src.visualisation.vis_config import VIOLIN_FACE_COLOR, SWARM_PLOT_COLOR, LINE_THICKNESS, LINE_THICKNESS_GRID

LINE_THICKNESS = 4
LINE_COLOR = sns.color_palette("colorblind")[7]
DPI = 600

# %%
# Load configuration and data
config = load_config("experiment_config.json")
experiment_id = config["experiment_id"]
output_data_path = config["data_paths"]["output_data"]
results_path = f"{output_data_path}/{experiment_id}/results.csv"
results_df = pd.read_csv(results_path)

# %%
# Filter data for cross-site vs same-site comparison
filtered_df = results_df[
    (results_df['model'] == "Random Forest") &
    (results_df['training_sample_strategy'] == "simple_random") &
    (results_df['training_sample_size'] == 1000) &
    (results_df['tuning_method'] == "no_tuning")
].copy()

print(f"Filtered data shape: {filtered_df.shape}")

# %%
# Prepare data for plotting
metrics = ["proportional_OA", "balanced_OA", "proportional_F1", "balanced_F1"]
titles = ['Proportional\nOverall Accuracy', 'Balanced\nOverall Accuracy', 'Proportional\nF1-Score', 'Balanced\nF1-Score']

# Get unique training sites and testing sites
training_sites = sorted(filtered_df['training_sites'].unique())
testing_sites = sorted(filtered_df['testing_site'].unique())
site_types = ["same_site", "cross_site"]

# Create melted data for plotting with combined site information
plot_data = []
for _, row in filtered_df.iterrows():
    train_site = row['training_sites']
    test_site = row['testing_site']
    
    for metric in metrics:
        # Same site data
        same_site_col = f"{metric}_same_site"
        if same_site_col in filtered_df.columns and not pd.isna(row[same_site_col]):
            # For same site: show train_site (same as test_site for same-site)
            x_label = f"{train_site}  \n→{train_site}"
            plot_data.append({
                'Site Comparison': x_label,
                'Metric': metric,
                'Value': row[same_site_col],
                'Type': 'same_site'
            })
        
        # Cross site data
        cross_site_col = f"{metric}_cross_site"
        if cross_site_col in filtered_df.columns and not pd.isna(row[cross_site_col]):
            # For cross site: show train_site -> test_site with line break
            x_label = f"{train_site}  \n→{test_site}"
            plot_data.append({
                'Site Comparison': x_label,
                'Metric': metric,
                'Value': row[cross_site_col],
                'Type': 'cross_site'
            })

plot_df = pd.DataFrame(plot_data)

# %%
# Create the plot
font_size = 18
title_size = 20
fig = plt.figure(figsize=(12.99, 4.72))

n_cols = 4
n_rows = 1

gs = fig.add_gridspec(n_rows, n_cols)
# Set line thickness and color for all subplots
plt.rcParams['axes.linewidth'] = LINE_THICKNESS  # Outline (spine) thickness
plt.rcParams['xtick.major.width'] = LINE_THICKNESS
plt.rcParams['ytick.major.width'] = LINE_THICKNESS
plt.rcParams['axes.edgecolor'] = LINE_COLOR  # Set spine color

for i, metric in enumerate(metrics):
    ax = fig.add_subplot(gs[i])
    
    # Filter data for this metric
    metric_data = plot_df[plot_df['Metric'] == metric]
    
    # Create violin plot without hue (flipped axes)
    sns.violinplot(
        data=metric_data,
        y='Site Comparison',
        x='Value',
        ax=ax,
        #linewidth=LINE_THICKNESS,
        inner=None,
        alpha=1,
        color='#0173B2',
        zorder=1
    )
    
    # Add individual points (flipped axes)
    sns.swarmplot(
        data=metric_data,
        y='Site Comparison',
        x='Value',
        ax=ax,
        alpha=1,
        size=4,
        color='#CB78BB',
        zorder=2
    )
    
    # Styling
    ax.set_title(titles[i], fontsize=title_size, fontweight='bold', pad=10)
    ax.set(xlabel=None, ylabel=None)
    
    # Set exactly three x-axis ticks: min, middle, max, buffer min and max inwards 
    buffer_factor = 0.05
    x_min, x_max = ax.get_xlim()
    buffer = (x_max - x_min) * buffer_factor
    x_middle = (x_min + x_max) / 2
    x_min_buffer = x_min + buffer
    x_max_buffer = x_max - buffer
    ax.set_xticks([x_min_buffer, x_middle, x_max_buffer])
    ax.set_xticklabels([f'{x:.2f}' for x in [x_min_buffer, x_middle, x_max_buffer]], rotation=45, fontsize=font_size)
    
    # Add horizontal grid lines (default behavior)
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=LINE_THICKNESS, color=LINE_COLOR,  zorder=0)
    

    # Remove y-axis tick labels for all except the first subplot
    if i != 0:
        ax.set_yticklabels([])
    else:
        ax.tick_params(axis='y', labelsize=font_size)
    
    # Set x-axis tick label size
    ax.tick_params(axis='x', labelsize=font_size)

    # Set spine thickness
    for spine in ax.spines.values():
        spine.set_linewidth(LINE_THICKNESS)

# Add overall title
'''plt.suptitle('Cross-Site vs Same-Site Performance Comparison\n(Random Forest, Simple Random, Size=500, No Tuning)', 
             fontsize=16, fontweight='bold', y=0.98)'''

plt.tight_layout(pad=0.5)

# Save the plot
plot_dir = f"{output_data_path}/{experiment_id}/plots"
os.makedirs(plot_dir, exist_ok=True)
output_path = f"{plot_dir}/cross_vs_same_site_comparison.png"
plt.savefig(output_path, dpi=DPI, transparent=True)

print(f"Plot saved to: {output_path}")

# %%
# Create summary statistics table
summary_stats = []

for _, row in filtered_df.iterrows():
    train_site = row['training_sites']
    test_site = row['testing_site']
    
    for metric in metrics:
        # Same site statistics
        same_site_col = f"{metric}_same_site"
        if same_site_col in filtered_df.columns and not pd.isna(row[same_site_col]):
            summary_stats.append({
                'Site Comparison': f"{train_site}\n→ {test_site}",
                'Type': 'same_site',
                'Metric': metric,
                'Value': row[same_site_col]
            })
        
        # Cross site statistics
        cross_site_col = f"{metric}_cross_site"
        if cross_site_col in filtered_df.columns and not pd.isna(row[cross_site_col]):
            summary_stats.append({
                'Site Comparison': f"{train_site}\n→ {test_site}",
                'Type': 'cross_site',
                'Metric': metric,
                'Value': row[cross_site_col]
            })

summary_df = pd.DataFrame(summary_stats)

# Calculate summary statistics
summary_table = summary_df.groupby(['Site Comparison', 'Type', 'Metric'])['Value'].agg(['mean', 'std', 'count']).round(3)

print("\nSummary Statistics:")
print(summary_table)

# %% 