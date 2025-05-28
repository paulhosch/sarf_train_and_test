#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAR Flood Mapping Evaluation Pipeline

This script runs the experiment defined in experiment_config.json, which compares
different ML models, sampling strategies, and parameter tuning methods for
flood mapping using SAR data.
"""

#%% Import required libraries
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns

# Import project modules
from src.data_handler import load_config
from src.training import run_experiment
from src.visualization import (
    plot_performance_metrics,
    plot_cross_site_vs_same_site,
    plot_confusion_matrices,
    plot_feature_importance
)
from src.utils import Timer

#%% Load and display configuration
config = load_config("experiment_config.json")

# Print experiment variables
print("Experiment ID:", config["experiment_id"])
print("\nData Paths:")
for path_name, path_value in config["data_paths"].items():
    print(f"- {path_name}: {path_value}")
    
print("\nExperiment Variables:")
for var_name, var_values in config["experiment_variables"].items():
    print(f"- {var_name}: {var_values}")

# Calculate total number of configurations
total_configs = (
    len(config["experiment_variables"]["models"]) *
    len(config["experiment_variables"]["training_sample_strategies"]) *
    len(config["experiment_variables"]["training_sample_sizes"]) *
    len(config["experiment_variables"]["tuning_methods"])
)

print(f"\nTotal configurations to test: {total_configs}")
print(f"Iterations per configuration: {config['experiment_constants']['experiment_iterations']}")
print(f"Case studies: {config['experiment_constants']['case_studies']}")

# Calculate total experiment runs
total_runs = (
    total_configs * 
    config['experiment_constants']['experiment_iterations'] * 
    len(config['experiment_constants']['case_studies'])
)

print(f"\nTotal experiment runs: {total_runs}")

#%% Run the experiment
# Set to False to skip running and just load existing results
run_full_experiment = True

new_samples_each_iteration = True  # Set to True to use per-iteration samples

if run_full_experiment:
    # Run the experiment and get results
    with Timer("Full experiment"):
        results_df = run_experiment("experiment_config.json", new_samples_each_iteration=new_samples_each_iteration)
else:
    # Load previously saved results
    experiment_id = config["experiment_id"]
    output_data_path = config["data_paths"]["output_data"]
    results_path = f"{output_data_path}/{experiment_id}/results.csv"
    
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        print(f"Loaded existing results from {results_path}")
    else:
        print(f"No existing results found at {results_path}. Set run_full_experiment = True to run the experiment.")
        # Set results_df to None so subsequent cells that check for it won't run
        results_df = None

# %%
