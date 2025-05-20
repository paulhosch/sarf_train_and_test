#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for visualizations.
Contains shared functionality used across different visualization scripts.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast
from typing import Dict, List, Tuple, Any, Optional, Union

# Add parent directory to path to import project modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from project
from src.visualization import create_output_dir
from src.visualisation.vis_config import DEFAULT_PLOT_PARAMS, FIGURE_SIZES, COLOR_PALETTES, PLOT_STYLE, OUTPUT_SETTINGS

def load_experiment_data():
    """
    Load experiment configuration and results data
    
    Returns:
        Tuple of (results DataFrame, config dict) or (None, config) or (None, None)
    """
    # Load configuration
    try:
        with open("experiment_config.json", 'r') as f:
            config = json.load(f)
            
        experiment_id = config["experiment_id"]
        output_data_path = config["data_paths"]["output_data"]
        results_path = f"{output_data_path}/{experiment_id}/results.csv"
        
        # Load results
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            print(f"Loaded {len(results_df)} results from {results_path}")
            return results_df, config
        else:
            print(f"Error: No results found at {results_path}")
            return None, config
    except Exception as e:
        print(f"Error loading experiment data: {str(e)}")
        return None, None

def parse_feature_importance(importance_str):
    """
    Parse feature importance string from results dataframe
    
    Args:
        importance_str: String representation of feature importance dictionary
        
    Returns:
        Dictionary of feature names and importance values
    """
    if pd.isna(importance_str) or importance_str == '':
        return {}
    
    try:
        # Handle dictionaries with NumPy arrays like {'FEATURE': array([0.1234, 0.1234])}
        if 'array(' in importance_str:
            importance_dict = {}
            # Remove the outer braces
            content = importance_str.strip('{}')
            # Split by comma outside of array brackets
            parts = []
            bracket_count = 0
            current_part = ""
            
            for char in content:
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                
                if char == ',' and bracket_count == 0:
                    parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char
            
            # Add the last part if not empty
            if current_part.strip():
                parts.append(current_part.strip())
            
            # Process each key-value pair
            for part in parts:
                if ':' in part:
                    key, value_str = part.split(':', 1)
                    key = key.strip().strip("'\"")
                    
                    # Extract the numeric value from array([...])
                    if 'array(' in value_str:
                        # Extract values between square brackets
                        array_content = value_str[value_str.find('[')+1:value_str.find(']')]
                        values = [float(v.strip()) for v in array_content.split(',')]
                        # Use the first value (they are typically duplicated for binary classification)
                        importance_dict[key] = values[0]
                    else:
                        # Handle plain numeric values
                        try:
                            importance_dict[key] = float(value_str.strip())
                        except ValueError:
                            # Skip invalid values
                            continue
            
            return importance_dict
        elif 'np.float64' in importance_str:
            # Handle older format with np.float64
            cleaned_str = importance_str.replace('np.float64(', '').replace(')', '')
            # Convert to dictionary
            importance_dict = {}
            pairs = cleaned_str.strip('{}').split(', ')
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    key = key.strip().strip("'\"")
                    value = float(value.strip())
                    importance_dict[key] = value
            return importance_dict
        else:
            # Try to evaluate as literal Python dict if no special values
            return ast.literal_eval(importance_str)
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing feature importance: {e}")
        print(f"Problematic string: {importance_str}")
        return {}
    except Exception as e:
        print(f"Unexpected error parsing feature importance: {e}")
        return {}

def apply_plot_style(ax, gridline_width=None, subplot_borderwidth=None):
    """Apply consistent styling to plot axes"""
    # Use default values if not provided
    gridline_width = gridline_width or PLOT_STYLE['grid_alpha']
    subplot_borderwidth = subplot_borderwidth or DEFAULT_PLOT_PARAMS['subplot_borderwidth']
    
    # Set transparent background
    ax.set_facecolor('none')
    
    # Set subplot border width
    for spine in ax.spines.values():
        spine.set_linewidth(subplot_borderwidth)
    
    # Add gridlines with specified width
    ax.grid(True, linestyle=PLOT_STYLE['grid_style'], alpha=PLOT_STYLE['grid_alpha'], linewidth=gridline_width)

def create_error_plot(error_message, plot_dir, comparison_type, fixed_model=None, fixed_sample_size=None, 
                     fixed_strategy=None, tune_method=None):
    """
    Create and save an error plot
    
    Args:
        error_message: Text to display in the error plot
        plot_dir: Directory to save the plot
        comparison_type: Type of comparison being performed
        fixed_model: Model filter that was applied
        fixed_sample_size: Sample size filter that was applied
        fixed_strategy: Strategy filter that was applied
        tune_method: Tuning method filter that was applied
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=FIGURE_SIZES['error'])
    plt.text(0.5, 0.5, error_message, ha='center', va='center', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    # Create filter string for the filename
    filter_str = ""
    if fixed_model:
        filter_str += f"model_{str(fixed_model).replace(' ', '_')}_"
    if fixed_sample_size:
        filter_str += f"size_{fixed_sample_size}_"
    if fixed_strategy:
        filter_str += f"strategy_{str(fixed_strategy).replace(' ', '_')}_"
    if not filter_str:
        filter_str = "all_"
        
    output_path = f"{plot_dir}/strategy_performance_{comparison_type}_{filter_str}{tune_method}_error.png"
    plt.savefig(output_path, dpi=OUTPUT_SETTINGS['dpi'], bbox_inches=OUTPUT_SETTINGS['bbox_inches'])
    plt.show()
    
    return output_path

def set_plot_fonts(tick_fontsize=None, subtitle_fontsize=None, axis_label_fontsize=None, legend_fontsize=None):
    """Set consistent font sizes for plots"""
    # Use default values if not provided
    tick_fontsize = tick_fontsize or DEFAULT_PLOT_PARAMS['tick_fontsize']
    subtitle_fontsize = subtitle_fontsize or DEFAULT_PLOT_PARAMS['subtitle_fontsize']
    axis_label_fontsize = axis_label_fontsize or DEFAULT_PLOT_PARAMS['axis_label_fontsize']
    legend_fontsize = legend_fontsize or DEFAULT_PLOT_PARAMS['legend_fontsize']
    
    plt.rcParams.update({
        'font.size': tick_fontsize,
        'axes.titlesize': subtitle_fontsize,
        'axes.labelsize': axis_label_fontsize,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize,
        'legend.fontsize': legend_fontsize,
    })

def get_output_directory(experiment_id, config=None):
    """Get or create output directory for plots"""
    if config:
        plot_dir = create_output_dir(experiment_id, config)
    else:
        plot_dir = f"../../data/experiments/{experiment_id}/plots"
        os.makedirs(plot_dir, exist_ok=True)
    
    return plot_dir
