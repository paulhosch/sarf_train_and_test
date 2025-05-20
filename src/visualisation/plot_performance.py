#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% 
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import ast
import json

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules from the project
from src.visualization import create_output_dir
from src.visualisation.vis_utils import (load_experiment_data, create_error_plot, 
                                        set_plot_fonts, apply_plot_style, get_output_directory)
from src.visualisation.vis_config import (DEFAULT_PLOT_PARAMS, FIGURE_SIZES, 
                                         COLOR_PALETTES, PLOT_STYLE, OUTPUT_SETTINGS, 
                                         TABLE_STYLE, TABLE_COLORMAP, LAYOUT_SPACING, DEFAULT_PLOT_PARAMS)

def plot_strategies_performance(results_df: pd.DataFrame,
                              comparison_type: str = "sample_size",
                              fixed_filter: Optional[Union[str, int]] = None,
                              fixed_sample_size: Optional[int] = None,  # Added for training_site comparison
                              fixed_model: Optional[str] = None,  # Added for training_site comparison
                              fixed_strategy: Optional[str] = None,  # Added for testing_site comparison
                              tune_method: str = "no_tuning",
                              experiment_id: str = "main_experiment",
                              site_type: str = "same_site",
                              config: Dict = None,
                              show_points: bool = True,
                              title_fontsize: int = 16,
                              subtitle_fontsize: int = 12,
                              axis_label_fontsize: int = 10,
                              tick_fontsize: int = 9,
                              legend_fontsize: int = 10,
                              gridline_width: float = 0.8,
                              boxplot_linewidth: float = 1.2,
                              bar_linewidth: float = 1.5,
                              subplot_borderwidth: float = 1.0) -> str:
    """
    Plot performance metrics comparing sampling strategies with flexible comparison options
    
    Args:
        results_df: DataFrame containing results
        comparison_type: What to compare as hue - options:
                        "sample_size" - Compare different sample sizes (hue) for a specific model
                        "model" - Compare different models (hue) for a specific sample size
                        "training_site" - Compare different training sites (hue) for specific model and sample size
                        "testing_site" - Compare performance across testing sites (x-axis) with same_site vs 
                                       cross_site metrics as hue, for a specific model and sampling strategy
        fixed_filter: [DEPRECATED] Use fixed_model or fixed_sample_size instead. Former parameter 
                     for backward compatibility - model name (for sample_size comparison) or
                     sample size value (for model comparison).
        fixed_sample_size: Sample size to filter results by (used with "model" or "training_site" comparison)
        fixed_model: Model name to filter results by (used with "sample_size", "training_site" or "testing_site")
        fixed_strategy: Sampling strategy to filter by (used with "testing_site" comparison)
        tune_method: Filter results to this tuning method (e.g., "no_tuning", "BayesSearchCV")
        experiment_id: ID of the experiment
        site_type: "same_site" or "cross_site" metrics (only used for non-testing_site comparison types)
        config: Experiment configuration with data paths
        show_points: Whether to show individual data points in the boxplot
        title_fontsize: Font size for main title
        subtitle_fontsize: Font size for subplot titles
        axis_label_fontsize: Font size for axis labels
        tick_fontsize: Font size for axis ticks
        legend_fontsize: Font size for legend
        gridline_width: Line width for gridlines
        boxplot_linewidth: Line width for boxplot outlines
        bar_linewidth: Line width for bar plots
        subplot_borderwidth: Line width for subplot borders
        
    Returns:
        Path to saved plot
        
    Examples:
        # Compare sample sizes for Random Forest
        plot_strategies_performance(results_df, "sample_size", fixed_model="Random Forest")
        
        # Compare models for sample size of 500
        plot_strategies_performance(results_df, "model", fixed_sample_size=500)
        
        # Compare training sites for Random Forest with sample size of 500
        plot_strategies_performance(results_df, "training_site", 
                                  fixed_model="Random Forest", 
                                  fixed_sample_size=500)
                                  
        # Compare testing sites with same_site vs cross_site metrics for Random Forest with 
        # "stratified" sampling strategy
        plot_strategies_performance(results_df, "testing_site",
                                  fixed_model="Random Forest",
                                  fixed_strategy="stratified")
    """
    # Filter results by tuning method
    if 'tuning_method' not in results_df.columns:
        # Extract tuning method from configuration name if not available
        results_df['tuning_method'] = results_df['configuration_name'].apply(
            lambda x: 'BayesSearchCV' if 'BayesSearchCV' in x else 'no_tuning' if 'no_tuning' in x else None
        )
    
    # Helper function to create error path and plot
    def create_error_plot(error_message, plot_dir, comparison_type, fixed_model, fixed_sample_size, fixed_strategy, tune_method):
        plt.figure(figsize=(10, 6))
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
        plt.savefig(output_path, dpi=300)
        plt.show()
        
        return output_path
    
    # Base filter for tuning method
    if comparison_type == "tuning_strategy":
        # Don't filter by tuning method when comparing tuning strategies
        filtered_df = results_df.copy()
    else:
        # Apply tuning method filter for other comparison types
        filtered_df = results_df[results_df['tuning_method'] == tune_method].copy()
    
    # Data preparation: if training_sites is a comma-separated string, normalize it for comparison
    if 'training_sites' in filtered_df.columns:
        # Check if it's stored as comma-separated values
        if len(filtered_df) > 0 and isinstance(filtered_df['training_sites'].iloc[0], str):
            # Normalize by sorting and removing trailing spaces
            filtered_df['training_sites'] = filtered_df['training_sites'].apply(
                lambda x: ','.join(sorted([site.strip() for site in x.split(',')]))
            )
    
    # Handle backward compatibility for fixed_filter
    if fixed_sample_size is None and fixed_model is None and fixed_filter is not None:
        if comparison_type == "sample_size":
            fixed_model = fixed_filter
        elif comparison_type == "model":
            fixed_sample_size = fixed_filter
    
    # Apply the appropriate filter based on comparison type
    if comparison_type == "sample_size":
        # Compare different sample sizes for a specific model
        if fixed_model is not None:
            filtered_df = filtered_df[filtered_df['model'] == fixed_model]
            title_filter = f"Model: {fixed_model}"
            hue_variable = 'training_sample_size'
            hue_title = 'Sample Size'
        else:
            # No model filter provided, use all models
            title_filter = "All Models"
            hue_variable = 'training_sample_size'
            hue_title = 'Sample Size'
    elif comparison_type == "model":
        # Compare different models for a specific sample size
        if fixed_sample_size is not None:
            filtered_df = filtered_df[filtered_df['training_sample_size'] == fixed_sample_size]
            title_filter = f"Sample Size: {fixed_sample_size}"
            hue_variable = 'model'
            hue_title = 'Model'
        else:
            # No sample size filter provided, use all sample sizes
            title_filter = "All Sample Sizes"
            hue_variable = 'model'
            hue_title = 'Model'
    elif comparison_type == "tuning_strategy":
        # Compare different tuning methods for a specific sample size and model
        title_parts = []
        
        # Apply model filter if provided
        if fixed_model is not None:
            filtered_df = filtered_df[filtered_df['model'] == fixed_model]
            title_parts.append(f"Model: {fixed_model}")
        
        # Apply sample size filter if provided
        if fixed_sample_size is not None:
            filtered_df = filtered_df[filtered_df['training_sample_size'] == fixed_sample_size]
            title_parts.append(f"Size: {fixed_sample_size}")
            
        title_filter = ", ".join(title_parts) if title_parts else "All Models and Sizes"
        hue_variable = 'tuning_method'
        hue_title = 'Tuning Method'
    elif comparison_type == "training_site":
        # Verify training_sites column exists
        if 'training_sites' not in filtered_df.columns:
            print(f"Error: 'training_sites' column not found in results DataFrame but comparison_type='training_site' was specified.")
            print(f"Available columns: {filtered_df.columns.tolist()}")
            
            if config:
                plot_dir = create_output_dir(experiment_id, config)
            else:
                plot_dir = f"../../data/experiments/{experiment_id}/plots"
                os.makedirs(plot_dir, exist_ok=True)
                
            return create_error_plot(
                "Error: 'training_sites' column not found", 
                plot_dir, 
                comparison_type, 
                fixed_model, 
                fixed_sample_size,
                fixed_strategy,
                tune_method
            )
        
        # Compare different training sites for specific model and sample size
        title_parts = []
        
        # Apply model filter if provided
        if fixed_model is not None:
            filtered_df = filtered_df[filtered_df['model'] == fixed_model]
            title_parts.append(f"Model: {fixed_model}")
        
        # Apply sample size filter if provided
        if fixed_sample_size is not None:
            filtered_df = filtered_df[filtered_df['training_sample_size'] == fixed_sample_size]
            title_parts.append(f"Sample Size: {fixed_sample_size}")
            
        title_filter = ", ".join(title_parts) if title_parts else "All Models and Sample Sizes"
        hue_variable = 'training_sites'
        hue_title = 'Training Sites'
    elif comparison_type == "testing_site":
        # Verify testing_site column exists
        if 'testing_site' not in filtered_df.columns:
            print(f"Error: 'testing_site' column not found in results DataFrame but comparison_type='testing_site' was specified.")
            print(f"Available columns: {filtered_df.columns.tolist()}")
            
            if config:
                plot_dir = create_output_dir(experiment_id, config)
            else:
                plot_dir = f"../../data/experiments/{experiment_id}/plots"
                os.makedirs(plot_dir, exist_ok=True)
                
            return create_error_plot(
                "Error: 'testing_site' column not found", 
                plot_dir, 
                comparison_type, 
                fixed_model, 
                fixed_sample_size,
                fixed_strategy,
                tune_method
            )
        
        # Compare different testing sites with same_site vs cross_site metrics
        title_parts = []
        
        # Apply model filter if provided
        if fixed_model is not None:
            filtered_df = filtered_df[filtered_df['model'] == fixed_model]
            title_parts.append(f"Model: {fixed_model}")
        
        # Apply sampling strategy filter if provided
        if fixed_strategy is not None:
            filtered_df = filtered_df[filtered_df['training_sample_strategy'] == fixed_strategy]
            title_parts.append(f"Strategy: {fixed_strategy}")
            
        # Apply sample size filter if provided
        if fixed_sample_size is not None:
            filtered_df = filtered_df[filtered_df['training_sample_size'] == fixed_sample_size]
            title_parts.append(f"Size: {fixed_sample_size}")
            
        title_filter = ", ".join(title_parts) if title_parts else "All Models and Strategies"
        hue_variable = 'site_type'  # This will be created during data preparation
        hue_title = 'Testing Site'
    else:
        # Default case (should not happen with proper input validation)
        title_filter = "Unknown Comparison"
        hue_variable = 'model'
        hue_title = 'Model'
        
    # Check if we have data after filtering
    if len(filtered_df) == 0:
        error_msg = f"No data found after filtering"
        if fixed_model:
            error_msg += f" for model={fixed_model}"
        if fixed_sample_size:
            error_msg += f" for sample_size={fixed_sample_size}"
        if fixed_strategy:
            error_msg += f" for strategy={fixed_strategy}"
        error_msg += f" with tuning method '{tune_method}'"
        
        print(error_msg)
        print(f"Available models: {results_df['model'].unique()}")
        print(f"Available sample sizes: {results_df['training_sample_size'].unique()}")
        print(f"Available tuning methods: {results_df['tuning_method'].unique()}")
        if comparison_type == "training_site":
            print(f"Available training sites: {results_df['training_sites'].unique() if 'training_sites' in results_df.columns else 'Not available'}")
        if comparison_type == "testing_site":
            print(f"Available testing sites: {results_df['testing_site'].unique() if 'testing_site' in results_df.columns else 'Not available'}")
            print(f"Available strategies: {results_df['training_sample_strategy'].unique() if 'training_sample_strategy' in results_df.columns else 'Not available'}")
        
        # Create and save error plot
        if config:
            plot_dir = create_output_dir(experiment_id, config)
        else:
            plot_dir = f"../../data/experiments/{experiment_id}/plots"
            os.makedirs(plot_dir, exist_ok=True)
            
        return create_error_plot(error_msg, plot_dir, comparison_type, fixed_model, fixed_sample_size, fixed_strategy, tune_method)
    
    # Setup metrics and plot structure
    if comparison_type == "testing_site":
        # For testing_site comparison, we use fixed metrics and compare same_site vs cross_site
        metrics = [
            "proportional_OA",
            "balanced_OA",
            "proportional_F1",
            "balanced_F1"
        ]
        
        # Check which metrics are available in both same_site and cross_site variants
        available_metrics = []
        for metric in metrics:
            same_site_metric = f"{metric}_same_site"
            cross_site_metric = f"{metric}_cross_site"
            
            if same_site_metric in filtered_df.columns and cross_site_metric in filtered_df.columns:
                available_metrics.append(metric)
        
        if not available_metrics:
            print("No matching metrics found with both same_site and cross_site variants")
            print(f"Available columns: {[col for col in filtered_df.columns if 'OA' in col or 'F1' in col]}")
            
            # Create error plot
            error_msg = "No matching metrics found with both same_site and cross_site variants"
            
            # Create and save error plot
            if config:
                plot_dir = create_output_dir(experiment_id, config)
            else:
                plot_dir = f"../../data/experiments/{experiment_id}/plots"
                os.makedirs(plot_dir, exist_ok=True)
                
            return create_error_plot(error_msg, plot_dir, comparison_type, fixed_model, fixed_sample_size, fixed_strategy, tune_method)
    else:
        # For other comparison types, use metrics based on site_type
        metrics = [
            f"proportional_OA_{site_type}",
            f"balanced_OA_{site_type}",
            f"proportional_F1_{site_type}",
            f"balanced_F1_{site_type}"
        ]
        
        # Check which metrics are available
        available_metrics = [m for m in metrics if m in filtered_df.columns]
        if not available_metrics:
            print(f"No '{site_type}' metrics found in results")
            available_columns = [col for col in filtered_df.columns if 'OA' in col or 'F1' in col]
            print(f"Available metric columns: {available_columns}")
            
            error_msg = f"No '{site_type}' metrics found in results"
            
            # Create and save error plot
            if config:
                plot_dir = create_output_dir(experiment_id, config)
            else:
                plot_dir = f"../../data/experiments/{experiment_id}/plots"
                os.makedirs(plot_dir, exist_ok=True)
                
            return create_error_plot(error_msg, plot_dir, comparison_type, fixed_model, fixed_sample_size, fixed_strategy, tune_method)

    # Create a figure for the boxplots and legend, with no background
    fig_plots = plt.figure(figsize=FIGURE_SIZES['boxplot_figure'], facecolor='none')
    
    # Define a grid layout with appropriate height ratios: 4 parts for plots, 0.5 for padding, 1 for legend
    grid = plt.GridSpec(6, 1, height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.8)
    
    # Create a 2x2 grid in the top section for boxplots
    boxplot_grid = grid[:4].subgridspec(2, 2, wspace=0.1, hspace=0.2)
    
    # Create the boxplot axes
    axs = np.array([[fig_plots.add_subplot(boxplot_grid[i, j]) for j in range(2)] for i in range(2)])
    
    # Create a subplot for the legend at the bottom
    ax_legend = fig_plots.add_subplot(grid[4:])
    ax_legend.axis('off')  # Hide axes
    
    # Set titles
    titles = [
        'Proportional Overall Accuracy', 
        'Balanced Overall Accuracy',
        'Proportional F1 Score', 
        'Balanced F1 Score'
    ]
    
    # Get unique hue values and strategies/x-axis values
    if comparison_type == "testing_site":
        # For testing_site, we have same_site and cross_site as the hue values
        hue_values = ["same_site", "cross_site"]
        
        # Use training sites on the x-axis instead of testing sites
        strategies = sorted(filtered_df['training_sites'].unique())
    else:
        # For other comparison types, use standard approach
        hue_values = sorted(filtered_df[hue_variable].unique())
        strategies = sorted(filtered_df['training_sample_strategy'].unique(), reverse=True)
    
    # Create colorblind-friendly palette - 'colorblind' is a seaborn colorblind-friendly palette
    palette = sns.color_palette("colorblind", len(hue_values))
    
    # Variables to store legend handles and labels
    all_handles = []
    all_labels = []
    
    # Variables to track y-axis limits for top and bottom rows
    top_ymin, top_ymax = float('inf'), float('-inf')
    bottom_ymin, bottom_ymax = float('inf'), float('-inf')
    
    # Set font sizes for all texts
    plt.rcParams.update({
        'font.size': tick_fontsize,
        'axes.titlesize': subtitle_fontsize,
        'axes.labelsize': axis_label_fontsize,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize,
        'legend.fontsize': legend_fontsize,
    })
    
    # Plot each metric
    for i, metric in enumerate(available_metrics):
        if comparison_type == "testing_site":
            # For testing_site comparison, we need to check if both metrics exist
            same_site_metric = f"{metric}_same_site"
            cross_site_metric = f"{metric}_cross_site"
            
            if same_site_metric not in filtered_df.columns or cross_site_metric not in filtered_df.columns:
                row, col = i // 2, i % 2
                axs[row, col].text(0.5, 0.5, f"Metric '{metric}' not available in both site types",
                                ha='center', va='center')
                continue
        elif metric not in filtered_df.columns:
            row, col = i // 2, i % 2
            axs[row, col].text(0.5, 0.5, f"Metric '{metric}' not available",
                            ha='center', va='center')
            continue
            
        # Create a melted DataFrame for this metric
        melted_data = []
        
        # Organize data by strategy and hue value
        for strategy in strategies:
            if comparison_type == "testing_site":
                # For testing_site, strategy is actually the training site
                for hue_val in hue_values:
                    # Get the appropriate metric column based on site type
                    metric_col = f"{metric}_{hue_val}"
                    
                    # Filter for this training site
                    train_site_data = filtered_df[filtered_df['training_sites'] == strategy]
                    
                    if len(train_site_data) > 0:
                        for value in train_site_data[metric_col]:
                            melted_data.append({
                                'Training Sites': strategy,
                                hue_title: hue_val,
                                'Value': value
                            })
            else:
                # Standard approach for other comparison types
                for hue_val in hue_values:
                    # Filter condition based on comparison type
                    if comparison_type == "model":
                        filtered_condition = (
                            (filtered_df['training_sample_strategy'] == strategy) & 
                            (filtered_df[hue_variable] == hue_val)
                        )
                    elif comparison_type == "sample_size":
                        filtered_condition = (
                            (filtered_df['training_sample_strategy'] == strategy) & 
                            (filtered_df[hue_variable] == hue_val)
                        )
                    elif comparison_type == "training_site":
                        filtered_condition = (
                            (filtered_df['training_sample_strategy'] == strategy) & 
                            (filtered_df[hue_variable] == hue_val)
                        )
                    else:
                        # Default fallback (should not happen with proper validation)
                        filtered_condition = (
                            (filtered_df['training_sample_strategy'] == strategy) & 
                            (filtered_df[hue_variable] == hue_val)
                        )
                    
                    strategy_data = filtered_df[filtered_condition]
                    
                    if len(strategy_data) > 0:
                        for value in strategy_data[metric]:
                            melted_data.append({
                                'Sampling Strategy': strategy,
                                hue_title: str(hue_val),
                                'Value': value
                            })
        
        # Convert to DataFrame
        plot_df = pd.DataFrame(melted_data)
        
        # Plot in the appropriate subplot
        row, col = i // 2, i % 2
        
        # Create boxplot with individual points
        ax = axs[row, col]
        
        # Set transparent subplot background
        ax.set_facecolor('none')
        
        if comparison_type == "testing_site":
            # For testing_site comparison, use 'Training Sites' on x-axis
            sns.boxplot(
                data=plot_df, 
                x='Training Sites', 
                y='Value', 
                hue=hue_title,
                palette=palette,
                ax=ax,
                showfliers=True,
                linewidth=boxplot_linewidth
            )
            
            # Add individual points if requested
            if show_points:
                sns.stripplot(
                    data=plot_df, 
                    x='Training Sites', 
                    y='Value', 
                    hue=hue_title,
                    palette=palette,
                    dodge=True,
                    alpha=0.5,
                    ax=ax
                )
        else:
            # Standard approach for other comparison types
            sns.boxplot(
                data=plot_df, 
                x='Sampling Strategy', 
                y='Value', 
                hue=hue_title,
                palette=palette,
                ax=ax,
                showfliers=True,
                linewidth=boxplot_linewidth
            )
            
            # Add individual points if requested
            if show_points:
                sns.stripplot(
                    data=plot_df, 
                    x='Sampling Strategy', 
                    y='Value', 
                    hue=hue_title,
                    palette=palette,
                    dodge=True,
                    alpha=0.5,
                    ax=ax
                )
            
        # Store handles and labels for the common legend (but only from the first subplot)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            # Only keep the first set of labels (from the boxplot, not stripplot)
            num_unique_hues = len(hue_values)
            all_handles = handles[:num_unique_hues]
            all_labels = labels[:num_unique_hues]
            
        # Remove the individual subplot legend
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        
        # Set title - make it bold
        if comparison_type == "testing_site":
            ax.set_title(titles[i], fontsize=subtitle_fontsize, fontweight='bold')
        else:
            ax.set_title(titles[i], fontsize=subtitle_fontsize, fontweight='bold')
        
        # Remove x-axis labels for top subplots
        if row == 0:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        else:
            if comparison_type == "testing_site":
                ax.set_xlabel('Training Sites', fontsize=axis_label_fontsize)
            else:
                ax.set_xlabel('Sampling Strategy', fontsize=axis_label_fontsize)
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=tick_fontsize)
        
        # Remove y-axis labels for right column subplots
        if col == 1:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Performance', fontsize=axis_label_fontsize)
        
        # Set tick font size
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # Set subplot border width
        for spine in ax.spines.values():
            spine.set_linewidth(subplot_borderwidth)
        
        # Add gridlines with specified width
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=gridline_width)
        
        # Track y-axis limits for synchronization
        y_min, y_max = ax.get_ylim()
        if row == 0:  # Top row
            top_ymin = min(top_ymin, y_min)
            top_ymax = max(top_ymax, y_max)
        else:  # Bottom row
            bottom_ymin = min(bottom_ymin, y_min)
            bottom_ymax = max(bottom_ymax, y_max)
    
    # Synchronize y-axis limits for top row
    for col in range(2):
        axs[0, col].set_ylim(top_ymin, top_ymax)
    
    # Synchronize y-axis limits for bottom row
    for col in range(2):
        axs[1, col].set_ylim(bottom_ymin, bottom_ymax)
    
    # Add overall title
    if comparison_type == "testing_site":
        plt.suptitle(f"Performance by Training Sites & {hue_title} across iterations\n(Filtered by: {title_filter} & Tuning: {tune_method})", 
                   fontsize=title_fontsize, y=0.98, fontweight='bold')
    elif comparison_type == "model":
        plt.suptitle(f"Performance by Sampling Strategy & {hue_title} across iterations and LOGO-Folds\n(Filtered by: {title_filter} & Tuning: {tune_method})", 
                   fontsize=title_fontsize, y=0.98, fontweight='bold') 
    elif comparison_type == "sample_size":
        plt.suptitle(f"Performance by Sampling Strategy & {hue_title} across iterations and LOGO-Folds\n(Filtered by: {title_filter} & Tuning: {tune_method})", 
                   fontsize=title_fontsize, y=0.98, fontweight='bold')
    elif comparison_type == "training_site":
        plt.suptitle(f"Performance by Sampling Strategy & {hue_title} across iterations\n(Filtered by: {title_filter} & Tuning: {tune_method})", 
                   fontsize=title_fontsize, y=0.98, fontweight='bold')
    elif comparison_type == "tuning_strategy":
        plt.suptitle(f"Performance by Sampling Strategy & {hue_title} across iterations and LOGO-Folds\n(Filtered by: {title_filter})", 
                   fontsize=title_fontsize, y=0.98, fontweight='bold')
    else:
        plt.suptitle(f"Performance by Sampling Strategy & {hue_title} across iterations and LOGO-Folds\n(Filtered by: {title_filter} & Tuning: {tune_method})", 
                   fontsize=title_fontsize, y=0.98, fontweight='bold')
    
    # Add a single legend at the bottom
    if all_handles and all_labels:
        # Add legend to the dedicated legend axes, no background no outline
        leg = ax_legend.legend(
            all_handles, 
            all_labels, 
            loc='center',
            ncol=min(len(all_labels), 5),
            title=hue_title,
            fontsize=legend_fontsize,
            frameon=False,
            facecolor='none',
            title_fontsize=legend_fontsize + 1
        )
    
    # Apply tight_layout to boxplot figure
    fig_plots.tight_layout()
    
    # Save the boxplot figure
    if config:
        plot_dir = create_output_dir(experiment_id, config)
    else:
        plot_dir = f"../../data/experiments/{experiment_id}/plots"
        os.makedirs(plot_dir, exist_ok=True)
        
    filter_str = ""
    if fixed_model:
        filter_str += f"model_{str(fixed_model).replace(' ', '_')}_"
    if fixed_sample_size:
        filter_str += f"size_{fixed_sample_size}_"
    if fixed_strategy:
        filter_str += f"strategy_{str(fixed_strategy).replace(' ', '_')}_"
    if not filter_str:
        filter_str = "all_"
        
    if comparison_type == "testing_site":
        output_path_plots = f"{plot_dir}/testing_site_performance_{filter_str}{tune_method}.png"
    else:
        site_suffix = "same_site" if site_type == "same_site" else "cross_site"
        output_path_plots = f"{plot_dir}/strategy_performance_{comparison_type}_{filter_str}{tune_method}_{site_suffix}.png"
        
    # Save the boxplot figure
    fig_plots.savefig(output_path_plots, dpi=300, bbox_inches='tight')
    
    # Now create the table data and figure
    summary_data = []
    
    # Define the 4 main metrics we want to display
    summary_metric_names = ["proportional_OA", "balanced_OA", "proportional_F1", "balanced_F1"]
    display_metric_names = ["Proportional OA", "Balanced OA", "Proportional F1", "Balanced F1"]
    
    # Create the dataframe with the restructured data
    if comparison_type == "testing_site":
        # For testing_site comparison, rows are training sites with hue values
        metrics_data = {}
        
        # Initialize columns for the metrics
        for metric_display in display_metric_names:
            metrics_data[metric_display] = []
        
        # Row labels will be combination of site and hue
        row_labels = []
        
        # Process each training site and hue combination
        for strategy in strategies:
            for hue_val in hue_values:
                # Row name combining site and evaluation type
                row_name = f"{strategy}_{hue_val}"
                row_labels.append(row_name)
                
                # Filter data for this site and hue
                site_data = filtered_df[filtered_df['training_sites'] == strategy]
                metric_suffix = hue_val  # same_site or cross_site
                
                # Calculate mean for each metric
                for i, base_metric in enumerate(summary_metric_names):
                    metric = f"{base_metric}_{metric_suffix}"
                    if metric in site_data.columns:
                        mean_value = site_data[metric].mean()
                        metrics_data[display_metric_names[i]].append(f"{mean_value:.3f}")
                    else:
                        metrics_data[display_metric_names[i]].append("N/A")
    else:
        # For other comparison types, rows are strategies with hue values
        metrics_data = {}
        
        # Initialize columns for the metrics
        for metric_display in display_metric_names:
            metrics_data[metric_display] = []
        
        # Row labels will be combination of strategy and hue
        row_labels = []
        
        # Process each strategy and hue combination
        for strategy in strategies:
            for hue_val in hue_values:
                # Create row name by combining strategy and hue value
                row_name = f"{strategy}_{hue_val}"
                row_labels.append(row_name)
                
                # Filter data for this strategy and hue
                strategy_data = filtered_df[
                    (filtered_df['training_sample_strategy'] == strategy) & 
                    (filtered_df[hue_variable] == hue_val)
                ]
                
                # Calculate mean for each metric
                for i, base_metric in enumerate(summary_metric_names):
                    metric = f"{base_metric}_{site_type}"
                    if metric in strategy_data.columns and len(strategy_data) > 0:
                        mean_value = strategy_data[metric].mean()
                        metrics_data[display_metric_names[i]].append(f"{mean_value:.3f}")
                    else:
                        metrics_data[display_metric_names[i]].append("N/A")
    
    # Create DataFrame with row labels as index
    summary_df = pd.DataFrame(metrics_data, index=row_labels)
    summary_df.index.name = f"Strategy/Site_{hue_title}"
    
    # Store numeric values for colormapping
    numeric_data = {}
    for col in summary_df.columns:
        numeric_data[col] = []
        for val in summary_df[col]:
            try:
                numeric_data[col].append(float(val))
            except (ValueError, TypeError):
                numeric_data[col].append(np.nan)
    
    numeric_df = pd.DataFrame(numeric_data, index=summary_df.index)
    
    # Only create the table if we have data
    if not summary_df.empty and len(summary_df.columns) > 0:
        # Calculate dynamic figure size for table
        n_rows = len(summary_df.index)
        base_height = 1  # minimum height including space for title
        height_per_row = 0.7  # additional height per row
        
        # Calculate title height based on number of lines in title
        if comparison_type == "testing_site":
            title = f"Performance by Training Sites & {hue_title} across iterations\n(Filtered by: {title_filter}, Tuning: {tune_method})"
        elif comparison_type == "tuning_strategy":
            title = f"Performance by Sampling Strategy & {hue_title} across iterations and LOGO-Folds\n(Filtered by: {title_filter})"
        else:
            title = f"Performance by Sampling Strategy & {hue_title} across iterations and LOGO-Folds\n(Filtered by: {title_filter}, Tuning: {tune_method})"
        
        title_lines = len(title.split('\n'))
        title_height = title_lines * 1  # 1 unit per title line
        
        # Calculate total figure height
        fig_height = max(base_height, (n_rows) * height_per_row + title_height)
        
        # Create a separate figure for the table with dynamic height
        fig_table = plt.figure(figsize=(16, fig_height))
        
        # Adjust subplot position to leave room for title
        plt.subplots_adjust(top=0.9, bottom=0.1)
        
        ax_table = fig_table.add_subplot(111)
        ax_table.axis('off')  # Hide axes
        
        # Add table title
        ax_table.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=10)
        
        # Convert DataFrame to arrays for the table
        cell_text = summary_df.values
        row_labels = summary_df.index.tolist()
        col_labels = summary_df.columns.tolist()
        
        # Create the table with adjusted position
        table = ax_table.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
        
        # Style the table using parameters from TABLE_STYLE
        table.auto_set_font_size(False)
        table.set_fontsize(TABLE_STYLE['fontsize'])
        
        # Scale table to fit the axes
        table.scale(1, 3)
        
        # Import matplotlib colormap for cell backgrounds (use existing import)
        import matplotlib.colors as mcolors
        
        # Create colormap instance - using the recommended approach instead of deprecated get_cmap
        cmap = plt.colormaps[TABLE_COLORMAP['cmap_name']]
        if TABLE_COLORMAP['invert_cmap']:
            cmap = cmap.reversed()
        
        # Function to determine text color based on background brightness
        def get_text_color(bg_color):
            """Determine whether to use black or white text based on background color"""
            if isinstance(bg_color, str):  # If it's a hex color
                bg_color = mcolors.to_rgb(bg_color)
            elif len(bg_color) == 4:  # If it's RGBA, ignore alpha
                bg_color = bg_color[:3]
            
            # Calculate relative luminance
            luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
            return 'black' if luminance > 0.5 else 'white'

        # Apply cell styling and colormapping
        for (row, col), cell in table.get_celld().items():
            # Set border style
            cell.set_edgecolor(TABLE_STYLE['cell_border_color'])
            cell.set_linewidth(TABLE_STYLE['cell_border_width'])
            
            if row == 0:  # Header row
                cell.set_facecolor(TABLE_STYLE['header_bg_color'])
                cell.set_text_props(
                    weight='bold', 
                    color=get_text_color(TABLE_STYLE['header_bg_color']),
                    fontsize=TABLE_STYLE['header_fontsize']
                )
            elif col == -1:  # Row labels
                cell.set_facecolor(TABLE_STYLE['row_header_bg_color'])
                cell.set_text_props(
                    weight='bold',
                    color=get_text_color(TABLE_STYLE['row_header_bg_color'])
                )
            else:  # Data cells
                # Apply colormapping if enabled and column is in the list
                if TABLE_COLORMAP['enabled'] and col < len(col_labels) and col_labels[col] in TABLE_COLORMAP['apply_to_columns']:
                    column_name = col_labels[col]
                    
                    # Get numeric data for this column
                    col_data = numeric_df[column_name].dropna()
                    
                    if len(col_data) > 0:
                        # Calculate min and max with buffer for the colormap range
                        min_val = col_data.min()
                        max_val = col_data.max()
                        
                        # Add buffer zone if values are close
                        if max_val - min_val < 0.001:
                            buffer = 0.001
                        else:
                            buffer = (max_val - min_val) * TABLE_COLORMAP['value_range_buffer']
                        
                        # Create normalized colormap range with buffer
                        norm = mcolors.Normalize(vmin=min_val - buffer, vmax=max_val + buffer)
                        
                        # Get cell value
                        try:
                            cell_value = numeric_df.iloc[row-1, col]
                            if not np.isnan(cell_value):
                                # Get color from colormap
                                cell_color = cmap(norm(cell_value))
                                
                                # Set cell background color
                                cell.set_facecolor(cell_color)
                                
                                # Set contrasting text color and make bold
                                text_color = get_text_color(cell_color)
                                cell.get_text().set_color(text_color)
                                cell.get_text().set_weight('bold')
                        except (ValueError, IndexError):
                            # Skip if there's an issue with the value
                            pass
                else:
                    # Set default text color for non-colored cells and make bold
                    cell.get_text().set_color('black')
                    cell.get_text().set_weight('bold')
        
        # Remove the tight_layout call for table figure
        #fig_table.tight_layout()
        
        # Save the table figure without background
        output_path_table = output_path_plots.replace('.png', '_table.png')
        fig_table.savefig(output_path_table, dpi=300, bbox_inches='tight', facecolor='none')
        
        # Show both figures
        plt.figure(fig_plots.number)
        plt.show()
        plt.figure(fig_table.number)
        plt.show()
    else:
        # Just show the boxplot figure if no table
        plt.figure(fig_plots.number)
    plt.show()
    
    # Return the path to the boxplot figure
    return output_path_plots

# %%
# Define shared plotting parameters
# Use default plot parameters from configuration
plot_params = DEFAULT_PLOT_PARAMS

# Only run this if the script is executed directly (not imported)
if __name__ == "__main__":
    # Load data when running as a standalone script
    results_df, config = load_experiment_data()
    
    if results_df is not None and config is not None:
        # Example 1: Compare different sample sizes for Random Forest model
        print("\nGenerating plot comparing different sample sizes for Random Forest model...")
        plot_strategies_performance(
            results_df=results_df,
            comparison_type="sample_size",
            fixed_model="Random Forest",
            tune_method="no_tuning",
            experiment_id=config["experiment_id"],
            site_type="same_site",
            config=config,
            show_points=True,
            **plot_params
        )
        
        # Example 2: Compare different models for sample size of 500
        print("\nGenerating plot comparing different models for sample size of 500...")
        plot_strategies_performance(
            results_df=results_df,
            comparison_type="model", 
            fixed_sample_size=500,
            tune_method="no_tuning",
            experiment_id=config["experiment_id"],
            site_type="same_site",
            config=config,
            show_points=True,
            **plot_params
        )
        
        # Example 3: Compare different training sites for Random Forest with sample size of 500
        print("\nGenerating plot comparing different training sites for Random Forest with sample size of 500...")
        plot_strategies_performance(
            results_df=results_df,
            comparison_type="training_site",
            fixed_model="Random Forest",
            fixed_sample_size=500,
            tune_method="no_tuning",
            experiment_id=config["experiment_id"],
            site_type="same_site",
            config=config,
            show_points=True,
            **plot_params
        )
        
        # Example 4: Compare testing sites with same_site vs cross_site metrics
        print("\nGenerating plot comparing testing sites with same-site vs cross-site performance...")
        plot_strategies_performance(
            results_df=results_df,
            comparison_type="testing_site",
            fixed_model="Random Forest",
            fixed_strategy="simple_random",  # Filter to just one sampling strategy
            fixed_sample_size=500,  # Add fixed sample size filter
            tune_method="no_tuning",
            experiment_id=config["experiment_id"],
            config=config,
            show_points=True,
            **plot_params
        )
        
        # Example 5: Compare different tuning strategies for Random Forest with sample size of 500
        print("\nGenerating plot comparing different tuning strategies...")
        plot_strategies_performance(
            results_df=results_df,
            comparison_type="tuning_strategy",
            fixed_model="Random Forest",
            fixed_sample_size=500,
            experiment_id=config["experiment_id"],
            site_type="same_site",
            config=config,
            show_points=True,
            **plot_params
        )
    else:
        print("Could not load experiment data. Please make sure experiment_config.json exists and contains valid paths.")

# %%
