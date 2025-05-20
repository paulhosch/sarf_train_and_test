# %%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re
from typing import Dict, Any, Optional

# %%
# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualisation.vis_utils import load_experiment_data
from src.visualisation.vis_config import DEFAULT_PLOT_PARAMS

# %%
def parse_best_params(param_str):
    """
    Parse best_params from various formats including OrderedDict and string representations.
    """
    if pd.isna(param_str) or param_str == '' or param_str == 'None':
        return {}
    
    try:
        # If it's already a dict
        if isinstance(param_str, dict):
            return param_str
        
        # If it's a string representation of OrderedDict (from collections)
        if isinstance(param_str, str) and 'OrderedDict' in param_str:
            # Extract contents from OrderedDict([('param', value), ...])
            params_content = param_str.replace('OrderedDict(', '').replace('])', '').replace('[', '')
            # Split by parentheses to get individual tuples
            param_pairs = re.findall(r"\('([^']+)', (\d+)\)", params_content)
            return {param: int(value) for param, value in param_pairs}
        
        # Try the default ast.literal_eval approach
        return ast.literal_eval(param_str)
    except Exception as e:
        print(f"Error parsing best_params: {e} - Value: {param_str[:100]}")
        return {}

# %%
def plot_best_params_by_param(results_df: pd.DataFrame,
                             model_filter: str = "Random Forest",
                             sample_size_filter: int = 1000,
                             tune_method: str = "BayesSearchCV",
                             experiment_id: str = "main_experiment",
                             config: Optional[Dict] = None,
                             title_fontsize: int = DEFAULT_PLOT_PARAMS['title_fontsize'],
                             subtitle_fontsize: int = DEFAULT_PLOT_PARAMS['subtitle_fontsize'],
                             axis_label_fontsize: int = DEFAULT_PLOT_PARAMS['axis_label_fontsize'],
                             tick_fontsize: int = DEFAULT_PLOT_PARAMS['tick_fontsize'],
                             legend_fontsize: int = DEFAULT_PLOT_PARAMS['legend_fontsize'],
                             gridline_width: float = DEFAULT_PLOT_PARAMS['gridline_width'],
                             subplot_borderwidth: float = DEFAULT_PLOT_PARAMS['subplot_borderwidth'],
                             bar_linewidth: float = DEFAULT_PLOT_PARAMS['bar_linewidth'],
                             boxplot_linewidth: float = DEFAULT_PLOT_PARAMS['boxplot_linewidth']) -> str:
    """
    Plot best_params for each parameter as a subplot, with param value on y, training_sites on x, colored by sampling strategy.
    Also includes horizontal lines for default parameter values and parameter search space.
    
    Args:
        results_df: DataFrame containing results
        model_filter: Filter to specific model
        sample_size_filter: Filter to specific sample size
        tune_method: Filter results to this tuning method (e.g., "BayesSearchCV")
        experiment_id: ID of the experiment
        config: Experiment configuration with data paths
        title_fontsize: Font size for main title
        subtitle_fontsize: Font size for subplot titles
        axis_label_fontsize: Font size for axis labels
        tick_fontsize: Font size for axis ticks
        legend_fontsize: Font size for legend
        gridline_width: Line width for gridlines
        subplot_borderwidth: Line width for subplot borders
        bar_linewidth: Not used, included for compatibility with DEFAULT_PLOT_PARAMS
        boxplot_linewidth: Not used, included for compatibility with DEFAULT_PLOT_PARAMS
        
    Returns:
        Path to saved plot
    """
    # Filter results
    # Check if tuning_method column exists, if not extract from configuration_name
    if 'tuning_method' not in results_df.columns and 'configuration_name' in results_df.columns:
        # Extract tuning method from configuration name
        results_df['tuning_method'] = results_df['configuration_name'].apply(
            lambda x: 'BayesSearchCV' if 'BayesSearchCV' in x else 'no_tuning' if 'no_tuning' in x else tune_method
        )
        
    filtered_df = results_df[
        (results_df['tuning_method'] == tune_method) &
        (results_df['model'] == model_filter) &
        (results_df['training_sample_size'] == sample_size_filter)
    ]
    if len(filtered_df) == 0:
        print(f"No data found for model={model_filter}, sample_size={sample_size_filter}, tune_method={tune_method}")
        return None

    # Parse best_params into columns
    if 'best_params' not in filtered_df.columns:
        print("Error: 'best_params' column not found in the filtered DataFrame")
        print(f"Available columns after filtering: {filtered_df.columns.tolist()}")
        return None
    
    # Show a sample of the best_params values to help diagnose parsing issues
    print("Sample best_params value:", filtered_df['best_params'].iloc[0] if len(filtered_df) > 0 else "No samples")
    
    # Create a copy to avoid SettingWithCopyWarning
    plot_df = filtered_df.copy()
    
    param_dicts = plot_df['best_params'].apply(parse_best_params)
    all_params = set()
    for d in param_dicts:
        all_params.update(d.keys())
    all_params = sorted(all_params)
    
    print(f"Parameters found: {all_params}")
    
    if not all_params:
        print("Warning: No parameters found in best_params")
        return None

    # Expand best_params into separate columns
    for param in all_params:
        plot_df.loc[:, param] = param_dicts.apply(lambda d: d.get(param, np.nan))

    # Prepare for plotting
    n_params = len(all_params)
    if n_params == 0:
        print("No parameters found in best_params. Check if the parameters are correctly formatted.")
        return None
    
    # Get parameter information from config
    if config and 'model_parameters' in config and model_filter in config['model_parameters']:
        model_config = config['model_parameters'][model_filter]
        default_params = model_config.get('default_parameters', {})
        tunable_params = model_config.get('tunable_parameters', {})
        print(f"Found default parameters: {default_params}")
        print(f"Found tunable parameters: {tunable_params}")
    else:
        default_params = {}
        tunable_params = {}
        print("Warning: Could not find parameter configuration in config")
    
    # Calculate figure dimensions based on content
    n_cols = 1
    n_rows = n_params
    
    # Calculate heights needed
    title_height = 1  # Space for title in inches
    subplot_height = 4  # Height per subplot in inches
    legend_height = 2  # Space for legend in inches
    
    # Total figure height
    total_height = title_height + (n_rows * subplot_height) + legend_height
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, total_height), squeeze=False)

    # Set up for combined legends
    strategy_handles = []
    param_handles = []
    param_labels = []
    
    # Create color palette - use a consistent one for all plots
    if 'training_sample_strategy' in plot_df.columns:
        unique_strategies = sorted(plot_df['training_sample_strategy'].unique())
        strategy_palette = sns.color_palette("colorblind", n_colors=len(unique_strategies))
    else:
        strategy_palette = sns.color_palette("colorblind", n_colors=1)
    
    # Force conversion to string then to numeric to handle different formats
    plot_df.loc[:, 'training_sites_str'] = plot_df['training_sites'].astype(str)
    
    try:
        # Try direct conversion to numeric
        plot_df.loc[:, 'training_sites_numeric'] = pd.to_numeric(plot_df['training_sites_str'], errors='raise')
        x_labels = sorted(plot_df['training_sites'].unique())
        x_positions = sorted(plot_df['training_sites_numeric'].unique())
    except (ValueError, TypeError) as e:
        print(f"Direct numeric conversion failed: {e}")
        # Use integer mapping instead
        unique_sites = sorted(plot_df['training_sites_str'].unique())
        site_mapping = {site: i for i, site in enumerate(unique_sites)}
        plot_df.loc[:, 'training_sites_numeric'] = plot_df['training_sites_str'].map(site_mapping)
        
        # Store the mapping for axis labels
        x_labels = unique_sites
        x_positions = list(range(len(unique_sites)))
    
    # Instead of random jitter, create a systematic offset based on sampling strategy
    if 'training_sample_strategy' in plot_df.columns:
        # Get unique strategies and create a mapping for offsets
        unique_strategies = sorted(plot_df['training_sample_strategy'].unique())
        strategy_offsets = {}
        
        # Calculate spacing between strategies
        n_strategies = len(unique_strategies)
        spacing = 0.8 / (n_strategies + 1)  # 0.8 to keep points within the x-tick bounds
        
        # Create offset mapping for each strategy
        for i, strategy in enumerate(unique_strategies):
            # Center strategies around the x-tick, with equal spacing
            offset = -0.4 + (spacing * (i + 1))
            strategy_offsets[strategy] = offset
        
        # Apply offsets based on strategy
        plot_df.loc[:, 'x_offset'] = plot_df['training_sample_strategy'].map(strategy_offsets)
        
        # Add small random jitter to avoid perfect overlap of multiple iterations
        small_jitter = np.random.normal(0, 0.02, size=len(plot_df))
        plot_df.loc[:, 'x_jittered'] = plot_df['training_sites_numeric'] + plot_df['x_offset'] + small_jitter
    else:
        # If no strategy column, add small jitter
        small_jitter = np.random.normal(0, 0.05, size=len(plot_df))
        plot_df.loc[:, 'x_jittered'] = plot_df['training_sites_numeric'] + small_jitter
    
    for i, param in enumerate(all_params):
        ax = axes[i, 0]
        
        # Check if necessary columns for plotting exist
        if 'training_sites' not in plot_df.columns:
            print(f"Error: 'training_sites' column not found, can't create scatterplot")
            return None
            
        # Use 'training_sample_strategy' for coloring if available, else use a constant color
        if 'training_sample_strategy' in plot_df.columns:
            # Scatterplot: x=training_sites, y=param value, color by training_sample_strategy
            scatter = sns.scatterplot(
                data=plot_df,
                x='x_jittered',   # Use jittered x values
                y=param,
                hue='training_sample_strategy',
                palette=strategy_palette,
                ax=ax,
                s=60,
                edgecolor='none',  # Remove marker outlines
                alpha=0.6,  # Set transparency to 0.6
                legend=False  # Don't show legend on individual plots
            )
            
            # We'll create manual legend patches later, so no need to collect handles here
        else:
            # No training_sample_strategy column, use constant color
            print("Warning: 'training_sample_strategy' column not found, using constant color")
            sns.scatterplot(
                data=plot_df,
                x='x_jittered',   # Use jittered x values
                y=param,
                color=strategy_palette[0],
                ax=ax,
                s=60,
                edgecolor='none',  # Remove marker outlines
                alpha=0.6,  # Set transparency to 0.6
            )
        
        # Add horizontal lines for default value and search space
        # Default parameter (solid black line)
        if param in default_params:
            default_value = default_params[param]
            # Handle 'null' value (None in Python)
            if default_value is None and param == 'max_depth':
                # For max_depth, None typically means no limit, so use a high value
                default_value = plot_df[param].max() * 1.1
            
            line = ax.axhline(y=default_value, color='black', linestyle='-', linewidth=1.5, 
                             alpha=0.8, zorder=1)
            
            # Add to legend handles only once
            if i == 0:
                param_handles.append(line)
                param_labels.append('Default value')
        
        # Parameter search space (dashed grey lines)
        if param in tunable_params:
            param_range = tunable_params[param]
            if isinstance(param_range, list):
                for j, value in enumerate(param_range):
                    # Handle 'null' value (None in Python)
                    if value is None and param == 'max_depth':
                        # For max_depth, None typically means no limit, so use a high value
                        value = plot_df[param].max() * 1.1
                    
                    # Only add first value to legend
                    if j == 0 and i == 0:
                        line = ax.axhline(y=value, color='grey', linestyle='--', linewidth=1, 
                                         alpha=0.6, zorder=0)
                        param_handles.append(line)
                        param_labels.append('Search space')
                    else:
                        ax.axhline(y=value, color='grey', linestyle='--', linewidth=1, 
                                  alpha=0.6, zorder=0)
        
        ax.set_title(f"Best {param}", fontsize=subtitle_fontsize, fontweight='bold')
        ax.set_xlabel('Training Sites', fontsize=axis_label_fontsize)
        ax.set_ylabel(param, fontsize=axis_label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # Remove gridlines
        ax.grid(False)
        
        # Set x-ticks to the original training_sites values, not the jittered ones
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        
        for spine in ax.spines.values():
            spine.set_linewidth(subplot_borderwidth)
        
        # Remove individual legends
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.suptitle(f"Best Parameters by Training Sites\nModel: {model_filter}, Sample Size: {sample_size_filter}", 
                fontsize=title_fontsize, fontweight='bold')
    
    # Add a single combined legend for both strategy colors and parameter lines
    # Calculate legend position to place it at the bottom of the figure
    legend_bottom = 0.05  # Position from bottom of figure (in figure coordinates)
    
    # Manually create legend handles for sampling strategies
    strategy_patches = []
    strategy_labels = []
    if 'training_sample_strategy' in plot_df.columns:
        unique_strategies = sorted(plot_df['training_sample_strategy'].unique())
        for i, strategy in enumerate(unique_strategies):
            color = strategy_palette[i]
            # Create larger, more visible markers for the legend
            patch = plt.Line2D([0], [0], marker='o', color='none', 
                             markerfacecolor=color, 
                             markersize=12,  # Larger marker
                             markeredgecolor='black', 
                             markeredgewidth=1.0,  # Thicker edge
                             alpha=1.0)  # Full opacity for legend
            strategy_patches.append(patch)
            strategy_labels.append(strategy)
    
    # Make sure we have the parameter labels
    if 'Default value' not in param_labels and len(param_handles) > 0:
        param_labels = ['Default value', 'Search space']
    
    # Combine manually created patches with parameter line handles
    combined_handles = param_handles + strategy_patches
    combined_labels = param_labels + strategy_labels
    
    # Create a single combined legend with transparent background and no frame
    if combined_handles and combined_labels:
        # Calculate how many rows we need for the legend (with at most 4 items per row)
        n_items = len(combined_handles)
        n_rows_legend = np.ceil(n_items / 4)
        
        # Adjust legend spacing based on number of rows
        legend_height_ratio = 0.15 + (0.03 * max(0, n_rows_legend - 2))  # Add space for additional rows
        
        combined_legend = fig.legend(
            handles=combined_handles, 
            labels=combined_labels, 
            loc='lower center', 
            bbox_to_anchor=(0.5, legend_height_ratio/2),  # Positioned in middle of reserved space
            ncol=min(4, n_items),  # At most 4 columns
            fontsize=legend_fontsize,
            frameon=False,  # No frame
            title="Parameters and Sampling Strategies",
            title_fontsize=legend_fontsize + 1,
            labelspacing=1.2,  # More vertical space between legend items
            handletextpad=0.8,  # More space between handle and text
            borderpad=0.5  # Padding between legend content and edge
        )
        # Set legend background to transparent
        combined_legend.get_frame().set_facecolor('none')
    
    # Adjust layout to make space for legend at bottom
    plt.tight_layout(rect=[0.05, legend_height_ratio, 0.95, 0.95])

    # Save plot
    if config:
        plot_dir = os.path.join(config['data_paths']['output_data'], experiment_id, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = f"../../data/experiments/{experiment_id}/plots"
        os.makedirs(plot_dir, exist_ok=True)
    output_path = os.path.join(plot_dir, f"best_params_model_{model_filter.replace(' ', '_')}_size_{sample_size_filter}_{tune_method}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    return output_path

# %%
# Example usage if run as script or interactively
results_df, config = load_experiment_data()
if results_df is not None and config is not None:
    plot_best_params_by_param(
        results_df=results_df,
        model_filter="Random Forest",
        sample_size_filter=1000,
        tune_method="BayesSearchCV",
        experiment_id=config["experiment_id"],
        config=config,
        **DEFAULT_PLOT_PARAMS
    )
else:
    print("Could not load experiment data. Please make sure experiment_config.json exists and contains valid paths.")

# %%
