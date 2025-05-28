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
import matplotlib.colors as mcolors

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules from the project
from src.visualisation.vis_utils import (load_experiment_data, parse_feature_importance, 
                                        set_plot_fonts, apply_plot_style, get_output_directory, create_output_dir)
from src.visualisation.vis_config import (DEFAULT_PLOT_PARAMS, FIGURE_SIZES, 
                                         COLOR_PALETTES, PLOT_STYLE, OUTPUT_SETTINGS,
                                         TABLE_STYLE, TABLE_COLORMAP)

# Define manual feature order
FEATURE_ORDER = [
    'VV_PRE', 'VH_PRE', 
    'VV_POST', 'VH_POST', 
    'VV_CHANGE', 'VH_CHANGE',
    'VV_VH_RATIO_PRE', 'VV_VH_RATIO_POST', 'VV_VH_RATIO_CHANGE',
    'HAND', 'EDTW', 'SLOPE',
    'LAND_COVER_Water', 'LAND_COVER_Trees', 'LAND_COVER_Flooded Vegetation',
    'LAND_COVER_Crops', 'LAND_COVER_Built Area', 'LAND_COVER_Bare Ground',
    'LAND_COVER_Snow and Ice', 'LAND_COVER_Clouds', 'LAND_COVER_Rangeland'
]
def format_feature_name(feature: str) -> str:
    """
    Format feature names for better display on plots.
    Shortens LAND_COVER_ prefixed features to more concise form.
    Places the land cover type and (LC) suffix on separate lines for better readability.
    
    Args:
        feature: Original feature name
        
    Returns:
        Formatted feature name for display
    """
    if feature.startswith('LAND_COVER_'):
        # Extract the part after LAND_COVER_ and add (LC) suffix on a new line
        land_cover_type = feature.replace('LAND_COVER_', '')
        return f"{land_cover_type}_LC"
    return feature

# After FEATURE_ORDER definition, add DISPLAY_FEATURE_ORDER
DISPLAY_FEATURE_ORDER = [format_feature_name(feature) for feature in FEATURE_ORDER]


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

def plot_feature_importance_by_strategy(results_df: pd.DataFrame,
                                      model_filter: str = "Random Forest",
                                      sample_size_filter: int = 1000,
                                      strategy_filter: Optional[str] = None,
                                      tune_method: str = "no_tuning",
                                      experiment_id: str = "main_experiment",
                                      group_by: str = "strategy",
                                      config: Dict = None,
                                      min_importance: float = 0.01,
                                      show_points: bool = True,
                                      title_fontsize: int = DEFAULT_PLOT_PARAMS['title_fontsize'],
                                      subtitle_fontsize: int = DEFAULT_PLOT_PARAMS['subtitle_fontsize'],
                                      axis_label_fontsize: int = DEFAULT_PLOT_PARAMS['axis_label_fontsize'],
                                      tick_fontsize: int = DEFAULT_PLOT_PARAMS['tick_fontsize'],
                                      legend_fontsize: int = DEFAULT_PLOT_PARAMS['legend_fontsize'],
                                      gridline_width: float = DEFAULT_PLOT_PARAMS['gridline_width'],
                                      bar_linewidth: float = DEFAULT_PLOT_PARAMS['bar_linewidth'],
                                      boxplot_linewidth: float = DEFAULT_PLOT_PARAMS['boxplot_linewidth'],
                                      subplot_borderwidth: float = DEFAULT_PLOT_PARAMS['subplot_borderwidth']) -> str:
    """
    Plot feature importance by sampling strategy or training site with separate subplots using bar plots
    
    Args:
        results_df: DataFrame containing results
        model_filter: Filter to specific model
        sample_size_filter: Filter to specific sample size
        strategy_filter: Filter to specific sampling strategy (when group_by='training_site')
        tune_method: Filter results to this tuning method (e.g., "no_tuning", "BayesSearchCV")
        experiment_id: ID of the experiment
        group_by: How to group the data - 'strategy' or 'training_site'
        config: Experiment configuration with data paths
        min_importance: Minimum importance value to include feature (aggregate others)
        show_points: Whether to show error bars on the bar plots
        title_fontsize: Font size for main title
        subtitle_fontsize: Font size for subplot titles
        axis_label_fontsize: Font size for axis labels
        tick_fontsize: Font size for axis ticks
        legend_fontsize: Font size for legend
        gridline_width: Line width for gridlines
        bar_linewidth: Line width for bars
        boxplot_linewidth: Line width for boxplots
        subplot_borderwidth: Line width for subplot borders
        
    Returns:
        Path to saved plot
    """
    # Filter results by tuning method, model, and sample size
    if 'tuning_method' not in results_df.columns:
        # Extract tuning method from configuration name if not available
        results_df['tuning_method'] = results_df['configuration_name'].apply(
            lambda x: 'BayesSearchCV' if 'BayesSearchCV' in x else 'no_tuning' if 'no_tuning' in x else None
        )
    
    # Base filters
    filtered_df = results_df[
        (results_df['tuning_method'] == tune_method) &
        (results_df['model'] == model_filter) &
        (results_df['training_sample_size'] == sample_size_filter)
    ]
    
    # Apply additional filter for sampling strategy if specified and grouping by training site
    if group_by == 'training_site' and strategy_filter is not None:
        filtered_df = filtered_df[filtered_df['training_sample_strategy'] == strategy_filter]
    
    # Check if we have data after filtering
    if len(filtered_df) == 0:
        error_msg = f"No data found after filtering for model={model_filter}, sample_size={sample_size_filter}, tuning_method={tune_method}"
        if group_by == 'training_site' and strategy_filter:
            error_msg += f", strategy={strategy_filter}"
        
        print(error_msg)
        print(f"Available models: {results_df['model'].unique()}")
        print(f"Available sample sizes: {results_df['training_sample_size'].unique()}")
        print(f"Available tuning methods: {results_df['tuning_method'].unique() if 'tuning_method' in results_df.columns else 'N/A'}")
        if group_by == 'training_site':
            print(f"Available strategies: {results_df['training_sample_strategy'].unique()}")
        
        # Create error plot
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, error_msg, ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        # Save the error plot
        if config:
            plot_dir = create_output_dir(experiment_id, config)
        else:
            plot_dir = f"../../data/experiments/{experiment_id}/plots"
            os.makedirs(plot_dir, exist_ok=True)
        
        # Create a descriptive filename
        filename_parts = ["feature_importance"]
        if group_by == 'strategy':
            filename_parts.append("by_strategy")
        else:
            filename_parts.append("by_site")
            if strategy_filter:
                filename_parts.append(f"strategy_{strategy_filter}")
                
        filename_parts.extend([
            f"model_{model_filter.replace(' ', '_')}",
            f"size_{sample_size_filter}",
            f"{tune_method}",
            "error.png"
        ])
        
        output_path = f"{plot_dir}/{'-'.join(filename_parts)}"
        plt.savefig(output_path, dpi=300)
        plt.show()
        
        return output_path
    
    # Determine what to group by - either sampling strategies or training sites
    if group_by == 'strategy':
        group_values = sorted(filtered_df['training_sample_strategy'].unique(), reverse=True)
        group_column = 'training_sample_strategy'
        group_title = 'Sampling Strategy'
    else:  # group_by == 'training_site'
        group_values = sorted(filtered_df['training_sites'].unique())
        group_column = 'training_sites'
        group_title = 'Training Sites'
    
    # Create a grid of subplots (one per strategy)
    n_groups = len(group_values)
    n_cols = min(1, n_groups)  # Max 2 columns
    n_rows = int(np.ceil(n_groups / n_cols))
    
    # Create figure
    fig = plt.figure(figsize=(16, n_rows * 8))
    
    # Set transparent figure background
    fig.patch.set_alpha(0)
    
    # Set font sizes for all texts
    plt.rcParams.update({
        'font.size': tick_fontsize,
        'axes.titlesize': subtitle_fontsize,
        'axes.labelsize': axis_label_fontsize,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize,
        'legend.fontsize': legend_fontsize,
    })
    
    # Feature importance column names
    importance_columns = [
        'feature_importance_mdi',
        'feature_importance_mda',
        'feature_importance_shap_importance'
    ]
    
    # Method display names for legend
    method_names = {
        'feature_importance_mdi': 'Mean Decrease in Impurity',
        'feature_importance_mda': 'Mean Decrease in Accuracy',
        'feature_importance_shap_importance': 'SHAP Importance'
    }
    
    # Create colorblind-friendly palette
    palette = sns.color_palette("colorblind", len(importance_columns))
    
    # Variables to store legend handles and labels
    all_handles = []
    all_labels = []
    
    # Variables to track global y-axis limits
    global_ymin = float('inf')
    global_ymax = float('-inf')

    # First pass to determine global y-axis limits
    for i, group_value in enumerate(group_values):
        group_data = filtered_df[filtered_df[group_column] == group_value]
        
        for method_col in importance_columns:
            if method_col not in group_data.columns:
                continue
                
            for _, row in group_data.iterrows():
                importance_dict = parse_feature_importance(row[method_col])
                if not importance_dict:
                    continue
                    
                for feature, value in importance_dict.items():
                    global_ymin = min(global_ymin, float(value))
                    global_ymax = max(global_ymax, float(value))

    # Add padding to y-axis limits
    y_range = global_ymax - global_ymin
    global_ymin -= y_range * 0.05
    global_ymax += y_range * 0.05

    # Process each group (strategy or training site) and create plots
    for i, group_value in enumerate(group_values):
        # Filter data for this group
        group_data = filtered_df[filtered_df[group_column] == group_value]
        
        # Create subplot
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # Set transparent subplot background
        ax.set_facecolor('none')
        
        # Process feature importance data
        all_features_data = []
        
        # Process each feature importance method
        for method_col in importance_columns:
            # Skip if column doesn't exist
            if method_col not in group_data.columns:
                continue
                
            # Process each row's feature importance
            for _, row in group_data.iterrows():
                # Parse feature importance
                importance_dict = parse_feature_importance(row[method_col])
                
                if not importance_dict:
                    continue
                
                # Add each feature as a row in our data
                for feature, value in importance_dict.items():
                    all_features_data.append({
                        'Feature': feature,
                        'Importance': float(value),
                        'Method': method_col
                    })
        
        # Convert to DataFrame
        if not all_features_data:
            ax.text(0.5, 0.5, f"No feature importance data for {group_value}",
                  ha='center', va='center')
            continue
            
        features_df = pd.DataFrame(all_features_data)
        
        # Calculate average importance per feature across all methods
        feature_avg_importance = features_df.groupby('Feature')['Importance'].mean().reset_index()
        
        # Determine top features based on average importance
        top_features = feature_avg_importance.sort_values('Importance', ascending=False)
        
        # Handle features below min_importance threshold
        other_features = top_features[top_features['Importance'] < min_importance]['Feature'].tolist()
        
        if other_features:
            # Replace minor features with "Other" category
            features_df.loc[features_df['Feature'].isin(other_features), 'Feature'] = 'Other'
            
            # Recalculate sums for the new "Other" category
            features_df = features_df.groupby(['Feature', 'Method'], as_index=False)['Importance'].mean()
        
        # Get final feature list in order of importance
        feature_order = features_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False).index.tolist()
        
        # Calculate mean importance by feature and method for bar plot
        mean_importance = features_df.groupby(['Feature', 'Method'])['Importance'].mean().reset_index()
        
        # Calculate error (standard deviation) for error bars if there are multiple values per feature/method
        error_data = features_df.groupby(['Feature', 'Method'])['Importance'].std().reset_index()
        error_data.rename(columns={'Importance': 'Error'}, inplace=True)
        # Replace NaN with 0 for features with only one measurement
        error_data['Error'] = error_data['Error'].fillna(0)
        
        # Merge mean and error data
        plot_data = pd.merge(mean_importance, error_data, on=['Feature', 'Method'], how='left')
        
        # Order features by importance
        plot_data['Feature'] = pd.Categorical(plot_data['Feature'], categories=FEATURE_ORDER, ordered=True)
        plot_data = plot_data.sort_values(['Feature', 'Method'])
        
        # Create bar plot with manual feature order
        g = sns.barplot(
            data=plot_data,
            x='Feature',
            y='Importance',
            hue='Method',
            order=FEATURE_ORDER,  # Use manual feature order
            hue_order=importance_columns,
            palette=palette,
            ax=ax,
            errorbar='sd',
            linewidth=bar_linewidth
        )
        
        # Update x-axis tick labels with formatted feature names
        # Get the current tick positions
        tick_positions = range(len(FEATURE_ORDER))
        # Set the tick positions explicitly first
        ax.set_xticks(tick_positions)
        # Then set the tick labels with formatted feature names
        formatted_labels = [format_feature_name(feature) for feature in FEATURE_ORDER]
        ax.set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=tick_fontsize)
        
        # Add error bars if show_points is True (instead of stripplot)
        if show_points:
            # Add error bars manually
            for idx, feature in enumerate(FEATURE_ORDER):
                feature_data = plot_data[plot_data['Feature'] == feature]
                
                for method_idx, method in enumerate([m for m in importance_columns if m in group_data.columns]):
                    method_data = feature_data[feature_data['Method'] == method]
                    if not method_data.empty:
                        # Calculate bar center position based on feature index and method index
                        # Each feature position has multiple methods side by side
                        num_methods = len([m for m in importance_columns if m in group_data.columns])
                        bar_width = 0.8 / num_methods  # Approximation of seaborn's bar width
                        
                        # Calculate center position of the specific bar
                        x_pos = idx + (method_idx - (num_methods-1)/2) * bar_width
                        height = method_data['Importance'].values[0]
                        err = method_data['Error'].values[0]
                        
                        if err > 0:
                            # Draw error bar
                            ax.errorbar(x=x_pos, y=height, yerr=err, 
                                      fmt='none', color='black', capsize=3, capthick=1, 
                                      linewidth=1, alpha=0.7)
        
        # Store handles and labels for common legend (only from first subplot)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            num_methods = len([col for col in importance_columns if col in group_data.columns])
            all_handles = handles[:num_methods]
            all_labels = [method_names.get(col, col) for col in importance_columns if col in group_data.columns]
        
        # Remove individual subplot legend
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        
        # Set title - make it bold
        ax.set_title(f" {group_value}", fontsize=subtitle_fontsize, fontweight='bold')
        
        # Set labels
        ax.set_xlabel('', fontsize=axis_label_fontsize)  # Remove 'Feature' label
        ax.set_ylabel('Importance', fontsize=axis_label_fontsize)
        
        # Set tick font size
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=tick_fontsize)
        
        # Set subplot border width
        for spine in ax.spines.values():
            spine.set_linewidth(subplot_borderwidth)
        
        # Add gridlines with specified width
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=gridline_width)
        
        # Set consistent y-axis limits for all subplots
        ax.set_ylim(global_ymin, global_ymax)
    
    # Add overall title
    title = "Feature Importance"
    if group_by == 'strategy':
        title += f" by Sampling Strategy & Importance Method accross Iterations and LOGO-Folds\n(Filtered by: Model: {model_filter}, Sample Size: {sample_size_filter}, Tuning Method: {tune_method})"
    elif group_by == 'training_site':
        title += f" by Training Site & Importance Method accross Iterations \n(Filtered by: Model: {model_filter}, Sample Size: {sample_size_filter}, Strategy: {strategy_filter}, Tuning Method: {tune_method})"
    else:
        title += f" by Training Site & Importance Method accross Iterations\n(Filtered by: Model: {model_filter}, Sample Size: {sample_size_filter}, Tuning Method: {tune_method})"

    
    plt.suptitle(title, fontsize=title_fontsize, y=0.98, fontweight='bold')
    
    # Adjust layout to make room for the legend at the bottom
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Add a single legend at the bottom
    if all_handles and all_labels:
        leg = fig.legend(
            all_handles, 
            all_labels, 
            loc='lower center', 
            bbox_to_anchor=(0.5, 0.01), 
            ncol=min(len(all_labels), 3),
            title="Importance Method",
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize + 1
        )
        # Remove legend frame and make background transparent
        leg.get_frame().set_linewidth(0)
        leg.get_frame().set_facecolor('none')
    
    # Create summary table data with restructured format
    summary_data = []
    
    # Second pass to create summary data
    for group_value in group_values:
        group_data = filtered_df[filtered_df[group_column] == group_value]
        
        for method_col in importance_columns:
            if method_col not in group_data.columns:
                continue
            
            # Create row identifier combining group and method
            method_display = {
                'feature_importance_mdi': 'MDI',
                'feature_importance_mda': 'MDA',
                'feature_importance_shap_importance': 'SHAP'
            }.get(method_col, method_col)
            
            row_name = f"{group_value} ({method_display})"
            
            # Initialize row data with zeros for all features in manual order
            row_data = {feature: 0.0 for feature in FEATURE_ORDER}
            feature_counts = {feature: 0 for feature in FEATURE_ORDER}
            
            # Collect all importance values
            for _, row in group_data.iterrows():
                importance_dict = parse_feature_importance(row[method_col])
                if importance_dict:
                    for feature, value in importance_dict.items():
                        if feature in row_data:  # Only include features in our manual order
                            row_data[feature] += float(value)
                            feature_counts[feature] += 1
            
            # Calculate means
            for feature in row_data:
                if feature_counts[feature] > 0:
                    row_data[feature] /= feature_counts[feature]
            
            # Add row name and data to summary
            row_data['Row'] = row_name
            summary_data.append(row_data)
    
    # Create DataFrame with Row as index and manual column order
    summary_df = pd.DataFrame(summary_data)
    summary_df.set_index('Row', inplace=True)
    # Reorder columns according to manual order
    summary_df = summary_df[FEATURE_ORDER]
    
    # Calculate dynamic figure size
    n_rows = len(summary_df.index)
    base_height = 1  # minimum height including space for title
    height_per_row = 0.7  # additional height per row

    
    # Calculate title height based on number of lines in title
    title = f"Feature Importance Summary by {group_title}\n"
    title += f"Model: {model_filter}, Sample Size: {sample_size_filter}"
    if strategy_filter:
        title += f", Strategy: {strategy_filter}"
    title_lines = len(title.split('\n'))
    title_height = title_lines * 1 # 2 units per title line
    
    # Calculate total figure height
    fig_height = max(base_height, (n_rows) * height_per_row + title_height)
    
    # Create table figure with dynamic height
    fig_table = plt.figure(figsize=(16, fig_height))
    
    # Adjust subplot position to leave room for title
    # bottom, top: leave more space at top for title
    plt.subplots_adjust(top=0.9, bottom=0.1)
    
    ax_table = fig_table.add_subplot(111)
    ax_table.axis('off')
    
    # Create table title with padding - use same title as boxplot figure
    title = "Feature Importance"
    if group_by == 'strategy':
        title += f" by Sampling Strategy & Importance Method accross Iterations and LOGO-Folds\n(Filtered by: Model: {model_filter}, Sample Size: {sample_size_filter}, Tuning Method: {tune_method})"
    elif group_by == 'training_site':
        title += f" by Training Site & Importance Method accross Iterations \n(Filtered by: Model: {model_filter}, Sample Size: {sample_size_filter}, Strategy: {strategy_filter}, Tuning Method: {tune_method})"
    else:
        title += f" by Training Site & Importance Method accross Iterations\n(Filtered by: Model: {model_filter}, Sample Size: {sample_size_filter}, Tuning Method: {tune_method})"
    
    ax_table.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=10)
    
    # Function to wrap text at underscores with width limit
    def wrap_text(text, width=5):
        """
        Wrap text at underscores when reaching width limit.
        Prioritizes breaking at underscores, then falls back to character width.
        Also ensures that for land cover features, "(LC)" is on a separate line.
        """
        # Special handling for land cover features that already have a newline with (LC)
        if '\n(LC)' in text:
            land_cover_type = text.replace('\n(LC)', '')
            
            # If the land cover type is short, just return as is
            if len(land_cover_type) <= width:
                return text
                
            # Otherwise, wrap the land cover type and keep (LC) on a separate line
            wrapped_land_cover = wrap_text(land_cover_type, width)
            return f"{wrapped_land_cover}\n(LC)"
        
        # Standard wrapping for other text
        if '_' not in text or len(text) <= width:
            return text
            
        parts = []
        current_part = ""
        
        # Split by underscore first
        segments = text.split('_')
        
        for i, segment in enumerate(segments):
            # If this is not the first segment and adding it would exceed width
            if current_part and (len(current_part) + len(segment) + 1 > width):
                parts.append(current_part)
                current_part = segment
            else:
                # Add underscore if not the first segment
                if current_part:
                    current_part += '_'
                current_part += segment
        
        # Add the last part if there is one
        if current_part:
            parts.append(current_part)
            
        return '\n'.join(parts)
    
    # Format cell values to 3 decimal places and wrap column headers
    cell_text = [[f"{val:.3f}" if isinstance(val, (int, float)) else val 
                  for val in row] for row in summary_df.values]
    
    # Format and wrap column headers
    formatted_headers = [format_feature_name(col) for col in summary_df.columns]
    # Apply wrap_text to already formatted headers for multi-line display
    wrapped_headers = [wrap_text(header) for header in formatted_headers]
    
    # Create and style the table
    table = ax_table.table(
        cellText=cell_text,
        rowLabels=summary_df.index,
        colLabels=wrapped_headers,
        loc='center',
        cellLoc='center'
    )
    
    # Apply table styling
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_STYLE['fontsize'])
    
    # Calculate the height multiplier based on the maximum number of lines in headers
    max_header_lines = max(len(header.split('\n')) for header in wrapped_headers)
    height_scale = max(3, max_header_lines * 1.2)  # At least 3, or more if headers are very long
    
    # Scale table with adjusted height
    table.scale(1, height_scale)
    
    # Create colormap
    cmap = plt.colormaps[TABLE_COLORMAP['cmap_name']]
    if TABLE_COLORMAP['invert_cmap']:
        cmap = cmap.reversed()
    
    # Apply cell styling with row-based coloring
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(TABLE_STYLE['cell_border_color'])
        cell.set_linewidth(TABLE_STYLE['cell_border_width'])
        
        if row == 0:  # Header row
            cell.set_facecolor(TABLE_STYLE['header_bg_color'])
            cell.set_text_props(weight='bold', color='black', fontsize=TABLE_STYLE['header_fontsize'])
        elif col == -1:  # Row labels
            cell.set_facecolor(TABLE_STYLE['row_header_bg_color'])
            cell.set_text_props(weight='bold', color='black')
        else:  # Data cells
            # Get numeric values for this row
            row_values = summary_df.iloc[row-1].values.astype(float)
            row_min = np.min(row_values)
            row_max = np.max(row_values)
            
            # Add buffer to min/max range
            value_range = row_max - row_min
            if value_range > 0:
                buffer = value_range * TABLE_COLORMAP['value_range_buffer']
                row_min -= buffer
                row_max += buffer
            
            # Get current cell value
            try:
                cell_value = float(cell_text[row-1][col])
                
                # Normalize value within this row
                if row_max > row_min:
                    norm_value = (cell_value - row_min) / (row_max - row_min)
                else:
                    norm_value = 0.5
                
                # Apply color with alpha
                cell_color = cmap(norm_value)
                # Convert to RGBA with configured alpha
                cell_color = (*cell_color[:3], TABLE_COLORMAP['alpha'])
                cell.set_facecolor(cell_color)
                
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
        
                # Get text color based on cell background color
                text_color = get_text_color(cell_color)
                cell.get_text().set_color(text_color)
                cell.get_text().set_weight('bold')
            except ValueError:
                # Handle non-numeric cells
                cell.get_text().set_color('black')
                cell.get_text().set_weight('bold')
    
    # Save figures
    if config:
        plot_dir = create_output_dir(experiment_id, config)
    else:
        plot_dir = f"../../data/experiments/{experiment_id}/plots"
        os.makedirs(plot_dir, exist_ok=True)
    
    # Create descriptive filenames
    filename_parts = ["feature_importance"]
    if group_by == 'strategy':
        filename_parts.append("by_strategy")
    else:
        filename_parts.append("by_site")
        if strategy_filter:
            filename_parts.append(f"strategy_{strategy_filter.replace(' ', '_')}")
            
    filename_parts.extend([
        f"model_{model_filter.replace(' ', '_')}",
        f"size_{sample_size_filter}",
        f"{tune_method}"
    ])
    
    base_output_path = f"{plot_dir}/{'-'.join(filename_parts)}"
    
    # Save plots
    fig.savefig(f"{base_output_path}.png", dpi=300, bbox_inches='tight')
    fig_table.savefig(f"{base_output_path}_table.png", dpi=300, bbox_inches='tight', facecolor='none')
    
    plt.show(fig)
    plt.show(fig_table)
    
    return f"{base_output_path}.png"

# %%
# Define shared plotting parameters
# Use default plot parameters from configuration
plot_params = DEFAULT_PLOT_PARAMS

# %%
# Only run this if the script is executed directly (not imported)
if __name__ == "__main__":
    # Load data when running as a standalone script
    results_df, config = load_experiment_data()
    
    if results_df is not None and config is not None:
        # Example 1: Visualizing feature importance by sampling strategy
        plot_feature_importance_by_strategy(
            results_df=results_df,
            model_filter="Random Forest",
            sample_size_filter=1000,
            group_by="strategy",  # Group by sampling strategy
            tune_method="no_tuning",
            experiment_id=config["experiment_id"],
            config=config,
            min_importance=0.01,
            show_points=True,
            **plot_params
        )
    else:
        print("Could not load experiment data. Please make sure experiment_config.json exists and contains valid paths.")

# %%
# Only run this if the script is executed directly (not imported)
if __name__ == "__main__":
    # Load data when running as a standalone script
    results_df, config = load_experiment_data()
    
    if results_df is not None and config is not None:
        # Example 2: Visualizing feature importance by training site for a specific strategy
        plot_feature_importance_by_strategy(
            results_df=results_df,
            model_filter="Balanced Random Forest",
            sample_size_filter=500,
            strategy_filter="balanced_random",  # Specify strategy when grouping by site
            group_by="training_site",  # Group by training site
            tune_method="no_tuning",
            experiment_id=config["experiment_id"],
            config=config,
            min_importance=0.01,
            show_points=True,
            **plot_params
        )
    else:
        print("Could not load experiment data. Please make sure experiment_config.json exists and contains valid paths.")

# %%
