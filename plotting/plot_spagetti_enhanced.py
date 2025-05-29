# I have a big csv file results.csv with the following columns:
#iteration	configuration_name	model	training_sample_size	training_sample_strategy	tuning_method	best_params	testing_site	training_sites	model_path	training_time	prediction_times	balanced_OA_same_site	balanced_F1_same_site	balanced_Confusion_Matrix_same_site	proportional_OA_same_site	proportional_F1_same_site	proportional_Confusion_Matrix_same_site	balanced_OA_cross_site	balanced_F1_cross_site	balanced_Confusion_Matrix_cross_site	proportional_OA_cross_site	proportional_F1_cross_site	proportional_Confusion_Matrix_cross_site
# 	feature_importance_mdi	feature_importance_mda	feature_importance_shap_importance

# I want to create a cross_site_score combining balanced_OA_cross_site, balanced_F1_cross_site and proportional_OA_cross_site, proportional_F1_cross_site into a single score.
# I also want to create a same_site_score combining balanced_OA_same_site, balanced_F1_same_site and proportional_OA_same_site, proportional_F1_same_site into a single score.
# the scorer should weight all metrics equally and average accross the iterations, and if not filtered also accross the testing_site 
# I have multiple experiment variables: model(Random Forest, Balanced Random Forest, Weihgted Random Forest, XGBoost), training_sample_size(100, 500, 1000), training_sample_strategy(simple_random, simple_systematic, ...), tuning_method(no_tuning, BayesSearchCV)

# I want to plot a parallel coordinate plot of either the cross_site_score or the same site score 

# the color of the line should be determined by the score, the labels should be the experiment variables 

# I want to use cividis color map and I want to highlight the highest score in red 

# I want to be able to filter for a column eg. testing_site = 'valencia' 

# print the combination with the highest score

# create functions within a interactive python script 

# %%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from typing import Dict, List, Optional, Any
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.visualisation.vis_utils import load_experiment_data
from src.visualisation.vis_config import DEFAULT_PLOT_PARAMS, LINE_THICKNESS, LINE_THICKNESS_GRID

# %%
def create_combined_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined scores from individual metrics.
    
    Args:
        df: DataFrame containing experiment results
        
    Returns:
        DataFrame with added cross_site_score and same_site_score columns
    """
    df = df.copy()
    
    # Cross-site score: average of 4 metrics
    cross_site_metrics = [
        'balanced_OA_cross_site',
        'balanced_F1_cross_site', 
        'proportional_OA_cross_site',
        'proportional_F1_cross_site'
    ]
    
    # Same-site score: average of 4 metrics  
    same_site_metrics = [
        'balanced_OA_same_site',
        'balanced_F1_same_site',
        'proportional_OA_same_site', 
        'proportional_F1_same_site'
    ]
    
    # Calculate combined scores (equal weights)
    df['cross_site_score'] = df[cross_site_metrics].mean(axis=1)
    df['same_site_score'] = df[same_site_metrics].mean(axis=1)
    
    return df

# %%
def aggregate_results(df: pd.DataFrame, group_by_testing_site: bool = True) -> pd.DataFrame:
    """
    Aggregate results by experiment variables, averaging across iterations and optionally testing sites.
    
    Args:
        df: DataFrame with experiment results
        group_by_testing_site: If True, keep testing_site separate. If False, average across testing sites too.
        
    Returns:
        Aggregated DataFrame
    """
    print(f"\n=== AGGREGATION DEBUGGING ===")
    print(f"Input DataFrame shape: {df.shape}")
    
    # Split training strategy into two variables
    df = split_training_strategy(df)
    
    # Core experiment variables - now using the split variables
    group_vars = ['model', 'training_sample_size', 'spatial_distribution', 'class_distribution', 'tuning_method']
    
    if group_by_testing_site:
        group_vars.append('testing_site')
    
    print(f"Grouping by variables: {group_vars}")
    
    # Check unique values for each variable
    for var in group_vars:
        if var in df.columns:
            unique_vals = df[var].unique()
            print(f"{var}: {len(unique_vals)} unique values -> {unique_vals}")
        else:
            print(f"WARNING: {var} not in DataFrame columns!")
    
    # Check how many unique combinations exist in the data
    unique_combinations = df[group_vars].drop_duplicates()
    print(f"Unique combinations in data: {len(unique_combinations)}")
    print("First 10 combinations:")
    print(unique_combinations.head(10))
    
    # Aggregate by taking mean of scores across iterations (and testing sites if not grouped)
    agg_df = df.groupby(group_vars).agg({
        'cross_site_score': 'mean',
        'same_site_score': 'mean'
    }).reset_index()
    
    print(f"After aggregation shape: {agg_df.shape}")
    print(f"Aggregated combinations: {len(agg_df)}")
    
    return agg_df

# %%
def plot_parallel_coordinates(df: pd.DataFrame, 
                            score_type: str = 'cross_site_score',
                            filter_dict: Optional[Dict[str, Any]] = None,
                            experiment_id: str = "main_experiment",
                            config: Optional[Dict] = None,
                            plot_raw_data: bool = False,
                            line_spacing: float = 0.002,
                            color_percentile_range: tuple = (5, 95),
                            color_min: Optional[float] = None,
                            color_max: Optional[float] = None,
                            tick_fontsize: int = 10,
                            axis_title_fontsize: int = 14,
                            plot_title_fontsize: int = 16,
                            score_line_width: float = 2.0,
                            axis_line_width: float = 1.5,
                            axis_line_color: str = 'black',
                            figure_size: tuple = (16, 8),
                            connect_to_colorbar: bool = False,
                            colormap: str = 'viridis',
                            text_color: str = 'black',
                            label_outline_width: float = 2.0,
                            output_dir: Optional[str] = None,
                            show_bean_plots: bool = True,
                            bean_plot_height: float = 0.25) -> str:
    """
    Create parallel coordinates plot for experiment results using smooth curves.
    
    Args:
        df: DataFrame with experiment results
        score_type: Either 'cross_site_score' or 'same_site_score'
        filter_dict: Dictionary of column filters e.g., {'testing_site': 'valencia'}
        experiment_id: ID of the experiment
        config: Experiment configuration
        plot_raw_data: If True, plot all individual runs (like example). If False, plot aggregated averages.
        line_spacing: Vertical spacing factor between lines (default: 0.002)
                     - Controls how much lines are vertically separated based on score ranking
                     - Larger values = more separation, smaller values = less separation
                     - Typical range: 0.001 (minimal) to 0.01 (strong separation)
                     - Set to 0.0 to disable spacing (lines may overlap)
        color_percentile_range: Tuple of (low, high) percentiles for color normalization (default: (5, 95))
                               - Focuses colormap on main data distribution for better distinction
                               - (0, 100) = full range (like original), (10, 90) = more focused
                               - Smaller range = better distinction in clustered values
                               - Ignored if color_min and color_max are provided
        color_min: Optional minimum value for color normalization (default: None)
                  - If provided (along with color_max), overrides percentile-based normalization
                  - Useful for making color scales comparable across different plots
        color_max: Optional maximum value for color normalization (default: None)
                  - If provided (along with color_min), overrides percentile-based normalization
                  - Useful for making color scales comparable across different plots
        tick_fontsize: Font size for tick labels (default: 10)
        axis_title_fontsize: Font size for axis titles (default: 14)
        plot_title_fontsize: Font size for main plot title (default: 16)
        score_line_width: Width of the actual score lines (default: 2.0)
        axis_line_width: Width of axis lines, ticks, and colorbar elements (default: 1.5)
        axis_line_color: Color of axis lines, ticks, and colorbar elements (default: 'black')
        figure_size: Tuple of (width, height) for figure size in inches (default: (16, 8))
        connect_to_colorbar: If True, extend lines from rightmost axis to colorbar at score position (default: False)
        colormap: Name of the colormap to use for coloring lines (default: 'viridis')
        text_color: Color for all text elements in the figure (default: 'black')
                   - Applied to axis titles, tick labels, colorbar labels, and plot title
        label_outline_width: Width of white outline around text labels for better readability (default: 2.0)
                             - Applied to Y-axis tick labels to make them readable over colored backgrounds
        output_dir: Optional output directory for saving the plot (default: None)
                    - If provided, plot will be saved to this directory with auto-generated filename
                    - If None, uses config-based directory or default experiment directory
        show_bean_plots: If True, show bean plots below each axis (default: True)
                        - Shows distribution of scores for each category of each variable
                        - Helps visualize performance variance within each variable
        bean_plot_height: Height of bean plots as fraction of main plot height (default: 0.25)
                         - Controls how much vertical space bean plots take up
                         - 0.25 means bean plots are 25% the height of main plot
        
    Returns:
        Path to saved plot
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.path import Path
    import matplotlib.patches as patches
    
    print(f"\n=== PARALLEL COORDINATES PLOT DEBUGGING ===")
    print(f"Initial DataFrame shape: {df.shape}")
    print(f"Score type: {score_type}")
    print(f"Filter dict: {filter_dict}")
    print(f"Plot raw data: {plot_raw_data}")
    
    # Apply filters if provided
    plot_df = df.copy()
    if filter_dict:
        for col, val in filter_dict.items():
            if col in plot_df.columns:
                print(f"Before filter {col}={val}: {len(plot_df)} rows")
                plot_df = plot_df[plot_df[col] == val]
                print(f"After filter {col}={val}: {len(plot_df)} rows")
    
    if len(plot_df) == 0:
        print("No data remaining after filtering!")
        return None
    
    # Create combined scores
    plot_df = create_combined_scores(plot_df)
    print(f"After creating combined scores: {len(plot_df)} rows")
    
    # Split training strategy into spatial and class distribution
    plot_df = split_training_strategy(plot_df)
    print(f"After splitting training strategy: {len(plot_df)} rows")
    
    # Aggregate results (optional - example doesn't do this)
    if plot_raw_data:
        print(f"Using raw data: {plot_df.shape[0]} individual experiment runs")
        agg_df = plot_df.copy()
    else:
        agg_df = aggregate_results(plot_df, group_by_testing_site='testing_site' not in (filter_dict or {}))
        print(f"Using aggregated data: {agg_df.shape[0]} averaged combinations (reduced from {plot_df.shape[0]} raw runs)")
    
    if len(agg_df) == 0:
        print("No data remaining after aggregation!")
        return None
    
    # Define variables for parallel coordinates in the requested order
    # Spatial Distribution -> Class Distribution -> Training Sample Size -> Model -> Tuning Method
    plot_vars = ['spatial_distribution', 'class_distribution', 'training_sample_size', 'model', 'tuning_method']
    if 'testing_site' in agg_df.columns and 'testing_site' not in (filter_dict or {}):
        plot_vars.append('testing_site')
    
    print(f"Plot variables: {plot_vars}")
    
    # Check each variable in detail
    for var in plot_vars:
        if var in agg_df.columns:
            unique_vals = sorted(agg_df[var].unique())
            print(f"{var}: {len(unique_vals)} unique values -> {unique_vals}")
    
    # Create variable names for display - axis titles with line breaks
    var_names = [format_label_text(var, add_line_breaks=True) for var in plot_vars]
    
    # Prepare data for parallel coordinates
    df_plot = agg_df[plot_vars + [score_type]].copy()
    print(f"Before dropna: {len(df_plot)} rows")
    df_plot = df_plot.dropna().reset_index(drop=True)
    print(f"After dropna: {len(df_plot)} rows")
    
    print(f"\n=== FINAL DATA FOR PLOTTING (NO NORMALIZATION) ===")
    print(f"Original data matrix (ym) sample (first 10 rows):")
    for j in range(min(10, df_plot.shape[0])):
        original_coords = []
        for i in range(df_plot.shape[1] - 1):
            val = df_plot.iloc[j, i]
            if isinstance(val, (int, float)):
                original_coords.append(f"{val:.1f}")
            else:
                original_coords.append(str(val))
        print(f"  Row {j}: [{', '.join(original_coords)}]")
    
    print(f"Data matrix shape: {df_plot.shape} - plotting all {df_plot.shape[0]} lines")
    
    # Convert categorical variables to numeric for plotting
    ym = []
    dics_vars = []
    
    # Define custom ordering for categorical variables
    custom_orders = {
        'spatial_distribution': ['random', 'grts', 'systematic'],
        'training_sample_size': [100, 500, 1000],
        'model': ['Random Forest', 'Weighted Random Forest', 'Balanced Random Forest'],
        'tuning_method': ['no_tuning', 'BayesSearchCV'],
        'class_distribution': ['simple', 'proportional', 'balanced']  # Added for completeness
    }
    
    for v, var in enumerate(plot_vars):
        if df_plot[var].dtype.kind not in ["i", "u", "f"] or var == 'training_sample_size':
            # Categorical variable OR training_sample_size (treat as categorical for positioning)
            unique_vals_in_data = df_plot[var].unique()
            
            # Use custom ordering if available, otherwise fall back to sorted
            if var in custom_orders:
                # Filter custom order to only include values present in data
                if var == 'training_sample_size':
                    # For training_sample_size, compare as numbers
                    unique_vals = [val for val in custom_orders[var] if val in unique_vals_in_data]
                else:
                    # For string variables, compare as strings (case-insensitive)
                    unique_vals_lower = [str(val).lower() for val in unique_vals_in_data]
                    unique_vals = [val for val in custom_orders[var] if val.lower() in unique_vals_lower]
                    # Map back to original case from data
                    val_mapping = {str(val).lower(): val for val in unique_vals_in_data}
                    unique_vals = [val_mapping.get(val.lower(), val) for val in unique_vals]
                
                print(f"Using custom order for {var}: {unique_vals}")
            else:
                # Fall back to alphabetical sorting for variables without custom order
                unique_vals = sorted(unique_vals_in_data)
                print(f"Using default alphabetical order for {var}: {unique_vals}")
            
            dic_var = dict([(val, c) for c, val in enumerate(unique_vals)])
            dics_vars.append(dic_var)
            numeric_values = [dic_var[i] for i in df_plot[var].tolist()]
            ym.append(numeric_values)
            print(f"Treating {var} as categorical: {len(unique_vals)} values -> {unique_vals}")
        else:
            # Numeric variable
            ym.append(df_plot[var].tolist())
            print(f"Treating {var} as numeric: range {df_plot[var].min()}-{df_plot[var].max()}")
    
    ym = np.array(ym).T
    print(f"Converted to numeric matrix shape: {ym.shape}")
    
    # Colors based on score using colormap with either explicit values or percentile-based normalization
    scores = df_plot[score_type].values
    
    if color_min is not None and color_max is not None:
        # Use explicit min/max values for color normalization (overrides percentile range)
        vmin_adjusted = color_min
        vmax_adjusted = color_max
        print(f"Using explicit color range: {vmin_adjusted:.4f} to {vmax_adjusted:.4f}")
    else:
        # Use percentile-based normalization to better distinguish clustered values
        # This focuses the colormap on where most data lies while still including outliers
        score_percentiles = np.percentile(scores, color_percentile_range)  # Use specified percentiles
        vmin_percentile, vmax_percentile = score_percentiles
        
        # Ensure we still include the absolute min/max values by extending the range slightly
        score_range = vmax_percentile - vmin_percentile
        vmin_adjusted = max(scores.min(), vmin_percentile - 0.1 * score_range)  # Include lowest values
        vmax_adjusted = min(scores.max(), vmax_percentile + 0.1 * score_range)  # Include highest values
        print(f"Using percentile-based color range ({color_percentile_range[0]}-{color_percentile_range[1]}%)")
        print(f"Percentile range: {vmin_percentile:.4f} to {vmax_percentile:.4f}")
    
    # Create normalization
    norm = plt.Normalize(vmin=vmin_adjusted, vmax=vmax_adjusted)
    colors = plt.cm.get_cmap(colormap)(norm(scores))
    
    # Use separate line widths for score lines vs axis elements
    line_thickness = score_line_width  # For actual score lines
    
    print(f"Color mapping: {len(scores)} scores -> {len(colors)} colors")
    print(f"Score range: {scores.min():.4f} to {scores.max():.4f}")
    print(f"Colormap normalization range: {vmin_adjusted:.4f} to {vmax_adjusted:.4f}")
    print(f"Colormap: {colormap}")
    print(f"Score line width: {score_line_width}")
    print(f"Axis line width: {axis_line_width}")
    print(f"Axis line color: {axis_line_color}")
    
    # Find highest score for highlighting
    max_score_idx = np.argmax(scores)
    print(f"Highest score: {scores[max_score_idx]:.4f} at index {max_score_idx}")
    
    # Create the plot - use proper size for parallel coordinates with optional bean plots
    if show_bean_plots:
        # Create subplots: main plot on top, bean plots below
        fig = plt.figure(figsize=figure_size)
        gs = fig.add_gridspec(2, 1, height_ratios=[1, bean_plot_height], hspace=0.1)
        host_ax = fig.add_subplot(gs[0])  # Main parallel coordinates plot
        bean_ax = fig.add_subplot(gs[1])  # Bean plots area
    else:
        # Original single plot layout
        fig, host_ax = plt.subplots(figsize=figure_size)
        bean_ax = None
    
    # Adjust subplot to leave more room for rightmost y-axis labels and colorbar
    if show_bean_plots:
        # With gridspec, adjust the layout differently
        gs.update(right=0.75)  # Leave room for colorbar
    else:
        plt.subplots_adjust(right=0.75)  # Increased horizontal spacing for colorbar
    
    # Make the axes
    axes = [host_ax] + [host_ax.twinx() for i in range(ym.shape[1] - 1)]
    
    # Track which categorical variable we're on
    dic_idx = 0
    
    print(f"\n=== SETTING UP AXES ===")
    for i, ax in enumerate(axes):
        ax.set_ylim(bottom=ym[:, i].min(), top=ym[:, i].max())
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.ticklabel_format(style='plain')
        
        # Apply line thickness and color from parameters to spines
        for spine in ax.spines.values():
            spine.set_linewidth(axis_line_width)
            spine.set_color(axis_line_color)
        
        # Apply line thickness and color to ticks
        ax.tick_params(width=axis_line_width, length=6, color=axis_line_color)
        
        if ax != host_ax:
            ax.spines.left.set_visible(False)
            ax.yaxis.set_ticks_position("right")
            # Scale the position to fit within the adjusted subplot area
            ax.spines.right.set_position(("axes", i / (ym.shape[1] - 1)))
        
        # Set up ticks for categorical variables
        if i < len(plot_vars) and (df_plot.iloc[:, i].dtype.kind not in ["i", "u", "f"] or plot_vars[i] == 'training_sample_size'):
            # Categorical variable OR training_sample_size - use dic_idx to access dics_vars
            if dic_idx < len(dics_vars):
                dic_var_i = dics_vars[dic_idx]
                print(f"Axis {i} ({plot_vars[i]}): Setting categorical ticks for {len(dic_var_i)} categories")
                
                # Set ticks and labels
                tick_positions = list(range(len(dic_var_i)))
                tick_labels = [format_label_text(str(key_val), add_line_breaks=True) for key_val in dic_var_i.keys()]
                
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels, fontsize=tick_fontsize, fontweight='bold')
                
                # Add white outlines to y-axis labels for better readability
                for label in ax.get_yticklabels():
                    label.set_path_effects([path_effects.withStroke(linewidth=label_outline_width, foreground='white')])
                    label.set_color(text_color)
                
                # For axis 0 (host axis), ensure all ticks are shown
                if i == 0:
                    from matplotlib.ticker import FixedLocator, FixedFormatter
                    ax.yaxis.set_major_locator(FixedLocator(tick_positions))
                    ax.yaxis.set_major_formatter(FixedFormatter(tick_labels))
                    print(f"  Host axis: Forced {len(tick_positions)} ticks with FixedLocator")
                
                dic_idx += 1  # Move to next categorical variable
            else:
                print(f"ERROR: dic_idx {dic_idx} >= len(dics_vars) {len(dics_vars)}")
        else:
            # For numeric variables, use default ticks
            print(f"Axis {i} ({plot_vars[i]}): Setting numeric ticks")
            ax.tick_params(labelsize=tick_fontsize)
            # Make numeric tick labels bold and add white outlines
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_path_effects([path_effects.withStroke(linewidth=label_outline_width, foreground='white')])
                label.set_color(text_color)
    
    # Set up x-axis
    host_ax.set_xlim(left=0, right=ym.shape[1] - 1)
    host_ax.set_xticks(range(ym.shape[1]))
    host_ax.set_xticklabels(var_names, fontsize=axis_title_fontsize, fontweight='bold', color=text_color)
    host_ax.tick_params(axis="x", which="major", pad=12, width=axis_line_width, length=6, color=axis_line_color)
    
    # Make the curves
    host_ax.spines.right.set_visible(False)
    host_ax.xaxis.tick_bottom()  # Changed from tick_top() to tick_bottom()
    
    print(f"\n=== PLOTTING {ym.shape[0]} CURVES ===")
    
    # Create normalized coordinates for proper parallel coordinate plotting
    # Each axis needs to be normalized to a common scale (0-1) for plotting
    ym_normalized = np.zeros_like(ym, dtype=float)
    
    print(f"Normalizing coordinates for each axis:")
    for i in range(ym.shape[1]):
        axis_min = ym[:, i].min()
        axis_max = ym[:, i].max()
        axis_range = axis_max - axis_min
        
        if axis_range == 0:  # Handle case where all values are the same
            ym_normalized[:, i] = 0.5  # Put in middle
        else:
            ym_normalized[:, i] = (ym[:, i] - axis_min) / axis_range
            
        print(f"  Axis {i} ({plot_vars[i]}): range {axis_min}-{axis_max} -> normalized to 0-1")
        print(f"    Sample values: {ym[:3, i]} -> {ym_normalized[:3, i].round(3)}")
    
    # Sort data by score and add fixed spacing to prevent line overlap
    score_order = np.argsort(scores)  # Sort indices by score (lowest to highest)
    
    # SPACING ALGORITHM EXPLANATION:
    # ===============================
    # We apply systematic vertical spacing based on score ranking to prevent line overlap
    # and create visual layering where higher-scoring lines appear "above" lower-scoring ones.
    #
    # Spacing calculation: offset = (rank - center_rank) * line_spacing
    # - rank: position in sorted order (0 = lowest score, n-1 = highest score)  
    # - center_rank: middle position (n/2), gets zero offset
    # - line_spacing: user-controlled spacing factor
    #
    # Examples with line_spacing=0.002 and 162 lines:
    # - Lowest score (rank 0):  offset = (0 - 81) * 0.002 = -0.162 (pushed down)
    # - Middle score (rank 81): offset = (81 - 81) * 0.002 = 0.000 (no change)
    # - Highest score (rank 161): offset = (161 - 81) * 0.002 = +0.160 (pushed up)
    
    # Create a copy for sorted plotting
    ym_plot = ym_normalized.copy()
    scores_plot = scores.copy()
    colors_plot = colors.copy()
    
    # Sort everything by score order (lowest to highest)
    ym_plot = ym_plot[score_order]
    scores_plot = scores_plot[score_order] 
    colors_plot = colors_plot[score_order]
    
    # Apply fixed spacing - higher scores get more positive offset (appear higher visually)
    if line_spacing > 0.0:  # Only apply spacing if parameter is positive
        center_rank = len(score_order) / 2  # Middle position gets zero offset
        
        for rank, original_idx in enumerate(score_order):
            # Calculate vertical offset based on rank relative to center
            # Negative for below-average scores, positive for above-average scores
            offset = (rank - center_rank) * line_spacing
            
            # UNIFORM SPACING: Apply offset uniformly to all lines, allowing outward spacing
            # This creates even spacing across all lines regardless of boundary position
            for coord_idx in range(ym_plot.shape[1]):
                original_coord = ym_plot[rank, coord_idx]
                
                # Calculate the new coordinate with offset (no clipping - allow outward spacing)
                new_coord = original_coord + offset
                ym_plot[rank, coord_idx] = new_coord
        
        # Calculate the expanded plot range needed to show all spaced lines
        y_min = ym_plot.min()
        y_max = ym_plot.max()
        
        # Add a small buffer to ensure all lines are visible
        y_range = y_max - y_min
        y_buffer = y_range * 0.02  # 2% buffer
        plot_y_min = y_min - y_buffer
        plot_y_max = y_max + y_buffer
        
        print(f"Applied uniform spacing (strength: {line_spacing:.6f}) based on score ranking")
        print(f"Spacing range: {-center_rank * line_spacing:.3f} to {+center_rank * line_spacing:.3f}")
        print(f"Expanded plot range: {plot_y_min:.3f} to {plot_y_max:.3f} (was 0.0 to 1.0)")
        print(f"Outward spacing: boundary lines can extend beyond original 0-1 range")
    else:
        print("Line spacing disabled (line_spacing = 0.0) - lines may overlap")
        # Set default range when no spacing
        plot_y_min = 0.0
        plot_y_max = 1.0
    
    # Update max_score_idx to reflect new sorted order
    max_score_idx_sorted = len(score_order) - 1  # Highest score is now last
    
    print(f"Lines sorted by score: lowest to highest")
    print(f"Best line is now at index {max_score_idx_sorted} (was {max_score_idx})")
    
    # Plot all curves using normalized coordinates with smooth curves
    lines_plotted = 0
    
    for j in range(ym_plot.shape[0]):
        if j != max_score_idx_sorted:  # Plot non-max curves first
            # Create smooth curve across all axes using normalized coordinates
            x_coords = list(range(ym_plot.shape[1]))
            y_coords = ym_plot[j, :].tolist()
            
            # Create smooth Bézier curve path
            if len(x_coords) >= 2:
                # Create control points for smooth curves
                path_coords = []
                path_codes = []
                
                # Start point
                path_coords.append((x_coords[0], y_coords[0]))
                path_codes.append(Path.MOVETO)
                
                # Add smooth curves between consecutive points
                for i in range(1, len(x_coords)):
                    x_start, y_start = x_coords[i-1], y_coords[i-1] 
                    x_end, y_end = x_coords[i], y_coords[i]
                    
                    # Control points for smooth curve (1/3 and 2/3 along x, but adjusted y)
                    control1_x = x_start + (x_end - x_start) * 0.33
                    control1_y = y_start
                    control2_x = x_start + (x_end - x_start) * 0.67  
                    control2_y = y_end
                    
                    # Add Bézier curve segment
                    path_coords.extend([(control1_x, control1_y), (control2_x, control2_y), (x_end, y_end)])
                    path_codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
                
                # Create and add the path
                path = Path(path_coords, path_codes)
                patch = patches.PathPatch(path, facecolor='none', edgecolor=colors_plot[j], 
                                        linewidth=line_thickness, alpha=0.7)
                host_ax.add_patch(patch)
            
            lines_plotted += 1
            
            if j < 3:  # Debug first few lines
                coords_str = ", ".join([f"{ym[score_order[j], k]:.1f}" for k in range(ym.shape[1])])
                norm_coords_str = ", ".join([f"{ym_plot[j, k]:.3f}" for k in range(ym_plot.shape[1])])
                print(f"Line {j}: actual=[{coords_str}] -> normalized+spaced=[{norm_coords_str}] (score: {scores_plot[j]:.4f})")
    
    # Plot the highest score curve on top (highlighted in red) with smooth curve
    j = max_score_idx_sorted
    x_coords = list(range(ym_plot.shape[1]))
    y_coords = ym_plot[j, :].tolist()
    
    # Calculate enhanced thickness for best line
    best_line_thickness = max(line_thickness + 1.0, 3.0)  # At least 3.0, or score-based + 1
    
    if len(x_coords) >= 2:
        # Create smooth curve for best line
        path_coords = []
        path_codes = []
        
        # Start point
        path_coords.append((x_coords[0], y_coords[0]))
        path_codes.append(Path.MOVETO)
        
        # Add smooth curves between consecutive points
        for i in range(1, len(x_coords)):
            x_start, y_start = x_coords[i-1], y_coords[i-1]
            x_end, y_end = x_coords[i], y_coords[i]
            
            # Control points for smooth curve
            control1_x = x_start + (x_end - x_start) * 0.33
            control1_y = y_start
            control2_x = x_start + (x_end - x_start) * 0.67
            control2_y = y_end
            
            # Add Bézier curve segment
            path_coords.extend([(control1_x, control1_y), (control2_x, control2_y), (x_end, y_end)])
            path_codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
        
        # Create and add the best path in red with enhanced thickness
        path = Path(path_coords, path_codes)
        patch = patches.PathPatch(path, facecolor='none', edgecolor='red', 
                                linewidth=best_line_thickness, alpha=0.9)
        host_ax.add_patch(patch)
    
    lines_plotted += 1
    coords_str = ", ".join([f"{ym[score_order[j], k]:.1f}" for k in range(ym.shape[1])])
    norm_coords_str = ", ".join([f"{ym_plot[j, k]:.3f}" for k in range(ym_plot.shape[1])])
    print(f"Best line {j}: actual=[{coords_str}] -> normalized+spaced=[{norm_coords_str}] (thickness: {best_line_thickness:.2f}, score: {scores_plot[j]:.4f})")
    
    print(f"Total smooth curves plotted: {lines_plotted}")
    
    # Update the main axis to use expanded range to accommodate spaced lines
    host_ax.set_ylim(plot_y_min, plot_y_max)
    
    # Update all axes to use expanded range and position their labels correctly
    for i, ax in enumerate(axes):
        ax.set_ylim(plot_y_min, plot_y_max)
        
        # For categorical axes, we need to place the labels at the correct normalized positions
        # NOTE: Labels stay at original 0-1 positions even though plot range is expanded
        if i < len(plot_vars) and (df_plot.iloc[:, i].dtype.kind not in ["i", "u", "f"] or plot_vars[i] == 'training_sample_size'):
            # Find which categorical variable this is
            cat_var_idx = 0
            for var_idx in range(i):
                if df_plot.iloc[:, var_idx].dtype.kind not in ["i", "u", "f"] or plot_vars[var_idx] == 'training_sample_size':
                    cat_var_idx += 1
            
            if cat_var_idx < len(dics_vars):
                dic_var = dics_vars[cat_var_idx]
                category_names = [format_label_text(str(name), add_line_breaks=True) for name in dic_var.keys()]
                n_categories = len(category_names)
                
                print(f"Axis {i} ({plot_vars[i]}): {n_categories} categories = {category_names}")
                
                if n_categories > 1:
                    # Calculate normalized positions for each category (keep at original 0-1 scale)
                    # Category 0 -> position 0, Category n-1 -> position 1
                    label_positions = [j / (n_categories - 1) for j in range(n_categories)]
                    print(f"  Label positions: {label_positions} (kept at original 0-1 scale)")
                    
                    # Clear existing ticks and set new ones
                    ax.set_yticks(label_positions)
                    ax.set_yticklabels(category_names, fontsize=tick_fontsize, fontweight='bold')
                    
                    # Add white outlines to y-axis labels for better readability
                    for label in ax.get_yticklabels():
                        label.set_path_effects([path_effects.withStroke(linewidth=label_outline_width, foreground='white')])
                        label.set_color(text_color)
                    
                    # For axis 0 (host axis), ensure all ticks are shown with FixedLocator
                    if i == 0:
                        from matplotlib.ticker import FixedLocator, FixedFormatter
                        ax.yaxis.set_major_locator(FixedLocator(label_positions))
                        ax.yaxis.set_major_formatter(FixedFormatter(category_names))
                        print(f"  Host axis final: Forced {len(label_positions)} ticks with FixedLocator at positions {label_positions}")
                else:
                    # Single category - place in middle of original 0-1 range
                    ax.set_yticks([0.5])
                    ax.set_yticklabels(category_names, fontsize=tick_fontsize, fontweight='bold')
                    # Add white outlines to y-axis labels for better readability
                    for label in ax.get_yticklabels():
                        label.set_path_effects([path_effects.withStroke(linewidth=label_outline_width, foreground='white')])
                        label.set_color(text_color)
                    if i == 0:
                        from matplotlib.ticker import FixedLocator, FixedFormatter
                        ax.yaxis.set_major_locator(FixedLocator([0.5]))
                        ax.yaxis.set_major_formatter(FixedFormatter(category_names))
            else:
                print(f"ERROR: cat_var_idx {cat_var_idx} >= len(dics_vars) {len(dics_vars)} for axis {i}")
        else:
            # Numeric axis - show original values at normalized positions (keep at original 0-1 scale)
            if i < len(plot_vars):
                var_name = plot_vars[i]
                unique_values = sorted(df_plot[var_name].unique())
                n_values = len(unique_values)
                
                print(f"Axis {i} ({var_name}): {n_values} numeric values = {unique_values}")
                
                if n_values > 1:
                    # Calculate normalized positions for numeric values (keep at original 0-1 scale)
                    label_positions = [j / (n_values - 1) for j in range(n_values)]
                    print(f"  Label positions: {label_positions} (kept at original 0-1 scale)")
                    
                    # Set ticks at normalized positions but show original values
                    ax.set_yticks(label_positions)
                    ax.set_yticklabels([str(val) for val in unique_values], 
                                     fontsize=tick_fontsize, fontweight='bold')
                    
                    # Add white outlines to y-axis labels for better readability
                    for label in ax.get_yticklabels():
                        label.set_path_effects([path_effects.withStroke(linewidth=label_outline_width, foreground='white')])
                        label.set_color(text_color)
                    
                    # For axis 0 (host axis), ensure all ticks are shown
                    if i == 0:
                        from matplotlib.ticker import FixedLocator, FixedFormatter  
                        ax.yaxis.set_major_locator(FixedLocator(label_positions))
                        ax.yaxis.set_major_formatter(FixedFormatter([str(val) for val in unique_values]))
                        print(f"  Host axis final: Forced {len(label_positions)} numeric ticks with FixedLocator")
                else:
                    # Single value - place in middle of original 0-1 range
                    ax.set_yticks([0.5])
                    ax.set_yticklabels([str(unique_values[0])], 
                                     fontsize=tick_fontsize, fontweight='bold')
                    # Add white outlines to y-axis labels for better readability
                    for label in ax.get_yticklabels():
                        label.set_path_effects([path_effects.withStroke(linewidth=label_outline_width, foreground='white')])
                        label.set_color(text_color)
                    if i == 0:
                        from matplotlib.ticker import FixedLocator, FixedFormatter
                        ax.yaxis.set_major_locator(FixedLocator([0.5]))
                        ax.yaxis.set_major_formatter(FixedFormatter([str(unique_values[0])]))
            else:
                print(f"Axis {i}: Using default numeric ticks for expanded range ({plot_y_min:.3f} to {plot_y_max:.3f})")
    
    print(f"All axes set to expanded range {plot_y_min:.3f} to {plot_y_max:.3f} with labels positioned at original 0-1 scale")
    
    # PRECISE COLORBAR POSITIONING: Calculate based on actual axis positions in figure coordinates
    # Ensure figure layout is established for accurate coordinate calculations
    fig.canvas.draw_idle()  # Force layout calculation
    
    # Get actual axis spine positions in figure coordinates (where axes are visually drawn)
    axis_positions_x = []
    for i, ax in enumerate(axes):
        if i == 0:
            # For host axis, get the left spine position (where the axis line is drawn)
            left_spine_pos = ax.spines['left'].get_position()
            if left_spine_pos[0] == 'axes':  # Position relative to axes
                spine_data_coords = (0, 0)  # Left edge of axis in data coordinates
            else:
                spine_data_coords = (0, 0)  # Fallback
            spine_display = ax.transData.transform(spine_data_coords)  # To display coordinates
            spine_fig = fig.transFigure.inverted().transform(spine_display)  # To figure coordinates
            axis_positions_x.append(spine_fig[0])
        else:
            # For twin axes, get the right spine position (where the axis line is actually drawn)
            right_spine_pos = ax.spines['right'].get_position()
            if hasattr(right_spine_pos, '__len__') and len(right_spine_pos) == 2:
                # Position is ('axes', fraction) - get the actual x position
                spine_x_fraction = right_spine_pos[1]
                # Transform from axes coordinates to figure coordinates
                spine_axes_coords = (spine_x_fraction, 0)  # Position in axes coordinates
                spine_display = ax.transAxes.transform(spine_axes_coords)  # To display coordinates
                spine_fig = fig.transFigure.inverted().transform(spine_display)  # To figure coordinates
                axis_positions_x.append(spine_fig[0])
            else:
                # Fallback: use right edge of axes
                right_edge_fig = ax.transAxes.transform((1, 0))  # Right edge in display coordinates  
                right_edge_fig = fig.transFigure.inverted().transform(right_edge_fig)  # To figure coordinates
                axis_positions_x.append(right_edge_fig[0])
    
    print(f"Actual axis spine positions in figure coordinates: {[f'{pos:.4f}' for pos in axis_positions_x]}")
    
    # Calculate spacing between consecutive axes
    if len(axis_positions_x) > 1:
        axis_spacings = [axis_positions_x[i+1] - axis_positions_x[i] for i in range(len(axis_positions_x)-1)]
        avg_axis_spacing = np.mean(axis_spacings)
        print(f"Axis spacings: {[f'{spacing:.4f}' for spacing in axis_spacings]}, average: {avg_axis_spacing:.4f}")
    else:
        avg_axis_spacing = 0.1  # Fallback for single axis
    
    # Position colorbar at same spacing from rightmost axis spine as spacing between axes
    rightmost_axis_x = axis_positions_x[-1]
    cbar_width = 0.02  # Fixed width for colorbar
    cbar_left = rightmost_axis_x + avg_axis_spacing-cbar_width/2

    # Get exact plot area height and vertical position (not the full axis bounding box)
    # Use the actual data coordinate limits transformed to figure coordinates
    plot_bottom_data = (0, plot_y_min)  # Bottom of actual plot area in data coordinates
    plot_top_data = (0, plot_y_max)     # Top of actual plot area in data coordinates
    
    plot_bottom_display = host_ax.transData.transform(plot_bottom_data)  # To display coordinates
    plot_top_display = host_ax.transData.transform(plot_top_data)        # To display coordinates
    
    plot_bottom_fig = fig.transFigure.inverted().transform(plot_bottom_display)  # To figure coordinates
    plot_top_fig = fig.transFigure.inverted().transform(plot_top_display)        # To figure coordinates
    
    axis_bottom = plot_bottom_fig[1]  # Y coordinate of plot bottom
    axis_height = plot_top_fig[1] - plot_bottom_fig[1]  # Height of actual plot area
    
    # Manually increase colorbar height by a minimal amount (3%)
    height_increase = axis_height * -0.04  # 3% increase
    cbar_height = axis_height + height_increase
    cbar_bottom = axis_bottom - (height_increase * 0.5)  # Center the extra height
    
    print(f"Precise plot area: bottom={axis_bottom:.4f}, height={axis_height:.4f}")
    print(f"Colorbar position: left={cbar_left:.4f}, bottom={cbar_bottom:.4f}, width={cbar_width:.4f}, height={cbar_height:.4f}")
    
    # Add colorbar with precise positioning and slightly increased height
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])  # Slightly increased height
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f'{score_type.replace("_", " ").title()}', 
                   fontsize=axis_title_fontsize, fontweight='bold', color=text_color)
    
    # Remove colorbar outline and ticks but keep labels
    cbar.outline.set_visible(False)  # Remove colorbar outline
    cbar.ax.tick_params(length=0, width=0)  # Remove tick marks but keep labels
    
    # Add red line indicator for highest score in colorbar
    max_score = scores.max()
    print(f"Adding red line indicator at highest score: {max_score:.4f}")
    cbar.ax.axhline(y=max_score, color='red', linewidth=axis_line_width*1.5, alpha=0.8, zorder=100)
    
    # Format colorbar tick labels to .2f and handle overlap with red max label
    tick_locations = cbar.get_ticks()
    tick_labels = []
    
    # Define threshold for "too close" to max score (5% of the colorbar range)
    colorbar_range = vmax_adjusted - vmin_adjusted
    proximity_threshold = colorbar_range * 0.05
    
    print(f"Colorbar tick processing: max_score={max_score:.4f}, threshold={proximity_threshold:.4f}")
    
    for tick_value in tick_locations:
        # Check if this tick is too close to the max score
        distance_to_max = abs(tick_value - max_score)
        if distance_to_max < proximity_threshold:
            # Hide this tick label by making it empty
            tick_labels.append('')
            print(f"  Hiding tick at {tick_value:.4f} (distance to max: {distance_to_max:.4f})")
        else:
            # Format to .2f
            tick_labels.append(f'{tick_value:.2f}')
            print(f"  Keeping tick at {tick_value:.4f} -> '{tick_value:.2f}'")
    
    # Apply the formatted labels
    cbar.ax.set_yticklabels(tick_labels)
    
    # Make remaining colorbar tick labels bold
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(tick_fontsize)
        label.set_color(text_color)
    
    # Add text label for the red line aligned with other colorbar labels
    # Use x=1.0 in colorbar axis coordinates to align with regular tick labels  
    red_max_label = cbar.ax.text(1.1, max_score, f'{max_score:.2f} (max)', 
                                transform=cbar.ax.get_yaxis_transform(), 
                                verticalalignment='center', fontsize=tick_fontsize, 
                                color='red', fontweight='bold',
                                horizontalalignment='left')  # Keep red color for max label emphasis
    
    # Connect lines to colorbar if requested
    if connect_to_colorbar:
        print(f"Adding {len(ym_plot)} curved connecting lines to colorbar")
        
        # IMPORTANT: Ensure figure layout is fully established before coordinate transformation
        fig.canvas.draw_idle()  # Force layout calculation
        
        # Get the rightmost axis (last in the axes list) 
        rightmost_axis = axes[-1]
        rightmost_x_data = len(axis_positions_x) - 1  # Data coordinate of rightmost axis
        
        # Transform coordinates for all lines including the red max line
        all_connections = []
        
        # Regular lines
        for j in range(ym_plot.shape[0]):
            rightmost_y = ym_plot[j, -1]  # Last coordinate (rightmost axis) 
            score_value = scores_plot[j]
            line_color = colors_plot[j]
            line_width = score_line_width * 0.7
            alpha = 0.6
            all_connections.append((rightmost_x_data, rightmost_y, score_value, line_color, line_width, alpha))
        
        # Add the red max line connection (it's already included above, but we want it on top)
        max_idx = max_score_idx_sorted
        max_rightmost_y = ym_plot[max_idx, -1]
        max_score_value = scores_plot[max_idx]
        all_connections.append((rightmost_x_data, max_rightmost_y, max_score_value, 'red', score_line_width, 0.9))
        
        # Draw all connections with curved lines
        for start_x_data, start_y_data, score_value, line_color, line_width, alpha in all_connections:
            # Calculate colorbar position for this score using the SAME percentile-based normalization as the colorbar
            # The 'norm' object already accounts for the color_percentile_range adjustment
            normalized_score = norm(score_value)  # This uses vmin_adjusted to vmax_adjusted from percentile range
            
            # Clamp normalized score to [0, 1] to handle edge cases outside percentile range
            normalized_score = np.clip(normalized_score, 0.0, 1.0)
            
            # Calculate y-position on colorbar using ACTUAL colorbar dimensions 
            # (colorbar now spans from cbar_bottom to cbar_bottom + cbar_height)
            cbar_y = cbar_bottom + (normalized_score * cbar_height)
            
            # PROPER COORDINATE TRANSFORMATION: Transform rightmost axis point to figure coordinates
            # The rightmost axis coordinates (start_x_data, start_y_data) need to be converted to figure space
            start_point_data = (start_x_data, start_y_data)
            start_point_fig = host_ax.transData.transform(start_point_data)  # To display coordinates
            start_point_fig = fig.transFigure.inverted().transform(start_point_fig)  # To figure coordinates
            
            start_x_fig, start_y_fig = start_point_fig
            
            # ADJUST FOR LINE THICKNESS: Calculate line thickness offset in figure coordinates
            # Convert line thickness from points to figure coordinates for seamless connection
            line_thickness_points = line_width  # Line width in points
            fig_dpi = fig.dpi if hasattr(fig, 'dpi') else 72  # Default DPI
            line_thickness_inches = line_thickness_points / 72.0  # Convert points to inches
            fig_height_inches = figure_size[1]  # Figure height in inches
            line_thickness_fig = line_thickness_inches / fig_height_inches  # Line thickness in figure coordinates
            
            # Adjust ONLY the colorbar end point based on where the line connects
            # Keep starting point at actual line position on rightmost axis
            # Lines connecting to bottom of colorbar: connect to top edge of colorbar position
            # Lines connecting to top of colorbar: connect to bottom edge of colorbar position
            if normalized_score < 0.5:  # Line connects to lower part of colorbar
                end_y_adjusted = cbar_y + (line_thickness_fig * 0.5)  # Connect to top edge of colorbar position
            else:  # Line connects to upper part of colorbar
                end_y_adjusted = cbar_y - (line_thickness_fig * 0.5)  # Connect to bottom edge of colorbar position
            
            # End: colorbar position (using percentile-normalized score position)
            end_x_fig = cbar_left
            end_y_fig = end_y_adjusted
            
            # Create curved connection using Bézier curve
            # Control points for smooth S-curve
            mid_x = (start_x_fig + end_x_fig) * 0.5
            control1_x = start_x_fig + (mid_x - start_x_fig) * 0.6
            control1_y = start_y_fig  # Use original starting point for control
            control2_x = end_x_fig - (end_x_fig - mid_x) * 0.6
            control2_y = end_y_fig
            
            # Create curved path
            path_coords = [
                (start_x_fig, start_y_fig),                           # Start point at actual line position
                (control1_x, control1_y),                            # Control point 1
                (control2_x, control2_y),                            # Control point 2  
                (end_x_fig, end_y_fig)                               # End point adjusted for line thickness
            ]
            path_codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            
            # Create and add curved path
            path = Path(path_coords, path_codes)
            patch = patches.PathPatch(
                path, 
                facecolor='none', 
                edgecolor=line_color,
                linewidth=line_width, 
                alpha=alpha,
                transform=fig.transFigure,
                clip_on=False,
                zorder=-10  # Very low z-order to ensure lines appear behind all axis elements
            )
            fig.patches.append(patch)
        
        print(f"Added curved connections using percentile-based normalization ({color_percentile_range[0]}-{color_percentile_range[1]}%)")
        print(f"Colorbar normalization range: {vmin_adjusted:.4f} to {vmax_adjusted:.4f}")
        print(f"Using proper coordinate transformation from rightmost axis to figure coordinates")
        
        # Ensure rightmost axis labels appear on top of connection lines
        rightmost_axis = axes[-1]
        for label in rightmost_axis.get_yticklabels():
            label.set_zorder(10)  # High z-order for axis labels
    
    # Title
    filter_str = ""
    if filter_dict:
        filter_str = f" (Filtered: {', '.join([f'{k}_{v}' for k, v in filter_dict.items()])})"
    
    # Create short score description with abbreviations
    if score_type == 'cross_site_score':
        score_desc = "Cross-Site Configuration Performance\nScore = Mean(Mean(bal_OA, bal_F1, prop_OA, prop_F1); i=10)"
    else:  # same_site_score
        score_desc = "Same-Site Configuration Performance\nScore = Mean(Mean(bal_OA, bal_F1, prop_OA, prop_F1); i=10)"
    
    host_ax.set_title(f'{score_desc}{filter_str}', 
                     fontsize=plot_title_fontsize, fontweight='bold', pad=20, color=text_color)
    
    # Add grid to main axis with line thickness and color from parameters
    host_ax.grid(True, alpha=0.3, axis='x', linewidth=axis_line_width, color=axis_line_color)
    
    # Create bean plots below each axis if requested
    if show_bean_plots and bean_ax is not None:
        print(f"\n=== CREATING BEAN PLOTS ===")
        
        # Prepare data for bean plots - use the original filtered data before aggregation
        bean_data = create_combined_scores(plot_df.copy())
        bean_data = split_training_strategy(bean_data)
        
        # Clear the bean plot area
        bean_ax.clear()
        bean_ax.set_xlim(-0.5, len(plot_vars) - 0.5)
        
        # Calculate global score range for consistent y-scale across all bean plots
        all_scores = bean_data[score_type].dropna()
        if color_min is not None and color_max is not None:
            score_min, score_max = color_min, color_max
        else:
            score_min, score_max = all_scores.min(), all_scores.max()
        score_range = score_max - score_min
        bean_ax.set_ylim(score_min - 0.1 * score_range, score_max + 0.1 * score_range)
        
        print(f"Bean plot score range: {score_min:.3f} to {score_max:.3f}")
        
        # Create bean plot for each variable
        for i, var in enumerate(plot_vars):
            print(f"Creating bean plot for {var}")
            
            # Get all scores for this variable (regardless of category)
            var_scores = bean_data[score_type].dropna()
            
            if len(var_scores) == 0:
                print(f"  Skipping {var}: no valid scores")
                continue
            
            # Create single violin plot for this variable
            try:
                violin = bean_ax.violinplot([var_scores.values], positions=[i], 
                                          widths=0.6, showmeans=True, showmedians=True)
                
                # Style the violin with a neutral color
                for pc in violin['bodies']:
                    pc.set_facecolor('lightgray')
                    pc.set_alpha(0.6)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(0.5)
                
                # Style the statistical lines
                for partname in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxs']:
                    if partname in violin:
                        violin[partname].set_edgecolor('black')
                        violin[partname].set_linewidth(1.0)
                
                print(f"  Created violin plot with {len(var_scores)} total scores")
                
            except Exception as e:
                print(f"  Error creating violin plot for {var}: {e}")
                continue
            
            # Add colored scatter points on top
            try:
                # Get unique categories for this variable and assign colors
                categories = sorted(bean_data[var].unique())
                n_categories = len(categories)
                
                if n_categories <= 1:
                    print(f"  Skipping scatter for {var}: only {n_categories} category")
                    continue
                
                # Generate distinct colors for categories
                colors_for_cats = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_categories))
                
                # Add scatter points for each category
                for j, cat in enumerate(categories):
                    cat_data = bean_data[bean_data[var] == cat]
                    cat_scores = cat_data[score_type].dropna()
                    
                    if len(cat_scores) == 0:
                        continue
                    
                    # Add small random horizontal jitter to avoid overlap
                    n_points = len(cat_scores)
                    jitter_strength = 0.15  # How much horizontal spread
                    x_positions = np.full(n_points, i) + np.random.normal(0, jitter_strength, n_points)
                    
                    # Create scatter plot
                    scatter = bean_ax.scatter(
                        x_positions, cat_scores.values,
                        c=[colors_for_cats[j]], 
                        label=f'{format_label_text(str(cat), add_line_breaks=False)}' if i == 0 else "",
                        alpha=0.7, 
                        s=20,  # Point size
                        edgecolors='black',
                        linewidths=0.3
                    )
                    
                    print(f"  Added {len(cat_scores)} scatter points for {cat}")
                
                print(f"  Successfully created scatter overlay with {n_categories} categories")
                
            except Exception as e:
                print(f"  Error creating scatter overlay for {var}: {e}")
        
        # Style the bean plot area
        bean_ax.set_xlabel('')
        bean_ax.set_ylabel(f'{score_type.replace("_", " ").title()}', 
                          fontsize=tick_fontsize, fontweight='bold', color=text_color)
        bean_ax.tick_params(axis='y', labelsize=tick_fontsize-2, colors=text_color)
        bean_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        # Set x-ticks to align with main plot axes
        bean_ax.set_xticks(range(len(plot_vars)))
        bean_ax.set_xticklabels([])  # No labels, they're in the main plot
        
        # Add subtle grid
        bean_ax.grid(True, alpha=0.2, axis='y', linewidth=axis_line_width*0.5, color=axis_line_color)
        
        # Style spines
        for spine in bean_ax.spines.values():
            spine.set_linewidth(axis_line_width*0.5)
            spine.set_color(axis_line_color)
        
        bean_ax.spines['top'].set_visible(False)
        bean_ax.spines['right'].set_visible(False)
        
        # Add legend for the scatter point colors (only show legend for first variable to avoid clutter)
        if len(plot_vars) > 0:
            try:
                legend = bean_ax.legend(loc='upper right', fontsize=tick_fontsize-2, 
                                      frameon=True, fancybox=True, shadow=True,
                                      title=format_label_text(plot_vars[0], add_line_breaks=False))
                legend.get_title().set_fontweight('bold')
                legend.get_title().set_color(text_color)
                for text in legend.get_texts():
                    text.set_color(text_color)
                print(f"Added legend for {plot_vars[0]} categories")
            except Exception as e:
                print(f"Error creating legend: {e}")
        
        print(f"Bean plots completed for {len(plot_vars)} variables")
    
    # Save plot
    if output_dir:
        # Use provided output directory
        plot_dir = output_dir
        os.makedirs(plot_dir, exist_ok=True)
    elif config:
        # Use config-based directory
        plot_dir = os.path.join(config['data_paths']['output_data'], experiment_id, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
    else:
        # Use default experiment directory
        plot_dir = f"../../data/experiments/{experiment_id}/plots"
        os.makedirs(plot_dir, exist_ok=True)
    
    filter_suffix = ""
    if filter_dict:
        filter_suffix = "_" + "_".join([f"{k}_{str(v).replace(' ', '_')}" for k, v in filter_dict.items()])
    
    # Add suffix to distinguish aggregated vs raw data
    agg_suffix = "_raw" if plot_raw_data else "_agg"
    
    output_path = os.path.join(plot_dir, f"parallel_coords_{score_type}{filter_suffix}{agg_suffix}.png")
    plt.savefig(output_path, dpi=1000, transparent=True)
    plt.show()
    
    return output_path

# %%
def find_best_combination(df: pd.DataFrame, score_type: str = 'cross_site_score') -> Dict:
    """
    Find the combination with the highest score.
    
    Args:
        df: DataFrame with experiment results
        score_type: Either 'cross_site_score' or 'same_site_score'
        
    Returns:
        Dictionary with best combination details
    """
    # Create combined scores
    df = create_combined_scores(df)
    
    # Split training strategy
    df = split_training_strategy(df)
    
    # Aggregate results
    agg_df = aggregate_results(df, group_by_testing_site=True)
    
    # Find best combination
    best_idx = agg_df[score_type].idxmax()
    best_row = agg_df.loc[best_idx]
    
    best_combination = {
        'score': best_row[score_type],
        'model': best_row['model'],
        'training_sample_size': best_row['training_sample_size'],
        'spatial_distribution': best_row['spatial_distribution'],
        'class_distribution': best_row['class_distribution'],
        'tuning_method': best_row['tuning_method']
    }
    
    if 'testing_site' in best_row:
        best_combination['testing_site'] = best_row['testing_site']
    
    return best_combination

# %%
def examine_data_combinations(df: pd.DataFrame) -> None:
    """
    Examine what combinations actually exist in the data.
    """
    print(f"\n=== RAW DATA EXAMINATION ===")
    print(f"Total rows in dataset: {len(df)}")
    
    # Split training strategy
    df = split_training_strategy(df)
    
    # Core experiment variables - now using split variables
    exp_vars = ['model', 'training_sample_size', 'spatial_distribution', 'class_distribution', 'tuning_method']
    
    # Check each variable
    for var in exp_vars:
        if var in df.columns:
            unique_vals = sorted(df[var].unique())
            print(f"{var}: {len(unique_vals)} unique values")
            print(f"  Values: {unique_vals}")
        else:
            print(f"WARNING: {var} not found in data!")
    
    # Check testing sites
    if 'testing_site' in df.columns:
        sites = sorted(df['testing_site'].unique())
        print(f"testing_site: {len(sites)} unique values")
        print(f"  Values: {sites}")
    
    # Calculate theoretical maximum combinations
    unique_counts = []
    for var in exp_vars:
        if var in df.columns:
            unique_counts.append(len(df[var].unique()))
    
    if unique_counts:
        theoretical_max = np.prod(unique_counts)
        print(f"\nTheoretical maximum combinations (without testing_site): {theoretical_max}")
        print(f"  = {' × '.join([str(c) for c in unique_counts])}")
    
    # Check actual unique combinations
    actual_combinations = df[exp_vars].drop_duplicates()
    print(f"Actual unique combinations in data: {len(actual_combinations)}")
    
    # Check combinations per testing site
    if 'testing_site' in df.columns:
        print(f"\nCombinations per testing site:")
        for site in sorted(df['testing_site'].unique()):
            site_data = df[df['testing_site'] == site]
            site_combinations = site_data[exp_vars].drop_duplicates()
            print(f"  {site}: {len(site_combinations)} combinations")
    
    # Check if any combinations are missing certain variables
    print(f"\nMissing data check:")
    for var in exp_vars:
        missing = df[var].isna().sum()
        if missing > 0:
            print(f"  {var}: {missing} missing values")
    
    # Show some example combinations
    print(f"\nFirst 10 unique combinations:")
    print(actual_combinations.head(10).to_string())

# %%
def split_training_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split training_sample_strategy into spatial_distribution and class_distribution.
    
    Args:
        df: DataFrame containing training_sample_strategy column
        
    Returns:
        DataFrame with added spatial_distribution and class_distribution columns
    """
    df = df.copy()
    
    # Extract class distribution (prefix: simple, proportional, balanced)
    df['class_distribution'] = df['training_sample_strategy'].str.split('_').str[0]
    
    # Extract spatial distribution (suffix: random, grts, systematic)
    df['spatial_distribution'] = df['training_sample_strategy'].str.split('_').str[1]
    
    print(f"Split training_sample_strategy into:")
    print(f"  class_distribution: {sorted(df['class_distribution'].unique())}")
    print(f"  spatial_distribution: {sorted(df['spatial_distribution'].unique())}")
    
    return df

# %%
def format_label_text(text: str, add_line_breaks: bool = True) -> str:
    """
    Format label text with proper capitalization and optional line breaks.
    
    Args:
        text: Original text to format
        add_line_breaks: If True, split at spaces with newlines
        
    Returns:
        Formatted text
    """
    # Handle special cases first
    if text.lower() == 'grts':
        formatted = 'GRTS'
    elif text == 'BayesSearchCV':
        formatted = 'Bayes\nSearchCV'  # Split into two lines
    else:
        # Standard capitalization
        formatted = text.replace('_', ' ').title()
    
    # Add line breaks at spaces if requested
    if add_line_breaks and ' ' in formatted:
        formatted = formatted.replace(' ', '\n')
    
    return formatted

# %%
# Example usage
def main():
    """Main function to demonstrate the functionality."""
    # Load data
    results_df, config = load_experiment_data()
    
    if results_df is None:
        print("Could not load results data!")
        return
    
    print(f"Loaded {len(results_df)} results")
    print(f"Columns: {list(results_df.columns)}")
    
    # Examine data combinations first
    examine_data_combinations(results_df)
    
    # Find best combinations
    print("\n=== BEST CROSS-SITE COMBINATION ===")
    best_cross = find_best_combination(results_df, 'cross_site_score')
    for key, value in best_cross.items():
        print(f"{key}: {value}")
    
    print("\n=== BEST SAME-SITE COMBINATION ===")
    best_same = find_best_combination(results_df, 'same_site_score')
    for key, value in best_same.items():
        print(f"{key}: {value}")
    
    # Create parallel coordinates plots
    print("\n=== CREATING PLOTS ===")
    
    # Example: For comparable color scales across plots, you can use explicit min/max values:
    # color_min=0.4, color_max=0.8  # Use same range for all plots to compare
    
    testing_site_to_plot = results_df['testing_site'].iloc[0]
    # Plot aggregated data (current approach - fewer lines but cleaner trends)
    print(f"\n1. Aggregated data for single testing site ({testing_site_to_plot}) with bean plots:")
    plot_path1 = plot_parallel_coordinates(
        results_df,
        score_type='cross_site_score',
        filter_dict={'testing_site': testing_site_to_plot},
        experiment_id=config["experiment_id"] if config else "main_experiment", 
        config=config,
        plot_raw_data=False,
        line_spacing=0.001,  # Same spacing for consistency
        #color_percentile_range=(10, 100),  # Same percentile range for consistency
        color_min=0.7,
        color_max=0.9,
        tick_fontsize=18,
        axis_title_fontsize=18,
        plot_title_fontsize=16,
        score_line_width=2.0,
        axis_line_width=3,
        axis_line_color='grey',
        figure_size=(16, 16),  # Increased height for bean plots
        connect_to_colorbar=True,  # Enable colorbar connections
        colormap='cividis',
        text_color='black',
        label_outline_width=3,
        output_dir=config["data_paths"]["output_data"]+"/plots/overall_perfromance",
        show_bean_plots=True,
        bean_plot_height=0.3  # Slightly larger bean plots
    )
    print(f"\n2. Same-site score with smaller bean plots:")
    plot_path2 = plot_parallel_coordinates(
        results_df,
        score_type='same_site_score',
        filter_dict={'testing_site': testing_site_to_plot},
        experiment_id=config["experiment_id"] if config else "main_experiment", 
        config=config,
        plot_raw_data=False,
        line_spacing=0.001,  # Same spacing for consistency
        color_min=0.7,
        color_max=0.9,
        tick_fontsize=16,
        axis_title_fontsize=18,
        plot_title_fontsize=16,
        score_line_width=2.0,
        axis_line_width=3,
        axis_line_color='grey',
        figure_size=(16, 16),  # Standard height
        connect_to_colorbar=True,  # Enable colorbar connections
        colormap='cividis',
        text_color='black',
        label_outline_width=2.0,
        output_dir=config["data_paths"]["output_data"]+"/plots/overall_perfromance",
        show_bean_plots=True,
        bean_plot_height=0.25  # Standard bean plot size
    )

    print(f"\nPlots saved at:")
    print(f"  Cross-site: {plot_path1}")
    print(f"  Same-site: {plot_path2}")

# %%
if __name__ == "__main__":
    main()







# %%

