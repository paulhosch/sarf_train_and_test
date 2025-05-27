#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration parameters for visualizations.
Contains shared settings for plot styling, colors, and default parameters.
"""


# Default plot parameters
DEFAULT_PLOT_PARAMS = {
    'title_fontsize': 14,
    'subtitle_fontsize': 14,
    'axis_label_fontsize': 14, 
    'tick_fontsize': 14,
    'legend_fontsize': 14,
    'gridline_width': 0.7,
    'boxplot_linewidth': 2,
    'bar_linewidth': 1.5,
    'subplot_borderwidth': 1.2
}

# Figure sizes
FIGURE_SIZES = {
    'boxplot_figure': (16, 16),  # Boxplot figure size
    'table_figure': (12, 18),    # Table figure size
    'feature_importance_figure': (16, 16),  # Feature importance figure size
    'standard': (12, 10),        # For backward compatibility
    'wide': (15, 10),            # For backward compatibility
    'single_column': (8, 6),
    'error': (10, 6)
}

# Color palettes (using colorblind-friendly options)
COLOR_PALETTES = {
    'default': 'colorblind',    # seaborn colorblind palette
    'categorical': 'colorblind',
    'sequential': 'viridis',
    'diverging': 'coolwarm'
}

# Specific plot colors
VIOLIN_FACE_COLOR = '#0B1D26'  # Blue
SWARM_PLOT_COLOR = '#B72F6A'   # Orange

# Plot styling defaults
PLOT_STYLE = {
    'grid_alpha': 0.7,
    'grid_style': '--',
    'background_alpha': 0,     # transparent background
    'marker_alpha': 0.5,
    'errorbar_capsize': 3,
    'errorbar_capthick': 1,
    'errorbar_linewidth': 1,
    'errorbar_alpha': 0.7
}

# Layout spacing parameters
LAYOUT_SPACING = {
    'subplot_to_legend_spacing': 0.02,   # Vertical space between subplots and legend
    'legend_to_table_spacing': 0.02,     # Vertical space between legend and table
    'figure_top_margin': 0.99,           # Top margin of the figure
    'figure_bottom_margin': 0.12,        # Bottom margin of the figure
    'bottom_padding_per_row': 0.03,      # Additional bottom padding per table row
    'subplot_horizontal_spacing': 0.1,   # Horizontal space between subplots
    'subplot_vertical_spacing': 0.8      # Vertical space between subplots
}

# Table styling parameters
TABLE_STYLE = {
    'fontsize': 10,
    'header_fontsize': 10,
    'cell_padding': 0.5,
    'header_bg_color': '#E0E0E0',   # Light gray
    'header_text_color': 'black',
    'cell_border_color': '#888888',
    'cell_border_width': 1,
    'scale_width': 1.0,
    'scale_height': 1.5,
    'max_cell_width': 20,       # Maximum characters before wrapping
    'row_header_bg_color': '#F0F0F0',  # Very light gray for row headers
    'alternating_row_colors': [None, '#F9F9F9'],  # None for white, light gray
    'text_wrap_format': False     # Whether to wrap text in cells
}

# Table cell colormapping settings
TABLE_COLORMAP = {
    'enabled': True,                    # Whether to use colormapping for cells
    'cmap_name': 'viridis',             # Colormap to use (viridis, plasma, inferno, magma, cividis)
    'value_range_buffer': 0.05,         # Buffer percentage on min/max for color mapping (0.05 = 5%)
    'invert_cmap': False,               # Whether to invert the colormap
    'text_color_threshold': 0.8,        # Threshold for switching text color (0-1, higher = more dark text)
    'dark_text_color': 'black',         # Text color for bright backgrounds
    'light_text_color': 'white',        # Text color for dark backgrounds
    'alpha': 0.8,                       # Transparency of the background color (0-1)
    'apply_to_columns': ['Proportional OA', 'Balanced OA', 'Proportional F1', 'Balanced F1']  # Columns to apply colormap to
}

# Output settings
OUTPUT_SETTINGS = {
    'dpi': 300,
    'format': 'png',
    'bbox_inches': 'tight'
}

# Line thickness settings
LINE_THICKNESS = 2
LINE_THICKNESS_GRID = 2
