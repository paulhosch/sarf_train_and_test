#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import os

# %% Helper functions

def create_output_dir(experiment_id: str, config: Dict) -> str:
    """
    Create output directory for plots
    
    Args:
        experiment_id: Name of the experiment
        config: Experiment configuration with data paths
        
    Returns:
        Path to the plots directory
    """
    output_data_path = config["data_paths"]["output_data"]
    plot_dir = f"{output_data_path}/{experiment_id}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

# %% Performance metric plots

def plot_performance_metrics(results_df: pd.DataFrame,
                           groupby: str,
                           experiment_id: str,
                           compare_by: str,
                           config: Dict,
                           distribution: str = "iterations") -> str:
    """
    Plot boxplots of OA and F1 scores for proportional and balanced metrics
    
    Args:
        results_df: DataFrame containing results
        groupby: Column to group results by
        experiment_id: ID of the experiment
        compare_by: What to compare on x-axis
        config: Experiment configuration with data paths
        distribution: What to use for boxplot distribution
        
    Returns:
        Path to saved plot
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Set titles for the subplots
    axs[0, 0].set_title('Proportional OA')
    axs[0, 1].set_title('Balanced OA')
    axs[1, 0].set_title('Proportional F1')
    axs[1, 1].set_title('Balanced F1')
    
    # Get unique values for x-axis
    if compare_by == "all":
        # Use configuration_name for all configs
        x_values = results_df["configuration_name"].unique()
        x_col = "configuration_name"
    else:
        # Check if the column exists
        if compare_by not in results_df.columns:
            print(f"WARNING: Column '{compare_by}' not found in results DataFrame.")
            print(f"Available columns: {results_df.columns.tolist()}")
            print(f"Using 'configuration_name' instead.")
            x_values = results_df["configuration_name"].unique()
            x_col = "configuration_name"
            
            # If we're comparing tuning methods, try to extract from configuration_name
            if compare_by == "tuning_method":
                # Extract tuning method from configuration name (last part after last _)
                results_df["tuning_method"] = results_df["configuration_name"].str.split("_").str[-1]
                if "tuning_method" in results_df.columns:
                    x_values = results_df["tuning_method"].unique()
                    x_col = "tuning_method"
        else:
            # Use a specific column
            x_values = results_df[compare_by].unique()
            x_col = compare_by
    
    # Define metrics for each subplot
    metrics = [
        "proportional_OA_same_site",
        "balanced_OA_same_site",
        "proportional_F1_same_site",
        "balanced_F1_same_site"
    ]
    
    # Generate boxplots
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        
        # Check if metric exists
        if metric not in results_df.columns:
            axs[row, col].text(0.5, 0.5, f"Metric '{metric}' not available", 
                              ha='center', va='center')
            continue
        
        # Group data for plotting
        if distribution == "iterations":
            # Boxplot across iterations
            plot_data = []
            for x_val in x_values:
                filtered_data = results_df[results_df[x_col] == x_val][metric]
                plot_data.append(filtered_data)
            
            axs[row, col].boxplot(plot_data, labels=x_values)
        
        elif distribution == "logo_folds":
            # Boxplot across logo folds (testing sites)
            sns.boxplot(data=results_df, x=x_col, y=metric, ax=axs[row, col])
        
        # Set labels
        axs[row, col].set_xlabel(x_col)
        axs[row, col].set_ylabel(metric)
        
        # Rotate x-axis labels if needed
        if len(str(x_values[0])) > 10:
            plt.setp(axs[row, col].get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_dir = create_output_dir(experiment_id, config)
    output_path = f"{plot_dir}/performance_metrics_{groupby}_{compare_by}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def plot_cross_site_vs_same_site(results_df: pd.DataFrame,
                                experiment_id: str,
                                config: Dict,
                                metric: str = "F1") -> str:
    """
    Plot comparison between same-site and cross-site performance
    
    Args:
        results_df: DataFrame containing results
        experiment_id: ID of the experiment
        config: Experiment configuration with data paths
        metric: Which metric to plot ('OA' or 'F1')
        
    Returns:
        Path to saved plot
    """
    # Check required columns
    same_site_col = f"proportional_{metric}_same_site"
    cross_site_col = f"proportional_{metric}_cross_site"
    
    required_columns = ["testing_site", "model", same_site_col, cross_site_col]
    missing_columns = [col for col in required_columns if col not in results_df.columns]
    
    if missing_columns:
        print(f"WARNING: Missing required columns: {missing_columns}")
        print(f"Available columns: {results_df.columns.tolist()}")
        
        # Create error plot
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f"Cannot create plot: Missing columns {missing_columns}", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        
        # Save error plot
        plot_dir = create_output_dir(experiment_id, config)
        output_path = f"{plot_dir}/cross_site_vs_same_site_{metric}_error.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    # Get unique testing sites
    testing_sites = results_df["testing_site"].unique()
    
    # Create figure
    fig, axs = plt.subplots(1, len(testing_sites), figsize=(15, 5))
    
    # Handle case with only one testing site
    if len(testing_sites) == 1:
        axs = [axs]
    
    # Plot for each testing site
    for i, site in enumerate(testing_sites):
        try:
            site_data = results_df[results_df["testing_site"] == site]
            
            # Check if we have data for this site
            if len(site_data) == 0:
                axs[i].text(0.5, 0.5, f"No data for site: {site}", 
                          ha='center', va='center')
                axs[i].axis('off')
                continue
                
            # Group by model type for clarity
            grouped_data = site_data.groupby("model").agg({
                same_site_col: ["mean", "std"],
                cross_site_col: ["mean", "std"]
            })
            
            # Plot bars
            x = np.arange(len(grouped_data.index))
            width = 0.35
            
            axs[i].bar(x - width/2, grouped_data[same_site_col]["mean"], 
                      width, yerr=grouped_data[same_site_col]["std"],
                      label='Same Site')
            
            axs[i].bar(x + width/2, grouped_data[cross_site_col]["mean"], 
                      width, yerr=grouped_data[cross_site_col]["std"],
                      label='Cross Site')
            
            # Add labels and title
            axs[i].set_xlabel('Model')
            axs[i].set_ylabel(f'Proportional {metric}')
            axs[i].set_title(f'Testing Site: {site}')
            axs[i].set_xticks(x)
            axs[i].set_xticklabels(grouped_data.index, rotation=45, ha='right')
            axs[i].legend()
        except Exception as e:
            print(f"ERROR plotting site {site}: {str(e)}")
            axs[i].text(0.5, 0.5, f"Error plotting site {site}:\n{str(e)}", 
                      ha='center', va='center', wrap=True)
            axs[i].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_dir = create_output_dir(experiment_id, config)
    output_path = f"{plot_dir}/cross_site_vs_same_site_{metric}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def plot_confusion_matrices(results_df: pd.DataFrame,
                          experiment_id: str,
                          config1: str,
                          config2: str,
                          config: Dict) -> str:
    """
    Plot confusion matrices comparing two configurations
    
    Args:
        results_df: DataFrame containing results
        experiment_id: ID of the experiment
        config1: Name of first configuration
        config2: Name of second configuration
        config: Experiment configuration with data paths
        
    Returns:
        Path to saved plot
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Get data for each configuration
    config1_data = results_df[results_df["configuration_name"] == config1]
    config2_data = results_df[results_df["configuration_name"] == config2]
    
    # Check if data is available
    if len(config1_data) == 0 or len(config2_data) == 0:
        print(f"WARNING: Missing data for one or both configurations")
        print(f"Config1 ({config1}): {len(config1_data)} rows")
        print(f"Config2 ({config2}): {len(config2_data)} rows")
        
        # Create error message plot
        for i in range(2):
            for j in range(2):
                axs[i, j].text(0.5, 0.5, "Configuration data unavailable", 
                             ha='center', va='center')
                axs[i, j].axis('off')
        
        plt.tight_layout()
        plot_dir = create_output_dir(experiment_id, config)
        output_path = f"{plot_dir}/confusion_matrices_error.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    # Helper function to parse confusion matrix
    def parse_matrix(matrix_data):
        if isinstance(matrix_data, str):
            try:
                # Replace single quotes with double quotes for proper JSON parsing
                matrix_data = matrix_data.replace("'", '"')
                return eval(matrix_data)  # Use eval as a fallback
            except:
                print(f"WARNING: Could not parse matrix: {matrix_data[:50]}...")
                return [[0, 0], [0, 0]]  # Return empty matrix as fallback
        else:
            return matrix_data
        
    # Compute average confusion matrices
    try:
        matrices = {
            "config1_proportional": np.mean([np.array(parse_matrix(cm)) for cm in config1_data["proportional_Confusion_Matrix_same_site"] if cm is not None], axis=0),
            "config1_balanced": np.mean([np.array(parse_matrix(cm)) for cm in config1_data["balanced_Confusion_Matrix_same_site"] if cm is not None], axis=0),
            "config2_proportional": np.mean([np.array(parse_matrix(cm)) for cm in config2_data["proportional_Confusion_Matrix_same_site"] if cm is not None], axis=0),
            "config2_balanced": np.mean([np.array(parse_matrix(cm)) for cm in config2_data["balanced_Confusion_Matrix_same_site"] if cm is not None], axis=0)
        }
    except Exception as e:
        print(f"ERROR computing matrices: {str(e)}")
        # Create error message plot
        for i in range(2):
            for j in range(2):
                axs[i, j].text(0.5, 0.5, f"Error computing matrices:\n{str(e)}", 
                             ha='center', va='center')
                axs[i, j].axis('off')
        
        plt.tight_layout()
        plot_dir = create_output_dir(experiment_id, config)
        output_path = f"{plot_dir}/confusion_matrices_error.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    # Plot each confusion matrix
    titles = [
        f"{config1} - Proportional",
        f"{config2} - Proportional",
        f"{config1} - Balanced",
        f"{config2} - Balanced"
    ]
    
    plot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    matrices_to_plot = [
        matrices["config1_proportional"],
        matrices["config2_proportional"],
        matrices["config1_balanced"],
        matrices["config2_balanced"]
    ]
    
    for (row, col), matrix, title in zip(plot_positions, matrices_to_plot, titles):
        # Create heatmap
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", ax=axs[row, col])
        
        # Set labels
        axs[row, col].set_xlabel("Predicted")
        axs[row, col].set_ylabel("Actual")
        axs[row, col].set_title(title)
        
        # Set tick labels
        axs[row, col].set_xticklabels(["No Flood", "Flood"])
        axs[row, col].set_yticklabels(["No Flood", "Flood"])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_dir = create_output_dir(experiment_id, config)
    output_path = f"{plot_dir}/confusion_matrices_{config1}_vs_{config2}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

# %% Feature importance plots

def plot_feature_importance(results_df: pd.DataFrame,
                          experiment_id: str,
                          config: Dict,
                          min_importance: float = 0.01) -> str:
    """
    Plot feature importance boxplots
    
    Args:
        results_df: DataFrame containing results
        experiment_id: ID of the experiment
        config: Experiment configuration with data paths
        min_importance: Minimum importance to show feature separately
        
    Returns:
        Path to saved plot
    """
    # Extract feature importance data
    methods = ["mdi", "mda", "shap_importance"]
    
    # Create a DataFrame for plotting
    plot_data = []
    
    for method in methods:
        method_col = f"feature_importance_{method}"
        
        # Check if this column exists
        if method_col not in results_df.columns:
            print(f"WARNING: Column '{method_col}' not found in results DataFrame")
            continue
        
        # Extract feature importances from all rows
        for _, row in results_df.iterrows():
            importances = row[method_col]
            
            # Convert string representation to dict if needed
            if isinstance(importances, str):
                try:
                    # Replace single quotes with double quotes for proper JSON parsing
                    importances = importances.replace("'", '"')
                    importances = eval(importances)  # Use eval as a fallback
                except:
                    print(f"WARNING: Could not parse importance string: {importances[:50]}...")
                    continue
            
            # Skip if not a dictionary
            if not isinstance(importances, dict):
                print(f"WARNING: Unexpected importance type: {type(importances)}")
                continue
                
            for feature, importance in importances.items():
                # Skip if importance is not a number
                if not isinstance(importance, (int, float)):
                    continue
                    
                plot_data.append({
                    "Method": method,
                    "Feature": feature,
                    "Importance": importance
                })
    
    # Check if we have data to plot
    if not plot_data:
        print("ERROR: No valid feature importance data found")
        
        # Create a simple error plot
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No valid feature importance data found", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        
        # Save the error plot
        plot_dir = create_output_dir(experiment_id, config)
        output_path = f"{plot_dir}/feature_importance_error.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    plot_df = pd.DataFrame(plot_data)
    
    # Calculate mean importance for each feature across all methods
    feature_means = plot_df.groupby("Feature")["Importance"].mean().reset_index()
    feature_means = feature_means.sort_values("Importance", ascending=False)
    
    # Combine features with low importance as "Others"
    important_features = feature_means[feature_means["Importance"] >= min_importance]["Feature"].tolist()
    
    plot_df["Feature_Group"] = plot_df["Feature"].apply(
        lambda x: x if x in important_features else "Others"
    )
    
    # For "Others" group, sum importances
    if "Others" in plot_df["Feature_Group"].unique():
        others_df = plot_df[plot_df["Feature_Group"] == "Others"].copy()
        others_sum = others_df.groupby(["Method", "Feature_Group"])["Importance"].sum().reset_index()
        
        # Remove individual "Others" entries and add the sum
        plot_df = plot_df[plot_df["Feature_Group"] != "Others"]
        plot_df = pd.concat([plot_df, others_sum])
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Create boxplot
    sns.boxplot(data=plot_df, x="Feature_Group", y="Importance", hue="Method")
    
    # Add labels and title
    plt.xlabel("Feature")
    plt.ylabel("Importance (0-1)")
    plt.title("Feature Importance Across Methods")
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    plt.legend(title="Method")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_dir = create_output_dir(experiment_id, config)
    output_path = f"{plot_dir}/feature_importance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path 