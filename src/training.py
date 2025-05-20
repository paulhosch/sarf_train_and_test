#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from itertools import product
import os
import json
from sklearn.preprocessing import StandardScaler
import time

# Import local modules
from src.data_handler import get_feature_columns, prepare_X_y, prepare_combined_training_data, load_samples
from src.models import create_model, tune_model, save_model
from src.evaluation import compute_metrics, compute_all_feature_importances

# %% Configuration handling

def generate_config_combinations(config: Dict) -> List[Dict]:
    """
    Generate all combinations of experiment variables
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        List of configuration dictionaries
    """
    # Get experiment variables
    models = config["experiment_variables"]["models"]
    training_strategies = config["experiment_variables"]["training_sample_strategies"]
    training_sizes = config["experiment_variables"]["training_sample_sizes"]
    tuning_methods = config["experiment_variables"]["tuning_methods"]
    
    # Generate all combinations
    combinations = list(product(models, training_strategies, training_sizes, tuning_methods))
    
    # Create configuration dictionaries
    config_list = []
    for model, strategy, size, tuning in combinations:
        config_dict = {
            "model": model,
            "training_sample_strategy": strategy,
            "training_sample_size": size,
            "tuning_method": tuning,
            "configuration_name": f"{model}_{strategy}_{size}_{tuning}"
        }
        config_list.append(config_dict)
    
    return config_list

# %% Training and evaluation

def run_logo_cv_fold(datasets: Dict,
                    config_dict: Dict,
                    main_config: Dict,
                    iteration: int,
                    testing_site: str,
                    new_samples_each_iteration: bool = False,
                    iteration_id: Optional[str] = None) -> Dict:
    """
    Run a single fold of Leave-One-Group-Out Cross Validation
    
    Args:
        datasets: Dictionary with datasets organized by case study
        config_dict: Configuration for this specific experiment
        main_config: Main experiment configuration
        iteration: Current iteration number
        testing_site: Site to use for testing
        new_samples_each_iteration: Flag to load new samples each iteration
        iteration_id: ID for the current iteration
        
    Returns:
        Dictionary with results for this fold
    """
    # Extract parameters from configurations
    model_name = config_dict["model"]
    training_strategy = config_dict["training_sample_strategy"]
    training_size = config_dict["training_sample_size"]
    tuning_method = config_dict["tuning_method"]
    
    # Get categorical and continuous feature columns
    categorical_cols, continuous_cols = get_feature_columns(main_config)
    feature_cols = categorical_cols + continuous_cols
    
    training_label_col = main_config["experiment_constants"]["training_label_column"]
    testing_label_col = main_config["experiment_constants"]["testing_label_column"]
    
    # Determine training sites (all except testing site)
    case_studies = main_config["experiment_constants"]["case_studies"]
    training_sites = [site for site in case_studies if site != testing_site]
    
    # Get dataset key for training
    training_key = f"{training_strategy}_{training_size}"
    
    # Get training data from all training sites
    training_data_dict = {}
    for site in training_sites:
        training_data_dict[site] = load_samples(
            site, "training", training_size, training_strategy, main_config,
            new_samples_each_iteration=new_samples_each_iteration, iteration_id=iteration_id
        )
    
    # Combine training data
    combined_training_data = prepare_combined_training_data(training_data_dict)
    X_train, y_train = prepare_X_y(combined_training_data, feature_cols, training_label_col, categorical_cols, main_config)
    
    # Initialize model
    default_params = main_config["model_parameters"][model_name]["default_parameters"]
    
    # Calculate sample weights for Weighted Random Forest
    sample_weight = None
    if model_name == "Weighted Random Forest":
        # Calculate inverse class frequencies
        class_counts = y_train.value_counts()
        total_samples = len(y_train)
        weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
        sample_weight = y_train.map(weights).values
    
    # Apply tuning if specified
    best_params = default_params.copy()
    training_time = None
    if tuning_method == "BayesSearchCV":
        start_train = time.time()
        best_params, tuned_model = tune_model(
            model_name, X_train, y_train, main_config, sample_weight
        )
        model = tuned_model  # Use the already trained model
        training_time = time.time() - start_train
    else:
        # Create and train model without tuning
        model = create_model(model_name, default_params)
        start_train = time.time()
        if sample_weight is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        training_time = time.time() - start_train
    
    # Save the trained model
    experiment_id = main_config["experiment_id"]
    config_name = config_dict["configuration_name"]
    model_path = save_model(model, experiment_id, f"{config_name}_{testing_site}", iteration, main_config)
    
    # Get the balanced and proportional testing strategies
    balanced_strategy = "balanced_systematic"
    proportional_strategy = "proportional_systematic"
    testing_size = main_config["experiment_constants"]["testing_sample_sizes"][0]  # Use the only testing size
    
    # Setup testing data for same-site evaluation by combining datasets from all training sites
    same_site_results = {}
    
    # 1. Combine and process balanced testing datasets from all training sites
    combined_balanced_test_data = []
    for site in training_sites:
        balanced_testing_key = f"{balanced_strategy}_{testing_size}"
        if balanced_testing_key in datasets[site]["testing"]:
            site_balanced_data = datasets[site]["testing"][balanced_testing_key]
            combined_balanced_test_data.append(site_balanced_data)
    
    if combined_balanced_test_data:
        # Combine all balanced test datasets
        combined_balanced_df = pd.concat(combined_balanced_test_data, ignore_index=True)
        
        # Limit to specified testing_size by random sampling if needed
        if len(combined_balanced_df) > testing_size:
            combined_balanced_df = combined_balanced_df.sample(testing_size, random_state=main_config["experiment_constants"]["random_seed"])
        
        # Prepare features and labels with categorical feature transformation
        X_test_balanced, y_test_balanced = prepare_X_y(combined_balanced_df, feature_cols, testing_label_col, categorical_cols, main_config)
        
        # For each prediction, track prediction time
        prediction_times = {}
        
        start_pred = time.time()
        y_pred_balanced = model.predict(X_test_balanced)
        prediction_times["same_site_balanced"] = time.time() - start_pred
        
        # Compute metrics - these are now "balanced" metrics because of the dataset
        balanced_metrics = compute_metrics(y_test_balanced, y_pred_balanced)
        
        # Store with "balanced" prefix
        same_site_results["balanced_OA_same_site"] = balanced_metrics["OA"]
        same_site_results["balanced_F1_same_site"] = balanced_metrics["F1"]
        same_site_results["balanced_Confusion_Matrix_same_site"] = balanced_metrics["Confusion_Matrix"]
    
    # 2. Combine and process proportional testing datasets from all training sites
    combined_proportional_test_data = []
    for site in training_sites:
        proportional_testing_key = f"{proportional_strategy}_{testing_size}"
        if proportional_testing_key in datasets[site]["testing"]:
            site_proportional_data = datasets[site]["testing"][proportional_testing_key]
            combined_proportional_test_data.append(site_proportional_data)
    
    if combined_proportional_test_data:
        # Combine all proportional test datasets
        combined_proportional_df = pd.concat(combined_proportional_test_data, ignore_index=True)
        
        # Limit to specified testing_size by random sampling if needed
        if len(combined_proportional_df) > testing_size:
            combined_proportional_df = combined_proportional_df.sample(testing_size, random_state=main_config["experiment_constants"]["random_seed"])
        
        # Prepare features and labels with categorical feature transformation
        X_test_proportional, y_test_proportional = prepare_X_y(combined_proportional_df, feature_cols, testing_label_col, categorical_cols, main_config)
        
        start_pred = time.time()
        y_pred_proportional = model.predict(X_test_proportional)
        prediction_times["same_site_proportional"] = time.time() - start_pred
        
        # Compute metrics - these are now "proportional" metrics because of the dataset
        proportional_metrics = compute_metrics(y_test_proportional, y_pred_proportional)
        
        # Store with "proportional" prefix
        same_site_results["proportional_OA_same_site"] = proportional_metrics["OA"]
        same_site_results["proportional_F1_same_site"] = proportional_metrics["F1"]
        same_site_results["proportional_Confusion_Matrix_same_site"] = proportional_metrics["Confusion_Matrix"]
    
    # Setup testing data for cross-site evaluation (test on left-out site)
    cross_site_results = {}
    
    # 1. Evaluate using the balanced testing dataset from the testing site
    balanced_testing_key = f"{balanced_strategy}_{testing_size}"
    if balanced_testing_key in datasets[testing_site]["testing"]:
        balanced_test_data = datasets[testing_site]["testing"][balanced_testing_key]
        X_test_balanced, y_test_balanced = prepare_X_y(balanced_test_data, feature_cols, testing_label_col, categorical_cols, main_config)
        
        start_pred = time.time()
        y_pred_balanced = model.predict(X_test_balanced)
        prediction_times["cross_site_balanced"] = time.time() - start_pred
        
        # Compute metrics
        balanced_metrics = compute_metrics(y_test_balanced, y_pred_balanced)
        
        # Store with "balanced" prefix
        cross_site_results["balanced_OA_cross_site"] = balanced_metrics["OA"]
        cross_site_results["balanced_F1_cross_site"] = balanced_metrics["F1"]
        cross_site_results["balanced_Confusion_Matrix_cross_site"] = balanced_metrics["Confusion_Matrix"]
    
    # 2. Evaluate using the proportional testing dataset from the testing site
    proportional_testing_key = f"{proportional_strategy}_{testing_size}" 
    if proportional_testing_key in datasets[testing_site]["testing"]:
        proportional_test_data = datasets[testing_site]["testing"][proportional_testing_key]
        X_test_proportional, y_test_proportional = prepare_X_y(proportional_test_data, feature_cols, testing_label_col, categorical_cols, main_config)
        
        start_pred = time.time()
        y_pred_proportional = model.predict(X_test_proportional)
        prediction_times["cross_site_proportional"] = time.time() - start_pred
        
        # Compute metrics
        proportional_metrics = compute_metrics(y_test_proportional, y_pred_proportional)
        
        # Store with "proportional" prefix
        cross_site_results["proportional_OA_cross_site"] = proportional_metrics["OA"]
        cross_site_results["proportional_F1_cross_site"] = proportional_metrics["F1"]
        cross_site_results["proportional_Confusion_Matrix_cross_site"] = proportional_metrics["Confusion_Matrix"]
    
    # Calculate feature importances
    importance_methods = main_config["experiment_constants"]["feature_importance_methods"]
    try:
        print(f"DEBUG: Starting feature importance calculation with methods: {importance_methods}")
        feature_importances = compute_all_feature_importances(
            model, X_train, y_train, importance_methods, main_config["experiment_constants"]["random_seed"]
        )
        print(f"DEBUG: Feature importance calculation completed")
    except Exception as e:
        print(f"DEBUG: Error calculating feature importances: {str(e)}")
        # Create empty feature importance dictionaries
        feature_importances = {method: {"error": f"Failed to calculate: {str(e)}"} for method in importance_methods}
    
    # Combine all results
    results = {
        "iteration": iteration,
        "configuration_name": config_dict["configuration_name"],
        "model": model_name,
        "training_sample_size": training_size,
        "training_sample_strategy": training_strategy,
        "best_params": best_params if tuning_method == "BayesSearchCV" else None,
        "testing_site": testing_site,
        "training_sites": ",".join(training_sites),
        "model_path": model_path,
        "training_time": training_time,
        "prediction_times": prediction_times
    }
    
    # Add metrics
    results.update(same_site_results)
    results.update(cross_site_results)
    
    # Add feature importances
    for method, importances in feature_importances.items():
        # Convert dictionary to string to avoid serialization issues
        results[f"feature_importance_{method}"] = str(importances)
    
    return results

def run_experiment(config_path: str = "experiment_config.json", new_samples_each_iteration: bool = False) -> pd.DataFrame:
    """
    Run the complete experiment with all configurations and iterations
    
    Args:
        config_path: Path to the experiment configuration file
        new_samples_each_iteration: Flag to load new samples each iteration
        
    Returns:
        DataFrame with all experiment results
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    experiment_id = config["experiment_id"]
    
    # Create output directories
    output_data_path = config["data_paths"]["output_data"]
    results_dir = f"{output_data_path}/{experiment_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate all configuration combinations
    config_combinations = generate_config_combinations(config)
    
    # Import modules here to avoid circular imports
    from src.data_handler import prepare_datasets_for_logo_cv
    
    # Prepare datasets for all case studies
    print("Loading datasets...")
    datasets = prepare_datasets_for_logo_cv(config)
    
    # Initialize results list
    all_results = []
    
    # Get number of iterations
    iterations = config["experiment_constants"]["experiment_iterations"]
    case_studies = config["experiment_constants"]["case_studies"]
    
    # Start timing
    start_time = time.time()
    
    # Run all iterations, configurations, and LOGO-CV folds
    total_runs = iterations * len(config_combinations) * len(case_studies)
    run_count = 0
    
    print(f"Starting experiment with {total_runs} total runs...")
    
    for iteration in range(1, iterations + 1):
        iteration_id = f"iteration_{iteration}"
        # Generate a new random seed for this iteration based on the base seed
        iteration_seed = config["experiment_constants"]["random_seed"] + iteration
        
        for config_dict in config_combinations:
            # Update the random_state in model parameters for this iteration
            for model_name, params in config_dict.get('model_parameters', {}).items():
                if 'random_state' in params:
                    params['random_state'] = iteration_seed
            
            for testing_site in case_studies:
                run_count += 1
                
                # Print progress
                elapsed_time = time.time() - start_time
                avg_time_per_run = elapsed_time / run_count
                estimated_total_time = avg_time_per_run * total_runs
                estimated_remaining_time = estimated_total_time - elapsed_time
                
                print(f"Run {run_count}/{total_runs} - "
                      f"Iteration {iteration} (seed: {iteration_seed}), Config: {config_dict['configuration_name']}, "
                      f"Testing site: {testing_site}")
                print(f"Elapsed: {elapsed_time:.2f}s, Estimated remaining: {estimated_remaining_time:.2f}s")
                
                # Run this specific experiment configuration
                try:
                    result = run_logo_cv_fold(
                        datasets, config_dict, config, iteration, testing_site,
                        new_samples_each_iteration=new_samples_each_iteration, iteration_id=iteration_id
                    )
                    all_results.append(result)
                    
                    # Save intermediate results
                    intermediate_df = pd.DataFrame(all_results)
                    intermediate_df.to_csv(f"{results_dir}/results_partial.csv", index=False)
                    
                except Exception as e:
                    print(f"Error in run: {e}")
                    # Continue with next run
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save final results
    results_df.to_csv(f"{results_dir}/results.csv", index=False)
    
    print(f"Experiment completed in {time.time() - start_time:.2f} seconds")
    print(f"Results saved to {results_dir}/results.csv")
    
    return results_df 