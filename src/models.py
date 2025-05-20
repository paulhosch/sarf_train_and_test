#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import joblib
import os

# %% Model factory function

def create_model(model_name: str, params: Dict = None) -> RandomForestClassifier:
    """
    Create a model instance based on name and parameters
    
    Args:
        model_name: Name of the model type
        params: Dictionary of model parameters
        
    Returns:
        Initialized model instance
    """
    if params is None:
        params = {}
    
    if model_name == "Random Forest":
        return RandomForestClassifier(**params)
    
    elif model_name == "Balanced Random Forest":
        # Ensure class_weight is set to 'balanced'
        if 'class_weight' not in params:
            params['class_weight'] = 'balanced'
        return RandomForestClassifier(**params)
    
    elif model_name == "Weighted Random Forest":
        # This will be configured with sample weights during training
        return RandomForestClassifier(**params)
    
    else:
        raise ValueError(f"Unknown model type: {model_name}")

# %% Model tuning

def create_search_space(tunable_params: Dict) -> Dict:
    """
    Convert parameter ranges to skopt search space
    
    Args:
        tunable_params: Dictionary with parameter ranges
        
    Returns:
        Dictionary with skopt search space objects
    """
    search_space = {}
    
    for param_name, param_values in tunable_params.items():
        # Handle numerical ranges (min/max values)
        if param_name == "n_estimators" or param_name == "max_leaf_nodes":
            # For parameters that should be integers within a range
            if isinstance(param_values, list) and len(param_values) == 2 and all(isinstance(v, int) for v in param_values):
                search_space[param_name] = Integer(param_values[0], param_values[1])
            elif isinstance(param_values, list) and all(isinstance(v, int) for v in param_values):
                # If a list of specific integer values is provided
                search_space[param_name] = Categorical(param_values)
            else:
                print(f"Warning: Unexpected format for {param_name}: {param_values}")
        
        elif param_name == "max_depth":
            # Handle None value in max_depth
            if isinstance(param_values, list) and len(param_values) == 2 and all(isinstance(v, int) or v is None for v in param_values):
                # If we have a range with None, need to handle it specially
                non_none_values = [v for v in param_values if v is not None]
                if len(non_none_values) == 2:
                    # If both values are not None, treat as Integer range
                    search_space[param_name] = Integer(min(non_none_values), max(non_none_values))
                elif len(non_none_values) == 1:
                    # If one value is None, use Categorical
                    search_space[param_name] = Categorical(non_none_values + [None])
            elif isinstance(param_values, list):
                # If a list of specific values is provided (including None)
                search_space[param_name] = Categorical(param_values)
            else:
                print(f"Warning: Unexpected format for {param_name}: {param_values}")
        
        elif param_name in ["min_samples_split", "min_samples_leaf"]:
            # Integer parameters
            if isinstance(param_values, list) and len(param_values) == 2 and all(isinstance(v, int) for v in param_values):
                search_space[param_name] = Integer(param_values[0], param_values[1])
            elif isinstance(param_values, list) and all(isinstance(v, int) for v in param_values):
                search_space[param_name] = Categorical(param_values)
            else:
                print(f"Warning: Unexpected format for {param_name}: {param_values}")
        
        elif param_name == "max_samples":
            # Handle float range for max_samples
            if isinstance(param_values, list) and len(param_values) == 2 and all(isinstance(v, (int, float)) for v in param_values):
                search_space[param_name] = Real(param_values[0], param_values[1])
            elif isinstance(param_values, list):
                search_space[param_name] = Categorical(param_values)
            else:
                print(f"Warning: Unexpected format for {param_name}: {param_values}")
                
        elif param_name in ["max_features", "criterion"]:
            # Categorical string parameters
            search_space[param_name] = Categorical(param_values)
            
        elif param_name == "bootstrap":
            # Boolean parameter
            if isinstance(param_values, list) and all(isinstance(v, bool) for v in param_values):
                search_space[param_name] = Categorical(param_values)
            else:
                print(f"Warning: Unexpected format for bootstrap: {param_values}")
        
        else:
            # Default case: treat as categorical
            print(f"Unrecognized parameter: {param_name}. Treating as categorical.")
            search_space[param_name] = Categorical(param_values)
    
    return search_space

def custom_scorer(estimator, X, y, sample_weight=None):
    """
    Custom scoring function combining OA and F1 with configurable weights
    
    Args:
        estimator: The fitted estimator to evaluate
        X: Test data features
        y: True labels for X
        sample_weight: Optional weights for samples
        
    Returns:
        Combined score value
    """
    print(f"DEBUG: Inside custom_scorer with estimator: {type(estimator)}")
    
    # Generate predictions
    y_pred = estimator.predict(X)
    
    # Calculate metrics
    proportional_oa = accuracy_score(y, y_pred, sample_weight=sample_weight)
    proportional_f1 = f1_score(y, y_pred, sample_weight=sample_weight)
    
    # Equal weighting (can be parameterized from config)
    combined_score = 0.5 * proportional_oa + 0.5 * proportional_f1
    
    print(f"DEBUG: Score calculated: {combined_score}")
    return combined_score

def tune_model(model_name: str, 
              X_train: pd.DataFrame, 
              y_train: pd.Series,
              config: Dict,
              sample_weight: Optional[np.ndarray] = None) -> Tuple[Dict, Any]:
    """
    Tune model hyperparameters using Bayesian optimization
    
    Args:
        model_name: Name of the model to tune
        X_train: Training features
        y_train: Training labels
        config: Experiment configuration
        sample_weight: Optional sample weights
        
    Returns:
        Tuple of (best parameters, best estimator)
    """
    print(f"DEBUG: Starting model tuning for {model_name}")
    
    # Get tunable parameters and default parameters from config
    tunable_params = config["model_parameters"][model_name]["tunable_parameters"]
    default_params = config["model_parameters"][model_name]["default_parameters"].copy()
    
    print(f"DEBUG: Tunable parameters: {tunable_params}")
    print(f"DEBUG: Default parameters: {default_params}")
    
    # Create base model with default parameters - ensure we have all required defaults
    # This includes basics like random_state that should be in all models
    if 'random_state' not in default_params:
        default_params['random_state'] = config["experiment_constants"]["random_seed"]
        
    base_model = create_model(model_name, default_params)
    print(f"DEBUG: Base model created: {type(base_model)}")
    
    # Create search space using skopt dimensions
    search_space = create_search_space(tunable_params)
    print(f"DEBUG: Search space created: {search_space}")
    
    # Configure BayesSearchCV
    tuning_iterations = config["experiment_constants"]["tuning_iterations"]
    tuning_cv_folds = config["experiment_constants"]["tuning_cv_folds"]
    
    print(f"DEBUG: Configuring BayesSearchCV with {tuning_iterations} iterations and {tuning_cv_folds} CV folds")
    
    try:
        # Configure BayesSearchCV with appropriate parameters
        optimizer = BayesSearchCV(
            base_model,
            search_space,
            n_iter=tuning_iterations,
            cv=tuning_cv_folds,
            scoring=custom_scorer,
            random_state=config["experiment_constants"]["random_seed"],
            n_jobs=-1,  # Use all available cores
            verbose=1   # Show progress
        )
        
        print(f"DEBUG: BayesSearchCV configured successfully")
        
        # Fit the optimizer
        print(f"DEBUG: Starting optimizer fitting with X shape {X_train.shape} and y shape {y_train.shape}")
        
        if model_name == "Weighted Random Forest" and sample_weight is not None:
            print(f"DEBUG: Fitting with sample weights, shape: {sample_weight.shape if hasattr(sample_weight, 'shape') else len(sample_weight)}")
            optimizer.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            print(f"DEBUG: Fitting without sample weights")
            optimizer.fit(X_train, y_train)
        
        print(f"DEBUG: Optimizer fitting completed")
        print(f"DEBUG: Best parameters: {optimizer.best_params_}")
        print(f"DEBUG: Best estimator type: {type(optimizer.best_estimator_)}")
        
        return optimizer.best_params_, optimizer.best_estimator_
    except Exception as e:
        print(f"DEBUG: Error during tuning: {str(e)}")
        # Return default parameters and a basic model if tuning fails
        print(f"DEBUG: Falling back to default model without tuning")
        model = create_model(model_name, default_params)
        if model_name == "Weighted Random Forest" and sample_weight is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        return default_params, model

# %% Model saving and loading

def save_model(model: Any, 
              experiment_id: str, 
              config_name: str, 
              iteration: int,
              config: Dict) -> str:
    """
    Save trained model to disk
    
    Args:
        model: Trained model instance
        experiment_id: ID of the experiment
        config_name: Name of the configuration
        iteration: Iteration number
        config: Experiment configuration with data paths
        
    Returns:
        Path to the saved model
    """
    # Create directory if it doesn't exist
    output_data_path = config["data_paths"]["output_data"]
    save_dir = f"{output_data_path}/{experiment_id}/models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename
    filename = f"{save_dir}/{config_name}_iter{iteration}.joblib"
    
    # Save model
    joblib.dump(model, filename)
    
    return filename

def load_model(model_path: str) -> Any:
    """
    Load trained model from disk
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model instance
    """
    return joblib.load(model_path)

# %% Testing function

def test_search_space_creation(config_file: str = "experiment_config.json") -> Dict:
    """
    Test function to verify search space creation
    
    Args:
        config_file: Path to experiment configuration file
        
    Returns:
        Dictionary with search spaces for each model
    """
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Test search space creation for each model
    result = {}
    for model_name in config["experiment_variables"]["models"]:
        tunable_params = config["model_parameters"][model_name]["tunable_parameters"]
        search_space = create_search_space(tunable_params)
        result[model_name] = search_space
        
        print(f"\nSearch space for {model_name}:")
        for param_name, param_space in search_space.items():
            print(f"  {param_name}: {type(param_space).__name__} - {param_space}")
    
    return result

# Uncomment to test
# if __name__ == "__main__":
#     test_search_space_creation() 