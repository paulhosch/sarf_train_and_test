#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
import json
from sklearn.preprocessing import OneHotEncoder

# %% Data loading functions

def load_config(config_path: str = "experiment_config.json") -> Dict:
    """
    Load experiment configuration from a JSON file
    
    Args:
        config_path: Path to the config file
        
    Returns:
        Dict containing the experiment configuration
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_sample_path(case_study: str, 
                   dataset_type: str, 
                   sample_size: int, 
                   strategy: str,
                   config: Dict,
                   new_samples_each_iteration: bool = False,
                   iteration_id: Optional[str] = None) -> str:
    """
    Generate the path to a specific sample dataset, supporting per-iteration samples if requested.
    """
    input_data_path = config["data_paths"]["input_data"]
    if new_samples_each_iteration and iteration_id is not None:
        # New logic: per-iteration sample sets
        return f"{input_data_path}/{case_study}/samples/{iteration_id}/{dataset_type}/{sample_size}/samples/{strategy}.csv"
    else:
        # Old logic
        return f"{input_data_path}/{case_study}/samples/{dataset_type}/{sample_size}/samples/{strategy}.csv"

def load_samples(case_study: str, 
                dataset_type: str, 
                sample_size: int, 
                strategy: str,
                config: Dict,
                new_samples_each_iteration: bool = False,
                iteration_id: Optional[str] = None) -> pd.DataFrame:
    """
    Load a specific sample dataset, supporting per-iteration samples if requested.
    """
    path = get_sample_path(case_study, dataset_type, sample_size, strategy, config, new_samples_each_iteration, iteration_id)
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Sample file not found: {path}")

# %% Feature selection and preprocessing

def get_feature_columns(config: Dict) -> Tuple[List[str], List[str]]:
    """
    Get the lists of categorical and continuous feature columns from the config
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        Tuple of (categorical_features, continuous_features)
    """
    if "categorical_features" in config["experiment_constants"] and "continuous_features" in config["experiment_constants"]:
        categorical_features = config["experiment_constants"]["categorical_features"]
        continuous_features = config["experiment_constants"]["continuous_features"]
    else:
        # For backward compatibility
        categorical_features = ["LAND_COVER"]  # Default categorical
        all_features = config["experiment_constants"]["sar_features"] + config["experiment_constants"]["contextual_features"]
        continuous_features = [f for f in all_features if f not in categorical_features]
    
    return categorical_features, continuous_features

def sort_categorical_features(df: pd.DataFrame, 
                             categorical_features: List[str],
                             label_col: str) -> pd.DataFrame:
    """
    Sort categorical features based on flood occurrence proportion
    
    Args:
        df: DataFrame containing the dataset
        categorical_features: List of categorical feature column names
        label_col: Name of the label column
        
    Returns:
        DataFrame with transformed categorical features
    """
    df_transformed = df.copy()
    
    for col in categorical_features:
        if col in df.columns:
            # Calculate the proportion of flood for each category
            flood_prop = df.groupby(col)[label_col].mean().sort_values()
            
            # Create mapping dictionary from category to order
            cat_mapping = {cat: order for order, cat in enumerate(flood_prop.index)}
            
            # Apply the mapping
            df_transformed[col] = df[col].map(cat_mapping)
            
            # Print the mapping for reference
            print(f"Transformed {col} categories based on flood proportion:")
            for cat, order in cat_mapping.items():
                prop = flood_prop[cat]
                print(f"  Category: {cat}, Flood Proportion: {prop:.4f}, New Value: {order}")
    
    return df_transformed

def prepare_datasets_for_logo_cv(config: Dict) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Prepare datasets for Leave-One-Group-Out Cross-Validation
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        Dictionary with datasets organized by case study and dataset type
    """
    case_studies = config["experiment_constants"]["case_studies"]
    training_strategies = config["experiment_variables"]["training_sample_strategies"]
    training_sizes = config["experiment_variables"]["training_sample_sizes"]
    testing_strategies = config["experiment_constants"]["testing_sample_strategies"]
    testing_sizes = config["experiment_constants"]["testing_sample_sizes"]
    
    # Initialize datasets dictionary
    datasets = {}
    
    # Load training datasets
    for case_study in case_studies:
        datasets[case_study] = {"training": {}, "testing": {}}
        
        # Load training datasets with different strategies and sizes
        for strategy in training_strategies:
            for size in training_sizes:
                try:
                    key = f"{strategy}_{size}"
                    datasets[case_study]["training"][key] = load_samples(
                        case_study, "training", size, strategy, config
                    )
                except FileNotFoundError as e:
                    print(f"Warning: {e}")
        
        # Load testing datasets
        for strategy in testing_strategies:
            for size in testing_sizes:
                try:
                    key = f"{strategy}_{size}"
                    datasets[case_study]["testing"][key] = load_samples(
                        case_study, "testing", size, strategy, config
                    )
                except FileNotFoundError as e:
                    print(f"Warning: {e}")
    
    return datasets

def prepare_combined_training_data(training_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine training data from multiple case studies
    
    Args:
        training_data_dict: Dictionary with DataFrames from different case studies
        
    Returns:
        Combined DataFrame for training
    """
    return pd.concat(list(training_data_dict.values()), ignore_index=True)

def clean_and_convert_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean data by removing NaN values and convert to float32
    
    Args:
        X: Feature matrix
        y: Label vector
        
    Returns:
        Tuple of cleaned (X, y) with features and labels as float32 and no NaN values
    """
    # Record original sample count
    original_count = len(X)
    
    # Remove rows with NaN values
    valid_indices = ~(X.isna().any(axis=1) | y.isna())
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    # Print number of removed samples
    removed_count = original_count - len(X)
    if removed_count > 0:
        print(f"Removed {removed_count} samples due to NaN values")
    
    # Convert continuous features to float32
    # For one-hot encoded columns, keep them as is (they're already binary)
    float_cols = X.select_dtypes(include=['float64', 'int64']).columns
    if not float_cols.empty:
        X.loc[:, float_cols] = X[float_cols].astype(np.float32)
    
    # Convert labels to float32
    y = y.astype(np.float32)
    
    return X, y

def prepare_X_y(data: pd.DataFrame, 
              feature_cols: List[str], 
              label_col: str,
              categorical_cols: Optional[List[str]] = None,
              config: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and label vector y from input data
    
    Args:
        data: Input DataFrame
        feature_cols: List of feature column names
        label_col: Name of label column
        categorical_cols: List of categorical feature columns
        config: Configuration dictionary containing categorical feature class names
        
    Returns:
        Tuple of (X, y) where X is feature matrix and y is label vector
    """
    # Extract features and labels
    X = data[feature_cols].copy()
    y = data[label_col].copy()
    
    # Remove rows where LAND_COVER is 0 (invalid/missing)
    if categorical_cols and 'LAND_COVER' in categorical_cols and 'LAND_COVER' in X.columns:
        invalid_mask = X['LAND_COVER'] == 0
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            print(f"Removed {n_invalid} samples due to invalid LAND_COVER == 0")
            X = X.loc[~invalid_mask]
            y = y.loc[X.index]
    
    # Handle categorical features with one-hot encoding
    if categorical_cols:
        categorical_features_in_X = [col for col in categorical_cols if col in feature_cols]
        if categorical_features_in_X:
            # Separate categorical and continuous features
            X_cat = X[categorical_features_in_X]
            X_cont = X.drop(columns=categorical_features_in_X)
            
            # Prepare categories for OneHotEncoder
            encoder_categories = []
            for feature in categorical_features_in_X:
                if (config and 'experiment_constants' in config 
                    and 'categorical_features_class_names' in config['experiment_constants']
                    and feature in config['experiment_constants']['categorical_features_class_names']):
                    class_mapping = config['experiment_constants']['categorical_features_class_names'][feature]
                    # Use sorted integer keys as categories
                    encoder_categories.append(sorted([int(k) for k in class_mapping.keys()]))
                else:
                    # Use unique values from data if not defined in config
                    encoder_categories.append(sorted(X_cat[feature].dropna().unique()))
            
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=encoder_categories)
            X_cat_encoded = encoder.fit_transform(X_cat)
            
            # Create feature names for encoded columns using predefined class names if available
            encoded_feature_names = []
            for i, feature in enumerate(categorical_features_in_X):
                if (config and 'experiment_constants' in config 
                    and 'categorical_features_class_names' in config['experiment_constants']
                    and feature in config['experiment_constants']['categorical_features_class_names']):
                    class_mapping = config['experiment_constants']['categorical_features_class_names'][feature]
                    data_categories = encoder.categories_[i]
                    feature_names = []
                    for cat in data_categories:
                        cat_str = str(int(cat)) if isinstance(cat, (int, float)) else str(cat)
                        if cat_str in class_mapping:
                            feature_names.append(f"{feature}_{class_mapping[cat_str]}")
                        else:
                            print(f"Warning: Category {cat} in {feature} not found in config mapping")
                            feature_names.append(f"{feature}_{cat}")
                    encoded_feature_names.extend(feature_names)
                    # Validate that we found all expected categories from config
                    expected_categories = set(class_mapping.keys())
                    found_categories = set(str(int(cat)) for cat in data_categories if isinstance(cat, (int, float)))
                    missing_categories = expected_categories - found_categories
                    extra_categories = found_categories - expected_categories
                    if missing_categories:
                        missing_classes = {f"{cat}: {class_mapping[cat]}" for cat in missing_categories}
                        print(f"Warning: Expected but missing {feature} categories: {missing_classes}")
                    if extra_categories:
                        print(f"Warning: Unexpected {feature} categories in data: {extra_categories}")
                else:
                    categories = encoder.categories_[i]
                    encoded_feature_names.extend([f"{feature}_{cat}" for cat in categories])
                    print(f"Warning: No predefined class names found for {feature}. Using values from data: {categories}")
            
            # Convert encoded array to DataFrame with proper column names
            X_cat_encoded_df = pd.DataFrame(
                X_cat_encoded, 
                columns=encoded_feature_names,
                index=X.index
            )
            
            # Combine one-hot encoded categorical features with continuous features
            X = pd.concat([X_cont, X_cat_encoded_df], axis=1)
    
    # Clean data and convert types
    X, y = clean_and_convert_data(X, y)
    
    return X, y 