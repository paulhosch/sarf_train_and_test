#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt

# %% Performance metrics

def compute_metrics(y_true: np.ndarray, 
                   y_pred: np.ndarray) -> Dict:
    """
    Compute performance metrics for flood mapping prediction
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with performance metrics
    """
    # Calculate overall accuracy
    oa = accuracy_score(y_true, y_pred)
    
    # Calculate F1 score (binary classification)
    f1 = f1_score(y_true, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        "OA": oa,
        "F1": f1,
        "Confusion_Matrix": cm
    }

def compute_all_metrics(y_true: np.ndarray, 
                       y_pred: np.ndarray, 
                       proportional_weights: Optional[np.ndarray] = None,
                       balanced_weights: Optional[np.ndarray] = None) -> Dict:
    """
    Compute all required metrics (proportional and balanced)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        proportional_weights: Weights for proportional sampling
        balanced_weights: Weights for balanced sampling
        
    Returns:
        Dictionary with all metrics
    """
    # Compute proportional metrics
    proportional_metrics = compute_metrics(y_true, y_pred, proportional_weights)
    
    # Compute balanced metrics
    balanced_metrics = compute_metrics(y_true, y_pred, balanced_weights)
    
    # Combine into a single dictionary with prefixes
    metrics = {
        "proportional_OA": proportional_metrics["OA"],
        "proportional_F1": proportional_metrics["F1"],
        "proportional_Confusion_Matrix": proportional_metrics["Confusion_Matrix"],
        "balanced_OA": balanced_metrics["OA"],
        "balanced_F1": balanced_metrics["F1"],
        "balanced_Confusion_Matrix": balanced_metrics["Confusion_Matrix"]
    }
    
    return metrics

# %% Feature importance

def mdi_feature_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
    """
    Calculate Mean Decrease in Impurity (MDI) feature importance
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    print(f"DEBUG: Inside MDI calculation, model type: {type(model)}")
    print(f"DEBUG: Feature names: {feature_names[:5]}...{len(feature_names)} total")
    
    try:
        importances = model.feature_importances_
        print(f"DEBUG: Feature importances shape/length: {importances.shape if hasattr(importances, 'shape') else len(importances)}")
        
        # Normalize to [0,1] scale
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        # Create dictionary mapping feature names to importance values
        importance_dict = dict(zip(feature_names, importances))
        
        return importance_dict
    except Exception as e:
        print(f"DEBUG: MDI calculation error: {str(e)}")
        raise

def mda_feature_importance(model: Any, 
                          X: pd.DataFrame, 
                          y: pd.Series,
                          random_seed: int = 42) -> Dict[str, float]:
    """
    Calculate Mean Decrease in Accuracy (MDA) via permutation importance
    
    Args:
        model: Trained model
        X: Feature data
        y: Target labels
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    print(f"DEBUG: Inside MDA calculation, model type: {type(model)}")
    print(f"DEBUG: X shape: {X.shape}, sample: {X.iloc[0]}")
    
    try:
        # Calculate permutation importance
        result = permutation_importance(
            model, X, y, n_repeats=10, random_state=random_seed
        )
        
        # Get mean importance scores
        importances = result.importances_mean
        
        # Normalize to [0,1] scale
        importances_sum = importances.sum()
        if importances_sum > 0:
            importances = importances / importances_sum
        
        # Create dictionary mapping feature names to importance values
        importance_dict = dict(zip(X.columns, importances))
        
        return importance_dict
    except Exception as e:
        print(f"DEBUG: MDA calculation error: {str(e)}")
        raise

def shap_feature_importance(model: Any, 
                          X: pd.DataFrame, 
                          n_samples: int = 100) -> Dict[str, float]:
    """
    Calculate SHAP feature importance
    
    Args:
        model: Trained model
        X: Feature data (can be a sample)
        n_samples: Number of samples to use
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    print(f"DEBUG: Inside SHAP calculation, model type: {type(model)}")
    print(f"DEBUG: X shape: {X.shape}")
    
    try:
        # If X is large, take a sample
        if len(X) > n_samples:
            X_sample = X.sample(n_samples, random_state=42)
        else:
            X_sample = X
        
        # Create explainer
        print(f"DEBUG: Creating TreeExplainer")
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        print(f"DEBUG: Calculating SHAP values")
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, shap_values is a list with two arrays
        # We use the positive class (index 1) for importance
        print(f"DEBUG: SHAP values type: {type(shap_values)}")
        if isinstance(shap_values, list):
            print(f"DEBUG: SHAP values is a list of length {len(shap_values)}")
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP values for each feature
        importances = np.abs(shap_values).mean(axis=0)
        
        # Normalize to [0,1] scale
        importances_sum = importances.sum()
        if importances_sum > 0:
            importances = importances / importances_sum
        
        # Create dictionary mapping feature names to importance values
        importance_dict = dict(zip(X_sample.columns, importances))
        
        return importance_dict
    except Exception as e:
        print(f"DEBUG: SHAP calculation error: {str(e)}")
        raise

def compute_all_feature_importances(model: Any, 
                                  X: pd.DataFrame, 
                                  y: pd.Series,
                                  methods: List[str],
                                  random_seed: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Compute feature importance using multiple methods
    
    Args:
        model: Trained model
        X: Feature data
        y: Target labels
        methods: List of methods to use ('mdi', 'mda', 'shap_importance')
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping method names to feature importance dictionaries
    """
    print(f"DEBUG: Computing feature importances with methods: {methods}")
    print(f"DEBUG: Model type: {type(model)}")
    print(f"DEBUG: X shape: {X.shape}, columns: {X.columns.tolist()}")
    
    importances = {}
    
    # Compute importance with each method
    if 'mdi' in methods:
        try:
            print(f"DEBUG: Computing MDI importance")
            importances['mdi'] = mdi_feature_importance(model, X.columns.tolist())
            print(f"DEBUG: MDI importance completed successfully")
        except Exception as e:
            print(f"DEBUG: Error in MDI calculation: {str(e)}")
            importances['mdi'] = {"error": str(e)}
    
    if 'mda' in methods:
        try:
            print(f"DEBUG: Computing MDA importance")
            importances['mda'] = mda_feature_importance(model, X, y, random_seed)
            print(f"DEBUG: MDA importance completed successfully")
        except Exception as e:
            print(f"DEBUG: Error in MDA calculation: {str(e)}")
            importances['mda'] = {"error": str(e)}
    
    if 'shap_importance' in methods:
        try:
            print(f"DEBUG: Computing SHAP importance")
            importances['shap_importance'] = shap_feature_importance(model, X)
            print(f"DEBUG: SHAP importance completed successfully")
        except Exception as e:
            print(f"DEBUG: Error in SHAP calculation: {str(e)}")
            importances['shap_importance'] = {"error": str(e)}
    
    return importances 