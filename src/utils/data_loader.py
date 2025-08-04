"""
Data loading utilities for the Diabetes dataset.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any


def load_diabetes_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Diabetes dataset from sklearn.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target variables
    """
    diabetes = load_diabetes()
    
    # Create DataFrame with feature names
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    target = pd.Series(diabetes.target, name='target')
    
    return df, target


def get_dataset_info() -> Dict[str, Any]:
    """
    Get information about the Diabetes dataset.
    
    Returns:
        Dict containing dataset information
    """
    diabetes = load_diabetes()
    
    return {
        'name': 'Diabetes Dataset',
        'description': diabetes.DESCR,
        'n_samples': diabetes.data.shape[0],
        'n_features': diabetes.data.shape[1],
        'feature_names': list(diabetes.feature_names),
        'target_name': 'disease progression',
        'task_type': 'regression'
    }


def split_data(X: pd.DataFrame, y: pd.Series, 
               test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state) 