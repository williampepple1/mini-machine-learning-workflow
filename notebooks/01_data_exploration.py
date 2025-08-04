"""
Data exploration script for the Diabetes dataset.
This script provides initial analysis and visualization of the dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.data_loader import load_diabetes_data, get_dataset_info, split_data


def main():
    """Main function to run the data exploration."""
    print("=== Diabetes Dataset Exploration ===\n")
    
    # 1. Load the dataset
    print("1. Loading dataset...")
    X, y = load_diabetes_data()
    
    print(f"Dataset shape:")
    print(f"Features: {X.shape}")
    print(f"Target: {y.shape}")
    print(f"Feature names: {X.columns.tolist()}\n")
    
    # 2. Get dataset information
    print("2. Dataset information:")
    info = get_dataset_info()
    print(f"Dataset: {info['name']}")
    print(f"Task: {info['task_type']}")
    print(f"Samples: {info['n_samples']}")
    print(f"Features: {info['n_features']}\n")
    
    # 3. Basic statistics
    print("3. Feature statistics:")
    print(X.describe())
    print(f"\nTarget statistics:")
    print(y.describe())
    print()
    
    # 4. Data split
    print("4. Train-test split:")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training target range: {y_train.min():.2f} - {y_train.max():.2f}")
    print(f"Test target range: {y_test.min():.2f} - {y_test.max():.2f}\n")
    
    # 5. Correlation analysis
    print("5. Top correlations with target:")
    df_with_target = X.copy()
    df_with_target['target'] = y
    correlations = df_with_target.corr()['target'].sort_values(ascending=False)
    print(correlations)
    
    print("\n=== Exploration Complete ===")


if __name__ == "__main__":
    main() 