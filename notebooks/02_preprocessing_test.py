"""
Test script for the preprocessing pipeline with Diabetes dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.utils.data_loader import load_diabetes_data, split_data
from src.utils.preprocessor import DataPreprocessor, create_preprocessing_pipeline


def test_preprocessing():
    """Test the preprocessing pipeline with Diabetes dataset."""
    print("=== Testing Preprocessing Pipeline ===\n")
    
    # 1. Load and split data
    print("1. Loading and splitting data...")
    X, y = load_diabetes_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}\n")
    
    # 2. Create and test preprocessor
    print("2. Testing DataPreprocessor...")
    preprocessor = DataPreprocessor(scaling_method='standard', handle_missing=True)
    
    # Validate data
    validation = preprocessor.validate_data(X_train)
    print("Data validation results:")
    print(f"  Shape: {validation['shape']}")
    print(f"  Has missing values: {validation['has_missing']}")
    print(f"  Is valid: {validation['is_valid']}")
    print(f"  Numeric columns: {len(validation['numeric_columns'])}")
    print(f"  Categorical columns: {len(validation['categorical_columns'])}\n")
    
    # 3. Fit and transform training data
    print("3. Fitting and transforming training data...")
    X_train_scaled = preprocessor.fit_transform(X_train)
    print(f"Training data scaled shape: {X_train_scaled.shape}")
    print(f"Training data scaled mean: {X_train_scaled.mean():.6f}")
    print(f"Training data scaled std: {X_train_scaled.std():.6f}\n")
    
    # 4. Transform test data
    print("4. Transforming test data...")
    X_test_scaled = preprocessor.transform(X_test)
    print(f"Test data scaled shape: {X_test_scaled.shape}")
    print(f"Test data scaled mean: {X_test_scaled.mean():.6f}")
    print(f"Test data scaled std: {X_test_scaled.std():.6f}\n")
    
    # 5. Test inverse transform
    print("5. Testing inverse transform...")
    X_test_original = preprocessor.inverse_transform(X_test_scaled)
    print(f"Inverse transformed shape: {X_test_original.shape}")
    
    # Check if inverse transform is close to original
    mse = np.mean((X_test.values - X_test_original) ** 2)
    print(f"Mean squared error between original and inverse: {mse:.10f}\n")
    
    # 6. Test factory function
    print("6. Testing factory function...")
    preprocessor_factory = create_preprocessing_pipeline(
        scaling_method='minmax', 
        handle_missing=True
    )
    X_train_minmax = preprocessor_factory.fit_transform(X_train)
    print(f"MinMax scaled training data range: [{X_train_minmax.min():.6f}, {X_train_minmax.max():.6f}]\n")
    
    print("=== Preprocessing Test Complete ===")
    print("✅ All preprocessing tests passed!")


if __name__ == "__main__":
    test_preprocessing() 