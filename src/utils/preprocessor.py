"""
Data preprocessing utilities for the ML workflow.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for ML workflows.
    """
    
    def __init__(self, scaling_method: str = 'standard', 
                 handle_missing: bool = True,
                 random_state: int = 42):
        """
        Initialize the preprocessor.
        
        Args:
            scaling_method: 'standard' or 'minmax'
            handle_missing: Whether to handle missing values
            random_state: Random seed for reproducibility
        """
        self.scaling_method = scaling_method
        self.handle_missing = handle_missing
        self.random_state = random_state
        
        # Initialize transformers
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaling_method must be 'standard' or 'minmax'")
            
        self.imputer = SimpleImputer(strategy='mean') if handle_missing else None
        self.is_fitted = False
        
    def validate_data(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data and return statistics.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'shape': X.shape,
            'missing_values': X.isnull().sum().to_dict(),
            'data_types': X.dtypes.to_dict(),
            'numeric_columns': X.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': X.select_dtypes(include=['object']).columns.tolist(),
            'has_missing': X.isnull().any().any(),
            'is_valid': True
        }
        
        # Check for issues
        if validation_results['has_missing'] and not self.handle_missing:
            logger.warning("Data contains missing values but handle_missing=False")
            validation_results['is_valid'] = False
            
        if len(validation_results['categorical_columns']) > 0:
            logger.warning("Data contains categorical columns - consider encoding")
            
        return validation_results
    
    def fit(self, X: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Training features DataFrame
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting preprocessor...")
        
        # Validate data
        validation = self.validate_data(X)
        if not validation['is_valid']:
            raise ValueError("Data validation failed")
        
        # Handle missing values if needed
        if self.handle_missing and validation['has_missing']:
            logger.info("Handling missing values...")
            X_imputed = self.imputer.fit_transform(X)
        else:
            X_imputed = X.values
            
        # Fit scaler
        logger.info(f"Fitting {self.scaling_method} scaler...")
        self.scaler.fit(X_imputed)
        
        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Features DataFrame to transform
            
        Returns:
            Transformed features as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming")
            
        logger.info("Transforming data...")
        
        # Handle missing values if needed
        if self.handle_missing and X.isnull().any().any():
            X_imputed = self.imputer.transform(X)
        else:
            X_imputed = X.values
            
        # Scale features
        X_scaled = self.scaler.transform(X_imputed)
        
        logger.info("Data transformation completed")
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Transformed features as numpy array
        """
        return self.fit(X).transform(X)
    
    def get_feature_names(self) -> list:
        """
        Get feature names after preprocessing.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        return [f"feature_{i}" for i in range(self.scaler.n_features_in_)]
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            X: Scaled features array
            
        Returns:
            Features in original scale
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
            
        return self.scaler.inverse_transform(X)


def create_preprocessing_pipeline(scaling_method: str = 'standard',
                                handle_missing: bool = True) -> DataPreprocessor:
    """
    Factory function to create a preprocessing pipeline.
    
    Args:
        scaling_method: Scaling method to use
        handle_missing: Whether to handle missing values
        
    Returns:
        Configured DataPreprocessor instance
    """
    return DataPreprocessor(
        scaling_method=scaling_method,
        handle_missing=handle_missing
    ) 