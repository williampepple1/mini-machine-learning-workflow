"""
Model training pipeline for the ML workflow.
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A comprehensive model training class for ML workflows.
    """
    
    def __init__(self, models_dir: str = "models", random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            models_dir: Directory to save trained models
            random_state: Random seed for reproducibility
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = float('-inf')
        
    def get_available_models(self) -> Dict[str, BaseEstimator]:
        """
        Get dictionary of available models for regression.
        
        Returns:
            Dictionary of model names and their instances
        """
        return {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=self.random_state),
            'lasso': Lasso(random_state=self.random_state),
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=self.random_state
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=self.random_state
            ),
            'svr': SVR(kernel='rbf')
        }
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict[str, List]]:
        """
        Get hyperparameter grids for grid search.
        
        Returns:
            Dictionary of hyperparameter grids for each model
        """
        return {
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'svr': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        }
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   use_grid_search: bool = False, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train a single model and evaluate its performance.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            use_grid_search: Whether to use grid search for hyperparameter tuning
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {model_name}...")
        
        available_models = self.get_available_models()
        if model_name not in available_models:
            raise ValueError(f"Model {model_name} not available")
        
        model = available_models[model_name]
        
        # Perform grid search if requested
        if use_grid_search and model_name in self.get_hyperparameter_grids():
            logger.info(f"Performing grid search for {model_name}...")
            param_grid = self.get_hyperparameter_grids()[model_name]
            grid_search = GridSearchCV(
                model, param_grid, cv=cv_folds, scoring='neg_mean_squared_error',
                n_jobs=-1, random_state=self.random_state
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            logger.info(f"Best parameters: {best_params}")
        else:
            best_params = {}
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv_folds, 
            scoring='neg_mean_squared_error'
        )
        cv_mse = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Store results
        results = {
            'model_name': model_name,
            'model': model,
            'best_params': best_params,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mse': cv_mse,
            'cv_std': cv_std,
            'training_time': datetime.now().isoformat()
        }
        
        self.models[model_name] = model
        self.results[model_name] = results
        
        # Update best model
        if test_r2 > self.best_score:
            self.best_score = test_r2
            self.best_model = model_name
        
        logger.info(f"{model_name} - Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}")
        
        return results
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        use_grid_search: bool = False, cv_folds: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Train all available models and compare their performance.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            use_grid_search: Whether to use grid search for hyperparameter tuning
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with results for all models
        """
        logger.info("Training all available models...")
        
        available_models = self.get_available_models()
        
        for model_name in available_models.keys():
            try:
                self.train_model(
                    model_name, X_train, y_train, X_test, y_test,
                    use_grid_search, cv_folds
                )
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        logger.info(f"Training complete. Best model: {self.best_model} (R²: {self.best_score:.4f})")
        
        return self.results
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get a summary of all trained models.
        
        Returns:
            DataFrame with model performance summary
        """
        if not self.results:
            raise ValueError("No models have been trained yet")
        
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Test R²': results['test_r2'],
                'Test MSE': results['test_mse'],
                'Test MAE': results['test_mae'],
                'CV MSE': results['cv_mse'],
                'CV Std': results['cv_std'],
                'Best Model': model_name == self.best_model
            })
        
        return pd.DataFrame(summary_data).sort_values('Test R²', ascending=False)
    
    def save_model(self, model_name: str, filename: Optional[str] = None) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filename: Optional custom filename
            
        Returns:
            Path to saved model file
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}.joblib"
        
        model_path = self.models_dir / filename
        joblib.dump(self.models[model_name], model_path)
        
        # Save model metadata
        metadata_path = model_path.with_suffix('.json')
        metadata = {
            'model_name': model_name,
            'training_results': self.results[model_name],
            'saved_at': datetime.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    def save_best_model(self, filename: Optional[str] = None) -> str:
        """
        Save the best performing model.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved model file
        """
        if self.best_model is None:
            raise ValueError("No models have been trained yet")
        
        return self.save_model(self.best_model, filename)
    
    def load_model(self, model_path: str) -> BaseEstimator:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Loaded model
        """
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model to use
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict(X)
    
    def predict_best(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the best performing model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if self.best_model is None:
            raise ValueError("No models have been trained yet")
        
        return self.predict(self.best_model, X) 