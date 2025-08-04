"""
Comprehensive model training script for the Diabetes dataset.
This script demonstrates the complete ML pipeline: data loading → preprocessing → training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.data_loader import load_diabetes_data, split_data
from src.utils.preprocessor import DataPreprocessor
from src.training.model_trainer import ModelTrainer


def train_diabetes_models():
    """Complete ML pipeline for Diabetes dataset."""
    print("=== Diabetes Model Training Pipeline ===\n")
    
    # 1. Load and split data
    print("1. Loading and preparing data...")
    X, y = load_diabetes_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f}\n")
    
    # 2. Preprocess data
    print("2. Preprocessing data...")
    preprocessor = DataPreprocessor(scaling_method='standard', handle_missing=True)
    
    # Validate data
    validation = preprocessor.validate_data(X_train)
    print(f"Data validation: {'✅ Valid' if validation['is_valid'] else '❌ Invalid'}")
    
    # Transform data
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    print(f"Preprocessing complete - Features scaled to mean=0, std=1\n")
    
    # 3. Initialize model trainer
    print("3. Initializing model trainer...")
    trainer = ModelTrainer(models_dir="models", random_state=42)
    
    available_models = trainer.get_available_models()
    print(f"Available models: {list(available_models.keys())}\n")
    
    # 4. Train all models (without grid search for speed)
    print("4. Training all models...")
    results = trainer.train_all_models(
        X_train_scaled, y_train.values,
        X_test_scaled, y_test.values,
        use_grid_search=False,  # Set to True for hyperparameter tuning
        cv_folds=5
    )
    
    print(f"\nTraining complete! Best model: {trainer.best_model}")
    print(f"Best R² score: {trainer.best_score:.4f}\n")
    
    # 5. Display model comparison
    print("5. Model Performance Comparison:")
    summary = trainer.get_model_summary()
    print(summary.to_string(index=False))
    print()
    
    # 6. Save the best model
    print("6. Saving best model...")
    best_model_path = trainer.save_best_model("diabetes_best_model.joblib")
    print(f"Best model saved to: {best_model_path}\n")
    
    # 7. Make predictions with best model
    print("7. Making predictions with best model...")
    y_pred_best = trainer.predict_best(X_test_scaled)
    
    # Calculate final metrics
    from sklearn.metrics import mean_squared_error, r2_score
    final_mse = mean_squared_error(y_test, y_pred_best)
    final_r2 = r2_score(y_test, y_pred_best)
    
    print(f"Final Test MSE: {final_mse:.4f}")
    print(f"Final Test R²: {final_r2:.4f}")
    
    # 8. Feature importance (for tree-based models)
    print("\n8. Feature importance analysis...")
    best_model = trainer.models[trainer.best_model]
    
    if hasattr(best_model, 'feature_importances_'):
        feature_names = X_train.columns
        importances = best_model.feature_importances_
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("Top 5 most important features:")
        print(importance_df.head().to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances)
        plt.xticks(range(len(importances)), feature_names, rotation=45)
        plt.title(f'Feature Importance - {trainer.best_model.replace("_", " ").title()}')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Feature importance plot saved to models/feature_importance.png")
    
    elif hasattr(best_model, 'coef_'):
        feature_names = X_train.columns
        coefficients = best_model.coef_
        
        # Create coefficient DataFrame
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        print("Top 5 features by absolute coefficient:")
        print(coef_df.head().to_string(index=False))
    
    print("\n=== Training Pipeline Complete ===")
    print("✅ All models trained and evaluated successfully!")
    
    return trainer, preprocessor, results


def compare_models_visualization(trainer):
    """Create visualization comparing model performances."""
    print("\n=== Creating Model Comparison Visualization ===")
    
    summary = trainer.get_model_summary()
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # R² scores
    ax1.bar(summary['Model'], summary['Test R²'])
    ax1.set_title('Test R² Scores')
    ax1.set_ylabel('R² Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # MSE scores
    ax2.bar(summary['Model'], summary['Test MSE'])
    ax2.set_title('Test MSE Scores')
    ax2.set_ylabel('MSE')
    ax2.tick_params(axis='x', rotation=45)
    
    # MAE scores
    ax3.bar(summary['Model'], summary['Test MAE'])
    ax3.set_title('Test MAE Scores')
    ax3.set_ylabel('MAE')
    ax3.tick_params(axis='x', rotation=45)
    
    # Cross-validation MSE
    ax4.bar(summary['Model'], summary['CV MSE'])
    ax4.set_title('Cross-Validation MSE')
    ax4.set_ylabel('CV MSE')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Model comparison plot saved to models/model_comparison.png")


if __name__ == "__main__":
    # Run the complete training pipeline
    trainer, preprocessor, results = train_diabetes_models()
    
    # Create visualizations
    compare_models_visualization(trainer)
    
    print("\n🎉 Complete ML pipeline executed successfully!")
    print("📁 Check the 'models/' directory for saved models and visualizations") 