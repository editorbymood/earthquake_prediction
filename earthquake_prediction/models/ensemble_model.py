"""
Ensemble model for earthquake magnitude prediction using RandomForest + XGBoost stacking
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
from config import ENSEMBLE_CONFIG


class EarthquakeEnsembleModel:
    """
    Stacked ensemble model combining RandomForest and XGBoost for earthquake magnitude prediction
    """
    
    def __init__(self, config=None):
        """
        Initialize the ensemble model
        
        Args:
            config (dict): Model configuration parameters
        """
        self.config = config or ENSEMBLE_CONFIG
        
        # Base models
        self.rf_model = None
        self.xgb_model = None
        self.meta_model = None
        
        # Model performance tracking
        self.is_fitted = False
        self.feature_names = None
        self.training_scores = {}
        
    def _initialize_models(self):
        """Initialize base models and meta-learner"""
        # RandomForest model
        self.rf_model = RandomForestRegressor(**self.config['random_forest'])
        
        # XGBoost model
        self.xgb_model = xgb.XGBRegressor(**self.config['xgboost'])
        
        # Meta-learner (Ridge regression)
        self.meta_model = Ridge(**self.config['meta_learner'])
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the stacked ensemble model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target values
            X_val (pd.DataFrame, optional): Validation features
            y_val (pd.Series, optional): Validation target values
            
        Returns:
            dict: Training results and metrics
        """
        print("Training Ensemble Model (RandomForest + XGBoost + Stacking)...")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Initialize models
        self._initialize_models()
        
        # Step 1: Generate base predictions using cross-validation
        print("Generating base model predictions...")
        
        # RandomForest predictions
        rf_pred_train = cross_val_predict(
            self.rf_model, X_train, y_train, cv=5, n_jobs=-1
        )
        
        # XGBoost predictions  
        xgb_pred_train = cross_val_predict(
            self.xgb_model, X_train, y_train, cv=5, n_jobs=-1
        )
        
        # Step 2: Create meta-features
        meta_features_train = np.column_stack((rf_pred_train, xgb_pred_train))
        
        # Step 3: Train base models on full training data
        print("Training base models...")
        self.rf_model.fit(X_train, y_train)
        self.xgb_model.fit(X_train, y_train)
        
        # Step 4: Train meta-learner
        print("Training meta-learner...")
        self.meta_model.fit(meta_features_train, y_train)
        
        # Calculate training metrics
        train_pred = self.predict(X_train)
        self.training_scores['train_mse'] = mean_squared_error(y_train, train_pred)
        self.training_scores['train_mae'] = mean_absolute_error(y_train, train_pred)
        self.training_scores['train_r2'] = r2_score(y_train, train_pred)
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            self.training_scores['val_mse'] = mean_squared_error(y_val, val_pred)
            self.training_scores['val_mae'] = mean_absolute_error(y_val, val_pred)
            self.training_scores['val_r2'] = r2_score(y_val, val_pred)
        
        self.is_fitted = True
        
        print(f"Training completed!")
        print(f"Train MSE: {self.training_scores['train_mse']:.4f}")
        print(f"Train MAE: {self.training_scores['train_mae']:.4f}")
        print(f"Train R²: {self.training_scores['train_r2']:.4f}")
        
        if 'val_mse' in self.training_scores:
            print(f"Val MSE: {self.training_scores['val_mse']:.4f}")
            print(f"Val MAE: {self.training_scores['val_mae']:.4f}")
            print(f"Val R²: {self.training_scores['val_r2']:.4f}")
        
        return self.training_scores
    
    def predict(self, X):
        """
        Make predictions using the stacked ensemble
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.ndarray: Predicted earthquake magnitudes
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get base model predictions
        rf_pred = self.rf_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        
        # Create meta-features
        meta_features = np.column_stack((rf_pred, xgb_pred))
        
        # Get final predictions from meta-learner
        final_pred = self.meta_model.predict(meta_features)
        
        return final_pred
    
    def get_feature_importance(self):
        """
        Get feature importance from base models
        
        Returns:
            dict: Feature importance from different models
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # RandomForest feature importance
        rf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'rf_importance': self.rf_model.feature_importances_
        })
        
        # XGBoost feature importance
        xgb_importance = pd.DataFrame({
            'feature': self.feature_names,
            'xgb_importance': self.xgb_model.feature_importances_
        })
        
        # Combine importances
        importance_df = rf_importance.merge(xgb_importance, on='feature')
        importance_df['avg_importance'] = (
            importance_df['rf_importance'] + importance_df['xgb_importance']
        ) / 2
        
        importance_df = importance_df.sort_values('avg_importance', ascending=False)
        
        return {
            'combined_importance': importance_df,
            'rf_importance': rf_importance.sort_values('rf_importance', ascending=False),
            'xgb_importance': xgb_importance.sort_values('xgb_importance', ascending=False),
            'meta_coef': self.meta_model.coef_ if hasattr(self.meta_model, 'coef_') else None
        }
    
    def get_base_predictions(self, X):
        """
        Get predictions from individual base models
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            dict: Predictions from each base model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return {
            'rf_predictions': self.rf_model.predict(X),
            'xgb_predictions': self.xgb_model.predict(X),
            'ensemble_predictions': self.predict(X)
        }
    
    def save_model(self, filepath):
        """
        Save the trained ensemble model
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'meta_model': self.meta_model,
            'feature_names': self.feature_names,
            'training_scores': self.training_scores,
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a pre-trained ensemble model
        
        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.rf_model = model_data['rf_model']
        self.xgb_model = model_data['xgb_model']
        self.meta_model = model_data['meta_model']
        self.feature_names = model_data['feature_names']
        self.training_scores = model_data['training_scores']
        self.config = model_data['config']
        self.is_fitted = True
        
        print(f"Model loaded from {filepath}")
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation on the ensemble model
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target values
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        from sklearn.model_selection import cross_val_score
        
        # Initialize models
        self._initialize_models()
        
        # Create a pipeline that mimics the stacking process
        # Note: This is a simplified cross-validation approach
        # In practice, you might want to implement proper nested CV
        
        cv_results = {}
        
        # RandomForest CV
        rf_scores = cross_val_score(
            self.rf_model, X, y, cv=cv, scoring='neg_mean_squared_error'
        )
        cv_results['rf_mse'] = -rf_scores
        cv_results['rf_mse_mean'] = -rf_scores.mean()
        cv_results['rf_mse_std'] = rf_scores.std()
        
        # XGBoost CV
        xgb_scores = cross_val_score(
            self.xgb_model, X, y, cv=cv, scoring='neg_mean_squared_error'
        )
        cv_results['xgb_mse'] = -xgb_scores
        cv_results['xgb_mse_mean'] = -xgb_scores.mean()
        cv_results['xgb_mse_std'] = xgb_scores.std()
        
        return cv_results


def create_ensemble_model(config=None):
    """
    Factory function to create an ensemble model
    
    Args:
        config (dict): Model configuration
        
    Returns:
        EarthquakeEnsembleModel: Configured ensemble model
    """
    return EarthquakeEnsembleModel(config)
