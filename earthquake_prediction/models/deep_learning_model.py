"""
Deep learning model for earthquake magnitude prediction using TensorFlow/Keras MLP
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from config import DL_CONFIG


class EarthquakeDeepLearningModel:
    """
    Multi-layer perceptron (MLP) model for earthquake magnitude prediction using TensorFlow/Keras
    """
    
    def __init__(self, config=None):
        """
        Initialize the deep learning model
        
        Args:
            config (dict): Model configuration parameters
        """
        self.config = config or DL_CONFIG
        
        # Set random seeds for reproducibility
        tf.random.set_seed(self.config.get('random_state', 42))
        np.random.seed(self.config.get('random_state', 42))
        
        # Model components
        self.model = None
        self.history = None
        self.is_fitted = False
        self.feature_names = None
        self.training_scores = {}
        
    def _build_model(self, input_dim):
        """
        Build the MLP architecture
        
        Args:
            input_dim (int): Number of input features
            
        Returns:
            keras.Model: Compiled model
        """
        # Input layer
        inputs = keras.Input(shape=(input_dim,), name='input_features')
        
        # Hidden layers
        x = inputs
        for i, units in enumerate(self.config['hidden_layers']):
            x = layers.Dense(
                units, 
                activation='relu',
                kernel_initializer='he_normal',
                name=f'dense_layer_{i+1}'
            )(x)
            
            # Batch normalization
            x = layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
            
            # Dropout for regularization
            x = layers.Dropout(
                self.config['dropout_rate'],
                name=f'dropout_{i+1}'
            )(x)
        
        # Output layer for regression
        outputs = layers.Dense(
            1, 
            activation='linear',
            name='magnitude_output'
        )(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='earthquake_mlp')
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _setup_callbacks(self):
        """
        Setup training callbacks
        
        Returns:
            list: List of Keras callbacks
        """
        callback_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config['reduce_lr_patience'],
            min_lr=1e-6,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Model checkpoint (optional)
        checkpoint = callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        callback_list.append(checkpoint)
        
        return callback_list
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the deep learning model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target values
            X_val (pd.DataFrame, optional): Validation features
            y_val (pd.Series, optional): Validation target values
            
        Returns:
            dict: Training results and metrics
        """
        print("Training Deep Learning Model (MLP with TensorFlow/Keras)...")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Convert to numpy arrays
        X_train_array = X_train.values
        y_train_array = y_train.values
        
        # Build model
        input_dim = X_train_array.shape[1]
        self.model = self._build_model(input_dim)
        
        # Print model summary
        print("\\nModel Architecture:")
        self.model.summary()
        
        # Setup callbacks
        callback_list = self._setup_callbacks()
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val.values, y_val.values)
        elif self.config.get('validation_split', 0) > 0:
            # Use validation split from training data
            pass  # Keras will handle this automatically
        
        # Train the model
        print("\\nStarting training...")
        self.history = self.model.fit(
            X_train_array, y_train_array,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=validation_data,
            validation_split=self.config.get('validation_split', 0.2) if validation_data is None else 0,
            callbacks=callback_list,
            verbose=1
        )
        
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
        
        # Print final results
        print(f"\\nTraining completed!")
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
        Make predictions using the trained model
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.ndarray: Predicted earthquake magnitudes
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_array = X.values
        predictions = self.model.predict(X_array, verbose=0)
        
        # Flatten if needed (Keras returns 2D array for single output)
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        
        return predictions
    
    def get_training_history(self):
        """
        Get training history for visualization
        
        Returns:
            dict: Training history data
        """
        if self.history is None:
            return None
        
        return {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history.get('val_loss', []),
            'mae': self.history.history['mae'],
            'val_mae': self.history.history.get('val_mae', []),
            'epochs': range(1, len(self.history.history['loss']) + 1)
        }
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target values
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        test_metrics = {
            'test_mse': mean_squared_error(y_test, y_pred),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        return test_metrics
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the Keras model
        self.model.save(filepath)
        
        # Save additional information
        metadata = {
            'feature_names': self.feature_names,
            'training_scores': self.training_scores,
            'config': self.config
        }
        
        import joblib
        metadata_path = filepath.replace('.h5', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model
        
        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load the Keras model
        self.model = keras.models.load_model(filepath)
        
        # Load metadata
        import joblib
        metadata_path = filepath.replace('.h5', '_metadata.pkl')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.feature_names = metadata['feature_names']
            self.training_scores = metadata['training_scores']
            self.config = metadata['config']
        
        self.is_fitted = True
        print(f"Model loaded from {filepath}")
    
    def get_feature_importance_approximation(self, X_sample, n_samples=1000):
        """
        Get approximate feature importance using permutation-based approach
        
        Args:
            X_sample (pd.DataFrame): Sample data for importance calculation
            n_samples (int): Number of samples to use
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Sample data if too large
        if len(X_sample) > n_samples:
            X_sample = X_sample.sample(n_samples, random_state=42)
        
        # Get baseline predictions
        baseline_pred = self.predict(X_sample)
        baseline_mse = np.mean((baseline_pred - baseline_pred.mean()) ** 2)
        
        feature_importance = []
        
        for feature in X_sample.columns:
            # Create a copy and shuffle the feature
            X_permuted = X_sample.copy()
            X_permuted[feature] = X_permuted[feature].sample(frac=1, random_state=42).values
            
            # Get predictions with permuted feature
            permuted_pred = self.predict(X_permuted)
            permuted_mse = np.mean((permuted_pred - baseline_pred.mean()) ** 2)
            
            # Calculate importance as increase in MSE
            importance = permuted_mse - baseline_mse
            feature_importance.append(importance)
        
        # Create DataFrame with results
        importance_df = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df


def create_deep_learning_model(config=None):
    """
    Factory function to create a deep learning model
    
    Args:
        config (dict): Model configuration
        
    Returns:
        EarthquakeDeepLearningModel: Configured deep learning model
    """
    return EarthquakeDeepLearningModel(config)
