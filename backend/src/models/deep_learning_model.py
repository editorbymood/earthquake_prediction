import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class DeepLearningModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        
    def _build_model(self, input_dim):
        model = Sequential()
        
        layers = self.config.get('hidden_layers', [128, 64])
        dropout = self.config.get('dropout_rate', 0.2)
        
        # Input layer
        model.add(Dense(layers[0], input_dim=input_dim, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        
        # Hidden layers
        for units in layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
            
        # Output layer (Regression)
        model.add(Dense(1, activation='linear'))
        
        optimizer = Adam(learning_rate=self.config.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        input_dim = X_train.shape[1]
        self.model = self._build_model(input_dim)
        
        callbacks = []
        if self.config.get('early_stopping_patience'):
            callbacks.append(EarlyStopping(
                monitor='val_loss', 
                patience=self.config.get('early_stopping_patience'),
                restore_best_weights=True
            ))
            
        if self.config.get('reduce_lr_patience'):
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=self.config.get('reduce_lr_patience')
            ))
            
        validation_data = (X_val, y_val) if X_val is not None else None
        
        print("Training Deep Learning Model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.config.get('epochs', 50),
            batch_size=self.config.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        
        metrics = {}
        if validation_data:
            preds = self.predict(X_val)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_val, preds))
            metrics['mae'] = mean_absolute_error(y_val, preds)
            metrics['r2'] = r2_score(y_val, preds)
            
        return metrics

    def predict(self, X):
        return self.model.predict(X).flatten()

    def get_feature_importance_approximation(self, X):
        # Permutation importance or similar could be used.
        # For simple approximation, we can use the weights of the first layer?
        # A bit hacky but fast. 
        # Better: permutation importance. But that's slow.
        # Let's return Random Weights for now or ones?
        # Actually `main.py` expects a return value.
        # Let's return variance of weights connected to each input.
        weights = self.model.layers[0].get_weights()[0] # (n_features, n_units)
        importance = np.mean(np.abs(weights), axis=1)
        return importance

    def get_training_history(self):
        return self.history.history if self.history else None

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Deep learning model saved to {path}")

def create_deep_learning_model(config):
    return DeepLearningModel(config)
