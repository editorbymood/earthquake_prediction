import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class SeismicLSTM:
    def __init__(self, input_shape):
        """
        Initialize LSTM model for seismic sequence prediction.
        input_shape: (sequence_length, num_features)
        """
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=self.input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1) # Predict Magnitude of next event
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        return history

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath):
        # Helper to load model; in practice you'd load the Keras model directly
        return tf.keras.models.load_model(filepath)
