"""
models.py
Defines the LSTM model architecture and helper to compile.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm_model(input_shape,
                     lstm_units=(128,64),
                     dropout_rate=0.2,
                     l2_reg=1e-4,
                     dense_units=32,
                     forecast_horizon=10):
    """
    Build and compile a regularized LSTM model for sequence-to-sequence forecasting.
    input_shape: (timesteps, features)
    """
    reg = tf.keras.regularizers.l2(l2_reg)
    inp = layers.Input(shape=input_shape, name="input_sequence")
    x = inp
    # First LSTM with return_sequences True if stacking
    x = layers.LSTM(lstm_units[0], return_sequences=True, kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LSTM(lstm_units[1], kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
    out = layers.Dense(forecast_horizon, name="forecast")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse", metrics=["mae"])
    return model
