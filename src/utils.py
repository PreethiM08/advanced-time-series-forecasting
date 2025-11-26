"""
utils.py
Utility functions for evaluation and rolling-origin splitting.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))

def mae(a, b):
    return mean_absolute_error(a, b)

def rolling_origin_split(X, y, n_splits=3, min_train_size=None):
    """
    Expandng-window (rolling-origin) splits.
    Yields tuples: (X_train, y_train, X_val, y_val)
    """
    n = len(X)
    if min_train_size is None:
        min_train_size = n // (n_splits + 1)
    fold_size = (n - min_train_size) // n_splits
    for i in range(n_splits):
        train_end = min_train_size + i * fold_size
        val_start = train_end
        val_end = train_end + fold_size
        yield X[:train_end], y[:train_end], X[val_start:val_end], y[val_start:val_end]
