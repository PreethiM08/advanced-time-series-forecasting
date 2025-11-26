"""
baseline.py
Implements the SARIMAX baseline for the target series.
It fits on the training period and forecasts the horizon for each validation fold.
"""

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

def sarimax_forecast_from_windows(X_train, y_train, X_val, target_index=0, order=(1,1,1), seasonal_order=(0,0,0,0)):
    """
    Fit SARIMAX on the flattened training 'target' series and forecast each step for len(X_val)
    Here we extract the target column from the original windows. This function assumes X arrays were generated in order.
    """
    # reconstruct 1D train series from windows: take the last available target in the training windows
    train_targets = []
    for i in range(len(X_train)):
        # X_train[i] is a window; the final time of window is the one before the forecast start
        train_targets.append(X_train[i][-1, target_index])
    train_series = np.array(train_targets)

    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.forecast(steps=len(X_val))
    # For multi-step horizon we repeat the single-step forecast to match shape (len(X_val), horizon) by naive method,
    # but better approach is to walk-forward with extended model; for conservativeness, we forecast only the first step and tile.
    return np.array(fc)
