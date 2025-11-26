"""
data_loader.py
Load and preprocess the S&P 500 OHLCV dataset (yfinance).
Includes stationarity checks, differencing utilities and sequence creation.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, kpss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_data(ticker="^GSPC", period="10y", interval="1d"):
    """Download OHLCV data for `ticker` using yfinance."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df = df.dropna()
    logger.info("Downloaded data shape: %s", df.shape)
    return df

def stationarity_tests(series, signif=0.05):
    """
    Run ADF and KPSS tests on a pandas Series.
    Returns dict with results and recommended transformation flag.
    """
    adf_res = adfuller(series.dropna(), autolag='AIC')
    kpss_res = kpss(series.dropna(), nlags="auto", regression='c')
    out = {
        "adf_stat": adf_res[0],
        "adf_pvalue": adf_res[1],
        "kpss_stat": kpss_res[0],
        "kpss_pvalue": kpss_res[1],
        "adf_critical": adf_res[4],
        "kpss_critical": kpss_res[3],
        "stationary": (adf_res[1] < signif) and (kpss_res[1] > signif)
    }
    return out

def difference_series(df, columns=None, order=1):
    """Difference specified columns to help stationarity. Returns new df."""
    df_diff = df.copy()
    cols = columns if columns else df.columns
    for c in cols:
        df_diff[c] = df_diff[c].diff(order)
    df_diff = df_diff.dropna()
    return df_diff

def scale_dataframe(df, scaler=None):
    """Scale DataFrame using MinMaxScaler. Returns scaled numpy array and scaler."""
    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.values)
    else:
        scaled = scaler.transform(df.values)
    return scaled, scaler

def create_multivariate_sequences(data, input_len=60, forecast_horizon=10, target_col=0):
    """
    Converts 2D array (n_samples, n_features) to windows.
    Returns X: (N, input_len, n_features), y: (N, forecast_horizon)
    """
    X, y = [], []
    n = len(data)
    for i in range(n - input_len - forecast_horizon + 1):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+forecast_horizon, target_col])
    X = np.array(X)
    y = np.array(y)
    return X, y
