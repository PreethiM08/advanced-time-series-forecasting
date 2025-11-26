"""
train.py
End-to-end training script:
- Loads data
- Preprocesses (optional differencing + scaling)
- Creates sequences
- Performs rolling-origin CV
- Trains LSTM model (with early stopping)
- Evaluates vs SARIMAX baseline
- Saves best model and results
"""

import os
import json
import numpy as np
from src.data_loader import download_data, stationarity_tests, difference_series, scale_dataframe, create_multivariate_sequences
from src.utils import rolling_origin_split, rmse, mae
from src.models import build_lstm_model
from src.baseline import sarimax_forecast_from_windows
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_experiment(ticker="^GSPC",
                   period="10y",
                   input_len=60,
                   forecast_horizon=10,
                   n_splits=3,
                   differencing=False,
                   epochs=100,
                   batch_size=32):
    # 1. Load
    df = download_data(ticker=ticker, period=period)
    # Choose features: Close, Volume, High, Low
    df = df[["Close", "Volume", "High", "Low", "Open"]].dropna()

    # 2. Stationarity check on Close
    st = stationarity_tests(df["Close"])
    print("Stationarity (Close):", st)

    # 3. Optionally difference to help stationarity (assignment requires discussion)
    if differencing:
        df = difference_series(df, columns=df.columns, order=1)

    # 4. Scale
    scaled, scaler = scale_dataframe(df)

    # 5. Create sequences
    X, y = create_multivariate_sequences(scaled, input_len=input_len, forecast_horizon=forecast_horizon, target_col=0)
    print("Sequences shapes:", X.shape, y.shape)

    # 6. Rolling-origin CV
    fold_results = []
    fold_num = 0
    for X_train, y_train, X_val, y_val in rolling_origin_split(X, y, n_splits=n_splits):
        fold_num += 1
        # Build model
        model = build_lstm_model(input_shape=X_train.shape[1:], forecast_horizon=forecast_horizon)
        model_name = f"model_fold{fold_num}.h5"
        mcp = ModelCheckpoint(os.path.join(RESULTS_DIR, model_name), save_best_only=True, monitor="val_loss", mode="min")
        es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

        # Train
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=epochs, batch_size=batch_size, callbacks=[es,mcp], verbose=1)
        # Predict
        preds = model.predict(X_val)
        # Evaluate multi-step by comparing predicted horizon vector
        fold_rmse = rmse(y_val.flatten(), preds.flatten())
        fold_mae = mae(y_val.flatten(), preds.flatten())

        # Baseline SARIMAX - only first-step baseline is produced here; we compare first step explicitly in report
        sar_pred = sarimax_forecast_from_windows(X_train, y_train, X_val)
        # Compare first step only
        sar_rmse = rmse(y_val[:,0], sar_pred[:len(y_val)])
        sar_mae = mae(y_val[:,0], sar_pred[:len(y_val)])

        fold_results.append({
            "fold": fold_num,
            "lstm_rmse": float(fold_rmse),
            "lstm_mae": float(fold_mae),
            "sarimax_rmse_first_step": float(sar_rmse),
            "sarimax_mae_first_step": float(sar_mae),
            "history": {k: [float(x) for x in v] for k,v in history.history.items()}
        })
        print(f"Fold {fold_num} results:", fold_results[-1])

    # Save results summary
    with open(os.path.join(RESULTS_DIR, "results_summary.json"), "w") as f:
        json.dump(fold_results, f, indent=2)
    print("Saved results to", RESULTS_DIR)
    return fold_results

if __name__ == "__main__":
    run_experiment()
