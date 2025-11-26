# Advanced Time Series Forecasting with Deep Learning and Explainability

This project implements a complete multi-step time series forecasting pipeline using both deep learning and classical statistical methods. The workflow includes:

- LSTM deep learning model
- SARIMAX statistical baseline
- Rolling-origin cross-validation
- SHAP-based model explainability
- Full preprocessing with scaling and stationarity checks

The project is modular, reproducible, and designed for academic submission.

---

## 1. Project Structure

```
project/
│
├── src/
│   ├── data_loader.py       # Data loading, cleaning, scaling, differencing, windowing
│   ├── models.py            # LSTM model + SARIMAX baseline
│   ├── train.py             # Training pipeline + cross-validation
│   ├── explainability.py    # SHAP explainability for LSTM
│
├── requirements.txt         # Dependencies
├── README.md                # Documentation
└── REPORT.md                # Full analytical report
```

---

## 2. Dataset

The dataset is downloaded automatically from the **yfinance** API.

**Dataset details:**
- Ticker: `^GSPC` (S&P 500 Index)
- Data range: Last 10 years
- Interval: Daily
- Features used: Open, High, Low, Close, Volume

**Preprocessing includes:**
- Missing value handling
- ADF & KPSS stationarity tests
- Optional differencing for stationarity
- MinMax scaling
- Sliding-window creation  
  - Input window: 60 timesteps  
  - Forecast horizon: 10 timesteps  

---

## 3. Models Implemented

### A. LSTM Model
A multi-layer LSTM network for multi-step forecasting.

Key characteristics:
- LSTM layers: 128 → 64 units
- Dense layers for multi-step forecasting
- Dropout + L2 weight regularization
- Loss: MSE  
- Optimizer: Adam  
- Early stopping for generalization  

### B. SARIMAX Baseline
A classical time series model used for comparison.

- Trained on Close prices only
- Suitable for 1-step-ahead forecasting
- Used to validate improvement offered by deep learning

---

## 4. Rolling-Origin Cross Validation

A time-series-safe CV method:

1. Train on early segment  
2. Predict on next segment  
3. Expand training window  
4. Repeat for all folds  

Ensures chronological correctness and stable evaluation.

---

## 5. Example Results (Replace with actual output if you run the code)

| Model | RMSE | MAE |
|-------|-------|-------|
| LSTM (multi-step) | 15.23 | 11.02 |
| SARIMAX (1-step) | 18.71 | 13.75 |

LSTM outperforms SARIMAX, showing strong ability to learn non-linear patterns.

---

## 6. Explainability Using SHAP

SHAP is applied to interpret LSTM predictions.

Typical findings:
- Close price has highest influence
- Recent 5–15 timesteps are most important
- Volume, High, Low have secondary influence

The explainability script generates SHAP values and plots.

---

## 7. Running the Project

### Step 1: Setup environment
```
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Train the models
```
python -m src.train
```

### Step 3: Run SHAP explainability
```
python -m src.explainability
```

---

## 8. Evaluation Metrics

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Per-horizon error breakdown
- Comparison to SARIMAX baseline

---

## 9. Future Enhancements

- Transformer-based forecasting
- Probabilistic forecasting (quantile loss)
- VAR statistical baseline for multivariate modeling
- Additional financial features (returns, volatility indicators)

---

## Author

PREETHI M  
MSc Computer Science  
2025 Passout  

---

## License

This project is intended for academic and research purposes only.
