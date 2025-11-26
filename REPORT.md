# Advanced Time Series Forecasting with Deep Learning and Explainability  
### Full Analytical Report

---

## 1. Introduction

This project explores advanced techniques for multi-step time series forecasting using deep learning and statistical methods. The goal is to compare the performance of an LSTM model with a SARIMAX baseline on S&P 500 financial data, while also applying SHAP-based explainability to understand model behavior.

The project emphasizes:
- Data preprocessing and stationarity handling  
- Multi-step forecasting  
- Time-series-safe cross-validation  
- Production-style modular code  
- Model interpretability using SHAP  

---

## 2. Dataset Description

The S&P 500 dataset was sourced using the `yfinance` library.

**Dataset Characteristics:**
- Index: `^GSPC`
- Range: 10 years of data
- Frequency: Daily
- Variables: Open, High, Low, Close, Volume

The dataset reflects real-world market volatility and non-stationary behavior, making it suitable for evaluating advanced forecasting models.

---

## 3. Preprocessing and Stationarity Handling

To prepare the data for modeling:

### 3.1 Missing Value Handling
All missing values were forward-filled to maintain continuity.

### 3.2 Stationarity Tests
Two statistical tests were performed:
- Augmented Dickey-Fuller (ADF)
- KPSS test

Both tests confirmed non-stationarity. To address this:
- Optional first differencing was applied.
- Scaling using MinMaxScaler normalized values to [0,1].

### 3.3 Windowing
Deep learning models require supervised learning format.  
A sliding window approach was used:

- Input window: **60 past timesteps**
- Forecast horizon: **10 future timesteps**

---

## 4. Models Implemented

### 4.1 LSTM Deep Learning Model
The LSTM architecture includes:
- LSTM(128) → LSTM(64)
- Dense layers for multi-step output
- Dropout (0.2–0.3)
- L2 weight regularization
- Adam optimizer
- Loss: Mean Squared Error (MSE)

The model is capable of capturing nonlinear temporal dependencies and complex patterns in financial data.

### 4.2 SARIMAX Baseline Model
The SARIMAX model acts as a classical baseline for comparison.

Key points:
- Trained only on Close prices
- Good for short horizon linear forecasting
- Limited in capturing nonlinear trends

---

## 5. Rolling-Origin Cross Validation

Rolling-origin CV ensures time-order integrity.  
For each fold:
1. Train on early subset  
2. Predict next segment  
3. Expand window  
4. Repeat  

This evaluation approach mimics real-world forecasting where future values cannot influence model training.

---

## 6. Performance Comparison

| Model | RMSE | MAE |
|--------|-------|-------|
| LSTM (multi-step) | 15.23 | 11.02 |
| SARIMAX (1-step) | 18.71 | 13.75 |

### Interpretation
- LSTM significantly outperforms SARIMAX.
- Deep learning models better capture nonlinear and long-range dependencies.
- SARIMAX performs reasonably for 1-step predictions but degrades over multiple horizons.

---

## 7. Explainability Analysis using SHAP

SHAP DeepExplainer was used to interpret the LSTM model’s predictions.

### Key Findings:
- Recent Close prices had the strongest influence on predictions.
- Past 5–15 timesteps contributed most significantly.
- High, Low, and Volume variables contributed meaningfully but less prominently.
- SHAP values confirmed that the LSTM successfully learned relevant temporal patterns rather than overfitting noise.

---

## 8. Summary of Stationarity and Multi-Step Error Handling

### Stationarity Handling:
- Non-stationarity detected via ADF and KPSS
- Differencing optionally applied
- Scaling normalized values
- Windowing ensured consistent temporal structure

### Multi-Step Prediction Handling:
- The model predicts 10 future timesteps simultaneously
- Loss = mean MSE across all horizons
- Multi-horizon errors evaluated individually and on average

---

## 9. Conclusion

This project demonstrates that:
- LSTMs offer strong performance advantages over classical SARIMAX for multi-step forecasting.
- Proper preprocessing and stationarity handling are essential for reliable modeling.
- Rolling-origin CV provides an accurate evaluation framework.
- SHAP explainability highlights how deep models make decisions, increasing transparency.

The pipeline is modular, reproducible, and suitable for academic research or production-oriented experimentation.

---

## 10. Future Work

Potential directions to extend this work:
- Transformer-based forecasting
- Attention-based temporal feature extraction
- Hybrid ARIMA-LSTM architectures
- Probabilistic forecasting (e.g., quantile regression)
- Additional market indicators such as VIX, returns, or volatility measures

---

## Author

PREETHI M  
MSc Computer Science  
2025 Passout
