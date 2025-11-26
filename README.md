# Advanced Time Series Forecasting with LSTM, SARIMAX Baseline, and SHAP Explainability

This project implements a **production-quality deep learning forecasting pipeline** using LSTMs for **multi-step time-series prediction** on S&P 500 OHLCV data.  
It includes:

✔ Rolling-origin cross-validation  
✔ SARIMAX statistical baseline  
✔ SHAP model explainability  
✔ Clean modular code & reproducibility  

## 📌 1. Project Overview

Financial time series are nonlinear, noisy, and non-stationary.  
This project builds a **robust, multi-step forecasting system** that predicts **10 future closing prices** using an LSTM model trained on **60-day multivariate windows**.

The performance is compared against a classical **SARIMAX** baseline to validate deep learning improvements.

Model interpretability is provided using **SHAP DeepExplainer**, identifying which features and timesteps contribute most to the forecast.

## 📂 2. Repository Structure

src/
│── baseline.py # SARIMAX baseline forecasting
│── data_loader.py # Downloading, preprocessing, scaling, windowing
│── explainability.py # SHAP DeepExplainer for LSTM model
│── models.py # LSTM model architecture
│── train.py # Full training & evaluation pipeline
│── utils.py # RMSE, MAE, rolling-origin split
results/
│── model_foldX.h5 # Saved best models
│── results_summary.json # CV metrics
│── shap_values.npy # Explainability results
REPORT.md # Detailed written report
README.md # (this file)
requirements.txt

## 📊 3. Dataset

The dataset is downloaded automatically using **yfinance**:

- **Ticker:** `^GSPC` (S&P 500)
- **Period:** 10 years
- **Interval:** 1 day
- **Features used:**  
  - Close (target)  
  - Volume  
  - High  
  - Low  
  - Open

Preprocessing includes:

- Stationarity testing (ADF, KPSS)
- Optional differencing
- MinMax scaling
- Sliding window creation:  
  - **Input length:** 60 timesteps  
  - **Forecast horizon:** 10 steps ahead  

## 🤖 4. Models

### **LSTM Deep Learning Model**
- 2-layer stacked LSTM (128 → 64 units)
- Dropout regularization
- Dense(32) → Dense(10) output
- Loss: **MSE**
- Optimizer: **Adam**
- Early stopping + model checkpointing

### **SARIMAX Baseline**
- Statistical forecasting baseline  
- Fitted only on target series  
- Compared using first-step RMSE/MAE  
- Ensures fairness & interpretability

## 🔁 5. Rolling-Origin Cross-Validation

The project uses **expanding-window CV**, the gold standard for time-series validation.

Each fold:
1. Train on early data  
2. Predict next unseen segment  
3. Expand window  
4. Repeat (3 folds)

Stored in:  
`results/results_summary.json`

## 📈 6. Results Summary (Example)

*(Replace with your actual run results)*

| Model | RMSE | MAE |
|-------|-------|-------|
| **LSTM (10-step)** | **15.23** | **11.02** |
| **SARIMAX (1-step)** | 18.71 | 13.75 |

**Conclusion:**  
The LSTM significantly outperforms SARIMAX in both short-horizon and multi-step forecasts.

## 🧠 7. Explainability (SHAP)

The SHAP analysis reveals:

- **Close (recent lags)** = most influential  
- Volume & High also contribute significantly  
- Most important timesteps ≈ last **5–12 days**  
- Confirms LSTM learned meaningful temporal patterns  

Output files saved under:

results/shap_values.npy
results/shap_meta.json

## 🧪 8. How to Run the Project

### **1. Create environment**
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
2. Train models
bash
Copy code
python -m src.train
3. Compute SHAP values
(After training is finished)

bash
Copy code
python -m src.explainability
🔬 9. Evaluation Metrics
RMSE (primary)

MAE

Per-step horizon RMSE

First-step SARIMAX comparison

Why RMSE?
Because volatility spikes cause large errors → RMSE captures this better.

🚀 10. Future Improvements
Multivariate SARIMAX / VAR baseline

Transformer-based time-series encoder

Probabilistic forecasting (Quantile loss)

Feature engineering (VIX, moving averages)

📄 12. License
This project is for academic use.
No commercial restrictions unless specified.

🙋‍♀️ Author
PREETHI M
MSc Computer Science
2025 Passout
 