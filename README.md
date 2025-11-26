# Advanced Time Series Forecasting with Deep Learning and Explainability

## Summary
This repository implements a production-quality LSTM model for multi-step forecasting on S&P 500 OHLCV data, compares it with a SARIMAX baseline, and uses SHAP for model explainability.

## Project structure
See top-level files and `src/` for code.

## Reproducibility
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
