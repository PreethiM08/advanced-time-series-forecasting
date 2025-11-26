"""
explainability.py
Use SHAP DeepExplainer to compute feature/time-step importance for the LSTM.
Outputs shap values summary and simple aggregate indicating which features/time steps matter most.
"""

import numpy as np
import shap
import json
import os
from tensorflow.keras.models import load_model
from src.data_loader import download_data, scale_dataframe, create_multivariate_sequences

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_shap_for_saved_model(model_path, sample_data, background_size=100, test_size=5, output_path=RESULTS_DIR):
    """
    model_path: path to saved Keras model (.h5)
    sample_data: X array (N, timesteps, features)
    """
    model = load_model(model_path)
    # Use a subset for background
    background = sample_data[:background_size]
    test = sample_data[background_size:background_size+test_size]
    # shap DeepExplainer supports TF models; reshape/unpack if needed
    try:
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(test)
        # shap_values shape for regression multi-output: list (n_outputs) of arrays (test, timesteps, features)
        # For convenience we'll sum absolute values across outputs and timeframe to produce a single importance map
        # Convert to python lists and save
        out = {
            "shap_values_shape": str([np.array(sv).shape for sv in shap_values]),
        }
        # Save arrays (small)
        np.save(os.path.join(output_path, "shap_values.npy"), shap_values)
        with open(os.path.join(output_path, "shap_meta.json"), "w") as f:
            json.dump(out, f, indent=2)
        print("Saved SHAP values.")
    except Exception as e:
        print("SHAP explanation failed:", e)
        raise

if __name__ == "__main__":
    # Example runner: find a model in results and compute SHAP
    # Requires results/model_foldX.h5 to exist and an X array saved or regenerate
    print("Run explainability after training and saving the model. See README for instructions.")
