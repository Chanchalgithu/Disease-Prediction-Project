
"""
helpers.py

Utility functions for Disease Prediction project:
- save_prediction: Save predicted record to CSV safely
- create_input_vector: Convert user-selected symptoms into model input vector
- get_disease_name: Map numeric prediction to disease name using label mapping

These functions are modular and reusable across app.py and model scripts.
"""

import pandas as pd
import os

def save_prediction(record, csv_path):
    """
    Save a single prediction record to CSV.
    If CSV doesn't exist, create with header.
    """
    if not os.path.exists(csv_path):
        pd.DataFrame([record]).to_csv(csv_path, index=False)
    else:
        pd.DataFrame([record]).to_csv(csv_path, mode="a", header=False, index=False)

def create_input_vector(symptoms, feature_columns):
    """
    Convert list of selected symptoms into a DataFrame suitable for model prediction.
    """
    input_data = {feature: 0 for feature in feature_columns}
    for s in symptoms:
        if s in input_data:
            input_data[s] = 1
    df = pd.DataFrame([input_data])
    return df

def get_disease_name(pred_index, label_mapping):
    """
    Convert numeric prediction to disease name using label mapping.
    """
    return label_mapping.get(str(pred_index), "Unknown Disease")

