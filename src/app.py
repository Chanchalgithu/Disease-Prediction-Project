from flask import Flask, render_template, request
import joblib
import pandas as pd
import json
import os

# ------------------------------
# Paths
# ------------------------------
MODEL_PATH = r"E:\Disease_Prediction\models\disease_model.pkl"
FEATURES_PATH = r"E:\Disease_Prediction\data\processed\feature_columns.json"
LABEL_MAP_PATH = r"E:\Disease_Prediction\data\processed\label_mapping.json"
PREDICTIONS_CSV = r"E:\Disease_Prediction\predictions.csv"

# ------------------------------
# Load Model & Metadata
# ------------------------------
model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    FEATURES = json.load(f)

with open(LABEL_MAP_PATH, "r") as f:
    LABEL_MAPPING = json.load(f)

# ------------------------------
# Flask App
# ------------------------------
app = Flask(
    __name__,
    template_folder=r"E:\Disease_Prediction\templates",
    static_folder=r"E:\Disease_Prediction\static"
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect form inputs
    name = request.form.get("name")
    gender = request.form.get("gender")
    age = request.form.get("age")

    # Collect symptoms (symptom1–symptom5)
    symptoms = []
    for i in range(1, 6):
        symptom = request.form.get(f"symptom{i}")
        if symptom:
            symptoms.append(symptom)

    # Create input vector
    input_data = {feature: 0 for feature in FEATURES}
    for s in symptoms:
        if s in input_data:
            input_data[s] = 1

    df = pd.DataFrame([input_data])

    # ------------------------------
    # Predict Disease safely
    # ------------------------------
    pred_class = model.predict(df)[0]

    # Use LABEL_MAPPING if numeric, else fallback to pred_class itself
    disease = LABEL_MAPPING.get(str(pred_class), str(pred_class))

    # ------------------------------
    # Save Prediction to CSV
    # ------------------------------
    record = {
        "Patient Name": name,
        "Gender": gender,
        "Age Group": age,
        "Symptoms": ", ".join(symptoms),
        "Predicted Disease": disease
    }

    if not os.path.exists(PREDICTIONS_CSV):
        pd.DataFrame([record]).to_csv(PREDICTIONS_CSV, index=False)
    else:
        pd.DataFrame([record]).to_csv(PREDICTIONS_CSV, mode="a", header=False, index=False)

    # ------------------------------
    # Render Result Page
    # ------------------------------
    return render_template("result.html", disease=disease)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
