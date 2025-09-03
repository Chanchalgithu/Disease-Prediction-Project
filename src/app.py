from flask import Flask, render_template, request
import joblib
import pandas as pd
import json
import os

# ==========================================================
# Paths (relative for Render or any server, not local drive)
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_columns.json")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "data", "processed", "label_mapping.json")
PREDICTIONS_CSV = os.path.join(BASE_DIR, "predictions.csv")

# ==========================================================
# Load Trained Model and Metadata
# ==========================================================
model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    FEATURES = json.load(f)

with open(LABEL_MAP_PATH, "r") as f:
    LABEL_MAPPING = json.load(f)

# ==========================================================
# Initialize Flask Application
# ==========================================================
app = Flask(__name__, template_folder="templates", static_folder="static")

# ==========================================================
# Home Page Route
# ==========================================================
@app.route("/")
def index():
    return render_template("index.html")

# ==========================================================
# Prediction Route
# ==========================================================
@app.route("/predict", methods=["POST"])
def predict():
    # ------------------------------
    # Collect Form Inputs
    # ------------------------------
    name = request.form.get("name")
    gender = request.form.get("gender")
    age = request.form.get("age")

    # ------------------------------
    # Collect Symptoms from Form
    # ------------------------------
    symptoms = []
    for i in range(1, 6):
        symptom = request.form.get(f"symptom{i}")
        if symptom:
            symptoms.append(symptom)

    # ------------------------------
    # Create Input Feature Vector
    # ------------------------------
    input_data = {feature: 0 for feature in FEATURES}
    for s in symptoms:
        if s in input_data:
            input_data[s] = 1

    df = pd.DataFrame([input_data])

    # ------------------------------
    # Predict Disease Safely
    # ------------------------------
    pred_class = model.predict(df)[0]

    # ------------------------------
    # Map Prediction to Disease Name
    # ------------------------------
    disease = LABEL_MAPPING.get(str(pred_class), str(pred_class))

    # ------------------------------
    # Save Prediction to CSV File
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

# ==========================================================
# Run Flask App with Render Port
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
