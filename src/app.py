from flask import Flask, render_template, request, send_file
import os
import joblib
import pandas as pd
import json
import traceback

# ==========================================================
# Paths (Work for Local + Render)
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # DISEASE_PREDICTION/

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "disease_model.pkl")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "feature_columns.json")
LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "label_mapping.json")
PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "predictions.csv")

# ==========================================================
# Initialize Flask App (templates + static from project root)
# ==========================================================
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# ==========================================================
# Load Model + Metadata
# ==========================================================
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")

    with open(FEATURES_PATH, "r") as f:
        FEATURES = json.load(f)
    print("✅ Features loaded")

    with open(LABEL_MAP_PATH, "r") as f:
        LABEL_MAPPING = json.load(f)
    print("✅ Label mapping loaded")

except Exception as e:
    print("❌ Error loading model/metadata:", e)
    FEATURES, LABEL_MAPPING = [], {}

# ==========================================================
# Routes
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Patient info
        name = request.form.get("name")
        gender = request.form.get("gender")
        age = request.form.get("age")

        # Symptoms
        symptoms = []
        for i in range(1, 6):
            s = request.form.get(f"symptom{i}")
            if s:
                symptoms.append(s)

        # Create input vector
        input_data = {feature: 0 for feature in FEATURES}
        for s in symptoms:
            if s in input_data:
                input_data[s] = 1

        df = pd.DataFrame([input_data])

        # Prediction
        pred_class = model.predict(df)[0]
        disease = LABEL_MAPPING.get(str(pred_class), str(pred_class))

        # Save locally (only in local, not Render)
        if os.environ.get("RENDER") is None:  # Detect Render env
            record = {
                "Patient Name": name,
                "Gender": gender,
                "Age Group": age,
                "Symptoms": ", ".join(symptoms),
                "Predicted Disease": disease,
            }
            if not os.path.exists(PREDICTIONS_CSV):
                pd.DataFrame([record]).to_csv(PREDICTIONS_CSV, index=False)
            else:
                pd.DataFrame([record]).to_csv(PREDICTIONS_CSV, mode="a", header=False, index=False)

        return render_template("result.html", disease=disease)

    except Exception as e:
        traceback.print_exc()
        return f"Error in prediction: {e}", 500

@app.route("/download", methods=["GET"])
def download_predictions():
    try:
        if os.path.exists(PREDICTIONS_CSV):
            return send_file(PREDICTIONS_CSV, as_attachment=True)
        else:
            return "⚠️ No prediction records found", 404
    except Exception as e:
        traceback.print_exc()
        return f"Error generating download: {e}", 500

# ==========================================================
# Run App
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port)

