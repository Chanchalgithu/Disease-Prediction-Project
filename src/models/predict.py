import joblib
import pandas as pd
import json
import os

# ------------------------------
# Paths (Work for Local + Render)
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # DISEASE_PREDICTION/

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "disease_model.pkl")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "feature_columns.json")
LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "label_mapping.json")
PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "predictions.csv")

# ------------------------------
# Load Model & Metadata
# ------------------------------
model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    FEATURES = json.load(f)

with open(LABEL_MAP_PATH, "r") as f:
    LABEL_MAPPING = json.load(f)

# ------------------------------
# Test Input (example)
# ------------------------------
name = "Test Patient"
gender = "Male"
age = "21-30"
symptoms = ["fatigue", "nausea", "yellowing_of_eyes"]

# ------------------------------
# Create input vector
# ------------------------------
input_data = {feature: 0 for feature in FEATURES}
for s in symptoms:
    if s in input_data:
        input_data[s] = 1

df = pd.DataFrame([input_data])
pred_class = model.predict(df)[0]
disease = LABEL_MAPPING.get(str(pred_class), str(pred_class))

# ------------------------------
# Save to CSV (works locally)
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
# Output
# ------------------------------
print("🧾 Symptoms:", symptoms)
print("✅ Predicted Disease:", disease)
print(f"📂 Saved in {PREDICTIONS_CSV}")


