import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import json
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Load processed data
# -------------------------------
train_df = pd.read_csv(r"E:\Disease_Prediction\data\processed\train_clean.csv")
test_df  = pd.read_csv(r"E:\Disease_Prediction\data\processed\test_clean.csv")

X = train_df.drop("prognosis", axis=1)
y = train_df["prognosis"]

# -------------------------------
# Encode y temporarily for training
# -------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -------------------------------
# Train model (XGBoost example)
# -------------------------------
model = XGBClassifier(eval_metric="mlogloss")
model.fit(X, y_encoded)

# -------------------------------
# Save model
# -------------------------------
joblib.dump(model, r"E:\Disease_Prediction\models\disease_model.pkl")

# -------------------------------
# Save label mapping
# -------------------------------
label_map = {int(i): cls for i, cls in enumerate(le.classes_)}
with open(r"E:\Disease_Prediction\data\processed\label_mapping.json", "w") as f:
    json.dump(label_map, f)

print("✅ Model and Label Mapping saved successfully!")


