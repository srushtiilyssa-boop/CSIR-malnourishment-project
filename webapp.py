"""
webapp.py
---------
Simple Flask UI to try out the trained malnutrition-screening model.

This loads the model + scaler saved by train_final_model.py (NOT the live
socket federation — that's what client.py/server.py are for). Think of this
as "take the model the FL demo produced, and let a health worker enter one
child's details to get a screening prediction."

Run order:
    python data_prep.py
    python train_final_model.py   # <-- creates model/ folder this app needs
    python webapp.py              # <-- then open http://127.0.0.1:5000
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

MODEL_DIR = Path("model")
TARGET_COL = "malnourished"

app = Flask(__name__)

model = None
scaler = None
feature_columns = None
load_error = None

try:
    model = joblib.load(MODEL_DIR / "malnutrition_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
except FileNotFoundError:
    load_error = (
        "Model files not found in model/. Run 'python train_final_model.py' "
        "first (after data_prep.py), then restart this app."
    )

# Field definitions for the form: (name, label, input type, extra)
# type is "select" (dropdown) or "number"
FIELDS = [
    ("child_age_months", "Child's age (months)", "number", {"min": 0, "max": 59, "step": 1}),
    ("child_age_years", "Child's age (years)", "number", {"min": 0, "max": 5, "step": 0.1}),
    ("gender", "Gender", "select", {"options": [(1, "Male"), (2, "Female")]}),
    ("birth_weight", "Birth size category (as recorded on health/survey card, 1=very large ... 5=very small)",
     "number", {"min": 1, "max": 8, "step": 1}),
    ("weight_kg", "Current weight (kg)", "number", {"min": 0, "max": 40, "step": 0.1}),
    ("height_cm", "Current height/length (cm)", "number", {"min": 30, "max": 130, "step": 0.1}),
    ("breastfeeding", "Breastfeeding duration (months; 95 = still breastfeeding, 98 = don't know)",
     "number", {"min": 0, "max": 98, "step": 1}),
    ("diarrhea", "Diarrhea in last 2 weeks?", "select", {"options": [(0, "No"), (1, "Yes")]}),
    ("fever", "Fever in last 2 weeks?", "select", {"options": [(0, "No"), (1, "Yes")]}),
    ("cough", "Cough in last 2 weeks?", "select", {"options": [(0, "No"), (1, "Yes")]}),
    ("mother_age", "Mother's age (years)", "number", {"min": 12, "max": 55, "step": 1}),
    ("mother_education", "Mother's education", "select",
     {"options": [(0, "No education"), (1, "Primary"), (2, "Secondary"), (3, "Higher")]}),
    ("mother_bmi", "Mother's BMI", "number", {"min": 10, "max": 60, "step": 0.1}),
    ("children_ever_born", "Children ever born (to mother)", "number", {"min": 1, "max": 15, "step": 1}),
    ("birth_order", "This child's birth order", "number", {"min": 1, "max": 15, "step": 1}),
    ("urban_rural", "Residence", "select", {"options": [(1, "Urban"), (2, "Rural")]}),
    ("wealth_index", "Household wealth index (1=poorest ... 5=richest)", "number",
     {"min": 1, "max": 5, "step": 1}),
    ("water_source", "Water source code (as per survey form)", "number", {"min": 10, "max": 99, "step": 1}),
    ("time_to_water", "Time to water source (minutes)", "number", {"min": 0, "max": 900, "step": 1}),
    ("sanitation_toilet_facility", "Toilet facility code (as per survey form)", "number",
     {"min": 10, "max": 99, "step": 1}),
    ("handwashing_facility", "Handwashing facility code (as per survey form)", "number",
     {"min": 10, "max": 99, "step": 1}),
    ("household_members", "Number of household members", "number", {"min": 1, "max": 30, "step": 1}),
    ("antenatal_visits", "Antenatal care visits during pregnancy", "number", {"min": 0, "max": 30, "step": 1}),
]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    submitted_values = {}

    if load_error:
        return render_template("index.html", fields=FIELDS, error=load_error,
                                result=None, values={})

    if request.method == "POST":
        try:
            row = {}
            for name, _, _, _ in FIELDS:
                raw = request.form.get(name, "")
                submitted_values[name] = raw
                if raw == "":
                    raise ValueError(f"Missing value for {name}")
                row[name] = float(raw)

            X = pd.DataFrame([row])[feature_columns]
            X_scaled = scaler.transform(X)
            pred = int(model.predict(X_scaled)[0])
            prob = float(model.predict_proba(X_scaled)[0, 1])

            result = {
                "label": "Malnourished (risk flagged)" if pred == 1 else "Not malnourished",
                "prob": round(prob * 100, 1),
                "pred": pred,
            }
        except Exception as e:
            error = f"Couldn't score this input: {e}"

    return render_template("index.html", fields=FIELDS, error=error,
                            result=result, values=submitted_values)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
