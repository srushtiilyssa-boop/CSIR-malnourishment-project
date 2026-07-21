import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("malnutrition_model.joblib")
scaler = joblib.load("scaler.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Parse inputs from frontend
        age = float(data["age"])
        gender = int(data["gender"])  # 1: Male, 2: Female
        height = float(data["height"])
        weight = float(data["weight"])

        # Format input dataframe
        input_data = pd.DataFrame(
            [[age, gender, height, weight]],
            columns=["child_age_months", "gender", "height_cm", "weight_kg"],
        )

        # Scale features and predict
        scaled_input = scaler.transform(input_data)
        prediction = int(model.predict(scaled_input)[0])
        probabilities = model.predict_proba(scaled_input)[0]
        confidence = float(probabilities[prediction] * 100)

        result_text = "Malnourished" if prediction == 1 else "Healthy"

        return jsonify(
            {
                "success": True,
                "prediction": result_text,
                "confidence": round(confidence, 2),
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)