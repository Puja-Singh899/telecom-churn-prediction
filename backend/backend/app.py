from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and encoder
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
feature_names = model_data["feature_names"]

with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

# Preprocess incoming data
def preprocess_input(data_dict):
    processed = []
    for feature in feature_names:
        val = data_dict.get(feature)
        if feature in encoders:
            val = encoders[feature].transform([val])[0]
        processed.append(val)
    return np.array(processed).reshape(1, -1)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = preprocess_input(data)
    prediction = model.predict(features)[0]
    result = "Churn" if prediction == 1 else "No Churn"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
