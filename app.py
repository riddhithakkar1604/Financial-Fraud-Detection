from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load saved files
model = pickle.load(open("logistic_regression_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    step = int(request.form["step"])
    transaction_type = request.form["type"]
    amount = float(request.form["amount"])
    oldbalanceOrg = float(request.form["oldbalanceOrg"])
    newbalanceOrig = float(request.form["newbalanceOrig"])
    oldbalanceDest = float(request.form["oldbalanceDest"])
    newbalanceDest = float(request.form["newbalanceDest"])

    # Encode transaction type
    transaction_type = label_encoder.transform([transaction_type])[0]

    # Create dataframe
    data = pd.DataFrame([[step,
                          transaction_type,
                          amount,
                          oldbalanceOrg,
                          newbalanceOrig,
                          oldbalanceDest,
                          newbalanceDest]],
                        columns=[
                            "step",
                            "type",
                            "amount",
                            "oldbalanceOrg",
                            "newbalanceOrig",
                            "oldbalanceDest",
                            "newbalanceDest"
                        ])

    # Scale data
    data_scaled = scaler.transform(data)

    # Predict
        # Predict
    prediction = model.predict(data_scaled)

    # Get prediction probabilities
    probabilities = model.predict_proba(data_scaled)[0]

    if prediction[0] == 1:
        result = "⚠ Fraud Transaction"
        confidence = probabilities[1] * 100
    else:
        result = "✅ Legitimate Transaction"
        confidence = probabilities[0] * 100

    # Determine risk level
    if confidence >= 90:
         risk = "HIGH"
    elif confidence >= 70:
         risk = "MEDIUM"
    else:
         risk = "LOW"

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2),
        risk=risk
    )
if __name__ == "__main__":
    app.run(debug=True)